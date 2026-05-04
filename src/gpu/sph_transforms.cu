// sph_transforms.cu — 2D 球面 (θ,φ) 譜變換的 orchestration。
//
// 實作說明:
//   - Gauss-Legendre 節點/權重由 Newton-Raphson 迭代求 Legendre roots(host 端)。
//   - Associated Legendre P_l^m(x) 由三項遞推公式計算,已乘上 4π-normalization。
//   - φ 方向用 cuFFT batched R2C / C2R(N_theta 條緯度同時做)。
//   - θ 方向的 Legendre transform 目前是 O(N²) naive(kernel 直接求和)。
//     後續若需要上 SHTns,可替換此函式而不動上層 API。

#include "sph_transforms.cuh"
#include "fas_common.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---- kernel forward decls (sph2d_spectral_kernels.cu 也會用到這兩個) ----
__global__ void k_sph_forward_legendre(const cufftDoubleComplex* fm_theta,
                                       const double* P_table,
                                       const double* weight,
                                       cufftDoubleComplex* out_lm,
                                       int N_theta, int N_mphi,
                                       int L_max, int n_lm);
__global__ void k_sph_inverse_legendre(const cufftDoubleComplex* in_lm,
                                       const double* P_table,
                                       cufftDoubleComplex* out_fm_theta,
                                       int N_theta, int N_mphi,
                                       int L_max);
__global__ void k_sph_scale_phys(double* f, double scale, int n);

#define CUFFT_CHECK(call) do {                                                   \
    cufftResult _r = (call);                                                     \
    if (_r != CUFFT_SUCCESS) {                                                   \
        std::fprintf(stderr, "cuFFT err %d at %s:%d\n", _r, __FILE__, __LINE__); \
        std::abort();                                                             \
    }                                                                             \
} while (0)

// ============================================================
// Gauss-Legendre quadrature nodes on [-1, 1]:
//   返回 N 個 Legendre 根 x_j 和權重 w_j。
// 用 Newton-Raphson 從 Chebyshev 近似 (cos((k-0.25)π/(N+0.5))) 起始收斂。
// 參考 Numerical Recipes gauleg。
// ============================================================
static void gauss_legendre_nodes(int N, std::vector<double>& x,
                                 std::vector<double>& w) {
    x.assign(N, 0.0);
    w.assign(N, 0.0);
    const double eps = 1e-15;
    int m = (N + 1) / 2;
    for (int i = 1; i <= m; ++i) {
        double z = std::cos(M_PI * (i - 0.25) / (N + 0.5));
        double z1, pp;
        do {
            double p1 = 1.0, p2 = 0.0;
            for (int j = 1; j <= N; ++j) {
                double p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
            }
            pp = N * (z * p1 - p2) / (z * z - 1.0);
            z1 = z;
            z  = z1 - p1 / pp;
        } while (std::fabs(z - z1) > eps);
        // 節點按 cos θ 從 +1 到 -1 的順序(θ 從 0 到 π)
        x[i - 1]       = -z;
        x[N - i]       =  z;
        w[i - 1]       = 2.0 / ((1.0 - z * z) * pp * pp);
        w[N - i]       = w[i - 1];
    }
}

// ============================================================
// Precompute P_l^m(x) · N_l^m for all (j, l, m):
//   用 standard recurrence for normalized P_l^m:
//     P̄_m^m(x) = √((2m+1)/(4π)) · ∏_{k=1}^{m} √((2k-1)/(2k)) · (1-x²)^(m/2)     (diagonal)
//     P̄_{m+1}^m(x) = √(2m+3) · x · P̄_m^m(x)
//     P̄_l^m(x) = a_l · x · P̄_{l-1}^m(x) - b_l · P̄_{l-2}^m(x)                   (l>m+1)
//   係數:
//     a_l = √((2l-1)(2l+1)/((l-m)(l+m)))
//     b_l = √((2l+1)(l-1-m)(l-1+m) / ((2l-3)(l-m)(l+m)))
//   這個 P̄ 已包含 √((2l+1)/(4π)·(l-m)!/(l+m)!) normalization,
//   符合 4π-orthonormal 約定:
//     ∫_{-1}^1 P̄_l^m(x)² dx · 2π = 1    (m=0)
//     ∫_{-1}^1 P̄_l^m(x)² dx · π  = 1    (m>0,因為 exp(imφ) 歸一額外 1/√2π)
//
// 為簡化,我們用的 normalization 約定:
//   對 m = 0:f 在物理和 (l,0) 譜空間雙向滿足 Parseval 時,P̄ 含 √((2l+1)/(4π))
//   對 m > 0:由於 R2C cuFFT 的複數 conj 對稱(f̂(-m) = conj(f̂(m))),
//            在物理總和 = 2·Re[Σ f̂(m)·exp(imφ)] 時需保持譜到物理的 2·Re 恢復正確。
//   具體實作我們讓 spatial_to_spectral 和 spectral_to_spatial 各自歸一自洽,
//   能在 Rossby 波 analytic roundtrip 下 err ≤ 1e-12 即可。
// ============================================================
static void precompute_P_table(int L_max, int N_theta,
                               const std::vector<double>& x,
                               std::vector<double>& P_table /* (N_theta, n_lm) */) {
    int n_lm = (L_max + 1) * (L_max + 2) / 2;
    P_table.assign((size_t)N_theta * n_lm, 0.0);

    const double pi = M_PI;
    for (int j = 0; j < N_theta; ++j) {
        double xj = x[j];
        double sin2 = 1.0 - xj * xj;
        double sinj = std::sqrt(std::max(sin2, 0.0));

        // 遞推 m = 0..L_max 的 diagonal Pmm
        // Pmm_raw = (-1)^m · (2m-1)!! · (sin θ)^m   (unnormalized)
        // 歸一後:P̄_m^m(x) = √((2m+1)/(4π) · 1/(2m)!) · (2m-1)!! · sin^m θ
        //                  = √((2m+1)/(4π)) · ∏_{k=1}^{m} √((2k-1)/(2k)) · sin^m θ
        // 用增量法避免階乘溢位。
        double pmm = std::sqrt(1.0 / (4.0 * pi));    // P̄_0^0 = 1/√(4π)
        P_table[(size_t)j * n_lm + 0 /* idx(0,0) */] = pmm;
        for (int m = 1; m <= L_max; ++m) {
            // P̄_m^m = -√((2m+1)/(2m)) · sin θ · P̄_{m-1}^{m-1}
            // (符號選擇:Condon-Shortley phase 用於 spherical harmonics,
            //  不帶 (-1)^m 會讓 Y_l^m 和 Y_l^{-m} 共軛關係符號不一致;
            //  這裡統一用正號約定,m<0 recovery 時額外 × (-1)^m)
            pmm = std::sqrt((2.0 * m + 1.0) / (2.0 * m)) * sinj * pmm;
            int idx = SphTransform::lm_to_idx(m, m);
            P_table[(size_t)j * n_lm + idx] = pmm;
        }

        // 對每個 m,遞推 l = m+1..L_max 的 P̄_l^m
        for (int m = 0; m <= L_max; ++m) {
            double pmm_val = P_table[(size_t)j * n_lm + SphTransform::lm_to_idx(m, m)];
            if (m + 1 > L_max) continue;
            // P̄_{m+1}^m = √(2m+3) · x · P̄_m^m
            double pmm1 = std::sqrt(2.0 * m + 3.0) * xj * pmm_val;
            int idx_mm1 = SphTransform::lm_to_idx(m + 1, m);
            P_table[(size_t)j * n_lm + idx_mm1] = pmm1;

            double p_lm2 = pmm_val;
            double p_lm1 = pmm1;
            for (int l = m + 2; l <= L_max; ++l) {
                double a = std::sqrt((2.0 * l - 1.0) * (2.0 * l + 1.0)
                                     / ((double)(l - m) * (l + m)));
                double b = std::sqrt((2.0 * l + 1.0) * (l - 1 - m) * (l - 1 + m)
                                     / ((2.0 * l - 3.0) * (l - m) * (l + m)));
                double p_l = a * xj * p_lm1 - b * p_lm2;
                int idx_l = SphTransform::lm_to_idx(l, m);
                P_table[(size_t)j * n_lm + idx_l] = p_l;
                p_lm2 = p_lm1;
                p_lm1 = p_l;
            }
        }
    }
}

// ============================================================
// init / destroy
// ============================================================
void SphTransform::init(int N_theta_, int N_phi_, int L_max_, double R_) {
    N_theta = N_theta_;
    N_phi   = N_phi_;
    N_mphi  = N_phi / 2 + 1;
    // 預設取 L_max ≤ min(N_theta - 1, N_mphi - 2) 確保不 alias
    L_max   = std::min(L_max_,
                       std::min(N_theta - 1, N_mphi - 2));
    n_lm    = (L_max + 1) * (L_max + 2) / 2;
    n_phys  = N_theta * N_phi;
    n_fper  = N_theta * N_mphi;
    R       = R_;

    // Gauss-Legendre 節點
    gauss_legendre_nodes(N_theta, h_cos_theta, h_weight);

    // Precompute P table (host)
    std::vector<double> h_P;
    precompute_P_table(L_max, N_theta, h_cos_theta, h_P);

    // 上傳到 device
    CUDA_CHECK(cudaMalloc(&d_cos_theta, N_theta * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_weight,    N_theta * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_P_table,
                          (size_t)N_theta * n_lm * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fmphi_tmp,
                          n_fper * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMemcpy(d_cos_theta, h_cos_theta.data(),
                          N_theta * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(),
                          N_theta * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_P_table, h_P.data(),
                          (size_t)N_theta * n_lm * sizeof(double),
                          cudaMemcpyHostToDevice));

    // cuFFT batched plans along φ (length = N_phi), batch = N_theta
    int n_fft[1] = { N_phi };
    int inembed_r[1] = { N_phi };
    int onembed_c[1] = { N_mphi };
    CUFFT_CHECK(cufftPlanMany(&plan_r2c, 1, n_fft,
                              inembed_r, 1, N_phi,
                              onembed_c, 1, N_mphi,
                              CUFFT_D2Z, N_theta));
    int inembed_c[1] = { N_mphi };
    int onembed_r[1] = { N_phi };
    CUFFT_CHECK(cufftPlanMany(&plan_c2r, 1, n_fft,
                              inembed_c, 1, N_mphi,
                              onembed_r, 1, N_phi,
                              CUFFT_Z2D, N_theta));

    std::fprintf(stderr,
        "  SphTransform: N_theta=%d N_phi=%d L_max=%d (n_lm=%d) R=%.3g\n",
        N_theta, N_phi, L_max, n_lm, R);
}

void SphTransform::destroy() {
    if (plan_r2c) { cufftDestroy(plan_r2c); plan_r2c = 0; }
    if (plan_c2r) { cufftDestroy(plan_c2r); plan_c2r = 0; }
    auto F = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
    F((void*&)d_cos_theta);
    F((void*&)d_weight);
    F((void*&)d_P_table);
    F((void*&)d_fmphi_tmp);
}

// ============================================================
// Forward: spatial f(θ_j, φ_i) → spectral f̂(l, m)
//
// Step 1: cuFFT batched R2C 沿 φ → f_m(θ_j) 複數 (N_theta × N_mphi)
//         cuFFT 輸出是 unnormalized (Σ_i f(θ,φ_i) · exp(-i·2π·m·i/N_phi)),
//         我們除以 N_phi 以轉成「離散 Fourier 係數」:
//           f_m(θ) = (1/N_phi) · Σ_i f(θ, φ_i) · exp(-imφ_i)
//         這樣 Σ_m f_m(θ) · exp(imφ_i) = f(θ, φ_i)(對稱 m 恢復)。
//
// Step 2: Gauss-Legendre quadrature along θ:
//           f̂_l^m = Σ_j w_j · P̄_l^m(x_j) · f_m(θ_j)
//         (w_j 是 [-1,1] 區間權重,∫dx = ∫sinθ dθ,自動吸收 sin θ Jacobian)
//
// 輸出:out_lm[idx_lm(l,m)],僅 m ≥ 0,m ≤ l。
// ============================================================
void SphTransform::spatial_to_spectral(const double* in_phys,
                                       cufftDoubleComplex* out_lm) {
    // cuFFT R2C 會寫入 out(N_theta × N_mphi)
    // (cuFFT D2Z 不保證保留 in,但 in_phys 是 const,我們需要拷貝先)
    // 為了簡化,用 d_fmphi_tmp 作為 complex 輸出 buffer。
    // cuFFT R2C 需要非 const double*;我們需要一個 mutable scratch,
    // 但其實 cuFFT 的文件說 R2C 會保留 input(僅 out-of-place) — 我們這裡傳同指標
    // 並使用 cuFFT out-of-place 自動保留 in。
    CUFFT_CHECK(cufftExecD2Z(plan_r2c,
                             const_cast<double*>(in_phys),
                             d_fmphi_tmp));
    // 標準化 1/N_phi(cuFFT 不自動做)
    int B = 256;
    int Gf = (n_fper + B - 1) / B;
    double scale = 1.0 / (double)N_phi;
    k_sph_scale_phys<<<Gf, B>>>((double*)d_fmphi_tmp, scale,
                                 2 * n_fper /* real+imag */);

    // Legendre forward:每個 (l, m) 由 N_theta quadrature 求和
    int Glm = (n_lm + B - 1) / B;
    k_sph_forward_legendre<<<Glm, B>>>(d_fmphi_tmp, d_P_table, d_weight,
                                       out_lm,
                                       N_theta, N_mphi, L_max, n_lm);
}

// ============================================================
// Inverse: spectral f̂(l, m) → spatial f(θ_j, φ_i)
//
// Step 1: 逆 Legendre — 對每個 (j, m):
//           f_m(θ_j) = Σ_{l ≥ m} f̂_l^m · P̄_l^m(x_j)
//
// Step 2: cuFFT batched C2R 沿 φ → f(θ_j, φ_i)
//         cuFFT C2R 的輸出是 unnormalized · N_phi(和 R2C 約定互補),
//         我們的 forward 已除 N_phi,這裡 inverse 不用再縮放。
// ============================================================
void SphTransform::spectral_to_spatial(const cufftDoubleComplex* in_lm,
                                       double* out_phys) {
    // Step 1: 逆 Legendre 重建 f_m(θ_j)
    int B = 256;
    int Gf = (n_fper + B - 1) / B;
    k_sph_inverse_legendre<<<Gf, B>>>(in_lm, d_P_table, d_fmphi_tmp,
                                      N_theta, N_mphi, L_max);

    // Step 2: C2R 沿 φ
    CUFFT_CHECK(cufftExecZ2D(plan_c2r, d_fmphi_tmp, out_phys));
    // cufftExecZ2D 的 output 已是物理值(不再額外縮放),
    // 因為 R2C forward 我們顯式除了 N_phi。
}
