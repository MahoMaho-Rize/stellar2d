// sph2d_spectral 的 device kernels 與 sph_transforms 的 Legendre transforms。
//
// 球諧指標 idx_lm = l*(l+1)/2 + m,只存 m ≥ 0。
// Fourier-per-latitude 複數 array 大小 N_theta × N_mphi,idx = j * N_mphi + m。
// 物理 array 大小 N_theta × N_phi,idx = j * N_phi + i。

#include <cuda_runtime.h>
#include <cufft.h>
#include <curand_kernel.h>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================
// Forward Legendre transform (sph_transforms 用):
//   對每個 (l, m),f̂_l^m = Σ_j w_j · P̄_l^m(x_j) · f_m(θ_j)
// 一個 thread 處理一個 (l, m)。
// ============================================================
__global__ void k_sph_forward_legendre(const cufftDoubleComplex* fm_theta,
                                       const double* P_table,
                                       const double* weight,
                                       cufftDoubleComplex* out_lm,
                                       int N_theta, int N_mphi,
                                       int L_max, int n_lm) {
    int idx_lm = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_lm >= n_lm) return;

    // 從 idx_lm 反求 (l, m)
    // l(l+1)/2 + m = idx  →  l = floor((√(1+8·idx) - 1)/2)
    int l = (int)((sqrt(1.0 + 8.0 * idx_lm) - 1.0) * 0.5);
    while ((l + 1) * (l + 2) / 2 <= idx_lm) l++;
    while (l * (l + 1) / 2 > idx_lm) l--;
    int m = idx_lm - l * (l + 1) / 2;
    if (m > l || m >= N_mphi) {
        out_lm[idx_lm].x = 0.0;
        out_lm[idx_lm].y = 0.0;
        return;
    }

    double acc_r = 0.0, acc_i = 0.0;
    // 沿 θ 做 quadrature,P_table[j, idx_lm] 已乘 N_l^m
    for (int j = 0; j < N_theta; ++j) {
        double P = P_table[(size_t)j * n_lm + idx_lm];
        double w = weight[j];
        cufftDoubleComplex fm = fm_theta[j * N_mphi + m];
        acc_r += w * P * fm.x;
        acc_i += w * P * fm.y;
    }
    // 歸一:Σ w_j = 2,對應 ∫_{-1}^1 dx;φ 已除過 N_phi。
    // 為了讓 forward/backward 是 exact inverse,我們讓 forward 輸出
    // 「係數」定義為 f̂_l^m 使得 Σ f̂·Y = f(θ,φ)。
    // 詳細因子由 P̄ 的 normalization 承擔,此處不再縮放。
    // 但對 m = 0:cuFFT R2C 產生 f_0(θ) = real,
    // 對 m > 0:cuFFT 產生 f_m(θ) 含 full 複數係數(not × 2)。
    // 我們 roundtrip 時由 inverse kernel 用 m=0 不倍 / m>0 倍 2·Re 處理。
    out_lm[idx_lm].x = 2.0 * M_PI * acc_r;
    out_lm[idx_lm].y = 2.0 * M_PI * acc_i;
}

// ============================================================
// Inverse Legendre transform:
//   對每個 (j, m),f_m(θ_j) = Σ_{l ≥ m} f̂_l^m · P̄_l^m(x_j)
// 一個 thread 處理一個 (j, m)。對 m ≥ N_mphi 的位置直接寫 0。
// ============================================================
__global__ void k_sph_inverse_legendre(const cufftDoubleComplex* in_lm,
                                       const double* P_table,
                                       cufftDoubleComplex* out_fm_theta,
                                       int N_theta, int N_mphi,
                                       int L_max) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N_theta * N_mphi;
    if (gid >= total) return;
    int j = gid / N_mphi;
    int m = gid - j * N_mphi;

    double acc_r = 0.0, acc_i = 0.0;
    if (m <= L_max) {
        int n_lm = (L_max + 1) * (L_max + 2) / 2;
        for (int l = m; l <= L_max; ++l) {
            int idx_lm = l * (l + 1) / 2 + m;
            double P = P_table[(size_t)j * n_lm + idx_lm];
            cufftDoubleComplex c = in_lm[idx_lm];
            acc_r += P * c.x;
            acc_i += P * c.y;
        }
        // forward 的 2π 因子在 inverse 端取消(避免重複縮放 Jacobian)
        double inv = 1.0 / (2.0 * M_PI);
        acc_r *= inv;
        acc_i *= inv;
    }
    out_fm_theta[gid].x = acc_r;
    out_fm_theta[gid].y = acc_i;
}

__global__ void k_sph_scale_phys(double* f, double scale, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    f[gid] *= scale;
}

// ============================================================
// sph2d 求解器的 kernels 從此開始
// ============================================================
//
// 狀態:ζ (渦度) 的球諧係數 ζ̂_l^m,大小 n_lm 複數。
//
// 流函數:  ψ̂_l^m = -R² / (l(l+1)) · ζ̂_l^m,對 l=0 設 0(均勻流不演化)
// 速度場(物理):
//   u_θ(θ,φ) = -(1/R) · (1/sin θ) · ∂ψ/∂φ
//   u_φ(θ,φ) =  (1/R) · ∂ψ/∂θ
// 對於 Jacobian u·∇ζ 我們在物理空間算。
//
// 絕對渦度:Z = ζ + 2Ω·cos θ。演化 ∂_t Z + u·∇Z = 0(+ ν∇²ζ + forcing + drag)
// 等價於 ∂_t ζ = -u·∇ζ - u·∇(2Ω cos θ) = -u·∇ζ + (2Ω/R²)·∂_φ ψ + dissipation
//         (用 u_θ·(-∂_θ(2Ω cosθ))/R = u_θ·(2Ω sinθ)/R;
//          u_θ sinθ = -(1/R)·∂_φ ψ,最終 = (2Ω/R²)·∂_φ ψ)
//
// IFRK3:對每個 (l,m) 獨立,exp(-(ν·l(l+1)/R²·[+hyper] + α)·dt·expo)。
//   hyper_p 時 l(l+1)/R² → (l(l+1)/R²)^p。
// ============================================================

// ζ̂ → (u_θ, u_φ) 譜空間的中間量:
//   ψ̂_l^m = R²/(l(l+1)) · ζ̂_l^m    (l=0 時 = 0)
//   (u_θ)_l^m 的計算需要 sin θ 的乘法,最方便是:
//     (u_θ sin θ)^_l^m = -(im/R) · ψ̂_l^m
//     (u_φ)_l^m = (1/R) · (dψ/dθ)_l^m   (dθ 在球諧空間有遞推式,略複雜)
// 為了盡量簡化實作,我們用更直接的路徑:
//   Step A: 譜空間求 ψ̂,乘以 im 得到 (∂ψ/∂φ)^;再逆變換到物理得 (∂ψ/∂φ)(θ,φ)
//   Step B: u_θ sin θ = -(1/R)·(∂ψ/∂φ);物理空間除以 sin θ(極軸附近要小心,
//           用 Gauss-Legendre 節點恰好 sin θ_j > 0)
//   Step C: u_φ 的計算我們繞開顯式 dψ/dθ,改用 Helmholtz 分解:
//             (u_θ sin θ, u_φ) 是從 ψ 的切向 gradient 2 個分量。
//           更簡潔的替代:用**流函數場本身的物理值**和渦度物理值做 Jacobian。
//           J(ψ, ζ) = (1/(R²·sin θ)) · (∂_θ ψ · ∂_φ ζ - ∂_φ ψ · ∂_θ ζ)
//           我們只算 4 個物理 partial derivatives,全用譜方法逐一求。
//
// 實作策略(簡潔優先,性能後續優化):
//   1. 從 ζ̂ 算 ψ̂(乘 R²/l(l+1))
//   2. 算 4 個標量物理場:ψ(θ,φ), ζ(θ,φ), ∂_φ ψ, ∂_φ ζ
//      其中 ∂_φ 在譜空間就是 iM 乘 — 不需要 θ 導數
//   3. ∂_θ 項 (∂_θ ψ, ∂_θ ζ) 採有限差分 in physical:
//      θ 方向用 central diff on Gauss-Legendre grid(非均勻,但我們只取一階精度的混合方案)
//      — 代價:Jacobian 項不再全譜精度,θ 方向降為 2 階。
//      這對 Rossby 波色散測試可接受(色散線性,只要譜 dispersion 正確);
//      對湍流 cascade 是次優但實作最快。
//
// 如果效果不夠:替代方案是用向量球諧 (vector spherical harmonics) 對速度場展開,
// 完全保持譜精度。此為 roadmap Phase A 的 perf iteration 目標。
// ============================================================

// ---- ζ̂ → ψ̂(球諧對角 Laplacian 逆) ----
//   ψ̂_l^m = R²/(l(l+1)) · ζ̂_l^m;  l=0 設 0。
__global__ void k_sph2d_vort_to_psi(const cufftDoubleComplex* zeta_hat,
                                    cufftDoubleComplex* psi_hat,
                                    int n_lm, double R) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_lm) return;
    // recover l from triangular index
    int l = (int)((sqrt(1.0 + 8.0 * idx) - 1.0) * 0.5);
    while ((l + 1) * (l + 2) / 2 <= idx) l++;
    while (l * (l + 1) / 2 > idx) l--;
    double denom = (double)l * (l + 1);
    if (denom < 0.5) {
        psi_hat[idx].x = 0.0;
        psi_hat[idx].y = 0.0;
    } else {
        double s = R * R / denom;
        psi_hat[idx].x = s * zeta_hat[idx].x;
        psi_hat[idx].y = s * zeta_hat[idx].y;
    }
}

// ---- 譜空間 ∂_φ:乘 im(im = i·m)----
__global__ void k_sph2d_dphi(const cufftDoubleComplex* in_lm,
                             cufftDoubleComplex* out_lm,
                             int n_lm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_lm) return;
    int l = (int)((sqrt(1.0 + 8.0 * idx) - 1.0) * 0.5);
    while ((l + 1) * (l + 2) / 2 <= idx) l++;
    while (l * (l + 1) / 2 > idx) l--;
    int m = idx - l * (l + 1) / 2;
    cufftDoubleComplex c = in_lm[idx];
    // ∂_φ (f · e^{imφ}) = im · f · e^{imφ}
    out_lm[idx].x = -(double)m * c.y;
    out_lm[idx].y =  (double)m * c.x;
}

// ---- Truncate spectral mode to zero for l > L_max 或 l=0 (DC for vort)----
__global__ void k_sph2d_zero_mean(cufftDoubleComplex* zeta_hat) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        zeta_hat[0].x = 0.0;
        zeta_hat[0].y = 0.0;
    }
}

// ============================================================
// 物理空間 Jacobian + β (Coriolis 球面貢獻):
//   rhs_phys(θ, φ) = -u·∇ζ - (2Ω/R²)·∂_φ ψ·(-1 從右邊移到左邊的符號)
//   = -(1/(R²·sin θ)) · (∂_θ ψ · ∂_φ ζ - ∂_φ ψ · ∂_θ ζ) - (2Ω·sin θ / R) · u_θ
//
// 但我們要的是 ∂_t ζ,等於:
//   ∂_t ζ = -u·∇Z = -u·∇ζ - u·∇(2Ω cos θ)
//          = -J(ψ,ζ)/(R²·sinθ) - (-2Ω sin θ) · u_θ / R
//          = -J(ψ,ζ)/(R²·sinθ) - (2Ω sin θ / R) · u_θ
//
// 注:u_θ / R = -(1/(R² sin θ))·∂_φ ψ → 故 β-項 = (2Ω/R²) · ∂_φ ψ
//
// 最終 rhs_phys = -(1/(R²·sinθ))·(∂_θ ψ · ∂_φ ζ - ∂_φ ψ · ∂_θ ζ) + (2Ω/R²)·∂_φ ψ
//
// 輸入:ψ_phys, zeta_phys, dpsi_dphi_phys, dzeta_dphi_phys, dpsi_dtheta_phys,
//       dzeta_dtheta_phys, sin_theta (per-latitude from host), Omega, R
// 輸出:rhs_phys(會被 R2C→LT→譜)
// ============================================================
__global__ void k_sph2d_compute_rhs_phys(const double* dpsi_dphi,
                                         const double* dzeta_dphi,
                                         const double* dpsi_dtheta,
                                         const double* dzeta_dtheta,
                                         const double* sin_theta,   // N_theta
                                         double Omega, double R,
                                         double* rhs_phys,
                                         int N_theta, int N_phi) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N_theta * N_phi;
    if (gid >= total) return;
    int j = gid / N_phi;
    double sth = sin_theta[j];
    double inv_sth = (sth > 1e-12) ? 1.0 / sth : 0.0;

    double dpsi_p = dpsi_dphi[gid];
    double dpsi_t = dpsi_dtheta[gid];
    double dz_p   = dzeta_dphi[gid];
    double dz_t   = dzeta_dtheta[gid];

    double J = (dpsi_t * dz_p - dpsi_p * dz_t);     // Jacobian numerator
    double invR2 = 1.0 / (R * R);
    double rhs_J = -invR2 * inv_sth * J;
    double rhs_beta = (2.0 * Omega) * invR2 * dpsi_p;

    rhs_phys[gid] = rhs_J + rhs_beta;
}

// ============================================================
// θ 方向 central difference on Gauss-Legendre grid (非均勻)
//   df/dθ[j] ≈ (f[j+1] - f[j-1]) / (θ[j+1] - θ[j-1])
//   邊界用 one-sided 2 階 forward/backward
// 輸入:f_phys(N_theta, N_phi), theta (N_theta) — host 傳入 θ_j 值
// 注:這是 2 階精度的妥協;完全譜版需向量球諧。
// ============================================================
__global__ void k_sph2d_dtheta_fd(const double* f_phys,
                                  const double* theta,   // N_theta, θ_j ∈ (0,π)
                                  double* df_dtheta,
                                  int N_theta, int N_phi) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N_theta * N_phi;
    if (gid >= total) return;
    int j = gid / N_phi;
    int i = gid - j * N_phi;

    double df;
    if (j == 0) {
        // forward 2nd order: df ≈ (-3f[0] + 4f[1] - f[2]) / (θ[2] - θ[0])
        int id0 = 0 * N_phi + i;
        int id1 = 1 * N_phi + i;
        int id2 = 2 * N_phi + i;
        double dth = theta[2] - theta[0];
        df = (-3.0 * f_phys[id0] + 4.0 * f_phys[id1] - f_phys[id2]) / dth;
    } else if (j == N_theta - 1) {
        int id0 = (N_theta - 1) * N_phi + i;
        int id1 = (N_theta - 2) * N_phi + i;
        int id2 = (N_theta - 3) * N_phi + i;
        double dth = theta[N_theta - 1] - theta[N_theta - 3];
        df = (3.0 * f_phys[id0] - 4.0 * f_phys[id1] + f_phys[id2]) / dth;
    } else {
        int idp = (j + 1) * N_phi + i;
        int idm = (j - 1) * N_phi + i;
        double dth = theta[j + 1] - theta[j - 1];
        df = (f_phys[idp] - f_phys[idm]) / dth;
    }
    df_dtheta[gid] = df;
}

// ============================================================
// IFRK3 級更新(同 pseudo_spectral 風格),per (l,m) 獨立:
//   y_out = (a·Ea)·y_orig + (b·Eb)·y_curr + (c_dt·Eb)·rhs
//   L·dt = (ν · [l(l+1)/R²]^p + α) · dt
// ============================================================
__global__ void k_sph2d_ifrk_combine(cufftDoubleComplex* y_out,
                                     const cufftDoubleComplex* y_orig,
                                     const cufftDoubleComplex* y_curr,
                                     const cufftDoubleComplex* rhs,
                                     double a, double b, double c_dt,
                                     double expo_a, double expo_b,
                                     double nu_dt, double drag_dt,
                                     double R, int hyper_p,
                                     int n_lm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_lm) return;
    int l = (int)((sqrt(1.0 + 8.0 * idx) - 1.0) * 0.5);
    while ((l + 1) * (l + 2) / 2 <= idx) l++;
    while (l * (l + 1) / 2 > idx) l--;
    double eig = (double)l * (l + 1) / (R * R);
    double eig_p = eig;
    for (int q = 1; q < hyper_p; ++q) eig_p *= eig;
    double Ldt = nu_dt * eig_p + drag_dt;
    double Ea = (Ldt * expo_a > 700.0) ? 0.0
              : (Ldt * expo_a < -700.0) ? __longlong_as_double(0x7FF0000000000000LL)
              : exp(-Ldt * expo_a);
    double Eb = (Ldt * expo_b > 700.0) ? 0.0
              : (Ldt * expo_b < -700.0) ? __longlong_as_double(0x7FF0000000000000LL)
              : exp(-Ldt * expo_b);
    double ca = a * Ea;
    double cb = b * Eb;
    double cc = c_dt * Eb;
    double xo = ca * y_orig[idx].x + cb * y_curr[idx].x + cc * rhs[idx].x;
    double yo = ca * y_orig[idx].y + cb * y_curr[idx].y + cc * rhs[idx].y;
    if (!isfinite(xo)) xo = 0.0;
    if (!isfinite(yo)) yo = 0.0;
    y_out[idx].x = xo;
    y_out[idx].y = yo;
}

// ============================================================
// 診斷 reductions
// ============================================================
// max|v| reduction — 同時得 max(|u_θ|²+|u_φ|²)^½
// 為簡化我們用 |∇_h ψ|² / R² 的 scalar proxy:
//   |u_h|² = |∇_h ψ|² / R²
// 物理空間 |∇_h ψ|² 近似:(1/R²)·((1/sin θ · ∂_φ ψ)² + (∂_θ ψ)²)
__global__ void k_sph2d_reduce_diag(const double* dpsi_dphi,
                                    const double* dpsi_dtheta,
                                    const double* zeta_phys,
                                    const double* sin_theta,
                                    const double* weight,       // GL weight per latitude
                                    double R, double R2,
                                    double N_phi_inv,           // 1/N_phi
                                    double* out, int N_theta, int N_phi) {
    extern __shared__ double shm[];
    double* s_maxv = shm;
    double* s_maxz = shm + blockDim.x;
    double* s_KE   = shm + 2 * blockDim.x;
    double* s_Z    = shm + 3 * blockDim.x;

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    double mv = 0.0, mz = 0.0, sK = 0.0, sZ = 0.0;
    if (gid < N_theta * N_phi) {
        int j = gid / N_phi;
        double sth = sin_theta[j];
        double w = weight[j];
        double inv_sth = (sth > 1e-12) ? 1.0 / sth : 0.0;
        double up = dpsi_dphi[gid];
        double ut = dpsi_dtheta[gid];
        // u_θ_sphere = -(1/(R·sin θ)) · ∂_φ ψ ; u_φ = (1/R)·∂_θ ψ
        double u_theta = -inv_sth * up / R;
        double u_phi   =  ut / R;
        double speed2  = u_theta * u_theta + u_phi * u_phi;
        mv = sqrt(speed2);
        double z = zeta_phys[gid];
        mz = fabs(z);
        // Surface area element = R² · sin θ · dθ dφ
        // Gauss-Legendre quadrature: ∫ f·sinθ dθ = Σ w_j · f(θ_j)   (w_j 已含 sinθ,在 [-1,1] dx)
        // dφ = 2π/N_phi → 等距 φ 權重 = 2π · N_phi_inv
        // 單位面積 2π/N_phi · w_j · R²
        double area_el = R2 * w * (2.0 * M_PI * N_phi_inv);
        sK = 0.5 * speed2 * area_el;
        sZ = 0.5 * z * z * area_el;
    }
    s_maxv[tid] = mv;
    s_maxz[tid] = mz;
    s_KE[tid]   = sK;
    s_Z[tid]    = sZ;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) {
            s_maxv[tid] = fmax(s_maxv[tid], s_maxv[tid + off]);
            s_maxz[tid] = fmax(s_maxz[tid], s_maxz[tid + off]);
            s_KE[tid]  += s_KE[tid + off];
            s_Z[tid]   += s_Z[tid + off];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[4 * blockIdx.x + 0] = s_maxv[0];
        out[4 * blockIdx.x + 1] = s_maxz[0];
        out[4 * blockIdx.x + 2] = s_KE[0];
        out[4 * blockIdx.x + 3] = s_Z[0];
    }
}

// 譜空間 E(l) bin:對每個 l,E_l = Σ_{m=-l}^{l} |ψ̂_l^m|² · l(l+1)/(2R²) · (Parseval-like)
// 用 atomicAdd 把 (l, m) 的貢獻加到 E_bins[l]。m 和 -m 的共軛對已由 Hermitian 天然成對,
// 對 R2C 只存 m ≥ 0,所以 m > 0 的模要 × 2。
__global__ void k_sph2d_spectrum(const cufftDoubleComplex* zeta_hat,
                                 double* E_bins,
                                 int n_lm, double R) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_lm) return;
    int l = (int)((sqrt(1.0 + 8.0 * idx) - 1.0) * 0.5);
    while ((l + 1) * (l + 2) / 2 <= idx) l++;
    while (l * (l + 1) / 2 > idx) l--;
    int m = idx - l * (l + 1) / 2;
    if (l == 0) return;
    double denom = (double)l * (l + 1);
    cufftDoubleComplex z = zeta_hat[idx];
    // |ψ̂|² = |ζ̂|²·R⁴/(l(l+1))²  → E_l = 0.5·|ψ̂|²·l(l+1)/R² = 0.5·|ζ̂|²·R²/(l(l+1))
    double mag2 = z.x * z.x + z.y * z.y;
    double contrib = 0.5 * mag2 * R * R / denom;
    double weight = (m == 0) ? 1.0 : 2.0;
    atomicAdd(&E_bins[l], weight * contrib);
}

// ============================================================
// 物理 field 計算誤差 (err_L2) 對比 analytic,球面 weighted L²:
//   err² = ∫_{S²} (f_num - f_exact)² dΩ
// GL quadrature: Σ_j w_j · (2π/N_phi) · Σ_i (f_num - f_exact)²(θ_j, φ_i)
// analytic 傳入 host 預計算的 f_exact_phys。
// ============================================================
__global__ void k_sph2d_err_reduce(const double* f_num,
                                   const double* f_exact,
                                   const double* weight,
                                   double N_phi_inv,
                                   double* out, int N_theta, int N_phi) {
    extern __shared__ double shm_e[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    double s = 0.0;
    if (gid < N_theta * N_phi) {
        int j = gid / N_phi;
        double w = weight[j];
        double d = f_num[gid] - f_exact[gid];
        s = d * d * w * (2.0 * M_PI * N_phi_inv);
    }
    shm_e[tid] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) shm_e[tid] += shm_e[tid + off];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = shm_e[0];
}

// ============================================================
// Forcing:薄殼 stochastic white-noise on shell l ∈ [l_min, l_max]
//   Δζ̂_l^m = √dt · σ · e^{iφ_rand}
// init 時枚舉 index_lm 進 d_forcing_idx;σ 由 host 反解 ε_inj 後下載。
// ============================================================
__global__ void k_sph2d_init_curand(curandStatePhilox4_32_10_t* states,
                                    uint64_t seed, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    curand_init(seed, (unsigned long long)gid, 0, &states[gid]);
}

__global__ void k_sph2d_apply_forcing(cufftDoubleComplex* zeta_hat,
                                      const int* idx, const double* sigma,
                                      curandStatePhilox4_32_10_t* states,
                                      double sqrt_dt, int n_forcing) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_forcing) return;
    curandStatePhilox4_32_10_t st = states[gid];
    double u = curand_uniform_double(&st);
    double phi = 2.0 * M_PI * u;
    states[gid] = st;
    double c, s;
    sincos(phi, &s, &c);
    int k_idx = idx[gid];
    double amp = sigma[gid] * sqrt_dt;
    zeta_hat[k_idx].x += amp * c;
    zeta_hat[k_idx].y += amp * s;
}

// Frame pool snapshot: ω (ζ), u_theta-sinθ 替代 (∂_φ ψ), ∂_θ ψ
__global__ void k_sph2d_snapshot_frame(const double* zeta, const double* dpsi_dphi,
                                       const double* dpsi_dtheta,
                                       double* slot, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    slot[0 * n + gid] = zeta[gid];
    slot[1 * n + gid] = dpsi_dphi[gid];
    slot[2 * n + gid] = dpsi_dtheta[gid];
}
