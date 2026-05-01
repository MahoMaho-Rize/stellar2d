// 偽譜法 2D 不可壓 NS 求解器 — orchestration。
//
// 每步流程:
//   1. 由 ω̂ 解 ψ̂ = ω̂ / |k|²,取 û = i ky ψ̂,v̂ = -i kx ψ̂;IFFT → u, v
//   2. 取 ∂ω/∂x̂ = i kx ω̂,∂ω/∂ŷ = i ky ω̂;IFFT → ∂ω/∂x, ∂ω/∂y
//   3. 物理空間計算 N(x) = u·∂ω/∂x + v·∂ω/∂y
//   4. FFT N → N̂
//   5. rhs_hat = -N̂ - ν|k|²ω̂ (套 2/3 dealias mask)
//   6. Shu-Osher RK3 更新 ω̂
//
// VRAM frame pool + binary VTK writer 風格複用 cart_ale2。

#include "pseudo_spectral_solver.cuh"
#include "fas_common.cuh"
#include <curand_kernel.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- kernel forward decls (pseudo_spectral_kernels.cu) ---
__global__ void k_init_wavenumbers(double*, double*, double*, int, int, int, double, double);
__global__ void k_spec_uv(const cufftDoubleComplex*, const double*, const double*,
    const double*, cufftDoubleComplex*, cufftDoubleComplex*, int, int);
__global__ void k_spec_grad_omega(const cufftDoubleComplex*, const double*, const double*,
    const double*, cufftDoubleComplex*, cufftDoubleComplex*, int, int);
__global__ void k_compute_convection(const double*, const double*, const double*, const double*,
    double*, int);
__global__ void k_scale_inplace(double*, double, int);
__global__ void k_apply_dealias(cufftDoubleComplex*, const double*, int);
__global__ void k_form_rhs_hat(const cufftDoubleComplex*, const cufftDoubleComplex*,
    const double*, const double*, const double*,
    cufftDoubleComplex*, double, double, int, int, int);
__global__ void k_form_rhs_adv_only(const cufftDoubleComplex*, const double*,
    cufftDoubleComplex*, int);
__global__ void k_compute_u_omega(const double*, const double*, const double*,
    double*, double*, int);
__global__ void k_form_rhs_skew(const cufftDoubleComplex*, const cufftDoubleComplex*,
    const cufftDoubleComplex*, const double*, const double*, const double*,
    cufftDoubleComplex*, int, int);
__global__ void k_form_rhs_conservative(const cufftDoubleComplex*, const cufftDoubleComplex*,
    const double*, const double*, const double*,
    cufftDoubleComplex*, int, int);
__global__ void k_rk_combine(cufftDoubleComplex*, const cufftDoubleComplex*,
    const cufftDoubleComplex*, const cufftDoubleComplex*,
    double, double, double, int);
__global__ void k_ifrk_combine(cufftDoubleComplex*, const cufftDoubleComplex*,
    const cufftDoubleComplex*, const cufftDoubleComplex*,
    const double*, const double*,
    double, double, double, double, double, double, double,
    int, int, int);
__global__ void k_reduce_diag(const double*, const double*, const double*,
    double*, int, double);
__global__ void k_reduce_k2E(const cufftDoubleComplex*, const double*, const double*,
    double*, int, int, int, double, int);
__global__ void k_reduce_tg_err(const cufftDoubleComplex*, const cufftDoubleComplex*,
    double, double*, int, int, double);
__global__ void k_clear_dc(cufftDoubleComplex*);
__global__ void k_snapshot_frame(const double*, const double*, const double*, double*, int);
__global__ void k_apply_forcing(cufftDoubleComplex*, const int*, const double*,
    const double*, const double*, double, int);
__global__ void k_init_curand_states(curandStatePhilox4_32_10_t*, uint64_t, int);
__global__ void k_apply_forcing_curand(cufftDoubleComplex*, const int*, const double*,
    curandStatePhilox4_32_10_t*, double, int);
__global__ void k_reduce_spectrum_bins(const cufftDoubleComplex*, const double*, const double*,
    double*, int, int, int, double, double, int);

#define CUFFT_CHECK(call) do {                                                   \
    cufftResult _r = (call);                                                     \
    if (_r != CUFFT_SUCCESS) {                                                   \
        std::fprintf(stderr, "cuFFT error %d at %s:%d\n", _r, __FILE__, __LINE__); \
        std::abort();                                                             \
    }                                                                             \
} while (0)

// 譜 → 物理 (C2R + 1/(nx·ny) 歸一)
static void spec_to_phys(cufftHandle plan, cufftDoubleComplex* in_hat,
                         double* out, int ncell) {
    CUFFT_CHECK(cufftExecZ2D(plan, in_hat, out));
    int B = 256, G = (ncell + B - 1) / B;
    k_scale_inplace<<<G, B>>>(out, 1.0 / (double)ncell, ncell);
}

// 物理 → 譜 (R2C)
static void phys_to_spec(cufftHandle plan, double* in, cufftDoubleComplex* out_hat) {
    CUFFT_CHECK(cufftExecD2Z(plan, in, out_hat));
}

// ============================================================
// rhs(ω̂) 計算:讀 omega_hat_in,寫 rhs_hat_out。
// 過程會覆寫 d_u, d_v, d_dwx, d_dwy, d_Nphys, d_tmp_hat。
// 不碰 d_k1_hat (由 RK3 驅動循環使用)。
// scratch 策略:用 d_tmp_hat 與 rhs_hat_out 作為兩個 hat scratch。
// ============================================================
static void compute_rhs_hat(PseudoSpectralSolver& s,
                            const cufftDoubleComplex* omega_hat_in,
                            cufftDoubleComplex* rhs_hat_out) {
    int B = 256;
    int Gh = (s.ncplx + B - 1) / B;
    int Gc = (s.ncell + B - 1) / B;
    cufftDoubleComplex* tmp = s.d_tmp_hat;

    // Stage A: û, v̂ → tmp, rhs_hat_out;IFFT → d_u, d_v
    k_spec_uv<<<Gh, B>>>(omega_hat_in, s.d_kx, s.d_ky, s.d_dealias,
                         tmp, rhs_hat_out, s.nx, s.nh);
    spec_to_phys(s.plan_c2r, tmp,          s.d_u, s.ncell);
    spec_to_phys(s.plan_c2r, rhs_hat_out,  s.d_v, s.ncell);

    // Stage B: ∂ωx̂, ∂ωŷ → tmp, rhs_hat_out;IFFT → d_dwx, d_dwy
    k_spec_grad_omega<<<Gh, B>>>(omega_hat_in, s.d_kx, s.d_ky, s.d_dealias,
                                 tmp, rhs_hat_out, s.nx, s.nh);
    spec_to_phys(s.plan_c2r, tmp,          s.d_dwx, s.ncell);
    spec_to_phys(s.plan_c2r, rhs_hat_out,  s.d_dwy, s.ncell);

    // Stage C: N = u·∂ω/∂x + v·∂ω/∂y   (物理空間)
    k_compute_convection<<<Gc, B>>>(s.d_u, s.d_v, s.d_dwx, s.d_dwy, s.d_Nphys, s.ncell);

    // Stage D: N → N̂ in tmp
    phys_to_spec(s.plan_r2c, s.d_Nphys, tmp);

    // Stage E: rhs_hat = -N̂ - (ν·|k|^(2p) + α) ω̂
    k_form_rhs_hat<<<Gh, B>>>(tmp, omega_hat_in, s.d_kx, s.d_ky, s.d_dealias,
                              rhs_hat_out, s.nu, s.drag_alpha, s.hyper_p,
                              s.nx, s.nh);
}

// IFRK 專用:只算 -N̂,黏性在 IFRK combine 裡解析處理。
// 輸入/scratch 用量與 compute_rhs_hat 一致 (d_u, d_v, d_dwx, d_dwy, d_Nphys,
// d_tmp_hat),不碰 d_k1_hat。
// 若 s.use_skew = true,採 skew-symmetric form N_S = ½(N_A + N_C);
// 否則採 advective form N_A。
static void compute_rhs_adv_only(PseudoSpectralSolver& s,
                                 const cufftDoubleComplex* omega_hat_in,
                                 cufftDoubleComplex* rhs_hat_out) {
    int B = 256;
    int Gh = (s.ncplx + B - 1) / B;
    int Gc = (s.ncell + B - 1) / B;
    cufftDoubleComplex* tmp = s.d_tmp_hat;

    // 共通: 得到 u, v, ∂ω/∂x, ∂ω/∂y, ω (物理) ---------------
    k_spec_uv<<<Gh, B>>>(omega_hat_in, s.d_kx, s.d_ky, s.d_dealias,
                         tmp, rhs_hat_out, s.nx, s.nh);
    spec_to_phys(s.plan_c2r, tmp,         s.d_u, s.ncell);
    spec_to_phys(s.plan_c2r, rhs_hat_out, s.d_v, s.ncell);

    k_spec_grad_omega<<<Gh, B>>>(omega_hat_in, s.d_kx, s.d_ky, s.d_dealias,
                                 tmp, rhs_hat_out, s.nx, s.nh);
    spec_to_phys(s.plan_c2r, tmp,         s.d_dwx, s.ncell);
    spec_to_phys(s.plan_c2r, rhs_hat_out, s.d_dwy, s.ncell);

    // Conservative (rotational) form: 只需 2 次 R2C FFT
    // N_C = ∂(uω)/∂x + ∂(vω)/∂y  (∇·u=0 下 ≡ N_A)
    if (s.use_conservative && !s.use_skew) {
        // 先把 ω 物理搞到 d_Nphys (借用為 ω scratch)
        CUDA_CHECK(cudaMemcpy(tmp, omega_hat_in,
                              s.ncplx * sizeof(cufftDoubleComplex),
                              cudaMemcpyDeviceToDevice));
        spec_to_phys(s.plan_c2r, tmp, s.d_Nphys, s.ncell);  // d_Nphys ← ω
        // uω, vω → d_dwx, d_dwy (覆寫,之後不再需要)
        k_compute_u_omega<<<Gc, B>>>(s.d_u, s.d_v, s.d_Nphys,
                                     s.d_dwx, s.d_dwy, s.ncell);
        phys_to_spec(s.plan_r2c, s.d_dwx, tmp);              // (uω)^
        phys_to_spec(s.plan_r2c, s.d_dwy, s.d_skew_hat);     // (vω)^
        k_form_rhs_conservative<<<Gh, B>>>(tmp, s.d_skew_hat,
                                           s.d_kx, s.d_ky, s.d_dealias,
                                           rhs_hat_out, s.nx, s.nh);
        return;
    }

    // N_A (物理) = u·∂ω/∂x + v·∂ω/∂y
    k_compute_convection<<<Gc, B>>>(s.d_u, s.d_v, s.d_dwx, s.d_dwy, s.d_Nphys, s.ncell);

    if (!s.use_skew) {
        // Advective: rhs = -N̂_A (dealiased)
        phys_to_spec(s.plan_r2c, s.d_Nphys, tmp);
        k_form_rhs_adv_only<<<Gh, B>>>(tmp, s.d_dealias, rhs_hat_out, s.ncplx);
        return;
    }

    // Skew-symmetric: 再計算 N_C = ∇·(uω)
    // 1. N̂_A 譜 (放 tmp)
    phys_to_spec(s.plan_r2c, s.d_Nphys, tmp);
    // tmp 現在 = N̂_A;但接下來要做 (uω)^、(vω)^,會覆寫 tmp。
    // 所以先把 N̂_A 挪到 rhs_hat_out (一會兒 k_form_rhs_skew 會再讀寫它)
    CUDA_CHECK(cudaMemcpy(rhs_hat_out, tmp,
                          s.ncplx * sizeof(cufftDoubleComplex),
                          cudaMemcpyDeviceToDevice));

    // 2. uω, vω 物理 → 覆寫 d_u, d_v (advective rhs 已算完,不再需要 u/v)
    // 但 ω 的物理空間值我們有嗎? 沒有 — d_omega 保留的是上一步的值。
    // 注意:本函式只接收 ω̂ (omega_hat_in),物理 ω 要另外算。
    // 幸好 d_dwx (即 ∂ω/∂x) 和 ∂ω/∂y 不需要,可以挪作它用?
    // 更乾淨:直接做一次 ω̂ → ω 的 IFFT 到一個 buffer。
    // 我們借用 d_Nphys 存 ω (因為 N_A 已轉成 N̂_A 存在 rhs_hat_out)。
    // 為此要 ω̂ 的一份拷貝進 tmp:
    CUDA_CHECK(cudaMemcpy(tmp, omega_hat_in,
                          s.ncplx * sizeof(cufftDoubleComplex),
                          cudaMemcpyDeviceToDevice));
    spec_to_phys(s.plan_c2r, tmp, s.d_Nphys, s.ncell);  // d_Nphys ← ω

    // uω → d_dwx, vω → d_dwy  (這兩個 array 現在不再需要)
    k_compute_u_omega<<<Gc, B>>>(s.d_u, s.d_v, s.d_Nphys,
                                 s.d_dwx, s.d_dwy, s.ncell);

    // 3. FFT uω → tmp,   FFT vω → d_tmp_hat? tmp 又被佔
    // 需要兩個 hat scratch。當前狀態:
    //   rhs_hat_out = N̂_A
    //   tmp = 可用 (ω̂ 拷貝已被 C2R 消耗)
    //   d_k1_hat = RK 循環狀態 (不可碰)
    // 方案: 用 tmp 當 (uω)^,再借 d_rhs_hat 本身 — 不,那就是 rhs_hat_out。
    // 只能分兩階段:
    //   (a) FFT uω → tmp;保留 (uω)^ 在 tmp
    //       FFT vω → ???  (無 hat scratch 可用)
    // 解法:先把 N̂_A 轉存 host? 太慢。
    // 最乾淨:再開一個持久 hat scratch d_skew_hat (solver init 時配)。
    // ——這裡採用這個方案,呼叫 s.d_skew_hat (下方補在 struct 裡)。
    phys_to_spec(s.plan_r2c, s.d_dwx, tmp);              // tmp ← (uω)^
    phys_to_spec(s.plan_r2c, s.d_dwy, s.d_skew_hat);     // skew_hat ← (vω)^

    // 4. rhs = -N̂_S = -½ (N̂_A + N̂_C),  N̂_C = i·kx·(uω)^ + i·ky·(vω)^
    k_form_rhs_skew<<<Gh, B>>>(rhs_hat_out,             // N̂_A in
                               tmp, s.d_skew_hat,       // (uω)^, (vω)^
                               s.d_kx, s.d_ky, s.d_dealias,
                               rhs_hat_out,             // out
                               s.nx, s.nh);
}

// ============================================================
// init / destroy
// ============================================================
void PseudoSpectralSolver::init(int nx_, int ny_, double Lx_, double Ly_,
                                double nu_, double cfl_) {
    nx = nx_; ny = ny_;
    nh = ny / 2 + 1;
    ncell = nx * ny;
    ncplx = nx * nh;
    Lx = Lx_; Ly = Ly_;
    dx = Lx / (double)nx;
    dy = Ly / (double)ny;
    nu = nu_;
    cfl = cfl_;

    CUDA_CHECK(cudaMalloc(&d_omega, ncell * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_u,     ncell * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v,     ncell * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dwx,   ncell * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dwy,   ncell * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Nphys, ncell * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&d_omega_hat, ncplx * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_rhs_hat,   ncplx * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_tmp_hat,   ncplx * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_k1_hat,    ncplx * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_skew_hat,  ncplx * sizeof(cufftDoubleComplex)));

    CUDA_CHECK(cudaMalloc(&d_kx,      nx * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ky,      nh * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dealias, ncplx * sizeof(double)));

    int B = 256, G = (ncplx + B - 1) / B;
    k_init_wavenumbers<<<G, B>>>(d_kx, d_ky, d_dealias, nx, ny, nh, Lx, Ly);
    CUDA_CHECK(cudaDeviceSynchronize());

    // cuFFT row-major: (n0 slow, n1 fast) = (nx, ny). 對應物理 index = ic*ny + jc。
    // R2C 輸出形狀 (nx, ny/2+1) → index = ic*nh + jc.
    CUFFT_CHECK(cufftPlan2d(&plan_r2c, nx, ny, CUFFT_D2Z));
    CUFFT_CHECK(cufftPlan2d(&plan_c2r, nx, ny, CUFFT_Z2D));

    reduce_blocks = (ncell + B - 1) / B;
    CUDA_CHECK(cudaMalloc(&d_reduce, 4 * reduce_blocks * sizeof(double)));

    // 譜 bin: 與 python compute_spectrum 一致
    //   dk = 2π/max(Lx,Ly),  nbins = min(nx,ny)/2 + 1
    nbins  = (std::min(nx, ny) / 2) + 1;
    dk_bin = 2.0 * M_PI / std::max(Lx, Ly);
    CUDA_CHECK(cudaMalloc(&d_E_bins, nbins * sizeof(double)));

    std::fprintf(stderr,
        "  PseudoSpectral init: %dx%d (nh=%d, cells=%d, cplx=%d), "
        "L=(%.3g,%.3g), ν=%.3g, cfl=%.3g, ifrk=%d, skew=%d\n",
        nx, ny, nh, ncell, ncplx, Lx, Ly, nu, cfl,
        use_ifrk ? 1 : 0, use_skew ? 1 : 0);
}

void PseudoSpectralSolver::destroy() {
    if (plan_r2c) { cufftDestroy(plan_r2c); plan_r2c = 0; }
    if (plan_c2r) { cufftDestroy(plan_c2r); plan_c2r = 0; }
    auto F = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
    F((void*&)d_omega); F((void*&)d_u); F((void*&)d_v);
    F((void*&)d_dwx);   F((void*&)d_dwy); F((void*&)d_Nphys);
    F((void*&)d_omega_hat); F((void*&)d_rhs_hat);
    F((void*&)d_tmp_hat);   F((void*&)d_k1_hat);
    F((void*&)d_skew_hat);
    F((void*&)d_kx); F((void*&)d_ky); F((void*&)d_dealias);
    F((void*&)d_reduce);
    F((void*&)d_forcing_idx);
    F((void*&)d_forcing_sigma);
    F((void*&)d_forcing_cos);
    F((void*&)d_forcing_sin);
    F((void*&)d_forcing_states);
    F((void*&)d_omega_hat_ic);
    F((void*&)d_E_bins);
    forcing_enabled = false;
    forcing_n = 0;
    has_analytic_ic = false;
    free_frame_buffer();
}

// ============================================================
// KH 初始條件
//   vx(y) = vshear · (2·band(y) - 1),  band = ½[tanh((y-y_low)/δ) - tanh((y-y_high)/δ)]
//   vy(x,y) = amp · sin(k·2π·x/Lx) · (G(y,y_low) + G(y,y_high)),  σ = 0.05 Ly
// 解析求 ω = ∂vy/∂x - ∂vx/∂y 後 FFT → ω̂。
// ============================================================
void PseudoSpectralSolver::init_kh_shear(double vshear, double amp, int k) {
    // 剪切層厚度: 用 max(8·dy, 0.02·Ly) — 較舊版 (4dy, 0.01Ly) 多倍,
    // 使 Fourier 係數 ~ sech²(k·δ) 在 dealias cut 前衰減到 <1e-10,
    // 高 k Gibbs 殘餘完全被譜截斷清除,不依賴 dealias mask 遮蔽。
    double delta   = std::fmax(8.0 * dy, 0.02 * Ly);
    double y_low   = 0.25 * Ly;
    double y_high  = 0.75 * Ly;
    double sigma   = 0.05 * Ly;
    double kx_phys = k * 2.0 * M_PI / Lx;

    std::vector<double> h_omega(ncell);
    for (int ic = 0; ic < nx; ++ic) {
        double x = (ic + 0.5) * dx;
        double cos_kx = std::cos(kx_phys * x);
        for (int jc = 0; jc < ny; ++jc) {
            double y = (jc + 0.5) * dy;
            double sech1 = 1.0 / std::cosh((y - y_low)  / delta);
            double sech2 = 1.0 / std::cosh((y - y_high) / delta);
            double dvxdy = vshear * (sech1 * sech1 - sech2 * sech2) / delta;
            double G1 = std::exp(-(y - y_low)  * (y - y_low)  / (sigma * sigma));
            double G2 = std::exp(-(y - y_high) * (y - y_high) / (sigma * sigma));
            double dvydx = amp * kx_phys * cos_kx * (G1 + G2);
            h_omega[ic * ny + jc] = dvydx - dvxdy;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_omega, h_omega.data(),
                          ncell * sizeof(double), cudaMemcpyHostToDevice));
    phys_to_spec(plan_r2c, d_omega, d_omega_hat);

    // 套 2/3 圓形 dealias mask (雙保險 — δ 加大後 Gibbs 已經被譜衰減壓住)
    int B = 256;
    int Gh = (ncplx + B - 1) / B;
    k_apply_dealias<<<Gh, B>>>(d_omega_hat, d_dealias, ncplx);
    k_clear_dc<<<1, 1>>>(d_omega_hat);

    dt_current = 0.0;
    step_count = 0;

    std::fprintf(stderr,
        "  PseudoSpectral KH shear: |vx|=%g, amp=%g, k=%d (δ=%.3g, σ=%.3g)\n",
        vshear, amp, k, delta, sigma);
}

// ============================================================
// 零場 IC (給 forced turbulence 用)
// ============================================================
void PseudoSpectralSolver::init_zero() {
    CUDA_CHECK(cudaMemset(d_omega, 0, ncell * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_omega_hat, 0, ncplx * sizeof(cufftDoubleComplex)));
    dt_current = 0.0;
    step_count = 0;
    std::fprintf(stderr, "  PseudoSpectral zero IC\n");
}

// ============================================================
// Taylor-Green vortex:純擴散解析解驗證用 IC
//   ω(x,y,0) = 2·k_phys·cos(kx·x)·cos(ky·y)
//   kx = ky = k·2π/L (假設 Lx=Ly=L;異向時按各自 2π/L 構造)
//   對流項 u·∇ω 在此幾何下嚴格為 0 → ω(t) = ω(0)·exp(-2ν·k_phys²·t)
// ============================================================
void PseudoSpectralSolver::init_taylor_green(int k) {
    double kx_phys = k * 2.0 * M_PI / Lx;
    double ky_phys = k * 2.0 * M_PI / Ly;

    std::vector<double> h_omega(ncell);
    for (int ic = 0; ic < nx; ++ic) {
        double x = (ic + 0.5) * dx;
        double cx = std::cos(kx_phys * x);
        for (int jc = 0; jc < ny; ++jc) {
            double y = (jc + 0.5) * dy;
            h_omega[ic * ny + jc] = 2.0 * kx_phys * cx * std::cos(ky_phys * y);
        }
    }
    CUDA_CHECK(cudaMemcpy(d_omega, h_omega.data(),
                          ncell * sizeof(double), cudaMemcpyHostToDevice));
    phys_to_spec(plan_r2c, d_omega, d_omega_hat);

    // 保存 IC 譜,用於 err_L2 解析解比對
    if (!d_omega_hat_ic) {
        CUDA_CHECK(cudaMalloc(&d_omega_hat_ic,
                              ncplx * sizeof(cufftDoubleComplex)));
    }
    CUDA_CHECK(cudaMemcpy(d_omega_hat_ic, d_omega_hat,
                          ncplx * sizeof(cufftDoubleComplex),
                          cudaMemcpyDeviceToDevice));
    has_analytic_ic = true;
    analytic_k = k;

    dt_current = 0.0;
    step_count = 0;
    std::fprintf(stderr,
        "  PseudoSpectral Taylor-Green: k=%d (kx=%.3g, ky=%.3g)\n",
        k, kx_phys, ky_phys);
}

// ============================================================
// 設置 stochastic forcing — 薄殼白噪聲相位
//
// 方法學 (Boffetta-Ecke 2012, Alvelius 1999):
//   每步對 ω̂ 加 Δω̂(k) = √dt · σ · e^{iφ}   (φ 每步重擲)
//   期望注入功率:  E[|Δω̂|²] = dt · σ²
//   對應動能注入率(2D 不可壓,Parseval, R2C Hermitian weight w(k)):
//     ε_inj = (Lx·Ly)/(dt·N⁴)·Σ_shell w(k)·|Δω̂|²/(2·|k|²)
//           = (Lx·Ly·σ²)/(N⁴) · Σ_shell w(k)/(2·|k|²)
//   其中 N² = nx·ny, w(k)=2 (一般模), w(k)=1 (jc=0 或 jc=nh-1 Nyquist)。
//   反解 σ:  σ = √( ε_inj·N⁴ / (Lx·Ly · Σ_shell w/(2·k²)) )
//
// shell: |k_mode|² ∈ [(kf-dk)², (kf+dk)²]  其中 k_mode = (shift(ic), jc)
// 為避免 DC / jc=0 的 Hermitian 自共軛問題,排除 jc=0 (Hermitian 被成對動會破壞實值性)。
// ============================================================
void PseudoSpectralSolver::init_forcing(int kf, int dk, double eps, uint64_t seed) {
    forcing_enabled = true;
    forcing_eps     = eps;
    forcing_kf      = kf;
    forcing_dk      = dk;
    forcing_seed    = seed;

    // host 端枚舉 shell 模 (排除 jc=0 避開 Hermitian 自共軛)
    // 各向異性域 (Lx≠Ly): shell 以物理 |k|² 判斷,而非 mode²
    std::vector<int>    h_idx;
    std::vector<double> h_kphys2;  // 物理 |k|²(2π 單位)
    double kx_unit = 2.0 * M_PI / Lx;
    double ky_unit = 2.0 * M_PI / Ly;
    // shell 中心物理 k (以 mode kf 為基準,取較短邊的 L 做尺度參考)
    double L_ref = std::min(Lx, Ly);
    double kc_phys = (2.0 * M_PI / L_ref) * (double)kf;
    double kw_phys = (2.0 * M_PI / L_ref) * (double)dk;
    double klo_phys2 = (kc_phys - kw_phys) * (kc_phys - kw_phys);
    double khi_phys2 = (kc_phys + kw_phys) * (kc_phys + kw_phys);

    for (int ic = 0; ic < nx; ++ic) {
        int i_shift = (ic <= nx / 2) ? ic : ic - nx;
        double kxv = (double)i_shift * kx_unit;
        for (int jc = 1; jc < nh; ++jc) {   // 跳過 jc=0 (Hermitian 自共軛帶)
            if ((ny % 2 == 0) && (jc == nh - 1)) continue;  // 跳過 Nyquist
            double kyv = (double)jc * ky_unit;
            double k_phys2 = kxv * kxv + kyv * kyv;
            if (k_phys2 >= klo_phys2 && k_phys2 <= khi_phys2) {
                h_idx.push_back(ic * nh + jc);
                h_kphys2.push_back(k_phys2);
            }
        }
    }
    forcing_n = (int)h_idx.size();
    if (forcing_n == 0) {
        std::fprintf(stderr,
            "  PS forcing: WARNING shell k_f=%d±%d empty (nx=%d,ny=%d,Lx=%g,Ly=%g)\n",
            kf, dk, nx, ny, Lx, Ly);
        forcing_enabled = false;
        return;
    }

    // ε_inj = (Lx·Ly / N⁴)·Σ_shell w(k)·σ²/(2·k²_phys)
    //   w(k)=2 (jc≥1 且非 Nyquist,皆已滿足),σ per-mode 同值 (isotropic shell)
    // 反解: σ² = ε·N⁴ / (Lx·Ly·Σ 1/k²_phys)
    double Nsum = 0.0;
    for (int i = 0; i < forcing_n; ++i) {
        Nsum += 1.0 / h_kphys2[i];
    }
    double sigma2 = forcing_eps * (double)ncell * (double)ncell
                    / (Lx * Ly * Nsum);
    double sigma_val = std::sqrt(sigma2);

    std::vector<double> h_sigma(forcing_n, sigma_val);

    CUDA_CHECK(cudaMalloc(&d_forcing_idx,   forcing_n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_forcing_sigma, forcing_n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_forcing_idx, h_idx.data(),
                          forcing_n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_forcing_sigma, h_sigma.data(),
                          forcing_n * sizeof(double), cudaMemcpyHostToDevice));

    if (forcing_use_curand) {
        CUDA_CHECK(cudaMalloc(&d_forcing_states,
                              forcing_n * sizeof(curandStatePhilox4_32_10_t)));
        int B = 256, G = (forcing_n + B - 1) / B;
        k_init_curand_states<<<G, B>>>(d_forcing_states, seed, forcing_n);
    } else {
        CUDA_CHECK(cudaMalloc(&d_forcing_cos, forcing_n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_forcing_sin, forcing_n * sizeof(double)));
    }

    std::fprintf(stderr,
        "  PS forcing: k_f=%d±%d (%d modes, %s), ε_inj=%.3e, σ=%.3e, seed=0x%lx\n",
        kf, dk, forcing_n,
        forcing_use_curand ? "cuRAND" : "host-mt19937",
        eps, sigma_val, (unsigned long)seed);
}

// 每步呼叫: cuRAND device-side 或 host mt19937 回退
void PseudoSpectralSolver::apply_forcing(double dt) {
    if (!forcing_enabled || forcing_n == 0 || dt <= 0.0) return;
    int B = 256, G = (forcing_n + B - 1) / B;
    double sqrt_dt = std::sqrt(dt);
    if (forcing_use_curand) {
        k_apply_forcing_curand<<<G, B>>>(d_omega_hat, d_forcing_idx,
                                         d_forcing_sigma, d_forcing_states,
                                         sqrt_dt, forcing_n);
        return;
    }
    static thread_local std::mt19937_64 rng(forcing_seed);
    static thread_local std::uniform_real_distribution<double> U(0.0, 2.0 * M_PI);
    std::vector<double> h_cos(forcing_n), h_sin(forcing_n);
    for (int i = 0; i < forcing_n; ++i) {
        double phi = U(rng);
        h_cos[i] = std::cos(phi);
        h_sin[i] = std::sin(phi);
    }
    CUDA_CHECK(cudaMemcpy(d_forcing_cos, h_cos.data(),
                          forcing_n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_forcing_sin, h_sin.data(),
                          forcing_n * sizeof(double), cudaMemcpyHostToDevice));
    k_apply_forcing<<<G, B>>>(d_omega_hat, d_forcing_idx, d_forcing_sigma,
                              d_forcing_cos, d_forcing_sin,
                              sqrt_dt, forcing_n);
}

// ============================================================
// dt — 對流 CFL (+ 顯式粘性 CFL, 若未啟用 IFRK)
// ============================================================
static double compute_dt(PseudoSpectralSolver& s, double max_v) {
    double dt_adv = s.cfl * std::fmin(s.dx, s.dy) / std::fmax(max_v, 1e-30);
    double dt = dt_adv;
    if (!s.use_ifrk) {
        // 顯式 hyperviscosity: |k|^(2p) 的 CFL 要跟著 p
        double kmax = std::sqrt((M_PI / s.dx) * (M_PI / s.dx)
                                + (M_PI / s.dy) * (M_PI / s.dy));
        double kmax2p = 1.0;
        for (int q = 0; q < s.hyper_p; ++q) kmax2p *= (kmax * kmax);
        double dt_visc = s.dt_visc_factor / std::fmax(s.nu * kmax2p, 1e-30);
        dt = std::fmin(dt, dt_visc);
    }
    dt = std::fmin(dt, s.dt_max);
    // PI smoother (Söderlind 2003 H211b 簡化):dt ≤ dt_prev·(1 + β),β = 0.1
    // 僅在啟用且 dt_prev 有效時生效,打住 max|v| 突變導致 CFL overshoot 的 dt 抖動。
    if (s.use_pi_dt && s.dt_prev > 0.0) {
        double dt_up = s.dt_prev * 1.10;    // 一步最多放大 10%
        double dt_dn = s.dt_prev * 0.50;    // 一步最多縮到 50%
        dt = std::fmax(std::fmin(dt, dt_up), std::fmin(dt_dn, dt));
    }
    return std::fmax(dt, s.dt_min);
}

static inline void copy_hat(cufftDoubleComplex* dst, const cufftDoubleComplex* src, int n) {
    CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice));
}

// ============================================================
// Shu-Osher RK3 在譜空間推進 ω̂。
// ============================================================
double PseudoSpectralSolver::step() {
    int B = 256;
    int Gh = (ncplx + B - 1) / B;

    // 第一步時 d_u/d_v 尚未有意義,跑一次 u, v 更新供 dt 估計與後續複用
    if (step_count == 0) {
        if (use_ifrk) compute_rhs_adv_only(*this, d_omega_hat, d_rhs_hat);
        else          compute_rhs_hat(*this, d_omega_hat, d_rhs_hat);
    }

    // max|v| reduction
    size_t shm = 4 * (size_t)B * sizeof(double);
    k_reduce_diag<<<reduce_blocks, B, shm>>>(d_u, d_v, d_omega, d_reduce, ncell, dx * dy);
    std::vector<double> h_red(4 * reduce_blocks);
    CUDA_CHECK(cudaMemcpy(h_red.data(), d_reduce,
                          h_red.size() * sizeof(double), cudaMemcpyDeviceToHost));
    double max_v = 0.0;
    for (int i = 0; i < reduce_blocks; ++i) max_v = std::fmax(max_v, h_red[4 * i + 0]);
    double dt = compute_dt(*this, max_v);
    dt_current = dt;

    // ── 隨機相位強迫 (若啟用) ──
    // 加在 RK3 之前,把 √dt·σ·e^{iφ} 注入 ω̂。
    // 用 Euler-Maruyama 意義:ω̂(t+dt) 裡包含強迫貢獻,RK3 演化的是「含初值強迫」的 ω̂。
    if (forcing_enabled) apply_forcing(dt);

    // y_orig ← ω̂_n (已含本步強迫),保存在 d_k1_hat
    copy_hat(d_k1_hat, d_omega_hat, ncplx);

    if (use_ifrk) {
        // ─── IFRK3: 黏性 + drag 經積分因子精確處理 ───
        //   Ldt = (ν·|k|^(2p) + α)·dt
        //   E_1 = exp(-Ldt),   E_½ = exp(-½Ldt),   E_-½ = exp(½Ldt)
        double nu_dt   = nu * dt;
        double drag_dt = drag_alpha * dt;

        // 級 1
        compute_rhs_adv_only(*this, d_omega_hat, d_rhs_hat);
        k_ifrk_combine<<<Gh, B>>>(d_omega_hat, d_k1_hat, d_omega_hat, d_rhs_hat,
                                  d_kx, d_ky,
                                  1.0, 0.0, dt,
                                  /*expo_a*/1.0, /*expo_b*/1.0,
                                  nu_dt, drag_dt, hyper_p, nx, nh);
        // 級 2
        compute_rhs_adv_only(*this, d_omega_hat, d_rhs_hat);
        k_ifrk_combine<<<Gh, B>>>(d_omega_hat, d_k1_hat, d_omega_hat, d_rhs_hat,
                                  d_kx, d_ky,
                                  0.75, 0.25, 0.25 * dt,
                                  /*expo_a*/0.5, /*expo_b*/-0.5,
                                  nu_dt, drag_dt, hyper_p, nx, nh);
        // 級 3
        compute_rhs_adv_only(*this, d_omega_hat, d_rhs_hat);
        k_ifrk_combine<<<Gh, B>>>(d_omega_hat, d_k1_hat, d_omega_hat, d_rhs_hat,
                                  d_kx, d_ky,
                                  1.0/3.0, 2.0/3.0, (2.0/3.0)*dt,
                                  /*expo_a*/1.0, /*expo_b*/0.5,
                                  nu_dt, drag_dt, hyper_p, nx, nh);
    } else {
        // ─── 顯式 SSP-RK3 (黏性也顯式, dt 受擴散 CFL 限制) ───
        // 級 1: y1 = y + dt · rhs(y)
        compute_rhs_hat(*this, d_omega_hat, d_rhs_hat);
        k_rk_combine<<<Gh, B>>>(d_omega_hat, d_k1_hat, d_omega_hat, d_rhs_hat,
                                1.0, 0.0, dt, ncplx);

        // 級 2: y2 = ¾ y + ¼ y1 + ¼ dt · rhs(y1)
        compute_rhs_hat(*this, d_omega_hat, d_rhs_hat);
        k_rk_combine<<<Gh, B>>>(d_omega_hat, d_k1_hat, d_omega_hat, d_rhs_hat,
                                0.75, 0.25, 0.25 * dt, ncplx);

        // 級 3: y_{n+1} = ⅓ y + ⅔ y2 + ⅔ dt · rhs(y2)
        compute_rhs_hat(*this, d_omega_hat, d_rhs_hat);
        k_rk_combine<<<Gh, B>>>(d_omega_hat, d_k1_hat, d_omega_hat, d_rhs_hat,
                                1.0 / 3.0, 2.0 / 3.0, (2.0 / 3.0) * dt, ncplx);
    }

    // DC mode 強制歸零(浮點 + forcing + IC 殘影可能累積非零 spurious mean vorticity)
    k_clear_dc<<<1, 1>>>(d_omega_hat);

    // 同步物理場 (供輸出與下一步 dt)。
    // cuFFT Z2D 預設會破壞輸入,所以對 d_omega_hat 做 C2R 前必須先拷貝。
    copy_hat(d_tmp_hat, d_omega_hat, ncplx);
    spec_to_phys(plan_c2r, d_tmp_hat, d_omega, ncell);
    k_spec_uv<<<Gh, B>>>(d_omega_hat, d_kx, d_ky, d_dealias,
                         d_tmp_hat, d_rhs_hat, nx, nh);
    spec_to_phys(plan_c2r, d_tmp_hat, d_u, ncell);
    spec_to_phys(plan_c2r, d_rhs_hat, d_v, ncell);

    dt_prev = dt;
    step_count++;
    return dt;
}

// ============================================================
// Diagnostics
// ============================================================
PseudoSpectralSolver::Diagnostics PseudoSpectralSolver::compute_diagnostics(double t_eval) {
    int B = 256;
    size_t shm = 4 * (size_t)B * sizeof(double);
    k_reduce_diag<<<reduce_blocks, B, shm>>>(d_u, d_v, d_omega, d_reduce, ncell, dx * dy);
    std::vector<double> h(4 * reduce_blocks);
    CUDA_CHECK(cudaMemcpy(h.data(), d_reduce, h.size() * sizeof(double), cudaMemcpyDeviceToHost));
    Diagnostics d{};
    d.err_L2 = std::nan("");
    d.t_eval = t_eval;
    for (int i = 0; i < reduce_blocks; ++i) {
        d.max_v           = std::fmax(d.max_v,     h[4 * i + 0]);
        d.max_omega       = std::fmax(d.max_omega, h[4 * i + 1]);
        d.total_KE       += h[4 * i + 2];
        d.total_enstrophy+= h[4 * i + 3];
    }

    // ε_Ω (enstrophy 耗散率) = ν·∫ω·(-Δ)^p·ω dA
    //   譜: Σ_k weight·|ω̂|²·k^(2p)·scale ;  scale = Lx·Ly/N²
    //   p=1 退化為 ν·∫|∇ω|² dA。
    int Gk = (ncplx + B - 1) / B;
    size_t shm2 = (size_t)B * sizeof(double);
    double scale = (Lx * Ly) / ((double)ncell * (double)ncell);
    k_reduce_k2E<<<Gk, B, shm2>>>(d_omega_hat, d_kx, d_ky,
                                   d_reduce, ncplx, nx, nh, scale, hyper_p);
    std::vector<double> h2(Gk);
    CUDA_CHECK(cudaMemcpy(h2.data(), d_reduce, Gk * sizeof(double), cudaMemcpyDeviceToHost));
    double sum_k2pE = 0.0;
    for (int i = 0; i < Gk; ++i) sum_k2pE += h2[i];
    d.eps_enstrophy = nu * sum_k2pE;

    // ε_KE (動能耗散率):
    //   對 p=1 (Laplacian): dKE/dt|_visc = -2ν·Ω;加 drag → -2α·KE
    //   對 p>1: dKE/dt|_hyper = -2ν·Σ k^(2p-2)·|ψ̂_k|²·k² = -2ν·Σ k^(2p-2)·|ω̂|²/k²·k²
    //         = -2ν·Σ k^(2p-2)·|ω̂|² (+ weight/scale)
    //   更直接:以 ω 方程 -(ν·|k|^(2p)+α)·ω 推動,對 KE 的影響是:
    //     dKE/dt = -2·Σ (ν·k^(2p-2) + α·k^{-2})·|ω̂|²·... (Parseval)
    //   當前實作:p=1 用 2ν·Ω;p>1 時以譜積分補上 hyperviscosity 貢獻。
    if (hyper_p == 1) {
        d.eps_KE = 2.0 * nu * d.total_enstrophy + 2.0 * drag_alpha * d.total_KE;
    } else {
        // eps_KE(hyper) = 2ν·Σ k^(2p-2)·|ω̂|²·scale_weight
        k_reduce_k2E<<<Gk, B, shm2>>>(d_omega_hat, d_kx, d_ky,
                                      d_reduce, ncplx, nx, nh, scale, hyper_p - 1);
        CUDA_CHECK(cudaMemcpy(h2.data(), d_reduce, Gk * sizeof(double),
                              cudaMemcpyDeviceToHost));
        double sum_k2pm2 = 0.0;
        for (int i = 0; i < Gk; ++i) sum_k2pm2 += h2[i];
        d.eps_KE = 2.0 * nu * sum_k2pm2 + 2.0 * drag_alpha * d.total_KE;
    }

    // 交叉檢查(p=1 Laplacian):物理空間 Ω 對比譜空間 Ω。
    //   Ω_phys = ½·Σ_cell ω²·dA = total_enstrophy
    //   Ω_spec = ½·Σ_k w(k)·|ω̂|²·(Lx·Ly/N²)   (p_eff = 0)
    //   兩者理論上完全相等 (Parseval)。發散則表示 FFT plan / dealias / 介面錯誤。
    if (hyper_p == 1) {
        k_reduce_k2E<<<Gk, B, shm2>>>(d_omega_hat, d_kx, d_ky,
                                      d_reduce, ncplx, nx, nh, scale, 0);
        CUDA_CHECK(cudaMemcpy(h2.data(), d_reduce, Gk * sizeof(double),
                              cudaMemcpyDeviceToHost));
        double sum_E = 0.0;
        for (int i = 0; i < Gk; ++i) sum_E += h2[i];
        double Omega_spec = 0.5 * sum_E;
        d.eps_KE_spec = 2.0 * nu * Omega_spec + 2.0 * drag_alpha * d.total_KE;
        double ref = std::fmax(std::fabs(d.eps_KE), 1e-300);
        double rel = std::fabs(d.eps_KE_spec - d.eps_KE) / ref;
        if (rel > 1e-6 && ref > 1e-20) {
            std::fprintf(stderr,
                "  PS diag WARN: eps_KE(phys)=%.6e vs spec=%.6e rel=%.3e "
                "(Ω_phys=%.6e Ω_spec=%.6e)\n",
                d.eps_KE, d.eps_KE_spec, rel,
                d.total_enstrophy, Omega_spec);
        }
    } else {
        d.eps_KE_spec = d.eps_KE;   // hyper_p>1 時 spec 交叉檢查不簡單,略過
    }

    // Taylor-Green 解析解誤差 (僅在 init_taylor_green 啟用後有意義)
    if (has_analytic_ic && d_omega_hat_ic) {
        // 解析解衰減因子:ω(t) = ω(0)·exp(-2ν·k_phys²·t)
        double kx_phys = analytic_k * 2.0 * M_PI / Lx;
        double ky_phys = analytic_k * 2.0 * M_PI / Ly;
        double kf2 = kx_phys * kx_phys + ky_phys * ky_phys;
        // hyperviscosity: exp(-2·ν·k²p·t) — 但 TG ω̂ 只在 4 個 mode,手動展開
        double k2p = kf2;
        for (int q = 1; q < hyper_p; ++q) k2p *= kf2;
        double decay = std::exp(-(nu * k2p + drag_alpha) * t_eval);  // 連續解析

        k_reduce_tg_err<<<Gk, B, shm2>>>(d_omega_hat, d_omega_hat_ic, decay,
                                          d_reduce, ncplx, nh, scale);
        CUDA_CHECK(cudaMemcpy(h2.data(), d_reduce, Gk * sizeof(double),
                              cudaMemcpyDeviceToHost));
        double err2 = 0.0;
        for (int i = 0; i < Gk; ++i) err2 += h2[i];
        d.err_L2 = std::sqrt(std::fmax(err2, 0.0));
    }

    return d;
}

// ============================================================
// GPU ring-integrated energy spectrum
//
// 對照 scripts/spectrum_pseudo_spectral.py:compute_spectrum
//   E_mode = (dA/N)·½·(|û|²+|v̂|²) = (Lx·Ly/N⁴)·½·|ω̂|²/k²
//   bin    = round(|k|/dk),  dk = 2π/max(Lx,Ly)
//   E(k)   = Σ_mode E_mode_in_bin / dk   (密度)
// 回傳向量 out 大小 = nbins,為譜密度。
// ============================================================
void PseudoSpectralSolver::compute_spectrum_bins(std::vector<double>& out) {
    CUDA_CHECK(cudaMemset(d_E_bins, 0, nbins * sizeof(double)));
    int B = 256;
    int Gh = (ncplx + B - 1) / B;
    double mode_scale = (Lx * Ly) / ((double)ncell * (double)ncell) * 0.5;
    int ny_even = (ny % 2 == 0) ? 1 : 0;
    k_reduce_spectrum_bins<<<Gh, B>>>(d_omega_hat, d_kx, d_ky, d_E_bins,
                                       nx, nh, nbins,
                                       1.0 / dk_bin, mode_scale, ny_even);
    out.resize(nbins);
    CUDA_CHECK(cudaMemcpy(out.data(), d_E_bins,
                          nbins * sizeof(double), cudaMemcpyDeviceToHost));
    // 轉密度:bin 內總能 / dk
    for (auto& v : out) v /= dk_bin;
}

void PseudoSpectralSolver::download_omega(std::vector<double>& h) {
    h.resize(ncell);
    CUDA_CHECK(cudaMemcpy(h.data(), d_omega, ncell * sizeof(double), cudaMemcpyDeviceToHost));
}
void PseudoSpectralSolver::download_uv(std::vector<double>& h_u, std::vector<double>& h_v) {
    h_u.resize(ncell); h_v.resize(ncell);
    CUDA_CHECK(cudaMemcpy(h_u.data(), d_u, ncell * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v.data(), d_v, ncell * sizeof(double), cudaMemcpyDeviceToHost));
}

// ============================================================
// ASCII VTK (小網格 debug 用)
// ============================================================
void PseudoSpectralSolver::write_vtk_2d(const char* filename) {
    std::vector<double> h_w, h_u, h_v;
    download_omega(h_w);
    download_uv(h_u, h_v);

    std::FILE* fp = std::fopen(filename, "w");
    if (!fp) return;
    int nnx = nx + 1, nny = ny + 1;
    std::fprintf(fp, "# vtk DataFile Version 3.0\npseudo_spectral output\nASCII\n");
    std::fprintf(fp, "DATASET STRUCTURED_GRID\nDIMENSIONS %d %d 1\n", nnx, nny);
    std::fprintf(fp, "POINTS %d double\n", nnx * nny);
    for (int jn = 0; jn < nny; ++jn) {
        double y = Ly * (double)jn / (double)(nny - 1);
        for (int in = 0; in < nnx; ++in) {
            double x = Lx * (double)in / (double)(nnx - 1);
            std::fprintf(fp, "%.10e %.10e %.10e\n", x, y, 0.0);
        }
    }
    std::fprintf(fp, "CELL_DATA %d\n", ncell);
    std::fprintf(fp, "SCALARS vorticity double 1\nLOOKUP_TABLE default\n");
    for (int jc = 0; jc < ny; ++jc)
        for (int ic = 0; ic < nx; ++ic)
            std::fprintf(fp, "%.10e\n", h_w[ic * ny + jc]);
    std::fprintf(fp, "VECTORS velocity double\n");
    for (int jc = 0; jc < ny; ++jc)
        for (int ic = 0; ic < nx; ++ic) {
            int c = ic * ny + jc;
            std::fprintf(fp, "%.10e %.10e 0.0\n", h_u[c], h_v[c]);
        }
    std::fclose(fp);
}

// ============================================================
// VRAM frame pool — 每幀 3 × ncell (ω, u, v)
// ============================================================
void PseudoSpectralSolver::alloc_frame_buffer(int headroom_mb) {
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    size_t headroom_b = (size_t)headroom_mb * 1024ull * 1024ull;
    if (free_b <= headroom_b) {
        std::fprintf(stderr,
            "  PS frame buffer: only %.2f GB free — disabling VRAM buffer\n",
            free_b / 1.0e9);
        frame_capacity = 0;
        return;
    }
    size_t pool_b = free_b - headroom_b;
    size_t per_frame_b = (size_t)ncell * 3ull * sizeof(double);
    frame_capacity = (int)(pool_b / per_frame_b);
    if (frame_capacity < 4) frame_capacity = 4;
    size_t actual_b = (size_t)frame_capacity * per_frame_b;
    if (cudaMalloc(&d_frame_pool, actual_b) != cudaSuccess) {
        frame_capacity = (int)(((size_t)(free_b * 0.5)) / per_frame_b);
        if (frame_capacity < 4) frame_capacity = 4;
        actual_b = (size_t)frame_capacity * per_frame_b;
        CUDA_CHECK(cudaMalloc(&d_frame_pool, actual_b));
    }
    frame_count = 0;
    total_frames = 0;
    frame_times.clear();
    frame_steps.clear();
    std::fprintf(stderr,
        "  PS frame buffer: %d frames × %.2f MB = %.2f GB\n",
        frame_capacity, per_frame_b / 1.0e6, actual_b / 1.0e9);
}

void PseudoSpectralSolver::capture_frame(double t, int step) {
    if (!d_frame_pool || frame_capacity == 0) return;
    if (frame_count >= frame_capacity) flush_frames_to_disk(frame_out_dir);
    int B = 256, G = (ncell + B - 1) / B;
    double* slot = d_frame_pool + (size_t)frame_count * 3ull * (size_t)ncell;
    k_snapshot_frame<<<G, B>>>(d_omega, d_u, d_v, slot, ncell);
    frame_times.push_back(t);
    frame_steps.push_back(step);

    // 順手做一份譜快照 (GPU reduce + D2H,~20 μs 量級)
    std::vector<double> E_k;
    compute_spectrum_bins(E_k);
    frame_spectra.push_back(std::move(E_k));

    frame_count++;
}

static inline double bswap8(double v) {
    union { double d; uint64_t u; } x; x.d = v;
    x.u = ((x.u & 0x00000000000000FFULL) << 56) |
          ((x.u & 0x000000000000FF00ULL) << 40) |
          ((x.u & 0x0000000000FF0000ULL) << 24) |
          ((x.u & 0x00000000FF000000ULL) <<  8) |
          ((x.u & 0x000000FF00000000ULL) >>  8) |
          ((x.u & 0x0000FF0000000000ULL) >> 24) |
          ((x.u & 0x00FF000000000000ULL) >> 40) |
          ((x.u & 0xFF00000000000000ULL) >> 56);
    return x.d;
}

// 為與 render_cart_ale.py 相容,幀內提供 cart_ale2 風格的場名:
//   density   = 1   (不可壓常數)
//   pressure  = 0
//   e_int     = ω   (主視覺通道 — 渦度)
//   velocity  = (u, v, 0)
//   mach      = |v| (不可壓無音速,此欄位顯示速度大小)
static void write_vtk_binary_frame_ps(const char* path, int nx, int ny,
                                      double Lx, double Ly,
                                      const double* omega, const double* u,
                                      const double* v) {
    std::FILE* fp = std::fopen(path, "wb");
    if (!fp) return;
    int nnx = nx + 1, nny = ny + 1;
    std::fprintf(fp, "# vtk DataFile Version 3.0\npseudo_spectral frame\nBINARY\n");
    std::fprintf(fp, "DATASET STRUCTURED_GRID\nDIMENSIONS %d %d 1\n", nnx, nny);
    std::fprintf(fp, "POINTS %d double\n", nnx * nny);

    std::vector<double> pts(3 * nnx * nny);
    size_t k = 0;
    for (int jn = 0; jn < nny; ++jn) {
        double y = Ly * (double)jn / (double)(nny - 1);
        for (int in = 0; in < nnx; ++in) {
            double x = Lx * (double)in / (double)(nnx - 1);
            pts[k++] = bswap8(x);
            pts[k++] = bswap8(y);
            pts[k++] = bswap8(0.0);
        }
    }
    std::fwrite(pts.data(), sizeof(double), pts.size(), fp);
    std::fputc('\n', fp);

    int nc = nx * ny;
    std::fprintf(fp, "CELL_DATA %d\n", nc);

    auto write_rearr = [&](const char* name, const double* arr) {
        std::fprintf(fp, "SCALARS %s double 1\nLOOKUP_TABLE default\n", name);
        std::vector<double> buf(nc);
        size_t idx = 0;
        for (int jc = 0; jc < ny; ++jc)
            for (int ic = 0; ic < nx; ++ic)
                buf[idx++] = bswap8(arr[ic * ny + jc]);
        std::fwrite(buf.data(), sizeof(double), buf.size(), fp);
        std::fputc('\n', fp);
    };
    auto write_const = [&](const char* name, double val) {
        std::fprintf(fp, "SCALARS %s double 1\nLOOKUP_TABLE default\n", name);
        std::vector<double> buf(nc, bswap8(val));
        std::fwrite(buf.data(), sizeof(double), buf.size(), fp);
        std::fputc('\n', fp);
    };

    write_const("density",  1.0);
    write_const("pressure", 0.0);
    write_rearr("e_int",    omega);   // 用 e_int 欄位承載渦度

    std::fprintf(fp, "VECTORS velocity double\n");
    std::vector<double> vbuf(3 * nc);
    {
        size_t idx = 0;
        for (int jc = 0; jc < ny; ++jc)
            for (int ic = 0; ic < nx; ++ic) {
                int c = ic * ny + jc;
                vbuf[idx++] = bswap8(u[c]);
                vbuf[idx++] = bswap8(v[c]);
                vbuf[idx++] = bswap8(0.0);
            }
        std::fwrite(vbuf.data(), sizeof(double), vbuf.size(), fp);
        std::fputc('\n', fp);
    }
    std::fprintf(fp, "SCALARS mach double 1\nLOOKUP_TABLE default\n");
    std::vector<double> mbuf(nc);
    {
        size_t idx = 0;
        for (int jc = 0; jc < ny; ++jc)
            for (int ic = 0; ic < nx; ++ic) {
                int c = ic * ny + jc;
                double sp = std::sqrt(u[c] * u[c] + v[c] * v[c]);
                mbuf[idx++] = bswap8(sp);
            }
        std::fwrite(mbuf.data(), sizeof(double), mbuf.size(), fp);
        std::fputc('\n', fp);
    }
    std::fclose(fp);
}

void PseudoSpectralSolver::flush_frames_to_disk(const std::string& run_dir) {
    if (frame_count == 0 || !d_frame_pool) return;
    frame_out_dir = run_dir;
    size_t per_frame = (size_t)ncell * 3ull;
    std::vector<double> host((size_t)frame_count * per_frame);
    CUDA_CHECK(cudaMemcpy(host.data(), d_frame_pool,
                          host.size() * sizeof(double), cudaMemcpyDeviceToHost));
    std::fprintf(stderr, "  PS flushing %d frames → %s ...",
                 frame_count, run_dir.c_str());
    std::fflush(stderr);
    char csv_path[512];
    std::snprintf(csv_path, sizeof(csv_path), "%s/frames.csv", run_dir.c_str());
    bool first_batch = (total_frames == 0);
    std::FILE* fcsv = std::fopen(csv_path, first_batch ? "w" : "a");
    if (fcsv && first_batch) std::fprintf(fcsv, "index,step,t\n");

    // spectrum.csv: header = index,step,t,<k0>,<k1>,...
    char spec_path[512];
    std::snprintf(spec_path, sizeof(spec_path), "%s/spectrum.csv", run_dir.c_str());
    std::FILE* fspec = std::fopen(spec_path, first_batch ? "w" : "a");
    if (fspec && first_batch) {
        std::fprintf(fspec, "index,step,t");
        for (int b = 0; b < nbins; ++b)
            std::fprintf(fspec, ",%.6e", b * dk_bin);
        std::fputc('\n', fspec);
    }

    int base_idx = total_frames;
    for (int f = 0; f < frame_count; ++f) {
        ++total_frames;
        const double* base = host.data() + (size_t)f * per_frame;
        const double* w = base;
        const double* u = base + ncell;
        const double* v = base + 2 * ncell;
        char path[512];
        std::snprintf(path, sizeof(path), "%s/output_%04d.vtk",
                      run_dir.c_str(), total_frames);
        write_vtk_binary_frame_ps(path, nx, ny, Lx, Ly, w, u, v);
        if (fcsv) {
            int idx = base_idx + f + 1;
            std::fprintf(fcsv, "%d,%d,%.10e\n", idx, frame_steps[f], frame_times[f]);
        }
        if (fspec && f < (int)frame_spectra.size()) {
            int idx = base_idx + f + 1;
            std::fprintf(fspec, "%d,%d,%.10e", idx, frame_steps[f], frame_times[f]);
            const auto& E = frame_spectra[f];
            for (double v : E) std::fprintf(fspec, ",%.6e", v);
            std::fputc('\n', fspec);
        }
    }
    if (fcsv)  std::fclose(fcsv);
    if (fspec) std::fclose(fspec);
    frame_spectra.clear();
    std::fprintf(stderr, " done (%d total)\n", total_frames);
    frame_count = 0;
    frame_times.clear();
    frame_steps.clear();
}

// ============================================================
// Checkpoint / Restart
//   格式:
//     magic (uint64)    = 0x50534350434B3031  "PSCPCK01"
//     nx, ny (int32 x2)
//     hyper_p (int32)
//     Lx, Ly, nu, drag_alpha (double x4)
//     step (int64), t (double)
//     ncplx (int64)
//     omega_hat (cufftDoubleComplex × ncplx,  little-endian host bytes)
// 注:RNG 狀態 (cuRAND) 不 dump;restart 會重啟一個新 seed sequence
//     (forcing 是 statistically stationary,統計量不受影響)。
// ============================================================
static constexpr uint64_t PS_CKPT_MAGIC = 0x50534350434B3031ULL;

void PseudoSpectralSolver::save_checkpoint(const std::string& path, double t) {
    std::FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) {
        std::fprintf(stderr, "  PS checkpoint: cannot open %s\n", path.c_str());
        return;
    }
    uint64_t magic = PS_CKPT_MAGIC;
    int32_t i_nx = nx, i_ny = ny, i_hp = hyper_p;
    int64_t i_step = step_count;
    int64_t i_ncplx = ncplx;
    std::fwrite(&magic,      sizeof(magic),      1, fp);
    std::fwrite(&i_nx,       sizeof(i_nx),       1, fp);
    std::fwrite(&i_ny,       sizeof(i_ny),       1, fp);
    std::fwrite(&i_hp,       sizeof(i_hp),       1, fp);
    std::fwrite(&Lx,         sizeof(Lx),         1, fp);
    std::fwrite(&Ly,         sizeof(Ly),         1, fp);
    std::fwrite(&nu,         sizeof(nu),         1, fp);
    std::fwrite(&drag_alpha, sizeof(drag_alpha), 1, fp);
    std::fwrite(&i_step,     sizeof(i_step),     1, fp);
    std::fwrite(&t,          sizeof(t),          1, fp);
    std::fwrite(&i_ncplx,    sizeof(i_ncplx),    1, fp);
    std::vector<cufftDoubleComplex> h(ncplx);
    CUDA_CHECK(cudaMemcpy(h.data(), d_omega_hat,
                          ncplx * sizeof(cufftDoubleComplex),
                          cudaMemcpyDeviceToHost));
    std::fwrite(h.data(), sizeof(cufftDoubleComplex), ncplx, fp);
    std::fclose(fp);
    std::fprintf(stderr,
        "  PS checkpoint saved: %s (step=%lld, t=%.6e)\n",
        path.c_str(), (long long)step_count, t);
}

bool PseudoSpectralSolver::load_checkpoint(const std::string& path, double& t_out) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        std::fprintf(stderr, "  PS checkpoint: cannot open %s for read\n", path.c_str());
        return false;
    }
    uint64_t magic = 0;
    int32_t i_nx = 0, i_ny = 0, i_hp = 0;
    double r_Lx = 0, r_Ly = 0, r_nu = 0, r_alpha = 0;
    int64_t i_step = 0, i_ncplx = 0;
    double  r_t = 0;
    std::fread(&magic,    sizeof(magic),    1, fp);
    if (magic != PS_CKPT_MAGIC) {
        std::fprintf(stderr, "  PS checkpoint: bad magic 0x%llx in %s\n",
                     (unsigned long long)magic, path.c_str());
        std::fclose(fp);
        return false;
    }
    std::fread(&i_nx,     sizeof(i_nx),     1, fp);
    std::fread(&i_ny,     sizeof(i_ny),     1, fp);
    std::fread(&i_hp,     sizeof(i_hp),     1, fp);
    std::fread(&r_Lx,     sizeof(r_Lx),     1, fp);
    std::fread(&r_Ly,     sizeof(r_Ly),     1, fp);
    std::fread(&r_nu,     sizeof(r_nu),     1, fp);
    std::fread(&r_alpha,  sizeof(r_alpha),  1, fp);
    std::fread(&i_step,   sizeof(i_step),   1, fp);
    std::fread(&r_t,      sizeof(r_t),      1, fp);
    std::fread(&i_ncplx,  sizeof(i_ncplx),  1, fp);
    if (i_nx != nx || i_ny != ny || i_ncplx != ncplx) {
        std::fprintf(stderr,
            "  PS checkpoint: size mismatch ckpt=%dx%d (ncplx=%lld) vs solver=%dx%d (%d)\n",
            i_nx, i_ny, (long long)i_ncplx, nx, ny, ncplx);
        std::fclose(fp);
        return false;
    }
    std::vector<cufftDoubleComplex> h(ncplx);
    size_t read_n = std::fread(h.data(), sizeof(cufftDoubleComplex), ncplx, fp);
    std::fclose(fp);
    if ((int)read_n != ncplx) {
        std::fprintf(stderr, "  PS checkpoint: truncated (%zu/%d)\n", read_n, ncplx);
        return false;
    }
    CUDA_CHECK(cudaMemcpy(d_omega_hat, h.data(),
                          ncplx * sizeof(cufftDoubleComplex),
                          cudaMemcpyHostToDevice));
    // 同步物理場
    copy_hat(d_tmp_hat, d_omega_hat, ncplx);
    spec_to_phys(plan_c2r, d_tmp_hat, d_omega, ncell);
    int B = 256, Gh = (ncplx + B - 1) / B;
    k_spec_uv<<<Gh, B>>>(d_omega_hat, d_kx, d_ky, d_dealias,
                         d_tmp_hat, d_rhs_hat, nx, nh);
    spec_to_phys(plan_c2r, d_tmp_hat, d_u, ncell);
    spec_to_phys(plan_c2r, d_rhs_hat, d_v, ncell);
    step_count = (int)i_step;
    t_out = r_t;
    std::fprintf(stderr,
        "  PS checkpoint loaded: %s (step=%lld, t=%.6e, ν_ckpt=%.3g, α_ckpt=%.3g)\n",
        path.c_str(), (long long)step_count, r_t, r_nu, r_alpha);
    return true;
}

void PseudoSpectralSolver::free_frame_buffer() {
    if (d_frame_pool) { cudaFree(d_frame_pool); d_frame_pool = nullptr; }
    frame_capacity = 0;
    frame_count = 0;
    frame_times.clear();
    frame_steps.clear();
    frame_spectra.clear();
}
