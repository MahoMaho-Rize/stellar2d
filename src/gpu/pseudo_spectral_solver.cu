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
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
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
__global__ void k_form_rhs_hat(const cufftDoubleComplex*, const cufftDoubleComplex*,
    const double*, const double*, const double*,
    cufftDoubleComplex*, double, int, int);
__global__ void k_form_rhs_adv_only(const cufftDoubleComplex*, const double*,
    cufftDoubleComplex*, int);
__global__ void k_rk_combine(cufftDoubleComplex*, const cufftDoubleComplex*,
    const cufftDoubleComplex*, const cufftDoubleComplex*,
    double, double, double, int);
__global__ void k_ifrk_combine(cufftDoubleComplex*, const cufftDoubleComplex*,
    const cufftDoubleComplex*, const cufftDoubleComplex*,
    const double*, const double*,
    double, double, double, double, double, double,
    int, int);
__global__ void k_reduce_diag(const double*, const double*, const double*,
    double*, int, double);
__global__ void k_snapshot_frame(const double*, const double*, const double*, double*, int);

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

    // Stage E: rhs_hat = -N̂ - ν|k|² ω̂
    k_form_rhs_hat<<<Gh, B>>>(tmp, omega_hat_in, s.d_kx, s.d_ky, s.d_dealias,
                              rhs_hat_out, s.nu, s.nx, s.nh);
}

// IFRK 專用:只算 -N̂,黏性在 IFRK combine 裡解析處理。
// 輸入/scratch 用量與 compute_rhs_hat 一致 (d_u, d_v, d_dwx, d_dwy, d_Nphys,
// d_tmp_hat),不碰 d_k1_hat。
static void compute_rhs_adv_only(PseudoSpectralSolver& s,
                                 const cufftDoubleComplex* omega_hat_in,
                                 cufftDoubleComplex* rhs_hat_out) {
    int B = 256;
    int Gh = (s.ncplx + B - 1) / B;
    int Gc = (s.ncell + B - 1) / B;
    cufftDoubleComplex* tmp = s.d_tmp_hat;

    k_spec_uv<<<Gh, B>>>(omega_hat_in, s.d_kx, s.d_ky, s.d_dealias,
                         tmp, rhs_hat_out, s.nx, s.nh);
    spec_to_phys(s.plan_c2r, tmp,         s.d_u, s.ncell);
    spec_to_phys(s.plan_c2r, rhs_hat_out, s.d_v, s.ncell);

    k_spec_grad_omega<<<Gh, B>>>(omega_hat_in, s.d_kx, s.d_ky, s.d_dealias,
                                 tmp, rhs_hat_out, s.nx, s.nh);
    spec_to_phys(s.plan_c2r, tmp,         s.d_dwx, s.ncell);
    spec_to_phys(s.plan_c2r, rhs_hat_out, s.d_dwy, s.ncell);

    k_compute_convection<<<Gc, B>>>(s.d_u, s.d_v, s.d_dwx, s.d_dwy, s.d_Nphys, s.ncell);

    phys_to_spec(s.plan_r2c, s.d_Nphys, tmp);

    // rhs = -N̂  (dealiased)
    k_form_rhs_adv_only<<<Gh, B>>>(tmp, s.d_dealias, rhs_hat_out, s.ncplx);
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

    std::fprintf(stderr,
        "  PseudoSpectral init: %dx%d (nh=%d, cells=%d, cplx=%d), "
        "L=(%.3g,%.3g), ν=%.3g, cfl=%.3g, ifrk=%d\n",
        nx, ny, nh, ncell, ncplx, Lx, Ly, nu, cfl, use_ifrk ? 1 : 0);
}

void PseudoSpectralSolver::destroy() {
    if (plan_r2c) { cufftDestroy(plan_r2c); plan_r2c = 0; }
    if (plan_c2r) { cufftDestroy(plan_c2r); plan_c2r = 0; }
    auto F = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
    F((void*&)d_omega); F((void*&)d_u); F((void*&)d_v);
    F((void*&)d_dwx);   F((void*&)d_dwy); F((void*&)d_Nphys);
    F((void*&)d_omega_hat); F((void*&)d_rhs_hat);
    F((void*&)d_tmp_hat);   F((void*&)d_k1_hat);
    F((void*&)d_kx); F((void*&)d_ky); F((void*&)d_dealias);
    F((void*&)d_reduce);
    free_frame_buffer();
}

// ============================================================
// KH 初始條件
//   vx(y) = vshear · (2·band(y) - 1),  band = ½[tanh((y-y_low)/δ) - tanh((y-y_high)/δ)]
//   vy(x,y) = amp · sin(k·2π·x/Lx) · (G(y,y_low) + G(y,y_high)),  σ = 0.05 Ly
// 解析求 ω = ∂vy/∂x - ∂vx/∂y 後 FFT → ω̂。
// ============================================================
void PseudoSpectralSolver::init_kh_shear(double vshear, double amp, int k) {
    double delta   = 2.0 * dy;
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

    dt_current = 0.0;
    step_count = 0;

    std::fprintf(stderr,
        "  PseudoSpectral KH shear: |vx|=%g, amp=%g, k=%d (δ=%.3g, σ=%.3g)\n",
        vshear, amp, k, delta, sigma);
}

// ============================================================
// dt — 對流 CFL (+ 顯式粘性 CFL, 若未啟用 IFRK)
// ============================================================
static double compute_dt(PseudoSpectralSolver& s, double max_v) {
    double dt_adv = s.cfl * std::fmin(s.dx, s.dy) / std::fmax(max_v, 1e-30);
    double dt = dt_adv;
    if (!s.use_ifrk) {
        double kmax2 = (M_PI / s.dx) * (M_PI / s.dx) + (M_PI / s.dy) * (M_PI / s.dy);
        double dt_visc = s.dt_visc_factor / std::fmax(s.nu * kmax2, 1e-30);
        dt = std::fmin(dt, dt_visc);
    }
    return std::fmin(dt, s.dt_max);
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

    // y_orig ← ω̂_n,保存在 d_k1_hat
    copy_hat(d_k1_hat, d_omega_hat, ncplx);

    if (use_ifrk) {
        // ─── IFRK3: 黏性經積分因子精確處理 ───
        //   E_1 = exp(-νk²Δt),   E_½ = exp(-νk²Δt·½),   E_-½ = exp(-νk²Δt·-½)
        //   級 1: y1 = E_1·yₙ + Δt·E_1·(-N̂(yₙ))
        //   級 2: y2 = ¾·E_½·yₙ + ¼·E_-½·y1 + ¼Δt·E_-½·(-N̂(y1))
        //   級 3: y₃ = ⅓·E_1·yₙ + ⅔·E_½·y2 + ⅔Δt·E_½·(-N̂(y2))
        double nu_dt = nu * dt;

        // 級 1
        compute_rhs_adv_only(*this, d_omega_hat, d_rhs_hat);
        k_ifrk_combine<<<Gh, B>>>(d_omega_hat, d_k1_hat, d_omega_hat, d_rhs_hat,
                                  d_kx, d_ky,
                                  1.0, 0.0, dt,
                                  /*expo_a*/1.0, /*expo_b*/1.0,
                                  nu_dt, nx, nh);
        // 級 2
        compute_rhs_adv_only(*this, d_omega_hat, d_rhs_hat);
        k_ifrk_combine<<<Gh, B>>>(d_omega_hat, d_k1_hat, d_omega_hat, d_rhs_hat,
                                  d_kx, d_ky,
                                  0.75, 0.25, 0.25 * dt,
                                  /*expo_a*/0.5, /*expo_b*/-0.5,
                                  nu_dt, nx, nh);
        // 級 3
        compute_rhs_adv_only(*this, d_omega_hat, d_rhs_hat);
        k_ifrk_combine<<<Gh, B>>>(d_omega_hat, d_k1_hat, d_omega_hat, d_rhs_hat,
                                  d_kx, d_ky,
                                  1.0/3.0, 2.0/3.0, (2.0/3.0)*dt,
                                  /*expo_a*/1.0, /*expo_b*/0.5,
                                  nu_dt, nx, nh);
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

    // 同步物理場 (供輸出與下一步 dt)。
    // cuFFT Z2D 預設會破壞輸入,所以對 d_omega_hat 做 C2R 前必須先拷貝。
    copy_hat(d_tmp_hat, d_omega_hat, ncplx);
    spec_to_phys(plan_c2r, d_tmp_hat, d_omega, ncell);
    k_spec_uv<<<Gh, B>>>(d_omega_hat, d_kx, d_ky, d_dealias,
                         d_tmp_hat, d_rhs_hat, nx, nh);
    spec_to_phys(plan_c2r, d_tmp_hat, d_u, ncell);
    spec_to_phys(plan_c2r, d_rhs_hat, d_v, ncell);

    step_count++;
    return dt;
}

// ============================================================
// Diagnostics
// ============================================================
PseudoSpectralSolver::Diagnostics PseudoSpectralSolver::compute_diagnostics() {
    int B = 256;
    size_t shm = 4 * (size_t)B * sizeof(double);
    k_reduce_diag<<<reduce_blocks, B, shm>>>(d_u, d_v, d_omega, d_reduce, ncell, dx * dy);
    std::vector<double> h(4 * reduce_blocks);
    CUDA_CHECK(cudaMemcpy(h.data(), d_reduce, h.size() * sizeof(double), cudaMemcpyDeviceToHost));
    Diagnostics d{};
    for (int i = 0; i < reduce_blocks; ++i) {
        d.max_v           = std::fmax(d.max_v,     h[4 * i + 0]);
        d.max_omega       = std::fmax(d.max_omega, h[4 * i + 1]);
        d.total_KE       += h[4 * i + 2];
        d.total_enstrophy+= h[4 * i + 3];
    }
    return d;
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
    }
    if (fcsv) std::fclose(fcsv);
    std::fprintf(stderr, " done (%d total)\n", total_frames);
    frame_count = 0;
    frame_times.clear();
    frame_steps.clear();
}

void PseudoSpectralSolver::free_frame_buffer() {
    if (d_frame_pool) { cudaFree(d_frame_pool); d_frame_pool = nullptr; }
    frame_capacity = 0;
    frame_count = 0;
    frame_times.clear();
    frame_steps.clear();
}
