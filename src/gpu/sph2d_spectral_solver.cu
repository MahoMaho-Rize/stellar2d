// sph2d_spectral_solver.cu — 2D 薄球殼 barotropic 不可壓 NS 偽譜求解器。
//
// Pipeline per RK stage:
//   1. 從 ζ̂ 算 ψ̂ (對角 Laplacian 逆)
//   2. 逆 transform 得 ζ, ψ 物理場
//   3. 譜空間 ∂_φ ζ, ∂_φ ψ → 逆 transform 得 ∂_φ ζ, ∂_φ ψ 物理
//   4. 物理空間 θ 方向 central FD 得 ∂_θ ζ, ∂_θ ψ
//   5. 組合 Jacobian + β 項 → rhs_phys
//   6. Forward transform → rhs_hat
//   7. IFRK3 combine(線性耗散 exp 積分)

#include "sph2d_spectral_solver.cuh"
#include "fas_common.cuh"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---- kernel forward decls ----
__global__ void k_sph2d_vort_to_psi(const cufftDoubleComplex*, cufftDoubleComplex*, int, double);
__global__ void k_sph2d_dphi(const cufftDoubleComplex*, cufftDoubleComplex*, int);
__global__ void k_sph2d_zero_mean(cufftDoubleComplex*);
__global__ void k_sph2d_compute_rhs_phys(const double*, const double*, const double*,
    const double*, const double*, double, double, double*, int, int);
__global__ void k_sph2d_dtheta_fd(const double*, const double*, double*, int, int);
__global__ void k_sph2d_ifrk_combine(cufftDoubleComplex*, const cufftDoubleComplex*,
    const cufftDoubleComplex*, const cufftDoubleComplex*,
    double, double, double, double, double, double, double,
    double, int, int);
__global__ void k_sph2d_reduce_diag(const double*, const double*, const double*,
    const double*, const double*, double, double, double, double*, int, int);
__global__ void k_sph2d_spectrum(const cufftDoubleComplex*, double*, int, double);
__global__ void k_sph2d_err_reduce(const double*, const double*, const double*,
    double, double*, int, int);
__global__ void k_sph2d_init_curand(curandStatePhilox4_32_10_t*, uint64_t, int);
__global__ void k_sph2d_apply_forcing(cufftDoubleComplex*, const int*, const double*,
    curandStatePhilox4_32_10_t*, double, int);
__global__ void k_sph2d_snapshot_frame(const double*, const double*, const double*,
                                        double*, int);

static inline void copy_hat(cufftDoubleComplex* dst, const cufftDoubleComplex* src, int n) {
    CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(cufftDoubleComplex),
                          cudaMemcpyDeviceToDevice));
}

// ============================================================
// 1 stage rhs:從 ζ̂ 得到 rhs_hat(不含線性耗散,IFRK 處理)
// 對流 + β 項全部在這裡;耗散被分到 k_sph2d_ifrk_combine。
// scratch:d_tmp_hat, d_psi_hat(後者是 struct member,可持久)
// ============================================================
static void compute_rhs_from_zeta(Sph2DSpectralSolver& s,
                                  const cufftDoubleComplex* zeta_hat_in,
                                  cufftDoubleComplex* rhs_hat_out) {
    int B = 256;
    int Glm = (s.n_lm + B - 1) / B;
    int Gphys = (s.n_phys + B - 1) / B;

    // 1. ψ̂ = R²/(l(l+1)) · ζ̂
    k_sph2d_vort_to_psi<<<Glm, B>>>(zeta_hat_in, s.d_psi_hat, s.n_lm, s.R);

    // 2. 逆 transform ζ̂, ψ̂ 到物理空間
    s.T.spectral_to_spatial(zeta_hat_in, s.d_zeta);
    s.T.spectral_to_spatial(s.d_psi_hat, s.d_psi);

    // 3. 譜空間 ∂_φ(im 乘),再逆 transform
    k_sph2d_dphi<<<Glm, B>>>(zeta_hat_in, s.d_tmp_hat, s.n_lm);
    s.T.spectral_to_spatial(s.d_tmp_hat, s.d_dzeta_dphi);
    k_sph2d_dphi<<<Glm, B>>>(s.d_psi_hat, s.d_tmp_hat, s.n_lm);
    s.T.spectral_to_spatial(s.d_tmp_hat, s.d_dpsi_dphi);

    // 4. θ 方向 central FD(物理)
    k_sph2d_dtheta_fd<<<Gphys, B>>>(s.d_zeta, s.d_theta, s.d_dzeta_dth,
                                     s.N_theta, s.N_phi);
    k_sph2d_dtheta_fd<<<Gphys, B>>>(s.d_psi, s.d_theta, s.d_dpsi_dth,
                                     s.N_theta, s.N_phi);

    // 5. 組合 Jacobian + β → rhs_phys
    k_sph2d_compute_rhs_phys<<<Gphys, B>>>(
        s.d_dpsi_dphi, s.d_dzeta_dphi,
        s.d_dpsi_dth,  s.d_dzeta_dth,
        s.d_sin_theta, s.Omega, s.R,
        s.d_rhs_phys, s.N_theta, s.N_phi);

    // 6. Forward transform → rhs_hat
    s.T.spatial_to_spectral(s.d_rhs_phys, rhs_hat_out);

    // DC mode 強制 0(均勻旋轉不演化)
    k_sph2d_zero_mean<<<1, 1>>>(rhs_hat_out);
}

// ============================================================
// init / destroy
// ============================================================
void Sph2DSpectralSolver::init(int N_theta_, int N_phi_, int L_max_,
                               double R_, double Omega_, double nu_, double cfl_) {
    T.init(N_theta_, N_phi_, L_max_, R_);
    N_theta = T.N_theta;
    N_phi   = T.N_phi;
    L_max   = T.L_max;
    n_lm    = T.n_lm;
    n_phys  = T.n_phys;
    n_fper  = T.n_fper;
    R       = T.R;
    Omega   = Omega_;
    nu      = nu_;
    cfl     = cfl_;

    // 譜 buffers
    CUDA_CHECK(cudaMalloc(&d_zeta_hat, n_lm * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_psi_hat,  n_lm * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_rhs_hat,  n_lm * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_k1_hat,   n_lm * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_tmp_hat,  n_lm * sizeof(cufftDoubleComplex)));

    // 物理 buffers
    CUDA_CHECK(cudaMalloc(&d_zeta,       n_phys * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_psi,        n_phys * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dpsi_dphi,  n_phys * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dzeta_dphi, n_phys * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dpsi_dth,   n_phys * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dzeta_dth,  n_phys * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rhs_phys,   n_phys * sizeof(double)));

    // Gauss-Legendre θ/sinθ/θ_arr(host 先算 θ_j = acos(x_j))
    h_sin_theta.resize(N_theta);
    h_theta.resize(N_theta);
    for (int j = 0; j < N_theta; ++j) {
        double x = T.h_cos_theta[j];
        h_theta[j] = std::acos(std::max(-1.0, std::min(1.0, x)));
        h_sin_theta[j] = std::sqrt(std::max(0.0, 1.0 - x * x));
    }
    CUDA_CHECK(cudaMalloc(&d_sin_theta, N_theta * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_theta,     N_theta * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_sin_theta, h_sin_theta.data(),
                          N_theta * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_theta, h_theta.data(),
                          N_theta * sizeof(double), cudaMemcpyHostToDevice));

    // 診斷 scratch
    int B = 256;
    reduce_blocks = (n_phys + B - 1) / B;
    CUDA_CHECK(cudaMalloc(&d_reduce, 4 * reduce_blocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_E_bins, (L_max + 1) * sizeof(double)));

    std::fprintf(stderr,
        "  Sph2DSpectral init: R=%.3g Ω=%.3g ν=%.3g cfl=%.3g\n",
        R, Omega, nu, cfl);
}

void Sph2DSpectralSolver::destroy() {
    auto F = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
    F((void*&)d_zeta_hat);
    F((void*&)d_psi_hat);
    F((void*&)d_rhs_hat);
    F((void*&)d_k1_hat);
    F((void*&)d_tmp_hat);
    F((void*&)d_zeta_hat_ic);
    F((void*&)d_zeta);
    F((void*&)d_psi);
    F((void*&)d_dpsi_dphi);
    F((void*&)d_dzeta_dphi);
    F((void*&)d_dpsi_dth);
    F((void*&)d_dzeta_dth);
    F((void*&)d_rhs_phys);
    F((void*&)d_sin_theta);
    F((void*&)d_theta);
    F((void*&)d_reduce);
    F((void*&)d_E_bins);
    F((void*&)d_forcing_idx);
    F((void*&)d_forcing_sigma);
    F((void*&)d_forcing_states);
    T.destroy();
    free_frame_buffer();
}

// ============================================================
// Rossby 波單模 IC:
//   ζ(θ,φ) = amp · Re[Y_l^m(θ,φ)]   (對 m=0 是 P_l(cos θ) scaled)
// 解析色散:ω_rw = -2Ω·m / (l·(l+1))
// analytic 傳播:ζ̂_l^m(t) = ζ̂_l^m(0) · exp(-i·ω_rw·t)
// (對實場,ζ̂_l^{-m} 共軛對演化 exp(+i·ω_rw·t) ;合起來仍是實的)
// ============================================================
void Sph2DSpectralSolver::init_rossby_wave(int l, int m, double amp) {
    // 直接設 ζ̂_l^m = amp (在 (l,m) 的複數位置),其餘 0
    CUDA_CHECK(cudaMemset(d_zeta_hat, 0, n_lm * sizeof(cufftDoubleComplex)));
    cufftDoubleComplex val{};
    val.x = amp;
    val.y = 0.0;
    int idx = SphTransform::lm_to_idx(l, m);
    CUDA_CHECK(cudaMemcpy(&d_zeta_hat[idx], &val,
                          sizeof(cufftDoubleComplex),
                          cudaMemcpyHostToDevice));

    // 保存 IC 譜
    if (!d_zeta_hat_ic) {
        CUDA_CHECK(cudaMalloc(&d_zeta_hat_ic, n_lm * sizeof(cufftDoubleComplex)));
    }
    copy_hat(d_zeta_hat_ic, d_zeta_hat, n_lm);
    has_rossby_ic = true;
    rossby_l = l;
    rossby_m = m;

    // 同步物理場(供 dt 首估)
    T.spectral_to_spatial(d_zeta_hat, d_zeta);
    dt_current = 0.0;
    step_count = 0;
    std::fprintf(stderr,
        "  Sph2D Rossby wave IC: (l,m)=(%d,%d) amp=%.3g, "
        "ω_rw = -2Ω·m/(l(l+1)) = %.6g\n",
        l, m, amp, -2.0 * Omega * m / (double)(l * (l + 1)));
}

void Sph2DSpectralSolver::init_zero() {
    CUDA_CHECK(cudaMemset(d_zeta_hat, 0, n_lm * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMemset(d_zeta, 0, n_phys * sizeof(double)));
    dt_current = 0.0;
    step_count = 0;
    std::fprintf(stderr, "  Sph2D zero IC\n");
}

// ============================================================
// Stochastic forcing on shell l ∈ [l_min, l_max]
//
//   Δζ̂_l^m = √dt · σ · e^{iφ_rand}, σ 常數(shell 內等同)
// ε_inj 對應動能注入率:
//   d(KE)/dt = ¼ · Σ w(k)·σ²/l(l+1)·R²   (對 ψ·∇²ψ = -ψ·ζ ; KE ∝ Σ ζ̂²/l(l+1))
// σ² = 2·ε / (R²·Σ 1/l(l+1) · w)        (省略細常數,調用者用 ε 下 close to 實測)
// ============================================================
void Sph2DSpectralSolver::init_forcing(int l_min, int l_max, double eps,
                                       uint64_t seed) {
    forcing_enabled = true;
    forcing_eps = eps;
    forcing_l_min = l_min;
    forcing_l_max = l_max;
    forcing_seed = seed;

    std::vector<int> h_idx;
    std::vector<double> h_ll1;
    for (int l = std::max(1, l_min); l <= std::min(L_max, l_max); ++l) {
        for (int m = 1; m <= l; ++m) {
            // 跳過 m=0 (Hermitian 自共軛,加 noise 破壞實值性)
            // 跳過 m = L_max (若剛好邊緣) — 為簡化不做額外檢查
            int idx = SphTransform::lm_to_idx(l, m);
            h_idx.push_back(idx);
            h_ll1.push_back((double)l * (l + 1));
        }
    }
    forcing_n = (int)h_idx.size();
    if (forcing_n == 0) {
        std::fprintf(stderr, "  Sph2D forcing: shell empty\n");
        forcing_enabled = false;
        return;
    }

    double Nsum = 0.0;
    for (double v : h_ll1) Nsum += 1.0 / v;
    // σ² = ε · (某正規化) / Nsum ;這裡不追嚴格 ε 匹配(與文獻約定多種),
    // 取 σ² = ε · 4π · L² / Nsum 作為 first-order scaling。
    double sigma2 = eps * 4.0 * M_PI * (double)(L_max + 1) * (L_max + 1)
                    / std::max(Nsum, 1e-30);
    double sigma_val = std::sqrt(std::max(sigma2, 0.0));
    std::vector<double> h_sigma(forcing_n, sigma_val);

    CUDA_CHECK(cudaMalloc(&d_forcing_idx,   forcing_n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_forcing_sigma, forcing_n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_forcing_states,
                          forcing_n * sizeof(curandStatePhilox4_32_10_t)));
    CUDA_CHECK(cudaMemcpy(d_forcing_idx, h_idx.data(),
                          forcing_n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_forcing_sigma, h_sigma.data(),
                          forcing_n * sizeof(double), cudaMemcpyHostToDevice));
    int B = 256, G = (forcing_n + B - 1) / B;
    k_sph2d_init_curand<<<G, B>>>(d_forcing_states, seed, forcing_n);

    std::fprintf(stderr,
        "  Sph2D forcing: l∈[%d,%d] (%d modes), ε=%.3e, σ=%.3e\n",
        l_min, l_max, forcing_n, eps, sigma_val);
}

void Sph2DSpectralSolver::apply_forcing(double dt) {
    if (!forcing_enabled || forcing_n == 0 || dt <= 0.0) return;
    int B = 256, G = (forcing_n + B - 1) / B;
    k_sph2d_apply_forcing<<<G, B>>>(d_zeta_hat, d_forcing_idx, d_forcing_sigma,
                                     d_forcing_states, std::sqrt(dt), forcing_n);
}

// ============================================================
// dt — CFL 對流 + drag 最大 |v| 估計;drag/ν/hyper 在 IFRK 處理,無需擴散 CFL。
// ============================================================
static double compute_dt_sph(Sph2DSpectralSolver& s, double max_v) {
    // 等效 dx 取 2πR / N_phi (最小經向格距在赤道)
    double dx_min = 2.0 * M_PI * s.R / (double)s.N_phi;
    double dt_adv = s.cfl * dx_min / std::fmax(max_v, 1e-30);
    double dt = std::fmin(dt_adv, s.dt_max);
    if (s.use_pi_dt && s.dt_prev > 0.0) {
        double up = s.dt_prev * 1.10;
        double dn = s.dt_prev * 0.50;
        dt = std::fmax(std::fmin(dt, up), std::fmin(dn, dt));
    }
    return std::fmax(dt, s.dt_min);
}

// ============================================================
// step — Shu-Osher IFRK3
// ============================================================
double Sph2DSpectralSolver::step() {
    int B = 256;
    int Glm = (n_lm + B - 1) / B;

    // 首步:先 evaluate rhs 以便有 ψ, ∂_ψ 估 max|v|
    if (step_count == 0) {
        compute_rhs_from_zeta(*this, d_zeta_hat, d_rhs_hat);
    }

    // max |v| reduction
    size_t shm = 4 * (size_t)B * sizeof(double);
    k_sph2d_reduce_diag<<<reduce_blocks, B, shm>>>(
        d_dpsi_dphi, d_dpsi_dth, d_zeta,
        d_sin_theta, T.d_weight, R, R * R,
        1.0 / (double)N_phi,
        d_reduce, N_theta, N_phi);
    std::vector<double> h_red(4 * reduce_blocks);
    CUDA_CHECK(cudaMemcpy(h_red.data(), d_reduce,
                          h_red.size() * sizeof(double), cudaMemcpyDeviceToHost));
    double max_v = 0.0;
    for (int i = 0; i < reduce_blocks; ++i)
        max_v = std::fmax(max_v, h_red[4 * i + 0]);
    // 再加上 Rossby / β 項的特徵速度 = 2Ω·R(保守 CFL)
    double v_eff = std::fmax(max_v, 2.0 * std::fabs(Omega) * R);
    double dt = compute_dt_sph(*this, v_eff);
    dt_current = dt;

    if (forcing_enabled) apply_forcing(dt);

    // y_orig
    copy_hat(d_k1_hat, d_zeta_hat, n_lm);

    double nu_dt = nu * dt;
    double drag_dt = drag_alpha * dt;

    // 級 1
    compute_rhs_from_zeta(*this, d_zeta_hat, d_rhs_hat);
    k_sph2d_ifrk_combine<<<Glm, B>>>(
        d_zeta_hat, d_k1_hat, d_zeta_hat, d_rhs_hat,
        1.0, 0.0, dt, 1.0, 1.0,
        nu_dt, drag_dt, R, hyper_p, n_lm);
    k_sph2d_zero_mean<<<1, 1>>>(d_zeta_hat);

    // 級 2
    compute_rhs_from_zeta(*this, d_zeta_hat, d_rhs_hat);
    k_sph2d_ifrk_combine<<<Glm, B>>>(
        d_zeta_hat, d_k1_hat, d_zeta_hat, d_rhs_hat,
        0.75, 0.25, 0.25 * dt, 0.5, -0.5,
        nu_dt, drag_dt, R, hyper_p, n_lm);
    k_sph2d_zero_mean<<<1, 1>>>(d_zeta_hat);

    // 級 3
    compute_rhs_from_zeta(*this, d_zeta_hat, d_rhs_hat);
    k_sph2d_ifrk_combine<<<Glm, B>>>(
        d_zeta_hat, d_k1_hat, d_zeta_hat, d_rhs_hat,
        1.0/3.0, 2.0/3.0, (2.0/3.0) * dt, 1.0, 0.5,
        nu_dt, drag_dt, R, hyper_p, n_lm);
    k_sph2d_zero_mean<<<1, 1>>>(d_zeta_hat);

    // 同步物理 ζ、ψ、∂ψ/∂φ、∂ψ/∂θ 供下一步 dt 估
    k_sph2d_vort_to_psi<<<Glm, B>>>(d_zeta_hat, d_psi_hat, n_lm, R);
    T.spectral_to_spatial(d_zeta_hat, d_zeta);
    T.spectral_to_spatial(d_psi_hat,  d_psi);
    k_sph2d_dphi<<<Glm, B>>>(d_psi_hat, d_tmp_hat, n_lm);
    T.spectral_to_spatial(d_tmp_hat, d_dpsi_dphi);
    int Gphys = (n_phys + B - 1) / B;
    k_sph2d_dtheta_fd<<<Gphys, B>>>(d_psi, d_theta, d_dpsi_dth,
                                     N_theta, N_phi);

    dt_prev = dt;
    step_count++;
    return dt;
}

// ============================================================
// Diagnostics
// ============================================================
Sph2DSpectralSolver::Diagnostics
Sph2DSpectralSolver::compute_diagnostics(double t_eval) {
    int B = 256;
    size_t shm = 4 * (size_t)B * sizeof(double);
    k_sph2d_reduce_diag<<<reduce_blocks, B, shm>>>(
        d_dpsi_dphi, d_dpsi_dth, d_zeta,
        d_sin_theta, T.d_weight, R, R * R,
        1.0 / (double)N_phi,
        d_reduce, N_theta, N_phi);
    std::vector<double> h(4 * reduce_blocks);
    CUDA_CHECK(cudaMemcpy(h.data(), d_reduce,
                          h.size() * sizeof(double), cudaMemcpyDeviceToHost));
    Diagnostics d{};
    d.err_L2 = std::nan("");
    d.t_eval = t_eval;
    for (int i = 0; i < reduce_blocks; ++i) {
        d.max_v           = std::fmax(d.max_v,     h[4 * i + 0]);
        d.max_zeta        = std::fmax(d.max_zeta,  h[4 * i + 1]);
        d.total_KE       += h[4 * i + 2];
        d.total_enstrophy+= h[4 * i + 3];
    }

    // Rossby analytic err_L2(譜空間對比)
    if (has_rossby_ic && d_zeta_hat_ic) {
        // ζ̂(t) should be ζ̂_IC · exp(-i·ω_rw·t) 在 (l_ros, m_ros) 上;其他 mode 0。
        // 差異 L² 在譜空間:|ζ̂_num - ζ̂_exact|² over all (l,m)
        // 用 host 端對比(只有幾個 mode 有數據),省 kernel
        std::vector<cufftDoubleComplex> h_num(n_lm), h_ic(n_lm);
        CUDA_CHECK(cudaMemcpy(h_num.data(), d_zeta_hat,
                              n_lm * sizeof(cufftDoubleComplex),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ic.data(), d_zeta_hat_ic,
                              n_lm * sizeof(cufftDoubleComplex),
                              cudaMemcpyDeviceToHost));
        double omega_rw = -2.0 * Omega * (double)rossby_m
                         / (double)(rossby_l * (rossby_l + 1));
        double phase = omega_rw * t_eval;
        double cp = std::cos(-phase), sp = std::sin(-phase);  // exp(-iω·t)
        double err2 = 0.0;
        for (int i = 0; i < n_lm; ++i) {
            double ex_r = cp * h_ic[i].x - sp * h_ic[i].y;
            double ex_i = sp * h_ic[i].x + cp * h_ic[i].y;
            double dr = h_num[i].x - ex_r;
            double di = h_num[i].y - ex_i;
            err2 += dr * dr + di * di;
        }
        d.err_L2 = std::sqrt(err2);
    }

    return d;
}

void Sph2DSpectralSolver::compute_spectrum(std::vector<double>& out) {
    CUDA_CHECK(cudaMemset(d_E_bins, 0, (L_max + 1) * sizeof(double)));
    int B = 256;
    int Glm = (n_lm + B - 1) / B;
    k_sph2d_spectrum<<<Glm, B>>>(d_zeta_hat, d_E_bins, n_lm, R);
    out.resize(L_max + 1);
    CUDA_CHECK(cudaMemcpy(out.data(), d_E_bins,
                          (L_max + 1) * sizeof(double), cudaMemcpyDeviceToHost));
}

void Sph2DSpectralSolver::download_zeta(std::vector<double>& h) {
    h.resize(n_phys);
    CUDA_CHECK(cudaMemcpy(h.data(), d_zeta, n_phys * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

void Sph2DSpectralSolver::download_uv(std::vector<double>& h_ut,
                                      std::vector<double>& h_up) {
    // u_θ = -(1/(R sin θ)) · ∂_φ ψ, u_φ = (1/R) · ∂_θ ψ (host 算)
    std::vector<double> h_dp(n_phys), h_dt(n_phys);
    CUDA_CHECK(cudaMemcpy(h_dp.data(), d_dpsi_dphi, n_phys * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dt.data(), d_dpsi_dth,  n_phys * sizeof(double),
                          cudaMemcpyDeviceToHost));
    h_ut.resize(n_phys);
    h_up.resize(n_phys);
    for (int j = 0; j < N_theta; ++j) {
        double sth = h_sin_theta[j];
        double inv_sth = (sth > 1e-12) ? 1.0 / sth : 0.0;
        for (int i = 0; i < N_phi; ++i) {
            int idx = j * N_phi + i;
            h_ut[idx] = -inv_sth * h_dp[idx] / R;
            h_up[idx] =  h_dt[idx] / R;
        }
    }
}

// ============================================================
// ASCII STRUCTURED_GRID VTK:lat-lon mesh + ω 標量 + velocity 向量
// 座標採球面 (x,y,z) = (R·sinθ·cosφ, R·sinθ·sinφ, R·cosθ),
// 便於 ParaView 直接渲染球面。
// ============================================================
void Sph2DSpectralSolver::write_vtk_2d(const char* filename) {
    std::vector<double> h_z, h_ut, h_up;
    download_zeta(h_z);
    download_uv(h_ut, h_up);
    std::FILE* fp = std::fopen(filename, "w");
    if (!fp) return;
    int nnx = N_phi + 1, nny = N_theta + 1;
    std::fprintf(fp, "# vtk DataFile Version 3.0\nsph2d_spectral\nASCII\n");
    std::fprintf(fp, "DATASET STRUCTURED_GRID\nDIMENSIONS %d %d 1\n", nnx, nny);
    std::fprintf(fp, "POINTS %d double\n", nnx * nny);
    for (int jn = 0; jn < nny; ++jn) {
        double th = (jn < N_theta) ? h_theta[jn] : M_PI;
        double sth = std::sin(th), cth = std::cos(th);
        for (int in = 0; in < nnx; ++in) {
            double ph = 2.0 * M_PI * (double)in / (double)N_phi;
            double x = R * sth * std::cos(ph);
            double y = R * sth * std::sin(ph);
            double z = R * cth;
            std::fprintf(fp, "%.6e %.6e %.6e\n", x, y, z);
        }
    }
    std::fprintf(fp, "CELL_DATA %d\n", N_theta * N_phi);
    std::fprintf(fp, "SCALARS vorticity double 1\nLOOKUP_TABLE default\n");
    for (int j = 0; j < N_theta; ++j)
        for (int i = 0; i < N_phi; ++i)
            std::fprintf(fp, "%.6e\n", h_z[j * N_phi + i]);
    std::fclose(fp);
}

// ============================================================
// VRAM frame pool — layout: [ω | ∂ψ/∂φ | ∂ψ/∂θ] × n_phys 每幀
// ============================================================
void Sph2DSpectralSolver::alloc_frame_buffer(int headroom_mb) {
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    size_t headroom_b = (size_t)headroom_mb * 1024ull * 1024ull;
    if (free_b <= headroom_b) {
        std::fprintf(stderr,
            "  Sph2D frame buffer: only %.2f GB free — disabling\n",
            free_b / 1.0e9);
        frame_capacity = 0;
        return;
    }
    size_t pool_b = free_b - headroom_b;
    size_t per_frame_b = (size_t)n_phys * 3ull * sizeof(double);
    frame_capacity = (int)(pool_b / per_frame_b);
    if (frame_capacity < 4) frame_capacity = 4;
    size_t actual_b = (size_t)frame_capacity * per_frame_b;
    if (cudaMalloc(&d_frame_pool, actual_b) != cudaSuccess) {
        frame_capacity = (int)(((size_t)(free_b * 0.5)) / per_frame_b);
        if (frame_capacity < 4) frame_capacity = 4;
        actual_b = (size_t)frame_capacity * per_frame_b;
        CUDA_CHECK(cudaMalloc(&d_frame_pool, actual_b));
    }
    frame_count = 0; total_frames = 0;
    std::fprintf(stderr,
        "  Sph2D frame buffer: %d × %.2f MB = %.2f GB\n",
        frame_capacity, per_frame_b / 1.0e6, actual_b / 1.0e9);
}

void Sph2DSpectralSolver::capture_frame(double t, int step) {
    if (!d_frame_pool || frame_capacity == 0) return;
    if (frame_count >= frame_capacity) flush_frames_to_disk(frame_out_dir);
    int B = 256, G = (n_phys + B - 1) / B;
    double* slot = d_frame_pool + (size_t)frame_count * 3ull * (size_t)n_phys;
    k_sph2d_snapshot_frame<<<G, B>>>(d_zeta, d_dpsi_dphi, d_dpsi_dth,
                                      slot, n_phys);
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

static void write_vtk_binary_frame_sph2d(const char* path,
                                         int N_theta, int N_phi,
                                         const double* theta_arr,
                                         double R,
                                         const double* zeta,
                                         const double* dpsi_dphi,
                                         const double* dpsi_dtheta,
                                         const double* sin_theta) {
    std::FILE* fp = std::fopen(path, "wb");
    if (!fp) return;
    int nnx = N_phi + 1, nny = N_theta + 1;
    std::fprintf(fp, "# vtk DataFile Version 3.0\nsph2d frame\nBINARY\n");
    std::fprintf(fp, "DATASET STRUCTURED_GRID\nDIMENSIONS %d %d 1\n", nnx, nny);
    std::fprintf(fp, "POINTS %d double\n", nnx * nny);
    std::vector<double> pts(3 * nnx * nny);
    size_t k = 0;
    for (int jn = 0; jn < nny; ++jn) {
        double th = (jn < N_theta) ? theta_arr[jn] : M_PI;
        double sth = std::sin(th), cth = std::cos(th);
        for (int in = 0; in < nnx; ++in) {
            double ph = 2.0 * M_PI * (double)in / (double)N_phi;
            double x = R * sth * std::cos(ph);
            double y = R * sth * std::sin(ph);
            double z = R * cth;
            pts[k++] = bswap8(x);
            pts[k++] = bswap8(y);
            pts[k++] = bswap8(z);
        }
    }
    std::fwrite(pts.data(), sizeof(double), pts.size(), fp);
    std::fputc('\n', fp);
    int nc = N_theta * N_phi;
    std::fprintf(fp, "CELL_DATA %d\n", nc);
    // vorticity
    auto write_arr = [&](const char* name, const double* arr) {
        std::fprintf(fp, "SCALARS %s double 1\nLOOKUP_TABLE default\n", name);
        std::vector<double> buf(nc);
        for (int ii = 0; ii < nc; ++ii) buf[ii] = bswap8(arr[ii]);
        std::fwrite(buf.data(), sizeof(double), buf.size(), fp);
        std::fputc('\n', fp);
    };
    write_arr("vorticity", zeta);

    // u_theta, u_phi (派生自 ∂ψ)
    std::vector<double> ut(nc), up(nc);
    for (int j = 0; j < N_theta; ++j) {
        double sth = sin_theta[j];
        double inv_sth = (sth > 1e-12) ? 1.0 / sth : 0.0;
        for (int i = 0; i < N_phi; ++i) {
            int idx = j * N_phi + i;
            ut[idx] = -inv_sth * dpsi_dphi[idx] / R;
            up[idx] =  dpsi_dtheta[idx] / R;
        }
    }
    write_arr("u_theta", ut.data());
    write_arr("u_phi",   up.data());
    // 速率
    std::vector<double> spd(nc);
    for (int ii = 0; ii < nc; ++ii)
        spd[ii] = std::sqrt(ut[ii] * ut[ii] + up[ii] * up[ii]);
    write_arr("speed", spd.data());
    std::fclose(fp);
}

void Sph2DSpectralSolver::flush_frames_to_disk(const std::string& run_dir) {
    if (frame_count == 0 || !d_frame_pool) return;
    frame_out_dir = run_dir;
    size_t per_frame = (size_t)n_phys * 3ull;
    std::vector<double> host((size_t)frame_count * per_frame);
    CUDA_CHECK(cudaMemcpy(host.data(), d_frame_pool,
                          host.size() * sizeof(double), cudaMemcpyDeviceToHost));
    char csv_path[512];
    std::snprintf(csv_path, sizeof(csv_path), "%s/frames.csv", run_dir.c_str());
    bool first_batch = (total_frames == 0);
    std::FILE* fcsv = std::fopen(csv_path, first_batch ? "w" : "a");
    if (fcsv && first_batch) std::fprintf(fcsv, "index,step,t\n");
    int base_idx = total_frames;
    std::fprintf(stderr, "  Sph2D flushing %d frames → %s ...",
                 frame_count, run_dir.c_str());
    std::fflush(stderr);
    for (int f = 0; f < frame_count; ++f) {
        ++total_frames;
        const double* base = host.data() + (size_t)f * per_frame;
        const double* z = base;
        const double* dp = base + n_phys;
        const double* dt = base + 2 * n_phys;
        char path[512];
        std::snprintf(path, sizeof(path), "%s/output_%04d.vtk",
                      run_dir.c_str(), total_frames);
        write_vtk_binary_frame_sph2d(path, N_theta, N_phi,
                                     h_theta.data(), R, z, dp, dt,
                                     h_sin_theta.data());
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

void Sph2DSpectralSolver::free_frame_buffer() {
    if (d_frame_pool) { cudaFree(d_frame_pool); d_frame_pool = nullptr; }
    frame_capacity = 0;
    frame_count = 0;
    frame_times.clear();
    frame_steps.clear();
}

// ============================================================
// Checkpoint / Restart
// ============================================================
static constexpr uint64_t SPH2D_CKPT_MAGIC = 0x5350483244435031ULL; // "SPH2DCP1"

void Sph2DSpectralSolver::save_checkpoint(const std::string& path, double t) {
    std::FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return;
    uint64_t m = SPH2D_CKPT_MAGIC;
    int32_t iNt = N_theta, iNp = N_phi, iL = L_max, iHp = hyper_p;
    int64_t istep = step_count;
    int64_t inlm = n_lm;
    std::fwrite(&m,    sizeof(m),    1, fp);
    std::fwrite(&iNt,  sizeof(iNt),  1, fp);
    std::fwrite(&iNp,  sizeof(iNp),  1, fp);
    std::fwrite(&iL,   sizeof(iL),   1, fp);
    std::fwrite(&iHp,  sizeof(iHp),  1, fp);
    std::fwrite(&R,          sizeof(R),          1, fp);
    std::fwrite(&Omega,      sizeof(Omega),      1, fp);
    std::fwrite(&nu,         sizeof(nu),         1, fp);
    std::fwrite(&drag_alpha, sizeof(drag_alpha), 1, fp);
    std::fwrite(&istep,      sizeof(istep),      1, fp);
    std::fwrite(&t,          sizeof(t),          1, fp);
    std::fwrite(&inlm,       sizeof(inlm),       1, fp);
    std::vector<cufftDoubleComplex> h(n_lm);
    CUDA_CHECK(cudaMemcpy(h.data(), d_zeta_hat,
                          n_lm * sizeof(cufftDoubleComplex),
                          cudaMemcpyDeviceToHost));
    std::fwrite(h.data(), sizeof(cufftDoubleComplex), n_lm, fp);
    std::fclose(fp);
    std::fprintf(stderr, "  Sph2D checkpoint saved: %s (step=%lld, t=%.6e)\n",
                 path.c_str(), (long long)step_count, t);
}

bool Sph2DSpectralSolver::load_checkpoint(const std::string& path, double& t_out) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    uint64_t m = 0;
    std::fread(&m, sizeof(m), 1, fp);
    if (m != SPH2D_CKPT_MAGIC) { std::fclose(fp); return false; }
    int32_t iNt, iNp, iL, iHp;
    double rR, rO, rNu, rA, rt;
    int64_t istep, inlm;
    std::fread(&iNt, sizeof(iNt), 1, fp);
    std::fread(&iNp, sizeof(iNp), 1, fp);
    std::fread(&iL,  sizeof(iL),  1, fp);
    std::fread(&iHp, sizeof(iHp), 1, fp);
    std::fread(&rR,  sizeof(rR),  1, fp);
    std::fread(&rO,  sizeof(rO),  1, fp);
    std::fread(&rNu, sizeof(rNu), 1, fp);
    std::fread(&rA,  sizeof(rA),  1, fp);
    std::fread(&istep, sizeof(istep), 1, fp);
    std::fread(&rt, sizeof(rt), 1, fp);
    std::fread(&inlm, sizeof(inlm), 1, fp);
    if (iNt != N_theta || iNp != N_phi || inlm != n_lm) {
        std::fclose(fp);
        std::fprintf(stderr, "  Sph2D checkpoint: size mismatch\n");
        return false;
    }
    std::vector<cufftDoubleComplex> h(n_lm);
    std::fread(h.data(), sizeof(cufftDoubleComplex), n_lm, fp);
    std::fclose(fp);
    CUDA_CHECK(cudaMemcpy(d_zeta_hat, h.data(),
                          n_lm * sizeof(cufftDoubleComplex),
                          cudaMemcpyHostToDevice));
    // 同步物理
    T.spectral_to_spatial(d_zeta_hat, d_zeta);
    step_count = (int)istep;
    t_out = rt;
    std::fprintf(stderr, "  Sph2D checkpoint loaded (step=%lld, t=%.6e)\n",
                 (long long)istep, rt);
    return true;
}
