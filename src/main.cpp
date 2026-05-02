#include "grid.h"
#include "state.h"
#include "eos.h"
#include "bc/boundary.h"
#include "hydro/flux.h"
#include "hydro/integrate.h"
#include "hydro/reconstruct.h"
#include "gravity/gmg.h"
#include "io/output.h"
#include "init/lane_emden.h"
#include "init/sedov.h"
#include "init/jeans.h"
#include "init/evrard.h"

#ifdef USE_GPU
#include <cuda_runtime.h>
#ifdef USE_AMGX
#include "gpu/gpu_solver.h"
#endif
#include "gpu/lowmach_solver.h"
#include "gpu/fas_solver.cuh"
#include "gpu/simple_solver.cuh"
#include "gpu/projection_solver.cuh"
#include "gpu/radial1d_solver.cuh"
#include "gpu/wb2d_solver.cuh"
#include "gpu/ale2d_solver.cuh"
#include "gpu/cart_lag_solver.cuh"
#include "gpu/cart_ale_solver.cuh"
#include "gpu/cart_ale2_solver.cuh"
#include "gpu/pseudo_spectral_solver.cuh"
#include "gpu/anelastic_sl_solver.cuh"
#include "gpu/sph2d_spectral_solver.cuh"
#endif

#include <cstdio>
#include <cmath>
#include <string>
#include <cstring>
#include <ctime>
#include <array>
#include <vector>
#include <csignal>
#include <functional>
#include <sys/stat.h>

static volatile sig_atomic_t g_interrupted = 0;
static void handle_sigint(int) { g_interrupted = 1; }

struct SimConfig {
    int nr = 128;
    int ntheta = 64;
    double R_outer = 1.0;
    double log_alpha = 2.0;
    double gamma = 5.0 / 3.0;
    double cfl = 0.4;
    double t_end = 1.0;
    int output_interval = 100;
    double G = 1.0;
    std::string test_case = "lane_emden";
    std::string mesh_type = "log";
    std::string solver_type = "compressible"; // "compressible" or "lowmach"
    std::string precond = "line_jacobi";         // preconditioner for lowmach solver
    Limiter limiter = Limiter::MINMOD;
    double perturb_amplitude = 1e-3;  // density perturbation for lane_emden_perturbed
    std::string bubble_mode = "pressure"; // "pressure" or "entropy"
    // cart_ale --test hse_bubble parameters
    double bubble_xc = 0.5;
    double bubble_yc = 0.3;
    double bubble_rb = 0.1;
    double bubble_alpha = -0.5;   // density multiplier: ρ = ρ_HSE·(1+α·exp(-r²/rb²))
    double bubble_beta  = 0.0;    // pressure multiplier
    // Multi-bubble: each --bubble "xc,yc,rb,alpha,beta" appends one bubble.
    // If none given, falls back to the single --bubble-* defaults.
    std::vector<std::array<double, 5>> bubbles;
    bool no_sponge = false;
    bool lm_hllc = false;
    int cart_ale_remap_order = 2; // cart_ale: 1 = donor-cell, 2 = MUSCL (default)
    std::string cart_ale_limiter = "vanleer"; // minmod / vanleer (default) / mc
    int diag_interval = 0;   // cart_ale: step interval for diagnostics+CSV; 0 = follow output_interval
    int vtk_interval  = 0;   // cart_ale: step interval for VTK dump; 0 = follow output_interval
    bool frame_buffer = false;   // cart_ale: buffer frames in VRAM, dump at wall/end
    int frame_headroom_mb = 1024; // cart_ale: leave this much free VRAM when sizing the pool
    double vtk_dt = 0.0;         // cart_ale: capture every T physical time; 0 = use vtk_interval (step count)
    double cart_ale_cq_lin  = 0.5;   // cart_ale: AV linear coefficient (default Caramana)
    double cart_ale_cq_quad = 2.0;   // cart_ale: AV quadratic coefficient
    bool   cart_ale_shear_aware = false; // cart_ale: reduce Q in shear-dominated cells
    std::string cart_ale2_bc_x = "reflect";  // cart_ale2: reflect / periodic
    std::string cart_ale2_bc_y = "reflect";
    bool cart_ale2_ppm = false;   // cart_ale2: PPM-in-remap (default OFF; falls back to MUSCL)
    std::string cart_ale2_ppm_limiter = "cs"; // cart_ale2: PPM limiter (cs | cw)
    std::string cart_ale2_ppm_space = "prim"; // cart_ale2: PPM recon space (prim | cons)
    bool cart_ale2_ppm_char = true;  // cart_ale2: project to characteristic variables (prim space only)
    int cart_ale2_kh_k = 0;   // cart_ale2: KH mode number (0 = IC default: k=2 shear, k=1 Lecoanet)
    // pseudo-spectral (偽譜法) 專用
    double ps_nu = 1e-4;          // 運動黏度
    double ps_Lx = 1.0;
    double ps_Ly = 1.0;
    double ps_vshear = 0.5;       // KH 基流速度
    int    ps_k = 4;              // KH 擾動模數
    bool   ps_explicit = false;   // 預設用 IFRK3 隱式黏性;此旗標強制改回全顯式 SSP-RK3
    bool   ps_adv_only = false;   // 預設用 skew-symmetric;此旗標強制回到 advective-only 對流
    // forced turbulence (Lilly/Alvelius-style stochastic forcing)
    double ps_forcing_eps = 0.0;  // 能量注入率 ε_inj (0 = 停用 forcing)
    int    ps_forcing_kf  = 32;   // forcing 中心波數 (mode 整數)
    int    ps_forcing_dk  = 1;    // forcing 殼半寬 (mode)
    uint64_t ps_forcing_seed = 0x5a5a5a5aULL;
    bool   ps_forcing_host_rng = false;   // 強制回到 host mt19937 + D2H (除錯用)
    double ps_drag_alpha   = 0.0;          // linear drag -α·ω, 破壞 condensate
    int    ps_hyper_p      = 1;            // hyperviscosity 冪次: 1=Laplacian, >1 高階
    bool   ps_conservative = false;        // 用 conservative(rotational)對流形式
    bool   ps_batched_fft  = false;        // 啟用 batched FFT pipeline (opt-in, 消費卡慢)
    bool   ps_pi_dt        = false;        // PI controller 自適應 dt
    int    ps_tg_k         = 2;            // Taylor-Green 波數 (for --test taylor_green)
    // vortex_merger 參數 (co-rotating Gaussian pair)
    double ps_vm_gamma     = 1.0;          // 峰值渦量 Γ
    double ps_vm_sigma     = 0.08;         // σ / min(Lx,Ly)
    double ps_vm_dist      = 0.20;         // d / min(Lx,Ly);d/σ ≲ 3 → 合併
    int    ps_ckpt_every   = 0;            // 每 N 步存 checkpoint (0 = 停用)
    std::string ps_resume;                 // restart 檔路徑 (空 = 從 IC 開始)
    std::string run_base   = "runs";       // 輸出根目錄 (--run-base)

    // --- sph2d_spectral 2D 薄球殼 偽譜 ---
    double sph_R = 1.0;
    double sph_Omega = 1.0;
    double sph_nu = 1e-4;
    int    sph_Lmax = 0;               // 0 = auto (min(N_theta-1, N_phi/2-1))
    double sph_drag = 0.0;
    int    sph_hyper = 1;
    bool   sph_pi_dt = false;
    int    sph_rossby_l = 4;
    int    sph_rossby_m = 2;
    double sph_rossby_amp = 1.0;
    double sph_forcing_eps = 0.0;
    int    sph_forcing_lmin = 20;
    int    sph_forcing_lmax = 30;
    uint64_t sph_forcing_seed = 0x5a5a5a5aULL;
    int    sph_ckpt_every = 0;
    std::string sph_resume;
    bool radial_only = false;  // enforce v_theta=0, skip theta-direction work (FAS/explicit only)
    double r_inner = -1.0;  // auto-set for mass mesh; override with --r-inner
    double M_core = 0.0;
};

static void extract_density(const Grid& grid, const State& state, std::vector<double>& rho_cells) {
    int nr = grid.nr, nt = grid.ntheta;
    rho_cells.resize(nr * nt);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nt; ++j)
            rho_cells[i * nt + j] = state.rho[grid.idx(i, j)];
}

static double compute_lane_emden_R_star(double n_poly, double K_poly, double rho_c, double G) {
    auto sol = solve_lane_emden(n_poly);
    double alpha2 = (n_poly + 1.0) * K_poly
                    * std::pow(rho_c, 1.0 / n_poly - 1.0)
                    / (4.0 * M_PI * G);
    double alpha = std::sqrt(alpha2);
    return alpha * sol.xi_1;
}

static double compute_lane_emden_R_outer(double n_poly, double K_poly, double rho_c, double G) {
    return compute_lane_emden_R_star(n_poly, K_poly, rho_c, G) * 1.1;
}

static void print_progress(double t, double t_end, int step, double dt,
                           std::timespec& t_start) {
    double frac = t / t_end;
    int pct = static_cast<int>(frac * 100.0);
    if (pct > 100) pct = 100;

    int bar_width = 30;
    int filled = static_cast<int>(frac * bar_width);

    std::timespec t_now;
    clock_gettime(CLOCK_MONOTONIC, &t_now);
    double elapsed = (t_now.tv_sec - t_start.tv_sec)
                   + (t_now.tv_nsec - t_start.tv_nsec) * 1e-9;

    double eta = (frac > 1e-6) ? elapsed / frac * (1.0 - frac) : 0.0;

    std::fprintf(stderr, "\r  [");
    for (int i = 0; i < bar_width; ++i)
        std::fputc(i < filled ? '#' : '.', stderr);
    std::fprintf(stderr, "] %3d%%  step %-8d  t=%.3e  dt=%.2e  elapsed %.0fs  ETA %.0fs  ",
                 pct, step, t, dt, elapsed, eta);
    std::fflush(stderr);
}

// Build a traceable run directory: runs/<test>_<nr>x<nt>_<timestamp>/
// Returns the path string (e.g. "runs/lane_emden_64x32_20260428_153012/").
static std::string make_run_dir(const SimConfig& cfg) {
    std::time_t now = std::time(nullptr);
    std::tm* lt = std::localtime(&now);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", lt);

    char dirname[512];
    std::snprintf(dirname, sizeof(dirname), "%s/%s_%dx%d_%s",
                  cfg.run_base.c_str(), cfg.test_case.c_str(),
                  cfg.nr, cfg.ntheta, ts);

    mkdir(cfg.run_base.c_str(), 0755);
    mkdir(dirname, 0755);
    return std::string(dirname);
}

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);
    std::signal(SIGTERM, handle_sigint);

    SimConfig cfg;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--test") == 0 && i + 1 < argc)
            cfg.test_case = argv[++i];
        else if (std::strcmp(argv[i], "--nr") == 0 && i + 1 < argc)
            cfg.nr = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ntheta") == 0 && i + 1 < argc)
            cfg.ntheta = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--tend") == 0 && i + 1 < argc)
            cfg.t_end = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--cfl") == 0 && i + 1 < argc)
            cfg.cfl = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--output-interval") == 0 && i + 1 < argc)
            cfg.output_interval = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--mesh") == 0 && i + 1 < argc)
            cfg.mesh_type = argv[++i];
        else if (std::strcmp(argv[i], "--solver") == 0 && i + 1 < argc)
            cfg.solver_type = argv[++i];
        else if (std::strcmp(argv[i], "--precond") == 0 && i + 1 < argc)
            cfg.precond = argv[++i];
        else if (std::strcmp(argv[i], "--perturb") == 0 && i + 1 < argc)
            cfg.perturb_amplitude = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--bubble-xc") == 0 && i + 1 < argc)
            cfg.bubble_xc = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--bubble-yc") == 0 && i + 1 < argc)
            cfg.bubble_yc = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--bubble-rb") == 0 && i + 1 < argc)
            cfg.bubble_rb = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--bubble-alpha") == 0 && i + 1 < argc)
            cfg.bubble_alpha = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--bubble-beta") == 0 && i + 1 < argc)
            cfg.bubble_beta = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--bubble") == 0 && i + 1 < argc) {
            // --bubble "xc,yc,rb,alpha,beta"  — may be repeated
            std::array<double, 5> b{};
            int n = std::sscanf(argv[++i], "%lf,%lf,%lf,%lf,%lf",
                                &b[0], &b[1], &b[2], &b[3], &b[4]);
            if (n < 5) {
                std::fprintf(stderr, "ERROR: --bubble needs xc,yc,rb,alpha,beta\n");
                return 1;
            }
            cfg.bubbles.push_back(b);
        }
        else if (std::strcmp(argv[i], "--limiter") == 0 && i + 1 < argc) {
            std::string lim = argv[++i];
            if (lim == "vanleer") cfg.limiter = Limiter::VAN_LEER;
            else if (lim == "mc") cfg.limiter = Limiter::MC;
            else cfg.limiter = Limiter::MINMOD;
        }
        else if (std::strcmp(argv[i], "--bubble-mode") == 0 && i + 1 < argc)
            cfg.bubble_mode = argv[++i];
        else if (std::strcmp(argv[i], "--no-sponge") == 0)
            cfg.no_sponge = true;
        else if (std::strcmp(argv[i], "--lm-hllc") == 0)
            cfg.lm_hllc = true;
        else if (std::strcmp(argv[i], "--radial-only") == 0)
            cfg.radial_only = true;
        else if (std::strcmp(argv[i], "--r-inner") == 0 && i + 1 < argc)
            cfg.r_inner = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--remap-order") == 0 && i + 1 < argc)
            cfg.cart_ale_remap_order = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--remap-limiter") == 0 && i + 1 < argc)
            cfg.cart_ale_limiter = argv[++i];
        else if (std::strcmp(argv[i], "--diag-interval") == 0 && i + 1 < argc)
            cfg.diag_interval = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--vtk-interval") == 0 && i + 1 < argc)
            cfg.vtk_interval = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--frame-buffer") == 0)
            cfg.frame_buffer = true;
        else if (std::strcmp(argv[i], "--frame-headroom-mb") == 0 && i + 1 < argc)
            cfg.frame_headroom_mb = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--vtk-dt") == 0 && i + 1 < argc)
            cfg.vtk_dt = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--cq-lin") == 0 && i + 1 < argc)
            cfg.cart_ale_cq_lin = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--cq-quad") == 0 && i + 1 < argc)
            cfg.cart_ale_cq_quad = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--shear-aware-av") == 0)
            cfg.cart_ale_shear_aware = true;
        else if (std::strcmp(argv[i], "--bc-x") == 0 && i + 1 < argc)
            cfg.cart_ale2_bc_x = argv[++i];
        else if (std::strcmp(argv[i], "--bc-y") == 0 && i + 1 < argc)
            cfg.cart_ale2_bc_y = argv[++i];
        else if (std::strcmp(argv[i], "--ppm") == 0)
            cfg.cart_ale2_ppm = true;
        else if (std::strcmp(argv[i], "--ppm-limiter") == 0 && i + 1 < argc)
            cfg.cart_ale2_ppm_limiter = argv[++i];
        else if (std::strcmp(argv[i], "--ppm-space") == 0 && i + 1 < argc)
            cfg.cart_ale2_ppm_space = argv[++i];
        else if (std::strcmp(argv[i], "--no-ppm-char") == 0)
            cfg.cart_ale2_ppm_char = false;
        else if (std::strcmp(argv[i], "--ppm-char") == 0)
            cfg.cart_ale2_ppm_char = true;
        else if (std::strcmp(argv[i], "--cart-ale2-kh-k") == 0 && i + 1 < argc)
            cfg.cart_ale2_kh_k = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-nu") == 0 && i + 1 < argc)
            cfg.ps_nu = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-Lx") == 0 && i + 1 < argc)
            cfg.ps_Lx = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-Ly") == 0 && i + 1 < argc)
            cfg.ps_Ly = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-vshear") == 0 && i + 1 < argc)
            cfg.ps_vshear = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-k") == 0 && i + 1 < argc)
            cfg.ps_k = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-explicit") == 0)
            cfg.ps_explicit = true;
        else if (std::strcmp(argv[i], "--ps-adv-only") == 0)
            cfg.ps_adv_only = true;
        else if (std::strcmp(argv[i], "--ps-forcing-eps") == 0 && i + 1 < argc)
            cfg.ps_forcing_eps = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-forcing-kf") == 0 && i + 1 < argc)
            cfg.ps_forcing_kf = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-forcing-dk") == 0 && i + 1 < argc)
            cfg.ps_forcing_dk = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-forcing-seed") == 0 && i + 1 < argc)
            cfg.ps_forcing_seed = std::strtoull(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--ps-forcing-host-rng") == 0)
            cfg.ps_forcing_host_rng = true;
        else if (std::strcmp(argv[i], "--ps-drag") == 0 && i + 1 < argc)
            cfg.ps_drag_alpha = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-hyper") == 0 && i + 1 < argc)
            cfg.ps_hyper_p = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-conservative") == 0)
            cfg.ps_conservative = true;
        else if (std::strcmp(argv[i], "--ps-batched-fft") == 0)
            cfg.ps_batched_fft = true;
        else if (std::strcmp(argv[i], "--ps-pi") == 0)
            cfg.ps_pi_dt = true;
        else if (std::strcmp(argv[i], "--ps-tg-k") == 0 && i + 1 < argc)
            cfg.ps_tg_k = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-vm-gamma") == 0 && i + 1 < argc)
            cfg.ps_vm_gamma = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-vm-sigma") == 0 && i + 1 < argc)
            cfg.ps_vm_sigma = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-vm-dist") == 0 && i + 1 < argc)
            cfg.ps_vm_dist = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-ckpt-every") == 0 && i + 1 < argc)
            cfg.ps_ckpt_every = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-resume") == 0 && i + 1 < argc)
            cfg.ps_resume = argv[++i];
        else if (std::strcmp(argv[i], "--run-base") == 0 && i + 1 < argc)
            cfg.run_base = argv[++i];
        // sph2d flags
        else if (std::strcmp(argv[i], "--sph-R") == 0 && i + 1 < argc)
            cfg.sph_R = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-Omega") == 0 && i + 1 < argc)
            cfg.sph_Omega = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-nu") == 0 && i + 1 < argc)
            cfg.sph_nu = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-Lmax") == 0 && i + 1 < argc)
            cfg.sph_Lmax = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-drag") == 0 && i + 1 < argc)
            cfg.sph_drag = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-hyper") == 0 && i + 1 < argc)
            cfg.sph_hyper = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-pi") == 0)
            cfg.sph_pi_dt = true;
        else if (std::strcmp(argv[i], "--sph-rossby-l") == 0 && i + 1 < argc)
            cfg.sph_rossby_l = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-rossby-m") == 0 && i + 1 < argc)
            cfg.sph_rossby_m = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-rossby-amp") == 0 && i + 1 < argc)
            cfg.sph_rossby_amp = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-forcing-eps") == 0 && i + 1 < argc)
            cfg.sph_forcing_eps = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-forcing-lmin") == 0 && i + 1 < argc)
            cfg.sph_forcing_lmin = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-forcing-lmax") == 0 && i + 1 < argc)
            cfg.sph_forcing_lmax = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-forcing-seed") == 0 && i + 1 < argc)
            cfg.sph_forcing_seed = std::strtoull(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--sph-ckpt-every") == 0 && i + 1 < argc)
            cfg.sph_ckpt_every = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-resume") == 0 && i + 1 < argc)
            cfg.sph_resume = argv[++i];
    }

    if (cfg.test_case == "lane_emden" || cfg.test_case == "lane_emden_perturbed"
        || cfg.test_case == "bubble") {
        if (cfg.mesh_type == "mass")
            cfg.R_outer = compute_lane_emden_R_star(1.5, 1.0, 1.0, cfg.G);
        else
            cfg.R_outer = compute_lane_emden_R_outer(1.5, 1.0, 1.0, cfg.G);
    }

    std::printf("stellar2d - 2D Axisymmetric Euler + Self-Gravity\n");
    std::printf("Test case: %s, mesh: %s\n", cfg.test_case.c_str(), cfg.mesh_type.c_str());
    std::printf("Grid: %d x %d, R_outer = %.6f\n", cfg.nr, cfg.ntheta, cfg.R_outer);

    Grid grid;
    if (cfg.mesh_type == "equimass" &&
        (cfg.test_case == "lane_emden" || cfg.test_case == "lane_emden_perturbed"
         || cfg.test_case == "bubble")) {
        double n_poly = 1.5, K_poly = 1.0, rho_c = 1.0, G = cfg.G;
        auto le_sol = solve_lane_emden(n_poly);
        double alpha2 = (n_poly + 1.0) * K_poly
                        * std::pow(rho_c, 1.0 / n_poly - 1.0) / (4.0 * M_PI * G);
        double alpha = std::sqrt(alpha2);

        auto rho_func = [&](double r) -> double {
            double xi = r / alpha;
            if (xi >= le_sol.xi_1) return 1e-20;
            // Interpolate Lane-Emden solution
            auto it = std::lower_bound(le_sol.xi.begin(), le_sol.xi.end(), xi);
            int idx = static_cast<int>(it - le_sol.xi.begin());
            if (idx <= 0) return rho_c;
            if (idx >= static_cast<int>(le_sol.xi.size())) return 1e-20;
            double x0 = le_sol.xi[idx - 1], x1 = le_sol.xi[idx];
            double t0 = le_sol.theta_le[idx - 1], t1 = le_sol.theta_le[idx];
            double frac = (xi - x0) / (x1 - x0);
            double theta_val = t0 + frac * (t1 - t0);
            return rho_c * std::pow(std::max(theta_val, 1e-15), n_poly);
        };

        grid.init_equimass(cfg.nr, cfg.ntheta, cfg.R_outer, rho_func);
        std::printf("Using equimass radial mesh based on Lane-Emden density\n");
    } else if (cfg.mesh_type == "mass" &&
               (cfg.test_case == "lane_emden" || cfg.test_case == "lane_emden_perturbed"
                || cfg.test_case == "bubble")) {
        double n_poly = 1.5, K_poly = 1.0, rho_c = 1.0, G = cfg.G;
        auto le_sol = solve_lane_emden(n_poly);
        double alpha2 = (n_poly + 1.0) * K_poly
                        * std::pow(rho_c, 1.0 / n_poly - 1.0) / (4.0 * M_PI * G);
        double alpha = std::sqrt(alpha2);

        auto rho_func = [&](double r) -> double {
            double xi = r / alpha;
            if (xi >= le_sol.xi_1) return 1e-20;
            auto it = std::lower_bound(le_sol.xi.begin(), le_sol.xi.end(), xi);
            int idx = static_cast<int>(it - le_sol.xi.begin());
            if (idx <= 0) return rho_c;
            if (idx >= static_cast<int>(le_sol.xi.size())) return 1e-20;
            double x0 = le_sol.xi[idx - 1], x1 = le_sol.xi[idx];
            double t0 = le_sol.theta_le[idx - 1], t1 = le_sol.theta_le[idx];
            double frac = (xi - x0) / (x1 - x0);
            double theta_val = t0 + frac * (t1 - t0);
            return rho_c * std::pow(std::max(theta_val, 1e-15), n_poly);
        };

        double r_inner = (cfg.r_inner >= 0) ? cfg.r_inner : 0.0;
        cfg.r_inner = r_inner;

        if (r_inner > 0) {
            const int nfine = 20000;
            double dr_f = r_inner / nfine;
            double m = 0.0;
            for (int ii = 0; ii < nfine; ++ii) {
                double r = (ii + 0.5) * dr_f;
                m += 4.0 * M_PI * rho_func(r) * r * r * dr_f;
            }
            cfg.M_core = m;
        }

        grid.init_mass_shell(cfg.nr, cfg.ntheta, cfg.R_outer, rho_func, r_inner);
        cfg.no_sponge = true;
        std::printf("Using hybrid mass-shell mesh (R_star=%.6f, r_inner=%.4f, M_core=%.5f, HSE outer BC)\n",
                    cfg.R_outer, r_inner, cfg.M_core);
    } else if (cfg.mesh_type == "uniform") {
        grid.init_uniform(cfg.nr, cfg.ntheta, cfg.R_outer);
        std::printf("Using uniform radial mesh\n");
    } else {
        grid.init(cfg.nr, cfg.ntheta, cfg.R_outer, cfg.log_alpha);
    }

    EOS eos(cfg.gamma);

    State state;
    state.allocate(grid);

    if (cfg.test_case == "lane_emden") {
        LaneEmdenParams lep;
        lep.n_poly = 1.5; lep.rho_c = 1.0; lep.K_poly = 1.0; lep.G = cfg.G;
        init_lane_emden(grid, state, lep, cfg.gamma);
    } else if (cfg.test_case == "lane_emden_perturbed") {
        LaneEmdenParams lep;
        lep.n_poly = 1.5; lep.rho_c = 1.0; lep.K_poly = 1.0; lep.G = cfg.G;
        init_lane_emden_perturbed(grid, state, lep, cfg.gamma, cfg.perturb_amplitude);
    } else if (cfg.test_case == "bubble") {
        LaneEmdenParams lep;
        lep.n_poly = 1.5; lep.rho_c = 1.0; lep.K_poly = 1.0; lep.G = cfg.G;
        if (cfg.bubble_mode == "entropy") {
            init_lane_emden_bubble_entropy(grid, state, lep, cfg.gamma,
                                           0.5, M_PI/3.0, 0.15, 0.5);
            std::printf("Bubble mode: entropy (constant pressure, density perturbation)\n");
        } else {
            init_lane_emden_bubble(grid, state, lep, cfg.gamma,
                                   0.5, M_PI/3.0, 0.15, 0.5);
        }
    } else if (cfg.test_case == "sedov") {
        SedovParams sp;
        sp.rho_0 = 1.0; sp.E_blast = 1.0; sp.r_blast = 0.05;
        init_sedov(grid, state, sp, cfg.gamma);
    } else if (cfg.test_case == "jeans") {
        JeansParams jp;
        jp.rho_0 = 1.0; jp.cs = 1.0; jp.G = cfg.G; jp.epsilon = 1e-3;
        jp.k_r = 2.0 * M_PI / cfg.R_outer; jp.k_theta = 2.0;
        init_jeans(grid, state, jp, cfg.gamma);
    } else if (cfg.test_case == "evrard") {
        EvrardParams ep;
        ep.M = 1.0; ep.R = 1.0; ep.G = cfg.G;
        init_evrard(grid, state, ep, cfg.gamma);
    } else if (cfg.test_case == "hse" || cfg.test_case == "hse_perturbed"
               || cfg.test_case == "hse_bubble" || cfg.test_case == "sod"
               || cfg.test_case == "kh_shear" || cfg.test_case == "forced_turb"
               || cfg.test_case == "kh_lecoanet"
               || cfg.test_case == "taylor_green"
               || cfg.test_case == "double_shear_layer"
               || cfg.test_case == "vortex_merger"
               || cfg.test_case == "quad_vortex_merger"
               || cfg.test_case == "rossby_wave"
               || cfg.test_case == "jovian_bands"
               || cfg.test_case == "sl_basis_check"
               || cfg.test_case == "sl_poisson_test"
               || cfg.test_case == "sl_poisson_test_boussinesq"
               || cfg.test_case == "kh_shear_boussinesq"
               || cfg.test_case == "gmode_pulsation") {
        // Cart-Lagrangian-only test cases — no Grid/State initialization needed;
        // cart_lag solver branch handles its own IC.
    } else {
        std::fprintf(stderr, "Unknown test case: %s\n", cfg.test_case.c_str());
        return 1;
    }

    std::string run_dir = make_run_dir(cfg);
    std::printf("Output directory: %s/\n", run_dir.c_str());

    {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/output_0000.vtk", run_dir.c_str());
        write_vtk(path, grid, state, cfg.gamma);
    }

    double t = 0.0;
    int step = 0;

    std::printf("Starting time integration...\n");

#ifdef USE_GPU
    // ── Solver adapter: type-erased callbacks for the time-stepping loop ──
    struct SolverOps {
        std::function<double(double, double)> step;
        std::function<void(const Grid&, State&, double dt)> download;
        std::function<void()> destroy;
        int progress_interval = 200;
    };

    auto snapshot_hse_if_needed = [&](auto& solver) {
        if (cfg.test_case == "lane_emden_perturbed" || cfg.test_case == "bubble") {
            State state_hse;
            state_hse.allocate(grid);
            LaneEmdenParams lep;
            lep.n_poly = 1.5; lep.rho_c = 1.0; lep.K_poly = 1.0; lep.G = cfg.G;
            init_lane_emden(grid, state_hse, lep, cfg.gamma);
            solver.upload_state(grid, state_hse);
            solver.snapshot_hse();
        }
    };

    auto configure_mass_mesh = [&](auto& solver) {
        if (cfg.mesh_type != "mass") return;
        solver.use_hse_outer_bc = true;
        solver.use_core_excision = (cfg.r_inner > 0);
        solver.M_core = cfg.M_core;
        solver.n_pole_avg = cfg.ntheta / 2;
        if (cfg.r_inner <= 0) {
            int n_uni = static_cast<int>(0.15 * cfg.R_outer / (cfg.R_outer / cfg.nr));
            solver.n_angular_avg = n_uni;
        }
    };

    auto run_time_loop = [&](SolverOps& ops) {
        std::timespec wall_start;
        clock_gettime(CLOCK_MONOTONIC, &wall_start);

        while (t < cfg.t_end && !g_interrupted) {
            double dt = ops.step(t, cfg.t_end);
            t += dt;
            step++;

            if (step % ops.progress_interval == 0)
                print_progress(t, cfg.t_end, step, dt, wall_start);

            if (step % cfg.output_interval == 0) {
                ops.download(grid, state, dt);
                Diagnostics diag = compute_diagnostics(grid, state, cfg.gamma);
                std::fprintf(stderr, "\n");
                std::printf("Step %6d  t = %.6e  dt = %.3e  M = %.10e  E = %.10e\n",
                            step, t, dt, diag.total_mass, diag.total_energy);
                char fname[512];
                std::snprintf(fname, sizeof(fname), "%s/output_%04d.vtk",
                              run_dir.c_str(), step / cfg.output_interval);
                write_vtk(fname, grid, state, cfg.gamma);
            }
        }
        std::fprintf(stderr, "\n");
        ops.download(grid, state, 0.0);
        ops.destroy();
    };

    if (cfg.solver_type == "radial1d") {
        // ===== 1D Lagrangian radial solver (MESA RSP-inspired) =====
        // Ignores the 2D Grid; uses nr as number of Lagrangian zones.
        // Lane-Emden specific; other test cases not supported yet.
        if (cfg.test_case != "lane_emden" && cfg.test_case != "lane_emden_perturbed") {
            std::fprintf(stderr, "ERROR: radial1d solver only supports lane_emden / lane_emden_perturbed\n");
            return 1;
        }
        Radial1DSolver r1d;
        r1d.init(cfg.nr, cfg.gamma, cfg.G, cfg.cfl);
        r1d.init_lane_emden(1.0, 1.0, 1.5);          // ρ_c=1, K=1, n=1.5
        r1d.snapshot_hse();
        if (cfg.test_case == "lane_emden_perturbed")
            r1d.apply_perturbation(cfg.perturb_amplitude);

        std::timespec wall_start;
        clock_gettime(CLOCK_MONOTONIC, &wall_start);

        // Simple text output (CSV format) for radial1d mode
        char csv_path[512];
        std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", run_dir.c_str());
        std::FILE* csv = std::fopen(csv_path, "w");
        std::fprintf(csv, "step,t,dt,mass,KE,IE,PE,total_E,max_mach,max_vr\n");

        int frame = 0;
        while (t < cfg.t_end && !g_interrupted) {
            double dt = r1d.step(t, cfg.t_end);
            t += dt;
            step++;

            if (step % 200 == 0)
                print_progress(t, cfg.t_end, step, dt, wall_start);

            if (step % cfg.output_interval == 0) {
                auto d = r1d.compute_diagnostics();
                std::fprintf(stderr, "\n");
                std::printf("Step %6d  t=%.6e dt=%.3e M=%.10e E=%.10e |v|_max=%.3e Mach_max=%.3e\n",
                            step, t, dt, d.total_mass, d.total_E, d.max_vr, d.max_mach);
                std::fprintf(csv, "%d,%.10e,%.6e,%.10e,%.10e,%.10e,%.10e,%.10e,%.6e,%.6e\n",
                             step, t, dt, d.total_mass, d.total_KE, d.total_internal_E,
                             d.total_grav_E, d.total_E, d.max_mach, d.max_vr);
                std::fflush(csv);

                // Dump profile as simple text
                std::vector<double> r_face, v_face, rho_cell, P_cell, e_cell;
                r1d.download_profile(r_face, v_face, rho_cell, P_cell, e_cell);
                char path[512];
                std::snprintf(path, sizeof(path), "%s/profile_%04d.txt", run_dir.c_str(), ++frame);
                std::FILE* fp = std::fopen(path, "w");
                std::fprintf(fp, "# t = %.10e  step = %d\n# k r_face v_face rho P e_int\n", t, step);
                for (int k = 0; k < r1d.lev.nz; ++k) {
                    std::fprintf(fp, "%d %.10e %.10e %.10e %.10e %.10e\n",
                                 k, r_face[k], v_face[k], rho_cell[k], P_cell[k], e_cell[k]);
                }
                // last face
                std::fprintf(fp, "%d %.10e %.10e - - -\n", r1d.lev.nz, r_face[r1d.lev.nz], v_face[r1d.lev.nz]);
                std::fclose(fp);
            }
        }
        std::fclose(csv);
        std::fprintf(stderr, "\n");
        r1d.destroy();

    } else if (cfg.solver_type == "projection") {
        ProjSolver proj;
        proj.init(grid, eos, cfg.G, cfg.cfl);
        if (cfg.no_sponge) proj.sponge_kappa = 0.0;
        configure_mass_mesh(proj);
        snapshot_hse_if_needed(proj);
        proj.upload_state(grid, state);

        SolverOps ops;
        ops.step = [&](double t_, double te) { return proj.step(t_, te); };
        ops.download = [&](const Grid& g, State& s, double) { proj.download_state(g, s); };
        ops.destroy = [&]() { proj.destroy(); };
        run_time_loop(ops);

    } else if (cfg.solver_type == "simple") {
        SimpleSolver sim;
        sim.init(grid, eos, cfg.G, cfg.cfl);
        if (cfg.no_sponge) sim.sponge_kappa = 0.0;
        configure_mass_mesh(sim);
        snapshot_hse_if_needed(sim);
        sim.upload_state(grid, state);

        SolverOps ops;
        ops.step = [&](double t_, double te) { return sim.step(t_, te); };
        ops.download = [&](const Grid& g, State& s, double) { sim.download_state(g, s); };
        ops.destroy = [&]() { sim.destroy(); };
        run_time_loop(ops);

    } else if (cfg.solver_type == "fas" || cfg.solver_type == "explicit") {
        bool use_explicit = (cfg.solver_type == "explicit");
        FasSolver fas;
        fas.use_simple_smoother = (cfg.precond != "block_jacobi");
        fas.limiter_type = static_cast<int>(cfg.limiter);
        fas.use_lm_hllc = cfg.lm_hllc;
        fas.radial_only = cfg.radial_only;
        fas.init(grid, eos, cfg.G, cfg.cfl);
        if (cfg.radial_only)
            std::printf("Radial-only mode: v_theta=0 enforced, theta fluxes and atm_reset skipped\n");
        if (cfg.no_sponge) fas.sponge_kappa = 0.0;
        configure_mass_mesh(fas);
        if (cfg.mesh_type == "mass" && cfg.r_inner <= 0)
            fas.central_damp_r = 0.15 * cfg.R_outer;
        snapshot_hse_if_needed(fas);
        fas.upload_state(grid, state);

        // GPU snapshot buffer: store frames in VRAM, write all at end
        FasLevel& fl = fas.levels[0];
        int snap_size = fl.total;  // per-variable size (with ghost)
        int max_snaps = static_cast<int>(cfg.t_end / (cfg.output_interval * 1e-5)) + 100;
        long long bytes_per_snap = 4LL * snap_size * sizeof(double);
        size_t mem_free = 0, mem_total = 0;
        cudaMemGetInfo(&mem_free, &mem_total);
        long long max_bytes = static_cast<long long>(mem_free) / 2;
        if (max_snaps > max_bytes / bytes_per_snap)
            max_snaps = static_cast<int>(max_bytes / bytes_per_snap);
        if (max_snaps < 1) max_snaps = 1;

        double* d_snap_buf = nullptr;
        CUDA_CHECK(cudaMalloc(&d_snap_buf, (long long)max_snaps * 4 * snap_size * sizeof(double)));
        std::vector<double> snap_times;
        std::vector<double> snap_dts;
        std::vector<int> snap_steps;
        int n_snaps = 0;

        SolverOps ops;
        ops.progress_interval = 2000;
        ops.step = [&](double t_, double te) -> double {
            return use_explicit ? fas.step_explicit(t_, te) : fas.step(t_, te);
        };
        ops.download = [&](const Grid&, State&, double dt_val) {
            if (n_snaps >= max_snaps) return;
            long long off = (long long)n_snaps * 4 * snap_size;
            CUDA_CHECK(cudaMemcpy(d_snap_buf + off, fl.d_rho, snap_size*sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_snap_buf + off + snap_size, fl.d_mr, snap_size*sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_snap_buf + off + 2*snap_size, fl.d_mt, snap_size*sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_snap_buf + off + 3*snap_size, fl.d_rhoE, snap_size*sizeof(double), cudaMemcpyDeviceToDevice));
            snap_times.push_back(t);
            snap_dts.push_back(dt_val);
            snap_steps.push_back(step);
            n_snaps++;
        };
        ops.destroy = [&]() {
            if (g_interrupted)
                std::printf("\nInterrupted at step %d, t=%.6e. ", step, t);
            std::printf("Writing %d snapshots from GPU...\n", n_snaps);
            std::vector<double> h_buf(4LL * snap_size);
            for (int s = 0; s < n_snaps; ++s) {
                long long off = (long long)s * 4 * snap_size;
                CUDA_CHECK(cudaMemcpy(h_buf.data(), d_snap_buf + off, 4*snap_size*sizeof(double), cudaMemcpyDeviceToHost));
                for (int ii = 0; ii < fl.nr; ++ii)
                    for (int jj = 0; jj < fl.nt; ++jj) {
                        int k = grid.idx(ii, jj);
                        int kg = (ii + fl.ng) * (fl.nt + 2*fl.ng) + (jj + fl.ng);
                        state.rho[k] = h_buf[kg];
                        state.mr[k] = h_buf[snap_size + kg];
                        state.mtheta[k] = h_buf[2*snap_size + kg];
                        state.E[k] = h_buf[3*snap_size + kg];
                    }
                Diagnostics diag = compute_diagnostics(grid, state, cfg.gamma);
                std::printf("Step %6d  t = %.6e  dt = %.3e  M = %.10e  E = %.10e\n",
                            snap_steps[s], snap_times[s], snap_dts[s], diag.total_mass, diag.total_energy);
                char fname[512];
                std::snprintf(fname, sizeof(fname), "%s/output_%04d.vtk", run_dir.c_str(), s + 1);
                write_vtk(fname, grid, state, cfg.gamma);
            }
            cudaFree(d_snap_buf);
            fas.download_state(grid, state);
            fas.destroy();
        };
        run_time_loop(ops);

    } else if (cfg.solver_type == "cart_lag") {
        // ===== Cartesian 2D Lagrangian (Caramana compatible, planar) =====
        // Runs independent of Grid/State: uses [0,Lx]×[0,Ly] box = [1,1].
        // IC: Sod shock tube (default) or uniform.
        CartLagSolver clag;
        // For HSE: square box. For Sod: thin strip.
        bool is_hse = (cfg.test_case == "hse" || cfg.test_case == "hse_perturbed");
        double Lx = is_hse ? 1.0 : 1.0;
        double Ly = is_hse ? 1.0 : 0.2;
        double gam = is_hse ? cfg.gamma : 1.4;  // Sod expects γ=1.4 by default
        clag.init(cfg.nr, cfg.ntheta, Lx, Ly, gam, cfg.cfl);
        if (is_hse) {
            // Two-step: first build HSE unperturbed, snapshot its discrete
            // force defect, then (if hse_perturbed) layer the perturbation on top.
            clag.init_hse_polytrope(1.0, 1.0, 0.0);
            clag.snapshot_hse_force();
            if (cfg.test_case == "hse_perturbed") {
                clag.init_hse_polytrope(1.0, 1.0, cfg.perturb_amplitude);
            }
        } else {
            clag.init_sod();
        }

        std::timespec wall_start;
        clock_gettime(CLOCK_MONOTONIC, &wall_start);

        char csv_path[512];
        std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", run_dir.c_str());
        std::FILE* csv = std::fopen(csv_path, "w");
        std::fprintf(csv, "step,t,dt,mass,KE,IE,PE,E,max_v,max_mach\n");

        int frame = 0;
        while (t < cfg.t_end && !g_interrupted) {
            double dt = clag.step(t, cfg.t_end);
            t += dt;
            step++;
            if (step % 200 == 0) print_progress(t, cfg.t_end, step, dt, wall_start);
            if (step % cfg.output_interval == 0 || t >= cfg.t_end) {
                auto d = clag.compute_diagnostics();
                std::fprintf(stderr, "\n");
                std::printf("Step %6d  t=%.6e dt=%.3e M=%.10e KE=%.4e IE=%.10e PE=%.10e E=%.10e |v|=%.3e\n",
                            step, t, dt, d.total_mass, d.total_KE, d.total_internal_E,
                            d.total_PE, d.total_E, d.max_v);
                std::fprintf(csv, "%d,%.10e,%.6e,%.10e,%.10e,%.10e,%.10e,%.10e,%.6e,%.6e\n",
                             step, t, dt, d.total_mass, d.total_KE, d.total_internal_E,
                             d.total_PE, d.total_E, d.max_v, d.max_mach);
                std::fflush(csv);
                std::vector<double> xv, rhov, Pv, vxv, ev;
                clag.download_xslice(xv, rhov, Pv, vxv, ev);
                char path[512];
                std::snprintf(path, sizeof(path), "%s/xslice_%04d.txt", run_dir.c_str(), ++frame);
                std::FILE* fp = std::fopen(path, "w");
                std::fprintf(fp, "# t=%.10e step=%d\n# x rho P vx e\n", t, step);
                for (int i = 0; i < (int)xv.size(); ++i)
                    std::fprintf(fp, "%.10e %.10e %.10e %.10e %.10e\n",
                                 xv[i], rhov[i], Pv[i], vxv[i], ev[i]);
                std::fclose(fp);
            }
        }
        std::fclose(csv);
        std::fprintf(stderr, "\n");
        clag.destroy();
    } else if (cfg.solver_type == "cart_ale") {
        // ===== Cartesian 2D ALE (Caramana Lagrangian + Eulerian rezone + swept remap) =====
        CartAleSolver cale;
        bool is_hse = (cfg.test_case == "hse" || cfg.test_case == "hse_perturbed"
                       || cfg.test_case == "hse_bubble");
        bool is_kh = (cfg.test_case == "kh_shear");
        double Lx = 1.0;
        double Ly = (is_hse || is_kh) ? 1.0 : 0.2;
        double gam = is_hse ? cfg.gamma : 1.4;
        cale.init(cfg.nr, cfg.ntheta, Lx, Ly, gam, cfg.cfl);
        cale.remap_order = cfg.cart_ale_remap_order;
        cale.CQ_lin  = cfg.cart_ale_cq_lin;
        cale.CQ_quad = cfg.cart_ale_cq_quad;
        cale.shear_aware_av = cfg.cart_ale_shear_aware ? 1 : 0;
        if      (cfg.cart_ale_limiter == "minmod")  cale.remap_limiter = 0;
        else if (cfg.cart_ale_limiter == "vanleer") cale.remap_limiter = 1;
        else if (cfg.cart_ale_limiter == "mc")      cale.remap_limiter = 2;
        else { std::fprintf(stderr, "unknown --remap-limiter %s; using vanleer\n",
                            cfg.cart_ale_limiter.c_str()); cale.remap_limiter = 1; }
        const char* lim_name = cale.remap_limiter == 0 ? "minmod"
                             : cale.remap_limiter == 1 ? "vanleer" : "mc";
        std::fprintf(stderr,
            "  CartAle remap_order = %d (%s)  limiter = %s  CQ_lin=%g CQ_quad=%g  shear_aware_av=%d\n",
            cale.remap_order,
            cale.remap_order >= 2 ? "MUSCL-in-remap" : "donor-cell",
            lim_name, cale.CQ_lin, cale.CQ_quad, cale.shear_aware_av);
        if (cfg.test_case == "hse_bubble") {
            std::vector<CartAleSolver::Bubble> blist;
            if (!cfg.bubbles.empty()) {
                for (const auto& b : cfg.bubbles)
                    blist.push_back({b[0], b[1], b[2], b[3], b[4]});
            } else {
                blist.push_back({cfg.bubble_xc, cfg.bubble_yc, cfg.bubble_rb,
                                 cfg.bubble_alpha, cfg.bubble_beta});
            }
            cale.init_hse_bubbles(1.0, 1.0, blist);
        } else if (cfg.test_case == "hse_perturbed") {
            cale.init_hse_polytrope(1.0, 1.0, cfg.perturb_amplitude);
        } else if (cfg.test_case == "hse") {
            cale.init_hse_polytrope(1.0, 1.0, 0.0);
        } else if (cfg.test_case == "kh_shear") {
            // Canonical KH (Athena++ parity): ρ_heavy/ρ_light=2, P0=2.5,
            // |vx|=0.5, amp=0.1 (fully nonlinear seed), k=2 (2 vortices
            // per interface — 256 cells resolve each vortex). --perturb
            // overrides amplitude.
            double amp = (cfg.perturb_amplitude > 0) ? cfg.perturb_amplitude : 0.1;
            cale.init_kh_shear(1.0, 2.0, 2.5, 0.5, amp, 2);
        } else {
            cale.init_sod();
        }

        std::timespec wall_start;
        clock_gettime(CLOCK_MONOTONIC, &wall_start);

        char csv_path[512];
        std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", run_dir.c_str());
        std::FILE* csv = std::fopen(csv_path, "w");
        std::fprintf(csv, "step,t,dt,mass,KE,IE,PE,E,max_v,max_mach\n");

        int diag_every = cfg.diag_interval > 0 ? cfg.diag_interval : cfg.output_interval;
        int vtk_every  = cfg.vtk_interval  > 0 ? cfg.vtk_interval  : cfg.output_interval;
        bool vtk_by_time = cfg.vtk_dt > 0.0;
        double next_vtk_t = 0.0;   // first capture at t=0+dt (after first step)
        if (vtk_by_time) {
            std::fprintf(stderr,
                "  CartAle diag every %d steps, VTK every dt=%g (physical time)%s\n",
                diag_every, cfg.vtk_dt,
                cfg.frame_buffer ? " (VRAM buffered)" : "");
        } else {
            std::fprintf(stderr, "  CartAle diag every %d steps, VTK every %d steps%s\n",
                         diag_every, vtk_every,
                         cfg.frame_buffer ? " (VRAM buffered)" : "");
        }
        if (cfg.frame_buffer) {
            cale.alloc_frame_buffer(cfg.frame_headroom_mb);
        }

        int frame = 0;
        while (t < cfg.t_end && !g_interrupted) {
            double dt = cale.step(t, cfg.t_end);
            t += dt;
            step++;
            if (step % 200 == 0) print_progress(t, cfg.t_end, step, dt, wall_start);
            bool do_diag = (step % diag_every == 0) || t >= cfg.t_end;
            bool do_vtk;
            if (vtk_by_time) {
                do_vtk = (t >= next_vtk_t) || t >= cfg.t_end;
                if (do_vtk) next_vtk_t = t + cfg.vtk_dt;
            } else {
                do_vtk = (step % vtk_every == 0) || t >= cfg.t_end;
            }
            if (do_diag) {
                auto d = cale.compute_diagnostics();
                std::fprintf(stderr, "\n");
                std::printf("Step %6d  t=%.6e dt=%.3e M=%.10e KE=%.4e IE=%.10e PE=%.10e E=%.10e |v|=%.3e\n",
                            step, t, dt, d.total_mass, d.total_KE, d.total_internal_E,
                            d.total_PE, d.total_E, d.max_v);
                std::fprintf(csv, "%d,%.10e,%.6e,%.10e,%.10e,%.10e,%.10e,%.10e,%.6e,%.6e\n",
                             step, t, dt, d.total_mass, d.total_KE, d.total_internal_E,
                             d.total_PE, d.total_E, d.max_v, d.max_mach);
                std::fflush(csv);
            }
            if (do_vtk) {
                if (cfg.frame_buffer && cale.frame_capacity > 0) {
                    if (cale.frame_count >= cale.frame_capacity) {
                        cale.flush_frames_to_disk(run_dir, Lx, Ly);
                    }
                    cale.capture_frame(t, step);
                } else {
                    ++frame;
                    char path[512];
                    std::snprintf(path, sizeof(path), "%s/output_%04d.vtk", run_dir.c_str(), frame);
                    cale.write_vtk_2d(path, Lx, Ly);
                }
            }
        }
        if (cfg.frame_buffer) {
            cale.flush_frames_to_disk(run_dir, Lx, Ly);
        }
        std::fclose(csv);
        std::fprintf(stderr, "\n");
        cale.destroy();
    } else if (cfg.solver_type == "cart_ale2") {
        // ===== cart_ale2: full periodic BC + PPM-in-remap (in development) =====
        CartAle2Solver cale;
        bool is_hse = (cfg.test_case == "hse" || cfg.test_case == "hse_perturbed"
                       || cfg.test_case == "hse_bubble");
        bool is_kh = (cfg.test_case == "kh_shear");
        bool is_kh_lec = (cfg.test_case == "kh_lecoanet");
        double Lx = 1.0;
        // Lecoanet: domain aspect 1:2 so shear layers at y=0.5, y=1.5 match
        // Athena++ iprob=4 geometry (z1=-0.5, z2=0.5 in centred coords).
        double Ly = is_kh_lec ? 2.0
                  : (is_hse || is_kh) ? 1.0
                  : 0.2;
        double gam = is_hse ? cfg.gamma : 1.4;
        cale.init(cfg.nr, cfg.ntheta, Lx, Ly, gam, cfg.cfl);
        cale.remap_order = cfg.cart_ale_remap_order;
        cale.CQ_lin  = cfg.cart_ale_cq_lin;
        cale.CQ_quad = cfg.cart_ale_cq_quad;
        cale.shear_aware_av = cfg.cart_ale_shear_aware ? 1 : 0;
        int bcm = 0;
        if (cfg.cart_ale2_bc_x == "periodic") bcm |= 1;
        if (cfg.cart_ale2_bc_y == "periodic") bcm |= 2;
        cale.bc_mode = bcm;
        cale.ppm_enabled = cfg.cart_ale2_ppm ? 1 : 0;
        if      (cfg.cart_ale2_ppm_limiter == "cs") cale.ppm_cs_limiter = 1;
        else if (cfg.cart_ale2_ppm_limiter == "cw") cale.ppm_cs_limiter = 0;
        else { std::fprintf(stderr, "unknown --ppm-limiter %s; using cs\n",
                            cfg.cart_ale2_ppm_limiter.c_str()); cale.ppm_cs_limiter = 1; }
        if      (cfg.cart_ale2_ppm_space == "prim") cale.ppm_primitive = 1;
        else if (cfg.cart_ale2_ppm_space == "cons") cale.ppm_primitive = 0;
        else { std::fprintf(stderr, "unknown --ppm-space %s; using prim\n",
                            cfg.cart_ale2_ppm_space.c_str()); cale.ppm_primitive = 1; }
        cale.ppm_char = cfg.cart_ale2_ppm_char ? 1 : 0;
        if      (cfg.cart_ale_limiter == "minmod")  cale.remap_limiter = 0;
        else if (cfg.cart_ale_limiter == "vanleer") cale.remap_limiter = 1;
        else if (cfg.cart_ale_limiter == "mc")      cale.remap_limiter = 2;
        else { std::fprintf(stderr, "unknown --remap-limiter %s; using vanleer\n",
                            cfg.cart_ale_limiter.c_str()); cale.remap_limiter = 1; }
        const char* lim_name = cale.remap_limiter == 0 ? "minmod"
                             : cale.remap_limiter == 1 ? "vanleer" : "mc";
        const char* recon_name;
        if (cale.remap_order < 2)      recon_name = "donor-cell";
        else if (!cale.ppm_enabled)    recon_name = "MUSCL-in-remap";
        else if (!cale.ppm_primitive)  recon_name = cale.ppm_cs_limiter ? "PPM/CS/cons" : "PPM/CW/cons";
        else if (cale.ppm_char)        recon_name = cale.ppm_cs_limiter ? "PPM/CS/char" : "PPM/CW/char";
        else                           recon_name = cale.ppm_cs_limiter ? "PPM/CS/prim" : "PPM/CW/prim";
        std::fprintf(stderr,
            "  CartAle2 remap_order = %d (%s)  limiter = %s  CQ_lin=%g CQ_quad=%g  shear_aware_av=%d  bc=(%s,%s)\n",
            cale.remap_order, recon_name,
            lim_name, cale.CQ_lin, cale.CQ_quad, cale.shear_aware_av,
            cfg.cart_ale2_bc_x.c_str(), cfg.cart_ale2_bc_y.c_str());
        if (cfg.test_case == "hse_bubble") {
            std::vector<CartAle2Solver::Bubble> blist;
            if (!cfg.bubbles.empty()) {
                for (const auto& b : cfg.bubbles)
                    blist.push_back({b[0], b[1], b[2], b[3], b[4]});
            } else {
                blist.push_back({cfg.bubble_xc, cfg.bubble_yc, cfg.bubble_rb,
                                 cfg.bubble_alpha, cfg.bubble_beta});
            }
            cale.init_hse_bubbles(1.0, 1.0, blist);
        } else if (cfg.test_case == "hse_perturbed") {
            cale.init_hse_polytrope(1.0, 1.0, cfg.perturb_amplitude);
        } else if (cfg.test_case == "hse") {
            cale.init_hse_polytrope(1.0, 1.0, 0.0);
        } else if (cfg.test_case == "kh_shear") {
            double amp = (cfg.perturb_amplitude > 0) ? cfg.perturb_amplitude : 0.1;
            cale.init_kh_shear(1.0, 2.0, 2.5, 0.5, amp, 2);
        } else if (cfg.test_case == "kh_lecoanet") {
            // Athena++ pgen/kh.cpp iprob=4 canonical parameters:
            //   vflow=1, amp=0.01, drho_rho0=0 (unstratified), P0=10.
            // --perturb overrides amp. Requires periodic BC in both x and y.
            double amp = (cfg.perturb_amplitude >= 0) ? cfg.perturb_amplitude : 0.01;
            if (cale.bc_mode != 3) {
                std::fprintf(stderr,
                    "  [warn] kh_lecoanet expects --bc-x periodic --bc-y periodic; "
                    "current bc_mode=%d\n", cale.bc_mode);
            }
            int k = (cfg.cart_ale2_kh_k > 0) ? cfg.cart_ale2_kh_k : 1;
            cale.init_kh_lecoanet(1.0, amp, 0.0, 10.0, k);
        } else {
            cale.init_sod();
        }

        std::timespec wall_start;
        clock_gettime(CLOCK_MONOTONIC, &wall_start);

        char csv_path[512];
        std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", run_dir.c_str());
        std::FILE* csv = std::fopen(csv_path, "w");
        std::fprintf(csv, "step,t,dt,mass,KE,IE,PE,E,max_v,max_mach\n");

        int diag_every = cfg.diag_interval > 0 ? cfg.diag_interval : cfg.output_interval;
        int vtk_every  = cfg.vtk_interval  > 0 ? cfg.vtk_interval  : cfg.output_interval;
        bool vtk_by_time = cfg.vtk_dt > 0.0;
        double next_vtk_t = 0.0;
        if (vtk_by_time) {
            std::fprintf(stderr,
                "  CartAle2 diag every %d steps, VTK every dt=%g (physical time)%s\n",
                diag_every, cfg.vtk_dt,
                cfg.frame_buffer ? " (VRAM buffered)" : "");
        } else {
            std::fprintf(stderr, "  CartAle2 diag every %d steps, VTK every %d steps%s\n",
                         diag_every, vtk_every,
                         cfg.frame_buffer ? " (VRAM buffered)" : "");
        }
        if (cfg.frame_buffer) cale.alloc_frame_buffer(cfg.frame_headroom_mb);

        int frame = 0;
        while (t < cfg.t_end && !g_interrupted) {
            double dt = cale.step(t, cfg.t_end);
            t += dt;
            step++;
            if (step % 200 == 0) print_progress(t, cfg.t_end, step, dt, wall_start);
            bool do_diag = (step % diag_every == 0) || t >= cfg.t_end;
            bool do_vtk;
            if (vtk_by_time) {
                do_vtk = (t >= next_vtk_t) || t >= cfg.t_end;
                if (do_vtk) next_vtk_t = t + cfg.vtk_dt;
            } else {
                do_vtk = (step % vtk_every == 0) || t >= cfg.t_end;
            }
            if (do_diag) {
                auto d = cale.compute_diagnostics();
                std::fprintf(stderr, "\n");
                std::printf("Step %6d  t=%.6e dt=%.3e M=%.10e KE=%.4e IE=%.10e PE=%.10e E=%.10e |v|=%.3e\n",
                            step, t, dt, d.total_mass, d.total_KE, d.total_internal_E,
                            d.total_PE, d.total_E, d.max_v);
                std::fprintf(csv, "%d,%.10e,%.6e,%.10e,%.10e,%.10e,%.10e,%.10e,%.6e,%.6e\n",
                             step, t, dt, d.total_mass, d.total_KE, d.total_internal_E,
                             d.total_PE, d.total_E, d.max_v, d.max_mach);
                std::fflush(csv);
            }
            if (do_vtk) {
                if (cfg.frame_buffer && cale.frame_capacity > 0) {
                    if (cale.frame_count >= cale.frame_capacity)
                        cale.flush_frames_to_disk(run_dir, Lx, Ly);
                    cale.capture_frame(t, step);
                } else {
                    ++frame;
                    char path[512];
                    std::snprintf(path, sizeof(path), "%s/output_%04d.vtk", run_dir.c_str(), frame);
                    cale.write_vtk_2d(path, Lx, Ly);
                }
            }
        }
        if (cfg.frame_buffer) cale.flush_frames_to_disk(run_dir, Lx, Ly);
        std::fclose(csv);
        std::fprintf(stderr, "\n");
        cale.destroy();
    } else if (cfg.solver_type == "pseudo_spectral") {
        // ===== 偽譜法 2D 不可壓 NS (渦度-流函數, cuFFT) =====
        if (cfg.test_case != "kh_shear" && cfg.test_case != "forced_turb"
            && cfg.test_case != "taylor_green"
            && cfg.test_case != "double_shear_layer"
            && cfg.test_case != "vortex_merger"
            && cfg.test_case != "quad_vortex_merger") {
            std::fprintf(stderr,
                "ERROR: pseudo_spectral supports --test {kh_shear, forced_turb, taylor_green, "
                "double_shear_layer, vortex_merger, quad_vortex_merger}\n");
            return 1;
        }
        PseudoSpectralSolver ps;
        ps.use_ifrk         = !cfg.ps_explicit;
        ps.use_skew         = !cfg.ps_adv_only && !cfg.ps_conservative;
        ps.use_conservative = cfg.ps_conservative && !cfg.ps_adv_only;
        ps.use_batched_fft  = cfg.ps_batched_fft;
        ps.drag_alpha       = cfg.ps_drag_alpha;
        ps.hyper_p          = std::max(1, cfg.ps_hyper_p);
        ps.use_pi_dt        = cfg.ps_pi_dt;
        ps.forcing_use_curand = !cfg.ps_forcing_host_rng;
        ps.init(cfg.nr, cfg.ntheta, cfg.ps_Lx, cfg.ps_Ly, cfg.ps_nu, cfg.cfl);
        if (cfg.test_case == "kh_shear") {
            double amp = (cfg.perturb_amplitude > 0) ? cfg.perturb_amplitude : 1e-2;
            ps.init_kh_shear(cfg.ps_vshear, amp, cfg.ps_k);
        } else if (cfg.test_case == "taylor_green") {
            ps.init_taylor_green(cfg.ps_tg_k);
        } else if (cfg.test_case == "double_shear_layer") {
            double amp = (cfg.perturb_amplitude > 0) ? cfg.perturb_amplitude : 5e-2;
            ps.init_double_shear_layer(cfg.ps_vshear, amp, cfg.ps_k);
        } else if (cfg.test_case == "vortex_merger") {
            ps.init_vortex_merger(cfg.ps_vm_gamma, cfg.ps_vm_sigma, cfg.ps_vm_dist);
        } else if (cfg.test_case == "quad_vortex_merger") {
            ps.init_quad_vortex_merger(cfg.ps_vm_gamma, cfg.ps_vm_sigma, cfg.ps_vm_dist);
        } else {
            // forced_turb: zero IC + stochastic forcing
            ps.init_zero();
            if (cfg.ps_forcing_eps <= 0.0) {
                std::fprintf(stderr,
                    "ERROR: --test forced_turb requires --ps-forcing-eps > 0\n");
                return 1;
            }
        }
        if (cfg.ps_forcing_eps > 0.0) {
            ps.init_forcing(cfg.ps_forcing_kf, cfg.ps_forcing_dk,
                            cfg.ps_forcing_eps, cfg.ps_forcing_seed);
        }
        // Restart (若提供):覆寫 IC 的 ω̂,更新 step/t。
        if (!cfg.ps_resume.empty()) {
            double t_ckpt = 0.0;
            if (ps.load_checkpoint(cfg.ps_resume, t_ckpt)) {
                t = t_ckpt;
                step = ps.step_count;
                // Taylor-Green IC-比對 buffer 在 restart 後對非-TG run 沒意義;
                // 對 TG run,IC 譜仍來自 init_taylor_green 的初次呼叫 → 已保存。
            } else {
                std::fprintf(stderr, "ERROR: --ps-resume failed\n");
                return 1;
            }
        }

        std::timespec wall_start;
        clock_gettime(CLOCK_MONOTONIC, &wall_start);

        char csv_path[512];
        std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", run_dir.c_str());
        std::FILE* csv = std::fopen(csv_path, "w");
        std::fprintf(csv,
            "step,t,dt,KE,enstrophy,max_v,max_omega,eps_KE,eps_enstrophy,err_L2\n");

        int diag_every = cfg.diag_interval > 0 ? cfg.diag_interval : cfg.output_interval;
        int vtk_every  = cfg.vtk_interval  > 0 ? cfg.vtk_interval  : cfg.output_interval;
        bool vtk_by_time = cfg.vtk_dt > 0.0;
        double next_vtk_t = 0.0;
        if (cfg.frame_buffer) ps.alloc_frame_buffer(cfg.frame_headroom_mb);

        int frame = 0;
        // t=0 幀 (便於 renderer 取首幀參考)
        if (cfg.frame_buffer && ps.frame_capacity > 0) ps.capture_frame(0.0, 0);
        else {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/output_%04d.vtk", run_dir.c_str(), ++frame);
            ps.write_vtk_2d(path);
        }

        while (t < cfg.t_end && !g_interrupted) {
            double dt = ps.step();
            t += dt;
            step++;
            if (step % 200 == 0) print_progress(t, cfg.t_end, step, dt, wall_start);

            bool do_diag = (step % diag_every == 0) || t >= cfg.t_end;
            bool do_vtk;
            if (vtk_by_time) {
                do_vtk = (t >= next_vtk_t) || t >= cfg.t_end;
                if (do_vtk) next_vtk_t = t + cfg.vtk_dt;
            } else {
                do_vtk = (step % vtk_every == 0) || t >= cfg.t_end;
            }

            if (do_diag) {
                auto d = ps.compute_diagnostics(t);
                std::fprintf(stderr, "\n");
                if (cfg.test_case == "taylor_green") {
                    std::printf("Step %6d  t=%.6e dt=%.3e KE=%.6e Ω=%.6e |v|=%.3e |ω|=%.3e εKE=%.3e errL2=%.3e\n",
                                step, t, dt, d.total_KE, d.total_enstrophy,
                                d.max_v, d.max_omega, d.eps_KE, d.err_L2);
                } else {
                    std::printf("Step %6d  t=%.6e dt=%.3e KE=%.6e Ω=%.6e |v|=%.3e |ω|=%.3e εKE=%.3e εΩ=%.3e\n",
                                step, t, dt, d.total_KE, d.total_enstrophy,
                                d.max_v, d.max_omega, d.eps_KE, d.eps_enstrophy);
                }
                std::fprintf(csv, "%d,%.10e,%.6e,%.10e,%.10e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                             step, t, dt, d.total_KE, d.total_enstrophy,
                             d.max_v, d.max_omega, d.eps_KE, d.eps_enstrophy,
                             d.err_L2);
                std::fflush(csv);
            }
            if (cfg.ps_ckpt_every > 0 && step % cfg.ps_ckpt_every == 0) {
                char cpath[512];
                std::snprintf(cpath, sizeof(cpath), "%s/checkpoint.bin",
                              run_dir.c_str());
                ps.save_checkpoint(cpath, t);
            }
            if (do_vtk) {
                if (cfg.frame_buffer && ps.frame_capacity > 0) {
                    if (ps.frame_count >= ps.frame_capacity)
                        ps.flush_frames_to_disk(run_dir);
                    ps.capture_frame(t, step);
                } else {
                    ++frame;
                    char path[512];
                    std::snprintf(path, sizeof(path), "%s/output_%04d.vtk",
                                  run_dir.c_str(), frame);
                    ps.write_vtk_2d(path);
                }
            }
        }
        if (cfg.frame_buffer) ps.flush_frames_to_disk(run_dir);
        std::fclose(csv);
        std::fprintf(stderr, "\n");
        ps.destroy();
    } else if (cfg.solver_type == "anelastic_sl") {
        // ===== Anelastic SL-spectral solver (Phase 1b: Poisson + self-test) =====
        if (cfg.test_case != "sl_basis_check"
            && cfg.test_case != "sl_poisson_test"
            && cfg.test_case != "sl_poisson_test_boussinesq"
            && cfg.test_case != "kh_shear_boussinesq"
            && cfg.test_case != "gmode_pulsation") {
            std::fprintf(stderr,
                "ERROR: anelastic_sl supports --test {sl_basis_check, "
                "sl_poisson_test[_boussinesq], kh_shear_boussinesq, "
                "gmode_pulsation}\n");
            return 1;
        }
        AnelasticSLSolver ansl;
        int n_modes = (cfg.nr > 0 ? cfg.nr / 2 : 128);
        ansl.init(cfg.ntheta, cfg.nr, n_modes,
                  cfg.ps_Lx, cfg.ps_Ly, cfg.ps_nu, cfg.cfl);

        std::string bg = (cfg.test_case == "kh_shear_boussinesq"
                          || cfg.test_case == "sl_poisson_test_boussinesq")
                         ? "boussinesq" : "lane_emden_1_5";
        ansl.set_background(bg, 0.01);

        if (cfg.test_case == "sl_poisson_test"
            || cfg.test_case == "sl_poisson_test_boussinesq") {
            ansl.manufactured_test();
        }

        if (cfg.test_case == "gmode_pulsation") {
            // Phase 1d: Lane-Emden n=3/2 background, k_y=1 sinusoid seed.
            double amp = (cfg.perturb_amplitude > 0) ? cfg.perturb_amplitude : 1e-3;
            int k_y = (cfg.ps_k > 0) ? cfg.ps_k : 1;
            ansl.init_gmode_pulsation(amp, k_y);
            std::fprintf(stderr,
                "  AnelasticSL gmode run: tend=%g, cfl=%g, amp=%g, k_y=%d\n",
                cfg.t_end, cfg.cfl, amp, k_y);
            double t = 0.0;
            int step = 0;
            std::timespec wall_start;
            clock_gettime(CLOCK_MONOTONIC, &wall_start);

            char probe_path[512];
            std::snprintf(probe_path, sizeof(probe_path), "%s/gmode_probe.csv",
                          run_dir.c_str());
            FILE* probe = std::fopen(probe_path, "w");
            std::fprintf(probe, "# t  v_center\n");

            while (t < cfg.t_end && !g_interrupted) {
                double dt = ansl.step();
                if (t + dt > cfg.t_end) dt = cfg.t_end - t;
                t += dt;
                ++step;
                double v_c = ansl.probe_v_center();
                std::fprintf(probe, "%.10e %.10e\n", t, v_c);
                if (step % 200 == 0 || t >= cfg.t_end) {
                    print_progress(t, cfg.t_end, step, dt, wall_start);
                }
            }
            std::fclose(probe);
            std::fprintf(stderr, "  gmode probe written to %s (%d samples)\n",
                         probe_path, step);
        }

        if (cfg.test_case == "kh_shear_boussinesq") {
            double amp = (cfg.perturb_amplitude > 0) ? cfg.perturb_amplitude : 1e-2;
            ansl.init_kh_shear(cfg.ps_vshear, amp, cfg.ps_k);
            std::fprintf(stderr,
                "  AnelasticSL KH run: tend=%g, cfl=%g, ν=%g, k=%d\n",
                cfg.t_end, cfg.cfl, cfg.ps_nu, cfg.ps_k);
            double t = 0.0;
            int step = 0;
            std::timespec wall_start;
            clock_gettime(CLOCK_MONOTONIC, &wall_start);
            while (t < cfg.t_end && !g_interrupted) {
                double dt = ansl.step();
                if (t + dt > cfg.t_end) dt = cfg.t_end - t;
                t += dt;
                ++step;
                if (step % 50 == 0 || t >= cfg.t_end) {
                    print_progress(t, cfg.t_end, step, dt, wall_start);
                }
            }
            // Dump final ω(x,y) to CSV for Python comparison.
            std::vector<double> h_omega;
            ansl.download_omega(h_omega);
            std::vector<double> h_u, h_v;
            ansl.download_uv(h_u, h_v);
            std::vector<double> y_cgl;
            ansl.download_y(y_cgl);
            std::vector<double> h_div;
            ansl.download_divergence(h_div);
            double max_div = 0.0;
            for (double d : h_div) if (std::fabs(d) > max_div) max_div = std::fabs(d);
            std::fprintf(stderr, "  final |∇·u|∞ (C++ side) = %.3e\n", max_div);
            char csv_path[512];
            std::snprintf(csv_path, sizeof(csv_path), "%s/kh_final.csv", run_dir.c_str());
            FILE* fp = std::fopen(csv_path, "w");
            std::fprintf(fp, "# nx=%d, ny=%d, Lx=%g, Ly=%g, nu=%g, t=%g, steps=%d\n",
                         ansl.nx, ansl.ny, ansl.Lx, ansl.Ly, ansl.nu, t, step);
            std::fprintf(fp, "# y_cgl (ny values):\n");
            for (int i = 0; i < ansl.ny; ++i)
                std::fprintf(fp, "%.15e\n", y_cgl[i]);
            std::fprintf(fp, "# omega (ny × nx row-major):\n");
            for (int i = 0; i < ansl.ncell; ++i)
                std::fprintf(fp, "%.15e\n", h_omega[i]);
            std::fprintf(fp, "# u (ny × nx row-major):\n");
            for (int i = 0; i < ansl.ncell; ++i)
                std::fprintf(fp, "%.15e\n", h_u[i]);
            std::fprintf(fp, "# v (ny × nx row-major):\n");
            for (int i = 0; i < ansl.ncell; ++i)
                std::fprintf(fp, "%.15e\n", h_v[i]);
            std::fclose(fp);
            std::fprintf(stderr, "  KH final state written to %s\n", csv_path);
        }

        // Phase 1a: dump SL basis to CSV for offline verification.
        char basis_path[512];
        std::snprintf(basis_path, sizeof(basis_path), "%s/sl_basis.csv", run_dir.c_str());
        FILE* bf = std::fopen(basis_path, "w");
        std::fprintf(bf, "# ny=%d, n_modes=%d, Ly=%g, background=%s\n",
                     ansl.ny, ansl.n_modes, ansl.Ly, bg.c_str());
        std::fprintf(bf, "# mu_n (n_modes values):\n");
        for (int m = 0; m < ansl.n_modes; ++m)
            std::fprintf(bf, "%.15e\n", ansl.h_mu[m]);
        std::fprintf(bf, "# y_cgl (ny values):\n");
        for (int i = 0; i < ansl.ny; ++i)
            std::fprintf(bf, "%.15e\n", ansl.h_y_cgl[i]);
        std::fprintf(bf, "# W_tilde (ny values):\n");
        for (int i = 0; i < ansl.ny; ++i)
            std::fprintf(bf, "%.15e\n", ansl.h_W_tilde[i]);
        std::fclose(bf);
        std::fprintf(stderr, "  SL basis written to %s\n", basis_path);
        std::fprintf(stderr, "\n");
    } else if (cfg.solver_type == "sph2d_spectral") {
        // ===== 2D 薄球殼 barotropic 偽譜 =====
        if (cfg.test_case != "rossby_wave" && cfg.test_case != "jovian_bands") {
            std::fprintf(stderr,
                "ERROR: sph2d_spectral supports --test {rossby_wave, jovian_bands}\n");
            return 1;
        }
        Sph2DSpectralSolver sph;
        sph.drag_alpha = cfg.sph_drag;
        sph.hyper_p    = std::max(1, cfg.sph_hyper);
        sph.use_pi_dt  = cfg.sph_pi_dt;
        int Lmax_req = (cfg.sph_Lmax > 0) ? cfg.sph_Lmax
                       : std::min(cfg.nr - 1, cfg.ntheta / 2 - 2);
        sph.init(cfg.nr, cfg.ntheta, Lmax_req,
                 cfg.sph_R, cfg.sph_Omega, cfg.sph_nu, cfg.cfl);
        if (cfg.test_case == "rossby_wave") {
            sph.init_rossby_wave(cfg.sph_rossby_l, cfg.sph_rossby_m,
                                 cfg.sph_rossby_amp);
        } else {
            sph.init_zero();
            if (cfg.sph_forcing_eps <= 0.0) {
                std::fprintf(stderr,
                    "ERROR: --test jovian_bands requires --sph-forcing-eps > 0\n");
                return 1;
            }
        }
        if (cfg.sph_forcing_eps > 0.0) {
            sph.init_forcing(cfg.sph_forcing_lmin, cfg.sph_forcing_lmax,
                             cfg.sph_forcing_eps, cfg.sph_forcing_seed);
        }
        if (!cfg.sph_resume.empty()) {
            double t_ckpt = 0.0;
            if (sph.load_checkpoint(cfg.sph_resume, t_ckpt)) {
                t = t_ckpt;
                step = sph.step_count;
            } else {
                std::fprintf(stderr, "ERROR: --sph-resume failed\n");
                return 1;
            }
        }

        std::timespec wall_start;
        clock_gettime(CLOCK_MONOTONIC, &wall_start);

        char csv_path[512];
        std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", run_dir.c_str());
        std::FILE* csv = std::fopen(csv_path, "w");
        std::fprintf(csv, "step,t,dt,KE,enstrophy,max_v,max_zeta,err_L2\n");

        int diag_every = cfg.diag_interval > 0 ? cfg.diag_interval : cfg.output_interval;
        int vtk_every  = cfg.vtk_interval  > 0 ? cfg.vtk_interval  : cfg.output_interval;
        bool vtk_by_time = cfg.vtk_dt > 0.0;
        double next_vtk_t = 0.0;
        if (cfg.frame_buffer) sph.alloc_frame_buffer(cfg.frame_headroom_mb);

        int frame = 0;
        if (cfg.frame_buffer && sph.frame_capacity > 0) sph.capture_frame(0.0, 0);
        else {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/output_%04d.vtk", run_dir.c_str(), ++frame);
            sph.write_vtk_2d(path);
        }

        while (t < cfg.t_end && !g_interrupted) {
            double dt = sph.step();
            t += dt;
            step++;
            if (step % 200 == 0) print_progress(t, cfg.t_end, step, dt, wall_start);

            bool do_diag = (step % diag_every == 0) || t >= cfg.t_end;
            bool do_vtk;
            if (vtk_by_time) {
                do_vtk = (t >= next_vtk_t) || t >= cfg.t_end;
                if (do_vtk) next_vtk_t = t + cfg.vtk_dt;
            } else {
                do_vtk = (step % vtk_every == 0) || t >= cfg.t_end;
            }

            if (do_diag) {
                auto d = sph.compute_diagnostics(t);
                std::fprintf(stderr, "\n");
                std::printf("Step %6d  t=%.6e dt=%.3e KE=%.6e Ω=%.6e |v|=%.3e |ζ|=%.3e err=%.3e\n",
                            step, t, dt, d.total_KE, d.total_enstrophy,
                            d.max_v, d.max_zeta, d.err_L2);
                std::fprintf(csv, "%d,%.10e,%.6e,%.10e,%.10e,%.6e,%.6e,%.6e\n",
                             step, t, dt, d.total_KE, d.total_enstrophy,
                             d.max_v, d.max_zeta, d.err_L2);
                std::fflush(csv);
            }
            if (cfg.sph_ckpt_every > 0 && step % cfg.sph_ckpt_every == 0) {
                char cpath[512];
                std::snprintf(cpath, sizeof(cpath), "%s/checkpoint.bin",
                              run_dir.c_str());
                sph.save_checkpoint(cpath, t);
            }
            if (do_vtk) {
                if (cfg.frame_buffer && sph.frame_capacity > 0) {
                    if (sph.frame_count >= sph.frame_capacity)
                        sph.flush_frames_to_disk(run_dir);
                    sph.capture_frame(t, step);
                } else {
                    ++frame;
                    char path[512];
                    std::snprintf(path, sizeof(path), "%s/output_%04d.vtk",
                                  run_dir.c_str(), frame);
                    sph.write_vtk_2d(path);
                }
            }
        }
        if (cfg.frame_buffer) sph.flush_frames_to_disk(run_dir);
        std::fclose(csv);
        std::fprintf(stderr, "\n");
        sph.destroy();
    } else if (cfg.solver_type == "ale2d") {
        // ===== 2D axisymmetric Lagrangian (Caramana compatible) =====
        if (cfg.test_case != "lane_emden" && cfg.test_case != "lane_emden_perturbed") {
            std::fprintf(stderr, "ERROR: ale2d currently supports lane_emden / lane_emden_perturbed only\n");
            return 1;
        }
        Ale2DSolver ale;
        ale.init(grid, eos, cfg.G, cfg.cfl);
        ale.init_lane_emden(1.0, 1.0, 1.5);
        ale.snapshot_hse();
        if (cfg.test_case == "lane_emden_perturbed")
            ale.apply_perturbation(cfg.perturb_amplitude);

        std::timespec wall_start;
        clock_gettime(CLOCK_MONOTONIC, &wall_start);

        char csv_path[512];
        std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", run_dir.c_str());
        std::FILE* csv = std::fopen(csv_path, "w");
        std::fprintf(csv, "step,t,dt,mass,KE,IE,PE,total_E,max_mach,max_v\n");

        int frame = 0;
        while (t < cfg.t_end && !g_interrupted) {
            double dt = ale.step(t, cfg.t_end);
            t += dt;
            step++;

            if (step % 200 == 0) print_progress(t, cfg.t_end, step, dt, wall_start);

            if (step % cfg.output_interval == 0) {
                auto d = ale.compute_diagnostics();
                std::fprintf(stderr, "\n");
                std::printf("Step %6d  t=%.6e dt=%.3e M=%.10e E=%.10e |v|_max=%.3e Mach_max=%.3e\n",
                            step, t, dt, d.total_mass, d.total_E, d.max_v, d.max_mach);
                std::fprintf(csv, "%d,%.10e,%.6e,%.10e,%.10e,%.10e,%.10e,%.10e,%.6e,%.6e\n",
                             step, t, dt, d.total_mass, d.total_KE, d.total_internal_E,
                             d.total_grav_E, d.total_E, d.max_mach, d.max_v);
                std::fflush(csv);

                std::vector<double> rp, rhop, Pp, ep, vrp;
                ale.download_radial_profile(rp, rhop, Pp, ep, vrp);
                char path[512];
                std::snprintf(path, sizeof(path), "%s/profile_%04d.txt", run_dir.c_str(), ++frame);
                std::FILE* fp = std::fopen(path, "w");
                std::fprintf(fp, "# t = %.10e  step = %d\n# ic r rho P e_int v_r\n", t, step);
                for (int ic = 0; ic < ale.nr; ++ic)
                    std::fprintf(fp, "%d %.10e %.10e %.10e %.10e %.10e\n",
                                 ic, rp[ic], rhop[ic], Pp[ic], ep[ic], vrp[ic]);
                std::fclose(fp);
            }
        }
        std::fclose(csv);
        std::fprintf(stderr, "\n");
        ale.destroy();
    } else if (cfg.solver_type == "wb2d") {
        // ===== Well-Balanced 2D Eulerian (MESA-stabilized) =====
        Wb2DSolver wb;
        wb.limiter_type = static_cast<int>(cfg.limiter);
        wb.use_lm_hllc = cfg.lm_hllc ? 1 : 0;
        wb.init(grid, eos, cfg.G, cfg.cfl);
        if (cfg.mesh_type == "mass") {
            wb.n_pole_avg = cfg.ntheta / 2;
            if (cfg.r_inner <= 0) {
                int n_uni = static_cast<int>(0.15 * cfg.R_outer / (cfg.R_outer / cfg.nr));
                wb.n_angular_avg = n_uni;
                wb.central_damp_r = 0.15 * cfg.R_outer;
            }
        }
        if (!cfg.no_sponge) wb.sponge_kappa = 100.0;

        if (cfg.test_case == "lane_emden_perturbed" || cfg.test_case == "bubble") {
            State state_hse;
            state_hse.allocate(grid);
            LaneEmdenParams lep;
            lep.n_poly = 1.5; lep.rho_c = 1.0; lep.K_poly = 1.0; lep.G = cfg.G;
            init_lane_emden(grid, state_hse, lep, cfg.gamma);
            wb.upload_state(grid, state_hse);
            wb.snapshot_hse();
        }
        wb.upload_state(grid, state);
        if (!(cfg.test_case == "lane_emden_perturbed" || cfg.test_case == "bubble"))
            wb.snapshot_hse();

        std::timespec wall_start;
        clock_gettime(CLOCK_MONOTONIC, &wall_start);

        while (t < cfg.t_end && !g_interrupted) {
            double dt = wb.step(t, cfg.t_end);
            t += dt;
            step++;

            if (step % 200 == 0)
                print_progress(t, cfg.t_end, step, dt, wall_start);

            if (step % cfg.output_interval == 0) {
                wb.download_state(grid, state);
                Diagnostics diag = compute_diagnostics(grid, state, cfg.gamma);
                std::fprintf(stderr, "\n");
                std::printf("Step %6d  t = %.6e  dt = %.3e  M = %.10e  E = %.10e\n",
                            step, t, dt, diag.total_mass, diag.total_energy);
                char fname[512];
                std::snprintf(fname, sizeof(fname), "%s/output_%04d.vtk",
                              run_dir.c_str(), step / cfg.output_interval);
                write_vtk(fname, grid, state, cfg.gamma);
            }
        }
        std::fprintf(stderr, "\n");
        wb.download_state(grid, state);
        wb.destroy();
    } else if (cfg.solver_type == "lowmach") {
        // ===== GPU low-Mach path =====
        PrecondType pc = PrecondType::LINE_JACOBI;
        if (cfg.precond == "none")          pc = PrecondType::NONE;
        else if (cfg.precond == "block_jacobi") pc = PrecondType::BLOCK_JACOBI;
        else if (cfg.precond == "simple")   pc = PrecondType::SIMPLE;
        else if (cfg.precond == "line_jacobi") pc = PrecondType::LINE_JACOBI;
        else if (cfg.precond == "block_schur") pc = PrecondType::BLOCK_SCHUR;
        else if (cfg.precond == "combined") pc = PrecondType::COMBINED;
        else if (cfg.precond == "pbp")      pc = PrecondType::PBP;

        LowMachSolver lm;
        lm.init(grid, eos, cfg.G, cfg.cfl, pc);
        if (cfg.no_sponge) lm.sponge_kappa = 0.0;
        configure_mass_mesh(lm);
        snapshot_hse_if_needed(lm);
        lm.upload_state(grid, state);

        std::timespec wall_start;
        clock_gettime(CLOCK_MONOTONIC, &wall_start);

        while (t < cfg.t_end && !g_interrupted) {
            double dt = lm.step(t, cfg.t_end);
            t += dt;
            step++;

            if (step % 200 == 0)
                print_progress(t, cfg.t_end, step, dt, wall_start);

            if (step % cfg.output_interval == 0) {
                lm.download_state(grid, state);
                Diagnostics diag = compute_diagnostics(grid, state, cfg.gamma);

                double max_vr = 0, max_vt = 0;
                double rho_thresh = lm.atm_rho_thresh;
                for (int i = 0; i < grid.nr; i++)
                    for (int j = 0; j < grid.ntheta; j++) {
                        int k = grid.idx(i, j);
                        if (state.rho[k] < rho_thresh) continue;
                        double rho = state.rho[k];
                        max_vr = std::max(max_vr, std::fabs(state.mr[k] / rho));
                        max_vt = std::max(max_vt, std::fabs(state.mtheta[k] / rho));
                    }

                std::fprintf(stderr, "\n");
                std::printf("Step %6d  t = %.6e  dt = %.3e  M = %.10e  E = %.10e  |vr|=%.3e |vt|=%.3e\n",
                            step, t, dt, diag.total_mass, diag.total_energy, max_vr, max_vt);

                char fname[512];
                std::snprintf(fname, sizeof(fname), "%s/output_%04d.vtk", run_dir.c_str(), step / cfg.output_interval);
                write_vtk(fname, grid, state, cfg.gamma);
            }
        }
        std::fprintf(stderr, "\n");

        lm.download_state(grid, state);
        lm.destroy();
    } else {
#ifdef USE_AMGX
        // ===== GPU compressible path (HLLC + JFNK) =====
        GpuSolver gpu;
        gpu.init(grid, eos, cfg.G, cfg.cfl, cfg.limiter);
        gpu.upload_state(grid, state);

        SolverOps ops;
        ops.step = [&](double t_, double te) { return gpu.step(t_, te); };
        ops.download = [&](const Grid& g, State& s, double) { gpu.download_state(g, s); };
        ops.destroy = [&]() { gpu.destroy(); };
        run_time_loop(ops);
#else
        std::fprintf(stderr, "ERROR: --solver compressible requires AmgX. "
                     "Rebuild with -DAMGX_DIR=/path/to/amgx, or use --solver lowmach.\n");
        return 1;
#endif
    }

#else
    // ===== CPU path =====
    State state_tmp;
    state_tmp.allocate(grid);

    PoissonGMG poisson_solver;
    poisson_solver.init(grid);

    FluxAccumulator acc;
    acc.allocate(grid.total_cells());

    std::vector<double> rho_cells;
    std::vector<double> poisson_rhs;

    std::timespec wall_start;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);

    while (t < cfg.t_end && !g_interrupted) {
        fill_ghost_cells(grid, state, cfg.gamma);

        double dt = compute_cfl_dt(grid, state, eos, cfg.cfl);
        if (t + dt > cfg.t_end) dt = cfg.t_end - t;

        state_tmp.copy_from(state);

        compute_rhs(grid, state, eos, acc, cfg.limiter);
        extract_density(grid, state, rho_cells);
        compute_poisson_rhs(grid, rho_cells, cfg.G, poisson_rhs);
        poisson_solver.solve(poisson_rhs.data(), state.phi.data());
        add_gravity_source(grid, state, acc);
        rk2_substep(grid, state, acc, dt);

        fill_ghost_cells(grid, state, cfg.gamma);
        compute_rhs(grid, state, eos, acc, cfg.limiter);
        extract_density(grid, state, rho_cells);
        compute_poisson_rhs(grid, rho_cells, cfg.G, poisson_rhs);
        poisson_solver.solve(poisson_rhs.data(), state.phi.data());
        add_gravity_source(grid, state, acc);
        rk2_substep(grid, state, acc, dt);

        int nr = grid.nr, nt = grid.ntheta;
        for (int i = 0; i < nr; ++i) {
            for (int j = 0; j < nt; ++j) {
                int k = grid.idx(i, j);
                state.rho[k] = 0.5 * (state_tmp.rho[k] + state.rho[k]);
                state.mr[k] = 0.5 * (state_tmp.mr[k] + state.mr[k]);
                state.mtheta[k] = 0.5 * (state_tmp.mtheta[k] + state.mtheta[k]);
                state.E[k] = 0.5 * (state_tmp.E[k] + state.E[k]);
            }
        }

        t += dt;
        step++;

        if (step % 200 == 0)
            print_progress(t, cfg.t_end, step, dt, wall_start);

        if (step % cfg.output_interval == 0) {
            std::fprintf(stderr, "\n");
            Diagnostics diag = compute_diagnostics(grid, state, cfg.gamma);
            std::printf("Step %6d  t = %.6e  dt = %.3e  M = %.10e  E = %.10e\n",
                        step, t, dt, diag.total_mass, diag.total_energy);

            char fname[512];
            std::snprintf(fname, sizeof(fname), "%s/output_%04d.vtk", run_dir.c_str(), step / cfg.output_interval);
            write_vtk(fname, grid, state, cfg.gamma);
        }
    }
    std::fprintf(stderr, "\n");
#endif

    Diagnostics diag = compute_diagnostics(grid, state, cfg.gamma);
    std::printf("Final: step %d  t = %.6e  M = %.10e  E = %.10e\n",
                step, t, diag.total_mass, diag.total_energy);
    {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/output_final.vtk", run_dir.c_str());
        write_vtk(path, grid, state, cfg.gamma);
    }

    std::printf("Done.\n");
    return 0;
}
