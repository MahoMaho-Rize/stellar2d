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
#include "gpu_solver.h"
#endif
#include "lowmach_solver.h"
#include "fas_solver.cuh"
#include "fas2_solver.cuh"
#include "simple_solver.cuh"
#include "projection_solver.cuh"
#include "radial1d_solver.cuh"
#include "wb2d_solver.cuh"
#include "ale2d_solver.cuh"
#include "cart_lag_solver.cuh"
#include "cart_ale_solver.cuh"
#include "cart_ale2_solver.cuh"
#include "cart_impl_solver.cuh"
#include "pseudo_spectral_solver.cuh"
#include "anelastic_sl_solver.cuh"
#include "stellar_profile.h"
#include "sph2d_spectral_solver.cuh"
#include "physics/helmholtz_eos.cuh"
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
#include <algorithm>
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
    bool radiation_enabled = false;
    double rad_c_light = 100.0;       // reduced speed of light (code units); full c = very slow
    bool nuclear_enabled = false;
    double nuc_X = 0.7;
    double nuc_Y = 0.28;
    double nuc_epsilon_scale = 1.0;
    double nuc_T_floor = 1.0e6;
    double nuc_T_scale = 1.0;
    double nuc_q_burn = 6.4e18;
    bool species_enabled = false;
    bool implicit_mode = false;      // radial1d: use BE + JFNK (step_implicit) instead of explicit RK2
    double dt_implicit = 0.0;        // fixed dt for --implicit;<=0 uses acoustic CFL × scale
    double dt_implicit_scale = 1.0;  // multiplier on CFL dt when dt_implicit<=0
    bool no_viallet = false;         // dev toggle: disable Viallet L/R scaling in implicit
    bool precond_tridiag = false;    // implicit: block-tridiag PC (assembled from 9 colored FD matvecs)
    bool jfnk_autodiff = false;      // implicit: exact J·v via Dual<1> AD (replaces FD matvec)
    bool no_rhse_subtract = false;   // implicit diagnostic: F = (U-Uⁿ)/dt - R(U) (skip R_hse term)
    double newton_tol_override = 0.0;// implicit: override Newton ||F|| convergence tol (0 = solver default 1e-8)
    int hse_resnap_interval = 0;     // implicit: re-snapshot R_hse every N steps (0=off)
    double dt_thermal_frac = 0.0;    // implicit: dt ≤ frac · IE/L_surf (τ_KH cap, 0=off)
    double dt_mach_cap = 0.0;        // implicit: shrink dt when max Mach exceeds this (0=off)
    double rad_T_phot_floor = 0.0;   // implicit rad-in-F: min T_phot for Stefan BC (K, 0=off)
    double nuc_compress_frac = 0.0;  // dynamic nuc scale: ε·dt/(cv·T) ≤ this (0=off)
    bool ic_solar = false;           // if true, Lane-Emden IC in physical cgs matching sun
    bool mlt_enabled = false;        // Böhm-Vitense MLT convection in BE rad solve
    double mlt_alpha = 1.5;          // mixing length / pressure scale height
    double ic_rho_c = -1.0;          // override central density; <0 = test default
    double ic_R_star = -1.0;         // override target stellar radius (cgs); <0 = derive from K
    double ic_n_poly = 1.5;          // polytropic index for --ic-solar
    std::string ic_mesa_path;        // non-empty ⇒ take IC from scripts/convert_mesa_ic.py output
    bool ic_mesa_seed_T = false;     // --ic-mesa-seed-T: seed (e,P) from Helm(ρ,T_MESA) instead of (ρ,P_MESA)
    int  ic_mesa_atm_zones = 0;      // --ic-mesa-atm-zones N: use hybrid zoning with N log-spaced outer atm zones (0=equal-mass)
    int  atm_split = 0;              // --atm-split N: operator-split rad in outer N zones (usually = ic_mesa_atm_zones)
    bool rich_profile = false;       // --rich-profile: emit T, κ, ∇_ad, ∇_rad, L, conv_vel per zone
    std::string bubble_mode = "pressure"; // "pressure" or "entropy"
    // EOS selection
    std::string eos_type = "ideal";   // "ideal" or "ideal_rad"
    double eos_mu = 1.0;
    double eos_rad_a = 0.1;           // radiation constant (code units)
    // Helmholtz EOS
    std::string helm_table_path = "third_party/helmholtz/helm_table.bin";
    double helm_Abar = 1.28;          // solar: X≈0.73, Y≈0.25, Z≈0.02
    double helm_Zbar = 1.13;
    // MESA opacity tables — radial1d reads two binaries (lowT + highT) and
    // stitches them at logT ≈ 4. --kap-prefix/--kap-lowT accept just the
    // family tag (e.g. `gs98`, `lowT_fa05_gs98`); we append `_z<Z>.kapbin`.
    std::string kap_highT_family = "gs98";
    std::string kap_lowT_family  = "lowT_fa05_gs98";
    double      kap_table_Z      = 0.02;
    std::string kap_data_dir     = "third_party/mesa_kap";
    bool        kap_use_table    = false;
    double      kap_logT_lo_end   = 3.9;
    double      kap_logT_hi_start = 4.1;
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
    // HLLC variant: 0=standard, 1=Rieper LM-HLLC, 2=Minoshima LHLLC.
    // --lm-hllc  → 1 (back-compat)
    // --lhllc    → 2 (low-dissipation HLLC, Athena++ port)
    // --hllc <standard|lm|lhllc> — explicit form
    int hllc_variant = 0;
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
    std::string cart_ale2_slab_file;   // cart_ale2 --test local_convection: slab stratification
    double cart_ale2_slab_perturb = 0.01;  // entropy seed amplitude at slab bottom
    int cart_ale2_slab_seed_k = 4;     // horizontal mode of entropy seed
    double cart_ale2_cool_tau = 0.0;   // cart_ale2: Newton-cooling timescale (s). 0 = disabled.
    // Bottom enthalpy-flux source (makes local_convection flux-driven so v_conv
    // saturates at the MLT scale). Either --heat-flux F directly [erg/cm²/s],
    // or --heat-lsun L + --heat-bot-R R → F = L/(4π R²).  0 = disabled.
    double cart_ale2_heat_flux = 0.0;
    double cart_ale2_heat_lsun = 0.0;
    double cart_ale2_heat_bot_R = 0.0;
    double cart_ale2_heat_bot_frac = 0.05; // e-fold of exp heating profile / Ly
    double cart_ale2_cool_top_frac = 0.3;  // cooling confined to top frac of column
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
        else if (std::strcmp(argv[i], "--eos") == 0 && i + 1 < argc)
            cfg.eos_type = argv[++i];
        else if (std::strcmp(argv[i], "--eos-mu") == 0 && i + 1 < argc)
            cfg.eos_mu = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--eos-rad-a") == 0 && i + 1 < argc)
            cfg.eos_rad_a = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--helm-table") == 0 && i + 1 < argc)
            cfg.helm_table_path = argv[++i];
        else if (std::strcmp(argv[i], "--helm-abar") == 0 && i + 1 < argc)
            cfg.helm_Abar = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--helm-zbar") == 0 && i + 1 < argc)
            cfg.helm_Zbar = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--kap") == 0)
            cfg.kap_use_table = true;
        else if (std::strcmp(argv[i], "--kap-highT") == 0 && i + 1 < argc)
            cfg.kap_highT_family = argv[++i];
        else if (std::strcmp(argv[i], "--kap-lowT") == 0 && i + 1 < argc)
            cfg.kap_lowT_family = argv[++i];
        else if (std::strcmp(argv[i], "--kap-Z") == 0 && i + 1 < argc)
            cfg.kap_table_Z = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--kap-dir") == 0 && i + 1 < argc)
            cfg.kap_data_dir = argv[++i];
        else if (std::strcmp(argv[i], "--kap-logT-lo-end") == 0 && i + 1 < argc)
            cfg.kap_logT_lo_end = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--kap-logT-hi-start") == 0 && i + 1 < argc)
            cfg.kap_logT_hi_start = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--radiation") == 0)
            cfg.radiation_enabled = true;
        else if (std::strcmp(argv[i], "--rad-c") == 0 && i + 1 < argc)
            cfg.rad_c_light = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--nuclear") == 0)
            cfg.nuclear_enabled = true;
        else if (std::strcmp(argv[i], "--nuc-x") == 0 && i + 1 < argc)
            cfg.nuc_X = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--nuc-scale") == 0 && i + 1 < argc)
            cfg.nuc_epsilon_scale = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--nuc-t-floor") == 0 && i + 1 < argc)
            cfg.nuc_T_floor = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--nuc-y") == 0 && i + 1 < argc)
            cfg.nuc_Y = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--nuc-q") == 0 && i + 1 < argc)
            cfg.nuc_q_burn = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--nuc-t-scale") == 0 && i + 1 < argc)
            cfg.nuc_T_scale = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--species") == 0)
            cfg.species_enabled = true;
        else if (std::strcmp(argv[i], "--ic-solar") == 0)
            cfg.ic_solar = true;
        else if (std::strcmp(argv[i], "--ic-rho-c") == 0 && i + 1 < argc)
            cfg.ic_rho_c = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ic-rstar") == 0 && i + 1 < argc)
            cfg.ic_R_star = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ic-n-poly") == 0 && i + 1 < argc)
            cfg.ic_n_poly = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ic-mesa") == 0 && i + 1 < argc)
            cfg.ic_mesa_path = argv[++i];
        else if (std::strcmp(argv[i], "--ic-mesa-seed-T") == 0)
            cfg.ic_mesa_seed_T = true;
        else if (std::strcmp(argv[i], "--ic-mesa-atm-zones") == 0 && i + 1 < argc)
            cfg.ic_mesa_atm_zones = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--atm-split") == 0 && i + 1 < argc)
            cfg.atm_split = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--rich-profile") == 0)
            cfg.rich_profile = true;
        else if (std::strcmp(argv[i], "--G") == 0 && i + 1 < argc)
            cfg.G = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--implicit") == 0)
            cfg.implicit_mode = true;
        else if (std::strcmp(argv[i], "--dt-implicit") == 0 && i + 1 < argc)
            cfg.dt_implicit = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--dt-implicit-scale") == 0 && i + 1 < argc)
            cfg.dt_implicit_scale = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--no-viallet") == 0)
            cfg.no_viallet = true;
        else if (std::strcmp(argv[i], "--precond-tridiag") == 0)
            cfg.precond_tridiag = true;
        else if (std::strcmp(argv[i], "--jfnk-autodiff") == 0)
            cfg.jfnk_autodiff = true;
        else if (std::strcmp(argv[i], "--no-rhse") == 0)
            cfg.no_rhse_subtract = true;
        else if (std::strcmp(argv[i], "--newton-tol") == 0 && i + 1 < argc)
            cfg.newton_tol_override = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--hse-resnap") == 0 && i + 1 < argc)
            cfg.hse_resnap_interval = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--dt-thermal-frac") == 0 && i + 1 < argc)
            cfg.dt_thermal_frac = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--dt-mach-cap") == 0 && i + 1 < argc)
            cfg.dt_mach_cap = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--rad-T-phot-floor") == 0 && i + 1 < argc)
            cfg.rad_T_phot_floor = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--nuc-compress") == 0 && i + 1 < argc)
            cfg.nuc_compress_frac = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--mlt") == 0)
            cfg.mlt_enabled = true;
        else if (std::strcmp(argv[i], "--mlt-alpha") == 0 && i + 1 < argc)
            cfg.mlt_alpha = std::atof(argv[++i]);
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
            cfg.hllc_variant = 1;
        else if (std::strcmp(argv[i], "--lhllc") == 0)
            cfg.hllc_variant = 2;
        else if (std::strcmp(argv[i], "--hllc") == 0 && i + 1 < argc) {
            std::string v = argv[++i];
            if (v == "lhllc") cfg.hllc_variant = 2;
            else if (v == "lm") cfg.hllc_variant = 1;
            else cfg.hllc_variant = 0;
        }
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
        else if (std::strcmp(argv[i], "--ic-slab") == 0 && i + 1 < argc)
            cfg.cart_ale2_slab_file = argv[++i];
        else if (std::strcmp(argv[i], "--slab-perturb") == 0 && i + 1 < argc)
            cfg.cart_ale2_slab_perturb = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--slab-seed-k") == 0 && i + 1 < argc)
            cfg.cart_ale2_slab_seed_k = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--cool-tau") == 0 && i + 1 < argc)
            cfg.cart_ale2_cool_tau = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--heat-flux") == 0 && i + 1 < argc)
            cfg.cart_ale2_heat_flux = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--heat-lsun") == 0 && i + 1 < argc)
            cfg.cart_ale2_heat_lsun = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--heat-bot-R") == 0 && i + 1 < argc)
            cfg.cart_ale2_heat_bot_R = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--heat-bot-frac") == 0 && i + 1 < argc)
            cfg.cart_ale2_heat_bot_frac = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--cool-top-frac") == 0 && i + 1 < argc)
            cfg.cart_ale2_cool_top_frac = std::atof(argv[++i]);
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
        // TEMP (sphere_impl preview): if --r-inner > 0 on uniform mesh, compute
        // M_core the same way as mass mesh so pole_avg + core_excision wire up.
        if (cfg.r_inner > 0 && (cfg.test_case == "lane_emden" || cfg.test_case == "lane_emden_perturbed")) {
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
            const int nfine = 20000;
            double dr_f = cfg.r_inner / nfine;
            double m = 0.0;
            for (int ii = 0; ii < nfine; ++ii) {
                double r = (ii + 0.5) * dr_f;
                m += 4.0 * M_PI * rho_func(r) * r * r * dr_f;
            }
            cfg.M_core = m;
            std::printf("Using uniform radial mesh with r_inner=%.4f, M_core=%.5f (sphere_impl preview)\n",
                        cfg.r_inner, cfg.M_core);
        } else {
            std::printf("Using uniform radial mesh\n");
        }
    } else {
        grid.init(cfg.nr, cfg.ntheta, cfg.R_outer, cfg.log_alpha);
    }

    EOS eos;
#ifdef USE_GPU
    // Helmholtz table lives through the whole run; destroyed after solver loop.
    HelmholtzTable helm_tbl;
    bool helm_loaded = false;
    KapTable kap_tbl_lowT, kap_tbl_highT;
    bool kap_loaded = false;
#endif
    if (cfg.eos_type == "ideal_rad") {
        eos = EOS::ideal_rad(cfg.gamma, cfg.eos_mu, cfg.eos_rad_a);
        std::printf("EOS: ideal + radiation (γ=%.3f, μ=%.3f, a=%.3e)\n",
                    cfg.gamma, cfg.eos_mu, cfg.eos_rad_a);
    } else if (cfg.eos_type == "pre_ms") {
        PreMsParams p;
        // Use reasonable defaults; override via CLI later if needed.
        // R_gas in code units: keep 1.0 (consistent with rest of code).
        p.R_gas = 1.0 / cfg.eos_mu;  // inverse μ for code-unit consistency
        p.a_rad = cfg.eos_rad_a;
        eos = EOS::pre_ms(p);
        std::printf("EOS: pre-MS Chabrier-Baraffe (T_diss=%.0f, T_ion=%.0f, μ_cold=%.2f → μ_hot=%.2f)\n",
                    p.T_diss, p.T_ion, p.mu_cold, p.mu_hot);
    } else if (cfg.eos_type == "helmholtz") {
#ifdef USE_GPU
        if (helm_tbl.load(nullptr, cfg.helm_table_path.c_str(), 0) != 0) {
            std::fprintf(stderr,
                "ERROR: failed to load Helmholtz table at %s — run\n"
                "  tools/helm_convert <ascii> %s\n",
                cfg.helm_table_path.c_str(), cfg.helm_table_path.c_str());
            return 1;
        }
        helm_tbl.view.Abar = cfg.helm_Abar;
        helm_tbl.view.Zbar = cfg.helm_Zbar;
        eos = EOS::helmholtz(helm_tbl.view);
        helm_loaded = true;
        std::printf("EOS: Helmholtz (cococubed %dx%d, Abar=%.3f, Zbar=%.3f)\n",
                    HELM_IMAX, HELM_JMAX, cfg.helm_Abar, cfg.helm_Zbar);
#else
        std::fprintf(stderr, "ERROR: --eos helmholtz requires USE_GPU build\n");
        return 1;
#endif
    } else {
        eos = EOS::ideal(cfg.gamma, cfg.eos_mu);
    }

#ifdef USE_GPU
    // Optional MESA kap table pair (lowT + highT). Loaded once before the
    // solver starts; the radial1d branch below hands the views to the solver.
    if (cfg.kap_use_table) {
        char z_buf[32];
        std::snprintf(z_buf, sizeof(z_buf), "%g", cfg.kap_table_Z);
        std::string z_tag = z_buf;
        std::string high_path = cfg.kap_data_dir + "/" + cfg.kap_highT_family
                              + "_z" + z_tag + ".kapbin";
        std::string low_path  = cfg.kap_data_dir + "/" + cfg.kap_lowT_family
                              + "_z" + z_tag + ".kapbin";
        int rc_hi = kap_tbl_highT.load(high_path.c_str(), 0);
        if (rc_hi != 0) {
            std::fprintf(stderr,
                "ERROR: failed to load high-T kap binary at %s (rc=%d)\n",
                high_path.c_str(), rc_hi);
            return 1;
        }
        int rc_lo = kap_tbl_lowT.load(low_path.c_str(), 0);
        if (rc_lo != 0) {
            std::fprintf(stderr,
                "ERROR: failed to load low-T kap binary at %s (rc=%d)\n",
                low_path.c_str(), rc_lo);
            return 1;
        }
        kap_loaded = true;
        std::printf("KAP: stitched {%s + %s} at Z=%s, seam logT ∈ [%.2f, %.2f]\n",
                    cfg.kap_highT_family.c_str(), cfg.kap_lowT_family.c_str(),
                    z_tag.c_str(), cfg.kap_logT_lo_end, cfg.kap_logT_hi_start);
    }
#endif

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
               || cfg.test_case == "gmode_pulsation"
               || cfg.test_case == "gmode_2d_evp"
               || cfg.test_case == "gmode_eigenmode_td"
               || cfg.test_case == "gmode_exp_k"
               || cfg.test_case == "dns_triad"
               || cfg.test_case == "dns_triad_coupled"
               || cfg.test_case == "local_convection") {
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
        // TEMP: extend to uniform mesh with --r-inner > 0 (sphere_impl preview test).
        // Treat "uniform + r_inner>0" like mass mesh for pole_avg / core_excision wiring.
        bool is_mass = (cfg.mesh_type == "mass");
        bool is_uniform_with_rinner = (cfg.mesh_type == "uniform" && cfg.r_inner > 0);
        if (!is_mass && !is_uniform_with_rinner) return;
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
        // Wire EOS: if user picked anything other than ideal, use EOS-aware kernels.
        if (cfg.eos_type != "ideal") {
            r1d.use_eos = true;
            r1d.eos = eos;
            std::printf("radial1d: EOS-aware kernels enabled (%s)\n", cfg.eos_type.c_str());
        }
        // Wire radiation diffusion
        if (cfg.radiation_enabled) {
            r1d.radiation_enabled = true;
            r1d.rad_c_light = cfg.rad_c_light;
            r1d.rad_a_rad = cfg.eos_rad_a > 0 ? cfg.eos_rad_a : 1.0;
            std::printf("radial1d: radiation diffusion ON (c=%.3e, a=%.3e)\n",
                        r1d.rad_c_light, r1d.rad_a_rad);
        }
        // Wire tabulated opacity, if loaded
        if (kap_loaded) {
            r1d.kap_use_table      = true;
            r1d.kap_view_lowT      = kap_tbl_lowT.view;
            r1d.kap_view_highT     = kap_tbl_highT.view;
            r1d.kap_logT_lo_end    = cfg.kap_logT_lo_end;
            r1d.kap_logT_hi_start  = cfg.kap_logT_hi_start;
            r1d.kap_hydrogen_X     = cfg.nuc_X;     // composition slice
            std::printf("radial1d: tabulated kap ON (X slice = %.3f)\n",
                        r1d.kap_hydrogen_X);
        }
        // Wire nuclear burning
        if (cfg.nuclear_enabled) {
            r1d.nuclear_enabled = true;
            r1d.nuc_X = cfg.nuc_X;
            r1d.nuc_Y = cfg.nuc_Y;
            r1d.nuc_epsilon_scale = cfg.nuc_epsilon_scale;
            r1d.nuc_T_floor = cfg.nuc_T_floor;
            r1d.nuc_T_scale = cfg.nuc_T_scale;
            r1d.nuc_q_burn = cfg.nuc_q_burn;
            std::printf("radial1d: pp-chain nuclear burning ON (X=%.2f, scale=%.3e, T_floor=%.3eK, T_scale=%.3e, q=%.3e)\n",
                        r1d.nuc_X, r1d.nuc_epsilon_scale, r1d.nuc_T_floor, r1d.nuc_T_scale, r1d.nuc_q_burn);
        }
        // Wire species tracking (requires --nuclear)
        if (cfg.species_enabled) {
            if (!cfg.nuclear_enabled) {
                std::fprintf(stderr, "WARN: --species without --nuclear has no effect; enabling nuclear\n");
                r1d.nuclear_enabled = true;
                r1d.nuc_X = cfg.nuc_X;
                r1d.nuc_Y = cfg.nuc_Y;
                r1d.nuc_epsilon_scale = cfg.nuc_epsilon_scale;
                r1d.nuc_T_floor = cfg.nuc_T_floor;
                r1d.nuc_T_scale = cfg.nuc_T_scale;
                r1d.nuc_q_burn = cfg.nuc_q_burn;
            }
            r1d.species_enabled = true;
            std::printf("radial1d: species tracking ON (X→Y burn-up)\n");
        }
        if (cfg.mlt_enabled) {
            r1d.mlt_enabled = true;
            r1d.mlt_alpha = cfg.mlt_alpha;
            std::printf("radial1d: MLT convection ON (α=%.2f)\n", cfg.mlt_alpha);
        }
        if (!cfg.ic_mesa_path.empty()) {
            std::printf("radial1d: MESA IC from %s (seed=%s, atm_zones=%d)\n",
                        cfg.ic_mesa_path.c_str(),
                        cfg.ic_mesa_seed_T ? "T" : "P",
                        cfg.ic_mesa_atm_zones);
            if (r1d.init_from_mesa(cfg.ic_mesa_path.c_str(),
                                   cfg.ic_mesa_seed_T,
                                   cfg.ic_mesa_atm_zones) != 0) {
                std::fprintf(stderr, "ERROR: init_from_mesa failed\n");
                return 1;
            }
        } else if (cfg.ic_solar) {
            // Physical cgs Lane-Emden IC with user-specified (ρ_c, R_star, n).
            // Derive K so that α·ξ_1 = R_star exactly.
            //   α² = (n+1) K ρ_c^(1/n − 1) / (4πG)
            //   ⇒ K = α² · 4πG · ρ_c^(1 − 1/n) / (n+1),  α = R_star / ξ_1
            // Pre-computed ξ_1 for common n:
            double n_pol = cfg.ic_n_poly;
            // crude ξ_1 table: n=1.5 → 3.65375; n=3.0 → 6.89685
            double xi1 = (std::fabs(n_pol - 1.5) < 1e-3) ? 3.65375
                       : (std::fabs(n_pol - 3.0) < 1e-3) ? 6.89685
                       : 3.65375;
            double rho_c = (cfg.ic_rho_c > 0) ? cfg.ic_rho_c : 80.0;     // g/cc
            double R_star = (cfg.ic_R_star > 0) ? cfg.ic_R_star : 7.0e10; // cm
            double G_cgs = cfg.G; // user must pass --G 6.674e-8 or equivalent
            double alpha = R_star / xi1;
            double K_poly = alpha * alpha * 4.0 * M_PI * G_cgs
                            * std::pow(rho_c, 1.0 - 1.0/n_pol)
                            / (n_pol + 1.0);
            std::printf("radial1d: solar polytrope IC  n=%.2f  ρ_c=%.3e g/cc  R⋆=%.3e cm  K=%.3e  G=%.3e\n",
                        n_pol, rho_c, R_star, K_poly, G_cgs);
            r1d.init_lane_emden(rho_c, K_poly, n_pol);
        } else {
            r1d.init_lane_emden(1.0, 1.0, 1.5);          // ρ_c=1, K=1, n=1.5
        }
        r1d.snapshot_hse();
        if (r1d.species_enabled) {
            r1d.init_species_uniform(r1d.nuc_X, r1d.nuc_Y);
        }
        // Capture R(U_hse) BEFORE any perturbation so the well-balanced
        // subtraction references the true HSE, not the perturbed state.
        if (cfg.implicit_mode) {
            r1d.implicit_enabled = true;
            if (cfg.no_viallet) r1d.use_viallet_scaling = false;
            r1d.precond_tridiag = cfg.precond_tridiag;
            r1d.jfnk_autodiff   = cfg.jfnk_autodiff;
            r1d.no_rhse_subtract = cfg.no_rhse_subtract;
            r1d.rad_T_phot_floor = cfg.rad_T_phot_floor;
            if (cfg.newton_tol_override > 0) r1d.newton_tol = cfg.newton_tol_override;
            r1d.hse_resnap_interval = cfg.hse_resnap_interval;
            r1d.nuc_compress_frac = cfg.nuc_compress_frac;
            r1d.nz_atm_split = cfg.atm_split;
            r1d.init_implicit();
            r1d.snapshot_hse_implicit();
        }
        if (cfg.test_case == "lane_emden_perturbed")
            r1d.apply_perturbation(cfg.perturb_amplitude);

        std::timespec wall_start;
        clock_gettime(CLOCK_MONOTONIC, &wall_start);

        // Simple text output (CSV format) for radial1d mode
        char csv_path[512];
        std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", run_dir.c_str());
        std::FILE* csv = std::fopen(csv_path, "w");
        if (cfg.species_enabled) {
            std::fprintf(csv, "step,t,dt,mass,KE,IE,PE,total_E,max_mach,max_vr,T_c,rho_c,L_nuc,mass_H,mass_He,X_core,X_surf,L_surf,conv_mfrac,r_conv_in,r_conv_out,max_super,T_phot,phot_zone,R_surf\n");
        } else {
            std::fprintf(csv, "step,t,dt,mass,KE,IE,PE,total_E,max_mach,max_vr,T_c,rho_c,L_nuc,L_surf,conv_mfrac,r_conv_in,r_conv_out,max_super,T_phot,phot_zone,R_surf\n");
        }

        int frame = 0;
        if (cfg.implicit_mode) {
            std::printf("radial1d: IMPLICIT mode ON (Viallet=%s, Newton tol=%.1e, GMRES tol=%.1e)\n",
                        r1d.use_viallet_scaling ? "on" : "off",
                        r1d.newton_tol, r1d.gmres_tol);
        }
        while (t < cfg.t_end && !g_interrupted) {
            double dt;
            if (cfg.implicit_mode) {
                // Adaptive dt control optimised for crossing τ_KH:
                //   - Start from --dt-implicit (seed).
                //   - On success, grow ×2 per accepted step; cap at --dt-implicit
                //     if given (explicit ceiling) or at 10·τ_dyn otherwise.
                //   - On failure, cut ×0.1 (not ×0.5) — aggressive because
                //     stiff nuclear flashes usually need a big reset.
                //   - Never fall back to explicit: accelerated nuclear would
                //     blow up.
                static double dt_req_state = 0.0;
                double dt_seed = (cfg.dt_implicit > 0) ? cfg.dt_implicit
                                                       : cfg.dt_implicit_scale * r1d.compute_dt();
                if (dt_req_state <= 0.0) dt_req_state = dt_seed;
                // Ceiling: explicit --dt-implicit if provided, else let it grow
                // unbounded (user controls via --tend).
                double dt_cap = (cfg.dt_implicit > 0) ? cfg.dt_implicit : 1e30;

                dt = r1d.step_implicit(t, cfg.t_end, dt_req_state);
                if (dt <= 0.0) {
                    dt_req_state *= 0.1;
                    if (dt_req_state < 1e-30) dt_req_state = 1e-30;
                    std::fprintf(stderr,
                        "  step %d: implicit FAILED, outer dt reset to %.3e\n",
                        step, dt_req_state);
                    dt = 0.0;
                } else {
                    // Geometric growth on success — cross many τ_dyn fast.
                    dt_req_state = std::min(dt_cap, dt_req_state * 2.0);

                    // Thermal-timescale cap: dt ≤ thermal_frac · IE / L_surf
                    // (Kelvin-Helmholtz). Without this, once rad coupling
                    // actually cools the star the outer dt grows ×2 per step
                    // and overshoots a full τ_KH in one step, blowing the
                    // solution past hydrostatic equilibrium.
                    if (cfg.dt_thermal_frac > 0.0 || cfg.dt_mach_cap > 0.0) {
                        Radial1DSolver::Diagnostics dg = r1d.compute_diagnostics();
                        double L_tot = std::fabs(r1d.rad_impl_L_surf);
                        if (cfg.dt_thermal_frac > 0.0
                            && L_tot > 1e-30 && dg.total_internal_E > 0.0) {
                            double tau_th = dg.total_internal_E / L_tot;
                            double dt_th  = cfg.dt_thermal_frac * tau_th;
                            if (dt_req_state > dt_th) dt_req_state = dt_th;
                        }
                        // Mach-based damping: if the last step's hydro solution
                        // has transient velocities (operator-split rad can
                        // leave momentum imbalance), shrink dt until the
                        // transient decays below mach_cap.
                        if (cfg.dt_mach_cap > 0.0 && dg.max_mach > cfg.dt_mach_cap) {
                            double shrink = cfg.dt_mach_cap / dg.max_mach;
                            if (shrink < 0.1) shrink = 0.1;
                            dt_req_state *= shrink;
                        }
                    }
                }
            } else {
                dt = r1d.step(t, cfg.t_end);
            }
            t += dt;
            step++;

            if (step % 200 == 0)
                print_progress(t, cfg.t_end, step, dt, wall_start);

            if (step % cfg.output_interval == 0) {
                auto d = r1d.compute_diagnostics();
                auto mlt = r1d.compute_convection_diag();
                std::fprintf(stderr, "\n");
                std::printf("Step %6d  t=%.6e dt=%.3e M=%.10e E=%.10e |v|_max=%.3e Mach_max=%.3e\n",
                            step, t, dt, d.total_mass, d.total_E, d.max_vr, d.max_mach);
                if (r1d.nuclear_enabled) {
                    double L_ratio = (r1d.rad_impl_L_surf > 0) ? d.L_nuc / r1d.rad_impl_L_surf : 0.0;
                    std::printf("    ignition: T_c=%.3e K  ρ_c=%.3e g/cc  L_nuc=%.3e erg/s  L_nuc/L_surf=%.3e\n",
                                d.T_c, d.rho_c, d.L_nuc, L_ratio);
                }
                if (r1d.radiation_enabled) {
                    std::printf("    rad_BC:   phot_zone=%d  T_phot=%.3e K  tau_sum=%.3e  L_surf=%.3e erg/s\n",
                                r1d.rad_impl_phot_zone, r1d.rad_impl_T_phot,
                                r1d.rad_impl_tau_surf, r1d.rad_impl_L_surf);
                }
                if (mlt.n_conv_zones > 0) {
                    std::printf("    MLT: conv_mass_frac=%.4f  r_conv=[%.3e,%.3e]  n_conv=%d  max_super=%.3e\n",
                                mlt.conv_mass_frac, mlt.r_conv_inner, mlt.r_conv_outer,
                                mlt.n_conv_zones, mlt.max_superadiab);
                }
                // R_surf = outermost face radius (read once per CSV row)
                double R_surf = 0.0;
                {
                    std::vector<double> r_f, v_f, rho_c3, P_c3, e_c3;
                    r1d.download_profile(r_f, v_f, rho_c3, P_c3, e_c3);
                    if (!r_f.empty()) R_surf = r_f.back();
                }
                if (r1d.species_enabled) {
                    std::vector<double> X_c, Y_c;
                    r1d.download_species(X_c, Y_c);
                    // Mass-weighted totals over zones: need dm; use e_cell/P_cell buffers via profile
                    std::vector<double> r_f, v_f, rho_c2, P_c2, e_c2;
                    r1d.download_profile(r_f, v_f, rho_c2, P_c2, e_c2);
                    const double PI43 = 4.188790204786391;
                    double mH = 0.0, mHe = 0.0;
                    for (int k = 0; k < r1d.lev.nz; ++k) {
                        double rL = r_f[k], rR = r_f[k+1];
                        double dmk = rho_c2[k] * PI43 * (rR*rR*rR - rL*rL*rL);
                        mH  += dmk * X_c[k];
                        mHe += dmk * Y_c[k];
                    }
                    double X_core = X_c.front();
                    double X_surf = X_c.back();
                    std::fprintf(csv, "%d,%.10e,%.6e,%.10e,%.10e,%.10e,%.10e,%.10e,%.6e,%.6e,%.6e,%.6e,%.6e,%.10e,%.10e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%d,%.6e\n",
                                 step, t, dt, d.total_mass, d.total_KE, d.total_internal_E,
                                 d.total_grav_E, d.total_E, d.max_mach, d.max_vr,
                                 d.T_c, d.rho_c, d.L_nuc,
                                 mH, mHe, X_core, X_surf, r1d.rad_impl_L_surf,
                                 mlt.conv_mass_frac, mlt.r_conv_inner, mlt.r_conv_outer, mlt.max_superadiab,
                                 r1d.rad_impl_T_phot, r1d.rad_impl_phot_zone, R_surf);
                } else {
                    std::fprintf(csv, "%d,%.10e,%.6e,%.10e,%.10e,%.10e,%.10e,%.10e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%d,%.6e\n",
                                 step, t, dt, d.total_mass, d.total_KE, d.total_internal_E,
                                 d.total_grav_E, d.total_E, d.max_mach, d.max_vr,
                                 d.T_c, d.rho_c, d.L_nuc, r1d.rad_impl_L_surf,
                                 mlt.conv_mass_frac, mlt.r_conv_inner, mlt.r_conv_outer, mlt.max_superadiab,
                                 r1d.rad_impl_T_phot, r1d.rad_impl_phot_zone, R_surf);
                }
                std::fflush(csv);

                // Dump profile as simple text. When --rich-profile is set,
                // also emit T, κ, Γ₁, ∇_ad, ∇_rad, L, mixing type, v_conv
                // so scripts/pk_mesa_radial1d.py can compare per-field.
                std::vector<double> r_face, v_face, rho_cell, P_cell, e_cell;
                std::vector<double> T_cell, kap_cell, g1_cell, ga_cell, gr_cell;
                std::vector<double> L_face, vc_cell;
                std::vector<int>    mt_cell;
                if (cfg.rich_profile) {
                    r1d.download_profile_rich(r_face, v_face, rho_cell, P_cell, e_cell,
                                              T_cell, kap_cell, g1_cell, ga_cell, gr_cell,
                                              L_face, mt_cell, vc_cell);
                } else {
                    r1d.download_profile(r_face, v_face, rho_cell, P_cell, e_cell);
                }
                std::vector<double> X_cell, Y_cell;
                if (r1d.species_enabled) r1d.download_species(X_cell, Y_cell);
                char path[512];
                std::snprintf(path, sizeof(path), "%s/profile_%04d.txt", run_dir.c_str(), ++frame);
                std::FILE* fp = std::fopen(path, "w");
                if (cfg.rich_profile) {
                    std::fprintf(fp,
                        "# t = %.10e  step = %d\n"
                        "# k r_face v_face rho P e_int T kap gamma1 grada gradr L_face mixing_type conv_vel%s\n",
                        t, step, r1d.species_enabled ? " X Y" : "");
                } else if (r1d.species_enabled) {
                    std::fprintf(fp, "# t = %.10e  step = %d\n# k r_face v_face rho P e_int X Y\n", t, step);
                } else {
                    std::fprintf(fp, "# t = %.10e  step = %d\n# k r_face v_face rho P e_int\n", t, step);
                }
                for (int k = 0; k < r1d.lev.nz; ++k) {
                    if (cfg.rich_profile) {
                        std::fprintf(fp,
                            "%d %.10e %.10e %.10e %.10e %.10e "
                            "%.10e %.10e %.6e %.6e %.6e %.6e %d %.6e",
                            k, r_face[k], v_face[k], rho_cell[k], P_cell[k], e_cell[k],
                            T_cell[k], kap_cell[k], g1_cell[k], ga_cell[k], gr_cell[k],
                            L_face[k], mt_cell[k], vc_cell[k]);
                        if (r1d.species_enabled)
                            std::fprintf(fp, " %.6e %.6e", X_cell[k], Y_cell[k]);
                        std::fprintf(fp, "\n");
                    } else if (r1d.species_enabled) {
                        std::fprintf(fp, "%d %.10e %.10e %.10e %.10e %.10e %.6e %.6e\n",
                                     k, r_face[k], v_face[k], rho_cell[k], P_cell[k], e_cell[k],
                                     X_cell[k], Y_cell[k]);
                    } else {
                        std::fprintf(fp, "%d %.10e %.10e %.10e %.10e %.10e\n",
                                     k, r_face[k], v_face[k], rho_cell[k], P_cell[k], e_cell[k]);
                    }
                }
                // last face (velocity only; cell-centred fields don't apply)
                std::fprintf(fp, "%d %.10e %.10e - - -\n", r1d.lev.nz, r_face[r1d.lev.nz], v_face[r1d.lev.nz]);
                std::fclose(fp);
            }
        }
        std::fclose(csv);
        std::fprintf(stderr, "\n");
        if (cfg.implicit_mode) r1d.destroy_implicit();
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

    } else if (cfg.solver_type == "fas2") {
        // ===== fas2: experimental fork of FAS for low-Mach robustness =====
        // Clone of FasSolver with incremental fixes:
        //   - CGS2 Gram-Schmidt (Knoll-Keyes 2004)
        //   - Unit-normalize v in JFNK matvec (Trilinos NOX)
        //   - Viallet 2016 eq 72 asymmetric L/R scaling
        //   - Line-implicit-in-r preconditioner (cherry-pick from line-jacobi-precond)
        FasSolver2 fas;
        fas.use_simple_smoother = (cfg.precond != "block_jacobi");
        // fas2-specific: --precond line_r toggles line-implicit-in-r JFNK preconditioner
        fas.use_line_precond_r = (cfg.precond == "line_r");
        fas.limiter_type = static_cast<int>(cfg.limiter);
        fas.hllc_variant = cfg.hllc_variant;
        fas.radial_only = cfg.radial_only;
        fas.init(grid, eos, cfg.G, cfg.cfl);
        if (cfg.radial_only)
            std::printf("fas2 radial-only mode\n");
        if (cfg.no_sponge) fas.sponge_kappa = 0.0;
        configure_mass_mesh(fas);
        if (cfg.mesh_type == "mass" && cfg.r_inner <= 0)
            fas.central_damp_r = 0.15 * cfg.R_outer;
        snapshot_hse_if_needed(fas);
        fas.upload_state(grid, state);

        FasLevel2& fl = fas.levels[0];
        int snap_size = fl.total;
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
        std::vector<double> snap_times, snap_dts;
        std::vector<int> snap_steps;
        int n_snaps = 0;

        SolverOps ops;
        ops.progress_interval = 2000;
        ops.step = [&](double t_, double te) { return fas.step(t_, te); };
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

    } else if (cfg.solver_type == "fas" || cfg.solver_type == "explicit") {
        bool use_explicit = (cfg.solver_type == "explicit");
        FasSolver fas;
        fas.use_simple_smoother = (cfg.precond != "block_jacobi");
        fas.limiter_type = static_cast<int>(cfg.limiter);
        fas.hllc_variant = cfg.hllc_variant;
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
    } else if (cfg.solver_type == "cart_impl") {
        // ===== cart_impl: 2D Cartesian BE + JFNK low-Mach =====
        // 借鑒 cart_ale2 的 Cartesian 均勻網格 + fas2 的 JFNK + Viallet scaling.
        // 放棄球極座標,目標是解決 256² 擾動能真正演化(不被 BE 凍結).
        CartImplSolver csol;
        bool is_hse = (cfg.test_case == "hse" || cfg.test_case == "hse_perturbed");
        if (!is_hse) {
            std::fprintf(stderr,
                "cart_impl only supports --test hse / hse_perturbed for now\n");
            return 1;
        }
        double Lx = 1.0, Ly = 1.0;
        double g_val = 1.0;
        double rho_base = 1.0;
        csol.hllc_variant = cfg.hllc_variant;
        csol.limiter_type = static_cast<int>(cfg.limiter);
        csol.init(cfg.nr, cfg.ntheta, Lx, Ly, eos, cfg.gamma, g_val, cfg.cfl);

        // Two-step init: HSE first, snapshot reference, then layer perturbation
        csol.init_hse_polytrope(rho_base, 0.0);
        csol.snapshot_hse();
        if (cfg.test_case == "hse_perturbed" && cfg.perturb_amplitude != 0.0) {
            csol.init_hse_polytrope(rho_base, cfg.perturb_amplitude);
        }

        std::timespec wall_start;
        clock_gettime(CLOCK_MONOTONIC, &wall_start);

        char csv_path[512];
        std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", run_dir.c_str());
        std::FILE* csv = std::fopen(csv_path, "w");
        std::fprintf(csv, "step,t,dt,mass,energy,max_v,max_mach\n");

        // Initial frame
        {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/output_0000.vtk", run_dir.c_str());
            csol.write_vtk(path);
        }

        int frame = 0;
        int diag_every = cfg.diag_interval > 0 ? cfg.diag_interval : cfg.output_interval;
        int vtk_every  = cfg.vtk_interval  > 0 ? cfg.vtk_interval  : cfg.output_interval;
        while (t < cfg.t_end && !g_interrupted) {
            double dt = csol.step(t, cfg.t_end);
            t += dt;
            step++;
            if (step % 200 == 0) print_progress(t, cfg.t_end, step, dt, wall_start);
            if (step % diag_every == 0 || t >= cfg.t_end) {
                auto d = csol.compute_diagnostics();
                std::fprintf(stderr, "\n");
                std::printf("Step %6d  t=%.6e dt=%.3e M=%.10e E=%.10e |v|=%.3e Ma=%.3e\n",
                            step, t, dt, d.mass, d.energy, d.max_v, d.max_mach);
                std::fprintf(csv, "%d,%.10e,%.6e,%.10e,%.10e,%.6e,%.6e\n",
                             step, t, dt, d.mass, d.energy, d.max_v, d.max_mach);
                std::fflush(csv);
            }
            if (step % vtk_every == 0 || t >= cfg.t_end) {
                ++frame;
                char path[512];
                std::snprintf(path, sizeof(path), "%s/output_%04d.vtk", run_dir.c_str(), frame);
                csol.write_vtk(path);
            }
        }
        std::fclose(csv);
        std::fprintf(stderr, "\n");
        csol.destroy();

    } else if (cfg.solver_type == "cart_ale2") {
        // ===== cart_ale2: full periodic BC + PPM-in-remap (in development) =====
        CartAle2Solver cale;
        bool is_hse = (cfg.test_case == "hse" || cfg.test_case == "hse_perturbed"
                       || cfg.test_case == "hse_bubble");
        bool is_kh = (cfg.test_case == "kh_shear");
        bool is_kh_lec = (cfg.test_case == "kh_lecoanet");
        bool is_loc_conv = (cfg.test_case == "local_convection");
        double Lx = 1.0;
        // Lecoanet: domain aspect 1:2 so shear layers at y=0.5, y=1.5 match
        // Athena++ iprob=4 geometry (z1=-0.5, z2=0.5 in centred coords).
        double Ly = is_kh_lec ? 2.0
                  : (is_hse || is_kh) ? 1.0
                  : 0.2;
        double gam = is_hse ? cfg.gamma : 1.4;
        // local_convection: read slab header to get real Ly, Lx, γ in cgs.
        if (is_loc_conv) {
            if (cfg.cart_ale2_slab_file.empty()) {
                std::fprintf(stderr,
                    "local_convection requires --ic-slab <file>\n");
                return 1;
            }
            std::FILE* fp = std::fopen(cfg.cart_ale2_slab_file.c_str(), "r");
            if (!fp) {
                std::fprintf(stderr,
                    "cannot open slab file %s\n", cfg.cart_ale2_slab_file.c_str());
                return 1;
            }
            char line[512];
            while (std::fgets(line, sizeof(line), fp)) {
                const char* s = line;
                while (*s == ' ' || *s == '\t') ++s;
                if (*s == '#' || *s == '\0' || *s == '\n') continue;
                double Ly_f, Lx_f, gy_f, gamma_f, rho_t, P_t, T_t, mu_f;
                if (std::sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf",
                                &Ly_f, &Lx_f, &gy_f, &gamma_f,
                                &rho_t, &P_t, &T_t, &mu_f) == 8) {
                    Ly = Ly_f; Lx = Lx_f; gam = gamma_f;
                    break;
                }
            }
            std::fclose(fp);
        }
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
        } else if (cfg.test_case == "local_convection") {
            // Recommend --bc-x periodic --bc-y reflect.
            if (!((cale.bc_mode & 1) && !(cale.bc_mode & 2))) {
                std::fprintf(stderr,
                    "  [warn] local_convection prefers --bc-x periodic --bc-y reflect; "
                    "current bc_mode=%d\n", cale.bc_mode);
            }
            cale.init_local_convection(cfg.cart_ale2_slab_file,
                                       cfg.cart_ale2_slab_perturb,
                                       cfg.cart_ale2_slab_seed_k);
            cale.tau_cool = cfg.cart_ale2_cool_tau;
            if (cale.tau_cool > 0.0)
                std::fprintf(stderr,
                    "    Newton cooling ON: τ_cool=%.3e s\n", cale.tau_cool);

            // Bottom enthalpy-flux heating.  Either F directly or L/(4π R²).
            double F_bot = cfg.cart_ale2_heat_flux;
            if (F_bot <= 0.0 && cfg.cart_ale2_heat_lsun > 0.0) {
                if (cfg.cart_ale2_heat_bot_R <= 0.0) {
                    std::fprintf(stderr,
                        "  [error] --heat-lsun requires --heat-bot-R "
                        "(bottom-face radius in cm)\n");
                    return 1;
                }
                double R = cfg.cart_ale2_heat_bot_R;
                F_bot = cfg.cart_ale2_heat_lsun / (4.0 * M_PI * R * R);
            }
            if (F_bot > 0.0 || cfg.cart_ale2_cool_top_frac < 1.0) {
                cale.configure_thermal(F_bot,
                                       cfg.cart_ale2_heat_bot_frac,
                                       cfg.cart_ale2_cool_top_frac);
            }
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
            && cfg.test_case != "gmode_pulsation"
            && cfg.test_case != "gmode_2d_evp"
            && cfg.test_case != "gmode_eigenmode_td"
            && cfg.test_case != "gmode_exp_k"
            && cfg.test_case != "dns_triad"
            && cfg.test_case != "dns_triad_coupled") {
            std::fprintf(stderr,
                "ERROR: anelastic_sl supports --test {sl_basis_check, "
                "sl_poisson_test[_boussinesq], kh_shear_boussinesq, "
                "gmode_pulsation, gmode_2d_evp, gmode_eigenmode_td, gmode_exp_k, "
                "dns_triad, dns_triad_coupled}\n");
            return 1;
        }
        AnelasticSLSolver ansl;
        int n_modes = (cfg.nr > 0 ? cfg.nr / 2 : 128);
        ansl.init(cfg.ntheta, cfg.nr, n_modes,
                  cfg.ps_Lx, cfg.ps_Ly, cfg.ps_nu, cfg.cfl);

        std::string bg;
        double bg_arg = 0.01;
        if (cfg.test_case == "kh_shear_boussinesq"
            || cfg.test_case == "sl_poisson_test_boussinesq") {
            bg = "boussinesq";
        } else if (cfg.test_case == "gmode_pulsation"
                   || cfg.test_case == "gmode_2d_evp"
                   || cfg.test_case == "gmode_eigenmode_td"
                   || cfg.test_case == "dns_triad"
                   || cfg.test_case == "dns_triad_coupled") {
            // dns_triad / dns_triad_coupled default to Lane-Emden n=3/2 with
            // rho_cut=0.05 (triad) or 0.1 (coupled, for better long-time
            // stability); other tests default to stratified_n2 unless
            // overridden via ANSL_BG / ANSL_RHO_CUT.
            if (cfg.test_case == "dns_triad") {
                bg = "lane_emden_1_5";
                bg_arg = 0.05;
                if (const char* rc = std::getenv("ANSL_RHO_CUT")) {
                    double v = std::atof(rc);
                    if (v > 0.0 && v < 1.0) bg_arg = v;
                }
            } else if (cfg.test_case == "dns_triad_coupled") {
                bg = "lane_emden_1_5";
                bg_arg = 0.1;
                if (const char* rc = std::getenv("ANSL_RHO_CUT")) {
                    double v = std::atof(rc);
                    if (v > 0.0 && v < 1.0) bg_arg = v;
                }
            } else {
                bg = "stratified_n2";
                // Re-use ps_vshear as the N² value knob for this test (default 1.0).
                bg_arg = (cfg.ps_vshear > 0.0) ? cfg.ps_vshear : 1.0;
            }
            // Variable-density override: ANSL_BG=lane_emden_1_5 flips to a
            // real stratified Lane-Emden n=3/2 background (uses ANSL_RHO_CUT
            // for surface truncation, default 0.01).  The EVP, SL-filter and
            // TANH coord-map downstream all work identically; only ρ₀(y),
            // N²(y), and W̃(y) change.
            if (const char* bg_env = std::getenv("ANSL_BG")) {
                std::string s(bg_env);
                if (s == "lane_emden_1_5" || s == "lane_emden") {
                    bg = "lane_emden_1_5";
                    bg_arg = 0.01;
                    if (const char* rc = std::getenv("ANSL_RHO_CUT")) {
                        double v = std::atof(rc);
                        if (v > 0.0 && v < 1.0) bg_arg = v;
                    }
                }
            }
        } else {
            bg = "lane_emden_1_5";
        }
        ansl.set_background(bg, bg_arg);

        if (cfg.test_case == "sl_poisson_test"
            || cfg.test_case == "sl_poisson_test_boussinesq") {
            ansl.manufactured_test();
        }

        if (cfg.test_case == "gmode_exp_k") {
            // CUDA port of scripts/gmode_exp_k_chebyshev_full.py
            int N = (cfg.nr > 0 ? cfg.nr : 64);
            int ell = 1;
            int n_modes_req = 10;
            double inner_cut = 1e-4, outer_cut = 0.9999;

            // Build CGL nodes on [inner_cut, outer_cut]
            std::vector<double> x_cgl(N + 1);
            for (int k = 0; k <= N; ++k) {
                int kk = N - k;
                x_cgl[k] = inner_cut
                         + (1.0 + std::cos(M_PI * kk / (double)N))
                         * (outer_cut - inner_cut) / 2.0;
            }
            std::sort(x_cgl.begin(), x_cgl.end());

            // If ANSL_POLY3_TXT is set, load GYRE's poly3.txt and CubicSpline-
            // interpolate to CGL.  Otherwise build our own n=3 polytrope.
            StellarProfile prof;
            const char* poly_path_env = std::getenv("ANSL_POLY3_TXT");
            if (poly_path_env) {
                StellarProfile raw;
                if (!read_gyre_structure_txt(poly_path_env, raw)) {
                    std::fprintf(stderr,
                        "gmode_exp_k: failed to load %s\n", poly_path_env);
                    return 1;
                }
                std::fprintf(stderr,
                    "  Loaded GYRE structure from %s (%d rows)\n",
                    poly_path_env, raw.n_points());
                // Linear-interpolate raw → CGL (upgrade to cubic spline later)
                auto interp = [&](const std::vector<double>& y) {
                    std::vector<double> out(x_cgl.size());
                    for (size_t i = 0; i < x_cgl.size(); ++i) {
                        double xq = x_cgl[i];
                        int j = 0;
                        while (j + 1 < (int)raw.x.size() && raw.x[j + 1] < xq) ++j;
                        if (j + 1 >= (int)raw.x.size()) j = (int)raw.x.size() - 2;
                        double f = (xq - raw.x[j]) / (raw.x[j + 1] - raw.x[j]);
                        out[i] = y[j] + f * (y[j + 1] - y[j]);
                    }
                    return out;
                };
                prof.x       = x_cgl;
                prof.V_2     = interp(raw.V_2);
                prof.A_star  = interp(raw.A_star);
                prof.U       = interp(raw.U);
                prof.c_1     = interp(raw.c_1);
                prof.Gamma_1 = interp(raw.Gamma_1);
            } else {
                prof = build_polytrope_profile_at(3.0, x_cgl);
            }

            std::vector<double> omega_sq, eigvecs_y1;
            ansl.solve_gmode_full_chebyshev(
                prof.x, prof.V_2, prof.U, prof.A_star, prof.c_1, prof.Gamma_1,
                ell, n_modes_req, omega_sq, eigvecs_y1);

            // Exp K EXPECTED vs GYRE (frozen values in the Python script)
            static const double EXPECTED[10] = {
                2.5159279360877496, 1.2857077544856306, 0.7757327764772477,
                0.5177759762324133, 0.36992549567563754, 0.2775028154601723,
                0.21592664733814718, 0.1728536032702941, 0.1415440904624304,
                0.11806842352742726,
            };
            std::fprintf(stderr,
                "  Exp K CUDA:  N=%d (DOF=%d), ell=%d, polytrope n=3\n",
                N, 4 * (N + 1), ell);
            std::fprintf(stderr,
                "  %3s  %16s  %16s  %10s\n", "n_g", "ω²_CUDA", "ω²_GYRE", "rel err");
            double max_rel = 0.0;
            int n_show = std::min((int)omega_sq.size(), 10);
            for (int k = 0; k < n_show; ++k) {
                double rel = std::fabs(omega_sq[k] - EXPECTED[k]) / EXPECTED[k];
                if (rel > max_rel) max_rel = rel;
                std::fprintf(stderr,
                    "  %3d  %16.10e  %16.10e  %10.3e\n",
                    k + 1, omega_sq[k], EXPECTED[k], rel);
            }
            std::fprintf(stderr, "  Exp K max_rel = %.3e\n", max_rel);

            // CSV dump
            char csv_path[512];
            std::snprintf(csv_path, sizeof(csv_path), "%s/gmode_exp_k.csv",
                          run_dir.c_str());
            FILE* fp = std::fopen(csv_path, "w");
            std::fprintf(fp, "# N=%d, ell=%d\n", N, ell);
            std::fprintf(fp, "# omega_sq (n_modes):\n");
            for (double w : omega_sq) std::fprintf(fp, "%.15e\n", w);
            std::fclose(fp);
            std::fprintf(stderr, "  Exp K CSV at %s\n", csv_path);
        }

        if (cfg.test_case == "gmode_2d_evp") {
            int kx_int = (cfg.ps_k > 0) ? cfg.ps_k : 1;
            double kx_phys = kx_int * 2.0 * M_PI / ansl.Lx;
            int n_modes_req = 10;
            std::vector<double> omega_sq, v_modes;
            enum class EvpPath { GALERKIN, QSPACE_FOURIER, QSPACE_SL };
            EvpPath path = EvpPath::GALERKIN;
            if (const char* s = std::getenv("ANSL_EVP_BASIS")) {
                std::string ss(s);
                if (ss == "qspace" || ss == "q" || ss == "phi")
                    path = EvpPath::QSPACE_FOURIER;
                else if (ss == "qspace_sl" || ss == "sl" || ss == "q_sl")
                    path = EvpPath::QSPACE_SL;
            }
            if (path == EvpPath::QSPACE_FOURIER) {
                std::vector<double> phi_modes;
                ansl.compute_2d_gmode_evp_qspace(kx_phys, n_modes_req,
                                                 omega_sq, phi_modes);
                const int n_out = (int)omega_sq.size();
                v_modes.assign((size_t)(ansl.ny - 2) * n_out, 0.0);
                for (int k = 0; k < n_out; ++k) {
                    for (int i = 0; i < ansl.ny - 2; ++i) {
                        int row = i + 1;
                        double phi = phi_modes[(size_t)row * n_out + k];
                        v_modes[(size_t)i + (size_t)k * (ansl.ny - 2)] =
                            phi / std::max(ansl.h_rho[row], 1e-30);
                    }
                }
            } else if (path == EvpPath::QSPACE_SL) {
                std::vector<double> v_full;
                ansl.compute_2d_gmode_evp_qspace_sl(kx_phys, n_modes_req,
                                                    omega_sq, v_full);
                const int n_out = (int)omega_sq.size();
                v_modes.assign((size_t)(ansl.ny - 2) * n_out, 0.0);
                for (int k = 0; k < n_out; ++k) {
                    for (int i = 0; i < ansl.ny - 2; ++i) {
                        int row = i + 1;
                        v_modes[(size_t)i + (size_t)k * (ansl.ny - 2)] =
                            v_full[(size_t)row * n_out + k];
                    }
                }
            } else {
                ansl.compute_2d_gmode_evp(kx_phys, n_modes_req, omega_sq, v_modes);
            }
            double N2 = bg_arg;
            std::fprintf(stderr,
                "  2D g-mode EVP at k_x_int=%d  (k_x_phys=%g, N²=%g)\n",
                kx_int, kx_phys, N2);
            std::fprintf(stderr,
                "  %3s  %16s  %14s  %12s\n",
                "n", "ω²_CUDA", "ω²_analytic", "rel err");
            for (size_t n = 0; n < omega_sq.size(); ++n) {
                double ky = (double)(n + 1) * M_PI / ansl.Ly;
                double ex = N2 * kx_phys * kx_phys /
                            (kx_phys * kx_phys + ky * ky);
                double rel = std::fabs(omega_sq[n] - ex) / ex;
                std::fprintf(stderr,
                    "  %3zu  %16.10e  %14.10e  %12.3e\n",
                    n + 1, omega_sq[n], ex, rel);
            }

            // Dump ω² and eigenvectors to CSV for downstream use.
            char csv_path[512];
            std::snprintf(csv_path, sizeof(csv_path), "%s/gmode_2d_evp.csv",
                          run_dir.c_str());
            FILE* fp = std::fopen(csv_path, "w");
            std::fprintf(fp, "# ny=%d, Lx=%g, Ly=%g, kx_int=%d, N2=%g\n",
                         ansl.ny, ansl.Lx, ansl.Ly, kx_int, N2);
            std::fprintf(fp, "# omega_sq (n_modes values):\n");
            for (double w : omega_sq) std::fprintf(fp, "%.15e\n", w);
            std::fprintf(fp, "# v_modes ((ny-2) × n_modes column-major):\n");
            for (double x : v_modes) std::fprintf(fp, "%.15e\n", x);
            std::fclose(fp);
            std::fprintf(stderr, "  2D EVP written to %s\n", csv_path);
        }

        if (cfg.test_case == "gmode_pulsation") {
            // Phase 1d: Lane-Emden n=3/2 background, k_y=1 sinusoid seed.
            double amp = (cfg.perturb_amplitude > 0) ? cfg.perturb_amplitude : 1e-3;
            int k_y = (cfg.ps_k > 0) ? cfg.ps_k : 1;
            ansl.init_gmode_pulsation(amp, k_y);
            if (std::getenv("ANSL_DUMP_STEP1")) {
                // Run exactly ONE step and dump v, b at y=Ly/2 for Python compare.
                double dt1 = ansl.step();
                std::vector<double> hv, hu;
                ansl.download_uv(hu, hv);
                std::vector<double> hb;
                ansl.download_b(hb);
                int jy = ansl.ny / 2, ix = ansl.nx / 4;
                int k = jy * ansl.nx + ix;
                std::fprintf(stderr,
                    "  [STEP1 DUMP] dt=%.6e  v[jy=%d,ix=%d]=%.6e  b[]=%.6e  u[]=%.6e\n",
                    dt1, jy, ix, hv[k], hb[k], hu[k]);
                std::exit(0);
            }
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

        if (cfg.test_case == "gmode_eigenmode_td") {
            // Phase 1e: exact 2D g-mode eigenmode IC → time-domain probe.
            // Background: stratified_n2 (ρ₀=1, constant N² via --ps-vshear).
            // Used to characterise the Chorin-splitting time error on a
            // clean single-frequency signal.
            double amp   = (cfg.perturb_amplitude > 0) ? cfg.perturb_amplitude : 1e-3;
            int kx_int   = (cfg.ps_k > 0) ? cfg.ps_k : 1;
            int n_g_env  = 1;
            if (const char* s = std::getenv("ANSL_NG")) n_g_env = std::atoi(s);
            if (n_g_env < 1) n_g_env = 1;

            double om2_evp = ansl.init_gmode_eigenmode(kx_int, n_g_env, amp);
            double om_evp  = std::sqrt(om2_evp);
            double T_period = 2.0 * M_PI / om_evp;
            std::fprintf(stderr,
                "  AnelasticSL eigenmode-TD run: tend=%g, cfl=%g, N²=%g, kx_int=%d, n_g=%d\n",
                cfg.t_end, cfg.cfl, bg_arg, kx_int, n_g_env);
            std::fprintf(stderr,
                "  EVP ω²=%.10e, ω=%.10e, period=%.6f (t_end covers %.2f periods)\n",
                om2_evp, om_evp, T_period, cfg.t_end / T_period);

            double t = 0.0;
            int step = 0;
            std::timespec wall_start;
            clock_gettime(CLOCK_MONOTONIC, &wall_start);

            char probe_path[512];
            std::snprintf(probe_path, sizeof(probe_path),
                          "%s/gmode_eigenmode_td.csv", run_dir.c_str());
            FILE* probe = std::fopen(probe_path, "w");
            std::fprintf(probe,
                "# kx_int=%d n_g=%d N2=%g omega_sq_evp=%.15e omega_evp=%.15e amp=%g\n",
                kx_int, n_g_env, bg_arg, om2_evp, om_evp, amp);
            std::fprintf(probe, "# t  v_center  eigmode_deviation\n");
            double v0 = ansl.probe_v_center();
            double dev0 = ansl.eigmode_deviation();
            std::fprintf(probe, "%.10e %.10e %.10e\n", 0.0, v0, dev0);

            // Path D: if ANSL_TD_KIND=assembled_linear, replace step() with
            // the linear-only assembled-matrix RK4 path
            // (docs/full_galerkin_closure_proof_2026-05-03.md).
            bool use_path_d = ansl.td_assembled_linear;
            if (use_path_d) {
                std::fprintf(stderr,
                    "  Path D linear TD: assembled L⁻¹R per kx, RK4\n");
            }
            while (t < cfg.t_end && !g_interrupted) {
                double dt = use_path_d ? ansl.step_assembled_linear()
                                       : ansl.step();
                if (t + dt > cfg.t_end) dt = cfg.t_end - t;
                t += dt;
                ++step;
                double v_c = ansl.probe_v_center();
                double dev = ansl.eigmode_deviation();
                std::fprintf(probe, "%.10e %.10e %.10e\n", t, v_c, dev);
                if (step % 200 == 0 || t >= cfg.t_end) {
                    print_progress(t, cfg.t_end, step, dt, wall_start);
                }
            }
            std::fclose(probe);
            std::fprintf(stderr,
                "  eigenmode-TD probe written to %s (%d samples)\n",
                probe_path, step);
        }

        if (cfg.test_case == "dns_triad") {
            // ─── Phase 3 / DNS Experiment A: 3-wave triad mode coupling ─────
            // (paper/DNS_PLAN.md).  Lane-Emden n=3/2 bg, eigenmode IC (n_g=1,
            // kx_int=1) at amp=1e-2, Strang-split RK4 (Path D linear block +
            // advection), 300 g-mode periods.  Per-period diagnostics:
            //   total energy E(t) = ½∫ρ(u²+v²) + ½∫b²/N²
            //   modal energy E_k(t) for k=1..4 (horizontal Fourier bands)
            //   eigmode deviation on the primary k_1 component.
            double amp   = (cfg.perturb_amplitude > 0) ? cfg.perturb_amplitude : 1e-2;
            int kx_int   = (cfg.ps_k > 0) ? cfg.ps_k : 1;
            int n_g      = 1;
            if (const char* s = std::getenv("ANSL_NG")) n_g = std::max(1, std::atoi(s));

            double om2_evp = ansl.init_gmode_eigenmode(kx_int, n_g, amp);
            double om_evp  = std::sqrt(om2_evp);
            double T_period = 2.0 * M_PI / om_evp;

            // For Strang (V, W, B) oscillator: consistent eigenmode IC is
            // V = V_EVP·sin(kx·x), W = 0, B = 0 (cosine-phase oscillator peak).
            // init_gmode_eigenmode fills d_b with the off-phase b = -(N²/ω²)·V
            // which is wrong for our (v, w, b) RK4 dynamics — zero it out.
            CUDA_CHECK(cudaMemset(ansl.d_b, 0, sizeof(double) * ansl.ncell));

            // Period count and fixed dt aligned to periods.
            int n_periods = 300;
            if (const char* s = std::getenv("ANSL_DNS_PERIODS")) {
                int v = std::atoi(s); if (v > 0) n_periods = v;
            }
            int steps_per_period = 32;
            if (const char* s = std::getenv("ANSL_DNS_SPP")) {
                int v = std::atoi(s); if (v > 0) steps_per_period = v;
            }
            double dt = T_period / (double)steps_per_period;
            int n_steps_total = n_periods * steps_per_period;

            std::fprintf(stderr,
                "  DNS Experiment A (triad):  bg=%s, ny=%d, nx=%d, kx_int=%d, n_g=%d\n",
                bg.c_str(), cfg.nr, cfg.ntheta, kx_int, n_g);
            std::fprintf(stderr,
                "  amp=%g, ω=%.6f, period=%.6f, dt=%.6f (%d steps/period), %d periods\n",
                amp, om_evp, T_period, dt, steps_per_period, n_periods);

            char probe_path[512];
            std::snprintf(probe_path, sizeof(probe_path),
                          "%s/dns_triad.csv", run_dir.c_str());
            FILE* probe = std::fopen(probe_path, "w");
            std::fprintf(probe,
                "# kx_int=%d n_g=%d amp=%g omega=%.15e period=%.15e dt=%.15e steps_per_period=%d\n",
                kx_int, n_g, amp, om_evp, T_period, dt, steps_per_period);
            std::fprintf(probe,
                "# t  v_center  eigmode_dev  E_total  E_k1  E_k2  E_k3  E_k4  max_abs_v\n");

            // Grab weights and grid for host-side diagnostics.
            std::vector<double> y_cgl, w_cc(cfg.nr, 0.0);
            ansl.download_y(y_cgl);
            {
                // Reproduce Clenshaw-Curtis weights on [0, Ly] (same recipe as
                // Python scripts/nonlinear_paths_infra.py).  Ly from config.
                int N = cfg.nr - 1;
                std::vector<double> w(N + 1, 0.0);
                for (int k = 0; k <= N; ++k) {
                    double s = 0.0;
                    int J = N / 2;
                    for (int j = 1; j <= J; ++j) {
                        double b = (2 * j != N) ? 2.0 : 1.0;
                        s += b / (4.0 * j * j - 1) *
                             std::cos(2.0 * j * k * M_PI / N);
                    }
                    w[k] = (1.0 - s) * 2.0 / (double)N;
                }
                w[0] /= 2.0; w[N] /= 2.0;
                for (int k = 0; k <= N; ++k) w_cc[k] = w[N - k] * cfg.ps_Ly / 2.0;
            }

            // Initial u is built in init_gmode_eigenmode.  Download initial state.
            auto diagnostics = [&](double t_now) {
                std::vector<double> h_u, h_v, h_b;
                ansl.rebuild_u_from_continuity();
                ansl.download_uv(h_u, h_v);
                ansl.download_b(h_b);

                const int ny = cfg.nr;
                const int nx = cfg.ntheta;

                // Background N²(y) and ρ(y) host-side: we need them for E_total.
                // AnelasticSLSolver exposes h_N2 and h_rho internally.  We hack
                // around by requesting them from the solver's diagnostic path.
                // (They're on host as ansl.h_N2, ansl.h_rho.)

                double E_kin = 0.0, E_pot = 0.0;
                for (int jy = 0; jy < ny; ++jy) {
                    double w = w_cc[jy];
                    double rho_y = ansl.h_rho[jy];
                    double N2_y  = ansl.h_N2[jy];
                    double KE_row = 0.0, PE_row = 0.0;
                    for (int ix = 0; ix < nx; ++ix) {
                        int k = jy * nx + ix;
                        double U = h_u[k], V = h_v[k], B = h_b[k];
                        KE_row += U * U + V * V;
                        if (N2_y > 1e-12) PE_row += B * B / N2_y;
                    }
                    E_kin += 0.5 * w * rho_y * KE_row / (double)nx;
                    E_pot += 0.5 * w               * PE_row / (double)nx;
                }
                double E_total = E_kin + E_pot;

                // Modal energy E_k for k=1..4:  integrate |v̂_k(y)|² + |û_k(y)|²
                // over y with ρ weight; ignore b contribution for compactness.
                // Use naive DFT at integer mode k (nx typically small).
                double Emk[5] = {0,0,0,0,0};
                for (int kmode = 1; kmode <= 4; ++kmode) {
                    double cumE = 0.0;
                    for (int jy = 0; jy < ny; ++jy) {
                        double w   = w_cc[jy];
                        double rho_y = ansl.h_rho[jy];
                        double vr = 0.0, vi = 0.0, ur = 0.0, ui = 0.0;
                        for (int ix = 0; ix < nx; ++ix) {
                            double x = (double)ix * (cfg.ps_Lx / (double)nx);
                            double ph = 2.0 * M_PI * (double)kmode * x / cfg.ps_Lx;
                            double c = std::cos(ph), s = std::sin(ph);
                            double V = h_v[jy * nx + ix];
                            double U = h_u[jy * nx + ix];
                            vr += V * c; vi -= V * s;
                            ur += U * c; ui -= U * s;
                        }
                        double inv_nx = 1.0 / (double)nx;
                        vr *= inv_nx; vi *= inv_nx;
                        ur *= inv_nx; ui *= inv_nx;
                        // single-sided mode has magnitude squared coefficient 2:
                        cumE += w * rho_y * 2.0 *
                                (vr * vr + vi * vi + ur * ur + ui * ui);
                    }
                    Emk[kmode] = 0.5 * cumE;
                }

                double v_c = ansl.probe_v_center();
                double dev = ansl.eigmode_deviation();
                double max_v = 0.0;
                for (double x : h_v) if (std::fabs(x) > max_v) max_v = std::fabs(x);
                std::fprintf(probe,
                    "%.10e %.10e %.10e %.15e %.15e %.15e %.15e %.15e %.10e\n",
                    t_now, v_c, dev, E_total,
                    Emk[1], Emk[2], Emk[3], Emk[4], max_v);
            };

            // Sample at t=0.
            diagnostics(0.0);

            std::timespec wall_start;
            clock_gettime(CLOCK_MONOTONIC, &wall_start);

            double t_now = 0.0;
            int samples = 1;
            for (int p = 0; p < n_periods && !g_interrupted; ++p) {
                for (int k = 0; k < steps_per_period && !g_interrupted; ++k) {
                    ansl.step_strang_nonlinear(dt);
                    t_now += dt;
                }
                diagnostics(t_now);
                ++samples;
                if ((p + 1) % 10 == 0 || p == n_periods - 1) {
                    print_progress(t_now, (double)n_periods * T_period,
                                   (p + 1) * steps_per_period, dt, wall_start);
                }
            }
            std::fclose(probe);
            std::fprintf(stderr,
                "  DNS triad probe written to %s (%d samples)\n",
                probe_path, samples);
        }

        if (cfg.test_case == "dns_triad_coupled") {
            // ─── Experiment E1: true three-wave resonant triad ──────────────
            // Seeds TWO eigenmodes (a, b) and watches the resonance partner
            // (c = a + b in wavenumber) grow from numerical noise.
            //
            // Default mode selection (from scripts/scan_resonance.py on
            // Lane-Emden n=3/2, rho_cut=0.1, Ny=128):
            //   a = (n_g=4, kx=1, ω≈0.6402)
            //   b = (n_g=3, kx=2, ω≈1.1751)
            //   c = (n_g=1, kx=3, ω≈1.8556)        ← should grow ≈ t²
            //   detune ≈ 2.2%, |V_abc| = 0.92 (strong overlap)
            //
            // Overridable via ANSL_NG_A, ANSL_KX_A, ANSL_AMP_A (same for B).
            std::vector<AnelasticSLSolver::ModeSpec> modes;
            auto env_int = [](const char* key, int fallback) {
                if (const char* s = std::getenv(key)) {
                    int v = std::atoi(s); if (v != 0) return v;
                }
                return fallback;
            };
            auto env_dbl = [](const char* key, double fallback) {
                if (const char* s = std::getenv(key)) {
                    double v = std::atof(s); if (v != 0.0) return v;
                }
                return fallback;
            };
            double amp_a = env_dbl("ANSL_AMP_A", cfg.perturb_amplitude > 0
                                   ? cfg.perturb_amplitude : 1e-3);
            double amp_b = env_dbl("ANSL_AMP_B", amp_a);
            int n_g_a = env_int("ANSL_NG_A", 4);
            int n_g_b = env_int("ANSL_NG_B", 3);
            int kx_a  = env_int("ANSL_KX_A", 1);
            int kx_b  = env_int("ANSL_KX_B", 2);
            // Phase choice: put mode a on sin(kx_a x) and mode b on cos(kx_b x)
            // so they are not coherent at t=0 when their x-wavenumbers align
            // (here kx_a != kx_b so both can be sin-phase).
            modes.push_back({n_g_a, kx_a, amp_a, false});
            modes.push_back({n_g_b, kx_b, amp_b, false});

            double om_a = ansl.init_multi_mode_ic(modes);
            double T_period = 2.0 * M_PI / om_a;

            int n_periods = env_int("ANSL_DNS_PERIODS", 200);
            int steps_per_period = env_int("ANSL_DNS_SPP", 32);
            double dt = T_period / (double)steps_per_period;

            // Expected c mode: kx_c = kx_a + kx_b, we only display it.
            int kx_c = kx_a + kx_b;
            std::fprintf(stderr,
                "  DNS E1 (three-wave triad):  (a,b)→c =  "
                "(%d,kx=%d) + (%d,kx=%d) → (?,kx=%d)\n",
                n_g_a, kx_a, n_g_b, kx_b, kx_c);
            std::fprintf(stderr,
                "  amp_a=%g, amp_b=%g, ω_a=%.6f (period=%.4f), "
                "dt=%.4f (%d spp), %d periods\n",
                amp_a, amp_b, om_a, T_period, dt, steps_per_period, n_periods);

            char probe_path[512];
            std::snprintf(probe_path, sizeof(probe_path),
                          "%s/dns_triad_coupled.csv", run_dir.c_str());
            FILE* probe = std::fopen(probe_path, "w");
            std::fprintf(probe,
                "# triad: a=(n%d,kx%d,amp%g)  b=(n%d,kx%d,amp%g)  c=(?,kx%d)\n",
                n_g_a, kx_a, amp_a, n_g_b, kx_b, amp_b, kx_c);
            std::fprintf(probe,
                "# omega_a=%.15e period_a=%.15e dt=%.15e spp=%d\n",
                om_a, T_period, dt, steps_per_period);
            std::fprintf(probe,
                "# t  max_v  E_kin_total  E_k1  E_k2  E_k3  E_k4  E_k5  E_k6  E_pot\n");

            // Host-side CC weights (reuse recipe from dns_triad).
            std::vector<double> y_cgl, w_cc(cfg.nr, 0.0);
            ansl.download_y(y_cgl);
            {
                int N = cfg.nr - 1;
                std::vector<double> w(N + 1, 0.0);
                for (int k = 0; k <= N; ++k) {
                    double s = 0.0;
                    int J = N / 2;
                    for (int j = 1; j <= J; ++j) {
                        double b = (2 * j != N) ? 2.0 : 1.0;
                        s += b / (4.0 * j * j - 1) *
                             std::cos(2.0 * j * k * M_PI / N);
                    }
                    w[k] = (1.0 - s) * 2.0 / (double)N;
                }
                w[0] /= 2.0; w[N] /= 2.0;
                for (int k = 0; k <= N; ++k) w_cc[k] = w[N - k] * cfg.ps_Ly / 2.0;
            }

            // Snapshot cadence (periods).  ANSL_DNS_SNAP_EVERY=N writes binary
            // (u, v, b) float32 cubes to <run_dir>/snapshots/snap_NNNN.bin
            // after every N periods; 0 disables snapshotting.
            int snap_every = env_int("ANSL_DNS_SNAP_EVERY", 0);
            std::string snap_dir = run_dir + "/snapshots";
            if (snap_every > 0) {
                std::string cmd = "mkdir -p '" + snap_dir + "'";
                (void)std::system(cmd.c_str());
            }
            int snap_idx = 0;

            auto diagnostics = [&](double t_now) {
                std::vector<double> h_u, h_v, h_b;
                ansl.rebuild_u_from_continuity();
                ansl.download_uv(h_u, h_v);
                ansl.download_b(h_b);
                const int ny = cfg.nr;
                const int nx = cfg.ntheta;

                // Total energy (anelastic functional).
                double E_kin = 0.0, E_pot = 0.0;
                double max_v = 0.0;
                for (int jy = 0; jy < ny; ++jy) {
                    double w = w_cc[jy];
                    double rho_y = ansl.h_rho[jy];
                    double N2_y  = ansl.h_N2[jy];
                    double KE_row = 0.0, PE_row = 0.0;
                    for (int ix = 0; ix < nx; ++ix) {
                        int k = jy * nx + ix;
                        double U = h_u[k], V = h_v[k], B = h_b[k];
                        KE_row += U * U + V * V;
                        if (N2_y > 1e-12) PE_row += B * B / N2_y;
                        if (std::fabs(V) > max_v) max_v = std::fabs(V);
                    }
                    E_kin += 0.5 * w * rho_y * KE_row / (double)nx;
                    E_pot += 0.5 * w               * PE_row / (double)nx;
                }
                double E_total = E_kin + E_pot;

                // Modal energy E_k for k=1..6 using naive DFT per jy.
                const int KMAX = 6;
                std::vector<double> Emk(KMAX + 1, 0.0);
                for (int kmode = 1; kmode <= KMAX; ++kmode) {
                    double cumE = 0.0;
                    for (int jy = 0; jy < ny; ++jy) {
                        double w   = w_cc[jy];
                        double rho_y = ansl.h_rho[jy];
                        double vr=0,vi=0,ur=0,ui=0;
                        for (int ix = 0; ix < nx; ++ix) {
                            double x = (double)ix * (cfg.ps_Lx / (double)nx);
                            double ph = 2.0 * M_PI * (double)kmode * x / cfg.ps_Lx;
                            double c = std::cos(ph), s = std::sin(ph);
                            double V = h_v[jy * nx + ix];
                            double U = h_u[jy * nx + ix];
                            vr += V * c; vi -= V * s;
                            ur += U * c; ui -= U * s;
                        }
                        double inv_nx = 1.0 / (double)nx;
                        vr *= inv_nx; vi *= inv_nx;
                        ur *= inv_nx; ui *= inv_nx;
                        cumE += w * rho_y * 2.0 *
                                (vr*vr + vi*vi + ur*ur + ui*ui);
                    }
                    Emk[kmode] = 0.5 * cumE;
                }

                // Kinetic-only modal sum as conservation check:  E_kin_total
                // is immune to the 1/N² surface-amplification of ∫ b²/N² that
                // dominates E_total on Lane-Emden.
                double E_kin_total = 0.0;
                for (int kmode = 1; kmode <= KMAX; ++kmode) E_kin_total += Emk[kmode];
                std::fprintf(probe,
                    "%.10e %.10e %.15e "
                    "%.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                    t_now, max_v, E_kin_total,
                    Emk[1], Emk[2], Emk[3], Emk[4], Emk[5], Emk[6], E_pot);
                std::fflush(probe);

                // Snapshot dump for visualisation.
                if (snap_every > 0 && (snap_idx % snap_every == 0)) {
                    char snap_path[768];
                    std::snprintf(snap_path, sizeof(snap_path),
                                  "%s/snap_%04d.bin", snap_dir.c_str(), snap_idx);
                    FILE* sf = std::fopen(snap_path, "wb");
                    if (sf) {
                        // Header: ny, nx, t (double).  Payload: u, v, b as
                        // float32 row-major (ny × nx), one field after the
                        // other.  Python np.fromfile / np.memmap can read this.
                        int32_t hdr[2] = { (int32_t)ny, (int32_t)nx };
                        std::fwrite(hdr, sizeof(int32_t), 2, sf);
                        std::fwrite(&t_now, sizeof(double), 1, sf);
                        int nc = ny * nx;
                        std::vector<float> buf(nc);
                        for (int i = 0; i < nc; ++i) buf[i] = (float)h_u[i];
                        std::fwrite(buf.data(), sizeof(float), nc, sf);
                        for (int i = 0; i < nc; ++i) buf[i] = (float)h_v[i];
                        std::fwrite(buf.data(), sizeof(float), nc, sf);
                        for (int i = 0; i < nc; ++i) buf[i] = (float)h_b[i];
                        std::fwrite(buf.data(), sizeof(float), nc, sf);
                        std::fclose(sf);
                    }
                }
                ++snap_idx;
            };

            diagnostics(0.0);

            std::timespec wall_start;
            clock_gettime(CLOCK_MONOTONIC, &wall_start);

            double t_now = 0.0;
            int samples = 1;
            for (int p = 0; p < n_periods && !g_interrupted; ++p) {
                for (int k = 0; k < steps_per_period && !g_interrupted; ++k) {
                    ansl.step_strang_nonlinear(dt);
                    t_now += dt;
                }
                diagnostics(t_now);
                ++samples;
                if ((p + 1) % 10 == 0 || p == n_periods - 1) {
                    print_progress(t_now, (double)n_periods * T_period,
                                   (p + 1) * steps_per_period, dt, wall_start);
                }
            }
            std::fclose(probe);
            std::fprintf(stderr,
                "  DNS triad-coupled probe written to %s (%d samples)\n",
                probe_path, samples);
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
        wb.hllc_variant = cfg.hllc_variant;
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

#ifdef USE_GPU
    if (helm_loaded) helm_tbl.destroy();
    if (kap_loaded) { kap_tbl_lowT.destroy(); kap_tbl_highT.destroy(); }
#endif

    std::printf("Done.\n");
    return 0;
}
