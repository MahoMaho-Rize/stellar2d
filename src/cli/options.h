#pragma once

#include "hydro/reconstruct.h"  // Limiter enum

#include <array>
#include <cstdint>
#include <string>
#include <vector>

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
    std::string ic_mesa_path;        // non-empty ⇒ take IC from scripts/mesa/convert_mesa_ic.py output
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
    int cart_ale2_rebuild_order = 0; // cart_ale2 node v-rebuild: 0=1st-order mass-weighted avg (default, stable on stratified/reflect), 1=2nd-order corner MUSCL w/ Barth-Jespersen (experimental — stable on smooth periodic flow, but 2nd-order remap + 2nd-order rebuild + reflect wall + long-t stratification triggers atmosphere-mode instability as of 2026-05-07)
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
    // Diagnostic tracer (cart_ale2_trace.h): per-cell cumulative KE/IE
    // accounting + per-step pick-cell time series. Fully VRAM-buffered;
    // CSV flushed only at VTK boundaries (cum) and at run end (picks).
    std::string cart_ale2_trace_cells; // "ic,jc;ic,jc;..." (empty = disabled)
    int cart_ale2_trace_step_cap = 0;  // max step rows per pick cell (0 = disabled)
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
    // cart_ale2 --test andrassy2022 (A&A 659 A193 idealized O-shell benchmark):
    // uses init_andrassy2022() which reads a 6-col slab from build_ic.py
    // (y, ρ, P, T, g(y), q̇(y)) and applies Eq. 6 density perturbation.
    double cart_ale2_andrassy_amp  = 5.0e-5;  // Eq. 6 δρ/ρ amplitude (paper value)
    int    cart_ale2_andrassy_seed = -1;      // -1 = paper-exact; ≥0 adds noise
    double cart_ale2_andrassy_noise = 0.0;    // ensemble noise amplitude
    // athena_vl2 scheme knobs (only used when --solver athena_vl2):
    //   xorder: 1 = donor-cell only (for debugging), 2 = PLM (default, matches
    //           Athena++ --input xorder=2). Stage-1 of vl2 is ALWAYS order=1
    //           (that's the vl2 design); xorder=1 forces stage-2 also to DC.
    //   limiter: 0 = van-Leer harmonic (Athena default at xorder=2),
    //            1 = minmod (Athena "2m" flag).
    int athena_vl2_xorder = 2;
    int athena_vl2_limiter = 0;
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
    // ---- T3 shear_mode (scheme characterization ν_eff probe) ----
    // Domain defaults to Lx=Ly=1.0, periodic both dirs, uniform rho=1, P=1.
    // IC: vx = shear_V0 · sin(shear_k · 2π y / Ly), vy = 0, g = 0.
    // Applies to --test shear_mode for cart_ale2 and athena_vl2.
    double shear_V0 = 0.01;
    int    shear_k  = 1;
    double shear_rho = 1.0;
    double shear_P   = 1.0;
    // ---- T1 entropy_wave (smooth convergence probe) ----
    // IC: ρ = ewave_rho0·(1 + ewave_A·sin(ewave_k·2π x/Lx)),
    //     P = ewave_P0, v = (ewave_u0, 0).  Periodic both dirs.
    //     t_end set to `ewave_periods · Lx / u0` in the driver.
    double ewave_rho0 = 1.0;
    double ewave_P0   = 1.0;
    double ewave_u0   = 1.0;
    double ewave_A    = 0.01;
    int    ewave_k    = 1;
    double ewave_periods = 1.0;
    // --compute-error: when set, the solver's compute_*_error() is called
    // at t_end for the active test_case (Athena++ compute_error pattern).
    // Emits <run-base>/<test_case>-errors.dat; pytest tst/ reads it.
    bool compute_error = false;

    bool radial_only = false;  // enforce v_theta=0, skip theta-direction work (FAS/explicit only)
    double r_inner = -1.0;  // auto-set for mass mesh; override with --r-inner
    double M_core = 0.0;
};

// Parse argv into cfg. Returns 0 on success, non-zero on usage error (value
// is suitable for returning from main).
int parse_cli(int argc, char** argv, SimConfig& cfg);
