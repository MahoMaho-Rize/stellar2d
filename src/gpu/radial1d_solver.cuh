#pragma once

#include "../grid.h"
#include "../state.h"
#include "../eos.h"
#include "../physics/opacity_table.cuh"

// 1D radial Lagrangian stellar hydrodynamics solver (MESA RSP-inspired).
//
// Primary design goals:
//   - Exact mass conservation (Lagrangian mass zones, dm_k fixed in time)
//   - Well-posed origin: v_center = 0 pinned at r = 0 (no wedge cell pathology)
//   - Stable surface: P_surf floor prevents rhoE collapse to KE
//   - Tscharnuter-Winkler artificial viscosity: shock-triggered only
//   - Compression-based dt limiter (not just acoustic CFL)
//
// Staggering:
//   - Faces (interfaces): r[k], v[k], k = 0..nz
//     r[0]    = 0 (center, pinned)
//     r[nz]   = R_star (surface)
//     v[0]    = 0 (reflective BC at center)
//   - Zones (cells):    Vol[k], e_int[k], rho[k], P[k], Pvsc[k], dm[k], k = 0..nz-1
//     Zone k sits between face k and face k+1
//     dm[k] = rho[k] * Vol_3d[k] is zone mass (constant over time)
//
// Equations (Lagrangian):
//   dr_k/dt        = v_k                           (face motion)
//   dv_k/dt        = -A_k * (XP_k - XP_{k-1}) / dm_bar_k - g_k   (momentum)
//   d(rho e)_k/dt  = -P_tot_k * dV_k/dt / V_k      (adiabatic)
// where
//   A_k       = 4π r_k²                                          (face area)
//   dm_bar_k  = 0.5 * (dm_{k-1} + dm_k)                          (mass around face)
//   XP_k      = P_k + Pvsc_k                                      (total pressure)
//   g_k       = G * M_k / r_k²                                    (gravity, M_k = Σ dm_j, j<k)
//   Pvsc_k    = (CQ/V_k) * max(0, Δv_k - ZSH√(PV))²              (artif viscosity)
//   Δv_k      = v_k - v_{k+1}  > 0 means compression
//
// Boundary conditions:
//   v[0] = 0 (pinned, reflective at center)
//   At the surface face k = nz: use P_surf_floor (ghost zone pressure) when computing XP gradient

struct Radial1DLevel {
    int nz;  // number of zones

    // Face-centered (size nz+1)
    double* d_r;        // face radius
    double* d_v;        // face velocity
    double* d_M;        // enclosed mass at face k (M[0] = 0, M[nz] = total)
    double* d_gr;       // face gravity g = G*M/r² (g[0] = 0 by L'Hopital)
    double* d_r_prev;   // previous face radius (for RK2)
    double* d_v_prev;   // previous face velocity

    // Zone-centered (size nz)
    double* d_dm;       // zone mass (constant)
    double* d_Vol;      // zone volume V = (4π/3)(r[k+1]³ - r[k]³)
    double* d_rho;      // zone density = dm / Vol
    double* d_e_int;    // zone specific internal energy
    double* d_P;        // zone gas pressure (from EOS)
    double* d_Pvsc;     // zone artificial viscosity
    double* d_rhoE;     // zone total energy density (for diagnostics / save)
    double* d_Vol_prev; // previous volume (for energy update)
    double* d_e_prev;   // previous specific internal energy

    // HSE reference (for diagnostics and surface floor)
    double* d_rho0;     // initial density profile (zone-centered)
    double* d_P0;       // initial pressure profile
    double* d_r0;       // initial face radii

    // Scratch
    double* d_dt_cell;  // per-zone dt (for reduction)
    double* d_scratch;  // generic scratch for reductions
};

// Forward-declared radiation diffusion params (defined in physics/radiation_diffusion.cuh)
struct RadDiffParams;

struct Radial1DSolver {
    // Physical / numerical parameters
    double gamma = 5.0/3.0;
    double G_const = 1.0;
    double cfl = 0.4;

    // Optional EOS. If use_eos == true, all primitive/CFL/diagnostic kernels
    // use the EOS struct (supporting ideal_rad, PRE_MS, etc.). Otherwise they
    // fall back to P = (γ-1)·ρ·e and cs = √(γP/ρ).
    bool use_eos = false;
    EOS eos;

    // Nuclear burning (pp-chain, simplified). If enabled, every hydro step adds
    // ε_pp(ρ, T, X) · dt to internal energy, and optionally burns X → Y when
    // species_enabled == true.
    bool nuclear_enabled = false;
    double nuc_X = 0.7;              // initial hydrogen mass fraction (fallback if no species)
    double nuc_Y = 0.28;             // initial helium mass fraction
    double nuc_T_floor = 1.0e6;
    double nuc_T_scale = 1.0;        // T_K = T_code · T_scale (1 for cgs, e.g. 1e8 for code units)
    double nuc_epsilon_scale = 1.0;
    double nuc_q_burn = 6.4e18;      // erg/g per gram of H burnt (26.73 MeV / 4 m_p)
    // Dynamic time-scale compression: pick ε_scale each step so that
    // ε·dt / (cv·T) ≤ nuc_compress_frac. Disabled when ≤ 0.
    double nuc_compress_frac = 0.0;

    // Species tracking. When enabled, per-zone X (hydrogen) and Y (helium) are
    // evolved by the nuclear kernel. In Lagrangian coords there is no advection
    // (mass parcels don't mix unless we add diffusion), so species evolve only
    // via burning.
    bool species_enabled = false;
    double* d_X = nullptr;           // (nz) hydrogen mass fraction
    double* d_Y = nullptr;           // (nz) helium mass fraction

    // Phase D: 13-species α-chain. Co-exists with (d_X, d_Y) but is the only
    // path that couples to alpha_net::advance_substep(). species_mode switches
    // which buffer the nuclear operator-split reads/writes.
    //   pp      : legacy X→Y burn through pp_burn_kernel (default)
    //   alpha13 : α-chain operator split (He4..Ni56), no pp-chain
    // Layout: d_X_spec[k * N_SPEC + s], N_SPEC=13 fixed by alpha_net.
    enum SpeciesMode : int { SPEC_PP = 0, SPEC_ALPHA13 = 1 };
    int species_mode = SPEC_PP;
    double* d_X_spec = nullptr;      // (nz * 13) alpha-chain composition
    // Temperature below which the α-chain burn kernel is a no-op.  rhs()
    // already short-circuits below 1e5 K; this floor avoids kernel launch
    // cost in the cold envelope.  10⁸ K is safely below C+O burning
    // threshold (T₉≈0.5) and 3α resonance (T₉≈0.1 for explosive).
    double alpha_burn_T_min = 1.0e8;

    // Radiation diffusion (explicit subcycled). If enabled, every hydro step
    // calls apply_radiation_diffusion(dt) which subcycles at the parabolic
    // radiation CFL (dt_rad ≪ dt_hydro typically; 1-10k subcycles per step).
    bool radiation_enabled = false;
    double rad_c_light = 1.0;
    double rad_a_rad = 1.0;
    // Opacity params stored as raw doubles (POD-safe, OpacityParams has
    // default constructor so we just need to hand over values).
    double rad_kappa_es = 0.2;
    double rad_kappa_ff_0 = 4.3e24;
    double rad_kappa_dust_0 = 2.0e-4;
    double rad_kappa_Hm_0 = 1.1e-25;
    double rad_T_dust_off = 1500.0;
    // Optional MESA-style tabulated opacity. Owned in main.cpp (lifetime
    // spans the solver); if either view is uninitialised we fall back to
    // the analytic grey formula. Use set_kap_tables() to attach.
    KapTableView kap_view_lowT  = {};
    KapTableView kap_view_highT = {};
    bool         kap_use_table  = false;
    double       kap_logT_lo_end   = 3.9;
    double       kap_logT_hi_start = 4.1;
    double       kap_hydrogen_X    = 0.7;   // slice at this X in the 3-D table
    // Scratch arrays for radiation diffusion
    double* d_T_work = nullptr;   // (nz)
    double* d_F_work = nullptr;   // (nz+1)
    double* d_dt_rad = nullptr;   // (nz)

    // Artificial viscosity (Tscharnuter-Winkler)
    double CQ = 2.0;      // viscosity coefficient
    double ZSH = 0.1;     // shock-detection threshold (fraction of sound crossing speed)

    // Surface pressure floor
    double P_surf_floor = 0.0;   // P at ghost zone beyond nz-1; set to initial P0[nz-1]

    // Inner boundary (Phase D): when M_inner > 0, there's a gravitating
    // point mass inside r[0] (proto-NS after core-collapse mass cut).
    // M_inner is ADDED into M[0] by k_rad1d_enclosed_mass_offset; gravity
    // at the inner face then follows g[0] = G·M_inner/r[0]².  v[0] is
    // still pinned to 0 (rigid piston wall).  M_inner == 0 and r[0] == 0
    // is the legacy r_in=0 origin boundary.
    double M_inner = 0.0;        // proto-NS mass (g)
    double r_inner_init = 0.0;   // r[0] at IC (cm) — diagnostic only

    // Compression dt limit (MESA: dt ≤ fraction * Δr / Δv)
    double comp_dt_fraction = 0.1;

    // State
    Radial1DLevel lev;
    double dt_current = 0.0;
    int step_count = 0;
    bool hse_set = false;

    // Populate an OpacityParams from the solver's rad_kappa_* fields +
    // optional table views. Inlined in the .cu once the include is in
    // scope; declared here so any caller can fill it consistently.
    void fill_opacity_params(struct OpacityParams& opa) const;

    // Lifecycle
    void init(int nz, double gamma, double G, double cfl);
    void destroy();

    // Initialize from a Lane-Emden polytrope (equal-mass shells).
    // Central density rho_c, polytropic constant K, index n.
    void init_lane_emden(double rho_c, double K_poly, double n_poly);

    // Initialize from a MESA profile written by scripts/convert_mesa_ic.py.
    // Re-binned to the solver's own `nz` Lagrangian shells by equal-mass
    // shells; (rho, T, P, e, X, Y) interpolated linearly in m_enc. Sets
    // v=0 everywhere, fills d_rho0/d_P0/d_r0 with the mapped HSE state,
    // initialises species (if `species_enabled`) from the profile.
    //
    // If `seed_T` is true (Helmholtz only), (e, P) are re-derived from a
    // forward helm_eval(ρ, T) instead of inverting MESA's P. This pins
    // our runtime T to MESA's T to machine precision, which in turn
    // makes κ(ρ, T, X) agree with MESA to ≤1 % (14 % → 1 % in practice).
    // The downside is MESA's P is slightly overwritten (1 % level) to
    // whatever Helm says at the same (ρ, T); HSE is re-balanced when
    // the first hydro step runs.
    //
    // Returns 0 on success, non-zero on parse/load failure.
    int init_from_mesa(const char* ic_path, bool seed_T = false,
                       int n_atm_zones = 0);

    // Phase D: load a Sukhbold+2018 13-species IC (output of
    // scripts/n49b/convert_sukhbold_ic.py).  On success M_inner and r[0]
    // are set from the header, 13-species X_spec is uploaded, species_mode
    // is switched to ALPHA13.
    //
    // bomb_E and bomb_dm specify a thermal bomb: e_int += bomb_E / bomb_dm
    // distributed over the innermost zones totalling bomb_dm in mass,
    // above r_inner.  Set bomb_E = 0 to skip.
    int init_from_sukhbold(const char* ic_path,
                           double bomb_E = 0.0,
                           double bomb_dm = 0.1 /*Msun*/);

    // Apply a radial-only pressure perturbation: P *= (1 + amp*sin(π r/R))
    // Also adjust density self-consistently for adiabatic perturbation:
    // rho *= (1 + amp*sin(πr/R))^(1/γ)  (isentropic along the perturbation)
    // Note: consistency with the 2D test requires a *density* perturbation
    //       ρ' = ρ₀ (1 + amp sin(πr/R)),
    //       P' = P₀ (1 + γ amp sin(πr/R))   (adiabatic).
    // See init_lane_emden_perturbed() in src/init/lane_emden.cpp.
    void apply_perturbation(double amplitude);

    // Time step (explicit RK2 with compression-limited dt)
    double step(double t, double t_end);

    // Compute dt from both acoustic CFL and compression limit
    double compute_dt();

    // Diagnostics: download profile to host (size nz zones, nz+1 faces)
    void download_profile(std::vector<double>& r_face,
                          std::vector<double>& v_face,
                          std::vector<double>& rho_cell,
                          std::vector<double>& P_cell,
                          std::vector<double>& e_cell);

    // Rich profile emission used by the Tier-2 MESA PK. Adds the physical
    // state variables a MESA profile exposes: T [K], κ_R [cm²/g], Γ₁,
    // ∇_ad, ∇_rad, L_rad at outer face [erg/s], convective-velocity
    // proxy and mixing-type flag. Single launch per call; all fields
    // computed from the current (r, v, ρ, P, e_int, X) state.
    //
    // mixing_type: 0 = stable (∇_rad < ∇_ad)
    //              1 = convective (∇_rad > ∇_ad)
    // conv_vel:    Böhm-Vitense v_conv estimate (0 in stable zones)
    void download_profile_rich(std::vector<double>& r_face,
                               std::vector<double>& v_face,
                               std::vector<double>& rho_cell,
                               std::vector<double>& P_cell,
                               std::vector<double>& e_cell,
                               std::vector<double>& T_cell,
                               std::vector<double>& kap_cell,
                               std::vector<double>& gamma1_cell,
                               std::vector<double>& grada_cell,
                               std::vector<double>& gradr_cell,
                               std::vector<double>& L_face,
                               std::vector<int>&    mixing_type,
                               std::vector<double>& conv_vel);

    // Compute conservation diagnostics (mass, total energy) on GPU
    struct Diagnostics {
        double total_mass;
        double total_KE;
        double total_internal_E;
        double total_grav_E;
        double total_E;
        double max_mach;
        double max_vr;
        double T_c;          // core temperature (innermost zone) — for ignition tracking
        double rho_c;        // core density
        double L_nuc;        // ∫ ε_pp·dm over all zones [erg/s] — 0 if --nuclear off
    };
    Diagnostics compute_diagnostics();

    // Snapshot current state as HSE reference (for floors)
    void snapshot_hse();

    // Apply radiation diffusion for total time dt_total using explicit
    // subcycles at the parabolic radiation CFL. Returns number of subcycles.
    int apply_radiation_diffusion(double dt_total);

    // Implicit (BE) radiation diffusion: single backward-Euler step with
    // Picard linearization of T⁴, tridiagonal solve. Includes Stefan-Boltzmann
    // photosphere BC at the outer face (F_surf = σ T_surf⁴ · A).
    // Unconditionally stable → can take arbitrary dt, suitable for τ_KH.
    // Returns number of Picard iterations.
    int apply_radiation_diffusion_implicit(double dt_total, int k_start = 0);
    int rad_impl_last_picard = 0;     // diagnostic: iterations last call
    double rad_impl_L_surf = 0.0;     // diagnostic: surface luminosity last call
    int    rad_impl_phot_zone = 0;    // diagnostic: τ=2/3 photosphere zone index
    double rad_impl_T_phot    = 0.0;  // diagnostic: T at photosphere [K]
    double rad_impl_tau_surf  = 0.0;  // diagnostic: total τ integrated down

    // ============================================================
    // MLT convection — Phase 6
    // ============================================================
    // Schwarzschild criterion: zone is convective if ∇_rad > ∇_ad.
    //   ∇_rad = d ln T / d ln P from radiative flux (3 κ ρ L P) / (16 π a c G M T⁴)
    //   ∇_ad  = (γ-1)/γ for ideal gas (corrected per EOS)
    // Compute on GPU each time diagnostics are requested. NOT fed back into
    // the solver yet — that's task #25.
    struct ConvectionDiag {
        double conv_mass_frac;   // Σ dm[k where ∇_rad > ∇_ad] / M_total
        double r_conv_inner;     // innermost radius of convection zone
        double r_conv_outer;     // outermost radius
        double max_superadiab;   // max_k (∇_rad − ∇_ad)
        int    n_conv_zones;     // count of convective zones
    };
    ConvectionDiag compute_convection_diag();

    // MLT is OFF by default (only diagnostic). Setting true wires MLT flux
    // into the radiation BE solve — implemented in task #25.
    bool mlt_enabled = false;
    double mlt_alpha = 1.5;    // Böhm-Vitense mixing-length / pressure scale height
    double* d_K_conv = nullptr; // (nz) per-zone MLT conductivity scratch

    // ============================================================
    // Phase 4: Implicit Backward-Euler + JFNK
    // ============================================================
    // State packed as (v[1..nz], r[1..nz], e[0..nz-1]); total DOF = 3·nz.
    // k=0 face pinned (v=0, r=0). Surface face k=nz uses P_surf_floor ghost.
    //
    // F(U) = (U - U^n)/dt - (R(U) - R_hse) = 0
    //
    // N_dof = n_fields · nz.  n_fields = 3 (v, r, e); species stays
    // operator-split for now (burn after Newton converges).
    bool implicit_enabled = false;
    int n_fields = 3;                  // (v, r, e); reserved 4 if species fold in
    int N_dof = 0;                     // n_fields · nz

    // Newton-Krylov scratch
    double* d_U = nullptr;             // current packed state, size N_dof
    double* d_Un = nullptr;            // U^n (saved at step start), size N_dof
    double* d_Ubak = nullptr;          // backup for JFNK perturbation, size N_dof
    double* d_R = nullptr;             // R(U), size N_dof
    double* d_R_hse = nullptr;         // R(U_hse), size N_dof (well-balanced)
    double* d_F = nullptr;             // F(U), size N_dof
    double* d_Fk = nullptr;            // F(U_k) cached during Newton, size N_dof
    double* d_scale_L = nullptr;       // Viallet L, size N_dof
    double* d_scale_R = nullptr;       // Viallet R, size N_dof
    double* d_scale_invL = nullptr;    // Viallet 1/L, size N_dof

    // GMRES(K)
    static constexpr int GMRES_K = 30;
    double* d_V[GMRES_K + 1] = {};     // Krylov basis
    double* d_Z[GMRES_K] = {};         // preconditioned directions
    double* d_gmres_w = nullptr;

    // ============================================================
    // Block-tridiag preconditioner (plan: autodiff_jacobian_plan.md §B).
    // Approximates the scaled Jacobian A = invL·J·R (the same matrix
    // jfnk_matvec_implicit produces) as a block-tridiagonal operator
    // with 3×3 blocks (one per zone, fields v/r/e). Assembled via 9
    // colored FD matvecs (3 colors × 3 fields), refreshed each Newton
    // step. Block-Thomas forward/backward sweep provides GMRES PC.
    //
    // Layout (row-major, per zone i ∈ [0, nz)):
    //   A_diag  [i]  : 3×3 block, column in U at zone i
    //   A_lower [i]  : 3×3 block, column in U at zone i-1 (lower diag; A_lower[0] unused)
    //   A_upper [i]  : 3×3 block, column in U at zone i+1 (upper; A_upper[nz-1] unused)
    // Blocks stored row-major in 9 doubles each.
    bool precond_tridiag = false;      // toggle: off = identity PC (legacy)
    // AD (forward-mode Dual) J·v matvec. Off = finite-difference (legacy,
    // cgs noise floor ~1e-3 relative). On = exact J·v via Dual<1>, one
    // residual evaluation per matvec, no noise floor.
    bool jfnk_autodiff = false;
    // Operator-split BE radiation (legacy path). Default OFF — rad is now
    // inside the implicit residual R(U), so Newton solves hydro + rad in
    // one step. Keep this flag for A/B comparisons.
    bool rad_be_split = false;
    // Split architecture: when nz_atm_split > 0, the outer nz_atm_split
    // zones are treated as a 1D atmosphere. Inside Newton, atm zones skip
    // the rad diffusion term (their e is effectively frozen from Newton's
    // POV). After Newton converges, operator-split BE-rad runs ONLY over
    // those atm zones to evolve T(τ) toward radiative equilibrium.
    // Physics: atm diffusion time ~(Δr)²·3κρ/c ≈ 10⁻⁷ s, already in
    // equilibrium on any reasonable implicit dt ≥ μs.
    int  nz_atm_split = 0;
    // Diagnostic: disable R_hse subtraction in F. When ON, F = (U-Uⁿ)/dt - R(U),
    // so Newton sees the full residual (gravity + pressure + rad). Tests
    // whether the well-balanced HSE subtraction absorbs rad-driven evolution.
    bool no_rhse_subtract = false;
    // Min photospheric T used in Stefan surface BC. Prevents atmospheric
    // cooling below the Helm-table floor (1000 K) from stalling KH — the
    // physical photospheric T of a pre-MS 1 M☉ on Hayashi is 3000-4000 K
    // (H⁻ opacity self-regulation). 0 = no floor.
    double rad_T_phot_floor = 0.0;
    double* d_A_diag  = nullptr;       // nz · 9 doubles
    double* d_A_lower = nullptr;
    double* d_A_upper = nullptr;
    // Thomas sweep scratch (host-side, nz small)
    double* d_thomas_rhs = nullptr;    // nz · 3 doubles (working rhs, copies of subvector)
    double* d_thomas_y   = nullptr;    // nz · 3 doubles (intermediate)
    double* d_matvec_scratch = nullptr;// N_dof working buffer for PC assembly probes

    // Params
    bool use_viallet_scaling = true;
    double viallet_alpha1 = 1e-5;
    double viallet_alpha2 = 1.0;
    double newton_tol = 1e-8;
    // Relative convergence target: accept if ||F||_new < newton_rel_tol·init_res.
    // 1e-4 = four orders drop from initial. Previous 0.5 (2× cut) let Newton
    // stop too early at high resolution.
    double newton_rel_tol = 1e-9;
    int newton_max_iter = 15;
    double gmres_tol = 1e-6;   // Tightened from 1e-3: at nr=1024 GMRES was
                               // exiting at j=1 with false convergence (PC
                               // too well-aligned to first Krylov direction
                               // but not to the global solution).
    int gmres_max_iter = GMRES_K;

    // If >0, re-snapshot R_hse every N accepted steps to track evolving
    // quasi-static state (KH contraction / ignition). Keeps F small when
    // the initial HSE reference no longer reflects the solution.
    int hse_resnap_interval = 0;

    // Public API (when implicit_enabled)
    void init_implicit();              // allocates implicit scratch, called from init() when needed
    void destroy_implicit();
    void pack_state_from_device();     // writes (d_v, d_r, d_e_int) → d_U
    void unpack_state_to_device();     // writes d_U → (d_v, d_r, d_e_int) + refreshes primitives
    void snapshot_hse_implicit();      // computes d_R_hse from current HSE state
    // Picard-lag refresh of MLT conductivity K_conv[0..nz-1] from current ρ/P/T.
    // Called once per Newton iteration so the rad-in-F residual can include a
    // convective-flux term F_conv ∝ K_face · (T_L − T_R), linear in T. K stays
    // constant during a Newton solve but updates across Newton outer iters.
    void refresh_K_conv_implicit();
    double step_implicit(double t, double t_end, double dt_try);  // returns dt actually taken
    int    newton_solve_implicit(double dt);
    int    gmres_solve_implicit(double* d_x, const double* d_b, double inv_dt, double tol, int max_iter);
    void   jfnk_matvec_implicit(const double* d_v_in, double* d_Jv, double inv_dt);
    // Exact J·v via Dual<1> — same call signature as FD matvec above but
    // uses the autodiff F kernel (radial1d_residual_dual.cuh). No α / FD.
    void   jfnk_matvec_ad(const double* d_v_in, double* d_Jv, double inv_dt);
    void   apply_precond_implicit(const double* d_v_in, double* d_Mv, double inv_dt);
    // Build the block-tridiag preconditioner approximation of the current
    // scaled Jacobian. Calls jfnk_matvec_implicit 9 times (3 colors × 3 fields)
    // to extract the 9-entry 3×3 block per zone. Cheap: 18 F evals vs ~60 in a
    // 30-iter GMRES cycle.
    void   build_precond_tridiag(double inv_dt);
    // Apply block-Thomas forward/backward sweep: M⁻¹ · v_in → Mv.
    void   apply_precond_tridiag(const double* d_v_in, double* d_Mv);
    // Launch the appropriate enclosed-mass kernel.  If M_inner > 0, calls
    // the offset variant so M[0] = M_inner (for proto-NS inner boundary).
    void   launch_enclosed_mass();
    void   compute_R_implicit();       // evaluates R(U) into d_R using current (d_v,d_r,d_e_int)
    void   compute_F_implicit(double inv_dt);  // d_F = (d_U - d_Un)/dt - (d_R - d_R_hse)
    double residual_norm_implicit();
    void   build_scaling_implicit();

    // Initialize uniform X/Y profiles (called after species buffers allocated).
    void init_species_uniform(double X0, double Y0);

    // Download species profile (size nz zones).
    void download_species(std::vector<double>& X_cell,
                          std::vector<double>& Y_cell);

    // Phase D alpha-chain helpers. Layout: X_host[k*13 + s].
    // init_species_alpha copies nz·13 doubles to d_X_spec; if species_mode is
    // not already ALPHA13 it is set here. download mirrors the device buffer
    // back into X_host (resized if too small).
    void init_species_alpha(const double* X_host);
    void download_species_alpha(std::vector<double>& X_host);
};
