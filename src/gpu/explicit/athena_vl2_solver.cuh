#pragma once

#include <string>
#include <vector>

// ============================================================
// AthenaVL2Solver — GPU port of Athena++ vl2 Eulerian Godunov
//
// Replicates Stone & Gardiner (2009) van Leer 2-stage predictor-corrector
// unsplit integrator AS IMPLEMENTED in Athena++ (src/task_list/time_integrator.cpp):
//
//   Stage 1 (predictor):  order=1 donor-cell reconstruction of w^n
//                         RiemannSolver(HLLC) → F^(1/2)
//                         u* = u^n − (dt/2) [∂_x F^(1/2) + ∂_y G^(1/2)]
//                         apply source term with β·dt, β = 0.5
//   Stage 2 (corrector):  order=2 PLM (primitive, van-Leer-harmonic
//                         limiter on uniform mesh) reconstruction of w*
//                         RiemannSolver(HLLC) → F^(n+1/2)
//                         u^{n+1} = u^n − dt [∂_x F^(n+1/2) + ∂_y G^(n+1/2)]
//                         apply source term with β·dt, β = 1.0
//
// Riemann: HLLC (Toro 10.4 / Athena++ src/hydro/rsolvers/hydro/hllc.cpp)
// EOS:     adiabatic γ = 5/3 (hardwired for Andrassy2022)
// Source:  variable g(y) gravity (per-row table) + q̇(y) volumetric heating
// Tracer:  nscalars = 1, advected conservatively as s = ρ·X along the
//          hydro fluxes (s_face = (F_ρ ≷ 0 ? X_upwind·F_ρ))
// BC:      x = periodic, y = reflecting (nscalars and hydro match Athena)
//
// Storage (conserved, total values, single register + u1 scratch):
//   u0[5 + 1, ny+2g, nx+2g] = (IDN, IM1, IM2, IM3=0, IEN, s0=ρX)
//
// Ghost width ng = 3 (PLM needs 2, we keep 3 to be generic with future PPM)
//
// Notes on faithfulness:
//  - PLM simple path with uniform mesh, NO characteristic projection
//    (matches --prob=andrassy default xorder=2, characteristic_projection_=false).
//  - "van-Leer harmonic" limiter (minmod_=false default) matches Athena
//    default at xorder=2. Set `minmod` for the `2m` Athena flag.
//  - Stage-1 uses order=1 donor-cell per time_integrator.cpp:1700.
//  - Source term ordering: Athena applies u^{(s)} update first, then
//    AddSourceTerms(dt_stage, flx, prim_old, cons_new) using prim^{n or *}
//    (the stage's initial state), where dt_stage = β·Mesh::dt.
//    We follow the same convention — source uses prim from START of stage.
// ============================================================

struct AthenaVL2Solver {
    // ---- grid ------------------------------------------------
    int nx = 0, ny = 0;
    int ng = 3;
    double dx = 0.0, dy = 0.0;
    double x_lo = 0.0, x_hi = 0.0;
    double y_lo = 0.0, y_hi = 0.0;
    double Lx = 0.0, Ly = 0.0;

    // ---- physics ---------------------------------------------
    double gamma = 5.0 / 3.0;
    double cfl = 0.4;            // user-supplied CFL
    double cfl_limit = 0.5;      // hard vl2 2D stability ceiling (Athena convention)

    // ---- scheme toggles --------------------------------------
    int  limiter = 0;            // 0 = van-Leer harmonic (default), 1 = minmod
    int  xorder = 2;             // 1, 2 supported (3 = PPM is future work)
    // BC in y. Default reflect (Andrassy2022 and other HSE flows). shear_mode
    // / pure-periodic flows set this true via init_shear_mode.
    bool y_periodic = false;

    // ---- tracer ----------------------------------------------
    bool tracer_enabled = false; // nscalars=1 if true

    // ---- state (device, conserved) ---------------------------
    // Stored as SoA: one big allocation per field, size = stride*strideY.
    // Fields: IDN, IM1, IM2, IEN   (IM3 dropped — pure 2D)
    double *d_rho = nullptr, *d_mx = nullptr, *d_my = nullptr, *d_E = nullptr;
    // Predictor scratch (u^*)
    double *d_rho1 = nullptr, *d_mx1 = nullptr, *d_my1 = nullptr, *d_E1 = nullptr;
    // Passive scalar (conservative s = ρ·X)
    double *d_s = nullptr, *d_s1 = nullptr;

    // ---- primitive scratch (same shape as conserved) ---------
    //   w = (ρ, u, v, P)
    double *d_w_rho = nullptr, *d_w_u = nullptr, *d_w_v = nullptr, *d_w_P = nullptr;
    // For passive-scalar, we also carry X = s/ρ (not the density-weighted s).
    double *d_w_X = nullptr;

    // ---- face flux buffers (one per dir, full stride) --------
    //  x-fluxes live at i+1/2 faces → size (nx+1+2g)*(ny+2g)
    //  y-fluxes live at j+1/2 faces → size (nx+2g)*(ny+1+2g)
    double *d_Fx_rho = nullptr, *d_Fx_mx = nullptr, *d_Fx_my = nullptr, *d_Fx_E = nullptr;
    double *d_Fy_rho = nullptr, *d_Fy_mx = nullptr, *d_Fy_my = nullptr, *d_Fy_E = nullptr;
    // Tracer fluxes (scalar advected by F_rho · X_upwind)
    double *d_Fx_s = nullptr, *d_Fy_s = nullptr;

    // ---- source term tables (per-row, device) ----------------
    //  g_row[j] = g(y_j)  on cell centers, ny values (no ghost)
    //  q_row[j] = q̇(y_j)  on cell centers, ny values
    double *d_g_row = nullptr;
    double *d_q_row = nullptr;
    std::vector<double> h_g_row;
    std::vector<double> h_q_row;

    // ---- CFL scratch -----------------------------------------
    double *d_cfl_buf = nullptr;

    // ---- bookkeeping -----------------------------------------
    int step_count = 0;
    double dt_current = 0.0;

    // ---- public API ------------------------------------------
    void init(int nx, int ny, double Lx, double Ly,
              double gamma, double cfl);
    void destroy();

    // Andrassy 2022 IC.  Reads the same 6-col slab as cart_ale2.init_andrassy2022:
    //   header: Ly Lx g_y gamma rho_top P_top T_top mu
    //   data:   (ny+1 face rows)  y ρ P T g q̇
    // Applies Eq. 6 δρ/ρ = Δ·[q̇(y)/q̇₀]·[sin(3π x') + cos(π x')] with
    // x' = 2(x - x_lo)/Lx − 1 ∈ [−1, 1].  Optionally add deterministic noise
    // via splitmix64(seed, i, j) so each seed gives a distinct realization.
    void init_andrassy2022(const std::string& slab_file,
                           double delta_rho_amp = 5.0e-5,
                           int noise_seed = -1,
                           double noise_amp = 0.0);

    // T3 linear shear-mode decay (ν_eff probe). Uniform rho/P, g=0,
    // vx = V0·sin(k·2π y/Ly), vy=0, periodic BC in both directions.
    // Analytic NS decay: max|vx|(t) = V0·exp(-ν·k²·t) with k = k·2π/Ly.
    void init_shear_mode(double rho, double P, double V0, int k);

    // Standard Sod shock tube in x: ρL=1, PL=1 for x<Lx/2, ρR=0.125,
    // PR=0.1 for x>=Lx/2.  v=0, γ=1.4.  BC: x-periodic by default (for
    // use with analytic comparison at a short t before the wave hits
    // the wrap), y-reflect.  Unit test uses a very short t (0.2).
    void init_sod();

    // T1 entropy wave (x-advection, periodic both dirs). Same definition
    // as CartAle2::init_entropy_wave: ρ = ρ0(1 + A·sin(k·2π x/Lx)),
    // P = P0 uniform, v = (u0, 0). After one period t = Lx/u0 the exact
    // solution returns to the IC.
    void init_entropy_wave(double rho0, double P0, double u0,
                           double A, int k);

    // One vl2 predictor-corrector step.  Returns the dt actually used.
    double step(double t, double t_end);

    // Diagnostics (computed host-side after download).
    struct Diagnostics {
        double total_mass;
        double total_KE;
        double total_internal_E;
        double total_PE;       // Σ ρ·Φ(y) dV, using Φ(y) = −∫₀ʸ g dy
        double total_E;        // KE + IE + PE
        double max_v;
        double max_mach;
    };
    Diagnostics compute_diagnostics();
    double total_species_mass();

    // VTK dump — same RECTILINEAR_GRID format cart_ale2 uses.
    void write_vtk_2d(const char* filename, double Lx, double Ly);

    // T1 entropy wave post-run error (Athena++ compute_error pattern).
    // See CartAle2Solver::compute_entropy_wave_error for the schema.
    // Writes one appended line to <run_dir>/entropy_wave-errors.dat.
    void compute_entropy_wave_error(double t_now, int ncycle,
                                    double rho0, double P0, double u0,
                                    double A, int k, double periods,
                                    const std::string& run_dir);

    // ---- internal helpers ------------------------------------
    int stride_x() const { return nx + 2 * ng; }
    int stride_y() const { return ny + 2 * ng; }
    int total_cells() const { return stride_x() * stride_y(); }
    int total_fx() const { return (nx + 1 + 2 * ng) * (ny + 2 * ng); }
    int total_fy() const { return (nx + 2 * ng) * (ny + 1 + 2 * ng); }

    void fill_ghost();
    void cons_to_prim();
    void calc_hydro_flux(int order);
    void apply_flux_divergence(double dt_stage);
    void add_source_terms(double dt_stage, bool from_u0);
    double compute_dt();
    // Stage workhorse: given a "read u" (via primitive scratch) produce
    // a "write u" = u^n − dt_flux·∇·F + dt_src·source(prim).
    void stage_advance(int stage, double dt);

    // Potential Φ(y_j) at cell-centered row j ∈ [0, ny),
    // Φ_j = Φ_{j-1} + g(y_j)·dy with Φ_0 anchored so Φ(y_lo) = 0.
    std::vector<double> h_phi_row;

    // Δρ_amp used at IC time, echoed to stderr for reproducibility.
    double ic_delta_amp = 0.0;
};
