#pragma once

#include <vector>
#include <string>

// Cartesian 2D staggered quad ALE (Arbitrary Lagrangian-Eulerian).
//
// Pipeline per step:
//   Phase L: Caramana-compatible Lagrangian substep
//            (identical numerics to cart_lag: kick-drift-kick,
//             subcell force ½(edge_km + edge_k), compatible energy).
//   Phase R: full Eulerian rezone — nodes snapped back to X0, Y0.
//   Phase M: swept-edge conservative remap of {dm, dm·e_int, p_x, p_y}
//            via donor-cell upwind flux on the (small) displacement.
//
// Topology: uniform (nx, ny) cells on [0, Lx] × [0, Ly]. Nodes never
// actually leave the initial uniform lattice — but the swept area of
// each edge (A→A', B→B') is what carries conserved quantities.
//
// This is a separate solver from cart_lag. cart_lag stays as a
// pure-Lagrangian baseline / reference.
//
// References:
//   - Caramana, Shashkov, Whalen 1998 (JCP 144) — compatible hydro
//   - Kucharik & Shashkov 2012 (JCP 231) — swept remap
//   - hal3d omp3/lagrange.c, advection.c — reference implementation

struct CartAle2Solver {
    // ---- Sizes ----
    int nx = 0, ny = 0;
    int nnode_x = 0, nnode_y = 0;
    int nnode = 0;
    int ncell = 0;
    int nsub = 0;                  // 4 * ncell

    // ---- Node arrays (nnode) ----
    double *d_X = nullptr, *d_Y = nullptr;        // current positions
    double *d_X0 = nullptr, *d_Y0 = nullptr;      // rezone target (uniform, never changes)
    double *d_vX = nullptr, *d_vY = nullptr;      // node velocities
    double *d_FX = nullptr, *d_FY = nullptr;      // net node forces
    double *d_mnode = nullptr;                    // node mass (= 0.25 Σ_adj dm)
    double *d_dX = nullptr, *d_dY = nullptr;      // Lagrangian-phase displacements

    // ---- Cell arrays (ncell) ----
    double *d_dm = nullptr;
    double *d_Vol = nullptr;
    double *d_Area0 = nullptr;
    double *d_rho = nullptr;
    double *d_e_int = nullptr;
    double *d_P = nullptr, *d_Q = nullptr, *d_cs = nullptr;
    double *d_minheight = nullptr;
    double *d_strain_rate = nullptr;

    // ---- Cell-centered momentum (for remap) ----
    double *d_px_cell = nullptr;     // Σ m_node_frac · vx  per cell (conservative)
    double *d_py_cell = nullptr;

    // ---- Scratch buffers for remap ----
    double *d_dm_new = nullptr;      // new cell mass after remap
    double *d_ie_new = nullptr;      // new cell internal-energy-density-integrated (dm·e)
    double *d_px_new = nullptr;      // new cell momentum
    double *d_py_new = nullptr;

    // ---- 2nd-order remap (MUSCL-in-remap, Kucharik-Shashkov 2012) ----
    // Per-cell densities on the reference uniform grid.
    double *d_rho_dens  = nullptr;   // dm/V0
    double *d_rhoE_dens = nullptr;   // (dm·e_int)/V0
    double *d_pxd_dens  = nullptr;   // px/V0
    double *d_pyd_dens  = nullptr;   // py/V0
    // Per-cell minmod-limited slopes (∂f/∂x, ∂f/∂y) for each of the 4 fields.
    double *d_rho_sx  = nullptr, *d_rho_sy  = nullptr;
    double *d_rhoE_sx = nullptr, *d_rhoE_sy = nullptr;
    double *d_pxd_sx  = nullptr, *d_pxd_sy  = nullptr;
    double *d_pyd_sx  = nullptr, *d_pyd_sy  = nullptr;
    // PPM face values (Colella-Woodward 1984): each cell owns left/right
    // face values in x (L/R subscript) and bottom/top in y (D/U subscript).
    // Parabolic profile inside cell:
    //   f(ξ_x, ξ_y) ≈ f_bar + s_x·(ξ_x−½)·dx + s_y·(ξ_y−½)·dy
    //                 + Δ²_x·((ξ_x−½)²−1/12) + Δ²_y·((ξ_y−½)²−1/12)
    // For remap swept integration we only need point evaluation, done in
    // the 2nd-order PPM remap kernels. Same four fields as MUSCL.
    double *d_rho_xL  = nullptr, *d_rho_xR  = nullptr;
    double *d_rho_yD  = nullptr, *d_rho_yU  = nullptr;
    double *d_rhoE_xL = nullptr, *d_rhoE_xR = nullptr;
    double *d_rhoE_yD = nullptr, *d_rhoE_yU = nullptr;
    double *d_pxd_xL  = nullptr, *d_pxd_xR  = nullptr;
    double *d_pxd_yD  = nullptr, *d_pxd_yU  = nullptr;
    double *d_pyd_xL  = nullptr, *d_pyd_xR  = nullptr;
    double *d_pyd_yD  = nullptr, *d_pyd_yU  = nullptr;
    // Uniform reference mesh spacing (cached for donor-center lookup in kernels).
    double dx_u = 0.0, dy_u = 0.0;

    // ---- Subcell forces (nsub) ----
    double *d_FSX = nullptr, *d_FSY = nullptr;

    // ---- CFL reduction scratch ----
    double *d_dt_cell = nullptr;
    double *d_reduce_buf = nullptr;

    // ---- Parameters ----
    double gamma = 1.4;
    double cfl = 0.25;
    double CQ_lin = 0.5;
    double CQ_quad = 2.0;
    double comp_dt_frac = 0.25;
    // Shear-aware AV: if 1, reduce Q in shear-dominated cells (pseudo-tensor
    // AV). Recommended ON for KH, convection, turbulence; OFF for pure
    // shocks (Sod) where the legacy scalar Q is well calibrated.
    int shear_aware_av = 0;
    // BC mode (bit 0: x-periodic, bit 1: y-periodic). 0 = reflective walls.
    int bc_mode = 0;
    double g_y = 0.0;                // downward gravity magnitude (pulls −y)
    // Remap order: 1 = donor-cell (legacy, kept for regression), 2 = MUSCL-in-remap
    // with slope limiter (Kucharik-Shashkov 2012). Default is 2.
    int remap_order = 2;
    // Limiter for 2nd-order remap: 0=minmod, 1=van Leer (default), 2=MC.
    int remap_limiter = 1;
    // Use PPM (Colella-Woodward 1984) piecewise parabolic reconstruction
    // instead of MUSCL linear. Only applies when remap_order >= 2.
    int ppm_enabled = 0;
    // PPM limiter: 0 = classical Colella-Woodward monotonization
    // (clamps smooth extrema), 1 = Colella-Sekora 2008 extremum-preserving
    // (default, matches Athena ppm.cpp uniform-mesh path).
    int ppm_cs_limiter = 1;
    // PPM reconstruction variable space: 0 = conservative (ρ, ρe, ρvx, ρvy),
    // 1 = primitive (ρ, P, vx, vy) — default. Primitive-space PPM is robust
    // to tanh shear interfaces where px=ρvx crosses zero; conservative PPM
    // overshoots there and crashes.
    int ppm_primitive = 1;
    // PPM characteristic projection (only meaningful when ppm_primitive=1):
    // 0 = reconstruct primitives directly,
    // 1 = project to (slow-acoustic, entropy, shear, fast-acoustic) wave
    //     modes, reconstruct each independently, project back. Required
    //     for smooth tanh shear layers (Lecoanet KH). Follows Athena
    //     ppm.cpp + characteristic.cpp (Stone+08 Appendix A, adiabatic hydro).
    int ppm_char = 1;

    // ---- Newton cooling + bottom enthalpy-flux heating (optional) ----
    // Newton cooling bleeds off heat at the top:
    //   e ← e + (e_ref(y) − e)·α_cool·s_cool(y),   α_cool = 1 − exp(−dt/τ)
    // where s_cool(y) is a cosine ramp: 0 below (1 − cool_top_frac)·Ly, 1 at top.
    // Bottom enthalpy source injects L⊙-equivalent flux as a volumetric
    // heating with exponential profile concentrated at the bottom:
    //   q(y) = F_bot · g(y),   g(y) = w(y)/∫w dy,   w(y) = exp(−y/(heat_bot_frac·Ly))
    //   de/dt = q(y) / ρ(y)
    // In steady state ∫(top cooling) = ∫(bottom heating) = F_bot·Lx — classical
    // Stein-Nordlund box convection driving. Density is never touched so mass &
    // HSE are exactly preserved.
    double *d_e_ref_y          = nullptr;  // per-row reference e_int (cooling target)
    double *d_cool_weight_y    = nullptr;  // per-row cooling ramp weight (0 at bot, 1 at top)
    double *d_heat_dedt_base_y = nullptr;  // per-row F_bot·g(y) [erg/(s·cm³)]
    double tau_cool            = 0.0;      // Newton cooling timescale (s); 0 = disabled
    double cool_top_frac       = 1.0;      // cooling active in top frac of column (1 = full)
    double bottom_heat_flux    = 0.0;      // F_bot [erg/cm²/s]; 0 = disabled
    double heat_bot_frac       = 0.05;     // heating depth fraction (e-fold of exp profile / Ly)
    void alloc_cooling_ref(const std::vector<double>& e_ref_per_row);
    void configure_thermal(double F_bot, double heat_bot_frac_, double cool_top_frac_);
    void apply_cooling(double dt);

    // ---- Bookkeeping ----
    double dt_current = 0.0;
    int step_count = 0;
    double remap_mass_drift = 0.0;   // |Σ dm_after − Σ dm_before| / Σ dm_before per step

    // ---- Lifecycle ----
    void init(int nx, int ny, double Lx, double Ly, double gamma, double cfl);
    void destroy();

    // Initial conditions
    void init_uniform(double rho, double P, double vx, double vy);
    void init_sod();
    void init_hse_polytrope(double rho_base, double g_val, double perturb_amp = 0.0);

    // HSE polytrope + N Gaussian bubble overlays (multiplicative, so bubbles
    // compound if they overlap).
    //   ρ(x,y) = ρ_HSE(y) · Π_i [1 + αᵢ · exp(-rᵢ²/rbᵢ²)]
    //   P(x,y) = P_HSE(y) · Π_i [1 + βᵢ · exp(-rᵢ²/rbᵢ²)]
    struct Bubble { double xc, yc, rb, alpha, beta; };
    void init_hse_bubbles(double rho_base, double g_val,
                          const std::vector<Bubble>& bubbles);

    // Zero-gravity Kelvin-Helmholtz shear test:
    //   y ∈ (0.25, 0.75):  ρ = rho_heavy,  vx = +vshear
    //   else             :  ρ = rho_light,  vx = −vshear
    // Pressure uniform P0 (isobaric). Seeded with vy = amp·sin(k·2π·x)
    // × Gaussian envelope around each interface (σ = 0.05) to trigger KH.
    // g_y is set to 0 inside this IC — ideal for watching pure KH roll-up.
    void init_kh_shear(double rho_light, double rho_heavy, double P0,
                       double vshear, double amp, int k);

    // Plane-parallel stratified slab from a MESA envelope strip.  Loads a
    // slab file emitted by scripts/render/make_local_convection_slab.py containing:
    //   header line:  Ly Lx g_y gamma rho_top P_top T_top mu
    //   data lines:   (ny+1) × (y, rho, P, T)   face-centered
    // Sets g_y from the file and pairs naturally with --bc-x periodic
    // --bc-y reflect.  Seeds a small entropy perturbation at the bottom to
    // trigger overturning convection.
    //   perturb_amp — δs/s relative bump (0.01 is plenty)
    //   seed_k      — horizontal mode for the perturbation (default 4)
    void init_local_convection(const std::string& slab_file,
                               double perturb_amp = 0.01,
                               int seed_k = 4);

    // Lecoanet (2015) canonical KH — dual tanh shear layers, fully periodic.
    // Matches Athena pgen/kh.cpp iprob=4 when k=1. Default parameters from
    // Athena inputs/hydro/athinput.lecoanet_kh: vflow=1, amp=0.01,
    // drho_rho0=0 (unstratified), k=1 (single mode per interface). Larger
    // k (e.g. 7) trades canonical fidelity for richer Kraichnan-like cascade.
    // Requires --bc-x periodic --bc-y periodic.
    void init_kh_lecoanet(double vflow, double amp,
                          double drho_rho0 = 0.0, double P0 = 10.0, int k = 1);

    // One ALE step: Lagrangian → Rezone → Remap
    double step(double t, double t_end);

    struct Diagnostics {
        double total_mass;
        double total_KE;
        double total_internal_E;
        double total_PE;
        double total_E;
        double max_v;
        double max_mach;
    };
    Diagnostics compute_diagnostics();

    void download_xslice(std::vector<double>& x,
                         std::vector<double>& rho,
                         std::vector<double>& P,
                         std::vector<double>& vx,
                         std::vector<double>& e_int);

    // Cartesian-structured VTK: full 2D fields for visualization.
    // Writes density, pressure, velocity vector (cell-centered).
    // Mesh is Lx × Ly uniform in node space.
    void write_vtk_2d(const char* filename, double Lx, double Ly);

    // ---- VRAM-buffered frame dump (high-frame-rate I/O) ----
    // Each captured frame stores 5 cell-centered fields (ρ, P, e_int, vx, vy)
    // in device memory. When the pool fills, the whole batch is copied to
    // host and written to binary VTK files as output_NNNN.vtk. The GPU
    // stalls during D2H but otherwise never pays per-frame I/O cost.
    //
    //   alloc_frame_buffer(headroom_mb=1024)  — size pool to (free−1 GB)
    //   capture_frame(t)                       — device-side snapshot
    //   flush_frames_to_disk(run_dir, Lx, Ly)  — D2H + write all frames
    //
    // Layout per frame: 5 contiguous ncell arrays (ρ, P, e_int, vx_cc, vy_cc).
    // t-stamp array is host-side.
    double *d_frame_pool = nullptr;
    int frame_capacity = 0;    // max frames the pool can hold
    int frame_count = 0;       // frames currently captured (buffered)
    int total_frames = 0;      // total frames written to disk so far
    std::vector<double> frame_times;   // host: t at each captured frame
    std::vector<int>    frame_steps;   // host: step index at each captured frame
    std::string frame_out_dir;         // cached for flush

    void alloc_frame_buffer(int headroom_mb);
    void capture_frame(double t, int step);
    void flush_frames_to_disk(const std::string& run_dir, double Lx, double Ly);
    void free_frame_buffer();
};
