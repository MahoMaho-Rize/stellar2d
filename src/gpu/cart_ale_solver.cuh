#pragma once

#include <vector>

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

struct CartAleSolver {
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
    double g_y = 0.0;                // downward gravity magnitude (pulls −y)
    // Remap order: 1 = donor-cell (legacy, kept for regression), 2 = MUSCL-in-remap
    // with slope limiter (Kucharik-Shashkov 2012). Default is 2.
    int remap_order = 2;
    // Limiter for 2nd-order remap: 0=minmod, 1=van Leer (default), 2=MC.
    int remap_limiter = 1;

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
};
