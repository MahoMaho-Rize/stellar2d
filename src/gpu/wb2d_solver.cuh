#pragma once

#include "../grid.h"
#include "../state.h"
#include "../eos.h"

// Well-Balanced 2D Eulerian solver with MESA-inspired stabilization.
//
// Dualistic spherical (r, θ): both directions treated symmetrically.
// Explicit RK2 time integration.
//
// Core mechanisms (see docs/wb2d_design.md):
//   - Flux-level well-balanced HSE subtraction (perturbation MUSCL on ρ, P)
//   - Tscharnuter-Winkler artificial viscosity in r AND θ
//   - CFL limit = min(acoustic_r, acoustic_t, compression_r, compression_t)
//   - Uniform pressure floor P ≥ P_floor_frac · P₀ (no conditional branches)
//   - Reused central v_r damping (conservative, θ-symmetric)
//   - 1D angle-averaged gravity (upgrade to 2D GMG later if needed)

struct Wb2DSolver {
    // --- State arrays (with ghost, size total = (nr+2*ng)*(nt+2*ng)) ---
    double *d_rho, *d_mr, *d_mt, *d_rhoE;

    // --- HSE reference (physical only, size phys = nr*nt) ---
    double *d_rho0, *d_P0;
    double *d_hse_defect;  // R_WB(U₀), size 4*phys, subtracted from R(U)

    // --- TW viscosity (cell-centered, size phys) ---
    double *d_Pvsc_r, *d_Pvsc_t;

    // --- 1D angle-averaged gravity (size nr) ---
    double *d_gr, *d_gr0, *d_shell_mass;

    // --- RK2 scratch ---
    double *d_Un;        // saved Uⁿ, size 4*phys
    double *d_res;       // residual R(U), size 4*phys
    double *d_dt_cell;   // per-cell dt, size phys

    // --- Reduction scratch (device, size phys) ---
    double *d_reduce_buf, *d_reduce_out;

    // --- Grid geometry on device ---
    double *d_r_face, *d_r_center, *d_dr;
    double *d_theta_face, *d_theta_center, *d_dtheta;
    double *d_cell_volume, *d_area_r, *d_area_theta;

    // --- Size info ---
    int nr = 0, nt = 0, ng = 0;
    int total = 0;   // (nr+2*ng)*(nt+2*ng)
    int phys = 0;    // nr*nt

    // --- Physical / numerical parameters ---
    double gamma = 5.0/3.0;
    double G_const = 1.0;
    double cfl_num = 0.4;

    double CQ = 2.0;              // TW viscosity coefficient
    double ZSH = 0.1;              // TW shock threshold
    double comp_dt_frac = 0.1;     // compression dt safety factor
    double P_floor_frac = 0.1;     // P ≥ P_floor_frac * P₀
    double rho_floor_frac = 0.01;  // ρ ≥ rho_floor_frac * ρ₀
    double central_damp_r = 0.0;   // v_r damping radius at origin (0=off)
    double central_damp_alpha = 5.0;
    int limiter_type = 0;           // 0=minmod, 1=vanleer, 2=MC
    int use_lm_hllc = 0;

    // Porting FAS stabilizers (safe under θ-symmetric IC;
    // disable n_angular_avg/n_pole_avg for non-axisymmetric cases like bubble).
    int n_angular_avg = 0;          // angular-average innermost n shells per step
    int n_pole_avg = 0;             // wedge-average n cells near each pole
    double sponge_r_start = 0.0;    // outer sponge layer inner radius (0 = off)
    double sponge_r_top = 0.0;
    double sponge_kappa = 0.0;      // max damping rate (0 = off)

    // --- Bookkeeping ---
    double dt_current = 0.0;
    int step_count = 0;
    bool hse_set = false;

    // --- Lifecycle ---
    void init(const Grid& grid, const EOS& eos, double G, double cfl);
    void destroy();

    // --- I/O ---
    void upload_state(const Grid& grid, const State& state);
    void download_state(const Grid& grid, State& state);

    // Capture current state as HSE reference (sets d_rho0, d_P0, d_gr0, d_hse_defect).
    // Also pre-computes R_WB(U₀) to cancel the spatial discretization HSE error.
    void snapshot_hse();

    // One explicit RK2 step, returns dt actually taken.
    double step(double t, double t_end);

    // CFL + compression dt estimate.
    double compute_cfl_dt();
};
