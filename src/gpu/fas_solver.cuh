#pragma once

#include "../grid.h"
#include "../state.h"
#include "../eos.h"
#include "gmg_gpu.cuh"

// FAS (Full Approximation Scheme) nonlinear multigrid for the low-Mach
// 4-DOF Euler system: (ρ, ρvr, ρvθ, ρe).
//
// Backward Euler: F(U) = (U - Uⁿ)/dt - R(U) = 0
//
// FAS V-cycle:
//   1. Pre-smooth: damped block Jacobi on F(u) = 0
//   2. Restrict: u_H = Î·u_h, f_H = R(Î·u_h) + Î·(f_h - R(u_h))
//      where f_h = Uⁿ/dt (source from time derivative)
//   3. Recurse on coarse level
//   4. Prolongate: u_h += P·(u_H - Î·u_h)
//   5. Post-smooth

struct FasLevel {
    int nr, nt, ng;
    int total;  // (nr + 2*ng) * (nt + 2*ng)
    int phys;   // nr * nt

    // Grid geometry (device)
    double *d_r_face, *d_r_center, *d_dr;
    double *d_theta_face, *d_theta_center, *d_dtheta;
    double *d_cell_volume, *d_area_r, *d_area_theta;
    double *d_sin_theta_face, *d_sin_theta_center;

    // State arrays with ghost cells: ρ, ρvr, ρvθ, ρe
    double *d_rho, *d_mr, *d_mt, *d_rhoE;

    // RHS for FAS (includes τ-correction): size 4*phys
    double *d_fas_rhs;

    // Residual scratch: size 4*phys
    double *d_res;

    // Saved Uⁿ and Uⁿ⁻¹ for time derivative: size 4*phys each
    double *d_Un;
    double *d_Un_prev;

    // Saved restricted state before coarse solve (for prolongation correction): size 4*phys
    double *d_save;

    // HSE reference for well-balanced residual
    double *d_rho0, *d_P0;

    // Pre-computed WB residual of HSE state: R_WB(U₀) on this level (4*phys)
    // Should be ~0 but nonzero due to discrete HSE inconsistency.
    // Subtracted from fas_rhs after restrict_defect to ensure F(U₀)=0 exactly.
    double *d_hse_defect;

    // 1D gravity arrays
    double *d_gr, *d_gr0;
    double *d_shell_mass;

    // Block Jacobian diagonal (4×4 per cell, 16*phys)
    double *d_blk_inv;

    // SIMPLE smoother scratch (per level)
    double *d_Ap;           // momentum diagonal (phys)
    double *d_vr_s, *d_vt_s; // predicted velocity (phys each)
    double *d_div_s;        // divergence scratch (phys)
    double *d_dp;           // pressure correction (phys)
    double *d_poisson_rhs;  // Poisson RHS (phys)
    double *d_inv_Ap;       // 1/Ap (phys)
    GmgGpu pressure_gmg;   // per-level pressure Poisson solver
};

struct FasSolver {
    void init(const Grid& grid, const EOS& eos, double G, double cfl);
    void destroy();

    // Solve F(U) = (U-Uⁿ)/dt - R(U) = 0 using FAS V-cycles.
    // State is in d_rho/d_mr/d_mt/d_rhoE on finest level.
    // Returns number of V-cycles used.
    int solve(double dt, double g0_over_dt, int max_cycles = 20, double tol = 1e-4);

    // Upload / download state from host
    void upload_state(const Grid& grid, const State& state);
    void download_state(const Grid& grid, State& state);

    // Snapshot HSE reference (call once on equilibrium state)
    void snapshot_hse();

    // Full time step (manages dt, calls solve)
    double step(double t, double t_end);

    // CFL-based dt estimate (|v|+cs signal speed)
    double compute_cfl_dt();

    int n_levels = 0;
    FasLevel levels[8];

    double gamma, G_const, cfl_num;
    double dt_current = 0.0;
    double dt_prev = 0.0;
    double atm_rho_thresh = 0.0;
    int step_count = 0;
    bool hse_set = false;
    bool use_simple_smoother = true; // true=SIMPLE (Poisson-based), false=block Jacobi

    double sponge_r_start = 0.0;
    double sponge_r_top = 0.0;
    double sponge_kappa = 10.0;

    static constexpr int NU1 = 2;     // pre-smooth iterations
    static constexpr int NU2 = 3;     // post-smooth iterations
    static constexpr double OMEGA = 0.7;  // damping factor

private:
    void build_level(int l, int nr, int nt, int ng,
                     const double* h_rf, const double* h_tf);
    void build_coarse_geometry(int fine, int coarse);

    // FAS operations
    void fas_vcycle(int l, double dt, double g0_over_dt);
    void smooth(int l, double dt, double g0_over_dt, int n_iters);
    void compute_residual(int l, int use_hllc = 0);
    void compute_F(int l, double g0_over_dt);
    void launch_ghost(int l);
    void compute_gravity_1d(int l);
    void apply_floor(int l);

    void restrict_state(int fine, int coarse);
    void restrict_defect(int fine, int coarse, double g0_over_dt);
    void prolongate_correct(int coarse, int fine);

    void assemble_smoother(int l, double g0_over_dt);

    // Diagnostics
    double residual_norm(int l);
};
