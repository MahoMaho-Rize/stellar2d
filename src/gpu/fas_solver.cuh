#pragma once

#include "../grid.h"
#include "../state.h"
#include "../eos.h"
#include "gmg_gpu.cuh"

// FAS (Full Approximation Scheme) nonlinear multigrid for the low-Mach
// 4-DOF Euler system: (ρ, ρvr, ρvθ, ρe).
//
// Smoother: SIMPLE (momentum predict → pressure Poisson → correct)
// Each FAS level has its own pressure GMG for the SIMPLE smoother.

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

    // Saved Uⁿ for time derivative: size 4*phys
    double *d_Un;

    // HSE reference for well-balanced residual
    double *d_rho0, *d_P0;

    // 1D gravity arrays
    double *d_gr, *d_gr0;
    double *d_shell_mass;

    // Block Jacobian diagonal (4×4 per cell, 16*phys)
    double *d_blk_inv;

    // SIMPLE smoother scratch (per level)
    double *d_Ap;           // momentum diagonal coefficient (phys)
    double *d_vr_s, *d_vt_s; // predicted velocity (phys each)
    double *d_div;          // divergence (phys)
    double *d_dp;           // pressure correction (phys)
    double *d_rhs_poisson;  // Poisson RHS (phys)
    double *d_inv_Ap;       // 1/Ap for variable-coefficient Poisson (phys)
    GmgGpu pressure_gmg;   // per-level pressure Poisson solver
};

struct FasSolver {
    void init(const Grid& grid, const EOS& eos, double G, double cfl);
    void destroy();

    // Solve F(U) = (U-Uⁿ)/dt - R(U) = 0 using FAS V-cycles.
    // State is in d_rho/d_mr/d_mt/d_rhoE on finest level.
    // Returns number of V-cycles used.
    int solve(double dt, int max_cycles = 20, double tol = 1e-4);

    // Upload / download state from host
    void upload_state(const Grid& grid, const State& state);
    void download_state(const Grid& grid, State& state);

    // Snapshot HSE reference (call once on equilibrium state)
    void snapshot_hse();

    // Full time step (manages dt, calls solve)
    double step(double t, double t_end);

    int n_levels = 0;
    FasLevel levels[8];

    double gamma, G_const, cfl_num;
    double dt_current = 0.0;
    double atm_rho_thresh = 0.0;
    int step_count = 0;
    bool hse_set = false;

    static constexpr int NU1 = 2;     // pre-smooth SIMPLE iterations
    static constexpr int NU2 = 2;     // post-smooth SIMPLE iterations

private:
    void build_level(int l, int nr, int nt, int ng,
                     const double* h_rf, const double* h_tf);
    void build_coarse_geometry(int fine, int coarse);

    // FAS operations
    void fas_vcycle(int l, double dt);
    void smooth(int l, double dt, int n_iters);
    void compute_residual(int l);
    void compute_F(int l, double dt);  // F = U/dt - Uⁿ/dt - R(U)
    void launch_ghost(int l);
    void compute_gravity_1d(int l);
    void apply_floor(int l);

    // Inter-grid transfers (4-DOF, conservative)
    void restrict_state(int fine, int coarse);
    void restrict_defect(int fine, int coarse, double dt);
    void prolongate_correct(int coarse, int fine);

    // Block Jacobi assembly
    void assemble_smoother(int l, double dt);

    // Diagnostics
    double residual_norm(int l);
};
