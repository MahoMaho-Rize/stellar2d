#pragma once

#include "../grid.h"
#include "../state.h"
#include "../eos.h"
#include "fas_common.cuh"
#include "gmg_gpu.cuh"

// SIMPLE pressure-correction solver for the compressible Euler equations.
// Single-level (no multigrid), block-Jacobi + SIMPLE iterations.
//
// Backward Euler: F(U) = R(U) - γ₀·U/dt + rhs = 0
// Each iteration:
//   1. Block-Jacobi: U -= ω · J⁻¹_diag · F(U)
//   2. SIMPLE: pressure Poisson correction for momentum + block-Jacobi for ρ/E

struct SimpleLevel {
    int nr, nt, ng;
    int total, phys;

    // Grid geometry (device)
    double *d_r_face, *d_r_center, *d_dr;
    double *d_theta_face, *d_theta_center, *d_dtheta;
    double *d_cell_volume, *d_area_r, *d_area_theta;
    double *d_sin_theta_face, *d_sin_theta_center;

    // State (with ghost)
    double *d_rho, *d_mr, *d_mt, *d_rhoE;

    // FAS-compatible arrays
    double *d_fas_rhs;   // RHS: Uⁿ/dt + BDF2 terms (4*phys)
    double *d_res;       // residual scratch (4*phys)
    double *d_Un;        // saved Uⁿ (4*phys)
    double *d_Un_prev;   // saved Uⁿ⁻¹ (4*phys)

    // HSE reference
    double *d_rho0, *d_P0;
    double *d_hse_defect;

    // 1D gravity
    double *d_gr, *d_gr0;
    double *d_shell_mass;

    // Block Jacobi (4×4 per cell)
    double *d_blk_inv;

    // SIMPLE scratch
    double *d_Ap, *d_vr_s, *d_vt_s;
    double *d_div_s, *d_dp, *d_poisson_rhs, *d_inv_Ap;

    GmgGpu pressure_gmg;
};

struct SimpleSolver {
    void init(const Grid& grid, const EOS& eos, double G, double cfl);
    void destroy();

    void upload_state(const Grid& grid, const State& state);
    void download_state(const Grid& grid, State& state);
    void snapshot_hse();

    double step(double t, double t_end);
    double step_explicit(double t, double t_end);
    double compute_cfl_dt();

    SimpleLevel lev;
    double gamma, G_const, cfl_num;
    double dt_current = 0.0, dt_prev = 0.0;
    double atm_rho_thresh = 0.0;
    int step_count = 0;
    bool hse_set = false;

    double sponge_r_start = 0.0, sponge_r_top = 0.0;
    double sponge_kappa = 100.0;

    static constexpr int N_INNER = 8;       // block-Jacobi + SIMPLE iterations per solve
    static constexpr int MAX_OUTER = 12;     // max outer Newton-like iterations
    static constexpr double OMEGA = 0.8;     // block-Jacobi damping
    static constexpr double TOL = 0.1;       // convergence tolerance on ||F||

private:
    void compute_residual();
    void compute_F(double g0_over_dt);
    void launch_ghost();
    void compute_gravity_1d();
    void apply_floor();
    void assemble_precond(double g0_over_dt);

    // One block-Jacobi + SIMPLE iteration
    void simple_iteration(double dt, double g0_over_dt);

    double residual_norm();
};
