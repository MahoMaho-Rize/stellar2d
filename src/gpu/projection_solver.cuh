#pragma once

#include "../grid.h"
#include "../state.h"
#include "../eos.h"
#include "fas_common.cuh"
#include "gmg_gpu.cuh"

// Semi-implicit pressure projection solver for compressible Euler + self-gravity.
//
// Algorithm (Chorin / Kwatra 2009 style):
//   1. Explicit predictor: U* = Uⁿ + dt·R(Uⁿ)  [full HLLC including pressure]
//   2. Pressure Poisson:   ∇·((dt/ρ)∇δp) = ∇·v*  [correct acoustic overshoot]
//   3. Correct momentum:   (ρv)ⁿ⁺¹ = (ρv)* - dt·∇δp
//   4. dt limited by ADVECTIVE CFL: dt = cfl · min(dr / |v|)
//
// One Poisson solve per step. No Newton, no block-Jacobi, no iterations.

struct ProjSolver {
    void init(const Grid& grid, const EOS& eos, double G, double cfl);
    void destroy();

    void upload_state(const Grid& grid, const State& state);
    void download_state(const Grid& grid, State& state);
    void snapshot_hse();

    double step(double t, double t_end);

    int nr, nt, ng;
    int total, phys;
    double gamma, G_const, cfl_num;
    double atm_rho_thresh = 0.0;
    double sponge_r_start = 0.0, sponge_r_top = 0.0, sponge_kappa = 10.0;
    int step_count = 0;
    bool hse_set = false;

    // Grid geometry (device)
    double *d_r_face, *d_r_center, *d_dr;
    double *d_theta_face, *d_theta_center, *d_dtheta;
    double *d_cell_volume, *d_area_r, *d_area_theta;
    double *d_sin_theta_face, *d_sin_theta_center;

    // State (with ghost)
    double *d_rho, *d_mr, *d_mt, *d_rhoE;

    // Scratch: spatial residual R(U) = 4*phys
    double *d_res;

    // HSE reference
    double *d_rho0, *d_P0;
    double *d_hse_defect;

    // 1D gravity
    double *d_gr, *d_gr0, *d_shell_mass;

    // Pressure projection scratch
    double *d_dp;           // pressure correction δp (phys)
    double *d_div;          // divergence ∇·v* (phys)
    double *d_alpha;        // dt/ρ coefficient (phys)
    double *d_poisson_rhs;  // Poisson RHS (phys)

    GmgGpu pressure_gmg;

private:
    void launch_ghost();
    void compute_gravity_1d();
    void apply_floor();
    void compute_residual();  // R(U) without HSE defect subtraction
    double compute_advective_cfl_dt();
};
