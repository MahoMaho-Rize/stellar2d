#pragma once

#ifdef USE_AMGX

#include "grid.h"
#include "state.h"
#include "eos.h"
#include "hydro/reconstruct.h"
#include "gmg_gpu.cuh"
#include <amgx_c.h>
#include <string>

// Fully-implicit GPU solver: Backward Euler + JFNK + FGMRES + AmgX preconditioner.
//
// Preconditioner: approximate Jacobian A ≈ I/dt - dR/dU assembled from
// 1st-order Rusanov flux Jacobian, stored as 4×4 block-CSR on device.
// AmgX applies one V-cycle of block-AMG as left preconditioner for FGMRES.
//
// AmgX handles persist for the entire simulation. The sparsity pattern
// is set once at init; only coefficients are replaced each Newton step
// via AMGX_matrix_replace_coefficients.

struct GpuSolver {
    void init(const Grid& grid, const EOS& eos, double G, double cfl,
              Limiter lim);
    void upload_state(const Grid& grid, const State& state);
    void download_state(const Grid& grid, State& state);
    double step(double t, double t_end);
    void destroy();

    int nr, nt, ng;
    double gamma, G_const, cfl_num;
    Limiter limiter;
    int total_phys, total_ghost;
    double dt_max;
    double dt_good = 0.0;   // best dt that recently converged (for fast recovery)
    int step_count = 0;
    double dt_adapt = 0.0;
    bool initialized = false;

    // Device grid arrays
    double *d_r_face, *d_r_center, *d_dr;
    double *d_theta_face, *d_theta_center, *d_dtheta;
    double *d_cell_volume, *d_area_r, *d_area_theta;

    // Conservative state (with ghost cells)
    double *d_rho, *d_mr, *d_mtheta, *d_E;

    // U^n for backward Euler (4*n packed)
    double *d_Un;

    // Gravity potential
    double *d_phi;

    // Background HSE
    double *d_rho0, *d_P0, *d_phi0;

    // HLLC flux scratch
    double *d_flux_r, *d_flux_t;

    // Residual & JFNK vectors (4*n packed)
    double *d_residual, *d_Fk;

    static constexpr int GMRES_RESTART = 60;
    double *d_gmres_V[GMRES_RESTART + 1];
    double *d_gmres_Z[GMRES_RESTART + 1]; // preconditioned vectors for FGMRES
    double *d_gmres_w;
    double *d_gmres_Uk;

    // Work buffers — MUST NOT alias each other across nested calls.
    // d_work_dot_*:  exclusively for gpu_dot inside FGMRES/Newton
    // d_work_grav_*: exclusively for gpu_reduce_sum inside solve_gravity
    // d_work_ls_*:   exclusively for line search compute_F
    double *d_work_dot_a, *d_work_dot_b;
    double *d_work_grav_a, *d_work_grav_b;
    double *d_rhs_poisson;
    double *d_residual_ls; // line search F output (separate from d_residual)

    // GMG for gravity
    GmgGpu gmg;

    // ===== AmgX block-AMG preconditioner =====
    // Persistent handles — created once, reused every step.
    AMGX_config_handle amgx_cfg;
    AMGX_resources_handle amgx_rsrc;
    AMGX_matrix_handle amgx_A;
    AMGX_vector_handle amgx_x, amgx_b;
    AMGX_solver_handle amgx_solver;
    bool amgx_setup_done = false;

    // Block-CSR Jacobian storage (device pointers, 4×4 blocks)
    // row_ptr: n+1 ints, col_idx: nnz ints, values: nnz*16 doubles
    int *d_jac_row_ptr, *d_jac_col_idx;
    double *d_jac_values;
    int jac_nnz; // number of 4×4 blocks

    // ===== Internal methods =====
    void compute_residual(double* d_res);
    void compute_F(double* d_F, double dt);
    void jfnk_matvec(const double* d_v, double* d_Jv, double dt);
    int fgmres_solve(double* d_x, const double* d_b, double dt,
                     double tol, int max_iter);
    void pack_state(double* d_packed);
    void unpack_delta(const double* d_delta, double sign);
    void apply_floor();
    void solve_gravity();
    double compute_cfl_dt();

    // Jacobian assembly + AmgX preconditioner
    void assemble_jacobian(double dt);
    void apply_preconditioner(const double* d_v, double* d_Mv);
};

#endif // USE_AMGX
