#pragma once

#include "../grid.h"
#include "../state.h"
#include "../eos.h"
#include "../hydro/reconstruct.h"
#include <string>

#ifdef USE_AMGX
#include <amgx_c.h>
#endif

struct GpuGridConst {
    int nr, ntheta, ng;
    double R_outer, gamma, G;
    double cfl;
};

struct GpuSolver {
    void init(const Grid& grid, const EOS& eos, double G, double cfl,
              Limiter lim, const std::string& amgx_config);
    void upload_state(const Grid& grid, const State& state);
    void download_state(const Grid& grid, State& state);
    double step(double t, double t_end);
    void destroy();

    GpuGridConst gc;
    Limiter limiter;
    int total_phys, total_ghost;

    // device grid arrays
    double *d_r_face, *d_r_center, *d_dr;
    double *d_theta_face, *d_theta_center, *d_dtheta;
    double *d_cell_volume, *d_area_r, *d_area_theta;

    // device state (with ghost cells)
    double *d_rho, *d_mr, *d_mtheta, *d_E;
    // saved state ρⁿ for pressure equation RHS
    double *d_rho_old;
    // phi (gravity potential) on physical grid
    double *d_phi;
    // pressure solution on physical grid
    double *d_pressure;

    // flux arrays (4 components each: rho, mr, mtheta, E)
    double *d_flux_r, *d_flux_t;

    // work buffers (reused for divergence, RHS, reduction, etc.)
    double *d_work_a, *d_work_b;

    // CFL reduction
    double *d_dt_min;

    // Poisson CSR (shared sparsity pattern for gravity and pressure)
    int *d_row_ptr, *d_col_idx;
    double *d_grav_values;    // gravity matrix values (constant)
    double *d_pres_values;    // pressure matrix values (rebuilt each step)
    double *d_rhs;            // RHS buffer for Poisson solves
    int poisson_n, poisson_nnz;

    double dt_max;

#ifdef USE_AMGX
    AMGX_config_handle amgx_cfg;
    AMGX_resources_handle amgx_rsrc;
    // gravity Poisson solver
    AMGX_matrix_handle amgx_A_grav;
    AMGX_vector_handle amgx_b_grav, amgx_x_grav;
    AMGX_solver_handle amgx_solver_grav;
    // pressure Poisson solver
    AMGX_matrix_handle amgx_A_pres;
    AMGX_vector_handle amgx_b_pres, amgx_x_pres;
    AMGX_solver_handle amgx_solver_pres;
#endif

    bool initialized = false;
};
