// ============================================================
// athena_mhd_solver.cu — host-side driver for AthenaMHDSolver
// Kernels live in athena_mhd_kernels.cu.
//
// Derivation dossier:  docs/mhd_derivations/manuscript.pdf (§A1-A11).
// ============================================================

#include "athena_mhd_solver.cuh"
#include "gpu/common/gpu_common.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// ---- kernel forward declarations ----------------------------
__global__ void k_athmhd_cons_to_prim(
    const double*, const double*, const double*, const double*, const double*,
    const double*, const double*, const double*,
    double*, double*, double*, double*, double*, double*, double*, double*,
    double*, double*, int, int, double);
__global__ void k_athmhd_ghost_x_periodic_cc(
    double*, double*, double*, double*, double*, double*,
    int, int, int, int);
__global__ void k_athmhd_ghost_y_periodic_cc(
    double*, double*, double*, double*, double*, double*,
    int, int, int, int);
__global__ void k_athmhd_ghost_y_reflect_cc(
    double*, double*, double*, double*, double*, double*,
    int, int, int, int);
__global__ void k_athmhd_ghost_x_outflow_cc(
    double*, double*, double*, double*, double*, double*,
    int, int, int, int);
__global__ void k_athmhd_ghost_y_outflow_cc(
    double*, double*, double*, double*, double*, double*,
    int, int, int, int);
__global__ void k_athmhd_ghost_x_periodic_face(
    double*, double*, int, int, int, int);
__global__ void k_athmhd_ghost_y_periodic_face(
    double*, double*, int, int, int, int);
__global__ void k_athmhd_ghost_y_reflect_face(
    double*, double*, int, int, int, int);
__global__ void k_athmhd_ghost_x_outflow_face(
    double*, double*, int, int, int, int);
__global__ void k_athmhd_ghost_y_outflow_face(
    double*, double*, int, int, int, int);
__global__ void k_athmhd_flux_x(
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*,
    double*, double*, double*, double*, double*, double*, double*, double*,
    int, int, int, int, int, int, double);
__global__ void k_athmhd_flux_y(
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*,
    double*, double*, double*, double*, double*, double*, double*, double*,
    int, int, int, int, int, int, double);
__global__ void k_athmhd_corner_emf(
    const double*, const double*, double*,
    int, int, int, int);
__global__ void k_athmhd_flux_divergence(
    const double*, const double*, const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, double*, double*,
    const double*, const double*, const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*, const double*, const double*,
    int, int, int, int, int, double, double, double);
__global__ void k_athmhd_update_face_B(
    const double*, const double*, double*, double*, const double*,
    int, int, int, int, int, double, double, double);
__global__ void k_athmhd_copy_face_B(
    const double*, const double*, double*, double*, int, int);
__global__ void k_athmhd_source_gravity(
    const double*, const double*, double*, double*, const double*,
    int, int, int, int, int, double);
__global__ void k_athmhd_source_wb_subtract(
    double*, double*, double*, double*, double*, double*,
    const double*, const double*, const double*,
    const double*, const double*, const double*,
    int, int, int, int, int, double);
__global__ void k_athmhd_cfl(
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    double*, int, int, int, int, int, double, double, double);
__global__ void k_athmhd_divB_cc(
    const double*, const double*, double*, int, int, int, int, double, double);
__global__ void k_athmhd_compute_T(
    const double*, const double*, double*, int, int);
__global__ void k_athmhd_ghost_T_y_reflect(double*, int, int, int, int);
__global__ void k_athmhd_ghost_T_y_periodic(double*, int, int, int, int);
__global__ void k_athmhd_ghost_T_y_outflow(double*, int, int, int, int);
__global__ void k_athmhd_ghost_T_x_periodic(double*, int, int, int, int);
__global__ void k_athmhd_ghost_T_x_outflow(double*, int, int, int, int);
__global__ void k_athmhd_conduction_flux_x(
    const double*, const double*, const double*, const double*,
    double*, int, int, int, int, int, double, double, double);
__global__ void k_athmhd_conduction_flux_y(
    const double*, const double*, const double*, const double*,
    double*, int, int, int, int, int, double, double, double);
__global__ void k_athmhd_apply_conduction(
    double*, const double*, const double*,
    int, int, int, int, int, double, double, double);
__global__ void k_athmhd_conduction_cfl(
    const double*, const double*, double*,
    int, int, int, int, double, double, double, double);
__global__ void k_athmhd_cool_townsend(
    const double*, const double*, double*,
    int, int, int, int,
    double, double, double, double, double, double);
__global__ void k_athmhd_driver_apply(
    const double*, double*, double*, double*, double*,
    const double*, const double*, const double*,
    int, double, int, int, int);

// ============================================================
// init / destroy
// ============================================================
void AthenaMHDSolver::init(int nx_, int ny_, double Lx_, double Ly_,
                           double gam_, double cfl_) {
    nx = nx_;
    ny = ny_;
    Lx = Lx_;
    Ly = Ly_;
    gamma = gam_;
    cfl = cfl_;
    x_lo = 0.0; x_hi = Lx;
    y_lo = 0.0; y_hi = Ly;
    dx = Lx / (double)nx;
    dy = Ly / (double)ny;

    int ncell = total_cells();
    int nfx   = total_fx();
    int nfy   = total_fy();
    int ncorn = total_corner();
    size_t nb_cell   = (size_t)ncell * sizeof(double);
    size_t nb_fx     = (size_t)nfx   * sizeof(double);
    size_t nb_fy     = (size_t)nfy   * sizeof(double);
    size_t nb_corner = (size_t)ncorn * sizeof(double);

    auto alloc_zero = [&](double** p, size_t nbytes) {
        CUDA_CHECK(cudaMalloc(p, nbytes));
        CUDA_CHECK(cudaMemset(*p, 0, nbytes));
    };

    // Cell-centred conservatives (5 fields)
    alloc_zero(&d_rho, nb_cell); alloc_zero(&d_mx, nb_cell);
    alloc_zero(&d_my,  nb_cell); alloc_zero(&d_mz, nb_cell);
    alloc_zero(&d_E,   nb_cell);
    alloc_zero(&d_rho1, nb_cell); alloc_zero(&d_mx1, nb_cell);
    alloc_zero(&d_my1,  nb_cell); alloc_zero(&d_mz1, nb_cell);
    alloc_zero(&d_E1,   nb_cell);
    // Cell-centred B (diagnostic; recomputed in cons_to_prim)
    alloc_zero(&d_Bx_cc, nb_cell);
    alloc_zero(&d_By_cc, nb_cell);
    alloc_zero(&d_Bz_cc, nb_cell);
    alloc_zero(&d_Bz1,   nb_cell);
    // Face-centred B
    alloc_zero(&d_Bxf,   nb_fx);
    alloc_zero(&d_Byf,   nb_fy);
    alloc_zero(&d_Bxf1,  nb_fx);
    alloc_zero(&d_Byf1,  nb_fy);
    // Corner EMF + partial face-EMFs
    alloc_zero(&d_Ez_corner, nb_corner);
    alloc_zero(&d_Ezx_face,  nb_fx);
    alloc_zero(&d_Ezy_face,  nb_fy);
    // Primitive scratch (8 fields)
    alloc_zero(&d_w_rho, nb_cell); alloc_zero(&d_w_u,  nb_cell);
    alloc_zero(&d_w_v,   nb_cell); alloc_zero(&d_w_w,  nb_cell);
    alloc_zero(&d_w_Bx,  nb_cell); alloc_zero(&d_w_By, nb_cell);
    alloc_zero(&d_w_Bz,  nb_cell); alloc_zero(&d_w_P,  nb_cell);
    // Face fluxes (7 per direction)
    alloc_zero(&d_Fx_rho, nb_fx); alloc_zero(&d_Fx_mx, nb_fx);
    alloc_zero(&d_Fx_my,  nb_fx); alloc_zero(&d_Fx_mz, nb_fx);
    alloc_zero(&d_Fx_By,  nb_fx); alloc_zero(&d_Fx_Bz, nb_fx);
    alloc_zero(&d_Fx_E,   nb_fx);
    alloc_zero(&d_Gy_rho, nb_fy); alloc_zero(&d_Gy_mx, nb_fy);
    alloc_zero(&d_Gy_my,  nb_fy); alloc_zero(&d_Gy_mz, nb_fy);
    alloc_zero(&d_Gy_Bx,  nb_fy); alloc_zero(&d_Gy_Bz, nb_fy);
    alloc_zero(&d_Gy_E,   nb_fy);
    // Source tables (optional)
    alloc_zero(&d_g_row, (size_t)ny * sizeof(double));
    h_g_row.assign(ny, 0.0);
    h_phi_row.assign(ny, 0.0);
    // §B4 well-balanced MHSE defect buffers (lazily-populated by snapshot_hse)
    // Two copies: stage-1 (donor-cell) vs stage-2 (PLM) produce different
    // discrete residuals; subtracting the matching one keeps U_hse fixed
    // to machine precision at *both* substages.
    alloc_zero(&d_rhs_hse_s1_rho, nb_cell);
    alloc_zero(&d_rhs_hse_s1_mx,  nb_cell);
    alloc_zero(&d_rhs_hse_s1_my,  nb_cell);
    alloc_zero(&d_rhs_hse_s1_mz,  nb_cell);
    alloc_zero(&d_rhs_hse_s1_E,   nb_cell);
    alloc_zero(&d_rhs_hse_s1_Bz,  nb_cell);
    alloc_zero(&d_rhs_hse_s2_rho, nb_cell);
    alloc_zero(&d_rhs_hse_s2_mx,  nb_cell);
    alloc_zero(&d_rhs_hse_s2_my,  nb_cell);
    alloc_zero(&d_rhs_hse_s2_mz,  nb_cell);
    alloc_zero(&d_rhs_hse_s2_E,   nb_cell);
    alloc_zero(&d_rhs_hse_s2_Bz,  nb_cell);
    wb_active = false;
    // §C6 conduction
    alloc_zero(&d_T_cc,        nb_cell);
    alloc_zero(&d_Fx_cond,     nb_fx);
    alloc_zero(&d_Gy_cond,     nb_fy);
    alloc_zero(&d_cond_dt_buf, (size_t)nx * (size_t)ny * sizeof(double));
    kappa0 = 0.0;
    // CFL
    alloc_zero(&d_cfl_buf, (size_t)nx * (size_t)ny * sizeof(double));

    step_count = 0;
    dt_current = 0.0;
}

void AthenaMHDSolver::destroy() {
    auto F = [](double*& p) { if (p) { cudaFree(p); p = nullptr; } };
    F(d_rho); F(d_mx); F(d_my); F(d_mz); F(d_E);
    F(d_rho1); F(d_mx1); F(d_my1); F(d_mz1); F(d_E1);
    F(d_Bx_cc); F(d_By_cc); F(d_Bz_cc); F(d_Bz1);
    F(d_Bxf); F(d_Byf); F(d_Bxf1); F(d_Byf1);
    F(d_Ez_corner); F(d_Ezx_face); F(d_Ezy_face);
    F(d_w_rho); F(d_w_u); F(d_w_v); F(d_w_w);
    F(d_w_Bx); F(d_w_By); F(d_w_Bz); F(d_w_P);
    F(d_Fx_rho); F(d_Fx_mx); F(d_Fx_my); F(d_Fx_mz);
    F(d_Fx_By); F(d_Fx_Bz); F(d_Fx_E);
    F(d_Gy_rho); F(d_Gy_mx); F(d_Gy_my); F(d_Gy_mz);
    F(d_Gy_Bx); F(d_Gy_Bz); F(d_Gy_E);
    F(d_g_row);
    F(d_rhs_hse_s1_rho); F(d_rhs_hse_s1_mx); F(d_rhs_hse_s1_my);
    F(d_rhs_hse_s1_mz);  F(d_rhs_hse_s1_E);  F(d_rhs_hse_s1_Bz);
    F(d_rhs_hse_s2_rho); F(d_rhs_hse_s2_mx); F(d_rhs_hse_s2_my);
    F(d_rhs_hse_s2_mz);  F(d_rhs_hse_s2_E);  F(d_rhs_hse_s2_Bz);
    F(d_T_cc); F(d_Fx_cond); F(d_Gy_cond); F(d_cond_dt_buf);
    F(d_driver_f); F(d_driver_amp); F(d_driver_phi);
    F(d_cfl_buf);
}

// ============================================================
// Ghost fill — driven by x_bc / y_bc flags
// ============================================================
void AthenaMHDSolver::fill_ghost() {
    int sx = stride_x(), sy = stride_y();
    dim3 bcc(64, 1), gcc((sy + 63) / 64, (ng + 1 - 1) / 1);
    // cell-centred: dispatch per direction based on x_bc / y_bc
    dim3 bx_grid((sy + 63) / 64, ng);
    if (x_bc == 2) {
        k_athmhd_ghost_x_outflow_cc<<<bx_grid, dim3(64, 1)>>>(
            d_rho, d_mx, d_my, d_mz, d_E, d_Bz_cc, nx, ng, sx, sy);
    } else {
        // Periodic (default); reflect-in-x not implemented for MHD —
        // users needing a wall should use y_bc=1 and rotate the tube.
        k_athmhd_ghost_x_periodic_cc<<<bx_grid, dim3(64, 1)>>>(
            d_rho, d_mx, d_my, d_mz, d_E, d_Bz_cc, nx, ng, sx, sy);
    }

    dim3 by_grid((sx + 63) / 64, ng);
    if (y_bc == 0) {
        k_athmhd_ghost_y_periodic_cc<<<by_grid, dim3(64, 1)>>>(
            d_rho, d_mx, d_my, d_mz, d_E, d_Bz_cc, ny, ng, sx, sy);
    } else if (y_bc == 2) {
        k_athmhd_ghost_y_outflow_cc<<<by_grid, dim3(64, 1)>>>(
            d_rho, d_mx, d_my, d_mz, d_E, d_Bz_cc, ny, ng, sx, sy);
    } else {
        k_athmhd_ghost_y_reflect_cc<<<by_grid, dim3(64, 1)>>>(
            d_rho, d_mx, d_my, d_mz, d_E, d_Bz_cc, ny, ng, sx, sy);
    }

    // Face B — use max(nx+1+2ng, ny+1+2ng) × ng grid
    dim3 bf_grid_x(((sy + 1) + 63) / 64, ng);
    if (x_bc == 2) {
        k_athmhd_ghost_x_outflow_face<<<bf_grid_x, dim3(64, 1)>>>(
            d_Bxf, d_Byf, nx, ny, ng, sy);
    } else {
        k_athmhd_ghost_x_periodic_face<<<bf_grid_x, dim3(64, 1)>>>(
            d_Bxf, d_Byf, nx, ny, ng, sy);
    }
    dim3 bf_grid_y(((sx + 1) + 63) / 64, ng);
    if (y_bc == 0) {
        k_athmhd_ghost_y_periodic_face<<<bf_grid_y, dim3(64, 1)>>>(
            d_Bxf, d_Byf, nx, ny, ng, sy);
    } else if (y_bc == 2) {
        k_athmhd_ghost_y_outflow_face<<<bf_grid_y, dim3(64, 1)>>>(
            d_Bxf, d_Byf, nx, ny, ng, sy);
    } else {
        k_athmhd_ghost_y_reflect_face<<<bf_grid_y, dim3(64, 1)>>>(
            d_Bxf, d_Byf, nx, ny, ng, sy);
    }
    (void)bcc; (void)gcc;   // unused
}

// ============================================================
// cons_to_prim
// ============================================================
void AthenaMHDSolver::cons_to_prim() {
    int sx = stride_x(), sy = stride_y();
    dim3 b(16, 16), g((sx + 15) / 16, (sy + 15) / 16);
    k_athmhd_cons_to_prim<<<g, b>>>(
        d_rho, d_mx, d_my, d_mz, d_E,
        d_Bxf, d_Byf, d_Bz_cc,
        d_w_rho, d_w_u, d_w_v, d_w_w,
        d_w_Bx, d_w_By, d_w_Bz, d_w_P,
        d_Bx_cc, d_By_cc,
        sx, sy, gamma - 1.0);
}

// ============================================================
// calc_hydro_flux_and_emf — populates:
//   Fx_* / Gy_* flux arrays
//   Ezx_face / Ezy_face  (partial EMF contributions)
//   Ez_corner            (GS05 average)
// using the primitive arrays currently in d_w_*.
// ============================================================
void AthenaMHDSolver::calc_hydro_flux_and_emf(int order) {
    int sx = stride_x(), sy = stride_y();
    dim3 b(32, 8);
    // x-flux
    dim3 gx((sx + b.x - 1) / b.x, (sy + b.y - 1) / b.y);
    k_athmhd_flux_x<<<gx, b>>>(
        d_w_rho, d_w_u, d_w_v, d_w_w,
        d_w_Bx, d_w_By, d_w_Bz, d_w_P,
        d_Bxf,
        d_Fx_rho, d_Fx_mx, d_Fx_my, d_Fx_mz,
        d_Fx_By, d_Fx_Bz, d_Fx_E,
        d_Ezx_face,
        nx, ng, sx, sy, order, limiter, gamma);
    // y-flux
    dim3 gy((sx + b.x - 1) / b.x, ((sy + 1) + b.y - 1) / b.y);
    k_athmhd_flux_y<<<gy, b>>>(
        d_w_rho, d_w_u, d_w_v, d_w_w,
        d_w_Bx, d_w_By, d_w_Bz, d_w_P,
        d_Byf,
        d_Gy_rho, d_Gy_mx, d_Gy_my, d_Gy_mz,
        d_Gy_Bx, d_Gy_Bz, d_Gy_E,
        d_Ezy_face,
        ny, ng, sx, sy, order, limiter, gamma);
    // corner EMF
    dim3 gc((sx + 1 + 15) / 16, (sy + 1 + 15) / 16);
    k_athmhd_corner_emf<<<gc, dim3(16, 16)>>>(
        d_Ezx_face, d_Ezy_face, d_Ez_corner, nx, ny, ng, sy);
}

// ============================================================
// apply_flux_divergence_and_ct — given the flux/EMF arrays currently
// populated, apply:
//   stage=1:  U_dst = U^n − (dt/2) ∇·F;       B_f_dst = B_f^n − (dt/2) curl E_z
//   stage=2:  U_dst = U^n − dt ∇·F;            B_f_dst = B_f^n − dt curl E_z
// The src cell-centred U is always d_rho etc. (U^n); the dst goes to d_rho1.
// ============================================================
void AthenaMHDSolver::apply_flux_divergence_and_ct(int stage, double dt) {
    int sx = stride_x(), sy = stride_y();
    double dt_stage = (stage == 1) ? 0.5 * dt : dt;

    dim3 b(16, 16), gg((nx + 15) / 16, (ny + 15) / 16);
    k_athmhd_flux_divergence<<<gg, b>>>(
        d_rho, d_mx, d_my, d_mz, d_Bz_cc, d_E,
        d_rho1, d_mx1, d_my1, d_mz1, d_Bz1, d_E1,
        d_Fx_rho, d_Fx_mx, d_Fx_my, d_Fx_mz, d_Fx_Bz, d_Fx_E,
        d_Gy_rho, d_Gy_mx, d_Gy_my, d_Gy_mz, d_Gy_Bz, d_Gy_E,
        nx, ny, ng, sx, sy, dx, dy, dt_stage);

    // Face-B CT update (src = Bxf/Byf at t^n, dst = Bxf1/Byf1)
    dim3 gbf((sx + 1 + 15) / 16, (sy + 1 + 15) / 16);
    k_athmhd_update_face_B<<<gbf, dim3(16, 16)>>>(
        d_Bxf, d_Byf, d_Bxf1, d_Byf1, d_Ez_corner,
        nx, ny, ng, sx, sy, dx, dy, dt_stage);

    // Optional gravity source term (applied to cons at t^n; uses
    // current w_*).
    bool any_grav = false;
    // Host check: if g_row sum is nonzero, run the kernel.
    for (double g : h_g_row) {
        if (g != 0.0) { any_grav = true; break; }
    }
    if (any_grav) {
        k_athmhd_source_gravity<<<gg, b>>>(
            d_w_rho, d_w_v, d_my1, d_E1, d_g_row,
            nx, ny, ng, sx, sy, dt_stage);
    }

    // §B4 well-balanced MHSE: subtract the frozen R(U_hse) residual so
    // an atmosphere at MHSE stays at MHSE to machine precision.  Only
    // active after snapshot_hse() has been called.  Per-stage defects
    // because predictor (order=1) and corrector (order=xorder) give
    // different discrete residuals.
    if (wb_active) {
        double* rhs_rho = (stage == 1) ? d_rhs_hse_s1_rho : d_rhs_hse_s2_rho;
        double* rhs_mx  = (stage == 1) ? d_rhs_hse_s1_mx  : d_rhs_hse_s2_mx;
        double* rhs_my  = (stage == 1) ? d_rhs_hse_s1_my  : d_rhs_hse_s2_my;
        double* rhs_mz  = (stage == 1) ? d_rhs_hse_s1_mz  : d_rhs_hse_s2_mz;
        double* rhs_E   = (stage == 1) ? d_rhs_hse_s1_E   : d_rhs_hse_s2_E;
        double* rhs_Bz  = (stage == 1) ? d_rhs_hse_s1_Bz  : d_rhs_hse_s2_Bz;
        k_athmhd_source_wb_subtract<<<gg, b>>>(
            d_rho1, d_mx1, d_my1, d_mz1, d_E1, d_Bz1,
            rhs_rho, rhs_mx, rhs_my, rhs_mz, rhs_E, rhs_Bz,
            nx, ny, ng, sx, sy, dt_stage);
    }
}

// ============================================================
// stage_advance
// Given the primitives in d_w_* and face-B in d_Bxf/d_Byf:
//   compute flux + EMF for given order
//   apply to U^n → U1 (d_rho1...) and Bxf^n → Bxf1/Byf1.
// ============================================================
void AthenaMHDSolver::stage_advance(int stage, double dt) {
    int order = (stage == 1) ? 1 : xorder;
    calc_hydro_flux_and_emf(order);
    apply_flux_divergence_and_ct(stage, dt);
}

// ============================================================
// One VL2 predictor-corrector MHD step
// ============================================================
double AthenaMHDSolver::step(double t, double t_end) {
    // At entrance: d_rho etc. = u^n; d_Bxf/d_Byf = B_f^n.
    fill_ghost();
    cons_to_prim();

    double dt = compute_dt();
    if (t + dt > t_end) dt = t_end - t;
    dt_current = dt;

    // ── stage 1: predictor (order=1) ───────────────
    stage_advance(/*stage=*/1, dt);
    // Now d_rho1 = u*, d_Bxf1/d_Byf1 = B_f*.  We need to recompute
    // prims at that star state before the corrector.
    //   Swap u^n and u* so d_rho holds u*, d_rho1 holds u^n.
    std::swap(d_rho, d_rho1);
    std::swap(d_mx,  d_mx1);
    std::swap(d_my,  d_my1);
    std::swap(d_mz,  d_mz1);
    std::swap(d_E,   d_E1);
    std::swap(d_Bz_cc, d_Bz1);
    std::swap(d_Bxf, d_Bxf1);
    std::swap(d_Byf, d_Byf1);
    fill_ghost();
    cons_to_prim();
    // Swap back so d_rho = u^n for stage-2 flux-divergence source.
    // (Face-B used at stage-2 CT update is ALSO B_f^n, not B_f*, because
    //  Stone-Gardiner 2009 VL2 uses EMF(u*) to advance B_f^n → B_f^{n+1}.)
    std::swap(d_rho, d_rho1);
    std::swap(d_mx,  d_mx1);
    std::swap(d_my,  d_my1);
    std::swap(d_mz,  d_mz1);
    std::swap(d_E,   d_E1);
    std::swap(d_Bz_cc, d_Bz1);
    std::swap(d_Bxf, d_Bxf1);
    std::swap(d_Byf, d_Byf1);
    // Now d_rho = u^n and d_w_* = primitives AT u* (what we want).
    // d_Bxf = B_f^n (needed as src for CT).

    // ── stage 2: corrector (order=xorder) ──────────
    stage_advance(/*stage=*/2, dt);

    // Final:  d_rho1 = u^{n+1}, d_Bxf1/d_Byf1 = B_f^{n+1}.
    std::swap(d_rho, d_rho1);
    std::swap(d_mx,  d_mx1);
    std::swap(d_my,  d_my1);
    std::swap(d_mz,  d_mz1);
    std::swap(d_E,   d_E1);
    std::swap(d_Bz_cc, d_Bz1);
    std::swap(d_Bxf, d_Bxf1);
    std::swap(d_Byf, d_Byf1);

    step_count++;
    return dt;
}

// ============================================================
// compute_dt
// ============================================================
double AthenaMHDSolver::compute_dt() {
    int sx = stride_x(), sy = stride_y();
    dim3 b(16, 16), g((nx + 15) / 16, (ny + 15) / 16);
    k_athmhd_cfl<<<g, b>>>(
        d_w_rho, d_w_u, d_w_v, d_w_w,
        d_w_Bx, d_w_By, d_w_Bz, d_w_P,
        d_cfl_buf, nx, ny, ng, sx, sy,
        dx, dy, gamma);
    std::vector<double> h_buf((size_t)nx * (size_t)ny);
    CUDA_CHECK(cudaMemcpy(h_buf.data(), d_cfl_buf,
                          h_buf.size() * sizeof(double),
                          cudaMemcpyDeviceToHost));
    double dt_min = 1e300;
    for (double v : h_buf) if (v > 0.0 && v < dt_min) dt_min = v;
    double use_cfl = std::min(cfl, cfl_limit);
    return use_cfl * dt_min;
}

// ============================================================
// Diagnostics — cell-centred conservatives + Bxf/Byf → div·B
// ============================================================
AthenaMHDSolver::Diagnostics AthenaMHDSolver::compute_diagnostics() {
    // Refresh cell-centred B_cc (re-derived from face-B in cons_to_prim)
    // so that diagnostics work at any time, including immediately after
    // an IC method that only wrote face-B values.  Cheap (one prim pass).
    fill_ghost();
    cons_to_prim();

    int sx = stride_x(), sy = stride_y();
    int ncell = sx * sy;
    size_t nb = (size_t)ncell * sizeof(double);
    std::vector<double> h_rho(ncell), h_mx(ncell), h_my(ncell),
                        h_mz(ncell), h_E(ncell);
    std::vector<double> h_Bxcc(ncell), h_Bycc(ncell), h_Bzcc(ncell);
    CUDA_CHECK(cudaMemcpy(h_rho.data(),  d_rho, nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mx.data(),   d_mx,  nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_my.data(),   d_my,  nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mz.data(),   d_mz,  nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_E.data(),    d_E,   nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Bxcc.data(), d_Bx_cc, nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Bycc.data(), d_By_cc, nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Bzcc.data(), d_Bz_cc, nb, cudaMemcpyDeviceToHost));

    // div·B via cell-centred kernel
    std::vector<double> h_divB(ncell, 0.0);
    {
        dim3 b(16, 16), g((nx + 15) / 16, (ny + 15) / 16);
        double* d_tmp = nullptr;
        CUDA_CHECK(cudaMalloc(&d_tmp, nb));
        CUDA_CHECK(cudaMemset(d_tmp, 0, nb));
        k_athmhd_divB_cc<<<g, b>>>(d_Bxf, d_Byf, d_tmp,
                                    nx, ny, ng, sy, dx, dy);
        CUDA_CHECK(cudaMemcpy(h_divB.data(), d_tmp, nb, cudaMemcpyDeviceToHost));
        cudaFree(d_tmp);
    }

    double M = 0, KE = 0, IE = 0, ME = 0, vmax = 0, Mmax = 0;
    double maxdivB = 0;
    double dV = dx * dy;
    double gm1 = gamma - 1.0;
    for (int jc = 0; jc < ny; ++jc)
        for (int ic = 0; ic < nx; ++ic) {
            int c = (ic + ng) * sy + (jc + ng);
            double r = std::max(h_rho[c], 1e-30);
            double u = h_mx[c] / r, v = h_my[c] / r, w = h_mz[c] / r;
            double ke = 0.5 * r * (u*u + v*v + w*w);
            double me = 0.5 * (h_Bxcc[c] * h_Bxcc[c]
                             + h_Bycc[c] * h_Bycc[c]
                             + h_Bzcc[c] * h_Bzcc[c]);
            double ie = h_E[c] - ke - me;
            double P  = std::max(gm1 * ie, 1e-30);
            double cs2 = gamma * P / r;
            double cA2 = (h_Bxcc[c]*h_Bxcc[c] + h_Bycc[c]*h_Bycc[c]
                         + h_Bzcc[c]*h_Bzcc[c]) / r;
            double cf = std::sqrt(cs2 + cA2);   // upper bound on fast speed
            double speed = std::sqrt(u*u + v*v + w*w);
            M += r * dV;
            KE += ke * dV;
            IE += ie * dV;
            ME += me * dV;
            vmax = std::max(vmax, speed);
            Mmax = std::max(Mmax, speed / cf);
            maxdivB = std::max(maxdivB, std::fabs(h_divB[c]));
        }
    return {M, KE, IE, ME, KE + IE + ME, maxdivB, vmax, Mmax};
}

// ============================================================
// VTK writer (cell-centred; B_* from cc reconstructions)
// ============================================================
void AthenaMHDSolver::write_vtk_2d(const char* filename,
                                   double Lx_in, double Ly_in) {
    (void)Lx_in; (void)Ly_in;
    int sx = stride_x(), sy = stride_y();
    int ncell = sx * sy;
    size_t nb = (size_t)ncell * sizeof(double);
    std::vector<double> h_rho(ncell), h_mx(ncell), h_my(ncell),
                        h_mz(ncell), h_E(ncell);
    std::vector<double> h_Bxcc(ncell), h_Bycc(ncell), h_Bzcc(ncell);
    CUDA_CHECK(cudaMemcpy(h_rho.data(),  d_rho, nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mx.data(),   d_mx,  nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_my.data(),   d_my,  nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mz.data(),   d_mz,  nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_E.data(),    d_E,   nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Bxcc.data(), d_Bx_cc, nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Bycc.data(), d_By_cc, nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Bzcc.data(), d_Bz_cc, nb, cudaMemcpyDeviceToHost));

    std::FILE* fp = std::fopen(filename, "w");
    if (!fp) return;
    int nnx = nx + 1, nny = ny + 1;
    std::fprintf(fp, "# vtk DataFile Version 3.0\n");
    std::fprintf(fp, "athena_mhd 2D Cartesian output\n");
    std::fprintf(fp, "ASCII\nDATASET STRUCTURED_GRID\n");
    std::fprintf(fp, "DIMENSIONS %d %d 1\n", nnx, nny);
    std::fprintf(fp, "POINTS %d double\n", nnx * nny);
    for (int jn = 0; jn < nny; ++jn) {
        double y = y_lo + Ly * (double)jn / (double)(nny - 1);
        for (int in = 0; in < nnx; ++in) {
            double x = x_lo + Lx * (double)in / (double)(nnx - 1);
            std::fprintf(fp, "%.10e %.10e %.10e\n", x, y, 0.0);
        }
    }
    std::fprintf(fp, "CELL_DATA %d\n", nx * ny);
    double gm1 = gamma - 1.0;
    auto cc = [&](const char* name, auto fn) {
        std::fprintf(fp, "SCALARS %s double 1\nLOOKUP_TABLE default\n", name);
        for (int jc = 0; jc < ny; ++jc)
            for (int ic = 0; ic < nx; ++ic) {
                int c = (ic + ng) * sy + (jc + ng);
                std::fprintf(fp, "%.10e\n", fn(c));
            }
    };
    cc("density",  [&](int c) { return h_rho[c]; });
    cc("pressure", [&](int c) {
        double r = std::max(h_rho[c], 1e-30);
        double u = h_mx[c]/r, v = h_my[c]/r, w = h_mz[c]/r;
        double ke = 0.5*r*(u*u + v*v + w*w);
        double me = 0.5*(h_Bxcc[c]*h_Bxcc[c] + h_Bycc[c]*h_Bycc[c]
                       + h_Bzcc[c]*h_Bzcc[c]);
        return std::max(gm1 * (h_E[c] - ke - me), 1e-30);
    });
    cc("Bx", [&](int c) { return h_Bxcc[c]; });
    cc("By", [&](int c) { return h_Bycc[c]; });
    cc("Bz", [&](int c) { return h_Bzcc[c]; });
    cc("Bmag", [&](int c) {
        return std::sqrt(h_Bxcc[c]*h_Bxcc[c] + h_Bycc[c]*h_Bycc[c]
                       + h_Bzcc[c]*h_Bzcc[c]);
    });
    std::fprintf(fp, "VECTORS velocity double\n");
    for (int jc = 0; jc < ny; ++jc)
        for (int ic = 0; ic < nx; ++ic) {
            int c = (ic + ng) * sy + (jc + ng);
            double r = std::max(h_rho[c], 1e-30);
            std::fprintf(fp, "%.10e %.10e %.10e\n",
                         h_mx[c]/r, h_my[c]/r, h_mz[c]/r);
        }
    std::fclose(fp);
}

// ============================================================
// Initial conditions (§A11 + field-loop + Brio-Wu + Orszag-Tang)
// ============================================================

// --- helper: write one cell-centred conserved state to host arrays
static inline void set_cell(std::vector<double>& rho, std::vector<double>& mx,
                            std::vector<double>& my, std::vector<double>& mz,
                            std::vector<double>& E, std::vector<double>& Bz_cc,
                            int c, double r, double u, double v, double w,
                            double Bx, double By, double Bz, double P,
                            double gamma) {
    double ke = 0.5 * r * (u*u + v*v + w*w);
    double me = 0.5 * (Bx*Bx + By*By + Bz*Bz);
    rho[c] = r;
    mx [c] = r * u;
    my [c] = r * v;
    mz [c] = r * w;
    E  [c] = P / (gamma - 1.0) + ke + me;
    Bz_cc[c] = Bz;
}

void AthenaMHDSolver::init_brio_wu() {
    // §A9 demonstration.  Brio-Wu 1988 Table 1 state.  γ=2, L_x=1.
    // B_x = 0.75 (common), v = 0 everywhere, B_z = 0.
    gamma = 2.0;
    x_bc = 2;        // outflow in x
    y_bc = 0;        // periodic in y (effectively 1D)
    x_lo = 0.0; x_hi = Lx;
    y_lo = 0.0; y_hi = Ly;
    dx = Lx / (double)nx;
    dy = Ly / (double)ny;

    int sx = stride_x(), sy = stride_y();
    int ncell = sx * sy;
    std::vector<double> h_rho(ncell, 0.0), h_mx(ncell, 0.0),
                        h_my(ncell, 0.0), h_mz(ncell, 0.0),
                        h_E(ncell, 0.0), h_Bz_cc(ncell, 0.0);
    int nfx = total_fx(), nfy = total_fy();
    std::vector<double> h_Bxf(nfx, 0.75);   // common B_x = 0.75
    std::vector<double> h_Byf(nfy, 0.0);

    for (int ic = 0; ic < nx; ++ic) {
        double xc = x_lo + (ic + 0.5) * dx;
        bool left = (xc < 0.5 * (x_lo + x_hi));
        double r  = left ? 1.0   : 0.125;
        double P  = left ? 1.0   : 0.1;
        double By = left ? +1.0  : -1.0;
        for (int jc = 0; jc < ny; ++jc) {
            int c = (ic + ng) * sy + (jc + ng);
            set_cell(h_rho, h_mx, h_my, h_mz, h_E, h_Bz_cc,
                     c, r, 0, 0, 0, 0.75, By, 0, P, gamma);
        }
    }
    // Byf: initialise the face-centred array from the cell-centred By.
    // For Brio-Wu the discontinuity is at x=Lx/2; we initialise face
    // values from the AVERAGE of adjacent cells (which is exact for a
    // piecewise-constant state).
    for (int ic = 0; ic < nx + 2 * ng; ++ic) {
        double xc = x_lo + ((ic - ng) + 0.5) * dx;
        bool left = (xc < 0.5 * (x_lo + x_hi));
        double By = left ? +1.0 : -1.0;
        for (int jf = 0; jf < ny + 1 + 2 * ng; ++jf) {
            h_Byf[ic * (sy + 1) + jf] = By;
        }
    }
    // Bxf: uniform = 0.75 (already set).

    size_t nb_cell = (size_t)ncell * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_rho, h_rho.data(), nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mx,  h_mx.data(),  nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_my,  h_my.data(),  nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mz,  h_mz.data(),  nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E,   h_E.data(),   nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bz_cc, h_Bz_cc.data(), nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bxf, h_Bxf.data(), (size_t)nfx * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Byf, h_Byf.data(), (size_t)nfy * sizeof(double),
                          cudaMemcpyHostToDevice));

    std::fprintf(stderr,
        "  AthenaMHD Brio-Wu IC: γ=%g, Lx=%g, nx=%d\n"
        "    B_x=0.75, B_y=±1, ρ=1/0.125, P=1/0.1, v=0\n",
        gamma, Lx, nx);
}

void AthenaMHDSolver::init_linear_wave(LinearWaveMode mode, int k, double A) {
    // §A11 Stone+08 Table 1 background.
    gamma = 5.0 / 3.0;
    x_bc = 0; y_bc = 0;          // fully periodic
    x_lo = 0.0; x_hi = Lx;
    y_lo = 0.0; y_hi = Ly;
    dx = Lx / (double)nx;
    dy = Ly / (double)ny;

    // Background (§A11).  c_f=2, c_Ax=1, c_s=1/2, c_s0=1.
    double rho0 = 1.0;
    double p0   = 1.0 / gamma;
    double Bx0  = 1.0;
    double By0  = std::sqrt(2.0);
    double Bz0  = 0.5;
    double v0x  = (mode == ENTROPY) ? 1.0 : 0.0;

    // Eigenvector components for each mode (Stone+08 Appendix B,
    // normalised for primitive wave equation A_W δW = λ δW).  We use
    // the following components ordered as (δρ, δv_x, δv_y, δv_z,
    // δB_y, δB_z, δP):
    //
    // Derived wave speeds (§A11):
    //   c_Ax² = 1,  c_s0² = 1,  c_A² = 1 + 2 + 0.25 = 3.25,
    //   sum = c_s0² + c_A² = 4.25;  disc = √(4.25² − 4·1·1) = √(14.0625) = 3.75
    //   c_f² = (4.25 + 3.75)/2 = 4   ⇒  c_f = 2
    //   c_s² = (4.25 − 3.75)/2 = 0.25 ⇒ c_s = 0.5
    // α_f² = (c_s0² − c_s²)/(c_f² − c_s²) = (1 − 0.25)/(4 − 0.25) = 0.75/3.75 = 0.2
    // α_s² = (c_f² − c_s0²)/(c_f² − c_s²) = (4 − 1)/3.75 = 0.8
    // β_y = B_y / √(B_y² + B_z²) = √2/√(2.25) ≈ 0.9428
    // β_z = B_z / √(B_y² + B_z²) = 0.5/√(2.25) ≈ 0.3333
    double cf   = 2.0;
    double cs   = 0.5;
    double cAx  = 1.0;
    double cs0  = 1.0;
    double B_perp_mag = std::sqrt(By0*By0 + Bz0*Bz0);
    double beta_y = By0 / B_perp_mag;
    double beta_z = Bz0 / B_perp_mag;
    double alpha_f = std::sqrt(0.2);
    double alpha_s = std::sqrt(0.8);

    // λ and eigenvector r (ρ, v_x, v_y, v_z, B_y, B_z, p).
    double lam = 0.0;
    double r_rho, r_vx, r_vy, r_vz, r_By, r_Bz, r_P;
    if (mode == FAST_M) {
        lam = v0x + cf;
        r_rho = rho0 * alpha_f;
        r_vx  = +alpha_f * cf;
        r_vy  = -alpha_s * cs * beta_y;
        r_vz  = -alpha_s * cs * beta_z;
        r_By  = alpha_s * cs0 * std::sqrt(rho0) * beta_y;
        r_Bz  = alpha_s * cs0 * std::sqrt(rho0) * beta_z;
        r_P   = alpha_f * gamma * p0;
    } else if (mode == ALFVEN) {
        lam = v0x + cAx;
        r_rho = 0.0;
        r_vx  = 0.0;
        r_vy  = -beta_z;
        r_vz  = +beta_y;
        r_By  = -beta_z * std::sqrt(rho0);
        r_Bz  = +beta_y * std::sqrt(rho0);
        r_P   = 0.0;
    } else if (mode == SLOW) {
        lam = v0x + cs;
        r_rho = rho0 * alpha_s;
        r_vx  = +alpha_s * cs;
        r_vy  = +alpha_f * cf * beta_y;
        r_vz  = +alpha_f * cf * beta_z;
        r_By  = -alpha_f * cs0 * std::sqrt(rho0) * beta_y;
        r_Bz  = -alpha_f * cs0 * std::sqrt(rho0) * beta_z;
        r_P   = alpha_s * gamma * p0;
    } else { // ENTROPY
        lam = v0x;
        r_rho = 1.0;
        r_vx = 0; r_vy = 0; r_vz = 0;
        r_By = 0; r_Bz = 0; r_P = 0;
    }
    (void)lam;

    // Normalise eigenvector to unit L² — to give clean convergence
    // diagnostics on the perturbation norm.
    double norm = std::sqrt(r_rho*r_rho + r_vx*r_vx + r_vy*r_vy + r_vz*r_vz
                          + r_By*r_By + r_Bz*r_Bz + r_P*r_P);
    if (norm > 0) {
        r_rho /= norm; r_vx /= norm; r_vy /= norm; r_vz /= norm;
        r_By  /= norm; r_Bz  /= norm; r_P  /= norm;
    }

    int sx = stride_x(), sy = stride_y();
    int ncell = sx * sy;
    int nfx = total_fx(), nfy = total_fy();
    std::vector<double> h_rho(ncell, 0.0), h_mx(ncell, 0.0),
                        h_my(ncell, 0.0), h_mz(ncell, 0.0),
                        h_E(ncell, 0.0), h_Bz_cc(ncell, 0.0);
    std::vector<double> h_Bxf(nfx, Bx0);
    std::vector<double> h_Byf(nfy, By0);

    const double two_pi = 2.0 * M_PI;
    for (int ic = 0; ic < nx; ++ic) {
        double xc = x_lo + (ic + 0.5) * dx;
        double phase = two_pi * k * (xc - x_lo) / Lx;
        double dW = A * std::cos(phase);
        double r  = rho0 + dW * r_rho;
        double ux = v0x + dW * r_vx;
        double uy =       dW * r_vy;
        double uz =       dW * r_vz;
        double By = By0 + dW * r_By;
        double Bz = Bz0 + dW * r_Bz;
        double P  = p0  + dW * r_P;
        for (int jc = 0; jc < ny; ++jc) {
            int c = (ic + ng) * sy + (jc + ng);
            set_cell(h_rho, h_mx, h_my, h_mz, h_E, h_Bz_cc,
                     c, r, ux, uy, uz, Bx0, By, Bz, P, gamma);
        }
    }
    // Byf at j-faces: same x-varying value as cell-centred By (1D wave)
    for (int ic = 0; ic < nx + 2 * ng; ++ic) {
        double xc = x_lo + ((ic - ng) + 0.5) * dx;
        double phase = two_pi * k * (xc - x_lo) / Lx;
        double dW = A * std::cos(phase);
        double By = By0 + dW * r_By;
        for (int jf = 0; jf < ny + 1 + 2 * ng; ++jf) {
            h_Byf[ic * (sy + 1) + jf] = By;
        }
    }
    // Bxf is uniform = Bx0 (k·δB_x = 0 automatically in 1D wave, §A11).

    size_t nb_cell = (size_t)ncell * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_rho, h_rho.data(), nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mx,  h_mx.data(),  nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_my,  h_my.data(),  nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mz,  h_mz.data(),  nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E,   h_E.data(),   nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bz_cc, h_Bz_cc.data(), nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bxf, h_Bxf.data(),
                          (size_t)nfx * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Byf, h_Byf.data(),
                          (size_t)nfy * sizeof(double), cudaMemcpyHostToDevice));
    std::fprintf(stderr,
        "  AthenaMHD linear-wave IC: mode=%d, k=%d, A=%.3e\n"
        "    cf=2, cA=1, cs=0.5, γ=5/3\n", (int)mode, k, A);
}

// ============================================================
// Oblique linear wave (§F1, Stone+08 §6.2)
// Rotated eigenvector on 2D periodic box with k = 2π(kx_int/Lx, ky_int/Ly).
// Derivation: docs/mhd_derivations/sections/f1_oblique_linwave.md
// ============================================================
void AthenaMHDSolver::init_linear_wave_oblique(LinearWaveMode mode,
                                               int kx_int, int ky_int,
                                               double A) {
    // §A11 Stone+08 Table 1 background (shared with init_linear_wave)
    gamma = 5.0 / 3.0;
    x_bc = 0; y_bc = 0;
    x_lo = 0.0; x_hi = Lx;
    y_lo = 0.0; y_hi = Ly;
    dx = Lx / (double)nx;
    dy = Ly / (double)ny;

    double rho0 = 1.0;
    double p0   = 1.0 / gamma;
    double Bx0_ref  = 1.0;
    double By0_ref  = std::sqrt(2.0);
    double Bz0  = 0.5;
    double v0x  = (mode == ENTROPY) ? 1.0 : 0.0;

    // Wave vector components (physical)
    const double two_pi = 2.0 * M_PI;
    const double kx = two_pi * (double)kx_int / Lx;
    const double ky = two_pi * (double)ky_int / Ly;
    const double kmag = std::sqrt(kx*kx + ky*ky);
    const double cosT = kx / kmag;
    const double sinT = ky / kmag;

    // Compute §A3 eigenvector in the unrotated (x̂-aligned) frame,
    // replacing c_Ax → c_{A,k̂} = (B·k̂)/√ρ (F1-cAk).
    //   Rotated-frame Bx' = Bx cosT + By sinT (along k̂)
    //   Rotated-frame By' = -Bx sinT + By cosT (transverse)
    const double Bx_k = Bx0_ref * cosT + By0_ref * sinT;
    const double By_k = -Bx0_ref * sinT + By0_ref * cosT;
    // Reference background in k̂-frame
    double Bz0_k = Bz0;  // (invariant)
    double Bx0_frame = Bx_k;
    double By0_frame = By_k;

    // §A3 wave speeds in k̂-frame
    double cAksq = Bx0_frame * Bx0_frame / rho0;   // (B·k̂)²/ρ
    double cAsq  = (Bx0_frame*Bx0_frame + By0_frame*By0_frame + Bz0_k*Bz0_k) / rho0;
    double cs0sq = gamma * p0 / rho0;
    double disc  = std::sqrt((cs0sq + cAsq) * (cs0sq + cAsq)
                             - 4.0 * cs0sq * cAksq);
    double cfsq  = 0.5 * ((cs0sq + cAsq) + disc);
    double cssq  = 0.5 * ((cs0sq + cAsq) - disc);
    double cf    = std::sqrt(cfsq);
    double cs    = std::sqrt(cssq);
    double cAx   = std::sqrt(cAksq);
    double cs0   = std::sqrt(cs0sq);
    double B_perp_mag = std::sqrt(By0_frame*By0_frame + Bz0_k*Bz0_k);
    double beta_y = (B_perp_mag > 1e-30) ? By0_frame / B_perp_mag : 1.0;
    double beta_z = (B_perp_mag > 1e-30) ? Bz0_k     / B_perp_mag : 0.0;
    double alpha_f_sq = (cs0sq - cssq) / (cfsq - cssq);
    double alpha_s_sq = (cfsq - cs0sq) / (cfsq - cssq);
    double alpha_f = std::sqrt(std::max(alpha_f_sq, 0.0));
    double alpha_s = std::sqrt(std::max(alpha_s_sq, 0.0));

    // Eigenvector in the k̂-frame (ρ, v_para, v_perp_y, v_perp_z, B_perp_y, B_perp_z, p)
    double lam = 0.0;
    double r_rho, r_vpara, r_vperp_y, r_vperp_z, r_By_k, r_Bz_k, r_P;
    if (mode == FAST_M) {
        lam = v0x + cf;
        r_rho = rho0 * alpha_f;
        r_vpara  = +alpha_f * cf;
        r_vperp_y  = -alpha_s * cs * beta_y;
        r_vperp_z  = -alpha_s * cs * beta_z;
        r_By_k  = alpha_s * cs0 * std::sqrt(rho0) * beta_y;
        r_Bz_k  = alpha_s * cs0 * std::sqrt(rho0) * beta_z;
        r_P   = alpha_f * gamma * p0;
    } else if (mode == ALFVEN) {
        lam = v0x + cAx;
        r_rho = 0.0;
        r_vpara  = 0.0;
        r_vperp_y  = -beta_z;
        r_vperp_z  = +beta_y;
        r_By_k  = -beta_z * std::sqrt(rho0);
        r_Bz_k  = +beta_y * std::sqrt(rho0);
        r_P   = 0.0;
    } else if (mode == SLOW) {
        lam = v0x + cs;
        r_rho = rho0 * alpha_s;
        r_vpara  = +alpha_s * cs;
        r_vperp_y  = +alpha_f * cf * beta_y;
        r_vperp_z  = +alpha_f * cf * beta_z;
        r_By_k  = -alpha_f * cs0 * std::sqrt(rho0) * beta_y;
        r_Bz_k  = -alpha_f * cs0 * std::sqrt(rho0) * beta_z;
        r_P   = alpha_s * gamma * p0;
    } else { // ENTROPY
        lam = v0x;
        r_rho = 1.0;
        r_vpara = 0.0;
        r_vperp_y = 0.0; r_vperp_z = 0.0;
        r_By_k = 0.0; r_Bz_k = 0.0; r_P = 0.0;
    }
    (void)lam;

    // Normalise
    double norm = std::sqrt(r_rho*r_rho + r_vpara*r_vpara
                          + r_vperp_y*r_vperp_y + r_vperp_z*r_vperp_z
                          + r_By_k*r_By_k + r_Bz_k*r_Bz_k + r_P*r_P);
    if (norm > 0) {
        r_rho /= norm; r_vpara /= norm;
        r_vperp_y /= norm; r_vperp_z /= norm;
        r_By_k /= norm; r_Bz_k /= norm; r_P /= norm;
    }

    // Rotate the k̂-frame vector components back to Cartesian (F1-rotation).
    // Velocity: (v_para, v_perp_y, v_perp_z) → (vx, vy, vz) where
    //   (vx, vy) = R(θ)(v_para, v_perp_y),  vz = v_perp_z.
    // B perturbation: in k̂-frame δB has components (0, δB_perp_y, δB_perp_z)
    // (no δB along k̂; 1D wave has δB·k̂ = 0 automatically).  So
    //   (δBx, δBy) = R(θ)(0, δB_perp_y) = (-sinT·δB_perp_y, cosT·δB_perp_y)
    //   δBz = δB_perp_z.

    auto rotate_v = [&](double vpar, double vperp_y,
                        double& vx_out, double& vy_out) {
        vx_out =  cosT * vpar  - sinT * vperp_y;
        vy_out =  sinT * vpar  + cosT * vperp_y;
    };

    int sx = stride_x(), sy = stride_y();
    int ncell = sx * sy;
    int nfx = total_fx(), nfy = total_fy();
    std::vector<double> h_rho(ncell, 0.0), h_mx(ncell, 0.0),
                        h_my(ncell, 0.0), h_mz(ncell, 0.0),
                        h_E(ncell, 0.0), h_Bz_cc(ncell, 0.0);
    std::vector<double> h_Bxf(nfx, 0.0);
    std::vector<double> h_Byf(nfy, 0.0);

    // B field: use vector potential A_z to enforce solenoidal seeding.
    //   uniform B₀ = (Bx0_ref, By0_ref, Bz0)
    //   δB in k̂-frame is (0, r_By_k, r_Bz_k) · A cos(k·x); rotate δB_perp_y
    //   to Cartesian: (δBx, δBy) = R(θ)(0, r_By_k·A cos) with the same r_By_k.
    // Build A_z such that Bx = ∂_y A_z, By = -∂_x A_z.
    //   Bx_pert = (cos(k·x) coefficient) = -sinT · r_By_k
    //     ⇒ A_z = (A · r_By_k) · sin(k·x) · (something); verify via:
    //        A_z(x, y) = A_z0 sin(kx x + ky y)
    //        ∂_y A_z = A_z0 ky cos(...); so A_z0 ky = A · (-sinT · r_By_k)
    //        ∂_x A_z = A_z0 kx cos(...); By_pert = -A_z0 kx cos(...)
    //                = -A_z0 kx cos = A · cosT · r_By_k
    //     consistency: -A_z0 kx = A cosT r_By_k  AND  A_z0 ky = -A sinT r_By_k
    //     both give A_z0 = -A · r_By_k / kmag (using kx = kmag cosT, ky = kmag sinT).
    const double Az0 = -A * r_By_k / kmag;
    auto Az = [&](double x, double y) -> double {
        return Az0 * std::sin(kx * x + ky * y);
    };

    // Bxf at (i+½, j-cell): Bx_pert = ∂_y A_z on the face
    for (int i_face = 0; i_face < nx + 1 + 2 * ng; ++i_face) {
        double xf = x_lo + ((i_face - ng) + 0.0) * dx;
        for (int jc = 0; jc < ny + 2 * ng; ++jc) {
            double y_top = y_lo + ((jc - ng) + 1.0) * dy;
            double y_bot = y_lo + ((jc - ng) + 0.0) * dy;
            double Bx_pert = (Az(xf, y_top) - Az(xf, y_bot)) / dy;
            h_Bxf[i_face * sy + jc] = Bx0_ref + Bx_pert;
        }
    }
    // Byf at (i-cell, j+½): By_pert = -∂_x A_z on the face
    for (int ic = 0; ic < nx + 2 * ng; ++ic) {
        double x_ip = x_lo + ((ic - ng) + 1.0) * dx;
        double x_im = x_lo + ((ic - ng) + 0.0) * dx;
        for (int j_face = 0; j_face < ny + 1 + 2 * ng; ++j_face) {
            double yf = y_lo + ((j_face - ng) + 0.0) * dy;
            double By_pert = -(Az(x_ip, yf) - Az(x_im, yf)) / dx;
            h_Byf[ic * (sy + 1) + j_face] = By0_ref + By_pert;
        }
    }
    // Cell-centred fields: reconstruct Bx_cc, By_cc from faces; velocity
    // and pressure direct from rotated eigenvector.
    for (int ic = 0; ic < nx; ++ic) {
        for (int jc = 0; jc < ny; ++jc) {
            double xc = x_lo + (ic + 0.5) * dx;
            double yc = y_lo + (jc + 0.5) * dy;
            double phase = kx * xc + ky * yc;
            double dW = A * std::cos(phase);
            double r = rho0 + dW * r_rho;
            double vx_cart, vy_cart;
            rotate_v(dW * r_vpara, dW * r_vperp_y, vx_cart, vy_cart);
            double ux = v0x + vx_cart;
            double uy =       vy_cart;
            double uz =       dW * r_vperp_z;
            double Bz_val = Bz0 + dW * r_Bz_k;
            double P = p0 + dW * r_P;
            // Bx_cc, By_cc from face averages (consistent with runtime)
            double Bxc = 0.5 * (h_Bxf[(ic + ng)     * sy + (jc + ng)]
                              + h_Bxf[(ic + ng + 1) * sy + (jc + ng)]);
            double Byc = 0.5 * (h_Byf[(ic + ng) * (sy + 1) + (jc + ng)]
                              + h_Byf[(ic + ng) * (sy + 1) + (jc + ng + 1)]);
            int c = (ic + ng) * sy + (jc + ng);
            set_cell(h_rho, h_mx, h_my, h_mz, h_E, h_Bz_cc,
                     c, r, ux, uy, uz, Bxc, Byc, Bz_val, P, gamma);
        }
    }
    size_t nb_cell = (size_t)ncell * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_rho, h_rho.data(), nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mx,  h_mx.data(),  nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_my,  h_my.data(),  nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mz,  h_mz.data(),  nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E,   h_E.data(),   nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bz_cc, h_Bz_cc.data(), nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bxf, h_Bxf.data(),
                          (size_t)nfx * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Byf, h_Byf.data(),
                          (size_t)nfy * sizeof(double), cudaMemcpyHostToDevice));
    std::fprintf(stderr,
        "  AthenaMHD linwave oblique IC: mode=%d, k=(%d,%d), θ=%.3f, A=%.3e\n",
        (int)mode, kx_int, ky_int, std::atan2(ky, kx), A);
}

void AthenaMHDSolver::init_field_loop(double v_adv_x, double v_adv_y,
                                      double R, double A0) {
    // Gardiner-Stone 2005 Fig 3.  Uniform background + cylindrical B
    // loop built from a z-directed vector potential:
    //   A_z(r) = max(A0·(R − r), 0),  r = √((x−x_c)² + (y−y_c)²)
    //   B_x = ∂_y A_z,  B_y = −∂_x A_z  (on faces)
    // The interior B field is uniform = A0·ê_φ on a disk of radius R,
    // zero outside.  Advected diagonally with v = (v_adv_x, v_adv_y).
    gamma = 5.0 / 3.0;
    x_bc = 0; y_bc = 0;
    x_lo = -0.5; x_hi = 0.5;
    y_lo = -0.5; y_hi = 0.5;
    Lx = x_hi - x_lo;
    Ly = y_hi - y_lo;
    dx = Lx / (double)nx;
    dy = Ly / (double)ny;

    int sx = stride_x(), sy = stride_y();
    int ncell = sx * sy;
    int nfx = total_fx(), nfy = total_fy();
    std::vector<double> h_rho(ncell, 1.0), h_mx(ncell, 0.0),
                        h_my(ncell, 0.0), h_mz(ncell, 0.0),
                        h_E(ncell, 0.0), h_Bz_cc(ncell, 0.0);
    std::vector<double> h_Bxf(nfx, 0.0);
    std::vector<double> h_Byf(nfy, 0.0);

    double rho0 = 1.0, P0 = 1.0;

    // Compute A_z at cell corners (i+½, j+½).  Cell corners relative
    // to the ghost-inclusive layout: corner (i_corner, j_corner) with
    // i_corner ∈ [0, nx+1+2g), j_corner ∈ [0, ny+1+2g).
    //   x_corner = x_lo + (i_corner - ng) * dx
    //   y_corner = y_lo + (j_corner - ng) * dy
    auto Az = [&](double x, double y) -> double {
        double r = std::sqrt(x*x + y*y);
        return (r < R) ? A0 * (R - r) : 0.0;
    };

    // Bxf lives at (i+½, j cell-centre-j) — face normal = ê_x, so
    // B_x = ∂_y A_z  → use finite diff in y across the face.
    // Bxf[i_face, jc] = (A_z(x_face, y_jc+½) − A_z(x_face, y_jc−½)) / dy
    for (int i_face = 0; i_face < nx + 1 + 2 * ng; ++i_face) {
        double xf = x_lo + ((i_face - ng) + 0.0) * dx;  // face at cell boundary
        for (int jc = 0; jc < ny + 2 * ng; ++jc) {
            double yc_jp = y_lo + ((jc - ng) + 1.0) * dy;  // upper corner
            double yc_jm = y_lo + ((jc - ng) + 0.0) * dy;  // lower corner
            h_Bxf[i_face * sy + jc] =
                (Az(xf, yc_jp) - Az(xf, yc_jm)) / dy;
        }
    }
    // Byf lives at (ic cell-centre, j+½) — face normal = ê_y, so
    // B_y = −∂_x A_z  → use finite diff in x across the face.
    // Byf[ic, j_face] = −(A_z(x_ic+½, y_face) − A_z(x_ic−½, y_face)) / dx
    for (int ic = 0; ic < nx + 2 * ng; ++ic) {
        double xc_ip = x_lo + ((ic - ng) + 1.0) * dx;
        double xc_im = x_lo + ((ic - ng) + 0.0) * dx;
        for (int j_face = 0; j_face < ny + 1 + 2 * ng; ++j_face) {
            double yf = y_lo + ((j_face - ng) + 0.0) * dy;
            h_Byf[ic * (sy + 1) + j_face] =
                -(Az(xc_ip, yf) - Az(xc_im, yf)) / dx;
        }
    }
    // Cell-centred state: uniform ρ=1, P=1, v=(v_adv_x, v_adv_y),
    // and cell-centred B reconstructed from Bxf/Byf face averages.
    for (int ic = 0; ic < nx; ++ic) {
        for (int jc = 0; jc < ny; ++jc) {
            int c = (ic + ng) * sy + (jc + ng);
            double Bxc = 0.5 * (h_Bxf[(ic + ng)     * sy + (jc + ng)]
                              + h_Bxf[(ic + ng + 1) * sy + (jc + ng)]);
            double Byc = 0.5 * (h_Byf[(ic + ng) * (sy + 1) + (jc + ng)]
                              + h_Byf[(ic + ng) * (sy + 1) + (jc + ng + 1)]);
            set_cell(h_rho, h_mx, h_my, h_mz, h_E, h_Bz_cc,
                     c, rho0, v_adv_x, v_adv_y, 0.0,
                     Bxc, Byc, 0.0, P0, gamma);
        }
    }

    size_t nb_cell = (size_t)ncell * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_rho, h_rho.data(), nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mx,  h_mx.data(),  nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_my,  h_my.data(),  nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mz,  h_mz.data(),  nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E,   h_E.data(),   nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bz_cc, h_Bz_cc.data(), nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bxf, h_Bxf.data(),
                          (size_t)nfx * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Byf, h_Byf.data(),
                          (size_t)nfy * sizeof(double), cudaMemcpyHostToDevice));
    std::fprintf(stderr,
        "  AthenaMHD field-loop IC: R=%g, A0=%g, v_adv=(%g, %g)\n",
        R, A0, v_adv_x, v_adv_y);
}

void AthenaMHDSolver::init_orszag_tang() {
    // Orszag-Tang 1979, 2D periodic MHD vortex.
    //   L = 2π in both directions; ρ = 25/(36π²),  P = 5/(12π²),  γ=5/3
    //   v = (−sin y, sin x),  B = (−sin y, sin 2x),  B_z = 0
    // Common convention in Stone+08 et al.  Uses a vector potential
    //   A_z = cos(2x)/2 + cos y    (so that B = ∇ × (A_z ê_z))
    // which yields B_x = ∂_y A_z = −sin y, B_y = −∂_x A_z = sin 2x.
    gamma = 5.0 / 3.0;
    x_bc = 0; y_bc = 0;
    x_lo = 0.0; x_hi = 2.0 * M_PI;
    y_lo = 0.0; y_hi = 2.0 * M_PI;
    Lx = x_hi - x_lo;
    Ly = y_hi - y_lo;
    dx = Lx / (double)nx;
    dy = Ly / (double)ny;

    double rho0 = 25.0 / (36.0 * M_PI * M_PI);
    double P0   =  5.0 / (12.0 * M_PI * M_PI);

    int sx = stride_x(), sy = stride_y();
    int ncell = sx * sy;
    int nfx = total_fx(), nfy = total_fy();
    std::vector<double> h_rho(ncell, rho0), h_mx(ncell, 0.0),
                        h_my(ncell, 0.0), h_mz(ncell, 0.0),
                        h_E(ncell, 0.0), h_Bz_cc(ncell, 0.0);
    std::vector<double> h_Bxf(nfx, 0.0);
    std::vector<double> h_Byf(nfy, 0.0);

    // A_z(x, y) = cos(2x) / 2 + cos(y)
    auto Az = [&](double x, double y) -> double {
        return 0.5 * std::cos(2.0 * x) + std::cos(y);
    };

    for (int i_face = 0; i_face < nx + 1 + 2 * ng; ++i_face) {
        double xf = x_lo + ((i_face - ng) + 0.0) * dx;
        for (int jc = 0; jc < ny + 2 * ng; ++jc) {
            double yc_jp = y_lo + ((jc - ng) + 1.0) * dy;
            double yc_jm = y_lo + ((jc - ng) + 0.0) * dy;
            h_Bxf[i_face * sy + jc] = (Az(xf, yc_jp) - Az(xf, yc_jm)) / dy;
        }
    }
    for (int ic = 0; ic < nx + 2 * ng; ++ic) {
        double xc_ip = x_lo + ((ic - ng) + 1.0) * dx;
        double xc_im = x_lo + ((ic - ng) + 0.0) * dx;
        for (int j_face = 0; j_face < ny + 1 + 2 * ng; ++j_face) {
            double yf = y_lo + ((j_face - ng) + 0.0) * dy;
            h_Byf[ic * (sy + 1) + j_face] =
                -(Az(xc_ip, yf) - Az(xc_im, yf)) / dx;
        }
    }
    for (int ic = 0; ic < nx; ++ic) {
        double xc = x_lo + (ic + 0.5) * dx;
        for (int jc = 0; jc < ny; ++jc) {
            double yc = y_lo + (jc + 0.5) * dy;
            int c = (ic + ng) * sy + (jc + ng);
            double u = -std::sin(yc);
            double v =  std::sin(xc);
            double Bxc = 0.5 * (h_Bxf[(ic + ng)     * sy + (jc + ng)]
                              + h_Bxf[(ic + ng + 1) * sy + (jc + ng)]);
            double Byc = 0.5 * (h_Byf[(ic + ng) * (sy + 1) + (jc + ng)]
                              + h_Byf[(ic + ng) * (sy + 1) + (jc + ng + 1)]);
            set_cell(h_rho, h_mx, h_my, h_mz, h_E, h_Bz_cc,
                     c, rho0, u, v, 0.0, Bxc, Byc, 0.0, P0, gamma);
        }
    }

    size_t nb_cell = (size_t)ncell * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_rho, h_rho.data(), nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mx,  h_mx.data(),  nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_my,  h_my.data(),  nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mz,  h_mz.data(),  nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E,   h_E.data(),   nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bz_cc, h_Bz_cc.data(), nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bxf, h_Bxf.data(),
                          (size_t)nfx * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Byf, h_Byf.data(),
                          (size_t)nfy * sizeof(double), cudaMemcpyHostToDevice));
    std::fprintf(stderr,
        "  AthenaMHD Orszag-Tang IC: ρ=%.3e, P=%.3e\n", rho0, P0);
}

// ============================================================
// Ryu-Jones 1995 RJ2a: "all 7 waves" shock tube.
// Stone+08 Fig 14; 1D along x, γ=5/3.
// Values from mhd_repro/standard_mhd_benchmarks.md §2a (original
// Gaussian units; Athena convention divides by √(4π)).
// ============================================================
void AthenaMHDSolver::init_rj2a() {
    gamma = 5.0 / 3.0;
    x_bc = 2;   // outflow in x
    y_bc = 0;   // periodic in y (1D tube)
    x_lo = -0.5; x_hi = 0.5;
    y_lo = 0.0;  y_hi = Ly;
    dx = Lx / (double)nx;
    dy = Ly / (double)ny;

    const double s4pi = std::sqrt(4.0 * M_PI);
    const double Bx0    = 2.0   / s4pi;
    const double By_L   = 3.6   / s4pi;
    const double By_R   = 4.0   / s4pi;
    const double Bz_cst = 2.0   / s4pi;

    int sx = stride_x(), sy = stride_y();
    int ncell = sx * sy;
    int nfx = total_fx(), nfy = total_fy();
    std::vector<double> h_rho(ncell, 0.0), h_mx(ncell, 0.0),
                        h_my(ncell, 0.0), h_mz(ncell, 0.0),
                        h_E(ncell, 0.0), h_Bz_cc(ncell, Bz_cst);
    std::vector<double> h_Bxf(nfx, Bx0);
    std::vector<double> h_Byf(nfy, 0.0);

    for (int ic = 0; ic < nx; ++ic) {
        double xc = x_lo + (ic + 0.5) * dx;
        bool left = (xc < 0.0);
        double r  = left ? 1.08 : 1.0;
        double ux = left ? 1.2  : 0.0;
        double uy = left ? 0.01 : 0.0;
        double uz = left ? 0.5  : 0.0;
        double P  = left ? 0.95 : 1.0;
        double By = left ? By_L : By_R;
        for (int jc = 0; jc < ny; ++jc) {
            int c = (ic + ng) * sy + (jc + ng);
            set_cell(h_rho, h_mx, h_my, h_mz, h_E, h_Bz_cc,
                     c, r, ux, uy, uz, Bx0, By, Bz_cst, P, gamma);
        }
    }
    // Byf from cell-centred (By is piecewise constant along x)
    for (int ic = 0; ic < nx + 2 * ng; ++ic) {
        double xc = x_lo + ((ic - ng) + 0.5) * dx;
        double By = (xc < 0.0) ? By_L : By_R;
        for (int jf = 0; jf < ny + 1 + 2 * ng; ++jf) {
            h_Byf[ic * (sy + 1) + jf] = By;
        }
    }
    size_t nb_cell = (size_t)ncell * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_rho,   h_rho.data(),   nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mx,    h_mx.data(),    nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_my,    h_my.data(),    nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mz,    h_mz.data(),    nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E,     h_E.data(),     nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bz_cc, h_Bz_cc.data(), nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bxf, h_Bxf.data(), (size_t)nfx * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Byf, h_Byf.data(), (size_t)nfy * sizeof(double),
                          cudaMemcpyHostToDevice));
    std::fprintf(stderr,
        "  AthenaMHD Ryu-Jones RJ2a: γ=%g, Bx=%.4f, L-state ρ=1.08 v=(1.2,0.01,0.5) P=0.95\n",
        gamma, Bx0);
}

// ============================================================
// RJ4d — switch-on slow rarefaction (Stone+08 Fig 15).
// ============================================================
void AthenaMHDSolver::init_rj4d() {
    gamma = 5.0 / 3.0;
    x_bc = 2;
    y_bc = 0;
    x_lo = -0.5; x_hi = 0.5;
    y_lo = 0.0;  y_hi = Ly;
    dx = Lx / (double)nx;
    dy = Ly / (double)ny;

    const double Bx0  = 0.7;
    const double By_L = +1.0;
    const double By_R = std::cos(3.0);      // ≈ -0.98999
    const double Bz_L = 0.0;
    const double Bz_R = std::sin(3.0);      // ≈ +0.14112

    int sx = stride_x(), sy = stride_y();
    int ncell = sx * sy;
    int nfx = total_fx(), nfy = total_fy();
    std::vector<double> h_rho(ncell, 0.0), h_mx(ncell, 0.0),
                        h_my(ncell, 0.0), h_mz(ncell, 0.0),
                        h_E(ncell, 0.0), h_Bz_cc(ncell, 0.0);
    std::vector<double> h_Bxf(nfx, Bx0);
    std::vector<double> h_Byf(nfy, 0.0);
    for (int ic = 0; ic < nx; ++ic) {
        double xc = x_lo + (ic + 0.5) * dx;
        bool left = (xc < 0.0);
        double r  = left ? 1.0  : 0.3;
        double P  = left ? 1.0  : 0.2;
        double By = left ? By_L : By_R;
        double Bz = left ? Bz_L : Bz_R;
        for (int jc = 0; jc < ny; ++jc) {
            int c = (ic + ng) * sy + (jc + ng);
            set_cell(h_rho, h_mx, h_my, h_mz, h_E, h_Bz_cc,
                     c, r, 0, 0, 0, Bx0, By, Bz, P, gamma);
        }
    }
    for (int ic = 0; ic < nx + 2 * ng; ++ic) {
        double xc = x_lo + ((ic - ng) + 0.5) * dx;
        double By = (xc < 0.0) ? By_L : By_R;
        for (int jf = 0; jf < ny + 1 + 2 * ng; ++jf) {
            h_Byf[ic * (sy + 1) + jf] = By;
        }
    }
    size_t nb_cell = (size_t)ncell * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_rho,   h_rho.data(),   nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mx,    h_mx.data(),    nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_my,    h_my.data(),    nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mz,    h_mz.data(),    nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E,     h_E.data(),     nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bz_cc, h_Bz_cc.data(), nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bxf, h_Bxf.data(), (size_t)nfx * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Byf, h_Byf.data(), (size_t)nfy * sizeof(double),
                          cudaMemcpyHostToDevice));
    std::fprintf(stderr,
        "  AthenaMHD Ryu-Jones RJ4d switch-on slow: Bx=%.2f, By_R=cos(3)=%.4f\n",
        Bx0, By_R);
}

// ============================================================
// CP Alfvén wave (1D, Tóth 2000; exact nonlinear MHD solution).
// Background: ρ=1, P=0.1, Bx=1 (so c_A = Bx/√ρ = 1).
//   δB⊥ = 0.1 · (sin 2πx, cos 2πx)
//   δv⊥ = ∓ 0.1 · (sin 2πx, cos 2πx)   (+ = right-traveling convention)
// v∥ = 0 traveling, v∥ = -1 standing.
// After t_f = 1 (one period), waveform should return exactly.
// ============================================================
void AthenaMHDSolver::init_cpaw_1d(bool traveling) {
    gamma = 5.0 / 3.0;
    x_bc = 0; y_bc = 0;   // periodic
    x_lo = 0.0; x_hi = Lx;
    y_lo = 0.0; y_hi = Ly;
    dx = Lx / (double)nx;
    dy = Ly / (double)ny;

    const double rho0 = 1.0, P0 = 0.1, Bx0 = 1.0;
    const double dB = 0.1;
    const double v_par = traveling ? 0.0 : -1.0;    // -c_A for standing
    const double sgn_v = traveling ? -1.0 : -1.0;   // right-travel: v = -δB
    const double two_pi = 2.0 * M_PI;

    int sx = stride_x(), sy = stride_y();
    int ncell = sx * sy;
    int nfx = total_fx(), nfy = total_fy();
    std::vector<double> h_rho(ncell, rho0), h_mx(ncell, 0.0),
                        h_my(ncell, 0.0), h_mz(ncell, 0.0),
                        h_E(ncell, 0.0), h_Bz_cc(ncell, 0.0);
    std::vector<double> h_Bxf(nfx, Bx0);
    std::vector<double> h_Byf(nfy, 0.0);
    for (int ic = 0; ic < nx; ++ic) {
        double xc = x_lo + (ic + 0.5) * dx;
        double phase = two_pi * (xc - x_lo) / Lx;
        double By = dB * std::sin(phase);
        double Bz = dB * std::cos(phase);
        double vy = sgn_v * dB * std::sin(phase);
        double vz = sgn_v * dB * std::cos(phase);
        for (int jc = 0; jc < ny; ++jc) {
            int c = (ic + ng) * sy + (jc + ng);
            set_cell(h_rho, h_mx, h_my, h_mz, h_E, h_Bz_cc,
                     c, rho0, v_par, vy, vz, Bx0, By, Bz, P0, gamma);
        }
    }
    for (int ic = 0; ic < nx + 2 * ng; ++ic) {
        double xc = x_lo + ((ic - ng) + 0.5) * dx;
        double phase = two_pi * (xc - x_lo) / Lx;
        double By = dB * std::sin(phase);
        for (int jf = 0; jf < ny + 1 + 2 * ng; ++jf) {
            h_Byf[ic * (sy + 1) + jf] = By;
        }
    }
    size_t nb_cell = (size_t)ncell * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_rho,   h_rho.data(),   nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mx,    h_mx.data(),    nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_my,    h_my.data(),    nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mz,    h_mz.data(),    nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E,     h_E.data(),     nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bz_cc, h_Bz_cc.data(), nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bxf, h_Bxf.data(), (size_t)nfx * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Byf, h_Byf.data(), (size_t)nfy * sizeof(double),
                          cudaMemcpyHostToDevice));
    std::fprintf(stderr,
        "  AthenaMHD CPAW 1D (%s): A=0.1, c_A=1\n",
        traveling ? "traveling" : "standing");
}

// ============================================================
// CP Alfvén 2D (GS05 §5.2).  Wave tilted θ=arctan(2) ≈ 63.4° to x.
// Domain √5 × √5/2 periodic.  B and v components from vector-potential
// rotation so CT ∇·B = 0 at roundoff.
// ============================================================
void AthenaMHDSolver::init_cpaw_2d(bool traveling) {
    gamma = 5.0 / 3.0;
    x_bc = 0; y_bc = 0;
    // Rewrite domain to match GS05
    Lx = std::sqrt(5.0);
    Ly = std::sqrt(5.0) / 2.0;
    x_lo = 0.0; x_hi = Lx;
    y_lo = 0.0; y_hi = Ly;
    dx = Lx / (double)nx;
    dy = Ly / (double)ny;

    const double rho0 = 1.0, P0 = 0.1;
    const double B1   = 1.0;    // parallel to wave (rotated axis)
    const double dB   = 0.1;
    const double cosT = 2.0 / std::sqrt(5.0);
    const double sinT = 1.0 / std::sqrt(5.0);
    const double v1   = traveling ? 0.0 : 1.0;

    int sx = stride_x(), sy = stride_y();
    int ncell = sx * sy;
    int nfx = total_fx(), nfy = total_fy();
    std::vector<double> h_rho(ncell, rho0), h_mx(ncell, 0.0),
                        h_my(ncell, 0.0), h_mz(ncell, 0.0),
                        h_E(ncell, 0.0), h_Bz_cc(ncell, 0.0);
    std::vector<double> h_Bxf(nfx, 0.0);
    std::vector<double> h_Byf(nfy, 0.0);

    // Vector-potential method for solenoidal B.
    // In rotated frame: B1 (parallel), B2 = dB sin(2π x₁), B3 = dB cos(2π x₁)
    //   x₁ = x cosθ + y sinθ
    // Cartesian:
    //   Bx = B1 cosθ − B2 sinθ
    //   By = B1 sinθ + B2 cosθ
    //   Bz = B3
    // With A_z(x,y) such that Bx = ∂A_z/∂y, By = -∂A_z/∂x:
    //   A_z(x,y) = B1·(x sinθ − y cosθ) + (dB/(2π))·cos(2π x₁)
    // (sign chosen so that curl matches Bx, By above — easy to verify
    // by differentiating: ∂A_z/∂y = B1 sinθ + (dB/(2π)) · (-sinθ · 2π)·
    // (-sin 2πx₁) = B1 sinθ + (B1 is wrong; the constant part gives B1
    // sinθ which should equal Bx_parallel component = B1 cosθ) — so the
    // uniform part must be handled separately via a *uniform* face value.
    //
    // Simpler: treat uniform B (B1) by setting uniform face values, and
    // the perturbation via A_z = (dB/(2π))·cos(2π x₁).
    auto Az_pert = [&](double x, double y) -> double {
        double x1 = x * cosT + y * sinT;
        return (dB / (2.0 * M_PI)) * std::cos(2.0 * M_PI * x1);
    };

    // Bxf face (perturbation):  B_x_pert = ∂_y A_z = (dB·sinT)·sin(2πx₁)
    // Uniform part of B_x = B1 cosT.
    for (int i_face = 0; i_face < nx + 1 + 2 * ng; ++i_face) {
        double xf = x_lo + ((i_face - ng) + 0.0) * dx;
        for (int jc = 0; jc < ny + 2 * ng; ++jc) {
            double y_top = y_lo + ((jc - ng) + 1.0) * dy;
            double y_bot = y_lo + ((jc - ng) + 0.0) * dy;
            double Bx_pert = (Az_pert(xf, y_top) - Az_pert(xf, y_bot)) / dy;
            h_Bxf[i_face * sy + jc] = B1 * cosT + Bx_pert;
        }
    }
    // Byf face (perturbation):  B_y_pert = -∂_x A_z = (dB·cosT)·sin(2πx₁)
    // Uniform part of B_y = B1 sinT.
    for (int ic = 0; ic < nx + 2 * ng; ++ic) {
        double x_ip = x_lo + ((ic - ng) + 1.0) * dx;
        double x_im = x_lo + ((ic - ng) + 0.0) * dx;
        for (int j_face = 0; j_face < ny + 1 + 2 * ng; ++j_face) {
            double yf = y_lo + ((j_face - ng) + 0.0) * dy;
            double By_pert = -(Az_pert(x_ip, yf) - Az_pert(x_im, yf)) / dx;
            h_Byf[ic * (sy + 1) + j_face] = B1 * sinT + By_pert;
        }
    }

    // Cell-centred values: v and B reconstructed from face averages.
    for (int ic = 0; ic < nx; ++ic) {
        for (int jc = 0; jc < ny; ++jc) {
            double xc = x_lo + (ic + 0.5) * dx;
            double yc = y_lo + (jc + 0.5) * dy;
            double x1 = xc * cosT + yc * sinT;
            double phase = 2.0 * M_PI * x1;
            double B2 = dB * std::sin(phase);
            double B3 = dB * std::cos(phase);
            double v2 = B2;                          // traveling: v=+δB
            double v3 = B3;
            // Cartesian velocity (v1 along wave, v2 perp-in-plane, v3 out-of-plane)
            double vx = v1 * cosT - v2 * sinT;
            double vy = v1 * sinT + v2 * cosT;
            double vz = v3;
            // B_cc from face averages (consistent with runtime cons_to_prim):
            double Bxc = 0.5 * (h_Bxf[(ic + ng)     * sy + (jc + ng)]
                              + h_Bxf[(ic + ng + 1) * sy + (jc + ng)]);
            double Byc = 0.5 * (h_Byf[(ic + ng) * (sy + 1) + (jc + ng)]
                              + h_Byf[(ic + ng) * (sy + 1) + (jc + ng + 1)]);
            double Bzc = B3;
            int c = (ic + ng) * sy + (jc + ng);
            set_cell(h_rho, h_mx, h_my, h_mz, h_E, h_Bz_cc,
                     c, rho0, vx, vy, vz, Bxc, Byc, Bzc, P0, gamma);
        }
    }
    size_t nb_cell = (size_t)ncell * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_rho,   h_rho.data(),   nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mx,    h_mx.data(),    nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_my,    h_my.data(),    nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mz,    h_mz.data(),    nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E,     h_E.data(),     nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bz_cc, h_Bz_cc.data(), nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bxf, h_Bxf.data(), (size_t)nfx * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Byf, h_Byf.data(), (size_t)nfy * sizeof(double),
                          cudaMemcpyHostToDevice));
    std::fprintf(stderr,
        "  AthenaMHD CPAW 2D (%s): domain %.3f×%.3f, θ=arctan(2)\n",
        traveling ? "traveling" : "standing", Lx, Ly);
}

// ============================================================
// MHD blast wave (Balsara-Spicer 1999 / Stone+08 Fig 28).
// Domain [-0.5, 0.5] × [-0.75, 0.75], periodic.
// High-pressure circular region (r<0.1, P=10) inside uniform B at 45°.
// ============================================================
void AthenaMHDSolver::init_blast() {
    gamma = 5.0 / 3.0;
    x_bc = 0; y_bc = 0;
    Lx = 1.0;
    Ly = 1.5;
    x_lo = -0.5; x_hi = 0.5;
    y_lo = -0.75; y_hi = 0.75;
    dx = Lx / (double)nx;
    dy = Ly / (double)ny;

    const double B0   = 1.0;
    const double Bx0  = B0 / std::sqrt(2.0);
    const double By0  = B0 / std::sqrt(2.0);
    const double rho0 = 1.0;
    const double P_in = 10.0, P_out = 0.1;
    const double R_b  = 0.1;

    int sx = stride_x(), sy = stride_y();
    int ncell = sx * sy;
    int nfx = total_fx(), nfy = total_fy();
    std::vector<double> h_rho(ncell, rho0), h_mx(ncell, 0.0),
                        h_my(ncell, 0.0), h_mz(ncell, 0.0),
                        h_E(ncell, 0.0), h_Bz_cc(ncell, 0.0);
    std::vector<double> h_Bxf(nfx, Bx0);
    std::vector<double> h_Byf(nfy, By0);

    for (int ic = 0; ic < nx; ++ic) {
        double xc = x_lo + (ic + 0.5) * dx;
        for (int jc = 0; jc < ny; ++jc) {
            double yc = y_lo + (jc + 0.5) * dy;
            double rr = std::sqrt(xc * xc + yc * yc);
            double P = (rr < R_b) ? P_in : P_out;
            int c = (ic + ng) * sy + (jc + ng);
            set_cell(h_rho, h_mx, h_my, h_mz, h_E, h_Bz_cc,
                     c, rho0, 0, 0, 0, Bx0, By0, 0, P, gamma);
        }
    }
    // Uniform B: face values already correct (initialised to Bx0, By0).

    size_t nb_cell = (size_t)ncell * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_rho,   h_rho.data(),   nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mx,    h_mx.data(),    nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_my,    h_my.data(),    nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mz,    h_mz.data(),    nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E,     h_E.data(),     nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bz_cc, h_Bz_cc.data(), nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bxf, h_Bxf.data(), (size_t)nfx * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Byf, h_Byf.data(), (size_t)nfy * sizeof(double),
                          cudaMemcpyHostToDevice));
    std::fprintf(stderr,
        "  AthenaMHD MHD blast: B0=%g @45°, β_out=%.3f, β_in=%.1f\n",
        B0, 2.0 * P_out / (B0 * B0), 2.0 * P_in / (B0 * B0));
}

// ============================================================
// MHD rotor (Tóth 2000 Rotor Test 1, Stone+08 Fig 25).
// Domain [-0.5, 0.5]², outflow BC.
// Inside r<0.1: ρ=10, rigid rotation ω=200 (peak |v|=20).
// Outside:      ρ=1, v=0.
// P=1 uniform, B=(5/√4π, 0, 0), γ=1.4.
// ============================================================
void AthenaMHDSolver::init_rotor() {
    gamma = 1.4;
    x_bc = 2; y_bc = 2;
    Lx = 1.0; Ly = 1.0;
    x_lo = -0.5; x_hi = 0.5;
    y_lo = -0.5; y_hi = 0.5;
    dx = Lx / (double)nx;
    dy = Ly / (double)ny;

    const double rho_in = 10.0, rho_out = 1.0;
    const double P0     = 1.0;
    const double r0     = 0.1;
    const double omega  = 200.0;
    const double Bx0    = 5.0 / std::sqrt(4.0 * M_PI);   // ≈ 1.41047

    int sx = stride_x(), sy = stride_y();
    int ncell = sx * sy;
    int nfx = total_fx(), nfy = total_fy();
    std::vector<double> h_rho(ncell, rho_out), h_mx(ncell, 0.0),
                        h_my(ncell, 0.0), h_mz(ncell, 0.0),
                        h_E(ncell, 0.0), h_Bz_cc(ncell, 0.0);
    std::vector<double> h_Bxf(nfx, Bx0);
    std::vector<double> h_Byf(nfy, 0.0);

    for (int ic = 0; ic < nx; ++ic) {
        double xc = x_lo + (ic + 0.5) * dx;
        for (int jc = 0; jc < ny; ++jc) {
            double yc = y_lo + (jc + 0.5) * dy;
            double rr = std::sqrt(xc * xc + yc * yc);
            bool inside = (rr <= r0);
            double rho = inside ? rho_in  : rho_out;
            double ux  = inside ? -omega * yc : 0.0;
            double uy  = inside ? +omega * xc : 0.0;
            int c = (ic + ng) * sy + (jc + ng);
            set_cell(h_rho, h_mx, h_my, h_mz, h_E, h_Bz_cc,
                     c, rho, ux, uy, 0.0, Bx0, 0.0, 0.0, P0, gamma);
        }
    }
    // Uniform Bx, zero By → face arrays already correct.

    size_t nb_cell = (size_t)ncell * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_rho,   h_rho.data(),   nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mx,    h_mx.data(),    nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_my,    h_my.data(),    nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mz,    h_mz.data(),    nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E,     h_E.data(),     nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bz_cc, h_Bz_cc.data(), nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bxf, h_Bxf.data(), (size_t)nfx * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Byf, h_Byf.data(), (size_t)nfy * sizeof(double),
                          cudaMemcpyHostToDevice));
    std::fprintf(stderr,
        "  AthenaMHD rotor: γ=1.4, ω=%.0f (peak v=%.1f), Bx0=%.4f\n",
        omega, omega * r0, Bx0);
}

// ============================================================
// Well-balanced MHSE: snapshot_hse()
//   Captures per-cell R(U_hse) + gravity(U_hse) so the step()
//   residual becomes [R(U) + g(U)] − [R(U_hse) + g(U_hse)].
//   Assumes the CURRENT state (d_rho, d_mx, d_my, d_mz, d_E, d_B*)
//   is an MHSE configuration and d_g_row is populated.
//
// Algorithm:
//   1. Make sure wb_active = false so the subtract kernel is a no-op.
//   2. Zero rhs_hse_* (defensive).
//   3. Do one order-1 predictor stage with dt_probe = 1 — this writes
//      d_rho1 = U^n + 1·(R + g).  (dt_stage = 0.5 at stage 1, so we use
//      dt_probe = 2 to get a full-unit residual, then take the
//      difference U1 − U^n.)
//   4. Copy (d_mx1 − d_mx) / dt_effective → d_rhs_hse_mx, etc.
//   5. Restore solver state (U^n, B_f^n untouched).
//   6. Set wb_active = true.
// ============================================================
__global__ void k_athmhd_wb_extract_defect(
    const double* rho,  const double* mx,  const double* my,
    const double* mz,   const double* E,   const double* Bz,
    const double* rho1, const double* mx1, const double* my1,
    const double* mz1,  const double* E1,  const double* Bz1,
    double* rhs_rho, double* rhs_mx, double* rhs_my,
    double* rhs_mz,  double* rhs_E,  double* rhs_Bz,
    int nx, int ny, int ng, int sy, double inv_dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + ng;
    int j = blockIdx.y * blockDim.y + threadIdx.y + ng;
    if (i >= ng + nx || j >= ng + ny) return;
    int c = i * sy + j;
    rhs_rho[c] = (rho1[c] - rho[c]) * inv_dt;
    rhs_mx [c] = (mx1 [c] - mx [c]) * inv_dt;
    rhs_my [c] = (my1 [c] - my [c]) * inv_dt;
    rhs_mz [c] = (mz1 [c] - mz [c]) * inv_dt;
    rhs_E  [c] = (E1  [c] - E  [c]) * inv_dt;
    rhs_Bz [c] = (Bz1 [c] - Bz [c]) * inv_dt;
}

void AthenaMHDSolver::snapshot_hse() {
    // Ensure clean state: kill any stale defect, disable subtract.
    wb_active = false;
    int ncell = total_cells();
    size_t nb_cell = (size_t)ncell * sizeof(double);
    CUDA_CHECK(cudaMemset(d_rhs_hse_s1_rho, 0, nb_cell));
    CUDA_CHECK(cudaMemset(d_rhs_hse_s1_mx,  0, nb_cell));
    CUDA_CHECK(cudaMemset(d_rhs_hse_s1_my,  0, nb_cell));
    CUDA_CHECK(cudaMemset(d_rhs_hse_s1_mz,  0, nb_cell));
    CUDA_CHECK(cudaMemset(d_rhs_hse_s1_E,   0, nb_cell));
    CUDA_CHECK(cudaMemset(d_rhs_hse_s1_Bz,  0, nb_cell));
    CUDA_CHECK(cudaMemset(d_rhs_hse_s2_rho, 0, nb_cell));
    CUDA_CHECK(cudaMemset(d_rhs_hse_s2_mx,  0, nb_cell));
    CUDA_CHECK(cudaMemset(d_rhs_hse_s2_my,  0, nb_cell));
    CUDA_CHECK(cudaMemset(d_rhs_hse_s2_mz,  0, nb_cell));
    CUDA_CHECK(cudaMemset(d_rhs_hse_s2_E,   0, nb_cell));
    CUDA_CHECK(cudaMemset(d_rhs_hse_s2_Bz,  0, nb_cell));

    int sy = stride_y();
    dim3 b(16, 16), gg((nx + 15) / 16, (ny + 15) / 16);

    // Capture residuals at U = U_hse for both reconstruction orders.
    // At runtime when U^n = U_hse AND stage-1 WB is perfect, U_star
    // will equal U_hse, so stage-2 will compute L_s2(U_hse) — matching
    // what we capture here.
    fill_ghost();
    cons_to_prim();

    // Stage-1 capture (order=1 predictor). With dt_probe=2 → dt_stage=1.
    stage_advance(/*stage=*/1, /*dt=*/2.0);
    k_athmhd_wb_extract_defect<<<gg, b>>>(
        d_rho,  d_mx,  d_my,  d_mz,  d_E,  d_Bz_cc,
        d_rho1, d_mx1, d_my1, d_mz1, d_E1, d_Bz1,
        d_rhs_hse_s1_rho, d_rhs_hse_s1_mx, d_rhs_hse_s1_my,
        d_rhs_hse_s1_mz,  d_rhs_hse_s1_E,  d_rhs_hse_s1_Bz,
        nx, ny, ng, sy, /*inv_dt=*/1.0);

    // Stage-2 capture (order=xorder) using SAME prim(U_hse) still in d_w_*.
    stage_advance(/*stage=*/2, /*dt=*/1.0);
    k_athmhd_wb_extract_defect<<<gg, b>>>(
        d_rho,  d_mx,  d_my,  d_mz,  d_E,  d_Bz_cc,
        d_rho1, d_mx1, d_my1, d_mz1, d_E1, d_Bz1,
        d_rhs_hse_s2_rho, d_rhs_hse_s2_mx, d_rhs_hse_s2_my,
        d_rhs_hse_s2_mz,  d_rhs_hse_s2_E,  d_rhs_hse_s2_Bz,
        nx, ny, ng, sy, /*inv_dt=*/1.0);

    // State U^n, B_f^n intact (stages only wrote _1 scratch).  Turn
    // on WB subtraction from next step() onward.
    wb_active = true;
    std::fprintf(stderr,
        "  [snapshot_hse] captured R(U_hse) defects (s1+s2) on %dx%d, wb_active=true\n",
        nx, ny);
}

// ============================================================
// Isothermal stratified atmosphere IC (§B4 test bed).
//   ρ(y) = ρ₀ exp(−y/H),  c_s² = g·H,  P = ρ·c_s²,  v = 0.
//   Optional uniform B₀ along ê_y (divergence-free by construction).
// After seeding, populates d_g_row and calls snapshot_hse().
// ============================================================
void AthenaMHDSolver::init_hse_atmosphere(double g_val, double H,
                                          double rho0, double B0_y) {
    gamma = 5.0 / 3.0;
    x_bc = 0;   // periodic in x
    y_bc = 1;   // reflective in y — prevents atmosphere from draining
    x_lo = 0.0; x_hi = Lx;
    y_lo = 0.0; y_hi = Ly;
    dx = Lx / (double)nx;
    dy = Ly / (double)ny;

    const double cs2 = g_val * H;   // isothermal sound speed²

    int sx = stride_x(), sy = stride_y();
    int ncell = sx * sy;
    int nfx = total_fx(), nfy = total_fy();
    std::vector<double> h_rho(ncell), h_mx(ncell, 0.0),
                        h_my(ncell, 0.0), h_mz(ncell, 0.0),
                        h_E(ncell), h_Bz_cc(ncell, 0.0);
    std::vector<double> h_Bxf(nfx, 0.0);
    std::vector<double> h_Byf(nfy, B0_y);

    // Cell-centred seeding: ρ, P, E.  Note Bx_cc = 0, By_cc = B0_y.
    for (int jc = 0; jc < ny; ++jc) {
        double yc = y_lo + (jc + 0.5) * dy;
        double rho = rho0 * std::exp(-yc / H);
        double P   = rho * cs2;
        for (int ic = 0; ic < nx; ++ic) {
            int c = (ic + ng) * sy + (jc + ng);
            set_cell(h_rho, h_mx, h_my, h_mz, h_E, h_Bz_cc,
                     c, rho, 0.0, 0.0, 0.0,
                     /*Bx_cc=*/0.0, /*By_cc=*/B0_y, /*Bz_cc=*/0.0,
                     P, gamma);
        }
    }

    size_t nb_cell = (size_t)ncell * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_rho,   h_rho.data(),   nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mx,    h_mx.data(),    nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_my,    h_my.data(),    nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mz,    h_mz.data(),    nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E,     h_E.data(),     nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bz_cc, h_Bz_cc.data(), nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bxf, h_Bxf.data(), (size_t)nfx * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Byf, h_Byf.data(), (size_t)nfy * sizeof(double),
                          cudaMemcpyHostToDevice));

    // Fill gravity row (constant g along −ê_y, all rows).
    h_g_row.assign(ny, g_val);
    CUDA_CHECK(cudaMemcpy(d_g_row, h_g_row.data(), (size_t)ny * sizeof(double),
                          cudaMemcpyHostToDevice));

    std::fprintf(stderr,
        "  AthenaMHD HSE atm: g=%.4g, H=%.4g, ρ₀=%.4g, cs²=%.4g, B0y=%.4g\n",
        g_val, H, rho0, cs2, B0_y);

    snapshot_hse();
}

// ============================================================
// §C6 Spitzer conduction — compute_conduction_dt + apply_conduction
// ============================================================
// Scalar T ghost fill — apply after compute_T to overwrite the ghost
// rows/columns that cons_to_prim computed using (potentially wrong)
// B_cc at the ghost row.  See comment on k_athmhd_ghost_T_y_reflect.
static void fill_T_ghost(AthenaMHDSolver* sv) {
    int sx = sv->stride_x(), sy = sv->stride_y();
    int ng = sv->ng;
    dim3 bx_grid((sy + 63) / 64, ng);
    if (sv->x_bc == 2) {
        k_athmhd_ghost_T_x_outflow<<<bx_grid, dim3(64, 1)>>>(
            sv->d_T_cc, sv->nx, ng, sx, sy);
    } else {
        k_athmhd_ghost_T_x_periodic<<<bx_grid, dim3(64, 1)>>>(
            sv->d_T_cc, sv->nx, ng, sx, sy);
    }
    dim3 by_grid((sx + 63) / 64, ng);
    if (sv->y_bc == 0) {
        k_athmhd_ghost_T_y_periodic<<<by_grid, dim3(64, 1)>>>(
            sv->d_T_cc, sv->ny, ng, sx, sy);
    } else if (sv->y_bc == 2) {
        k_athmhd_ghost_T_y_outflow<<<by_grid, dim3(64, 1)>>>(
            sv->d_T_cc, sv->ny, ng, sx, sy);
    } else {
        k_athmhd_ghost_T_y_reflect<<<by_grid, dim3(64, 1)>>>(
            sv->d_T_cc, sv->ny, ng, sx, sy);
    }
}

double AthenaMHDSolver::compute_conduction_dt() {
    if (kappa0 <= 0.0) return 1e30;
    fill_ghost();
    cons_to_prim();
    int sx = stride_x(), sy = stride_y();
    dim3 bT(16, 16);
    dim3 gT((sx + bT.x - 1) / bT.x, (sy + bT.y - 1) / bT.y);
    k_athmhd_compute_T<<<gT, bT>>>(d_w_rho, d_w_P, d_T_cc, sx, sy);
    fill_T_ghost(this);
    // CFL buffer: per-cell (nx × ny).
    dim3 gc((nx + 15) / 16, (ny + 15) / 16);
    k_athmhd_conduction_cfl<<<gc, dim3(16, 16)>>>(
        d_w_rho, d_T_cc, d_cond_dt_buf, nx, ny, ng, sy,
        dx, dy, kappa0, gamma - 1.0);
    std::vector<double> h_buf((size_t)nx * (size_t)ny);
    CUDA_CHECK(cudaMemcpy(h_buf.data(), d_cond_dt_buf,
                          h_buf.size() * sizeof(double),
                          cudaMemcpyDeviceToHost));
    double dt_min = 1e300;
    for (double v : h_buf) if (v > 0.0 && v < dt_min) dt_min = v;
    return dt_min;
}

void AthenaMHDSolver::apply_conduction(double dt_target) {
    if (kappa0 <= 0.0 || dt_target <= 0.0) return;
    int sx = stride_x(), sy = stride_y();
    dim3 bT(16, 16);
    dim3 gT((sx + bT.x - 1) / bT.x, (sy + bT.y - 1) / bT.y);
    dim3 gfx(((nx + 1) + 15) / 16, (ny + 15) / 16);
    dim3 gfy((nx + 15) / 16, ((ny + 1) + 15) / 16);
    dim3 gc((nx + 15) / 16, (ny + 15) / 16);

    double t_remaining = dt_target;
    int n_sub = 0;
    const int max_sub = 100000;
    while (t_remaining > 0.0 && n_sub < max_sub) {
        fill_ghost();
        cons_to_prim();
        k_athmhd_compute_T<<<gT, bT>>>(d_w_rho, d_w_P, d_T_cc, sx, sy);
        // T ghost: scalar mirror per BC.  Overwrites the bogus ghost
        // T that compute_T produced from (ρ_ghost, P_ghost where
        // P_ghost uses reflected face-B → wrong B_cc → wrong P).
        fill_T_ghost(this);

        // CFL for this sub-step
        k_athmhd_conduction_cfl<<<gc, dim3(16, 16)>>>(
            d_w_rho, d_T_cc, d_cond_dt_buf, nx, ny, ng, sy,
            dx, dy, kappa0, gamma - 1.0);
        std::vector<double> h_buf((size_t)nx * (size_t)ny);
        CUDA_CHECK(cudaMemcpy(h_buf.data(), d_cond_dt_buf,
                              h_buf.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
        double dt_cond = 1e300;
        for (double v : h_buf) if (v > 0.0 && v < dt_cond) dt_cond = v;
        // Take a fraction of CFL for margin; cap at remaining time.
        double dt_sub = 0.45 * dt_cond;
        if (dt_sub > t_remaining) dt_sub = t_remaining;
        if (dt_sub <= 0.0) break;

        // Face fluxes
        k_athmhd_conduction_flux_x<<<gfx, dim3(16, 16)>>>(
            d_T_cc, d_w_Bx, d_w_By, d_w_Bz, d_Fx_cond,
            nx, ny, ng, sx, sy, dx, dy, kappa0);
        k_athmhd_conduction_flux_y<<<gfy, dim3(16, 16)>>>(
            d_T_cc, d_w_Bx, d_w_By, d_w_Bz, d_Gy_cond,
            nx, ny, ng, sx, sy, dx, dy, kappa0);

        // Apply -∇·F_c · dt_sub to E
        k_athmhd_apply_conduction<<<gc, dim3(16, 16)>>>(
            d_E, d_Fx_cond, d_Gy_cond,
            nx, ny, ng, sx, sy, dx, dy, dt_sub);

        t_remaining -= dt_sub;
        ++n_sub;
    }
    if (n_sub >= max_sub) {
        std::fprintf(stderr,
            "  [apply_conduction] WARNING: hit max_sub=%d (t_remaining=%g)\n",
            max_sub, t_remaining);
    }
}

// ============================================================
// §C7 Townsend closed-form optically-thin cooling.
//   Exact integration of dT/dt = -C T^α per cell, unconditionally
//   stable → no subcycle, single kernel launch.
// ============================================================
void AthenaMHDSolver::apply_cooling(double dt) {
    if (!cool_on || dt <= 0.0) return;
    int sx = stride_x(), sy = stride_y();
    (void)sx;
    fill_ghost();
    cons_to_prim();
    dim3 bc(16, 16);
    dim3 gc((nx + bc.x - 1) / bc.x, (ny + bc.y - 1) / bc.y);
    k_athmhd_cool_townsend<<<gc, bc>>>(
        d_w_rho, d_w_P, d_E,
        nx, ny, ng, sy,
        dt, gamma - 1.0,
        cool_Lambda0, cool_Tref, cool_alpha, cool_Tfloor);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// §E1 stochastic broadband driver.
//   init_stochastic_driver: pre-computes log-spaced frequencies,
//     iid-uniform phases, and normalised per-mode amplitudes s.t.
//     ⟨v_x²⟩_{t → ∞} = A_rms².
//   apply_driver(t): evaluate waveform and write into the j=ng row.
// ============================================================
void AthenaMHDSolver::init_stochastic_driver(double A_rms, double f_min,
                                             double f_max, int N_modes,
                                             unsigned seed) {
    if (N_modes <= 0 || A_rms <= 0.0 || f_min <= 0.0 || f_max <= f_min) {
        std::fprintf(stderr,
            "[init_stochastic_driver] invalid params (A_rms=%g, "
            "f_min=%g, f_max=%g, N=%d)\n", A_rms, f_min, f_max, N_modes);
        return;
    }
    driver_Arms   = A_rms;
    driver_fmin   = f_min;
    driver_fmax   = f_max;
    driver_Nmodes = N_modes;

    // Host-side mode table.
    std::vector<double> h_f(N_modes), h_amp(N_modes), h_phi(N_modes);
    double ln_ratio = std::log(f_max / f_min);
    // Amplitude normalisation: Σ_N (A_N)² / 2 = A_rms²
    //   with A_N = A_rms · √(2/N) per log-octave bin.  (§E1, Parseval
    //   identity on N iid-phase sinusoids.)
    double per_mode_amp = A_rms * std::sqrt(2.0 / (double)N_modes);

    // Deterministic RNG: linear congruential, seed-controlled.
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 2.0 * M_PI);
    for (int n = 0; n < N_modes; ++n) {
        double log_f = std::log(f_min) + ln_ratio * (n + 0.5) / (double)N_modes;
        h_f[n]   = std::exp(log_f);
        h_amp[n] = per_mode_amp;     // flat per-mode; log-spacing gives 1/ω total
        h_phi[n] = uniform(rng);
    }

    size_t nb = (size_t)N_modes * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_driver_f,   nb));
    CUDA_CHECK(cudaMalloc(&d_driver_amp, nb));
    CUDA_CHECK(cudaMalloc(&d_driver_phi, nb));
    CUDA_CHECK(cudaMemcpy(d_driver_f,   h_f.data(),   nb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_driver_amp, h_amp.data(), nb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_driver_phi, h_phi.data(), nb, cudaMemcpyHostToDevice));

    driver_on = true;
    std::printf("  [init_stochastic_driver] N=%d modes on [%.3g, %.3g] Hz, "
                "A_rms=%.3g, seed=%u\n",
                N_modes, f_min, f_max, A_rms, seed);
}

void AthenaMHDSolver::apply_driver(double t) {
    if (!driver_on) return;
    int sy = stride_y();
    dim3 b(64, 1), g((nx + 63) / 64, 1);
    k_athmhd_driver_apply<<<g, b>>>(
        d_rho, d_mx, d_my, d_mz, d_E,
        d_driver_f, d_driver_amp, d_driver_phi,
        driver_Nmodes, t, nx, ng, sy);
    CUDA_CHECK(cudaGetLastError());
}
