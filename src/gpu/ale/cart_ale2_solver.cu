// Cartesian 2D ALE solver — orchestration.
//
// Pipeline per step:
//   1. Lagrangian substep: geom, EOS+Q, forces, BC, node update,
//      compatible energy update.           (identical to cart_lag)
//   2. Cell momentum snapshot (from node velocities).
//   3. Swept-edge remap (east then north edges).
//   4. Rebuild node velocities from post-remap cell momentum.
//   5. Snap nodes back to (X0, Y0).  (Eulerian rezone)
//   6. Refresh node mass from (possibly unchanged) dm.

#include "cart_ale2_solver.cuh"
#include "cart_ale2_trace.h"
#include "gpu_common.cuh"
#include "gpu_linalg.cuh"
#include "sod_exact.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <sys/stat.h>
#include <string>

// ==== Forward decls of all kernels in cart_ale_kernels.cu =====
__global__ void k_cale2_geometry(const double*, const double*, double*, double*, int, int);
__global__ void k_cale2_eos_and_q(const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, double*, int, int, double, double, double, int);
__global__ void k_cale2_node_forces(const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, int, int);
__global__ void k_cale2_zero(double*, int);
__global__ void k_cale2_add_gravity(const double*, double*, double, int);
__global__ void k_cale2_add_gravity_var(const double*, double*, const double*, int, int);
__global__ void k_cale2_bc_reflective(const double*, const double*, double*, double*,
    double*, double*, double*, double*, int, int, int);
__global__ void k_cale2_node_update(double*, double*, double*, double*,
    const double*, const double*, const double*, double*, double*, double, int);
__global__ void k_cale2_energy_update(int, int, const double*, const double*,
    const double*, const double*, const double*, double*);
__global__ void k_cale2_cfl(const double*, const double*, const double*,
    const double*, const double*, int, int, double, double, double*);
__global__ void k_cale2_init_nodes(double*, double*, double*, double*, double, double, int, int);
__global__ void k_cale2_reset_mesh(const double*, const double*, double*, double*, int);
__global__ void k_cale2_node_mass(const double*, double*, int, int, int);
__global__ void k_cale2_cell_momentum(const double*, const double*, const double*,
    double*, double*, double*, int, int);
__global__ void k_cale2_remap_init(const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, int);
__global__ void k_cale2_remap_east(const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, int, int, int);
__global__ void k_cale2_remap_north(const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, int, int, int);
__global__ void k_cale2_remap_finalize_cells(const double*, const double*,
                                              const double*, const double*,
                                              const double*, const double*,
                                              const double*, const double*,
                                              double*, double*, int);
__global__ void k_cale2_species_init_scratch(const double*, double*, int);
__global__ void k_cale2_species_remap_east(const double*, const double*, const double*, const double*,
    const double*, const double*, double*, int, int, int);
__global__ void k_cale2_species_remap_north(const double*, const double*, const double*, const double*,
    const double*, const double*, double*, int, int, int);
__global__ void k_cale2_species_finalize(const double*, const double*, double*, double*, int);
__global__ void k_cale2_species_density(const double*, const double*, double*, int);
__global__ void k_cale2_species_slopes(const double*, double*, double*, int, int, double, double, int, int);
__global__ void k_cale2_species_remap_east_2nd(const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, double*, int, int, double, double, int);
__global__ void k_cale2_species_remap_north_2nd(const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, double*, int, int, double, double, int);
__global__ void k_cale2_rebuild_node_v(const double*, const double*, const double*,
    double*, double*,
    double*, double*,  /* e_int_incr, dKE_node_out */
    int, int, int);
__global__ void k_cale2_bc_velocity(double*, double*, int, int, int);
__global__ void k_cale2_periodic_sync_node(double*, double*, int, int, int, int);
__global__ void k_cale2_cell_densities(const double*, const double*, const double*, const double*,
    const double*, double*, double*, double*, double*, int);
__global__ void k_cale2_slopes_minmod(const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, double*, double*, double*, double*,
    int, int, double, double, int, int);
__global__ void k_cale2_remap_east_2nd(const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, int, int, double, double, int);
__global__ void k_cale2_remap_north_2nd(const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, int, int, double, double, int);
__global__ void k_cale2_node_KE(const double*, const double*, const double*,
                                int, int, int, double*);
__global__ void k_cale2_copy(const double*, double*, int);
__global__ void k_cale2_apply_ie_incr(double*, double*, int);
__global__ void k_cale2_zero_buf(double*, int);
__global__ void k_cale2_compute_node_dKE(const double*, const double*,
                                         const double*,
                                         const double*,
                                         const double*, const double*,
                                         double*, double*,
                                         int, int, int);
__global__ void k_cale2_cell_velocity(const double*, const double*, const double*,
                                      double*, double*, int);
__global__ void k_cale2_velocity_slopes(const double*, const double*,
                                        double*, double*, double*, double*,
                                        int, int, double, double, int, int);
__global__ void k_cale2_rebuild_node_v_2nd(const double*, const double*,
                                           const double*, const double*,
                                           const double*, const double*,
                                           const double*,
                                           double*, double*,
                                           double*, double*,  /* e_int_incr, dKE_node_out */
                                           int, int, double, double, int);
__global__ void k_cale2_snapshot(const double*, const double*, const double*,
    const double*, const double*, double*, int, int);
__global__ void k_cale2_ppm_reconstruct(const double*, const double*, const double*, const double*,
    double*, double*, double*, double*,
    double*, double*, double*, double*,
    double*, double*, double*, double*,
    double*, double*, double*, double*,
    int, int, int, int);
__global__ void k_cale2_cell_primitives(const double*, const double*, const double*, const double*,
    const double*, double*, double*, double*, double*, int, double);
__global__ void k_cale2_ppm_reconstruct_char(const double*, const double*, const double*, const double*,
    double*, double*, double*, double*,
    double*, double*, double*, double*,
    double*, double*, double*, double*,
    double*, double*, double*, double*,
    int, int, int, int, double);
__global__ void k_cale2_remap_east_ppm_prim(const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, int, int, double, double, double, int);
__global__ void k_cale2_remap_north_ppm_prim(const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, int, int, double, double, double, int);
__global__ void k_cale2_remap_east_ppm(const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, int, int, double, double, int);
__global__ void k_cale2_remap_north_ppm(const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, int, int, double, double, int);

// Stash Lx/Ly at module scope (used by non-member IC loaders).
static double g_Lx = 0.0, g_Ly = 0.0;

void CartAle2Solver::init(int nx_in, int ny_in, double Lx, double Ly,
                         double gam, double cfl_in) {
    nx = nx_in; ny = ny_in;
    nnode_x = nx + 1; nnode_y = ny + 1;
    nnode = nnode_x * nnode_y;
    ncell = nx * ny;
    nsub = 4 * ncell;
    gamma = gam;
    cfl = cfl_in;
    g_Lx = Lx; g_Ly = Ly;

    auto mal = [](double** p, size_t nbytes) {
        CUDA_CHECK(cudaMalloc(p, nbytes));
        CUDA_CHECK(cudaMemset(*p, 0, nbytes));
    };

    mal(&d_X,  nnode*sizeof(double));  mal(&d_Y,  nnode*sizeof(double));
    mal(&d_X0, nnode*sizeof(double));  mal(&d_Y0, nnode*sizeof(double));
    mal(&d_vX, nnode*sizeof(double));  mal(&d_vY, nnode*sizeof(double));
    mal(&d_FX, nnode*sizeof(double));  mal(&d_FY, nnode*sizeof(double));
    mal(&d_mnode, nnode*sizeof(double));
    mal(&d_dX, nnode*sizeof(double));  mal(&d_dY, nnode*sizeof(double));

    mal(&d_dm,     ncell*sizeof(double));
    mal(&d_Vol,    ncell*sizeof(double));
    mal(&d_Area0,  ncell*sizeof(double));
    mal(&d_rho,    ncell*sizeof(double));
    mal(&d_e_int,  ncell*sizeof(double));
    mal(&d_P,      ncell*sizeof(double));
    mal(&d_Q,      ncell*sizeof(double));
    mal(&d_cs,     ncell*sizeof(double));
    mal(&d_minheight,   ncell*sizeof(double));
    mal(&d_strain_rate, ncell*sizeof(double));

    mal(&d_px_cell, ncell*sizeof(double));
    mal(&d_py_cell, ncell*sizeof(double));
    mal(&d_dm_new,  ncell*sizeof(double));
    mal(&d_ie_new,  ncell*sizeof(double));
    mal(&d_px_new,  ncell*sizeof(double));
    mal(&d_py_new,  ncell*sizeof(double));
    mal(&d_m_node_ref,       nnode*sizeof(double));
    mal(&d_vX_pre,           nnode*sizeof(double));
    mal(&d_vY_pre,           nnode*sizeof(double));
    mal(&d_KE_node_snap_pre, nnode*sizeof(double));
    mal(&d_KE_node_snap_post,nnode*sizeof(double));
    mal(&d_dKE_node_local,   nnode*sizeof(double));
    mal(&d_e_int_incr,       ncell*sizeof(double));
    mal(&d_node_reduce_buf,  nnode*sizeof(double));
    // e_int_incr must start zeroed — rebuild atomic-adds into it.
    cudaMemset(d_e_int_incr, 0, ncell*sizeof(double));
    mal(&d_vxc, ncell*sizeof(double));
    mal(&d_vyc, ncell*sizeof(double));
    mal(&d_vxc_sx, ncell*sizeof(double));
    mal(&d_vxc_sy, ncell*sizeof(double));
    mal(&d_vyc_sx, ncell*sizeof(double));
    mal(&d_vyc_sy, ncell*sizeof(double));

    mal(&d_rho_dens,  ncell*sizeof(double));
    mal(&d_rhoE_dens, ncell*sizeof(double));
    mal(&d_pxd_dens,  ncell*sizeof(double));
    mal(&d_pyd_dens,  ncell*sizeof(double));
    mal(&d_rho_sx,  ncell*sizeof(double)); mal(&d_rho_sy,  ncell*sizeof(double));
    mal(&d_rhoE_sx, ncell*sizeof(double)); mal(&d_rhoE_sy, ncell*sizeof(double));
    mal(&d_pxd_sx,  ncell*sizeof(double)); mal(&d_pxd_sy,  ncell*sizeof(double));
    mal(&d_pyd_sx,  ncell*sizeof(double)); mal(&d_pyd_sy,  ncell*sizeof(double));

    // PPM face values (only if used; alloc anyway, cheap)
    mal(&d_rho_xL,  ncell*sizeof(double)); mal(&d_rho_xR,  ncell*sizeof(double));
    mal(&d_rho_yD,  ncell*sizeof(double)); mal(&d_rho_yU,  ncell*sizeof(double));
    mal(&d_rhoE_xL, ncell*sizeof(double)); mal(&d_rhoE_xR, ncell*sizeof(double));
    mal(&d_rhoE_yD, ncell*sizeof(double)); mal(&d_rhoE_yU, ncell*sizeof(double));
    mal(&d_pxd_xL,  ncell*sizeof(double)); mal(&d_pxd_xR,  ncell*sizeof(double));
    mal(&d_pxd_yD,  ncell*sizeof(double)); mal(&d_pxd_yU,  ncell*sizeof(double));
    mal(&d_pyd_xL,  ncell*sizeof(double)); mal(&d_pyd_xR,  ncell*sizeof(double));
    mal(&d_pyd_yD,  ncell*sizeof(double)); mal(&d_pyd_yU,  ncell*sizeof(double));

    dx_u = Lx / (double)nx;
    dy_u = Ly / (double)ny;

    mal(&d_FSX, nsub*sizeof(double));
    mal(&d_FSY, nsub*sizeof(double));

    mal(&d_dt_cell,    ncell*sizeof(double));
    mal(&d_reduce_buf, ncell*sizeof(double));

    int B = 256;
    k_cale2_init_nodes<<<(nnode+B-1)/B, B>>>(d_X, d_Y, d_X0, d_Y0,
                                            Lx, Ly, nnode_x, nnode_y);
    k_cale2_geometry<<<(ncell+B-1)/B, B>>>(d_X, d_Y, d_Vol, d_minheight, nx, ny);
    CUDA_CHECK(cudaMemcpy(d_Area0, d_Vol, ncell*sizeof(double), cudaMemcpyDeviceToDevice));

    std::fprintf(stderr,
        "CartAle2Solver: %dx%d cells, box=[%g,%g]x[%g,%g], γ=%g, CFL=%g (Eulerian rezone)\n",
        nx, ny, 0.0, Lx, 0.0, Ly, gamma, cfl);
}

void CartAle2Solver::destroy() {
    free_frame_buffer();
    auto f = [](double* p) { if (p) cudaFree(p); };
    f(d_X); f(d_Y); f(d_X0); f(d_Y0); f(d_vX); f(d_vY);
    f(d_FX); f(d_FY); f(d_mnode); f(d_dX); f(d_dY);
    f(d_dm); f(d_Vol); f(d_Area0); f(d_rho); f(d_e_int);
    f(d_P); f(d_Q); f(d_cs); f(d_minheight); f(d_strain_rate);
    f(d_px_cell); f(d_py_cell);
    f(d_dm_new); f(d_ie_new); f(d_px_new); f(d_py_new);
    f(d_m_node_ref); f(d_vX_pre); f(d_vY_pre);
    f(d_KE_node_snap_pre); f(d_KE_node_snap_post);
    f(d_dKE_node_local); f(d_e_int_incr); f(d_node_reduce_buf);
    f(d_vxc); f(d_vyc);
    f(d_vxc_sx); f(d_vxc_sy); f(d_vyc_sx); f(d_vyc_sy);
    f(d_rho_dens); f(d_rhoE_dens); f(d_pxd_dens); f(d_pyd_dens);
    f(d_rho_sx); f(d_rho_sy); f(d_rhoE_sx); f(d_rhoE_sy);
    f(d_pxd_sx); f(d_pxd_sy); f(d_pyd_sx); f(d_pyd_sy);
    f(d_rho_xL);  f(d_rho_xR);  f(d_rho_yD);  f(d_rho_yU);
    f(d_rhoE_xL); f(d_rhoE_xR); f(d_rhoE_yD); f(d_rhoE_yU);
    f(d_pxd_xL);  f(d_pxd_xR);  f(d_pxd_yD);  f(d_pxd_yU);
    f(d_pyd_xL);  f(d_pyd_xR);  f(d_pyd_yD);  f(d_pyd_yU);
    f(d_FSX); f(d_FSY);
    f(d_dt_cell); f(d_reduce_buf);
    f(d_e_ref_y);
    f(d_cool_weight_y);
    f(d_heat_dedt_base_y);
    f(d_gy_node);
    f(d_species_X); f(d_mX); f(d_mX_new);
    f(d_mXd); f(d_mXd_sx); f(d_mXd_sy);
    std::memset(this, 0, sizeof(*this));
}

// ============================================================
// Passive species tracer X ∈ [0, 1] — conservative donor-cell swept flux
// via species mass mX = X·dm.  Initialises X with Andrassy Eq. 3 cosine
// ramp over (y_lo, y_hi).  Below y_lo: X=0 (μ₀ fluid); above y_hi: X=1
// (μ₁ fluid).  Default ramp in cart_ale2 slab coords: y∈(Y_CB−1/16, Y_CB+1/16)
// with Y_CB = 1 (paper's y=2 shifted to local y=1).
// ============================================================
void CartAle2Solver::init_tracer_ramp(double y_lo, double y_hi) {
    if (d_species_X == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_species_X, ncell * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_mX,        ncell * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_mX_new,    ncell * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_mXd,       ncell * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_mXd_sx,    ncell * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_mXd_sy,    ncell * sizeof(double)));
    }
    std::vector<double> h_Y(nnode);
    CUDA_CHECK(cudaMemcpy(h_Y.data(), d_Y, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_dm(ncell);
    CUDA_CHECK(cudaMemcpy(h_dm.data(), d_dm, ncell*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_X(ncell), h_mX(ncell);
    for (int ic = 0; ic < nx; ++ic)
        for (int jc = 0; jc < ny; ++jc) {
            int flat = ic*ny + jc;
            int I0 = ic*nnode_y + jc;
            int I3 = ic*nnode_y + jc + 1;
            double Yc = 0.5 * (h_Y[I0] + h_Y[I3]);
            double X_val;
            if      (Yc <= y_lo) X_val = 0.0;
            else if (Yc >= y_hi) X_val = 1.0;
            else {
                double s = (Yc - y_lo) / (y_hi - y_lo);
                X_val = 0.5 * (1.0 - std::cos(M_PI * s));
            }
            h_X[flat]  = X_val;
            h_mX[flat] = X_val * h_dm[flat];
        }
    CUDA_CHECK(cudaMemcpy(d_species_X, h_X.data(),  ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mX,        h_mX.data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    tracer_enabled = true;
    std::fprintf(stderr,
        "  [tracer] species X enabled: ramp y∈[%.4f, %.4f] (Andrassy Eq. 3)\n",
        y_lo, y_hi);
}

double CartAle2Solver::total_species_mass() {
    if (!tracer_enabled || d_mX == nullptr) return 0.0;
    std::vector<double> h_mX(ncell);
    CUDA_CHECK(cudaMemcpy(h_mX.data(), d_mX, ncell*sizeof(double),
                          cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (int c = 0; c < ncell; ++c) sum += h_mX[c];
    return sum;
}

// ============================================================
// Newton cooling + bottom enthalpy-flux heating:
//   e ← e + (e_ref(y) − e)·α_cool·s_cool(y) + q_base(y)/ρ · dt
// α_cool = 1 − exp(−dt/τ); q_base(y) = F_bot·g(y), ∫g dy = 1.
// s_cool(y) is a cosine ramp active in the top cool_top_frac of the column.
// Density (dm/V) is untouched → mass conservation is exact.
// ============================================================
__global__ static void k_cale2_thermal_step(double* __restrict__ e_int,
                                            const double* __restrict__ dm,
                                            const double* __restrict__ Area0,
                                            const double* __restrict__ e_ref_y,
                                            const double* __restrict__ w_cool_y,
                                            const double* __restrict__ q_base_y,
                                            double alpha_cool, double dt,
                                            int nx, int ny, int has_cool, int has_heat) {
    int ic = blockIdx.x * blockDim.x + threadIdx.x;
    int jc = blockIdx.y * blockDim.y + threadIdx.y;
    if (ic >= nx || jc >= ny) return;
    int idx = ic * ny + jc;
    double e = e_int[idx];
    if (has_cool) {
        double w = w_cool_y[jc];
        double eref = e_ref_y[jc];
        e += (eref - e) * alpha_cool * w;
    }
    if (has_heat) {
        // q_base(y) is volumetric power density [erg/(s·cm³)].
        // Per-cell Δe = q · dt / ρ, with ρ = dm / (Area0 · 1 cm depth-equiv).
        double rho = dm[idx] / Area0[idx];
        e += q_base_y[jc] * dt / rho;
    }
    e_int[idx] = e;
}

void CartAle2Solver::alloc_cooling_ref(const std::vector<double>& e_ref_per_row) {
    if (d_e_ref_y) { cudaFree(d_e_ref_y); d_e_ref_y = nullptr; }
    if ((int)e_ref_per_row.size() != ny) {
        std::fprintf(stderr,
            "  [alloc_cooling_ref] e_ref size=%zu != ny=%d\n",
            e_ref_per_row.size(), ny);
        std::abort();
    }
    CUDA_CHECK(cudaMalloc(&d_e_ref_y, ny * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_e_ref_y, e_ref_per_row.data(),
                          ny * sizeof(double), cudaMemcpyHostToDevice));
}

// ============================================================
// Variable gravity g(y) configuration.  Input: one g value per node row
// (length = nnode_y = ny + 1) in the same units as scalar g_y (downward
// positive — force is -g·m).  Also caches the reference potential
// Φ(Y_row) = -∫₀^Y g(y') dy' (trapezoid on the uniform initial grid) for
// use in the total-PE diagnostic after particles drift.
// ============================================================
void CartAle2Solver::configure_variable_gravity(
        const std::vector<double>& gy_per_node_row) {
    if ((int)gy_per_node_row.size() != nnode_y) {
        std::fprintf(stderr,
            "configure_variable_gravity: expected %d entries, got %zu\n",
            nnode_y, gy_per_node_row.size());
        return;
    }
    h_gy_node_ref = gy_per_node_row;

    // Φ(Y_node) = -∫₀^Y g dy.  In cart_ale2 gravity is defined so that the
    // node force is FY += -g·m (i.e. g is the magnitude of downward pull,
    // and Y increases upward), so Φ(Y) = +∫₀^Y g dy (raising a mass
    // against gravity costs +g·dy of potential energy).
    h_phi_node_ref.assign(nnode_y, 0.0);
    double dy = g_Ly / (double)ny;
    for (int jn = 1; jn < nnode_y; ++jn) {
        double g_avg = 0.5 * (h_gy_node_ref[jn - 1] + h_gy_node_ref[jn]);
        h_phi_node_ref[jn] = h_phi_node_ref[jn - 1] + g_avg * dy;
    }

    // Upload to device.
    if (d_gy_node == nullptr)
        CUDA_CHECK(cudaMalloc(&d_gy_node, nnode_y * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_gy_node, h_gy_node_ref.data(),
                          nnode_y * sizeof(double), cudaMemcpyHostToDevice));

    // Report.
    double g_min = h_gy_node_ref[0], g_max = h_gy_node_ref[0];
    for (double v : h_gy_node_ref) { g_min = std::min(g_min, v); g_max = std::max(g_max, v); }
    std::fprintf(stderr,
        "  [gravity] variable g(y): range [%.4e, %.4e], Φ(top)=%.4e\n",
        g_min, g_max, h_phi_node_ref.back());
}

// Replace the canned exp(-y/H) heating shape with a user-supplied
// volumetric profile.  Used by Andrassy 2022 pilot which wants
// q̇(y) = q̇₀ sin(8π(y-1)) on a narrow band [1, 9/8].
void CartAle2Solver::configure_heating_profile(
        const std::vector<double>& qdot_per_row,
        double cool_top_frac_) {
    if ((int)qdot_per_row.size() != ny) {
        std::fprintf(stderr,
            "configure_heating_profile: expected %d entries, got %zu\n",
            ny, qdot_per_row.size());
        return;
    }
    cool_top_frac = cool_top_frac_;

    // Total integrated heating for a log line; F_bot isn't strictly defined
    // here (no single "bottom flux" since heating may be interior) but we
    // repurpose it as ∫q̇ dy · 1 cell thickness ≈ total volumetric power
    // per unit x-extent.  Setting bottom_heat_flux > 0 is the existing
    // "has_heat" gate in apply_cooling.
    double dy = g_Ly / (double)ny;
    double integral = 0.0;
    for (double q : qdot_per_row) integral += q * dy;
    bottom_heat_flux = std::max(integral, 1e-300);  // non-zero ⇒ has_heat=true

    // Cooling weight: reset per cool_top_frac.
    std::vector<double> h_wcool(ny, 1.0);
    if (cool_top_frac < 1.0) {
        double y_on = (1.0 - cool_top_frac) * g_Ly;
        for (int jc = 0; jc < ny; ++jc) {
            double yc = (jc + 0.5) * dy;
            if (yc <= y_on) h_wcool[jc] = 0.0;
            else {
                double u = (yc - y_on) / (g_Ly - y_on);
                h_wcool[jc] = 0.5 * (1.0 - std::cos(M_PI * u));
            }
        }
    }
    if (d_cool_weight_y) { cudaFree(d_cool_weight_y); d_cool_weight_y = nullptr; }
    CUDA_CHECK(cudaMalloc(&d_cool_weight_y, ny * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_cool_weight_y, h_wcool.data(),
                          ny * sizeof(double), cudaMemcpyHostToDevice));

    if (d_heat_dedt_base_y) { cudaFree(d_heat_dedt_base_y); d_heat_dedt_base_y = nullptr; }
    CUDA_CHECK(cudaMalloc(&d_heat_dedt_base_y, ny * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_heat_dedt_base_y, qdot_per_row.data(),
                          ny * sizeof(double), cudaMemcpyHostToDevice));

    std::fprintf(stderr,
        "  [thermal] custom q̇(y) profile, ∫q̇ dy = %.6e, cooling top frac=%.3f\n",
        integral, cool_top_frac);
}

void CartAle2Solver::configure_thermal(double F_bot,
                                       double heat_bot_frac_,
                                       double cool_top_frac_) {
    bottom_heat_flux = F_bot;
    if (heat_bot_frac_ > 0.0) heat_bot_frac = heat_bot_frac_;
    if (cool_top_frac_ > 0.0) cool_top_frac = cool_top_frac_;

    double Ly = g_Ly;
    double dy = Ly / ny;

    std::vector<double> h_wcool(ny, 1.0);
    if (cool_top_frac < 1.0) {
        double y_on = (1.0 - cool_top_frac) * Ly;    // cooling starts here
        for (int jc = 0; jc < ny; ++jc) {
            double yc = (jc + 0.5) * dy;
            if (yc <= y_on) {
                h_wcool[jc] = 0.0;
            } else {
                double u = (yc - y_on) / (Ly - y_on);   // 0 at start → 1 at top
                h_wcool[jc] = 0.5 * (1.0 - std::cos(M_PI * u));
            }
        }
    }
    if (d_cool_weight_y) { cudaFree(d_cool_weight_y); d_cool_weight_y = nullptr; }
    CUDA_CHECK(cudaMalloc(&d_cool_weight_y, ny * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_cool_weight_y, h_wcool.data(),
                          ny * sizeof(double), cudaMemcpyHostToDevice));

    std::vector<double> h_qbase(ny, 0.0);
    if (F_bot > 0.0) {
        double H = heat_bot_frac * Ly;
        double wsum = 0.0;
        std::vector<double> w(ny);
        for (int jc = 0; jc < ny; ++jc) {
            double yc = (jc + 0.5) * dy;
            w[jc] = std::exp(-yc / H);
            wsum += w[jc] * dy;
        }
        // q(y) = F_bot · g(y), ∫g dy = 1  → volumetric power density [erg/s/cm³]
        for (int jc = 0; jc < ny; ++jc)
            h_qbase[jc] = F_bot * w[jc] / wsum;
    }
    if (d_heat_dedt_base_y) { cudaFree(d_heat_dedt_base_y); d_heat_dedt_base_y = nullptr; }
    CUDA_CHECK(cudaMalloc(&d_heat_dedt_base_y, ny * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_heat_dedt_base_y, h_qbase.data(),
                          ny * sizeof(double), cudaMemcpyHostToDevice));

    if (F_bot > 0.0) {
        std::fprintf(stderr,
            "  [thermal] bottom heat F=%.3e erg/cm²/s, e-fold H=%.3e cm (%.2f Ly); "
            "cooling top frac=%.2f\n",
            F_bot, heat_bot_frac * Ly, heat_bot_frac, cool_top_frac);
    }
}

void CartAle2Solver::apply_cooling(double dt) {
    bool has_cool = (tau_cool > 0.0 && d_e_ref_y != nullptr);
    bool has_heat = (bottom_heat_flux > 0.0 && d_heat_dedt_base_y != nullptr);
    if (!has_cool && !has_heat) return;

    double alpha = has_cool ? (1.0 - std::exp(-dt / tau_cool)) : 0.0;

    // Cool-weight buffer is allocated lazily on the first call when cooling is
    // enabled but configure_thermal() was never invoked.
    if (has_cool && d_cool_weight_y == nullptr) {
        std::vector<double> ones(ny, 1.0);
        CUDA_CHECK(cudaMalloc(&d_cool_weight_y, ny * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_cool_weight_y, ones.data(),
                              ny * sizeof(double), cudaMemcpyHostToDevice));
    }

    dim3 B(16, 16);
    dim3 G((nx + B.x - 1) / B.x, (ny + B.y - 1) / B.y);
    k_cale2_thermal_step<<<G, B>>>(d_e_int, d_dm, d_Area0,
                                   d_e_ref_y, d_cool_weight_y, d_heat_dedt_base_y,
                                   alpha, dt, nx, ny,
                                   has_cool ? 1 : 0, has_heat ? 1 : 0);
}

// ============================================================
// ICs
// ============================================================
void CartAle2Solver::init_uniform(double rho, double P, double vx, double vy) {
    std::vector<double> h_Vol(ncell);
    CUDA_CHECK(cudaMemcpy(h_Vol.data(), d_Vol, ncell*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_dm(ncell, 0), h_e(ncell, 0);
    for (int c = 0; c < ncell; ++c) {
        h_dm[c] = rho * h_Vol[c];
        h_e[c]  = P / ((gamma - 1.0) * rho);
    }
    CUDA_CHECK(cudaMemcpy(d_dm,   h_dm.data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_e_int, h_e.data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    std::vector<double> h_vX(nnode, vx), h_vY(nnode, vy);
    CUDA_CHECK(cudaMemcpy(d_vX, h_vX.data(), nnode*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vY, h_vY.data(), nnode*sizeof(double), cudaMemcpyHostToDevice));
    int B = 256;
    k_cale2_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nx, ny, bc_mode);
    std::fprintf(stderr, "  CartAle Uniform IC: ρ=%g, P=%g, v=(%g,%g)\n", rho, P, vx, vy);
}

void CartAle2Solver::init_shear_mode(double rho, double P, double V0, int k) {
    g_y = 0.0;
    double Ly = g_Ly;
    std::vector<double> h_Vol(ncell);
    CUDA_CHECK(cudaMemcpy(h_Vol.data(), d_Vol, ncell*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_dm(ncell), h_e(ncell);
    double e_unif = P / ((gamma - 1.0) * rho);
    for (int c = 0; c < ncell; ++c) {
        h_dm[c] = rho * h_Vol[c];
        h_e[c]  = e_unif;
    }
    CUDA_CHECK(cudaMemcpy(d_dm,    h_dm.data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_e_int, h_e.data(),  ncell*sizeof(double), cudaMemcpyHostToDevice));
    std::vector<double> h_Y(nnode);
    CUDA_CHECK(cudaMemcpy(h_Y.data(), d_Y, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_vX(nnode, 0.0), h_vY(nnode, 0.0);
    for (int in = 0; in < nnode_x; ++in)
        for (int jn = 0; jn < nnode_y; ++jn) {
            int f = in * nnode_y + jn;
            double y = h_Y[f];
            h_vX[f] = V0 * std::sin(k * 2.0 * M_PI * y / Ly);
        }
    CUDA_CHECK(cudaMemcpy(d_vX, h_vX.data(), nnode*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vY, h_vY.data(), nnode*sizeof(double), cudaMemcpyHostToDevice));
    int B = 256;
    k_cale2_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nx, ny, bc_mode);
    std::fprintf(stderr,
        "  CartAle2 shear_mode IC: rho=%g P=%g V0=%g k=%d  (Ly=%g, k_phys=%g)\n",
        rho, P, V0, k, Ly, k * 2.0 * M_PI / Ly);
}

// ----- Linear acoustic wave ----------------------------------
// Background (ρ₀, P₀, v=0), right-acoustic perturbation along R⁺ =
// (1, c₀/ρ₀, 0, c₀²). One period T = Lx/c₀ returns state to IC exactly.
void CartAle2Solver::init_acoustic_wave(double rho0, double P0, double A, int k) {
    g_y = 0.0;
    double Lx = g_Lx;
    const double c0 = std::sqrt(gamma * P0 / rho0);
    std::vector<double> h_X(nnode), h_Y(nnode);
    CUDA_CHECK(cudaMemcpy(h_X.data(), d_X, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Y.data(), d_Y, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_Vol(ncell);
    CUDA_CHECK(cudaMemcpy(h_Vol.data(), d_Vol, ncell*sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> h_dm(ncell), h_e(ncell);
    for (int ic = 0; ic < nx; ++ic)
        for (int jc = 0; jc < ny; ++jc) {
            int flat = ic*ny + jc;
            int I[4] = { ic*nnode_y + jc, (ic+1)*nnode_y + jc,
                         (ic+1)*nnode_y + (jc+1), ic*nnode_y + (jc+1) };
            double Xc = 0.25 * (h_X[I[0]] + h_X[I[1]] + h_X[I[2]] + h_X[I[3]]);
            double s = std::sin(k * 2.0 * M_PI * Xc / Lx);
            double rho = rho0 + A * rho0 * s;
            double P   = P0   + A * rho0 * c0 * c0 * s;
            h_dm[flat] = rho * h_Vol[flat];
            h_e[flat]  = P / ((gamma - 1.0) * rho);
        }
    CUDA_CHECK(cudaMemcpy(d_dm,    h_dm.data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_e_int, h_e.data(),  ncell*sizeof(double), cudaMemcpyHostToDevice));

    // Node velocities: δvx = (c0/ρ0)·δρ = A·c0·sin(k·2π·X/Lx), vy=0.
    std::vector<double> h_vX(nnode), h_vY(nnode, 0.0);
    for (int in = 0; in < nnode_x; ++in)
        for (int jn = 0; jn < nnode_y; ++jn) {
            int f = in * nnode_y + jn;
            double s = std::sin(k * 2.0 * M_PI * h_X[f] / Lx);
            h_vX[f] = A * c0 * s;
        }
    CUDA_CHECK(cudaMemcpy(d_vX, h_vX.data(), nnode*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vY, h_vY.data(), nnode*sizeof(double), cudaMemcpyHostToDevice));
    int B = 256;
    k_cale2_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nx, ny, bc_mode);
    std::fprintf(stderr,
        "  CartAle2 acoustic_wave IC: rho0=%g P0=%g A=%g k=%d c0=%g (Lx=%g, period=%g)\n",
        rho0, P0, A, k, c0, Lx, Lx / c0);
}

void CartAle2Solver::init_entropy_wave(double rho0, double P0, double u0,
                                       double A, int k) {
    g_y = 0.0;
    double Lx = g_Lx;
    std::vector<double> h_X(nnode), h_Y(nnode);
    CUDA_CHECK(cudaMemcpy(h_X.data(), d_X, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Y.data(), d_Y, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_Vol(ncell);
    CUDA_CHECK(cudaMemcpy(h_Vol.data(), d_Vol, ncell*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_dm(ncell), h_e(ncell);
    for (int ic = 0; ic < nx; ++ic)
        for (int jc = 0; jc < ny; ++jc) {
            int flat = ic*ny + jc;
            int I[4] = { ic*nnode_y + jc, (ic+1)*nnode_y + jc,
                         (ic+1)*nnode_y + (jc+1), ic*nnode_y + (jc+1) };
            double Xc = 0.25 * (h_X[I[0]] + h_X[I[1]] + h_X[I[2]] + h_X[I[3]]);
            double rho = rho0 * (1.0 + A * std::sin(k * 2.0 * M_PI * Xc / Lx));
            h_dm[flat] = rho * h_Vol[flat];
            h_e[flat]  = P0 / ((gamma - 1.0) * rho);
        }
    CUDA_CHECK(cudaMemcpy(d_dm,    h_dm.data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_e_int, h_e.data(),  ncell*sizeof(double), cudaMemcpyHostToDevice));
    std::vector<double> h_vX(nnode, u0), h_vY(nnode, 0.0);
    CUDA_CHECK(cudaMemcpy(d_vX, h_vX.data(), nnode*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vY, h_vY.data(), nnode*sizeof(double), cudaMemcpyHostToDevice));
    int B = 256;
    k_cale2_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nx, ny, bc_mode);
    std::fprintf(stderr,
        "  CartAle2 entropy_wave IC: rho0=%g P0=%g u0=%g A=%g k=%d  (Lx=%g, period=%g)\n",
        rho0, P0, u0, A, k, Lx, Lx / u0);
}

void CartAle2Solver::init_sod() {
    std::vector<double> h_X(nnode), h_Y(nnode);
    CUDA_CHECK(cudaMemcpy(h_X.data(), d_X, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Y.data(), d_Y, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_Vol(ncell);
    CUDA_CHECK(cudaMemcpy(h_Vol.data(), d_Vol, ncell*sizeof(double), cudaMemcpyDeviceToHost));
    double Lx = g_Lx;
    std::vector<double> h_dm(ncell), h_e(ncell);
    for (int ic = 0; ic < nx; ++ic)
        for (int jc = 0; jc < ny; ++jc) {
            int flat = ic*ny + jc;
            int I[4] = { ic*nnode_y + jc, (ic+1)*nnode_y + jc,
                         (ic+1)*nnode_y + (jc+1), ic*nnode_y + (jc+1) };
            double Xc = 0.25 * (h_X[I[0]] + h_X[I[1]] + h_X[I[2]] + h_X[I[3]]);
            double rho, P;
            if (Xc < 0.5 * Lx) { rho = 1.0;   P = 1.0; }
            else               { rho = 0.125; P = 0.1; }
            h_dm[flat] = rho * h_Vol[flat];
            h_e[flat]  = P / ((gamma - 1.0) * rho);
        }
    CUDA_CHECK(cudaMemcpy(d_dm,    h_dm.data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_e_int, h_e.data(),  ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_vX, 0, nnode*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_vY, 0, nnode*sizeof(double)));
    int B = 256;
    k_cale2_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nx, ny, bc_mode);
    std::fprintf(stderr, "  CartAle Sod IC: ρL=1.0 PL=1.0 | ρR=0.125 PR=0.1\n");
}

// ----- 2D Sedov-Taylor cylindrical blast ------------------------
void CartAle2Solver::init_sedov(double rho0, double p_amb,
                                double E0, double r_exp) {
    std::vector<double> h_X(nnode), h_Y(nnode);
    CUDA_CHECK(cudaMemcpy(h_X.data(), d_X, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Y.data(), d_Y, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_Vol(ncell);
    CUDA_CHECK(cudaMemcpy(h_Vol.data(), d_Vol, ncell*sizeof(double), cudaMemcpyDeviceToHost));
    double Lx = g_Lx, Ly = g_Ly;
    double xc_dom = 0.5 * Lx, yc_dom = 0.5 * Ly;
    double r_exp2 = r_exp * r_exp;
    // Sum volume of cells inside r_exp so E0 deposits conservatively.
    double V_hot = 0.0;
    std::vector<double> h_Xc(ncell), h_Yc(ncell);
    for (int ic = 0; ic < nx; ++ic)
        for (int jc = 0; jc < ny; ++jc) {
            int flat = ic*ny + jc;
            int I[4] = { ic*nnode_y + jc, (ic+1)*nnode_y + jc,
                         (ic+1)*nnode_y + (jc+1), ic*nnode_y + (jc+1) };
            double Xc = 0.25 * (h_X[I[0]] + h_X[I[1]] + h_X[I[2]] + h_X[I[3]]);
            double Yc = 0.25 * (h_Y[I[0]] + h_Y[I[1]] + h_Y[I[2]] + h_Y[I[3]]);
            h_Xc[flat] = Xc; h_Yc[flat] = Yc;
            double dx = Xc - xc_dom, dy = Yc - yc_dom;
            if (dx*dx + dy*dy < r_exp2) V_hot += h_Vol[flat];
        }
    // e_hot chosen so Σ ρ·e·Vol over hot cells = E0.
    double e_hot = (V_hot > 0.0) ? (E0 / (rho0 * V_hot)) : 0.0;
    double e_amb = p_amb / ((gamma - 1.0) * rho0);
    std::vector<double> h_dm(ncell), h_e(ncell);
    int n_hot = 0;
    for (int c = 0; c < ncell; ++c) {
        double dx = h_Xc[c] - xc_dom, dy = h_Yc[c] - yc_dom;
        bool hot = (dx*dx + dy*dy < r_exp2);
        h_dm[c] = rho0 * h_Vol[c];
        h_e[c]  = hot ? e_hot : e_amb;
        if (hot) ++n_hot;
    }
    CUDA_CHECK(cudaMemcpy(d_dm,    h_dm.data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_e_int, h_e.data(),  ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_vX, 0, nnode*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_vY, 0, nnode*sizeof(double)));
    int B = 256;
    k_cale2_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nx, ny, bc_mode);
    g_y = 0.0;
    std::fprintf(stderr,
        "  CartAle2 Sedov IC: ρ0=%g, p_amb=%g, E0=%g, r_exp=%g "
        "(n_hot=%d, V_hot=%g, e_hot=%g)\n",
        rho0, p_amb, E0, r_exp, n_hot, V_hot, e_hot);
}

// ----- 2D Noh implosion ----------------------------------------
void CartAle2Solver::init_noh() {
    std::vector<double> h_X(nnode), h_Y(nnode);
    CUDA_CHECK(cudaMemcpy(h_X.data(), d_X, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Y.data(), d_Y, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_Vol(ncell);
    CUDA_CHECK(cudaMemcpy(h_Vol.data(), d_Vol, ncell*sizeof(double), cudaMemcpyDeviceToHost));
    double Lx = g_Lx, Ly = g_Ly;
    double xc_dom = 0.5 * Lx, yc_dom = 0.5 * Ly;
    const double rho0 = 1.0;
    const double p0   = 1.0e-6;
    double e0 = p0 / ((gamma - 1.0) * rho0);
    std::vector<double> h_dm(ncell, 0), h_e(ncell, 0);
    for (int c = 0; c < ncell; ++c) {
        h_dm[c] = rho0 * h_Vol[c];
        h_e[c]  = e0;
    }
    CUDA_CHECK(cudaMemcpy(d_dm,    h_dm.data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_e_int, h_e.data(),  ncell*sizeof(double), cudaMemcpyHostToDevice));
    // Nodes: v = −r̂ (unit inflow); centre node v=0.
    std::vector<double> h_vX(nnode, 0.0), h_vY(nnode, 0.0);
    for (int in = 0; in < nnode_x; ++in)
        for (int jn = 0; jn < nnode_y; ++jn) {
            int f = in * nnode_y + jn;
            double dx = h_X[f] - xc_dom, dy = h_Y[f] - yc_dom;
            double r  = std::sqrt(dx*dx + dy*dy);
            if (r < 1e-14) continue;
            h_vX[f] = -dx / r;
            h_vY[f] = -dy / r;
        }
    CUDA_CHECK(cudaMemcpy(d_vX, h_vX.data(), nnode*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vY, h_vY.data(), nnode*sizeof(double), cudaMemcpyHostToDevice));
    int B = 256;
    k_cale2_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nx, ny, bc_mode);
    g_y = 0.0;
    std::fprintf(stderr,
        "  CartAle2 Noh IC: ρ=1, p=%g, v=-r̂, γ=%g (expect ρ_post=16 at t=2)\n",
        p0, gamma);
}

// ----- Gresho stationary vortex --------------------------------
void CartAle2Solver::init_gresho() {
    std::vector<double> h_X(nnode), h_Y(nnode);
    CUDA_CHECK(cudaMemcpy(h_X.data(), d_X, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Y.data(), d_Y, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_Vol(ncell);
    CUDA_CHECK(cudaMemcpy(h_Vol.data(), d_Vol, ncell*sizeof(double), cudaMemcpyDeviceToHost));
    double Lx = g_Lx, Ly = g_Ly;
    double xc_dom = 0.5 * Lx, yc_dom = 0.5 * Ly;
    const double rho0 = 1.0;
    auto P_gresho = [](double r) {
        if (r < 0.2) return 5.0 + 12.5 * r * r;
        if (r < 0.4) return 9.0 + 12.5 * r * r - 20.0 * r + 4.0 * std::log(5.0 * r);
        return 3.0 + 4.0 * std::log(2.0);
    };
    std::vector<double> h_dm(ncell), h_e(ncell);
    for (int ic = 0; ic < nx; ++ic)
        for (int jc = 0; jc < ny; ++jc) {
            int flat = ic*ny + jc;
            int I[4] = { ic*nnode_y + jc, (ic+1)*nnode_y + jc,
                         (ic+1)*nnode_y + (jc+1), ic*nnode_y + (jc+1) };
            double Xc = 0.25 * (h_X[I[0]] + h_X[I[1]] + h_X[I[2]] + h_X[I[3]]);
            double Yc = 0.25 * (h_Y[I[0]] + h_Y[I[1]] + h_Y[I[2]] + h_Y[I[3]]);
            double dx = Xc - xc_dom, dy = Yc - yc_dom;
            double r  = std::sqrt(dx*dx + dy*dy);
            double P  = P_gresho(r);
            h_dm[flat] = rho0 * h_Vol[flat];
            h_e[flat]  = P / ((gamma - 1.0) * rho0);
        }
    CUDA_CHECK(cudaMemcpy(d_dm,    h_dm.data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_e_int, h_e.data(),  ncell*sizeof(double), cudaMemcpyHostToDevice));
    std::vector<double> h_vX(nnode, 0.0), h_vY(nnode, 0.0);
    auto vphi = [](double r) {
        if (r < 0.2) return 5.0 * r;
        if (r < 0.4) return 2.0 - 5.0 * r;
        return 0.0;
    };
    for (int in = 0; in < nnode_x; ++in)
        for (int jn = 0; jn < nnode_y; ++jn) {
            int f = in * nnode_y + jn;
            double dx = h_X[f] - xc_dom, dy = h_Y[f] - yc_dom;
            double r  = std::sqrt(dx*dx + dy*dy);
            if (r < 1e-14) continue;
            double vp = vphi(r);
            h_vX[f] = -vp * dy / r;
            h_vY[f] =  vp * dx / r;
        }
    CUDA_CHECK(cudaMemcpy(d_vX, h_vX.data(), nnode*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vY, h_vY.data(), nnode*sizeof(double), cudaMemcpyHostToDevice));
    int B = 256;
    k_cale2_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nx, ny, bc_mode);
    g_y = 0.0;
    std::fprintf(stderr,
        "  CartAle2 Gresho IC: ρ=1, γ=%g, centred at (%g,%g); v_max=1 at r=0.2\n",
        gamma, xc_dom, yc_dom);
}

// ----- Yee-Vinokur-Djomehri isentropic vortex ------------------
void CartAle2Solver::init_yee_vortex(double beta, double u_inf, double v_inf) {
    std::vector<double> h_X(nnode), h_Y(nnode);
    CUDA_CHECK(cudaMemcpy(h_X.data(), d_X, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Y.data(), d_Y, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_Vol(ncell);
    CUDA_CHECK(cudaMemcpy(h_Vol.data(), d_Vol, ncell*sizeof(double), cudaMemcpyDeviceToHost));
    double Lx = g_Lx, Ly = g_Ly;
    double xc_dom = 0.5 * Lx, yc_dom = 0.5 * Ly;
    const double gm1 = gamma - 1.0;
    const double coef = beta / (2.0 * M_PI);
    // T(r²) = 1 − (γ−1)β²/(8γπ²) · exp(1 − r²);
    // ρ = T^{1/(γ−1)}; P = ρ^γ → e = T/(γ−1) (ρ·e = P).
    auto compute_rho_e = [&](double x, double y, double& rho, double& e) {
        double dx = x - xc_dom, dy = y - yc_dom;
        double r2 = dx*dx + dy*dy;
        double T  = 1.0 - gm1 * beta * beta / (8.0 * gamma * M_PI * M_PI)
                        * std::exp(1.0 - r2);
        rho = std::pow(T, 1.0 / gm1);
        e   = T / gm1;
    };
    std::vector<double> h_dm(ncell), h_e(ncell);
    for (int ic = 0; ic < nx; ++ic)
        for (int jc = 0; jc < ny; ++jc) {
            int flat = ic*ny + jc;
            int I[4] = { ic*nnode_y + jc, (ic+1)*nnode_y + jc,
                         (ic+1)*nnode_y + (jc+1), ic*nnode_y + (jc+1) };
            double Xc = 0.25 * (h_X[I[0]] + h_X[I[1]] + h_X[I[2]] + h_X[I[3]]);
            double Yc = 0.25 * (h_Y[I[0]] + h_Y[I[1]] + h_Y[I[2]] + h_Y[I[3]]);
            double rho, e;
            compute_rho_e(Xc, Yc, rho, e);
            h_dm[flat] = rho * h_Vol[flat];
            h_e[flat]  = e;
        }
    CUDA_CHECK(cudaMemcpy(d_dm,    h_dm.data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_e_int, h_e.data(),  ncell*sizeof(double), cudaMemcpyHostToDevice));
    std::vector<double> h_vX(nnode), h_vY(nnode);
    for (int in = 0; in < nnode_x; ++in)
        for (int jn = 0; jn < nnode_y; ++jn) {
            int f = in * nnode_y + jn;
            double dx = h_X[f] - xc_dom, dy = h_Y[f] - yc_dom;
            double r2 = dx*dx + dy*dy;
            double fac = coef * std::exp(0.5 * (1.0 - r2));
            h_vX[f] = u_inf - fac * dy;
            h_vY[f] = v_inf + fac * dx;
        }
    CUDA_CHECK(cudaMemcpy(d_vX, h_vX.data(), nnode*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vY, h_vY.data(), nnode*sizeof(double), cudaMemcpyHostToDevice));
    int B = 256;
    k_cale2_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nx, ny, bc_mode);
    g_y = 0.0;
    std::fprintf(stderr,
        "  CartAle2 Yee vortex: β=%g, (u∞,v∞)=(%g,%g), γ=%g; periodic [%g,%g]×[%g,%g]\n",
        beta, u_inf, v_inf, gamma, 0.0, Lx, 0.0, Ly);
}

void CartAle2Solver::init_hse_polytrope(double rho_base, double g_val, double amp) {
    g_y = g_val;
    double Ly = g_Ly;
    double n = 1.0 / (gamma - 1.0);
    double K = (gamma - 1.0) / gamma * g_val * Ly * std::pow(rho_base, gamma - 1.0);

    std::vector<double> h_X(nnode), h_Y(nnode);
    CUDA_CHECK(cudaMemcpy(h_X.data(), d_X, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Y.data(), d_Y, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_Vol(ncell);
    CUDA_CHECK(cudaMemcpy(h_Vol.data(), d_Vol, ncell*sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> h_dm(ncell), h_e(ncell);
    double rho_floor = 1e-6 * rho_base;
    for (int ic = 0; ic < nx; ++ic)
        for (int jc = 0; jc < ny; ++jc) {
            int flat = ic*ny + jc;
            int I[4] = { ic*nnode_y + jc, (ic+1)*nnode_y + jc,
                         (ic+1)*nnode_y + (jc+1), ic*nnode_y + (jc+1) };
            double Yc = 0.25 * (h_Y[I[0]] + h_Y[I[1]] + h_Y[I[2]] + h_Y[I[3]]);
            double theta_val = std::max(1.0 - Yc / Ly, 1e-10);
            double rho = std::max(rho_base * std::pow(theta_val, n), rho_floor);
            double P   = K * std::pow(rho, gamma);
            double e   = P / ((gamma - 1.0) * rho);
            if (amp != 0.0) {
                double delta = amp * std::sin(M_PI * Yc / Ly);
                e *= (1.0 + gamma * delta) / (1.0 + delta);
            }
            h_dm[flat] = rho * h_Vol[flat];
            h_e[flat]  = e;
        }
    CUDA_CHECK(cudaMemcpy(d_dm,    h_dm.data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_e_int, h_e.data(),  ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_vX, 0, nnode*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_vY, 0, nnode*sizeof(double)));

    int B = 256;
    k_cale2_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nx, ny, bc_mode);

    std::fprintf(stderr,
        "  CartAle HSE polytrope: ρ_b=%g, g=%g, K=%g, y_top=%g, perturb=%g\n",
        rho_base, g_val, K, Ly, amp);
}

// ============================================================
// HSE polytrope + N Gaussian bubble overlays
// ============================================================
void CartAle2Solver::init_hse_bubbles(double rho_base, double g_val,
                                     const std::vector<Bubble>& bubbles) {
    g_y = g_val;
    double Ly = g_Ly;
    double n = 1.0 / (gamma - 1.0);
    double K = (gamma - 1.0) / gamma * g_val * Ly * std::pow(rho_base, gamma - 1.0);

    std::vector<double> h_X(nnode), h_Y(nnode);
    CUDA_CHECK(cudaMemcpy(h_X.data(), d_X, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Y.data(), d_Y, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_Vol(ncell);
    CUDA_CHECK(cudaMemcpy(h_Vol.data(), d_Vol, ncell*sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> h_dm(ncell), h_e(ncell);
    double rho_floor = 1e-6 * rho_base;
    for (int ic = 0; ic < nx; ++ic)
        for (int jc = 0; jc < ny; ++jc) {
            int flat = ic*ny + jc;
            int I[4] = { ic*nnode_y + jc, (ic+1)*nnode_y + jc,
                         (ic+1)*nnode_y + (jc+1), ic*nnode_y + (jc+1) };
            double Xc = 0.25 * (h_X[I[0]] + h_X[I[1]] + h_X[I[2]] + h_X[I[3]]);
            double Yc = 0.25 * (h_Y[I[0]] + h_Y[I[1]] + h_Y[I[2]] + h_Y[I[3]]);
            double theta_val = std::max(1.0 - Yc / Ly, 1e-10);
            double rho_hse = std::max(rho_base * std::pow(theta_val, n), rho_floor);
            double P_hse   = K * std::pow(rho_hse, gamma);

            double rho_mul = 1.0, P_mul = 1.0;
            for (const auto& b : bubbles) {
                double dx = Xc - b.xc, dy = Yc - b.yc;
                double bump = std::exp(-(dx*dx + dy*dy) / (b.rb * b.rb));
                rho_mul *= (1.0 + b.alpha * bump);
                P_mul   *= (1.0 + b.beta  * bump);
            }
            double rho = std::max(rho_hse * rho_mul, rho_floor);
            double P   = std::max(P_hse   * P_mul,   1e-30);
            double e   = P / ((gamma - 1.0) * rho);

            h_dm[flat] = rho * h_Vol[flat];
            h_e[flat]  = e;
        }
    CUDA_CHECK(cudaMemcpy(d_dm,    h_dm.data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_e_int, h_e.data(),  ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_vX, 0, nnode*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_vY, 0, nnode*sizeof(double)));

    int B = 256;
    k_cale2_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nx, ny, bc_mode);

    std::fprintf(stderr,
        "  CartAle HSE bubbles: ρ_b=%g, g=%g, N=%zu\n",
        rho_base, g_val, bubbles.size());
    for (size_t i = 0; i < bubbles.size(); ++i) {
        const auto& b = bubbles[i];
        std::fprintf(stderr,
            "    bubble[%zu]: center=(%g,%g), rb=%g, α=%g, β=%g\n",
            i, b.xc, b.yc, b.rb, b.alpha, b.beta);
    }
}

// ============================================================
// Plane-parallel stratified slab (from a MESA envelope strip) + small
// entropy seed at the bottom to trigger Rayleigh-Taylor-like overturning.
// Reads the file written by scripts/render/make_local_convection_slab.py.
// ============================================================
void CartAle2Solver::init_local_convection(const std::string& slab_file,
                                           double perturb_amp,
                                           int seed_k) {
    // Parse header + (ny_slab+1) face rows.
    std::FILE* fp = std::fopen(slab_file.c_str(), "r");
    if (!fp) {
        std::fprintf(stderr,
            "  init_local_convection: cannot open %s\n", slab_file.c_str());
        std::abort();
    }
    auto skip_comments = [&](char* buf, int cap) -> bool {
        while (std::fgets(buf, cap, fp)) {
            const char* s = buf;
            while (*s == ' ' || *s == '\t') ++s;
            if (*s == '#' || *s == '\0' || *s == '\n') continue;
            return true;
        }
        return false;
    };
    char line[512];
    if (!skip_comments(line, sizeof(line))) {
        std::fprintf(stderr, "  init_local_convection: header row missing\n");
        std::fclose(fp); std::abort();
    }
    double Ly_file, Lx_file, g_file, gamma_file;
    double rho_top, P_top, T_top, mu_file;
    if (std::sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf",
                    &Ly_file, &Lx_file, &g_file, &gamma_file,
                    &rho_top, &P_top, &T_top, &mu_file) != 8) {
        std::fprintf(stderr, "  init_local_convection: bad header line\n");
        std::fclose(fp); std::abort();
    }
    std::vector<double> ys, rhos, Ps;
    std::vector<double> gs_from_file;       // optional column 5: g(y)
    std::vector<double> qs_from_file;       // optional column 6: q̇(y) [erg/s/cm³]
    // We can't use skip_comments alone here because of an optional
    // trailing "# g_profile" section.  Do manual parsing: read all
    // non-comment lines; classify by token count.
    while (std::fgets(line, (int)sizeof(line), fp)) {
        const char* s = line;
        while (*s == ' ' || *s == '\t') ++s;
        if (*s == '#' || *s == '\0' || *s == '\n') continue;
        double y, r, p, T, g_val, q_val;
        int k = std::sscanf(line, "%lf %lf %lf %lf %lf %lf",
                            &y, &r, &p, &T, &g_val, &q_val);
        if (k >= 4) {
            ys.push_back(y); rhos.push_back(r); Ps.push_back(p);
            if (k >= 5) gs_from_file.push_back(g_val);
            if (k >= 6) qs_from_file.push_back(q_val);
        }
    }
    std::fclose(fp);
    int n_face = (int)ys.size();
    if (n_face < 2) {
        std::fprintf(stderr, "  init_local_convection: too few rows\n");
        std::abort();
    }
    // Sanity: slab Ly should match init() Ly (to ~1e-6).  We trust init().
    if (std::fabs(g_Ly - Ly_file) / Ly_file > 1e-4) {
        std::fprintf(stderr,
            "  [warn] init_local_convection: slab Ly=%g vs init Ly=%g — "
            "using init Ly and rescaling.\n", Ly_file, g_Ly);
    }

    g_y = g_file;
    // Slab uses ideal γ = 5/3; cart_ale2's `gamma` was fixed at init().
    if (std::fabs(gamma - gamma_file) > 1e-6) {
        std::fprintf(stderr,
            "  [warn] init_local_convection: slab γ=%g vs solver γ=%g\n",
            gamma_file, gamma);
    }

    std::vector<double> h_X(nnode), h_Y(nnode);
    CUDA_CHECK(cudaMemcpy(h_X.data(), d_X, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Y.data(), d_Y, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_Vol(ncell);
    CUDA_CHECK(cudaMemcpy(h_Vol.data(), d_Vol, ncell*sizeof(double), cudaMemcpyDeviceToHost));

    // Helper: log-interpolate (ρ, P) at a given y using the slab data.
    auto slab_lookup = [&](double y, double& rho_out, double& P_out) {
        if (y <= ys.front()) { rho_out = rhos.front(); P_out = Ps.front(); return; }
        if (y >= ys.back())  { rho_out = rhos.back();  P_out = Ps.back();  return; }
        int lo = 0, hi = n_face - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) / 2;
            if (ys[mid] <= y) lo = mid; else hi = mid;
        }
        double t = (y - ys[lo]) / (ys[hi] - ys[lo]);
        rho_out = std::exp((1.0 - t) * std::log(rhos[lo]) + t * std::log(rhos[hi]));
        P_out   = std::exp((1.0 - t) * std::log(Ps[lo])   + t * std::log(Ps[hi]));
    };

    std::vector<double> h_dm(ncell), h_e(ncell);
    std::vector<double> h_e_ref_y(ny, 0.0);
    double Lx_box = g_Lx;
    for (int ic = 0; ic < nx; ++ic)
        for (int jc = 0; jc < ny; ++jc) {
            int flat = ic*ny + jc;
            int I[4] = { ic*nnode_y + jc, (ic+1)*nnode_y + jc,
                         (ic+1)*nnode_y + (jc+1), ic*nnode_y + (jc+1) };
            double Xc = 0.25 * (h_X[I[0]] + h_X[I[1]] + h_X[I[2]] + h_X[I[3]]);
            double Yc = 0.25 * (h_Y[I[0]] + h_Y[I[1]] + h_Y[I[2]] + h_Y[I[3]]);
            double rho_hse, P_hse;
            slab_lookup(Yc, rho_hse, P_hse);
            if (ic == 0)
                h_e_ref_y[jc] = P_hse / ((gamma - 1.0) * rho_hse);
            // Bottom 10 % in y: add a sin(k·2π·x/Lx) entropy bump so the
            // slab doesn't sit in perfect HSE forever.  Bump is δs/s, so
            // we hold P and perturb ρ by δρ/ρ = -δs/(γ·s) · P_exp... simpler:
            // perturb ρ directly by -δε·sin(...), keeping P fixed (this is
            // an entropy perturbation because s = P/ρ^γ rises when ρ falls).
            double env = 1.0;
            double y_decay = 0.1 * g_Ly;
            if (Yc < y_decay) {
                env = std::exp(-Yc / (0.3 * y_decay));
            } else {
                env = 0.0;
            }
            double phase = 2.0 * M_PI * seed_k * Xc / Lx_box;
            double d_rho = -perturb_amp * env * std::sin(phase);
            double rho = rho_hse * (1.0 + d_rho);
            double P   = P_hse;
            double e   = P / ((gamma - 1.0) * rho);
            h_dm[flat] = rho * h_Vol[flat];
            h_e[flat]  = e;
        }
    CUDA_CHECK(cudaMemcpy(d_dm,    h_dm.data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_e_int, h_e.data(),  ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_vX, 0, nnode*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_vY, 0, nnode*sizeof(double)));

    int B = 256;
    k_cale2_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nx, ny, bc_mode);

    // Stash per-row e_ref(y) so Newton cooling can relax toward HSE
    // whenever the caller sets tau_cool > 0.
    alloc_cooling_ref(h_e_ref_y);

    // Optional variable gravity g(y): interpolate slab column 5 (if present)
    // to node rows using the Y_node spacing and call configure_variable_gravity.
    if (!gs_from_file.empty() && (int)gs_from_file.size() == n_face) {
        std::vector<double> g_node(nnode_y, 0.0);
        double dy_node = g_Ly / (double)ny;
        // Build interp from slab (ys, gs_from_file) — same monotone table.
        auto g_lookup = [&](double y) -> double {
            if (y <= ys.front()) return gs_from_file.front();
            if (y >= ys.back())  return gs_from_file.back();
            int lo = 0, hi = n_face - 1;
            while (hi - lo > 1) {
                int mid = (lo + hi) / 2;
                if (ys[mid] <= y) lo = mid; else hi = mid;
            }
            double t = (y - ys[lo]) / (ys[hi] - ys[lo]);
            return (1.0 - t) * gs_from_file[lo] + t * gs_from_file[hi];
        };
        for (int jn = 0; jn < nnode_y; ++jn)
            g_node[jn] = g_lookup(jn * dy_node);
        configure_variable_gravity(g_node);
    }

    // Optional q̇(y) heating profile: interpolate slab column 6 onto cell rows.
    if (!qs_from_file.empty() && (int)qs_from_file.size() == n_face) {
        std::vector<double> q_cell(ny, 0.0);
        double dy_cell = g_Ly / (double)ny;
        auto q_lookup = [&](double y) -> double {
            if (y <= ys.front()) return qs_from_file.front();
            if (y >= ys.back())  return qs_from_file.back();
            int lo = 0, hi = n_face - 1;
            while (hi - lo > 1) {
                int mid = (lo + hi) / 2;
                if (ys[mid] <= y) lo = mid; else hi = mid;
            }
            double t = (y - ys[lo]) / (ys[hi] - ys[lo]);
            return (1.0 - t) * qs_from_file[lo] + t * qs_from_file[hi];
        };
        for (int jc = 0; jc < ny; ++jc)
            q_cell[jc] = q_lookup((jc + 0.5) * dy_cell);
        configure_heating_profile(q_cell, /*cool_top_frac=*/1.0);
    }

    double cs_bot = std::sqrt(gamma * Ps.front() / rhos.front());
    double cs_top = std::sqrt(gamma * Ps.back()  / rhos.back());
    std::fprintf(stderr,
        "  CartAle2 local_convection: slab=%s\n"
        "    Ly=%.3e  Lx=%.3e  g=%.3e  γ=%.3f  (top ρ=%.3e, P=%.3e, T=%.3e)\n"
        "    c_s top=%.3e bot=%.3e,  τ_dyn=Ly/c_s_top=%.3e  perturb=%.3g @ k=%d\n",
        slab_file.c_str(), g_Ly, g_Lx, g_y, gamma, rho_top, P_top, T_top,
        cs_top, cs_bot, g_Ly / cs_top, perturb_amp, seed_k);
    if (d_gy_node != nullptr)
        std::fprintf(stderr, "    Variable gravity g(y) enabled (%d nodes)\n", nnode_y);
    if (!qs_from_file.empty())
        std::fprintf(stderr, "    q̇(y) profile from slab file (%zu entries) loaded\n",
                     qs_from_file.size());
}

// ============================================================
// Andrassy 2022 IC: same slab format as init_local_convection, but the
// density perturbation follows Andrassy Eq. 6 exactly instead of the
// init_local_convection's simple k-mode exp-envelope seed.
//
// Eq. 6 (2D projection, z-factor dropped):
//   δρ(x,y)/ρ₀(y) = Δ · [q̇(y)/q̇₀] · [sin(3π·x') + cos(π·x')]
// with x' = 2x/Lx - 1 ∈ [-1, 1] per Andrassy's "x ∈ [-1, 1]" convention.
// q̇(y)/q̇₀ auto-envelopes the perturbation to the heating layer.
// ============================================================
void CartAle2Solver::init_andrassy2022(const std::string& slab_file,
                                       double delta_rho_amp,
                                       int noise_seed,
                                       double noise_amp) {
    // Parse slab file — reuse the same 6-col reader from init_local_convection.
    std::FILE* fp = std::fopen(slab_file.c_str(), "r");
    if (!fp) {
        std::fprintf(stderr,
            "  init_andrassy2022: cannot open %s\n", slab_file.c_str());
        std::abort();
    }
    auto skip_comments = [&](char* buf, int cap) -> bool {
        while (std::fgets(buf, cap, fp)) {
            const char* s = buf;
            while (*s == ' ' || *s == '\t') ++s;
            if (*s == '#' || *s == '\0' || *s == '\n') continue;
            return true;
        }
        return false;
    };
    char line[512];
    if (!skip_comments(line, sizeof(line))) {
        std::fprintf(stderr, "  init_andrassy2022: header missing\n");
        std::fclose(fp); std::abort();
    }
    double Ly_f, Lx_f, g_f, gam_f, rho_top, P_top, T_top, mu_f;
    if (std::sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf",
                    &Ly_f, &Lx_f, &g_f, &gam_f,
                    &rho_top, &P_top, &T_top, &mu_f) != 8) {
        std::fprintf(stderr, "  init_andrassy2022: bad header\n");
        std::fclose(fp); std::abort();
    }
    std::vector<double> ys, rhos, Ps;
    std::vector<double> gs_file, qs_file;
    while (std::fgets(line, (int)sizeof(line), fp)) {
        const char* s = line;
        while (*s == ' ' || *s == '\t') ++s;
        if (*s == '#' || *s == '\0' || *s == '\n') continue;
        double y, r, p, T, gg, qq;
        int k = std::sscanf(line, "%lf %lf %lf %lf %lf %lf",
                            &y, &r, &p, &T, &gg, &qq);
        if (k >= 4) {
            ys.push_back(y); rhos.push_back(r); Ps.push_back(p);
            if (k >= 5) gs_file.push_back(gg);
            if (k >= 6) qs_file.push_back(qq);
        }
    }
    std::fclose(fp);
    int n_face = (int)ys.size();
    if (n_face < 2) {
        std::fprintf(stderr, "  init_andrassy2022: too few rows\n");
        std::abort();
    }
    if (qs_file.empty()) {
        std::fprintf(stderr,
            "  init_andrassy2022: slab missing q̇ column (need 6-col format)\n");
        std::abort();
    }

    g_y = g_f;
    if (std::fabs(gamma - gam_f) > 1e-6)
        std::fprintf(stderr,
            "  [warn] init_andrassy2022: slab γ=%g vs solver γ=%g\n", gam_f, gamma);

    // Node coord download for cell centers.
    std::vector<double> h_X(nnode), h_Y(nnode);
    CUDA_CHECK(cudaMemcpy(h_X.data(), d_X, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Y.data(), d_Y, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_Vol(ncell);
    CUDA_CHECK(cudaMemcpy(h_Vol.data(), d_Vol, ncell*sizeof(double), cudaMemcpyDeviceToHost));

    // Monotone interp helpers on the slab.
    auto interp = [&](const std::vector<double>& vals, double y) -> double {
        if (y <= ys.front()) return vals.front();
        if (y >= ys.back())  return vals.back();
        int lo = 0, hi = n_face - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) / 2;
            if (ys[mid] <= y) lo = mid; else hi = mid;
        }
        double t = (y - ys[lo]) / (ys[hi] - ys[lo]);
        return (1.0 - t) * vals[lo] + t * vals[hi];
    };
    double q0_max = 0.0;
    for (double q : qs_file) q0_max = std::max(q0_max, q);
    if (q0_max <= 0.0) q0_max = 1.0;  // fallback; shouldn't happen

    std::vector<double> h_dm(ncell), h_e(ncell);
    std::vector<double> h_e_ref_y(ny, 0.0);
    double Lx_box = g_Lx;
    // Deterministic PRNG for ensemble noise: splitmix64 hash keyed by
    // (seed, ic, jc) so repeat runs with same seed are bit-identical.
    auto hash_rand = [&](int seed, int ic, int jc) -> double {
        uint64_t s = ((uint64_t)(uint32_t)seed * 0x9E3779B97F4A7C15ULL)
                   ^ ((uint64_t)(uint32_t)ic  * 0xD1B54A32D192ED03ULL)
                   ^ ((uint64_t)(uint32_t)jc  * 0xAEF17502108EF2D9ULL);
        s ^= s >> 30; s *= 0xBF58476D1CE4E5B9ULL;
        s ^= s >> 27; s *= 0x94D049BB133111EBULL;
        s ^= s >> 31;
        return ((double)(s & 0x7FFFFFFFFFFFFFFFULL)) /
               (double)0x7FFFFFFFFFFFFFFFULL * 2.0 - 1.0;
    };
    for (int ic = 0; ic < nx; ++ic)
        for (int jc = 0; jc < ny; ++jc) {
            int flat = ic*ny + jc;
            int I[4] = { ic*nnode_y + jc, (ic+1)*nnode_y + jc,
                         (ic+1)*nnode_y + (jc+1), ic*nnode_y + (jc+1) };
            double Xc = 0.25 * (h_X[I[0]] + h_X[I[1]] + h_X[I[2]] + h_X[I[3]]);
            double Yc = 0.25 * (h_Y[I[0]] + h_Y[I[1]] + h_Y[I[2]] + h_Y[I[3]]);
            double rho_hse = interp(rhos, Yc);
            double P_hse   = interp(Ps,   Yc);
            double q_env   = interp(qs_file, Yc) / q0_max;  // ∈ [0, 1]
            if (ic == 0)
                h_e_ref_y[jc] = P_hse / ((gamma - 1.0) * rho_hse);

            // Andrassy Eq. 6 x-factor using x' = 2x/Lx - 1 ∈ [-1, 1].
            double xp = 2.0 * Xc / Lx_box - 1.0;
            double x_factor = std::sin(3.0 * M_PI * xp) + std::cos(M_PI * xp);
            // Density perturbation.  Pressure untouched (isobaric seed, same
            // convention as Andrassy and our original init_local_convection).
            double d_rho_rel = delta_rho_amp * q_env * x_factor;
            // Ensemble noise: add uniform [-1, 1]·noise_amp·q_env on top.
            // Gated by noise_seed >= 0 so paper-exact IC (seed=-1) stays
            // bit-identical.
            if (noise_seed >= 0 && noise_amp > 0.0) {
                d_rho_rel += noise_amp * q_env * hash_rand(noise_seed, ic, jc);
            }
            double rho = rho_hse * (1.0 + d_rho_rel);
            double P   = P_hse;
            double e   = P / ((gamma - 1.0) * rho);
            h_dm[flat] = rho * h_Vol[flat];
            h_e[flat]  = e;
        }
    CUDA_CHECK(cudaMemcpy(d_dm,    h_dm.data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_e_int, h_e.data(),  ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_vX, 0, nnode*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_vY, 0, nnode*sizeof(double)));

    int B = 256;
    k_cale2_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nx, ny, bc_mode);

    // HSE reference for optional Newton cooling (not used by Andrassy 2022
    // spec but we upload so τ_cool > 0 still works if the user flips it).
    alloc_cooling_ref(h_e_ref_y);

    // Variable gravity from slab col 5.
    if (!gs_file.empty() && (int)gs_file.size() == n_face) {
        std::vector<double> g_node(nnode_y, 0.0);
        double dy_node = g_Ly / (double)ny;
        for (int jn = 0; jn < nnode_y; ++jn)
            g_node[jn] = interp(gs_file, jn * dy_node);
        configure_variable_gravity(g_node);
    }

    // Heating profile from slab col 6.
    std::vector<double> q_cell(ny, 0.0);
    double dy_cell = g_Ly / (double)ny;
    for (int jc = 0; jc < ny; ++jc)
        q_cell[jc] = interp(qs_file, (jc + 0.5) * dy_cell);
    configure_heating_profile(q_cell, /*cool_top_frac=*/1.0);

    // Passive species tracer X (Andrassy Eq. 3 μ₁ mass fraction).
    // Paper η₁(y) ramps on y_paper ∈ [2 − 1/16, 2 + 1/16].  In slab coords
    // (y_slab = y_paper − 1) that's y_slab ∈ [15/16, 17/16].
    init_tracer_ramp(15.0 / 16.0, 17.0 / 16.0);

    double cs_bot = std::sqrt(gamma * Ps.front() / rhos.front());
    double cs_top = std::sqrt(gamma * Ps.back()  / rhos.back());
    std::fprintf(stderr,
        "  CartAle2 Andrassy 2022 IC: slab=%s\n"
        "    Ly=%.3e  Lx=%.3e  γ=%.3f  (top ρ=%.3e, P=%.3e, T=%.3e)\n"
        "    c_s top=%.3e bot=%.3e,  τ_sc=Ly/c_s_top=%.3e\n"
        "    δρ/ρ Eq. 6 amplitude=%.3g (paper: 5e-5)\n",
        slab_file.c_str(), g_Ly, g_Lx, gamma, rho_top, P_top, T_top,
        cs_top, cs_bot, g_Ly / cs_top, delta_rho_amp);
}

// ============================================================
// Classic KH shear: two horizontal bands with opposite vx, isobaric,
// zero gravity. Gaussian perturbation in vy near each interface seeds
// the instability.
// ============================================================
void CartAle2Solver::init_kh_shear(double rho_light, double rho_heavy,
                                  double P0, double vshear,
                                  double amp, int k) {
    g_y = 0.0;
    double Lx = g_Lx, Ly = g_Ly;

    std::vector<double> h_X(nnode), h_Y(nnode);
    CUDA_CHECK(cudaMemcpy(h_X.data(), d_X, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Y.data(), d_Y, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_Vol(ncell);
    CUDA_CHECK(cudaMemcpy(h_Vol.data(), d_Vol, ncell*sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> h_dm(ncell), h_e(ncell);
    // Smooth the ρ and vx jump across a few cells so AV doesn't spike
    // at the interface. Transition thickness δ = 2·h in each direction.
    double dy = Ly / ny;
    double delta = 2.0 * dy;
    double y_low  = 0.25 * Ly;
    double y_high = 0.75 * Ly;
    for (int ic = 0; ic < nx; ++ic)
        for (int jc = 0; jc < ny; ++jc) {
            int flat = ic*ny + jc;
            int I[4] = { ic*nnode_y + jc, (ic+1)*nnode_y + jc,
                         (ic+1)*nnode_y + (jc+1), ic*nnode_y + (jc+1) };
            double Yc = 0.25 * (h_Y[I[0]] + h_Y[I[1]] + h_Y[I[2]] + h_Y[I[3]]);
            // tanh blend: inside=1, outside=0
            double band = 0.5 * (std::tanh((Yc - y_low)  / delta)
                               - std::tanh((Yc - y_high) / delta));
            double rho = rho_light + (rho_heavy - rho_light) * band;
            double e = P0 / ((gamma - 1.0) * rho);
            h_dm[flat] = rho * h_Vol[flat];
            h_e[flat]  = e;
        }
    CUDA_CHECK(cudaMemcpy(d_dm,    h_dm.data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_e_int, h_e.data(),  ncell*sizeof(double), cudaMemcpyHostToDevice));

    // Node velocities: tanh-blended vx, Gaussian-envelope sin(k·2π·x) vy seed.
    std::vector<double> h_vX(nnode, 0.0), h_vY(nnode, 0.0);
    double sigma = 0.05 * Ly;
    for (int in = 0; in < nnode_x; ++in)
        for (int jn = 0; jn < nnode_y; ++jn) {
            int f = in * nnode_y + jn;
            double x = h_X[f], y = h_Y[f];
            double band = 0.5 * (std::tanh((y - y_low)  / delta)
                               - std::tanh((y - y_high) / delta));
            // vx: +vshear inside the band, −vshear outside
            double vx = vshear * (2.0 * band - 1.0);
            // vy seed: amp·sin(k·2π·x/Lx) × Gaussian centered on each interface
            double gy1 = std::exp(-(y - y_low)*(y - y_low) / (sigma*sigma));
            double gy2 = std::exp(-(y - y_high)*(y - y_high) / (sigma*sigma));
            double vy = amp * std::sin(k * 2.0 * M_PI * x / Lx) * (gy1 + gy2);
            h_vX[f] = vx;
            h_vY[f] = vy;
        }
    CUDA_CHECK(cudaMemcpy(d_vX, h_vX.data(), nnode*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vY, h_vY.data(), nnode*sizeof(double), cudaMemcpyHostToDevice));

    int B = 256;
    k_cale2_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nx, ny, bc_mode);

    std::fprintf(stderr,
        "  CartAle KH shear: ρ_light=%g, ρ_heavy=%g, P0=%g, |vx|=%g, amp=%g, k=%d (g=0)\n",
        rho_light, rho_heavy, P0, vshear, amp, k);
}

// ============================================================
// Lecoanet (2015) canonical KH test — dual tanh shear layers,
// fully periodic in x and y, zero gravity. Mirrors Athena
// src/pgen/kh.cpp iprob=4 up to coordinate origin shift.
//
// Shear layer centres at y = ¼·Ly and ¾·Ly (zones z1, z2).
// Profiles (with a = thickness, σ = perturbation width):
//   ρ(y)  = 1 + ½·drho_rho0 · [tanh((y-z1)/a) - tanh((y-z2)/a)]
//   vx(y) = vflow · [tanh((y-z1)/a) - tanh((y-z2)/a) - 1]
//   vy(x,y) = -amp·ave_sin(x) · [exp(-(y-z1)²/σ²) + exp(-(y-z2)²/σ²)]
// where ave_sin averages sin(2π·x/Lx) with its x-shifted counterpart
// to suppress FP-asymmetry over long integrations (Athena kh.cpp:348).
// Pressure uniform P0 ⇒ e_int = P0/((γ-1)·ρ). g_y forced to 0.
// Recommended BC: x and y periodic (--bc-x periodic --bc-y periodic).
// ============================================================
void CartAle2Solver::init_kh_lecoanet(double vflow, double amp,
                                      double drho_rho0, double P0, int k) {
    g_y = 0.0;
    double Lx = g_Lx, Ly = g_Ly;

    std::vector<double> h_X(nnode), h_Y(nnode);
    CUDA_CHECK(cudaMemcpy(h_X.data(), d_X, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Y.data(), d_Y, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_Vol(ncell);
    CUDA_CHECK(cudaMemcpy(h_Vol.data(), d_Vol, ncell*sizeof(double), cudaMemcpyDeviceToHost));

    const double z1 = 0.25 * Ly;
    const double z2 = 0.75 * Ly;
    const double a_width = 0.05 * Ly;    // Athena uses 0.05
    const double sigma   = 0.2  * Ly;    // Athena uses 0.2
    const double x_mid   = 0.5  * Lx;    // symmetry axis for FP-symmetric sin

    std::vector<double> h_dm(ncell), h_e(ncell);
    for (int ic = 0; ic < nx; ++ic)
        for (int jc = 0; jc < ny; ++jc) {
            int flat = ic*ny + jc;
            int I[4] = { ic*nnode_y + jc, (ic+1)*nnode_y + jc,
                         (ic+1)*nnode_y + (jc+1), ic*nnode_y + (jc+1) };
            double Yc = 0.25 * (h_Y[I[0]] + h_Y[I[1]] + h_Y[I[2]] + h_Y[I[3]]);
            double dT = std::tanh((Yc - z1)/a_width) - std::tanh((Yc - z2)/a_width);
            double rho = 1.0 + 0.5 * drho_rho0 * dT;
            double e   = P0 / ((gamma - 1.0) * rho);
            h_dm[flat] = rho * h_Vol[flat];
            h_e[flat]  = e;
        }
    CUDA_CHECK(cudaMemcpy(d_dm,    h_dm.data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_e_int, h_e.data(),  ncell*sizeof(double), cudaMemcpyHostToDevice));

    // DIAGNOSTIC: force uniform vx to isolate whether the tanh(y) vx profile
    // is the source of |v| accumulation. Set FORCE_UNIFORM_VX=1 → vx=vflow
    // everywhere; otherwise normal Lecoanet dual-tanh profile.
    bool force_uniform = (std::getenv("FORCE_UNIFORM_VX") != nullptr);
    std::vector<double> h_vX(nnode, 0.0), h_vY(nnode, 0.0);
    for (int in = 0; in < nnode_x; ++in)
        for (int jn = 0; jn < nnode_y; ++jn) {
            int f = in * nnode_y + jn;
            double x = h_X[f], y = h_Y[f];
            double dT = std::tanh((y - z1)/a_width) - std::tanh((y - z2)/a_width);
            double vx = force_uniform ? vflow : vflow * (dT - 1.0);
            // FP-symmetric sin centred on x_mid: average sin at x and at shifted partner.
            double x_shift = x - x_mid;
            double phase = k * 2.0 * M_PI * x_shift / Lx;
            double ave_sin = std::sin(phase);
            double x_partner = (x_shift > 0.0)
                              ? (x_shift - 0.5 * Lx)
                              : (x_shift + 0.5 * Lx);
            ave_sin -= std::sin(k * 2.0 * M_PI * x_partner / Lx);
            ave_sin *= 0.5;
            double env = std::exp(-(y - z1)*(y - z1) / (sigma*sigma))
                       + std::exp(-(y - z2)*(y - z2) / (sigma*sigma));
            double vy = -amp * ave_sin * env;
            h_vX[f] = vx;
            h_vY[f] = vy;
        }
    CUDA_CHECK(cudaMemcpy(d_vX, h_vX.data(), nnode*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vY, h_vY.data(), nnode*sizeof(double), cudaMemcpyHostToDevice));

    int B = 256;
    k_cale2_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nx, ny, bc_mode);

    std::fprintf(stderr,
        "  CartAle2 KH Lecoanet: vflow=%g, amp=%g, drho_rho0=%g, P0=%g, "
        "k=%d, z1=%g, z2=%g, a=%g, σ=%g (g=0, requires periodic BC)\n",
        vflow, amp, drho_rho0, P0, k, z1, z2, a_width, sigma);
}

// ============================================================
// One ALE step
// ============================================================
double CartAle2Solver::step(double t, double t_end) {
    int B = 256;
    int BCell = (ncell + B - 1) / B;
    int BNode = (nnode + B - 1) / B;

    if (trace) trace->snapshot_pre_lag(*this);

    // --- Phase L: Lagrangian substep -------------------------
    // Mesh is always X0/Y0 at step entry (reset by previous step's Phase R),
    // so Vol ≡ Area0 and minheight is constant — both cached at init time.
    // Skipping the per-step geometry kernel saves one launch per step.
    // Exception: in pure-Lagrangian mode (remap_order == 0) Phase R is
    // skipped, so mesh drifts and Vol/minheight must be recomputed.
    if (remap_order == 0) {
        k_cale2_geometry<<<BCell, B>>>(d_X, d_Y, d_Vol, d_minheight, nx, ny);
    }
    const double* d_V_eos = (remap_order == 0) ? d_Vol : d_Area0;
    k_cale2_eos_and_q<<<BCell, B>>>(
        d_X, d_Y, d_vX, d_vY, d_dm, d_V_eos, d_Area0, d_e_int,
        d_rho, d_P, d_Q, d_cs, d_strain_rate,
        nx, ny, gamma, CQ_lin, CQ_quad, shear_aware_av);

    k_cale2_cfl<<<BCell, B>>>(
        d_minheight, d_cs, d_strain_rate, d_vX, d_vY,
        nx, ny, cfl, comp_dt_frac, d_dt_cell);
    double dt_min = gpu_reduce_min(d_dt_cell, d_reduce_buf, ncell);
    double dt = (dt_min > 0.0 && dt_min < 1e29) ? dt_min : 1e-12;
    if (dt < 1e-30) dt = 1e-12;
    if (t + dt > t_end) dt = t_end - t;

    k_cale2_zero<<<BNode, B>>>(d_FX, nnode);
    k_cale2_zero<<<BNode, B>>>(d_FY, nnode);
    k_cale2_node_forces<<<BCell, B>>>(d_X, d_Y, d_P, d_Q,
                                     d_FX, d_FY, d_FSX, d_FSY, nx, ny);
    if (d_gy_node != nullptr)
        k_cale2_add_gravity_var<<<BNode, B>>>(d_mnode, d_FY, d_gy_node, nnode_x, nnode_y);
    else if (g_y != 0.0)
        k_cale2_add_gravity<<<BNode, B>>>(d_mnode, d_FY, g_y, nnode);

    k_cale2_bc_reflective<<<BNode, B>>>(d_X0, d_Y0, d_X, d_Y, d_vX, d_vY,
                                       d_FX, d_FY, nnode_x, nnode_y, bc_mode);
    // Forces accumulated via cell-parallel atomicAdd — periodic duplicates
    // only received contributions from their local-side cells, so sum
    // across partners to recover the full physical force on each node.
    if (bc_mode) k_cale2_periodic_sync_node<<<BNode, B>>>(d_FX, d_FY,
                                                         nnode_x, nnode_y, bc_mode, /*mode=sum*/ 1);

    k_cale2_node_update<<<BNode, B>>>(d_X, d_Y, d_vX, d_vY, d_FX, d_FY,
                                     d_mnode, d_dX, d_dY, dt, nnode);

    k_cale2_bc_reflective<<<BNode, B>>>(d_X0, d_Y0, d_X, d_Y, d_vX, d_vY,
                                       d_FX, d_FY, nnode_x, nnode_y, bc_mode);
    if (bc_mode) {
        // State variables — average partners to keep duplicates bit-identical
        // (FP roundoff cancellation); sum would incorrectly double the value.
        k_cale2_periodic_sync_node<<<BNode, B>>>(d_vX, d_vY, nnode_x, nnode_y, bc_mode, /*mode=copy*/ 0);
        k_cale2_periodic_sync_node<<<BNode, B>>>(d_dX, d_dY, nnode_x, nnode_y, bc_mode, /*mode=copy*/ 0);
    }

    k_cale2_energy_update<<<BCell, B>>>(nx, ny, d_FSX, d_FSY,
                                       d_dX, d_dY, d_dm, d_e_int);

    if (trace) { trace->begin_step(t, step_count, dt); trace->snapshot_post_lag(*this); }

    // --- Phase M: Remap --------------------------------------
    // remap_order == 0 → pure Lagrangian: skip rezone + remap entirely,
    // so Phase R does NOT reset X to X0. Used for diagnostic comparison
    // (Gresho, Yee) where we want to isolate Lagrangian substep accuracy.
    // Mesh will drift and eventually tangle; only use for short runs.
    if (remap_order == 0) {
        // Recompute geometry for the drifted mesh so the next step sees
        // the correct Vol / minheight (Phase L normally assumes X=X0).
        k_cale2_geometry<<<BCell, B>>>(d_X, d_Y, d_Vol, d_minheight, nx, ny);
        // Node mass stays the same (no mass redistribution without remap).
        step_count++;
        dt_current = dt;
        if (step_count <= 10 || step_count % 500 == 0)
            std::fprintf(stderr,
                "  [cart_ale PURE-LAG] step %d  t=%.4e  dt=%.3e\n",
                step_count, t+dt, dt);
        return dt;
    }

    // Snapshot pre-remap node velocities AND node mass for Phase-M
    // compensation. The diagnostic KE uses the current (post-step)
    // m_node·v_node, so consecutive-step E conservation requires
    //   KE_before = Σ m_node_pre · ½ v_pre²    (matches last step's diag)
    //   KE_after  = Σ m_node_post · ½ v_post²  (matches this step's diag)
    // ΔKE = KE_before − KE_after is therefore exactly the diagnostic KE
    // loss across the step, and adding it back as IE makes E_diag invariant.
    // (Mixing m_post with v_pre would introduce a (m_post−m_pre)·v_pre²
    // bias that accumulates as ~1e-6 per step on Sod.)
    k_cale2_copy<<<BNode, B>>>(d_vX, d_vX_pre, nnode);
    k_cale2_copy<<<BNode, B>>>(d_vY, d_vY_pre, nnode);
    k_cale2_copy<<<BNode, B>>>(d_mnode, d_m_node_ref, nnode);

    // Snapshot cell-centered momentum from current node velocities BEFORE rezone.
    // Jensen #1 loss (node v → cell v averaging) is deposited in-place into
    // d_e_int for this cell — strictly local, no global reduction (P33 fix).
    k_cale2_cell_momentum<<<BCell, B>>>(d_vX, d_vY, d_dm,
                                        d_px_cell, d_py_cell,
                                        d_e_int,  // Jensen #1 deposit
                                        nx, ny);
    if (trace) trace->after_cell_mom(*this);

    // Vol was set to pre-Lagrangian geom; we need pre-Lagrangian donor volume.
    // In Eulerian rezone the "old" mesh for swept-remap is X0/Y0 (uniform),
    // so V_donor = Area0. That's what we pass in.
    k_cale2_remap_init<<<BCell, B>>>(d_dm, d_e_int, d_px_cell, d_py_cell,
                                    d_dm_new, d_ie_new, d_px_new, d_py_new, ncell);

    bool x_per = (bc_mode & 1) != 0;
    bool y_per = (bc_mode & 2) != 0;
    int n_east  = x_per ? nx * ny       : (nx - 1) * ny;
    int n_north = y_per ? nx * ny       : nx * (ny - 1);

    if (remap_order >= 2) {
        if (ppm_enabled && ppm_primitive) {
            // Primitive-space PPM: reconstruct (ρ, P, vx, vy) instead of
            // (ρ, ρe, ρvx, ρvy). Avoids the px=ρvx sign-flip overshoot
            // pathology at smooth shear interfaces (e.g. Lecoanet KH).
            // Semantically reuse the density buffers as primitive storage:
            //   d_rho_dens ← ρ, d_rhoE_dens ← P, d_pxd_dens ← vx, d_pyd_dens ← vy.
            k_cale2_cell_primitives<<<BCell, B>>>(
                d_dm, d_e_int, d_px_cell, d_py_cell, d_Area0,
                d_rho_dens, d_rhoE_dens, d_pxd_dens, d_pyd_dens, ncell, gamma);
            if (ppm_char) {
                k_cale2_ppm_reconstruct_char<<<BCell, B>>>(
                    d_rho_dens, d_rhoE_dens, d_pxd_dens, d_pyd_dens,
                    d_rho_xL,  d_rho_xR,  d_rho_yD,  d_rho_yU,
                    d_rhoE_xL, d_rhoE_xR, d_rhoE_yD, d_rhoE_yU,
                    d_pxd_xL,  d_pxd_xR,  d_pxd_yD,  d_pxd_yU,
                    d_pyd_xL,  d_pyd_xR,  d_pyd_yD,  d_pyd_yU,
                    nx, ny, bc_mode, ppm_cs_limiter, gamma);
            } else {
                k_cale2_ppm_reconstruct<<<BCell, B>>>(
                    d_rho_dens, d_rhoE_dens, d_pxd_dens, d_pyd_dens,
                    d_rho_xL,  d_rho_xR,  d_rho_yD,  d_rho_yU,
                    d_rhoE_xL, d_rhoE_xR, d_rhoE_yD, d_rhoE_yU,
                    d_pxd_xL,  d_pxd_xR,  d_pxd_yD,  d_pxd_yU,
                    d_pyd_xL,  d_pyd_xR,  d_pyd_yD,  d_pyd_yU,
                    nx, ny, bc_mode, ppm_cs_limiter);
            }
            if (n_east > 0) {
                int BE = (n_east + B - 1) / B;
                k_cale2_remap_east_ppm_prim<<<BE, B>>>(
                    d_X0, d_Y0, d_X, d_Y,
                    d_rho_dens, d_rhoE_dens, d_pxd_dens, d_pyd_dens,
                    d_rho_xL,  d_rho_xR,  d_rho_yD,  d_rho_yU,
                    d_rhoE_xL, d_rhoE_xR, d_rhoE_yD, d_rhoE_yU,
                    d_pxd_xL,  d_pxd_xR,  d_pxd_yD,  d_pxd_yU,
                    d_pyd_xL,  d_pyd_xR,  d_pyd_yD,  d_pyd_yU,
                    d_dm_new, d_ie_new, d_px_new, d_py_new,
                    nx, ny, dx_u, dy_u, gamma, bc_mode);
            }
            if (n_north > 0) {
                int BN = (n_north + B - 1) / B;
                k_cale2_remap_north_ppm_prim<<<BN, B>>>(
                    d_X0, d_Y0, d_X, d_Y,
                    d_rho_dens, d_rhoE_dens, d_pxd_dens, d_pyd_dens,
                    d_rho_xL,  d_rho_xR,  d_rho_yD,  d_rho_yU,
                    d_rhoE_xL, d_rhoE_xR, d_rhoE_yD, d_rhoE_yU,
                    d_pxd_xL,  d_pxd_xR,  d_pxd_yD,  d_pxd_yU,
                    d_pyd_xL,  d_pyd_xR,  d_pyd_yD,  d_pyd_yU,
                    d_dm_new, d_ie_new, d_px_new, d_py_new,
                    nx, ny, dx_u, dy_u, gamma, bc_mode);
            }
        } else if (ppm_enabled) {
            // Legacy conservative-variable PPM (retained for A/B comparison).
            k_cale2_cell_densities<<<BCell, B>>>(
                d_dm, d_e_int, d_px_cell, d_py_cell, d_Area0,
                d_rho_dens, d_rhoE_dens, d_pxd_dens, d_pyd_dens, ncell);
            k_cale2_ppm_reconstruct<<<BCell, B>>>(
                d_rho_dens, d_rhoE_dens, d_pxd_dens, d_pyd_dens,
                d_rho_xL,  d_rho_xR,  d_rho_yD,  d_rho_yU,
                d_rhoE_xL, d_rhoE_xR, d_rhoE_yD, d_rhoE_yU,
                d_pxd_xL,  d_pxd_xR,  d_pxd_yD,  d_pxd_yU,
                d_pyd_xL,  d_pyd_xR,  d_pyd_yD,  d_pyd_yU,
                nx, ny, bc_mode, ppm_cs_limiter);
            if (n_east > 0) {
                int BE = (n_east + B - 1) / B;
                k_cale2_remap_east_ppm<<<BE, B>>>(
                    d_X0, d_Y0, d_X, d_Y,
                    d_rho_dens, d_rhoE_dens, d_pxd_dens, d_pyd_dens,
                    d_rho_xL,  d_rho_xR,  d_rho_yD,  d_rho_yU,
                    d_rhoE_xL, d_rhoE_xR, d_rhoE_yD, d_rhoE_yU,
                    d_pxd_xL,  d_pxd_xR,  d_pxd_yD,  d_pxd_yU,
                    d_pyd_xL,  d_pyd_xR,  d_pyd_yD,  d_pyd_yU,
                    d_dm_new, d_ie_new, d_px_new, d_py_new,
                    nx, ny, dx_u, dy_u, bc_mode);
            }
            if (n_north > 0) {
                int BN = (n_north + B - 1) / B;
                k_cale2_remap_north_ppm<<<BN, B>>>(
                    d_X0, d_Y0, d_X, d_Y,
                    d_rho_dens, d_rhoE_dens, d_pxd_dens, d_pyd_dens,
                    d_rho_xL,  d_rho_xR,  d_rho_yD,  d_rho_yU,
                    d_rhoE_xL, d_rhoE_xR, d_rhoE_yD, d_rhoE_yU,
                    d_pxd_xL,  d_pxd_xR,  d_pxd_yD,  d_pxd_yU,
                    d_pyd_xL,  d_pyd_xR,  d_pyd_yD,  d_pyd_yU,
                    d_dm_new, d_ie_new, d_px_new, d_py_new,
                    nx, ny, dx_u, dy_u, bc_mode);
            }
        } else {
            // MUSCL path uses conservative densities — build them here.
            k_cale2_cell_densities<<<BCell, B>>>(
                d_dm, d_e_int, d_px_cell, d_py_cell, d_Area0,
                d_rho_dens, d_rhoE_dens, d_pxd_dens, d_pyd_dens, ncell);
            // MUSCL path (original).
            k_cale2_slopes_minmod<<<BCell, B>>>(
                d_rho_dens, d_rhoE_dens, d_pxd_dens, d_pyd_dens,
                d_rho_sx,  d_rho_sy,
                d_rhoE_sx, d_rhoE_sy,
                d_pxd_sx,  d_pxd_sy,
                d_pyd_sx,  d_pyd_sy,
                nx, ny, dx_u, dy_u, remap_limiter, bc_mode);
            if (n_east > 0) {
                int BE = (n_east + B - 1) / B;
                k_cale2_remap_east_2nd<<<BE, B>>>(
                    d_X0, d_Y0, d_X, d_Y,
                    d_rho_dens, d_rhoE_dens, d_pxd_dens, d_pyd_dens,
                    d_rho_sx,  d_rho_sy,
                    d_rhoE_sx, d_rhoE_sy,
                    d_pxd_sx,  d_pxd_sy,
                    d_pyd_sx,  d_pyd_sy,
                    d_dm_new, d_ie_new, d_px_new, d_py_new,
                    nx, ny, dx_u, dy_u, bc_mode);
            }
            if (n_north > 0) {
                int BN = (n_north + B - 1) / B;
                k_cale2_remap_north_2nd<<<BN, B>>>(
                    d_X0, d_Y0, d_X, d_Y,
                    d_rho_dens, d_rhoE_dens, d_pxd_dens, d_pyd_dens,
                    d_rho_sx,  d_rho_sy,
                    d_rhoE_sx, d_rhoE_sy,
                    d_pxd_sx,  d_pxd_sy,
                    d_pyd_sx,  d_pyd_sy,
                    d_dm_new, d_ie_new, d_px_new, d_py_new,
                    nx, ny, dx_u, dy_u, bc_mode);
            }
        }
    } else {
        if (n_east > 0) {
            int BE = (n_east + B - 1) / B;
            k_cale2_remap_east<<<BE, B>>>(d_X0, d_Y0, d_X, d_Y,
                                         d_dm, d_e_int, d_px_cell, d_py_cell, d_Area0,
                                         d_dm_new, d_ie_new, d_px_new, d_py_new, nx, ny, bc_mode);
        }
        if (n_north > 0) {
            int BN = (n_north + B - 1) / B;
            k_cale2_remap_north<<<BN, B>>>(d_X0, d_Y0, d_X, d_Y,
                                          d_dm, d_e_int, d_px_cell, d_py_cell, d_Area0,
                                          d_dm_new, d_ie_new, d_px_new, d_py_new, nx, ny, bc_mode);
        }
    }

    // Finalize dm, e_int from accumulators + per-cell Jensen #2 deposit (P33).
    // Pass pre-remap (dm, e_int, px_cell, py_cell) AND post-remap new state
    // so the kernel can compute fixed-mass KE difference in place.
    if (trace) trace->after_remap(*this);  // note: uses dm_new/px_new BEFORE finalize

    k_cale2_remap_finalize_cells<<<BCell, B>>>(d_dm, d_e_int, d_px_cell, d_py_cell,
                                               d_dm_new, d_ie_new, d_px_new, d_py_new,
                                               d_dm, d_e_int, ncell);

    // Passive species tracer X: conservative swept flux of species mass mX=X·dm.
    // Order matches hydro remap_order:  1 = donor-cell, ≥2 = MUSCL minmod-limited
    // linear reconstruction at swept-region centroid (Kucharik-Shashkov 2012).
    // MUSCL cuts tracer numerical diffusion by ~10×, essential for M_e diagnostic.
    if (tracer_enabled && d_mX != nullptr) {
        k_cale2_species_init_scratch<<<BCell, B>>>(d_mX, d_mX_new, ncell);
        if (remap_order >= 2) {
            k_cale2_species_density<<<BCell, B>>>(d_mX, d_Area0, d_mXd, ncell);
            k_cale2_species_slopes<<<BCell, B>>>(d_mXd, d_mXd_sx, d_mXd_sy,
                nx, ny, dx_u, dy_u, remap_limiter, bc_mode);
            if (n_east > 0) {
                int BE = (n_east + B - 1) / B;
                k_cale2_species_remap_east_2nd<<<BE, B>>>(
                    d_X0, d_Y0, d_X, d_Y,
                    d_mXd, d_mXd_sx, d_mXd_sy, d_mX_new,
                    nx, ny, dx_u, dy_u, bc_mode);
            }
            if (n_north > 0) {
                int BN = (n_north + B - 1) / B;
                k_cale2_species_remap_north_2nd<<<BN, B>>>(
                    d_X0, d_Y0, d_X, d_Y,
                    d_mXd, d_mXd_sx, d_mXd_sy, d_mX_new,
                    nx, ny, dx_u, dy_u, bc_mode);
            }
        } else {
            if (n_east > 0) {
                int BE = (n_east + B - 1) / B;
                k_cale2_species_remap_east<<<BE, B>>>(
                    d_X0, d_Y0, d_X, d_Y, d_mX, d_Area0, d_mX_new,
                    nx, ny, bc_mode);
            }
            if (n_north > 0) {
                int BN = (n_north + B - 1) / B;
                k_cale2_species_remap_north<<<BN, B>>>(
                    d_X0, d_Y0, d_X, d_Y, d_mX, d_Area0, d_mX_new,
                    nx, ny, bc_mode);
            }
        }
        k_cale2_species_finalize<<<BCell, B>>>(d_mX_new, d_dm, d_mX, d_species_X, ncell);
    }

    // Rebuild node velocities from remapped momentum and mass.
    // 2nd-order MUSCL-style: each node samples 4 corner-extrapolated
    // cell velocities (linear reconstruction with same limiter as hydro
    // remap). Symmetric corner offsets cancel slope contributions per
    // cell → momentum-conservative, but v-field is preserved exactly
    // on linear/rotational flows (fixes Gresho KE smoothing).  Falls
    // back to 1st-order mass-weighted average when remap_order < 2.
    // Rebuild node velocity from remapped cell momenta (mass-weighted avg →
    // Jensen-convex, necessarily loses KE). KE accounting happens in a
    // dedicated kernel below, so rebuild itself stays pure.
    if (rebuild_order >= 1 && remap_order >= 2) {
        k_cale2_cell_velocity<<<BCell, B>>>(d_px_new, d_py_new, d_dm_new,
                                           d_vxc, d_vyc, ncell);
        k_cale2_velocity_slopes<<<BCell, B>>>(d_vxc, d_vyc,
                                              d_vxc_sx, d_vxc_sy,
                                              d_vyc_sx, d_vyc_sy,
                                              nx, ny, dx_u, dy_u,
                                              remap_limiter, bc_mode);
        k_cale2_rebuild_node_v_2nd<<<BNode, B>>>(d_vxc, d_vyc,
                                                 d_vxc_sx, d_vxc_sy,
                                                 d_vyc_sx, d_vyc_sy,
                                                 d_dm_new, d_vX, d_vY,
                                                 nullptr, nullptr,  // unused legacy params
                                                 nx, ny, dx_u, dy_u, bc_mode);
    } else {
        k_cale2_rebuild_node_v<<<BNode, B>>>(d_px_new, d_py_new, d_dm_new,
                                            d_vX, d_vY,
                                            nullptr, nullptr,
                                            nx, ny, bc_mode);
    }
    k_cale2_bc_velocity<<<BNode, B>>>(d_vX, d_vY, nnode_x, nnode_y, bc_mode);
    if (bc_mode) k_cale2_periodic_sync_node<<<BNode, B>>>(d_vX, d_vY,
                                                         nnode_x, nnode_y, bc_mode, /*mode=copy*/ 0);

    // --- Phase R: snap mesh back to uniform ------------------
    k_cale2_reset_mesh<<<BNode, B>>>(d_X0, d_Y0, d_X, d_Y, nnode);

    // Refresh node mass (dm redistributes a bit each step).
    k_cale2_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nx, ny, bc_mode);

    // (A) Jensen #3: cell v → node v averaging in rebuild. Compute entirely
    //     from post-remap state (self-consistent, no old/new-mass mixing).
    //     Deposit equally to N ≤ 4 adjacent cells' IE via e_int_incr.
    //     Together with Jensen #1 (in cell_momentum) and Jensen #2 (in
    //     remap_east/north edge kernels), this covers all three mass-
    //     averaging operations in a full step — E is conserved exactly
    //     to machine precision, locally.
    k_cale2_compute_node_dKE<<<BNode, B>>>(d_vX, d_vY,
                                           d_mnode,
                                           d_dm_new,
                                           d_px_new, d_py_new,
                                           d_e_int_incr,
                                           d_dKE_node_local,
                                           nx, ny, bc_mode);
    k_cale2_apply_ie_incr<<<BCell, B>>>(d_e_int, d_e_int_incr, ncell);

    if (trace) trace->after_rebuild(*this);

    // (B) Diagnostic KE snapshots — consistent-mass caliper, independent of
    //     compensation path. Currently written but not read in the hot loop;
    //     left available for step-by-step E-budget debugging scripts.
    k_cale2_node_KE<<<BNode, B>>>(d_vX_pre, d_vY_pre, d_m_node_ref,
                                  nnode_x, nnode_y, bc_mode, d_KE_node_snap_pre);
    k_cale2_node_KE<<<BNode, B>>>(d_vX,     d_vY,     d_m_node_ref,
                                  nnode_x, nnode_y, bc_mode, d_KE_node_snap_post);

    // Optional Newton cooling toward the IC stratification (only applied
    // if alloc_cooling_ref was called and tau_cool > 0).
    apply_cooling(dt);
    if (trace) { trace->after_heating(*this); trace->end_step(*this); }

    step_count++;
    dt_current = dt;
    if (step_count <= 10 || step_count % 500 == 0)
        std::fprintf(stderr, "  [cart_ale] step %d  t=%.4e  dt=%.3e\n",
                     step_count, t+dt, dt);
    return dt;
}

// ============================================================
// Diagnostics
// ============================================================
CartAle2Solver::Diagnostics CartAle2Solver::compute_diagnostics() {
    Diagnostics d{};
    std::vector<double> h_dm(ncell), h_e(ncell), h_cs(ncell);
    CUDA_CHECK(cudaMemcpy(h_dm.data(),  d_dm,    ncell*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_e.data(),   d_e_int, ncell*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cs.data(),  d_cs,    ncell*sizeof(double), cudaMemcpyDeviceToHost));
    for (int c = 0; c < ncell; ++c) {
        d.total_mass       += h_dm[c];
        d.total_internal_E += h_dm[c] * h_e[c];
    }
    std::vector<double> h_vX(nnode), h_vY(nnode), h_m(nnode), h_Y(nnode);
    CUDA_CHECK(cudaMemcpy(h_vX.data(), d_vX,    nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vY.data(), d_vY,    nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_m.data(),  d_mnode, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Y.data(),  d_Y,     nnode*sizeof(double), cudaMemcpyDeviceToHost));
    // Periodic BC: nodes at in=nnode_x-1 are the same physical point as in=0
    // (and ditto for jn). Skip the duplicate copies when accumulating KE/PE
    // to avoid double-counting.
    bool x_per = (bc_mode & 1) != 0;
    bool y_per = (bc_mode & 2) != 0;
    double max_vy = 0.0;
    // PE: prefer variable-g Φ(Y_node) if configured, else scalar g_y·Y.
    // Under variable gravity the reference Φ is cached in h_phi_node_ref,
    // keyed by the node-row index jn (same as during upload).  For a node
    // whose Y has drifted from the nominal row value we interpolate
    // Φ linearly in Y.
    const bool use_var_g = !h_phi_node_ref.empty();
    // Pre-build a monotone (Y_ref, Φ_ref) table from h_phi_node_ref (this
    // is Y_init at node rows, same uniform spacing as the initial grid).
    std::vector<double> y_tab, phi_tab;
    if (use_var_g) {
        y_tab.resize(nnode_y);
        phi_tab = h_phi_node_ref;
        double dy = g_Ly / (double)ny;
        for (int jn = 0; jn < nnode_y; ++jn) y_tab[jn] = jn * dy;
    }
    auto phi_at = [&](double y) {
        if (!use_var_g) return g_y * y;
        // Linear interpolation into (y_tab, phi_tab), clamp outside.
        if (y <= y_tab.front()) return phi_tab.front();
        if (y >= y_tab.back())  return phi_tab.back();
        // Uniform spacing → direct index.
        double dy = y_tab[1] - y_tab[0];
        double f = y / dy;
        int j = (int)f; if (j < 0) j = 0; if (j >= nnode_y - 1) j = nnode_y - 2;
        double t = f - j;
        return (1.0 - t) * phi_tab[j] + t * phi_tab[j + 1];
    };
    for (int in = 0; in < nnode_x; ++in) {
        if (x_per && in == nnode_x - 1) continue;
        for (int jn = 0; jn < nnode_y; ++jn) {
            if (y_per && jn == nnode_y - 1) continue;
            int n = in * nnode_y + jn;
            double v2 = h_vX[n]*h_vX[n] + h_vY[n]*h_vY[n];
            d.total_KE += 0.5 * h_m[n] * v2;
            d.max_v = std::max(d.max_v, std::sqrt(v2));
            max_vy = std::max(max_vy, std::fabs(h_vY[n]));
            d.total_PE += h_m[n] * phi_at(h_Y[n]);
        }
    }
    // Temporary diagnostic: print max |vy| to stderr to help trace spurious
    // cross-stream velocity accumulation in uniform-advection tests.
    if (std::getenv("DEBUG_MAXVY") != nullptr) {
        std::vector<double> h_P(ncell), h_Q(ncell), h_FX(nnode), h_FY(nnode);
        CUDA_CHECK(cudaMemcpy(h_P.data(),  d_P,  ncell*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_Q.data(),  d_Q,  ncell*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_FX.data(), d_FX, nnode*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_FY.data(), d_FY, nnode*sizeof(double), cudaMemcpyDeviceToHost));
        double Pmin=h_P[0], Pmax=h_P[0], Qmin=h_Q[0], Qmax=h_Q[0];
        for (int c = 0; c < ncell; ++c) {
            Pmin = std::min(Pmin, h_P[c]); Pmax = std::max(Pmax, h_P[c]);
            Qmin = std::min(Qmin, h_Q[c]); Qmax = std::max(Qmax, h_Q[c]);
        }
        double FXmax=0, FYmax=0;
        for (int n=0;n<nnode;++n) {
            FXmax = std::max(FXmax, std::fabs(h_FX[n]));
            FYmax = std::max(FYmax, std::fabs(h_FY[n]));
        }
        std::fprintf(stderr, "  [diag] max|vy|=%.6e P=[%.6e,%.6e] ΔP=%.2e Q=[%.6e,%.6e] |FX|=%.3e |FY|=%.3e\n",
                     max_vy, Pmin, Pmax, Pmax-Pmin, Qmin, Qmax, FXmax, FYmax);
    }
    d.total_E = d.total_KE + d.total_internal_E + d.total_PE;
    double cs_max = 0.0;
    for (int c = 0; c < ncell; ++c) cs_max = std::max(cs_max, h_cs[c]);
    d.max_mach = (cs_max > 0.0) ? d.max_v / cs_max : 0.0;
    return d;
}

void CartAle2Solver::download_xslice(std::vector<double>& x,
                                     std::vector<double>& rho,
                                     std::vector<double>& P,
                                     std::vector<double>& vx,
                                     std::vector<double>& e_int) {
    x.assign(nx, 0); rho.assign(nx, 0); P.assign(nx, 0);
    vx.assign(nx, 0); e_int.assign(nx, 0);
    std::vector<double> h_X(nnode), h_vX(nnode);
    CUDA_CHECK(cudaMemcpy(h_X.data(),  d_X,  nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vX.data(), d_vX, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_rho(ncell), h_P(ncell), h_e(ncell), h_Vol(ncell);
    CUDA_CHECK(cudaMemcpy(h_rho.data(), d_rho,   ncell*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_P.data(),   d_P,     ncell*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_e.data(),   d_e_int, ncell*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Vol.data(), d_Vol,   ncell*sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> w(nx, 0.0);
    for (int ic = 0; ic < nx; ++ic)
        for (int jc = 0; jc < ny; ++jc) {
            int flat = ic*ny + jc;
            int I[4] = { ic*nnode_y + jc, (ic+1)*nnode_y + jc,
                         (ic+1)*nnode_y + (jc+1), ic*nnode_y + (jc+1) };
            double Xc = 0.25 * (h_X[I[0]] + h_X[I[1]] + h_X[I[2]] + h_X[I[3]]);
            double vXc = 0.25 * (h_vX[I[0]] + h_vX[I[1]] + h_vX[I[2]] + h_vX[I[3]]);
            double wv = h_Vol[flat];
            x[ic]   += wv * Xc;
            rho[ic] += wv * h_rho[flat];
            P[ic]   += wv * h_P[flat];
            vx[ic]  += wv * vXc;
            e_int[ic] += wv * h_e[flat];
            w[ic] += wv;
        }
    for (int ic = 0; ic < nx; ++ic) {
        double wi = (w[ic] > 0) ? 1.0 / w[ic] : 0.0;
        x[ic] *= wi; rho[ic] *= wi; P[ic] *= wi; vx[ic] *= wi; e_int[ic] *= wi;
    }
}

// ============================================================
// T1 entropy wave compute_error (Athena++ compute_error pattern).
// Append one line to <run_dir>/entropy_wave-errors.dat with L1/Linf and a
// phase-aligned L1 (shift fit removes bulk timing drift). python/pytest
// only reads the .dat — never replicates the analytic solution.
// ============================================================
void CartAle2Solver::compute_entropy_wave_error(double t_now, int ncycle,
                                                double rho0, double P0, double u0,
                                                double A, int k, double periods,
                                                const std::string& run_dir) {
    std::vector<double> xs, rhos, Ps, vxs, es;
    download_xslice(xs, rhos, Ps, vxs, es);

    const double Lx = g_Lx;
    const double twopi_k = 2.0 * M_PI * (double)k / Lx;
    double l1 = 0.0, linf = 0.0;
    for (int i = 0; i < (int)xs.size(); ++i) {
        double expected = rho0 * (1.0 + A * std::sin(twopi_k * xs[i]));
        double e = std::fabs(rhos[i] - expected);
        l1  += e;
        linf = std::max(linf, e);
    }
    l1 /= (double)std::max<size_t>(xs.size(), 1);

    // Phase-aligned L1: fit best shift s in ρ_exact(x − s). The Lagrangian
    // rezone injects a small bulk timing drift that is NOT a dissipation
    // error. Scan shifts on a regular grid fine enough to resolve O(1/Nx):
    // 2001 samples on [-Lx, Lx] gives < 0.1 * dx precision for Nx ≤ 1024.
    const int N_SHIFT = 2001;
    double best_l1 = l1, best_s = 0.0;
    for (int si = 0; si < N_SHIFT; ++si) {
        double s = -Lx + 2.0 * Lx * (double)si / (double)(N_SHIFT - 1);
        double sum = 0.0;
        for (int i = 0; i < (int)xs.size(); ++i) {
            double expected = rho0 *
                (1.0 + A * std::sin(twopi_k * (xs[i] - s)));
            sum += std::fabs(rhos[i] - expected);
        }
        sum /= (double)std::max<size_t>(xs.size(), 1);
        if (sum < best_l1) { best_l1 = sum; best_s = s; }
    }

    // Ensure the run_dir exists (no-op if already there) and append.
    mkdir(run_dir.c_str(), 0755);
    std::string path = run_dir + "/entropy_wave-errors.dat";
    bool new_file = true;
    {
        FILE* f = std::fopen(path.c_str(), "r");
        if (f) { new_file = false; std::fclose(f); }
    }
    FILE* f = std::fopen(path.c_str(), "a");
    if (!f) {
        std::fprintf(stderr, "compute_entropy_wave_error: cannot open %s\n",
                     path.c_str());
        return;
    }
    if (new_file) {
        std::fprintf(f, "# schema: Nx Ny Ncycle t_end A k u0 L1 Linf L1_phase phase_shift\n");
    }
    std::fprintf(f, "%d %d %d %.15e %.6e %d %.6e %.10e %.10e %.10e %.10e\n",
                 nx, ny, ncycle, t_now, A, k, u0, l1, linf, best_l1, best_s);
    std::fclose(f);
    std::fprintf(stderr,
        "entropy_wave error Nx=%d Ny=%d Ncycle=%d  L1=%.4e Linf=%.4e  "
        "L1_phase=%.4e (shift=%+.4f)\n",
        nx, ny, ncycle, l1, linf, best_l1, best_s);
}

// ============================================================
// Linear acoustic wave compute_error (Athena++ linwave pattern).
// After integer periods, exact ρ returns to IC. Same phase-aligned
// L1 as entropy_wave (Lagrangian rezone injects a bulk timing drift
// that shouldn't count as dissipation).
// ============================================================
void CartAle2Solver::compute_acoustic_wave_error(double t_now, int ncycle,
                                                 double rho0, double P0,
                                                 double A, int k, double periods,
                                                 const std::string& run_dir) {
    (void)periods;
    std::vector<double> xs, rhos, Ps, vxs, es;
    download_xslice(xs, rhos, Ps, vxs, es);

    const double Lx = g_Lx;
    const double c0 = std::sqrt(gamma * P0 / rho0);
    const double twopi_k = 2.0 * M_PI * (double)k / Lx;
    // Exact at t_end (periodic): ρ₀·(1 + A·sin(k·2π·x/Lx)).
    double l1 = 0.0, linf = 0.0;
    for (int i = 0; i < (int)xs.size(); ++i) {
        double expected = rho0 * (1.0 + A * std::sin(twopi_k * xs[i]));
        double e = std::fabs(rhos[i] - expected);
        l1  += e;
        linf = std::max(linf, e);
    }
    l1 /= (double)std::max<size_t>(xs.size(), 1);

    // Phase-aligned best-shift L1 (same treatment as entropy wave).
    const int N_SHIFT = 2001;
    double best_l1 = l1, best_s = 0.0;
    for (int si = 0; si < N_SHIFT; ++si) {
        double s = -Lx + 2.0 * Lx * (double)si / (double)(N_SHIFT - 1);
        double sum = 0.0;
        for (int i = 0; i < (int)xs.size(); ++i) {
            double expected = rho0 *
                (1.0 + A * std::sin(twopi_k * (xs[i] - s)));
            sum += std::fabs(rhos[i] - expected);
        }
        sum /= (double)std::max<size_t>(xs.size(), 1);
        if (sum < best_l1) { best_l1 = sum; best_s = s; }
    }

    mkdir(run_dir.c_str(), 0755);
    std::string path = run_dir + "/acoustic_wave-errors.dat";
    bool new_file = true;
    {
        FILE* f = std::fopen(path.c_str(), "r");
        if (f) { new_file = false; std::fclose(f); }
    }
    FILE* f = std::fopen(path.c_str(), "a");
    if (!f) {
        std::fprintf(stderr, "compute_acoustic_wave_error: cannot open %s\n",
                     path.c_str());
        return;
    }
    if (new_file) {
        std::fprintf(f, "# schema: Nx Ny Ncycle t_end A k c0 L1 Linf L1_phase phase_shift\n");
    }
    std::fprintf(f, "%d %d %d %.15e %.6e %d %.6e %.10e %.10e %.10e %.10e\n",
                 nx, ny, ncycle, t_now, A, k, c0, l1, linf, best_l1, best_s);
    std::fclose(f);
    std::fprintf(stderr,
        "acoustic_wave error Nx=%d Ny=%d Ncycle=%d  L1=%.4e Linf=%.4e  "
        "L1_phase=%.4e (shift=%+.4f)\n",
        nx, ny, ncycle, l1, linf, best_l1, best_s);
}

// ============================================================
// Sod shock tube compute_error — compare y-averaged ρ to Toro
// exact at t_now. Analytic solver lives in sod_exact.h (shared
// with athena_vl2). Appends schema-line + one data row to
// <run_dir>/sod-errors.dat.
// ============================================================
void CartAle2Solver::compute_sod_error(double t_now, int ncycle,
                                       const std::string& run_dir) {
    std::vector<double> xs, rhos, Ps, vxs, es;
    download_xslice(xs, rhos, Ps, vxs, es);

    sod_exact::Params P;
    P.x0 = 0.5 * g_Lx;
    // Exclude the 5% wrap-pollution zones near x=0 and x=Lx: when a
    // solver uses x-periodic BC (athena_vl2), waves at t ≳ 0.1 have
    // wrapped and spuriously contaminate the edges. cart_ale2 with
    // reflect BC is fine on the full domain but we use the same window
    // for consistency so both solvers get scored the same way.
    const double x_lo_win = 0.05 * g_Lx;
    const double x_hi_win = 0.95 * g_Lx;
    double l1 = 0.0, linf = 0.0;
    int n_scored = 0;
    for (int i = 0; i < (int)xs.size(); ++i) {
        if (xs[i] < x_lo_win || xs[i] > x_hi_win) continue;
        double rho_e = sod_exact::rho_at(P, xs[i], t_now);
        double e = std::fabs(rhos[i] - rho_e);
        l1 += e;
        linf = std::max(linf, e);
        ++n_scored;
    }
    l1 /= (double)std::max(n_scored, 1);

    mkdir(run_dir.c_str(), 0755);
    std::string path = run_dir + "/sod-errors.dat";
    bool new_file = true;
    {
        FILE* f = std::fopen(path.c_str(), "r");
        if (f) { new_file = false; std::fclose(f); }
    }
    FILE* f = std::fopen(path.c_str(), "a");
    if (!f) {
        std::fprintf(stderr, "compute_sod_error: cannot open %s\n",
                     path.c_str());
        return;
    }
    if (new_file) {
        std::fprintf(f, "# schema: Nx Ny Ncycle t_end L1 Linf\n");
    }
    std::fprintf(f, "%d %d %d %.15e %.10e %.10e\n",
                 nx, ny, ncycle, t_now, l1, linf);
    std::fclose(f);
    std::fprintf(stderr,
        "sod error Nx=%d Ny=%d Ncycle=%d  L1=%.4e Linf=%.4e\n",
        nx, ny, ncycle, l1, linf);
}

// ============================================================
// Gresho stationary vortex compute_error.
// Exact: vφ(r) = 5r (r<0.2), 2-5r (0.2≤r<0.4), 0 (r≥0.4) on [0,Lx]²
// with domain centre (Lx/2, Lx/2). Stationary solution → IC at every t.
// We download node velocities, average onto cell centers, compute
// |v_sim − v_exact| inside r < 0.5 (the vortex disk) where the analytic
// profile is non-trivial.
// ============================================================
void CartAle2Solver::compute_gresho_error(double t_now, int ncycle,
                                          const std::string& run_dir) {
    std::vector<double> h_X(nnode), h_Y(nnode);
    std::vector<double> h_vX(nnode), h_vY(nnode);
    CUDA_CHECK(cudaMemcpy(h_X.data(),  d_X,  nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Y.data(),  d_Y,  nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vX.data(), d_vX, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vY.data(), d_vY, nnode*sizeof(double), cudaMemcpyDeviceToHost));

    const double xc0 = 0.5 * g_Lx;
    const double yc0 = 0.5 * g_Ly;
    double l1 = 0.0, linf = 0.0, v_max_sim = 0.0;
    int n_scored = 0;
    for (int ic = 0; ic < nx; ++ic) {
        for (int jc = 0; jc < ny; ++jc) {
            int I[4] = { ic*nnode_y + jc, (ic+1)*nnode_y + jc,
                         (ic+1)*nnode_y + (jc+1), ic*nnode_y + (jc+1) };
            double Xc  = 0.25 * (h_X[I[0]] + h_X[I[1]] + h_X[I[2]] + h_X[I[3]]);
            double Yc  = 0.25 * (h_Y[I[0]] + h_Y[I[1]] + h_Y[I[2]] + h_Y[I[3]]);
            double vx  = 0.25 * (h_vX[I[0]] + h_vX[I[1]] + h_vX[I[2]] + h_vX[I[3]]);
            double vy  = 0.25 * (h_vY[I[0]] + h_vY[I[1]] + h_vY[I[2]] + h_vY[I[3]]);
            double speed_sim = std::sqrt(vx*vx + vy*vy);
            if (speed_sim > v_max_sim) v_max_sim = speed_sim;

            double dx = Xc - xc0, dy = Yc - yc0;
            double r  = std::sqrt(dx*dx + dy*dy);
            if (r >= 0.5) continue;

            double vphi;
            if      (r < 0.2) vphi = 5.0 * r;
            else if (r < 0.4) vphi = 2.0 - 5.0 * r;
            else              vphi = 0.0;
            // Gresho speed equals |vφ| since flow is purely azimuthal.
            double e = std::fabs(speed_sim - vphi);
            l1 += e;
            linf = std::max(linf, e);
            ++n_scored;
        }
    }
    l1 /= (double)std::max(n_scored, 1);

    mkdir(run_dir.c_str(), 0755);
    std::string path = run_dir + "/gresho-errors.dat";
    bool new_file = true;
    {
        FILE* f = std::fopen(path.c_str(), "r");
        if (f) { new_file = false; std::fclose(f); }
    }
    FILE* f = std::fopen(path.c_str(), "a");
    if (!f) {
        std::fprintf(stderr, "compute_gresho_error: cannot open %s\n", path.c_str());
        return;
    }
    if (new_file) {
        std::fprintf(f, "# schema: Nx Ny Ncycle t_end L1 Linf v_max_sim\n");
    }
    std::fprintf(f, "%d %d %d %.15e %.10e %.10e %.10e\n",
                 nx, ny, ncycle, t_now, l1, linf, v_max_sim);
    std::fclose(f);
    std::fprintf(stderr,
        "gresho error Nx=%d Ny=%d Ncycle=%d  L1=%.4e Linf=%.4e  v_max_sim=%.4f\n",
        nx, ny, ncycle, l1, linf, v_max_sim);
}

// ============================================================
// Yee-Vinokur-Djomehri isentropic vortex round-trip compute_error.
// Domain [0,Lx]×[0,Ly] with centre (Lx/2, Ly/2), γ=1.4, advected at
// (u_inf, v_inf). After t = Lx/u_inf the solution returns to IC by
// periodicity. We compute L1/Linf on ρ − ρ_IC across the full domain.
// ============================================================
void CartAle2Solver::compute_yee_error(double t_now, int ncycle,
                                       double beta, double u_inf, double v_inf,
                                       const std::string& run_dir) {
    (void)u_inf; (void)v_inf;  // periodicity returns state to IC exactly
    std::vector<double> h_X(nnode), h_Y(nnode);
    std::vector<double> h_rho(ncell);
    CUDA_CHECK(cudaMemcpy(h_X.data(),   d_X,   nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Y.data(),   d_Y,   nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rho.data(), d_rho, ncell*sizeof(double), cudaMemcpyDeviceToHost));

    const double xc0 = 0.5 * g_Lx;
    const double yc0 = 0.5 * g_Ly;
    const double gm1 = gamma - 1.0;
    const double pref = gm1 * beta * beta / (8.0 * gamma * M_PI * M_PI);
    double l1 = 0.0, linf = 0.0;
    double rho_min = 1e30, rho_max = -1e30;
    for (int ic = 0; ic < nx; ++ic) {
        for (int jc = 0; jc < ny; ++jc) {
            int I[4] = { ic*nnode_y + jc, (ic+1)*nnode_y + jc,
                         (ic+1)*nnode_y + (jc+1), ic*nnode_y + (jc+1) };
            double Xc = 0.25 * (h_X[I[0]] + h_X[I[1]] + h_X[I[2]] + h_X[I[3]]);
            double Yc = 0.25 * (h_Y[I[0]] + h_Y[I[1]] + h_Y[I[2]] + h_Y[I[3]]);
            double dx = Xc - xc0, dy = Yc - yc0;
            double r2 = dx*dx + dy*dy;
            double T  = 1.0 - pref * std::exp(1.0 - r2);
            double rho_e = std::pow(T, 1.0 / gm1);

            int flat = ic*ny + jc;
            double rho = h_rho[flat];
            if (rho < rho_min) rho_min = rho;
            if (rho > rho_max) rho_max = rho;
            double e = std::fabs(rho - rho_e);
            l1 += e;
            linf = std::max(linf, e);
        }
    }
    l1 /= (double)std::max(ncell, 1);

    mkdir(run_dir.c_str(), 0755);
    std::string path = run_dir + "/yee-errors.dat";
    bool new_file = true;
    {
        FILE* f = std::fopen(path.c_str(), "r");
        if (f) { new_file = false; std::fclose(f); }
    }
    FILE* f = std::fopen(path.c_str(), "a");
    if (!f) {
        std::fprintf(stderr, "compute_yee_error: cannot open %s\n", path.c_str());
        return;
    }
    if (new_file) {
        std::fprintf(f, "# schema: Nx Ny Ncycle t_end L1 Linf rho_min rho_max\n");
    }
    std::fprintf(f, "%d %d %d %.15e %.10e %.10e %.10e %.10e\n",
                 nx, ny, ncycle, t_now, l1, linf, rho_min, rho_max);
    std::fclose(f);
    std::fprintf(stderr,
        "yee error Nx=%d Ny=%d Ncycle=%d  L1=%.4e Linf=%.4e  rho=[%.3f,%.3f]\n",
        nx, ny, ncycle, l1, linf, rho_min, rho_max);
}

// ============================================================
// Write 2D Cartesian VTK (STRUCTURED_GRID) with density, pressure,
// internal energy, velocity, mach. Nodes are always uniform in ALE,
// so the mesh block is just nnode_x × nnode_y lattice points.
// ============================================================
void CartAle2Solver::write_vtk_2d(const char* filename, double Lx, double Ly) {
    std::vector<double> h_rho(ncell), h_P(ncell), h_e(ncell), h_cs(ncell);
    std::vector<double> h_vX(nnode), h_vY(nnode);
    CUDA_CHECK(cudaMemcpy(h_rho.data(), d_rho,   ncell*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_P.data(),   d_P,     ncell*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_e.data(),   d_e_int, ncell*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cs.data(),  d_cs,    ncell*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vX.data(),  d_vX,    nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vY.data(),  d_vY,    nnode*sizeof(double), cudaMemcpyDeviceToHost));

    std::FILE* fp = std::fopen(filename, "w");
    if (!fp) return;
    int nnx = nnode_x, nny = nnode_y;
    std::fprintf(fp, "# vtk DataFile Version 3.0\n");
    std::fprintf(fp, "cart_ale 2D Cartesian ALE output\n");
    std::fprintf(fp, "ASCII\n");
    std::fprintf(fp, "DATASET STRUCTURED_GRID\n");
    std::fprintf(fp, "DIMENSIONS %d %d 1\n", nnx, nny);
    std::fprintf(fp, "POINTS %d double\n", nnx * nny);
    // VTK structured grid indexes X fastest then Y, so iterate jn outer, in inner.
    for (int jn = 0; jn < nny; ++jn) {
        double y = Ly * (double)jn / (double)(nny - 1);
        for (int in = 0; in < nnx; ++in) {
            double x = Lx * (double)in / (double)(nnx - 1);
            std::fprintf(fp, "%.10e %.10e %.10e\n", x, y, 0.0);
        }
    }
    std::fprintf(fp, "CELL_DATA %d\n", ncell);

    auto cell_scalar = [&](const char* name, const std::vector<double>& arr) {
        std::fprintf(fp, "SCALARS %s double 1\nLOOKUP_TABLE default\n", name);
        for (int jc = 0; jc < ny; ++jc)
            for (int ic = 0; ic < nx; ++ic)
                std::fprintf(fp, "%.10e\n", arr[ic*ny + jc]);
    };
    cell_scalar("density",  h_rho);
    cell_scalar("pressure", h_P);
    cell_scalar("e_int",    h_e);

    // Cell-centered velocity (average of 4 corner node velocities)
    std::fprintf(fp, "VECTORS velocity double\n");
    for (int jc = 0; jc < ny; ++jc)
        for (int ic = 0; ic < nx; ++ic) {
            int I[4] = { ic*nny + jc, (ic+1)*nny + jc,
                         (ic+1)*nny + (jc+1), ic*nny + (jc+1) };
            double vx = 0.25 * (h_vX[I[0]] + h_vX[I[1]] + h_vX[I[2]] + h_vX[I[3]]);
            double vy = 0.25 * (h_vY[I[0]] + h_vY[I[1]] + h_vY[I[2]] + h_vY[I[3]]);
            std::fprintf(fp, "%.10e %.10e %.10e\n", vx, vy, 0.0);
        }

    std::fprintf(fp, "SCALARS mach double 1\nLOOKUP_TABLE default\n");
    for (int jc = 0; jc < ny; ++jc)
        for (int ic = 0; ic < nx; ++ic) {
            int c = ic*ny + jc;
            int I[4] = { ic*nny + jc, (ic+1)*nny + jc,
                         (ic+1)*nny + (jc+1), ic*nny + (jc+1) };
            double vx = 0.25 * (h_vX[I[0]] + h_vX[I[1]] + h_vX[I[2]] + h_vX[I[3]]);
            double vy = 0.25 * (h_vY[I[0]] + h_vY[I[1]] + h_vY[I[2]] + h_vY[I[3]]);
            double speed = std::sqrt(vx*vx + vy*vy);
            double cs = std::fmax(h_cs[c], 1e-30);
            std::fprintf(fp, "%.10e\n", speed / cs);
        }

    // Passive species tracer X (μ₁ mass fraction) if enabled.
    if (tracer_enabled && d_species_X != nullptr) {
        std::vector<double> h_X(ncell);
        CUDA_CHECK(cudaMemcpy(h_X.data(), d_species_X, ncell*sizeof(double),
                              cudaMemcpyDeviceToHost));
        std::fprintf(fp, "SCALARS species_X double 1\nLOOKUP_TABLE default\n");
        for (int jc = 0; jc < ny; ++jc)
            for (int ic = 0; ic < nx; ++ic)
                std::fprintf(fp, "%.10e\n", h_X[ic*ny + jc]);
    }

    std::fclose(fp);
}

// ============================================================
// VRAM-buffered frame dump
//
// Sizing: use (free VRAM − headroom_mb) / (5·ncell·8B) frames.
// We keep the whole pool resident; when it fills we copy-out the
// whole batch in one cudaMemcpy, write VTK binary files sequentially,
// then reset the count.
// ============================================================
void CartAle2Solver::alloc_frame_buffer(int headroom_mb) {
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    size_t headroom_b = (size_t)headroom_mb * 1024ull * 1024ull;
    if (free_b <= headroom_b) {
        std::fprintf(stderr,
            "  CartAle frame buffer: only %.2f GB free, %.2f GB headroom — disabling VRAM buffer\n",
            free_b / 1.0e9, headroom_b / 1.0e9);
        frame_capacity = 0;
        return;
    }
    size_t pool_b = free_b - headroom_b;
    size_t per_frame_b = (size_t)ncell * 5ull * sizeof(double);
    frame_capacity = (int)(pool_b / per_frame_b);
    if (frame_capacity < 4) frame_capacity = 4;
    size_t actual_b = (size_t)frame_capacity * per_frame_b;
    if (cudaMalloc(&d_frame_pool, actual_b) != cudaSuccess) {
        // fallback: try smaller
        frame_capacity = (int)(((size_t)(free_b * 0.5)) / per_frame_b);
        if (frame_capacity < 4) frame_capacity = 4;
        actual_b = (size_t)frame_capacity * per_frame_b;
        CUDA_CHECK(cudaMalloc(&d_frame_pool, actual_b));
    }
    frame_count = 0;
    total_frames = 0;
    frame_times.clear();
    frame_steps.clear();
    std::fprintf(stderr,
        "  CartAle frame buffer: %d frames × %.2f MB = %.2f GB "
        "(free was %.2f GB, headroom %.2f GB)\n",
        frame_capacity, per_frame_b / 1.0e6, actual_b / 1.0e9,
        free_b / 1.0e9, headroom_b / 1.0e9);
}

void CartAle2Solver::capture_frame(double t, int step) {
    if (!d_frame_pool || frame_capacity == 0) return;
    if (frame_count >= frame_capacity) {
        // Caller is expected to flush before capturing more; guard anyway.
        flush_frames_to_disk(frame_out_dir, 1.0, 1.0);
    }
    int B = 256;
    int BCell = (ncell + B - 1) / B;
    double* slot = d_frame_pool + (size_t)frame_count * 5ull * (size_t)ncell;
    k_cale2_snapshot<<<BCell, B>>>(d_rho, d_P, d_e_int, d_vX, d_vY,
                                  slot, nx, ny);
    frame_times.push_back(t);
    frame_steps.push_back(step);
    frame_count++;
}

// Write one frame (5 fields) from a host pointer to binary VTK.
// Big-endian required by VTK legacy binary — swap on little-endian hosts.
static void write_vtk_binary_frame(const char* path, int nx, int ny,
                                   double Lx, double Ly,
                                   const double* rho, const double* P,
                                   const double* e_int, const double* vx,
                                   const double* vy, double gamma) {
    std::FILE* fp = std::fopen(path, "wb");
    if (!fp) return;
    int nnx = nx + 1, nny = ny + 1;
    std::fprintf(fp, "# vtk DataFile Version 3.0\n");
    std::fprintf(fp, "cart_ale binary frame\n");
    std::fprintf(fp, "BINARY\n");
    std::fprintf(fp, "DATASET STRUCTURED_GRID\n");
    std::fprintf(fp, "DIMENSIONS %d %d 1\n", nnx, nny);
    std::fprintf(fp, "POINTS %d double\n", nnx * nny);

    auto bswap8 = [](double v) {
        union { double d; uint64_t u; } x; x.d = v;
        x.u = ((x.u & 0x00000000000000FFULL) << 56) |
              ((x.u & 0x000000000000FF00ULL) << 40) |
              ((x.u & 0x0000000000FF0000ULL) << 24) |
              ((x.u & 0x00000000FF000000ULL) <<  8) |
              ((x.u & 0x000000FF00000000ULL) >>  8) |
              ((x.u & 0x0000FF0000000000ULL) >> 24) |
              ((x.u & 0x00FF000000000000ULL) >> 40) |
              ((x.u & 0xFF00000000000000ULL) >> 56);
        return x.d;
    };

    // Points in i-fastest, j-next order (consistent with existing ASCII writer).
    std::vector<double> pts(3 * nnx * nny);
    size_t k = 0;
    for (int jn = 0; jn < nny; ++jn) {
        double y = Ly * (double)jn / (double)(nny - 1);
        for (int in = 0; in < nnx; ++in) {
            double x = Lx * (double)in / (double)(nnx - 1);
            pts[k++] = bswap8(x);
            pts[k++] = bswap8(y);
            pts[k++] = bswap8(0.0);
        }
    }
    std::fwrite(pts.data(), sizeof(double), pts.size(), fp);
    std::fputc('\n', fp);

    int nc = nx * ny;
    std::fprintf(fp, "CELL_DATA %d\n", nc);

    auto write_scalar = [&](const char* name, const double* arr) {
        std::fprintf(fp, "SCALARS %s double 1\nLOOKUP_TABLE default\n", name);
        std::vector<double> buf(nc);
        // rearrange from ic-outer, jc-inner (flat=ic*ny+jc) to jc-outer, ic-inner
        // to match VTK cell-data order (x-fastest).
        size_t idx = 0;
        for (int jc = 0; jc < ny; ++jc)
            for (int ic = 0; ic < nx; ++ic)
                buf[idx++] = bswap8(arr[ic*ny + jc]);
        std::fwrite(buf.data(), sizeof(double), buf.size(), fp);
        std::fputc('\n', fp);
    };
    write_scalar("density",  rho);
    write_scalar("pressure", P);
    write_scalar("e_int",    e_int);

    std::fprintf(fp, "VECTORS velocity double\n");
    std::vector<double> vbuf(3 * nc);
    {
        size_t idx = 0;
        for (int jc = 0; jc < ny; ++jc)
            for (int ic = 0; ic < nx; ++ic) {
                int c = ic*ny + jc;
                vbuf[idx++] = bswap8(vx[c]);
                vbuf[idx++] = bswap8(vy[c]);
                vbuf[idx++] = bswap8(0.0);
            }
        std::fwrite(vbuf.data(), sizeof(double), vbuf.size(), fp);
        std::fputc('\n', fp);
    }

    // mach = |v| / cs,  cs = sqrt(γ·P/ρ)
    std::fprintf(fp, "SCALARS mach double 1\nLOOKUP_TABLE default\n");
    std::vector<double> mbuf(nc);
    {
        size_t idx = 0;
        for (int jc = 0; jc < ny; ++jc)
            for (int ic = 0; ic < nx; ++ic) {
                int c = ic*ny + jc;
                double rhoc = std::fmax(rho[c], 1e-30);
                double pc   = std::fmax(P[c],   1e-30);
                double cs   = std::sqrt(gamma * pc / rhoc);
                double sp   = std::sqrt(vx[c]*vx[c] + vy[c]*vy[c]);
                mbuf[idx++] = bswap8(sp / std::fmax(cs, 1e-30));
            }
        std::fwrite(mbuf.data(), sizeof(double), mbuf.size(), fp);
        std::fputc('\n', fp);
    }

    std::fclose(fp);
}

void CartAle2Solver::flush_frames_to_disk(const std::string& run_dir,
                                        double Lx, double Ly) {
    if (frame_count == 0 || !d_frame_pool) return;
    frame_out_dir = run_dir;
    size_t per_frame = (size_t)ncell * 5ull;
    std::vector<double> host((size_t)frame_count * per_frame);
    CUDA_CHECK(cudaMemcpy(host.data(), d_frame_pool,
                          host.size() * sizeof(double),
                          cudaMemcpyDeviceToHost));
    std::fprintf(stderr, "  CartAle flushing %d buffered frames → %s ...",
                 frame_count, run_dir.c_str());
    std::fflush(stderr);
    // frames.csv: cumulative index of every frame ever written, with its
    // exact physical time and step number. Enables the renderer to show
    // true-t in the title and (optionally) resample to strict uniform-t.
    char csv_path[512];
    std::snprintf(csv_path, sizeof(csv_path), "%s/frames.csv", run_dir.c_str());
    bool first_batch = (total_frames == 0);
    std::FILE* fcsv = std::fopen(csv_path, first_batch ? "w" : "a");
    if (fcsv && first_batch) std::fprintf(fcsv, "index,step,t\n");
    // frame_times/frame_steps hold the *currently buffered* frames;
    // pair them with a running total index.
    int base_idx = total_frames;
    for (int f = 0; f < frame_count; ++f) {
        ++total_frames;
        const double* base = host.data() + (size_t)f * per_frame;
        const double* rho_p = base;
        const double* P_p   = base + ncell;
        const double* e_p   = base + 2*ncell;
        const double* vx_p  = base + 3*ncell;
        const double* vy_p  = base + 4*ncell;
        char path[512];
        std::snprintf(path, sizeof(path),
                      "%s/output_%04d.vtk", run_dir.c_str(), total_frames);
        write_vtk_binary_frame(path, nx, ny, Lx, Ly,
                               rho_p, P_p, e_p, vx_p, vy_p, gamma);
        if (fcsv) {
            int idx = base_idx + f + 1;   // match %04d filename
            std::fprintf(fcsv, "%d,%d,%.10e\n",
                         idx, frame_steps[f], frame_times[f]);
        }
    }
    if (fcsv) std::fclose(fcsv);
    std::fprintf(stderr, " done (%d total)\n", total_frames);
    frame_count = 0;
    frame_times.clear();
    frame_steps.clear();
}

void CartAle2Solver::free_frame_buffer() {
    if (d_frame_pool) cudaFree(d_frame_pool);
    d_frame_pool = nullptr;
    frame_capacity = frame_count = total_frames = 0;
    frame_times.clear();
    frame_steps.clear();
}
