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
#include "fas_common.cuh"
#include "fas_linalg.cuh"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

// ==== Forward decls of all kernels in cart_ale_kernels.cu =====
__global__ void k_cale2_geometry(const double*, const double*, double*, double*, int, int);
__global__ void k_cale2_eos_and_q(const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, double*, int, int, double, double, double, int);
__global__ void k_cale2_node_forces(const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, int, int);
__global__ void k_cale2_zero(double*, int);
__global__ void k_cale2_add_gravity(const double*, double*, double, int);
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
    double*, double*, int, int);
__global__ void k_cale2_remap_init(const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, int);
__global__ void k_cale2_remap_east(const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, int, int, int);
__global__ void k_cale2_remap_north(const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, int, int, int);
__global__ void k_cale2_remap_finalize_cells(const double*, const double*, double*, double*, int);
__global__ void k_cale2_rebuild_node_v(const double*, const double*, const double*,
    double*, double*, int, int, int);
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
    std::memset(this, 0, sizeof(*this));
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
    while (skip_comments(line, sizeof(line))) {
        double y, r, p, T;
        if (std::sscanf(line, "%lf %lf %lf %lf", &y, &r, &p, &T) != 4) break;
        ys.push_back(y); rhos.push_back(r); Ps.push_back(p);
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

    double cs_bot = std::sqrt(gamma * Ps.front() / rhos.front());
    double cs_top = std::sqrt(gamma * Ps.back()  / rhos.back());
    std::fprintf(stderr,
        "  CartAle2 local_convection: slab=%s\n"
        "    Ly=%.3e  Lx=%.3e  g=%.3e  γ=%.3f  (top ρ=%.3e, P=%.3e, T=%.3e)\n"
        "    c_s top=%.3e bot=%.3e,  τ_dyn=Ly/c_s_top=%.3e  perturb=%.3g @ k=%d\n",
        slab_file.c_str(), g_Ly, g_Lx, g_y, gamma, rho_top, P_top, T_top,
        cs_top, cs_bot, g_Ly / cs_top, perturb_amp, seed_k);
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

    // --- Phase L: Lagrangian substep -------------------------
    // Mesh is always X0/Y0 at step entry (reset by previous step's Phase R),
    // so Vol ≡ Area0 and minheight is constant — both cached at init time.
    // Skipping the per-step geometry kernel saves one launch per step.
    k_cale2_eos_and_q<<<BCell, B>>>(
        d_X, d_Y, d_vX, d_vY, d_dm, d_Area0, d_Area0, d_e_int,
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
    if (g_y != 0.0)
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

    // --- Phase M: Remap --------------------------------------
    // Snapshot cell-centered momentum from current node velocities BEFORE rezone.
    k_cale2_cell_momentum<<<BCell, B>>>(d_vX, d_vY, d_dm, d_px_cell, d_py_cell, nx, ny);

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

    // Finalize dm, e_int from accumulators
    k_cale2_remap_finalize_cells<<<BCell, B>>>(d_dm_new, d_ie_new, d_dm, d_e_int, ncell);

    // Rebuild node velocities from remapped momentum and mass
    k_cale2_rebuild_node_v<<<BNode, B>>>(d_px_new, d_py_new, d_dm_new,
                                        d_vX, d_vY, nx, ny, bc_mode);
    k_cale2_bc_velocity<<<BNode, B>>>(d_vX, d_vY, nnode_x, nnode_y, bc_mode);
    if (bc_mode) k_cale2_periodic_sync_node<<<BNode, B>>>(d_vX, d_vY,
                                                         nnode_x, nnode_y, bc_mode, /*mode=copy*/ 0);

    // --- Phase R: snap mesh back to uniform ------------------
    k_cale2_reset_mesh<<<BNode, B>>>(d_X0, d_Y0, d_X, d_Y, nnode);

    // Refresh node mass (dm redistributes a bit each step)
    k_cale2_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nx, ny, bc_mode);

    // Optional Newton cooling toward the IC stratification (only applied
    // if alloc_cooling_ref was called and tau_cool > 0).
    apply_cooling(dt);

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
    for (int in = 0; in < nnode_x; ++in) {
        if (x_per && in == nnode_x - 1) continue;
        for (int jn = 0; jn < nnode_y; ++jn) {
            if (y_per && jn == nnode_y - 1) continue;
            int n = in * nnode_y + jn;
            double v2 = h_vX[n]*h_vX[n] + h_vY[n]*h_vY[n];
            d.total_KE += 0.5 * h_m[n] * v2;
            d.max_v = std::max(d.max_v, std::sqrt(v2));
            max_vy = std::max(max_vy, std::fabs(h_vY[n]));
            d.total_PE += h_m[n] * g_y * h_Y[n];
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
