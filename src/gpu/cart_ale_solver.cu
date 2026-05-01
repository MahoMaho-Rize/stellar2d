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

#include "cart_ale_solver.cuh"
#include "fas_common.cuh"
#include "fas_linalg.cuh"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

// ==== Forward decls of all kernels in cart_ale_kernels.cu =====
__global__ void k_cale_geometry(const double*, const double*, double*, double*, int, int);
__global__ void k_cale_eos_and_q(const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, double*, int, int, double, double, double);
__global__ void k_cale_node_forces(const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, int, int);
__global__ void k_cale_zero(double*, int);
__global__ void k_cale_add_gravity(const double*, double*, double, int);
__global__ void k_cale_bc_reflective(const double*, const double*, double*, double*,
    double*, double*, double*, double*, int, int);
__global__ void k_cale_node_update(double*, double*, double*, double*,
    const double*, const double*, const double*, double*, double*, double, int);
__global__ void k_cale_energy_update(int, int, const double*, const double*,
    const double*, const double*, const double*, double*);
__global__ void k_cale_cfl(const double*, const double*, const double*,
    const double*, const double*, int, int, double, double, double*);
__global__ void k_cale_init_nodes(double*, double*, double*, double*, double, double, int, int);
__global__ void k_cale_reset_mesh(const double*, const double*, double*, double*, int);
__global__ void k_cale_node_mass(const double*, double*, int, int);
__global__ void k_cale_cell_momentum(const double*, const double*, const double*,
    double*, double*, int, int);
__global__ void k_cale_remap_init(const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, int);
__global__ void k_cale_remap_east(const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, int, int);
__global__ void k_cale_remap_north(const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, int, int);
__global__ void k_cale_remap_finalize_cells(const double*, const double*, double*, double*, int);
__global__ void k_cale_rebuild_node_v(const double*, const double*, const double*,
    double*, double*, int, int);
__global__ void k_cale_bc_velocity(double*, double*, int, int);
__global__ void k_cale_cell_densities(const double*, const double*, const double*, const double*,
    const double*, double*, double*, double*, double*, int);
__global__ void k_cale_slopes_minmod(const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, double*, double*, double*, double*,
    int, int, double, double);
__global__ void k_cale_remap_east_2nd(const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, int, int, double, double);
__global__ void k_cale_remap_north_2nd(const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, int, int, double, double);

// Stash Lx/Ly at module scope (used by non-member IC loaders).
static double g_Lx = 0.0, g_Ly = 0.0;

void CartAleSolver::init(int nx_in, int ny_in, double Lx, double Ly,
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

    dx_u = Lx / (double)nx;
    dy_u = Ly / (double)ny;

    mal(&d_FSX, nsub*sizeof(double));
    mal(&d_FSY, nsub*sizeof(double));

    mal(&d_dt_cell,    ncell*sizeof(double));
    mal(&d_reduce_buf, ncell*sizeof(double));

    int B = 256;
    k_cale_init_nodes<<<(nnode+B-1)/B, B>>>(d_X, d_Y, d_X0, d_Y0,
                                            Lx, Ly, nnode_x, nnode_y);
    k_cale_geometry<<<(ncell+B-1)/B, B>>>(d_X, d_Y, d_Vol, d_minheight, nx, ny);
    CUDA_CHECK(cudaMemcpy(d_Area0, d_Vol, ncell*sizeof(double), cudaMemcpyDeviceToDevice));

    std::fprintf(stderr,
        "CartAleSolver: %dx%d cells, box=[%g,%g]x[%g,%g], γ=%g, CFL=%g (Eulerian rezone)\n",
        nx, ny, 0.0, Lx, 0.0, Ly, gamma, cfl);
}

void CartAleSolver::destroy() {
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
    f(d_FSX); f(d_FSY);
    f(d_dt_cell); f(d_reduce_buf);
    std::memset(this, 0, sizeof(*this));
}

// ============================================================
// ICs
// ============================================================
void CartAleSolver::init_uniform(double rho, double P, double vx, double vy) {
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
    k_cale_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nx, ny);
    std::fprintf(stderr, "  CartAle Uniform IC: ρ=%g, P=%g, v=(%g,%g)\n", rho, P, vx, vy);
}

void CartAleSolver::init_sod() {
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
    k_cale_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nx, ny);
    std::fprintf(stderr, "  CartAle Sod IC: ρL=1.0 PL=1.0 | ρR=0.125 PR=0.1\n");
}

void CartAleSolver::init_hse_polytrope(double rho_base, double g_val, double amp) {
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
    k_cale_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nx, ny);

    std::fprintf(stderr,
        "  CartAle HSE polytrope: ρ_b=%g, g=%g, K=%g, y_top=%g, perturb=%g\n",
        rho_base, g_val, K, Ly, amp);
}

// ============================================================
// HSE polytrope + N Gaussian bubble overlays
// ============================================================
void CartAleSolver::init_hse_bubbles(double rho_base, double g_val,
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
    k_cale_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nx, ny);

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
// One ALE step
// ============================================================
double CartAleSolver::step(double t, double t_end) {
    int B = 256;
    int BCell = (ncell + B - 1) / B;
    int BNode = (nnode + B - 1) / B;

    // --- Phase L: Lagrangian substep -------------------------
    k_cale_geometry<<<BCell, B>>>(d_X, d_Y, d_Vol, d_minheight, nx, ny);

    k_cale_eos_and_q<<<BCell, B>>>(
        d_X, d_Y, d_vX, d_vY, d_dm, d_Vol, d_Area0, d_e_int,
        d_rho, d_P, d_Q, d_cs, d_strain_rate,
        nx, ny, gamma, CQ_lin, CQ_quad);

    k_cale_cfl<<<BCell, B>>>(
        d_minheight, d_cs, d_strain_rate, d_vX, d_vY,
        nx, ny, cfl, comp_dt_frac, d_dt_cell);
    double dt_min = gpu_reduce_min(d_dt_cell, d_reduce_buf, ncell);
    double dt = (dt_min > 0.0 && dt_min < 1e29) ? dt_min : 1e-12;
    if (dt < 1e-30) dt = 1e-12;
    if (t + dt > t_end) dt = t_end - t;

    k_cale_zero<<<BNode, B>>>(d_FX, nnode);
    k_cale_zero<<<BNode, B>>>(d_FY, nnode);
    k_cale_node_forces<<<BCell, B>>>(d_X, d_Y, d_P, d_Q,
                                     d_FX, d_FY, d_FSX, d_FSY, nx, ny);
    if (g_y != 0.0)
        k_cale_add_gravity<<<BNode, B>>>(d_mnode, d_FY, g_y, nnode);

    k_cale_bc_reflective<<<BNode, B>>>(d_X0, d_Y0, d_X, d_Y, d_vX, d_vY,
                                       d_FX, d_FY, nnode_x, nnode_y);

    k_cale_node_update<<<BNode, B>>>(d_X, d_Y, d_vX, d_vY, d_FX, d_FY,
                                     d_mnode, d_dX, d_dY, dt, nnode);

    k_cale_bc_reflective<<<BNode, B>>>(d_X0, d_Y0, d_X, d_Y, d_vX, d_vY,
                                       d_FX, d_FY, nnode_x, nnode_y);

    k_cale_energy_update<<<BCell, B>>>(nx, ny, d_FSX, d_FSY,
                                       d_dX, d_dY, d_dm, d_e_int);

    // --- Phase M: Remap --------------------------------------
    // Snapshot cell-centered momentum from current node velocities BEFORE rezone.
    k_cale_cell_momentum<<<BCell, B>>>(d_vX, d_vY, d_dm, d_px_cell, d_py_cell, nx, ny);

    // Vol was set to pre-Lagrangian geom; we need pre-Lagrangian donor volume.
    // In Eulerian rezone the "old" mesh for swept-remap is X0/Y0 (uniform),
    // so V_donor = Area0. That's what we pass in.
    k_cale_remap_init<<<BCell, B>>>(d_dm, d_e_int, d_px_cell, d_py_cell,
                                    d_dm_new, d_ie_new, d_px_new, d_py_new, ncell);

    int n_east = (nx - 1) * ny;
    int n_north = nx * (ny - 1);

    if (remap_order >= 2) {
        // Prepare cell-average densities + minmod-limited slopes on the
        // (uniform) reference mesh, then 2nd-order swept-edge remap.
        k_cale_cell_densities<<<BCell, B>>>(
            d_dm, d_e_int, d_px_cell, d_py_cell, d_Area0,
            d_rho_dens, d_rhoE_dens, d_pxd_dens, d_pyd_dens, ncell);
        k_cale_slopes_minmod<<<BCell, B>>>(
            d_rho_dens, d_rhoE_dens, d_pxd_dens, d_pyd_dens,
            d_rho_sx,  d_rho_sy,
            d_rhoE_sx, d_rhoE_sy,
            d_pxd_sx,  d_pxd_sy,
            d_pyd_sx,  d_pyd_sy,
            nx, ny, dx_u, dy_u);
        if (n_east > 0) {
            int BE = (n_east + B - 1) / B;
            k_cale_remap_east_2nd<<<BE, B>>>(
                d_X0, d_Y0, d_X, d_Y,
                d_rho_dens, d_rhoE_dens, d_pxd_dens, d_pyd_dens,
                d_rho_sx,  d_rho_sy,
                d_rhoE_sx, d_rhoE_sy,
                d_pxd_sx,  d_pxd_sy,
                d_pyd_sx,  d_pyd_sy,
                d_dm_new, d_ie_new, d_px_new, d_py_new,
                nx, ny, dx_u, dy_u);
        }
        if (n_north > 0) {
            int BN = (n_north + B - 1) / B;
            k_cale_remap_north_2nd<<<BN, B>>>(
                d_X0, d_Y0, d_X, d_Y,
                d_rho_dens, d_rhoE_dens, d_pxd_dens, d_pyd_dens,
                d_rho_sx,  d_rho_sy,
                d_rhoE_sx, d_rhoE_sy,
                d_pxd_sx,  d_pxd_sy,
                d_pyd_sx,  d_pyd_sy,
                d_dm_new, d_ie_new, d_px_new, d_py_new,
                nx, ny, dx_u, dy_u);
        }
    } else {
        if (n_east > 0) {
            int BE = (n_east + B - 1) / B;
            k_cale_remap_east<<<BE, B>>>(d_X0, d_Y0, d_X, d_Y,
                                         d_dm, d_e_int, d_px_cell, d_py_cell, d_Area0,
                                         d_dm_new, d_ie_new, d_px_new, d_py_new, nx, ny);
        }
        if (n_north > 0) {
            int BN = (n_north + B - 1) / B;
            k_cale_remap_north<<<BN, B>>>(d_X0, d_Y0, d_X, d_Y,
                                          d_dm, d_e_int, d_px_cell, d_py_cell, d_Area0,
                                          d_dm_new, d_ie_new, d_px_new, d_py_new, nx, ny);
        }
    }

    // Finalize dm, e_int from accumulators
    k_cale_remap_finalize_cells<<<BCell, B>>>(d_dm_new, d_ie_new, d_dm, d_e_int, ncell);

    // Rebuild node velocities from remapped momentum and mass
    k_cale_rebuild_node_v<<<BNode, B>>>(d_px_new, d_py_new, d_dm_new,
                                        d_vX, d_vY, nx, ny);
    k_cale_bc_velocity<<<BNode, B>>>(d_vX, d_vY, nnode_x, nnode_y);

    // --- Phase R: snap mesh back to uniform ------------------
    k_cale_reset_mesh<<<BNode, B>>>(d_X0, d_Y0, d_X, d_Y, nnode);

    // Refresh node mass (dm redistributes a bit each step)
    k_cale_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nx, ny);

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
CartAleSolver::Diagnostics CartAleSolver::compute_diagnostics() {
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
    for (int n = 0; n < nnode; ++n) {
        double v2 = h_vX[n]*h_vX[n] + h_vY[n]*h_vY[n];
        d.total_KE += 0.5 * h_m[n] * v2;
        d.max_v = std::max(d.max_v, std::sqrt(v2));
        d.total_PE += h_m[n] * g_y * h_Y[n];
    }
    d.total_E = d.total_KE + d.total_internal_E + d.total_PE;
    double cs_max = 0.0;
    for (int c = 0; c < ncell; ++c) cs_max = std::max(cs_max, h_cs[c]);
    d.max_mach = (cs_max > 0.0) ? d.max_v / cs_max : 0.0;
    return d;
}

void CartAleSolver::download_xslice(std::vector<double>& x,
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
void CartAleSolver::write_vtk_2d(const char* filename, double Lx, double Ly) {
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
