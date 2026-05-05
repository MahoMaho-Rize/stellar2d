// Cartesian 2D staggered quad Lagrangian — orchestration.

#include "cart_lag_solver.cuh"
#include "gpu_common.cuh"
#include "gpu_linalg.cuh"   // gpu_reduce_min
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

// Forward decls (kernels in cart_lag_kernels.cu)
__global__ void k_clag_geometry(const double*, const double*, double*, double*, int, int);
__global__ void k_clag_eos_and_q(const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, double*, int, int, double, double, double);
__global__ void k_clag_node_forces(const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, int, int);
__global__ void k_clag_zero(double*, int);
__global__ void k_clag_add_gravity(const double*, double*, double, int);
__global__ void k_clag_bc_reflective(double*, double*, double*, double*, double*, double*,
    double, double, int, int);
__global__ void k_clag_node_update(double*, double*, double*, double*,
    const double*, const double*, const double*, double*, double*, double, int);
__global__ void k_clag_energy_update(const int, const int,
    const double*, const double*, const double*, const double*, const double*, double*);
__global__ void k_clag_cfl(const double*, const double*, const double*,
    const double*, const double*, int, int, double, double, double*);
__global__ void k_clag_init_nodes(double*, double*, double, double, int, int);
__global__ void k_clag_node_mass(const double*, double*, int, int);

// Stash Lx/Ly on the solver as members — add via static map for now (simpler)
static double g_Lx = 0.0, g_Ly = 0.0;

void CartLagSolver::init(int nx_in, int ny_in, double Lx, double Ly,
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
    mal(&d_vX, nnode*sizeof(double));  mal(&d_vY, nnode*sizeof(double));
    mal(&d_FX, nnode*sizeof(double));  mal(&d_FY, nnode*sizeof(double));
    mal(&d_FX_hse, nnode*sizeof(double)); mal(&d_FY_hse, nnode*sizeof(double));
    mal(&d_mnode, nnode*sizeof(double));
    mal(&d_dX, nnode*sizeof(double));  mal(&d_dY, nnode*sizeof(double));

    mal(&d_dm,   ncell*sizeof(double));
    mal(&d_Vol,  ncell*sizeof(double));
    mal(&d_Area0,ncell*sizeof(double));
    mal(&d_rho,  ncell*sizeof(double));
    mal(&d_e_int,ncell*sizeof(double));
    mal(&d_P,    ncell*sizeof(double));
    mal(&d_Q,    ncell*sizeof(double));
    mal(&d_cs,   ncell*sizeof(double));
    mal(&d_minheight,   ncell*sizeof(double));
    mal(&d_strain_rate, ncell*sizeof(double));

    mal(&d_FSX, nsub*sizeof(double));
    mal(&d_FSY, nsub*sizeof(double));

    mal(&d_dt_cell,    ncell*sizeof(double));
    mal(&d_reduce_buf, ncell*sizeof(double));

    // Place nodes on a uniform grid
    int B = 256;
    k_clag_init_nodes<<<(nnode+B-1)/B, B>>>(d_X, d_Y, Lx, Ly, nnode_x, nnode_y);

    // Initial geometry → Area0
    k_clag_geometry<<<(ncell+B-1)/B, B>>>(d_X, d_Y, d_Vol, d_minheight, nx, ny);
    CUDA_CHECK(cudaMemcpy(d_Area0, d_Vol, ncell*sizeof(double), cudaMemcpyDeviceToDevice));

    std::fprintf(stderr,
        "CartLagSolver: %dx%d cells, box=[%g,%g]x[%g,%g], γ=%g, CFL=%g\n",
        nx, ny, 0.0, Lx, 0.0, Ly, gamma, cfl);
}

void CartLagSolver::destroy() {
    auto f = [](double* p) { if (p) cudaFree(p); };
    f(d_X); f(d_Y); f(d_vX); f(d_vY); f(d_FX); f(d_FY);
    f(d_FX_hse); f(d_FY_hse);
    f(d_mnode); f(d_dX); f(d_dY);
    f(d_dm); f(d_Vol); f(d_Area0); f(d_rho); f(d_e_int);
    f(d_P); f(d_Q); f(d_cs); f(d_minheight); f(d_strain_rate);
    f(d_FSX); f(d_FSY);
    f(d_dt_cell); f(d_reduce_buf);
    std::memset(this, 0, sizeof(*this));
}

// ============================================================
// Sod shock tube IC
// Left  (x < Lx/2):  ρ = 1.0,   P = 1.0,    v = 0
// Right (x ≥ Lx/2):  ρ = 0.125, P = 0.1,    v = 0
// ============================================================
void CartLagSolver::init_sod() {
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
    k_clag_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nx, ny);

    std::fprintf(stderr, "  CartLag Sod IC loaded: ρL=1.0 PL=1.0 | ρR=0.125 PR=0.1\n");
}

// Uniform IC (useful for sanity tests — should stay stationary or translate cleanly)
void CartLagSolver::init_uniform(double rho, double P, double vx, double vy) {
    std::vector<double> h_Vol(ncell);
    CUDA_CHECK(cudaMemcpy(h_Vol.data(), d_Vol, ncell*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_dm(ncell, 0), h_e(ncell, 0);
    for (int c = 0; c < ncell; ++c) {
        h_dm[c] = rho * h_Vol[c];
        h_e[c]  = P / ((gamma - 1.0) * rho);
    }
    CUDA_CHECK(cudaMemcpy(d_dm, h_dm.data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_e_int, h_e.data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    std::vector<double> h_vX(nnode, vx), h_vY(nnode, vy);
    CUDA_CHECK(cudaMemcpy(d_vX, h_vX.data(), nnode*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vY, h_vY.data(), nnode*sizeof(double), cudaMemcpyHostToDevice));

    int B = 256;
    k_clag_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nx, ny);
    std::fprintf(stderr, "  CartLag Uniform IC: ρ=%g, P=%g, v=(%g,%g)\n", rho, P, vx, vy);
}

// ============================================================
// Polytropic HSE IC
//   ρ(y) = ρ_b · θ(y)^n,     n = 1/(γ-1),  θ = 1 - y/y_top
//   P(y) = K · ρ^γ,           K = (γ-1)/γ · g · y_top · ρ_b^(γ-1)
// Satisfies dP/dy = -ρ·g exactly (smooth solution; the discrete cell-averaged
// version has small O(h²) defect). Top of atmosphere at y = Ly (θ → 0).
// ============================================================
void CartLagSolver::init_hse_polytrope(double rho_base, double g_val, double amp) {
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
            // optional radial-style perturbation (sin in y, then adiabatic on e)
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
    k_clag_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nx, ny);

    std::fprintf(stderr,
        "  CartLag HSE polytrope: ρ_b=%g, g=%g, K=%g, y_top=%g, perturb=%g\n",
        rho_base, g_val, K, Ly, amp);
}

// ============================================================
// Capture HSE residual force so it can be subtracted each step.
// Must be called when state is at rest (v=0) and is meant to be HSE.
// ============================================================
void CartLagSolver::snapshot_hse_force() {
    int B = 256;
    int BCell = (ncell + B - 1) / B;
    int BNode = (nnode + B - 1) / B;

    CUDA_CHECK(cudaMemset(d_vX, 0, nnode*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_vY, 0, nnode*sizeof(double)));

    k_clag_geometry<<<BCell, B>>>(d_X, d_Y, d_Vol, d_minheight, nx, ny);
    k_clag_eos_and_q<<<BCell, B>>>(
        d_X, d_Y, d_vX, d_vY, d_dm, d_Vol, d_Area0, d_e_int,
        d_rho, d_P, d_Q, d_cs, d_strain_rate,
        nx, ny, gamma, CQ_lin, CQ_quad);

    k_clag_zero<<<BNode, B>>>(d_FX, nnode);
    k_clag_zero<<<BNode, B>>>(d_FY, nnode);
    k_clag_node_forces<<<BCell, B>>>(d_X, d_Y, d_P, d_Q,
                                     d_FX, d_FY, d_FSX, d_FSY, nx, ny);
    if (g_y != 0.0)
        k_clag_add_gravity<<<BNode, B>>>(d_mnode, d_FY, g_y, nnode);

    // Do NOT apply BC here — store the raw force; BC zeros wall-normal components
    // each step and this remains consistent since BC is applied post-subtraction.

    CUDA_CHECK(cudaMemcpy(d_FX_hse, d_FX, nnode*sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_FY_hse, d_FY, nnode*sizeof(double), cudaMemcpyDeviceToDevice));

    // Report max |F|/m so we see how big the discrete HSE defect was
    std::vector<double> h_FX(nnode), h_FY(nnode), h_m(nnode);
    CUDA_CHECK(cudaMemcpy(h_FX.data(), d_FX,    nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_FY.data(), d_FY,    nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_m.data(),  d_mnode, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    double max_a = 0.0;
    for (int n = 0; n < nnode; ++n) {
        if (h_m[n] <= 0.0) continue;
        double a = std::sqrt(h_FX[n]*h_FX[n] + h_FY[n]*h_FY[n]) / h_m[n];
        if (a > max_a) max_a = a;
    }
    hse_force_set = true;
    std::fprintf(stderr, "  CartLag HSE force snapshot: max|F|/m = %.3e\n", max_a);
}

// ============================================================
// One kick-drift-kick step
// ============================================================
double CartLagSolver::step(double t, double t_end) {
    int B = 256;
    int BCell = (ncell + B - 1) / B;
    int BNode = (nnode + B - 1) / B;

    k_clag_geometry<<<BCell, B>>>(d_X, d_Y, d_Vol, d_minheight, nx, ny);

    k_clag_eos_and_q<<<BCell, B>>>(
        d_X, d_Y, d_vX, d_vY, d_dm, d_Vol, d_Area0, d_e_int,
        d_rho, d_P, d_Q, d_cs, d_strain_rate,
        nx, ny, gamma, CQ_lin, CQ_quad);

    k_clag_cfl<<<BCell, B>>>(
        d_minheight, d_cs, d_strain_rate, d_vX, d_vY,
        nx, ny, cfl, comp_dt_frac, d_dt_cell);
    double dt_min = gpu_reduce_min(d_dt_cell, d_reduce_buf, ncell);
    double dt = (dt_min > 0.0 && dt_min < 1e29) ? dt_min : 1e-12;
    if (dt < 1e-30) dt = 1e-12;
    if (t + dt > t_end) dt = t_end - t;

    k_clag_zero<<<BNode, B>>>(d_FX, nnode);
    k_clag_zero<<<BNode, B>>>(d_FY, nnode);

    k_clag_node_forces<<<BCell, B>>>(d_X, d_Y, d_P, d_Q,
                                     d_FX, d_FY, d_FSX, d_FSY, nx, ny);

    // Gravity: F_y += -g · m_node  (added to nodes only; does NOT go through
    // the compatible subcell force path, so it acts as an external body force.
    // Energy update: gravitational PE is handled by diagnostics, not by
    // subtracting work here — gravity is *conservative* and transfers KE↔PE
    // automatically through the Pos + v update.)
    if (g_y != 0.0) {
        k_clag_add_gravity<<<BNode, B>>>(d_mnode, d_FY, g_y, nnode);
    }

    // HSE residual-force subtraction: F ← F - F_HSE.
    // Guarantees that the captured initial-state force field produces zero
    // acceleration. Valid because on a small neighborhood of HSE the force
    // varies smoothly in mesh position, so any true perturbation force is
    // preserved (the subtraction only removes the static "phantom" defect).
    if (hse_force_set) {
        k_fas_axpy_v<<<BNode, B>>>(d_FX, -1.0, d_FX_hse, nnode);
        k_fas_axpy_v<<<BNode, B>>>(d_FY, -1.0, d_FY_hse, nnode);
    }

    // BCs: zero normal component of force on wall edges
    k_clag_bc_reflective<<<BNode, B>>>(d_X, d_Y, d_vX, d_vY, d_FX, d_FY,
                                       g_Lx, g_Ly, nnode_x, nnode_y);

    k_clag_node_update<<<BNode, B>>>(d_X, d_Y, d_vX, d_vY, d_FX, d_FY,
                                     d_mnode, d_dX, d_dY, dt, nnode);

    // Re-apply BC after update (pin positions on walls)
    k_clag_bc_reflective<<<BNode, B>>>(d_X, d_Y, d_vX, d_vY, d_FX, d_FY,
                                       g_Lx, g_Ly, nnode_x, nnode_y);

    k_clag_energy_update<<<BCell, B>>>(nx, ny, d_FSX, d_FSY,
                                       d_dX, d_dY, d_dm, d_e_int);

    step_count++;
    dt_current = dt;
    if (step_count <= 10 || step_count % 500 == 0)
        std::fprintf(stderr, "  [cart_lag] step %d  t=%.4e  dt=%.3e\n",
                     step_count, t+dt, dt);
    return dt;
}

// ============================================================
// Diagnostics
// ============================================================
CartLagSolver::Diagnostics CartLagSolver::compute_diagnostics() {
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
        d.total_PE += h_m[n] * g_y * h_Y[n];  // PE = m·g·y (g_y = magnitude, pulling in -y)
    }
    d.total_E = d.total_KE + d.total_internal_E + d.total_PE;
    double cs_max = 0.0;
    for (int c = 0; c < ncell; ++c) cs_max = std::max(cs_max, h_cs[c]);
    d.max_mach = (cs_max > 0.0) ? d.max_v / cs_max : 0.0;
    return d;
}

// 1D slice along the horizontal midline (y ≈ Ly/2), volume-weighted over j.
void CartLagSolver::download_xslice(std::vector<double>& x,
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
