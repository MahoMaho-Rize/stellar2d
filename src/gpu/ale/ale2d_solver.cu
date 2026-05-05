// ALE2D orchestration: init, step, diagnostics, destroy.
// Kernels live in ale2d_kernels.cu.

#include "ale2d_solver.cuh"
#include "gpu_common.cuh"
#include "gpu_linalg.cuh"    // gpu_reduce_min
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

// ======= Forward declarations (definitions in ale2d_kernels.cu) =======
__global__ void k_ale_geometry(const double*, const double*, double*, double*, double*, int, int);
__global__ void k_ale_eos_and_q(const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, double*, int, int, double, double, double);
__global__ void k_ale_node_forces(const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, int, int);
__global__ void k_ale_zero_nodes(double*, int);
__global__ void k_ale_add_gravity(const double*, const double*, const double*, const double*,
    double*, double*, int, int, double);
__global__ void k_ale_node_update(double*, double*, double*, double*,
    const double*, const double*, const double*, double*, double*, double, int);
__global__ void k_ale_energy_update(const int, const int,
    const double*, const double*, const double*, const double*, const double*, double*);
__global__ void k_ale_bc_axis(double*, double*, double*, int, int);
__global__ void k_ale_bc_origin(double*, double*, double*, double*, double*, double*, int);
__global__ void k_ale_cfl(const double*, const double*, const double*, const double*,
    const double*, const double*, int, int, double, double, double*);
__global__ void k_ale_shell_mass(const double*, double*, int, int);
__global__ void k_ale_enclosed_mass(const double*, double*, int);
__global__ void k_ale_init_from_rth(const double*, const double*, double*, double*, int, int);
__global__ void k_ale_node_mass(const double*, double*, int, int);

// ============================================================
// init / destroy
// ============================================================
void Ale2DSolver::init(const Grid& grid, const EOS& eos, double G, double cfl_in) {
    gamma = eos.gamma;
    G_const = G;
    cfl = cfl_in;

    nr = grid.nr;
    nt = grid.ntheta;
    nnode_r = nr + 1;
    nnode_t = nt + 1;
    nnode = nnode_r * nnode_t;
    ncell = nr * nt;
    nsub = 4 * ncell;

    auto mal = [](double** p, size_t nbytes) {
        CUDA_CHECK(cudaMalloc(p, nbytes));
        CUDA_CHECK(cudaMemset(*p, 0, nbytes));
    };

    mal(&d_R,  nnode*sizeof(double));  mal(&d_Z,  nnode*sizeof(double));
    mal(&d_R_prev, nnode*sizeof(double)); mal(&d_Z_prev, nnode*sizeof(double));
    mal(&d_vR, nnode*sizeof(double));  mal(&d_vZ, nnode*sizeof(double));
    mal(&d_FR, nnode*sizeof(double));  mal(&d_FZ, nnode*sizeof(double));
    mal(&d_mnode, nnode*sizeof(double));

    mal(&d_dm, ncell*sizeof(double));
    mal(&d_Vol, ncell*sizeof(double));  mal(&d_Vol_prev, ncell*sizeof(double));
    mal(&d_Area, ncell*sizeof(double)); mal(&d_Area0, ncell*sizeof(double));
    mal(&d_rho, ncell*sizeof(double));
    mal(&d_e_int, ncell*sizeof(double)); mal(&d_e_prev, ncell*sizeof(double));
    mal(&d_P, ncell*sizeof(double));     mal(&d_Q, ncell*sizeof(double));
    mal(&d_cs, ncell*sizeof(double));
    mal(&d_minheight, ncell*sizeof(double));
    mal(&d_strain_rate, ncell*sizeof(double));

    mal(&d_rho0, ncell*sizeof(double));
    mal(&d_P0,   ncell*sizeof(double));
    mal(&d_e0,   ncell*sizeof(double));
    mal(&d_FR_hse, nnode*sizeof(double));
    mal(&d_FZ_hse, nnode*sizeof(double));

    mal(&d_FSR, nsub*sizeof(double));
    mal(&d_FSZ, nsub*sizeof(double));

    mal(&d_shell_mass, nr*sizeof(double));
    mal(&d_M_enc, nnode_r*sizeof(double));

    mal(&d_dt_cell, ncell*sizeof(double));
    mal(&d_reduce_buf, ncell*sizeof(double));
    mal(&d_reduce_out, ncell*sizeof(double));

    // Initial node positions from grid r_face × theta_face
    int nnr_bytes = nnode_r * sizeof(double);
    int nnt_bytes = nnode_t * sizeof(double);
    double *d_rf = nullptr, *d_tf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rf, nnr_bytes));
    CUDA_CHECK(cudaMalloc(&d_tf, nnt_bytes));
    CUDA_CHECK(cudaMemcpy(d_rf, grid.r_face.data(), nnr_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tf, grid.theta_face.data(), nnt_bytes, cudaMemcpyHostToDevice));
    {
        int B = 256;
        k_ale_init_from_rth<<<(nnode+B-1)/B, B>>>(
            d_rf, d_tf, d_R, d_Z, nnode_r, nnode_t);
    }
    cudaFree(d_rf); cudaFree(d_tf);

    // Compute Area/Vol/minheight for the *initial* mesh so we can grab Area0
    {
        int B = 256;
        k_ale_geometry<<<(ncell+B-1)/B, B>>>(
            d_R, d_Z, d_Vol, d_Area, d_minheight, nr, nt);
        CUDA_CHECK(cudaMemcpy(d_Area0, d_Area, ncell*sizeof(double),
                              cudaMemcpyDeviceToDevice));
    }

    std::fprintf(stderr,
        "Ale2DSolver initialized: %dx%d cells, %d nodes, γ=%g, CFL=%g, CQ_q=%g, CQ_l=%g\n",
        nr, nt, nnode, gamma, cfl, CQ_quad, CQ_lin);
}

void Ale2DSolver::destroy() {
    auto f = [](double* p) { if (p) cudaFree(p); };
    f(d_R); f(d_Z); f(d_R_prev); f(d_Z_prev);
    f(d_vR); f(d_vZ); f(d_FR); f(d_FZ); f(d_mnode);
    f(d_dm); f(d_Vol); f(d_Vol_prev); f(d_Area); f(d_Area0);
    f(d_rho); f(d_e_int); f(d_e_prev); f(d_P); f(d_Q); f(d_cs);
    f(d_minheight); f(d_strain_rate);
    f(d_rho0); f(d_P0); f(d_e0);
    f(d_FR_hse); f(d_FZ_hse);
    f(d_FSR); f(d_FSZ);
    f(d_shell_mass); f(d_M_enc);
    f(d_dt_cell); f(d_reduce_buf); f(d_reduce_out);
    std::memset(this, 0, sizeof(*this));
}

// ============================================================
// Lane-Emden initializer: compute ρ,P per cell (from theta_LE at cell centroid
// in spherical coords), then dm = ρ · Vol.
// The initial mesh is whatever Grid built (log/mass-shell/uniform).
// ============================================================

// Minimal Lane-Emden RK4 solver (shared with radial1d) — inlined to avoid
// coupling to src/init/lane_emden.h (which uses Grid+State).
struct AleLE {
    std::vector<double> xi, theta;
    double xi_1;
};

static AleLE ale_solve_lane_emden_cpu(double n_poly, double dxi = 1e-4) {
    AleLE sol;
    double xi = 1e-10, theta = 1.0, dtheta = 0.0;
    sol.xi.push_back(0.0); sol.theta.push_back(1.0);
    auto f2 = [&](double x, double t, double dt_val) -> double {
        if (x < 1e-10) return -t / 3.0;
        double tn = (t > 0) ? std::pow(t, n_poly) : 0.0;
        return -tn - 2.0 * dt_val / x;
    };
    while (theta > 0.0 && xi < 100.0) {
        double k1y1 = dtheta,                               k1y2 = f2(xi, theta, dtheta);
        double k2y1 = dtheta + 0.5*dxi*k1y2,                k2y2 = f2(xi+0.5*dxi, theta+0.5*dxi*k1y1, dtheta+0.5*dxi*k1y2);
        double k3y1 = dtheta + 0.5*dxi*k2y2,                k3y2 = f2(xi+0.5*dxi, theta+0.5*dxi*k2y1, dtheta+0.5*dxi*k2y2);
        double k4y1 = dtheta + dxi*k3y2,                    k4y2 = f2(xi+dxi, theta+dxi*k3y1, dtheta+dxi*k3y2);
        theta  += dxi/6.0 * (k1y1 + 2*k2y1 + 2*k3y1 + k4y1);
        dtheta += dxi/6.0 * (k1y2 + 2*k2y2 + 2*k3y2 + k4y2);
        xi += dxi;
        sol.xi.push_back(xi);
        sol.theta.push_back(std::max(theta, 0.0));
        if (theta <= 0.0) break;
    }
    sol.xi_1 = xi;
    return sol;
}

static double ale_interp_le(const AleLE& sol, double xi_val) {
    if (xi_val <= 0.0) return 1.0;
    if (xi_val >= sol.xi_1) return 0.0;
    auto it = std::lower_bound(sol.xi.begin(), sol.xi.end(), xi_val);
    int idx = (int)(it - sol.xi.begin());
    if (idx <= 0) return 1.0;
    if (idx >= (int)sol.xi.size()) return 0.0;
    double x0 = sol.xi[idx-1], x1 = sol.xi[idx];
    double t0 = sol.theta[idx-1], t1 = sol.theta[idx];
    double frac = (xi_val - x0) / (x1 - x0);
    return t0 + frac * (t1 - t0);
}

void Ale2DSolver::init_lane_emden(double rho_c, double K_poly, double n_poly) {
    AleLE le = ale_solve_lane_emden_cpu(n_poly);
    double alpha2 = (n_poly + 1.0) * K_poly * std::pow(rho_c, 1.0/n_poly - 1.0)
                    / (4.0 * M_PI * G_const);
    double alpha = std::sqrt(alpha2);

    // Download node positions, derive per-cell centroid r, then ρ/P/e_int.
    std::vector<double> h_R(nnode), h_Z(nnode);
    CUDA_CHECK(cudaMemcpy(h_R.data(), d_R, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Z.data(), d_Z, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_Vol(ncell), h_Area0(ncell);
    CUDA_CHECK(cudaMemcpy(h_Vol.data(),   d_Vol,   ncell*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Area0.data(), d_Area0, ncell*sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> h_dm(ncell), h_rho(ncell), h_P(ncell), h_e(ncell);
    for (int ic = 0; ic < nr; ++ic)
        for (int jc = 0; jc < nt; ++jc) {
            int flat = ic*nt + jc;
            // centroid in (R,Z) of the 4 corners
            int I[4] = {
                ic*nnode_t + jc, (ic+1)*nnode_t + jc,
                (ic+1)*nnode_t + (jc+1), ic*nnode_t + (jc+1)
            };
            double Rc = 0.25*(h_R[I[0]] + h_R[I[1]] + h_R[I[2]] + h_R[I[3]]);
            double Zc = 0.25*(h_Z[I[0]] + h_Z[I[1]] + h_Z[I[2]] + h_Z[I[3]]);
            double r = std::sqrt(Rc*Rc + Zc*Zc);
            double xi = r / alpha;
            double th_v = ale_interp_le(le, xi);
            double rho_v = rho_c * std::pow(std::max(th_v, 1e-15), n_poly);
            double P_v   = K_poly * std::pow(rho_v, 1.0 + 1.0/n_poly);
            double e_v   = P_v / ((gamma - 1.0) * std::max(rho_v, 1e-30));

            h_rho[flat] = rho_v;
            h_P[flat]   = P_v;
            h_e[flat]   = e_v;
            h_dm[flat]  = rho_v * std::max(h_Vol[flat], 1e-30);
        }

    CUDA_CHECK(cudaMemcpy(d_dm,    h_dm.data(),  ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rho,   h_rho.data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_P,     h_P.data(),   ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_e_int, h_e.data(),   ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_vR, 0, nnode*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_vZ, 0, nnode*sizeof(double)));

    // shell_mass and M_enc (constant for the whole run)
    int B = 256;
    k_ale_shell_mass<<<(nr+B-1)/B, B>>>(d_dm, d_shell_mass, nr, nt);
    k_ale_enclosed_mass<<<1, 1>>>(d_shell_mass, d_M_enc, nr);

    // Node masses: ¼ of each adjacent cell's dm
    k_ale_node_mass<<<(nnode+B-1)/B, B>>>(d_dm, d_mnode, nr, nt);

    double M_total;
    CUDA_CHECK(cudaMemcpy(&M_total, d_M_enc + nr, sizeof(double), cudaMemcpyDeviceToHost));

    // Surface pressure floor = outermost shell's initial P (equatorial column)
    int eq = nt / 2;
    P_surf_floor = h_P[(nr-1)*nt + eq];

    std::fprintf(stderr,
        "  Ale2D Lane-Emden: nr=%d nt=%d, M_total=%.6f, P_surf=%.3e\n",
        nr, nt, M_total, P_surf_floor);
}

void Ale2DSolver::apply_perturbation(double amplitude) {
    // Load state to host, apply sin(π r/R) perturbation on e_int (adiabatic),
    // keeping dm and geometry unchanged. This matches
    // init_lane_emden_perturbed in 2D (ρ *= 1+δ, P *= 1+γδ → e = P/((γ-1)ρ))
    std::vector<double> h_R(nnode), h_Z(nnode);
    CUDA_CHECK(cudaMemcpy(h_R.data(), d_R, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Z.data(), d_Z, nnode*sizeof(double), cudaMemcpyDeviceToHost));

    double R_star = 0.0;
    for (int jn = 0; jn < nnode_t; ++jn) {
        double rr = std::sqrt(h_R[nr*nnode_t + jn]*h_R[nr*nnode_t + jn]
                              + h_Z[nr*nnode_t + jn]*h_Z[nr*nnode_t + jn]);
        R_star = std::max(R_star, rr);
    }

    std::vector<double> h_e(ncell);
    CUDA_CHECK(cudaMemcpy(h_e.data(), d_e_int, ncell*sizeof(double), cudaMemcpyDeviceToHost));

    for (int ic = 0; ic < nr; ++ic)
        for (int jc = 0; jc < nt; ++jc) {
            int flat = ic*nt + jc;
            int I[4] = {
                ic*nnode_t + jc, (ic+1)*nnode_t + jc,
                (ic+1)*nnode_t + (jc+1), ic*nnode_t + (jc+1)
            };
            double Rc = 0.25*(h_R[I[0]] + h_R[I[1]] + h_R[I[2]] + h_R[I[3]]);
            double Zc = 0.25*(h_Z[I[0]] + h_Z[I[1]] + h_Z[I[2]] + h_Z[I[3]]);
            double r = std::sqrt(Rc*Rc + Zc*Zc);
            double delta = amplitude * std::sin(M_PI * r / R_star);
            // Adiabatic: P_new/ρ_new = P_old (1+γδ) / ρ_old(1+δ) ∝ e_new
            double e_new = h_e[flat] * (1.0 + gamma*delta) / (1.0 + delta);
            h_e[flat] = e_new;
        }
    CUDA_CHECK(cudaMemcpy(d_e_int, h_e.data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    std::fprintf(stderr, "  Ale2D: applied radial perturbation amp=%.3e\n", amplitude);
}

void Ale2DSolver::snapshot_hse() {
    int B = 256;
    int BCell = (ncell + B - 1) / B;
    int BNode = (nnode + B - 1) / B;

    // Evaluate full node force at current (assumed-HSE) state.
    // Zero v_R, v_Z to avoid any transient viscosity contribution.
    CUDA_CHECK(cudaMemset(d_vR, 0, nnode*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_vZ, 0, nnode*sizeof(double)));

    k_ale_geometry<<<BCell, B>>>(d_R, d_Z, d_Vol, d_Area, d_minheight, nr, nt);
    k_ale_eos_and_q<<<BCell, B>>>(
        d_R, d_Z, d_vR, d_vZ, d_dm, d_Vol, d_Area0, d_e_int,
        d_rho, d_P, d_Q, d_cs, d_strain_rate,
        nr, nt, gamma, CQ_lin, CQ_quad);

    k_ale_zero_nodes<<<BNode, B>>>(d_FR, nnode);
    k_ale_zero_nodes<<<BNode, B>>>(d_FZ, nnode);
    k_ale_node_forces<<<BCell, B>>>(d_R, d_Z, d_P, d_Q,
                                    d_FR, d_FZ, d_FSR, d_FSZ, nr, nt);
    k_ale_add_gravity<<<BNode, B>>>(d_R, d_Z, d_mnode, d_M_enc,
                                    d_FR, d_FZ, nnode_r, nnode_t, G_const);
    // Capture these as the HSE reference force so HSE stays exactly stationary.
    CUDA_CHECK(cudaMemcpy(d_FR_hse, d_FR, nnode*sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_FZ_hse, d_FZ, nnode*sizeof(double), cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaMemcpy(d_rho0, d_rho,   ncell*sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_P0,   d_P,     ncell*sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_e0,   d_e_int, ncell*sizeof(double), cudaMemcpyDeviceToDevice));

    // Print max |F|/m — the "numerical HSE defect" driving spurious motion.
    std::vector<double> h_FR(nnode), h_FZ(nnode), h_m(nnode);
    CUDA_CHECK(cudaMemcpy(h_FR.data(), d_FR,    nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_FZ.data(), d_FZ,    nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_m.data(),  d_mnode, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    double max_accel = 0.0, max_F = 0.0;
    int argmax = -1;
    for (int n = 0; n < nnode; ++n) {
        if (h_m[n] <= 0.0) continue;
        double Fmag = std::sqrt(h_FR[n]*h_FR[n] + h_FZ[n]*h_FZ[n]);
        double a = Fmag / h_m[n];
        if (a > max_accel) { max_accel = a; max_F = Fmag; argmax = n; }
    }
    int in_m = (argmax >= 0) ? (argmax / nnode_t) : -1;
    int jn_m = (argmax >= 0) ? (argmax % nnode_t) : -1;
    std::fprintf(stderr,
        "  Ale2D HSE defect: max|F|/m = %.3e  (|F|=%.3e, node in=%d jn=%d)\n",
        max_accel, max_F, in_m, jn_m);

    hse_set = true;
}

// ============================================================
// step: one explicit kick-drift-kick
// ============================================================
double Ale2DSolver::step(double t, double t_end) {
    int B = 256;
    int BCell = (ncell + B - 1) / B;
    int BNode = (nnode + B - 1) / B;

    // 1. Geometry (Vol, Area, minheight) from current node positions
    k_ale_geometry<<<BCell, B>>>(d_R, d_Z, d_Vol, d_Area, d_minheight, nr, nt);

    // 2. EOS + strain_rate + Q
    k_ale_eos_and_q<<<BCell, B>>>(
        d_R, d_Z, d_vR, d_vZ, d_dm, d_Vol, d_Area0, d_e_int,
        d_rho, d_P, d_Q, d_cs, d_strain_rate,
        nr, nt, gamma, CQ_lin, CQ_quad);

    // 3. CFL
    k_ale_cfl<<<BCell, B>>>(
        d_minheight, d_cs, d_strain_rate, d_Area0, d_vR, d_vZ,
        nr, nt, cfl, comp_dt_frac, d_dt_cell);
    double dt_min = gpu_reduce_min(d_dt_cell, d_reduce_buf, ncell);
    double dt = (dt_min > 0.0 && dt_min < 1e29) ? dt_min : 1e-12;
    if (dt < 1e-30) dt = 1e-12;
    if (t + dt > t_end) dt = t_end - t;

    // 4. Zero node force buffers
    k_ale_zero_nodes<<<BNode, B>>>(d_FR, nnode);
    k_ale_zero_nodes<<<BNode, B>>>(d_FZ, nnode);

    // 5. Accumulate subcell & node pressure+viscosity forces
    k_ale_node_forces<<<BCell, B>>>(d_R, d_Z, d_P, d_Q,
                                    d_FR, d_FZ, d_FSR, d_FSZ, nr, nt);

    // 6. Gravity (using constant shell mass; node-wise)
    k_ale_add_gravity<<<BNode, B>>>(d_R, d_Z, d_mnode, d_M_enc,
                                    d_FR, d_FZ, nnode_r, nnode_t, G_const);

    // (HSE subtraction removed: freezing force at initial positions is unsafe
    //  once the mesh moves. Proper fix is to make the discrete HSE exactly
    //  satisfied, which we debug below.)

    // 7. BCs: zero FR on axis, pin origin nodes
    {
        int Bi = std::min(nnode_r, 256);
        k_ale_bc_axis<<<(nnode_r+Bi-1)/Bi, Bi>>>(d_R, d_vR, d_FR, nnode_r, nnode_t);
    }
    {
        int Bj = std::min(nnode_t, 256);
        k_ale_bc_origin<<<(nnode_t+Bj-1)/Bj, Bj>>>(d_R, d_Z, d_vR, d_vZ,
                                                   d_FR, d_FZ, nnode_t);
    }

    // 8. Node update: writes displacement (dR, dZ) into d_FSR/d_FSZ's scratch space?
    //    We need per-node displacement arrays — reuse d_FR, d_FZ as scratch? No,
    //    these are forces. Allocate temporary via R_prev / Z_prev storage (yes:
    //    d_R_prev and d_Z_prev serve as displacement scratch).
    k_ale_node_update<<<BNode, B>>>(d_R, d_Z, d_vR, d_vZ, d_FR, d_FZ, d_mnode,
                                    d_R_prev, d_Z_prev, dt, nnode);

    // 9. Re-apply axis/origin BCs after update (R on axis must stay 0)
    {
        int Bi = std::min(nnode_r, 256);
        k_ale_bc_axis<<<(nnode_r+Bi-1)/Bi, Bi>>>(d_R, d_vR, d_FR, nnode_r, nnode_t);
    }
    {
        int Bj = std::min(nnode_t, 256);
        k_ale_bc_origin<<<(nnode_t+Bj-1)/Bj, Bj>>>(d_R, d_Z, d_vR, d_vZ,
                                                   d_FR, d_FZ, nnode_t);
    }

    // 10. Caramana-compatible energy: e_int -= Σ(F_sub · dPos_node) / dm
    k_ale_energy_update<<<BCell, B>>>(nr, nt, d_FSR, d_FSZ,
                                      d_R_prev, d_Z_prev, d_dm, d_e_int);

    step_count++;
    dt_current = dt;
    if (step_count <= 10 || step_count % 1000 == 0)
        std::fprintf(stderr, "  [ale2d] step %d  t=%.4e  dt=%.3e\n", step_count, t+dt, dt);
    return dt;
}

// ============================================================
// Diagnostics
// ============================================================
Ale2DSolver::Diagnostics Ale2DSolver::compute_diagnostics() {
    Diagnostics d{};

    std::vector<double> h_dm(ncell), h_e(ncell), h_rho(ncell), h_P(ncell);
    CUDA_CHECK(cudaMemcpy(h_dm.data(),  d_dm,    ncell*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_e.data(),   d_e_int, ncell*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rho.data(), d_rho,   ncell*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_P.data(),   d_P,     ncell*sizeof(double), cudaMemcpyDeviceToHost));

    for (int c = 0; c < ncell; ++c) {
        d.total_mass       += h_dm[c];
        d.total_internal_E += h_dm[c] * h_e[c];
    }

    std::vector<double> h_vR(nnode), h_vZ(nnode), h_m(nnode), h_R(nnode), h_Z(nnode);
    CUDA_CHECK(cudaMemcpy(h_vR.data(), d_vR,    nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vZ.data(), d_vZ,    nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_m.data(),  d_mnode, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_R.data(),  d_R,     nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Z.data(),  d_Z,     nnode*sizeof(double), cudaMemcpyDeviceToHost));

    for (int n = 0; n < nnode; ++n) {
        double v2 = h_vR[n]*h_vR[n] + h_vZ[n]*h_vZ[n];
        d.total_KE += 0.5 * h_m[n] * v2;
        double v = std::sqrt(v2);
        d.max_v = std::max(d.max_v, v);
    }

    // Gravitational PE via shells (stellar2d convention: PE = -½ G Σ dm_i M_enc(r_i)/r_i)
    std::vector<double> h_Menc(nnode_r);
    CUDA_CHECK(cudaMemcpy(h_Menc.data(), d_M_enc, nnode_r*sizeof(double), cudaMemcpyDeviceToHost));
    for (int ic = 0; ic < nr; ++ic) {
        // Use mid-radius of the shell: average of node positions on in=ic and in=ic+1
        double r_in = 0.0, r_out = 0.0;
        int cnt = 0;
        for (int jn = 0; jn < nnode_t; ++jn) {
            r_in  += std::sqrt(h_R[ic*nnode_t + jn]*h_R[ic*nnode_t + jn]
                             + h_Z[ic*nnode_t + jn]*h_Z[ic*nnode_t + jn]);
            r_out += std::sqrt(h_R[(ic+1)*nnode_t + jn]*h_R[(ic+1)*nnode_t + jn]
                             + h_Z[(ic+1)*nnode_t + jn]*h_Z[(ic+1)*nnode_t + jn]);
            cnt++;
        }
        r_in  /= std::max(cnt,1);
        r_out /= std::max(cnt,1);
        double r_mid = 0.5 * (r_in + r_out);
        double M_mid = 0.5 * (h_Menc[ic] + h_Menc[ic+1]);
        double shell_dm = 0.0;
        for (int jc = 0; jc < nt; ++jc) shell_dm += h_dm[ic*nt + jc];
        d.total_grav_E += -G_const * shell_dm * M_mid / std::max(r_mid, 1e-30);
    }
    d.total_grav_E *= 0.5;  // consistent with PE = -½ · self-interaction

    d.total_E = d.total_KE + d.total_internal_E + d.total_grav_E;

    double cs_max = 0.0;
    std::vector<double> h_cs(ncell);
    CUDA_CHECK(cudaMemcpy(h_cs.data(), d_cs, ncell*sizeof(double), cudaMemcpyDeviceToHost));
    for (int c = 0; c < ncell; ++c) cs_max = std::max(cs_max, h_cs[c]);
    d.max_mach = (cs_max > 0.0) ? d.max_v / cs_max : 0.0;
    return d;
}

// ============================================================
// Volume-weighted radial profile (for comparison with radial1d)
// ============================================================
void Ale2DSolver::download_radial_profile(
    std::vector<double>& r_cell,
    std::vector<double>& rho_cell,
    std::vector<double>& P_cell,
    std::vector<double>& e_cell,
    std::vector<double>& vr_cell)
{
    r_cell.assign(nr, 0.0); rho_cell.assign(nr, 0.0);
    P_cell.assign(nr, 0.0); e_cell.assign(nr, 0.0); vr_cell.assign(nr, 0.0);

    std::vector<double> h_R(nnode), h_Z(nnode), h_vR(nnode), h_vZ(nnode);
    CUDA_CHECK(cudaMemcpy(h_R.data(),  d_R,  nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Z.data(),  d_Z,  nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vR.data(), d_vR, nnode*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vZ.data(), d_vZ, nnode*sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> h_rho(ncell), h_P(ncell), h_e(ncell), h_Vol(ncell);
    CUDA_CHECK(cudaMemcpy(h_rho.data(), d_rho,   ncell*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_P.data(),   d_P,     ncell*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_e.data(),   d_e_int, ncell*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Vol.data(), d_Vol,   ncell*sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> w(nr, 0.0);
    for (int ic = 0; ic < nr; ++ic)
        for (int jc = 0; jc < nt; ++jc) {
            int flat = ic*nt + jc;
            double wv = h_Vol[flat];
            // cell centroid r
            int I[4] = {
                ic*nnode_t + jc, (ic+1)*nnode_t + jc,
                (ic+1)*nnode_t + (jc+1), ic*nnode_t + (jc+1)
            };
            double Rc = 0.25*(h_R[I[0]] + h_R[I[1]] + h_R[I[2]] + h_R[I[3]]);
            double Zc = 0.25*(h_Z[I[0]] + h_Z[I[1]] + h_Z[I[2]] + h_Z[I[3]]);
            double r = std::sqrt(Rc*Rc + Zc*Zc);
            // radial velocity at cell center (average of node radial velocities)
            double vr_cc = 0.0;
            for (int k = 0; k < 4; ++k) {
                double r_n = std::sqrt(h_R[I[k]]*h_R[I[k]] + h_Z[I[k]]*h_Z[I[k]]);
                if (r_n > 1e-14)
                    vr_cc += 0.25 * (h_vR[I[k]]*h_R[I[k]]/r_n + h_vZ[I[k]]*h_Z[I[k]]/r_n);
            }
            r_cell[ic]   += wv * r;
            rho_cell[ic] += wv * h_rho[flat];
            P_cell[ic]   += wv * h_P[flat];
            e_cell[ic]   += wv * h_e[flat];
            vr_cell[ic]  += wv * vr_cc;
            w[ic] += wv;
        }
    for (int ic = 0; ic < nr; ++ic) {
        double winv = (w[ic] > 0.0) ? 1.0 / w[ic] : 0.0;
        r_cell[ic]   *= winv;
        rho_cell[ic] *= winv;
        P_cell[ic]   *= winv;
        e_cell[ic]   *= winv;
        vr_cell[ic]  *= winv;
    }
}
