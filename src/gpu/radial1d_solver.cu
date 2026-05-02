// 1D Lagrangian radial stellar hydrodynamics solver (MESA RSP-inspired).
//
// Explicit RK2 time integration with:
//   - Lagrangian mass coordinates (exact mass conservation)
//   - Tscharnuter-Winkler shock-triggered artificial viscosity
//   - Compression-limited dt (not just acoustic CFL)
//   - Surface pressure floor
//   - Pinned center (r=0, v=0)

#include "radial1d_solver.cuh"
#include "radial1d_kernels.cuh"
#include "../physics/radiation_diffusion.cuh"
#include "fas_common.cuh"      // CUDA_CHECK macro
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

// --- tree-reduction helpers: min of a float array, sum of a float array ---
static double gpu_reduce_min_simple(const double* d_arr, int n) {
    std::vector<double> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d_arr, n*sizeof(double), cudaMemcpyDeviceToHost));
    double m = h[0];
    for (int i = 1; i < n; ++i) if (h[i] < m) m = h[i];
    return m;
}
static double gpu_reduce_sum_simple(const double* d_arr, int n) {
    std::vector<double> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d_arr, n*sizeof(double), cudaMemcpyDeviceToHost));
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += h[i];
    return s;
}
static double gpu_reduce_max_simple(const double* d_arr, int n) {
    std::vector<double> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d_arr, n*sizeof(double), cudaMemcpyDeviceToHost));
    double m = h[0];
    for (int i = 1; i < n; ++i) if (h[i] > m) m = h[i];
    return m;
}

// ============================================================
// Lane-Emden ODE solver (RK4).  Returns θ(ξ) and ξ_1 where θ = 0.
// ============================================================
struct LaneEmden {
    std::vector<double> xi, theta;
    double xi_1;
};

static LaneEmden solve_lane_emden_cpu(double n_poly, double dxi = 1e-4) {
    LaneEmden sol;
    double xi = 1e-10, theta = 1.0, dtheta = 0.0;
    sol.xi.push_back(0.0);
    sol.theta.push_back(1.0);
    auto f2 = [&](double x, double t, double dt_val) -> double {
        if (x < 1e-10) return -t / 3.0;
        double tn = (t > 0) ? std::pow(t, n_poly) : 0.0;
        return -tn - 2.0 * dt_val / x;
    };
    while (theta > 0.0 && xi < 100.0) {
        double k1y1 = dtheta;
        double k1y2 = f2(xi, theta, dtheta);
        double k2y1 = dtheta + 0.5*dxi*k1y2;
        double k2y2 = f2(xi+0.5*dxi, theta+0.5*dxi*k1y1, dtheta+0.5*dxi*k1y2);
        double k3y1 = dtheta + 0.5*dxi*k2y2;
        double k3y2 = f2(xi+0.5*dxi, theta+0.5*dxi*k2y1, dtheta+0.5*dxi*k2y2);
        double k4y1 = dtheta + dxi*k3y2;
        double k4y2 = f2(xi+dxi, theta+dxi*k3y1, dtheta+dxi*k3y2);
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

static double interp_le(const LaneEmden& sol, double xi_val) {
    if (xi_val <= 0.0) return 1.0;
    if (xi_val >= sol.xi_1) return 0.0;
    auto it = std::lower_bound(sol.xi.begin(), sol.xi.end(), xi_val);
    int idx = (int)(it - sol.xi.begin());
    if (idx == 0) idx = 1;
    if (idx >= (int)sol.xi.size()) return 0.0;
    double x0 = sol.xi[idx-1], x1 = sol.xi[idx];
    double t0 = sol.theta[idx-1], t1 = sol.theta[idx];
    double t = (xi_val - x0) / (x1 - x0);
    return t0 + t * (t1 - t0);
}

// ============================================================
// init: allocate device memory for given nz
// ============================================================
void Radial1DSolver::init(int nz_in, double gam, double G, double cfl_in) {
    lev.nz = nz_in;
    gamma = gam;
    G_const = G;
    cfl = cfl_in;

    int nf = nz_in + 1;  // faces
    int nz = nz_in;

    auto mal = [](double** p, int n) {
        CUDA_CHECK(cudaMalloc(p, n*sizeof(double)));
        CUDA_CHECK(cudaMemset(*p, 0, n*sizeof(double)));
    };
    mal(&lev.d_r, nf);       mal(&lev.d_v, nf);       mal(&lev.d_M, nf);
    mal(&lev.d_gr, nf);      mal(&lev.d_r_prev, nf);  mal(&lev.d_v_prev, nf);
    mal(&lev.d_dm, nz);      mal(&lev.d_Vol, nz);     mal(&lev.d_rho, nz);
    mal(&lev.d_e_int, nz);   mal(&lev.d_P, nz);       mal(&lev.d_Pvsc, nz);
    mal(&lev.d_rhoE, nz);    mal(&lev.d_Vol_prev, nz);mal(&lev.d_e_prev, nz);
    mal(&lev.d_rho0, nz);    mal(&lev.d_P0, nz);      mal(&lev.d_r0, nf);
    mal(&lev.d_dt_cell, nz); mal(&lev.d_scratch, std::max(nf, nz));

    // Radiation diffusion scratch (allocated regardless; tiny)
    mal(&d_T_work, nz);
    mal(&d_F_work, nf);
    mal(&d_dt_rad, nz);

    std::fprintf(stderr, "Radial1DSolver initialized: nz=%d zones, γ=%g, CFL=%g, CQ=%g, ZSH=%g\n",
                 nz, gamma, cfl, CQ, ZSH);
}

void Radial1DSolver::destroy() {
    auto f = [](double* p) { if (p) cudaFree(p); };
    f(lev.d_r); f(lev.d_v); f(lev.d_M); f(lev.d_gr); f(lev.d_r_prev); f(lev.d_v_prev);
    f(lev.d_dm); f(lev.d_Vol); f(lev.d_rho); f(lev.d_e_int); f(lev.d_P); f(lev.d_Pvsc);
    f(lev.d_rhoE); f(lev.d_Vol_prev); f(lev.d_e_prev);
    f(lev.d_rho0); f(lev.d_P0); f(lev.d_r0);
    f(lev.d_dt_cell); f(lev.d_scratch);
    f(d_T_work); f(d_F_work); f(d_dt_rad);
    std::memset(&lev, 0, sizeof(lev));
}

// ============================================================
// Lane-Emden initialization on equal-mass shells.
// Strategy: solve Lane-Emden ODE to get ρ(r), then integrate mass
// M(r) = ∫4πr²ρdr, invert to get r(M) on uniform mass grid.
// ============================================================
void Radial1DSolver::init_lane_emden(double rho_c, double K_poly, double n_poly) {
    // 1. Solve Lane-Emden ODE
    LaneEmden le = solve_lane_emden_cpu(n_poly);
    double alpha2 = (n_poly + 1.0) * K_poly * std::pow(rho_c, 1.0/n_poly - 1.0)
                    / (4.0 * M_PI * G_const);
    double alpha = std::sqrt(alpha2);
    double R_star = alpha * le.xi_1;

    // 2. Build fine radial grid, compute M(r) and ρ(r)
    int nfine = 20000;
    std::vector<double> r_fine(nfine+1), M_fine(nfine+1), rho_fine(nfine+1);
    double dr_f = R_star / nfine;
    for (int i = 0; i <= nfine; ++i) {
        double r = i * dr_f;
        r_fine[i] = r;
        double xi = r / alpha;
        double theta_v = interp_le(le, xi);
        rho_fine[i] = rho_c * std::pow(std::max(theta_v, 1e-15), n_poly);
    }
    M_fine[0] = 0.0;
    for (int i = 1; i <= nfine; ++i) {
        double rmid = 0.5 * (r_fine[i-1] + r_fine[i]);
        double rho_mid = 0.5 * (rho_fine[i-1] + rho_fine[i]);
        double dV = 4.0 * M_PI * rmid * rmid * dr_f;
        M_fine[i] = M_fine[i-1] + rho_mid * dV;
    }
    double M_total = M_fine[nfine];

    // 3. Build Lagrangian mass shells: equal-mass dm = M_total / nz
    int nz = lev.nz;
    std::vector<double> h_r(nz+1), h_dm(nz), h_rho(nz), h_P(nz), h_e(nz);
    h_r[0] = 0.0;
    double dm = M_total / nz;
    std::vector<double> M_target(nz+1);
    for (int k = 0; k <= nz; ++k) M_target[k] = k * dm;

    // Invert M(r) to find r at each target mass: linear search (nfine large enough)
    int ifine = 0;
    for (int k = 1; k <= nz; ++k) {
        while (ifine < nfine && M_fine[ifine+1] < M_target[k]) ifine++;
        // Linear interpolation between ifine and ifine+1
        double M0 = M_fine[ifine], M1 = M_fine[ifine+1];
        double r0 = r_fine[ifine], r1 = r_fine[ifine+1];
        double t = (M1 > M0) ? (M_target[k] - M0) / (M1 - M0) : 0.0;
        h_r[k] = r0 + t * (r1 - r0);
    }
    // Enforce exact surface radius
    h_r[nz] = R_star;

    // 4. Compute zone-centered quantities from r-grid
    for (int k = 0; k < nz; ++k) {
        double rL = h_r[k], rR = h_r[k+1];
        double V_k = (4.0*M_PI/3.0) * (rR*rR*rR - rL*rL*rL);
        h_dm[k] = dm;
        h_rho[k] = dm / V_k;
        h_P[k] = K_poly * std::pow(h_rho[k], 1.0 + 1.0/n_poly);
        h_e[k] = h_P[k] / ((gamma - 1.0) * h_rho[k]);  // specific internal energy
    }

    // Upload
    CUDA_CHECK(cudaMemcpy(lev.d_r,   h_r.data(),   (nz+1)*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_r0,  h_r.data(),   (nz+1)*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_dm,  h_dm.data(),  nz*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_e_int, h_e.data(), nz*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_rho0, h_rho.data(),nz*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_P0,   h_P.data(),  nz*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(lev.d_v, 0, (nz+1)*sizeof(double)));

    // Initial surface pressure floor = initial surface P (tiny but > 0)
    P_surf_floor = h_P[nz-1];
    hse_set = true;

    std::fprintf(stderr, "  Lane-Emden init: nz=%d zones, M_star=%.6f, R_star=%.6f, ρ_c=%.4e, P_surf_floor=%.3e\n",
                 nz, M_total, R_star, rho_c, P_surf_floor);
}

// ============================================================
// Apply radial-only perturbation (matching init_lane_emden_perturbed in 2D code):
//   ρ *= (1 + A sin(π r/R))
//   P *= (1 + γ A sin(π r/R))      (adiabatic)
// Zone k uses r_center = 0.5*(r[k] + r[k+1]).
// ============================================================
void Radial1DSolver::apply_perturbation(double amplitude) {
    int nz = lev.nz;
    std::vector<double> h_r(nz+1), h_rho(nz), h_P(nz), h_dm(nz);
    CUDA_CHECK(cudaMemcpy(h_r.data(),   lev.d_r,   (nz+1)*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rho.data(), lev.d_rho0,nz*sizeof(double),     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_P.data(),   lev.d_P0,  nz*sizeof(double),     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dm.data(),  lev.d_dm,  nz*sizeof(double),     cudaMemcpyDeviceToHost));

    double R_star = h_r[nz];
    // In 2D test, perturbation is ρ *= (1+δ), P *= (1+γδ),
    // which means specific internal energy e = P/(γ-1)/ρ changes as:
    //   e_new = P_old*(1+γδ) / ((γ-1) * ρ_old*(1+δ))
    //
    // But in Lagrangian form, dm is fixed and ρ = dm/V, so changing ρ means
    // changing V, i.e. moving the face radii. A fully-consistent adiabatic
    // compression would shift face positions.
    //
    // Approach: perturb ρ via volume change — we adjust r_face so that
    // the zone density matches the target. Simpler approximation:
    // perturb only e (specific internal energy) via P_new/ρ_new.
    //
    // For small amplitudes this is equivalent. Let's do it by re-solving
    // faces: r[k] is moved to enclose the perturbed mass distribution.
    //
    // But since dm is constant and ρ is perturbed, V_new = dm/ρ_new.
    // We can solve r[k+1] from V[k] using forward sweep:
    //   r[k+1]³ = r[k]³ + 3 V[k] / (4π)
    // with r[0] = 0.

    std::vector<double> h_rho_new(nz), h_P_new(nz), h_V_new(nz), h_r_new(nz+1);
    for (int k = 0; k < nz; ++k) {
        double r_cell = 0.5 * (h_r[k] + h_r[k+1]);
        double delta = amplitude * std::sin(M_PI * r_cell / R_star);
        h_rho_new[k] = h_rho[k] * (1.0 + delta);
        h_P_new[k]   = h_P[k]   * (1.0 + gamma * delta);
        h_V_new[k]   = h_dm[k] / h_rho_new[k];
    }
    h_r_new[0] = 0.0;
    for (int k = 0; k < nz; ++k) {
        double r3_new = h_r_new[k]*h_r_new[k]*h_r_new[k] + 3.0*h_V_new[k]/(4.0*M_PI);
        h_r_new[k+1] = std::cbrt(r3_new);
    }

    std::vector<double> h_e_new(nz);
    for (int k = 0; k < nz; ++k) {
        h_e_new[k] = h_P_new[k] / ((gamma - 1.0) * h_rho_new[k]);
    }

    CUDA_CHECK(cudaMemcpy(lev.d_r,     h_r_new.data(), (nz+1)*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_e_int, h_e_new.data(), nz*sizeof(double),     cudaMemcpyHostToDevice));

    std::fprintf(stderr, "  Applied radial perturbation: amp=%.3e, R_new=%.6f (was %.6f)\n",
                 amplitude, h_r_new[nz], R_star);
}

// ============================================================
// Snapshot current state as HSE reference
// Helper: dispatch primitives to γ or EOS kernel
static void launch_primitives(const Radial1DLevel& lev, int nz, bool use_eos,
                              double gamma, EOS eos, int B)
{
    if (use_eos) {
        k_rad1d_zone_primitives_eos<<<(nz+B-1)/B, B>>>(
            lev.d_r, lev.d_dm, lev.d_e_int, lev.d_Vol, lev.d_rho, lev.d_P, nz, eos);
    } else {
        launch_primitives(lev, nz, use_eos, gamma, eos, B);
    }
}

// ============================================================
void Radial1DSolver::snapshot_hse() {
    int nz = lev.nz;
    // compute primitives so d_rho, d_P are fresh
    int B = 256;
    launch_primitives(lev, nz, use_eos, gamma, eos, B);
    CUDA_CHECK(cudaMemcpy(lev.d_rho0, lev.d_rho, nz*sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_P0,   lev.d_P,   nz*sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_r0,   lev.d_r,   (nz+1)*sizeof(double), cudaMemcpyDeviceToDevice));

    // Surface pressure floor = initial surface pressure (small but positive)
    double h_Psurf;
    CUDA_CHECK(cudaMemcpy(&h_Psurf, lev.d_P0 + (nz-1), sizeof(double), cudaMemcpyDeviceToHost));
    P_surf_floor = h_Psurf;
    hse_set = true;
    std::fprintf(stderr, "  HSE snapshot taken: P_surf_floor=%.3e\n", P_surf_floor);
}

// ============================================================
// Compute dt from acoustic CFL and compression limit
// ============================================================
double Radial1DSolver::compute_dt() {
    int nz = lev.nz, B = 256;
    // Ensure primitives (V, ρ, P) are fresh
    launch_primitives(lev, nz, use_eos, gamma, eos, B);
    if (use_eos) {
        k_rad1d_cfl_eos<<<(nz+B-1)/B, B>>>(
            lev.d_r, lev.d_v, lev.d_rho, lev.d_P, lev.d_dt_cell,
            nz, eos, comp_dt_fraction);
    } else {
        k_rad1d_cfl<<<(nz+B-1)/B, B>>>(
            lev.d_r, lev.d_v, lev.d_rho, lev.d_P, lev.d_dt_cell,
            nz, gamma, comp_dt_fraction);
    }
    return cfl * gpu_reduce_min_simple(lev.d_dt_cell, nz);
}

// ============================================================
// One RK2 time step.
// ============================================================
double Radial1DSolver::step(double t, double t_end) {
    int nz = lev.nz, nf = nz + 1;
    int B = 256;

    // Fresh primitives & dt
    double dt = compute_dt();
    if (dt < 1e-30) dt = 1e-12;
    if (t + dt > t_end) dt = t_end - t;

    // Enclosed mass + gravity (stays fixed during step: dm doesn't change)
    k_rad1d_enclosed_mass<<<1, 1>>>(lev.d_dm, lev.d_M, nz);
    k_rad1d_gravity<<<(nf+B-1)/B, B>>>(lev.d_r, lev.d_M, lev.d_gr, nz, G_const);

    // Save state
    k_rad1d_save_state<<<(nf+B-1)/B, B>>>(lev.d_r, lev.d_r_prev, lev.d_v, lev.d_v_prev, nf);
    k_rad1d_save_cells<<<(nz+B-1)/B, B>>>(lev.d_Vol, lev.d_Vol_prev, lev.d_e_int, lev.d_e_prev, nz);

    // Pre-step: ensure current primitives fresh (done by compute_dt above)
    k_rad1d_artificial_viscosity<<<(nz+B-1)/B, B>>>(
        lev.d_v, lev.d_Vol, lev.d_P, lev.d_Pvsc, nz, CQ, ZSH);

    // ===== RK2 Stage 1 =====
    // Momentum with current P, Pvsc → new v*
    k_rad1d_momentum_update<<<(nf+B-1)/B, B>>>(
        lev.d_r, lev.d_dm, lev.d_P, lev.d_Pvsc, lev.d_gr, lev.d_v,
        nz, P_surf_floor, dt);
    // Positions with new v*
    k_rad1d_position_update<<<(nf+B-1)/B, B>>>(lev.d_v, lev.d_r, nz, dt);
    // Volumes/densities with new r (but keep old P, Pvsc for energy work)
    {
        int B_ = B;
        launch_primitives(lev, nz, use_eos, gamma, eos, B_);
    }
    // ^ This overwrote P. For a perfectly adiabatic update we should use
    // the PRE-step P (saved in d_P before position move). Since ZonePrimitives
    // computes P from the *old* e_int (we haven't updated it yet), the
    // pressure it produces is not the pre-step one nor the new one. For this
    // first version we accept this approximation; small-dt it's O(dt²).
    // TODO: add d_P_prev buffer.
    k_rad1d_energy_update<<<(nz+B-1)/B, B>>>(
        lev.d_Vol, lev.d_Vol_prev, lev.d_P, lev.d_Pvsc, lev.d_dm, lev.d_e_int, nz, dt);
    // Refresh primitives with new e_int
    launch_primitives(lev, nz, use_eos, gamma, eos, B);

    // ===== RK2 Stage 2: recompute R at U*, apply dt again =====
    k_rad1d_enclosed_mass<<<1, 1>>>(lev.d_dm, lev.d_M, nz);
    k_rad1d_gravity<<<(nf+B-1)/B, B>>>(lev.d_r, lev.d_M, lev.d_gr, nz, G_const);
    k_rad1d_artificial_viscosity<<<(nz+B-1)/B, B>>>(
        lev.d_v, lev.d_Vol, lev.d_P, lev.d_Pvsc, nz, CQ, ZSH);

    // Save V* before the next update
    k_rad1d_save_cells<<<(nz+B-1)/B, B>>>(lev.d_Vol, lev.d_Vol_prev, lev.d_e_int, lev.d_e_prev, nz);

    k_rad1d_momentum_update<<<(nf+B-1)/B, B>>>(
        lev.d_r, lev.d_dm, lev.d_P, lev.d_Pvsc, lev.d_gr, lev.d_v,
        nz, P_surf_floor, dt);
    k_rad1d_position_update<<<(nf+B-1)/B, B>>>(lev.d_v, lev.d_r, nz, dt);
    launch_primitives(lev, nz, use_eos, gamma, eos, B);
    k_rad1d_energy_update<<<(nz+B-1)/B, B>>>(
        lev.d_Vol, lev.d_Vol_prev, lev.d_P, lev.d_Pvsc, lev.d_dm, lev.d_e_int, nz, dt);

    // RK2 average: U^{n+1} = 0.5 (U^n + U**)
    k_rad1d_rk_average_faces<<<(nf+B-1)/B, B>>>(
        lev.d_r_prev, lev.d_r, lev.d_v_prev, lev.d_v, nz);
    k_rad1d_rk_average_cells<<<(nz+B-1)/B, B>>>(lev.d_e_prev, lev.d_e_int, nz);

    // Final primitives
    launch_primitives(lev, nz, use_eos, gamma, eos, B);

    // Radiation diffusion (operator split, after hydro)
    int rad_sub = 0;
    if (radiation_enabled) {
        rad_sub = apply_radiation_diffusion(dt);
        // Refresh primitives after e_int update
        launch_primitives(lev, nz, use_eos, gamma, eos, B);
    }

    step_count++;
    dt_current = dt;
    if (step_count <= 10 || step_count % 1000 == 0) {
        if (radiation_enabled)
            std::fprintf(stderr, "  [radial1d] step %d  t=%.4e  dt=%.3e  rad_sub=%d\n",
                         step_count, t+dt, dt, rad_sub);
        else
            std::fprintf(stderr, "  [radial1d] step %d  t=%.4e  dt=%.3e\n", step_count, t+dt, dt);
    }
    return dt;
}

// ============================================================
// Download profile
// ============================================================
void Radial1DSolver::download_profile(
    std::vector<double>& r_face, std::vector<double>& v_face,
    std::vector<double>& rho_cell, std::vector<double>& P_cell,
    std::vector<double>& e_cell)
{
    int nz = lev.nz;
    r_face.resize(nz+1);
    v_face.resize(nz+1);
    rho_cell.resize(nz);
    P_cell.resize(nz);
    e_cell.resize(nz);
    CUDA_CHECK(cudaMemcpy(r_face.data(),  lev.d_r,     (nz+1)*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_face.data(),  lev.d_v,     (nz+1)*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rho_cell.data(),lev.d_rho,   nz*sizeof(double),     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(P_cell.data(),  lev.d_P,     nz*sizeof(double),     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(e_cell.data(),  lev.d_e_int, nz*sizeof(double),     cudaMemcpyDeviceToHost));
}

// ============================================================
// Diagnostics
// ============================================================
Radial1DSolver::Diagnostics Radial1DSolver::compute_diagnostics() {
    int nz = lev.nz, B = 256;
    // Make sure primitives + enclosed mass are fresh
    launch_primitives(lev, nz, use_eos, gamma, eos, B);
    k_rad1d_enclosed_mass<<<1, 1>>>(lev.d_dm, lev.d_M, nz);

    // 5 scratch arrays of size nz — reuse existing buffers where safe
    std::vector<double*> scratch(6);
    for (int i = 0; i < 6; ++i) CUDA_CHECK(cudaMalloc(&scratch[i], nz*sizeof(double)));

    if (use_eos) {
        k_rad1d_diag_per_zone_eos<<<(nz+B-1)/B, B>>>(
            lev.d_dm, lev.d_e_int, lev.d_rho, lev.d_P, lev.d_v, lev.d_M, lev.d_r,
            scratch[0], scratch[1], scratch[2], scratch[3], scratch[4], scratch[5],
            nz, eos, G_const);
    } else {
        k_rad1d_diag_per_zone<<<(nz+B-1)/B, B>>>(
            lev.d_dm, lev.d_e_int, lev.d_rho, lev.d_P, lev.d_v, lev.d_M, lev.d_r,
            scratch[0], scratch[1], scratch[2], scratch[3], scratch[4], scratch[5],
            nz, gamma, G_const);
    }

    Diagnostics d;
    d.total_mass      = gpu_reduce_sum_simple(scratch[0], nz);
    d.total_KE        = gpu_reduce_sum_simple(scratch[1], nz);
    d.total_internal_E= gpu_reduce_sum_simple(scratch[2], nz);
    d.total_grav_E    = gpu_reduce_sum_simple(scratch[3], nz);
    d.total_E         = d.total_KE + d.total_internal_E + d.total_grav_E;
    d.max_mach        = gpu_reduce_max_simple(scratch[4], nz);
    d.max_vr          = gpu_reduce_max_simple(scratch[5], nz);

    for (int i = 0; i < 6; ++i) cudaFree(scratch[i]);
    return d;
}

// ============================================================
// Explicit radiation diffusion: subcycle at parabolic CFL
// ============================================================
int Radial1DSolver::apply_radiation_diffusion(double dt_total) {
    if (!radiation_enabled) return 0;
    int nz = lev.nz;
    int nf = nz + 1;
    int B = 256;

    RadDiffParams pars;
    pars.c_light = rad_c_light;
    pars.a_rad   = rad_a_rad;
    pars.opacity.kappa_es      = rad_kappa_es;
    pars.opacity.kappa_ff_0    = rad_kappa_ff_0;
    pars.opacity.kappa_dust_0  = rad_kappa_dust_0;
    pars.opacity.kappa_Hm_0    = rad_kappa_Hm_0;
    pars.opacity.T_dust_off    = rad_T_dust_off;

    double t_rad = 0.0;
    int sub = 0;
    const int max_sub = 100000;  // safety cap to avoid infinite subcycle
    while (t_rad < dt_total && sub < max_sub) {
        // Phase 1: T from EOS
        k_rad_diffusion_1d<<<(nz+B-1)/B, B>>>(
            lev.d_e_int, lev.d_rho, lev.d_r, lev.d_dm,
            d_T_work, d_F_work, nz, eos, pars, 0.0);
        // Determine dt_sub from parabolic CFL
        k_rad_diffusion_dt<<<(nz+B-1)/B, B>>>(
            lev.d_rho, d_T_work, lev.d_r, d_dt_rad, nz, pars);
        double dt_sub = gpu_reduce_min_simple(d_dt_rad, nz);
        if (dt_sub > dt_total - t_rad) dt_sub = dt_total - t_rad;
        if (dt_sub <= 0.0) break;

        // Phase 2: flux at faces
        k_rad_diffusion_1d_flux<<<(nf+B-1)/B, B>>>(
            d_T_work, lev.d_rho, lev.d_r, d_F_work, nz, eos, pars);
        // Phase 3: update e_int
        k_rad_diffusion_1d_update<<<(nz+B-1)/B, B>>>(
            lev.d_e_int, d_F_work, lev.d_dm, nz, dt_sub);

        t_rad += dt_sub;
        ++sub;
    }
    return sub;
}
