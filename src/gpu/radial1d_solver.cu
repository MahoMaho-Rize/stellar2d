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
    // Species arrays (allocated regardless; only used when species_enabled)
    mal(&d_X, nz);
    mal(&d_Y, nz);

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
    f(d_X); f(d_Y);
    d_X = nullptr; d_Y = nullptr;
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
        // Specific internal energy must be consistent with the EOS actually
        // in use. For ideal gas this is P/((γ−1)ρ); for ideal_rad / PRE_MS
        // we invert the EOS (P, ρ) → e. Otherwise the γ-only guess produces
        // grossly wrong T in the radiation-dominated regime.
        if (use_eos) {
            h_e[k] = eos.internal_energy(h_rho[k], h_P[k]);
        } else {
            h_e[k] = h_P[k] / ((gamma - 1.0) * h_rho[k]);
        }
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
        k_rad1d_zone_primitives<<<(nz+B-1)/B, B>>>(
            lev.d_r, lev.d_dm, lev.d_e_int, lev.d_Vol, lev.d_rho, lev.d_P, nz, gamma);
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

    // Nuclear burning (operator split; adds ε_pp · dt to e_int)
    if (nuclear_enabled && use_eos) {
        NuclearPPParams npars;
        npars.X_hydrogen = nuc_X;
        npars.T_floor = nuc_T_floor;
        npars.T_scale = nuc_T_scale;
        npars.epsilon_scale = nuc_epsilon_scale;
        npars.q_burn = nuc_q_burn;
        if (species_enabled) {
            k_rad1d_nuclear_pp_species<<<(nz+B-1)/B, B>>>(
                lev.d_e_int, d_X, d_Y, lev.d_rho, nz, eos, npars, dt);
        } else {
            k_rad1d_nuclear_pp<<<(nz+B-1)/B, B>>>(
                lev.d_e_int, lev.d_rho, nz, eos, npars, dt);
        }
        launch_primitives(lev, nz, use_eos, gamma, eos, B);
    }

    // Radiation diffusion (operator split, after hydro + burning)
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
// Species init + download
// ============================================================
void Radial1DSolver::init_species_uniform(double X0, double Y0) {
    int nz = lev.nz;
    std::vector<double> hX(nz, X0), hY(nz, Y0);
    CUDA_CHECK(cudaMemcpy(d_X, hX.data(), nz*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, hY.data(), nz*sizeof(double), cudaMemcpyHostToDevice));
    std::fprintf(stderr, "  Species init: X0=%.3f, Y0=%.3f (uniform)\n", X0, Y0);
}

void Radial1DSolver::download_species(std::vector<double>& X_cell,
                                      std::vector<double>& Y_cell) {
    int nz = lev.nz;
    X_cell.resize(nz); Y_cell.resize(nz);
    CUDA_CHECK(cudaMemcpy(X_cell.data(), d_X, nz*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Y_cell.data(), d_Y, nz*sizeof(double), cudaMemcpyDeviceToHost));
}

// ============================================================
// Implicit (BE) radiation diffusion — tridiagonal solve with Picard
// linearization of T⁴ and Stefan-Boltzmann photosphere BC.
//
// Energy eq (Lagrangian, zone k):
//   ρ cv · ∂T/∂t = -(1/dm) · ∂L/∂m      (in spherical Lagrangian form)
// where L = -4πr² D ∂(aT⁴)/∂r is the luminosity across the face.
//
// Discretize face k (between zones k-1 and k):
//   L_k = A_k · D_k · a · (T_{k-1}⁴ − T_k⁴) / Δr_zc_k         (interior)
//   L_0 = 0                                                    (center)
//   L_nz = A_nz · c · a · T_nz⁴ / 4   (Stefan-Boltzmann photosphere)
//
// For BE + Picard we linearize T⁴ ≈ T_p⁴ + 4 T_p³ (T − T_p) around the
// previous Picard iterate T_p. This gives a tridiag system in δT, then
// T_{p+1} = T_p + δT. 2-4 Picard iterations converge for reasonable dt.
// ============================================================

// Kernel: compute T from (ρ, e) into T_work (used both for init + Picard)
__global__ static void k_rad1d_T_from_rhoe(
    const double* rho, const double* e_int, double* T, int nz, EOS eos)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nz) return;
    double r = fmax(rho[k], 1e-30);
    double e = fmax(e_int[k], 1e-30);
    T[k] = fmax(eos.temperature_from_rho_e(r, e), 1e-12);
}

// Kernel: assemble tridiag for BE rad diffusion with Picard T⁴ linearization.
// Unknowns: δT[k] for k=0..nz-1 (zone-centered). Solves
//   A[k]·δT[k-1] + B[k]·δT[k] + C[k]·δT[k+1] = R[k]
// where the residual R[k] is the BE energy equation evaluated at T_p.
//
// Rearranging the BE eq in specific internal energy form:
//   ρ_k cv (T_k^{n+1} − T_k^n) · V_k / dt = -(L_{k+1} − L_k)
// => dm_k cv (T_k^{n+1} − T_k^n)/dt = L_k − L_{k+1}
// Picard: T_k^{n+1} ≈ T_p + δT_k, linearize T⁴ around T_p.
//
// k_face_factor_k = 4π r_k² · D_k · a · 4 T_p_{k-1}³  (coefficient on T_{k-1})
// etc. We split coefficient on T_{k-1} vs T_k since T⁴ linearization uses
// each zone's T_p.
__global__ static void k_rad1d_be_assemble(
    const double* T_p,      // (nz) current Picard iterate
    const double* T_n,      // (nz) T^n (start of BE step)
    const double* rho,      // (nz)
    const double* r,        // (nz+1)
    const double* dm,       // (nz)
    double* A_diag,         // (nz)  lower diag
    double* B_diag,         // (nz)  main diag
    double* C_diag,         // (nz)  upper diag
    double* rhs,            // (nz)
    int nz, EOS eos,
    double c_light, double a_rad, OpacityParams opa,
    double sigma_sb,        // σ_SB in code units (=c·a/4)
    double dt)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nz) return;

    double r_k   = r[k];           // inner face of zone k
    double r_k1  = r[k+1];         // outer face
    double dmk   = fmax(dm[k], 1e-30);
    double rho_k = fmax(rho[k],1e-30);
    double Tk    = fmax(T_p[k], 1e-12);
    double Tkn   = fmax(T_n[k], 1e-12);
    double cv_k  = eos.cv();

    // ---- Inner face L_k (between k-1 and k) ----
    // L_k = A_k · D_k · a · (T_{k-1}⁴ − T_k⁴) / Δr_zc_k
    // Linearize about (T_p_{k-1}, T_p_k):
    //   L_k ≈ L_k(T_p) + 4 A D a T_p_{k-1}³/Δr · δT_{k-1}  -  4 A D a T_p_k³/Δr · δT_k
    // Coefficient in eq for δT_k picks up -(-) sign...
    //
    // BE residual: dmk cv (Tkn+δTk − Tkn)/dt - (L_k − L_{k+1}) = 0
    // i.e. (dmk cv / dt) δT_k = L_k(T_p+δT) − L_{k+1}(T_p+δT) − (dmk cv / dt)(T_p - T_n)
    //
    // Sign convention: rhs[k] = -F_k + correction, where F_k is residual at T_p.

    double A_coef = 0.0, C_coef = 0.0;
    double L_face_in = 0.0;        // L_k at T_p
    double L_face_out = 0.0;       // L_{k+1} at T_p
    double dL_in_dTkm1 = 0.0;      // ∂L_k/∂T_{k-1}
    double dL_in_dTk   = 0.0;      // ∂L_k/∂T_k
    double dL_out_dTk  = 0.0;      // ∂L_{k+1}/∂T_k
    double dL_out_dTkp1= 0.0;      // ∂L_{k+1}/∂T_{k+1}

    // Inner face (k): exists for k >= 1, closed at k == 0
    if (k >= 1) {
        double Tkm1 = 1.0;  // placeholder, we can't read T_p[k-1] in kernel w/o shared — do coalesced read
        // direct read is fine for nz up to a few thousand
        Tkm1 = fmax(T_p[k-1], 1e-12);
        double rho_km1 = fmax(rho[k-1], 1e-30);
        double T_face = 0.5 * (Tkm1 + Tk);
        double rho_face = 0.5 * (rho_km1 + rho_k);
        double kap = grey_opacity(rho_face, T_face, opa);
        double D = c_light / (3.0 * kap * rho_face);
        double A_face = 4.0 * 3.14159265358979323846 * r_k * r_k;
        // Zone-center spacing
        double rc_lo = 0.5 * (r[k-1] + r[k]);
        double rc_hi = 0.5 * (r[k]   + r[k+1]);
        double dr_zc = fmax(rc_hi - rc_lo, 1e-30);
        double coef  = A_face * D * a_rad / dr_zc;
        double Tkm1_3 = Tkm1*Tkm1*Tkm1;
        double Tk_3   = Tk*Tk*Tk;
        double Tkm1_4 = Tkm1_3 * Tkm1;
        double Tk_4   = Tk_3   * Tk;
        L_face_in    = coef * (Tkm1_4 - Tk_4);
        dL_in_dTkm1  =  4.0 * coef * Tkm1_3;
        dL_in_dTk    = -4.0 * coef * Tk_3;
    }

    // Outer face (k+1)
    if (k < nz - 1) {
        double Tkp1 = fmax(T_p[k+1], 1e-12);
        double rho_kp1 = fmax(rho[k+1], 1e-30);
        double T_face = 0.5 * (Tk + Tkp1);
        double rho_face = 0.5 * (rho_k + rho_kp1);
        double kap = grey_opacity(rho_face, T_face, opa);
        double D = c_light / (3.0 * kap * rho_face);
        double A_face = 4.0 * 3.14159265358979323846 * r_k1 * r_k1;
        double rc_lo = 0.5 * (r[k]   + r[k+1]);
        double rc_hi = 0.5 * (r[k+1] + r[k+2]);
        double dr_zc = fmax(rc_hi - rc_lo, 1e-30);
        double coef  = A_face * D * a_rad / dr_zc;
        double Tkp1_3 = Tkp1*Tkp1*Tkp1;
        double Tk_3   = Tk*Tk*Tk;
        double Tkp1_4 = Tkp1_3 * Tkp1;
        double Tk_4   = Tk_3   * Tk;
        L_face_out   = coef * (Tk_4 - Tkp1_4);
        dL_out_dTk   =  4.0 * coef * Tk_3;
        dL_out_dTkp1 = -4.0 * coef * Tkp1_3;
    } else {
        // Surface face: L_surf = A_surf · σ_SB · T_surf⁴  (Stefan-Boltzmann)
        // Take T_surf = T[nz-1] (last zone) as photospheric temperature — crude
        // but standard for grey 1D stellar models. A more accurate τ=2/3 boundary
        // would find T at optical depth 2/3; we can refine later.
        double A_surf = 4.0 * 3.14159265358979323846 * r_k1 * r_k1;
        double Tk_3   = Tk*Tk*Tk;
        double Tk_4   = Tk_3 * Tk;
        L_face_out   = A_surf * sigma_sb * Tk_4;
        dL_out_dTk   = 4.0 * A_surf * sigma_sb * Tk_3;
        dL_out_dTkp1 = 0.0;
    }

    // Assemble tridiag row
    // LHS: (dmk cv/dt) δT_k - (∂L_k/∂T) term + (∂L_{k+1}/∂T) term
    // The BE eq with δT:  dmk cv (Tp+δTk − Tkn)/dt = L_k(Tp+δT) − L_{k+1}(Tp+δT)
    // Collect δT terms:
    //   (dmk cv / dt) δT_k - dL_in_dTkm1 δT_{k-1} - dL_in_dTk δT_k
    //    + dL_out_dTk δT_k + dL_out_dTkp1 δT_{k+1}
    //   = [-dmk cv(Tp-Tkn)/dt + L_in(Tp) − L_out(Tp)]
    //
    double a_ = -dL_in_dTkm1;
    double c_ =  dL_out_dTkp1;
    double b_ = (dmk * cv_k / dt) - dL_in_dTk + dL_out_dTk;
    double rhs_val = -dmk * cv_k * (Tk - Tkn) / dt + L_face_in - L_face_out;

    A_diag[k] = a_;
    B_diag[k] = b_;
    C_diag[k] = c_;
    rhs[k]    = rhs_val;
}

// Host-side Thomas algorithm (nz is typically 128-2048 — trivial).
static void thomas_solve(std::vector<double>& a, std::vector<double>& b,
                         std::vector<double>& c, std::vector<double>& d,
                         std::vector<double>& x)
{
    int n = (int)b.size();
    std::vector<double> cp(n), dp(n);
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];
    for (int i = 1; i < n; ++i) {
        double m = b[i] - a[i] * cp[i-1];
        if (std::fabs(m) < 1e-300) m = 1e-300;
        cp[i] = (i < n - 1) ? c[i] / m : 0.0;
        dp[i] = (d[i] - a[i] * dp[i-1]) / m;
    }
    x[n-1] = dp[n-1];
    for (int i = n - 2; i >= 0; --i) x[i] = dp[i] - cp[i] * x[i+1];
}

// Kernel: update e_int from ΔT (ρ cv ΔT = Δe per mass)
__global__ static void k_rad1d_apply_dT(
    double* e_int, const double* T_new, const double* T_start,
    int nz, EOS eos)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nz) return;
    double cv = eos.cv();
    double dT = T_new[k] - T_start[k];
    e_int[k] = fmax(e_int[k] + cv * dT, 1e-30);
}

// Kernel: apply δT onto T_p (Picard update)
__global__ static void k_rad1d_apply_delta_T(
    double* T_p, const double* dT, int nz, double damp)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nz) return;
    double newT = T_p[k] + damp * dT[k];
    if (newT < 1e-12) newT = 1e-12;
    T_p[k] = newT;
}

int Radial1DSolver::apply_radiation_diffusion_implicit(double dt_total) {
    if (!radiation_enabled || !use_eos) return 0;
    int nz = lev.nz, B = 256;

    OpacityParams opa;
    opa.kappa_es     = rad_kappa_es;
    opa.kappa_ff_0   = rad_kappa_ff_0;
    opa.kappa_dust_0 = rad_kappa_dust_0;
    opa.kappa_Hm_0   = rad_kappa_Hm_0;
    opa.T_dust_off   = rad_T_dust_off;
    double sigma_sb = rad_c_light * rad_a_rad / 4.0;

    // Allocate tridiag scratch (tiny)
    static double *d_A = nullptr, *d_Bm = nullptr, *d_Cm = nullptr,
                  *d_R = nullptr, *d_Tp = nullptr, *d_Tn = nullptr,
                  *d_dT = nullptr;
    static int cached_nz = 0;
    if (cached_nz != nz) {
        auto f = [](double* p){ if (p) cudaFree(p); };
        f(d_A); f(d_Bm); f(d_Cm); f(d_R); f(d_Tp); f(d_Tn); f(d_dT);
        auto a = [nz](double** p){ CUDA_CHECK(cudaMalloc(p, nz*sizeof(double))); };
        a(&d_A); a(&d_Bm); a(&d_Cm); a(&d_R); a(&d_Tp); a(&d_Tn); a(&d_dT);
        cached_nz = nz;
    }

    // Compute T^n from current (ρ, e_int)
    k_rad1d_T_from_rhoe<<<(nz+B-1)/B, B>>>(lev.d_rho, lev.d_e_int, d_Tn, nz, eos);
    // Initial Picard iterate: T_p = T_n
    CUDA_CHECK(cudaMemcpy(d_Tp, d_Tn, nz*sizeof(double), cudaMemcpyDeviceToDevice));

    const int max_picard = 6;
    const double tol_rel = 1e-4;
    std::vector<double> h_a(nz), h_b(nz), h_c(nz), h_rhs(nz), h_dT(nz);
    int it = 0;
    double last_rel = 0.0;
    for (it = 0; it < max_picard; ++it) {
        k_rad1d_be_assemble<<<(nz+B-1)/B, B>>>(
            d_Tp, d_Tn, lev.d_rho, lev.d_r, lev.d_dm,
            d_A, d_Bm, d_Cm, d_R,
            nz, eos, rad_c_light, rad_a_rad, opa, sigma_sb, dt_total);

        CUDA_CHECK(cudaMemcpy(h_a.data(),   d_A,  nz*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_b.data(),   d_Bm, nz*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_c.data(),   d_Cm, nz*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_rhs.data(), d_R,  nz*sizeof(double), cudaMemcpyDeviceToHost));
        thomas_solve(h_a, h_b, h_c, h_rhs, h_dT);

        // Compute relative change, upload δT, apply with damping if T would go <= 0
        double max_rel = 0.0;
        std::vector<double> h_Tp(nz);
        CUDA_CHECK(cudaMemcpy(h_Tp.data(), d_Tp, nz*sizeof(double), cudaMemcpyDeviceToHost));
        double damp = 1.0;
        for (int k = 0; k < nz; ++k) {
            double Tnew = h_Tp[k] + h_dT[k];
            if (Tnew <= 0.5 * h_Tp[k]) {
                double d_allow = 0.5 * h_Tp[k] - h_Tp[k];     // allow T to drop at most 50%
                if (h_dT[k] < 0) {
                    double local_damp = d_allow / h_dT[k];
                    if (local_damp < damp) damp = local_damp;
                }
            }
            if (h_Tp[k] > 1.0) {
                double rel = std::fabs(h_dT[k]) / h_Tp[k];
                if (rel > max_rel) max_rel = rel;
            }
        }
        if (damp < 0.1) damp = 0.1;
        CUDA_CHECK(cudaMemcpy(d_dT, h_dT.data(), nz*sizeof(double), cudaMemcpyHostToDevice));
        k_rad1d_apply_delta_T<<<(nz+B-1)/B, B>>>(d_Tp, d_dT, nz, damp);

        last_rel = max_rel;
        if (max_rel < tol_rel && damp >= 0.99) { it++; break; }
    }

    // Apply final ΔT back to e_int: Δe = cv · (T_final − T_n)
    k_rad1d_apply_dT<<<(nz+B-1)/B, B>>>(lev.d_e_int, d_Tp, d_Tn, nz, eos);

    // Diagnostic: surface luminosity L = A_surf · σ · T_surf⁴
    {
        double T_surf;
        CUDA_CHECK(cudaMemcpy(&T_surf, d_Tp + nz - 1, sizeof(double), cudaMemcpyDeviceToHost));
        double r_surf;
        CUDA_CHECK(cudaMemcpy(&r_surf, lev.d_r + nz, sizeof(double), cudaMemcpyDeviceToHost));
        double A_s = 4.0 * M_PI * r_surf * r_surf;
        rad_impl_L_surf = A_s * sigma_sb * T_surf * T_surf * T_surf * T_surf;
    }
    rad_impl_last_picard = it;
    if (step_count < 3 || (step_count % 200 == 0)) {
        std::fprintf(stderr, "  [BE-rad] step=%d picard=%d dt=%.2e L_surf=%.3e\n",
                     step_count, it, dt_total, rad_impl_L_surf);
    }
    return it;
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
