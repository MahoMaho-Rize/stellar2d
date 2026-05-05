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
#include "physics/radiation_diffusion.cuh"
#include "gpu_common.cuh"      // CUDA_CHECK macro
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

// Helper: fill an OpacityParams from the solver's rad_kappa_* coefficients
// plus any attached MESA kap table views. Used at every call site that
// needs to dispatch `grey_opacity()`.
void Radial1DSolver::fill_opacity_params(OpacityParams& opa) const {
    opa.kappa_es     = rad_kappa_es;
    opa.kappa_ff_0   = rad_kappa_ff_0;
    opa.kappa_dust_0 = rad_kappa_dust_0;
    opa.kappa_Hm_0   = rad_kappa_Hm_0;
    opa.T_dust_off   = rad_T_dust_off;
    opa.use_table    = kap_use_table;
    if (kap_use_table) {
        opa.table_lowT    = kap_view_lowT;
        opa.table_highT   = kap_view_highT;
        opa.logT_lo_end   = kap_logT_lo_end;
        opa.logT_hi_start = kap_logT_hi_start;
        opa.hydrogen_X    = kap_hydrogen_X;
    }
}

// Bulk P→e inversion on the device (EOS struct holds device table pointers
// for the Helmholtz branch). Each thread handles one zone.
static __global__ void k_rad1d_e_from_rhoP(const double* d_rho, const double* d_P,
                                           double* d_e, int n, EOS eos) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    d_e[k] = eos.internal_energy(d_rho[k], d_P[k]);
}

// T-seeded variant: read (ρ, T), write (e, P_out) via a Helm forward eval.
// Used by init_from_mesa --ic-mesa-seed-T so runtime T matches MESA exactly
// and κ(ρ, T) is consistent. Overwrites d_P with the Helm-computed P(ρ, T)
// to keep the internal HSE self-consistent; any discrepancy vs MESA's P
// is then a pure EOS-choice diagnostic (same ρ, same T → different P means
// blend differs), not a mismatch with our own runtime state.
#ifdef __CUDACC__
static __global__ void k_rad1d_eP_from_rhoT(const double* d_rho,
                                            const double* d_T,
                                            double* d_e, double* d_P,
                                            int n, EOS eos) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    double rho = d_rho[k], T = d_T[k];
    if (eos.type == (int)EosType::HELMHOLTZ) {
        HelmState s = helm_eval(rho, T, eos.helm);
        d_e[k] = s.e;
        d_P[k] = s.P;
    } else {
        // Ideal gas fallback: e = cv·T, P = (γ−1)·ρ·e = ρ·R_gas·T
        double cv_val = eos.cv();
        d_e[k] = cv_val * T;
        d_P[k] = (eos.gamma - 1.0) * rho * d_e[k];
    }
}
#endif

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
    f(d_K_conv);
    d_X = nullptr; d_Y = nullptr; d_K_conv = nullptr;
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
    bool eos_needs_device = use_eos
        && eos.type == static_cast<int>(EosType::HELMHOLTZ);
    for (int k = 0; k < nz; ++k) {
        double rL = h_r[k], rR = h_r[k+1];
        double V_k = (4.0*M_PI/3.0) * (rR*rR*rR - rL*rL*rL);
        h_dm[k] = dm;
        h_rho[k] = dm / V_k;
        h_P[k] = K_poly * std::pow(h_rho[k], 1.0 + 1.0/n_poly);
        // Specific internal energy must be consistent with the EOS actually
        // in use. For ideal gas this is P/((γ−1)ρ); for ideal_rad / PRE_MS
        // we invert the EOS (P, ρ) → e. Helmholtz does it on the device
        // (below) since its table lives in GPU memory.
        if (use_eos && !eos_needs_device) {
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
    if (eos_needs_device) {
        int block = 64, grid = (nz + block - 1) / block;
        k_rad1d_e_from_rhoP<<<grid, block>>>(lev.d_rho0, lev.d_P0, lev.d_e_int, nz, eos);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaMemset(lev.d_v, 0, (nz+1)*sizeof(double)));

    // Initial surface pressure floor = initial surface P (tiny but > 0)
    P_surf_floor = h_P[nz-1];
    hse_set = true;

    std::fprintf(stderr, "  Lane-Emden init: nz=%d zones, M_star=%.6f, R_star=%.6f, ρ_c=%.4e, P_surf_floor=%.3e\n",
                 nz, M_total, R_star, rho_c, P_surf_floor);
}

// ============================================================
// init_from_mesa: read scripts/mesa/convert_mesa_ic.py output, remap MESA's
// non-uniform zones onto our equal-mass Lagrangian shells.
//
// MESA profile ordering: index 0 = surface, N-1 = core. We flip to core→
// surface (monotonic increasing in both r and M_enc) and linearly
// interpolate (rho, T, P, X, Y) against M_enc at the target shell
// centres. The outermost face r[nz] is pinned to R_star from the IC
// header.
// ============================================================
int Radial1DSolver::init_from_mesa(const char* ic_path, bool seed_T,
                                    int n_atm_zones) {
    std::FILE* fp = std::fopen(ic_path, "r");
    if (!fp) {
        std::fprintf(stderr, "  init_from_mesa: cannot open %s\n", ic_path);
        return 1;
    }
    char line[1024];
    double M_star = -1.0, R_star = -1.0;
    int n_mesa = -1;
    while (std::fgets(line, sizeof(line), fp)) {
        if (line[0] != '#') { std::ungetc('\n', fp); std::fseek(fp, -(long)std::strlen(line), SEEK_CUR); break; }
        double v;
        if (std::sscanf(line, "# M_star_g %lf", &v) == 1) M_star = v;
        else if (std::sscanf(line, "# R_star_cm %lf", &v) == 1) R_star = v;
        else if (std::sscanf(line, "# n_zones %d", &n_mesa) == 1) {}
    }
    if (n_mesa <= 0 || !(M_star > 0.0) || !(R_star > 0.0)) {
        std::fprintf(stderr,
            "  init_from_mesa: missing/invalid header in %s "
            "(n_mesa=%d, M=%g, R=%g)\n",
            ic_path, n_mesa, M_star, R_star);
        std::fclose(fp);
        return 2;
    }

    // Read zone rows. MESA order = surface → core; store as-is, flip later.
    std::vector<double> m_s(n_mesa), r_s(n_mesa), rho_s(n_mesa);
    std::vector<double> T_s(n_mesa), P_s(n_mesa);
    std::vector<double> X_s(n_mesa), Y_s(n_mesa), Z_s(n_mesa);
    int i = 0;
    while (i < n_mesa && std::fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        int n = std::sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf",
                            &m_s[i], &r_s[i], &rho_s[i],
                            &T_s[i], &P_s[i],
                            &X_s[i], &Y_s[i], &Z_s[i]);
        if (n != 8) {
            std::fprintf(stderr, "  init_from_mesa: bad row %d in %s\n", i, ic_path);
            std::fclose(fp);
            return 3;
        }
        ++i;
    }
    std::fclose(fp);
    if (i != n_mesa) {
        std::fprintf(stderr,
            "  init_from_mesa: expected %d rows, got %d\n", n_mesa, i);
        return 4;
    }

    // Flip to core→surface so everything is monotonically increasing.
    std::reverse(m_s.begin(),   m_s.end());
    std::reverse(r_s.begin(),   r_s.end());
    std::reverse(rho_s.begin(), rho_s.end());
    std::reverse(T_s.begin(),   T_s.end());
    std::reverse(P_s.begin(),   P_s.end());
    std::reverse(X_s.begin(),   X_s.end());
    std::reverse(Y_s.begin(),   Y_s.end());
    std::reverse(Z_s.begin(),   Z_s.end());

    // Monotonicity sanity.
    for (int k = 1; k < n_mesa; ++k) {
        if (!(m_s[k] >= m_s[k-1])) {
            std::fprintf(stderr, "  init_from_mesa: m_enc non-monotone at k=%d\n", k);
            return 5;
        }
    }

    // Build target shell masses. Default = equal-mass Lagrangian.
    // When n_atm_zones > 0, use hybrid: inner (nz - n_atm_zones) equal-mass
    // covering [0, M_cut], outer n_atm_zones log-spaced in "depth from surface"
    // covering [M_cut, M_star], so the photosphere is resolved with multiple
    // zones instead of being compressed into a single outer shell.
    //
    // Physics: MESA's late_preMS atmosphere has 300+ zones in <10⁻⁵ of mass.
    // Equal-mass zoning at nz=128 collapses all into zone nz-1, whose
    // interpolated T at the shell midpoint is a deep-interior T (~7e6 K),
    // not the photospheric 4500 K. The Eddington BC then misfires on a wrong
    // T_k input → T_eff ~630 K → L_surf ~10⁻⁴ L☉ → KH contraction stalled.
    int nz = lev.nz;
    std::vector<double> M_target(nz + 1);
    int n_atm = n_atm_zones;
    if (n_atm < 0) n_atm = 0;
    if (n_atm > nz / 2) n_atm = nz / 2;    // keep at least half as inner core
    if (n_atm > 0) {
        // Hybrid zoning. M_cut = M_star · (1 - M_atm_frac). Choose
        // M_atm_frac so outer N_atm zones span the MESA atmosphere.
        // For 1 M⊙ pre-MS, MESA's photosphere (T ~ 4500 K) sits at depth
        // fraction ~10⁻¹² to 10⁻¹¹. For the outermost zone's mass-weighted
        // center to actually sample that cold material, we need
        // dm_{nz-1} ≲ 2·(1e-11·M_star). With 20 geometric zones at ratio
        // 0.05, outermost dm ≈ 0.14·M_atm_frac·M_star → need
        // M_atm_frac ≲ 1e-11. Outer zones beyond that become rarified
        // atmosphere (ρ ~ 10⁻⁷, many orders below Helm table edge ρ_min
        // = 10⁻¹² which is safe).
        double M_atm_frac = 1.0e-11;
        double M_cut = M_star * (1.0 - M_atm_frac);
        int nz_inner = nz - n_atm;
        double dm_inner = M_cut / nz_inner;
        for (int k = 0; k <= nz_inner; ++k) M_target[k] = k * dm_inner;
        // Outer zones: geometric series in depth (M_star - M).
        // M_target[nz_inner + j] = M_star - (M_star - M_cut) · r^{j/n_atm}
        // where r < 1 so the outermost zone (j = n_atm) sits at M_star.
        // Use r = 1/n_atm to compress outer zones exponentially.
        // Outer shell dm_j = M_star·M_atm_frac·(r^((j-1)/n_atm) - r^(j/n_atm))
        // Monotonic small outer dm — exactly what MESA resolves.
        double M_depth = M_star - M_cut;  // = M_star · M_atm_frac
        double ratio = 1.0 / (double)n_atm;
        for (int j = 1; j <= n_atm; ++j) {
            double frac = std::pow(ratio, (double)(n_atm - j) / (double)n_atm);
            // frac runs from ratio^1 (j=1, deep atm) → 1 (j=n_atm, surface)
            // so M_target climbs from M_cut + small to M_star.
            M_target[nz_inner + j] = M_star - M_depth * (1.0 - frac);
        }
        // Pin exactly at surface (avoid roundoff).
        M_target[nz] = M_star;
    } else {
        double dm = M_star / nz;
        for (int k = 0; k <= nz; ++k) M_target[k] = k * dm;
    }

    // Interpolate r at each face mass (linear in m).
    std::vector<double> h_r(nz + 1);
    h_r[0] = 0.0;                     // innermost face at r=0
    // For faces beyond MESA's innermost zone (mass < m_s[0]), fall back to
    // core-linear scaling r ~ (m/m_s[0])^{1/3} · r_s[0] (uniform-density
    // core approximation).
    int j = 0;
    for (int k = 1; k <= nz; ++k) {
        double Mt = M_target[k];
        if (Mt <= m_s[0]) {
            double frac = Mt / m_s[0];
            h_r[k] = std::pow(frac, 1.0 / 3.0) * r_s[0];
            continue;
        }
        while (j + 1 < n_mesa && m_s[j + 1] < Mt) ++j;
        if (j + 1 >= n_mesa) { h_r[k] = R_star; continue; }
        double t = (Mt - m_s[j]) / (m_s[j + 1] - m_s[j]);
        h_r[k] = r_s[j] + t * (r_s[j + 1] - r_s[j]);
    }
    h_r[nz] = R_star;

    // Outer-shell spacing floor (equal-mass mode only). For equal-mass
    // zoning, MESA atmospheres over-sample the outermost mass fractions
    // (300+ zones in < 10⁻⁵ of mass), producing a tiny Δr for the single
    // outer radial1d shell. This inflates ρ_geom = dm/Vol, produces a
    // density spike at k = nz-1, and makes Newton struggle.
    //
    // Remedy: cap the outermost 2 shells' Δr at a floor relative to the
    // interior spacing.
    //
    // In hybrid zoning (n_atm > 0) we deliberately want small outer Δr to
    // resolve the atmosphere; skip the floor.
    if (n_atm == 0) {
        double dr_deep = h_r[nz-2] - h_r[nz-3];
        double dr_top  = h_r[nz]   - h_r[nz-1];
        double dr_mid  = h_r[nz-1] - h_r[nz-2];
        double dr_min  = 0.7 * dr_deep;
        if (dr_top < dr_min) {
            h_r[nz-1] = h_r[nz] - dr_min;
            dr_mid = h_r[nz-1] - h_r[nz-2];
        }
        if (dr_mid < dr_min) {
            h_r[nz-2] = h_r[nz-1] - dr_min;
        }
    }

    // Zone-centred quantities: interpolate MESA data at the shell's
    // center-of-mass (midpoint between M_target[k] and M_target[k+1]).
    std::vector<double> h_dm(nz), h_rho(nz), h_T(nz), h_P(nz), h_e(nz);
    std::vector<double> h_X(nz), h_Y(nz);
    j = 0;
    for (int k = 0; k < nz; ++k) {
        double dm_k = M_target[k+1] - M_target[k];
        double Mc = M_target[k] + 0.5 * dm_k;
        if (Mc <= m_s[0]) {
            h_rho[k] = rho_s[0]; h_T[k] = T_s[0]; h_P[k] = P_s[0];
            h_X[k] = X_s[0]; h_Y[k] = Y_s[0];
        } else if (Mc >= m_s[n_mesa - 1]) {
            h_rho[k] = rho_s[n_mesa - 1]; h_T[k] = T_s[n_mesa - 1];
            h_P[k]   = P_s[n_mesa - 1];
            h_X[k]   = X_s[n_mesa - 1]; h_Y[k] = Y_s[n_mesa - 1];
        } else {
            while (j + 1 < n_mesa && m_s[j + 1] < Mc) ++j;
            double t = (Mc - m_s[j]) / (m_s[j + 1] - m_s[j]);
            h_rho[k] = rho_s[j] + t * (rho_s[j + 1] - rho_s[j]);
            h_T[k]   = T_s[j]   + t * (T_s[j + 1]   - T_s[j]);
            h_P[k]   = P_s[j]   + t * (P_s[j + 1]   - P_s[j]);
            h_X[k]   = X_s[j]   + t * (X_s[j + 1]   - X_s[j]);
            h_Y[k]   = Y_s[j]   + t * (Y_s[j + 1]   - Y_s[j]);
        }
        h_dm[k] = dm_k;

        // --- Geometric consistency override ---
        // MESA's mass → radius map can be sampled far more densely at the
        // surface than our nz allows. The outermost equal-mass shell ends
        // up occupying a finite Δr where MESA has a sharp atmosphere drop,
        // so the mass-averaged ρ_geom = dm/Vol is much larger than
        // rho_s interpolated at Mc (the MESA surface ρ). Using the MESA ρ
        // directly gives e(ρ_mesa, T_mesa) ≈ 1e-30 floor for the outermost
        // shell — Helm can't stably invert. But d_rho at runtime is
        // dm/Vol, so the EOS call sees a completely different state.
        //
        // Fix: override h_rho[k] = dm / Vol_k for shells where Vol_k is
        // what radial1d will actually use. Keep T from MESA (temperature is
        // the physically meaningful atmosphere quantity we want preserved);
        // P is then re-derived via Helm in k_rad1d_eP_from_rhoT.
        double r_lo = h_r[k];
        double r_hi = h_r[k+1];
        double Vol  = (4.0/3.0) * M_PI * (r_hi*r_hi*r_hi - r_lo*r_lo*r_lo);
        if (Vol > 0.0) {
            double rho_geom = dm_k / Vol;
            // Only use geom ρ when it's larger than MESA ρ (outer zones
            // where MESA atmosphere is unresolved). Core zones where
            // sampling is fine keep MESA ρ.
            if (rho_geom > h_rho[k]) h_rho[k] = rho_geom;
        }
        // Host-side e seeding (non-Helm path). For Helm we defer to the
        // device kernel below where table pointers live.
        bool eos_needs_device = use_eos
            && eos.type == static_cast<int>(EosType::HELMHOLTZ);
        if (use_eos && !eos_needs_device) {
            h_e[k] = eos.internal_energy(h_rho[k], h_P[k]);
        } else {
            h_e[k] = h_P[k] / ((gamma - 1.0) * h_rho[k]);
        }
    }

    // Upload
    CUDA_CHECK(cudaMemcpy(lev.d_r,     h_r.data(),   (nz+1)*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_r0,    h_r.data(),   (nz+1)*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_dm,    h_dm.data(),  nz*sizeof(double),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_e_int, h_e.data(),   nz*sizeof(double),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_rho0,  h_rho.data(), nz*sizeof(double),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_P0,    h_P.data(),   nz*sizeof(double),     cudaMemcpyHostToDevice));
    bool eos_needs_device = use_eos
        && eos.type == static_cast<int>(EosType::HELMHOLTZ);
    if (eos_needs_device) {
        int block = 64, grid = (nz + block - 1) / block;
        if (seed_T) {
            // Forward Helm eval from (ρ, T_MESA); overwrites d_P with Helm's
            // P(ρ, T) so HSE and runtime κ(ρ, T) are internally consistent.
            double* d_T_tmp = nullptr;
            CUDA_CHECK(cudaMalloc(&d_T_tmp, nz * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_T_tmp, h_T.data(),
                                  nz * sizeof(double), cudaMemcpyHostToDevice));
            k_rad1d_eP_from_rhoT<<<grid, block>>>(
                lev.d_rho0, d_T_tmp, lev.d_e_int, lev.d_P0, nz, eos);
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaFree(d_T_tmp);
        } else {
            k_rad1d_e_from_rhoP<<<grid, block>>>(
                lev.d_rho0, lev.d_P0, lev.d_e_int, nz, eos);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    CUDA_CHECK(cudaMemset(lev.d_v, 0, (nz+1)*sizeof(double)));

    // Diagnostic: outer 3 zones after IC write (read back e_int from device)
    if (n_atm > 0) {
        std::vector<double> h_e_check(nz);
        CUDA_CHECK(cudaMemcpy(h_e_check.data(), lev.d_e_int,
                              nz*sizeof(double), cudaMemcpyDeviceToHost));
        std::fprintf(stderr, "  init_from_mesa hybrid IC (outer 3):\n");
        for (int k = std::max(0, nz-3); k < nz; ++k) {
            std::fprintf(stderr,
                "    k=%d rho=%.3e T_MESA=%.3e e=%.3e\n",
                k, h_rho[k], h_T[k], h_e_check[k]);
        }
    }

    // Species, if wired
    if (species_enabled && d_X && d_Y) {
        CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), nz*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Y, h_Y.data(), nz*sizeof(double), cudaMemcpyHostToDevice));
    }

    if (seed_T && eos_needs_device) {
        // d_P0 was overwritten by Helm; read back the outermost shell for
        // the surface-floor BC so it matches the live state.
        double P_surf_live = 0.0;
        CUDA_CHECK(cudaMemcpy(&P_surf_live, lev.d_P0 + (nz - 1),
                              sizeof(double), cudaMemcpyDeviceToHost));
        P_surf_floor = P_surf_live;
    } else {
        P_surf_floor = h_P[nz - 1];
    }
    hse_set = true;

    std::fprintf(stderr,
        "  MESA IC loaded: nz=%d zones (from %d MESA), M=%.4e g, R=%.4e cm, "
        "T_core=%.3e K, ρ_core=%.3e g/cc, P_surf_floor=%.3e [%s]\n",
        nz, n_mesa, M_star, R_star, h_T[0], h_rho[0], P_surf_floor,
        seed_T ? "seed_T" : "seed_P");
    return 0;
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

    // Core state + integrated nuclear luminosity. Both need temperature, which
    // for the Helmholtz EOS must be evaluated on device (table pointers live in
    // GPU memory). Use a small device kernel to compute per-zone T and
    // ε_pp(ρ, T, X)·dm, then copy to host and reduce on CPU.
    std::vector<double> h_rho(nz), h_e(nz), h_dm(nz), h_X, h_L(nz, 0.0), h_T(nz);
    CUDA_CHECK(cudaMemcpy(h_rho.data(), lev.d_rho,   nz*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_e.data(),   lev.d_e_int, nz*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dm.data(),  lev.d_dm,    nz*sizeof(double), cudaMemcpyDeviceToHost));

    if (nuclear_enabled && use_eos) {
        // Kernel: per-zone T_k + ε_pp·dm, stored in scratch[0] (unused after this).
        NuclearPPParams npars;
        npars.X_hydrogen = nuc_X;
        npars.T_floor = nuc_T_floor;
        npars.T_scale = nuc_T_scale;
        npars.epsilon_scale = nuc_epsilon_scale;
        npars.q_burn = nuc_q_burn;
        if (species_enabled) {
            k_rad1d_nuclear_L_species<<<(nz+B-1)/B, B>>>(
                lev.d_rho, lev.d_e_int, d_X, lev.d_dm, scratch[0], nz, eos, npars);
        } else {
            k_rad1d_nuclear_L<<<(nz+B-1)/B, B>>>(
                lev.d_rho, lev.d_e_int, lev.d_dm, scratch[0], nz, eos, npars);
        }
        d.L_nuc = gpu_reduce_sum_simple(scratch[0], nz);
    } else {
        d.L_nuc = 0.0;
    }

    // Core state: innermost zone at k=0
    d.rho_c = h_rho[0];
    if (use_eos) {
        // Need a device eval for Helmholtz branch. Reuse scratch[1] for T[0] only.
        k_rad1d_T_from_rho_e<<<1, 1>>>(lev.d_rho, lev.d_e_int, scratch[1], 1, eos);
        CUDA_CHECK(cudaMemcpy(&d.T_c, scratch[1], sizeof(double), cudaMemcpyDeviceToHost));
    } else {
        d.T_c = (gamma - 1.0) * h_e[0];
    }

    for (int i = 0; i < 6; ++i) cudaFree(scratch[i]);
    return d;
}

// ============================================================
// MLT Schwarzschild diagnostic (Phase 6)
//
// For each zone k compute:
//   ∇_rad = d ln T / d ln P |_rad, from radiative flux balance:
//           ∇_rad = (3 κ ρ L_rad P) / (16π a c G M T⁴)   (diffusion limit)
//   ∇_ad  = (γ-1)/γ  (approximate; ideal gas; corrections for radiation
//                     pressure & ionization handled by real EOS later).
//   super = ∇_rad − ∇_ad    (positive ⇒ convective by Schwarzschild)
//
// L_rad[k] at zone k is the luminosity across the *outer* face (face k+1).
// We estimate it from local grad aT⁴ using the same formula as the BE rad
// diffusion kernel — this keeps the diagnostic consistent with the solver.
// ============================================================
__global__ static void k_rad1d_mlt_diag(
    const double* r,          // (nz+1)
    const double* rho,        // (nz)
    const double* P,          // (nz)
    const double* e_int,      // (nz)
    const double* M,          // (nz+1)
    const double* dm,         // (nz)
    double* out_super,        // (nz) ∇_rad − ∇_ad
    double* out_isconv,       // (nz) 1 if convective, 0 else
    double* out_conv_mass,    // (nz) dm if convective, 0 else
    double* out_rad_L,        // (nz) L_rad at outer face
    int nz, EOS eos,
    double a_rad, double c_light, double G_const,
    OpacityParams opa)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nz) return;

    double rho_k = fmax(rho[k], 1e-30);
    double e_k   = fmax(e_int[k], 1e-30);
    double T_k   = fmax(eos.temperature_from_rho_e(rho_k, e_k), 1e-12);
    double P_k   = fmax(P[k], 1e-30);
    double kap   = grey_opacity(rho_k, T_k, opa);

    // L_rad at outer face k+1 using diffusion formula, consistent with BE
    // solver. L = -4πr² · D · a · ∂T⁴/∂r  where D = c/(3κρ).
    double L_rad = 0.0;
    double r_kp1 = r[k+1];
    if (k < nz - 1) {
        double rho_kp1 = fmax(rho[k+1], 1e-30);
        double e_kp1   = fmax(e_int[k+1], 1e-30);
        double T_kp1   = fmax(eos.temperature_from_rho_e(rho_kp1, e_kp1), 1e-12);
        double T_face  = 0.5 * (T_k + T_kp1);
        double rho_face= 0.5 * (rho_k + rho_kp1);
        double kap_f   = grey_opacity(rho_face, T_face, opa);
        double D       = c_light / (3.0 * kap_f * rho_face);
        double A_face  = 4.0 * 3.14159265358979323846 * r_kp1 * r_kp1;
        double rc_lo   = 0.5 * (r[k]   + r[k+1]);
        double rc_hi   = 0.5 * (r[k+1] + r[k+2]);
        double dr_zc   = fmax(rc_hi - rc_lo, 1e-30);
        double T_k_4   = T_k*T_k*T_k*T_k;
        double T_kp1_4 = T_kp1*T_kp1*T_kp1*T_kp1;
        L_rad = A_face * D * a_rad * (T_k_4 - T_kp1_4) / dr_zc;
    } else {
        // At surface, L_rad = photospheric σ T⁴ A (same as BE surface BC).
        double A_s = 4.0 * 3.14159265358979323846 * r_kp1 * r_kp1;
        double sigma_sb = c_light * a_rad / 4.0;
        double T_k_4 = T_k*T_k*T_k*T_k;
        L_rad = A_s * sigma_sb * T_k_4;
    }
    out_rad_L[k] = L_rad;

    // Enclosed mass at outer face for this zone's L_rad balance.
    double M_k = M[k+1];
    if (M_k < 1e-30) M_k = 1e-30;
    double T_k_4 = T_k*T_k*T_k*T_k;
    // ∇_rad = (3 κ ρ L P) / (16π a c G M T⁴)
    double denom = 16.0 * 3.14159265358979323846 * a_rad * c_light * G_const * M_k * T_k_4;
    double grad_rad = (denom > 1e-300) ? (3.0 * kap * rho_k * L_rad * P_k) / denom : 0.0;
    // ∇_ad from EOS γ. For ideal_rad this is γ-1/γ; for PRE_MS use cv()
    // derived γ — close enough for diagnostic.
    double gam = eos.gamma;
    double grad_ad = (gam - 1.0) / gam;

    double super = grad_rad - grad_ad;
    out_super[k]     = super;
    out_isconv[k]    = (super > 0.0) ? 1.0 : 0.0;
    out_conv_mass[k] = (super > 0.0) ? dm[k] : 0.0;
}

// Rich per-zone diagnostic kernel used by download_profile_rich (Tier-2
// MESA PK). Writes T, κ, Γ₁, ∇_ad, ∇_rad, L_rad at outer face, an int
// mixing flag (0=stable, 1=convective), and a v_conv proxy.
//
// v_conv proxy: Böhm-Vitense-style estimate
//   v_conv^2 = g · H_P · (∇_rad − ∇_ad)    for ∇_rad > ∇_ad, else 0
// using H_P = P/(ρg), g = G M_enc / r². This is what MESA records as
// `conv_vel` in a fully-developed MLT envelope.
__global__ static void k_rad1d_rich_diag(
    const double* r,          // (nz+1)
    const double* rho,        // (nz)
    const double* P,          // (nz)
    const double* e_int,      // (nz)
    const double* M,          // (nz+1)
    double* out_T,            // (nz)
    double* out_kap,          // (nz)
    double* out_gamma1,       // (nz)
    double* out_grada,        // (nz)
    double* out_gradr,        // (nz)
    double* out_Lface,        // (nz) L at outer face
    int*    out_mixtype,      // (nz)
    double* out_conv_vel,     // (nz)
    int nz, EOS eos,
    double a_rad, double c_light, double G_const,
    OpacityParams opa)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nz) return;
    const double PI = 3.14159265358979323846;

    double rho_k = fmax(rho[k], 1e-30);
    double e_k   = fmax(e_int[k], 1e-30);
    double T_k   = fmax(eos.temperature_from_rho_e(rho_k, e_k), 1e-12);
    double P_k   = fmax(P[k], 1e-30);
    double kap   = grey_opacity(rho_k, T_k, opa);

    out_T[k]   = T_k;
    out_kap[k] = kap;

    // Γ₁ and ∇_ad — prefer Helm's exact derivatives when available.
    double gam = eos.gamma;
    double gamma1_val = gam;
    double grada_val  = (gam - 1.0) / gam;    // ideal-gas default
#ifdef __CUDA_ARCH__
    if (eos.type == (int)EosType::HELMHOLTZ) {
        HelmState hs = helm_eval(rho_k, T_k, eos.helm);
        gamma1_val = hs.gamma1;
        grada_val  = hs.grada;
    }
#endif
    out_gamma1[k] = gamma1_val;

    // L_rad at outer face, identical logic to k_rad1d_mlt_diag
    double L_rad = 0.0;
    double r_kp1 = r[k+1];
    if (k < nz - 1) {
        double rho_kp1 = fmax(rho[k+1], 1e-30);
        double e_kp1   = fmax(e_int[k+1], 1e-30);
        double T_kp1   = fmax(eos.temperature_from_rho_e(rho_kp1, e_kp1), 1e-12);
        double T_face   = 0.5 * (T_k + T_kp1);
        double rho_face = 0.5 * (rho_k + rho_kp1);
        double kap_f    = grey_opacity(rho_face, T_face, opa);
        double D        = c_light / (3.0 * kap_f * rho_face);
        double A_face   = 4.0 * PI * r_kp1 * r_kp1;
        double rc_lo    = 0.5 * (r[k]   + r[k+1]);
        double rc_hi    = 0.5 * (r[k+1] + r[k+2]);
        double dr_zc    = fmax(rc_hi - rc_lo, 1e-30);
        double T_k_4    = T_k * T_k * T_k * T_k;
        double T_kp1_4  = T_kp1 * T_kp1 * T_kp1 * T_kp1;
        L_rad = A_face * D * a_rad * (T_k_4 - T_kp1_4) / dr_zc;
    } else {
        double A_s = 4.0 * PI * r_kp1 * r_kp1;
        double sigma_sb = c_light * a_rad / 4.0;
        double T_k_4 = T_k * T_k * T_k * T_k;
        L_rad = A_s * sigma_sb * T_k_4;
    }
    out_Lface[k] = L_rad;

    // ∇_rad from radiative-diffusion flux formula. ∇_ad already computed
    // above (Helm-exact when EOS supports it, ideal-gas fallback else).
    double M_k = M[k+1];
    if (M_k < 1e-30) M_k = 1e-30;
    double T_k_4  = T_k * T_k * T_k * T_k;
    double denom  = 16.0 * PI * a_rad * c_light * G_const * M_k * T_k_4;
    double gradr  = (denom > 1e-300) ? (3.0 * kap * rho_k * L_rad * P_k) / denom : 0.0;

    out_gradr[k]   = gradr;
    out_grada[k]   = grada_val;
    out_mixtype[k] = (gradr > grada_val) ? 1 : 0;

    // Böhm-Vitense MLT with radiative-efficiency saturation (Henyey 1965 /
    // Kippenhahn-Weigert-Weiss eq. 7.7 — same classical MLT MESA uses).
    //
    // Inputs: ∇_rad, ∇_ad, ρ, T, P, c_P, δ = χ_T/χ_ρ, κ, g, H_P, α.
    // The actual ∇_T lies between ∇_ad (efficient) and ∇_rad (inefficient).
    // Let ξ² = ∇_T − ∇_ad; Henyey's derivation gives the cubic
    //   ξ³ + U·ξ² + U²·ξ − U²·W = 0,   W = ∇_rad − ∇_ad
    //   U = (12·σ_SB·T³)/(c_P·ρ²·κ·ℓ²) · √(8·H_P/(g·δ))
    //
    // Deep envelope limit U→0: ξ² → W (no loss).
    // Surface limit U→∞:        ξ² → W/U² (radiative loss dominates).
    // Solve via Cardano (real root guaranteed positive for W>0).
    double g     = G_const * M_k / fmax(r_kp1 * r_kp1, 1e-30);
    double H_P   = P_k / fmax(rho_k * g, 1e-30);
    double W_raw = gradr - grada_val;
    double v_conv = 0.0;
    if (W_raw > 0.0 && H_P > 0.0) {
        // Thermodynamic derivatives — Helm-exact when available.
        double delta_val = 1.0;   // δ = −∂lnρ/∂lnT|_P = χ_T/χ_ρ
        double cP_val    = 2.5e8; // fallback ≈ ideal gas μ=1, stellar T
#ifdef __CUDA_ARCH__
        if (eos.type == (int)EosType::HELMHOLTZ) {
            HelmState hs_k = helm_eval(rho_k, T_k, eos.helm);
            delta_val = (hs_k.chiRho > 1e-30) ? hs_k.chiT / hs_k.chiRho : 1.0;
            cP_val    = (hs_k.cP > 1e-30) ? hs_k.cP : 2.5e8;
        }
#endif
        const double alpha_mlt = 1.5;     // mixing-length parameter
        double ell = alpha_mlt * H_P;
        double sigma_sb_loc = c_light * a_rad / 4.0;
        double T3 = T_k * T_k * T_k;
        double U_num = 12.0 * sigma_sb_loc * T3;
        double U_den = cP_val * rho_k * rho_k * kap * ell * ell;
        double U_aux = (g * delta_val > 1e-30)
                       ? sqrt(8.0 * H_P / (g * delta_val))
                       : 0.0;
        double U = (U_den > 1e-30) ? (U_num / U_den) * U_aux : 0.0;

        // The cubic  ξ³ + U·ξ² + U²·ξ − U²·W = 0  has exactly one positive
        // real root between 0 and √W. Limits:
        //   U → 0   (deep efficient):   ξ³ ≈ U²·W    ⇒ ξ ≈ (U²·W)^{1/3}
        //                               → super_eff ≈ (U²·W)^{2/3} → 0
        //   U → ∞   (surface radiative): ξ² ≈ W       ⇒ ξ ≈ √W
        //                               → super_eff → W  (no convection happening)
        double xi;
        if (U < 1e-30) {
            xi = 0.0;                     // fully efficient limit → zero super
        } else {
            // Seed from the small-U balance, which is usually where MLT sits.
            double seed = cbrt(U * U * W_raw);
            xi = seed;
            for (int it = 0; it < 15; ++it) {
                double f  = xi * xi * xi + U * xi * xi + U * U * xi - U * U * W_raw;
                double fp = 3.0 * xi * xi + 2.0 * U * xi + U * U;
                if (fp < 1e-30) break;
                double dxi = f / fp;
                xi -= dxi;
                if (xi < 0.0) xi = 0.0;
                if (fabs(dxi) < 1e-12 * (fabs(xi) + 1e-30)) break;
            }
            if (xi < 0.0) xi = 0.0;
            if (xi > sqrt(W_raw)) xi = sqrt(W_raw);
        }
        double xi2 = xi * xi;
        // v_conv² = (1/8) · g · δ · ℓ² · ξ² / H_P · (H_P/ℓ)²   simplified:
        //         = (g · δ · ℓ² · ξ²) / (8 · H_P)
        // Equivalent classical form v_conv = α · √(g·δ·H_P·ξ²/8)
        double v2 = (g * delta_val * ell * ell * xi2) / fmax(8.0 * H_P, 1e-30);
        v_conv = v2 > 0.0 ? sqrt(v2) : 0.0;
    }
    out_conv_vel[k] = v_conv;
}

// Host wrapper: download all rich profile fields.
void Radial1DSolver::download_profile_rich(
    std::vector<double>& r_face, std::vector<double>& v_face,
    std::vector<double>& rho_cell, std::vector<double>& P_cell,
    std::vector<double>& e_cell, std::vector<double>& T_cell,
    std::vector<double>& kap_cell, std::vector<double>& gamma1_cell,
    std::vector<double>& grada_cell, std::vector<double>& gradr_cell,
    std::vector<double>& L_face, std::vector<int>& mixing_type,
    std::vector<double>& conv_vel)
{
    int nz = lev.nz, B = 256;
    r_face.resize(nz+1);
    v_face.resize(nz+1);
    rho_cell.resize(nz); P_cell.resize(nz); e_cell.resize(nz);
    T_cell.resize(nz); kap_cell.resize(nz); gamma1_cell.resize(nz);
    grada_cell.resize(nz); gradr_cell.resize(nz);
    L_face.resize(nz); mixing_type.resize(nz); conv_vel.resize(nz);

    // Refresh primitives + enclosed mass so rho, P, M are consistent.
    launch_primitives(lev, nz, use_eos, gamma, eos, B);
    k_rad1d_enclosed_mass<<<1, 1>>>(lev.d_dm, lev.d_M, nz);

    // Allocate device scratch for rich fields.
    double *d_T, *d_kap, *d_g1, *d_ga, *d_gr, *d_Lf, *d_vc;
    int *d_mt;
    CUDA_CHECK(cudaMalloc(&d_T,   nz*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_kap, nz*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_g1,  nz*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ga,  nz*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gr,  nz*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Lf,  nz*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vc,  nz*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mt,  nz*sizeof(int)));

    OpacityParams opa;
    fill_opacity_params(opa);
    k_rad1d_rich_diag<<<(nz+B-1)/B, B>>>(
        lev.d_r, lev.d_rho, lev.d_P, lev.d_e_int, lev.d_M,
        d_T, d_kap, d_g1, d_ga, d_gr, d_Lf, d_mt, d_vc,
        nz, eos, rad_a_rad, rad_c_light, G_const, opa);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download everything.
    CUDA_CHECK(cudaMemcpy(r_face.data(),   lev.d_r,    (nz+1)*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_face.data(),   lev.d_v,    (nz+1)*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rho_cell.data(), lev.d_rho,  nz*sizeof(double),     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(P_cell.data(),   lev.d_P,    nz*sizeof(double),     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(e_cell.data(),   lev.d_e_int,nz*sizeof(double),     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(T_cell.data(),   d_T,        nz*sizeof(double),     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kap_cell.data(), d_kap,      nz*sizeof(double),     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gamma1_cell.data(), d_g1,    nz*sizeof(double),     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grada_cell.data(),  d_ga,    nz*sizeof(double),     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gradr_cell.data(),  d_gr,    nz*sizeof(double),     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(L_face.data(),      d_Lf,    nz*sizeof(double),     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(conv_vel.data(),    d_vc,    nz*sizeof(double),     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mixing_type.data(), d_mt,    nz*sizeof(int),        cudaMemcpyDeviceToHost));

    cudaFree(d_T); cudaFree(d_kap); cudaFree(d_g1); cudaFree(d_ga);
    cudaFree(d_gr); cudaFree(d_Lf); cudaFree(d_vc); cudaFree(d_mt);
}

Radial1DSolver::ConvectionDiag Radial1DSolver::compute_convection_diag() {
    ConvectionDiag d{};
    if (!use_eos) return d;  // need EOS for T from e
    int nz = lev.nz, B = 256;

    // Refresh primitives + enclosed mass
    launch_primitives(lev, nz, use_eos, gamma, eos, B);
    k_rad1d_enclosed_mass<<<1, 1>>>(lev.d_dm, lev.d_M, nz);

    std::vector<double*> sc(4);
    for (int i = 0; i < 4; ++i) CUDA_CHECK(cudaMalloc(&sc[i], nz*sizeof(double)));

    OpacityParams opa;
    fill_opacity_params(opa);

    k_rad1d_mlt_diag<<<(nz+B-1)/B, B>>>(
        lev.d_r, lev.d_rho, lev.d_P, lev.d_e_int, lev.d_M, lev.d_dm,
        sc[0], sc[1], sc[2], sc[3],
        nz, eos, rad_a_rad, rad_c_light, G_const, opa);

    // Host-side reductions + r_conv_inner/outer from host array
    std::vector<double> h_super(nz), h_isc(nz), h_cmass(nz);
    CUDA_CHECK(cudaMemcpy(h_super.data(), sc[0], nz*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_isc.data(),   sc[1], nz*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cmass.data(), sc[2], nz*sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> h_r(nz+1), h_dm(nz);
    CUDA_CHECK(cudaMemcpy(h_r.data(),  lev.d_r,  (nz+1)*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dm.data(), lev.d_dm, nz*sizeof(double),     cudaMemcpyDeviceToHost));

    double M_tot = 0.0, M_conv = 0.0, max_super = -1e300;
    int first = -1, last = -1, n_conv = 0;
    for (int k = 0; k < nz; ++k) {
        M_tot += h_dm[k];
        M_conv += h_cmass[k];
        if (h_super[k] > max_super) max_super = h_super[k];
        if (h_isc[k] > 0.5) {
            if (first < 0) first = k;
            last = k;
            ++n_conv;
        }
    }
    d.conv_mass_frac = (M_tot > 0) ? M_conv / M_tot : 0.0;
    d.max_superadiab = max_super;
    d.n_conv_zones   = n_conv;
    d.r_conv_inner   = (first >= 0) ? h_r[first]      : 0.0;
    d.r_conv_outer   = (last  >= 0) ? h_r[last + 1]   : 0.0;

    for (int i = 0; i < 4; ++i) cudaFree(sc[i]);
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

// ============================================================
// MLT conductivity per zone (Böhm-Vitense simplified).
// We use a "diffusive MLT" formulation (Eggleton 1971 / Henyey):
//   F_conv = -K_conv · dT/dr
// where
//   K_conv = ρ cp · ℓ_m · v_conv
//   ℓ_m    = α · H_P       (H_P = P / (ρ g))
//   v_conv = sqrt( g δ H_P (∇-∇_ad) ) / 2        (δ ≈ 1 for ideal gas)
// This is enabled only if ∇_rad > ∇_ad (Schwarzschild).
//
// Picard-lagged: K_conv is computed from T_p and treated as constant in
// the tridiag linear solve. This keeps the BE Jacobian clean — MLT flux
// is then linear in T (no T⁴ nonlinearity) and shows up as a standard
// conduction diffusion term.
// ============================================================
__global__ static void k_rad1d_mlt_cond(
    const double* T_p,       // (nz)
    const double* rho,       // (nz)
    const double* P,         // (nz)
    const double* M,         // (nz+1) enclosed mass at faces
    const double* r,         // (nz+1)
    double* K_conv,          // (nz)   output
    int nz, EOS eos, double G_const, double alpha_mlt,
    double a_rad, double c_light,
    OpacityParams opa)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nz) return;

    double rho_k = fmax(rho[k], 1e-30);
    double P_k   = fmax(P[k],   1e-30);
    double T_k   = fmax(T_p[k], 1e-12);
    double cv    = eos.cv();
    double gam   = eos.gamma;
    double cp    = gam * cv;
    // Cell-centre radius + gravity
    double r_c   = 0.5 * (r[k] + r[k+1]);
    double M_c   = 0.5 * (M[k] + M[k+1]);
    double g     = (r_c > 1e-30) ? G_const * M_c / (r_c * r_c) : 0.0;
    if (g <= 0.0) { K_conv[k] = 0.0; return; }

    // Pressure scale height H_P = P / (ρ g)
    double HP = P_k / (rho_k * g);
    double ell = alpha_mlt * HP;

    // Recover ∇_rad and ∇_ad — identical formulas to k_rad1d_mlt_diag.
    // Compute local L_rad at outer face of zone k (needed for ∇_rad).
    double kap_k = grey_opacity(rho_k, T_k, opa);
    double T_k_4 = T_k*T_k*T_k*T_k;
    double L_rad = 0.0;
    if (k < nz - 1) {
        double rho_kp1 = fmax(rho[k+1], 1e-30);
        double T_kp1   = fmax(T_p[k+1], 1e-12);
        double T_face  = 0.5 * (T_k + T_kp1);
        double rho_face= 0.5 * (rho_k + rho_kp1);
        double kap_f   = grey_opacity(rho_face, T_face, opa);
        double D       = c_light / (3.0 * kap_f * rho_face);
        double A_face  = 4.0 * 3.14159265358979323846 * r[k+1] * r[k+1];
        double rc_lo   = 0.5 * (r[k]   + r[k+1]);
        double rc_hi   = 0.5 * (r[k+1] + r[k+2]);
        double dr_zc   = fmax(rc_hi - rc_lo, 1e-30);
        double T_kp1_4 = T_kp1*T_kp1*T_kp1*T_kp1;
        L_rad = A_face * D * a_rad * (T_k_4 - T_kp1_4) / dr_zc;
    } else {
        double A_s = 4.0 * 3.14159265358979323846 * r[k+1] * r[k+1];
        double sigma_sb = c_light * a_rad / 4.0;
        L_rad = A_s * sigma_sb * T_k_4;
    }
    double M_out = M[k+1];
    double denom = 16.0 * 3.14159265358979323846 * a_rad * c_light * G_const * fmax(M_out, 1e-30) * T_k_4;
    double grad_rad = (denom > 1e-300) ? (3.0 * kap_k * rho_k * L_rad * P_k) / denom : 0.0;
    double grad_ad  = (gam - 1.0) / gam;
    double super    = grad_rad - grad_ad;
    if (super <= 0.0) { K_conv[k] = 0.0; return; }

    // v_conv from Böhm-Vitense:  v² = (g·δ·HP/8) · (∇−∇_ad)
    // δ ≈ 1 for ideal gas without ionization; factor 1/8 from (α/2)²/2.
    double v_conv = sqrt(fmax(g * HP * super / 8.0, 0.0));
    // Cap v_conv at c_s (Mach=1) so BE doesn't produce runaway transport.
    double cs = eos.sound_speed(rho_k, P_k);
    if (v_conv > cs) v_conv = cs;

    // Diffusion-limit cap on K_conv: don't let the convective flux exceed
    // what radiative diffusion delivers at the same level × some multiplier.
    //   D_rad = c / (3 κ ρ)    (rad diffusivity, cm²/s)
    //   K_rad_equiv = ρ cp D_rad   (erg/s/cm/K equivalent)
    // Convection in pre-MS envelopes really is ~100-1000× more efficient
    // than rad — leave plenty of headroom but don't blow up.
    double kap_c  = kap_k > 1e-30 ? kap_k : 1e-30;
    double D_rad  = c_light / (3.0 * kap_c * rho_k);
    double K_rad  = rho_k * cp * D_rad;
    double K_cap  = 1.0e4 * K_rad;  // 10⁴× rad diffusion; plenty of margin

    double K_raw = rho_k * cp * ell * v_conv;
    K_conv[k] = fmin(K_raw, K_cap);
}

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

// Picard-lag MLT refresh for rad-in-F path. Builds K_conv[0..nz-1] from
// current device state (ρ, P, T(ρ,e)). Called from the implicit Newton outer
// loop so each iter sees an up-to-date but frozen convective conductivity.
void Radial1DSolver::refresh_K_conv_implicit() {
    if (!mlt_enabled || !radiation_enabled || !use_eos) return;
    int nz = lev.nz, B = 256;
    if (d_K_conv == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_K_conv, nz * sizeof(double)));
    }
    // Need enclosed mass M[0..nz] at faces and T at zone centres. Scratch
    // T buffer reused per call — small (nz doubles).
    static double* d_T_scratch = nullptr;
    static int cached_nz = 0;
    if (cached_nz != nz) {
        if (d_T_scratch) cudaFree(d_T_scratch);
        CUDA_CHECK(cudaMalloc(&d_T_scratch, nz * sizeof(double)));
        cached_nz = nz;
    }
    k_rad1d_enclosed_mass<<<1, 1>>>(lev.d_dm, lev.d_M, nz);
    k_rad1d_T_from_rhoe<<<(nz+B-1)/B, B>>>(
        lev.d_rho, lev.d_e_int, d_T_scratch, nz, eos);
    OpacityParams opa;
    fill_opacity_params(opa);
    k_rad1d_mlt_cond<<<(nz+B-1)/B, B>>>(
        d_T_scratch, lev.d_rho, lev.d_P, lev.d_M, lev.d_r,
        d_K_conv, nz, eos, G_const, mlt_alpha,
        rad_a_rad, rad_c_light, opa);
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
    double dt,
    const double* K_conv,   // (nz) MLT conductivity, nullptr ⇒ no MLT
    int k_start)            // solve only zones [k_start, nz). 0 = full grid.
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nz) return;
    // Outside the active range: assemble identity row (zero δT solution).
    if (k < k_start) {
        A_diag[k] = 0.0;
        B_diag[k] = 1.0;
        C_diag[k] = 0.0;
        rhs[k]    = 0.0;
        return;
    }

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

        // ---- MLT convective flux at inner face k ----
        // F_conv_face = A · K_face · (T_{k-1} − T_k) / dr_zc
        // Linear in T ⇒ constant Jacobian (after Picard lag on K_face).
        if (K_conv != nullptr) {
            double K_face = 0.5 * (K_conv[k-1] + K_conv[k]);
            if (K_face > 0.0) {
                double cf = A_face * K_face / dr_zc;
                L_face_in    += cf * (Tkm1 - Tk);
                dL_in_dTkm1  +=  cf;
                dL_in_dTk    += -cf;
            }
        }
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

        // ---- MLT convective flux at outer face k+1 ----
        if (K_conv != nullptr) {
            double K_face = 0.5 * (K_conv[k] + K_conv[k+1]);
            if (K_face > 0.0) {
                double cf = A_face * K_face / dr_zc;
                L_face_out   += cf * (Tk - Tkp1);
                dL_out_dTk   +=  cf;
                dL_out_dTkp1 += -cf;
            }
        }
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
    // For k = k_start, T[k-1] is Dirichlet (part of the implicitly-solved
    // inner block, not a BE unknown here). Zero its Jacobian coupling so
    // the tridiag doesn't try to solve for δT_{k-1} via the a_ off-diagonal.
    // L_face_in at T_p(T_{k-1}, T_k) remains correct (it's the nonlinear
    // value used on the RHS).
    double a_coup = (k == k_start) ? 0.0 : dL_in_dTkm1;
    double a_ = -a_coup;
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

// Kernel: update e_int given T_new from BE-rad Picard solve.
//
// Correctness matters here: for Helmholtz EOS, e is a highly non-linear
// function of T (ideal ions + radiation ~T⁴ + partially degenerate
// electrons). The earlier linearization Δe = eos.cv() · ΔT used the
// IDEAL-GAS cv from EOS::cv() = 1/μ/(γ-1) — for Helm that gives ≈1 erg/g/K
// while the real pre-MS core cv is ≈1e8 erg/g/K. That undercounted
// radiation cooling by eight orders of magnitude and froze T_c.
//
// Fix: evaluate the EOS directly at (ρ, T_new) to get e exactly. For IDEAL
// / IDEAL_RAD we still use the local-cv linearization since those paths
// don't have a cheap T→e inverse kernel.
__global__ static void k_rad1d_apply_dT(
    double* e_int, const double* rho, const double* T_new, const double* T_start,
    int nz, EOS eos, int k_start)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nz) return;
    if (k < k_start) return;  // split mode: don't overwrite inner e
    double rho_k = fmax(rho[k], 1e-30);
    double T_k   = fmax(T_new[k], 1e-12);
#ifdef USE_GPU
    if (eos.type == (int)EosType::HELMHOLTZ) {
        HelmState s = helm_eval(rho_k, T_k, eos.helm);
        e_int[k] = fmax(s.e, 1e-30);
        return;
    }
#endif
    if (eos.type == (int)EosType::IDEAL_RAD) {
        // e_tot = cv·T + a·T^4 / ρ
        double cv_gas = eos.cv();
        double T4 = T_k * T_k * T_k * T_k;
        double e_tot = cv_gas * T_k + eos.radiation_a * T4 / rho_k;
        e_int[k] = fmax(e_tot, 1e-30);
        return;
    }
    // IDEAL (and PRE_MS as a last-resort linearization)
    double cv = eos.cv();
    double dT = T_k - T_start[k];
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

int Radial1DSolver::apply_radiation_diffusion_implicit(double dt_total, int k_start) {
    if (!radiation_enabled || !use_eos) return 0;
    int nz = lev.nz, B = 256;
    if (k_start < 0) k_start = 0;
    if (k_start >= nz) return 0;

    OpacityParams opa;
    fill_opacity_params(opa);
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
    // Allocate d_K_conv lazily when MLT is enabled (persists in struct).
    if (mlt_enabled && d_K_conv == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_K_conv, nz*sizeof(double)));
    }
    const double* K_conv_ptr = mlt_enabled ? d_K_conv : nullptr;

    // Refresh enclosed mass for MLT gravity
    if (mlt_enabled) {
        k_rad1d_enclosed_mass<<<1, 1>>>(lev.d_dm, lev.d_M, nz);
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
        if (mlt_enabled) {
            k_rad1d_mlt_cond<<<(nz+B-1)/B, B>>>(
                d_Tp, lev.d_rho, lev.d_P, lev.d_M, lev.d_r,
                d_K_conv, nz, eos, G_const, mlt_alpha,
                rad_a_rad, rad_c_light, opa);
        }
        k_rad1d_be_assemble<<<(nz+B-1)/B, B>>>(
            d_Tp, d_Tn, lev.d_rho, lev.d_r, lev.d_dm,
            d_A, d_Bm, d_Cm, d_R,
            nz, eos, rad_c_light, rad_a_rad, opa, sigma_sb, dt_total,
            K_conv_ptr, k_start);

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

    // Apply final T back to e_int. For Helm/IDEAL_RAD we re-evaluate the EOS
    // at (ρ, T_new) exactly; for IDEAL we linearize Δe = cv · ΔT.
    k_rad1d_apply_dT<<<(nz+B-1)/B, B>>>(lev.d_e_int, lev.d_rho, d_Tp, d_Tn, nz, eos, k_start);

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
    fill_opacity_params(pars.opacity);

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
