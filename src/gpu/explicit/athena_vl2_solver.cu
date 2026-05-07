// ============================================================
// athena_vl2_solver.cu — driver-level host code for the Athena++ vl2
// port to CUDA.  Kernels live in athena_vl2_kernels.cu.
// ============================================================

#include "athena_vl2_solver.cuh"
#include "gpu/common/gpu_common.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// ---- kernels (forward decl, defined in athena_vl2_kernels.cu) ----
__global__ void k_athvl2_cons_to_prim(
    const double* rho, const double* mx, const double* my, const double* E,
    const double* s,
    double* w_rho, double* w_u, double* w_v, double* w_P,
    double* w_X,
    int sx, int sy, double gm1);
__global__ void k_athvl2_fill_ghost_x_periodic(
    double* rho, double* mx, double* my, double* E, double* s,
    int nx, int ng, int sx, int sy);
__global__ void k_athvl2_fill_ghost_y_reflect(
    double* rho, double* mx, double* my, double* E, double* s,
    int ny, int ng, int sx, int sy);
__global__ void k_athvl2_fill_ghost_y_periodic(
    double* rho, double* mx, double* my, double* E, double* s,
    int ny, int ng, int sx, int sy);
__global__ void k_athvl2_flux_x(
    const double* w_rho, const double* w_u, const double* w_v,
    const double* w_P, const double* w_X,
    double* Fx_rho, double* Fx_mx, double* Fx_my, double* Fx_E,
    double* Fx_s,
    int nx, int ng, int sx, int sy,
    int order, int limiter, double gamma);
__global__ void k_athvl2_flux_y(
    const double* w_rho, const double* w_u, const double* w_v,
    const double* w_P, const double* w_X,
    double* Fy_rho, double* Fy_mx, double* Fy_my, double* Fy_E,
    double* Fy_s,
    int ny, int ng, int sx, int sy,
    int order, int limiter, double gamma);
__global__ void k_athvl2_flux_divergence(
    const double* u_rho, const double* u_mx, const double* u_my, const double* u_E,
    const double* u_s,
    double* u_rho_dst, double* u_mx_dst, double* u_my_dst, double* u_E_dst,
    double* u_s_dst,
    const double* Fx_rho, const double* Fx_mx, const double* Fx_my, const double* Fx_E,
    const double* Fx_s,
    const double* Fy_rho, const double* Fy_mx, const double* Fy_my, const double* Fy_E,
    const double* Fy_s,
    int nx, int ny, int ng, int sx, int sy,
    double dx, double dy, double dt);
__global__ void k_athvl2_source_terms(
    const double* w_rho, const double* w_v,
    double* u_mx, double* u_my, double* u_E,
    const double* g_row, const double* q_row,
    int nx, int ny, int ng, int sx, int sy, double dt);
__global__ void k_athvl2_cfl(
    const double* w_rho, const double* w_u, const double* w_v, const double* w_P,
    double* dt_buf,
    int nx, int ny, int ng, int sx, int sy,
    double dx, double dy, double gamma);

// ============================================================
// init / destroy
// ============================================================
void AthenaVL2Solver::init(int nx_, int ny_, double Lx_, double Ly_,
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

    int sx = stride_x();
    int sy = stride_y();
    int ncell = sx * sy;
    int nfx = total_fx();
    int nfy = total_fy();
    size_t nb_cell = (size_t)ncell * sizeof(double);
    size_t nb_fx   = (size_t)nfx   * sizeof(double);
    size_t nb_fy   = (size_t)nfy   * sizeof(double);

    auto alloc_zero = [&](double** p, size_t nbytes) {
        CUDA_CHECK(cudaMalloc(p, nbytes));
        CUDA_CHECK(cudaMemset(*p, 0, nbytes));
    };
    alloc_zero(&d_rho, nb_cell); alloc_zero(&d_mx, nb_cell);
    alloc_zero(&d_my,  nb_cell); alloc_zero(&d_E,  nb_cell);
    alloc_zero(&d_rho1, nb_cell); alloc_zero(&d_mx1, nb_cell);
    alloc_zero(&d_my1,  nb_cell); alloc_zero(&d_E1,  nb_cell);
    alloc_zero(&d_w_rho, nb_cell); alloc_zero(&d_w_u, nb_cell);
    alloc_zero(&d_w_v,   nb_cell); alloc_zero(&d_w_P, nb_cell);

    alloc_zero(&d_Fx_rho, nb_fx); alloc_zero(&d_Fx_mx, nb_fx);
    alloc_zero(&d_Fx_my,  nb_fx); alloc_zero(&d_Fx_E,  nb_fx);
    alloc_zero(&d_Fy_rho, nb_fy); alloc_zero(&d_Fy_mx, nb_fy);
    alloc_zero(&d_Fy_my,  nb_fy); alloc_zero(&d_Fy_E,  nb_fy);

    alloc_zero(&d_g_row, (size_t)ny * sizeof(double));
    alloc_zero(&d_q_row, (size_t)ny * sizeof(double));
    h_g_row.assign(ny, 0.0);
    h_q_row.assign(ny, 0.0);
    h_phi_row.assign(ny, 0.0);

    alloc_zero(&d_cfl_buf, (size_t)nx * (size_t)ny * sizeof(double));

    // Scalar (allocated only on init_andrassy2022 when tracer_enabled=true).
    d_s = nullptr; d_s1 = nullptr; d_w_X = nullptr;
    d_Fx_s = nullptr; d_Fy_s = nullptr;

    step_count = 0;
    dt_current = 0.0;
    ic_delta_amp = 0.0;
}

void AthenaVL2Solver::destroy() {
    auto F = [](double*& p) { if (p) { cudaFree(p); p = nullptr; } };
    F(d_rho); F(d_mx); F(d_my); F(d_E);
    F(d_rho1); F(d_mx1); F(d_my1); F(d_E1);
    F(d_w_rho); F(d_w_u); F(d_w_v); F(d_w_P);
    F(d_s); F(d_s1); F(d_w_X);
    F(d_Fx_rho); F(d_Fx_mx); F(d_Fx_my); F(d_Fx_E); F(d_Fx_s);
    F(d_Fy_rho); F(d_Fy_mx); F(d_Fy_my); F(d_Fy_E); F(d_Fy_s);
    F(d_g_row); F(d_q_row);
    F(d_cfl_buf);
}

// ============================================================
// Andrassy 2022 IC — reads the same 6-col slab as cart_ale2 and applies
// Eq. 6 δρ/ρ + optional hash-based ensemble noise.  Bit-identical to
// cart_ale2's IC on paper-exact seed (noise_seed < 0).
// ============================================================
void AthenaVL2Solver::init_andrassy2022(const std::string& slab_file,
                                        double delta_rho_amp,
                                        int noise_seed,
                                        double noise_amp) {
    std::FILE* fp = std::fopen(slab_file.c_str(), "r");
    if (!fp) {
        std::fprintf(stderr,
            "  AthenaVL2 init_andrassy2022: cannot open %s\n", slab_file.c_str());
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
        std::fprintf(stderr, "  AthenaVL2 init_andrassy2022: header missing\n");
        std::fclose(fp); std::abort();
    }
    double Ly_f, Lx_f, g_f, gam_f, rho_top, P_top, T_top, mu_f;
    if (std::sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf",
                    &Ly_f, &Lx_f, &g_f, &gam_f,
                    &rho_top, &P_top, &T_top, &mu_f) != 8) {
        std::fprintf(stderr, "  AthenaVL2 init_andrassy2022: bad header\n");
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
    if (n_face < 2 || qs_file.empty()) {
        std::fprintf(stderr,
            "  AthenaVL2 init_andrassy2022: slab must be 6-col with ≥ 2 rows\n");
        std::abort();
    }
    if (std::fabs(gamma - gam_f) > 1e-6) {
        std::fprintf(stderr,
            "  [warn] AthenaVL2: slab γ=%g vs solver γ=%g\n", gam_f, gamma);
    }
    // The paper uses a y ∈ [1, 3] slab.  We keep the same physical coords
    // (y_lo = 1, y_hi = 3) — init() already set y_lo=0 based on Ly, so
    // override here for Andrassy2022.
    y_lo = ys.front();
    y_hi = ys.back();
    Ly = y_hi - y_lo;
    dy = Ly / (double)ny;
    // x stays in [-Lx/2, +Lx/2] convention (Andrassy uses [-1, 1]).
    // Our `Lx` was set from the slab's Lx_f (matches paper).
    x_lo = -0.5 * Lx;
    x_hi =  0.5 * Lx;
    dx = Lx / (double)nx;

    // Linear interpolation on the slab.
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
    if (q0_max <= 0.0) q0_max = 1.0;

    // Build host conserved arrays (interior only; ghosts filled in first
    // fill_ghost() call).
    int sx = stride_x();
    int sy = stride_y();
    int ncell = sx * sy;
    std::vector<double> h_rho(ncell, 0.0), h_mx(ncell, 0.0),
                        h_my(ncell, 0.0), h_E(ncell, 0.0);
    std::vector<double> h_s(ncell, 0.0);
    bool want_tracer = true;    // Andrassy2022 tracks μ₁ mass fraction
    tracer_enabled = want_tracer;

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

    // Andrassy Eq. 3 ramp (Y_CB = 2, δ = 1/16) in paper coordinates.
    // Our slab uses slab coordinate yc ∈ [0, Ly] with slab origin at
    // paper y = 1, so y_paper = yc + 1.  Equivalently the ramp center in
    // slab coords is Y_CB_slab = 1 with the same half-width 1/16.
    auto eta1 = [](double yc_slab) -> double {
        double Y_CB = 1.0;           // slab coord of convective boundary
        double DY  = 1.0 / 16.0;
        double lo = Y_CB - DY, hi = Y_CB + DY;
        if (yc_slab <= lo) return 0.0;
        if (yc_slab >= hi) return 1.0;
        double s = (yc_slab - lo) / (2.0 * DY);
        return 0.5 * (1.0 - std::cos(M_PI * s));
    };

    double gm1 = gamma - 1.0;
    for (int ic = 0; ic < nx; ++ic) {
        double xc = x_lo + (ic + 0.5) * dx;
        for (int jc = 0; jc < ny; ++jc) {
            double yc = y_lo + (jc + 0.5) * dy;
            double rho_hse = interp(rhos,    yc);
            double P_hse   = interp(Ps,      yc);
            double q_env   = interp(qs_file, yc) / q0_max;
            // Eq. 6  x' = 2x/Lx with Lx = 2 ⇒ x' = x ∈ [−1, 1]
            double xp = 2.0 * xc / Lx;
            double x_factor = std::sin(3.0 * M_PI * xp) + std::cos(M_PI * xp);
            double d_rho_rel = delta_rho_amp * q_env * x_factor;
            if (noise_seed >= 0 && noise_amp > 0.0) {
                d_rho_rel += noise_amp * q_env * hash_rand(noise_seed, ic, jc);
            }
            double rho = rho_hse * (1.0 + d_rho_rel);
            double P   = P_hse;
            double e   = P / (gm1 * rho);       // internal energy per mass
            double X   = eta1(yc);
            // Layout matches kernels: cell flat = (ic+ng)*sy + (jc+ng)
            int c = (ic + ng) * sy + (jc + ng);
            h_rho[c] = rho;
            h_mx [c] = 0.0;
            h_my [c] = 0.0;
            h_E  [c] = rho * e;                 // KE = 0 at t = 0
            h_s  [c] = rho * X;
        }
    }

    size_t nb_cell = (size_t)ncell * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_rho, h_rho.data(), nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mx,  h_mx.data(),  nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_my,  h_my.data(),  nb_cell, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E,   h_E.data(),   nb_cell, cudaMemcpyHostToDevice));

    if (tracer_enabled) {
        CUDA_CHECK(cudaMalloc(&d_s,  nb_cell));
        CUDA_CHECK(cudaMalloc(&d_s1, nb_cell));
        CUDA_CHECK(cudaMalloc(&d_w_X, nb_cell));
        CUDA_CHECK(cudaMemset(d_s1, 0, nb_cell));
        CUDA_CHECK(cudaMemcpy(d_s, h_s.data(), nb_cell, cudaMemcpyHostToDevice));
        int nfx = total_fx(), nfy = total_fy();
        CUDA_CHECK(cudaMalloc(&d_Fx_s, (size_t)nfx * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Fy_s, (size_t)nfy * sizeof(double)));
    }

    // g(y) and q̇(y) at cell centres
    for (int jc = 0; jc < ny; ++jc) {
        double yc = y_lo + (jc + 0.5) * dy;
        h_g_row[jc] = interp(gs_file, yc);
        h_q_row[jc] = interp(qs_file, yc);
    }
    CUDA_CHECK(cudaMemcpy(d_g_row, h_g_row.data(),
                          ny * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_row, h_q_row.data(),
                          ny * sizeof(double), cudaMemcpyHostToDevice));

    // Potential table for PE diagnostic: Φ(y_j) with Φ(y_lo) = 0.
    h_phi_row.assign(ny, 0.0);
    double phi = 0.0;
    double y_prev = y_lo;
    double g_prev = h_g_row[0];
    for (int jc = 0; jc < ny; ++jc) {
        double yc = y_lo + (jc + 0.5) * dy;
        double g_mid = 0.5 * (g_prev + h_g_row[jc]);
        phi += g_mid * (yc - y_prev);            // Φ = ∫ g dy (positive upward)
        h_phi_row[jc] = phi;
        y_prev = yc;
        g_prev = h_g_row[jc];
    }

    ic_delta_amp = delta_rho_amp;

    double cs_bot = std::sqrt(gamma * Ps.front() / rhos.front());
    double cs_top = std::sqrt(gamma * Ps.back()  / rhos.back());
    std::fprintf(stderr,
        "  AthenaVL2 Andrassy 2022 IC: slab=%s\n"
        "    y=[%.3e, %.3e]  Lx=%.3e  γ=%.3f\n"
        "    c_s top=%.3e bot=%.3e,  τ_sc=Ly/c_s_top=%.3e\n"
        "    δρ/ρ Eq. 6 amp=%.3g  seed=%d noise=%.3g  (tracer=%s)\n",
        slab_file.c_str(), y_lo, y_hi, Lx, gamma,
        cs_top, cs_bot, Ly / cs_top, delta_rho_amp, noise_seed, noise_amp,
        tracer_enabled ? "ON" : "OFF");
}

// ============================================================
// init_shear_mode — T3 ν_eff probe IC (periodic BC both dirs)
// ============================================================
void AthenaVL2Solver::init_shear_mode(double rho, double P, double V0, int k) {
    // Anchor the domain at [0, Lx] × [0, Ly].
    x_lo = 0.0; x_hi = Lx;
    y_lo = 0.0; y_hi = Ly;
    dx = Lx / (double)nx;
    dy = Ly / (double)ny;
    y_periodic = true;
    tracer_enabled = false;

    int sx = stride_x();
    int sy = stride_y();
    int ncell = sx * sy;
    std::vector<double> h_rho(ncell, rho);
    std::vector<double> h_mx(ncell, 0.0);
    std::vector<double> h_my(ncell, 0.0);
    std::vector<double> h_E(ncell, 0.0);
    double e_int = P / (gamma - 1.0);
    double kphys = k * 2.0 * M_PI / Ly;
    for (int jc = 0; jc < ny; ++jc) {
        double yc = y_lo + (jc + 0.5) * dy;
        double vx = V0 * std::sin(kphys * yc);
        double ke = 0.5 * rho * vx * vx;
        for (int ic = 0; ic < nx; ++ic) {
            int cidx = (ic + ng) * sy + (jc + ng);
            h_rho[cidx] = rho;
            h_mx [cidx] = rho * vx;
            h_my [cidx] = 0.0;
            h_E  [cidx] = e_int + ke;
        }
    }
    // Zero per-row gravity/heating source tables.
    h_g_row.assign(ny, 0.0);
    h_q_row.assign(ny, 0.0);
    h_phi_row.assign(ny, 0.0);
    CUDA_CHECK(cudaMemcpy(d_rho, h_rho.data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mx,  h_mx .data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_my,  h_my .data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E,   h_E  .data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g_row, h_g_row.data(), ny*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_row, h_q_row.data(), ny*sizeof(double), cudaMemcpyHostToDevice));

    std::fprintf(stderr,
        "  AthenaVL2 shear_mode IC: rho=%g P=%g V0=%g k=%d  (Ly=%g k_phys=%g)\n",
        rho, P, V0, k, Ly, kphys);
}

// ============================================================
// init_entropy_wave — T1 smooth-convergence probe (periodic BC both dirs)
// ============================================================
void AthenaVL2Solver::init_entropy_wave(double rho0, double P0, double u0,
                                        double A, int k) {
    x_lo = 0.0; x_hi = Lx;
    y_lo = 0.0; y_hi = Ly;
    dx = Lx / (double)nx;
    dy = Ly / (double)ny;
    y_periodic = true;
    tracer_enabled = false;

    int sx = stride_x();
    int sy = stride_y();
    int ncell = sx * sy;
    std::vector<double> h_rho(ncell, rho0);
    std::vector<double> h_mx(ncell, 0.0);
    std::vector<double> h_my(ncell, 0.0);
    std::vector<double> h_E(ncell, 0.0);
    double gm1 = gamma - 1.0;
    for (int ic = 0; ic < nx; ++ic) {
        double xc = x_lo + (ic + 0.5) * dx;
        double rho = rho0 * (1.0 + A * std::sin(k * 2.0 * M_PI * xc / Lx));
        double e_int = P0 / gm1;
        double ke = 0.5 * rho * u0 * u0;
        for (int jc = 0; jc < ny; ++jc) {
            int cidx = (ic + ng) * sy + (jc + ng);
            h_rho[cidx] = rho;
            h_mx [cidx] = rho * u0;
            h_my [cidx] = 0.0;
            h_E  [cidx] = e_int + ke;
        }
    }
    h_g_row.assign(ny, 0.0);
    h_q_row.assign(ny, 0.0);
    h_phi_row.assign(ny, 0.0);
    CUDA_CHECK(cudaMemcpy(d_rho, h_rho.data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mx,  h_mx .data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_my,  h_my .data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E,   h_E  .data(), ncell*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g_row, h_g_row.data(), ny*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_row, h_q_row.data(), ny*sizeof(double), cudaMemcpyHostToDevice));

    std::fprintf(stderr,
        "  AthenaVL2 entropy_wave IC: rho0=%g P0=%g u0=%g A=%g k=%d  (Lx=%g, period=%g)\n",
        rho0, P0, u0, A, k, Lx, Lx / u0);
}

// ============================================================
// fill_ghost / cons_to_prim: dispatch helpers
// ============================================================
void AthenaVL2Solver::fill_ghost() {
    int sx = stride_x(), sy = stride_y();
    dim3 bx(64, 4), gx((sy + bx.x - 1) / bx.x, (ng + bx.y - 1) / bx.y);
    k_athvl2_fill_ghost_x_periodic<<<gx, bx>>>(
        d_rho, d_mx, d_my, d_E, d_s, nx, ng, sx, sy);
    dim3 by(64, 4), gy((sx + by.x - 1) / by.x, (ng + by.y - 1) / by.y);
    if (y_periodic) {
        k_athvl2_fill_ghost_y_periodic<<<gy, by>>>(
            d_rho, d_mx, d_my, d_E, d_s, ny, ng, sx, sy);
    } else {
        k_athvl2_fill_ghost_y_reflect<<<gy, by>>>(
            d_rho, d_mx, d_my, d_E, d_s, ny, ng, sx, sy);
    }
}

void AthenaVL2Solver::cons_to_prim() {
    int sx = stride_x(), sy = stride_y();
    double gm1 = gamma - 1.0;
    dim3 b(16, 16), g((sx + 15) / 16, (sy + 15) / 16);
    k_athvl2_cons_to_prim<<<g, b>>>(
        d_rho, d_mx, d_my, d_E, d_s,
        d_w_rho, d_w_u, d_w_v, d_w_P, d_w_X,
        sx, sy, gm1);
}

// ============================================================
// Stage advance: read u^n from (d_rho, d_mx, d_my, d_E),
//                read flux-input prim from (d_w_*),
//                write new state into the "dst" conserved arrays.
// beta = 0.5 for predictor, 1.0 for corrector.
// ============================================================
void AthenaVL2Solver::stage_advance(int stage, double dt) {
    int sx = stride_x(), sy = stride_y();

    // Flux computation: order depends on stage.
    int order = (stage == 1) ? 1 : xorder;   // Athena vl2: stage-1 is always donor-cell
    {
        // x flux: faces at i ∈ [ng-1, ng+nx] → width nx+2 faces, but the
        // kernel guard accepts the full range and simply no-ops outside.
        dim3 b(32, 8);
        dim3 gx((sx + b.x - 1) / b.x, (sy + b.y - 1) / b.y);
        k_athvl2_flux_x<<<gx, b>>>(
            d_w_rho, d_w_u, d_w_v, d_w_P, d_w_X,
            d_Fx_rho, d_Fx_mx, d_Fx_my, d_Fx_E, d_Fx_s,
            nx, ng, sx, sy, order, limiter, gamma);
        int sxp1 = sx;           // same stride layout (sx cols for Fy; sy+1 rows)
        dim3 gy((sxp1 + b.x - 1) / b.x, (sy + 1 + b.y - 1) / b.y);
        k_athvl2_flux_y<<<gy, b>>>(
            d_w_rho, d_w_u, d_w_v, d_w_P, d_w_X,
            d_Fy_rho, d_Fy_mx, d_Fy_my, d_Fy_E, d_Fy_s,
            ny, ng, sx, sy, order, limiter, gamma);
    }

    // Destination arrays
    double *dst_rho, *dst_mx, *dst_my, *dst_E, *dst_s;
    double *src_rho, *src_mx, *src_my, *src_E, *src_s;
    if (stage == 1) {
        // u* = u^n − (dt/2) ∇·F(u^n) + src(w^n; dt/2)
        src_rho = d_rho; src_mx = d_mx; src_my = d_my; src_E = d_E; src_s = d_s;
        dst_rho = d_rho1; dst_mx = d_mx1; dst_my = d_my1; dst_E = d_E1; dst_s = d_s1;
    } else {
        // u^{n+1} = u^n − dt ∇·F(u^*) + src(w^*; dt)
        src_rho = d_rho; src_mx = d_mx; src_my = d_my; src_E = d_E; src_s = d_s;
        dst_rho = d_rho1; dst_mx = d_mx1; dst_my = d_my1; dst_E = d_E1; dst_s = d_s1;
    }

    double dt_stage = (stage == 1) ? 0.5 * dt : dt;
    dim3 b(16, 16), gk((nx + 15) / 16, (ny + 15) / 16);
    k_athvl2_flux_divergence<<<gk, b>>>(
        src_rho, src_mx, src_my, src_E, src_s,
        dst_rho, dst_mx, dst_my, dst_E, dst_s,
        d_Fx_rho, d_Fx_mx, d_Fx_my, d_Fx_E, d_Fx_s,
        d_Fy_rho, d_Fy_mx, d_Fy_my, d_Fy_E, d_Fy_s,
        nx, ny, ng, sx, sy, dx, dy, dt_stage);

    // Source terms with stage's own β·dt, using the PRIMITIVE at START
    // of this stage (which is currently in d_w_*, populated by the most
    // recent cons_to_prim call).
    k_athvl2_source_terms<<<gk, b>>>(
        d_w_rho, d_w_v,
        dst_mx, dst_my, dst_E,
        d_g_row, d_q_row,
        nx, ny, ng, sx, sy, dt_stage);
}

// ============================================================
// One vl2 predictor-corrector step
// ============================================================
double AthenaVL2Solver::step(double t, double t_end) {
    // Ghost fill + prim at state u^n
    fill_ghost();
    cons_to_prim();

    double dt = compute_dt();
    if (t + dt > t_end) dt = t_end - t;
    dt_current = dt;

    // ---- stage 1 (predictor) ----
    // Uses w^n (currently in d_w_*), order=1 donor-cell, β=0.5.
    stage_advance(/*stage=*/1, dt);
    // Swap u^n → d_rho1 becomes u^*; we need d_rho to hold u^n for stage 2
    // so we DON'T overwrite d_rho.  Instead, use d_rho1 as the "read" for prim.
    // Update: re-fill-ghosts and cons_to_prim on (d_rho1, d_mx1, d_my1, d_E1, d_s1).
    std::swap(d_rho, d_rho1);
    std::swap(d_mx,  d_mx1);
    std::swap(d_my,  d_my1);
    std::swap(d_E,   d_E1);
    if (d_s) std::swap(d_s, d_s1);
    fill_ghost();     // now ghosts of u*
    cons_to_prim();   // now w* in d_w_*
    // Swap back so d_rho = u^n, d_rho1 = u* — stage_advance reads src = d_rho (u^n)
    // and prim from d_w_* (currently w*).
    std::swap(d_rho, d_rho1);
    std::swap(d_mx,  d_mx1);
    std::swap(d_my,  d_my1);
    std::swap(d_E,   d_E1);
    if (d_s) std::swap(d_s, d_s1);

    // ---- stage 2 (corrector) ----
    // Reads src u^n, flux uses w* (in d_w_*), β=1.0.
    stage_advance(/*stage=*/2, dt);

    // Final: u^{n+1} in d_rho1, promote to d_rho.
    std::swap(d_rho, d_rho1);
    std::swap(d_mx,  d_mx1);
    std::swap(d_my,  d_my1);
    std::swap(d_E,   d_E1);
    if (d_s) std::swap(d_s, d_s1);

    step_count++;
    return dt;
}

// ============================================================
// dt = CFL·min over interior cells
// ============================================================
double AthenaVL2Solver::compute_dt() {
    int sx = stride_x(), sy = stride_y();
    dim3 b(16, 16), g((nx + 15) / 16, (ny + 15) / 16);
    k_athvl2_cfl<<<g, b>>>(
        d_w_rho, d_w_u, d_w_v, d_w_P,
        d_cfl_buf, nx, ny, ng, sx, sy,
        dx, dy, gamma);
    // Host reduce (tiny, nx·ny ≤ 512²=262144).  Future: device reduction.
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
// Diagnostics
// ============================================================
AthenaVL2Solver::Diagnostics AthenaVL2Solver::compute_diagnostics() {
    int sx = stride_x(), sy = stride_y();
    int ncell = sx * sy;
    std::vector<double> h_rho(ncell), h_mx(ncell), h_my(ncell), h_E(ncell);
    size_t nb = (size_t)ncell * sizeof(double);
    CUDA_CHECK(cudaMemcpy(h_rho.data(), d_rho, nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mx.data(),  d_mx,  nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_my.data(),  d_my,  nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_E.data(),   d_E,   nb, cudaMemcpyDeviceToHost));

    double M = 0, KE = 0, IE = 0, PE = 0, vmax = 0, Mmax = 0;
    double dV = dx * dy;
    double gm1 = gamma - 1.0;
    for (int jc = 0; jc < ny; ++jc) {
        double phi = h_phi_row[jc];
        for (int ic = 0; ic < nx; ++ic) {
            int c = (ic + ng) * sy + (jc + ng);
            double r = h_rho[c];
            double u = h_mx[c] / std::max(r, 1e-30);
            double v = h_my[c] / std::max(r, 1e-30);
            double ke = 0.5 * r * (u*u + v*v);
            double ie = h_E[c] - ke;
            double P  = gm1 * ie;
            double cs = std::sqrt(gamma * std::max(P, 1e-30) / std::max(r, 1e-30));
            double speed = std::sqrt(u*u + v*v);
            M  += r * dV;
            KE += ke * dV;
            IE += ie * dV;
            PE += r * phi * dV;
            vmax = std::max(vmax, speed);
            Mmax = std::max(Mmax, speed / cs);
        }
    }
    return {M, KE, IE, PE, KE + IE + PE, vmax, Mmax};
}

double AthenaVL2Solver::total_species_mass() {
    if (!tracer_enabled || d_s == nullptr) return 0.0;
    int sx = stride_x(), sy = stride_y();
    int ncell = sx * sy;
    std::vector<double> h_s(ncell);
    CUDA_CHECK(cudaMemcpy(h_s.data(), d_s,
                          (size_t)ncell * sizeof(double),
                          cudaMemcpyDeviceToHost));
    double dV = dx * dy;
    double M = 0.0;
    for (int jc = 0; jc < ny; ++jc)
        for (int ic = 0; ic < nx; ++ic) {
            int c = (ic + ng) * sy + (jc + ng);
            M += h_s[c] * dV;
        }
    return M;
}

// ============================================================
// VTK write (same STRUCTURED_GRID format as cart_ale2, x ∈ [x_lo, x_hi])
// ============================================================
void AthenaVL2Solver::write_vtk_2d(const char* filename, double Lx_in, double Ly_in) {
    (void)Lx_in; (void)Ly_in;   // keep signature compatible with cart_ale2's
    int sx = stride_x(), sy = stride_y();
    int ncell = sx * sy;
    std::vector<double> h_rho(ncell), h_mx(ncell), h_my(ncell), h_E(ncell);
    size_t nb = (size_t)ncell * sizeof(double);
    CUDA_CHECK(cudaMemcpy(h_rho.data(), d_rho, nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mx.data(),  d_mx,  nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_my.data(),  d_my,  nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_E.data(),   d_E,   nb, cudaMemcpyDeviceToHost));
    std::vector<double> h_s;
    if (tracer_enabled && d_s) {
        h_s.resize(ncell);
        CUDA_CHECK(cudaMemcpy(h_s.data(), d_s, nb, cudaMemcpyDeviceToHost));
    }

    std::FILE* fp = std::fopen(filename, "w");
    if (!fp) return;
    int nnx = nx + 1, nny = ny + 1;
    std::fprintf(fp, "# vtk DataFile Version 3.0\n");
    std::fprintf(fp, "athena_vl2 2D Cartesian Eulerian output\n");
    std::fprintf(fp, "ASCII\n");
    std::fprintf(fp, "DATASET STRUCTURED_GRID\n");
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

    auto cell_scalar_cb = [&](const char* name, auto fn) {
        std::fprintf(fp, "SCALARS %s double 1\nLOOKUP_TABLE default\n", name);
        for (int jc = 0; jc < ny; ++jc)
            for (int ic = 0; ic < nx; ++ic) {
                int c = (ic + ng) * sy + (jc + ng);
                std::fprintf(fp, "%.10e\n", fn(c));
            }
    };
    cell_scalar_cb("density",  [&](int c) { return h_rho[c]; });
    cell_scalar_cb("pressure", [&](int c) {
        double r = std::max(h_rho[c], 1e-30);
        double u = h_mx[c] / r, v = h_my[c] / r;
        double ke = 0.5 * r * (u*u + v*v);
        return std::max(gm1 * (h_E[c] - ke), 1e-30);
    });
    cell_scalar_cb("e_int", [&](int c) {
        double r = std::max(h_rho[c], 1e-30);
        double u = h_mx[c] / r, v = h_my[c] / r;
        double ke = 0.5 * r * (u*u + v*v);
        return std::max(h_E[c] - ke, 1e-30) / r;
    });
    // Velocity vector (cell-centered — vl2 is cell-centered natively, no nodes to average)
    std::fprintf(fp, "VECTORS velocity double\n");
    for (int jc = 0; jc < ny; ++jc)
        for (int ic = 0; ic < nx; ++ic) {
            int c = (ic + ng) * sy + (jc + ng);
            double r = std::max(h_rho[c], 1e-30);
            double u = h_mx[c] / r, v = h_my[c] / r;
            std::fprintf(fp, "%.10e %.10e %.10e\n", u, v, 0.0);
        }
    std::fprintf(fp, "SCALARS mach double 1\nLOOKUP_TABLE default\n");
    for (int jc = 0; jc < ny; ++jc)
        for (int ic = 0; ic < nx; ++ic) {
            int c = (ic + ng) * sy + (jc + ng);
            double r = std::max(h_rho[c], 1e-30);
            double u = h_mx[c] / r, v = h_my[c] / r;
            double ke = 0.5 * r * (u*u + v*v);
            double P  = std::max(gm1 * (h_E[c] - ke), 1e-30);
            double cs = std::sqrt(gamma * P / r);
            double speed = std::sqrt(u*u + v*v);
            std::fprintf(fp, "%.10e\n", speed / cs);
        }
    if (tracer_enabled && !h_s.empty()) {
        std::fprintf(fp, "SCALARS species_X double 1\nLOOKUP_TABLE default\n");
        for (int jc = 0; jc < ny; ++jc)
            for (int ic = 0; ic < nx; ++ic) {
                int c = (ic + ng) * sy + (jc + ng);
                double X = h_s[c] / std::max(h_rho[c], 1e-30);
                std::fprintf(fp, "%.10e\n", X);
            }
    }
    std::fclose(fp);
}
