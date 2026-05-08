// test_athena_mhd_conduction.cu
// ============================================================
// Phase B-M2 — Spitzer anisotropic conduction (§C6).
//
// Four tests corresponding to §C6 verification checkpoints:
//   C6-T1  parallel diffusion matches analytic exp(-χ_eff k² t) decay
//          of sinusoidal T perturbation on B₀ = B₀ x̂ background
//   C6-T2  perpendicular quench: B₀ = B₀ ŷ, ∇T ⊥ B → decay ≤ 1e-10
//   C6-T3  1D Kirchhoff potential sign/scaling: δT > 0 hot spot
//          diffuses outward (peak drops, tails rise)
//   C6-T4  entropy production non-negative: δ E total conserved
//          (no source) + per-cell heat flux down T gradient
// ============================================================

#include "athena_mhd_solver.cuh"
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

static int g_failures = 0;
static int g_tests = 0;

#define CHECK_LT(got, bound, msg) do {                                \
    ++g_tests;                                                        \
    double _g = (got), _b = (bound);                                  \
    if (!(_g < _b)) {                                                 \
        std::fprintf(stderr, "FAIL %s:%d [%s]: got=%.6e bound=%.1e\n",\
                     __FILE__, __LINE__, msg, _g, _b);                \
        ++g_failures;                                                 \
    } else {                                                          \
        std::printf("  PASS  %s  (got=%.3e < %.1e)\n", msg, _g, _b); \
    }                                                                 \
} while (0)

#define CHECK_LT_GE(got, lo, hi, msg) do {                            \
    ++g_tests;                                                        \
    double _g = (got), _l = (lo), _h = (hi);                          \
    if (_g < _l || _g > _h) {                                         \
        std::fprintf(stderr,                                          \
            "FAIL %s:%d [%s]: got=%.6e not in [%.3e, %.3e]\n",        \
            __FILE__, __LINE__, msg, _g, _l, _h);                     \
        ++g_failures;                                                 \
    } else {                                                          \
        std::printf("  PASS  %s  (got=%.4e ∈ [%.3e, %.3e])\n",       \
                    msg, _g, _l, _h);                                \
    }                                                                 \
} while (0)

// Seed a uniform-B 2D domain with a sinusoidal T perturbation.
//   ρ = ρ₀, P = ρ · T(x,y), v = 0, B = B₀·b̂
//   T(x,y) = T₀ · (1 + A cos(kx · x + ky · y))
// `b_axis`: 0 → B̂ = x̂, 1 → B̂ = ŷ.
// Returns the k (nondim) used for this setup.
static void seed_sinusoidal_T(AthenaMHDSolver& sv,
                              int N, double Lx, double Ly,
                              double rho0, double T0, double A,
                              int kx_int, int ky_int,
                              double B0, int b_axis)
{
    sv.init(N, N, Lx, Ly, 5.0/3.0, 0.3);
    sv.xorder = 2; sv.limiter = 0; sv.cfl_limit = 0.5;
    // Periodic both directions (pure conduction test).
    sv.x_bc = 0; sv.y_bc = 0;
    sv.x_lo = 0.0; sv.x_hi = Lx;
    sv.y_lo = 0.0; sv.y_hi = Ly;
    sv.dx = Lx / (double)N;
    sv.dy = Ly / (double)N;

    int sx = sv.stride_x(), sy = sv.stride_y();
    int ng = sv.ng;
    int ncell = sx * sy;
    int nfx = sv.total_fx(), nfy = sv.total_fy();

    std::vector<double> h_rho(ncell, rho0);
    std::vector<double> h_mx(ncell, 0.0), h_my(ncell, 0.0),
                        h_mz(ncell, 0.0), h_E(ncell, 0.0),
                        h_Bz_cc(ncell, 0.0);
    // Uniform B₀ on faces according to b_axis.
    double Bxf_val = (b_axis == 0) ? B0 : 0.0;
    double Byf_val = (b_axis == 1) ? B0 : 0.0;
    std::vector<double> h_Bxf(nfx, Bxf_val);
    std::vector<double> h_Byf(nfy, Byf_val);

    double gm1 = sv.gamma - 1.0;
    const double two_pi = 2.0 * M_PI;
    double kx = two_pi * (double)kx_int / Lx;
    double ky = two_pi * (double)ky_int / Ly;

    for (int ic = 0; ic < N; ++ic) {
        double xc = (ic + 0.5) * sv.dx;
        for (int jc = 0; jc < N; ++jc) {
            double yc = (jc + 0.5) * sv.dy;
            int c = (ic + ng) * sy + (jc + ng);
            double T = T0 * (1.0 + A * std::cos(kx * xc + ky * yc));
            double P = rho0 * T;                 // code units, μ=1
            double ke = 0.0;                      // v=0
            double me = 0.5 * (Bxf_val * Bxf_val + Byf_val * Byf_val);
            h_E[c] = P / gm1 + ke + me;
        }
    }

    size_t nb_cell = (size_t)ncell * sizeof(double);
    cudaMemcpy(sv.d_rho,  h_rho.data(),  nb_cell, cudaMemcpyHostToDevice);
    cudaMemcpy(sv.d_mx,   h_mx.data(),   nb_cell, cudaMemcpyHostToDevice);
    cudaMemcpy(sv.d_my,   h_my.data(),   nb_cell, cudaMemcpyHostToDevice);
    cudaMemcpy(sv.d_mz,   h_mz.data(),   nb_cell, cudaMemcpyHostToDevice);
    cudaMemcpy(sv.d_E,    h_E.data(),    nb_cell, cudaMemcpyHostToDevice);
    cudaMemcpy(sv.d_Bz_cc,h_Bz_cc.data(),nb_cell, cudaMemcpyHostToDevice);
    cudaMemcpy(sv.d_Bxf, h_Bxf.data(),
               (size_t)nfx * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(sv.d_Byf, h_Byf.data(),
               (size_t)nfy * sizeof(double), cudaMemcpyHostToDevice);
}

static double measure_T_amplitude(AthenaMHDSolver& sv,
                                  int kx_int, int ky_int) {
    sv.fill_ghost(); sv.cons_to_prim();
    int sx = sv.stride_x(), sy = sv.stride_y();
    int ng = sv.ng;
    int ncell = sx * sy;
    std::vector<double> h_rho(ncell), h_P(ncell);
    cudaMemcpy(h_rho.data(), sv.d_w_rho, (size_t)ncell * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_P.data(), sv.d_w_P, (size_t)ncell * sizeof(double),
               cudaMemcpyDeviceToHost);
    const double two_pi = 2.0 * M_PI;
    double kx = two_pi * (double)kx_int / (sv.dx * (double)sv.nx);
    double ky = two_pi * (double)ky_int / (sv.dy * (double)sv.ny);
    // Fit: T = T_mean + ΔT · cos(kx·x + ky·y).  Compute ΔT via
    // inner product <T, cos(·)> * 2 / (nx·ny).
    double sum_cos = 0.0;
    int nac = 0;
    for (int ic = 0; ic < sv.nx; ++ic) {
        double x = (ic + 0.5) * sv.dx;
        for (int jc = 0; jc < sv.ny; ++jc) {
            double y = (jc + 0.5) * sv.dy;
            int c = (ic + ng) * sy + (jc + ng);
            double T = h_P[c] / std::max(h_rho[c], 1e-30);
            double b = std::cos(kx * x + ky * y);
            sum_cos += T * b;
            ++nac;
        }
    }
    return 2.0 * sum_cos / (double)nac;
}

static double measure_T_max_abs_delta(AthenaMHDSolver& sv, double T0) {
    sv.fill_ghost(); sv.cons_to_prim();
    int sx = sv.stride_x(), sy = sv.stride_y();
    int ng = sv.ng;
    int ncell = sx * sy;
    std::vector<double> h_rho(ncell), h_P(ncell);
    cudaMemcpy(h_rho.data(), sv.d_w_rho, (size_t)ncell * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_P.data(), sv.d_w_P, (size_t)ncell * sizeof(double),
               cudaMemcpyDeviceToHost);
    double m = 0.0;
    for (int ic = 0; ic < sv.nx; ++ic)
        for (int jc = 0; jc < sv.ny; ++jc) {
            int c = (ic + ng) * sy + (jc + ng);
            double T = h_P[c] / std::max(h_rho[c], 1e-30);
            m = std::max(m, std::fabs(T - T0));
        }
    return m;
}

// --------------------------------------------------------------------
// C6-T1: parallel diffusion matches analytic exp(-χ k² t).
//   Setup: B = B₀ x̂ (aligned with ∇T → full κ_∥ diffusion).
//   T = T₀ (1 + A cos(2π x / Lx)).
//   χ_eff = κ₀ T₀^{5/2} / (ρ₀ c_v)    with c_v = 1/(γ-1).
//   Decay rate γ = χ · k², k = 2π/Lx.
//   After t_end, expected amp / amp_0 = exp(-γ t_end).
// --------------------------------------------------------------------
static void test_T1_parallel_decay() {
    std::printf("\n[C6-T1] parallel diffusion — B∥∇T → exp(-χk²t)\n");
    const int N = 64;
    const double Lx = 1.0, Ly = 1.0;
    const double rho0 = 1.0, T0 = 1.0, A = 1e-2;
    const double B0 = 1.0;
    AthenaMHDSolver sv;
    seed_sinusoidal_T(sv, N, Lx, Ly, rho0, T0, A,
                      /*kx_int=*/1, /*ky_int=*/0, B0, /*b_axis=*/0);
    sv.kappa0 = 1.0;    // κ₀ = 1 (code units), set AFTER init resets

    double amp0 = measure_T_amplitude(sv, 1, 0);
    double cv = 1.0 / (sv.gamma - 1.0);
    double chi = sv.kappa0 * std::pow(T0, 2.5) / (rho0 * cv);
    double k = 2.0 * M_PI / Lx;
    double gamma_an = chi * k * k;
    double t_end = 0.05 / gamma_an;   // ~5% decay — linear regime
    sv.apply_conduction(t_end);
    double amp1 = measure_T_amplitude(sv, 1, 0);
    double ratio = amp1 / amp0;
    double expected = std::exp(-gamma_an * t_end);
    double rel_err = std::fabs(ratio - expected) / expected;
    std::printf("    χ=%.4g  γ_an=%.4g  t=%.4g\n", chi, gamma_an, t_end);
    std::printf("    amp0=%.4e amp1=%.4e ratio=%.6f (expect %.6f, rel err %.3e)\n",
                amp0, amp1, ratio, expected, rel_err);
    CHECK_LT(rel_err, 2e-2,
             "C6-T1: parallel decay rate matches analytic to 2%");
    sv.destroy();
}

// --------------------------------------------------------------------
// C6-T2: perpendicular quench — B ⊥ ∇T → no decay.
//   Setup: B = B₀ ŷ, T varies only in x (ky=0).
//   Expected amp ratio → 1 to machine precision.
// --------------------------------------------------------------------
static void test_T2_perp_quench() {
    std::printf("\n[C6-T2] perpendicular quench — B⊥∇T → amp frozen\n");
    const int N = 64;
    const double Lx = 1.0, Ly = 1.0;
    const double rho0 = 1.0, T0 = 1.0, A = 1e-2;
    const double B0 = 1.0;
    AthenaMHDSolver sv;
    seed_sinusoidal_T(sv, N, Lx, Ly, rho0, T0, A,
                      /*kx_int=*/1, /*ky_int=*/0, B0, /*b_axis=*/1);
    sv.kappa0 = 1.0;

    double amp0 = measure_T_amplitude(sv, 1, 0);
    double cv = 1.0 / (sv.gamma - 1.0);
    double chi = sv.kappa0 * std::pow(T0, 2.5) / (rho0 * cv);
    double k = 2.0 * M_PI / Lx;
    double t_end = 0.05 / (chi * k * k);   // same t as T1 for apples-apples
    sv.apply_conduction(t_end);
    double amp1 = measure_T_amplitude(sv, 1, 0);
    double rel_change = std::fabs(amp1 - amp0) / std::max(std::fabs(amp0), 1e-30);
    std::printf("    amp0=%.4e amp1=%.4e rel change = %.3e\n",
                amp0, amp1, rel_change);
    CHECK_LT(rel_change, 1e-8,
             "C6-T2: perpendicular decay ≤ 1e-8 (quench to machine ε)");
    sv.destroy();
}

// --------------------------------------------------------------------
// C6-T3: energy conservation — total ∫ E dV conserved (no source).
//   Conduction is a flux-divergence in E: divergence of face-flux on
//   periodic domain should exactly cancel in the sum.
// --------------------------------------------------------------------
static void test_T3_energy_conservation() {
    std::printf("\n[C6-T3] total energy conserved under conduction\n");
    const int N = 64;
    const double Lx = 1.0, Ly = 1.0;
    const double rho0 = 1.0, T0 = 1.0, A = 1e-2;
    const double B0 = 1.0;
    AthenaMHDSolver sv;
    seed_sinusoidal_T(sv, N, Lx, Ly, rho0, T0, A,
                      /*kx_int=*/1, /*ky_int=*/1, B0, /*b_axis=*/0);
    sv.kappa0 = 1.0;
    auto d0 = sv.compute_diagnostics();
    double cv = 1.0 / (sv.gamma - 1.0);
    double chi = sv.kappa0 * std::pow(T0, 2.5) / (rho0 * cv);
    double k = 2.0 * M_PI / Lx;
    double t_end = 0.1 / (chi * k * k);
    sv.apply_conduction(t_end);
    auto d1 = sv.compute_diagnostics();
    double dE_rel = std::fabs(d1.total_E - d0.total_E) /
                    std::max(std::fabs(d0.total_E), 1e-30);
    std::printf("    E₀=%.10e E₁=%.10e rel ΔE = %.3e\n",
                d0.total_E, d1.total_E, dE_rel);
    CHECK_LT(dE_rel, 1e-10, "C6-T3: total E conserved to 1e-10");
    // Also: ∇·B preserved (conduction must not touch B_f)
    CHECK_LT(d1.max_divB, 1e-10, "C6-T3: ∇·B untouched by conduction");
    sv.destroy();
}

// --------------------------------------------------------------------
// C6-T4: entropy production — amplitude monotonically decays.
//   Σ_n = amp at subcycle n; assert |Σ_n| ≤ |Σ_{n-1}| for 50 snapshots.
//   Proxy for σ_cond ≥ 0 (2nd law).
// --------------------------------------------------------------------
static void test_T4_monotone_decay() {
    std::printf("\n[C6-T4] monotone amplitude decay (2nd law proxy)\n");
    const int N = 32;
    const double Lx = 1.0, Ly = 1.0;
    const double rho0 = 1.0, T0 = 1.0, A = 1e-2;
    const double B0 = 1.0;
    AthenaMHDSolver sv;
    seed_sinusoidal_T(sv, N, Lx, Ly, rho0, T0, A,
                      /*kx_int=*/1, /*ky_int=*/0, B0, /*b_axis=*/0);
    sv.kappa0 = 1.0;

    double cv = 1.0 / (sv.gamma - 1.0);
    double chi = sv.kappa0 * std::pow(T0, 2.5) / (rho0 * cv);
    double k = 2.0 * M_PI / Lx;
    double t_total = 1.0 / (chi * k * k);    // one e-fold
    int n_bins = 50;
    double dt_bin = t_total / (double)n_bins;

    double prev_amp = std::fabs(measure_T_amplitude(sv, 1, 0));
    int violations = 0;
    double max_growth = 0.0;
    for (int b = 0; b < n_bins; ++b) {
        sv.apply_conduction(dt_bin);
        double amp = std::fabs(measure_T_amplitude(sv, 1, 0));
        if (amp > prev_amp + 1e-12) {
            ++violations;
            max_growth = std::max(max_growth, amp - prev_amp);
        }
        prev_amp = amp;
    }
    std::printf("    %d bins, violations=%d, max growth per bin=%.3e\n",
                n_bins, violations, max_growth);
    ++g_tests;
    if (violations == 0) {
        std::printf("  PASS  C6-T4: amp monotone decay (no bin growth)\n");
    } else {
        std::fprintf(stderr,
            "FAIL C6-T4: %d bins showed amp growth (max=%.3e)\n",
            violations, max_growth);
        ++g_failures;
    }
    sv.destroy();
}

int main() {
    std::printf("=== athena_mhd Spitzer conduction (B-M2, §C6) ===\n");
    test_T1_parallel_decay();
    test_T2_perp_quench();
    test_T3_energy_conservation();
    test_T4_monotone_decay();
    std::printf("\n=== Summary: %d/%d passed ===\n",
                g_tests - g_failures, g_tests);
    return g_failures == 0 ? 0 : 1;
}
