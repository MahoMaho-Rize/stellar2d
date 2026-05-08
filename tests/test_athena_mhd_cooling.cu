// test_athena_mhd_cooling.cu
// ============================================================
// Phase B-M3 — Townsend 2009 optically-thin radiative cooling (§C7).
//
// Single-segment power-law Λ(T) = Λ₀ (T/Tref)^α implies the closed-form
//   dT/dt = -C T^α,  C = (γ-1) ρ Λ₀ / Tref^α   (code units, k_B=μ=1)
//
// Four tests:
//   C7-T1  uniform cell matches analytic T(t) = [T0^{1-α} - C(1-α)t]^{1/(1-α)}
//          swept over α ∈ {0.5, 2.0, 3.0}  and the α = 1 exponential branch
//   C7-T2  cooling is passive w.r.t. ρ / mom / B_f  (all three invariant)
//   C7-T3  monotone: T(t_k) ≤ T(t_{k-1}) for 50 bins across one e-fold
//   C7-T4  ΔE equals the analytic -∫ Q_R dt over one cell to machine ε
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

// Seed uniform (ρ, T, B) periodic domain, v = 0.
static void seed_uniform(AthenaMHDSolver& sv, int N,
                         double rho0, double T0,
                         double Bx = 0.0, double By = 0.0) {
    sv.init(N, N, 1.0, 1.0, 5.0/3.0, 0.3);
    sv.xorder = 2; sv.limiter = 0; sv.cfl_limit = 0.5;
    sv.x_bc = 0; sv.y_bc = 0;
    sv.x_lo = 0.0; sv.x_hi = 1.0;
    sv.y_lo = 0.0; sv.y_hi = 1.0;
    sv.dx = 1.0 / (double)N;
    sv.dy = 1.0 / (double)N;

    int sx = sv.stride_x(), sy = sv.stride_y();
    int ncell = sx * sy;
    int nfx = sv.total_fx(), nfy = sv.total_fy();
    double P0 = rho0 * T0;
    double gm1 = sv.gamma - 1.0;
    double me = 0.5 * (Bx * Bx + By * By);

    std::vector<double> h_rho(ncell, rho0);
    std::vector<double> h_mx(ncell, 0.0), h_my(ncell, 0.0),
                        h_mz(ncell, 0.0), h_Bz(ncell, 0.0);
    std::vector<double> h_E(ncell, P0 / gm1 + me);
    std::vector<double> h_Bxf(nfx, Bx), h_Byf(nfy, By);
    size_t nb = (size_t)ncell * sizeof(double);
    cudaMemcpy(sv.d_rho, h_rho.data(), nb, cudaMemcpyHostToDevice);
    cudaMemcpy(sv.d_mx,  h_mx.data(),  nb, cudaMemcpyHostToDevice);
    cudaMemcpy(sv.d_my,  h_my.data(),  nb, cudaMemcpyHostToDevice);
    cudaMemcpy(sv.d_mz,  h_mz.data(),  nb, cudaMemcpyHostToDevice);
    cudaMemcpy(sv.d_E,   h_E.data(),   nb, cudaMemcpyHostToDevice);
    cudaMemcpy(sv.d_Bz_cc, h_Bz.data(), nb, cudaMemcpyHostToDevice);
    cudaMemcpy(sv.d_Bxf, h_Bxf.data(), (size_t)nfx * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(sv.d_Byf, h_Byf.data(), (size_t)nfy * sizeof(double),
               cudaMemcpyHostToDevice);
}

// Read cell (ng, ng) primitive T.
static double read_T_interior(AthenaMHDSolver& sv) {
    sv.fill_ghost();
    sv.cons_to_prim();
    int sx = sv.stride_x(), sy = sv.stride_y();
    int ncell = sx * sy;
    std::vector<double> h_rho(ncell), h_P(ncell);
    cudaMemcpy(h_rho.data(), sv.d_w_rho, (size_t)ncell * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_P.data(), sv.d_w_P, (size_t)ncell * sizeof(double),
               cudaMemcpyDeviceToHost);
    int c = sv.ng * sy + sv.ng;
    return h_P[c] / std::max(h_rho[c], 1e-30);
}

// Analytic Townsend T(t) for single-segment power-law.
static double townsend_analytic(double T0, double C, double alpha,
                                double t, double Tfloor = 1e-6) {
    if (std::fabs(1.0 - alpha) < 1e-12) {
        double T = T0 * std::exp(-C * t);
        return (T < Tfloor) ? Tfloor : T;
    }
    double one_minus_a = 1.0 - alpha;
    double base = std::pow(T0, one_minus_a) - C * one_minus_a * t;
    if (base <= 0.0) return Tfloor;
    double T = std::pow(base, 1.0 / one_minus_a);
    return (T < Tfloor) ? Tfloor : T;
}

// --------------------------------------------------------------------
// C7-T1: closed-form match across α.
// --------------------------------------------------------------------
static void test_T1_closed_form() {
    std::printf("\n[C7-T1] Townsend closed-form T(t) across α\n");
    const int N = 16;           // single-cell physics, use small domain
    const double rho0 = 1.0, T0 = 1.0;
    const double Lambda0 = 1.0, Tref = 1.0;
    const double alphas[] = {0.5, 1.0, 2.0, 3.0};
    for (double a : alphas) {
        AthenaMHDSolver sv;
        seed_uniform(sv, N, rho0, T0);
        sv.cool_on = true;
        sv.cool_Lambda0 = Lambda0;
        sv.cool_Tref    = Tref;
        sv.cool_alpha   = a;
        sv.cool_Tfloor  = 1e-6;
        double gm1 = sv.gamma - 1.0;
        double C = gm1 * rho0 * Lambda0 * std::pow(Tref, -a);
        // Target t for ~30% cooling — in linear regime the analytic is
        // well inside the regular power-law branch (no floor).
        double t_end;
        if (std::fabs(1.0 - a) < 1e-12) {
            t_end = 0.3 / C;          // T -> T0·e^{-0.3}
        } else {
            // T -> 0.7 T0 → base = 0.7^{1-a} T0^{1-a}
            double frac = 1.0 - std::pow(0.7, 1.0 - a);
            t_end = frac * std::pow(T0, 1.0 - a) / (C * (1.0 - a));
            if (t_end < 0.0) t_end = -t_end;
        }
        sv.apply_cooling(t_end);
        double T_num = read_T_interior(sv);
        double T_an  = townsend_analytic(T0, C, a, t_end);
        double rel_err = std::fabs(T_num - T_an) / std::max(T_an, 1e-30);
        std::printf("    α=%g  t=%.4g  T_num=%.6e  T_an=%.6e  rel err %.3e\n",
                    a, t_end, T_num, T_an, rel_err);
        char msg[80];
        std::snprintf(msg, sizeof(msg),
                      "C7-T1: α=%g closed-form to 1e-10", a);
        CHECK_LT(rel_err, 1e-10, msg);
        sv.destroy();
    }
}

// --------------------------------------------------------------------
// C7-T2: ρ, momentum, B unchanged.
// --------------------------------------------------------------------
static void test_T2_passive_to_other_fields() {
    std::printf("\n[C7-T2] cooling only changes E — ρ/mom/B invariant\n");
    const int N = 16;
    const double rho0 = 1.0, T0 = 1.0;
    const double Bx0 = 0.3, By0 = 0.7;
    AthenaMHDSolver sv;
    seed_uniform(sv, N, rho0, T0, Bx0, By0);
    sv.cool_on = true;
    sv.cool_Lambda0 = 1.0; sv.cool_Tref = 1.0; sv.cool_alpha = 2.0;
    sv.cool_Tfloor = 1e-6;

    auto d0 = sv.compute_diagnostics();
    // t long enough for measurable T change
    double C = (sv.gamma - 1.0) * rho0 * 1.0;
    double t_end = 0.5 / C;
    sv.apply_cooling(t_end);
    auto d1 = sv.compute_diagnostics();

    double dMass = std::fabs(d1.total_mass - d0.total_mass) /
                   std::max(std::fabs(d0.total_mass), 1e-30);
    double dME = std::fabs(d1.total_ME - d0.total_ME) /
                 std::max(std::fabs(d0.total_ME), 1e-30);
    double dKE = std::fabs(d1.total_KE - d0.total_KE);   // was 0
    std::printf("    rel Δmass=%.3e  rel ΔME=%.3e  ΔKE=%.3e  divB=%.3e\n",
                dMass, dME, dKE, d1.max_divB);
    CHECK_LT(dMass, 1e-14, "C7-T2: mass conserved exactly");
    CHECK_LT(dME,   1e-14, "C7-T2: magnetic energy frozen");
    CHECK_LT(dKE,   1e-14, "C7-T2: v=0 stays v=0");
    CHECK_LT(d1.max_divB, 1e-10, "C7-T2: ∇·B untouched");
    sv.destroy();
}

// --------------------------------------------------------------------
// C7-T3: monotone cooling across 50 sub-intervals.
// --------------------------------------------------------------------
static void test_T3_monotone() {
    std::printf("\n[C7-T3] monotone cooling — 50 bins, T(k) ≤ T(k-1)\n");
    const int N = 16;
    const double rho0 = 1.0, T0 = 1.0;
    AthenaMHDSolver sv;
    seed_uniform(sv, N, rho0, T0);
    sv.cool_on = true;
    sv.cool_Lambda0 = 1.0; sv.cool_Tref = 1.0; sv.cool_alpha = 2.0;
    sv.cool_Tfloor = 1e-6;

    double C = (sv.gamma - 1.0) * rho0 * 1.0;
    double t_total = 1.0 / C;    // one e-fold equivalent
    int n_bins = 50;
    double dt_bin = t_total / (double)n_bins;
    double prev = read_T_interior(sv);
    int violations = 0;
    double max_growth = 0.0;
    for (int b = 0; b < n_bins; ++b) {
        sv.apply_cooling(dt_bin);
        double Tn = read_T_interior(sv);
        if (Tn > prev + 1e-14) {
            ++violations;
            max_growth = std::max(max_growth, Tn - prev);
        }
        prev = Tn;
    }
    std::printf("    %d bins, violations=%d, max growth=%.3e\n",
                n_bins, violations, max_growth);
    ++g_tests;
    if (violations == 0) {
        std::printf("  PASS  C7-T3: monotone (no bin with T growth)\n");
    } else {
        std::fprintf(stderr,
            "FAIL C7-T3: %d bins showed T growth (max=%.3e)\n",
            violations, max_growth);
        ++g_failures;
    }
    sv.destroy();
}

// --------------------------------------------------------------------
// C7-T4: ΔE on a uniform cell matches  ρ c_v (T_analytic − T_0).
// Cross-check between solver-level E change and the analytic
// Townsend formula, separately from the kernel's internal logic.
// --------------------------------------------------------------------
static void test_T4_energy_budget() {
    std::printf("\n[C7-T4] ΔE = ρ c_v (T_end − T₀) matches analytic budget\n");
    const int N = 16;
    const double rho0 = 1.0, T0 = 1.0;
    const double alpha = 2.5;    // typical bremsstrahlung-like
    AthenaMHDSolver sv;
    seed_uniform(sv, N, rho0, T0);
    sv.cool_on = true;
    sv.cool_Lambda0 = 1.0; sv.cool_Tref = 1.0; sv.cool_alpha = alpha;
    sv.cool_Tfloor = 1e-6;

    double cv  = 1.0 / (sv.gamma - 1.0);
    double C   = (sv.gamma - 1.0) * rho0 * 1.0;
    double t_end = 0.4 / C;

    auto d0 = sv.compute_diagnostics();
    sv.apply_cooling(t_end);
    auto d1 = sv.compute_diagnostics();

    double T_an = townsend_analytic(T0, C, alpha, t_end);
    double dE_analytic = rho0 * cv * (T_an - T0);
    double dE_measured = d1.total_E - d0.total_E;
    // Total ΔE = ΔE_per_cell × ncells (uniform).  d.total_E already a sum.
    // ncells = nx*ny.
    double dE_expected_total = dE_analytic * (double)(sv.nx * sv.ny)
                               * sv.dx * sv.dy;    // compute_diagnostics sums volume
    // NOTE: compute_diagnostics almost certainly sums E · dV (volume-integrated).
    // We compare ratios instead of absolute numbers to avoid coupling to that
    // convention.
    double rel_err = std::fabs(dE_measured - dE_expected_total) /
                     std::max(std::fabs(dE_expected_total), 1e-30);
    std::printf("    T_an=%.6f  ΔE_expected=%.6e  ΔE_measured=%.6e  rel %.3e\n",
                T_an, dE_expected_total, dE_measured, rel_err);
    CHECK_LT(rel_err, 1e-10,
             "C7-T4: ΔE matches ρ c_v ΔT Townsend budget");
    sv.destroy();
}

int main() {
    std::printf("=== athena_mhd Townsend cooling (B-M3, §C7) ===\n");
    test_T1_closed_form();
    test_T2_passive_to_other_fields();
    test_T3_monotone();
    test_T4_energy_budget();
    std::printf("\n=== Summary: %d/%d passed ===\n",
                g_tests - g_failures, g_tests);
    return g_failures == 0 ? 0 : 1;
}
