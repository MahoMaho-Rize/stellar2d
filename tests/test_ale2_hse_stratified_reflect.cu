// test_ale2_hse_stratified_reflect.cu
// ============================================================
// Regression lock for the 2026-05-07 Andrassy scan failure:
// the `rebuild_order=1` default (2nd-order corner MUSCL) introduced
// at `214a7d9` passed all 5 canonical benchmarks (Sod/Sedov/Noh/
// Gresho/Yee) but immediately went checker-board unstable on the
// Andrassy O-shell convection (reflect wall in y, x-periodic,
// stratified HSE IC, long-time).
//
// This test is the minimal reproducer of that failure mode — a
// stratified HSE column with x-periodic / y-reflect BC, 100 steps
// of zero-velocity evolution.  A stable scheme keeps KE at
// round-off; the bad scheme grows KE by orders of magnitude.
//
// Tested combinations:
//   rebuild_order = 0 (1st-order mass-weighted avg, current default)
//   rebuild_order = 1 (2nd-order corner MUSCL, Barth-Jespersen limited)
//
// Checks (per rebuild_order):
//   U1. max(KE)/E0 over 100 steps < ε_KE
//       ε_KE(0) = 1e-2  (1st-order baseline — cart_ale2 is NOT well-
//                        balanced, so a small residual KE is expected;
//                        the fail mode is O(1) blow-up, not residuals)
//       ε_KE(1) = 1e-2  (2nd-order also allowed up to the same floor)
//   U2. |ΔE|/E0 < 1e-2  (compensation holds to the % level; machine
//                        precision is wrong for a non-well-balanced
//                        stratified column after Lagrangian move)
//   U3. M_end/M_0 - 1 < 1e-12    (mass is conserved to round-off)
//   U4. rebuild=1 KE ≤ 10 · rebuild=0 KE   (the Andrassy regression
//                                           specifically made rebuild=1
//                                           blow 100-1000× vs rebuild=0)
//
// If the 2026-05-07 Andrassy regression returns, U4 will fire (ratio
// was ~10^3 in the failing state) and most likely U1 too (the failing
// rebuild=1 scheme has max_KE ~ 1e-1).
// ============================================================

#include "cart_ale2_solver.cuh"
#include <cstdio>
#include <cmath>
#include <algorithm>

static int g_failures = 0;
static int g_tests = 0;

#define CHECK_TRUE(cond, msg) do {                                   \
    ++g_tests;                                                       \
    if (!(cond)) {                                                   \
        std::fprintf(stderr, "FAIL %s:%d [%s]\n",                   \
                     __FILE__, __LINE__, msg);                       \
        ++g_failures;                                                \
    } else { std::printf("  PASS  %s\n", msg); }                     \
} while (0)

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

struct RunResult {
    double max_KE_over_E0;
    double rel_E_drift;
    double rel_M_drift;
};

static RunResult run_hse(int rebuild_order, int nsteps) {
    const int nx = 64, ny = 64;
    const double Lx = 1.0, Ly = 1.0;
    const double gamma = 5.0 / 3.0;
    const double cfl   = 0.3;
    const double rho_base = 1.0;
    const double g_val    = 1.0;   // pulls -y

    CartAle2Solver sol;
    sol.init(nx, ny, Lx, Ly, gamma, cfl);
    sol.bc_mode = 1;  // bit 0 = x-periodic, bit 1 = y-reflect (0)
    sol.rebuild_order = rebuild_order;
    sol.init_hse_polytrope(rho_base, g_val, 0.0);

    auto d0 = sol.compute_diagnostics();
    const double E0 = d0.total_E;
    const double M0 = d0.total_mass;
    double max_KE = 0.0;

    double t = 0.0;
    const double t_end_cap = 1e9;
    for (int s = 0; s < nsteps; ++s) {
        double dt = sol.step(t, t_end_cap);
        t += dt;
        auto d = sol.compute_diagnostics();
        if (d.total_KE > max_KE) max_KE = d.total_KE;
        if (std::isnan(d.total_KE) || std::isnan(d.total_E)) break;
    }
    auto de = sol.compute_diagnostics();

    RunResult r;
    r.max_KE_over_E0 = max_KE / std::fabs(E0);
    r.rel_E_drift    = std::fabs(de.total_E - E0) / std::fabs(E0);
    r.rel_M_drift    = std::fabs(de.total_mass - M0) / std::fabs(M0);
    sol.destroy();
    return r;
}

int main() {
    std::printf("=== cart_ale2 HSE-stratified reflect (Andrassy regression lock) ===\n\n");

    // ---- rebuild_order = 0 (current stable default) ----
    std::printf(">> rebuild_order = 0 (1st-order, stable)\n");
    RunResult r0 = run_hse(/*rebuild_order=*/0, /*nsteps=*/100);
    std::printf("  max_KE/E0 = %.3e\n", r0.max_KE_over_E0);
    std::printf("  |ΔE|/E0   = %.3e\n", r0.rel_E_drift);
    std::printf("  |ΔM|/M0   = %.3e\n", r0.rel_M_drift);
    CHECK_LT(r0.max_KE_over_E0, 1e-2, "U1a: rebuild=0  max KE / E0 < 1e-2");
    CHECK_LT(r0.rel_E_drift,    1e-2, "U2a: rebuild=0  |ΔE|/E0 < 1e-2");
    CHECK_LT(r0.rel_M_drift,   1e-12, "U3a: rebuild=0  |ΔM|/M0 < 1e-12");

    std::printf("\n");

    // ---- rebuild_order = 1 (2nd-order) ----
    std::printf(">> rebuild_order = 1 (2nd-order)\n");
    RunResult r1 = run_hse(/*rebuild_order=*/1, /*nsteps=*/100);
    std::printf("  max_KE/E0 = %.3e\n", r1.max_KE_over_E0);
    std::printf("  |ΔE|/E0   = %.3e\n", r1.rel_E_drift);
    std::printf("  |ΔM|/M0   = %.3e\n", r1.rel_M_drift);
    CHECK_LT(r1.max_KE_over_E0, 1e-2, "U1b: rebuild=1  max KE / E0 < 1e-2");
    CHECK_LT(r1.rel_E_drift,    1e-2, "U2b: rebuild=1  |ΔE|/E0 < 1e-2");
    CHECK_LT(r1.rel_M_drift,   1e-12, "U3b: rebuild=1  |ΔM|/M0 < 1e-12");

    // U4: 2nd-order rebuild should NOT be massively worse than 1st-order
    //     on stratified HSE.  Historically the regression made rebuild=1
    //     blow 100-1000× vs rebuild=0; we lock this ratio below 10× so
    //     future "clever" rebuild changes can't silently reintroduce the
    //     Andrassy checker-board mode.
    double ratio = r1.max_KE_over_E0 / std::max(r0.max_KE_over_E0, 1e-30);
    std::printf("\n  KE ratio (rebuild=1 / rebuild=0) = %.2f\n", ratio);
    CHECK_LT(ratio, 10.0, "U4: rebuild=1 KE within 10× of rebuild=0");

    std::printf("\n=== Summary: %d/%d passed ===\n",
                g_tests - g_failures, g_tests);
    return g_failures == 0 ? 0 : 1;
}
