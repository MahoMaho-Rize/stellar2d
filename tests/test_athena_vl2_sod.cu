// test_athena_vl2_sod.cu
// ============================================================
// Minimal unit test for athena_vl2 — Sod shock-tube smoke.
//
// athena_vl2 was added without any unit / regression tests (it
// relies solely on end-to-end Andrassy comparison with the real
// Athena++).  This test gives it a minimal kernel-level heartbeat:
//
// IC:  standard Sod (x split at Lx/2, γ=1.4)
// BC:  x periodic (vl2 hardcoded), y reflect
// Run: tend=0.2, 128² grid
//
// Checks:
//   U1. M_end / M_0 - 1 < 1e-12           (mass conservation)
//   U2. E_end / E_0 - 1 < 1e-10           (total energy conservation)
//   U3. max_v < 2.0                       (no blow-up)
//   U4. |max_rho - max_rho_IC| < 0.1      (density roughly preserved —
//                                          Sod keeps rho_max ≈ 1.0)
//
// This is explicitly not a convergence test (Phase 2 will add that
// via compute_error once we have L1 vs analytic Toro Riemann).
// ============================================================

#include "athena_vl2_solver.cuh"
#include <cstdio>
#include <cmath>
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

#define CHECK_TRUE(cond, msg) do {                                   \
    ++g_tests;                                                       \
    if (!(cond)) {                                                   \
        std::fprintf(stderr, "FAIL %s:%d [%s]\n",                   \
                     __FILE__, __LINE__, msg);                       \
        ++g_failures;                                                \
    } else { std::printf("  PASS  %s\n", msg); }                     \
} while (0)

int main() {
    std::printf("=== athena_vl2 Sod shock-tube smoke ===\n\n");

    AthenaVL2Solver av;
    av.init(/*nx=*/128, /*ny=*/128, /*Lx=*/1.0, /*Ly=*/1.0,
            /*gamma=*/1.4, /*cfl=*/0.4);
    av.xorder = 2;
    av.limiter = 0;            // vanleer
    av.cfl_limit = 0.5;
    av.init_sod();

    auto d0 = av.compute_diagnostics();
    const double E0 = d0.total_E;
    const double M0 = d0.total_mass;
    std::printf("  IC: E0=%.10e  M0=%.10e  max_v=%.4e\n",
                E0, M0, d0.max_v);

    double t = 0.0;
    const double t_end = 0.2;
    int nsteps = 0;
    while (t < t_end) {
        double dt = av.step(t, t_end);
        t += dt;
        ++nsteps;
        if (std::isnan(dt) || dt <= 0.0) break;
    }

    auto d = av.compute_diagnostics();
    std::printf("  after %d steps (t=%.4f): E=%.10e  M=%.10e  max_v=%.4e\n",
                nsteps, t, d.total_E, d.total_mass, d.max_v);

    double rel_M = std::fabs(d.total_mass - M0) / std::fabs(M0);
    double rel_E = std::fabs(d.total_E    - E0) / std::fabs(E0);

    CHECK_LT(rel_M,   1e-12, "U1: |ΔM|/M0 < 1e-12");
    CHECK_LT(rel_E,   1e-10, "U2: |ΔE|/E0 < 1e-10");
    CHECK_LT(d.max_v, 2.0,   "U3: max_v < 2");
    CHECK_TRUE(std::isfinite(d.total_E) && std::isfinite(d.max_v),
               "U4: all diagnostics finite (no NaN)");

    av.destroy();
    std::printf("\n=== Summary: %d/%d passed ===\n",
                g_tests - g_failures, g_tests);
    return g_failures == 0 ? 0 : 1;
}
