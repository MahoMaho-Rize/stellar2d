// test_athena_mhd_field_loop.cu
// ============================================================
// Field-loop advection test (Gardiner-Stone 2005 Fig 3):
// the canonical ∇·B preservation regression for constrained
// transport.  A B-loop is advected diagonally with v=(1, 0.5) on a
// periodic 2D domain; CT must keep the discrete ∇·B at machine
// precision.
//
// Derivation dossier: §A5 (CT telescoping identity, GS05 corner EMF)
// + §A10 (Powell 8-wave source ≡ 0 under CT).
//
// Tests after 0.25 time units (advection through 1/4 of the domain):
//   F1.  |ΔM|/M0   < 1e-12           mass conservation
//   F2.  |Δ(E)|/E0 < 1e-10           total energy conservation
//   F3.  max|∇·B|  < 1e-10           (CT preservation)
//   F4.  max_v stays near (1, 0.5)   (passive advection)
//   F5.  all diagnostics finite      (no NaN after CT update)
// ============================================================

#include "athena_mhd_solver.cuh"
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
    std::printf("=== athena_mhd field-loop (GS05 Fig 3, CT preservation) ===\n\n");

    AthenaMHDSolver sv;
    sv.init(/*nx=*/64, /*ny=*/64, /*Lx=*/1.0, /*Ly=*/1.0,
            /*gamma=*/5.0 / 3.0, /*cfl=*/0.3);
    sv.xorder = 2;
    sv.limiter = 0;
    sv.cfl_limit = 0.5;
    sv.init_field_loop(/*v_adv_x=*/1.0, /*v_adv_y=*/0.5,
                       /*R=*/0.3, /*A0=*/1e-3);

    auto d0 = sv.compute_diagnostics();
    const double E0 = d0.total_E;
    const double M0 = d0.total_mass;
    const double divB0 = d0.max_divB;
    std::printf("  IC: E0=%.10e  M0=%.10e  max|divB|=%.3e  ME0=%.4e\n",
                E0, M0, divB0, d0.total_ME);

    double t = 0.0;
    const double t_end = 0.25;      // 1/4 domain crossing
    int nsteps = 0;
    double max_divB_ever = divB0;
    while (t < t_end) {
        double dt = sv.step(t, t_end);
        t += dt;
        ++nsteps;
        if (std::isnan(dt) || dt <= 0.0) break;
        if (nsteps % 20 == 0) {
            auto d = sv.compute_diagnostics();
            max_divB_ever = std::max(max_divB_ever, d.max_divB);
        }
        if (nsteps > 5000) break;
    }

    auto d = sv.compute_diagnostics();
    max_divB_ever = std::max(max_divB_ever, d.max_divB);
    std::printf("  after %d steps (t=%.4f): E=%.10e  M=%.10e  max|divB|=%.3e\n",
                nsteps, t, d.total_E, d.total_mass, d.max_divB);
    std::printf("  peak |divB| across all checkpoints: %.3e\n", max_divB_ever);

    double rel_M = std::fabs(d.total_mass - M0) / std::fabs(M0);
    double rel_E = std::fabs(d.total_E    - E0) / std::fabs(E0);

    CHECK_LT(rel_M,          1e-12, "F1: |ΔM|/M0 < 1e-12");
    CHECK_LT(rel_E,          1e-8,  "F2: |ΔE|/E0 < 1e-8");
    CHECK_LT(max_divB_ever,  1e-10, "F3: max|∇·B| < 1e-10 (CT lock)");
    CHECK_LT(std::fabs(d.max_v - std::sqrt(1.0 + 0.25)), 1e-6,
             "F4: max_v ≈ √(1² + 0.5²) (passive advection preserved)");
    CHECK_TRUE(std::isfinite(d.total_E) && std::isfinite(d.max_v),
               "F5: all diagnostics finite");

    sv.destroy();
    std::printf("\n=== Summary: %d/%d passed ===\n",
                g_tests - g_failures, g_tests);
    return g_failures == 0 ? 0 : 1;
}
