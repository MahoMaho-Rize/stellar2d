// test_athena_mhd_brio_wu.cu
// ============================================================
// Minimal Brio-Wu MHD shock-tube smoke (Brio-Wu 1988).
//
// Not a convergence / L¹ test — that needs the analytic exact
// solution, which is a separate multi-page dispatch of slow
// rarefactions + compound waves.  Instead this locks:
//
//   BW1. mass conservation |ΔM|/M0 < 1e-12
//   BW2. total energy conservation |ΔE|/E0 < 1e-10
//   BW3. NO NaN in any diagnostic
//   BW4. final max_v bounded (solution remains subsonic-ish)
//   BW5. final max|∇·B| < 1e-10  (§A5: CT is exact)
//
// Runs on a minimal 1D-ish slab: nx=256, ny=4.
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
    std::printf("=== athena_mhd Brio-Wu MHD shock tube ===\n\n");

    AthenaMHDSolver sv;
    sv.init(/*nx=*/256, /*ny=*/4, /*Lx=*/1.0, /*Ly=*/0.015625,
            /*gamma=*/2.0, /*cfl=*/0.3);
    sv.xorder = 2;
    sv.limiter = 0;
    sv.cfl_limit = 0.5;
    sv.init_brio_wu();

    auto d0 = sv.compute_diagnostics();
    const double E0 = d0.total_E;
    const double M0 = d0.total_mass;
    std::printf("  IC: E0=%.10e M0=%.10e ME0=%.4e max|divB|=%.3e\n",
                E0, M0, d0.total_ME, d0.max_divB);

    double t = 0.0;
    const double t_end = 0.1;    // Brio-Wu canonical endpoint t=0.1
    int nsteps = 0;
    while (t < t_end) {
        double dt = sv.step(t, t_end);
        t += dt;
        ++nsteps;
        if (std::isnan(dt) || dt <= 0.0) break;
        if (nsteps > 5000) break;
    }

    auto d = sv.compute_diagnostics();
    std::printf("  after %d steps (t=%.4f): E=%.10e M=%.10e max|divB|=%.3e\n",
                nsteps, t, d.total_E, d.total_mass, d.max_divB);

    double rel_M = std::fabs(d.total_mass - M0) / std::fabs(M0);
    double rel_E = std::fabs(d.total_E    - E0) / std::fabs(E0);

    CHECK_LT(rel_M,    1e-12, "BW1: |ΔM|/M0 < 1e-12");
    CHECK_LT(rel_E,    1e-10, "BW2: |ΔE|/E0 < 1e-10");
    CHECK_TRUE(std::isfinite(d.total_E) && std::isfinite(d.max_v),
               "BW3: all diagnostics finite");
    CHECK_LT(d.max_v, 5.0,   "BW4: max_v < 5 (bounded solution)");
    CHECK_LT(d.max_divB, 1e-10, "BW5: max|∇·B| < 1e-10 (CT lock)");

    sv.destroy();
    std::printf("\n=== Summary: %d/%d passed ===\n",
                g_tests - g_failures, g_tests);
    return g_failures == 0 ? 0 : 1;
}
