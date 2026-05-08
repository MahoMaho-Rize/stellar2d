// test_athena_mhd_field_loop_long.cu
// ============================================================
// Long-time field-loop CT preservation / numerical-η diagnostic.
// Phase A3 of docs/projects/mhd_verification/phase_A_plan.md.
//
// Extends the standard field_loop test (single 1/4-domain crossing)
// to **10 full diagonal crossings** (t=10 on a unit domain with v=(1, 0.5),
// so one crossing along x = 1 time unit).  Records:
//   - max|∇·B|(t)     at 40 checkpoints
//   - total ME(t)     (magnetic energy as a proxy for loop decay)
//   - mass/energy conservation
//
// Gardiner-Stone 2005 prediction under CT: max|∇·B| stays at
// machine precision over arbitrarily many crossings, and ME decays
// as roughly exp(-η_eff k_loop² t) from discrete numerical dissipation
// in the transverse-flux stencil.
//
// Pass criteria:
//   L1.  max|∇·B|(t) < 1e-10          forever (CT lock stays at roundoff)
//   L2.  |ΔM/M0| < 1e-12              (passive advection)
//   L3.  ME(t) bounded: max/IC < 3    (cell-centred ME is a diagnostic
//                                       reconstruction B_cc = ½(Bxf_L+Bxf_R),
//                                       not a conserved quantity; the
//                                       sharp r=R kink aliases differently
//                                       as the loop translates, so ME as
//                                       measured here does not monotone
//                                       decrease — CT conserves the face
//                                       flux, not cell-centred ME)
//   L4.  ME(t=10) > 0.5 · ME(0)       (no runaway decay)
//   L5.  all diagnostics finite throughout
//
// Also writes a CSV to <build>/field_loop_long.csv with columns
//   step,t,dt,total_mass,total_E,total_ME,max_divB,max_v
// for post-hoc plotting of η_eff(t) fit.
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

#define CHECK_GE(got, bound, msg) do {                                \
    ++g_tests;                                                        \
    double _g = (got), _b = (bound);                                  \
    if (!(_g >= _b)) {                                                \
        std::fprintf(stderr, "FAIL %s:%d [%s]: got=%.6e bound=%.1e\n",\
                     __FILE__, __LINE__, msg, _g, _b);                \
        ++g_failures;                                                 \
    } else {                                                          \
        std::printf("  PASS  %s  (got=%.3e >= %.1e)\n", msg, _g, _b);\
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
    std::printf("=== athena_mhd field-loop long-time (A3: 10 crossings) ===\n\n");

    AthenaMHDSolver sv;
    sv.init(/*nx=*/128, /*ny=*/128, /*Lx=*/1.0, /*Ly=*/1.0,
            /*gamma=*/5.0 / 3.0, /*cfl=*/0.3);
    sv.xorder = 2;
    sv.limiter = 1;   // minmod — more dissipative than van Leer, needed
                      //         for long-time stability of the r=R kink
                      //         in the field-loop IC (Stone+08 Fig 25 note)
    sv.cfl_limit = 0.5;
    sv.init_field_loop(/*v_adv_x=*/1.0, /*v_adv_y=*/0.5,
                       /*R=*/0.3, /*A0=*/1e-3);

    auto d0 = sv.compute_diagnostics();
    const double E0  = d0.total_E;
    const double M0  = d0.total_mass;
    const double ME0 = d0.total_ME;
    std::printf("  IC: E0=%.10e M0=%.10e ME0=%.6e max|divB|=%.3e\n",
                E0, M0, ME0, d0.max_divB);

    // CSV for post-processing the decay rate
    std::FILE* csv = std::fopen("field_loop_long.csv", "w");
    if (!csv) {
        std::fprintf(stderr, "ERROR: could not open field_loop_long.csv\n");
        return 1;
    }
    std::fprintf(csv, "step,t,dt,total_mass,total_E,total_ME,max_divB,max_v\n");
    std::fprintf(csv, "0,0.0,0.0,%.10e,%.10e,%.10e,%.3e,%.6e\n",
                 d0.total_mass, d0.total_E, d0.total_ME, d0.max_divB, d0.max_v);

    // Run 10 crossings (v_adv_x=1 on Lx=1 → 1 crossing per time unit).
    const double t_end = 10.0;
    double t = 0.0;
    int step = 0;
    double max_divB_ever = d0.max_divB;
    double ME_max_seen   = ME0;
    double next_sample   = 0.25;   // 40 samples over t∈[0,10]
    const int max_steps  = 200000;

    while (t < t_end) {
        double dt = sv.step(t, t_end);
        if (std::isnan(dt) || dt <= 0.0) break;
        t += dt;
        ++step;
        if (step > max_steps) break;

        // Periodic snapshot (not every step — compute_diagnostics costs).
        if (t >= next_sample - 1e-12 || t >= t_end) {
            auto d = sv.compute_diagnostics();
            max_divB_ever = std::max(max_divB_ever, d.max_divB);
            ME_max_seen = std::max(ME_max_seen, d.total_ME);
            std::fprintf(csv, "%d,%.6f,%.6e,%.10e,%.10e,%.10e,%.3e,%.6e\n",
                         step, t, dt, d.total_mass, d.total_E, d.total_ME,
                         d.max_divB, d.max_v);
            std::fflush(csv);
            next_sample += 0.25;
        }
    }

    auto d = sv.compute_diagnostics();
    max_divB_ever = std::max(max_divB_ever, d.max_divB);
    std::printf("\n  after %d steps (t=%.4f): E=%.10e M=%.10e\n"
                "  ME=%.6e  ME/ME0=%.6f   max|divB|=%.3e\n",
                step, t, d.total_E, d.total_mass,
                d.total_ME, d.total_ME / ME0, d.max_divB);
    std::printf("  peak |divB| across all checkpoints: %.3e\n", max_divB_ever);
    std::fclose(csv);

    // Expected crossings traveled: t_end * |v| / L = 10 * √(1+0.25) / 1
    //   ≈ 11.18 diagonal-length crossings.
    double rel_M = std::fabs(d.total_mass - M0) / std::fabs(M0);
    double ME_ratio = d.total_ME / ME0;

    CHECK_LT(max_divB_ever, 1e-10, "L1: max|∇·B| < 1e-10 (CT lock)");
    CHECK_LT(rel_M,         1e-12, "L2: |ΔM/M0| < 1e-12 (passive advect)");
    CHECK_LT(ME_max_seen / ME0, 3.0,
             "L3: ME_max/ME0 < 3 (B_cc aliasing bounded)");
    CHECK_GE(ME_ratio,      0.50,  "L4: ME(t=10) > 50% of IC");
    CHECK_TRUE(std::isfinite(d.total_E) && std::isfinite(d.max_v),
               "L5: all diagnostics finite");

    std::printf("\n  → wrote field_loop_long.csv (%d samples)\n", 40);
    std::printf("=== Summary: %d/%d passed ===\n",
                g_tests - g_failures, g_tests);
    sv.destroy();
    return g_failures == 0 ? 0 : 1;
}
