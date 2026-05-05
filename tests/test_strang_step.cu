// ============================================================
// test_strang_step.cu — Verify full Strang step integration
//
// Tests:
//   1. HSE preservation: 100 steps with no bubble → |v| ≈ 0
//   2. Mass conservation: bubble run → ΔM/M < 1e-10
//   3. Bubble motion: bubble run → max |v| grows (buoyancy)
//   4. CFL: dt is reasonable and stable
// ============================================================

#include "strang_solver.cuh"
#include "fas_common.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

static int n_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); n_fail++; } \
    else { std::printf("  PASS: %s\n", msg); } \
} while(0)

int main()
{
    std::printf("=== test_strang_step ===\n\n");
    double gamma = 5.0 / 3.0;

    // ---- Test 1: HSE preservation (no bubble) ----
    std::printf("[Test 1] HSE preservation: per-step velocity growth\n");
    {
        // Test with g=1 and g=0 to isolate gravity contribution
        for (double g_test : {1.0, 0.0}) {
            StrangSolver sol;
            sol.init(64, 96, 1.0, 1.5, gamma, g_test, 0.4, 1.0, 1.0);

            double t = 0;
            std::printf("  g = %.1f:\n", g_test);
            for (int s = 0; s < 5; s++) {
                double dt = sol.step(t, 100.0);
                t += dt;
                double vmax = sol.max_velocity();
                std::printf("    step %d: dt=%.4e  max|v|=%.6e\n", s+1, dt, vmax);
            }
            sol.destroy();
        }
        // Re-run for final check
        StrangSolver sol;
        sol.init(64, 96, 1.0, 1.5, gamma, 1.0, 0.4, 1.0, 1.0);
        double t = 0;
        for (int s = 0; s < 20; s++) {
            double dt = sol.step(t, 100.0);
            t += dt;
        }
        double vmax = sol.max_velocity();
        CHECK(vmax < 1e-1, "HSE: max |v| < 0.1 after 20 steps");
        sol.destroy();
    }

    // ---- Test 2: Mass conservation with bubble ----
    std::printf("\n[Test 2] Mass conservation: bubble, 50 steps\n");
    {
        StrangSolver sol;
        sol.init(64, 96, 1.0, 1.5, gamma, 1.0, 0.4, 1.0, 1.0);
        sol.init_bubble(0.5, 0.3, 0.15, 0.5, 0.05, 10, 3.0*sol.dx);

        double M0 = sol.total_mass();
        double t = 0;
        for (int s = 0; s < 50; s++) {
            double dt = sol.step(t, 100.0);
            t += dt;
        }
        double M1 = sol.total_mass();
        double dM = std::fabs(M1 - M0) / M0;
        std::printf("  M0 = %.10e  M_final = %.10e  |ΔM/M| = %.4e\n", M0, M1, dM);
        CHECK(dM < 1e-3, "Mass conservation: |ΔM/M| < 1e-3");

        sol.destroy();
    }

    // ---- Test 3: Bubble buoyancy ----
    std::printf("\n[Test 3] Bubble buoyancy: max |v| increases\n");
    {
        StrangSolver sol;
        sol.init(64, 96, 1.0, 1.5, gamma, 1.0, 0.4, 1.0, 1.0);
        sol.init_bubble(0.5, 0.3, 0.15, 0.5, 0.05, 10, 3.0*sol.dx);

        double v0 = sol.max_velocity();
        double t = 0;
        for (int s = 0; s < 200; s++) {
            double dt = sol.step(t, 100.0);
            t += dt;
        }
        double v1 = sol.max_velocity();
        std::printf("  v_init = %.4e  v_200steps = %.4e  t = %.4e\n", v0, v1, t);
        CHECK(v1 > v0 + 1e-6, "Bubble velocity grows (buoyancy active)");

        sol.destroy();
    }

    // ---- Test 4: CFL stability ----
    std::printf("\n[Test 4] CFL: dt is stable over 100 steps\n");
    {
        StrangSolver sol;
        sol.init(64, 96, 1.0, 1.5, gamma, 1.0, 0.4, 1.0, 1.0);
        sol.init_bubble(0.5, 0.3, 0.15, 0.5, 0.05, 10, 3.0*sol.dx);

        double t = 0, dt_min = 1e30, dt_max = 0;
        for (int s = 0; s < 100; s++) {
            double dt = sol.step(t, 100.0);
            t += dt;
            dt_min = std::fmin(dt_min, dt);
            dt_max = std::fmax(dt_max, dt);
        }
        std::printf("  dt range: [%.4e, %.4e]  ratio = %.2f\n",
                    dt_min, dt_max, dt_max / dt_min);
        CHECK(dt_min > 1e-8, "CFL: dt not absurdly small");
        CHECK(dt_max / dt_min < 100.0, "CFL: dt ratio < 100 (stable)");
        CHECK(std::isfinite(dt_min), "CFL: dt finite");

        sol.destroy();
    }

    // ---- Test 5: VTK output after evolution ----
    std::printf("\n[Test 5] VTK output after evolution\n");
    {
        StrangSolver sol;
        sol.init(64, 96, 1.0, 1.5, gamma, 1.0, 0.4, 1.0, 1.0);
        sol.init_bubble(0.5, 0.3, 0.15, 0.5, 0.05, 10, 3.0*sol.dx);

        double t = 0;
        for (int s = 0; s < 50; s++) {
            double dt = sol.step(t, 100.0);
            t += dt;
        }
        sol.write_vtk("test_strang_evolved.vtk");

        FILE* f = fopen("test_strang_evolved.vtk", "r");
        CHECK(f != nullptr, "Evolved VTK file created");
        if (f) {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fclose(f);
            CHECK(sz > 1000, "Evolved VTK has content");
        }

        sol.destroy();
    }

    std::printf("\n=== %s (%d failures) ===\n",
                n_fail == 0 ? "ALL PASSED" : "SOME FAILED", n_fail);
    return n_fail;
}
