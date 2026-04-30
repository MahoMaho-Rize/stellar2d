// ============================================================
// test_strang_init.cu — Verify Strang solver initialization
//
// Tests:
//   1. HSE background: dp̄/dy ≈ −ρ̄ g (discrete check)
//   2. Bubble init: ρ' < 0 at center, ≈ 0 far away
//   3. Total mass > 0 and finite
//   4. Pressure equilibrium: E' ≈ 0 everywhere
//   5. VTK file written successfully
// ============================================================

#include "gpu/strang_solver.cuh"
#include <cstdio>
#include <cmath>
#include <cstdlib>

static int n_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL: %s\n", msg); \
        n_fail++; \
    } else { \
        std::printf("  PASS: %s\n", msg); \
    } \
} while(0)

int main()
{
    std::printf("=== test_strang_init ===\n\n");

    // Parameters
    int nx = 64, ny = 96;
    double Lx = 1.0, Ly = 1.5;
    double gamma = 5.0 / 3.0;
    double g = 1.0, cfl = 0.4;
    double K = 1.0, rho0 = 1.0;

    StrangSolver solver;
    solver.init(nx, ny, Lx, Ly, gamma, g, cfl, K, rho0);

    // ---- Test 1: HSE discrete balance ----
    std::printf("\n[Test 1] HSE balance: dp̄/dy + ρ̄ g ≈ 0\n");
    {
        double max_err = 0.0;
        double dy = solver.dy;
        for (int j = 1; j < ny; ++j) {
            double dp_dy = (solver.h_p_bar[j] - solver.h_p_bar[j-1]) / dy;
            double rho_avg = 0.5 * (solver.h_rho_bar[j] + solver.h_rho_bar[j-1]);
            double err = std::fabs(dp_dy + rho_avg * g);
            max_err = std::fmax(max_err, err);
        }
        std::printf("  max |dp/dy + ρg| = %.4e\n", max_err);
        CHECK(max_err < 1e-2, "HSE balance error < 1e-2");
    }

    // ---- Test 2: Bubble initialization ----
    std::printf("\n[Test 2] Bubble initialization\n");
    {
        double x0 = 0.5, y0 = 0.3, R0 = 0.15;
        double dS = 0.5, eps = 0.05;
        int k_mode = 10;
        double dr = 3.0 * solver.dx;

        solver.init_bubble(x0, y0, R0, dS, eps, k_mode, dr);

        // Download ρ' and check
        int N = solver.total_cells();
        std::vector<double> h_rho(N);
        cudaMemcpy(h_rho.data(), solver.d_rho, N * sizeof(double), cudaMemcpyDeviceToHost);

        int str = solver.stride();

        // Check center of bubble: ρ' should be negative (lower density)
        int ic = (int)(x0 / solver.dx);
        int jc = (int)(y0 / solver.dy);
        int kc = (jc + solver.ng) * str + (ic + solver.ng);
        double rho_center = h_rho[kc];
        std::printf("  ρ'(center) = %.6e\n", rho_center);
        CHECK(rho_center < -0.01, "ρ' < 0 at bubble center (buoyant)");

        // Check far from bubble: ρ' ≈ 0
        int i_far = nx - 1;
        int j_far = ny - 1;
        int k_far = (j_far + solver.ng) * str + (i_far + solver.ng);
        double rho_far = h_rho[k_far];
        std::printf("  ρ'(corner) = %.6e\n", rho_far);
        CHECK(std::fabs(rho_far) < 1e-10, "ρ' ≈ 0 far from bubble");
    }

    // ---- Test 3: Total mass ----
    std::printf("\n[Test 3] Total mass\n");
    {
        double M = solver.total_mass();
        std::printf("  Total mass = %.10e\n", M);
        CHECK(M > 0.0 && std::isfinite(M), "Mass > 0 and finite");
    }

    // ---- Test 4: Pressure equilibrium ----
    std::printf("\n[Test 4] Energy perturbation (pressure equilibrium)\n");
    {
        int N = solver.total_cells();
        std::vector<double> h_E(N);
        cudaMemcpy(h_E.data(), solver.d_E, N * sizeof(double), cudaMemcpyDeviceToHost);

        int str = solver.stride();
        double max_Ep = 0.0;
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                double Ep = std::fabs(h_E[(j + solver.ng) * str + (i + solver.ng)]);
                max_Ep = std::fmax(max_Ep, Ep);
            }
        std::printf("  max |E'| = %.4e\n", max_Ep);
        CHECK(max_Ep < 1e-10, "E' ≈ 0 (pressure equilibrium at init)");
    }

    // ---- Test 5: VTK output ----
    std::printf("\n[Test 5] VTK output\n");
    {
        const char* fname = "test_strang_init.vtk";
        solver.write_vtk(fname);
        FILE* f = fopen(fname, "r");
        CHECK(f != nullptr, "VTK file created");
        if (f) {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fclose(f);
            std::printf("  VTK file size: %ld bytes\n", sz);
            CHECK(sz > 1000, "VTK file has substantial content");
        }
    }

    // ---- Test 6: step() doesn't crash (stub) ----
    std::printf("\n[Test 6] step() smoke test\n");
    {
        double dt = solver.step(0.0, 1.0);
        std::printf("  dt = %.6e\n", dt);
        CHECK(dt > 0 && std::isfinite(dt), "step() returns finite dt > 0");
    }

    solver.destroy();

    std::printf("\n=== %s (%d failures) ===\n",
                n_fail == 0 ? "ALL PASSED" : "SOME FAILED", n_fail);
    return n_fail;
}
