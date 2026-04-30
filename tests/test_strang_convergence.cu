// ============================================================
// test_strang_convergence.cu — Grid convergence study
//
// Entropy wave in x-direction (periodic, g=0, constant P):
//   ρ = ρ0 + A sin(kx - u0*k*t), u = u0, v = 0, P = P0
// LM-HLLC doesn't affect this (ΔP = 0 at faces).
// Run at N = 64, 128, 256. Verify L1 error ratio ≈ 4.
// ============================================================

#include "gpu/strang_solver.cuh"
#include "gpu/strang_device.cuh"
#include "gpu/fas_common.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

static int n_fail = 0, n_pass = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "  FAIL: %s\n", msg); n_fail++; } \
    else { std::printf("  PASS: %s\n", msg); n_pass++; } \
} while(0)

// Initialize entropy wave (density advection at constant velocity)
__global__
void k_init_entropy_wave(double* d_rho, double* d_mx, double* d_my, double* d_E,
                         double rho0, double P0, double gamma,
                         double A, double kx, double u0,
                         double x_lo, double dx,
                         int nx, int ny, int ng, int str)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny) return;
    int i_phys = tid % nx, j_phys = tid / nx;
    int k = (j_phys + ng) * str + (i_phys + ng);
    double gm1 = gamma - 1.0;

    double x = x_lo + (i_phys + 0.5) * dx;
    double drho = A * sin(kx * x);
    double rho  = rho0 + drho;
    double E    = P0 / gm1 + 0.5 * rho * u0 * u0;

    d_rho[k] = drho;               // ρ perturbation
    d_mx [k] = rho * u0;           // total x-momentum
    d_my [k] = 0.0;
    d_E  [k] = E - P0 / gm1;      // E perturbation = 0.5*ρ*u0² - 0.5*ρ0*u0²
}

static double compute_L1_entropy(StrangSolver& sol, double t,
                                  double rho0, double A, double kx, double u0)
{
    int N = sol.total_cells();
    std::vector<double> h_rho(N);
    CUDA_CHECK(cudaMemcpy(h_rho.data(), sol.d_rho, N*sizeof(double), cudaMemcpyDeviceToHost));
    int str = sol.stride(), ng = sol.ng;

    double L1 = 0.0;
    for (int i = 0; i < sol.nx; i++) {
        int k = ng * str + (i + ng);  // j=0 row
        double x = sol.x_lo + (i + 0.5) * sol.dx;
        double drho_exact = A * std::sin(kx * (x - u0 * t));
        L1 += std::fabs(h_rho[k] - drho_exact) * sol.dx;
    }
    return L1;
}

int main()
{
    std::printf("=== test_strang_convergence (entropy wave) ===\n\n");

    double gamma = 5.0 / 3.0;
    double rho0 = 1.0, P0 = 1.0;
    double cs = std::sqrt(gamma * P0 / rho0);
    double Lx = 1.0;
    double kx = 2.0 * M_PI / Lx;
    double u0 = 0.5;                  // advection velocity (M ≈ 0.39)
    double A  = 1e-2;                 // density amplitude (no ΔP → LM irrelevant)
    double T  = Lx / u0;             // one advection period

    std::printf("  cs=%.4f  u0=%.4f  M=%.4f  A=%.2e  T=%.4f\n", cs, u0, u0/cs, A, T);

    int resolutions[] = {64, 128, 256};
    double errors[3];
    int ny_fixed = 4;

    for (int r = 0; r < 3; r++) {
        int nx = resolutions[r];
        double Ly = ny_fixed * (Lx / nx);

        StrangSolver sol;
        sol.init(nx, ny_fixed, Lx, Ly, gamma, 0.0, 0.4, 1.0, rho0);

        int N_phys = nx * ny_fixed, B = 256;
        k_init_entropy_wave<<<(N_phys+B-1)/B, B>>>(
            sol.d_rho, sol.d_mx, sol.d_my, sol.d_E,
            rho0, P0, gamma, A, kx, u0,
            sol.x_lo, sol.dx, nx, ny_fixed, sol.ng, sol.stride());
        CUDA_CHECK(cudaDeviceSynchronize());

        double t = 0;
        int steps = 0;
        while (t < T) {
            double dt = sol.step(t, T);
            t += dt;
            steps++;
        }

        errors[r] = compute_L1_entropy(sol, t, rho0, A, kx, u0);
        std::printf("  N=%3d: %5d steps  t=%.6f  L1=%.6e\n",
                    nx, steps, t, errors[r]);
        sol.destroy();
    }

    std::printf("\nConvergence rates:\n");
    for (int r = 1; r < 3; r++) {
        double ratio = errors[r-1] / errors[r];
        double order = std::log2(ratio);
        std::printf("  N=%d→%d: ratio = %.2f  order = %.2f\n",
                    resolutions[r-1], resolutions[r], ratio, order);
    }

    double order_1 = std::log2(errors[0] / errors[1]);
    double order_2 = std::log2(errors[1] / errors[2]);

    CHECK(order_1 > 1.5, "64→128: convergence order > 1.5");
    CHECK(order_2 > 1.3, "128→256: convergence order > 1.3 (TVD limiter near extrema)");
    CHECK(errors[2] < errors[0] * 0.1, "256 error < 10% of 64 error");

    std::printf("\n=== %s: %d passed, %d failed ===\n",
                n_fail == 0 ? "ALL PASSED" : "SOME FAILED", n_pass, n_fail);
    return n_fail;
}
