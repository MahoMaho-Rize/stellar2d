// test_strang_linwave_convergence.cu
// ============================================================
// Linear-wave (acoustic) grid-convergence study for the Strang-split
// explicit solver — complements the existing entropy-wave convergence
// test (test_strang_convergence.cu) by exercising the **compressible**
// characteristic:  entropy_wave has Δρ but no ΔP, so HLLC degenerates
// to pure upwind advection.  A single right-going acoustic mode on
// the other hand excites all four HLLC waves and the predictor-
// corrector pressure gradient coupling.
//
// Right-going sound-wave IC:
//   δρ  = A · ρ₀ · sin(k x)
//   δvx = A · c₀ · sin(k x)              (from ρ₀·δvx = c₀·δρ)
//   δP  = A · γ · P₀ · sin(k x)          (from δP = c₀²·δρ)
// Evolved over exactly one wave period  T = Lx / c₀, the exact
// solution returns to the IC.  Periodic-x BC (default for strang
// uniform-HSE).
//
// Requires `sol.use_lm_fix = false` — the Rieper LM-HLLC fix that is
// on by default multiplies the S* pressure-jump term by fM ≈ M_local,
// which for an acoustic wave is just the wave amplitude A.  That
// clamps the pressure physics out of the Riemann solver and yields
// non-convergent L1 (measured: err ≈ 40–60 % of IC amplitude at all
// N).  The LM fix is correct for low-Mach convective flow (Andrassy)
// but wrong for waves whose core physics IS the pressure jump.
//
// Setup matches the existing entropy-wave test:
//   γ = 5/3, ρ₀ = 1, P₀ = 1, A = 1e-4, k = 2π/Lx, CFL = 0.4,
//   Lx = 1, ny = 4 (thin strip — same as strang_convergence).
//   Resolutions: N = 64, 128, 256.
//
// Assertions (fast bucket, < 3 s):
//   A1. Observed convergence order on ρ is ≥ 1.5 at 64→128 and ≥ 1.3
//       at 128→256.  Nominal MUSCL+HLLC Strang is p=2; we lock 1.5 /
//       1.3 to ride out TVD limiter activation near sinusoid peaks,
//       exactly like test_strang_convergence does for entropy wave.
//       Measured: 64→128 = 1.94, 128→256 = 1.48.
//
//   A2. L1(ρ) at N=256 must be < 0.1 × L1(ρ) at N=64 (4× improvement
//       minimum over a 4× refinement; formal p≥2 would give 16×).
//
//   A3. No NaN: all three L1 values finite.
// ============================================================

#include "strang_solver.cuh"
#include "gpu_common.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

static int n_fail = 0, n_pass = 0;
#define CHECK(cond, msg) do {                                        \
    if (!(cond)) { std::fprintf(stderr, "  FAIL: %s\n", msg); n_fail++; } \
    else         { std::printf("  PASS: %s\n", msg); n_pass++; }     \
} while (0)

// Initialise the right-going acoustic mode ON TOP OF the HSE snapshot
// that strang::init already wrote.  Strang stores perturbation
// (ρ, mx, my, E-P₀/gm1) in d_rho/d_mx/d_my/d_E — so we write exactly
// the perturbation fields, same convention as test_strang_convergence.
__global__
void k_init_acoustic_wave(double* d_rho, double* d_mx, double* d_my, double* d_E,
                          double rho0, double P0, double gamma,
                          double A, double kx,
                          double x_lo, double dx,
                          int nx, int ny, int ng, int str)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny) return;
    int i_phys = tid % nx, j_phys = tid / nx;
    int k = (j_phys + ng) * str + (i_phys + ng);

    double gm1 = gamma - 1.0;
    double c0  = sqrt(gamma * P0 / rho0);

    double x = x_lo + (i_phys + 0.5) * dx;
    double s = sin(kx * x);
    double drho = A * rho0 * s;
    double rho  = rho0 + drho;
    double vx   = A * c0  * s;             // right-going wave
    double dP   = A * gamma * P0 * s;
    double P    = P0 + dP;
    double E_tot = P / gm1 + 0.5 * rho * vx * vx;

    d_rho[k] = drho;                       // perturbation
    d_mx [k] = rho * vx;                   // total momentum (vx=0 background)
    d_my [k] = 0.0;
    d_E  [k] = E_tot - P0 / gm1;          // perturbation
}

// After one period, analytic solution is identical to IC.  We score
// ρ only (cleanest one-field compare; u/P follow trivially).
static double compute_L1_rho_vs_ic(StrangSolver& sol,
                                   double rho0, double A, double kx)
{
    int N = sol.total_cells();
    std::vector<double> h_rho(N);
    CUDA_CHECK(cudaMemcpy(h_rho.data(), sol.d_rho, N * sizeof(double),
                          cudaMemcpyDeviceToHost));
    int str = sol.stride(), ng = sol.ng;

    double L1 = 0.0;
    for (int i = 0; i < sol.nx; ++i) {
        int k = ng * str + (i + ng);      // j = 0 row
        double x = sol.x_lo + (i + 0.5) * sol.dx;
        double drho_exact = A * rho0 * std::sin(kx * x);
        L1 += std::fabs(h_rho[k] - drho_exact) * sol.dx;
    }
    return L1;
}

int main()
{
    std::printf("=== test_strang_linwave_convergence (acoustic right-going) ===\n\n");

    const double gamma = 5.0 / 3.0;
    const double rho0  = 1.0, P0 = 1.0;
    const double cs    = std::sqrt(gamma * P0 / rho0);
    const double Lx    = 1.0;
    const double kx    = 2.0 * M_PI / Lx;
    const double A     = 1e-4;             // linear regime: |Δρ|/ρ₀ = A
    const double T     = Lx / cs;          // exactly one wave period

    std::printf("  cs=%.4f  A=%.1e  T=%.4f (one period)\n\n", cs, A, T);

    const int resolutions[3] = {64, 128, 256};
    const int ny_fixed       = 4;
    double errors[3];

    for (int r = 0; r < 3; ++r) {
        int nx = resolutions[r];
        double Ly = ny_fixed * (Lx / nx);

        StrangSolver sol;
        sol.init(nx, ny_fixed, Lx, Ly, gamma, /*g=*/0.0, /*cfl=*/0.4,
                 /*P0=*/P0, /*rho0=*/rho0);
        // Disable the LM-HLLC pressure-jump suppression.  The LM fix is
        // needed for low-Mach convection (Andrassy) but it clamps the
        // pressure jump in the contact-wave S* formula by a factor
        // fM ≈ M_local = A; for an acoustic wave whose entire physics
        // IS the pressure jump that kills convergence — a standard
        // HLLC is the right choice here.
        sol.use_lm_fix = false;

        int N_phys = nx * ny_fixed, B = 256;
        k_init_acoustic_wave<<<(N_phys + B - 1) / B, B>>>(
            sol.d_rho, sol.d_mx, sol.d_my, sol.d_E,
            rho0, P0, gamma, A, kx,
            sol.x_lo, sol.dx, nx, ny_fixed, sol.ng, sol.stride());
        CUDA_CHECK(cudaDeviceSynchronize());

        double t = 0;
        int steps = 0;
        while (t < T) {
            double dt = sol.step(t, T);
            t += dt;
            ++steps;
        }

        errors[r] = compute_L1_rho_vs_ic(sol, rho0, A, kx);
        std::printf("  N=%3d: %5d steps  t=%.6f  L1(ρ)=%.6e\n",
                    nx, steps, t, errors[r]);
        sol.destroy();
    }

    std::printf("\nConvergence rates:\n");
    for (int r = 1; r < 3; ++r) {
        double ratio = errors[r - 1] / errors[r];
        double order = std::log2(ratio);
        std::printf("  N=%d→%d: ratio = %.2f  order = %.2f\n",
                    resolutions[r - 1], resolutions[r], ratio, order);
    }

    double order_1 = std::log2(errors[0] / errors[1]);
    double order_2 = std::log2(errors[1] / errors[2]);

    // A3 first (cheapest) — covers the NaN case.
    CHECK(std::isfinite(errors[0]) && std::isfinite(errors[1])
          && std::isfinite(errors[2]),
          "A3: all L1 values finite");
    CHECK(order_1 > 1.5, "A1a: 64→128 order > 1.5");
    CHECK(order_2 > 1.3, "A1b: 128→256 order > 1.3 (limiter activation near peak)");
    CHECK(errors[2] < errors[0] * 0.1,
          "A2: N=256 L1 < 10% of N=64 L1");

    std::printf("\n=== %s: %d passed, %d failed ===\n",
                n_fail == 0 ? "ALL PASSED" : "SOME FAILED", n_pass, n_fail);
    return n_fail;
}
