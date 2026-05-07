// test_pseudo_spectral_taylor_green.cu
// ============================================================
// GPU test: 2D incompressible NS pseudo-spectral solver against the
// Taylor-Green analytic decay solution.
//
// Taylor-Green vortex (the 2D streamfunction form built into
// PseudoSpectralSolver::init_taylor_green):
//   ω(x,y,0) = 2k · cos(k·2π·x/Lx) · cos(k·2π·y/Ly)
// On a doubly-periodic square domain this is an eigenfunction of
// advection (convection term is identically zero for the single-mode
// initial condition), so the ONLY dynamics is linear viscous decay:
//   ω(x,y,t) = ω(x,y,0) · exp(-2 · ν · k_phys² · t)
// where k_phys² = (k·2π/L)² + (k·2π/L)² = 2·(k·2π/L)².
//
// This is the gold-standard smoke test for any 2D spectral NS code —
// it is the single test that simultaneously exercises:
//   - FFT conventions (sign of kx, ky, R2C/C2R normalization)
//   - ω → ψ Laplacian inverse
//   - IFRK3 integrating-factor (analytic viscous exponential per mode)
//   - 2/3 dealias mask (mustn't clip the fundamental k)
//   - convection kernel (must vanish for this IC to round-off)
//
// If the code has any of those wrong, the measured decay rate either
// differs from -2νk² or a non-zero L2 error grows faster than
// truncation.
//
// Tests:
//   TG1. Short-time decay rate matches analytic.
//        Evolve to t1 and verify
//            |log(||ω||_2(t1)/||ω||_2(0)) / (-2·ν·k_phys²·t1) − 1| < 1e-2
//
//   TG2. End-of-run L2 error against the analytic field.
//        Use compute_diagnostics(t_eval) which ALREADY computes
//        err_L2 = ||ω_num − ω_exact||_L2 because init_taylor_green
//        stashes the IC spectrum (has_analytic_ic = true).
//        Require err_L2 / ||ω(0)||_L2 < 1e-3 at t = 0.5.
//
//   TG3. Convection term stays sub-dominant.
//        Since analytic ω = f(x)·f(y), the convection kernel should
//        produce ω̂ changes that are bounded by rounding only.  We
//        proxy this via total_enstrophy(t) / total_enstrophy(0):
//        the ratio must equal exp(-4·ν·k_phys²·t) to <1%.  If the
//        convection term leaks (e.g. skew-symmetric form is wrong),
//        enstrophy decays too fast or too slow.
//
// Parameters chosen so the test runs in <5s on a 4070 / H100:
//   N = 128, ν = 1e-3, k = 2, t_end = 0.5 → ~500 steps at dt=1e-3.
// ============================================================

#include "pseudo_spectral_solver.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

static int g_failures = 0;
static int g_tests = 0;

#define CHECK_TRUE(cond, msg)                                        \
    do {                                                             \
        ++g_tests;                                                   \
        if (!(cond)) {                                               \
            std::fprintf(stderr, "FAIL %s:%d [%s]\n",               \
                         __FILE__, __LINE__, msg);                   \
            ++g_failures;                                            \
        }                                                            \
    } while (0)

#define CHECK_REL(got, expected, tol, msg)                           \
    do {                                                             \
        ++g_tests;                                                   \
        double _g = (got), _e = (expected);                          \
        double _rel = std::fabs(_g - _e) / std::max(std::fabs(_e), 1e-30); \
        if (_rel > (tol)) {                                          \
            std::fprintf(stderr, "FAIL %s:%d [%s]: got=%.6e "       \
                         "expected=%.6e rel=%.3e tol=%.1e\n",       \
                         __FILE__, __LINE__, msg, _g, _e, _rel, tol); \
            ++g_failures;                                            \
        }                                                            \
    } while (0)

int main() {
    std::printf("=== pseudo_spectral Taylor-Green decay ===\n\n");

    const int    N      = 128;
    const double Lx     = 2.0 * M_PI;
    const double Ly     = 2.0 * M_PI;
    const double nu     = 1.0e-3;
    const double cfl    = 0.5;
    const int    k_mode = 2;
    const double t_end  = 0.5;

    // Analytic decay rate  κ = 2·ν·|k_phys|²
    // k_phys = (k·2π/L, k·2π/L) → |k_phys|² = 2·(k·2π/L)²
    const double kph2 = 2.0 * std::pow(k_mode * 2.0 * M_PI / Lx, 2);
    const double decay_rate = 2.0 * nu * kph2;                 // for ω
    const double decay_rate_Z = 2.0 * decay_rate;              // for ½|ω|² = enstrophy

    std::printf("  N=%d  ν=%.1e  k=%d  |k_phys|²=%.4f  κ_ω=%.4f\n",
                N, nu, k_mode, kph2, decay_rate);
    std::printf("  t_end=%.3f → analytic ||ω|| ratio = %.6f\n",
                t_end, std::exp(-decay_rate * t_end));

    PseudoSpectralSolver sol;
    sol.init(N, N, Lx, Ly, nu, cfl);
    sol.init_taylor_green(k_mode);

    // === TG0: initial diagnostic (t=0 should give err_L2 ~ 0) ===
    auto d0 = sol.compute_diagnostics(0.0);
    double Z0  = d0.total_enstrophy;
    double KE0 = d0.total_KE;
    std::printf("  TG0: initial Ω = %.6e, KE = %.6e, err_L2 @ t=0 = %.3e\n",
                Z0, KE0, d0.err_L2);
    CHECK_TRUE(d0.err_L2 < 1e-10,
        "TG0: err_L2 at t=0 is floating-point noise");
    CHECK_TRUE(Z0 > 1e-10, "TG0: initial enstrophy non-zero (IC non-trivial)");

    // === Evolve to t_end ===
    double t = 0.0;
    int steps = 0;
    while (t < t_end) {
        double dt = sol.step();
        t += dt;
        ++steps;
        if (steps > 200000) {
            std::fprintf(stderr, "  ABORT: step count runaway (%d)\n", steps);
            break;
        }
    }
    std::printf("  advanced %d steps to t = %.6f\n", steps, t);

    // === TG1 + TG3: diagnostics vs analytic decay ===
    auto d1 = sol.compute_diagnostics(t);
    double Z1  = d1.total_enstrophy;
    double KE1 = d1.total_KE;

    double ratio_Z_num  = Z1 / Z0;
    double ratio_Z_ana  = std::exp(-decay_rate_Z * t);
    double ratio_KE_num = KE1 / KE0;
    // Taylor-Green geometry:  ∫½u² = (1/k_phys²)·∫½ω² ⇒ KE decays at
    // same rate as Ω, both ∝ exp(-2·decay_rate·t).
    double ratio_KE_ana = ratio_Z_ana;

    std::printf("\n  TG1: ||ω||_2 decay\n"
                "       numeric Ω(t)/Ω(0) = %.6e\n"
                "       analytic          = %.6e\n"
                "       rel err           = %.3e\n",
                ratio_Z_num, ratio_Z_ana,
                std::fabs(ratio_Z_num - ratio_Z_ana) / ratio_Z_ana);

    std::printf("\n  TG3: KE decay\n"
                "       numeric KE(t)/KE(0) = %.6e\n"
                "       analytic             = %.6e\n"
                "       rel err              = %.3e\n",
                ratio_KE_num, ratio_KE_ana,
                std::fabs(ratio_KE_num - ratio_KE_ana) / ratio_KE_ana);

    CHECK_REL(ratio_Z_num,  ratio_Z_ana,  1e-2,  "TG1: enstrophy decays at analytic rate");
    CHECK_REL(ratio_KE_num, ratio_KE_ana, 1e-2,  "TG3: KE decays at analytic rate (no spurious convection)");

    // === TG2: pointwise L2 error field vs analytic ===
    // err_L2 = ||ω_num − ω_exact||_L2, exact = ω(0) · exp(-decay_rate·t)
    // Normalize by ||ω(0)||_L2 = sqrt(2·Z0) for a dimensionless comparison.
    double omega_norm_0 = std::sqrt(2.0 * Z0);
    double rel_err = d1.err_L2 / std::max(omega_norm_0, 1e-30);
    std::printf("\n  TG2: err_L2 @ t=%.3f     = %.3e\n"
                "       ||ω(0)||_L2          = %.3e\n"
                "       relative             = %.3e\n",
                t, d1.err_L2, omega_norm_0, rel_err);

    CHECK_TRUE(rel_err < 1e-3, "TG2: pointwise L2 error bounded");

    sol.destroy();

    std::printf("\n=== %d/%d tests passed ===\n",
                g_tests - g_failures, g_tests);
    return g_failures > 0 ? 1 : 0;
}
