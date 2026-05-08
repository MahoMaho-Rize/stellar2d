// test_radial1d_hse_static.cu
// ============================================================
// HSE bounded-stability regression for Radial1DSolver (explicit RK2 path).
//
// Physics:
//   Lane-Emden polytrope with n = 3/2 and solver γ = 5/3 is an *exact*
//   static solution of the continuum equations:  P = K·ρ^γ matches the
//   adiabatic EOS when γ = 1 + 1/n, so v ≡ 0 is a fixed point.
//
//   But the *discrete* explicit RK2 solver is NOT well-balanced
//   (well-balancing in radial1d lives in the implicit JFNK path —
//   F = (U−Uⁿ)/dt − (R(U) − R_hse) — and that's the only place where
//   the discrete imbalance is subtracted out; see CLAUDE.md).  On the
//   explicit path the residual gravity−pressure imbalance is O(Δr²)
//   and drives a low-amplitude bounded oscillation about HSE, not a
//   stationary state.
//
//   So what this test really locks is:
//     • No runaway / no NaN (regression in momentum/energy kernel,
//       enclosed-mass build, gravity kernel, surface-floor BC).
//     • Mass conservation (Lagrangian, bit-exact).
//     • Bounded pulsation amplitude (R, v, E all stay in a few-%
//       envelope instead of going to the Moon).
//
//   The magnitudes we lock are budgeted against the actual pulsation
//   signature observed on this IC at nz=128, 400 steps (~30 acoustic
//   crossings):  R drifts ~2%, max|v|/c_s ~1%, E drifts ~0.1%.  A
//   bug in the Lagrangian kernels (e.g. a sign flip in gravity,
//   momentum-update stencil regression, or wrong ghost-P convention)
//   blows these up by orders of magnitude.
//
// Assertions (fast bucket, < 2 s):
//
//   R1. NO-NAN — every zone/face finite after the run.
//
//   R2. MASS CONSERVATION (Lagrangian, bit-exact modulo reduction
//       order):  |Σ dm(t_end) − Σ dm(0)| / Σ dm(0) < 1e-12.
//
//   R3. SURFACE-RADIUS BOUNDED:  |R(t_end) − R(0)| / R(0) < 5e-2.
//       The measured drift on a correct solver is ~2%; we budget
//       2.5× that.  A runaway (e.g. wrong gravity sign) produces
//       collapse in < 10 steps.
//
//   R4. VELOCITY BOUNDED:  max_k |v_face[k]| / c_s0  < 5e-2.  The
//       measured max|v|/c_s on a correct solver is ~1%; we budget 5×
//       that.  A runaway drives |v| ~ c_s within a few steps.
//
//   R5. TOTAL-ENERGY BOUNDED: |E(t_end)/E(0) − 1| < 1e-2.  Measured
//       drift is ~1e-3; we budget 10×.  Artificial viscosity is not
//       meant to fire on this quasi-static IC, and if it accidentally
//       does, |ΔE/E| crosses this threshold.
//
// Knobs:
//   nz = 128, n_poly = 1.5, K_poly = 0.5, ρ_c = 1, G = 1, CFL = 0.4,
//   N_STEPS = 400.  c_s0 analytic = √(γ·K·ρ_c^γ) = √(5/3·0.5) ≈ 0.913.
//
// This is the explicit path only — the JFNK implicit path has its own
// regression via test_residual_dual.cu.
// ============================================================

#include "radial1d_solver.cuh"

#include <cmath>
#include <cstdio>
#include <vector>

static int g_failures = 0;
static int g_tests = 0;

#define CHECK_TRUE(cond, msg)                                        \
    do {                                                             \
        ++g_tests;                                                   \
        if (!(cond)) {                                               \
            std::fprintf(stderr, "FAIL %s:%d [%s]\n",                \
                         __FILE__, __LINE__, msg);                   \
            ++g_failures;                                            \
        }                                                            \
    } while (0)

int main()
{
    std::fprintf(stderr,
        "=== radial1d HSE static-stability regression ===\n");

    // Polytrope + grid params.
    const int    nz       = 128;
    const double n_poly   = 1.5;
    const double K_poly   = 0.5;
    const double rho_c    = 1.0;
    const double gamma    = 5.0 / 3.0;   // matches 1 + 1/n for n=1.5
    const double G_const  = 1.0;
    const double cfl      = 0.4;
    const int    N_STEPS  = 400;

    Radial1DSolver r1d;
    r1d.init(nz, gamma, G_const, cfl);
    r1d.init_lane_emden(rho_c, K_poly, n_poly);

    // compute_diagnostics() itself calls launch_primitives, so we run it
    // once before download_profile to ensure (d_rho, d_P) are populated;
    // otherwise download_profile reads the zero-initialised allocations.
    Radial1DSolver::Diagnostics d0 = r1d.compute_diagnostics();

    // Capture initial state.
    std::vector<double> r0, v0, rho0, P0, e0;
    r1d.download_profile(r0, v0, rho0, P0, e0);

    const double R_star_0 = r0[nz];
    // Analytic central sound speed: cs² = γ·K·ρ_c^γ = γ · P(ρ_c).
    const double cs0 = std::sqrt(gamma * K_poly * std::pow(rho_c, gamma));
    std::fprintf(stderr,
        "  IC: R_star=%.6f, ρ_c=%.6f, P_c=%.6f, c_s0=%.6f,\n"
        "      M_total=%.6e, E_total=%.6e\n",
        R_star_0, rho0[0], P0[0], cs0, d0.total_mass, d0.total_E);

    // Run.
    double t = 0.0;
    const double t_end = 1.0e30;   // let solver's own dt govern
    for (int s = 0; s < N_STEPS; ++s) {
        r1d.step(t, t_end);
        // Solver sets dt_current internally; t increment for bookkeeping
        // (t_end is effectively infinite, so step() advances by dt_current
        //  without truncation). We don't depend on t here.
        t += r1d.dt_current;
    }

    std::vector<double> r1, v1, rho1, P1, e1;
    r1d.download_profile(r1, v1, rho1, P1, e1);
    Radial1DSolver::Diagnostics d1 = r1d.compute_diagnostics();

    // ── R1. NO-NAN on every field.
    {
        bool finite_all = true;
        for (double x : r1)   if (!std::isfinite(x)) finite_all = false;
        for (double x : v1)   if (!std::isfinite(x)) finite_all = false;
        for (double x : rho1) if (!std::isfinite(x)) finite_all = false;
        for (double x : P1)   if (!std::isfinite(x)) finite_all = false;
        for (double x : e1)   if (!std::isfinite(x)) finite_all = false;
        CHECK_TRUE(finite_all, "R1: all fields finite after run");
    }

    // ── R2. Mass conservation (Lagrangian: bit-exact modulo reduction order).
    {
        const double rel = std::fabs(d1.total_mass / d0.total_mass - 1.0);
        std::fprintf(stderr,
            "  mass: M0=%.14e  M(t)=%.14e  |ΔM/M|=%.3e\n",
            d0.total_mass, d1.total_mass, rel);
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "R2: |ΔM/M| = %.3e (< 1e-12)", rel);
        CHECK_TRUE(rel < 1e-12, msg);
    }

    // ── R3. Surface radius bounded (discrete HSE imbalance → pulsation).
    {
        const double R1 = r1[nz];
        const double rel = std::fabs(R1 / R_star_0 - 1.0);
        std::fprintf(stderr,
            "  R_star: R0=%.6f   R(t)=%.6f   |ΔR/R|=%.3e\n",
            R_star_0, R1, rel);
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "R3: |ΔR_star/R| = %.3e (< 5e-2)", rel);
        CHECK_TRUE(rel < 5e-2, msg);
    }

    // ── R4. Velocity bounded (max |v_face| / c_s0 < 5e-2).
    {
        double v_max = 0.0;
        for (double v : v1) v_max = std::max(v_max, std::fabs(v));
        const double rel = v_max / cs0;
        std::fprintf(stderr,
            "  v: max|v_face| = %.3e   max|v|/c_s0 = %.3e\n",
            v_max, rel);
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "R4: max|v|/c_s0 = %.3e (< 5e-2)", rel);
        CHECK_TRUE(rel < 5e-2, msg);
    }

    // ── R5. Total energy bounded.
    {
        const double rel = std::fabs(d1.total_E / d0.total_E - 1.0);
        std::fprintf(stderr,
            "  E: E0=%.6e   E(t)=%.6e   |ΔE/E|=%.3e\n",
            d0.total_E, d1.total_E, rel);
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "R5: |ΔE/E| = %.3e (< 1e-2)", rel);
        CHECK_TRUE(rel < 1e-2, msg);
    }

    r1d.destroy();

    std::fprintf(stderr,
        "=== %d/%d tests passed ===\n",
        g_tests - g_failures, g_tests);
    return g_failures > 0 ? 1 : 0;
}
