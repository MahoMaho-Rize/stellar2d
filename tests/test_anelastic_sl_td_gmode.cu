// test_anelastic_sl_td_gmode.cu
// ============================================================
// Time-domain integration test for AnelasticSLSolver.
//
// Why this test:
//   `test_anelastic_sl_gmode_convergence` locks the EVP (spatial
//   operator eigenvalue) to machine precision.  But the eigenvalue
//   alone doesn't guarantee correct long-time evolution — the TD
//   integrator has to preserve the linear Hamiltonian operator's
//   invariants.  A regression in `step_exp_propagator` (e.g. a bug
//   in Q·diag(f(ω))·Q⁻¹ assembly, or the per-kx apply kernel, or
//   the W ↔ V coupling) would show up only here, not in the EVP test.
//
// Physics:
//   Boussinesq + constant N², IC = EVP eigenmode V(y)·sin(kx·x),
//   W(0) = 0, b(0) = b(y)·sin(kx·x).  The continuous evolution is
//   exactly  v(t) = V(y)·cos(ω·t)·sin(kx·x).
//   `step_exp_propagator` applies cos(√M·dt)·V + sin(√M·dt)/√M·W
//   per-kx in Fourier space, which for the linear Hamiltonian
//   V̈ = −M·V is **exactly symplectic and phase-preserving**.
//
// Assertions (fast bucket, < 5 s):
//   H1. Hamiltonian conservation:
//         |H(t=10·T) / H(t=0) − 1| < 1e-10
//       `hamiltonian_im()` computes  ½⟨W,W⟩_CC + ½⟨V, MV⟩_CC
//       which is the exact conserved Hamiltonian of the discrete
//       V̈ = −M·V.  For an exact-over-dt exponential propagator
//       this should sit at round-off independently of the number
//       of steps — any drift indicates Q/Q⁻¹ conditioning or
//       apply-kernel cancellation problem.
//
//   H2. Phase accuracy over 10 periods:
//         |⟨v(10T), v(0)⟩ / ⟨v(0), v(0)⟩ − 1| < 1e-6
//       With exact per-dt propagation of cos(√M·dt), after 10
//       integer periods v should coincide with v(0) up to the
//       cached-dt build precision (Q·diag·Q⁻¹ is O(ny³) floating
//       ops, so 1e-6 is a healthy budget over 10·T).
//
//   H3. EVP ω matches analytic  (redundant with C3 in the other
//       test but this binary exercises `init_gmode_eigenmode`
//       + `step_exp_propagator`'s dt-setting end-to-end, so we
//       re-check at the runtime omega used to choose dt):
//         |ω²_EVP − ω²_exact| / ω²_exact < 1e-8.
// ============================================================

#include "anelastic_sl_solver.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
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
        "=== anelastic_sl TD g-mode Hamiltonian + phase ===\n");

    // Geometry + physics (match test_anelastic_sl_gmode_convergence for
    // easy cross-reference of the analytic ω).
    const int    nx      = 32;
    const int    ny      = 48;
    const int    n_modes = 8;
    const int    kx_int  = 2;
    const int    n_g     = 1;
    const double Lx      = 2.0 * M_PI;
    const double Ly      = 2.0 * M_PI;
    const double N2_val  = 1.0;
    const double amp     = 1.0e-3;

    const double kx_phys = kx_int * 2.0 * M_PI / Lx;
    const double ky_phys = n_g * M_PI / Ly;
    const double om2_exact = N2_val * kx_phys*kx_phys
                           / (kx_phys*kx_phys + ky_phys*ky_phys);
    const double om_exact  = std::sqrt(om2_exact);
    const double T_period  = 2.0 * M_PI / om_exact;

    std::fprintf(stderr,
        "  ω_exact = %.14e    T_period = %.14e\n", om_exact, T_period);

    AnelasticSLSolver sl;
    sl.init(nx, ny, n_modes, Lx, Ly, /*nu=*/1e-4, /*cfl=*/0.4);
    sl.set_background("stratified_n2", N2_val);
    double om2_evp = sl.init_gmode_eigenmode(kx_int, n_g, amp);
    double om_evp  = std::sqrt(om2_evp);

    // ── H3. EVP ω against analytic.
    {
        double rel = std::fabs(om2_evp - om2_exact) / om2_exact;
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "H3: EVP ω² rel err = %.3e (< 1e-8)", rel);
        CHECK_TRUE(rel < 1e-8, msg);
    }

    // Choose dt so 10 periods × N_STEPS_PER_PERIOD lands on an integer
    // step count; step_exp_propagator caches propagator matrices keyed
    // on dt, so we need a single dt for the whole run.
    const int    N_PERIODS         = 10;
    const int    N_STEPS_PER_PERIOD = 20;
    const double dt                 = T_period / N_STEPS_PER_PERIOD;
    const int    total_steps        = N_PERIODS * N_STEPS_PER_PERIOD;

    std::fprintf(stderr,
        "  dt = %.6e, total %d steps over %d periods\n",
        dt, total_steps, N_PERIODS);

    // hamiltonian_im() returns NaN if M_per_kx hasn't been assembled
    // yet (assemble is lazy-called from step_exp_propagator).  Force
    // assembly without advancing the state.
    sl.assemble_path_d_operators();

    // Grab IC Hamiltonian and v-snapshot for phase check.
    const double H0 = sl.hamiltonian_im();
    std::vector<double> h_u0, h_v0;
    sl.download_uv(h_u0, h_v0);
    const int ncell = (int)h_v0.size();

    // Run.
    for (int s = 0; s < total_steps; ++s) {
        sl.step_exp_propagator(dt);
    }

    const double H1 = sl.hamiltonian_im();
    std::vector<double> h_u1, h_v1;
    sl.download_uv(h_u1, h_v1);

    // ── H1. Hamiltonian conservation.
    {
        double rel = std::fabs(H1 / H0 - 1.0);
        std::fprintf(stderr,
            "  H0 = %.14e    H(10T) = %.14e    |ΔH/H| = %.3e\n",
            H0, H1, rel);
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "H1: |H(10T)/H(0) − 1| = %.3e (< 1e-10)", rel);
        CHECK_TRUE(rel < 1e-10, msg);
    }

    // ── H2. Phase accuracy: ⟨v(10T), v(0)⟩ / ⟨v(0), v(0)⟩ → 1 at integer T.
    {
        double inner = 0.0, norm0 = 0.0;
        for (int k = 0; k < ncell; ++k) {
            inner += h_v0[k] * h_v1[k];
            norm0 += h_v0[k] * h_v0[k];
        }
        double c = inner / std::max(norm0, 1e-300);   // should = cos(ω·10T) = 1
        double rel = std::fabs(c - 1.0);
        std::fprintf(stderr,
            "  ⟨v(10T), v(0)⟩ / ⟨v(0), v(0)⟩ = %.14e    |c − 1| = %.3e\n",
            c, rel);
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "H2: phase error after 10T = %.3e (< 1e-6)", rel);
        CHECK_TRUE(rel < 1e-6, msg);
    }

    std::fprintf(stderr,
        "=== %d/%d tests passed ===\n",
        g_tests - g_failures, g_tests);
    return g_failures > 0 ? 1 : 0;
}
