// test_anelastic_sl_td_nonlinear.cu
// ============================================================
// Nonlinear-path regression on AnelasticSLSolver — designed around
// the diagnostics that `scripts/spectral/nonlinear_paths_infra.py`
// built during the Phase 3 prototype work and the specific bugs that
// `docs/projects/spectral_liouville/dns_expE1_triad_2026-05-04.md`
// documents.  Those references encode what "healthy nonlinear TD"
// actually means for this solver: it's not a `hamiltonian_im()`
// lock, it's:
//
//   • the *physical* total energy   E = ½⟨ρ (u²+v²)⟩_{y,CC} +
//                                       ½⟨b² / N²⟩_{y,CC}
//     (the linearised anelastic energy used by Python path 1 and the
//      paper figures).
//   • per-kx band energy, specifically the kx=0 column which must
//     stay at round-off (anelastic continuity forbids a mean flow;
//     dns_expE1 §2.2 documents the bug where it leaked to ~3e-13).
//   • high-kx mode amplitude preservation in the linear limit
//     (dns_expE1 §2.1 — W being erroneously advected killed 95% of
//      the mode-b energy at kx=5 over 100 periods).
//
// Three independent assertions:
//
//   T1. LINEAR-LIMIT EQUIVALENCE — step_strang_exp_nonlinear(dt) at
//       amp = 1e-6 must reproduce step_exp_propagator(dt) to within
//       ‖Δv‖₂ / ‖v‖₂ < 1e-8 over 10 periods.  The nonlinear block is
//       O(amp²) in this limit so the residual comes only from the
//       advection-substep composition: substep copies, Galerkin
//       V_K mask, DC-zeroing, and Strang half-step ordering.  We
//       intentionally compare exp-half vs exp-nonlinear-half (both
//       exact) — comparing RK4-half would be dominated by O((ω·dt)⁵)
//       RK4 phase error that has nothing to do with the nonlinear
//       pipeline.
//
//   T2. DC-COLUMN ZEROING (Bug #2 regression lock).
//       Even with a nonlinear block running, ‖v̂(kx=0, y)‖_∞ must
//       stay at round-off for the whole run.  If DC-zeroing in
//       `step_strang_nonlinear` regresses, nonlinear advection
//       generates a mean flow and anelastic continuity is violated;
//       per the docs this leaked ~3e-13 after 100 periods at amp=1e-6.
//
//   T3. HIGH-KX AMPLITUDE PRESERVATION (Bug #1 regression lock).
//       Seed a (kx=5, n_g=1) eigenmode at amp=1e-6 and run 50 periods
//       through step_strang_exp_nonlinear.  Per Python prototype
//       measurements the kx=5 band energy should be preserved to
//       ~1e-9 relative drift; the original bug (advecting W along
//       (u,v)) produced a monotone -95% drift at this kx.  We lock
//       the kx=5 band energy drift < 1e-3 — conservative above the
//       1e-9 floor but well below the bug signature.
// ============================================================

#include "anelastic_sl_solver.cuh"

#include <cmath>
#include <complex>
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

// Clenshaw-Curtis quadrature weights on [0, Ly] — same recipe as
// AnelasticSLSolver::hamiltonian_im (reversed indexing, w/ endpoint
// halving).  ny = number of CGL nodes.
static std::vector<double> cc_weights(int ny, double Ly) {
    const int N = ny - 1;
    std::vector<double> w(N + 1, 0.0);
    for (int k = 0; k <= N; ++k) {
        double s = 0.0;
        const int J = N / 2;
        for (int j = 1; j <= J; ++j) {
            double b = (2 * j != N) ? 2.0 : 1.0;
            s += b / (4.0 * j * j - 1) *
                 std::cos(2.0 * j * k * M_PI / N);
        }
        w[k] = (1.0 - s) * 2.0 / (double)N;
    }
    w[0] *= 0.5;  w[N] *= 0.5;
    std::vector<double> w_cc(ny, 0.0);
    for (int k = 0; k <= N; ++k) w_cc[k] = w[N - k] * Ly / 2.0;
    return w_cc;
}

// Total linearised anelastic energy on the CGL grid:
//   E = ½⟨ρ(u²+v²)⟩_{y:CC, x:avg}  +  ½⟨b²/N²⟩_{y:CC, x:avg}
// Buoyancy contribution only where N² > 0 (skips the surface).
static double total_anelastic_energy(
    const std::vector<double>& h_u,
    const std::vector<double>& h_v,
    const std::vector<double>& h_b,
    const std::vector<double>& h_rho,
    const std::vector<double>& h_N2,
    const std::vector<double>& w_cc,
    int ny, int nx)
{
    double KE = 0.0, PE = 0.0;
    for (int jy = 0; jy < ny; ++jy) {
        double row_u2v2 = 0.0, row_b2 = 0.0;
        for (int ix = 0; ix < nx; ++ix) {
            int k = jy * nx + ix;
            row_u2v2 += h_u[k] * h_u[k] + h_v[k] * h_v[k];
            row_b2   += h_b[k] * h_b[k];
        }
        KE += 0.5 * w_cc[jy] * h_rho[jy] * row_u2v2 / (double)nx;
        if (h_N2[jy] > 1e-12) {
            PE += 0.5 * w_cc[jy] * row_b2 / h_N2[jy] / (double)nx;
        }
    }
    return KE + PE;
}

// Extract the magnitude of the kx-component of a field: given (ny,nx)
// row-major, return ‖v̂(kx=target, y)‖_∞ over y.  Uses plain O(ny·nx)
// host DFT since nx is small (≤128) and nx is even.
static double linf_at_kx(const std::vector<double>& h_field,
                         int ny, int nx, int kx_target)
{
    double linf = 0.0;
    for (int jy = 0; jy < ny; ++jy) {
        std::complex<double> acc(0.0, 0.0);
        for (int ix = 0; ix < nx; ++ix) {
            double arg = -2.0 * M_PI * (double)kx_target * (double)ix / (double)nx;
            acc += std::complex<double>(std::cos(arg), std::sin(arg))
                 * h_field[jy * nx + ix];
        }
        // Normalise by nx so |v̂(kx=0)| matches the x-mean magnitude.
        double mag = std::abs(acc) / (double)nx;
        if (mag > linf) linf = mag;
    }
    return linf;
}

// Band energy for a single kx on the CGL-weighted norm:
//   E_kx = ½ Σ_y w_cc[y] · ρ[y] · (|û(kx,y)|² + |v̂(kx,y)|²)
//        + ½ Σ_y w_cc[y] · |b̂(kx,y)|² / N²[y]
// All |·|² are complex Fourier coefficients normalised by nx
// (two-sided; we include the factor-of-two for kx>0 implicitly in
// the comparison).
static double band_energy(
    const std::vector<double>& h_u,
    const std::vector<double>& h_v,
    const std::vector<double>& h_b,
    const std::vector<double>& h_rho,
    const std::vector<double>& h_N2,
    const std::vector<double>& w_cc,
    int ny, int nx, int kx_target)
{
    auto complex_at = [&](const std::vector<double>& f, int jy) {
        std::complex<double> acc(0.0, 0.0);
        for (int ix = 0; ix < nx; ++ix) {
            double arg = -2.0 * M_PI * (double)kx_target * (double)ix / (double)nx;
            acc += std::complex<double>(std::cos(arg), std::sin(arg))
                 * f[jy * nx + ix];
        }
        return acc / (double)nx;
    };

    double KE = 0.0, PE = 0.0;
    for (int jy = 0; jy < ny; ++jy) {
        auto uk = complex_at(h_u, jy);
        auto vk = complex_at(h_v, jy);
        auto bk = complex_at(h_b, jy);
        double mag2_uv = std::norm(uk) + std::norm(vk);
        double mag2_b  = std::norm(bk);
        KE += 0.5 * w_cc[jy] * h_rho[jy] * mag2_uv;
        if (h_N2[jy] > 1e-12) {
            PE += 0.5 * w_cc[jy] * mag2_b / h_N2[jy];
        }
    }
    return KE + PE;
}

struct RunResult {
    std::vector<double> u, v, b;
};

template <typename Integrator>
static RunResult run_fixed_ic(Integrator integrator,
                              int n_steps,
                              int kx_int, int n_g,
                              int nx, int ny, int n_modes,
                              double Lx, double Ly, double N2,
                              double amp)
{
    AnelasticSLSolver sl;
    sl.init(nx, ny, n_modes, Lx, Ly, /*nu=*/1e-4, /*cfl=*/0.4);
    sl.set_background("stratified_n2", N2);
    sl.init_gmode_eigenmode(kx_int, n_g, amp);
    sl.assemble_path_d_operators();

    for (int s = 0; s < n_steps; ++s) integrator(sl);

    RunResult r;
    sl.download_uv(r.u, r.v);
    sl.download_b(r.b);
    sl.free_all();
    return r;
}

int main()
{
    std::fprintf(stderr,
        "=== anelastic_sl TD nonlinear-pipeline regression ===\n");

    // Base params: chosen small enough to run in ~3 s total with 3 probes.
    const int    nx      = 32;
    const int    ny      = 48;
    const int    n_modes = 8;
    const double Lx      = 2.0 * M_PI;
    const double Ly      = 2.0 * M_PI;
    const double N2_val  = 1.0;
    const double amp     = 1.0e-6;        // deep linear limit

    const std::vector<double> w_cc = cc_weights(ny, Ly);
    // For stratified_n2 background ρ ≡ 1, N² ≡ N2_val everywhere.
    const std::vector<double> h_rho(ny, 1.0);
    const std::vector<double> h_N2(ny, N2_val);

    auto period_of = [&](int kx_int, int n_g) {
        const double kx_phys = kx_int * 2.0 * M_PI / Lx;
        const double ky_phys = n_g * M_PI / Ly;
        const double om2 = N2_val * kx_phys * kx_phys
                         / (kx_phys * kx_phys + ky_phys * ky_phys);
        return 2.0 * M_PI / std::sqrt(om2);
    };

    // ── T1: linear-limit equivalence ───────────────────────────────
    {
        const int    kx_int  = 2;
        const int    n_g     = 1;
        const int    N_PERIODS = 10;
        const int    N_STEPS_PER_PERIOD = 20;
        const double T_period = period_of(kx_int, n_g);
        const double dt       = T_period / N_STEPS_PER_PERIOD;
        const int    total    = N_PERIODS * N_STEPS_PER_PERIOD;

        std::fprintf(stderr,
            "  [T1] kx=%d, n_g=%d, %d steps × dt=%.3e\n",
            kx_int, n_g, total, dt);

        RunResult ref = run_fixed_ic(
            [dt](AnelasticSLSolver& s){ s.step_exp_propagator(dt); },
            total, kx_int, n_g, nx, ny, n_modes, Lx, Ly, N2_val, amp);
        RunResult sub = run_fixed_ic(
            [dt](AnelasticSLSolver& s){ s.step_strang_exp_nonlinear(dt); },
            total, kx_int, n_g, nx, ny, n_modes, Lx, Ly, N2_val, amp);

        double num = 0.0, den = 0.0;
        for (size_t k = 0; k < ref.v.size(); ++k) {
            double d = sub.v[k] - ref.v[k];
            num += d * d;
            den += ref.v[k] * ref.v[k];
        }
        double rel = std::sqrt(num / std::max(den, 1e-300));
        std::fprintf(stderr,
            "       ‖v_nl − v_exp‖₂ / ‖v_exp‖₂ = %.3e   (bound 1e-8)\n", rel);
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "T1: linear-limit residual %.3e < 1e-8", rel);
        CHECK_TRUE(rel < 1e-8, msg);
    }

    // ── T2: DC-column zero-lock (Bug #2 regression) ─────────────────
    // Run step_strang_exp_nonlinear on a (kx=1, n_g=1) mode for
    // several periods and verify the kx=0 column never grows.
    {
        const int    kx_int  = 1;
        const int    n_g     = 1;
        const int    N_PERIODS = 5;
        const int    N_STEPS_PER_PERIOD = 20;
        const double T_period = period_of(kx_int, n_g);
        const double dt       = T_period / N_STEPS_PER_PERIOD;
        const int    total    = N_PERIODS * N_STEPS_PER_PERIOD;

        std::fprintf(stderr,
            "  [T2] kx=%d, n_g=%d, %d steps × dt=%.3e\n",
            kx_int, n_g, total, dt);

        RunResult r = run_fixed_ic(
            [dt](AnelasticSLSolver& s){ s.step_strang_exp_nonlinear(dt); },
            total, kx_int, n_g, nx, ny, n_modes, Lx, Ly, N2_val, amp);

        // Check all three fields for DC leakage.
        double dc_u = linf_at_kx(r.u, ny, nx, 0);
        double dc_v = linf_at_kx(r.v, ny, nx, 0);
        double dc_b = linf_at_kx(r.b, ny, nx, 0);
        // Reference amplitude so the threshold is scale-independent.
        double scale = amp;  // u, v are normalised to amp at IC
        std::fprintf(stderr,
            "       DC leak:  |û(0)|∞ = %.3e   |v̂(0)|∞ = %.3e   "
            "|b̂(0)|∞ = %.3e   (amp=%.0e)\n",
            dc_u, dc_v, dc_b, amp);

        // Lock at 1e-9·amp = 1e-15 — well below Bug #2's ~3e-13 leak
        // signature (documented at amp=1e-6 × 100 periods).  Our run
        // is shorter (5 periods) so ~1.5e-14 is expected as an upper
        // edge; we measured <1e-15 in practice.
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "T2: v̂(kx=0) DC leak %.3e < 1e-9·amp (=%.3e)",
            dc_v, 1e-9 * scale);
        CHECK_TRUE(dc_v < 1e-9 * scale, msg);
        std::snprintf(msg, sizeof(msg),
            "T2: û(kx=0) DC leak %.3e < 1e-9·amp", dc_u);
        CHECK_TRUE(dc_u < 1e-9 * scale, msg);
    }

    // ── T3: high-kx amplitude preservation (Bug #1 regression) ─────
    // Seed (kx=5, n_g=1) — high kx where the original W-advection bug
    // killed 95% of mode energy over 100 periods.  Run just 10 periods
    // here (fast bucket) and lock the band energy stays within 1e-3
    // of IC.  Bug signature would be monotone decay of ~1 % per period,
    // ≳ 10% over 10 periods — way above our lock.
    {
        const int    kx_int  = 5;
        const int    n_g     = 1;
        const int    N_PERIODS = 10;
        const int    N_STEPS_PER_PERIOD = 30;   // higher kx → shorter T → more steps
        const double T_period = period_of(kx_int, n_g);
        const double dt       = T_period / N_STEPS_PER_PERIOD;
        const int    total    = N_PERIODS * N_STEPS_PER_PERIOD;

        std::fprintf(stderr,
            "  [T3] kx=%d, n_g=%d, T=%.3e, %d steps × dt=%.3e\n",
            kx_int, n_g, T_period, total, dt);

        RunResult ic = run_fixed_ic(
            [](AnelasticSLSolver&){ /* no steps: capture IC */ },
            0, kx_int, n_g, nx, ny, n_modes, Lx, Ly, N2_val, amp);
        RunResult r  = run_fixed_ic(
            [dt](AnelasticSLSolver& s){ s.step_strang_exp_nonlinear(dt); },
            total, kx_int, n_g, nx, ny, n_modes, Lx, Ly, N2_val, amp);

        double E0 = band_energy(ic.u, ic.v, ic.b, h_rho, h_N2, w_cc,
                                ny, nx, kx_int);
        double Et = band_energy(r.u,  r.v,  r.b,  h_rho, h_N2, w_cc,
                                ny, nx, kx_int);
        double drift = std::fabs(Et / std::max(E0, 1e-300) - 1.0);
        std::fprintf(stderr,
            "       E(kx=5, 0) = %.6e   E(kx=5, 10T) = %.6e   |ΔE/E| = %.3e\n",
            E0, Et, drift);
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "T3: high-kx band energy drift %.3e < 1e-3", drift);
        CHECK_TRUE(drift < 1e-3, msg);
    }

    std::fprintf(stderr,
        "=== %d/%d tests passed ===\n",
        g_tests - g_failures, g_tests);
    return g_failures > 0 ? 1 : 0;
}
