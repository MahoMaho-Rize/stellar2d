// test_anelastic_sl_gmode_convergence.cu
// ============================================================
// First convergence test for AnelasticSLSolver.
//
// Physics:
//   Boussinesq-limit g-mode in a uniform-density layer with constant
//   Brunt-Väisälä frequency N².  Geometry: periodic in x, Dirichlet
//   in y.  The exact dispersion of a (kx, n_g) g-mode is
//
//        ω²_exact  =  N² · kx² / (kx² + (n_g · π / Ly)²)
//
//   with  kx = kx_int · 2π / Lx  and  n_g the vertical mode number
//   (n_g = 1 is the fundamental).  This analytic relation is what
//   `compute_2d_gmode_evp_qspace_sl` — the operator-consistent EVP
//   path — should reproduce once the number of SL basis modes is
//   large enough to resolve the eigenfunction sin(n_g · π · y / Ly).
//
// Why this test matters:
//   anelastic_sl had ZERO numerical-convergence coverage until this
//   test.  The solver is currently doing paper work for g-mode triad
//   resonance, so an undetected regression in any of the five
//   dependencies of `init_gmode_eigenmode`
//     (SL basis μ_n / ψ_n, Clenshaw-Curtis quadrature, the Galerkin
//      matrix H_nm = ⟨ψ_n, N² ψ_m⟩, the ρ₀-scaled Helmholtz LHS,
//      and the Xgeev-based diagonalisation)
//   would silently pollute every downstream figure.  Here we lock the
//   first eigenvalue of the (kx_int = 2, n_g = 1) mode against the
//   analytic Boussinesq dispersion at three resolutions.
//
// Assertions (the whole test lives in fast-bucket, < 5 s):
//   C1. Absolute accuracy.  At ny = 64 the relative error
//         | ω_num − ω_exact | / ω_exact < 1e-8.
//       The SL basis is spectral, so this is an "eye-watering tight"
//       bound; anything looser means the operator is assembled wrong.
//
//   C2. Spectral-convergence witness.  The ratio
//         err(48) / err(32)  must be < 0.5  AND
//         err(64) / err(48)  must be < 0.5
//       i.e. the error halves at each step.  A textbook spectral
//       method would hit err ~ 1e-12 by ny = 64 giving ratios of
//       ~1e-4; we lock 0.5 because below ~1e-12 the ratio becomes
//       dominated by round-off.  Any regression to finite-difference
//       order (ratio ≈ 0.25 per DOUBLING = 0.7 per 32 → 48 step) fires.
//
//   C3. `init_gmode_eigenmode` (which returns ω², not ω — despite the
//       variable name `omega` inside the function) agrees with the
//       analytic value.  The default path is the v-space Galerkin EVP
//       (distinct from the qspace_sl path exercised in C1/C2); this
//       covers the currently-default code path.
//
//   C4. Galerkin v-space EVP convergence witness.  Same error-halving
//       criterion as C2 but applied to the default path
//       (`compute_2d_gmode_evp`).  This catches a regression in the
//       v-space assembly that wouldn't show up in qspace_sl.
// ============================================================

#include "anelastic_sl_solver.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
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

enum class EvpPath { QSPACE_SL, GALERKIN_V };

// Single EVP probe: returns omega_num at resolution ny on the requested path.
static double run_evp_probe(int ny, int kx_int, int n_g,
                            double Lx, double Ly, double N2_const,
                            EvpPath path)
{
    AnelasticSLSolver sl;
    // nx is only used by the FFT-x machinery; set it to a token even
    // power-of-two.  nu/cfl are ignored by the EVP routines but the init
    // routine wants something sensible.
    const int nx = 16;
    const int n_modes = std::max(n_g + 4, 8);
    sl.init(nx, ny, n_modes, Lx, Ly, /*nu=*/1e-4, /*cfl=*/0.4);
    sl.set_background("stratified_n2", /*N²=*/N2_const);

    const double kx_phys = kx_int * 2.0 * M_PI / Lx;
    std::vector<double> omega_sq;
    std::vector<double> v_modes_full;
    if (path == EvpPath::QSPACE_SL) {
        sl.compute_2d_gmode_evp_qspace_sl(kx_phys, n_modes, omega_sq, v_modes_full);
    } else {
        sl.compute_2d_gmode_evp(kx_phys, n_modes, omega_sq, v_modes_full);
    }
    sl.free_all();

    if ((int)omega_sq.size() < n_g) {
        std::fprintf(stderr,
            "EVP probe: ny=%d returned only %zu modes (needed %d).\n",
            ny, omega_sq.size(), n_g);
        std::exit(1);
    }
    return std::sqrt(omega_sq[n_g - 1]);
}

int main()
{
    std::fprintf(stderr,
        "=== anelastic_sl g-mode EVP convergence test ===\n");

    // Parameters: fundamental mode kx_int = 2, n_g = 1 in a square domain
    // with N² = 1.  Keep Lx = Ly so the analytic ω has a tidy closed form.
    const int    kx_int  = 2;
    const int    n_g     = 1;
    const double Lx      = 2.0 * M_PI;
    const double Ly      = 2.0 * M_PI;
    const double N2_val  = 1.0;

    const double kx_phys  = kx_int * 2.0 * M_PI / Lx;
    const double ky_phys  = n_g * M_PI / Ly;
    const double kx_sq    = kx_phys * kx_phys;
    const double ky_sq    = ky_phys * ky_phys;
    const double om2_exact = N2_val * kx_sq / (kx_sq + ky_sq);
    const double om_exact  = std::sqrt(om2_exact);

    std::fprintf(stderr,
        "  analytic: ω²_exact = %.12e   ω_exact = %.12e\n",
        om2_exact, om_exact);

    // Run three resolutions on the qspace_sl path.
    std::fprintf(stderr, "  [path = qspace_sl]\n");
    const int NY_LIST[3] = {32, 48, 64};
    double err_abs[3] = {0, 0, 0};
    double om_num[3]  = {0, 0, 0};

    for (int r = 0; r < 3; ++r) {
        int ny = NY_LIST[r];
        double om = run_evp_probe(ny, kx_int, n_g, Lx, Ly, N2_val,
                                  EvpPath::QSPACE_SL);
        om_num[r]  = om;
        err_abs[r] = std::fabs(om - om_exact);
        std::fprintf(stderr,
            "  ny=%3d  ω_num = %.14e  |err| = %.3e  rel = %.3e\n",
            ny, om, err_abs[r], err_abs[r] / om_exact);
    }

    // ── C1. Absolute accuracy at the finest grid.
    {
        const double rel64 = err_abs[2] / om_exact;
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "C1: rel err at ny=64 = %.3e (< 1e-8)", rel64);
        CHECK_TRUE(rel64 < 1e-8, msg);
    }

    // ── C2. Spectral convergence witness: err must halve at each step.
    //       (Textbook spectral is much faster; we lock 0.5 because below
    //       ~1e-12 the ratio floors on double-precision round-off.)
    {
        // Guard against the err(48) → round-off case where err(32) is not
        // yet at the floor but err(48) already is.  If err(48) is below
        // 1e-12 the ratio test is meaningless; check absolute instead.
        const double r1 = (err_abs[0] > 0) ? (err_abs[1] / err_abs[0]) : 0.0;
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "C2a: err(48)/err(32) = %.3e (< 0.5)", r1);
        CHECK_TRUE(err_abs[1] < 1e-12 || r1 < 0.5, msg);

        const double r2 = (err_abs[1] > 0) ? (err_abs[2] / err_abs[1]) : 0.0;
        std::snprintf(msg, sizeof(msg),
            "C2b: err(64)/err(48) = %.3e (< 0.5)", r2);
        CHECK_TRUE(err_abs[2] < 1e-12 || r2 < 0.5, msg);
    }

    // ── C3. init_gmode_eigenmode returns **ω²** (not ω; the internal
    //       variable is labelled `om2` inside the function and that is
    //       the returned value).  Default path is v-space Galerkin, so
    //       compare against the analytic ω²_exact directly with a looser
    //       tolerance (Galerkin is spectral but slower-converging than
    //       qspace_sl on a discrete grid).
    {
        AnelasticSLSolver sl;
        const int ny_probe = 64;
        const int nx       = 16;
        const int n_modes  = std::max(n_g + 4, 8);
        sl.init(nx, ny_probe, n_modes, Lx, Ly, 1e-4, 0.4);
        sl.set_background("stratified_n2", N2_val);
        double om2_from_init = sl.init_gmode_eigenmode(kx_int, n_g, /*amp=*/1e-3);
        sl.free_all();

        double rel = std::fabs(om2_from_init - om2_exact) / om2_exact;
        std::fprintf(stderr,
            "  C3: init_gmode_eigenmode ω² = %.14e  (analytic = %.14e, rel = %.3e)\n",
            om2_from_init, om2_exact, rel);
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "C3: init_gmode_eigenmode ω² rel err = %.3e (< 1e-6 on default v-space path)",
            rel);
        CHECK_TRUE(rel < 1e-6, msg);
    }

    // ── C4. Galerkin v-space EVP convergence.  Re-run the same 3-grid
    //       sweep on the default path (`compute_2d_gmode_evp`).  This is
    //       what `init_gmode_eigenmode` uses by default, so a regression
    //       here would poison every nonlinear triad paper-run.
    std::fprintf(stderr, "  [path = galerkin_v]\n");
    double err_v[3] = {0, 0, 0};
    for (int r = 0; r < 3; ++r) {
        int ny = NY_LIST[r];
        double om = run_evp_probe(ny, kx_int, n_g, Lx, Ly, N2_val,
                                  EvpPath::GALERKIN_V);
        err_v[r] = std::fabs(om - om_exact);
        std::fprintf(stderr,
            "  ny=%3d  ω_num = %.14e  |err| = %.3e  rel = %.3e\n",
            ny, om, err_v[r], err_v[r] / om_exact);
    }
    {
        // Galerkin v-space on Boussinesq/stratified_n2: since ρ₀ is
        // uniform, v-space operator degenerates to a standard Helmholtz
        // eigenproblem — SL basis is spectral.  Same halving bound as C2.
        char msg[256];
        const double r1 = (err_v[0] > 0) ? (err_v[1] / err_v[0]) : 0.0;
        std::snprintf(msg, sizeof(msg),
            "C4a: v-space err(48)/err(32) = %.3e (< 0.5)", r1);
        CHECK_TRUE(err_v[1] < 1e-12 || r1 < 0.5, msg);
        const double r2 = (err_v[1] > 0) ? (err_v[2] / err_v[1]) : 0.0;
        std::snprintf(msg, sizeof(msg),
            "C4b: v-space err(64)/err(48) = %.3e (< 0.5)", r2);
        CHECK_TRUE(err_v[2] < 1e-12 || r2 < 0.5, msg);
    }

    std::fprintf(stderr,
        "=== %d/%d tests passed ===\n",
        g_tests - g_failures, g_tests);
    return g_failures > 0 ? 1 : 0;
}
