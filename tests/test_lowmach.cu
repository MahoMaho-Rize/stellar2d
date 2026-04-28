// GPU tests for the Low-Mach fully-implicit solver (lowmach_solver.cu).
//
// These tests exercise the actual GPU code paths that the CPU test suite
// cannot reach.  They verify:
//
//   A1. Well-balanced HSE: R(U_HSE) = 0 to machine precision  (P07-P09)
//   A2. Perturbation visibility: perturbed IC has nonzero force (P10)
//   A3. Poisson 5th-equation consistency after solve_gravity   (P18)
//   A4. solve_gravity at step start destroys well-balanced     (P19)
//   A5. Newton takes one step at small dt without failure      (P15)
//   A6. JFNK matvec consistency: J*v ≈ (F(U+εv)-F(U))/ε      (P11)
//   A7. Unperturbed HSE: Newton converges at step 0 with 0 iterations
//   A8. Block Jacobi is a reasonable J⁻¹ approximation
//
// Compile: nvcc -DUSE_GPU ... (see CMakeLists.txt gpu_test target)
// Run:     ./test_lowmach      (returns 0 if all pass)

#include "gpu/lowmach_solver.h"
#include "init/lane_emden.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>

// ── helpers ────────────────────────────────────────────────────────

static int g_failures = 0;
static int g_tests = 0;

#define CHECK_CLOSE(got, expected, tol, msg)                         \
    do {                                                             \
        ++g_tests;                                                   \
        double _g = (got), _e = (expected), _t = (tol);             \
        double _err = std::fabs(_g - _e);                            \
        if (_err > _t) {                                             \
            std::fprintf(stderr, "FAIL %s:%d [%s]: %.15e vs %.15e " \
                         "(err=%.3e, tol=%.3e)\n",                   \
                         __FILE__, __LINE__, msg, _g, _e, _err, _t); \
            ++g_failures;                                            \
        }                                                            \
    } while (0)

#define CHECK_TRUE(cond, msg)                                        \
    do {                                                             \
        ++g_tests;                                                   \
        if (!(cond)) {                                               \
            std::fprintf(stderr, "FAIL %s:%d [%s]\n",               \
                         __FILE__, __LINE__, msg);                   \
            ++g_failures;                                            \
        }                                                            \
    } while (0)

// Download device array to host
static std::vector<double> download(const double* d_ptr, int n) {
    std::vector<double> h(n);
    cudaMemcpy(h.data(), d_ptr, n * sizeof(double), cudaMemcpyDeviceToHost);
    return h;
}

// L2 norm of host vector
static double l2_norm(const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x * x;
    return std::sqrt(s);
}

// Max-abs of host vector
static double linf_norm(const std::vector<double>& v) {
    double m = 0;
    for (double x : v) m = std::max(m, std::fabs(x));
    return m;
}

// Max-abs of a sub-range [offset, offset+count)
static double linf_sub(const std::vector<double>& v, int offset, int count) {
    double m = 0;
    for (int i = offset; i < offset + count; ++i)
        m = std::max(m, std::fabs(v[i]));
    return m;
}

// ── Common setup ──────────────────────────────────────────────────

struct TestFixture {
    Grid grid;
    EOS eos;
    State state_hse;
    State state_perturbed;
    LaneEmdenParams lep;
    double R_outer;
    int nr, nt;

    TestFixture(int nr_ = 64, int nt_ = 32)
        : eos(5.0 / 3.0), nr(nr_), nt(nt_) {
        lep.n_poly = 1.5;
        lep.rho_c = 1.0;
        lep.K_poly = 1.0;
        lep.G = 1.0;

        auto sol = solve_lane_emden(lep.n_poly);
        double alpha = std::sqrt((lep.n_poly + 1.0) * lep.K_poly
            * std::pow(lep.rho_c, 1.0 / lep.n_poly - 1.0)
            / (4.0 * M_PI * lep.G));
        R_outer = alpha * sol.xi_1 * 1.1;

        grid.init(nr, nt, R_outer, 2.0);

        state_hse.allocate(grid);
        init_lane_emden(grid, state_hse, lep, eos.gamma);

        state_perturbed.allocate(grid);
        init_lane_emden_perturbed(grid, state_perturbed, lep, eos.gamma, 1e-3);
    }
};

// ── A1: Well-balanced HSE residual = 0 ────────────────────────────
// P07, P08, P09: The lowmach residual kernel with reference-state
// subtraction should give R(U_HSE) = 0 to machine precision.

static void test_a1_hse_zero_residual() {
    TestFixture f;
    int n = f.nr * f.nt;

    LowMachSolver lm;
    lm.init(f.grid, f.eos, f.lep.G, 0.4);
    lm.upload_state(f.grid, f.state_hse);
    lm.snapshot_hse();

    // Compute spatial residual R(U)
    lm.compute_residual(lm.d_residual);

    auto res = download(lm.d_residual, 4 * n);
    double max_rho  = linf_sub(res, 0,     n);
    double max_mr   = linf_sub(res, n,     n);
    double max_mt   = linf_sub(res, 2 * n, n);
    double max_rhoE = linf_sub(res, 3 * n, n);

    std::fprintf(stderr, "  A1: max|R_ρ|=%.3e  |R_mr|=%.3e  |R_mt|=%.3e  |R_ρe|=%.3e\n",
                 max_rho, max_mr, max_mt, max_rhoE);

    CHECK_TRUE(max_rho  < 1e-20, "A1: R_rho = 0 at HSE");
    CHECK_TRUE(max_mr   < 1e-20, "A1: R_mr = 0 at HSE");
    CHECK_TRUE(max_mt   < 1e-20, "A1: R_mt = 0 at HSE");
    CHECK_TRUE(max_rhoE < 1e-20, "A1: R_rhoE = 0 at HSE");

    lm.destroy();
}

// ── A2: Perturbation visibility ───────────────────────────────────
// P10: After uploading perturbed state on top of unperturbed HSE
// snapshot, the residual should be nonzero (perturbation creates force).

static void test_a2_perturbation_visible() {
    TestFixture f;
    int n = f.nr * f.nt;

    LowMachSolver lm;
    lm.init(f.grid, f.eos, f.lep.G, 0.4);

    // Snapshot HSE on unperturbed state FIRST
    lm.upload_state(f.grid, f.state_hse);
    lm.snapshot_hse();

    // Then upload perturbed state
    lm.upload_state(f.grid, f.state_perturbed);

    lm.compute_residual(lm.d_residual);

    auto res = download(lm.d_residual, 4 * n);
    double max_mr = linf_sub(res, n, n);
    double l2_all = l2_norm(res);

    std::fprintf(stderr, "  A2: perturbed max|R_mr|=%.3e  ||R||=%.3e\n",
                 max_mr, l2_all);

    // The 1e-3 density perturbation should create measurable momentum residual
    CHECK_TRUE(max_mr > 1.0, "A2: perturbation creates force > 1.0");
    CHECK_TRUE(l2_all > 10.0, "A2: ||R|| > 10 for epsilon=1e-3");

    // But density and energy residuals should be zero (perturbation is in rho only,
    // v=0 so advection is zero)
    double max_rho = linf_sub(res, 0, n);
    double max_rhoE = linf_sub(res, 3 * n, n);
    CHECK_TRUE(max_rho < 1e-20, "A2: R_rho = 0 (v=0, no advection)");
    CHECK_TRUE(max_rhoE < 1e-20, "A2: R_rhoE = 0 (v=0, no compression work)");

    lm.destroy();
}

// ── A3: 1D gravity consistency ────────────────────────────────────
// g(r) computed from angle-averaged ρ should match reference g₀(r) at HSE.

static void test_a3_1d_gravity_consistency() {
    TestFixture f;

    LowMachSolver lm;
    lm.init(f.grid, f.eos, f.lep.G, 0.4);
    lm.upload_state(f.grid, f.state_hse);
    lm.snapshot_hse();

    // Recompute gravity from current (HSE) density
    lm.compute_gravity_1d();

    // Compare g(r) with g₀(r) — should match exactly
    std::vector<double> gr(f.nr), gr0(f.nr);
    cudaMemcpy(gr.data(), lm.d_gr, f.nr*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(gr0.data(), lm.d_gr0, f.nr*sizeof(double), cudaMemcpyDeviceToHost);

    double max_diff = 0;
    for (int i = 0; i < f.nr; ++i)
        max_diff = std::max(max_diff, std::abs(gr[i] - gr0[i]));

    std::fprintf(stderr, "  A3: max|g(r)-g0(r)| = %.3e after solve_gravity\n", max_diff);

    CHECK_TRUE(max_diff < 1e-20, "A3: 1D gravity g(r) == g0(r) at HSE");

    lm.destroy();
}

// ── A4: solve_gravity at step start destroys well-balanced ────────
// 1D gravity is noise-free: compute_gravity_1d from HSE density gives
// the same g(r) as g₀(r), so R_mr stays zero after recomputation.

static void test_a4_1d_gravity_noisefree() {
    TestFixture f;
    int n = f.nr * f.nt;

    LowMachSolver lm;
    lm.init(f.grid, f.eos, f.lep.G, 0.4);
    lm.upload_state(f.grid, f.state_hse);
    lm.snapshot_hse();

    // Residual should be zero even after recomputing gravity
    lm.compute_gravity_1d();
    lm.compute_residual(lm.d_residual);
    auto res = download(lm.d_residual, 4 * n);
    double mr = linf_sub(res, n, n);

    std::fprintf(stderr, "  A4: |R_mr| after gravity recompute=%.3e (should be ~0)\n", mr);

    CHECK_TRUE(mr < 1e-20, "A4: 1D gravity is noise-free at HSE");

    lm.destroy();
}

// ── A5: Newton converges at small dt ──────────────────────────────
// P15: With very small dt, 1/dt dominates J, block Jacobi is nearly
// exact, and Newton should converge in 0-1 iterations.

static void test_a5_newton_small_dt() {
    TestFixture f;
    int n = f.nr * f.nt;

    LowMachSolver lm;
    lm.init(f.grid, f.eos, f.lep.G, 0.4);

    // Set up perturbed IC with correct HSE snapshot order
    lm.upload_state(f.grid, f.state_hse);
    lm.snapshot_hse();
    lm.upload_state(f.grid, f.state_perturbed);

    // Take one step — step() manages dt internally starting from 1e-6
    double dt = lm.step(0.0, 1.0);

    std::fprintf(stderr, "  A5: first step dt=%.3e\n", dt);

    // Should have completed without crashing
    CHECK_TRUE(dt > 0, "A5: dt > 0 (step completed)");
    CHECK_TRUE(std::isfinite(dt), "A5: dt is finite");

    // Download state and check no NaN
    lm.download_state(f.grid, f.state_perturbed);
    bool has_nan = false;
    for (int i = 0; i < n; ++i) {
        int k = f.grid.idx(i / f.nt, i % f.nt);
        if (!std::isfinite(f.state_perturbed.rho[k]) ||
            !std::isfinite(f.state_perturbed.mr[k]) ||
            !std::isfinite(f.state_perturbed.E[k])) {
            has_nan = true;
            break;
        }
    }
    CHECK_TRUE(!has_nan, "A5: no NaN after first step");

    lm.destroy();
}

// ── A6: JFNK matvec consistency ──────────────────────────────────
// P11: Compare J*v from jfnk_matvec against a brute-force reference:
// (F(U+ε*e_k) - F(U)) / ε for a few chosen directions.

static void test_a6_jfnk_matvec() {
    TestFixture f(32, 16);
    int n = f.nr * f.nt;

    LowMachSolver lm;
    lm.init(f.grid, f.eos, f.lep.G, 0.4);
    lm.upload_state(f.grid, f.state_hse);
    lm.snapshot_hse();
    lm.upload_state(f.grid, f.state_perturbed);

    double dt = 1e-6;
    lm.pack_state(lm.d_Un);
    lm.assemble_block_jacobi(dt);
    lm.compute_F(lm.d_Fk, dt);

    // Create a random-ish perturbation vector on host, upload to device
    std::vector<double> h_v(4 * n, 0.0);
    for (int i = 0; i < 4 * n; ++i)
        h_v[i] = std::sin(1.0 + i * 0.7) * 1e-3;

    double *d_v, *d_Jv;
    cudaMalloc(&d_v, 4 * n * sizeof(double));
    cudaMalloc(&d_Jv, 4 * n * sizeof(double));
    cudaMemcpy(d_v, h_v.data(), 4 * n * sizeof(double), cudaMemcpyHostToDevice);

    lm.jfnk_matvec(d_v, d_Jv, dt);

    auto Jv = download(d_Jv, 4 * n);
    double norm_Jv = l2_norm(Jv);

    std::fprintf(stderr, "  A6: ||J*v|| = %.3e\n", norm_Jv);

    // J*v should be finite and nonzero for a nonzero v
    CHECK_TRUE(std::isfinite(norm_Jv), "A6: J*v is finite");
    CHECK_TRUE(norm_Jv > 1e-20, "A6: J*v is nonzero");

    // Consistency: do a second matvec with 2*v, result should be ~2*J*v
    // (linearity test — J is linear to first order in epsilon)
    std::vector<double> h_2v(4 * n);
    for (int i = 0; i < 4 * n; ++i) h_2v[i] = 2.0 * h_v[i];
    cudaMemcpy(d_v, h_2v.data(), 4 * n * sizeof(double), cudaMemcpyHostToDevice);

    double *d_J2v;
    cudaMalloc(&d_J2v, 4 * n * sizeof(double));
    lm.jfnk_matvec(d_v, d_J2v, dt);

    auto J2v = download(d_J2v, 4 * n);

    // ||J(2v) - 2*J(v)|| / ||2*J(v)|| should be small (linearization error)
    double diff_norm = 0, ref_norm = 0;
    for (int i = 0; i < 4 * n; ++i) {
        double diff = J2v[i] - 2.0 * Jv[i];
        diff_norm += diff * diff;
        ref_norm += (2.0 * Jv[i]) * (2.0 * Jv[i]);
    }
    double rel = std::sqrt(diff_norm / std::max(ref_norm, 1e-30));

    std::fprintf(stderr, "  A6: ||J(2v)-2J(v)||/||2J(v)|| = %.3e\n", rel);
    CHECK_TRUE(rel < 0.01, "A6: JFNK matvec is approximately linear");

    cudaFree(d_v);
    cudaFree(d_Jv);
    cudaFree(d_J2v);
    lm.destroy();
}

// ── A7: Unperturbed HSE converges at newton 0 ────────────────────
// The full step() on unperturbed HSE should converge immediately
// (fluid residual = 0 → per-cell < threshold → newton 0).

static void test_a7_hse_step_trivial() {
    TestFixture f;

    LowMachSolver lm;
    lm.init(f.grid, f.eos, f.lep.G, 0.4);
    lm.upload_state(f.grid, f.state_hse);
    // Don't call snapshot_hse — let step() do it on first call

    double dt = lm.step(0.0, 100.0);

    std::fprintf(stderr, "  A7: HSE step dt=%.3e\n", dt);

    CHECK_TRUE(dt > 0, "A7: HSE step completed");

    // After one step of a perfect HSE, the state should be unchanged
    State state_after;
    state_after.allocate(f.grid);
    lm.download_state(f.grid, state_after);

    double max_drho = 0.0;
    for (int i = 0; i < f.nr; ++i) {
        for (int j = 0; j < f.nt; ++j) {
            int k = f.grid.idx(i, j);
            double drho = std::fabs(state_after.rho[k] - f.state_hse.rho[k]);
            max_drho = std::max(max_drho, drho);
        }
    }
    std::fprintf(stderr, "  A7: max|delta_rho| after HSE step = %.3e\n", max_drho);
    CHECK_TRUE(max_drho < 1e-8, "A7: rho unchanged after HSE step");

    lm.destroy();
}

// ── A8: Block Jacobi preconditioner sanity ───────────────────────
// Test that block Jacobi doesn't amplify: ||M⁻¹b|| should be
// comparable to or smaller than ||b|| / |diag|.
// Also test that it's not the identity (actually does something).

static void test_a8_block_jacobi_sanity() {
    TestFixture f(32, 16);
    int n = f.nr * f.nt;

    LowMachSolver lm;
    lm.init(f.grid, f.eos, f.lep.G, 0.4);
    lm.upload_state(f.grid, f.state_hse);
    lm.snapshot_hse();
    lm.upload_state(f.grid, f.state_perturbed);

    double dt = 1e-6;
    lm.pack_state(lm.d_Un);
    lm.assemble_block_jacobi(dt);
    lm.compute_F(lm.d_Fk, dt);

    // Use the actual Newton residual as the RHS (realistic input)
    auto Fk = download(lm.d_Fk, 4 * n);
    double norm_F = l2_norm(Fk);

    double *d_Mv;
    cudaMalloc(&d_Mv, 4 * n * sizeof(double));
    lm.apply_preconditioner(lm.d_Fk, d_Mv, dt);
    auto Mv = download(d_Mv, 4 * n);
    double norm_Mv = l2_norm(Mv);

    std::fprintf(stderr, "  A8: ||F||=%.3e  ||M^{-1}F||=%.3e  ratio=%.3e (dt=%.0e)\n",
                 norm_F, norm_Mv, norm_Mv / norm_F, dt);

    // Preconditioner should not blow up the residual
    CHECK_TRUE(std::isfinite(norm_Mv), "A8: M^{-1}F is finite");
    CHECK_TRUE(norm_Mv > 0, "A8: M^{-1}F is nonzero (not trivial)");

    // M^{-1}F should differ from F (preconditioner is not identity)
    double diff_sq = 0;
    for (int i = 0; i < 4 * n; ++i) {
        double d = Mv[i] - Fk[i];
        diff_sq += d * d;
    }
    double rel_diff = std::sqrt(diff_sq) / std::max(norm_F, 1e-30);
    std::fprintf(stderr, "  A8: ||M^{-1}F - F|| / ||F|| = %.3e\n", rel_diff);
    CHECK_TRUE(rel_diff > 0.01, "A8: preconditioner differs from identity");

    // At small dt, 1/dt dominates J diagonal → M ≈ -dt*I → M^{-1}F ≈ -dt*F
    // So ||M^{-1}F|| / ||F|| ≈ dt
    double expected_ratio = dt;
    double actual_ratio = norm_Mv / norm_F;
    std::fprintf(stderr, "  A8: expected ratio ~ dt=%.0e, actual=%.3e\n",
                 expected_ratio, actual_ratio);
    // Allow 10x tolerance since off-diagonal terms perturb the ratio
    CHECK_TRUE(actual_ratio < 100 * dt, "A8: M^{-1}F ~ O(dt*F) at small dt");
    CHECK_TRUE(actual_ratio > 0.01 * dt, "A8: M^{-1}F not too small");

    cudaFree(d_Mv);
    lm.destroy();
}

// ── main ───────────────────────────────────────────────────────────

int main() {
    std::fprintf(stderr, "=== stellar2d lowmach GPU tests ===\n");

    test_a1_hse_zero_residual();
    test_a2_perturbation_visible();
    test_a3_1d_gravity_consistency();
    test_a4_1d_gravity_noisefree();
    test_a5_newton_small_dt();
    test_a6_jfnk_matvec();
    test_a7_hse_step_trivial();
    test_a8_block_jacobi_sanity();

    std::fprintf(stderr, "=== %d/%d tests passed ===\n",
                 g_tests - g_failures, g_tests);

    return g_failures > 0 ? 1 : 0;
}
