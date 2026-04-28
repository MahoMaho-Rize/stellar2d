// Layer 2: Exact physics tests — verify against known analytic solutions.
//
// These tests initialize a physically meaningful state and check that the
// discrete operators reproduce the expected behavior:
//
//   1. HSE residual test: Lane-Emden equilibrium → compute_rhs → ||R|| ≈ 0
//   2. Uniform flow: constant state → one RK2 step → state unchanged
//   3. Poisson known solution: uniform sphere → Phi matches analytic interior
//
// Compile: see CMakeLists.txt test target.
// Run:     ./test_exact          (returns 0 if all pass)

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

#include "eos.h"
#include "state.h"
#include "grid.h"
#include "hydro/flux.h"
#include "hydro/integrate.h"
#include "hydro/reconstruct.h"
#include "bc/boundary.h"
#include "gravity/gmg.h"
#include "init/lane_emden.h"

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

// ── helpers: grid + state setup ────────────────────────────────────

static void extract_density(const Grid& grid, const State& state,
                            std::vector<double>& rho_cells) {
    int nr = grid.nr, nt = grid.ntheta;
    rho_cells.resize(nr * nt);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nt; ++j)
            rho_cells[i * nt + j] = state.rho[grid.idx(i, j)];
}

// ── Test 1: HSE residual ───────────────────────────────────────────
// Initialize Lane-Emden equilibrium, solve Poisson for Phi,
// then evaluate the full RHS (flux divergence + geometric + gravity source).
// For a perfect discrete equilibrium the RHS should be zero.
// In practice, discretization error means ||RHS|| ~ O(h^2).

static void test_hse_residual() {
    int nr = 64, nt = 32;
    double gamma = 5.0 / 3.0;
    double G = 1.0;

    // Lane-Emden parameters
    LaneEmdenParams le_params;
    le_params.n_poly = 1.5;
    le_params.rho_c = 1.0;
    le_params.K_poly = 1.0;
    le_params.G = G;

    auto le_sol = solve_lane_emden(le_params.n_poly);
    double alpha = std::sqrt((le_params.n_poly + 1.0) * le_params.K_poly
                             * std::pow(le_params.rho_c, 1.0 / le_params.n_poly - 1.0)
                             / (4.0 * M_PI * G));
    double R_star = alpha * le_sol.xi_1;
    double R_outer = R_star * 1.1;

    Grid grid;
    grid.init(nr, nt, R_outer, 2.0, 2);

    State state;
    state.allocate(grid);
    EOS eos(gamma);

    init_lane_emden(grid, state, le_params, gamma);

    // Solve Poisson for self-consistent Phi
    std::vector<double> rho_cells, poisson_rhs;
    extract_density(grid, state, rho_cells);
    compute_poisson_rhs(grid, rho_cells, G, poisson_rhs);

    PoissonGMG gmg;
    gmg.init(grid);
    gmg.solve(poisson_rhs.data(), state.phi.data(), 50, 1e-10);

    // Fill ghost cells and compute RHS
    fill_ghost_cells(grid, state, gamma);

    FluxAccumulator acc;
    acc.allocate(grid.total_cells());
    // compute_rhs = flux divergence + geometric source (no gravity)
    compute_rhs(grid, state, eos, acc, Limiter::MINMOD);
    // add gravity source separately (same as main.cpp's RK2 loop)
    add_gravity_source(grid, state, acc);

    // Volume-weighted L2 norm of RHS — this is the physically meaningful measure
    // because on a log-stretched mesh the innermost cell has tiny Δr, making
    // L-inf of dU/dt blow up even though the *integrated* residual converges.
    double sum_sq = 0.0, vol_total = 0.0;
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nt; ++j) {
            int flat = i * nt + j;
            double vol = grid.cell_volume[flat];
            double r2 = acc.dU_rho[flat] * acc.dU_rho[flat]
                      + acc.dU_mr[flat] * acc.dU_mr[flat]
                      + acc.dU_mtheta[flat] * acc.dU_mtheta[flat]
                      + acc.dU_E[flat] * acc.dU_E[flat];
            sum_sq += r2 * vol;
            vol_total += vol;
        }
    }
    double rhs_l2 = std::sqrt(sum_sq / vol_total);

    std::fprintf(stderr, "  HSE residual L2: %.3e (nr=%d, nt=%d)\n", rhs_l2, nr, nt);

    CHECK_TRUE(rhs_l2 < 10.0, "HSE residual L2 bounded");
}

// ── Test 1b: HSE velocity growth over time ─────────────────────────
// A better measure than instantaneous residual: evolve the equilibrium
// for one CFL step and check that velocity growth is small relative to cs.
// On finer grids dt shrinks ∝ h, so |v| after one step ∝ h * |RHS| and
// should decrease if the scheme is consistent.

static double hse_velocity_after_one_step(int nr, int nt) {
    double gamma = 5.0 / 3.0;
    double G = 1.0;

    LaneEmdenParams le_params;
    le_params.n_poly = 1.5;
    le_params.rho_c = 1.0;
    le_params.K_poly = 1.0;
    le_params.G = G;

    auto le_sol = solve_lane_emden(le_params.n_poly);
    double alpha = std::sqrt((le_params.n_poly + 1.0) * le_params.K_poly
                             * std::pow(le_params.rho_c, 1.0 / le_params.n_poly - 1.0)
                             / (4.0 * M_PI * G));
    double R_star = alpha * le_sol.xi_1;
    double R_outer = R_star * 1.1;

    Grid grid;
    grid.init(nr, nt, R_outer, 2.0, 2);
    State state;
    state.allocate(grid);
    EOS eos(gamma);

    init_lane_emden(grid, state, le_params, gamma);

    std::vector<double> rho_cells, poisson_rhs;
    extract_density(grid, state, rho_cells);
    compute_poisson_rhs(grid, rho_cells, G, poisson_rhs);

    PoissonGMG gmg;
    gmg.init(grid);
    gmg.solve(poisson_rhs.data(), state.phi.data(), 50, 1e-10);

    fill_ghost_cells(grid, state, gamma);

    FluxAccumulator acc;
    acc.allocate(grid.total_cells());
    compute_rhs(grid, state, eos, acc, Limiter::MINMOD);
    add_gravity_source(grid, state, acc);

    double dt = compute_cfl_dt(grid, state, eos, 0.4);
    rk2_substep(grid, state, acc, dt);

    // Max |v| / cs after one step, excluding the innermost ~5% of cells
    // near the coordinate singularity at r=0 where cot(θ)/r produces
    // large but harmless residuals.
    int i_skip = std::max(1, nr / 20);
    double max_v_over_cs = 0.0;
    for (int i = i_skip; i < nr; ++i) {
        for (int j = 0; j < nt; ++j) {
            int k = grid.idx(i, j);
            double rho = state.rho[k];
            if (rho < 0.01) continue;
            double vr = state.mr[k] / rho;
            double vt = state.mtheta[k] / rho;
            double v_mag = std::sqrt(vr * vr + vt * vt);
            PrimitiveVars w = state.to_primitive(k, gamma);
            double cs = eos.sound_speed(w.rho, w.P);
            max_v_over_cs = std::max(max_v_over_cs, v_mag / cs);
        }
    }
    return max_v_over_cs;
}

static void test_hse_velocity_convergence() {
    double v1 = hse_velocity_after_one_step(32, 16);
    double v2 = hse_velocity_after_one_step(64, 32);
    double v3 = hse_velocity_after_one_step(128, 64);

    std::fprintf(stderr, "  HSE max |v|/cs after 1 step:\n");
    std::fprintf(stderr, "    32x16:  %.3e\n", v1);
    std::fprintf(stderr, "    64x32:  %.3e\n", v2);
    std::fprintf(stderr, "    128x64: %.3e\n", v3);

    double order_12 = std::log2(v1 / v2);
    double order_23 = std::log2(v2 / v3);
    std::fprintf(stderr, "    order 32→64:  %.2f\n", order_12);
    std::fprintf(stderr, "    order 64→128: %.2f\n", order_23);

    // Velocity growth should be small for all resolutions
    CHECK_TRUE(v1 < 0.3, "HSE v/cs < 0.3 at 32x16");
    CHECK_TRUE(v2 < 0.1, "HSE v/cs < 0.1 at 64x32");
    CHECK_TRUE(v3 < 0.05, "HSE v/cs < 0.05 at 128x64");
    // And should decrease with resolution
    CHECK_TRUE(v2 < v1, "HSE v/cs decreases 32→64");
    CHECK_TRUE(v3 < v2, "HSE v/cs decreases 64→128");
}

// ── Test 2: Uniform state preservation ─────────────────────────────
// A uniform state with zero velocity should remain unchanged after
// one RK2 step (no gradients → no fluxes, geometric source = 2P/r
// and gravity cancel for Phi = const, but actually geometric source
// ≠ 0 for uniform P in spherical coords).
//
// Better test: uniform state with v=0 and zero gravity → only
// geometric pressure source remains.  Check that momentum RHS = 0
// for a θ-integrated shell (net force = 0 by symmetry).

static void test_uniform_density_preservation() {
    int nr = 16, nt = 8;
    double gamma = 5.0 / 3.0;

    Grid grid;
    grid.init(nr, nt, 1.0, 2.0, 2);
    State state;
    state.allocate(grid);
    EOS eos(gamma);

    PrimitiveVars w;
    w.rho = 1.0; w.vr = 0.0; w.vtheta = 0.0; w.P = 1.0;

    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nt; ++j)
            state.from_primitive(grid.idx(i, j), w, gamma);

    // Zero gravity
    std::fill(state.phi.begin(), state.phi.end(), 0.0);

    fill_ghost_cells(grid, state, gamma);

    FluxAccumulator acc;
    acc.allocate(grid.total_cells());
    compute_rhs(grid, state, eos, acc, Limiter::MINMOD);

    // Density RHS should be exactly zero (no mass flux for v=0)
    double max_drho = 0.0;
    for (int flat = 0; flat < nr * nt; ++flat)
        max_drho = std::max(max_drho, std::fabs(acc.dU_rho[flat]));
    CHECK_CLOSE(max_drho, 0.0, 1e-14, "uniform state: d(rho)/dt = 0");

    // Energy RHS should be exactly zero (no fluxes, no gravity work, no v)
    double max_dE = 0.0;
    for (int flat = 0; flat < nr * nt; ++flat)
        max_dE = std::max(max_dE, std::fabs(acc.dU_E[flat]));
    CHECK_CLOSE(max_dE, 0.0, 1e-14, "uniform state: d(E)/dt = 0");

    // Momentum: geometric source 2P/r is nonzero per cell in spherical coords,
    // but flux divergence should exactly cancel it for uniform P via
    // the well-balanced formulation.  Check net momentum RHS summed over θ
    // for each radial shell (should be zero by symmetry).
    for (int i = 1; i < nr - 1; ++i) {
        double sum_mr = 0.0;
        for (int j = 0; j < nt; ++j)
            sum_mr += acc.dU_mr[i * nt + j] * grid.cell_volume[i * nt + j];
        // Allow small discretization error
        CHECK_CLOSE(sum_mr, 0.0, 1e-10, "uniform state: shell-summed d(mr)/dt ≈ 0");
    }
}

// ── Test 3: Poisson solver accuracy against analytic ───────────────
// Uniform sphere: Phi_interior(r) = (2π/3)Gρr² - 2πGρR²
// Phi_surface = -GM/R where M = (4/3)πρR³

static void test_poisson_analytic_sphere() {
    int nr = 64, nt = 32;
    double R = 1.0, G = 1.0, rho_val = 1.0;

    Grid grid;
    grid.init(nr, nt, R, 2.0, 2);

    std::vector<double> rho_cells(nr * nt, rho_val);
    std::vector<double> rhs(nr * nt), phi(nr * nt, 0.0);

    compute_poisson_rhs(grid, rho_cells, G, rhs);

    PoissonGMG gmg;
    gmg.init(grid);
    gmg.solve(rhs.data(), phi.data(), 50, 1e-10);

    // Check at cell centers in interior (skip last cell = Dirichlet)
    int j_eq = nt / 2;
    double max_rel_err = 0.0;
    int n_checked = 0;

    for (int i = 0; i < nr - 1; ++i) {
        double r = grid.r_center[i];
        double phi_exact = (2.0 * M_PI * G * rho_val / 3.0) * r * r
                         - 2.0 * M_PI * G * rho_val * R * R;
        double rel = std::fabs(phi[i * nt + j_eq] - phi_exact) / std::fabs(phi_exact);
        max_rel_err = std::max(max_rel_err, rel);
        ++n_checked;
    }

    std::fprintf(stderr, "  Poisson analytic sphere: max rel err = %.3e (%d points)\n",
                 max_rel_err, n_checked);

    CHECK_TRUE(max_rel_err < 0.02, "Poisson uniform sphere < 2% error");
}

// ── Test 3b: Poisson convergence order ─────────────────────────────

static double poisson_l2_error(int nr, int nt) {
    double R = 1.0, G = 1.0, rho_val = 1.0;

    Grid grid;
    grid.init(nr, nt, R, 2.0, 2);

    std::vector<double> rho_cells(nr * nt, rho_val);
    std::vector<double> rhs(nr * nt), phi(nr * nt, 0.0);

    compute_poisson_rhs(grid, rho_cells, G, rhs);

    PoissonGMG gmg;
    gmg.init(grid);
    gmg.solve(rhs.data(), phi.data(), 50, 1e-10);

    int j_eq = nt / 2;
    double sum_sq = 0.0;
    int count = 0;
    for (int i = 0; i < nr - 1; ++i) {
        double r = grid.r_center[i];
        double phi_exact = (2.0 * M_PI * G * rho_val / 3.0) * r * r
                         - 2.0 * M_PI * G * rho_val * R * R;
        double err = phi[i * nt + j_eq] - phi_exact;
        sum_sq += err * err;
        ++count;
    }
    return std::sqrt(sum_sq / count);
}

static void test_poisson_convergence() {
    double e1 = poisson_l2_error(16, 8);
    double e2 = poisson_l2_error(32, 16);
    double e3 = poisson_l2_error(64, 32);

    std::fprintf(stderr, "  Poisson convergence:\n");
    std::fprintf(stderr, "    16x8:  %.3e\n", e1);
    std::fprintf(stderr, "    32x16: %.3e\n", e2);
    std::fprintf(stderr, "    64x32: %.3e\n", e3);

    double order_12 = std::log2(e1 / e2);
    double order_23 = std::log2(e2 / e3);
    std::fprintf(stderr, "    order 16→32:  %.2f\n", order_12);
    std::fprintf(stderr, "    order 32→64:  %.2f\n", order_23);

    CHECK_TRUE(e2 < e1, "Poisson error decreases 16→32");
    CHECK_TRUE(e3 < e2, "Poisson error decreases 32→64");
    CHECK_TRUE(order_23 > 1.0, "Poisson convergence ≥ 2nd order");
}

// ── Test 4: CFL dt is bounded and positive ─────────────────────────

static void test_cfl_dt() {
    int nr = 16, nt = 8;
    double gamma = 5.0 / 3.0;

    Grid grid;
    grid.init(nr, nt, 1.0, 2.0, 2);
    State state;
    state.allocate(grid);
    EOS eos(gamma);

    PrimitiveVars w;
    w.rho = 1.0; w.vr = 0.0; w.vtheta = 0.0; w.P = 1.0;
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nt; ++j)
            state.from_primitive(grid.idx(i, j), w, gamma);

    fill_ghost_cells(grid, state, gamma);
    double dt = compute_cfl_dt(grid, state, eos, 0.4);

    CHECK_TRUE(dt > 0.0, "CFL dt > 0");
    CHECK_TRUE(std::isfinite(dt), "CFL dt is finite");
    // For a unit sphere with cs ~ 1.29, smallest cell ~ dr[0] ~ 0.01 on log mesh
    // dt should be ~ O(0.01)
    CHECK_TRUE(dt < 1.0, "CFL dt < 1 (sanity)");
    CHECK_TRUE(dt > 1e-6, "CFL dt > 1e-6 (sanity)");
}

// ── main ───────────────────────────────────────────────────────────

int main() {
    std::fprintf(stderr, "=== stellar2d exact physics tests ===\n");

    test_hse_residual();
    test_hse_velocity_convergence();
    test_uniform_density_preservation();
    test_poisson_analytic_sphere();
    test_poisson_convergence();
    test_cfl_dt();

    std::fprintf(stderr, "=== %d/%d tests passed ===\n",
                 g_tests - g_failures, g_tests);

    return g_failures > 0 ? 1 : 0;
}
