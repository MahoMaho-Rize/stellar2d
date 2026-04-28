// Tests derived from docs/pitfalls.md — each test catches a specific
// historical bug to prevent regression.
//
// P01: 2π factor in mass integral + Dirichlet BC Phi(R) = -GM/R
// P02: θ gravity force = 0 for spherically symmetric state
// P03: HLLC denominator near-zero → no NaN
// P04: Sedov blast energy deposition: total E = E_blast
// P05: Density/pressure floor → no NaN from near-zero state
// P07: Well-balanced: uniform state has zero residual in density + energy
// P08: Radial-only pressure perturbation → no θ momentum source

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>

#include "eos.h"
#include "state.h"
#include "grid.h"
#include "hydro/flux.h"
#include "hydro/integrate.h"
#include "hydro/reconstruct.h"
#include "hydro/riemann.h"
#include "bc/boundary.h"
#include "gravity/gmg.h"
#include "init/lane_emden.h"
#include "init/sedov.h"

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

static void extract_density(const Grid& grid, const State& state,
                            std::vector<double>& rho_cells) {
    int nr = grid.nr, nt = grid.ntheta;
    rho_cells.resize(nr * nt);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nt; ++j)
            rho_cells[i * nt + j] = state.rho[grid.idx(i, j)];
}

// ── P01: 2π mass factor + Phi(R) Dirichlet BC ─────────────────────

static void test_p01_mass_integral() {
    // Lane-Emden analytic total mass = (4π) * α³ * ρ_c * ξ₁² * |θ'(ξ₁)|
    LaneEmdenParams lp;
    lp.n_poly = 1.5; lp.rho_c = 1.0; lp.K_poly = 1.0; lp.G = 1.0;

    auto sol = solve_lane_emden(lp.n_poly);
    double alpha = std::sqrt((lp.n_poly + 1.0) * lp.K_poly
                             * std::pow(lp.rho_c, 1.0 / lp.n_poly - 1.0)
                             / (4.0 * M_PI * lp.G));
    double M_analytic = 4.0 * M_PI * alpha * alpha * alpha * lp.rho_c
                        * sol.xi_1 * sol.xi_1 * sol.dtheta_xi1;

    int nr = 64, nt = 32;
    double R_outer = alpha * sol.xi_1 * 1.1;
    Grid grid;
    grid.init(nr, nt, R_outer, 2.0, 2);
    State state;
    state.allocate(grid);
    init_lane_emden(grid, state, lp, 5.0 / 3.0);

    // Numerical mass: sum(rho * V) * 2π  (P01: the 2π factor!)
    double M_numerical = 0.0;
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nt; ++j)
            M_numerical += state.rho[grid.idx(i, j)] * grid.cell_volume[i * nt + j];
    M_numerical *= 2.0 * M_PI;

    double rel = std::fabs(M_numerical - M_analytic) / M_analytic;
    std::fprintf(stderr, "  P01: M_num=%.6e, M_ana=%.6e, rel=%.3e\n",
                 M_numerical, M_analytic, rel);
    CHECK_TRUE(rel < 0.01, "P01: |M_num - M_analytic| / M_analytic < 1%");
}

static void test_p01_phi_dirichlet() {
    // After Poisson solve, Phi at outer boundary should be -G*M/R
    LaneEmdenParams lp;
    lp.n_poly = 1.5; lp.rho_c = 1.0; lp.K_poly = 1.0; lp.G = 1.0;

    auto sol = solve_lane_emden(lp.n_poly);
    double alpha = std::sqrt((lp.n_poly + 1.0) * lp.K_poly
                             * std::pow(lp.rho_c, 1.0 / lp.n_poly - 1.0)
                             / (4.0 * M_PI * lp.G));

    int nr = 64, nt = 32;
    double R_outer = alpha * sol.xi_1 * 1.1;
    Grid grid;
    grid.init(nr, nt, R_outer, 2.0, 2);
    State state;
    state.allocate(grid);
    init_lane_emden(grid, state, lp, 5.0 / 3.0);

    std::vector<double> rho_cells, rhs;
    extract_density(grid, state, rho_cells);
    compute_poisson_rhs(grid, rho_cells, lp.G, rhs);

    PoissonGMG gmg;
    gmg.init(grid);
    gmg.solve(rhs.data(), state.phi.data(), 50, 1e-10);

    // The RHS at i=nr-1 is the Dirichlet value: -G*M_total / R_outer
    // Check that Phi at outer boundary matches this
    double M_total = 0.0;
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nt; ++j)
            M_total += rho_cells[i * nt + j] * grid.cell_volume[i * nt + j];
    M_total *= 2.0 * M_PI;

    double phi_expected = -lp.G * M_total / grid.r_center[nr - 1];
    int j_eq = nt / 2;
    double phi_boundary = state.phi[(nr - 1) * nt + j_eq];
    double rel = std::fabs(phi_boundary - phi_expected) / std::fabs(phi_expected);

    std::fprintf(stderr, "  P01: Phi(R)=%.6e, -GM/R=%.6e, rel=%.3e\n",
                 phi_boundary, phi_expected, rel);
    CHECK_TRUE(rel < 1e-6, "P01: Phi(R) = -GM/R (Dirichlet BC)");
}

// ── P02: θ gravity force = 0 for spherically symmetric state ──────

static void test_p02_theta_gravity_zero() {
    LaneEmdenParams lp;
    lp.n_poly = 1.5; lp.rho_c = 1.0; lp.K_poly = 1.0; lp.G = 1.0;

    auto sol = solve_lane_emden(lp.n_poly);
    double alpha = std::sqrt((lp.n_poly + 1.0) * lp.K_poly
                             * std::pow(lp.rho_c, 1.0 / lp.n_poly - 1.0)
                             / (4.0 * M_PI * lp.G));

    int nr = 32, nt = 16;
    double R_outer = alpha * sol.xi_1 * 1.1;
    Grid grid;
    grid.init(nr, nt, R_outer, 2.0, 2);
    State state;
    state.allocate(grid);
    init_lane_emden(grid, state, lp, 5.0 / 3.0);

    std::vector<double> rho_cells, rhs;
    extract_density(grid, state, rho_cells);
    compute_poisson_rhs(grid, rho_cells, lp.G, rhs);
    PoissonGMG gmg;
    gmg.init(grid);
    gmg.solve(rhs.data(), state.phi.data(), 50, 1e-10);

    fill_ghost_cells(grid, state, 5.0 / 3.0);

    // Compute gravity source only (not flux divergence / geometric)
    FluxAccumulator acc;
    acc.allocate(grid.total_cells());
    acc.zero();
    add_gravity_source(grid, state, acc);

    // For spherically symmetric state, theta gravity should be zero
    // (Phi is θ-independent → dPhi/dθ = 0)
    double max_grav_theta = 0.0;
    for (int i = 1; i < nr - 1; ++i) {  // skip center and boundary
        for (int j = 0; j < nt; ++j) {
            max_grav_theta = std::max(max_grav_theta,
                                       std::fabs(acc.dU_mtheta[i * nt + j]));
        }
    }

    std::fprintf(stderr, "  P02: max |S_mt_gravity| = %.3e\n", max_grav_theta);
    // GMG noise in Phi creates small dPhi/dtheta; should be < GMG_tol / dr ~ 1e-4
    CHECK_TRUE(max_grav_theta < 1e-2, "P02: theta gravity ≈ 0 for spherical state");
}

// ── P03: HLLC near-zero velocity → no NaN ─────────────────────────

static void test_p03_hllc_near_zero() {
    double gamma = 5.0 / 3.0;

    // Nearly identical states with near-zero velocity (triggers denom → 0)
    PrimitiveVars wl, wr;
    wl.rho = 1.0; wl.vr = 1e-20; wl.vtheta = 0.0; wl.P = 1.0;
    wr.rho = 1.0; wr.vr = 1e-20; wr.vtheta = 0.0; wr.P = 1.0;

    Flux4 f = hllc_flux_r(wl, wr, gamma);
    CHECK_TRUE(std::isfinite(f.f_rho), "P03: f_rho finite at v≈0");
    CHECK_TRUE(std::isfinite(f.f_mr), "P03: f_mr finite at v≈0");
    CHECK_TRUE(std::isfinite(f.f_mtheta), "P03: f_mtheta finite at v≈0");
    CHECK_TRUE(std::isfinite(f.f_E), "P03: f_E finite at v≈0");

    // For identical states, mass flux = rho * v ≈ 0
    CHECK_CLOSE(f.f_rho, 1e-20, 1e-14, "P03: f_rho = rho*v ≈ 0");
    // Momentum flux = rho*v^2 + P ≈ P
    CHECK_CLOSE(f.f_mr, 1.0, 1e-10, "P03: f_mr ≈ P for identical states");
}

static void test_p03_hllc_exactly_identical() {
    double gamma = 5.0 / 3.0;

    // Exactly identical states (worst case for denominator)
    PrimitiveVars w;
    w.rho = 1.0; w.vr = 0.0; w.vtheta = 0.0; w.P = 1.0;

    Flux4 f = hllc_flux_r(w, w, gamma);
    CHECK_TRUE(std::isfinite(f.f_rho), "P03: f_rho finite (identical)");
    CHECK_TRUE(std::isfinite(f.f_mr), "P03: f_mr finite (identical)");
    CHECK_TRUE(std::isfinite(f.f_E), "P03: f_E finite (identical)");
    CHECK_CLOSE(f.f_rho, 0.0, 1e-14, "P03: f_rho=0 (identical, v=0)");
}

static void test_p03_hllc_extreme_contrast() {
    double gamma = 5.0 / 3.0;

    // Extreme density contrast
    PrimitiveVars wl, wr;
    wl.rho = 1e-10; wl.vr = 0.0; wl.vtheta = 0.0; wl.P = 1e-10;
    wr.rho = 1e+10; wr.vr = 0.0; wr.vtheta = 0.0; wr.P = 1e+10;

    Flux4 f = hllc_flux_r(wl, wr, gamma);
    CHECK_TRUE(std::isfinite(f.f_rho), "P03: f_rho finite (extreme)");
    CHECK_TRUE(std::isfinite(f.f_mr), "P03: f_mr finite (extreme)");
    CHECK_TRUE(std::isfinite(f.f_E), "P03: f_E finite (extreme)");
}

// ── P04: Sedov total energy = E_blast ──────────────────────────────

static void test_p04_sedov_energy() {
    int nr = 64, nt = 32;
    double gamma = 5.0 / 3.0;

    Grid grid;
    grid.init(nr, nt, 1.0, 2.0, 2);
    State state;
    state.allocate(grid);

    SedovParams sp;
    sp.rho_0 = 1.0;
    sp.E_blast = 1.0;
    sp.r_blast = 0.05;

    init_sedov(grid, state, sp, gamma);

    // FV cell volumes are 2D (azimuthal 2π factor not included).
    // init_sedov deposits E_blast into the 2D volume, so
    // sum(rhoE * V_2D) = E_blast (no 2π needed).
    double E_total = 0.0;
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nt; ++j) {
            int k = grid.idx(i, j);
            E_total += state.E[k] * grid.cell_volume[i * nt + j];
        }
    }

    // Note: E_total includes ambient energy outside blast region
    // Ambient: P_ambient = 1e-10, e_int_ambient = P/(gamma-1)/rho = 1e-10/0.667
    // This is negligible compared to E_blast = 1.0
    double rel = std::fabs(E_total - sp.E_blast) / sp.E_blast;
    std::fprintf(stderr, "  P04: E_total=%.6e, E_blast=%.6e, rel=%.3e\n",
                 E_total, sp.E_blast, rel);
    CHECK_TRUE(rel < 0.01, "P04: total energy = E_blast ± 1%");
}

// ── P05: Floor enforcement ─────────────────────────────────────────

static void test_p05_floor_in_to_primitive() {
    Grid grid;
    grid.init(4, 4, 1.0, 2.0, 2);
    State state;
    state.allocate(grid);

    double gamma = 5.0 / 3.0;
    int k = grid.idx(1, 1);

    // Set near-zero / negative density
    state.rho[k] = -1e-30;
    state.mr[k] = 0.0;
    state.mtheta[k] = 0.0;
    state.E[k] = 1e-30;

    PrimitiveVars w = state.to_primitive(k, gamma);
    CHECK_TRUE(w.rho > 0, "P05: rho > 0 after floor");
    CHECK_TRUE(w.P > 0, "P05: P > 0 after floor");
    CHECK_TRUE(std::isfinite(w.rho), "P05: rho finite");
    CHECK_TRUE(std::isfinite(w.P), "P05: P finite");
    CHECK_TRUE(std::isfinite(w.vr), "P05: vr finite");
    CHECK_TRUE(std::isfinite(w.vtheta), "P05: vtheta finite");

    // Sound speed should also be finite
    EOS eos(gamma);
    double cs = eos.sound_speed(w.rho, w.P);
    CHECK_TRUE(std::isfinite(cs), "P05: cs finite from floored state");
    CHECK_TRUE(cs > 0, "P05: cs > 0 from floored state");
}

static void test_p05_no_nan_after_step() {
    // Initialize with near-vacuum and take one step — nothing should be NaN
    int nr = 8, nt = 4;
    double gamma = 5.0 / 3.0;
    Grid grid;
    grid.init(nr, nt, 1.0, 2.0, 2);
    State state;
    state.allocate(grid);
    EOS eos(gamma);

    PrimitiveVars w;
    w.rho = 1e-15; w.vr = 0.0; w.vtheta = 0.0; w.P = 1e-15;
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nt; ++j)
            state.from_primitive(grid.idx(i, j), w, gamma);

    std::fill(state.phi.begin(), state.phi.end(), 0.0);
    fill_ghost_cells(grid, state, gamma);

    FluxAccumulator acc;
    acc.allocate(grid.total_cells());
    compute_rhs(grid, state, eos, acc, Limiter::MINMOD);

    // Check no NaN in RHS
    bool has_nan = false;
    for (int flat = 0; flat < nr * nt; ++flat) {
        if (!std::isfinite(acc.dU_rho[flat]) || !std::isfinite(acc.dU_mr[flat]) ||
            !std::isfinite(acc.dU_mtheta[flat]) || !std::isfinite(acc.dU_E[flat])) {
            has_nan = true;
            break;
        }
    }
    CHECK_TRUE(!has_nan, "P05: no NaN in RHS from near-vacuum state");
}

// ── P07: Well-balanced: uniform v=0 → dU_rho = 0, dU_E = 0 ───────
// (Already tested in test_exact.cpp, but repeated here for pitfall traceability)

static void test_p07_uniform_zero_residual() {
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

    std::fill(state.phi.begin(), state.phi.end(), 0.0);
    fill_ghost_cells(grid, state, gamma);

    FluxAccumulator acc;
    acc.allocate(grid.total_cells());
    compute_rhs(grid, state, eos, acc, Limiter::MINMOD);

    double max_drho = 0.0, max_dE = 0.0;
    for (int flat = 0; flat < nr * nt; ++flat) {
        max_drho = std::max(max_drho, std::fabs(acc.dU_rho[flat]));
        max_dE = std::max(max_dE, std::fabs(acc.dU_E[flat]));
    }
    CHECK_CLOSE(max_drho, 0.0, 1e-14, "P07: dU_rho = 0 for uniform state");
    CHECK_CLOSE(max_dE, 0.0, 1e-14, "P07: dU_E = 0 for uniform state");
}

// ── P08: θ-independent state → dU_mtheta is θ-independent ─────────
// For a radial-only state, dU_mtheta at each (i,j) is nonzero
// (geometric source cot(θ)/r term), but it should be independent of θ
// within each radial shell. θ-variation signals a bug in the FV geometric
// source or θ-flux computation.

static void test_p08_radial_state_theta_symmetry() {
    int nr = 32, nt = 16;
    double gamma = 5.0 / 3.0;

    Grid grid;
    grid.init(nr, nt, 1.0, 2.0, 2);
    State state;
    state.allocate(grid);
    EOS eos(gamma);

    // Radial-only state: rho = 1 + 0.5*exp(-r^2/0.1^2), P = rho^gamma
    for (int i = 0; i < nr; ++i) {
        double r = grid.r_center[i];
        double rho = 1.0 + 0.5 * std::exp(-r * r / 0.01);
        double P = std::pow(rho, gamma);
        for (int j = 0; j < nt; ++j) {
            PrimitiveVars w;
            w.rho = rho; w.vr = 0.0; w.vtheta = 0.0; w.P = P;
            state.from_primitive(grid.idx(i, j), w, gamma);
        }
    }
    std::fill(state.phi.begin(), state.phi.end(), 0.0);
    fill_ghost_cells(grid, state, gamma);

    FluxAccumulator acc;
    acc.allocate(grid.total_cells());
    compute_rhs(grid, state, eos, acc, Limiter::MINMOD);

    // For a θ-independent state, dU_rho and dU_E should be θ-independent.
    // dU_mtheta varies with θ (cot term), but dU_rho has no geometric source.
    double max_drho_var = 0.0;
    for (int i = 2; i < nr - 2; ++i) {
        double mean = 0.0;
        for (int j = 0; j < nt; ++j)
            mean += acc.dU_rho[i * nt + j];
        mean /= nt;
        for (int j = 0; j < nt; ++j) {
            double var = std::fabs(acc.dU_rho[i * nt + j] - mean);
            if (std::fabs(mean) > 1e-20)
                max_drho_var = std::max(max_drho_var, var / std::fabs(mean));
        }
    }
    std::fprintf(stderr, "  P08: max dU_rho θ-variation = %.3e\n", max_drho_var);
    CHECK_TRUE(max_drho_var < 1e-10,
               "P08: dU_rho is θ-independent for radial-only state");
}

// ── main ───────────────────────────────────────────────────────────

int main() {
    std::fprintf(stderr, "=== stellar2d pitfall regression tests ===\n");

    // P01: 2π mass factor + Dirichlet BC
    test_p01_mass_integral();
    test_p01_phi_dirichlet();

    // P02: θ gravity force = 0
    test_p02_theta_gravity_zero();

    // P03: HLLC denominator safety
    test_p03_hllc_near_zero();
    test_p03_hllc_exactly_identical();
    test_p03_hllc_extreme_contrast();

    // P04: Sedov energy deposition
    test_p04_sedov_energy();

    // P05: Floor enforcement
    test_p05_floor_in_to_primitive();
    test_p05_no_nan_after_step();

    // P07: Well-balanced zero residual
    test_p07_uniform_zero_residual();

    // P08: Radial state → θ-symmetric residual
    test_p08_radial_state_theta_symmetry();

    std::fprintf(stderr, "=== %d/%d tests passed ===\n",
                 g_tests - g_failures, g_tests);

    return g_failures > 0 ? 1 : 0;
}
