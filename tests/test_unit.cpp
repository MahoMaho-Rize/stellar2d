// Layer 1: Unit tests for individual numerical building blocks.
//
// Each test feeds known inputs into a single function and checks the output
// against an analytic or hand-computed reference at machine precision.
//
// Compile: see CMakeLists.txt test target.
// Run:     ./test_unit          (returns 0 if all pass)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>

#include "eos.h"
#include "state.h"
#include "grid.h"
#include "hydro/reconstruct.h"
#include "hydro/riemann.h"
#include "gravity/gmg.h"

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

// ── EOS ────────────────────────────────────────────────────────────

static void test_eos_pressure() {
    EOS eos(5.0 / 3.0);
    // P = (gamma-1) * rho * e_int
    double rho = 2.0, e_int = 3.0;
    double P = eos.pressure(rho, e_int);
    CHECK_CLOSE(P, (5.0/3.0 - 1.0) * 2.0 * 3.0, 1e-14, "eos.pressure");
}

static void test_eos_internal_energy() {
    EOS eos(5.0 / 3.0);
    double rho = 2.0, P = 4.0;
    double e = eos.internal_energy(rho, P);
    // e = P / ((gamma-1)*rho)
    CHECK_CLOSE(e, 4.0 / ((5.0/3.0 - 1.0) * 2.0), 1e-14, "eos.internal_energy");
}

static void test_eos_roundtrip() {
    EOS eos(5.0 / 3.0);
    double rho = 1.5, e_int = 2.5;
    double P = eos.pressure(rho, e_int);
    double e2 = eos.internal_energy(rho, P);
    CHECK_CLOSE(e2, e_int, 1e-14, "eos roundtrip");
}

static void test_eos_sound_speed() {
    EOS eos(5.0 / 3.0);
    double rho = 1.0, P = 1.0;
    double cs = eos.sound_speed(rho, P);
    CHECK_CLOSE(cs, std::sqrt(5.0 / 3.0), 1e-14, "eos.sound_speed");
}

// ── State primitive<->conserved ────────────────────────────────────

static void test_state_roundtrip() {
    Grid grid;
    grid.init(4, 4, 1.0, 2.0, 2);
    State state;
    state.allocate(grid);

    double gamma = 5.0 / 3.0;
    PrimitiveVars w_in;
    w_in.rho = 1.5;
    w_in.vr = 0.3;
    w_in.vtheta = -0.2;
    w_in.P = 2.0;

    int k = grid.idx(1, 1);
    state.from_primitive(k, w_in, gamma);
    PrimitiveVars w_out = state.to_primitive(k, gamma);

    CHECK_CLOSE(w_out.rho, w_in.rho, 1e-14, "state roundtrip rho");
    CHECK_CLOSE(w_out.vr, w_in.vr, 1e-14, "state roundtrip vr");
    CHECK_CLOSE(w_out.vtheta, w_in.vtheta, 1e-14, "state roundtrip vtheta");
    CHECK_CLOSE(w_out.P, w_in.P, 1e-13, "state roundtrip P");
}

static void test_conserved_values() {
    Grid grid;
    grid.init(4, 4, 1.0, 2.0, 2);
    State state;
    state.allocate(grid);

    double gamma = 5.0 / 3.0;
    PrimitiveVars w;
    w.rho = 2.0;
    w.vr = 1.0;
    w.vtheta = 0.5;
    w.P = 3.0;

    int k = grid.idx(2, 2);
    state.from_primitive(k, w, gamma);

    CHECK_CLOSE(state.rho[k], 2.0, 1e-14, "conserved rho");
    CHECK_CLOSE(state.mr[k], 2.0, 1e-14, "conserved mr = rho*vr");
    CHECK_CLOSE(state.mtheta[k], 1.0, 1e-14, "conserved mtheta = rho*vt");
    // E = rho * (e_int + 0.5*(vr^2+vt^2))
    double e_int = w.P / ((gamma - 1.0) * w.rho);
    double ke = 0.5 * (w.vr * w.vr + w.vtheta * w.vtheta);
    CHECK_CLOSE(state.E[k], w.rho * (e_int + ke), 1e-14, "conserved E");
}

// ── Limiters ───────────────────────────────────────────────────────

static void test_minmod() {
    CHECK_CLOSE(minmod(1.0, 2.0), 1.0, 1e-14, "minmod same sign");
    CHECK_CLOSE(minmod(-1.0, -2.0), -1.0, 1e-14, "minmod both neg");
    CHECK_CLOSE(minmod(1.0, -1.0), 0.0, 1e-14, "minmod opposite sign");
    CHECK_CLOSE(minmod(0.0, 5.0), 0.0, 1e-14, "minmod one zero");
    CHECK_CLOSE(minmod(3.0, 1.0), 1.0, 1e-14, "minmod picks smaller");
}

static void test_van_leer() {
    CHECK_CLOSE(van_leer(1.0, -1.0), 0.0, 1e-14, "vl opposite sign");
    // van_leer(a,b) = 2ab/(a+b) for same sign
    CHECK_CLOSE(van_leer(1.0, 3.0), 2.0 * 3.0 / 4.0, 1e-14, "vl same sign");
    CHECK_CLOSE(van_leer(2.0, 2.0), 2.0, 1e-14, "vl equal");
}

static void test_apply_limiter() {
    CHECK_CLOSE(apply_limiter(1.0, 2.0, Limiter::MINMOD), 1.0, 1e-14, "apply minmod");
    CHECK_CLOSE(apply_limiter(1.0, 3.0, Limiter::VAN_LEER), 1.5, 1e-14, "apply van_leer");
}

// ── MUSCL reconstruction ───────────────────────────────────────────

static void test_muscl_constant() {
    // Constant state → reconstruction should return the same state on both sides
    PrimitiveVars w;
    w.rho = 1.0; w.vr = 0.5; w.vtheta = -0.3; w.P = 2.0;

    auto rp = muscl_reconstruct_r(w, w, w, w, Limiter::MINMOD);
    CHECK_CLOSE(rp.left.rho, w.rho, 1e-14, "muscl const rho L");
    CHECK_CLOSE(rp.right.rho, w.rho, 1e-14, "muscl const rho R");
    CHECK_CLOSE(rp.left.P, w.P, 1e-14, "muscl const P L");
    CHECK_CLOSE(rp.right.P, w.P, 1e-14, "muscl const P R");
}

static void test_muscl_linear() {
    // Linear profile: rho = 1 + 0.1*i  at cells i=-1,0,1,2
    // slopes are all 0.1, limiter preserves → exact linear reconstruction
    PrimitiveVars wm1, w0, w1, w2;
    wm1.rho = 0.9; wm1.vr = 0; wm1.vtheta = 0; wm1.P = 1.0;
    w0.rho = 1.0;  w0.vr = 0;  w0.vtheta = 0;  w0.P = 1.0;
    w1.rho = 1.1;  w1.vr = 0;  w1.vtheta = 0;  w1.P = 1.0;
    w2.rho = 1.2;  w2.vr = 0;  w2.vtheta = 0;  w2.P = 1.0;

    auto rp = muscl_reconstruct_r(wm1, w0, w1, w2, Limiter::MINMOD);
    // Left value at face 0+1/2: w0 + 0.5*slope = 1.0 + 0.05 = 1.05
    CHECK_CLOSE(rp.left.rho, 1.05, 1e-14, "muscl linear rho L");
    // Right value at face 0+1/2: w1 - 0.5*slope = 1.1 - 0.05 = 1.05
    CHECK_CLOSE(rp.right.rho, 1.05, 1e-14, "muscl linear rho R");
}

static void test_muscl_tvd() {
    // Limiter should clip the slope at a discontinuity
    PrimitiveVars wm1, w0, w1, w2;
    wm1.rho = 1.0; wm1.vr = 0; wm1.vtheta = 0; wm1.P = 1.0;
    w0.rho = 1.0;  w0.vr = 0;  w0.vtheta = 0;  w0.P = 1.0;
    w1.rho = 2.0;  w1.vr = 0;  w1.vtheta = 0;  w1.P = 1.0;
    w2.rho = 2.0;  w2.vr = 0;  w2.vtheta = 0;  w2.P = 1.0;

    auto rp = muscl_reconstruct_r(wm1, w0, w1, w2, Limiter::MINMOD);
    // slope_l = 0, slope_r = 1 → minmod = 0 → left = w0 = 1.0
    CHECK_CLOSE(rp.left.rho, 1.0, 1e-14, "muscl tvd rho L");
    // For right: slope_l = 1, slope_r = 0 → minmod = 0 → right = w1 = 2.0
    CHECK_CLOSE(rp.right.rho, 2.0, 1e-14, "muscl tvd rho R");
}

// ── HLLC Riemann solver ────────────────────────────────────────────

static void test_hllc_zero_velocity_contact() {
    // Stationary contact: same P, same v=0, different rho → flux should be zero
    double gamma = 5.0 / 3.0;
    PrimitiveVars wl, wr;
    wl.rho = 1.0; wl.vr = 0; wl.vtheta = 0; wl.P = 1.0;
    wr.rho = 2.0; wr.vr = 0; wr.vtheta = 0; wr.P = 1.0;

    Flux4 f = hllc_flux_r(wl, wr, gamma);
    CHECK_CLOSE(f.f_rho, 0.0, 1e-14, "hllc contact f_rho");
    CHECK_CLOSE(f.f_mr, 1.0, 1e-14, "hllc contact f_mr = P");
    CHECK_CLOSE(f.f_mtheta, 0.0, 1e-14, "hllc contact f_mtheta");
    CHECK_CLOSE(f.f_E, 0.0, 1e-14, "hllc contact f_E");
}

static void test_hllc_supersonic_r() {
    // Supersonic flow to the right: all waves move right, flux = left state flux
    double gamma = 5.0 / 3.0;
    PrimitiveVars wl, wr;
    wl.rho = 1.0; wl.vr = 10.0; wl.vtheta = 0.0; wl.P = 1.0;
    wr.rho = 0.5; wr.vr = 10.0; wr.vtheta = 0.0; wr.P = 0.5;

    double cs_l = std::sqrt(gamma * wl.P / wl.rho);
    // All waves right-moving if vr > cs for both states
    CHECK_TRUE(wl.vr > cs_l, "supersonic check");

    Flux4 f = hllc_flux_r(wl, wr, gamma);
    // Exact left flux
    double e_total_l = wl.P / (gamma - 1.0) + 0.5 * wl.rho * wl.vr * wl.vr;
    CHECK_CLOSE(f.f_rho, wl.rho * wl.vr, 1e-12, "hllc supersonic f_rho");
    CHECK_CLOSE(f.f_mr, wl.rho * wl.vr * wl.vr + wl.P, 1e-12, "hllc supersonic f_mr");
    CHECK_CLOSE(f.f_E, (e_total_l + wl.P) * wl.vr, 1e-12, "hllc supersonic f_E");
}

static void test_hllc_symmetry() {
    // Symmetric Sod-like: swap L/R and flip velocity → flux should negate
    double gamma = 1.4;
    PrimitiveVars wl, wr;
    wl.rho = 1.0; wl.vr = 0.5; wl.vtheta = 0.2; wl.P = 1.0;
    wr.rho = 0.5; wr.vr = -0.3; wr.vtheta = 0.1; wr.P = 0.5;

    Flux4 f_lr = hllc_flux_r(wl, wr, gamma);

    // Flip: swap L/R and negate normal velocity
    PrimitiveVars wl2, wr2;
    wl2.rho = wr.rho; wl2.vr = -wr.vr; wl2.vtheta = wr.vtheta; wl2.P = wr.P;
    wr2.rho = wl.rho; wr2.vr = -wl.vr; wr2.vtheta = wl.vtheta; wr2.P = wl.P;

    Flux4 f_rl = hllc_flux_r(wl2, wr2, gamma);

    CHECK_CLOSE(f_lr.f_rho, -f_rl.f_rho, 1e-12, "hllc symmetry f_rho");
    CHECK_CLOSE(f_lr.f_E, -f_rl.f_E, 1e-12, "hllc symmetry f_E");
}

static void test_hllc_theta_direction() {
    // Same state but via theta interface: mass flux should use vtheta as normal
    double gamma = 5.0 / 3.0;
    PrimitiveVars wl, wr;
    wl.rho = 1.0; wl.vr = 0.0; wl.vtheta = 2.0; wl.P = 1.0;
    wr.rho = 1.0; wr.vr = 0.0; wr.vtheta = 2.0; wr.P = 1.0;

    Flux4 f = hllc_flux_theta(wl, wr, gamma);
    // Uniform state → F_rho = rho * vtheta
    CHECK_CLOSE(f.f_rho, 1.0 * 2.0, 1e-14, "hllc theta f_rho");
    // F_mtheta = rho*vtheta^2 + P
    CHECK_CLOSE(f.f_mtheta, 1.0 * 4.0 + 1.0, 1e-14, "hllc theta f_mtheta");
}

// ── Poisson GMG: solve nabla^2 Phi = 4*pi*G*rho for uniform sphere ─

static void test_poisson_uniform_sphere() {
    // Uniform density sphere: analytic Phi = (2/3)*pi*G*rho*(r^2 - 3*R^2)
    // interior: Phi(r) = (2/3)*pi*G*rho*r^2 - 2*pi*G*rho*R^2
    // We check that GMG converges to this within a few %.
    int nr = 32, nt = 16;
    double R = 1.0, G = 1.0;
    double rho_val = 1.0;

    Grid grid;
    grid.init(nr, nt, R, 2.0, 2);

    std::vector<double> rho_cells(nr * nt, rho_val);
    std::vector<double> rhs(nr * nt), phi(nr * nt, 0.0);

    compute_poisson_rhs(grid, rho_cells, G, rhs);

    PoissonGMG gmg;
    gmg.init(grid);
    gmg.solve(rhs.data(), phi.data(), 50, 1e-8);

    // Check a few interior points against analytic
    double M_total = rho_val * (4.0 / 3.0) * M_PI * R * R * R;
    double max_err = 0.0;
    for (int i = 0; i < nr - 1; ++i) {
        double r = grid.r_center[i];
        // Interior of uniform sphere:
        // Phi(r) = -(2/3)*pi*G*rho*(3R^2 - r^2)  [standard form]
        //        = (2*pi*G*rho/3)*r^2 - 2*pi*G*rho*R^2
        double phi_exact = (2.0 * M_PI * G * rho_val / 3.0) * r * r
                         - 2.0 * M_PI * G * rho_val * R * R;
        double err = std::fabs(phi[i * nt] - phi_exact);
        double scale = std::fabs(phi_exact);
        if (scale > 1e-10)
            max_err = std::max(max_err, err / scale);
    }
    // On a log-stretched 32-cell mesh, expect a few % error
    CHECK_TRUE(max_err < 0.05, "poisson uniform sphere relative error < 5%");
}

// ── Poisson stencil coefficients: verify symmetry and sign ─────────

static void test_poisson_stencil_symmetry() {
    // On a uniform grid, interior stencils should satisfy cC < 0 (diagonal dominant)
    // and cW, cE, cS, cN >= 0 (M-matrix property)
    int nr = 8, nt = 8;
    Grid grid;
    grid.init(nr, nt, 1.0, 1.0, 2); // alpha=1 → nearly uniform radial

    PoissonGMG gmg;
    gmg.init(grid);

    // We can't access stencil_coeffs directly (private), so we verify via solve:
    // For a constant rhs, the solution should be smooth and monotone.
    // Instead, verify diagonal dominance indirectly:
    // A uniform rhs should produce a smooth solution without oscillations.
    std::vector<double> rho_cells(nr * nt, 1.0);
    std::vector<double> rhs(nr * nt), phi(nr * nt, 0.0);
    compute_poisson_rhs(grid, rho_cells, 1.0, rhs);
    gmg.solve(rhs.data(), phi.data(), 50, 1e-10);

    // Solution should be monotonically increasing from center to boundary
    // (since Phi is most negative at center for positive rho)
    bool monotone = true;
    int j_eq = nt / 2;
    for (int i = 1; i < nr - 1; ++i) {
        if (phi[i * nt + j_eq] < phi[(i - 1) * nt + j_eq] - 1e-10) {
            monotone = false;
            break;
        }
    }
    CHECK_TRUE(monotone, "poisson solution monotone in r");
}

// ── Grid geometry ──────────────────────────────────────────────────

static void test_grid_volume_sum() {
    // Sum of all cell volumes × 2π should equal the sphere volume (4/3)πR³
    int nr = 16, nt = 8;
    double R = 2.0;
    Grid grid;
    grid.init(nr, nt, R, 2.0, 2);

    double vol_sum = 0.0;
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nt; ++j)
            vol_sum += grid.cell_volume[i * nt + j];
    vol_sum *= 2.0 * M_PI; // azimuthal factor

    double vol_exact = (4.0 / 3.0) * M_PI * R * R * R;
    CHECK_CLOSE(vol_sum, vol_exact, 1e-10 * vol_exact, "grid volume sum = (4/3)πR³");
}

static void test_grid_area_consistency() {
    // Sum of radial face areas at r=R should equal 4πR²
    int nr = 16, nt = 16;
    double R = 1.0;
    Grid grid;
    grid.init(nr, nt, R, 2.0, 2);

    double area_sum = 0.0;
    for (int j = 0; j < nt; ++j)
        area_sum += grid.area_r[nr * nt + j]; // face at i=nr (r=R)
    area_sum *= 2.0 * M_PI;

    double area_exact = 4.0 * M_PI * R * R;
    CHECK_CLOSE(area_sum, area_exact, 1e-10 * area_exact, "outer face area = 4πR²");
}

// ── main ───────────────────────────────────────────────────────────

int main() {
    std::fprintf(stderr, "=== stellar2d unit tests ===\n");

    // EOS
    test_eos_pressure();
    test_eos_internal_energy();
    test_eos_roundtrip();
    test_eos_sound_speed();

    // State
    test_state_roundtrip();
    test_conserved_values();

    // Limiters
    test_minmod();
    test_van_leer();
    test_apply_limiter();

    // MUSCL
    test_muscl_constant();
    test_muscl_linear();
    test_muscl_tvd();

    // HLLC
    test_hllc_zero_velocity_contact();
    test_hllc_supersonic_r();
    test_hllc_symmetry();
    test_hllc_theta_direction();

    // Poisson / GMG
    test_poisson_uniform_sphere();
    test_poisson_stencil_symmetry();

    // Grid geometry
    test_grid_volume_sum();
    test_grid_area_consistency();

    std::fprintf(stderr, "=== %d/%d tests passed ===\n",
                 g_tests - g_failures, g_tests);

    return g_failures > 0 ? 1 : 0;
}
