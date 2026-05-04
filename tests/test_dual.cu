// Unit test for src/physics/dual.cuh forward-mode AD.
//
// Validates Dual<N> arithmetic + math functions against either hand-analytic
// derivatives or high-accuracy central finite differences. Covers:
//   - basic ops (+, -, *, /, unary minus)
//   - transcendentals (sqrt, exp, log, log10, pow, fabs)
//   - composition (chain rule)
//   - pp-chain ε(ρ, T, X) Jacobian (3-variable real physics use)

#include "../src/physics/dual.cuh"
#include "../src/physics/nuclear_pp.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

using dual::Dual;

static int g_fail = 0;

static bool close(double a, double b, double rtol = 1e-10, double atol = 1e-14) {
    double d = std::fabs(a - b);
    double s = std::fabs(a) + std::fabs(b);
    return d <= atol + rtol * s;
}

#define CHECK_CLOSE(label, got, want) do {                                      \
    if (!close((got), (want), 1e-9, 1e-14)) {                                   \
        std::fprintf(stderr, "FAIL  %-40s  got=%.17g  want=%.17g  diff=%.3e\n", \
                     (label), (double)(got), (double)(want),                    \
                     std::fabs((double)(got) - (double)(want)));                \
        ++g_fail;                                                               \
    }                                                                           \
} while (0)

// -----------------------------------------------------------------------------
// Basic arithmetic
// -----------------------------------------------------------------------------
static void test_basic_arith() {
    // f(x) = x² + 3x + 1  at x=2  →  f=11, f'=2x+3=7
    Dual<1> x = Dual<1>::seed(2.0, 0);
    Dual<1> f = x * x + 3.0 * x + 1.0;
    CHECK_CLOSE("arith f(2)",  f.v,    11.0);
    CHECK_CLOSE("arith f'(2)", f.g[0], 7.0);

    // g(x, y) = x² y + x / y  at (3, 2) → value, ∂/∂x = 2xy + 1/y = 12.5, ∂/∂y = x² - x/y² = 8.25
    Dual<2> x2 = Dual<2>::seed(3.0, 0);
    Dual<2> y2 = Dual<2>::seed(2.0, 1);
    Dual<2> g = x2 * x2 * y2 + x2 / y2;
    double want_v  = 3.0 * 3.0 * 2.0 + 3.0 / 2.0;      // 19.5
    double want_gx = 2.0 * 3.0 * 2.0 + 1.0 / 2.0;      // 12.5
    double want_gy = 3.0 * 3.0 - 3.0 / (2.0 * 2.0);    // 8.25
    CHECK_CLOSE("arith g(3,2).v",    g.v,    want_v);
    CHECK_CLOSE("arith g(3,2).gx",   g.g[0], want_gx);
    CHECK_CLOSE("arith g(3,2).gy",   g.g[1], want_gy);

    // Compound assignment
    Dual<1> a = Dual<1>::seed(1.5, 0);
    a += 0.5; CHECK_CLOSE("compound += double v", a.v, 2.0);
    a *= 2.0; CHECK_CLOSE("compound *= double v", a.v, 4.0);
    CHECK_CLOSE("compound *= double g",           a.g[0], 2.0);

    // unary minus
    Dual<1> n = -x;
    CHECK_CLOSE("unary neg v", n.v,    -2.0);
    CHECK_CLOSE("unary neg g", n.g[0], -1.0);
}

// -----------------------------------------------------------------------------
// Transcendentals
// -----------------------------------------------------------------------------
static void test_transcendentals() {
    using dual::sqrt; using dual::exp; using dual::log; using dual::pow; using dual::log10;

    // sqrt(x) at x=4  → 2, 1/(2·2)=0.25
    Dual<1> x = Dual<1>::seed(4.0, 0);
    Dual<1> r = sqrt(x);
    CHECK_CLOSE("sqrt(4)",  r.v,    2.0);
    CHECK_CLOSE("sqrt'(4)", r.g[0], 0.25);

    // exp(x) at x=1
    Dual<1> e1 = exp(Dual<1>::seed(1.0, 0));
    CHECK_CLOSE("exp(1)",  e1.v,    std::exp(1.0));
    CHECK_CLOSE("exp'(1)", e1.g[0], std::exp(1.0));

    // log(x) at x=3 → ln 3, 1/3
    Dual<1> l3 = log(Dual<1>::seed(3.0, 0));
    CHECK_CLOSE("log(3)",  l3.v,    std::log(3.0));
    CHECK_CLOSE("log'(3)", l3.g[0], 1.0 / 3.0);

    // log10(10) → 1, 1/(10 ln10)
    Dual<1> l10 = log10(Dual<1>::seed(10.0, 0));
    CHECK_CLOSE("log10(10)",  l10.v,    1.0);
    CHECK_CLOSE("log10'(10)", l10.g[0], 1.0 / (10.0 * std::log(10.0)));

    // pow(x, 2.5) at x=2 → 2^2.5, 2.5·2^1.5
    Dual<1> p = pow(Dual<1>::seed(2.0, 0), 2.5);
    CHECK_CLOSE("pow(2,2.5)",  p.v,    std::pow(2.0, 2.5));
    CHECK_CLOSE("pow'(2,2.5)", p.g[0], 2.5 * std::pow(2.0, 1.5));

    // Chain rule: f(x) = sqrt(exp(x) + 1) at x=0 → sqrt(2), exp(0)/(2 sqrt(2))
    Dual<1> y = Dual<1>::seed(0.0, 0);
    Dual<1> f = sqrt(exp(y) + 1.0);
    CHECK_CLOSE("chain f(0)",  f.v,    std::sqrt(2.0));
    CHECK_CLOSE("chain f'(0)", f.g[0], 1.0 / (2.0 * std::sqrt(2.0)));
}

// -----------------------------------------------------------------------------
// pp-chain ε(ρ, T, X) full 3-D Jacobian vs central FD.
// Since nuclear_pp_epsilon uses `pow(double, ...)` with builtin pow, we must
// route through a templated lambda that uses dual::pow. Easiest: inline the
// formula here in terms of T<Dual> ops.
// -----------------------------------------------------------------------------
template <typename T>
static T pp_epsilon(const T& rho, const T& Tin, const T& X) {
    using namespace dual;
    using std::pow;
    using std::exp;
    T T_K  = Tin;                 // already in K
    T T9   = T_K * 1e-9;
    T T913 = pow(T9,  1.0 / 3.0);
    T T9m23= pow(T9, -2.0 / 3.0);
    T eps  = 2.57e4 * X * X * rho * T9m23 * exp(-3.381 / T913);
    return eps;
}

// Float specialization (no ADL on dual::)
static double pp_epsilon_d(double rho, double T, double X) {
    double T9 = T * 1e-9;
    return 2.57e4 * X * X * rho * std::pow(T9, -2.0/3.0)
         * std::exp(-3.381 / std::pow(T9, 1.0/3.0));
}

static void test_pp_chain_jacobian() {
    double rho = 100.0;          // g/cc
    double T   = 1.5e7;          // K
    double X   = 0.7;

    // Seed Dual<3>: d/dρ at idx 0, d/dT at idx 1, d/dX at idx 2.
    Dual<3> rho_d = Dual<3>::seed(rho, 0);
    Dual<3> T_d   = Dual<3>::seed(T,   1);
    Dual<3> X_d   = Dual<3>::seed(X,   2);

    Dual<3> eps_d = pp_epsilon(rho_d, T_d, X_d);

    // Reference value
    double ref = pp_epsilon_d(rho, T, X);
    CHECK_CLOSE("pp.value", eps_d.v, ref);

    // Central-difference reference derivatives (h chosen per-axis)
    auto fd_deriv = [](auto f, double x, double h) {
        return (f(x + h) - f(x - h)) / (2.0 * h);
    };
    double depsdrho_fd = fd_deriv([&](double r) { return pp_epsilon_d(r, T, X); }, rho, 1e-3);
    double depsdT_fd   = fd_deriv([&](double t) { return pp_epsilon_d(rho, t, X); }, T,   1e2);
    double depsdX_fd   = fd_deriv([&](double xx){ return pp_epsilon_d(rho, T, xx); }, X,  1e-5);

    // AD matches FD to ~6 digits (FD truncation/rounding limited)
    auto close_fd = [](double a, double b) {
        double rel = std::fabs(a - b) / (std::fabs(a) + std::fabs(b) + 1e-300);
        return rel < 1e-5;
    };
    if (!close_fd(eps_d.g[0], depsdrho_fd)) {
        std::fprintf(stderr, "FAIL  pp.depsdrho  AD=%.6e  FD=%.6e\n", eps_d.g[0], depsdrho_fd);
        ++g_fail;
    }
    if (!close_fd(eps_d.g[1], depsdT_fd)) {
        std::fprintf(stderr, "FAIL  pp.depsdT    AD=%.6e  FD=%.6e\n", eps_d.g[1], depsdT_fd);
        ++g_fail;
    }
    if (!close_fd(eps_d.g[2], depsdX_fd)) {
        std::fprintf(stderr, "FAIL  pp.depsdX    AD=%.6e  FD=%.6e\n", eps_d.g[2], depsdX_fd);
        ++g_fail;
    }

    // Also check analytic d ln ε / d T from nuclear_pp.h matches AD
    //   d ln ε / d T = (eps_d.g[1]) / eps_d.v
    NuclearPPParams pars;
    pars.T_floor = 0.0;
    pars.T_scale = 1.0;
    double dedT_analytic = nuclear_pp_dedT(rho, T, pars);  // d ε / d T (computed in nuclear_pp.h)
    if (std::fabs(eps_d.g[1] - dedT_analytic) / std::fabs(dedT_analytic) > 1e-5) {
        std::fprintf(stderr, "FAIL  pp.dedT vs nuclear_pp_dedT  AD=%.6e  analytic=%.6e\n",
                     eps_d.g[1], dedT_analytic);
        ++g_fail;
    }
}

int main() {
    test_basic_arith();
    test_transcendentals();
    test_pp_chain_jacobian();
    if (g_fail == 0) {
        std::printf("all Dual<N> tests passed\n");
        return 0;
    }
    std::printf("Dual<N> tests FAILED: %d check(s)\n", g_fail);
    return 1;
}
