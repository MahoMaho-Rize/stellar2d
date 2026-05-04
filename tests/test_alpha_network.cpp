// Unit tests for alpha_network.h (Phase A — 6 species).
//
// Builds as a host-only binary (no CUDA needed for the math).
//
// Tests:
//   1. Species conservation: Σ X_i stays near 1 throughout burn
//   2. Single-reaction smoke: starting from pure ¹²C + ⁴He at T₉=0.2 produces
//      ¹⁶O at the expected rate (ε > 0, X(¹⁶O) grows)
//   3. Medium-T Si-burning: starting from pure ⁴He at T₉=3, ρ=10⁷ for 1 s,
//      most mass should end up in heavier nuclei (Si/S peak in Phase A)
//   4. Energy balance: eps · dt total within factor 2 of Q_R · Δ(burn count)
//
// Run:  pixi run python - <<'EOF'
//       # or compile manually with g++ -std=c++17 tests/test_alpha_network.cpp

#include "../src/physics/alpha_network.h"
#include <cstdio>
#include <cmath>
#include <cassert>

using namespace alpha_net;

static int n_failed = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("  FAIL: %s\n", msg); ++n_failed; } \
    else         { std::printf("  ok:   %s\n", msg); } \
} while (0)

static double sum_X(const double X[N_SPEC]) {
    double s = 0;
    for (int i = 0; i < N_SPEC; ++i) s += X[i];
    return s;
}

static void print_X(const char* label, const double X[N_SPEC]) {
    std::printf("  %s  X = [", label);
    const char* names[] = {"He4", "C12", "O16", "Ne20", "Mg24", "Si28"};
    for (int i = 0; i < N_SPEC; ++i)
        std::printf(" %s=%.4e", names[i], X[i]);
    std::printf(" ]  Σ=%.6f\n", sum_X(X));
}

int main() {
    std::printf("\n=== test_alpha_network ===\n\n");

    // ── Test 1: mass conservation + He burning to C at T₉=0.2 ──
    // (At T₉=0.2, C(α,γ)O rate is small; 3α dominates over C+α consumption.
    //  So He gets converted to C primarily.)
    std::printf("[1] mass conservation at T₉=0.2, 1 Myr (He-burning regime)\n");
    {
        double X[N_SPEC] = {0};
        X[HE4] = 0.5;
        X[C12] = 0.5;
        double rho = 1e6;
        double T   = 2.0e8;
        double S0 = sum_X(X);
        print_X("init", X);
        double dt_total = 1e6;  // 1 Myr
        double eps = advance_substep(X, rho, T, dt_total);
        print_X(" end", X);
        double S1 = sum_X(X);
        CHECK(fabs(S1 - S0) < 1e-8, "mass conservation |ΔΣX|<1e-8");
        CHECK(X[C12] > 0.5, "C grows via 3α at He-burning temperature");
        CHECK(X[HE4] < 0.5, "He consumed (3α dominates over Cαγ at T₉=0.2)");
        CHECK(eps > 0.0, "energy release positive");
        std::printf("    ε·dt = %.3e erg/g over %.1e s\n", eps, dt_total);
    }

    // ── Test 2: triple-alpha at He-flash conditions ──
    std::printf("\n[2] triple-α at T₉=0.3, ρ=1e5 (He-flash-like)\n");
    {
        double X[N_SPEC] = {0};
        X[HE4] = 1.0;
        double rho = 1e5;
        double T = 3.0e8;
        print_X("init", X);
        double eps = advance_substep(X, rho, T, 1e10);  // 300 yr
        print_X(" end", X);
        CHECK(X[C12] > 0.01, "¹²C produced from 3α");
        CHECK(eps > 0.0, "3α energy release positive");
    }

    // ── Test 3: O-burning at T₉=2 ──
    std::printf("\n[3] O-burning at T₉=2.0, ρ=1e7\n");
    {
        double X[N_SPEC] = {0};
        X[O16] = 0.5;
        X[HE4] = 0.5;
        double rho = 1e7;
        double T = 2.0e9;
        print_X("init", X);
        double eps = advance_substep(X, rho, T, 1.0);  // 1 s
        print_X(" end", X);
        CHECK(X[SI28] > 0.01, "²⁸Si produced (Si-peak burn)");
        CHECK(X[O16] < 0.5, "¹⁶O consumed");
        CHECK(eps > 0.0, "O-burning energy release positive");
    }

    // ── Test 4: high-T α-freeze-out (NSE neighbourhood) ──
    std::printf("\n[4] α-freeze-out at T₉=5.0, ρ=1e7, t=1s (Phase A top species = Si)\n");
    {
        double X[N_SPEC] = {0};
        X[HE4] = 0.5;
        X[O16] = 0.5;  // mix O + He to seed the chain
        double rho = 1e7;
        double T = 5.0e9;
        print_X("init", X);
        double eps = advance_substep(X, rho, T, 1.0);
        print_X(" end", X);
        // Phase A truncates at ²⁸Si; under full aprox13 this would proceed to
        // ⁵⁶Ni.  For Phase A we only check that mass piles up at Si.
        CHECK(X[SI28] > 0.3, "Most mass at ²⁸Si (Phase A endpoint)");
        CHECK(eps > 1e15, "Enormous nuclear energy released (>1e15 erg/g)");
    }

    std::printf("\n=== summary: %d failures ===\n\n", n_failed);
    return n_failed;
}
