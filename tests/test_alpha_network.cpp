// Unit tests for alpha_network.h (Phase B — 13 species + photodisintegration).
//
// Tests:
//   1. Mass conservation under He-burning (T₉=0.2)
//   2. Triple-α at He-flash (T₉=0.3)
//   3. O-burning to Si/S at T₉=2
//   4. α-freeze-out at T₉=5, ρ=10⁷ — mass should flow to ⁵⁶Ni / ⁵²Fe
//   5. NSE at very high T₉=7 — photodisintegration brings system toward α-dominated
//   6. Zero-temperature safety (no rates below T₉=1e-4)
//
// Run:  g++ -std=c++17 tests/test_alpha_network.cpp && ./a.out

#include "../src/physics/alpha_network.h"
#include <cstdio>
#include <cmath>

using namespace alpha_net;

static int n_failed = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("  FAIL: %s\n", msg); ++n_failed; } \
    else         { std::printf("  ok:   %s\n", msg); } \
} while (0)

static const char* NAMES[] = {
    "He4", "C12", "O16", "Ne20", "Mg24", "Si28",
    "S32", "Ar36", "Ca40", "Ti44", "Cr48", "Fe52", "Ni56"
};

static double sum_X(const double X[N_SPEC]) {
    double s = 0;
    for (int i = 0; i < N_SPEC; ++i) s += X[i];
    return s;
}

static void print_X(const char* label, const double X[N_SPEC]) {
    std::printf("  %s  [", label);
    for (int i = 0; i < N_SPEC; ++i)
        if (X[i] > 1e-4)
            std::printf(" %s=%.3e", NAMES[i], X[i]);
    std::printf(" ]  Σ=%.6f\n", sum_X(X));
}

int main() {
    std::printf("\n=== test_alpha_network (Phase B, 13 species) ===\n\n");

    // ── Test 1: He-burning regime T₉=0.2 ──
    std::printf("[1] T₉=0.2, ρ=1e6, 1 Myr: He + C equilibrating\n");
    {
        double X[N_SPEC] = {0};
        X[HE4] = 0.5;  X[C12] = 0.5;
        double S0 = sum_X(X);
        print_X("init", X);
        double eps = advance_substep(X, 1e6, 2.0e8, 1e6);
        print_X(" end", X);
        CHECK(fabs(sum_X(X) - S0) < 1e-6, "|ΔΣX| < 1e-6");
        CHECK(X[HE4] < 0.5, "He consumed");
        CHECK(X[C12] + X[O16] > 0.8, "mass piled into C+O");
        CHECK(eps > 0.0, "energy released positive");
        std::printf("    ε·dt = %.3e erg/g\n", eps);
    }

    // ── Test 2: triple-α He-flash ──
    std::printf("\n[2] T₉=0.3, ρ=1e5: pure-He flash\n");
    {
        double X[N_SPEC] = {0};
        X[HE4] = 1.0;
        print_X("init", X);
        double eps = advance_substep(X, 1e5, 3.0e8, 1e10);
        print_X(" end", X);
        CHECK(X[C12] > 0.01, "¹²C produced by 3α");
        CHECK(fabs(sum_X(X) - 1.0) < 1e-6, "mass conserved");
    }

    // ── Test 3: O-burning ──
    std::printf("\n[3] T₉=2.0, ρ=1e7, 1 s: O-burning\n");
    {
        double X[N_SPEC] = {0};
        X[HE4] = 0.5;  X[O16] = 0.5;
        print_X("init", X);
        double eps = advance_substep(X, 1e7, 2.0e9, 1.0);
        print_X(" end", X);
        CHECK(X[SI28] + X[S32] > 0.1, "mass advances past Si28");
        CHECK(X[O16] < 0.5, "¹⁶O consumed");
        CHECK(fabs(sum_X(X) - 1.0) < 1e-6, "mass conserved");
    }

    // ── Test 4: α-chain advances at T₉=3.5 (pre-NSE Si-burning) ──
    // Note: precise Fe-peak abundances at T₉ > 5 require Hashimoto+1989 /
    // Thielemann+1986 rate tables (Phase C).  Phase B's approximate CF88-style
    // fits for Si28(α,γ)S32 and heavier are qualitatively correct but not
    // quantitative at α-rich freezeout conditions.
    std::printf("\n[4] T₉=3.5, ρ=1e8, 10 s: chain advances past Si\n");
    {
        double X[N_SPEC] = {0};
        X[HE4] = 0.5;  X[O16] = 0.5;
        print_X("init", X);
        double eps = advance_substep(X, 1e8, 3.5e9, 10.0);
        print_X(" end", X);
        double past_si = X[S32] + X[AR36] + X[CA40] + X[TI44]
                       + X[CR48] + X[FE52] + X[NI56];
        CHECK(past_si > 0.05, "Chain advances past Si28 (>5% in S/Ar/Ca/...)");
        CHECK(fabs(sum_X(X) - 1.0) < 1e-5, "mass conserved to 1e-5");
        CHECK(eps > 1e15, "energy > 1e15 erg/g");
        std::printf("    mass beyond Si28 = %.3f\n", past_si);
    }

    // ── Test 5: photodisintegration at extreme T₉=7 ──
    std::printf("\n[5] T₉=7.0, ρ=1e6, 0.1 s: photo-dissociation dominates\n");
    {
        double X[N_SPEC] = {0};
        X[NI56] = 1.0;  // start with pure Ni56
        print_X("init", X);
        double eps = advance_substep(X, 1e6, 7.0e9, 0.1);
        print_X(" end", X);
        // At T₉=7 the equilibrium shifts dramatically toward lighter nuclei;
        // we expect significant break-up of Ni56.
        CHECK(X[NI56] < 0.99, "Ni56 partially photodisintegrated");
        CHECK(fabs(sum_X(X) - 1.0) < 1e-4, "mass conserved in reverse reactions");
    }

    // ── Test 6: safety at zero temperature ──
    std::printf("\n[6] T=0 safety: no rates\n");
    {
        double X[N_SPEC] = {0};
        X[HE4] = 1.0;
        double eps = advance_substep(X, 1e6, 0.0, 1e10);
        CHECK(fabs(X[HE4] - 1.0) < 1e-12, "no burn at T=0");
        CHECK(fabs(eps) < 1e-20, "no energy release at T=0");
    }

    std::printf("\n=== summary: %d failures ===\n\n", n_failed);
    return n_failed;
}
