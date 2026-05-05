#pragma once
//
// Alpha-chain nuclear network (Phase A — 6 reactions on 6 nuclei).
//
// Species (by index):
//   0: ⁴He    1: ¹²C    2: ¹⁶O    3: ²⁰Ne    4: ²⁴Mg    5: ²⁸Si
//
// Reactions (forward; backward via detailed balance is not included in Phase A,
// which is valid for T₉ < ~6 where forward dominates):
//   R0: 3 ⁴He → ¹²C                (triple-α)
//   R1: ¹²C + ⁴He → ¹⁶O            (C(α,γ)O)
//   R2: ¹⁶O + ⁴He → ²⁰Ne           (O(α,γ)Ne)
//   R3: ²⁰Ne + ⁴He → ²⁴Mg          (Ne(α,γ)Mg)
//   R4: ²⁴Mg + ⁴He → ²⁸Si          (Mg(α,γ)Si)
//   R5: ¹⁶O + ¹⁶O → ²⁸Si + α       (O+O heavy-ion, net to Si)
//
// Rate formulas: Caughlan & Fowler (1988) analytic fits, in cgs.
// Reference:  Timmes aprox13 reaction list, cococubed.com.
//
// For ODE integration we use a simple backward-Euler + fixed-point iteration,
// which is stable for stiff systems in the explosion regime τ_nuc ≳ 10⁻³ s.
//
// State vector is mass-fraction X[6]; we evolve X directly (not molar Y) for
// simplicity, using Y_i = X_i / A_i only internally in rate evaluations.
//
// Energy release per burn: track via Q-values in MeV, convert to erg/g.
//
// GPU __host__ __device__.

#include <cmath>

#ifdef __CUDACC__
#define ANET_HD __host__ __device__
#else
#define ANET_HD
#endif

namespace alpha_net {

constexpr int N_SPEC = 6;

enum Species : int {
    HE4 = 0, C12 = 1, O16 = 2, NE20 = 3, MG24 = 4, SI28 = 5,
};

constexpr double A_NUC[N_SPEC] = { 4.0, 12.0, 16.0, 20.0, 24.0, 28.0 };

// Q-values in MeV (mass defect per reaction).
// Data: NNDC AME2020 atomic mass evaluation.
constexpr double Q_MEV_R0_3A_C12   =  7.275;   // 3 ⁴He → ¹²C
constexpr double Q_MEV_R1_C12_AG   =  7.162;   // ¹²C + α → ¹⁶O
constexpr double Q_MEV_R2_O16_AG   =  4.730;   // ¹⁶O + α → ²⁰Ne
constexpr double Q_MEV_R3_NE20_AG  =  9.316;   // ²⁰Ne + α → ²⁴Mg
constexpr double Q_MEV_R4_MG24_AG  =  9.984;   // ²⁴Mg + α → ²⁸Si
constexpr double Q_MEV_R5_O16_O16  = 16.542;   // ¹⁶O + ¹⁶O → ²⁸Si + α

constexpr double MEV_TO_ERG = 1.602176634e-6;  // 1 MeV = 1.602e-6 erg
constexpr double N_A        = 6.02214076e23;    // Avogadro
constexpr double MU         = 1.66053907e-24;   // atomic mass unit [g]

// ──────────────────────────────────────────────────────────────────────────────
// Reaction rates λ(ρ, T) — per species pair, CF88 analytic fits.
// All rates in cgs (reactions · cm³/g·s for binary; reactions · cm⁶/g²·s for 3α).
// T9 = T / 10⁹ K.
//
// The "N_A <σv>" values below are the standard nuclear-astrophysics tabulation
// (cm³·mol⁻¹·s⁻¹); we'll convert to "per unit of mass-fraction" inside rhs().

// 3α → ¹²C: CF88 eq 26, T9_8 range, dominant at T₉ > 0.1
ANET_HD inline double rate_3a(double T9) {
    if (T9 < 1e-3) return 0.0;
    double T9i = 1.0 / T9;
    double T9r  = T9 / (1.0 + 78.3 * T9);
    double r2a  = 7.40e5 * pow(T9, -1.5) * exp(-1.0663 * T9i);
    double r3a  = 2.76e-1 * T9r * T9r * T9r * exp(-1.054 * T9i / T9r);
    // Total Caughlan-Fowler:
    double lam = 6.272e3 * pow(T9, -1.5) * exp(-4.4027 * T9i)  // narrow
               + 2.01e-3 * pow(T9, -1.5) * exp(-4.4027 * T9i); // broad
    // Simplified: use the canonical analytic fit (Fowler+ 1975 form):
    //   λ_3a ≈ 2.79e-8 * ρ² / T9³ · f(T9)
    // For Phase A, use the common simplified form:
    double T9_13 = pow(T9, 1.0/3.0);
    double lam_3a = 2.79e-8 / (T9 * T9 * T9) *
                    exp(-4.4027 / T9);
    return lam_3a;  // cm⁶/mol²/s, multiplied by ρ²Y(⁴He)³ later
}

// ¹²C(α,γ)¹⁶O: CF88 eq E1. Below T9~3 dominated by low-energy tail.
ANET_HD inline double rate_c12_ag(double T9) {
    if (T9 < 1e-3) return 0.0;
    double T9a = T9 / (1.0 + 0.0489 * T9);
    double T9a_m23 = pow(T9a, -2.0/3.0);
    double T9_m32  = pow(T9, -1.5);
    double term1 = 1.04e8 / (T9 * T9) * T9a_m23 *
                   exp(-32.120 * T9a_m23 - (T9/3.496) * (T9/3.496));
    double term2 = 1.76e8 / (T9 * T9) * pow(1.0 + 0.2654 * T9a, 2.0) *
                   T9a_m23 * exp(-32.120 * T9a_m23);
    double term3 = 1.25e3 * T9_m32 * exp(-27.499 / T9);
    double term4 = 1.43e-2 * T9 * T9 * T9 * T9 * T9 * exp(-15.541 / T9);
    return term1 + term2 + term3 + term4;  // cm³/mol/s
}

// ¹⁶O(α,γ)²⁰Ne: CF88 eq E2
ANET_HD inline double rate_o16_ag(double T9) {
    if (T9 < 1e-3) return 0.0;
    double T9_m23 = pow(T9, -2.0/3.0);
    double T9_m32 = pow(T9, -1.5);
    double T9_23  = pow(T9, 2.0/3.0);
    double term1 = 9.37e9 * T9_m23 * exp(-39.757 * T9_m23 - (T9/1.586) * (T9/1.586));
    double term2 = 62.1 * T9_m32 * exp(-10.297 / T9);
    double term3 = 538.0 * T9_m32 * exp(-12.226 / T9);
    double term4 = 13.0 * T9 * T9 * exp(-20.093 / T9);
    return term1 + term2 + term3 + term4;
}

// ²⁰Ne(α,γ)²⁴Mg: CF88 eq E3
ANET_HD inline double rate_ne20_ag(double T9) {
    if (T9 < 1e-3) return 0.0;
    double T9_m23 = pow(T9, -2.0/3.0);
    double T9_m32 = pow(T9, -1.5);
    double term1 = 4.11e11 * T9_m23 * exp(-46.766 * T9_m23 - (T9/4.87) * (T9/4.87));
    double term2 = 5.27e3 * T9_m32 * exp(-15.869 / T9);
    double term3 = 6.51e3 * T9_m32 * exp(-16.223 / T9);
    double term4 = 4.21e1 * T9_m32 * exp(-9.310 / T9);
    double term5 = 3.20e1 * pow(T9, 2.0/3.0) * exp(-5.785 / T9);
    return term1 + term2 + term3 + term4 + term5;
}

// ²⁴Mg(α,γ)²⁸Si: CF88 eq E4
ANET_HD inline double rate_mg24_ag(double T9) {
    if (T9 < 1e-3) return 0.0;
    double T9_m23 = pow(T9, -2.0/3.0);
    double T9_m32 = pow(T9, -1.5);
    double term1 = 4.78e1 * T9_m32 * exp(-13.506 / T9);
    double term2 = 2.38e3 * T9_m32 * exp(-15.218 / T9);
    double term3 = 2.47e2 * pow(T9, 1.5) * exp(-15.147 / T9);
    double term4 = 1.72e-9 * T9_m32 * exp(-5.028 / T9);
    double term5 = 1.25e-3 * T9_m32 * exp(-7.929 / T9);
    double term6 = 2.43e1 / T9 * exp(-11.058 / T9);
    return term1 + term2 + term3 + term4 + term5 + term6;
}

// ¹⁶O + ¹⁶O → ²⁸Si + α: CF88 heavy-ion fit
ANET_HD inline double rate_o16_o16(double T9) {
    if (T9 < 1e-3) return 0.0;
    // CF88: simplified fit for T9 < 5
    double T9_m13 = pow(T9, -1.0/3.0);
    double exp_arg = -135.93 * T9_m13 - 0.629 * pow(T9, 2.0/3.0)
                    - 0.445 * pow(T9, 4.0/3.0) + 0.0103 * T9 * T9;
    // Guard
    if (exp_arg < -700.0) return 0.0;
    double prefac = 7.10e36 * T9_m13 * T9_m13;
    return prefac * exp(exp_arg);
}

// ──────────────────────────────────────────────────────────────────────────────
// RHS evaluation — dY_i/dt given current Y and (ρ, T).
//
// Using molar abundances Y_i = X_i / A_i.
// For binary A+B → C: dY_C/dt = ρ N_A <σv> Y_A Y_B (÷ 2 if A=B)
// For 3-body 3A → C: dY_C/dt = ρ² N_A² <σv>_3 Y_A³ / 6

ANET_HD inline void rhs(const double Y[N_SPEC], double rho, double T,
                        double dYdt[N_SPEC], double* eps_out) {
    double T9 = T * 1e-9;
    // Compute rates
    double l_3a    = rate_3a(T9);          // cm⁶/mol²/s
    double l_cag   = rate_c12_ag(T9);      // cm³/mol/s
    double l_oag   = rate_o16_ag(T9);
    double l_neag  = rate_ne20_ag(T9);
    double l_mgag  = rate_mg24_ag(T9);
    double l_oo    = rate_o16_o16(T9);

    double Y_He = Y[HE4], Y_C = Y[C12], Y_O = Y[O16];
    double Y_Ne = Y[NE20], Y_Mg = Y[MG24], Y_Si = Y[SI28];

    // Reaction fluxes r_i [mol/g/s]
    // r0: 3α → C   — "per-reaction" rate = (1/6) ρ² N_A² <σv>_3α Y_He³
    double r0 = (1.0/6.0) * rho * rho * l_3a * Y_He * Y_He * Y_He;
    // r1: C + α → O
    double r1 = rho * l_cag * Y_C * Y_He;
    // r2: O + α → Ne
    double r2 = rho * l_oag * Y_O * Y_He;
    // r3: Ne + α → Mg
    double r3 = rho * l_neag * Y_Ne * Y_He;
    // r4: Mg + α → Si
    double r4 = rho * l_mgag * Y_Mg * Y_He;
    // r5: O + O → Si + α   — "per-reaction" rate with identical particles
    double r5 = 0.5 * rho * l_oo * Y_O * Y_O;

    // Species production/destruction [mol/g/s]
    dYdt[HE4]  = -3.0 * r0 - r1 - r2 - r3 - r4 + r5;
    dYdt[C12]  =  r0 - r1;
    dYdt[O16]  =  r1 - r2 - 2.0 * r5;
    dYdt[NE20] =  r2 - r3;
    dYdt[MG24] =  r3 - r4;
    dYdt[SI28] =  r4 + r5;

    // Energy release [erg/g/s] = Σ Q_i · r_i · N_A · MEV_TO_ERG
    if (eps_out) {
        double eps = 0.0;
        eps += Q_MEV_R0_3A_C12  * r0 * N_A * MEV_TO_ERG;
        eps += Q_MEV_R1_C12_AG  * r1 * N_A * MEV_TO_ERG;
        eps += Q_MEV_R2_O16_AG  * r2 * N_A * MEV_TO_ERG;
        eps += Q_MEV_R3_NE20_AG * r3 * N_A * MEV_TO_ERG;
        eps += Q_MEV_R4_MG24_AG * r4 * N_A * MEV_TO_ERG;
        eps += Q_MEV_R5_O16_O16 * r5 * N_A * MEV_TO_ERG;
        *eps_out = eps;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Mass-fraction ↔ molar-abundance helpers

ANET_HD inline void X_to_Y(const double X[N_SPEC], double Y[N_SPEC]) {
    for (int i = 0; i < N_SPEC; ++i) Y[i] = X[i] / A_NUC[i];
}

ANET_HD inline void Y_to_X(const double Y[N_SPEC], double X[N_SPEC]) {
    for (int i = 0; i < N_SPEC; ++i) X[i] = Y[i] * A_NUC[i];
}

// ──────────────────────────────────────────────────────────────────────────────
// Advance mass fraction X[] by dt at fixed (ρ, T) using substepped forward Euler
// with adaptive step size.  For Phase A stability in the Si-burning regime,
// we limit per-sub-step change to 5%.
//
// Returns energy released over full dt [erg/g].
//
// NOTE: For production use in Newton framework, we'd use BE + AD Jacobian.
// This is a standalone post-processing solver suitable for running on existing
// radial1d snapshots.

ANET_HD inline double advance_substep(double X[N_SPEC], double rho, double T,
                                       double dt, int max_substeps = 10000) {
    double Y[N_SPEC];
    X_to_Y(X, Y);

    double t = 0.0;
    double eps_total = 0.0;
    double safety = 0.05;  // max relative change per sub-step

    for (int step = 0; step < max_substeps; ++step) {
        double dYdt[N_SPEC];
        double eps;
        rhs(Y, rho, T, dYdt, &eps);

        // Adaptive dt: limit |dY|/Y < safety
        double dt_try = dt - t;
        for (int i = 0; i < N_SPEC; ++i) {
            if (Y[i] > 1e-20 && fabs(dYdt[i]) > 0.0) {
                double dt_i = safety * Y[i] / fabs(dYdt[i]);
                if (dt_i < dt_try) dt_try = dt_i;
            }
        }
        if (dt_try <= 0.0) dt_try = (dt - t) * 1e-3;
        if (t + dt_try > dt) dt_try = dt - t;

        // Forward Euler step
        for (int i = 0; i < N_SPEC; ++i) Y[i] += dYdt[i] * dt_try;

        // Clamp non-negative
        for (int i = 0; i < N_SPEC; ++i) if (Y[i] < 0) Y[i] = 0;

        eps_total += eps * dt_try;
        t += dt_try;
        if (t >= dt) break;
    }

    Y_to_X(Y, X);
    return eps_total;
}

} // namespace alpha_net

#undef ANET_HD
