#pragma once
//
// α-chain nuclear network — Phase B (13 species, full aprox13 scope).
//
// Species (by index):
//   0:  ⁴He    1:  ¹²C     2:  ¹⁶O    3:  ²⁰Ne   4:  ²⁴Mg
//   5:  ²⁸Si   6:  ³²S     7:  ³⁶Ar   8:  ⁴⁰Ca   9:  ⁴⁴Ti
//  10:  ⁴⁸Cr  11:  ⁵²Fe   12:  ⁵⁶Ni
//
// Reactions (15 forward + 12 photodisintegration reverses):
//   R0:   3 ⁴He → ¹²C                        (triple-α)
//   R1:   ¹²C + ⁴He → ¹⁶O
//   R2:   ¹⁶O + ⁴He → ²⁰Ne
//   R3:   ²⁰Ne + ⁴He → ²⁴Mg
//   R4:   ²⁴Mg + ⁴He → ²⁸Si
//   R5:   ²⁸Si + ⁴He → ³²S
//   R6:   ³²S  + ⁴He → ³⁶Ar
//   R7:   ³⁶Ar + ⁴He → ⁴⁰Ca
//   R8:   ⁴⁰Ca + ⁴He → ⁴⁴Ti
//   R9:   ⁴⁴Ti + ⁴He → ⁴⁸Cr
//   R10:  ⁴⁸Cr + ⁴He → ⁵²Fe
//   R11:  ⁵²Fe + ⁴He → ⁵⁶Ni
//   R12:  ¹⁶O + ¹⁶O → ²⁸Si + α              (O+O heavy-ion)
//   R13:  ¹²C + ¹²C → ²⁰Ne + α              (C+C heavy-ion)
//   R14:  ¹²C + ¹⁶O → ²⁴Mg + α              (C+O heavy-ion)
//
//   Reverse (photodisintegration via detailed balance from forward rate):
//     R̄0:  ¹²C → 3 α                          (triple-α reverse, 3-body)
//     R̄i:  X(γ,α)X'  for i = 1..11 (Ni→Fe, Fe→Cr, ..., O→C)
//
// Rate formulas:
//   R0–R4, R12:       Caughlan & Fowler 1988 (CF88) analytic fits
//   R5–R11:           CF88-style statistical-model forms with Hashimoto+1989 fits
//   R13, R14:         CF88 heavy-ion fits
//   R̄i (binary):      Fowler+1975 detailed balance from forward:
//                       λ_γ = α_rev · T₉^(3/2) · exp(-11.605 Q / T₉) · λ_fwd
//   R̄0 (3α):         Fowler+1964 eq 16 / Timmes aprox13
//
// Q-values: NNDC AME2020.
// Partition-function ratios in detailed balance: using ground-state weights
// (2J+1=1 for all even-even nuclei in this chain, G_ratio=1), which is
// standard for α-network approximations at T₉ < 6.
//
// Integration: substepped forward-Euler with adaptive dt limited to 5%
// relative change per sub-step.  For T₉ > 4 where photodisintegration brings
// the network to NSE, this still works but takes more sub-steps.  For
// production Newton-framework use, re-implement advance_substep() as a
// backward-Euler solve with exact J·v via the existing Dual<N> AD.
//
// GPU __host__ __device__.
//

#include <cmath>

#ifdef __CUDACC__
#define ANET_HD __host__ __device__
#else
#define ANET_HD
#endif

namespace alpha_net {

constexpr int N_SPEC = 13;

enum Species : int {
    HE4 = 0, C12 = 1, O16 = 2, NE20 = 3, MG24 = 4,
    SI28 = 5, S32 = 6, AR36 = 7, CA40 = 8, TI44 = 9,
    CR48 = 10, FE52 = 11, NI56 = 12,
};

constexpr double A_NUC[N_SPEC] = {
    4.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0,
};

// Q-values in MeV (mass defect per reaction).
// Data: NNDC AME2020 atomic mass evaluation.
constexpr double Q_3A     =  7.275;  // 3 ⁴He → ¹²C
constexpr double Q_CAG    =  7.162;  // ¹²C(α,γ)¹⁶O
constexpr double Q_OAG    =  4.730;  // ¹⁶O(α,γ)²⁰Ne
constexpr double Q_NEAG   =  9.316;  // ²⁰Ne(α,γ)²⁴Mg
constexpr double Q_MGAG   =  9.984;  // ²⁴Mg(α,γ)²⁸Si
constexpr double Q_SIAG   =  6.948;  // ²⁸Si(α,γ)³²S
constexpr double Q_SAG    =  6.645;  // ³²S(α,γ)³⁶Ar
constexpr double Q_ARAG   =  7.041;  // ³⁶Ar(α,γ)⁴⁰Ca
constexpr double Q_CAAG   =  5.128;  // ⁴⁰Ca(α,γ)⁴⁴Ti
constexpr double Q_TIAG   =  7.695;  // ⁴⁴Ti(α,γ)⁴⁸Cr
constexpr double Q_CRAG   =  7.941;  // ⁴⁸Cr(α,γ)⁵²Fe
constexpr double Q_FEAG   =  7.990;  // ⁵²Fe(α,γ)⁵⁶Ni
constexpr double Q_OO     = 16.542;  // ¹⁶O + ¹⁶O → ²⁸Si + α
constexpr double Q_CC     =  4.616;  // ¹²C + ¹²C → ²⁰Ne + α
constexpr double Q_CO     = 16.753;  // ¹²C + ¹⁶O → ²⁴Mg + α

constexpr double MEV_TO_ERG = 1.602176634e-6;  // 1 MeV = 1.602e-6 erg
constexpr double N_A        = 6.02214076e23;    // Avogadro

// ──────────────────────────────────────────────────────────────────────────────
// Reaction rates λ(ρ, T) — CF88 analytic fits, cgs.
// N_A <σv>  [cm³/mol/s]  for binary reactions.
// Symmetry factors (for identical particles) are applied in rhs().

// R0: 3α → ¹²C. Simplified CF88 fit, robust at T₉ > 0.1.
ANET_HD inline double rate_3a(double T9) {
    if (T9 < 1e-3) return 0.0;
    double lam = 2.79e-8 / (T9 * T9 * T9) * exp(-4.4027 / T9);  // cm⁶/mol²/s
    return lam;
}

// R1: ¹²C(α,γ)¹⁶O  [CF88 E1]
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
    return term1 + term2 + term3 + term4;
}

// R2: ¹⁶O(α,γ)²⁰Ne  [CF88 E2]
ANET_HD inline double rate_o16_ag(double T9) {
    if (T9 < 1e-3) return 0.0;
    double T9_m23 = pow(T9, -2.0/3.0);
    double T9_m32 = pow(T9, -1.5);
    double term1 = 9.37e9 * T9_m23 * exp(-39.757 * T9_m23 - (T9/1.586) * (T9/1.586));
    double term2 = 62.1 * T9_m32 * exp(-10.297 / T9);
    double term3 = 538.0 * T9_m32 * exp(-12.226 / T9);
    double term4 = 13.0 * T9 * T9 * exp(-20.093 / T9);
    return term1 + term2 + term3 + term4;
}

// R3: ²⁰Ne(α,γ)²⁴Mg  [CF88 E3]
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

// R4: ²⁴Mg(α,γ)²⁸Si  [CF88 E4]
ANET_HD inline double rate_mg24_ag(double T9) {
    if (T9 < 1e-3) return 0.0;
    double T9_m32 = pow(T9, -1.5);
    double term1 = 4.78e1 * T9_m32 * exp(-13.506 / T9);
    double term2 = 2.38e3 * T9_m32 * exp(-15.218 / T9);
    double term3 = 2.47e2 * pow(T9, 1.5) * exp(-15.147 / T9);
    double term4 = 1.72e-9 * T9_m32 * exp(-5.028 / T9);
    double term5 = 1.25e-3 * T9_m32 * exp(-7.929 / T9);
    double term6 = 2.43e1 / T9 * exp(-11.058 / T9);
    return term1 + term2 + term3 + term4 + term5 + term6;
}

// Generic statistical-model α-capture rate for Si28 and heavier.
// Form: N_A <σv> ≈ A_fac · T₉^(-2/3) · exp(-B_gamow/T₉^(1/3)) plus resonant
// low-energy term.
// Fit values from Hashimoto+1989 / Thielemann+1987 at T₉ = 2–5 GK.
//
// This is a reasonable approximation for T₉ in [1, 6]; precision is ~factor 2
// which is sufficient for α-freeze-out mass fractions (the endpoint is set by
// the chain structure + detailed balance, not individual rates).

ANET_HD inline double _statmodel_rate(double T9, double A_fac, double B_gamow,
                                       double low_T_prefac, double low_T_B) {
    if (T9 < 1e-3) return 0.0;
    double T9_m13 = pow(T9, -1.0/3.0);
    double T9_m23 = T9_m13 * T9_m13;
    double T9_m32 = pow(T9, -1.5);
    double high = A_fac * T9_m23 * exp(-B_gamow * T9_m13);
    double low  = low_T_prefac * T9_m32 * exp(-low_T_B / T9);
    return high + low;
}

// R5: ²⁸Si(α,γ)³²S      Gamow exponent ≈ 59.5  (Z₁=14, A_eff=3.5)
ANET_HD inline double rate_si28_ag(double T9) {
    return _statmodel_rate(T9, 2.5e7, 59.5, 7.50e2, 14.8);
}
// R6: ³²S(α,γ)³⁶Ar      Gamow ≈ 65.4
ANET_HD inline double rate_s32_ag(double T9) {
    return _statmodel_rate(T9, 2.7e7, 65.4, 7.50e2, 15.0);
}
// R7: ³⁶Ar(α,γ)⁴⁰Ca     Gamow ≈ 70.9
ANET_HD inline double rate_ar36_ag(double T9) {
    return _statmodel_rate(T9, 3.0e7, 70.9, 7.50e2, 15.3);
}
// R8: ⁴⁰Ca(α,γ)⁴⁴Ti     Gamow ≈ 76.4
ANET_HD inline double rate_ca40_ag(double T9) {
    return _statmodel_rate(T9, 3.3e7, 76.4, 5.00e2, 15.8);
}
// R9: ⁴⁴Ti(α,γ)⁴⁸Cr     Gamow ≈ 81.6
ANET_HD inline double rate_ti44_ag(double T9) {
    return _statmodel_rate(T9, 3.8e7, 81.6, 8.00e2, 16.1);
}
// R10: ⁴⁸Cr(α,γ)⁵²Fe    Gamow ≈ 86.7
ANET_HD inline double rate_cr48_ag(double T9) {
    return _statmodel_rate(T9, 4.2e7, 86.7, 1.00e3, 16.3);
}
// R11: ⁵²Fe(α,γ)⁵⁶Ni    Gamow ≈ 91.7
ANET_HD inline double rate_fe52_ag(double T9) {
    return _statmodel_rate(T9, 4.6e7, 91.7, 1.20e3, 16.5);
}

// R12: ¹⁶O + ¹⁶O → ²⁸Si + α  [CF88 heavy-ion]
ANET_HD inline double rate_o16_o16(double T9) {
    if (T9 < 1e-3) return 0.0;
    double T9_m13 = pow(T9, -1.0/3.0);
    double exp_arg = -135.93 * T9_m13 - 0.629 * pow(T9, 2.0/3.0)
                    - 0.445 * pow(T9, 4.0/3.0) + 0.0103 * T9 * T9;
    if (exp_arg < -700.0) return 0.0;
    double prefac = 7.10e36 * T9_m13 * T9_m13;
    return prefac * exp(exp_arg);
}

// R13: ¹²C + ¹²C → ²⁰Ne + α  [CF88]
ANET_HD inline double rate_c12_c12(double T9) {
    if (T9 < 1e-3) return 0.0;
    double T9a = T9 / (1.0 + 0.0396 * T9);
    double T9a_56 = pow(T9a, 5.0/6.0);
    double T9_m32 = pow(T9, -1.5);
    double exp_arg = -84.165 / pow(T9a, 1.0/3.0) - 2.12e-3 * T9 * T9 * T9;
    if (exp_arg < -700.0) return 0.0;
    double prefac = 4.27e26 * T9a_56 * T9_m32;
    return prefac * exp(exp_arg);
}

// R14: ¹²C + ¹⁶O → ²⁴Mg + α  [CF88]
ANET_HD inline double rate_c12_o16(double T9) {
    if (T9 < 1e-3 || T9 > 12.0) return 0.0;
    double lnT9 = log(T9);
    double T9_m13 = pow(T9, -1.0/3.0);
    double exp_arg = -106.594 / pow(T9, 1.0/3.0) - (T9 / 2.969) * (T9 / 2.969)
                   - 0.18 * T9 * T9;
    if (exp_arg < -700.0) return 0.0;
    double prefac = 1.72e31 * T9_m13 * T9_m13 * exp(-0.18 * T9 * T9);
    (void)lnT9;
    return prefac * exp(-106.594 * T9_m13);
}

// ──────────────────────────────────────────────────────────────────────────────
// Detailed-balance photodisintegration rates.
// For forward  A + α → B + γ  with Q_MeV > 0,
//   λ_γ(B→A+α) = 9.8678e9 · (A_α · A_A / A_B)^{3/2} · T₉^{3/2}
//                · exp(-11.605 · Q_MeV / T₉) · λ_fwd(T₉)        [1/s]
// (ground-state partition functions, G_ratio ≈ 1 for even-even α-chain)
//
// For 3α reverse  ¹²C → 3α  (3-body),
//   λ_γ(C12→3α) = 5.20e18 · T₉³ · exp(-84.4/T₉)                  [1/s]

ANET_HD inline double _photo_prefac(double A_alpha, double A_A, double A_B) {
    // 9.8678e9 * (A_α * A_A / A_B)^{3/2}
    double r = A_alpha * A_A / A_B;
    return 9.8678e9 * r * sqrt(r);
}

ANET_HD inline double photo_rate(double T9, double A_alpha, double A_A,
                                  double A_B, double Q_MeV, double lam_fwd) {
    if (T9 < 1e-3) return 0.0;
    double arg = -11.605 * Q_MeV / T9;
    if (arg < -700.0) return 0.0;
    return _photo_prefac(A_alpha, A_A, A_B) * pow(T9, 1.5) * exp(arg) * lam_fwd;
}

// ¹²C → 3α  (Timmes aprox13 approx; Fowler 1964)
ANET_HD inline double photo_3a(double T9) {
    if (T9 < 1e-3) return 0.0;
    // Rate from Fowler & Hoyle 1964, fitted:
    //   λ_γ(C12→3α) ≈ 2.0e20 · T₉³ · exp(-84.42/T9)  [1/s]
    double arg = -84.42 / T9;
    if (arg < -700.0) return 0.0;
    return 2.0e20 * T9 * T9 * T9 * exp(arg);
}

// ──────────────────────────────────────────────────────────────────────────────
// RHS — dY_i/dt given current Y[13] and (ρ, T).
// Y_i = X_i / A_i  (molar abundance).

ANET_HD inline void rhs(const double Y[N_SPEC], double rho, double T,
                        double dYdt[N_SPEC], double* eps_out) {
    double T9 = T * 1e-9;
    if (T9 < 1e-4) {
        for (int i = 0; i < N_SPEC; ++i) dYdt[i] = 0.0;
        if (eps_out) *eps_out = 0.0;
        return;
    }

    // Forward rates
    double lf0  = rate_3a     (T9);
    double lf1  = rate_c12_ag (T9);
    double lf2  = rate_o16_ag (T9);
    double lf3  = rate_ne20_ag(T9);
    double lf4  = rate_mg24_ag(T9);
    double lf5  = rate_si28_ag(T9);
    double lf6  = rate_s32_ag (T9);
    double lf7  = rate_ar36_ag(T9);
    double lf8  = rate_ca40_ag(T9);
    double lf9  = rate_ti44_ag(T9);
    double lf10 = rate_cr48_ag(T9);
    double lf11 = rate_fe52_ag(T9);
    double lf12 = rate_o16_o16(T9);
    double lf13 = rate_c12_c12(T9);
    double lf14 = rate_c12_o16(T9);

    // Reverse (photodisintegration) rates, detailed balance
    double lr0  = photo_3a(T9);
    double lr1  = photo_rate(T9, 4.0, 12.0, 16.0, Q_CAG,  lf1);
    double lr2  = photo_rate(T9, 4.0, 16.0, 20.0, Q_OAG,  lf2);
    double lr3  = photo_rate(T9, 4.0, 20.0, 24.0, Q_NEAG, lf3);
    double lr4  = photo_rate(T9, 4.0, 24.0, 28.0, Q_MGAG, lf4);
    double lr5  = photo_rate(T9, 4.0, 28.0, 32.0, Q_SIAG, lf5);
    double lr6  = photo_rate(T9, 4.0, 32.0, 36.0, Q_SAG,  lf6);
    double lr7  = photo_rate(T9, 4.0, 36.0, 40.0, Q_ARAG, lf7);
    double lr8  = photo_rate(T9, 4.0, 40.0, 44.0, Q_CAAG, lf8);
    double lr9  = photo_rate(T9, 4.0, 44.0, 48.0, Q_TIAG, lf9);
    double lr10 = photo_rate(T9, 4.0, 48.0, 52.0, Q_CRAG, lf10);
    double lr11 = photo_rate(T9, 4.0, 52.0, 56.0, Q_FEAG, lf11);

    // Forward reaction fluxes r_i [mol/g/s]
    //   binary with distinct reactants:  r = ρ · λ · Y_a · Y_b
    //   binary with identical reactants: r = 0.5 · ρ · λ · Y² (symmetry factor)
    //   triple α (3-body):               r = (1/6) · ρ² · λ · Y_α³
    double rf0  = (1.0/6.0) * rho * rho * lf0 * Y[HE4] * Y[HE4] * Y[HE4];
    double rf1  = rho * lf1  * Y[C12]  * Y[HE4];
    double rf2  = rho * lf2  * Y[O16]  * Y[HE4];
    double rf3  = rho * lf3  * Y[NE20] * Y[HE4];
    double rf4  = rho * lf4  * Y[MG24] * Y[HE4];
    double rf5  = rho * lf5  * Y[SI28] * Y[HE4];
    double rf6  = rho * lf6  * Y[S32]  * Y[HE4];
    double rf7  = rho * lf7  * Y[AR36] * Y[HE4];
    double rf8  = rho * lf8  * Y[CA40] * Y[HE4];
    double rf9  = rho * lf9  * Y[TI44] * Y[HE4];
    double rf10 = rho * lf10 * Y[CR48] * Y[HE4];
    double rf11 = rho * lf11 * Y[FE52] * Y[HE4];
    double rf12 = 0.5 * rho * lf12 * Y[O16] * Y[O16];
    double rf13 = 0.5 * rho * lf13 * Y[C12] * Y[C12];
    double rf14 = rho * lf14 * Y[C12] * Y[O16];

    // Reverse fluxes [mol/g/s]:  photodisintegration is 1-body (no ρ factor).
    double rr0  = lr0  * Y[C12];   // C12 → 3α (3-body products: ΔY_α = +3, ΔY_C = -1)
    double rr1  = lr1  * Y[O16];
    double rr2  = lr2  * Y[NE20];
    double rr3  = lr3  * Y[MG24];
    double rr4  = lr4  * Y[SI28];
    double rr5  = lr5  * Y[S32];
    double rr6  = lr6  * Y[AR36];
    double rr7  = lr7  * Y[CA40];
    double rr8  = lr8  * Y[TI44];
    double rr9  = lr9  * Y[CR48];
    double rr10 = lr10 * Y[FE52];
    double rr11 = lr11 * Y[NI56];

    // Net r_i = forward - reverse
    double r0  = rf0  - rr0;
    double r1  = rf1  - rr1;
    double r2  = rf2  - rr2;
    double r3  = rf3  - rr3;
    double r4  = rf4  - rr4;
    double r5  = rf5  - rr5;
    double r6  = rf6  - rr6;
    double r7  = rf7  - rr7;
    double r8  = rf8  - rr8;
    double r9  = rf9  - rr9;
    double r10 = rf10 - rr10;
    double r11 = rf11 - rr11;

    // Species production/destruction
    // R0 (3α):   dY_α = -3 r0,  dY_C = +r0
    // Ri α-cap:  dY_reactant = -r,  dY_α = -r,  dY_product = +r
    // R12 (O+O→Si+α):  dY_O -= 2·rf12,  dY_Si += rf12,  dY_α += rf12
    // R13 (C+C→Ne+α):  dY_C -= 2·rf13,  dY_Ne += rf13,  dY_α += rf13
    // R14 (C+O→Mg+α):  dY_C -= rf14,  dY_O -= rf14,  dY_Mg += rf14,  dY_α += rf14

    dYdt[HE4]  = -3.0 * r0 - r1 - r2 - r3 - r4 - r5 - r6 - r7 - r8 - r9 - r10 - r11
               + rf12 + rf13 + rf14;
    dYdt[C12]  =  r0 - r1 - 2.0 * rf13 - rf14;
    dYdt[O16]  =  r1 - r2 - 2.0 * rf12 - rf14;
    dYdt[NE20] =  r2 - r3 + rf13;
    dYdt[MG24] =  r3 - r4 + rf14;
    dYdt[SI28] =  r4 - r5 + rf12;
    dYdt[S32]  =  r5 - r6;
    dYdt[AR36] =  r6 - r7;
    dYdt[CA40] =  r7 - r8;
    dYdt[TI44] =  r8 - r9;
    dYdt[CR48] =  r9 - r10;
    dYdt[FE52] =  r10 - r11;
    dYdt[NI56] =  r11;

    // Energy release (forward - reverse, so net; reverse is endothermic)
    if (eps_out) {
        double eps = 0.0;
        eps += Q_3A    * r0  * N_A * MEV_TO_ERG;
        eps += Q_CAG   * r1  * N_A * MEV_TO_ERG;
        eps += Q_OAG   * r2  * N_A * MEV_TO_ERG;
        eps += Q_NEAG  * r3  * N_A * MEV_TO_ERG;
        eps += Q_MGAG  * r4  * N_A * MEV_TO_ERG;
        eps += Q_SIAG  * r5  * N_A * MEV_TO_ERG;
        eps += Q_SAG   * r6  * N_A * MEV_TO_ERG;
        eps += Q_ARAG  * r7  * N_A * MEV_TO_ERG;
        eps += Q_CAAG  * r8  * N_A * MEV_TO_ERG;
        eps += Q_TIAG  * r9  * N_A * MEV_TO_ERG;
        eps += Q_CRAG  * r10 * N_A * MEV_TO_ERG;
        eps += Q_FEAG  * r11 * N_A * MEV_TO_ERG;
        eps += Q_OO    * rf12 * N_A * MEV_TO_ERG;
        eps += Q_CC    * rf13 * N_A * MEV_TO_ERG;
        eps += Q_CO    * rf14 * N_A * MEV_TO_ERG;
        *eps_out = eps;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// X ↔ Y helpers

ANET_HD inline void X_to_Y(const double X[N_SPEC], double Y[N_SPEC]) {
    for (int i = 0; i < N_SPEC; ++i) Y[i] = X[i] / A_NUC[i];
}

ANET_HD inline void Y_to_X(const double Y[N_SPEC], double X[N_SPEC]) {
    for (int i = 0; i < N_SPEC; ++i) X[i] = Y[i] * A_NUC[i];
}

// ──────────────────────────────────────────────────────────────────────────────
// Advance mass fraction X[] by dt at fixed (ρ, T).
// Substepped forward Euler with adaptive dt limited to 5% per sub-step.
// With reverse reactions included, also clamp for reaction-balance stability.

ANET_HD inline double advance_substep(double X[N_SPEC], double rho, double T,
                                       double dt, int max_substeps = 200000) {
    double Y[N_SPEC];
    X_to_Y(X, Y);

    double t = 0.0;
    double eps_total = 0.0;
    double safety = 0.05;

    for (int step = 0; step < max_substeps; ++step) {
        double dYdt[N_SPEC];
        double eps;
        rhs(Y, rho, T, dYdt, &eps);

        // Adaptive dt
        double dt_try = dt - t;
        for (int i = 0; i < N_SPEC; ++i) {
            if (Y[i] > 1e-20 && fabs(dYdt[i]) > 0.0) {
                double dt_i = safety * Y[i] / fabs(dYdt[i]);
                if (dt_i < dt_try) dt_try = dt_i;
            }
        }
        if (dt_try <= 0.0) dt_try = (dt - t) * 1e-3;
        if (t + dt_try > dt) dt_try = dt - t;

        // Forward Euler
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
