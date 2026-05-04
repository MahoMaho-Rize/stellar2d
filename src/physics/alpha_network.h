#pragma once
//
// α-chain nuclear network — Phase C (13 species, aprox13-identical rates).
//
// This is a direct C++ port of the α-chain reaction rates used in
// AMReX-Astro Microphysics (networks/aprox13, rates/aprox_rates.H), which
// itself is the long-standing Timmes aprox13 implementation used in the
// published explosive-nucleosynthesis literature (Tur+2007, Magkotsios+2010,
// Sato & Suwa 2023, and many others).
//
// Source: https://github.com/AMReX-Astro/Microphysics rates/aprox_rates.H
// License: BSD-3-Clause (see third_party/amrex_microphysics/LICENSE)
//
// Species (by index):
//   0:  ⁴He    1:  ¹²C     2:  ¹⁶O    3:  ²⁰Ne   4:  ²⁴Mg
//   5:  ²⁸Si   6:  ³²S     7:  ³⁶Ar   8:  ⁴⁰Ca   9:  ⁴⁴Ti
//  10:  ⁴⁸Cr  11:  ⁵²Fe   12:  ⁵⁶Ni
//
// Reactions (forward + reverse):
//   R0:   3 ⁴He ↔ ¹²C                (triple-α + γ→3α)
//   R1:   ¹²C  + ⁴He ↔ ¹⁶O
//   R2:   ¹⁶O  + ⁴He ↔ ²⁰Ne
//   R3:   ²⁰Ne + ⁴He ↔ ²⁴Mg
//   R4:   ²⁴Mg + ⁴He ↔ ²⁸Si
//   R5:   ²⁸Si + ⁴He ↔ ³²S
//   R6:   ³²S  + ⁴He ↔ ³⁶Ar
//   R7:   ³⁶Ar + ⁴He ↔ ⁴⁰Ca
//   R8:   ⁴⁰Ca + ⁴He ↔ ⁴⁴Ti
//   R9:   ⁴⁴Ti + ⁴He ↔ ⁴⁸Cr
//   R10:  ⁴⁸Cr + ⁴He ↔ ⁵²Fe
//   R11:  ⁵²Fe + ⁴He ↔ ⁵⁶Ni
//   R12:  ¹⁶O + ¹⁶O → ²⁸Si + α (no reverse)
//   R13:  ¹²C + ¹²C → ²⁰Ne + α (no reverse)
//   R14:  ¹²C + ¹⁶O → ²⁴Mg + α (no reverse, below T₉=0.5 turned off)
//
// The aprox13 rates embed:
//   - C12(α,γ)O16 with "1.7 × CF88" renormalisation (standard in Timmes+)
//   - Hashimoto+1989 statistical-model fits for Si+α through Fe+α
//   - Inline reverse rates computed using detailed balance with numeric
//     prefactors from Iliadis 2007 (e.g., 5.13e10 for O16 reverse)
//
// Integration: substepped forward-Euler with adaptive dt (5% max change
// per sub-step).  Stiff regions (T₉ > 4) require many sub-steps but converge.
//
// GPU __host__ __device__.

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

// Q-values in MeV (NNDC AME2020).  Used ONLY for the energy-release diagnostic.
constexpr double Q_3A     =  7.275;
constexpr double Q_CAG    =  7.162;
constexpr double Q_OAG    =  4.730;
constexpr double Q_NEAG   =  9.316;
constexpr double Q_MGAG   =  9.984;
constexpr double Q_SIAG   =  6.948;
constexpr double Q_SAG    =  6.645;
constexpr double Q_ARAG   =  7.041;
constexpr double Q_CAAG   =  5.128;
constexpr double Q_TIAG   =  7.695;
constexpr double Q_CRAG   =  7.941;
constexpr double Q_FEAG   =  7.990;
constexpr double Q_OO     = 16.542;
constexpr double Q_CC     =  4.616;
constexpr double Q_CO     = 16.753;

constexpr double MEV_TO_ERG = 1.602176634e-6;
constexpr double N_A        = 6.02214076e23;

// ──────────────────────────────────────────────────────────────────────────────
// Temperature-factor struct (matches aprox13 tf_t{}).
// t9, t9i, t9i13, t913, t923, t932, t9i23, t9i32, t92, t93, t95, t912
// Used to avoid recomputing pow() in every rate.

struct TFactors {
    double t9;
    double t9i;   double t9i13; double t9i23; double t9i32;
    double t913;  double t923;  double t932;  double t912;
    double t92;   double t93;   double t95;   double t943;
    double t953;
};

ANET_HD inline TFactors make_tfactors(double T_K) {
    TFactors tf;
    tf.t9    = T_K * 1e-9;
    double t9 = tf.t9;
    tf.t9i   = 1.0 / t9;
    tf.t913  = cbrt(t9);
    tf.t923  = tf.t913 * tf.t913;
    tf.t9i13 = 1.0 / tf.t913;
    tf.t9i23 = 1.0 / tf.t923;
    tf.t912  = sqrt(t9);
    tf.t932  = t9 * tf.t912;
    tf.t9i32 = 1.0 / tf.t932;
    tf.t92   = t9 * t9;
    tf.t93   = tf.t92 * t9;
    tf.t95   = tf.t92 * tf.t93;
    tf.t943  = tf.t913 * t9;
    tf.t953  = tf.t923 * t9;
    return tf;
}

// ──────────────────────────────────────────────────────────────────────────────
// Rate functions — direct port of AMReX Microphysics rates/aprox_rates.H.
// Each returns (fr, rr) in the same conventions as aprox13:
//   fr = forward rate per unit mass of each reactant (includes den factor for
//        binary reactions)
//   rr = reverse rate (photodisintegration)
//
// For α-capture  A + α → B:
//   fr [s⁻¹·g/mol]  — binary rate, to be multiplied by Y_A · Y_α
//   rr [s⁻¹]        — photo rate on B, to be multiplied by Y_B
//
// For triple-α:
//   fr already has den² (so fr·Y_α³ /6 gives reaction flux)
//   rr [s⁻¹] photo on C12

// R0: triple-α (both forward and reverse)
ANET_HD inline void rate_3a(const TFactors& tf, double den,
                             double& fr, double& rr) {
    constexpr double rc28 = 0.1;
    constexpr double q1 = 1.0/0.009604;
    constexpr double q2 = 1.0/0.055225;

    double aa = 7.40e5 * tf.t9i32 * exp(-1.0663 * tf.t9i);
    double bb = 4.164e9 * tf.t9i23
              * exp(-13.49 * tf.t9i13 - tf.t92 * q1);
    double cc = 1.0 + 0.031*tf.t913 + 8.009*tf.t923 + 1.732*tf.t9
              + 49.883*tf.t943 + 27.426*tf.t953;
    double r2abe = aa + bb * cc;

    double dd = 130.0 * tf.t9i32 * exp(-3.3364 * tf.t9i);
    double ee = 2.510e7 * tf.t9i23
              * exp(-23.57 * tf.t9i13 - tf.t92 * q2);
    double ff = 1.0 + 0.018*tf.t913 + 5.249*tf.t923 + 0.650*tf.t9
              + 19.176*tf.t943 + 6.034*tf.t953;
    double rbeac = dd + ee * ff;

    double xx = rc28 * 1.35e-07 * tf.t9i32 * exp(-24.811 * tf.t9i);

    double term;
    if (tf.t9 > 0.08) {
        term = 2.90e-16 * r2abe * rbeac + xx;
    } else {
        double uu = 0.8 * exp(-pow(0.025 * tf.t9i, 3.263));
        double yy = 0.2 + uu;
        double vv = 4.0 * exp(-pow(tf.t9 / 0.025, 9.227));
        double zz = 1.0 + vv;
        double f1 = 0.01 + yy / zz;
        term = 2.90e-16 * r2abe * rbeac * f1 + xx;
    }

    fr = term * den * den;  // den² for 3-body
    double rev = 2.00e20 * tf.t93 * exp(-84.424 * tf.t9i);
    rr = rev * term;        // per-Y_C12; 1-body
}

// R1: C12(α,γ)O16  [aprox13 CF88 × 1.7]
ANET_HD inline void rate_c12ag(const TFactors& tf, double den,
                                double& fr, double& rr) {
    constexpr double q1 = 1.0/12.222016;

    double aa = 1.0 + 0.0489 * tf.t9i23;
    double bb = tf.t92 * aa * aa;
    double cc = exp(-32.120*tf.t9i13 - tf.t92*q1);
    double dd = 1.0 + 0.2654 * tf.t9i23;
    double ee = tf.t92 * dd * dd;
    double ff = exp(-32.120 * tf.t9i13);
    double gg = 1.25e3 * tf.t9i32 * exp(-27.499 * tf.t9i);
    double hh = 1.43e-2 * tf.t95 * exp(-15.541 * tf.t9i);

    double f1 = cc / bb;
    double f2 = ff / ee;
    double term = 1.04e8*f1 + 1.76e8*f2 + gg + hh;
    term *= 1.7;  // aprox13 renormalisation

    fr = term * den;
    double rev = 5.13e10 * tf.t932 * exp(-83.111 * tf.t9i);
    rr = rev * term;
}

// R2: O16(α,γ)Ne20  [aprox13 CF88]
ANET_HD inline void rate_o16ag(const TFactors& tf, double den,
                                double& fr, double& rr) {
    constexpr double q1 = 1.0/2.515396;

    double term1 = 9.37e9 * tf.t9i23 * exp(-39.757*tf.t9i13 - tf.t92*q1);
    double aa = 62.1 * tf.t9i32 * exp(-10.297 * tf.t9i);
    double bb = 538.0 * tf.t9i32 * exp(-12.226 * tf.t9i);
    double cc = 13.0 * tf.t92 * exp(-20.093 * tf.t9i);
    double term = term1 + aa + bb + cc;

    fr = term * den;
    double rev = 5.65e10 * tf.t932 * exp(-54.937 * tf.t9i);
    rr = rev * term;
}

// R3: Ne20(α,γ)Mg24  [aprox13 CF88]
ANET_HD inline void rate_ne20ag(const TFactors& tf, double den,
                                 double& fr, double& rr) {
    constexpr double rc102 = 0.1;
    constexpr double q1 = 1.0/4.923961;

    double aa = 4.11e11 * tf.t9i23 * exp(-46.766*tf.t9i13 - tf.t92*q1);
    double bb = 1.0 + 0.009*tf.t913 + 0.882*tf.t923 + 0.055*tf.t9
              + 0.749*tf.t943 + 0.119*tf.t953;
    double term1 = aa * bb;

    aa = 5.27e3 * tf.t9i32 * exp(-15.869 * tf.t9i);
    bb = 6.51e3 * tf.t912 * exp(-16.223 * tf.t9i);
    double term2 = aa + bb;

    aa = 42.1 * tf.t9i32 * exp(-9.115 * tf.t9i);
    bb = 32.0 * tf.t9i23 * exp(-9.383 * tf.t9i);
    double term3 = rc102 * (aa + bb);

    aa = 5.0 * exp(-18.960 * tf.t9i);
    bb = 1.0 + aa;
    double term = (term1 + term2 + term3) / bb;

    fr = term * den;
    double rev = 6.01e10 * tf.t932 * exp(-108.059 * tf.t9i);
    rr = rev * term;
}

// R4: Mg24(α,γ)Si28
ANET_HD inline void rate_mg24ag(const TFactors& tf, double den,
                                 double& fr, double& rr) {
    constexpr double rc121 = 0.1;

    double aa = 4.78e1 * tf.t9i32 * exp(-13.506 * tf.t9i);
    double bb = 2.38e3 * tf.t9i32 * exp(-15.218 * tf.t9i);
    double cc = 2.47e2 * tf.t932 * exp(-15.147 * tf.t9i);
    double dd = rc121 * 1.72e-9 * tf.t9i32 * exp(-5.028 * tf.t9i);
    double ee = rc121 * 1.25e-3 * tf.t9i32 * exp(-7.929 * tf.t9i);
    double ff = rc121 * 2.43e1 * tf.t9i * exp(-11.523 * tf.t9i);
    double gg = 5.0 * exp(-15.882 * tf.t9i);
    double hh = 1.0 + gg;
    double term = (aa + bb + cc + dd + ee + ff) / hh;

    fr = term * den;
    double rev = 6.27e10 * tf.t932 * exp(-115.862 * tf.t9i);
    rr = rev * term;
}

// R5: Si28(α,γ)S32  [Hashimoto statmodel]
ANET_HD inline void rate_si28ag(const TFactors& tf, double den,
                                 double& fr, double& rr) {
    double z  = (tf.t9 < 10.0) ? tf.t9 : 10.0;
    double z2 = z*z, z3 = z2*z;
    double aa = 1.0 + 6.340e-2*z + 2.541e-3*z2 - 2.900e-4*z3;
    double term = 4.82e22 * tf.t9i23 * exp(-61.015 * tf.t9i13 * aa);
    fr = term * den;
    double rev = 6.461e10 * tf.t932 * exp(-80.643 * tf.t9i);
    rr = rev * term;
}

// R6: S32(α,γ)Ar36
ANET_HD inline void rate_s32ag(const TFactors& tf, double den,
                                double& fr, double& rr) {
    double z  = (tf.t9 < 10.0) ? tf.t9 : 10.0;
    double z2 = z*z, z3 = z2*z;
    double aa = 1.0 + 4.913e-2*z + 4.637e-3*z2 - 4.067e-4*z3;
    double term = 1.16e24 * tf.t9i23 * exp(-66.690 * tf.t9i13 * aa);
    fr = term * den;
    double rev = 6.616e10 * tf.t932 * exp(-77.080 * tf.t9i);
    rr = rev * term;
}

// R7: Ar36(α,γ)Ca40
ANET_HD inline void rate_ar36ag(const TFactors& tf, double den,
                                 double& fr, double& rr) {
    double z  = (tf.t9 < 10.0) ? tf.t9 : 10.0;
    double z2 = z*z, z3 = z2*z;
    double aa = 1.0 + 1.458e-1*z - 1.069e-2*z2 + 3.790e-4*z3;
    double term = 2.81e30 * tf.t9i23 * exp(-78.271 * tf.t9i13 * aa);
    fr = term * den;
    double rev = 6.740e10 * tf.t932 * exp(-81.711 * tf.t9i);
    rr = rev * term;
}

// R8: Ca40(α,γ)Ti44
ANET_HD inline void rate_ca40ag(const TFactors& tf, double den,
                                 double& fr, double& rr) {
    double z  = (tf.t9 < 10.0) ? tf.t9 : 10.0;
    double z2 = z*z, z3 = z2*z;
    double aa = 1.0 + 1.650e-2*z + 5.973e-3*z2 - 3.889e-4*z3;
    double term = 4.66e24 * tf.t9i23 * exp(-76.435 * tf.t9i13 * aa);
    fr = term * den;
    double rev = 6.843e10 * tf.t932 * exp(-59.510 * tf.t9i);
    rr = rev * term;
}

// R9: Ti44(α,γ)Cr48
ANET_HD inline void rate_ti44ag(const TFactors& tf, double den,
                                 double& fr, double& rr) {
    double z  = (tf.t9 < 10.0) ? tf.t9 : 10.0;
    double z2 = z*z, z3 = z2*z;
    double aa = 1.0 + 1.066e-1*z - 1.102e-2*z2 + 5.324e-4*z3;
    double term = 1.37e26 * tf.t9i23 * exp(-81.227 * tf.t9i13 * aa);
    fr = term * den;
    double rev = 6.928e10 * tf.t932 * exp(-89.289 * tf.t9i);
    rr = rev * term;
}

// R10: Cr48(α,γ)Fe52
ANET_HD inline void rate_cr48ag(const TFactors& tf, double den,
                                 double& fr, double& rr) {
    double z  = (tf.t9 < 10.0) ? tf.t9 : 10.0;
    double z2 = z*z, z3 = z2*z;
    double aa = 1.0 + 6.325e-2*z - 5.671e-3*z2 + 2.848e-4*z3;
    double term = 1.04e23 * tf.t9i23 * exp(-81.420 * tf.t9i13 * aa);
    fr = term * den;
    double rev = 7.001e10 * tf.t932 * exp(-92.177 * tf.t9i);
    rr = rev * term;
}

// R11: Fe52(α,γ)Ni56
ANET_HD inline void rate_fe52ag(const TFactors& tf, double den,
                                 double& fr, double& rr) {
    double z  = (tf.t9 < 10.0) ? tf.t9 : 10.0;
    double z2 = z*z, z3 = z2*z;
    double aa = 1.0 + 7.846e-2*z - 7.430e-3*z2 + 3.723e-4*z3;
    double term = 1.05e27 * tf.t9i23 * exp(-91.674 * tf.t9i13 * aa);
    fr = term * den;
    double rev = 7.064e10 * tf.t932 * exp(-92.850 * tf.t9i);
    rr = rev * term;
}

// R12: O16+O16 → Si28+α  (no reverse in aprox13)
ANET_HD inline void rate_o16o16(const TFactors& tf, double den,
                                 double& fr, double& rr) {
    double term = 7.10e36 * tf.t9i23
                * exp(-135.93 * tf.t9i13 - 0.629*tf.t923
                      - 0.445*tf.t943 + 0.0103*tf.t92);
    fr = term * den;
    rr = 0.0;
}

// R13: C12+C12 → Ne20+α
ANET_HD inline void rate_c12c12(const TFactors& tf, double den,
                                 double& fr, double& rr) {
    double aa = 1.0 + 0.0396*tf.t9;
    double t9a = tf.t9 / aa;
    double t9a13 = cbrt(t9a);
    double t9a56 = pow(t9a, 5.0/6.0);
    double term = 4.27e26 * t9a56 * tf.t9i32
                * exp(-84.165/t9a13 - 2.12e-3 * tf.t93);
    fr = term * den;
    rr = 0.0;
}

// R14: C12+O16 → Mg24+α  (zero below T₉=0.5 per aprox13)
ANET_HD inline void rate_c12o16(const TFactors& tf, double den,
                                 double& fr, double& rr) {
    if (tf.t9 < 0.5) { fr = 0.0; rr = 0.0; return; }
    double aa = 1.0 + 0.055 * tf.t9;
    double t9a = tf.t9 / aa;
    double t9a13 = cbrt(t9a);
    double t9a23 = t9a13 * t9a13;
    double t9a56 = pow(t9a, 5.0/6.0);
    double a  = exp(-0.18 * t9a * t9a);
    double b  = 1.06e-3 * exp(2.562 * t9a23);
    double c  = a + b;
    double term = 1.72e31 * t9a56 * tf.t9i32 * exp(-106.594 / t9a13) / c;
    fr = term * den;
    rr = 0.0;
}

// ──────────────────────────────────────────────────────────────────────────────
// RHS — dY/dt given Y and (ρ, T)

ANET_HD inline void rhs(const double Y[N_SPEC], double rho, double T,
                        double dYdt[N_SPEC], double* eps_out) {
    if (T < 1e5) {
        for (int i = 0; i < N_SPEC; ++i) dYdt[i] = 0.0;
        if (eps_out) *eps_out = 0.0;
        return;
    }

    TFactors tf = make_tfactors(T);

    double fr0, rr0, fr1, rr1, fr2, rr2, fr3, rr3, fr4, rr4, fr5, rr5;
    double fr6, rr6, fr7, rr7, fr8, rr8, fr9, rr9, fr10, rr10, fr11, rr11;
    double fr12, rr12, fr13, rr13, fr14, rr14;

    rate_3a     (tf, rho, fr0,  rr0);   // fr0 already contains ρ²
    rate_c12ag  (tf, rho, fr1,  rr1);
    rate_o16ag  (tf, rho, fr2,  rr2);
    rate_ne20ag (tf, rho, fr3,  rr3);
    rate_mg24ag (tf, rho, fr4,  rr4);
    rate_si28ag (tf, rho, fr5,  rr5);
    rate_s32ag  (tf, rho, fr6,  rr6);
    rate_ar36ag (tf, rho, fr7,  rr7);
    rate_ca40ag (tf, rho, fr8,  rr8);
    rate_ti44ag (tf, rho, fr9,  rr9);
    rate_cr48ag (tf, rho, fr10, rr10);
    rate_fe52ag (tf, rho, fr11, rr11);
    rate_o16o16 (tf, rho, fr12, rr12);
    rate_c12c12 (tf, rho, fr13, rr13);
    rate_c12o16 (tf, rho, fr14, rr14);

    // Reaction fluxes [mol/g/s]
    // 3α:  (1/6) · fr0 · Y_α³  forward;   rr0 · Y_C12  reverse
    double r0  = (1.0/6.0) * fr0 * Y[HE4]*Y[HE4]*Y[HE4] - rr0 * Y[C12];
    // A + α:  fr · Y_A · Y_α  forward;   rr · Y_B  reverse
    double r1  = fr1  * Y[C12]  * Y[HE4] - rr1  * Y[O16];
    double r2  = fr2  * Y[O16]  * Y[HE4] - rr2  * Y[NE20];
    double r3  = fr3  * Y[NE20] * Y[HE4] - rr3  * Y[MG24];
    double r4  = fr4  * Y[MG24] * Y[HE4] - rr4  * Y[SI28];
    double r5  = fr5  * Y[SI28] * Y[HE4] - rr5  * Y[S32];
    double r6  = fr6  * Y[S32]  * Y[HE4] - rr6  * Y[AR36];
    double r7  = fr7  * Y[AR36] * Y[HE4] - rr7  * Y[CA40];
    double r8  = fr8  * Y[CA40] * Y[HE4] - rr8  * Y[TI44];
    double r9  = fr9  * Y[TI44] * Y[HE4] - rr9  * Y[CR48];
    double r10 = fr10 * Y[CR48] * Y[HE4] - rr10 * Y[FE52];
    double r11 = fr11 * Y[FE52] * Y[HE4] - rr11 * Y[NI56];
    // Heavy-ion identical-particle: 0.5 · fr · Y²; no reverse
    double r12 = 0.5 * fr12 * Y[O16] * Y[O16];
    double r13 = 0.5 * fr13 * Y[C12] * Y[C12];
    double r14 =       fr14 * Y[C12] * Y[O16];

    // Species balance
    dYdt[HE4]  = -3.0*r0 - r1 - r2 - r3 - r4 - r5 - r6 - r7 - r8 - r9 - r10 - r11
               + r12 + r13 + r14;
    dYdt[C12]  =  r0 - r1 - 2.0*r13 - r14;
    dYdt[O16]  =  r1 - r2 - 2.0*r12 - r14;
    dYdt[NE20] =  r2 - r3 + r13;
    dYdt[MG24] =  r3 - r4 + r14;
    dYdt[SI28] =  r4 - r5 + r12;
    dYdt[S32]  =  r5 - r6;
    dYdt[AR36] =  r6 - r7;
    dYdt[CA40] =  r7 - r8;
    dYdt[TI44] =  r8 - r9;
    dYdt[CR48] =  r9 - r10;
    dYdt[FE52] =  r10 - r11;
    dYdt[NI56] =  r11;

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
        eps += Q_OO    * r12 * N_A * MEV_TO_ERG;
        eps += Q_CC    * r13 * N_A * MEV_TO_ERG;
        eps += Q_CO    * r14 * N_A * MEV_TO_ERG;
        *eps_out = eps;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// X ↔ Y helpers

ANET_HD inline double A_of(int i) {
    // Duplicate the table locally so device code doesn't reach out to a
    // host-only constexpr array. Kept in sync with A_NUC above by construction.
    constexpr double a[N_SPEC] = {
        4.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0,
    };
    return a[i];
}

ANET_HD inline void X_to_Y(const double X[N_SPEC], double Y[N_SPEC]) {
    for (int i = 0; i < N_SPEC; ++i) Y[i] = X[i] / A_of(i);
}

ANET_HD inline void Y_to_X(const double Y[N_SPEC], double X[N_SPEC]) {
    for (int i = 0; i < N_SPEC; ++i) X[i] = Y[i] * A_of(i);
}

// ──────────────────────────────────────────────────────────────────────────────
// Advance X by dt — substepped forward Euler, adaptive dt.

ANET_HD inline double advance_substep(double X[N_SPEC], double rho, double T,
                                       double dt, int max_substeps = 500000) {
    double Y[N_SPEC];
    X_to_Y(X, Y);

    double t = 0.0;
    double eps_total = 0.0;
    double safety = 0.05;

    for (int step = 0; step < max_substeps; ++step) {
        double dYdt[N_SPEC];
        double eps;
        rhs(Y, rho, T, dYdt, &eps);

        double dt_try = dt - t;
        for (int i = 0; i < N_SPEC; ++i) {
            if (Y[i] > 1e-20 && fabs(dYdt[i]) > 0.0) {
                double dt_i = safety * Y[i] / fabs(dYdt[i]);
                if (dt_i < dt_try) dt_try = dt_i;
            }
        }
        if (dt_try <= 0.0) dt_try = (dt - t) * 1e-3;
        if (t + dt_try > dt) dt_try = dt - t;

        for (int i = 0; i < N_SPEC; ++i) Y[i] += dYdt[i] * dt_try;
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
