#pragma once

// pp-chain nuclear energy generation rate (simplified).
//
// Net reaction: 4 p → ⁴He + 2 e⁺ + 2 νe + 26.73 MeV
// ε_pp [erg/g/s] = 2.57e4 · ψ · f_11 · X² · ρ · T₉^{-2/3} · exp(-3.381 / T₉^{1/3})
//
// where:
//   X = hydrogen mass fraction (0.7 solar)
//   T₉ = T / 10⁹ K
//   ψ, f_11 ≈ 1 for our purposes (no screening, no higher corrections)
//
// Reference: Kippenhahn, Weigert & Weiss 2012, §18.5 eq 18.63
// Also Angulo+ 1999 NACRE rates (this simplified form is the standard
// undergraduate fit, accurate to ~5% in the T ~ 10⁷ K range).
//
// GPU __host__ __device__.
// Reusable by Path B.

#include <cmath>

#ifdef __CUDACC__
#define NUC_HD __host__ __device__
#else
#define NUC_HD
#endif

struct NuclearPPParams {
    double X_hydrogen = 0.7;       // hydrogen mass fraction
    double T_floor = 1.0e6;         // below this, ε = 0 (avoid divide-by-small)
    double epsilon_scale = 1.0;     // code-unit scaling factor (ε_code = ε_cgs · scale)
    // If running in physical cgs, epsilon_scale = 1. Otherwise user sets so that
    // d e_int / dt [code units] = ε_pp * rho in code units consistently.
};

// Specific energy generation rate ε_pp(ρ, T) in whatever units rho/T are in
// (results need epsilon_scale to convert to code units).
NUC_HD inline double nuclear_pp_epsilon(double rho, double T, const NuclearPPParams& p) {
    if (T < p.T_floor) return 0.0;
    double T9 = T * 1.0e-9;
    if (T9 < 1.0e-6) return 0.0;
    double T9_13 = pow(T9, 1.0/3.0);
    double T9_m23 = pow(T9, -2.0/3.0);
    double exp_arg = -3.381 / T9_13;
    // Underflow guard
    if (exp_arg < -700.0) return 0.0;
    double eps = 2.57e4 * p.X_hydrogen * p.X_hydrogen
               * rho * T9_m23 * exp(exp_arg);
    return eps * p.epsilon_scale;
}

// Approximate d ε / d T (for implicit stability / info only)
NUC_HD inline double nuclear_pp_dedT(double rho, double T, const NuclearPPParams& p) {
    if (T < p.T_floor) return 0.0;
    double eps = nuclear_pp_epsilon(rho, T, p);
    double T9 = T * 1.0e-9;
    // d/dT[T⁻²/³ · exp(-3.381/T¹/³)] = ... · (-2/(3T)) + · (3.381/(3T⁴/³))
    // net: d ln ε / dT ≈ (−2/3 + 3.381/(3·T₉^{1/3})) / T
    double d_ln_eps = (-2.0/3.0 + 3.381 / (3.0 * pow(T9, 1.0/3.0))) / T;
    return eps * d_ln_eps;
}

#undef NUC_HD
