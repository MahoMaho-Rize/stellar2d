#pragma once

// Stellar opacity (grey, Rosseland mean).
//
// Physical regimes:
//   - Low T (T < 1000 K): dust / molecular absorption. Power law κ ∝ T²
//   - 1000 < T < 10000 K: H⁻ bound-free, κ dominated by Bell-Lin fit
//   - T > 10000 K: electron scattering + Kramers (free-free + bound-free)
//                  κ_ff ∝ ρ · T^{-3.5}
//                  κ_es ≈ 0.2 (1 + X) cm²/g  (constant)
//
// For pre-MS protostellar physics we bridge all three. Use simplified
// multi-component fit (following Semenov et al. 2003 for dust, Iglesias-Rogers
// style for high T).
//
// All GPU __host__ __device__.

#include <cmath>
#ifdef USE_GPU
#include "opacity_table.cuh"
#endif

#ifdef __CUDACC__
#define OPA_HD __host__ __device__
#else
#define OPA_HD
#endif

struct OpacityParams {
    // Electron scattering floor
    double kappa_es = 0.2;
    // Kramers free-free coefficient: κ_ff = kappa_ff_0 · ρ · T^{-3.5}
    double kappa_ff_0 = 4.3e24;
    // Low-T dust (Semenov-like): κ_dust = kappa_dust_0 · T²  for T < T_dust
    double kappa_dust_0 = 2.0e-4;
    double T_dust_off = 1500.0;   // dust sublimation
    // Hydrogen scattering off (H⁻ bound-free): κ_H- = kappa_Hm_0 · √ρ · T^7.7 / (...)
    double kappa_Hm_0 = 1.1e-25;
    // Simpler: use piecewise maximum of the four components.
    // User must have T, ρ in physical cgs or code-unit-consistent system.

#ifdef USE_GPU
    // Tabulated MESA-style opacity, takes priority over the analytic fallback
    // when `use_table` is true and both view pointers are non-null.
    //
    // Stitching: if logT ≤ logT_lo_end use `table_lowT` only; if
    // logT ≥ logT_hi_start use `table_highT` only; otherwise take the
    // harmonic mean of the two (equivalent to MESA's "min of logκ" overlap
    // choice but smoother and C⁰ at the seams). Default seams match the
    // MESA kap Type-1 / Ferguson overlap (3.9, 4.1 dex).
    KapTableView table_lowT = {};
    KapTableView table_highT = {};
    double hydrogen_X = 0.7;         // composition used to slice the 3-D table
    double logT_lo_end = 3.9;        // upper edge of the pure-lowT regime
    double logT_hi_start = 4.1;      // lower edge of the pure-highT regime
    bool   use_table = false;
#endif
};

#ifdef USE_GPU
// Stitched table lookup. Assumes both views point at valid device memory.
// Regime selection follows MESA's Kap_Type1 + lowT split:
//   logT ≤ logT_lo_end        → pure lowT (Ferguson)
//   logT ≥ logT_hi_start      → pure highT (OPAL / OPLIB / OP)
//   lo_end < logT < hi_start  → C⁰ blend by logT fraction in logκ-space
// That last branch is what MESA's `kap_eval_blend_logT` does; the linear
// blend in log space is the same recipe.
OPA_HD inline double kap_stitch_eval(double rho, double T,
                                     const OpacityParams& p) {
    double T_use = T > 1.0 ? T : 1.0;
    double rho_use = rho > 1e-30 ? rho : 1e-30;
    double logT = log10(T_use);
    double logR = log10(rho_use) - 3.0 * logT + 18.0;

    double k_lo = 0.0, k_hi = 0.0;
    if (logT <= p.logT_hi_start) {
        k_lo = kap_eval(p.table_lowT,  p.hydrogen_X, logT, logR);
    }
    if (logT >= p.logT_lo_end) {
        k_hi = kap_eval(p.table_highT, p.hydrogen_X, logT, logR);
    }
    if (logT <= p.logT_lo_end) return k_lo;
    if (logT >= p.logT_hi_start) return k_hi;

    double w = (logT - p.logT_lo_end) / (p.logT_hi_start - p.logT_lo_end);
    if (w < 0.0) w = 0.0; else if (w > 1.0) w = 1.0;
    // Linear in log κ — matches MESA's blend scheme.
    double lk_lo = log10(k_lo > 1e-30 ? k_lo : 1e-30);
    double lk_hi = log10(k_hi > 1e-30 ? k_hi : 1e-30);
    double lk = (1.0 - w) * lk_lo + w * lk_hi;
#ifdef __CUDA_ARCH__
    return exp10(lk);
#else
    return pow(10.0, lk);
#endif
}
#endif

// Grey Rosseland opacity (cm²/g if parameters are cgs; code-unit consistent).
// Dispatches to the MESA table when one is attached; otherwise falls back
// to the analytic piecewise-max formula.
OPA_HD inline double grey_opacity(double rho, double T, const OpacityParams& p) {
#ifdef USE_GPU
    if (p.use_table) return kap_stitch_eval(rho, T, p);
#endif
    double T_use = T > 1.0 ? T : 1.0;
    double rho_use = rho > 1e-30 ? rho : 1e-30;

    // Dust regime (low T)
    double kap_dust = (T_use < p.T_dust_off)
                    ? p.kappa_dust_0 * T_use * T_use
                    : 0.0;

    // H⁻ bound-free (intermediate T, rough fit). Only valid while H⁻
    // survives (T ≲ 10⁴ K); above that it rapidly photoionises. Cut the
    // tail aggressively so the T^7.7 power doesn't go to infinity at
    // stellar interior temperatures (where Kramers/Thomson dominate).
    double kap_Hm = 0.0;
    if (T_use < 1.2e4) {
        kap_Hm = p.kappa_Hm_0 * sqrt(rho_use) * pow(T_use, 7.7);
    }

    // Kramers free-free (hot T)
    double kap_ff = p.kappa_ff_0 * rho_use * pow(T_use, -3.5);

    // Thomson electron scattering floor
    double kap_es = p.kappa_es;

    // Combine: for grey Rosseland, 1/κ = Σ 1/κ_i (parallel resistances) is
    // more accurate for limiting regimes; for simplicity here take max of
    // main contributors + es floor.
    double kap_max = kap_dust;
    if (kap_Hm > kap_max) kap_max = kap_Hm;
    if (kap_ff > kap_max) kap_max = kap_ff;
    return kap_max + kap_es;
}

#undef OPA_HD
