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
};

// Grey Rosseland opacity (cm²/g if parameters are cgs; code-unit consistent).
// Takes max of components plus electron scattering floor.
OPA_HD inline double grey_opacity(double rho, double T, const OpacityParams& p) {
    double T_use = T > 1.0 ? T : 1.0;
    double rho_use = rho > 1e-30 ? rho : 1e-30;

    // Dust regime (low T)
    double kap_dust = (T_use < p.T_dust_off)
                    ? p.kappa_dust_0 * T_use * T_use
                    : 0.0;

    // H⁻ bound-free (intermediate T, rough fit)
    double kap_Hm = p.kappa_Hm_0 * sqrt(rho_use) * pow(T_use, 7.7);

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
