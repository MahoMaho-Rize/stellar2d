#pragma once

// Pre-main-sequence EOS with H₂ dissociation and H ionization.
// Follows Chabrier-Baraffe 2000 style fit: smooth interpolation between
// neutral H₂, atomic H, ionized H+e⁻.
//
// Protostellar relevance:
//   - First core (T ~ 1000-2000 K): neutral H₂ + He, γ_ad ~ 7/5
//   - H₂ dissociation plateau (T ~ 2000-4000 K): γ_ad ~ 1.1-1.2, collapse accelerates
//   - Atomic H (T ~ 4000-10000 K): γ_ad ~ 5/3
//   - H ionization (T ~ 10000-30000 K): γ_ad drops again, another plateau
//   - Fully ionized (T > 30000 K): γ_ad ~ 5/3 back to monatomic ideal gas
//
// Reusable by Path B (cubed-sphere) — no geometry assumptions.

#include <cmath>

#ifdef __CUDACC__
#define PMS_HD __host__ __device__
#else
#define PMS_HD
#endif

// Code units: all physical constants normalized.
// Reference binding energies in code units (will be configurable):
//   χ_H2 = 4.48 eV / (k_B · T_ref) where T_ref is a code-unit reference.
// For now expose as free parameters on the EOS struct.

// Smooth Saha-like ionization fraction.
// x = 1 / (1 + exp((T_trans - T) / dT_width))
// Goes from 0 (cold, bound) → 1 (hot, dissociated/ionized) over ~dT_width.
PMS_HD inline double pms_frac(double T, double T_trans, double dT_width) {
    double arg = (T_trans - T) / (dT_width > 1e-30 ? dT_width : 1e-30);
    // exp overflow guard
    if (arg > 50.0) return 0.0;
    if (arg < -50.0) return 1.0;
    return 1.0 / (1.0 + std::exp(arg));
}

PMS_HD inline double pms_dfrac_dT(double T, double T_trans, double dT_width) {
    double f = pms_frac(T, T_trans, dT_width);
    return f * (1.0 - f) / (dT_width > 1e-30 ? dT_width : 1e-30);
}

// Parameters for pre-MS EOS (Chabrier-Baraffe fit simplified).
// T_diss, T_ion: dissociation / ionization midpoint temperatures (code units)
// dT_diss, dT_ion: transition widths
// chi_diss: H₂ dissociation energy per unit mass (4.48 eV / 2m_p in cgs)
// chi_ion:  H ionization energy per unit mass (13.6 eV / m_p)
// mu_cold: neutral H₂+He mean molecular weight (~2.3 for solar)
// mu_warm: atomic H+He mean molecular weight (~1.3)
// mu_hot:  ionized H+He mean molecular weight (~0.6)
struct PreMsParams {
    double T_diss     = 3000.0;
    double dT_diss    = 500.0;
    double T_ion      = 15000.0;
    double dT_ion     = 3000.0;
    double chi_diss   = 1.0e5;   // specific energy of H₂ dissociation (code units)
    double chi_ion    = 3.0e5;   // specific energy of H ionization (code units)
    double mu_cold    = 2.3;
    double mu_warm    = 1.3;
    double mu_hot     = 0.6;
    double R_gas      = 1.0;     // R = k_B / m_u (code units; 1.0 if already normalized)
    double a_rad      = 0.0;     // radiation constant; 0 disables
};

// Total specific internal energy:
//   e(T) = e_thermal(T) + f_diss(T)·chi_diss + f_ion(T)·chi_ion + e_rad(T, ρ)
// where
//   e_thermal = (3/2) R T / μ(T)   for monatomic;
//             the factor 3/2 approximates mean between 5/2 (H₂ diatomic low T)
//             and 3/2 (monatomic hot). Simplified single coefficient.
//   e_rad = a·T⁴ / ρ
//
// μ(T) smoothly interpolates between mu_cold → mu_warm (across diss) →
// mu_hot (across ion).
PMS_HD inline double pms_mu(double T, const PreMsParams& p) {
    double f_diss = pms_frac(T, p.T_diss, p.dT_diss);
    double f_ion  = pms_frac(T, p.T_ion,  p.dT_ion);
    // mu_cold when nothing dissociated, mu_warm after diss, mu_hot after ion
    double mu = p.mu_cold * (1.0 - f_diss)
              + p.mu_warm * f_diss * (1.0 - f_ion)
              + p.mu_hot  * f_ion;
    return mu > 1e-6 ? mu : 1e-6;
}

PMS_HD inline double pms_energy(double rho, double T, const PreMsParams& p) {
    double mu = pms_mu(T, p);
    double e_therm = 1.5 * p.R_gas * T / mu;
    double e_diss  = p.chi_diss * pms_frac(T, p.T_diss, p.dT_diss);
    double e_ion   = p.chi_ion  * pms_frac(T, p.T_ion,  p.dT_ion);
    double e_rad   = (p.a_rad > 0.0 && rho > 0.0)
                        ? p.a_rad * T*T*T*T / rho : 0.0;
    return e_therm + e_diss + e_ion + e_rad;
}

PMS_HD inline double pms_pressure(double rho, double T, const PreMsParams& p) {
    double mu = pms_mu(T, p);
    double P_gas = rho * p.R_gas * T / mu;
    double P_rad = p.a_rad > 0.0 ? p.a_rad * T*T*T*T / 3.0 : 0.0;
    return P_gas + P_rad;
}

// Solve T from (ρ, e_int) via bracketed bisection (GPU-safe, ~40 iter max).
PMS_HD inline double pms_T_from_rho_e(double rho, double e_int, const PreMsParams& p) {
    // Bracket: lower bound at tiny T, upper bound at very hot T.
    double T_lo = 1.0;
    double T_hi = 1.0e7;
    // Adjust brackets if necessary
    for (int guard = 0; guard < 5; ++guard) {
        if (pms_energy(rho, T_hi, p) < e_int) T_hi *= 10.0;
        else break;
    }
    if (pms_energy(rho, T_lo, p) > e_int) {
        // e_int below floor — fallback to minimal T
        return T_lo;
    }
    for (int it = 0; it < 50; ++it) {
        double T_mid = 0.5 * (T_lo + T_hi);
        double e_mid = pms_energy(rho, T_mid, p);
        if (e_mid > e_int) T_hi = T_mid;
        else T_lo = T_mid;
        if ((T_hi - T_lo) < 1e-6 * (T_hi + T_lo)) break;
    }
    return 0.5 * (T_lo + T_hi);
}

// Sound speed (adiabatic).
// γ_1 = ∂ln P / ∂ln ρ |_s.
// We use thermodynamic identity:
//   γ_1 = (ρ/P)·(∂P/∂ρ)_T + (T/(ρ·c_v·P))·((∂P/∂T)_ρ)²
// with c_v = (∂e/∂T)_ρ.
PMS_HD inline double pms_sound_speed(double rho, double e_int, const PreMsParams& p) {
    double T = pms_T_from_rho_e(rho, e_int, p);
    double P = pms_pressure(rho, T, p);
    if (P < 1e-30) P = 1e-30;

    // Numerical partials (central differences)
    double dT = 1e-4 * T + 1.0;
    double drho = 1e-4 * rho + 1e-10;

    double P_Tp = pms_pressure(rho, T + dT, p);
    double P_Tm = pms_pressure(rho, T - dT, p);
    double dP_dT = (P_Tp - P_Tm) / (2.0 * dT);

    double e_Tp = pms_energy(rho, T + dT, p);
    double e_Tm = pms_energy(rho, T - dT, p);
    double cv_eff = (e_Tp - e_Tm) / (2.0 * dT);
    if (cv_eff < 1e-30) cv_eff = 1e-30;

    double P_rhop = pms_pressure(rho + drho, T, p);
    double P_rhom = pms_pressure(rho - drho, T, p);
    double dP_drho_T = (P_rhop - P_rhom) / (2.0 * drho);

    double gamma1 = (rho / P) * dP_drho_T
                  + (T / (rho * cv_eff * P)) * dP_dT * dP_dT;
    if (gamma1 < 1.01) gamma1 = 1.01;
    return std::sqrt(gamma1 * P / rho);
}

#undef PMS_HD
