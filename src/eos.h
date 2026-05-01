#pragma once

#include <cmath>

#ifdef __CUDACC__
#define EOS_HD __host__ __device__
#else
#define EOS_HD
#endif

// Single POD struct (no virtuals) so it can cross host/device boundary.
// Represents either:
//   EosType::IDEAL       — P = (γ-1)·ρ·e
//   EosType::IDEAL_RAD   — P = ρ·R_gas·T + a·T⁴/3, e = cv·T + a·T⁴/ρ
//
// For ideal gas, radiation_a is ignored (treated as 0).
// Newton solve for T(ρ,e) uses bounded fixed iteration count, GPU-safe.

enum class EosType { IDEAL = 0, IDEAL_RAD = 1 };

struct EOS {
    double gamma;
    double mu;            // mean molecular weight (R_gas = 1/mu in code units)
    double radiation_a;   // radiation constant a (code units); 0 ⇒ pure ideal
    int type;             // EosType as int for trivial POD copy

    EOS_HD EOS()
        : gamma(5.0/3.0), mu(1.0), radiation_a(0.0),
          type(static_cast<int>(EosType::IDEAL)) {}

    EOS_HD explicit EOS(double gamma_, double mu_ = 1.0)
        : gamma(gamma_), mu(mu_), radiation_a(0.0),
          type(static_cast<int>(EosType::IDEAL)) {}

    static EOS_HD EOS ideal(double gamma_, double mu_ = 1.0) {
        EOS e(gamma_, mu_);
        e.type = static_cast<int>(EosType::IDEAL);
        return e;
    }

    static EOS_HD EOS ideal_rad(double gamma_, double mu_, double radiation_a_) {
        EOS e;
        e.gamma = gamma_; e.mu = mu_; e.radiation_a = radiation_a_;
        e.type = static_cast<int>(EosType::IDEAL_RAD);
        return e;
    }

    EOS_HD double gas_constant() const { return 1.0 / mu; }
    EOS_HD double cv() const { return gas_constant() / (gamma - 1.0); }

    EOS_HD double pressure(double rho, double e_int) const {
        if (type == (int)EosType::IDEAL_RAD) {
            double T = temperature_from_rho_e(rho, e_int);
            return rho * gas_constant() * T + radiation_a * T * T * T * T / 3.0;
        }
        return (gamma - 1.0) * rho * e_int;
    }

    EOS_HD double internal_energy(double rho, double P) const {
        if (type == (int)EosType::IDEAL_RAD) {
            double T = temperature_from_rho_p(rho, P);
            return cv() * T + radiation_a * T * T * T * T / rho;
        }
        return P / ((gamma - 1.0) * rho);
    }

    EOS_HD double sound_speed(double rho, double P) const {
        if (type == (int)EosType::IDEAL_RAD) {
            double T = temperature_from_rho_p(rho, P);
            double Pgas = rho * gas_constant() * T;
            double T4 = T * T * T * T;
            double Prad = radiation_a * T4 / 3.0;
            double Ptot = Pgas + Prad;
            if (Ptot < 1e-30) Ptot = 1e-30;
            double chi_rho = Pgas / Ptot;
            double chi_T   = (Pgas + 4.0 * Prad) / Ptot;
            double cv_tot  = cv() + 4.0 * radiation_a * T * T * T / rho;
            double g3m1 = Ptot * chi_T / (rho * T * cv_tot);
            double gamma1 = chi_rho + chi_T * g3m1;
            if (gamma1 < 1.0 + 1e-8) gamma1 = 1.0 + 1e-8;
            return sqrt(gamma1 * Ptot / rho);
        }
        return sqrt(gamma * P / rho);
    }

    // ∂P/∂(ρe) at fixed ρ — critical for FAS SIMPLE smoother Jacobian
    EOS_HD double dP_drhoe(double rho, double e_int) const {
        if (type == (int)EosType::IDEAL_RAD) {
            double T = temperature_from_rho_e(rho, e_int);
            double aT3 = radiation_a * T * T * T;
            double dP_dT = rho * gas_constant() + 4.0 / 3.0 * aT3;
            double de_dT_per_rho = cv() + 4.0 * aT3 / rho; // ∂e/∂T at fixed ρ
            // d(ρe)/dT = ρ · de/dT; so ∂(ρe)/∂T = ρ·de_dT_per_rho
            return dP_dT / (rho * de_dT_per_rho);
        }
        return gamma - 1.0;
    }

    // Temperature from (ρ, e) — used internally and for diagnostics
    EOS_HD double temperature_from_rho_e(double rho, double e_int) const {
        if (type != (int)EosType::IDEAL_RAD)
            return e_int / cv();
        double cv_gas = cv();
        double T_gas = e_int / (cv_gas > 1e-30 ? cv_gas : 1e-30);
        double aval = radiation_a > 1e-30 ? radiation_a : 1e-30;
        double T_rad = pow((e_int * rho / aval) > 0.0 ? e_int * rho / aval : 0.0, 0.25);
        double T_hi = T_gas > T_rad ? T_gas : T_rad;
        if (T_hi < 1e-12) T_hi = 1e-12;
        double T_lo = 0.0;
        double T = 0.5 * T_hi;
        if (T < 1e-12) T = 1e-12;
        for (int it = 0; it < 40; ++it) {
            double T4 = T*T*T*T;
            double f  = cv_gas * T + radiation_a * T4 / rho - e_int;
            if (fabs(f) < 1e-12 * (fabs(e_int) + 1e-30)) break;
            double df = cv_gas + 4.0 * radiation_a * T*T*T / rho;
            if (f > 0.0) T_hi = T; else T_lo = T;
            double T_new = T - f / (df > 1e-30 ? df : 1e-30);
            T = (T_new <= T_lo || T_new >= T_hi) ? 0.5*(T_lo + T_hi) : T_new;
        }
        return T > 1e-12 ? T : 1e-12;
    }

    EOS_HD double temperature_from_rho_p(double rho, double P) const {
        if (type != (int)EosType::IDEAL_RAD)
            return P / (rho * gas_constant());
        double Rg = rho * gas_constant();
        double T_gas = P / (Rg > 1e-30 ? Rg : 1e-30);
        double aval = radiation_a > 1e-30 ? radiation_a : 1e-30;
        double T_rad = pow((3.0 * P / aval) > 0.0 ? 3.0 * P / aval : 0.0, 0.25);
        double T_hi = T_gas > T_rad ? T_gas : T_rad;
        if (T_hi < 1e-12) T_hi = 1e-12;
        double T_lo = 0.0;
        double T = 0.5 * T_hi;
        if (T < 1e-12) T = 1e-12;
        for (int it = 0; it < 40; ++it) {
            double T4 = T*T*T*T;
            double f  = Rg * T + radiation_a * T4 / 3.0 - P;
            if (fabs(f) < 1e-12 * (fabs(P) + 1e-30)) break;
            double df = Rg + 4.0 * radiation_a * T*T*T / 3.0;
            if (f > 0.0) T_hi = T; else T_lo = T;
            double T_new = T - f / (df > 1e-30 ? df : 1e-30);
            T = (T_new <= T_lo || T_new >= T_hi) ? 0.5*(T_lo + T_hi) : T_new;
        }
        return T > 1e-12 ? T : 1e-12;
    }
};

#undef EOS_HD
