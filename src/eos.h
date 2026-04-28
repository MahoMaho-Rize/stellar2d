#pragma once

#include <cmath>
#include <algorithm>
#include <stdexcept>

struct EOS {
    double gamma;
    double mu;

    explicit EOS(double gamma_, double mu_) : gamma(gamma_), mu(mu_) {}
    virtual ~EOS() = default;

    virtual double pressure_from_rho_e(double rho, double e_int) const = 0;
    virtual double internal_energy_from_rho_p(double rho, double P) const = 0;
    virtual double sound_speed(double rho, double P) const = 0;
    virtual double temperature_from_rho_e(double rho, double e_int) const = 0;
    virtual double temperature_from_rho_p(double rho, double P) const = 0;
};

struct IdealGasEOS final : public EOS {
    explicit IdealGasEOS(double gamma_ = 5.0 / 3.0, double mu_ = 1.0)
        : EOS(gamma_, mu_) {}

    double pressure_from_rho_e(double rho, double e_int) const override {
        return (gamma - 1.0) * rho * e_int; // Eq. (1.2)
    }

    double internal_energy_from_rho_p(double rho, double P) const override {
        return P / ((gamma - 1.0) * rho); // Eq. (1.2) inverted
    }

    double sound_speed(double rho, double P) const override {
        return std::sqrt(gamma * P / rho); // Eq. (1.3)
    }

    double temperature_from_rho_e(double, double e_int) const override {
        return e_int / specific_heat_cv();
    }

    double temperature_from_rho_p(double rho, double P) const override {
        return P / (rho * gas_constant());
    }

    double gas_constant() const {
        return 1.0 / mu;
    }

    double specific_heat_cv() const {
        return gas_constant() / (gamma - 1.0);
    }
};

struct IdealGasRadiationEOS final : public EOS {
    double radiation_a;

    explicit IdealGasRadiationEOS(double gamma_ = 5.0 / 3.0,
                                  double mu_ = 1.0,
                                  double radiation_a_ = 0.1)
        : EOS(gamma_, mu_), radiation_a(radiation_a_) {}

    double pressure_from_rho_e(double rho, double e_int) const override {
        double T = temperature_from_rho_e(rho, e_int);
        return gas_pressure(rho, T) + radiation_pressure(T);
    }

    double internal_energy_from_rho_p(double rho, double P) const override {
        double T = temperature_from_rho_p(rho, P);
        return specific_heat_cv() * T + radiation_a * std::pow(T, 4) / rho;
    }

    double sound_speed(double rho, double P) const override {
        double T = temperature_from_rho_p(rho, P);
        double Pgas = gas_pressure(rho, T);
        double Prad = radiation_pressure(T);
        double Ptot = std::max(Pgas + Prad, 1e-30);

        double beta = Pgas / Ptot;
        double chi_rho = beta;
        double chi_T = (Pgas + 4.0 * Prad) / Ptot;
        double cv = specific_heat_cv() + 4.0 * radiation_a * std::pow(T, 3) / rho;
        double gamma3_minus1 = Ptot * chi_T / (rho * T * cv);
        double gamma1 = chi_rho + chi_T * gamma3_minus1;
        gamma1 = std::max(gamma1, 1.0 + 1e-8);

        return std::sqrt(gamma1 * Ptot / rho);
    }

    double temperature_from_rho_e(double rho, double e_int) const override {
        double cv = specific_heat_cv();
        double T_gas = e_int / std::max(cv, 1e-30);
        double T_rad = std::pow(std::max(e_int * rho / std::max(radiation_a, 1e-30), 0.0), 0.25);
        double T_hi = std::max({T_gas, T_rad, 1e-12});
        auto residual = [&](double T) {
            return cv * T + radiation_a * std::pow(T, 4) / rho - e_int;
        };
        auto deriv = [&](double T) {
            return cv + 4.0 * radiation_a * std::pow(T, 3) / rho;
        };
        return solve_monotone_root(residual, deriv, T_hi);
    }

    double temperature_from_rho_p(double rho, double P) const override {
        double T_gas = P / std::max(rho * gas_constant(), 1e-30);
        double T_rad = std::pow(std::max(3.0 * P / std::max(radiation_a, 1e-30), 0.0), 0.25);
        double T_hi = std::max({T_gas, T_rad, 1e-12});
        auto residual = [&](double T) {
            return gas_pressure(rho, T) + radiation_pressure(T) - P;
        };
        auto deriv = [&](double T) {
            return rho * gas_constant() + 4.0 * radiation_a * std::pow(T, 3) / 3.0;
        };
        return solve_monotone_root(residual, deriv, T_hi);
    }

    double gas_constant() const {
        return 1.0 / mu;
    }

    double specific_heat_cv() const {
        return gas_constant() / (gamma - 1.0);
    }

    double gas_pressure(double rho, double T) const {
        return rho * gas_constant() * T;
    }

    double radiation_pressure(double T) const {
        return radiation_a * std::pow(T, 4) / 3.0;
    }

    template <typename Residual, typename Derivative>
    double solve_monotone_root(const Residual& residual,
                               const Derivative& derivative,
                               double T_hi) const {
        double T_lo = 0.0;
        double T = std::max(0.5 * T_hi, 1e-12);

        for (int iter = 0; iter < 40; ++iter) {
            double f = residual(T);
            if (std::abs(f) < 1e-12) break;

            if (f > 0.0) {
                T_hi = T;
            } else {
                T_lo = T;
            }

            double df = derivative(T);
            double T_newton = T - f / std::max(df, 1e-30);
            if (T_newton <= T_lo || T_newton >= T_hi) {
                T = 0.5 * (T_lo + T_hi);
            } else {
                T = T_newton;
            }
        }

        return std::max(T, 1e-12);
    }
};
