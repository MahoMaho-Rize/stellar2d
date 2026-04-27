#pragma once

#include <cmath>

struct EOS {
    double gamma;

    explicit EOS(double gamma_ = 5.0 / 3.0) : gamma(gamma_) {}

    double pressure(double rho, double e_int) const {
        return (gamma - 1.0) * rho * e_int;
    }

    double internal_energy(double rho, double P) const {
        return P / ((gamma - 1.0) * rho);
    }

    double sound_speed(double rho, double P) const {
        return std::sqrt(gamma * P / rho);
    }
};
