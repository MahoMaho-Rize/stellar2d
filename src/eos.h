#pragma once

#include <cmath>

struct EOS {
    double gamma;

    explicit EOS(double gamma_ = 5.0 / 3.0) : gamma(gamma_) {}

    double pressure(double rho, double e_int) const {
        return (gamma - 1.0) * rho * e_int; // Eq. (1.2)
    }

    double internal_energy(double rho, double P) const {
        return P / ((gamma - 1.0) * rho); // Eq. (1.2) inverted
    }

    double sound_speed(double rho, double P) const {
        return std::sqrt(gamma * P / rho); // Eq. (1.3)
    }
};
