#include "pp_chain.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

OneZoneState clamp_state(const OneZoneState& in, double x_floor) {
    OneZoneState out = in;
    out.rho = std::max(out.rho, 1.0e-20);
    out.e_int = std::max(out.e_int, 1.0e-20);
    out.X_h = std::clamp(out.X_h, x_floor, 1.0 - x_floor);
    out.X_he = std::max(out.X_he, x_floor);
    double norm = out.X_h + out.X_he;
    if (norm > 0.0) {
        out.X_h /= norm;
        out.X_he /= norm;
    }
    return out;
}

OneZoneState add_scaled(const OneZoneState& s, const BurnDerivatives& k, double scale) {
    OneZoneState out = s;
    out.e_int += scale * k.de_int_dt;
    out.X_h += scale * k.dX_h_dt;
    out.X_he += scale * k.dX_he_dt;
    return out;
}

} // namespace

BurnDerivatives pp_chain_rhs(const OneZoneState& raw_state, const EOS& eos, const PPChainParams& params) {
    OneZoneState state = clamp_state(raw_state, params.x_floor);

    double temperature = eos.temperature_from_rho_e(state.rho, state.e_int);
    double pressure = eos.pressure_from_rho_e(state.rho, state.e_int);

    double theta = std::max(temperature / std::max(params.temperature_ref, 1.0e-30), 1.0e-12);
    double rate = params.lambda0 * state.rho * state.X_h * state.X_h
                * std::pow(theta, params.temperature_exponent);

    BurnDerivatives rhs;
    rhs.dX_h_dt = -rate;
    rhs.dX_he_dt = rate;
    rhs.eps_nuc = params.q_pp * rate;
    rhs.de_int_dt = rhs.eps_nuc;
    rhs.rate = rate;
    rhs.temperature = temperature;
    rhs.pressure = pressure;
    return rhs;
}

double estimate_pp_burn_dt(const OneZoneState& state, const EOS& eos,
                           const PPChainParams& params, double frac_change_limit) {
    BurnDerivatives rhs = pp_chain_rhs(state, eos, params);
    double dt_x = std::numeric_limits<double>::max();
    if (std::abs(rhs.dX_h_dt) > 1.0e-30) {
        dt_x = frac_change_limit * std::max(state.X_h, params.x_floor) / std::abs(rhs.dX_h_dt);
    }

    double dt_e = std::numeric_limits<double>::max();
    if (std::abs(rhs.de_int_dt) > 1.0e-30) {
        dt_e = frac_change_limit * std::max(state.e_int, 1.0e-20) / std::abs(rhs.de_int_dt);
    }

    double dt = std::min(dt_x, dt_e);
    if (!std::isfinite(dt) || dt <= 0.0) {
        dt = 1.0e-6;
    }
    return std::max(dt, 1.0e-12);
}

void advance_pp_chain_rk4(OneZoneState& state, const EOS& eos,
                          const PPChainParams& params, double dt) {
    OneZoneState s0 = clamp_state(state, params.x_floor);

    BurnDerivatives k1 = pp_chain_rhs(s0, eos, params);
    BurnDerivatives k2 = pp_chain_rhs(add_scaled(s0, k1, 0.5 * dt), eos, params);
    BurnDerivatives k3 = pp_chain_rhs(add_scaled(s0, k2, 0.5 * dt), eos, params);
    BurnDerivatives k4 = pp_chain_rhs(add_scaled(s0, k3, dt), eos, params);

    state = s0;
    state.X_h += dt / 6.0 * (k1.dX_h_dt + 2.0 * k2.dX_h_dt + 2.0 * k3.dX_h_dt + k4.dX_h_dt);
    state.X_he += dt / 6.0 * (k1.dX_he_dt + 2.0 * k2.dX_he_dt + 2.0 * k3.dX_he_dt + k4.dX_he_dt);
    state.e_int += dt / 6.0 * (k1.de_int_dt + 2.0 * k2.de_int_dt + 2.0 * k3.de_int_dt + k4.de_int_dt);
    state = clamp_state(state, params.x_floor);
}
