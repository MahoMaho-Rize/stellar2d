#pragma once

#include "../eos.h"

struct OneZoneState {
    double rho;
    double e_int;
    double X_h;
    double X_he;
};

struct PPChainParams {
    double lambda0 = 1.0e-2;       // effective base rate coefficient
    double temperature_ref = 1.0;  // reference temperature for nondimensionalization
    double temperature_exponent = 4.0;
    double q_pp = 1.0;             // specific energy release per unit H mass fraction burnt
    double x_floor = 1.0e-12;
};

struct BurnDerivatives {
    double dX_h_dt;
    double dX_he_dt;
    double de_int_dt;
    double rate;
    double eps_nuc;
    double temperature;
    double pressure;
};

BurnDerivatives pp_chain_rhs(const OneZoneState& state, const EOS& eos, const PPChainParams& params);

double estimate_pp_burn_dt(const OneZoneState& state, const EOS& eos,
                           const PPChainParams& params, double frac_change_limit = 0.02);

void advance_pp_chain_rk4(OneZoneState& state, const EOS& eos,
                          const PPChainParams& params, double dt);
