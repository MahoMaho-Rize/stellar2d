#pragma once

#include "../state.h"

struct Flux4 {
    double f_rho;
    double f_mr;
    double f_mtheta;
    double f_E;
};

Flux4 hllc_flux_r(const PrimitiveVars& wl, const PrimitiveVars& wr, double gamma);

Flux4 hllc_flux_theta(const PrimitiveVars& wl, const PrimitiveVars& wr, double gamma);
