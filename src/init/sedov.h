#pragma once

#include "../grid.h"
#include "../state.h"

struct SedovParams {
    double rho_0;        // ambient density
    double E_blast;      // total blast energy
    double r_blast;      // initial energy deposition radius
};

void init_sedov(const Grid& grid, State& state,
                const SedovParams& params, double gamma);
