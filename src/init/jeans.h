#pragma once

#include "../eos.h"
#include "../grid.h"
#include "../state.h"

struct JeansParams {
    double rho_0;
    double cs;         // sound speed
    double G;
    double epsilon;    // perturbation amplitude
    double k_r;        // radial wavenumber
    double k_theta;    // theta wavenumber
};

void init_jeans(const Grid& grid, State& state,
                const JeansParams& params, const EOS& eos);
