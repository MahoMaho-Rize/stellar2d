#pragma once

#include "../eos.h"
#include "../grid.h"
#include "../state.h"

struct EvrardParams {
    double M;     // total mass
    double R;     // initial radius
    double G;     // gravitational constant
};

void init_evrard(const Grid& grid, State& state,
                 const EvrardParams& params, const EOS& eos);
