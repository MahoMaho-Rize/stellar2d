#pragma once

#include "../grid.h"
#include "../state.h"
#include "../eos.h"
#include "flux.h"
#include "reconstruct.h"

double compute_cfl_dt(const Grid& grid, const State& state, const EOS& eos, double cfl);

void rk2_substep(const Grid& grid, State& state, const FluxAccumulator& acc, double dt);

void compute_rhs(const Grid& grid, State& state, const EOS& eos,
                 FluxAccumulator& acc, Limiter lim);
