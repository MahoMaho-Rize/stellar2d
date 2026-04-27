#pragma once

#include "../grid.h"
#include "../state.h"
#include "../eos.h"
#include "reconstruct.h"
#include "riemann.h"
#include <vector>

struct FluxAccumulator {
    std::vector<double> dU_rho;
    std::vector<double> dU_mr;
    std::vector<double> dU_mtheta;
    std::vector<double> dU_E;

    void allocate(int n);
    void zero();
};

void compute_flux_divergence(
    const Grid& grid, const State& state, const EOS& eos,
    FluxAccumulator& acc, Limiter lim);

void add_geometric_source(
    const Grid& grid, const State& state, const EOS& eos,
    FluxAccumulator& acc);

void add_gravity_source(
    const Grid& grid, const State& state,
    FluxAccumulator& acc);
