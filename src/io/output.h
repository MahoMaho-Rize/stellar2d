#pragma once

#include "../eos.h"
#include "../grid.h"
#include "../state.h"
#include <string>

struct Diagnostics {
    double total_mass;
    double total_energy;
    double kinetic_energy;
    double thermal_energy;
    double gravitational_energy;
};

Diagnostics compute_diagnostics(const Grid& grid, const State& state);

void write_vtk(const std::string& filename, const Grid& grid, const State& state, const EOS& eos);
