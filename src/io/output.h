#pragma once

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

Diagnostics compute_diagnostics(const Grid& grid, const State& state, double gamma);

void write_vtk(const std::string& filename, const Grid& grid, const State& state, double gamma);
