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
    double max_mach;
    double max_vr;
    double max_vtheta;
    double mass_conservation_error;  // relative to initial mass
};

Diagnostics compute_diagnostics(const Grid& grid, const State& state, double gamma);

void write_vtk(const std::string& filename, const Grid& grid, const State& state, double gamma);

// Append a one-line per-step diagnostic record to a CSV file.
// Creates the file with header on first call.
void write_diagnostics_csv(const std::string& filename, int step, double t, double dt,
                           const Diagnostics& diag, int newton_iters, int gmres_iters);
