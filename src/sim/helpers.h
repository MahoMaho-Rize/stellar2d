#pragma once

#include "grid.h"
#include "state.h"
#include "cli/options.h"

#include <ctime>
#include <string>
#include <vector>

// Extract density field into a flat [nr*nt] buffer (cell-centered order).
void extract_density(const Grid& grid, const State& state, std::vector<double>& rho_cells);

// Lane-Emden physical radius (code units, n=1.5 polytrope by default).
// R_star is the zero-density surface; R_outer adds 10% margin for ghost atm.
double compute_lane_emden_R_star(double n_poly, double K_poly, double rho_c, double G);
double compute_lane_emden_R_outer(double n_poly, double K_poly, double rho_c, double G);

// Progress-bar to stderr: "[####......] 42%  step 12345  t=... dt=... ETA ...".
// Called every N steps from each solver's time loop.
void print_progress(double t, double t_end, int step, double dt,
                    std::timespec& t_start);

// Build a traceable run directory: runs/<test>_<nr>x<nt>_<YYYYMMDD_HHMMSS>/
// Creates both the run-base and the timestamped sub-dir; returns the latter.
std::string make_run_dir(const SimConfig& cfg);
