#pragma once

#include "../grid.h"
#include "../state.h"

void fill_ghost_cells(const Grid& grid, State& state);

void apply_polar_cap_stabilizer(const Grid& grid, State& state, double blend = 1.0);
