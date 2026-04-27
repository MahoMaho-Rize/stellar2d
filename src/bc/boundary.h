#pragma once

#include "../grid.h"
#include "../state.h"
#include "../eos.h"

void fill_ghost_cells(const Grid& grid, State& state, double gamma);
