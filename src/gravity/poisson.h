#pragma once

#include "../grid.h"
#include <vector>

struct PoissonMatrix {
    int n;
    int nnz;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> values;
    std::vector<double> rhs;

    void assemble(const Grid& grid, double G);

    void set_rhs(const Grid& grid, const std::vector<double>& rho_cells, double G);
};
