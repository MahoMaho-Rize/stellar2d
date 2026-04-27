#include "poisson.h"
#include <cmath>
#include <algorithm>

void PoissonMatrix::assemble(const Grid& grid, double G) {
    int nr = grid.nr, nt = grid.ntheta;
    n = nr * nt;

    row_ptr.clear();
    col_idx.clear();
    values.clear();
    row_ptr.reserve(n + 1);
    col_idx.reserve(5 * n);
    values.reserve(5 * n);

    row_ptr.push_back(0);

    for (int i = 0; i < nr; ++i) {
        double ri = grid.r_center[i];
        double ri2 = ri * ri;

        for (int j = 0; j < nt; ++j) {
            int row = i * nt + j;
            double diag = 0.0;

            std::vector<std::pair<int, double>> entries;

            // Radial part: 1/r^2 * d/dr(r^2 dPhi/dr)
            if (i > 0) {
                double r_lo = grid.r_face[i];
                double dr_lo = grid.r_center[i] - grid.r_center[i - 1];
                double coeff = r_lo * r_lo / (ri2 * dr_lo * grid.dr[i]);
                // corrected: use average dr for the cell
                double coeff_corr = r_lo * r_lo / (ri2 * dr_lo);
                // Laplacian factor: coefficient for Phi_{i-1,j}
                double c = r_lo * r_lo / (ri2 * grid.dr[i] * dr_lo);
                entries.push_back({(i - 1) * nt + j, c});
                diag -= c;
            }

            if (i < nr - 1) {
                double r_hi = grid.r_face[i + 1];
                double dr_hi = grid.r_center[i + 1] - grid.r_center[i];
                double c = r_hi * r_hi / (ri2 * grid.dr[i] * dr_hi);
                entries.push_back({(i + 1) * nt + j, c});
                diag -= c;
            }

            // Theta part: 1/(r^2 sin(theta)) * d/dtheta(sin(theta) dPhi/dtheta)
            double sin_j = std::sin(grid.theta_center[j]);
            double dtheta = grid.dtheta[j];

            if (j > 0) {
                double sin_lo = std::sin(grid.theta_face[j]);
                double dtheta_lo = grid.theta_center[j] - grid.theta_center[j - 1];
                double c = sin_lo / (ri2 * sin_j * dtheta * dtheta_lo);
                entries.push_back({i * nt + (j - 1), c});
                diag -= c;
            }

            if (j < nt - 1) {
                double sin_hi = std::sin(grid.theta_face[j + 1]);
                double dtheta_hi = grid.theta_center[j + 1] - grid.theta_center[j];
                double c = sin_hi / (ri2 * sin_j * dtheta * dtheta_hi);
                entries.push_back({i * nt + (j + 1), c});
                diag -= c;
            }

            // Boundary modifications:
            // r=0 (i==0): Neumann dPhi/dr=0 => Phi_{-1} = Phi_0, already handled by skip
            // r=R (i==nr-1): Dirichlet Phi = -GM/r => handled in RHS, row is already correct
            //   Actually we enforce Dirichlet by modifying this row to set Phi directly.
            if (i == nr - 1) {
                // Dirichlet at outer boundary: replace row with identity
                entries.clear();
                diag = 1.0;
            }

            // theta=0 and theta=pi: Neumann dPhi/dtheta=0, handled by skip above

            // Sort entries by column index for CSR
            entries.push_back({row, diag});
            std::sort(entries.begin(), entries.end());

            for (auto& [col, val] : entries) {
                col_idx.push_back(col);
                values.push_back(val);
            }
            row_ptr.push_back(static_cast<int>(col_idx.size()));
        }
    }

    nnz = static_cast<int>(col_idx.size());
    rhs.resize(n, 0.0);
}

void PoissonMatrix::set_rhs(const Grid& grid, const std::vector<double>& rho_cells,
                             double G) {
    int nr = grid.nr, nt = grid.ntheta;

    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nt; ++j) {
            int flat = i * nt + j;

            if (i == nr - 1) {
                // Dirichlet BC: Phi = -G * M_total / r
                // Approximate M_total from density integration
                double M_total = 0.0;
                for (int ii = 0; ii < nr; ++ii) {
                    for (int jj = 0; jj < nt; ++jj) {
                        M_total += rho_cells[ii * nt + jj] * grid.cell_volume[ii * nt + jj];
                    }
                }
                rhs[flat] = -G * M_total / grid.r_center[i];
            } else {
                rhs[flat] = 4.0 * M_PI * G * rho_cells[flat];
            }
        }
    }
}
