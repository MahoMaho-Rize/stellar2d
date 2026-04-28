#pragma once

#include "../grid.h"
#include <vector>

void compute_poisson_rhs(const Grid& grid, const std::vector<double>& rho_cells,
                         double G, std::vector<double>& rhs);

struct PoissonGMG {
    void init(const Grid& grid);
    void solve(const double* rhs, double* phi, int max_cycles = 50, double tol = 1e-6);

private:
    struct Level {
        int nr, nt;
        std::vector<double> r_face, r_center, dr;
        std::vector<double> theta_face, theta_center, dtheta;
        std::vector<double> sin_theta_face, sin_theta_center;
        std::vector<double> cell_volume;
        std::vector<double> phi, rhs, res;
    };

    std::vector<Level> levels_;
    int n_levels_;
    static constexpr int NU1 = 5, NU2 = 5;

    void build_level(int l, const std::vector<double>& rf, const std::vector<double>& tf);
    void stencil_coeffs(const Level& lev, int i, int j,
                        double& cW, double& cE, double& cS, double& cN, double& cC) const;
    void smooth(int l, int n_iters);
    void compute_residual(int l);
    void restrict_level(int fine, int coarse);
    void prolongate_and_correct(int coarse, int fine);
    void vcycle(int l);
};
