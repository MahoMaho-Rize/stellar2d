#pragma once

#include "../eos.h"
#include "../grid.h"
#include "../state.h"

struct LaneEmdenParams {
    double n_poly;     // polytropic index
    double rho_c;      // central density
    double K_poly;     // polytropic constant P = K * rho^(1+1/n)
    double G;          // gravitational constant
};

struct LaneEmdenSolution {
    std::vector<double> xi;
    std::vector<double> theta_le; // Lane-Emden theta (not angle)
    double xi_1;                  // first zero
    double dtheta_xi1;            // -theta'(xi_1)
};

LaneEmdenSolution solve_lane_emden(double n_poly, int npoints = 10000);

void init_lane_emden(const Grid& grid, State& state,
                     const LaneEmdenParams& params, const EOS& eos);

void init_lane_emden_perturbed(const Grid& grid, State& state,
                                const LaneEmdenParams& params, const EOS& eos,
                                double amplitude);
