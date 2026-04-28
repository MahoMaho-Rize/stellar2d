#include "lane_emden.h"
#include <cmath>
#include <algorithm>

LaneEmdenSolution solve_lane_emden(double n_poly, int npoints) {
    LaneEmdenSolution sol;

    double dxi = 0.001;
    double xi = 1e-10;
    double theta = 1.0;
    double dtheta = 0.0;

    sol.xi.push_back(0.0);
    sol.theta_le.push_back(1.0);

    // Eq. (9.1): integrate Lane-Emden ODE via RK4
    // dy1/dxi = y2,  dy2/dxi = -Theta^n - 2*y2/xi
    while (theta > 0.0 && xi < 100.0) {
        auto f2 = [&](double x, double t, double dt_val) -> double {
            if (x < 1e-10) return -t / 3.0; // L'Hopital at origin
            double tn = (t > 0) ? std::pow(t, n_poly) : 0.0;
            return -tn - 2.0 * dt_val / x; // Eq. (9.1) rewritten as first-order system
        };

        double k1_y1 = dtheta;
        double k1_y2 = f2(xi, theta, dtheta);

        double k2_y1 = dtheta + 0.5 * dxi * k1_y2;
        double k2_y2 = f2(xi + 0.5 * dxi, theta + 0.5 * dxi * k1_y1, dtheta + 0.5 * dxi * k1_y2);

        double k3_y1 = dtheta + 0.5 * dxi * k2_y2;
        double k3_y2 = f2(xi + 0.5 * dxi, theta + 0.5 * dxi * k2_y1, dtheta + 0.5 * dxi * k2_y2);

        double k4_y1 = dtheta + dxi * k3_y2;
        double k4_y2 = f2(xi + dxi, theta + dxi * k3_y1, dtheta + dxi * k3_y2);

        theta += dxi / 6.0 * (k1_y1 + 2.0 * k2_y1 + 2.0 * k3_y1 + k4_y1);
        dtheta += dxi / 6.0 * (k1_y2 + 2.0 * k2_y2 + 2.0 * k3_y2 + k4_y2);
        xi += dxi;

        sol.xi.push_back(xi);
        sol.theta_le.push_back(std::max(theta, 0.0));

        if (theta <= 0.0) break;
    }

    sol.xi_1 = xi;
    sol.dtheta_xi1 = -dtheta;

    return sol;
}

static double interp_le(const LaneEmdenSolution& sol, double xi_val) {
    if (xi_val <= 0.0) return 1.0;
    if (xi_val >= sol.xi_1) return 0.0;

    auto it = std::lower_bound(sol.xi.begin(), sol.xi.end(), xi_val);
    int idx = static_cast<int>(it - sol.xi.begin());
    if (idx == 0) idx = 1;
    if (idx >= static_cast<int>(sol.xi.size())) return 0.0;

    double x0 = sol.xi[idx - 1], x1 = sol.xi[idx];
    double t0 = sol.theta_le[idx - 1], t1 = sol.theta_le[idx];
    double t = (xi_val - x0) / (x1 - x0);
    return t0 + t * (t1 - t0);
}

void init_lane_emden(const Grid& grid, State& state,
                     const LaneEmdenParams& params, const EOS& eos) {
    auto sol = solve_lane_emden(params.n_poly);

    // Eq. (9.2): Lane-Emden length scale
    double alpha2 = (params.n_poly + 1.0) * params.K_poly
                    * std::pow(params.rho_c, 1.0 / params.n_poly - 1.0)
                    / (4.0 * M_PI * params.G);
    double alpha = std::sqrt(alpha2);

    for (int i = 0; i < grid.nr; ++i) {
        double r = grid.r_center[i];
        double xi_val = r / alpha;
        double theta_val = interp_le(sol, xi_val);
        double rho = params.rho_c * std::pow(std::max(theta_val, 1e-15), params.n_poly); // Eq. (9.4)
        double P = params.K_poly * std::pow(rho, 1.0 + 1.0 / params.n_poly);             // Eq. (9.5)

        for (int j = 0; j < grid.ntheta; ++j) {
            int k = grid.idx(i, j);
            PrimitiveVars w;
            w.rho = std::max(rho, 1e-20);
            w.vr = 0.0;
            w.vtheta = 0.0;
            w.P = std::max(P, 1e-30);
            state.from_primitive(k, w, eos);
        }
    }
}

void init_lane_emden_perturbed(const Grid& grid, State& state,
                                const LaneEmdenParams& params, const EOS& eos,
                                double amplitude) {
    init_lane_emden(grid, state, params, eos);

    for (int i = 0; i < grid.nr; ++i) {
        double r = grid.r_center[i];
        for (int j = 0; j < grid.ntheta; ++j) {
            int k = grid.idx(i, j);
            PrimitiveVars w = state.to_primitive(k, eos);
            double delta = amplitude * std::sin(M_PI * r / grid.R_outer);
            w.rho *= (1.0 + delta);
            w.P *= (1.0 + eos.gamma * delta); // adiabatic perturbation
            state.from_primitive(k, w, eos);
        }
    }
}
