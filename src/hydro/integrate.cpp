#include "integrate.h"
#include <cmath>
#include <limits>

double compute_cfl_dt(const Grid& grid, const State& state, const EOS& eos, double cfl) {
    double dt_min = std::numeric_limits<double>::max();
    double gamma = eos.gamma;

    for (int i = 0; i < grid.nr; ++i) {
        for (int j = 0; j < grid.ntheta; ++j) {
            int k = grid.idx(i, j);
            PrimitiveVars w = state.to_primitive(k, gamma);
            double cs = eos.sound_speed(w.rho, w.P);

            double dt_r = grid.dr[i] / (std::abs(w.vr) + cs);
            double dt_t = grid.r_center[i] * grid.dtheta[j] / (std::abs(w.vtheta) + cs);
            double dt_local = std::min(dt_r, dt_t);
            dt_min = std::min(dt_min, dt_local);
        }
    }

    return cfl * dt_min;
}

void rk2_substep(const Grid& grid, State& state, const FluxAccumulator& acc, double dt) {
    int nr = grid.nr, nt = grid.ntheta;
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nt; ++j) {
            int k = grid.idx(i, j);
            int flat = i * nt + j;
            state.rho[k] += dt * acc.dU_rho[flat];
            state.mr[k] += dt * acc.dU_mr[flat];
            state.mtheta[k] += dt * acc.dU_mtheta[flat];
            state.E[k] += dt * acc.dU_E[flat];
        }
    }
}

void compute_rhs(const Grid& grid, State& state, const EOS& eos,
                 FluxAccumulator& acc, Limiter lim) {
    acc.zero();
    compute_flux_divergence(grid, state, eos, acc, lim);
    add_geometric_source(grid, state, eos, acc);
}
