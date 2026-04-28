#include "integrate.h"
#include "../parallel.h"
#include <cmath>
#include <limits>

double compute_cfl_dt(const Grid& grid, const State& state, const EOS& eos, double cfl) {
    double dt_min = std::numeric_limits<double>::max();

#ifdef _OPENMP
    #pragma omp parallel for collapse(2) reduction(min:dt_min)
#endif
    for (int i = 0; i < grid.nr; ++i) {
        for (int j = 0; j < grid.ntheta; ++j) {
            int k = grid.idx(i, j);
            PrimitiveVars w = state.to_primitive(k, eos);
            double cs = eos.sound_speed(w.rho, w.P); // Eq. (1.3)

            // Eq. (7.4)
            double dt_r = grid.dr[i] / (std::abs(w.vr) + cs);
            double dt_t = grid.r_center[i] * grid.dtheta[j] / (std::abs(w.vtheta) + cs);
            double dt_local = std::min(dt_r, dt_t);
            dt_min = std::min(dt_min, dt_local);
        }
    }

    return cfl * dt_min; // Eq. (7.4)
}

void rk2_substep(const Grid& grid, State& state, const FluxAccumulator& acc, double dt) {
    int nr = grid.nr, nt = grid.ntheta;
#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nt; ++j) {
            int k = grid.idx(i, j);
            int flat = i * nt + j;
            // Eq. (7.1) or (7.2): U += dt * R(U)
            state.rho[k] += dt * acc.dU_rho[flat];
            state.mr[k] += dt * acc.dU_mr[flat];
            state.mtheta[k] += dt * acc.dU_mtheta[flat];
            state.E[k] += dt * acc.dU_E[flat];
        }
    }
}

void compute_rhs(const Grid& grid, State& state, const EOS& eos,
                 FluxAccumulator& acc, Limiter lim,
                 PolarThetaGeomMode polar_theta_geom_mode) {
    acc.zero();
    compute_flux_divergence(grid, state, eos, acc, lim); // Eq. (2.5)
    add_geometric_source(grid, state, eos, acc, polar_theta_geom_mode); // Eq. (5.1), (5.2)
}
