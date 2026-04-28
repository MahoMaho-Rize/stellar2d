#include "boundary.h"

void fill_ghost_cells(const Grid& grid, State& state) {
    int nr = grid.nr, nt = grid.ntheta, ng = grid.ng;

    // Eq. (8.1): inner radial BC (r=0), reflecting symmetry
    for (int g = 1; g <= ng; ++g) {
        for (int j = 0; j < nt; ++j) {
            int k_ghost = grid.idx(-g, j);
            int k_phys = grid.idx(g - 1, j);

            state.rho[k_ghost] = state.rho[k_phys];
            state.mr[k_ghost] = -state.mr[k_phys]; // Eq. (8.1): reflect v_r
            state.mtheta[k_ghost] = state.mtheta[k_phys];
            state.E[k_ghost] = state.E[k_phys];
        }
    }

    // Eq. (8.2): outer radial BC (r=R), zero-gradient outflow
    for (int g = 0; g < ng; ++g) {
        for (int j = 0; j < nt; ++j) {
            int k_ghost = grid.idx(nr + g, j);
            int k_phys = grid.idx(nr - 1, j);

            state.rho[k_ghost] = state.rho[k_phys];
            state.mr[k_ghost] = state.mr[k_phys];
            state.mtheta[k_ghost] = state.mtheta[k_phys];
            state.E[k_ghost] = state.E[k_phys];
        }
    }

    // Eq. (8.3): north pole (theta=0), reflecting symmetry in v_theta
    for (int g = 1; g <= ng; ++g) {
        for (int i = -ng; i < nr + ng; ++i) {
            int k_ghost = grid.idx(i, -g);
            int k_phys = grid.idx(i, g - 1);

            state.rho[k_ghost] = state.rho[k_phys];
            state.mr[k_ghost] = state.mr[k_phys];
            state.mtheta[k_ghost] = -state.mtheta[k_phys]; // Eq. (8.3): reflect v_theta
            state.E[k_ghost] = state.E[k_phys];
        }
    }

    // Eq. (8.3): south pole (theta=pi), reflecting symmetry in v_theta
    for (int g = 0; g < ng; ++g) {
        for (int i = -ng; i < nr + ng; ++i) {
            int k_ghost = grid.idx(i, nt + g);
            int k_phys = grid.idx(i, nt - 1 - g);

            state.rho[k_ghost] = state.rho[k_phys];
            state.mr[k_ghost] = state.mr[k_phys];
            state.mtheta[k_ghost] = -state.mtheta[k_phys]; // Eq. (8.3): reflect v_theta
            state.E[k_ghost] = state.E[k_phys];
        }
    }
}

void apply_polar_cap_stabilizer(const Grid& grid, State& state, double blend) {
    int nr = grid.nr, nt = grid.ntheta;
    if (nt < 4) return;

    double w = (blend < 0.0) ? 0.0 : ((blend > 1.0) ? 1.0 : blend);

    for (int i = 0; i < nr; ++i) {
        int k_north = grid.idx(i, 0);
        int k_north_ref = grid.idx(i, 1);
        int k_south = grid.idx(i, nt - 1);
        int k_south_ref = grid.idx(i, nt - 2);

        state.rho[k_north] = (1.0 - w) * state.rho[k_north] + w * state.rho[k_north_ref];
        state.mr[k_north] = (1.0 - w) * state.mr[k_north] + w * state.mr[k_north_ref];
        state.E[k_north] = (1.0 - w) * state.E[k_north] + w * state.E[k_north_ref];
        state.mtheta[k_north] = 0.0;

        state.rho[k_south] = (1.0 - w) * state.rho[k_south] + w * state.rho[k_south_ref];
        state.mr[k_south] = (1.0 - w) * state.mr[k_south] + w * state.mr[k_south_ref];
        state.E[k_south] = (1.0 - w) * state.E[k_south] + w * state.E[k_south_ref];
        state.mtheta[k_south] = 0.0;
    }
}
