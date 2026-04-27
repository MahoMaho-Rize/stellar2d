#include "boundary.h"

void fill_ghost_cells(const Grid& grid, State& state, double gamma) {
    int nr = grid.nr, nt = grid.ntheta, ng = grid.ng;

    // r = 0 (inner boundary): reflecting
    // ghost cells at i = -1, -2 mirror from i = 0, 1
    for (int g = 1; g <= ng; ++g) {
        for (int j = 0; j < nt; ++j) {
            int k_ghost = grid.idx(-g, j);
            int k_phys = grid.idx(g - 1, j);

            state.rho[k_ghost] = state.rho[k_phys];
            state.mr[k_ghost] = -state.mr[k_phys]; // reflect v_r
            state.mtheta[k_ghost] = state.mtheta[k_phys];
            state.E[k_ghost] = state.E[k_phys];
        }
    }

    // r = R (outer boundary): zero-gradient outflow
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

    // theta = 0 (north pole): axis symmetry
    for (int g = 1; g <= ng; ++g) {
        for (int i = -ng; i < nr + ng; ++i) {
            int k_ghost = grid.idx(i, -g);
            int k_phys = grid.idx(i, g - 1);

            state.rho[k_ghost] = state.rho[k_phys];
            state.mr[k_ghost] = state.mr[k_phys];
            state.mtheta[k_ghost] = -state.mtheta[k_phys]; // reflect v_theta
            state.E[k_ghost] = state.E[k_phys];
        }
    }

    // theta = pi (south pole): axis symmetry
    for (int g = 0; g < ng; ++g) {
        for (int i = -ng; i < nr + ng; ++i) {
            int k_ghost = grid.idx(i, nt + g);
            int k_phys = grid.idx(i, nt - 1 - g);

            state.rho[k_ghost] = state.rho[k_phys];
            state.mr[k_ghost] = state.mr[k_phys];
            state.mtheta[k_ghost] = -state.mtheta[k_phys]; // reflect v_theta
            state.E[k_ghost] = state.E[k_phys];
        }
    }
}
