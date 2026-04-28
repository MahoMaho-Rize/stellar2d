#include "sedov.h"
#include <cmath>

void init_sedov(const Grid& grid, State& state,
                const SedovParams& params, double gamma) {
    double P_ambient = 1e-10;
    double vol_blast = 0.0;

    for (int i = 0; i < grid.nr; ++i) {
        if (grid.r_center[i] > params.r_blast) break;
        for (int j = 0; j < grid.ntheta; ++j) {
            vol_blast += grid.cell_volume[i * grid.ntheta + j]; // Eq. (2.2)
        }
    }

    // Eq. (9.6): specific internal energy in blast region
    double e_blast = params.E_blast / vol_blast / params.rho_0;

    for (int i = 0; i < grid.nr; ++i) {
        for (int j = 0; j < grid.ntheta; ++j) {
            int k = grid.idx(i, j);
            PrimitiveVars w;
            w.rho = params.rho_0;
            w.vr = 0.0;
            w.vtheta = 0.0;

            if (grid.r_center[i] <= params.r_blast) {
                w.P = (gamma - 1.0) * params.rho_0 * e_blast; // Eq. (9.6)
            } else {
                w.P = P_ambient;
            }

            state.from_primitive(k, w, gamma);
        }
    }
}
