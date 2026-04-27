#include "evrard.h"
#include <cmath>

void init_evrard(const Grid& grid, State& state,
                 const EvrardParams& params, double gamma) {
    // rho(r) = M / (2*pi*R^3) * 1/r  for r < R
    // e = 0.05 * G*M/R (cold but nonzero thermal energy)
    double e_spec = 0.05 * params.G * params.M / params.R;

    for (int i = 0; i < grid.nr; ++i) {
        double r = grid.r_center[i];
        for (int j = 0; j < grid.ntheta; ++j) {
            int k = grid.idx(i, j);
            PrimitiveVars w;

            if (r < params.R && r > 0.0) {
                w.rho = params.M / (2.0 * M_PI * params.R * params.R * params.R * r);
            } else {
                w.rho = 1e-10 * params.M / (params.R * params.R * params.R);
            }

            w.vr = 0.0;
            w.vtheta = 0.0;
            w.P = (gamma - 1.0) * w.rho * e_spec;

            state.from_primitive(k, w, gamma);
        }
    }
}
