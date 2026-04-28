#include "jeans.h"
#include <cmath>

void init_jeans(const Grid& grid, State& state,
                const JeansParams& params, const EOS& eos) {
    double P_0 = params.rho_0 * params.cs * params.cs / eos.gamma;

    for (int i = 0; i < grid.nr; ++i) {
        double r = grid.r_center[i];
        for (int j = 0; j < grid.ntheta; ++j) {
            double theta = grid.theta_center[j];
            int k = grid.idx(i, j);

            // Eq. (9.8)
            double pert = params.epsilon * std::cos(params.k_r * r) * std::cos(params.k_theta * theta);

            PrimitiveVars w;
            w.rho = params.rho_0 * (1.0 + pert);       // Eq. (9.8)
            w.vr = 0.0;
            w.vtheta = 0.0;
            w.P = P_0 * (1.0 + eos.gamma * pert);

            state.from_primitive(k, w, eos);
        }
    }
}
