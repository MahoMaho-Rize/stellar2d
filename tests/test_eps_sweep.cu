// Perturbation amplitude vs dt ceiling sweep.
// Tests the hypothesis: smaller initial perturbation → larger acceptable dt.

#include "lowmach_solver.h"
#include "init/lane_emden.h"
#include <cstdio>
#include <cmath>
#include <vector>

int main() {
    int nr = 64, nt = 32;
    double gamma = 5.0 / 3.0;
    double G = 1.0;

    LaneEmdenParams lep;
    lep.n_poly = 1.5; lep.rho_c = 1.0; lep.K_poly = 1.0; lep.G = G;

    auto sol = solve_lane_emden(lep.n_poly);
    double alpha = std::sqrt((lep.n_poly + 1.0) * lep.K_poly
        * std::pow(lep.rho_c, 1.0 / lep.n_poly - 1.0)
        / (4.0 * M_PI * lep.G));
    double R_outer = alpha * sol.xi_1 * 1.1;

    Grid grid;
    grid.init(nr, nt, R_outer, 2.0);
    EOS eos(gamma);

    State state_hse;
    state_hse.allocate(grid);
    init_lane_emden(grid, state_hse, lep, gamma);

    double eps_values[] = {1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 0.0};
    int n_eps = 7;

    double dt_targets[] = {1e-7, 5e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};
    int n_dt = 8;

    std::fprintf(stderr, "=== Perturbation amplitude vs dt ceiling ===\n");
    std::fprintf(stderr, "Grid: %dx%d, Lane-Emden\n\n", nr, nt);
    std::fprintf(stderr, "  %-10s  %-12s", "eps", "||R0||");
    for (int d = 0; d < n_dt; d++) std::fprintf(stderr, "  dt=%.0e", dt_targets[d]);
    std::fprintf(stderr, "\n");

    for (int e = 0; e < n_eps; e++) {
        double eps = eps_values[e];

        State state_pert;
        state_pert.allocate(grid);
        if (eps > 0)
            init_lane_emden_perturbed(grid, state_pert, lep, gamma, eps);
        else
            init_lane_emden(grid, state_pert, lep, gamma);

        // Measure initial residual
        LowMachSolver lm_tmp;
        lm_tmp.init(grid, eos, G, 0.4);
        lm_tmp.upload_state(grid, state_hse);
        lm_tmp.snapshot_hse();
        lm_tmp.upload_state(grid, state_pert);
        lm_tmp.compute_gravity_1d();
        lm_tmp.compute_residual(lm_tmp.d_residual);
        int n = nr * nt;
        std::vector<double> res(4*n);
        cudaMemcpy(res.data(), lm_tmp.d_residual, 4*n*sizeof(double), cudaMemcpyDeviceToHost);
        double l2 = 0;
        for (int i = 0; i < 4*n; i++) l2 += res[i]*res[i];
        l2 = std::sqrt(l2);
        lm_tmp.destroy();

        std::fprintf(stderr, "  %-10.0e  %-12.3e", eps, l2);

        for (int d = 0; d < n_dt; d++) {
            LowMachSolver lm;
            lm.init(grid, eos, G, 0.4);
            lm.upload_state(grid, state_hse);
            lm.snapshot_hse();
            lm.upload_state(grid, state_pert);

            lm.dt_current = dt_targets[d];
            double dt_acc = lm.step(0.0, 1.0);
            double ratio = dt_acc / dt_targets[d];

            if (ratio >= 0.99) std::fprintf(stderr, "  %7s", "OK");
            else if (ratio > 0.1) std::fprintf(stderr, "  %6.0f%%", ratio*100);
            else std::fprintf(stderr, "  %7s", "FAIL");

            lm.destroy();
        }
        std::fprintf(stderr, "\n");
    }

    std::fprintf(stderr, "\n=== Done ===\n");
    return 0;
}
