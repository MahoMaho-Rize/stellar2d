#include "grid.h"
#include "state.h"
#include "eos.h"
#include "bc/boundary.h"
#include "hydro/flux.h"
#include "hydro/integrate.h"
#include "hydro/reconstruct.h"
#include "gravity/poisson.h"
#include "gravity/amgx_solver.h"
#include "io/output.h"
#include "init/lane_emden.h"
#include "init/sedov.h"
#include "init/jeans.h"
#include "init/evrard.h"

#include <cstdio>
#include <cmath>
#include <string>
#include <cstring>

struct SimConfig {
    int nr = 128;
    int ntheta = 64;
    double R_outer = 1.0;
    double log_alpha = 2.0;
    double gamma = 5.0 / 3.0;
    double cfl = 0.4;
    double t_end = 1.0;
    int output_interval = 100;
    double G = 1.0;
    std::string test_case = "lane_emden";
    std::string amgx_config = "config/amgx.json";
    Limiter limiter = Limiter::MINMOD;
};

static void extract_density(const Grid& grid, const State& state, std::vector<double>& rho_cells) {
    int nr = grid.nr, nt = grid.ntheta;
    rho_cells.resize(nr * nt);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nt; ++j)
            rho_cells[i * nt + j] = state.rho[grid.idx(i, j)];
}

static double compute_lane_emden_R_outer(double n_poly, double K_poly, double rho_c, double G) {
    auto sol = solve_lane_emden(n_poly);
    // Eq. (9.2): alpha^2 = (n+1)*K*rho_c^(1/n-1) / (4*pi*G)
    double alpha2 = (n_poly + 1.0) * K_poly
                    * std::pow(rho_c, 1.0 / n_poly - 1.0)
                    / (4.0 * M_PI * G);
    double alpha = std::sqrt(alpha2);
    double R_star = alpha * sol.xi_1; // Eq. (9.3)
    return R_star * 1.1; // 10% padding beyond stellar surface
}

int main(int argc, char** argv) {
    SimConfig cfg;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--test") == 0 && i + 1 < argc)
            cfg.test_case = argv[++i];
        else if (std::strcmp(argv[i], "--nr") == 0 && i + 1 < argc)
            cfg.nr = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ntheta") == 0 && i + 1 < argc)
            cfg.ntheta = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--tend") == 0 && i + 1 < argc)
            cfg.t_end = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--cfl") == 0 && i + 1 < argc)
            cfg.cfl = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--output-interval") == 0 && i + 1 < argc)
            cfg.output_interval = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--amgx-config") == 0 && i + 1 < argc)
            cfg.amgx_config = argv[++i];
    }

    // Bug 1 fix: compute R_outer from Lane-Emden solution before grid init
    // Eq. (9.3): R_star = alpha * xi_1
    if (cfg.test_case == "lane_emden" || cfg.test_case == "lane_emden_perturbed") {
        cfg.R_outer = compute_lane_emden_R_outer(1.5, 1.0, 1.0, cfg.G);
    }

    std::printf("stellar2d POC - 2D Axisymmetric Euler + Self-Gravity\n");
    std::printf("Test case: %s\n", cfg.test_case.c_str());
    std::printf("Grid: %d x %d, R_outer = %.6f\n", cfg.nr, cfg.ntheta, cfg.R_outer);

    Grid grid;
    grid.init(cfg.nr, cfg.ntheta, cfg.R_outer, cfg.log_alpha);

    EOS eos(cfg.gamma);

    State state, state_tmp;
    state.allocate(grid);
    state_tmp.allocate(grid);

    if (cfg.test_case == "lane_emden") {
        LaneEmdenParams lep;
        lep.n_poly = 1.5;
        lep.rho_c = 1.0;
        lep.K_poly = 1.0;
        lep.G = cfg.G;
        init_lane_emden(grid, state, lep, cfg.gamma);
    } else if (cfg.test_case == "lane_emden_perturbed") {
        LaneEmdenParams lep;
        lep.n_poly = 1.5;
        lep.rho_c = 1.0;
        lep.K_poly = 1.0;
        lep.G = cfg.G;
        init_lane_emden_perturbed(grid, state, lep, cfg.gamma, 1e-3);
    } else if (cfg.test_case == "sedov") {
        SedovParams sp;
        sp.rho_0 = 1.0;
        sp.E_blast = 1.0;
        sp.r_blast = 0.05;
        init_sedov(grid, state, sp, cfg.gamma);
    } else if (cfg.test_case == "jeans") {
        JeansParams jp;
        jp.rho_0 = 1.0;
        jp.cs = 1.0;
        jp.G = cfg.G;
        jp.epsilon = 1e-3;
        jp.k_r = 2.0 * M_PI / cfg.R_outer;
        jp.k_theta = 2.0;
        init_jeans(grid, state, jp, cfg.gamma);
    } else if (cfg.test_case == "evrard") {
        EvrardParams ep;
        ep.M = 1.0;
        ep.R = 1.0;
        ep.G = cfg.G;
        init_evrard(grid, state, ep, cfg.gamma);
    } else {
        std::fprintf(stderr, "Unknown test case: %s\n", cfg.test_case.c_str());
        return 1;
    }

    PoissonMatrix poisson_mat;
    poisson_mat.assemble(grid, cfg.G);

    AmgxPoissonSolver poisson_solver;
    poisson_solver.init(cfg.amgx_config);
    poisson_solver.setup(poisson_mat);

    FluxAccumulator acc;
    acc.allocate(grid.total_cells());

    std::vector<double> rho_cells;

    double t = 0.0;
    int step = 0;

    write_vtk("output_0000.vtk", grid, state, cfg.gamma);

    std::printf("Starting time integration...\n");

    while (t < cfg.t_end) {
        fill_ghost_cells(grid, state, cfg.gamma);                   // Eq. (8.1)-(8.3)

        double dt = compute_cfl_dt(grid, state, eos, cfg.cfl);     // Eq. (7.4)
        if (t + dt > cfg.t_end) dt = cfg.t_end - t;

        // === Eq. (7.1): RK2 Stage 1 — U* = U^n + dt * R(U^n) ===
        state_tmp.copy_from(state);

        compute_rhs(grid, state, eos, acc, cfg.limiter);            // Eq. (2.5), (5.1)-(5.4)

        extract_density(grid, state, rho_cells);
        poisson_mat.set_rhs(grid, rho_cells, cfg.G);               // Eq. (1.8), (6.6)
        poisson_solver.solve(poisson_mat.rhs.data(), state.phi.data());

        add_gravity_source(grid, state, acc);                       // Eq. (6.1)-(6.3)

        rk2_substep(grid, state, acc, dt);                          // Eq. (7.1)

        // === Eq. (7.2): RK2 Stage 2 — U** = U* + dt * R(U*) ===
        fill_ghost_cells(grid, state, cfg.gamma);                   // Eq. (8.1)-(8.3)

        compute_rhs(grid, state, eos, acc, cfg.limiter);            // Eq. (2.5), (5.1)-(5.4)

        extract_density(grid, state, rho_cells);
        poisson_mat.set_rhs(grid, rho_cells, cfg.G);               // Eq. (1.8), (6.6)
        poisson_solver.solve(poisson_mat.rhs.data(), state.phi.data());

        add_gravity_source(grid, state, acc);                       // Eq. (6.1)-(6.3)

        rk2_substep(grid, state, acc, dt);                          // Eq. (7.2)

        // Eq. (7.3): U^{n+1} = 0.5 * (U^n + U**)
        int nr = grid.nr, nt = grid.ntheta;
        for (int i = 0; i < nr; ++i) {
            for (int j = 0; j < nt; ++j) {
                int k = grid.idx(i, j);
                state.rho[k] = 0.5 * (state_tmp.rho[k] + state.rho[k]);
                state.mr[k] = 0.5 * (state_tmp.mr[k] + state.mr[k]);
                state.mtheta[k] = 0.5 * (state_tmp.mtheta[k] + state.mtheta[k]);
                state.E[k] = 0.5 * (state_tmp.E[k] + state.E[k]);
            }
        }

        t += dt;
        step++;

        if (step % cfg.output_interval == 0) {
            Diagnostics diag = compute_diagnostics(grid, state, cfg.gamma);
            std::printf("Step %6d  t = %.6e  dt = %.3e  M = %.10e  E = %.10e\n",
                        step, t, dt, diag.total_mass, diag.total_energy);

            char fname[256];
            std::snprintf(fname, sizeof(fname), "output_%04d.vtk", step / cfg.output_interval);
            write_vtk(fname, grid, state, cfg.gamma);
        }
    }

    Diagnostics diag = compute_diagnostics(grid, state, cfg.gamma);
    std::printf("Final: step %d  t = %.6e  M = %.10e  E = %.10e\n",
                step, t, diag.total_mass, diag.total_energy);
    write_vtk("output_final.vtk", grid, state, cfg.gamma);

    poisson_solver.destroy();
    std::printf("Done.\n");
    return 0;
}
