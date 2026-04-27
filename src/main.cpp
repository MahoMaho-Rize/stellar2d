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
    double G = 1.0; // dimensionless
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

    std::printf("stellar2d POC - 2D Axisymmetric Euler + Self-Gravity\n");
    std::printf("Test case: %s\n", cfg.test_case.c_str());
    std::printf("Grid: %d x %d, R_outer = %.4f\n", cfg.nr, cfg.ntheta, cfg.R_outer);

    Grid grid;
    grid.init(cfg.nr, cfg.ntheta, cfg.R_outer, cfg.log_alpha);

    EOS eos(cfg.gamma);

    State state, state_tmp;
    state.allocate(grid);
    state_tmp.allocate(grid);

    // Initialize based on test case
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

    // Setup Poisson solver
    PoissonMatrix poisson_mat;
    poisson_mat.assemble(grid, cfg.G);

    AmgxPoissonSolver poisson_solver;
    poisson_solver.init(cfg.amgx_config);
    poisson_solver.setup(poisson_mat);

    // Allocate work arrays
    FluxAccumulator acc;
    acc.allocate(grid.total_cells());

    std::vector<double> rho_cells;

    double t = 0.0;
    int step = 0;

    // Initial output
    write_vtk("output_0000.vtk", grid, state, cfg.gamma);

    std::printf("Starting time integration...\n");

    while (t < cfg.t_end) {
        fill_ghost_cells(grid, state, cfg.gamma);

        double dt = compute_cfl_dt(grid, state, eos, cfg.cfl);
        if (t + dt > cfg.t_end) dt = cfg.t_end - t;

        // === RK2 Stage 1 ===
        state_tmp.copy_from(state);

        // Hydro RHS (flux divergence + geometric source)
        compute_rhs(grid, state, eos, acc, cfg.limiter);

        // Poisson solve for gravity
        extract_density(grid, state, rho_cells);
        poisson_mat.set_rhs(grid, rho_cells, cfg.G);
        poisson_solver.solve(poisson_mat.rhs.data(), state.phi.data());

        // Add gravity source
        add_gravity_source(grid, state, acc);

        // Update: U* = U^n + dt * R(U^n)
        rk2_substep(grid, state, acc, dt);

        // === RK2 Stage 2 ===
        fill_ghost_cells(grid, state, cfg.gamma);

        compute_rhs(grid, state, eos, acc, cfg.limiter);

        extract_density(grid, state, rho_cells);
        poisson_mat.set_rhs(grid, rho_cells, cfg.G);
        poisson_solver.solve(poisson_mat.rhs.data(), state.phi.data());

        add_gravity_source(grid, state, acc);

        // U^{n+1} = 0.5 * U^n + 0.5 * (U* + dt * R(U*))
        rk2_substep(grid, state, acc, dt);

        // Average: U^{n+1} = 0.5*(U^n + U*_updated)
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
