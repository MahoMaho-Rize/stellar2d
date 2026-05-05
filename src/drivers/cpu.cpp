#include "drivers/drivers.h"

#ifndef USE_GPU
#include "bc/boundary.h"
#include "hydro/flux.h"
#include "hydro/integrate.h"
#include "hydro/reconstruct.h"
#include "gravity/gmg.h"
#include "io/output.h"
#include "sim/helpers.h"
#include "sim/run_loop.h"   // g_interrupted extern decl

#include <cstdio>
#include <ctime>
#include <vector>

// Main time integrator for the CPU-only build. Uses FAS-free axisymmetric
// RK2 with a central-difference Poisson solver (gravity/gmg) and
// flux-accumulator RHS. Keeps behavior bit-for-bit identical to the code
// that used to live inlined in main.cpp under the USE_GPU=OFF branch.
int run_cpu(SimConfig& cfg, SimContext& ctx, double& t, int& step) {
    State state_tmp;
    state_tmp.allocate(ctx.grid);

    PoissonGMG poisson_solver;
    poisson_solver.init(ctx.grid);

    FluxAccumulator acc;
    acc.allocate(ctx.grid.total_cells());

    std::vector<double> rho_cells;
    std::vector<double> poisson_rhs;

    std::timespec wall_start;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);

    while (t < cfg.t_end && !g_interrupted) {
        fill_ghost_cells(ctx.grid, ctx.state, cfg.gamma);

        double dt = compute_cfl_dt(ctx.grid, ctx.state, ctx.eos, cfg.cfl);
        if (t + dt > cfg.t_end) dt = cfg.t_end - t;

        state_tmp.copy_from(ctx.state);

        compute_rhs(ctx.grid, ctx.state, ctx.eos, acc, cfg.limiter);
        extract_density(ctx.grid, ctx.state, rho_cells);
        compute_poisson_rhs(ctx.grid, rho_cells, cfg.G, poisson_rhs);
        poisson_solver.solve(poisson_rhs.data(), ctx.state.phi.data());
        add_gravity_source(ctx.grid, ctx.state, acc);
        rk2_substep(ctx.grid, ctx.state, acc, dt);

        fill_ghost_cells(ctx.grid, ctx.state, cfg.gamma);
        compute_rhs(ctx.grid, ctx.state, ctx.eos, acc, cfg.limiter);
        extract_density(ctx.grid, ctx.state, rho_cells);
        compute_poisson_rhs(ctx.grid, rho_cells, cfg.G, poisson_rhs);
        poisson_solver.solve(poisson_rhs.data(), ctx.state.phi.data());
        add_gravity_source(ctx.grid, ctx.state, acc);
        rk2_substep(ctx.grid, ctx.state, acc, dt);

        int nr = ctx.grid.nr, nt = ctx.grid.ntheta;
        for (int i = 0; i < nr; ++i) {
            for (int j = 0; j < nt; ++j) {
                int k = ctx.grid.idx(i, j);
                ctx.state.rho[k] = 0.5 * (state_tmp.rho[k] + ctx.state.rho[k]);
                ctx.state.mr[k] = 0.5 * (state_tmp.mr[k] + ctx.state.mr[k]);
                ctx.state.mtheta[k] = 0.5 * (state_tmp.mtheta[k] + ctx.state.mtheta[k]);
                ctx.state.E[k] = 0.5 * (state_tmp.E[k] + ctx.state.E[k]);
            }
        }

        t += dt;
        step++;

        if (step % 200 == 0)
            print_progress(t, cfg.t_end, step, dt, wall_start);

        if (step % cfg.output_interval == 0) {
            std::fprintf(stderr, "\n");
            Diagnostics diag = compute_diagnostics(ctx.grid, ctx.state, cfg.gamma);
            std::printf("Step %6d  t = %.6e  dt = %.3e  M = %.10e  E = %.10e\n",
                        step, t, dt, diag.total_mass, diag.total_energy);

            char fname[512];
            std::snprintf(fname, sizeof(fname), "%s/output_%04d.vtk",
                          ctx.run_dir.c_str(), step / cfg.output_interval);
            write_vtk(fname, ctx.grid, ctx.state, cfg.gamma);
        }
    }
    std::fprintf(stderr, "\n");
    return 0;
}
#endif
