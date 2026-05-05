#include "grid.h"
#include "state.h"
#include "eos.h"
#include "bc/boundary.h"
#include "hydro/flux.h"
#include "hydro/integrate.h"
#include "hydro/reconstruct.h"
#include "gravity/gmg.h"
#include "io/output.h"
#include "init/lane_emden.h"
#include "init/sedov.h"
#include "init/jeans.h"
#include "init/evrard.h"

#ifdef USE_GPU
#include <cuda_runtime.h>
#ifdef USE_AMGX
#include "gpu_solver.h"
#endif
#include "lowmach_solver.h"
#include "fas_solver.cuh"
#include "fas2_solver.cuh"
#include "simple_solver.cuh"
#include "projection_solver.cuh"
#include "radial1d_solver.cuh"
#include "wb2d_solver.cuh"
#include "ale2d_solver.cuh"
#include "cart_lag_solver.cuh"
#include "cart_ale_solver.cuh"
#include "cart_ale2_solver.cuh"
#include "cart_impl_solver.cuh"
#include "pseudo_spectral_solver.cuh"
#include "anelastic_sl_solver.cuh"
#include "stellar_profile.h"
#include "sph2d_spectral_solver.cuh"
#include "physics/helmholtz_eos.cuh"
#endif

#include <cstdio>
#include <cmath>
#include <string>
#include <cstring>
#include <ctime>
#include <array>
#include <vector>
#include <csignal>
#include <functional>
#include <algorithm>
#include <sys/stat.h>

#include "cli/options.h"
#include "sim/helpers.h"
#include "sim/setup.h"
#include "sim/run_loop.h"
#include "drivers/drivers.h"

// g_interrupted is defined in sim/run_loop.cpp.
static void handle_sigint(int) { g_interrupted = 1; }


int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);
    std::signal(SIGTERM, handle_sigint);

    SimConfig cfg;

    if (int rc = parse_cli(argc, argv, cfg); rc != 0) return rc;

    SimContext ctx;
    if (int rc = setup_simulation(cfg, ctx); rc != 0) return rc;

    // Local aliases so the (still-inlined) solver branches below keep working
    // without modification. The coming steps migrate each branch into its own
    // run_xxx(cfg, ctx) driver; until then we expose ctx members as bare names.
    Grid& grid = ctx.grid;
    State& state = ctx.state;
    EOS& eos = ctx.eos;
    std::string& run_dir = ctx.run_dir;
#ifdef USE_GPU
    HelmholtzTable& helm_tbl = ctx.helm_tbl;
    bool& helm_loaded = ctx.helm_loaded;
    KapTable& kap_tbl_lowT = ctx.kap_tbl_lowT;
    KapTable& kap_tbl_highT = ctx.kap_tbl_highT;
    bool& kap_loaded = ctx.kap_loaded;
#endif

    double t = 0.0;
    int step = 0;

    std::printf("Starting time integration...\n");

#ifdef USE_GPU
    if (cfg.solver_type == "radial1d") {
        if (int rc = run_radial1d(cfg, ctx, t, step); rc != 0) return rc;
    } else if (cfg.solver_type == "projection") {
        if (int rc = run_projection(cfg, ctx, t, step); rc != 0) return rc;
    } else if (cfg.solver_type == "simple") {
        if (int rc = run_simple(cfg, ctx, t, step); rc != 0) return rc;
    } else if (cfg.solver_type == "fas2") {
        if (int rc = run_fas2(cfg, ctx, t, step); rc != 0) return rc;
    } else if (cfg.solver_type == "fas" || cfg.solver_type == "explicit") {
        if (int rc = run_fas(cfg, ctx, t, step); rc != 0) return rc;


    } else if (cfg.solver_type == "cart_lag") {
        if (int rc = run_cart_lag(cfg, ctx, t, step); rc != 0) return rc;
    } else if (cfg.solver_type == "cart_ale") {
        if (int rc = run_cart_ale(cfg, ctx, t, step); rc != 0) return rc;
    } else if (cfg.solver_type == "cart_impl") {
        if (int rc = run_cart_impl(cfg, ctx, t, step); rc != 0) return rc;
    } else if (cfg.solver_type == "cart_ale2") {
        if (int rc = run_cart_ale2(cfg, ctx, t, step); rc != 0) return rc;

    } else if (cfg.solver_type == "pseudo_spectral") {
        if (int rc = run_pseudo_spectral(cfg, ctx, t, step); rc != 0) return rc;

    } else if (cfg.solver_type == "anelastic_sl") {
        if (int rc = run_anelastic_sl(cfg, ctx, t, step); rc != 0) return rc;

    } else if (cfg.solver_type == "sph2d_spectral") {
        if (int rc = run_sph2d_spectral(cfg, ctx, t, step); rc != 0) return rc;

    } else if (cfg.solver_type == "ale2d") {
        if (int rc = run_ale2d(cfg, ctx, t, step); rc != 0) return rc;
    } else if (cfg.solver_type == "wb2d") {
        if (int rc = run_wb2d(cfg, ctx, t, step); rc != 0) return rc;
    } else if (cfg.solver_type == "lowmach") {
        if (int rc = run_lowmach(cfg, ctx, t, step); rc != 0) return rc;
    } else {
#ifdef USE_AMGX
        if (int rc = run_compressible(cfg, ctx, t, step); rc != 0) return rc;
#else
        std::fprintf(stderr, "ERROR: --solver compressible requires AmgX. "
                     "Rebuild with -DAMGX_DIR=/path/to/amgx, or use --solver lowmach.\n");
        return 1;
#endif
    }


#else
    // ===== CPU path =====
    State state_tmp;
    state_tmp.allocate(grid);

    PoissonGMG poisson_solver;
    poisson_solver.init(grid);

    FluxAccumulator acc;
    acc.allocate(grid.total_cells());

    std::vector<double> rho_cells;
    std::vector<double> poisson_rhs;

    std::timespec wall_start;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);

    while (t < cfg.t_end && !g_interrupted) {
        fill_ghost_cells(grid, state, cfg.gamma);

        double dt = compute_cfl_dt(grid, state, eos, cfg.cfl);
        if (t + dt > cfg.t_end) dt = cfg.t_end - t;

        state_tmp.copy_from(state);

        compute_rhs(grid, state, eos, acc, cfg.limiter);
        extract_density(grid, state, rho_cells);
        compute_poisson_rhs(grid, rho_cells, cfg.G, poisson_rhs);
        poisson_solver.solve(poisson_rhs.data(), state.phi.data());
        add_gravity_source(grid, state, acc);
        rk2_substep(grid, state, acc, dt);

        fill_ghost_cells(grid, state, cfg.gamma);
        compute_rhs(grid, state, eos, acc, cfg.limiter);
        extract_density(grid, state, rho_cells);
        compute_poisson_rhs(grid, rho_cells, cfg.G, poisson_rhs);
        poisson_solver.solve(poisson_rhs.data(), state.phi.data());
        add_gravity_source(grid, state, acc);
        rk2_substep(grid, state, acc, dt);

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

        if (step % 200 == 0)
            print_progress(t, cfg.t_end, step, dt, wall_start);

        if (step % cfg.output_interval == 0) {
            std::fprintf(stderr, "\n");
            Diagnostics diag = compute_diagnostics(grid, state, cfg.gamma);
            std::printf("Step %6d  t = %.6e  dt = %.3e  M = %.10e  E = %.10e\n",
                        step, t, dt, diag.total_mass, diag.total_energy);

            char fname[512];
            std::snprintf(fname, sizeof(fname), "%s/output_%04d.vtk", run_dir.c_str(), step / cfg.output_interval);
            write_vtk(fname, grid, state, cfg.gamma);
        }
    }
    std::fprintf(stderr, "\n");
#endif

    Diagnostics diag = compute_diagnostics(grid, state, cfg.gamma);
    std::printf("Final: step %d  t = %.6e  M = %.10e  E = %.10e\n",
                step, t, diag.total_mass, diag.total_energy);
    {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/output_final.vtk", run_dir.c_str());
        write_vtk(path, grid, state, cfg.gamma);
    }

#ifdef USE_GPU
    ctx.destroy_tables();
#endif

    std::printf("Done.\n");
    return 0;
}
