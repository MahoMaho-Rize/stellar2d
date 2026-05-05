#include "drivers/drivers.h"

#ifdef USE_GPU
#include "lowmach_solver.h"
#include "io/output.h"
#include "sim/helpers.h"
#include "sim/run_loop.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>

int run_lowmach(SimConfig& cfg, SimContext& ctx, double& t, int& step) {
    // ===== GPU low-Mach path =====
    PrecondType pc = PrecondType::LINE_JACOBI;
    if (cfg.precond == "none")          pc = PrecondType::NONE;
    else if (cfg.precond == "block_jacobi") pc = PrecondType::BLOCK_JACOBI;
    else if (cfg.precond == "simple")   pc = PrecondType::SIMPLE;
    else if (cfg.precond == "line_jacobi") pc = PrecondType::LINE_JACOBI;
    else if (cfg.precond == "block_schur") pc = PrecondType::BLOCK_SCHUR;
    else if (cfg.precond == "combined") pc = PrecondType::COMBINED;
    else if (cfg.precond == "pbp")      pc = PrecondType::PBP;

    LowMachSolver lm;
    lm.init(ctx.grid, ctx.eos, cfg.G, cfg.cfl, pc);
    if (cfg.no_sponge) lm.sponge_kappa = 0.0;
    configure_mass_mesh(cfg, lm);
    snapshot_hse_if_needed(cfg, ctx, lm);
    lm.upload_state(ctx.grid, ctx.state);

    std::timespec wall_start;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);

    while (t < cfg.t_end && !g_interrupted) {
        double dt = lm.step(t, cfg.t_end);
        t += dt;
        step++;

        if (step % 200 == 0)
            print_progress(t, cfg.t_end, step, dt, wall_start);

        if (step % cfg.output_interval == 0) {
            lm.download_state(ctx.grid, ctx.state);
            Diagnostics diag = compute_diagnostics(ctx.grid, ctx.state, cfg.gamma);

            double max_vr = 0, max_vt = 0;
            double rho_thresh = lm.atm_rho_thresh;
            for (int i = 0; i < ctx.grid.nr; i++)
                for (int j = 0; j < ctx.grid.ntheta; j++) {
                    int k = ctx.grid.idx(i, j);
                    if (ctx.state.rho[k] < rho_thresh) continue;
                    double rho = ctx.state.rho[k];
                    max_vr = std::max(max_vr, std::fabs(ctx.state.mr[k] / rho));
                    max_vt = std::max(max_vt, std::fabs(ctx.state.mtheta[k] / rho));
                }

            std::fprintf(stderr, "\n");
            std::printf("Step %6d  t = %.6e  dt = %.3e  M = %.10e  E = %.10e  |vr|=%.3e |vt|=%.3e\n",
                        step, t, dt, diag.total_mass, diag.total_energy, max_vr, max_vt);

            char fname[512];
            std::snprintf(fname, sizeof(fname), "%s/output_%04d.vtk",
                          ctx.run_dir.c_str(), step / cfg.output_interval);
            write_vtk(fname, ctx.grid, ctx.state, cfg.gamma);
        }
    }
    std::fprintf(stderr, "\n");

    lm.download_state(ctx.grid, ctx.state);
    lm.destroy();
    return 0;
}

#ifdef USE_AMGX
#include "gpu_solver.h"

int run_compressible(SimConfig& cfg, SimContext& ctx, double& t, int& step) {
    // ===== GPU compressible path (HLLC + JFNK) =====
    GpuSolver gpu;
    gpu.init(ctx.grid, ctx.eos, cfg.G, cfg.cfl, cfg.limiter);
    gpu.upload_state(ctx.grid, ctx.state);

    SolverOps ops;
    ops.step = [&](double t_, double te) { return gpu.step(t_, te); };
    ops.download = [&](const Grid& g, State& s, double) { gpu.download_state(g, s); };
    ops.destroy = [&]() { gpu.destroy(); };
    run_time_loop(cfg, ctx, t, step, ops);
    return 0;
}
#endif
#endif
