#include "sim/run_loop.h"

#include "io/output.h"

#include <cstdio>
#include <ctime>

volatile std::sig_atomic_t g_interrupted = 0;

#ifdef USE_GPU
void run_time_loop(const SimConfig& cfg, SimContext& ctx,
                   double& t, int& step, SolverOps& ops) {
    std::timespec wall_start;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);

    while (t < cfg.t_end && !g_interrupted) {
        double dt = ops.step(t, cfg.t_end);
        t += dt;
        step++;

        if (step % ops.progress_interval == 0)
            print_progress(t, cfg.t_end, step, dt, wall_start);

        if (step % cfg.output_interval == 0) {
            ops.download(ctx.grid, ctx.state, dt);
            Diagnostics diag = compute_diagnostics(ctx.grid, ctx.state, cfg.gamma);
            std::fprintf(stderr, "\n");
            std::printf("Step %6d  t = %.6e  dt = %.3e  M = %.10e  E = %.10e\n",
                        step, t, dt, diag.total_mass, diag.total_energy);
            char fname[512];
            std::snprintf(fname, sizeof(fname), "%s/output_%04d.vtk",
                          ctx.run_dir.c_str(), step / cfg.output_interval);
            write_vtk(fname, ctx.grid, ctx.state, cfg.gamma);
        }
    }
    std::fprintf(stderr, "\n");
    ops.download(ctx.grid, ctx.state, 0.0);
    ops.destroy();
}
#endif
