#include "drivers/drivers.h"

#ifdef USE_GPU
#include "wb2d_solver.cuh"
#include "init/lane_emden.h"
#include "io/output.h"
#include "sim/helpers.h"
#include "sim/run_loop.h"

#include <cstdio>
#include <ctime>

int run_wb2d(SimConfig& cfg, SimContext& ctx, double& t, int& step) {
    // ===== Well-Balanced 2D Eulerian (MESA-stabilized) =====
    Wb2DSolver wb;
    wb.limiter_type = static_cast<int>(cfg.limiter);
    wb.hllc_variant = cfg.hllc_variant;
    wb.init(ctx.grid, ctx.eos, cfg.G, cfg.cfl);
    if (cfg.mesh_type == "mass") {
        wb.n_pole_avg = cfg.ntheta / 2;
        if (cfg.r_inner <= 0) {
            int n_uni = static_cast<int>(0.15 * cfg.R_outer / (cfg.R_outer / cfg.nr));
            wb.n_angular_avg = n_uni;
            wb.central_damp_r = 0.15 * cfg.R_outer;
        }
    }
    if (!cfg.no_sponge) wb.sponge_kappa = 100.0;

    if (cfg.test_case == "lane_emden_perturbed" || cfg.test_case == "bubble") {
        State state_hse;
        state_hse.allocate(ctx.grid);
        LaneEmdenParams lep;
        lep.n_poly = 1.5; lep.rho_c = 1.0; lep.K_poly = 1.0; lep.G = cfg.G;
        init_lane_emden(ctx.grid, state_hse, lep, cfg.gamma);
        wb.upload_state(ctx.grid, state_hse);
        wb.snapshot_hse();
    }
    wb.upload_state(ctx.grid, ctx.state);
    if (!(cfg.test_case == "lane_emden_perturbed" || cfg.test_case == "bubble"))
        wb.snapshot_hse();

    std::timespec wall_start;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);

    while (t < cfg.t_end && !g_interrupted) {
        double dt = wb.step(t, cfg.t_end);
        t += dt;
        step++;

        if (step % 200 == 0)
            print_progress(t, cfg.t_end, step, dt, wall_start);

        if (step % cfg.output_interval == 0) {
            wb.download_state(ctx.grid, ctx.state);
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
    wb.download_state(ctx.grid, ctx.state);
    wb.destroy();
    return 0;
}
#endif
