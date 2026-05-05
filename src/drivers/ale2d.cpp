#include "drivers/drivers.h"

#ifdef USE_GPU
#include "ale2d_solver.cuh"
#include "sim/helpers.h"
#include "sim/run_loop.h"

#include <cstdio>
#include <ctime>
#include <vector>

int run_ale2d(SimConfig& cfg, SimContext& ctx, double& t, int& step) {
    // ===== 2D axisymmetric Lagrangian (Caramana compatible) =====
    if (cfg.test_case != "lane_emden" && cfg.test_case != "lane_emden_perturbed") {
        std::fprintf(stderr, "ERROR: ale2d currently supports lane_emden / lane_emden_perturbed only\n");
        return 1;
    }
    Ale2DSolver ale;
    ale.init(ctx.grid, ctx.eos, cfg.G, cfg.cfl);
    ale.init_lane_emden(1.0, 1.0, 1.5);
    ale.snapshot_hse();
    if (cfg.test_case == "lane_emden_perturbed")
        ale.apply_perturbation(cfg.perturb_amplitude);

    std::timespec wall_start;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);

    char csv_path[512];
    std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", ctx.run_dir.c_str());
    std::FILE* csv = std::fopen(csv_path, "w");
    std::fprintf(csv, "step,t,dt,mass,KE,IE,PE,total_E,max_mach,max_v\n");

    int frame = 0;
    while (t < cfg.t_end && !g_interrupted) {
        double dt = ale.step(t, cfg.t_end);
        t += dt;
        step++;

        if (step % 200 == 0) print_progress(t, cfg.t_end, step, dt, wall_start);

        if (step % cfg.output_interval == 0) {
            auto d = ale.compute_diagnostics();
            std::fprintf(stderr, "\n");
            std::printf("Step %6d  t=%.6e dt=%.3e M=%.10e E=%.10e |v|_max=%.3e Mach_max=%.3e\n",
                        step, t, dt, d.total_mass, d.total_E, d.max_v, d.max_mach);
            std::fprintf(csv, "%d,%.10e,%.6e,%.10e,%.10e,%.10e,%.10e,%.10e,%.6e,%.6e\n",
                         step, t, dt, d.total_mass, d.total_KE, d.total_internal_E,
                         d.total_grav_E, d.total_E, d.max_mach, d.max_v);
            std::fflush(csv);

            std::vector<double> rp, rhop, Pp, ep, vrp;
            ale.download_radial_profile(rp, rhop, Pp, ep, vrp);
            char path[512];
            std::snprintf(path, sizeof(path), "%s/profile_%04d.txt", ctx.run_dir.c_str(), ++frame);
            std::FILE* fp = std::fopen(path, "w");
            std::fprintf(fp, "# t = %.10e  step = %d\n# ic r rho P e_int v_r\n", t, step);
            for (int ic = 0; ic < ale.nr; ++ic)
                std::fprintf(fp, "%d %.10e %.10e %.10e %.10e %.10e\n",
                             ic, rp[ic], rhop[ic], Pp[ic], ep[ic], vrp[ic]);
            std::fclose(fp);
        }
    }
    std::fclose(csv);
    std::fprintf(stderr, "\n");
    ale.destroy();
    return 0;
}
#endif
