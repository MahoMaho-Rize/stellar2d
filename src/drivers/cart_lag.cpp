#include "drivers/drivers.h"

#ifdef USE_GPU
#include "cart_lag_solver.cuh"
#include "sim/helpers.h"
#include "sim/run_loop.h"

#include <cstdio>
#include <ctime>
#include <vector>

int run_cart_lag(SimConfig& cfg, SimContext& ctx, double& t, int& step) {
    // ===== Cartesian 2D Lagrangian (Caramana compatible, planar) =====
    // Runs independent of Grid/State: uses [0,Lx]×[0,Ly] box = [1,1].
    // IC: Sod shock tube (default) or uniform.
    CartLagSolver clag;
    // For HSE: square box. For Sod: thin strip.
    bool is_hse = (cfg.test_case == "hse" || cfg.test_case == "hse_perturbed");
    double Lx = is_hse ? 1.0 : 1.0;
    double Ly = is_hse ? 1.0 : 0.2;
    double gam = is_hse ? cfg.gamma : 1.4;  // Sod expects γ=1.4 by default
    clag.init(cfg.nr, cfg.ntheta, Lx, Ly, gam, cfg.cfl);
    if (is_hse) {
        // Two-step: first build HSE unperturbed, snapshot its discrete
        // force defect, then (if hse_perturbed) layer the perturbation on top.
        clag.init_hse_polytrope(1.0, 1.0, 0.0);
        clag.snapshot_hse_force();
        if (cfg.test_case == "hse_perturbed") {
            clag.init_hse_polytrope(1.0, 1.0, cfg.perturb_amplitude);
        }
    } else {
        clag.init_sod();
    }

    std::timespec wall_start;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);

    char csv_path[512];
    std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", ctx.run_dir.c_str());
    std::FILE* csv = std::fopen(csv_path, "w");
    std::fprintf(csv, "step,t,dt,mass,KE,IE,PE,E,max_v,max_mach\n");

    int frame = 0;
    while (t < cfg.t_end && !g_interrupted) {
        double dt = clag.step(t, cfg.t_end);
        t += dt;
        step++;
        if (step % 200 == 0) print_progress(t, cfg.t_end, step, dt, wall_start);
        if (step % cfg.output_interval == 0 || t >= cfg.t_end) {
            auto d = clag.compute_diagnostics();
            std::fprintf(stderr, "\n");
            std::printf("Step %6d  t=%.6e dt=%.3e M=%.10e KE=%.4e IE=%.10e PE=%.10e E=%.10e |v|=%.3e\n",
                        step, t, dt, d.total_mass, d.total_KE, d.total_internal_E,
                        d.total_PE, d.total_E, d.max_v);
            std::fprintf(csv, "%d,%.10e,%.6e,%.10e,%.10e,%.10e,%.10e,%.10e,%.6e,%.6e\n",
                         step, t, dt, d.total_mass, d.total_KE, d.total_internal_E,
                         d.total_PE, d.total_E, d.max_v, d.max_mach);
            std::fflush(csv);
            std::vector<double> xv, rhov, Pv, vxv, ev;
            clag.download_xslice(xv, rhov, Pv, vxv, ev);
            char path[512];
            std::snprintf(path, sizeof(path), "%s/xslice_%04d.txt", ctx.run_dir.c_str(), ++frame);
            std::FILE* fp = std::fopen(path, "w");
            std::fprintf(fp, "# t=%.10e step=%d\n# x rho P vx e\n", t, step);
            for (int i = 0; i < (int)xv.size(); ++i)
                std::fprintf(fp, "%.10e %.10e %.10e %.10e %.10e\n",
                             xv[i], rhov[i], Pv[i], vxv[i], ev[i]);
            std::fclose(fp);
        }
    }
    std::fclose(csv);
    std::fprintf(stderr, "\n");
    clag.destroy();
    return 0;
}
#endif
