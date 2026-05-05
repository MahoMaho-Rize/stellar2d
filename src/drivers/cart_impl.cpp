#include "drivers/drivers.h"

#ifdef USE_GPU
#include "cart_impl_solver.cuh"
#include "sim/helpers.h"
#include "sim/run_loop.h"

#include <cstdio>
#include <ctime>

int run_cart_impl(SimConfig& cfg, SimContext& ctx, double& t, int& step) {
    // ===== cart_impl: 2D Cartesian BE + JFNK low-Mach =====
    // 借鑒 cart_ale2 的 Cartesian 均勻網格 + fas2 的 JFNK + Viallet scaling.
    // 放棄球極座標,目標是解決 256² 擾動能真正演化(不被 BE 凍結).
    CartImplSolver csol;
    bool is_hse = (cfg.test_case == "hse" || cfg.test_case == "hse_perturbed");
    if (!is_hse) {
        std::fprintf(stderr,
            "cart_impl only supports --test hse / hse_perturbed for now\n");
        return 1;
    }
    double Lx = 1.0, Ly = 1.0;
    double g_val = 1.0;
    double rho_base = 1.0;
    csol.hllc_variant = cfg.hllc_variant;
    csol.limiter_type = static_cast<int>(cfg.limiter);
    csol.init(cfg.nr, cfg.ntheta, Lx, Ly, ctx.eos, cfg.gamma, g_val, cfg.cfl);

    // Two-step init: HSE first, snapshot reference, then layer perturbation
    csol.init_hse_polytrope(rho_base, 0.0);
    csol.snapshot_hse();
    if (cfg.test_case == "hse_perturbed" && cfg.perturb_amplitude != 0.0) {
        csol.init_hse_polytrope(rho_base, cfg.perturb_amplitude);
    }

    std::timespec wall_start;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);

    char csv_path[512];
    std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", ctx.run_dir.c_str());
    std::FILE* csv = std::fopen(csv_path, "w");
    std::fprintf(csv, "step,t,dt,mass,energy,max_v,max_mach\n");

    // Initial frame
    {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/output_0000.vtk", ctx.run_dir.c_str());
        csol.write_vtk(path);
    }

    int frame = 0;
    int diag_every = cfg.diag_interval > 0 ? cfg.diag_interval : cfg.output_interval;
    int vtk_every  = cfg.vtk_interval  > 0 ? cfg.vtk_interval  : cfg.output_interval;
    while (t < cfg.t_end && !g_interrupted) {
        double dt = csol.step(t, cfg.t_end);
        t += dt;
        step++;
        if (step % 200 == 0) print_progress(t, cfg.t_end, step, dt, wall_start);
        if (step % diag_every == 0 || t >= cfg.t_end) {
            auto d = csol.compute_diagnostics();
            std::fprintf(stderr, "\n");
            std::printf("Step %6d  t=%.6e dt=%.3e M=%.10e E=%.10e |v|=%.3e Ma=%.3e\n",
                        step, t, dt, d.mass, d.energy, d.max_v, d.max_mach);
            std::fprintf(csv, "%d,%.10e,%.6e,%.10e,%.10e,%.6e,%.6e\n",
                         step, t, dt, d.mass, d.energy, d.max_v, d.max_mach);
            std::fflush(csv);
        }
        if (step % vtk_every == 0 || t >= cfg.t_end) {
            ++frame;
            char path[512];
            std::snprintf(path, sizeof(path), "%s/output_%04d.vtk", ctx.run_dir.c_str(), frame);
            csol.write_vtk(path);
        }
    }
    std::fclose(csv);
    std::fprintf(stderr, "\n");
    csol.destroy();
    return 0;
}
#endif
