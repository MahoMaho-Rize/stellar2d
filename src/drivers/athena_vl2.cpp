#include "drivers/drivers.h"

#ifdef USE_GPU
#include "athena_vl2_solver.cuh"
#include "sim/helpers.h"
#include "sim/run_loop.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

int run_athena_vl2(SimConfig& cfg, SimContext& ctx, double& t, int& step) {
    // ===== athena_vl2: GPU port of Athena++ vl2 Eulerian Godunov =====
    // Only supports the andrassy2022 test case (Route A benchmark).  New
    // tests can be added by expanding the if-chain here and by calling
    // init_andrassy2022 or any future IC method on the solver.
    bool is_andrassy = (cfg.test_case == "andrassy2022");
    bool is_shear    = (cfg.test_case == "shear_mode");
    bool is_ewave    = (cfg.test_case == "entropy_wave");
    bool is_sod      = (cfg.test_case == "sod");
    if (!is_andrassy && !is_shear && !is_ewave && !is_sod) {
        std::fprintf(stderr,
            "athena_vl2: supports --test {andrassy2022, shear_mode, entropy_wave, sod}\n");
        return 1;
    }
    double Lx = 1.0, Ly = 1.0;
    double gam = 5.0 / 3.0;
    if (is_andrassy) {
        if (cfg.cart_ale2_slab_file.empty()) {
            std::fprintf(stderr,
                "athena_vl2 andrassy2022 requires --ic-slab <file>\n");
            return 1;
        }
        // Peek slab header for Lx / Ly / gamma.
        std::FILE* fp = std::fopen(cfg.cart_ale2_slab_file.c_str(), "r");
        if (!fp) {
            std::fprintf(stderr,
                "cannot open slab file %s\n", cfg.cart_ale2_slab_file.c_str());
            return 1;
        }
        char line[512];
        while (std::fgets(line, sizeof(line), fp)) {
            const char* s = line;
            while (*s == ' ' || *s == '\t') ++s;
            if (*s == '#' || *s == '\0' || *s == '\n') continue;
            double Ly_f, Lx_f, gy_f, gamma_f, rho_t, P_t, T_t, mu_f;
            if (std::sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf",
                            &Ly_f, &Lx_f, &gy_f, &gamma_f,
                            &rho_t, &P_t, &T_t, &mu_f) == 8) {
                Ly = Ly_f; Lx = Lx_f; gam = gamma_f;
                break;
            }
        }
        std::fclose(fp);
    } else if (is_sod) {
        // Sod shock tube: Lx = Ly = 1, γ = 1.4 (standard).
        Lx = 1.0; Ly = 1.0; gam = 1.4;
    } else {
        // shear_mode / entropy_wave: Lx = Ly = 1, γ = 5/3.
        Lx = 1.0; Ly = 1.0; gam = 5.0 / 3.0;
    }

    AthenaVL2Solver av;
    av.init(cfg.nr, cfg.ntheta, Lx, Ly, gam, cfg.cfl);
    av.xorder = cfg.athena_vl2_xorder;
    av.limiter = cfg.athena_vl2_limiter;
    // Force Athena vl2 2D stability limit (Athena hardcodes this to 0.5).
    av.cfl_limit = 0.5;
    if (is_andrassy) {
        av.init_andrassy2022(cfg.cart_ale2_slab_file,
                             cfg.cart_ale2_andrassy_amp,
                             cfg.cart_ale2_andrassy_seed,
                             cfg.cart_ale2_andrassy_noise);
    } else if (is_shear) {
        av.init_shear_mode(cfg.shear_rho, cfg.shear_P,
                           cfg.shear_V0, cfg.shear_k);
    } else if (is_sod) {
        av.init_sod();
    } else {
        av.init_entropy_wave(cfg.ewave_rho0, cfg.ewave_P0,
                             cfg.ewave_u0, cfg.ewave_A, cfg.ewave_k);
        if (cfg.t_end == 1.0) {
            cfg.t_end = cfg.ewave_periods * Lx / cfg.ewave_u0;
            std::fprintf(stderr,
                "  entropy_wave: auto t_end = %g (%.3g periods)\n",
                cfg.t_end, cfg.ewave_periods);
        }
    }

    const char* lim_name = (av.limiter == 1) ? "minmod" : "vanleer";
    const char* order_name = (av.xorder == 1) ? "DC" : "PLM";
    std::fprintf(stderr,
        "  AthenaVL2 solver  nx=%d ny=%d  γ=%.4f CFL=%.3f (limit=%.3f)\n"
        "    xorder=%d (%s) limiter=%s  seed=%d noise=%g\n",
        av.nx, av.ny, av.gamma, av.cfl, av.cfl_limit,
        av.xorder, order_name, lim_name,
        cfg.cart_ale2_andrassy_seed, cfg.cart_ale2_andrassy_noise);

    std::timespec wall_start;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);

    char csv_path[512];
    std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", ctx.run_dir.c_str());
    std::FILE* csv = std::fopen(csv_path, "w");
    if (av.tracer_enabled)
        std::fprintf(csv, "step,t,dt,mass,KE,IE,PE,E,max_v,max_mach,species_mass\n");
    else
        std::fprintf(csv, "step,t,dt,mass,KE,IE,PE,E,max_v,max_mach\n");

    int diag_every = cfg.diag_interval > 0 ? cfg.diag_interval : cfg.output_interval;
    int vtk_every  = cfg.vtk_interval  > 0 ? cfg.vtk_interval  : cfg.output_interval;
    bool vtk_by_time = cfg.vtk_dt > 0.0;
    double next_vtk_t = 0.0;
    if (vtk_by_time) {
        std::fprintf(stderr,
            "  AthenaVL2 diag every %d steps, VTK every dt=%g (physical)\n",
            diag_every, cfg.vtk_dt);
    } else {
        std::fprintf(stderr,
            "  AthenaVL2 diag every %d steps, VTK every %d steps\n",
            diag_every, vtk_every);
    }

    int frame = 0;
    while (t < cfg.t_end && !g_interrupted) {
        double dt = av.step(t, cfg.t_end);
        t += dt;
        step++;
        if (step % 200 == 0) print_progress(t, cfg.t_end, step, dt, wall_start);
        bool do_diag = (step % diag_every == 0) || t >= cfg.t_end;
        bool do_vtk;
        if (vtk_by_time) {
            do_vtk = (t >= next_vtk_t) || t >= cfg.t_end;
            if (do_vtk) next_vtk_t = t + cfg.vtk_dt;
        } else {
            do_vtk = (step % vtk_every == 0) || t >= cfg.t_end;
        }
        if (do_diag) {
            auto d = av.compute_diagnostics();
            std::fprintf(stderr, "\n");
            std::printf("Step %6d  t=%.6e dt=%.3e M=%.10e KE=%.4e IE=%.10e "
                        "PE=%.10e E=%.10e |v|=%.3e\n",
                        step, t, dt, d.total_mass, d.total_KE, d.total_internal_E,
                        d.total_PE, d.total_E, d.max_v);
            if (av.tracer_enabled) {
                double m_species = av.total_species_mass();
                std::fprintf(csv,
                    "%d,%.10e,%.6e,%.10e,%.10e,%.10e,%.10e,%.10e,%.6e,%.6e,%.10e\n",
                    step, t, dt, d.total_mass, d.total_KE, d.total_internal_E,
                    d.total_PE, d.total_E, d.max_v, d.max_mach, m_species);
            } else {
                std::fprintf(csv,
                    "%d,%.10e,%.6e,%.10e,%.10e,%.10e,%.10e,%.10e,%.6e,%.6e\n",
                    step, t, dt, d.total_mass, d.total_KE, d.total_internal_E,
                    d.total_PE, d.total_E, d.max_v, d.max_mach);
            }
            std::fflush(csv);
        }
        if (do_vtk) {
            ++frame;
            char path[512];
            std::snprintf(path, sizeof(path), "%s/output_%04d.vtk",
                          ctx.run_dir.c_str(), frame);
            av.write_vtk_2d(path, Lx, Ly);
        }
    }
    std::fclose(csv);
    std::fprintf(stderr, "\n");
    if (cfg.compute_error && is_ewave) {
        av.compute_entropy_wave_error(
            t, step,
            cfg.ewave_rho0, cfg.ewave_P0, cfg.ewave_u0,
            cfg.ewave_A, cfg.ewave_k, cfg.ewave_periods,
            ctx.run_dir);
    }
    if (cfg.compute_error && is_sod) {
        av.compute_sod_error(t, step, ctx.run_dir);
    }
    av.destroy();
    return 0;
}
#endif
