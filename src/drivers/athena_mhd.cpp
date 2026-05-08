#include "drivers/drivers.h"

#ifdef USE_GPU
#include "athena_mhd_solver.cuh"
#include "sim/helpers.h"
#include "sim/run_loop.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

int run_athena_mhd(SimConfig& cfg, SimContext& ctx, double& t, int& step) {
    // ===== athena_mhd: GPU 2D Eulerian ideal-MHD (vl2 + HLLD + CT) =====
    // Derivation dossier: docs/mhd_derivations/manuscript.pdf
    //   §A1-A11, §B4, §C5.
    // Tests available:
    //   brio_wu          — 1D MHD shock tube (Brio-Wu 1988).  γ=2.
    //   linwave_mhd_{f,a,s,e} — Stone+08 Table 1 linear wave (§A11).
    //   field_loop       — GS05 Fig 3, ∇·B preservation (§A5).
    //   orszag_tang      — Orszag-Tang 2D MHD vortex.
    const std::string& tc = cfg.test_case;
    const bool is_brio   = (tc == "brio_wu");
    const bool is_linwf  = (tc == "linwave_mhd_f");
    const bool is_linwa  = (tc == "linwave_mhd_a");
    const bool is_linws  = (tc == "linwave_mhd_s");
    const bool is_linwe  = (tc == "linwave_mhd_e");
    const bool is_loop   = (tc == "field_loop");
    const bool is_ot     = (tc == "orszag_tang");
    const bool is_rj2a   = (tc == "rj2a");
    const bool is_rj4d   = (tc == "rj4d");
    const bool is_cpaw1  = (tc == "cpaw_1d");
    const bool is_cpaw2  = (tc == "cpaw_2d");
    const bool is_blast  = (tc == "mhd_blast");
    const bool is_rotor  = (tc == "mhd_rotor");
    if (!is_brio && !is_linwf && !is_linwa && !is_linws && !is_linwe
        && !is_loop && !is_ot
        && !is_rj2a && !is_rj4d && !is_cpaw1 && !is_cpaw2
        && !is_blast && !is_rotor) {
        std::fprintf(stderr,
            "athena_mhd: supports --test {brio_wu, linwave_mhd_{f,a,s,e},"
            " field_loop, orszag_tang, rj2a, rj4d, cpaw_1d, cpaw_2d,"
            " mhd_blast, mhd_rotor}\n");
        return 1;
    }

    double Lx = 1.0, Ly = 1.0, gam = 5.0 / 3.0;
    if (is_brio) {
        Lx = 1.0;
        Ly = 2.0 / std::max(1, cfg.ntheta);   // thin slab (ny minimal in 1D test)
        // Keep Ly ≥ 2·dx to make 2D kernels well-defined.
        Ly = std::max(Ly, 4.0 / (double)cfg.nr);
        gam = 2.0;
    } else if (is_loop) {
        Lx = 1.0; Ly = 1.0;
        gam = 5.0 / 3.0;
    } else if (is_ot) {
        Lx = 2.0 * M_PI; Ly = 2.0 * M_PI;
        gam = 5.0 / 3.0;
    } else if (is_rj2a || is_rj4d) {
        Lx = 1.0;
        Ly = std::max(2.0 / std::max(1, cfg.ntheta), 4.0 / (double)cfg.nr);
        gam = 5.0 / 3.0;
    } else if (is_cpaw1) {
        Lx = 1.0; Ly = 1.0;
        gam = 5.0 / 3.0;
    } else if (is_cpaw2) {
        // IC overrides Lx, Ly
        Lx = std::sqrt(5.0); Ly = std::sqrt(5.0) / 2.0;
        gam = 5.0 / 3.0;
    } else if (is_blast) {
        Lx = 1.0; Ly = 1.5;
        gam = 5.0 / 3.0;
    } else if (is_rotor) {
        Lx = 1.0; Ly = 1.0;
        gam = 1.4;
    } else {
        // linwave: Lx = Ly = 1
        Lx = 1.0; Ly = 1.0;
        gam = 5.0 / 3.0;
    }

    AthenaMHDSolver sv;
    sv.init(cfg.nr, cfg.ntheta, Lx, Ly, gam, cfg.cfl);
    sv.xorder  = cfg.athena_vl2_xorder;    // reuse CLI flag
    sv.limiter = cfg.athena_vl2_limiter;
    sv.cfl_limit = 0.5;                     // Stone+08 2D stability limit

    if (is_brio) {
        sv.init_brio_wu();
    } else if (is_linwf) {
        sv.init_linear_wave(AthenaMHDSolver::FAST_M,   /*k=*/1);
    } else if (is_linwa) {
        sv.init_linear_wave(AthenaMHDSolver::ALFVEN,   /*k=*/1);
    } else if (is_linws) {
        sv.init_linear_wave(AthenaMHDSolver::SLOW,     /*k=*/1);
    } else if (is_linwe) {
        sv.init_linear_wave(AthenaMHDSolver::ENTROPY,  /*k=*/1);
    } else if (is_loop) {
        sv.init_field_loop();
    } else if (is_ot) {
        sv.init_orszag_tang();
    } else if (is_rj2a) {
        sv.init_rj2a();
    } else if (is_rj4d) {
        sv.init_rj4d();
    } else if (is_cpaw1) {
        sv.init_cpaw_1d(/*traveling=*/true);
    } else if (is_cpaw2) {
        sv.init_cpaw_2d(/*traveling=*/true);
    } else if (is_blast) {
        sv.init_blast();
    } else if (is_rotor) {
        sv.init_rotor();
    }

    const char* lim_name = (sv.limiter == 1) ? "minmod" : "vanleer";
    const char* order_name = (sv.xorder == 1) ? "DC" : "PLM";
    std::fprintf(stderr,
        "  AthenaMHD solver  nx=%d ny=%d  γ=%.4f CFL=%.3f (limit=%.3f)\n"
        "    xorder=%d (%s) limiter=%s  test=%s\n",
        sv.nx, sv.ny, sv.gamma, sv.cfl, sv.cfl_limit,
        sv.xorder, order_name, lim_name, tc.c_str());

    std::timespec wall_start;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);

    char csv_path[512];
    std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", ctx.run_dir.c_str());
    std::FILE* csv = std::fopen(csv_path, "w");
    std::fprintf(csv, "step,t,dt,mass,KE,IE,ME,E,max_divB,max_v,max_fast_mach\n");

    int diag_every = cfg.diag_interval > 0 ? cfg.diag_interval : cfg.output_interval;
    int vtk_every  = cfg.vtk_interval  > 0 ? cfg.vtk_interval  : cfg.output_interval;

    int frame = 0;
    while (t < cfg.t_end && !g_interrupted) {
        double dt = sv.step(t, cfg.t_end);
        t += dt;
        step++;
        if (step % 200 == 0) print_progress(t, cfg.t_end, step, dt, wall_start);
        const bool do_diag = (step % diag_every == 0) || t >= cfg.t_end;
        const bool do_vtk  = (step % vtk_every  == 0) || t >= cfg.t_end;
        if (do_diag) {
            auto d = sv.compute_diagnostics();
            std::fprintf(stderr, "\n");
            std::printf("Step %6d  t=%.6e dt=%.3e M=%.10e KE=%.4e IE=%.10e "
                        "ME=%.4e E=%.10e max|divB|=%.3e\n",
                        step, t, dt, d.total_mass, d.total_KE, d.total_IE,
                        d.total_ME, d.total_E, d.max_divB);
            std::fprintf(csv,
                "%d,%.10e,%.6e,%.10e,%.10e,%.10e,%.10e,%.10e,%.3e,%.6e,%.6e\n",
                step, t, dt, d.total_mass, d.total_KE, d.total_IE,
                d.total_ME, d.total_E, d.max_divB, d.max_v, d.max_fast_mach);
            std::fflush(csv);
        }
        if (do_vtk) {
            ++frame;
            char path[512];
            std::snprintf(path, sizeof(path), "%s/output_%04d.vtk",
                          ctx.run_dir.c_str(), frame);
            sv.write_vtk_2d(path, Lx, Ly);
        }
    }
    std::fclose(csv);
    std::fprintf(stderr, "\n");

    sv.destroy();
    return 0;
}
#endif
