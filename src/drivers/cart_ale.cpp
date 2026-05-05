#include "drivers/drivers.h"

#ifdef USE_GPU
#include "cart_ale_solver.cuh"
#include "sim/helpers.h"
#include "sim/run_loop.h"

#include <cstdio>
#include <ctime>
#include <vector>

int run_cart_ale(SimConfig& cfg, SimContext& ctx, double& t, int& step) {
    // ===== Cartesian 2D ALE (Caramana Lagrangian + Eulerian rezone + swept remap) =====
    CartAleSolver cale;
    bool is_hse = (cfg.test_case == "hse" || cfg.test_case == "hse_perturbed"
                   || cfg.test_case == "hse_bubble");
    bool is_kh = (cfg.test_case == "kh_shear");
    double Lx = 1.0;
    double Ly = (is_hse || is_kh) ? 1.0 : 0.2;
    double gam = is_hse ? cfg.gamma : 1.4;
    cale.init(cfg.nr, cfg.ntheta, Lx, Ly, gam, cfg.cfl);
    cale.remap_order = cfg.cart_ale_remap_order;
    cale.CQ_lin  = cfg.cart_ale_cq_lin;
    cale.CQ_quad = cfg.cart_ale_cq_quad;
    cale.shear_aware_av = cfg.cart_ale_shear_aware ? 1 : 0;
    if      (cfg.cart_ale_limiter == "minmod")  cale.remap_limiter = 0;
    else if (cfg.cart_ale_limiter == "vanleer") cale.remap_limiter = 1;
    else if (cfg.cart_ale_limiter == "mc")      cale.remap_limiter = 2;
    else { std::fprintf(stderr, "unknown --remap-limiter %s; using vanleer\n",
                        cfg.cart_ale_limiter.c_str()); cale.remap_limiter = 1; }
    const char* lim_name = cale.remap_limiter == 0 ? "minmod"
                         : cale.remap_limiter == 1 ? "vanleer" : "mc";
    std::fprintf(stderr,
        "  CartAle remap_order = %d (%s)  limiter = %s  CQ_lin=%g CQ_quad=%g  shear_aware_av=%d\n",
        cale.remap_order,
        cale.remap_order >= 2 ? "MUSCL-in-remap" : "donor-cell",
        lim_name, cale.CQ_lin, cale.CQ_quad, cale.shear_aware_av);
    if (cfg.test_case == "hse_bubble") {
        std::vector<CartAleSolver::Bubble> blist;
        if (!cfg.bubbles.empty()) {
            for (const auto& b : cfg.bubbles)
                blist.push_back({b[0], b[1], b[2], b[3], b[4]});
        } else {
            blist.push_back({cfg.bubble_xc, cfg.bubble_yc, cfg.bubble_rb,
                             cfg.bubble_alpha, cfg.bubble_beta});
        }
        cale.init_hse_bubbles(1.0, 1.0, blist);
    } else if (cfg.test_case == "hse_perturbed") {
        cale.init_hse_polytrope(1.0, 1.0, cfg.perturb_amplitude);
    } else if (cfg.test_case == "hse") {
        cale.init_hse_polytrope(1.0, 1.0, 0.0);
    } else if (cfg.test_case == "kh_shear") {
        // Canonical KH (Athena++ parity): ρ_heavy/ρ_light=2, P0=2.5,
        // |vx|=0.5, amp=0.1 (fully nonlinear seed), k=2 (2 vortices
        // per interface — 256 cells resolve each vortex). --perturb
        // overrides amplitude.
        double amp = (cfg.perturb_amplitude > 0) ? cfg.perturb_amplitude : 0.1;
        cale.init_kh_shear(1.0, 2.0, 2.5, 0.5, amp, 2);
    } else {
        cale.init_sod();
    }

    std::timespec wall_start;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);

    char csv_path[512];
    std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", ctx.run_dir.c_str());
    std::FILE* csv = std::fopen(csv_path, "w");
    std::fprintf(csv, "step,t,dt,mass,KE,IE,PE,E,max_v,max_mach\n");

    int diag_every = cfg.diag_interval > 0 ? cfg.diag_interval : cfg.output_interval;
    int vtk_every  = cfg.vtk_interval  > 0 ? cfg.vtk_interval  : cfg.output_interval;
    bool vtk_by_time = cfg.vtk_dt > 0.0;
    double next_vtk_t = 0.0;   // first capture at t=0+dt (after first step)
    if (vtk_by_time) {
        std::fprintf(stderr,
            "  CartAle diag every %d steps, VTK every dt=%g (physical time)%s\n",
            diag_every, cfg.vtk_dt,
            cfg.frame_buffer ? " (VRAM buffered)" : "");
    } else {
        std::fprintf(stderr, "  CartAle diag every %d steps, VTK every %d steps%s\n",
                     diag_every, vtk_every,
                     cfg.frame_buffer ? " (VRAM buffered)" : "");
    }
    if (cfg.frame_buffer) {
        cale.alloc_frame_buffer(cfg.frame_headroom_mb);
    }

    int frame = 0;
    while (t < cfg.t_end && !g_interrupted) {
        double dt = cale.step(t, cfg.t_end);
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
            auto d = cale.compute_diagnostics();
            std::fprintf(stderr, "\n");
            std::printf("Step %6d  t=%.6e dt=%.3e M=%.10e KE=%.4e IE=%.10e PE=%.10e E=%.10e |v|=%.3e\n",
                        step, t, dt, d.total_mass, d.total_KE, d.total_internal_E,
                        d.total_PE, d.total_E, d.max_v);
            std::fprintf(csv, "%d,%.10e,%.6e,%.10e,%.10e,%.10e,%.10e,%.10e,%.6e,%.6e\n",
                         step, t, dt, d.total_mass, d.total_KE, d.total_internal_E,
                         d.total_PE, d.total_E, d.max_v, d.max_mach);
            std::fflush(csv);
        }
        if (do_vtk) {
            if (cfg.frame_buffer && cale.frame_capacity > 0) {
                if (cale.frame_count >= cale.frame_capacity) {
                    cale.flush_frames_to_disk(ctx.run_dir, Lx, Ly);
                }
                cale.capture_frame(t, step);
            } else {
                ++frame;
                char path[512];
                std::snprintf(path, sizeof(path), "%s/output_%04d.vtk", ctx.run_dir.c_str(), frame);
                cale.write_vtk_2d(path, Lx, Ly);
            }
        }
    }
    if (cfg.frame_buffer) {
        cale.flush_frames_to_disk(ctx.run_dir, Lx, Ly);
    }
    std::fclose(csv);
    std::fprintf(stderr, "\n");
    cale.destroy();
    return 0;
}
#endif
