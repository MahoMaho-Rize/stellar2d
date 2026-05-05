#include "drivers/drivers.h"

#ifdef USE_GPU
#include "sph2d_spectral_solver.cuh"
#include "sim/helpers.h"
#include "sim/run_loop.h"

#include <algorithm>
#include <cstdio>
#include <ctime>

int run_sph2d_spectral(SimConfig& cfg, SimContext& ctx, double& t, int& step) {
    // ===== 2D 薄球殼 barotropic 偽譜 =====
    if (cfg.test_case != "rossby_wave" && cfg.test_case != "jovian_bands") {
        std::fprintf(stderr,
            "ERROR: sph2d_spectral supports --test {rossby_wave, jovian_bands}\n");
        return 1;
    }
    Sph2DSpectralSolver sph;
    sph.drag_alpha = cfg.sph_drag;
    sph.hyper_p    = std::max(1, cfg.sph_hyper);
    sph.use_pi_dt  = cfg.sph_pi_dt;
    int Lmax_req = (cfg.sph_Lmax > 0) ? cfg.sph_Lmax
                   : std::min(cfg.nr - 1, cfg.ntheta / 2 - 2);
    sph.init(cfg.nr, cfg.ntheta, Lmax_req,
             cfg.sph_R, cfg.sph_Omega, cfg.sph_nu, cfg.cfl);
    if (cfg.test_case == "rossby_wave") {
        sph.init_rossby_wave(cfg.sph_rossby_l, cfg.sph_rossby_m,
                             cfg.sph_rossby_amp);
    } else {
        sph.init_zero();
        if (cfg.sph_forcing_eps <= 0.0) {
            std::fprintf(stderr,
                "ERROR: --test jovian_bands requires --sph-forcing-eps > 0\n");
            return 1;
        }
    }
    if (cfg.sph_forcing_eps > 0.0) {
        sph.init_forcing(cfg.sph_forcing_lmin, cfg.sph_forcing_lmax,
                         cfg.sph_forcing_eps, cfg.sph_forcing_seed);
    }
    if (!cfg.sph_resume.empty()) {
        double t_ckpt = 0.0;
        if (sph.load_checkpoint(cfg.sph_resume, t_ckpt)) {
            t = t_ckpt;
            step = sph.step_count;
        } else {
            std::fprintf(stderr, "ERROR: --sph-resume failed\n");
            return 1;
        }
    }

    std::timespec wall_start;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);

    char csv_path[512];
    std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", ctx.run_dir.c_str());
    std::FILE* csv = std::fopen(csv_path, "w");
    std::fprintf(csv, "step,t,dt,KE,enstrophy,max_v,max_zeta,err_L2\n");

    int diag_every = cfg.diag_interval > 0 ? cfg.diag_interval : cfg.output_interval;
    int vtk_every  = cfg.vtk_interval  > 0 ? cfg.vtk_interval  : cfg.output_interval;
    bool vtk_by_time = cfg.vtk_dt > 0.0;
    double next_vtk_t = 0.0;
    if (cfg.frame_buffer) sph.alloc_frame_buffer(cfg.frame_headroom_mb);

    int frame = 0;
    if (cfg.frame_buffer && sph.frame_capacity > 0) sph.capture_frame(0.0, 0);
    else {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/output_%04d.vtk", ctx.run_dir.c_str(), ++frame);
        sph.write_vtk_2d(path);
    }

    while (t < cfg.t_end && !g_interrupted) {
        double dt = sph.step();
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
            auto d = sph.compute_diagnostics(t);
            std::fprintf(stderr, "\n");
            std::printf("Step %6d  t=%.6e dt=%.3e KE=%.6e Ω=%.6e |v|=%.3e |ζ|=%.3e err=%.3e\n",
                        step, t, dt, d.total_KE, d.total_enstrophy,
                        d.max_v, d.max_zeta, d.err_L2);
            std::fprintf(csv, "%d,%.10e,%.6e,%.10e,%.10e,%.6e,%.6e,%.6e\n",
                         step, t, dt, d.total_KE, d.total_enstrophy,
                         d.max_v, d.max_zeta, d.err_L2);
            std::fflush(csv);
        }
        if (cfg.sph_ckpt_every > 0 && step % cfg.sph_ckpt_every == 0) {
            char cpath[512];
            std::snprintf(cpath, sizeof(cpath), "%s/checkpoint.bin",
                          ctx.run_dir.c_str());
            sph.save_checkpoint(cpath, t);
        }
        if (do_vtk) {
            if (cfg.frame_buffer && sph.frame_capacity > 0) {
                if (sph.frame_count >= sph.frame_capacity)
                    sph.flush_frames_to_disk(ctx.run_dir);
                sph.capture_frame(t, step);
            } else {
                ++frame;
                char path[512];
                std::snprintf(path, sizeof(path), "%s/output_%04d.vtk",
                              ctx.run_dir.c_str(), frame);
                sph.write_vtk_2d(path);
            }
        }
    }
    if (cfg.frame_buffer) sph.flush_frames_to_disk(ctx.run_dir);
    std::fclose(csv);
    std::fprintf(stderr, "\n");
    sph.destroy();
    return 0;
}
#endif
