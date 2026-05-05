#include "drivers/drivers.h"

#ifdef USE_GPU
#include "pseudo_spectral_solver.cuh"
#include "sim/helpers.h"
#include "sim/run_loop.h"

#include <algorithm>
#include <cstdio>
#include <ctime>

int run_pseudo_spectral(SimConfig& cfg, SimContext& ctx, double& t, int& step) {
    // ===== 偽譜法 2D 不可壓 NS (渦度-流函數, cuFFT) =====
    if (cfg.test_case != "kh_shear" && cfg.test_case != "forced_turb"
        && cfg.test_case != "taylor_green"
        && cfg.test_case != "double_shear_layer"
        && cfg.test_case != "vortex_merger"
        && cfg.test_case != "quad_vortex_merger") {
        std::fprintf(stderr,
            "ERROR: pseudo_spectral supports --test {kh_shear, forced_turb, taylor_green, "
            "double_shear_layer, vortex_merger, quad_vortex_merger}\n");
        return 1;
    }
    PseudoSpectralSolver ps;
    ps.use_ifrk         = !cfg.ps_explicit;
    ps.use_skew         = !cfg.ps_adv_only && !cfg.ps_conservative;
    ps.use_conservative = cfg.ps_conservative && !cfg.ps_adv_only;
    ps.use_batched_fft  = cfg.ps_batched_fft;
    ps.drag_alpha       = cfg.ps_drag_alpha;
    ps.hyper_p          = std::max(1, cfg.ps_hyper_p);
    ps.use_pi_dt        = cfg.ps_pi_dt;
    ps.forcing_use_curand = !cfg.ps_forcing_host_rng;
    ps.init(cfg.nr, cfg.ntheta, cfg.ps_Lx, cfg.ps_Ly, cfg.ps_nu, cfg.cfl);
    if (cfg.test_case == "kh_shear") {
        double amp = (cfg.perturb_amplitude > 0) ? cfg.perturb_amplitude : 1e-2;
        ps.init_kh_shear(cfg.ps_vshear, amp, cfg.ps_k);
    } else if (cfg.test_case == "taylor_green") {
        ps.init_taylor_green(cfg.ps_tg_k);
    } else if (cfg.test_case == "double_shear_layer") {
        double amp = (cfg.perturb_amplitude > 0) ? cfg.perturb_amplitude : 5e-2;
        ps.init_double_shear_layer(cfg.ps_vshear, amp, cfg.ps_k);
    } else if (cfg.test_case == "vortex_merger") {
        ps.init_vortex_merger(cfg.ps_vm_gamma, cfg.ps_vm_sigma, cfg.ps_vm_dist);
    } else if (cfg.test_case == "quad_vortex_merger") {
        ps.init_quad_vortex_merger(cfg.ps_vm_gamma, cfg.ps_vm_sigma, cfg.ps_vm_dist);
    } else {
        // forced_turb: zero IC + stochastic forcing
        ps.init_zero();
        if (cfg.ps_forcing_eps <= 0.0) {
            std::fprintf(stderr,
                "ERROR: --test forced_turb requires --ps-forcing-eps > 0\n");
            return 1;
        }
    }
    if (cfg.ps_forcing_eps > 0.0) {
        ps.init_forcing(cfg.ps_forcing_kf, cfg.ps_forcing_dk,
                        cfg.ps_forcing_eps, cfg.ps_forcing_seed);
    }
    // Restart (若提供):覆寫 IC 的 ω̂,更新 step/t。
    if (!cfg.ps_resume.empty()) {
        double t_ckpt = 0.0;
        if (ps.load_checkpoint(cfg.ps_resume, t_ckpt)) {
            t = t_ckpt;
            step = ps.step_count;
            // Taylor-Green IC-比對 buffer 在 restart 後對非-TG run 沒意義;
            // 對 TG run,IC 譜仍來自 init_taylor_green 的初次呼叫 → 已保存。
        } else {
            std::fprintf(stderr, "ERROR: --ps-resume failed\n");
            return 1;
        }
    }

    std::timespec wall_start;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);

    char csv_path[512];
    std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", ctx.run_dir.c_str());
    std::FILE* csv = std::fopen(csv_path, "w");
    std::fprintf(csv,
        "step,t,dt,KE,enstrophy,max_v,max_omega,eps_KE,eps_enstrophy,err_L2\n");

    int diag_every = cfg.diag_interval > 0 ? cfg.diag_interval : cfg.output_interval;
    int vtk_every  = cfg.vtk_interval  > 0 ? cfg.vtk_interval  : cfg.output_interval;
    bool vtk_by_time = cfg.vtk_dt > 0.0;
    double next_vtk_t = 0.0;
    if (cfg.frame_buffer) ps.alloc_frame_buffer(cfg.frame_headroom_mb);

    int frame = 0;
    // t=0 幀 (便於 renderer 取首幀參考)
    if (cfg.frame_buffer && ps.frame_capacity > 0) ps.capture_frame(0.0, 0);
    else {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/output_%04d.vtk", ctx.run_dir.c_str(), ++frame);
        ps.write_vtk_2d(path);
    }

    while (t < cfg.t_end && !g_interrupted) {
        double dt = ps.step();
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
            auto d = ps.compute_diagnostics(t);
            std::fprintf(stderr, "\n");
            if (cfg.test_case == "taylor_green") {
                std::printf("Step %6d  t=%.6e dt=%.3e KE=%.6e Ω=%.6e |v|=%.3e |ω|=%.3e εKE=%.3e errL2=%.3e\n",
                            step, t, dt, d.total_KE, d.total_enstrophy,
                            d.max_v, d.max_omega, d.eps_KE, d.err_L2);
            } else {
                std::printf("Step %6d  t=%.6e dt=%.3e KE=%.6e Ω=%.6e |v|=%.3e |ω|=%.3e εKE=%.3e εΩ=%.3e\n",
                            step, t, dt, d.total_KE, d.total_enstrophy,
                            d.max_v, d.max_omega, d.eps_KE, d.eps_enstrophy);
            }
            std::fprintf(csv, "%d,%.10e,%.6e,%.10e,%.10e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                         step, t, dt, d.total_KE, d.total_enstrophy,
                         d.max_v, d.max_omega, d.eps_KE, d.eps_enstrophy,
                         d.err_L2);
            std::fflush(csv);
        }
        if (cfg.ps_ckpt_every > 0 && step % cfg.ps_ckpt_every == 0) {
            char cpath[512];
            std::snprintf(cpath, sizeof(cpath), "%s/checkpoint.bin",
                          ctx.run_dir.c_str());
            ps.save_checkpoint(cpath, t);
        }
        if (do_vtk) {
            if (cfg.frame_buffer && ps.frame_capacity > 0) {
                if (ps.frame_count >= ps.frame_capacity)
                    ps.flush_frames_to_disk(ctx.run_dir);
                ps.capture_frame(t, step);
            } else {
                ++frame;
                char path[512];
                std::snprintf(path, sizeof(path), "%s/output_%04d.vtk",
                              ctx.run_dir.c_str(), frame);
                ps.write_vtk_2d(path);
            }
        }
    }
    if (cfg.frame_buffer) ps.flush_frames_to_disk(ctx.run_dir);
    std::fclose(csv);
    std::fprintf(stderr, "\n");
    ps.destroy();
    return 0;
}
#endif
