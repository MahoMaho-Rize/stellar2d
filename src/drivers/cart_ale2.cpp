#include "drivers/drivers.h"

#ifdef USE_GPU
#include "cart_ale2_solver.cuh"
#include "cart_ale2_trace.h"
#include "sim/helpers.h"
#include "sim/run_loop.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

int run_cart_ale2(SimConfig& cfg, SimContext& ctx, double& t, int& step) {
    // ===== cart_ale2: full periodic BC + PPM-in-remap (in development) =====
    CartAle2Solver cale;
    bool is_hse = (cfg.test_case == "hse" || cfg.test_case == "hse_perturbed"
                   || cfg.test_case == "hse_bubble");
    bool is_kh = (cfg.test_case == "kh_shear");
    bool is_kh_lec = (cfg.test_case == "kh_lecoanet");
    bool is_loc_conv = (cfg.test_case == "local_convection");
    bool is_andrassy = (cfg.test_case == "andrassy2022");
    bool is_sedov  = (cfg.test_case == "sedov2d");
    bool is_noh    = (cfg.test_case == "noh");
    bool is_gresho = (cfg.test_case == "gresho");
    bool is_yee    = (cfg.test_case == "yee_vortex");
    bool is_shear  = (cfg.test_case == "shear_mode");
    bool is_ewave  = (cfg.test_case == "entropy_wave");
    bool is_sod    = (cfg.test_case == "sod");
    double Lx = 1.0;
    // Lecoanet: domain aspect 1:2 so shear layers at y=0.5, y=1.5 match
    // Athena++ iprob=4 geometry (z1=-0.5, z2=0.5 in centred coords).
    // Standard-test domains: Sedov/Gresho = 1×1, Noh = 1×1 (centred rescale
    // of the canonical [-1,1]²), Yee vortex = 10×10 (standard [-5,5]²).
    double Ly = is_kh_lec ? 2.0
              : (is_hse || is_kh || is_sedov || is_noh || is_gresho) ? 1.0
              : is_yee ? 10.0
              : is_shear ? 1.0
              : is_ewave ? 1.0
              : 0.2;
    if (is_yee) Lx = 10.0;
    double gam = is_hse ? cfg.gamma : 1.4;
    // andrassy2022 uses γ=5/3 (paper §2.2).
    if (is_andrassy) gam = 5.0 / 3.0;
    // Noh problem canonical γ=5/3.
    if (is_noh) gam = 5.0 / 3.0;
    // local_convection / andrassy2022: read slab header to get real Ly, Lx, γ.
    if (is_loc_conv || is_andrassy) {
        if (cfg.cart_ale2_slab_file.empty()) {
            std::fprintf(stderr,
                "local_convection requires --ic-slab <file>\n");
            return 1;
        }
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
    }
    cale.init(cfg.nr, cfg.ntheta, Lx, Ly, gam, cfg.cfl);
    cale.remap_order = cfg.cart_ale_remap_order;
    cale.CQ_lin  = cfg.cart_ale_cq_lin;
    cale.CQ_quad = cfg.cart_ale_cq_quad;
    cale.shear_aware_av = cfg.cart_ale_shear_aware ? 1 : 0;
    cale.rebuild_order  = cfg.cart_ale2_rebuild_order;
    int bcm = 0;
    if (cfg.cart_ale2_bc_x == "periodic") bcm |= 1;
    if (cfg.cart_ale2_bc_y == "periodic") bcm |= 2;
    cale.bc_mode = bcm;
    cale.ppm_enabled = cfg.cart_ale2_ppm ? 1 : 0;
    if      (cfg.cart_ale2_ppm_limiter == "cs") cale.ppm_cs_limiter = 1;
    else if (cfg.cart_ale2_ppm_limiter == "cw") cale.ppm_cs_limiter = 0;
    else { std::fprintf(stderr, "unknown --ppm-limiter %s; using cs\n",
                        cfg.cart_ale2_ppm_limiter.c_str()); cale.ppm_cs_limiter = 1; }
    if      (cfg.cart_ale2_ppm_space == "prim") cale.ppm_primitive = 1;
    else if (cfg.cart_ale2_ppm_space == "cons") cale.ppm_primitive = 0;
    else { std::fprintf(stderr, "unknown --ppm-space %s; using prim\n",
                        cfg.cart_ale2_ppm_space.c_str()); cale.ppm_primitive = 1; }
    cale.ppm_char = cfg.cart_ale2_ppm_char ? 1 : 0;
    if      (cfg.cart_ale_limiter == "minmod")  cale.remap_limiter = 0;
    else if (cfg.cart_ale_limiter == "vanleer") cale.remap_limiter = 1;
    else if (cfg.cart_ale_limiter == "mc")      cale.remap_limiter = 2;
    else { std::fprintf(stderr, "unknown --remap-limiter %s; using vanleer\n",
                        cfg.cart_ale_limiter.c_str()); cale.remap_limiter = 1; }
    const char* lim_name = cale.remap_limiter == 0 ? "minmod"
                         : cale.remap_limiter == 1 ? "vanleer" : "mc";
    const char* recon_name;
    if (cale.remap_order < 2)      recon_name = "donor-cell";
    else if (!cale.ppm_enabled)    recon_name = "MUSCL-in-remap";
    else if (!cale.ppm_primitive)  recon_name = cale.ppm_cs_limiter ? "PPM/CS/cons" : "PPM/CW/cons";
    else if (cale.ppm_char)        recon_name = cale.ppm_cs_limiter ? "PPM/CS/char" : "PPM/CW/char";
    else                           recon_name = cale.ppm_cs_limiter ? "PPM/CS/prim" : "PPM/CW/prim";
    std::fprintf(stderr,
        "  CartAle2 remap_order = %d (%s)  limiter = %s  CQ_lin=%g CQ_quad=%g  shear_aware_av=%d  bc=(%s,%s)\n",
        cale.remap_order, recon_name,
        lim_name, cale.CQ_lin, cale.CQ_quad, cale.shear_aware_av,
        cfg.cart_ale2_bc_x.c_str(), cfg.cart_ale2_bc_y.c_str());
    if (cfg.test_case == "hse_bubble") {
        std::vector<CartAle2Solver::Bubble> blist;
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
        double amp = (cfg.perturb_amplitude > 0) ? cfg.perturb_amplitude : 0.1;
        cale.init_kh_shear(1.0, 2.0, 2.5, 0.5, amp, 2);
    } else if (cfg.test_case == "kh_lecoanet") {
        // Athena++ pgen/kh.cpp iprob=4 canonical parameters:
        //   vflow=1, amp=0.01, drho_rho0=0 (unstratified), P0=10.
        // --perturb overrides amp. Requires periodic BC in both x and y.
        double amp = (cfg.perturb_amplitude >= 0) ? cfg.perturb_amplitude : 0.01;
        if (cale.bc_mode != 3) {
            std::fprintf(stderr,
                "  [warn] kh_lecoanet expects --bc-x periodic --bc-y periodic; "
                "current bc_mode=%d\n", cale.bc_mode);
        }
        int k = (cfg.cart_ale2_kh_k > 0) ? cfg.cart_ale2_kh_k : 1;
        cale.init_kh_lecoanet(1.0, amp, 0.0, 10.0, k);
    } else if (cfg.test_case == "local_convection") {
        // Recommend --bc-x periodic --bc-y reflect.
        if (!((cale.bc_mode & 1) && !(cale.bc_mode & 2))) {
            std::fprintf(stderr,
                "  [warn] local_convection prefers --bc-x periodic --bc-y reflect; "
                "current bc_mode=%d\n", cale.bc_mode);
        }
        cale.init_local_convection(cfg.cart_ale2_slab_file,
                                   cfg.cart_ale2_slab_perturb,
                                   cfg.cart_ale2_slab_seed_k);
        cale.tau_cool = cfg.cart_ale2_cool_tau;
        if (cale.tau_cool > 0.0)
            std::fprintf(stderr,
                "    Newton cooling ON: τ_cool=%.3e s\n", cale.tau_cool);

        // Bottom enthalpy-flux heating.  Either F directly or L/(4π R²).
        double F_bot = cfg.cart_ale2_heat_flux;
        if (F_bot <= 0.0 && cfg.cart_ale2_heat_lsun > 0.0) {
            if (cfg.cart_ale2_heat_bot_R <= 0.0) {
                std::fprintf(stderr,
                    "  [error] --heat-lsun requires --heat-bot-R "
                    "(bottom-face radius in cm)\n");
                return 1;
            }
            double R = cfg.cart_ale2_heat_bot_R;
            F_bot = cfg.cart_ale2_heat_lsun / (4.0 * M_PI * R * R);
        }
        if (F_bot > 0.0 || cfg.cart_ale2_cool_top_frac < 1.0) {
            cale.configure_thermal(F_bot,
                                   cfg.cart_ale2_heat_bot_frac,
                                   cfg.cart_ale2_cool_top_frac);
        }
    } else if (cfg.test_case == "andrassy2022") {
        // Andrassy 2022 uses reflective walls in y, periodic in x.
        if (!((cale.bc_mode & 1) && !(cale.bc_mode & 2))) {
            std::fprintf(stderr,
                "  [warn] andrassy2022 requires --bc-x periodic --bc-y reflect; "
                "current bc_mode=%d\n", cale.bc_mode);
        }
        if (cfg.cart_ale2_slab_file.empty()) {
            std::fprintf(stderr,
                "  [error] andrassy2022 requires --ic-slab "
                "(6-col slab from scripts/andrassy2022/build_ic.py)\n");
            return 1;
        }
        cale.init_andrassy2022(cfg.cart_ale2_slab_file,
                               cfg.cart_ale2_andrassy_amp,
                               cfg.cart_ale2_andrassy_seed,
                               cfg.cart_ale2_andrassy_noise);
        // No Newton cooling per paper §2.2; heating comes from slab column 6,
        // set inside init_andrassy2022 via configure_heating_profile.
        cale.tau_cool = 0.0;
    } else if (cfg.test_case == "sedov2d") {
        // 2D cylindrical Sedov. Standard test: E0=1, rho=1, p_amb=1e-5.
        // r_exp chosen so a 256² grid has ~4 cells inside the hot spot.
        double r_exp = 2.5 * (Lx / std::max(cfg.nr, 1));
        cale.init_sedov(1.0, 1.0e-5, 1.0, r_exp);
    } else if (cfg.test_case == "noh") {
        cale.init_noh();
    } else if (cfg.test_case == "gresho") {
        cale.init_gresho();
    } else if (cfg.test_case == "yee_vortex") {
        cale.init_yee_vortex(5.0, 1.0, 1.0);
    } else if (is_shear) {
        if (cale.bc_mode != 3) {
            std::fprintf(stderr,
                "  [warn] shear_mode requires --bc-x periodic --bc-y periodic; "
                "current bc_mode=%d\n", cale.bc_mode);
        }
        cale.init_shear_mode(cfg.shear_rho, cfg.shear_P,
                             cfg.shear_V0, cfg.shear_k);
    } else if (is_ewave) {
        if (cale.bc_mode != 3) {
            std::fprintf(stderr,
                "  [warn] entropy_wave requires --bc-x periodic --bc-y periodic; "
                "current bc_mode=%d\n", cale.bc_mode);
        }
        cale.init_entropy_wave(cfg.ewave_rho0, cfg.ewave_P0,
                               cfg.ewave_u0, cfg.ewave_A, cfg.ewave_k);
        // Set t_end to N periods (period = Lx / u0) unless user overrode tend.
        if (cfg.t_end == 1.0) {
            cfg.t_end = cfg.ewave_periods * Lx / cfg.ewave_u0;
            std::fprintf(stderr,
                "  entropy_wave: auto t_end = %g (%.3g periods)\n",
                cfg.t_end, cfg.ewave_periods);
        }
    } else {
        cale.init_sod();
    }

    std::timespec wall_start;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);

    char csv_path[512];
    std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", ctx.run_dir.c_str());
    std::FILE* csv = std::fopen(csv_path, "w");
    if (cale.tracer_enabled)
        std::fprintf(csv, "step,t,dt,mass,KE,IE,PE,E,max_v,max_mach,species_mass\n");
    else
        std::fprintf(csv, "step,t,dt,mass,KE,IE,PE,E,max_v,max_mach\n");

    int diag_every = cfg.diag_interval > 0 ? cfg.diag_interval : cfg.output_interval;
    int vtk_every  = cfg.vtk_interval  > 0 ? cfg.vtk_interval  : cfg.output_interval;
    bool vtk_by_time = cfg.vtk_dt > 0.0;
    double next_vtk_t = 0.0;
    if (vtk_by_time) {
        std::fprintf(stderr,
            "  CartAle2 diag every %d steps, VTK every dt=%g (physical time)%s\n",
            diag_every, cfg.vtk_dt,
            cfg.frame_buffer ? " (VRAM buffered)" : "");
    } else {
        std::fprintf(stderr, "  CartAle2 diag every %d steps, VTK every %d steps%s\n",
                     diag_every, vtk_every,
                     cfg.frame_buffer ? " (VRAM buffered)" : "");
    }
    if (cfg.frame_buffer) cale.alloc_frame_buffer(cfg.frame_headroom_mb);

    // Optional diagnostic tracer hook
    TraceHook trace;
    bool trace_enabled = false;
    {
        std::vector<int> pick_ic, pick_jc;
        bool has_picks = !cfg.cart_ale2_trace_cells.empty() &&
                         parse_trace_cells(cfg.cart_ale2_trace_cells, pick_ic, pick_jc);
        bool want_trace = has_picks || cfg.cart_ale2_trace_step_cap > 0;
        if (want_trace) {
            int cap = cfg.cart_ale2_trace_step_cap > 0
                      ? cfg.cart_ale2_trace_step_cap : 0;
            trace.allocate(cale, cap, pick_ic, pick_jc, ctx.run_dir);
            cale.trace = &trace;
            trace_enabled = true;
        }
    }
    int trace_frame = 0;

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
            if (cale.tracer_enabled) {
                double m_species = cale.total_species_mass();
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
            if (cfg.frame_buffer && cale.frame_capacity > 0) {
                if (cale.frame_count >= cale.frame_capacity)
                    cale.flush_frames_to_disk(ctx.run_dir, Lx, Ly);
                cale.capture_frame(t, step);
            } else {
                ++frame;
                char path[512];
                std::snprintf(path, sizeof(path), "%s/output_%04d.vtk", ctx.run_dir.c_str(), frame);
                cale.write_vtk_2d(path, Lx, Ly);
            }
            if (trace_enabled) {
                trace.flush_cum_to_csv(t, trace_frame++);
            }
        }
    }
    if (cfg.frame_buffer) cale.flush_frames_to_disk(ctx.run_dir, Lx, Ly);
    if (trace_enabled) {
        trace.flush_pick_to_csv();
        cale.trace = nullptr;
        trace.destroy();
    }
    std::fclose(csv);
    std::fprintf(stderr, "\n");
    if (cfg.compute_error && is_ewave) {
        cale.compute_entropy_wave_error(
            t, step,
            cfg.ewave_rho0, cfg.ewave_P0, cfg.ewave_u0,
            cfg.ewave_A, cfg.ewave_k, cfg.ewave_periods,
            ctx.run_dir);
    }
    if (cfg.compute_error && is_sod) {
        cale.compute_sod_error(t, step, ctx.run_dir);
    }
    if (cfg.compute_error && is_gresho) {
        cale.compute_gresho_error(t, step, ctx.run_dir);
    }
    if (cfg.compute_error && is_yee) {
        cale.compute_yee_error(t, step, /*beta=*/5.0,
                                /*u_inf=*/1.0, /*v_inf=*/1.0,
                                ctx.run_dir);
    }
    cale.destroy();
    return 0;
}
#endif
