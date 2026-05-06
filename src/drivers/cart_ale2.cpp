#include "drivers/drivers.h"

#ifdef USE_GPU
#include "cart_ale2_solver.cuh"
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
    double Lx = 1.0;
    // Lecoanet: domain aspect 1:2 so shear layers at y=0.5, y=1.5 match
    // Athena++ iprob=4 geometry (z1=-0.5, z2=0.5 in centred coords).
    double Ly = is_kh_lec ? 2.0
              : (is_hse || is_kh) ? 1.0
              : 0.2;
    double gam = is_hse ? cfg.gamma : 1.4;
    // andrassy2022 uses γ=5/3 (paper §2.2).
    if (is_andrassy) gam = 5.0 / 3.0;
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
                               cfg.cart_ale2_andrassy_amp);
        // No Newton cooling per paper §2.2; heating comes from slab column 6,
        // set inside init_andrassy2022 via configure_heating_profile.
        cale.tau_cool = 0.0;
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
        }
    }
    if (cfg.frame_buffer) cale.flush_frames_to_disk(ctx.run_dir, Lx, Ly);
    std::fclose(csv);
    std::fprintf(stderr, "\n");
    cale.destroy();
    return 0;
}
#endif
