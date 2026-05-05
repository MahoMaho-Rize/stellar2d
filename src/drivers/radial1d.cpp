#include "drivers/drivers.h"

#ifdef USE_GPU
#include "radial1d_solver.cuh"
#include "sim/helpers.h"
#include "sim/run_loop.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <vector>

int run_radial1d(SimConfig& cfg, SimContext& ctx, double& t, int& step) {
    // ===== 1D Lagrangian radial solver (MESA RSP-inspired) =====
    // Ignores the 2D Grid; uses nr as number of Lagrangian zones.
    // Lane-Emden specific; other test cases not supported yet.
    if (cfg.test_case != "lane_emden" && cfg.test_case != "lane_emden_perturbed") {
        std::fprintf(stderr, "ERROR: radial1d solver only supports lane_emden / lane_emden_perturbed\n");
        return 1;
    }
    Radial1DSolver r1d;
    r1d.init(cfg.nr, cfg.gamma, cfg.G, cfg.cfl);
    // Wire EOS: if user picked anything other than ideal, use EOS-aware kernels.
    if (cfg.eos_type != "ideal") {
        r1d.use_eos = true;
        r1d.eos = ctx.eos;
        std::printf("radial1d: EOS-aware kernels enabled (%s)\n", cfg.eos_type.c_str());
    }
    // Wire radiation diffusion
    if (cfg.radiation_enabled) {
        r1d.radiation_enabled = true;
        r1d.rad_c_light = cfg.rad_c_light;
        r1d.rad_a_rad = cfg.eos_rad_a > 0 ? cfg.eos_rad_a : 1.0;
        std::printf("radial1d: radiation diffusion ON (c=%.3e, a=%.3e)\n",
                    r1d.rad_c_light, r1d.rad_a_rad);
    }
    // Wire tabulated opacity, if loaded
    if (ctx.kap_loaded) {
        r1d.kap_use_table      = true;
        r1d.kap_view_lowT      = ctx.kap_tbl_lowT.view;
        r1d.kap_view_highT     = ctx.kap_tbl_highT.view;
        r1d.kap_logT_lo_end    = cfg.kap_logT_lo_end;
        r1d.kap_logT_hi_start  = cfg.kap_logT_hi_start;
        r1d.kap_hydrogen_X     = cfg.nuc_X;     // composition slice
        std::printf("radial1d: tabulated kap ON (X slice = %.3f)\n",
                    r1d.kap_hydrogen_X);
    }
    // Wire nuclear burning
    if (cfg.nuclear_enabled) {
        r1d.nuclear_enabled = true;
        r1d.nuc_X = cfg.nuc_X;
        r1d.nuc_Y = cfg.nuc_Y;
        r1d.nuc_epsilon_scale = cfg.nuc_epsilon_scale;
        r1d.nuc_T_floor = cfg.nuc_T_floor;
        r1d.nuc_T_scale = cfg.nuc_T_scale;
        r1d.nuc_q_burn = cfg.nuc_q_burn;
        std::printf("radial1d: pp-chain nuclear burning ON (X=%.2f, scale=%.3e, T_floor=%.3eK, T_scale=%.3e, q=%.3e)\n",
                    r1d.nuc_X, r1d.nuc_epsilon_scale, r1d.nuc_T_floor, r1d.nuc_T_scale, r1d.nuc_q_burn);
    }
    // Wire species tracking (requires --nuclear)
    if (cfg.species_enabled) {
        if (!cfg.nuclear_enabled) {
            std::fprintf(stderr, "WARN: --species without --nuclear has no effect; enabling nuclear\n");
            r1d.nuclear_enabled = true;
            r1d.nuc_X = cfg.nuc_X;
            r1d.nuc_Y = cfg.nuc_Y;
            r1d.nuc_epsilon_scale = cfg.nuc_epsilon_scale;
            r1d.nuc_T_floor = cfg.nuc_T_floor;
            r1d.nuc_T_scale = cfg.nuc_T_scale;
            r1d.nuc_q_burn = cfg.nuc_q_burn;
        }
        r1d.species_enabled = true;
        std::printf("radial1d: species tracking ON (X→Y burn-up)\n");
    }
    if (cfg.mlt_enabled) {
        r1d.mlt_enabled = true;
        r1d.mlt_alpha = cfg.mlt_alpha;
        std::printf("radial1d: MLT convection ON (α=%.2f)\n", cfg.mlt_alpha);
    }
    if (!cfg.ic_mesa_path.empty()) {
        std::printf("radial1d: MESA IC from %s (seed=%s, atm_zones=%d)\n",
                    cfg.ic_mesa_path.c_str(),
                    cfg.ic_mesa_seed_T ? "T" : "P",
                    cfg.ic_mesa_atm_zones);
        if (r1d.init_from_mesa(cfg.ic_mesa_path.c_str(),
                               cfg.ic_mesa_seed_T,
                               cfg.ic_mesa_atm_zones) != 0) {
            std::fprintf(stderr, "ERROR: init_from_mesa failed\n");
            return 1;
        }
    } else if (cfg.ic_solar) {
        // Physical cgs Lane-Emden IC with user-specified (ρ_c, R_star, n).
        // Derive K so that α·ξ_1 = R_star exactly.
        //   α² = (n+1) K ρ_c^(1/n − 1) / (4πG)
        //   ⇒ K = α² · 4πG · ρ_c^(1 − 1/n) / (n+1),  α = R_star / ξ_1
        // Pre-computed ξ_1 for common n:
        double n_pol = cfg.ic_n_poly;
        // crude ξ_1 table: n=1.5 → 3.65375; n=3.0 → 6.89685
        double xi1 = (std::fabs(n_pol - 1.5) < 1e-3) ? 3.65375
                   : (std::fabs(n_pol - 3.0) < 1e-3) ? 6.89685
                   : 3.65375;
        double rho_c = (cfg.ic_rho_c > 0) ? cfg.ic_rho_c : 80.0;     // g/cc
        double R_star = (cfg.ic_R_star > 0) ? cfg.ic_R_star : 7.0e10; // cm
        double G_cgs = cfg.G; // user must pass --G 6.674e-8 or equivalent
        double alpha = R_star / xi1;
        double K_poly = alpha * alpha * 4.0 * M_PI * G_cgs
                        * std::pow(rho_c, 1.0 - 1.0/n_pol)
                        / (n_pol + 1.0);
        std::printf("radial1d: solar polytrope IC  n=%.2f  ρ_c=%.3e g/cc  R⋆=%.3e cm  K=%.3e  G=%.3e\n",
                    n_pol, rho_c, R_star, K_poly, G_cgs);
        r1d.init_lane_emden(rho_c, K_poly, n_pol);
    } else {
        r1d.init_lane_emden(1.0, 1.0, 1.5);          // ρ_c=1, K=1, n=1.5
    }
    r1d.snapshot_hse();
    if (r1d.species_enabled) {
        r1d.init_species_uniform(r1d.nuc_X, r1d.nuc_Y);
    }
    // Capture R(U_hse) BEFORE any perturbation so the well-balanced
    // subtraction references the true HSE, not the perturbed state.
    if (cfg.implicit_mode) {
        r1d.implicit_enabled = true;
        if (cfg.no_viallet) r1d.use_viallet_scaling = false;
        r1d.precond_tridiag = cfg.precond_tridiag;
        r1d.jfnk_autodiff   = cfg.jfnk_autodiff;
        r1d.no_rhse_subtract = cfg.no_rhse_subtract;
        r1d.rad_T_phot_floor = cfg.rad_T_phot_floor;
        if (cfg.newton_tol_override > 0) r1d.newton_tol = cfg.newton_tol_override;
        r1d.hse_resnap_interval = cfg.hse_resnap_interval;
        r1d.nuc_compress_frac = cfg.nuc_compress_frac;
        r1d.nz_atm_split = cfg.atm_split;
        r1d.init_implicit();
        r1d.snapshot_hse_implicit();
    }
    if (cfg.test_case == "lane_emden_perturbed")
        r1d.apply_perturbation(cfg.perturb_amplitude);

    std::timespec wall_start;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);

    // Simple text output (CSV format) for radial1d mode
    char csv_path[512];
    std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", ctx.run_dir.c_str());
    std::FILE* csv = std::fopen(csv_path, "w");
    if (cfg.species_enabled) {
        std::fprintf(csv, "step,t,dt,mass,KE,IE,PE,total_E,max_mach,max_vr,T_c,rho_c,L_nuc,mass_H,mass_He,X_core,X_surf,L_surf,conv_mfrac,r_conv_in,r_conv_out,max_super,T_phot,phot_zone,R_surf\n");
    } else {
        std::fprintf(csv, "step,t,dt,mass,KE,IE,PE,total_E,max_mach,max_vr,T_c,rho_c,L_nuc,L_surf,conv_mfrac,r_conv_in,r_conv_out,max_super,T_phot,phot_zone,R_surf\n");
    }

    int frame = 0;
    if (cfg.implicit_mode) {
        std::printf("radial1d: IMPLICIT mode ON (Viallet=%s, Newton tol=%.1e, GMRES tol=%.1e)\n",
                    r1d.use_viallet_scaling ? "on" : "off",
                    r1d.newton_tol, r1d.gmres_tol);
    }
    while (t < cfg.t_end && !g_interrupted) {
        double dt;
        if (cfg.implicit_mode) {
            // Adaptive dt control optimised for crossing τ_KH:
            //   - Start from --dt-implicit (seed).
            //   - On success, grow ×2 per accepted step; cap at --dt-implicit
            //     if given (explicit ceiling) or at 10·τ_dyn otherwise.
            //   - On failure, cut ×0.1 (not ×0.5) — aggressive because
            //     stiff nuclear flashes usually need a big reset.
            //   - Never fall back to explicit: accelerated nuclear would
            //     blow up.
            static double dt_req_state = 0.0;
            double dt_seed = (cfg.dt_implicit > 0) ? cfg.dt_implicit
                                                   : cfg.dt_implicit_scale * r1d.compute_dt();
            if (dt_req_state <= 0.0) dt_req_state = dt_seed;
            // Ceiling: explicit --dt-implicit if provided, else let it grow
            // unbounded (user controls via --tend).
            double dt_cap = (cfg.dt_implicit > 0) ? cfg.dt_implicit : 1e30;

            dt = r1d.step_implicit(t, cfg.t_end, dt_req_state);
            if (dt <= 0.0) {
                dt_req_state *= 0.1;
                if (dt_req_state < 1e-30) dt_req_state = 1e-30;
                std::fprintf(stderr,
                    "  step %d: implicit FAILED, outer dt reset to %.3e\n",
                    step, dt_req_state);
                dt = 0.0;
            } else {
                // Geometric growth on success — cross many τ_dyn fast.
                dt_req_state = std::min(dt_cap, dt_req_state * 2.0);

                // Thermal-timescale cap: dt ≤ thermal_frac · IE / L_surf
                // (Kelvin-Helmholtz). Without this, once rad coupling
                // actually cools the star the outer dt grows ×2 per step
                // and overshoots a full τ_KH in one step, blowing the
                // solution past hydrostatic equilibrium.
                if (cfg.dt_thermal_frac > 0.0 || cfg.dt_mach_cap > 0.0) {
                    Radial1DSolver::Diagnostics dg = r1d.compute_diagnostics();
                    double L_tot = std::fabs(r1d.rad_impl_L_surf);
                    if (cfg.dt_thermal_frac > 0.0
                        && L_tot > 1e-30 && dg.total_internal_E > 0.0) {
                        double tau_th = dg.total_internal_E / L_tot;
                        double dt_th  = cfg.dt_thermal_frac * tau_th;
                        if (dt_req_state > dt_th) dt_req_state = dt_th;
                    }
                    // Mach-based damping: if the last step's hydro solution
                    // has transient velocities (operator-split rad can
                    // leave momentum imbalance), shrink dt until the
                    // transient decays below mach_cap.
                    if (cfg.dt_mach_cap > 0.0 && dg.max_mach > cfg.dt_mach_cap) {
                        double shrink = cfg.dt_mach_cap / dg.max_mach;
                        if (shrink < 0.1) shrink = 0.1;
                        dt_req_state *= shrink;
                    }
                }
            }
        } else {
            dt = r1d.step(t, cfg.t_end);
        }
        t += dt;
        step++;

        if (step % 200 == 0)
            print_progress(t, cfg.t_end, step, dt, wall_start);

        if (step % cfg.output_interval == 0) {
            auto d = r1d.compute_diagnostics();
            auto mlt = r1d.compute_convection_diag();
            std::fprintf(stderr, "\n");
            std::printf("Step %6d  t=%.6e dt=%.3e M=%.10e E=%.10e |v|_max=%.3e Mach_max=%.3e\n",
                        step, t, dt, d.total_mass, d.total_E, d.max_vr, d.max_mach);
            if (r1d.nuclear_enabled) {
                double L_ratio = (r1d.rad_impl_L_surf > 0) ? d.L_nuc / r1d.rad_impl_L_surf : 0.0;
                std::printf("    ignition: T_c=%.3e K  ρ_c=%.3e g/cc  L_nuc=%.3e erg/s  L_nuc/L_surf=%.3e\n",
                            d.T_c, d.rho_c, d.L_nuc, L_ratio);
            }
            if (r1d.radiation_enabled) {
                std::printf("    rad_BC:   phot_zone=%d  T_phot=%.3e K  tau_sum=%.3e  L_surf=%.3e erg/s\n",
                            r1d.rad_impl_phot_zone, r1d.rad_impl_T_phot,
                            r1d.rad_impl_tau_surf, r1d.rad_impl_L_surf);
            }
            if (mlt.n_conv_zones > 0) {
                std::printf("    MLT: conv_mass_frac=%.4f  r_conv=[%.3e,%.3e]  n_conv=%d  max_super=%.3e\n",
                            mlt.conv_mass_frac, mlt.r_conv_inner, mlt.r_conv_outer,
                            mlt.n_conv_zones, mlt.max_superadiab);
            }
            // R_surf = outermost face radius (read once per CSV row)
            double R_surf = 0.0;
            {
                std::vector<double> r_f, v_f, rho_c3, P_c3, e_c3;
                r1d.download_profile(r_f, v_f, rho_c3, P_c3, e_c3);
                if (!r_f.empty()) R_surf = r_f.back();
            }
            if (r1d.species_enabled) {
                std::vector<double> X_c, Y_c;
                r1d.download_species(X_c, Y_c);
                // Mass-weighted totals over zones: need dm; use e_cell/P_cell buffers via profile
                std::vector<double> r_f, v_f, rho_c2, P_c2, e_c2;
                r1d.download_profile(r_f, v_f, rho_c2, P_c2, e_c2);
                const double PI43 = 4.188790204786391;
                double mH = 0.0, mHe = 0.0;
                for (int k = 0; k < r1d.lev.nz; ++k) {
                    double rL = r_f[k], rR = r_f[k+1];
                    double dmk = rho_c2[k] * PI43 * (rR*rR*rR - rL*rL*rL);
                    mH  += dmk * X_c[k];
                    mHe += dmk * Y_c[k];
                }
                double X_core = X_c.front();
                double X_surf = X_c.back();
                std::fprintf(csv, "%d,%.10e,%.6e,%.10e,%.10e,%.10e,%.10e,%.10e,%.6e,%.6e,%.6e,%.6e,%.6e,%.10e,%.10e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%d,%.6e\n",
                             step, t, dt, d.total_mass, d.total_KE, d.total_internal_E,
                             d.total_grav_E, d.total_E, d.max_mach, d.max_vr,
                             d.T_c, d.rho_c, d.L_nuc,
                             mH, mHe, X_core, X_surf, r1d.rad_impl_L_surf,
                             mlt.conv_mass_frac, mlt.r_conv_inner, mlt.r_conv_outer, mlt.max_superadiab,
                             r1d.rad_impl_T_phot, r1d.rad_impl_phot_zone, R_surf);
            } else {
                std::fprintf(csv, "%d,%.10e,%.6e,%.10e,%.10e,%.10e,%.10e,%.10e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%d,%.6e\n",
                             step, t, dt, d.total_mass, d.total_KE, d.total_internal_E,
                             d.total_grav_E, d.total_E, d.max_mach, d.max_vr,
                             d.T_c, d.rho_c, d.L_nuc, r1d.rad_impl_L_surf,
                             mlt.conv_mass_frac, mlt.r_conv_inner, mlt.r_conv_outer, mlt.max_superadiab,
                             r1d.rad_impl_T_phot, r1d.rad_impl_phot_zone, R_surf);
            }
            std::fflush(csv);

            // Dump profile as simple text. When --rich-profile is set,
            // also emit T, κ, Γ₁, ∇_ad, ∇_rad, L, mixing type, v_conv
            // so scripts/mesa/pk_mesa_radial1d.py can compare per-field.
            std::vector<double> r_face, v_face, rho_cell, P_cell, e_cell;
            std::vector<double> T_cell, kap_cell, g1_cell, ga_cell, gr_cell;
            std::vector<double> L_face, vc_cell;
            std::vector<int>    mt_cell;
            if (cfg.rich_profile) {
                r1d.download_profile_rich(r_face, v_face, rho_cell, P_cell, e_cell,
                                          T_cell, kap_cell, g1_cell, ga_cell, gr_cell,
                                          L_face, mt_cell, vc_cell);
            } else {
                r1d.download_profile(r_face, v_face, rho_cell, P_cell, e_cell);
            }
            std::vector<double> X_cell, Y_cell;
            if (r1d.species_enabled) r1d.download_species(X_cell, Y_cell);
            char path[512];
            std::snprintf(path, sizeof(path), "%s/profile_%04d.txt", ctx.run_dir.c_str(), ++frame);
            std::FILE* fp = std::fopen(path, "w");
            if (cfg.rich_profile) {
                std::fprintf(fp,
                    "# t = %.10e  step = %d\n"
                    "# k r_face v_face rho P e_int T kap gamma1 grada gradr L_face mixing_type conv_vel%s\n",
                    t, step, r1d.species_enabled ? " X Y" : "");
            } else if (r1d.species_enabled) {
                std::fprintf(fp, "# t = %.10e  step = %d\n# k r_face v_face rho P e_int X Y\n", t, step);
            } else {
                std::fprintf(fp, "# t = %.10e  step = %d\n# k r_face v_face rho P e_int\n", t, step);
            }
            for (int k = 0; k < r1d.lev.nz; ++k) {
                if (cfg.rich_profile) {
                    std::fprintf(fp,
                        "%d %.10e %.10e %.10e %.10e %.10e "
                        "%.10e %.10e %.6e %.6e %.6e %.6e %d %.6e",
                        k, r_face[k], v_face[k], rho_cell[k], P_cell[k], e_cell[k],
                        T_cell[k], kap_cell[k], g1_cell[k], ga_cell[k], gr_cell[k],
                        L_face[k], mt_cell[k], vc_cell[k]);
                    if (r1d.species_enabled)
                        std::fprintf(fp, " %.6e %.6e", X_cell[k], Y_cell[k]);
                    std::fprintf(fp, "\n");
                } else if (r1d.species_enabled) {
                    std::fprintf(fp, "%d %.10e %.10e %.10e %.10e %.10e %.6e %.6e\n",
                                 k, r_face[k], v_face[k], rho_cell[k], P_cell[k], e_cell[k],
                                 X_cell[k], Y_cell[k]);
                } else {
                    std::fprintf(fp, "%d %.10e %.10e %.10e %.10e %.10e\n",
                                 k, r_face[k], v_face[k], rho_cell[k], P_cell[k], e_cell[k]);
                }
            }
            // last face (velocity only; cell-centred fields don't apply)
            std::fprintf(fp, "%d %.10e %.10e - - -\n", r1d.lev.nz, r_face[r1d.lev.nz], v_face[r1d.lev.nz]);
            std::fclose(fp);
        }
    }
    std::fclose(csv);
    std::fprintf(stderr, "\n");
    if (cfg.implicit_mode) r1d.destroy_implicit();
    r1d.destroy();
    return 0;
}
#endif
