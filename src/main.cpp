#include "grid.h"
#include "state.h"
#include "eos.h"
#include "bc/boundary.h"
#include "hydro/flux.h"
#include "hydro/integrate.h"
#include "hydro/reconstruct.h"
#include "gravity/gmg.h"
#include "io/output.h"
#include "init/lane_emden.h"
#include "init/sedov.h"
#include "init/jeans.h"
#include "init/evrard.h"

#ifdef USE_GPU
#include <cuda_runtime.h>
#ifdef USE_AMGX
#include "gpu_solver.h"
#endif
#include "lowmach_solver.h"
#include "fas_solver.cuh"
#include "fas2_solver.cuh"
#include "simple_solver.cuh"
#include "projection_solver.cuh"
#include "radial1d_solver.cuh"
#include "wb2d_solver.cuh"
#include "ale2d_solver.cuh"
#include "cart_lag_solver.cuh"
#include "cart_ale_solver.cuh"
#include "cart_ale2_solver.cuh"
#include "cart_impl_solver.cuh"
#include "pseudo_spectral_solver.cuh"
#include "anelastic_sl_solver.cuh"
#include "stellar_profile.h"
#include "sph2d_spectral_solver.cuh"
#include "physics/helmholtz_eos.cuh"
#endif

#include <cstdio>
#include <cmath>
#include <string>
#include <cstring>
#include <ctime>
#include <array>
#include <vector>
#include <csignal>
#include <functional>
#include <algorithm>
#include <sys/stat.h>

#include "cli/options.h"
#include "sim/helpers.h"
#include "sim/setup.h"
#include "sim/run_loop.h"
#include "drivers/drivers.h"

// g_interrupted is defined in sim/run_loop.cpp.
static void handle_sigint(int) { g_interrupted = 1; }


int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);
    std::signal(SIGTERM, handle_sigint);

    SimConfig cfg;

    if (int rc = parse_cli(argc, argv, cfg); rc != 0) return rc;

    SimContext ctx;
    if (int rc = setup_simulation(cfg, ctx); rc != 0) return rc;

    // Local aliases so the (still-inlined) solver branches below keep working
    // without modification. The coming steps migrate each branch into its own
    // run_xxx(cfg, ctx) driver; until then we expose ctx members as bare names.
    Grid& grid = ctx.grid;
    State& state = ctx.state;
    EOS& eos = ctx.eos;
    std::string& run_dir = ctx.run_dir;
#ifdef USE_GPU
    HelmholtzTable& helm_tbl = ctx.helm_tbl;
    bool& helm_loaded = ctx.helm_loaded;
    KapTable& kap_tbl_lowT = ctx.kap_tbl_lowT;
    KapTable& kap_tbl_highT = ctx.kap_tbl_highT;
    bool& kap_loaded = ctx.kap_loaded;
#endif

    double t = 0.0;
    int step = 0;

    std::printf("Starting time integration...\n");

#ifdef USE_GPU
    // ── Solver adapter: type-erased callbacks for the time-stepping loop ──

    if (cfg.solver_type == "radial1d") {
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
            r1d.eos = eos;
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
        if (kap_loaded) {
            r1d.kap_use_table      = true;
            r1d.kap_view_lowT      = kap_tbl_lowT.view;
            r1d.kap_view_highT     = kap_tbl_highT.view;
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
        std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", run_dir.c_str());
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
                std::snprintf(path, sizeof(path), "%s/profile_%04d.txt", run_dir.c_str(), ++frame);
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

    } else if (cfg.solver_type == "projection") {
        if (int rc = run_projection(cfg, ctx, t, step); rc != 0) return rc;
    } else if (cfg.solver_type == "simple") {
        if (int rc = run_simple(cfg, ctx, t, step); rc != 0) return rc;
    } else if (cfg.solver_type == "fas2") {
        // ===== fas2: experimental fork of FAS for low-Mach robustness =====
        // Clone of FasSolver with incremental fixes:
        //   - CGS2 Gram-Schmidt (Knoll-Keyes 2004)
        //   - Unit-normalize v in JFNK matvec (Trilinos NOX)
        //   - Viallet 2016 eq 72 asymmetric L/R scaling
        //   - Line-implicit-in-r preconditioner (cherry-pick from line-jacobi-precond)
        FasSolver2 fas;
        fas.use_simple_smoother = (cfg.precond != "block_jacobi");
        // fas2-specific: --precond line_r toggles line-implicit-in-r JFNK preconditioner
        fas.use_line_precond_r = (cfg.precond == "line_r");
        fas.limiter_type = static_cast<int>(cfg.limiter);
        fas.hllc_variant = cfg.hllc_variant;
        fas.radial_only = cfg.radial_only;
        fas.init(grid, eos, cfg.G, cfg.cfl);
        if (cfg.radial_only)
            std::printf("fas2 radial-only mode\n");
        if (cfg.no_sponge) fas.sponge_kappa = 0.0;
        configure_mass_mesh(cfg, fas);
        if (cfg.mesh_type == "mass" && cfg.r_inner <= 0)
            fas.central_damp_r = 0.15 * cfg.R_outer;
        snapshot_hse_if_needed(cfg, ctx, fas);
        fas.upload_state(grid, state);

        FasLevel2& fl = fas.levels[0];
        int snap_size = fl.total;
        int max_snaps = static_cast<int>(cfg.t_end / (cfg.output_interval * 1e-5)) + 100;
        long long bytes_per_snap = 4LL * snap_size * sizeof(double);
        size_t mem_free = 0, mem_total = 0;
        cudaMemGetInfo(&mem_free, &mem_total);
        long long max_bytes = static_cast<long long>(mem_free) / 2;
        if (max_snaps > max_bytes / bytes_per_snap)
            max_snaps = static_cast<int>(max_bytes / bytes_per_snap);
        if (max_snaps < 1) max_snaps = 1;

        double* d_snap_buf = nullptr;
        CUDA_CHECK(cudaMalloc(&d_snap_buf, (long long)max_snaps * 4 * snap_size * sizeof(double)));
        std::vector<double> snap_times, snap_dts;
        std::vector<int> snap_steps;
        int n_snaps = 0;

        SolverOps ops;
        ops.progress_interval = 2000;
        ops.step = [&](double t_, double te) { return fas.step(t_, te); };
        ops.download = [&](const Grid&, State&, double dt_val) {
            if (n_snaps >= max_snaps) return;
            long long off = (long long)n_snaps * 4 * snap_size;
            CUDA_CHECK(cudaMemcpy(d_snap_buf + off, fl.d_rho, snap_size*sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_snap_buf + off + snap_size, fl.d_mr, snap_size*sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_snap_buf + off + 2*snap_size, fl.d_mt, snap_size*sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_snap_buf + off + 3*snap_size, fl.d_rhoE, snap_size*sizeof(double), cudaMemcpyDeviceToDevice));
            snap_times.push_back(t);
            snap_dts.push_back(dt_val);
            snap_steps.push_back(step);
            n_snaps++;
        };
        ops.destroy = [&]() {
            if (g_interrupted)
                std::printf("\nInterrupted at step %d, t=%.6e. ", step, t);
            std::printf("Writing %d snapshots from GPU...\n", n_snaps);
            std::vector<double> h_buf(4LL * snap_size);
            for (int s = 0; s < n_snaps; ++s) {
                long long off = (long long)s * 4 * snap_size;
                CUDA_CHECK(cudaMemcpy(h_buf.data(), d_snap_buf + off, 4*snap_size*sizeof(double), cudaMemcpyDeviceToHost));
                for (int ii = 0; ii < fl.nr; ++ii)
                    for (int jj = 0; jj < fl.nt; ++jj) {
                        int k = grid.idx(ii, jj);
                        int kg = (ii + fl.ng) * (fl.nt + 2*fl.ng) + (jj + fl.ng);
                        state.rho[k] = h_buf[kg];
                        state.mr[k] = h_buf[snap_size + kg];
                        state.mtheta[k] = h_buf[2*snap_size + kg];
                        state.E[k] = h_buf[3*snap_size + kg];
                    }
                Diagnostics diag = compute_diagnostics(grid, state, cfg.gamma);
                std::printf("Step %6d  t = %.6e  dt = %.3e  M = %.10e  E = %.10e\n",
                            snap_steps[s], snap_times[s], snap_dts[s], diag.total_mass, diag.total_energy);
                char fname[512];
                std::snprintf(fname, sizeof(fname), "%s/output_%04d.vtk", run_dir.c_str(), s + 1);
                write_vtk(fname, grid, state, cfg.gamma);
            }
            cudaFree(d_snap_buf);
            fas.download_state(grid, state);
            fas.destroy();
        };
        run_time_loop(cfg, ctx, t, step, ops);

    } else if (cfg.solver_type == "fas" || cfg.solver_type == "explicit") {
        bool use_explicit = (cfg.solver_type == "explicit");
        FasSolver fas;
        fas.use_simple_smoother = (cfg.precond != "block_jacobi");
        fas.limiter_type = static_cast<int>(cfg.limiter);
        fas.hllc_variant = cfg.hllc_variant;
        fas.radial_only = cfg.radial_only;
        fas.init(grid, eos, cfg.G, cfg.cfl);
        if (cfg.radial_only)
            std::printf("Radial-only mode: v_theta=0 enforced, theta fluxes and atm_reset skipped\n");
        if (cfg.no_sponge) fas.sponge_kappa = 0.0;
        configure_mass_mesh(cfg, fas);
        if (cfg.mesh_type == "mass" && cfg.r_inner <= 0)
            fas.central_damp_r = 0.15 * cfg.R_outer;
        snapshot_hse_if_needed(cfg, ctx, fas);
        fas.upload_state(grid, state);

        // GPU snapshot buffer: store frames in VRAM, write all at end
        FasLevel& fl = fas.levels[0];
        int snap_size = fl.total;  // per-variable size (with ghost)
        int max_snaps = static_cast<int>(cfg.t_end / (cfg.output_interval * 1e-5)) + 100;
        long long bytes_per_snap = 4LL * snap_size * sizeof(double);
        size_t mem_free = 0, mem_total = 0;
        cudaMemGetInfo(&mem_free, &mem_total);
        long long max_bytes = static_cast<long long>(mem_free) / 2;
        if (max_snaps > max_bytes / bytes_per_snap)
            max_snaps = static_cast<int>(max_bytes / bytes_per_snap);
        if (max_snaps < 1) max_snaps = 1;

        double* d_snap_buf = nullptr;
        CUDA_CHECK(cudaMalloc(&d_snap_buf, (long long)max_snaps * 4 * snap_size * sizeof(double)));
        std::vector<double> snap_times;
        std::vector<double> snap_dts;
        std::vector<int> snap_steps;
        int n_snaps = 0;

        SolverOps ops;
        ops.progress_interval = 2000;
        ops.step = [&](double t_, double te) -> double {
            return use_explicit ? fas.step_explicit(t_, te) : fas.step(t_, te);
        };
        ops.download = [&](const Grid&, State&, double dt_val) {
            if (n_snaps >= max_snaps) return;
            long long off = (long long)n_snaps * 4 * snap_size;
            CUDA_CHECK(cudaMemcpy(d_snap_buf + off, fl.d_rho, snap_size*sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_snap_buf + off + snap_size, fl.d_mr, snap_size*sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_snap_buf + off + 2*snap_size, fl.d_mt, snap_size*sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_snap_buf + off + 3*snap_size, fl.d_rhoE, snap_size*sizeof(double), cudaMemcpyDeviceToDevice));
            snap_times.push_back(t);
            snap_dts.push_back(dt_val);
            snap_steps.push_back(step);
            n_snaps++;
        };
        ops.destroy = [&]() {
            if (g_interrupted)
                std::printf("\nInterrupted at step %d, t=%.6e. ", step, t);
            std::printf("Writing %d snapshots from GPU...\n", n_snaps);
            std::vector<double> h_buf(4LL * snap_size);
            for (int s = 0; s < n_snaps; ++s) {
                long long off = (long long)s * 4 * snap_size;
                CUDA_CHECK(cudaMemcpy(h_buf.data(), d_snap_buf + off, 4*snap_size*sizeof(double), cudaMemcpyDeviceToHost));
                for (int ii = 0; ii < fl.nr; ++ii)
                    for (int jj = 0; jj < fl.nt; ++jj) {
                        int k = grid.idx(ii, jj);
                        int kg = (ii + fl.ng) * (fl.nt + 2*fl.ng) + (jj + fl.ng);
                        state.rho[k] = h_buf[kg];
                        state.mr[k] = h_buf[snap_size + kg];
                        state.mtheta[k] = h_buf[2*snap_size + kg];
                        state.E[k] = h_buf[3*snap_size + kg];
                    }
                Diagnostics diag = compute_diagnostics(grid, state, cfg.gamma);
                std::printf("Step %6d  t = %.6e  dt = %.3e  M = %.10e  E = %.10e\n",
                            snap_steps[s], snap_times[s], snap_dts[s], diag.total_mass, diag.total_energy);
                char fname[512];
                std::snprintf(fname, sizeof(fname), "%s/output_%04d.vtk", run_dir.c_str(), s + 1);
                write_vtk(fname, grid, state, cfg.gamma);
            }
            cudaFree(d_snap_buf);
            fas.download_state(grid, state);
            fas.destroy();
        };
        run_time_loop(cfg, ctx, t, step, ops);

    } else if (cfg.solver_type == "cart_lag") {
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
        std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", run_dir.c_str());
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
                std::snprintf(path, sizeof(path), "%s/xslice_%04d.txt", run_dir.c_str(), ++frame);
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
    } else if (cfg.solver_type == "cart_ale") {
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
        std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", run_dir.c_str());
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
                        cale.flush_frames_to_disk(run_dir, Lx, Ly);
                    }
                    cale.capture_frame(t, step);
                } else {
                    ++frame;
                    char path[512];
                    std::snprintf(path, sizeof(path), "%s/output_%04d.vtk", run_dir.c_str(), frame);
                    cale.write_vtk_2d(path, Lx, Ly);
                }
            }
        }
        if (cfg.frame_buffer) {
            cale.flush_frames_to_disk(run_dir, Lx, Ly);
        }
        std::fclose(csv);
        std::fprintf(stderr, "\n");
        cale.destroy();
    } else if (cfg.solver_type == "cart_impl") {
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
        csol.init(cfg.nr, cfg.ntheta, Lx, Ly, eos, cfg.gamma, g_val, cfg.cfl);

        // Two-step init: HSE first, snapshot reference, then layer perturbation
        csol.init_hse_polytrope(rho_base, 0.0);
        csol.snapshot_hse();
        if (cfg.test_case == "hse_perturbed" && cfg.perturb_amplitude != 0.0) {
            csol.init_hse_polytrope(rho_base, cfg.perturb_amplitude);
        }

        std::timespec wall_start;
        clock_gettime(CLOCK_MONOTONIC, &wall_start);

        char csv_path[512];
        std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", run_dir.c_str());
        std::FILE* csv = std::fopen(csv_path, "w");
        std::fprintf(csv, "step,t,dt,mass,energy,max_v,max_mach\n");

        // Initial frame
        {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/output_0000.vtk", run_dir.c_str());
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
                std::snprintf(path, sizeof(path), "%s/output_%04d.vtk", run_dir.c_str(), frame);
                csol.write_vtk(path);
            }
        }
        std::fclose(csv);
        std::fprintf(stderr, "\n");
        csol.destroy();

    } else if (cfg.solver_type == "cart_ale2") {
        // ===== cart_ale2: full periodic BC + PPM-in-remap (in development) =====
        CartAle2Solver cale;
        bool is_hse = (cfg.test_case == "hse" || cfg.test_case == "hse_perturbed"
                       || cfg.test_case == "hse_bubble");
        bool is_kh = (cfg.test_case == "kh_shear");
        bool is_kh_lec = (cfg.test_case == "kh_lecoanet");
        bool is_loc_conv = (cfg.test_case == "local_convection");
        double Lx = 1.0;
        // Lecoanet: domain aspect 1:2 so shear layers at y=0.5, y=1.5 match
        // Athena++ iprob=4 geometry (z1=-0.5, z2=0.5 in centred coords).
        double Ly = is_kh_lec ? 2.0
                  : (is_hse || is_kh) ? 1.0
                  : 0.2;
        double gam = is_hse ? cfg.gamma : 1.4;
        // local_convection: read slab header to get real Ly, Lx, γ in cgs.
        if (is_loc_conv) {
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
        } else {
            cale.init_sod();
        }

        std::timespec wall_start;
        clock_gettime(CLOCK_MONOTONIC, &wall_start);

        char csv_path[512];
        std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", run_dir.c_str());
        std::FILE* csv = std::fopen(csv_path, "w");
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
                std::fprintf(csv, "%d,%.10e,%.6e,%.10e,%.10e,%.10e,%.10e,%.10e,%.6e,%.6e\n",
                             step, t, dt, d.total_mass, d.total_KE, d.total_internal_E,
                             d.total_PE, d.total_E, d.max_v, d.max_mach);
                std::fflush(csv);
            }
            if (do_vtk) {
                if (cfg.frame_buffer && cale.frame_capacity > 0) {
                    if (cale.frame_count >= cale.frame_capacity)
                        cale.flush_frames_to_disk(run_dir, Lx, Ly);
                    cale.capture_frame(t, step);
                } else {
                    ++frame;
                    char path[512];
                    std::snprintf(path, sizeof(path), "%s/output_%04d.vtk", run_dir.c_str(), frame);
                    cale.write_vtk_2d(path, Lx, Ly);
                }
            }
        }
        if (cfg.frame_buffer) cale.flush_frames_to_disk(run_dir, Lx, Ly);
        std::fclose(csv);
        std::fprintf(stderr, "\n");
        cale.destroy();
    } else if (cfg.solver_type == "pseudo_spectral") {
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
        std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", run_dir.c_str());
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
            std::snprintf(path, sizeof(path), "%s/output_%04d.vtk", run_dir.c_str(), ++frame);
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
                              run_dir.c_str());
                ps.save_checkpoint(cpath, t);
            }
            if (do_vtk) {
                if (cfg.frame_buffer && ps.frame_capacity > 0) {
                    if (ps.frame_count >= ps.frame_capacity)
                        ps.flush_frames_to_disk(run_dir);
                    ps.capture_frame(t, step);
                } else {
                    ++frame;
                    char path[512];
                    std::snprintf(path, sizeof(path), "%s/output_%04d.vtk",
                                  run_dir.c_str(), frame);
                    ps.write_vtk_2d(path);
                }
            }
        }
        if (cfg.frame_buffer) ps.flush_frames_to_disk(run_dir);
        std::fclose(csv);
        std::fprintf(stderr, "\n");
        ps.destroy();
    } else if (cfg.solver_type == "anelastic_sl") {
        // ===== Anelastic SL-spectral solver (Phase 1b: Poisson + self-test) =====
        if (cfg.test_case != "sl_basis_check"
            && cfg.test_case != "sl_poisson_test"
            && cfg.test_case != "sl_poisson_test_boussinesq"
            && cfg.test_case != "kh_shear_boussinesq"
            && cfg.test_case != "gmode_pulsation"
            && cfg.test_case != "gmode_2d_evp"
            && cfg.test_case != "gmode_eigenmode_td"
            && cfg.test_case != "gmode_exp_k"
            && cfg.test_case != "dns_triad"
            && cfg.test_case != "dns_triad_coupled") {
            std::fprintf(stderr,
                "ERROR: anelastic_sl supports --test {sl_basis_check, "
                "sl_poisson_test[_boussinesq], kh_shear_boussinesq, "
                "gmode_pulsation, gmode_2d_evp, gmode_eigenmode_td, gmode_exp_k, "
                "dns_triad, dns_triad_coupled}\n");
            return 1;
        }
        AnelasticSLSolver ansl;
        int n_modes = (cfg.nr > 0 ? cfg.nr / 2 : 128);
        ansl.init(cfg.ntheta, cfg.nr, n_modes,
                  cfg.ps_Lx, cfg.ps_Ly, cfg.ps_nu, cfg.cfl);

        std::string bg;
        double bg_arg = 0.01;
        if (cfg.test_case == "kh_shear_boussinesq"
            || cfg.test_case == "sl_poisson_test_boussinesq") {
            bg = "boussinesq";
        } else if (cfg.test_case == "gmode_pulsation"
                   || cfg.test_case == "gmode_2d_evp"
                   || cfg.test_case == "gmode_eigenmode_td"
                   || cfg.test_case == "dns_triad"
                   || cfg.test_case == "dns_triad_coupled") {
            // dns_triad / dns_triad_coupled default to Lane-Emden n=3/2 with
            // rho_cut=0.05 (triad) or 0.1 (coupled, for better long-time
            // stability); other tests default to stratified_n2 unless
            // overridden via ANSL_BG / ANSL_RHO_CUT.
            if (cfg.test_case == "dns_triad") {
                bg = "lane_emden_1_5";
                bg_arg = 0.05;
                if (const char* rc = std::getenv("ANSL_RHO_CUT")) {
                    double v = std::atof(rc);
                    if (v > 0.0 && v < 1.0) bg_arg = v;
                }
            } else if (cfg.test_case == "dns_triad_coupled") {
                bg = "lane_emden_1_5";
                bg_arg = 0.1;
                if (const char* rc = std::getenv("ANSL_RHO_CUT")) {
                    double v = std::atof(rc);
                    if (v > 0.0 && v < 1.0) bg_arg = v;
                }
            } else {
                bg = "stratified_n2";
                // Re-use ps_vshear as the N² value knob for this test (default 1.0).
                bg_arg = (cfg.ps_vshear > 0.0) ? cfg.ps_vshear : 1.0;
            }
            // Variable-density override: ANSL_BG=lane_emden_1_5 flips to a
            // real stratified Lane-Emden n=3/2 background (uses ANSL_RHO_CUT
            // for surface truncation, default 0.01).  The EVP, SL-filter and
            // TANH coord-map downstream all work identically; only ρ₀(y),
            // N²(y), and W̃(y) change.
            if (const char* bg_env = std::getenv("ANSL_BG")) {
                std::string s(bg_env);
                if (s == "lane_emden_1_5" || s == "lane_emden") {
                    bg = "lane_emden_1_5";
                    bg_arg = 0.01;
                    if (const char* rc = std::getenv("ANSL_RHO_CUT")) {
                        double v = std::atof(rc);
                        if (v > 0.0 && v < 1.0) bg_arg = v;
                    }
                }
            }
        } else {
            bg = "lane_emden_1_5";
        }
        ansl.set_background(bg, bg_arg);

        if (cfg.test_case == "sl_poisson_test"
            || cfg.test_case == "sl_poisson_test_boussinesq") {
            ansl.manufactured_test();
        }

        if (cfg.test_case == "gmode_exp_k") {
            // CUDA port of scripts/gmode/gmode_exp_k_chebyshev_full.py
            int N = (cfg.nr > 0 ? cfg.nr : 64);
            int ell = 1;
            int n_modes_req = 10;
            double inner_cut = 1e-4, outer_cut = 0.9999;

            // Build CGL nodes on [inner_cut, outer_cut]
            std::vector<double> x_cgl(N + 1);
            for (int k = 0; k <= N; ++k) {
                int kk = N - k;
                x_cgl[k] = inner_cut
                         + (1.0 + std::cos(M_PI * kk / (double)N))
                         * (outer_cut - inner_cut) / 2.0;
            }
            std::sort(x_cgl.begin(), x_cgl.end());

            // If ANSL_POLY3_TXT is set, load GYRE's poly3.txt and CubicSpline-
            // interpolate to CGL.  Otherwise build our own n=3 polytrope.
            StellarProfile prof;
            const char* poly_path_env = std::getenv("ANSL_POLY3_TXT");
            if (poly_path_env) {
                StellarProfile raw;
                if (!read_gyre_structure_txt(poly_path_env, raw)) {
                    std::fprintf(stderr,
                        "gmode_exp_k: failed to load %s\n", poly_path_env);
                    return 1;
                }
                std::fprintf(stderr,
                    "  Loaded GYRE structure from %s (%d rows)\n",
                    poly_path_env, raw.n_points());
                // Linear-interpolate raw → CGL (upgrade to cubic spline later)
                auto interp = [&](const std::vector<double>& y) {
                    std::vector<double> out(x_cgl.size());
                    for (size_t i = 0; i < x_cgl.size(); ++i) {
                        double xq = x_cgl[i];
                        int j = 0;
                        while (j + 1 < (int)raw.x.size() && raw.x[j + 1] < xq) ++j;
                        if (j + 1 >= (int)raw.x.size()) j = (int)raw.x.size() - 2;
                        double f = (xq - raw.x[j]) / (raw.x[j + 1] - raw.x[j]);
                        out[i] = y[j] + f * (y[j + 1] - y[j]);
                    }
                    return out;
                };
                prof.x       = x_cgl;
                prof.V_2     = interp(raw.V_2);
                prof.A_star  = interp(raw.A_star);
                prof.U       = interp(raw.U);
                prof.c_1     = interp(raw.c_1);
                prof.Gamma_1 = interp(raw.Gamma_1);
            } else {
                prof = build_polytrope_profile_at(3.0, x_cgl);
            }

            std::vector<double> omega_sq, eigvecs_y1;
            ansl.solve_gmode_full_chebyshev(
                prof.x, prof.V_2, prof.U, prof.A_star, prof.c_1, prof.Gamma_1,
                ell, n_modes_req, omega_sq, eigvecs_y1);

            // Exp K EXPECTED vs GYRE (frozen values in the Python script)
            static const double EXPECTED[10] = {
                2.5159279360877496, 1.2857077544856306, 0.7757327764772477,
                0.5177759762324133, 0.36992549567563754, 0.2775028154601723,
                0.21592664733814718, 0.1728536032702941, 0.1415440904624304,
                0.11806842352742726,
            };
            std::fprintf(stderr,
                "  Exp K CUDA:  N=%d (DOF=%d), ell=%d, polytrope n=3\n",
                N, 4 * (N + 1), ell);
            std::fprintf(stderr,
                "  %3s  %16s  %16s  %10s\n", "n_g", "ω²_CUDA", "ω²_GYRE", "rel err");
            double max_rel = 0.0;
            int n_show = std::min((int)omega_sq.size(), 10);
            for (int k = 0; k < n_show; ++k) {
                double rel = std::fabs(omega_sq[k] - EXPECTED[k]) / EXPECTED[k];
                if (rel > max_rel) max_rel = rel;
                std::fprintf(stderr,
                    "  %3d  %16.10e  %16.10e  %10.3e\n",
                    k + 1, omega_sq[k], EXPECTED[k], rel);
            }
            std::fprintf(stderr, "  Exp K max_rel = %.3e\n", max_rel);

            // CSV dump
            char csv_path[512];
            std::snprintf(csv_path, sizeof(csv_path), "%s/gmode_exp_k.csv",
                          run_dir.c_str());
            FILE* fp = std::fopen(csv_path, "w");
            std::fprintf(fp, "# N=%d, ell=%d\n", N, ell);
            std::fprintf(fp, "# omega_sq (n_modes):\n");
            for (double w : omega_sq) std::fprintf(fp, "%.15e\n", w);
            std::fclose(fp);
            std::fprintf(stderr, "  Exp K CSV at %s\n", csv_path);
        }

        if (cfg.test_case == "gmode_2d_evp") {
            int kx_int = (cfg.ps_k > 0) ? cfg.ps_k : 1;
            double kx_phys = kx_int * 2.0 * M_PI / ansl.Lx;
            int n_modes_req = 10;
            std::vector<double> omega_sq, v_modes;
            enum class EvpPath { GALERKIN, QSPACE_FOURIER, QSPACE_SL };
            EvpPath path = EvpPath::GALERKIN;
            if (const char* s = std::getenv("ANSL_EVP_BASIS")) {
                std::string ss(s);
                if (ss == "qspace" || ss == "q" || ss == "phi")
                    path = EvpPath::QSPACE_FOURIER;
                else if (ss == "qspace_sl" || ss == "sl" || ss == "q_sl")
                    path = EvpPath::QSPACE_SL;
            }
            if (path == EvpPath::QSPACE_FOURIER) {
                std::vector<double> phi_modes;
                ansl.compute_2d_gmode_evp_qspace(kx_phys, n_modes_req,
                                                 omega_sq, phi_modes);
                const int n_out = (int)omega_sq.size();
                v_modes.assign((size_t)(ansl.ny - 2) * n_out, 0.0);
                for (int k = 0; k < n_out; ++k) {
                    for (int i = 0; i < ansl.ny - 2; ++i) {
                        int row = i + 1;
                        double phi = phi_modes[(size_t)row * n_out + k];
                        v_modes[(size_t)i + (size_t)k * (ansl.ny - 2)] =
                            phi / std::max(ansl.h_rho[row], 1e-30);
                    }
                }
            } else if (path == EvpPath::QSPACE_SL) {
                std::vector<double> v_full;
                ansl.compute_2d_gmode_evp_qspace_sl(kx_phys, n_modes_req,
                                                    omega_sq, v_full);
                const int n_out = (int)omega_sq.size();
                v_modes.assign((size_t)(ansl.ny - 2) * n_out, 0.0);
                for (int k = 0; k < n_out; ++k) {
                    for (int i = 0; i < ansl.ny - 2; ++i) {
                        int row = i + 1;
                        v_modes[(size_t)i + (size_t)k * (ansl.ny - 2)] =
                            v_full[(size_t)row * n_out + k];
                    }
                }
            } else {
                ansl.compute_2d_gmode_evp(kx_phys, n_modes_req, omega_sq, v_modes);
            }
            double N2 = bg_arg;
            std::fprintf(stderr,
                "  2D g-mode EVP at k_x_int=%d  (k_x_phys=%g, N²=%g)\n",
                kx_int, kx_phys, N2);
            std::fprintf(stderr,
                "  %3s  %16s  %14s  %12s\n",
                "n", "ω²_CUDA", "ω²_analytic", "rel err");
            for (size_t n = 0; n < omega_sq.size(); ++n) {
                double ky = (double)(n + 1) * M_PI / ansl.Ly;
                double ex = N2 * kx_phys * kx_phys /
                            (kx_phys * kx_phys + ky * ky);
                double rel = std::fabs(omega_sq[n] - ex) / ex;
                std::fprintf(stderr,
                    "  %3zu  %16.10e  %14.10e  %12.3e\n",
                    n + 1, omega_sq[n], ex, rel);
            }

            // Dump ω² and eigenvectors to CSV for downstream use.
            char csv_path[512];
            std::snprintf(csv_path, sizeof(csv_path), "%s/gmode_2d_evp.csv",
                          run_dir.c_str());
            FILE* fp = std::fopen(csv_path, "w");
            std::fprintf(fp, "# ny=%d, Lx=%g, Ly=%g, kx_int=%d, N2=%g\n",
                         ansl.ny, ansl.Lx, ansl.Ly, kx_int, N2);
            std::fprintf(fp, "# omega_sq (n_modes values):\n");
            for (double w : omega_sq) std::fprintf(fp, "%.15e\n", w);
            std::fprintf(fp, "# v_modes ((ny-2) × n_modes column-major):\n");
            for (double x : v_modes) std::fprintf(fp, "%.15e\n", x);
            std::fclose(fp);
            std::fprintf(stderr, "  2D EVP written to %s\n", csv_path);
        }

        if (cfg.test_case == "gmode_pulsation") {
            // Phase 1d: Lane-Emden n=3/2 background, k_y=1 sinusoid seed.
            double amp = (cfg.perturb_amplitude > 0) ? cfg.perturb_amplitude : 1e-3;
            int k_y = (cfg.ps_k > 0) ? cfg.ps_k : 1;
            ansl.init_gmode_pulsation(amp, k_y);
            if (std::getenv("ANSL_DUMP_STEP1")) {
                // Run exactly ONE step and dump v, b at y=Ly/2 for Python compare.
                double dt1 = ansl.step();
                std::vector<double> hv, hu;
                ansl.download_uv(hu, hv);
                std::vector<double> hb;
                ansl.download_b(hb);
                int jy = ansl.ny / 2, ix = ansl.nx / 4;
                int k = jy * ansl.nx + ix;
                std::fprintf(stderr,
                    "  [STEP1 DUMP] dt=%.6e  v[jy=%d,ix=%d]=%.6e  b[]=%.6e  u[]=%.6e\n",
                    dt1, jy, ix, hv[k], hb[k], hu[k]);
                std::exit(0);
            }
            std::fprintf(stderr,
                "  AnelasticSL gmode run: tend=%g, cfl=%g, amp=%g, k_y=%d\n",
                cfg.t_end, cfg.cfl, amp, k_y);
            double t = 0.0;
            int step = 0;
            std::timespec wall_start;
            clock_gettime(CLOCK_MONOTONIC, &wall_start);

            char probe_path[512];
            std::snprintf(probe_path, sizeof(probe_path), "%s/gmode_probe.csv",
                          run_dir.c_str());
            FILE* probe = std::fopen(probe_path, "w");
            std::fprintf(probe, "# t  v_center\n");

            while (t < cfg.t_end && !g_interrupted) {
                double dt = ansl.step();
                if (t + dt > cfg.t_end) dt = cfg.t_end - t;
                t += dt;
                ++step;
                double v_c = ansl.probe_v_center();
                std::fprintf(probe, "%.10e %.10e\n", t, v_c);
                if (step % 200 == 0 || t >= cfg.t_end) {
                    print_progress(t, cfg.t_end, step, dt, wall_start);
                }
            }
            std::fclose(probe);
            std::fprintf(stderr, "  gmode probe written to %s (%d samples)\n",
                         probe_path, step);
        }

        if (cfg.test_case == "gmode_eigenmode_td") {
            // Phase 1e: exact 2D g-mode eigenmode IC → time-domain probe.
            // Background: stratified_n2 (ρ₀=1, constant N² via --ps-vshear).
            // Used to characterise the Chorin-splitting time error on a
            // clean single-frequency signal.
            double amp   = (cfg.perturb_amplitude > 0) ? cfg.perturb_amplitude : 1e-3;
            int kx_int   = (cfg.ps_k > 0) ? cfg.ps_k : 1;
            int n_g_env  = 1;
            if (const char* s = std::getenv("ANSL_NG")) n_g_env = std::atoi(s);
            if (n_g_env < 1) n_g_env = 1;

            double om2_evp = ansl.init_gmode_eigenmode(kx_int, n_g_env, amp);
            double om_evp  = std::sqrt(om2_evp);
            double T_period = 2.0 * M_PI / om_evp;
            std::fprintf(stderr,
                "  AnelasticSL eigenmode-TD run: tend=%g, cfl=%g, N²=%g, kx_int=%d, n_g=%d\n",
                cfg.t_end, cfg.cfl, bg_arg, kx_int, n_g_env);
            std::fprintf(stderr,
                "  EVP ω²=%.10e, ω=%.10e, period=%.6f (t_end covers %.2f periods)\n",
                om2_evp, om_evp, T_period, cfg.t_end / T_period);

            double t = 0.0;
            int step = 0;
            std::timespec wall_start;
            clock_gettime(CLOCK_MONOTONIC, &wall_start);

            char probe_path[512];
            std::snprintf(probe_path, sizeof(probe_path),
                          "%s/gmode_eigenmode_td.csv", run_dir.c_str());
            FILE* probe = std::fopen(probe_path, "w");
            std::fprintf(probe,
                "# kx_int=%d n_g=%d N2=%g omega_sq_evp=%.15e omega_evp=%.15e amp=%g\n",
                kx_int, n_g_env, bg_arg, om2_evp, om_evp, amp);
            std::fprintf(probe, "# t  v_center  eigmode_deviation\n");
            double v0 = ansl.probe_v_center();
            double dev0 = ansl.eigmode_deviation();
            std::fprintf(probe, "%.10e %.10e %.10e\n", 0.0, v0, dev0);

            // Path D: if ANSL_TD_KIND=assembled_linear, replace step() with
            // the linear-only assembled-matrix RK4 path
            // (docs/full_galerkin_closure_proof_2026-05-03.md).
            bool use_path_d = ansl.td_assembled_linear;
            if (use_path_d) {
                std::fprintf(stderr,
                    "  Path D linear TD: assembled L⁻¹R per kx, RK4\n");
            }
            while (t < cfg.t_end && !g_interrupted) {
                double dt = use_path_d ? ansl.step_assembled_linear()
                                       : ansl.step();
                if (t + dt > cfg.t_end) dt = cfg.t_end - t;
                t += dt;
                ++step;
                double v_c = ansl.probe_v_center();
                double dev = ansl.eigmode_deviation();
                std::fprintf(probe, "%.10e %.10e %.10e\n", t, v_c, dev);
                if (step % 200 == 0 || t >= cfg.t_end) {
                    print_progress(t, cfg.t_end, step, dt, wall_start);
                }
            }
            std::fclose(probe);
            std::fprintf(stderr,
                "  eigenmode-TD probe written to %s (%d samples)\n",
                probe_path, step);
        }

        if (cfg.test_case == "dns_triad") {
            // ─── Phase 3 / DNS Experiment A: 3-wave triad mode coupling ─────
            // (paper/DNS_PLAN.md).  Lane-Emden n=3/2 bg, eigenmode IC (n_g=1,
            // kx_int=1) at amp=1e-2, Strang-split RK4 (Path D linear block +
            // advection), 300 g-mode periods.  Per-period diagnostics:
            //   total energy E(t) = ½∫ρ(u²+v²) + ½∫b²/N²
            //   modal energy E_k(t) for k=1..4 (horizontal Fourier bands)
            //   eigmode deviation on the primary k_1 component.
            double amp   = (cfg.perturb_amplitude > 0) ? cfg.perturb_amplitude : 1e-2;
            int kx_int   = (cfg.ps_k > 0) ? cfg.ps_k : 1;
            int n_g      = 1;
            if (const char* s = std::getenv("ANSL_NG")) n_g = std::max(1, std::atoi(s));

            double om2_evp = ansl.init_gmode_eigenmode(kx_int, n_g, amp);
            double om_evp  = std::sqrt(om2_evp);
            double T_period = 2.0 * M_PI / om_evp;

            // For Strang (V, W, B) oscillator: consistent eigenmode IC is
            // V = V_EVP·sin(kx·x), W = 0, B = 0 (cosine-phase oscillator peak).
            // init_gmode_eigenmode fills d_b with the off-phase b = -(N²/ω²)·V
            // which is wrong for our (v, w, b) RK4 dynamics — zero it out.
            CUDA_CHECK(cudaMemset(ansl.d_b, 0, sizeof(double) * ansl.ncell));

            // Period count and fixed dt aligned to periods.
            int n_periods = 300;
            if (const char* s = std::getenv("ANSL_DNS_PERIODS")) {
                int v = std::atoi(s); if (v > 0) n_periods = v;
            }
            int steps_per_period = 32;
            if (const char* s = std::getenv("ANSL_DNS_SPP")) {
                int v = std::atoi(s); if (v > 0) steps_per_period = v;
            }
            double dt = T_period / (double)steps_per_period;
            int n_steps_total = n_periods * steps_per_period;

            std::fprintf(stderr,
                "  DNS Experiment A (triad):  bg=%s, ny=%d, nx=%d, kx_int=%d, n_g=%d\n",
                bg.c_str(), cfg.nr, cfg.ntheta, kx_int, n_g);
            std::fprintf(stderr,
                "  amp=%g, ω=%.6f, period=%.6f, dt=%.6f (%d steps/period), %d periods\n",
                amp, om_evp, T_period, dt, steps_per_period, n_periods);

            char probe_path[512];
            std::snprintf(probe_path, sizeof(probe_path),
                          "%s/dns_triad.csv", run_dir.c_str());
            FILE* probe = std::fopen(probe_path, "w");
            std::fprintf(probe,
                "# kx_int=%d n_g=%d amp=%g omega=%.15e period=%.15e dt=%.15e steps_per_period=%d\n",
                kx_int, n_g, amp, om_evp, T_period, dt, steps_per_period);
            std::fprintf(probe,
                "# t  v_center  eigmode_dev  E_total  E_k1  E_k2  E_k3  E_k4  max_abs_v\n");

            // Grab weights and grid for host-side diagnostics.
            std::vector<double> y_cgl, w_cc(cfg.nr, 0.0);
            ansl.download_y(y_cgl);
            {
                // Reproduce Clenshaw-Curtis weights on [0, Ly] (same recipe as
                // Python scripts/spectral/nonlinear_paths_infra.py).  Ly from config.
                int N = cfg.nr - 1;
                std::vector<double> w(N + 1, 0.0);
                for (int k = 0; k <= N; ++k) {
                    double s = 0.0;
                    int J = N / 2;
                    for (int j = 1; j <= J; ++j) {
                        double b = (2 * j != N) ? 2.0 : 1.0;
                        s += b / (4.0 * j * j - 1) *
                             std::cos(2.0 * j * k * M_PI / N);
                    }
                    w[k] = (1.0 - s) * 2.0 / (double)N;
                }
                w[0] /= 2.0; w[N] /= 2.0;
                for (int k = 0; k <= N; ++k) w_cc[k] = w[N - k] * cfg.ps_Ly / 2.0;
            }

            // Initial u is built in init_gmode_eigenmode.  Download initial state.
            auto diagnostics = [&](double t_now) {
                std::vector<double> h_u, h_v, h_b;
                ansl.rebuild_u_from_continuity();
                ansl.download_uv(h_u, h_v);
                ansl.download_b(h_b);

                const int ny = cfg.nr;
                const int nx = cfg.ntheta;

                // Background N²(y) and ρ(y) host-side: we need them for E_total.
                // AnelasticSLSolver exposes h_N2 and h_rho internally.  We hack
                // around by requesting them from the solver's diagnostic path.
                // (They're on host as ansl.h_N2, ansl.h_rho.)

                double E_kin = 0.0, E_pot = 0.0;
                for (int jy = 0; jy < ny; ++jy) {
                    double w = w_cc[jy];
                    double rho_y = ansl.h_rho[jy];
                    double N2_y  = ansl.h_N2[jy];
                    double KE_row = 0.0, PE_row = 0.0;
                    for (int ix = 0; ix < nx; ++ix) {
                        int k = jy * nx + ix;
                        double U = h_u[k], V = h_v[k], B = h_b[k];
                        KE_row += U * U + V * V;
                        if (N2_y > 1e-12) PE_row += B * B / N2_y;
                    }
                    E_kin += 0.5 * w * rho_y * KE_row / (double)nx;
                    E_pot += 0.5 * w               * PE_row / (double)nx;
                }
                double E_total = E_kin + E_pot;

                // Modal energy E_k for k=1..4:  integrate |v̂_k(y)|² + |û_k(y)|²
                // over y with ρ weight; ignore b contribution for compactness.
                // Use naive DFT at integer mode k (nx typically small).
                double Emk[5] = {0,0,0,0,0};
                for (int kmode = 1; kmode <= 4; ++kmode) {
                    double cumE = 0.0;
                    for (int jy = 0; jy < ny; ++jy) {
                        double w   = w_cc[jy];
                        double rho_y = ansl.h_rho[jy];
                        double vr = 0.0, vi = 0.0, ur = 0.0, ui = 0.0;
                        for (int ix = 0; ix < nx; ++ix) {
                            double x = (double)ix * (cfg.ps_Lx / (double)nx);
                            double ph = 2.0 * M_PI * (double)kmode * x / cfg.ps_Lx;
                            double c = std::cos(ph), s = std::sin(ph);
                            double V = h_v[jy * nx + ix];
                            double U = h_u[jy * nx + ix];
                            vr += V * c; vi -= V * s;
                            ur += U * c; ui -= U * s;
                        }
                        double inv_nx = 1.0 / (double)nx;
                        vr *= inv_nx; vi *= inv_nx;
                        ur *= inv_nx; ui *= inv_nx;
                        // single-sided mode has magnitude squared coefficient 2:
                        cumE += w * rho_y * 2.0 *
                                (vr * vr + vi * vi + ur * ur + ui * ui);
                    }
                    Emk[kmode] = 0.5 * cumE;
                }

                double v_c = ansl.probe_v_center();
                double dev = ansl.eigmode_deviation();
                double max_v = 0.0;
                for (double x : h_v) if (std::fabs(x) > max_v) max_v = std::fabs(x);
                std::fprintf(probe,
                    "%.10e %.10e %.10e %.15e %.15e %.15e %.15e %.15e %.10e\n",
                    t_now, v_c, dev, E_total,
                    Emk[1], Emk[2], Emk[3], Emk[4], max_v);
            };

            // Sample at t=0.
            diagnostics(0.0);

            std::timespec wall_start;
            clock_gettime(CLOCK_MONOTONIC, &wall_start);

            double t_now = 0.0;
            int samples = 1;
            for (int p = 0; p < n_periods && !g_interrupted; ++p) {
                for (int k = 0; k < steps_per_period && !g_interrupted; ++k) {
                    ansl.step_strang_nonlinear(dt);
                    t_now += dt;
                }
                diagnostics(t_now);
                ++samples;
                if ((p + 1) % 10 == 0 || p == n_periods - 1) {
                    print_progress(t_now, (double)n_periods * T_period,
                                   (p + 1) * steps_per_period, dt, wall_start);
                }
            }
            std::fclose(probe);
            std::fprintf(stderr,
                "  DNS triad probe written to %s (%d samples)\n",
                probe_path, samples);
        }

        if (cfg.test_case == "dns_triad_coupled") {
            // ─── Experiment E1: true three-wave resonant triad ──────────────
            // Seeds TWO eigenmodes (a, b) and watches the resonance partner
            // (c = a + b in wavenumber) grow from numerical noise.
            //
            // Default mode selection (from scripts/spectral/scan_resonance.py on
            // Lane-Emden n=3/2, rho_cut=0.1, Ny=128):
            //   a = (n_g=4, kx=1, ω≈0.6402)
            //   b = (n_g=3, kx=2, ω≈1.1751)
            //   c = (n_g=1, kx=3, ω≈1.8556)        ← should grow ≈ t²
            //   detune ≈ 2.2%, |V_abc| = 0.92 (strong overlap)
            //
            // Overridable via ANSL_NG_A, ANSL_KX_A, ANSL_AMP_A (same for B).
            std::vector<AnelasticSLSolver::ModeSpec> modes;
            auto env_int = [](const char* key, int fallback) {
                if (const char* s = std::getenv(key)) {
                    int v = std::atoi(s); if (v != 0) return v;
                }
                return fallback;
            };
            auto env_dbl = [](const char* key, double fallback) {
                if (const char* s = std::getenv(key)) {
                    double v = std::atof(s); if (v != 0.0) return v;
                }
                return fallback;
            };
            double amp_a = env_dbl("ANSL_AMP_A", cfg.perturb_amplitude > 0
                                   ? cfg.perturb_amplitude : 1e-3);
            double amp_b = env_dbl("ANSL_AMP_B", amp_a);
            int n_g_a = env_int("ANSL_NG_A", 4);
            int n_g_b = env_int("ANSL_NG_B", 3);
            int kx_a  = env_int("ANSL_KX_A", 1);
            int kx_b  = env_int("ANSL_KX_B", 2);
            // Phase choice: put mode a on sin(kx_a x) and mode b on cos(kx_b x)
            // so they are not coherent at t=0 when their x-wavenumbers align
            // (here kx_a != kx_b so both can be sin-phase).
            modes.push_back({n_g_a, kx_a, amp_a, false});
            modes.push_back({n_g_b, kx_b, amp_b, false});

            double om_a = ansl.init_multi_mode_ic(modes);
            double T_period = 2.0 * M_PI / om_a;

            int n_periods = env_int("ANSL_DNS_PERIODS", 200);
            int steps_per_period = env_int("ANSL_DNS_SPP", 32);
            double dt = T_period / (double)steps_per_period;

            // Expected c mode: kx_c = kx_a + kx_b, we only display it.
            int kx_c = kx_a + kx_b;
            std::fprintf(stderr,
                "  DNS E1 (three-wave triad):  (a,b)→c =  "
                "(%d,kx=%d) + (%d,kx=%d) → (?,kx=%d)\n",
                n_g_a, kx_a, n_g_b, kx_b, kx_c);
            std::fprintf(stderr,
                "  amp_a=%g, amp_b=%g, ω_a=%.6f (period=%.4f), "
                "dt=%.4f (%d spp), %d periods\n",
                amp_a, amp_b, om_a, T_period, dt, steps_per_period, n_periods);

            char probe_path[512];
            std::snprintf(probe_path, sizeof(probe_path),
                          "%s/dns_triad_coupled.csv", run_dir.c_str());
            FILE* probe = std::fopen(probe_path, "w");
            std::fprintf(probe,
                "# triad: a=(n%d,kx%d,amp%g)  b=(n%d,kx%d,amp%g)  c=(?,kx%d)\n",
                n_g_a, kx_a, amp_a, n_g_b, kx_b, amp_b, kx_c);
            std::fprintf(probe,
                "# omega_a=%.15e period_a=%.15e dt=%.15e spp=%d\n",
                om_a, T_period, dt, steps_per_period);
            std::fprintf(probe,
                "# t  max_v  E_kin_total  E_k1  E_k2  E_k3  E_k4  E_k5  E_k6  E_pot\n");

            // Host-side CC weights (reuse recipe from dns_triad).
            std::vector<double> y_cgl, w_cc(cfg.nr, 0.0);
            ansl.download_y(y_cgl);
            {
                int N = cfg.nr - 1;
                std::vector<double> w(N + 1, 0.0);
                for (int k = 0; k <= N; ++k) {
                    double s = 0.0;
                    int J = N / 2;
                    for (int j = 1; j <= J; ++j) {
                        double b = (2 * j != N) ? 2.0 : 1.0;
                        s += b / (4.0 * j * j - 1) *
                             std::cos(2.0 * j * k * M_PI / N);
                    }
                    w[k] = (1.0 - s) * 2.0 / (double)N;
                }
                w[0] /= 2.0; w[N] /= 2.0;
                for (int k = 0; k <= N; ++k) w_cc[k] = w[N - k] * cfg.ps_Ly / 2.0;
            }

            // Snapshot cadence (periods).  ANSL_DNS_SNAP_EVERY=N writes binary
            // (u, v, b) float32 cubes to <run_dir>/snapshots/snap_NNNN.bin
            // after every N periods; 0 disables snapshotting.
            int snap_every = env_int("ANSL_DNS_SNAP_EVERY", 0);
            std::string snap_dir = run_dir + "/snapshots";
            if (snap_every > 0) {
                std::string cmd = "mkdir -p '" + snap_dir + "'";
                (void)std::system(cmd.c_str());
            }
            int snap_idx = 0;

            auto diagnostics = [&](double t_now) {
                std::vector<double> h_u, h_v, h_b;
                ansl.rebuild_u_from_continuity();
                ansl.download_uv(h_u, h_v);
                ansl.download_b(h_b);
                const int ny = cfg.nr;
                const int nx = cfg.ntheta;

                // Total energy (anelastic functional).
                double E_kin = 0.0, E_pot = 0.0;
                double max_v = 0.0;
                for (int jy = 0; jy < ny; ++jy) {
                    double w = w_cc[jy];
                    double rho_y = ansl.h_rho[jy];
                    double N2_y  = ansl.h_N2[jy];
                    double KE_row = 0.0, PE_row = 0.0;
                    for (int ix = 0; ix < nx; ++ix) {
                        int k = jy * nx + ix;
                        double U = h_u[k], V = h_v[k], B = h_b[k];
                        KE_row += U * U + V * V;
                        if (N2_y > 1e-12) PE_row += B * B / N2_y;
                        if (std::fabs(V) > max_v) max_v = std::fabs(V);
                    }
                    E_kin += 0.5 * w * rho_y * KE_row / (double)nx;
                    E_pot += 0.5 * w               * PE_row / (double)nx;
                }
                double E_total = E_kin + E_pot;

                // Modal energy E_k for k=1..6 using naive DFT per jy.
                const int KMAX = 6;
                std::vector<double> Emk(KMAX + 1, 0.0);
                for (int kmode = 1; kmode <= KMAX; ++kmode) {
                    double cumE = 0.0;
                    for (int jy = 0; jy < ny; ++jy) {
                        double w   = w_cc[jy];
                        double rho_y = ansl.h_rho[jy];
                        double vr=0,vi=0,ur=0,ui=0;
                        for (int ix = 0; ix < nx; ++ix) {
                            double x = (double)ix * (cfg.ps_Lx / (double)nx);
                            double ph = 2.0 * M_PI * (double)kmode * x / cfg.ps_Lx;
                            double c = std::cos(ph), s = std::sin(ph);
                            double V = h_v[jy * nx + ix];
                            double U = h_u[jy * nx + ix];
                            vr += V * c; vi -= V * s;
                            ur += U * c; ui -= U * s;
                        }
                        double inv_nx = 1.0 / (double)nx;
                        vr *= inv_nx; vi *= inv_nx;
                        ur *= inv_nx; ui *= inv_nx;
                        cumE += w * rho_y * 2.0 *
                                (vr*vr + vi*vi + ur*ur + ui*ui);
                    }
                    Emk[kmode] = 0.5 * cumE;
                }

                // Kinetic-only modal sum as conservation check:  E_kin_total
                // is immune to the 1/N² surface-amplification of ∫ b²/N² that
                // dominates E_total on Lane-Emden.
                double E_kin_total = 0.0;
                for (int kmode = 1; kmode <= KMAX; ++kmode) E_kin_total += Emk[kmode];
                std::fprintf(probe,
                    "%.10e %.10e %.15e "
                    "%.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                    t_now, max_v, E_kin_total,
                    Emk[1], Emk[2], Emk[3], Emk[4], Emk[5], Emk[6], E_pot);
                std::fflush(probe);

                // Snapshot dump for visualisation.
                if (snap_every > 0 && (snap_idx % snap_every == 0)) {
                    char snap_path[768];
                    std::snprintf(snap_path, sizeof(snap_path),
                                  "%s/snap_%04d.bin", snap_dir.c_str(), snap_idx);
                    FILE* sf = std::fopen(snap_path, "wb");
                    if (sf) {
                        // Header: ny, nx, t (double).  Payload: u, v, b as
                        // float32 row-major (ny × nx), one field after the
                        // other.  Python np.fromfile / np.memmap can read this.
                        int32_t hdr[2] = { (int32_t)ny, (int32_t)nx };
                        std::fwrite(hdr, sizeof(int32_t), 2, sf);
                        std::fwrite(&t_now, sizeof(double), 1, sf);
                        int nc = ny * nx;
                        std::vector<float> buf(nc);
                        for (int i = 0; i < nc; ++i) buf[i] = (float)h_u[i];
                        std::fwrite(buf.data(), sizeof(float), nc, sf);
                        for (int i = 0; i < nc; ++i) buf[i] = (float)h_v[i];
                        std::fwrite(buf.data(), sizeof(float), nc, sf);
                        for (int i = 0; i < nc; ++i) buf[i] = (float)h_b[i];
                        std::fwrite(buf.data(), sizeof(float), nc, sf);
                        std::fclose(sf);
                    }
                }
                ++snap_idx;
            };

            diagnostics(0.0);

            std::timespec wall_start;
            clock_gettime(CLOCK_MONOTONIC, &wall_start);

            double t_now = 0.0;
            int samples = 1;
            for (int p = 0; p < n_periods && !g_interrupted; ++p) {
                for (int k = 0; k < steps_per_period && !g_interrupted; ++k) {
                    ansl.step_strang_nonlinear(dt);
                    t_now += dt;
                }
                diagnostics(t_now);
                ++samples;
                if ((p + 1) % 10 == 0 || p == n_periods - 1) {
                    print_progress(t_now, (double)n_periods * T_period,
                                   (p + 1) * steps_per_period, dt, wall_start);
                }
            }
            std::fclose(probe);
            std::fprintf(stderr,
                "  DNS triad-coupled probe written to %s (%d samples)\n",
                probe_path, samples);
        }

        if (cfg.test_case == "kh_shear_boussinesq") {
            double amp = (cfg.perturb_amplitude > 0) ? cfg.perturb_amplitude : 1e-2;
            ansl.init_kh_shear(cfg.ps_vshear, amp, cfg.ps_k);
            std::fprintf(stderr,
                "  AnelasticSL KH run: tend=%g, cfl=%g, ν=%g, k=%d\n",
                cfg.t_end, cfg.cfl, cfg.ps_nu, cfg.ps_k);
            double t = 0.0;
            int step = 0;
            std::timespec wall_start;
            clock_gettime(CLOCK_MONOTONIC, &wall_start);
            while (t < cfg.t_end && !g_interrupted) {
                double dt = ansl.step();
                if (t + dt > cfg.t_end) dt = cfg.t_end - t;
                t += dt;
                ++step;
                if (step % 50 == 0 || t >= cfg.t_end) {
                    print_progress(t, cfg.t_end, step, dt, wall_start);
                }
            }
            // Dump final ω(x,y) to CSV for Python comparison.
            std::vector<double> h_omega;
            ansl.download_omega(h_omega);
            std::vector<double> h_u, h_v;
            ansl.download_uv(h_u, h_v);
            std::vector<double> y_cgl;
            ansl.download_y(y_cgl);
            std::vector<double> h_div;
            ansl.download_divergence(h_div);
            double max_div = 0.0;
            for (double d : h_div) if (std::fabs(d) > max_div) max_div = std::fabs(d);
            std::fprintf(stderr, "  final |∇·u|∞ (C++ side) = %.3e\n", max_div);
            char csv_path[512];
            std::snprintf(csv_path, sizeof(csv_path), "%s/kh_final.csv", run_dir.c_str());
            FILE* fp = std::fopen(csv_path, "w");
            std::fprintf(fp, "# nx=%d, ny=%d, Lx=%g, Ly=%g, nu=%g, t=%g, steps=%d\n",
                         ansl.nx, ansl.ny, ansl.Lx, ansl.Ly, ansl.nu, t, step);
            std::fprintf(fp, "# y_cgl (ny values):\n");
            for (int i = 0; i < ansl.ny; ++i)
                std::fprintf(fp, "%.15e\n", y_cgl[i]);
            std::fprintf(fp, "# omega (ny × nx row-major):\n");
            for (int i = 0; i < ansl.ncell; ++i)
                std::fprintf(fp, "%.15e\n", h_omega[i]);
            std::fprintf(fp, "# u (ny × nx row-major):\n");
            for (int i = 0; i < ansl.ncell; ++i)
                std::fprintf(fp, "%.15e\n", h_u[i]);
            std::fprintf(fp, "# v (ny × nx row-major):\n");
            for (int i = 0; i < ansl.ncell; ++i)
                std::fprintf(fp, "%.15e\n", h_v[i]);
            std::fclose(fp);
            std::fprintf(stderr, "  KH final state written to %s\n", csv_path);
        }

        // Phase 1a: dump SL basis to CSV for offline verification.
        char basis_path[512];
        std::snprintf(basis_path, sizeof(basis_path), "%s/sl_basis.csv", run_dir.c_str());
        FILE* bf = std::fopen(basis_path, "w");
        std::fprintf(bf, "# ny=%d, n_modes=%d, Ly=%g, background=%s\n",
                     ansl.ny, ansl.n_modes, ansl.Ly, bg.c_str());
        std::fprintf(bf, "# mu_n (n_modes values):\n");
        for (int m = 0; m < ansl.n_modes; ++m)
            std::fprintf(bf, "%.15e\n", ansl.h_mu[m]);
        std::fprintf(bf, "# y_cgl (ny values):\n");
        for (int i = 0; i < ansl.ny; ++i)
            std::fprintf(bf, "%.15e\n", ansl.h_y_cgl[i]);
        std::fprintf(bf, "# W_tilde (ny values):\n");
        for (int i = 0; i < ansl.ny; ++i)
            std::fprintf(bf, "%.15e\n", ansl.h_W_tilde[i]);
        std::fclose(bf);
        std::fprintf(stderr, "  SL basis written to %s\n", basis_path);
        std::fprintf(stderr, "\n");
    } else if (cfg.solver_type == "sph2d_spectral") {
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
        std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", run_dir.c_str());
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
            std::snprintf(path, sizeof(path), "%s/output_%04d.vtk", run_dir.c_str(), ++frame);
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
                              run_dir.c_str());
                sph.save_checkpoint(cpath, t);
            }
            if (do_vtk) {
                if (cfg.frame_buffer && sph.frame_capacity > 0) {
                    if (sph.frame_count >= sph.frame_capacity)
                        sph.flush_frames_to_disk(run_dir);
                    sph.capture_frame(t, step);
                } else {
                    ++frame;
                    char path[512];
                    std::snprintf(path, sizeof(path), "%s/output_%04d.vtk",
                                  run_dir.c_str(), frame);
                    sph.write_vtk_2d(path);
                }
            }
        }
        if (cfg.frame_buffer) sph.flush_frames_to_disk(run_dir);
        std::fclose(csv);
        std::fprintf(stderr, "\n");
        sph.destroy();
    } else if (cfg.solver_type == "ale2d") {
        if (int rc = run_ale2d(cfg, ctx, t, step); rc != 0) return rc;
    } else if (cfg.solver_type == "wb2d") {
        if (int rc = run_wb2d(cfg, ctx, t, step); rc != 0) return rc;
    } else if (cfg.solver_type == "lowmach") {
        if (int rc = run_lowmach(cfg, ctx, t, step); rc != 0) return rc;
    } else {
#ifdef USE_AMGX
        if (int rc = run_compressible(cfg, ctx, t, step); rc != 0) return rc;
#else
        std::fprintf(stderr, "ERROR: --solver compressible requires AmgX. "
                     "Rebuild with -DAMGX_DIR=/path/to/amgx, or use --solver lowmach.\n");
        return 1;
#endif
    }


#else
    // ===== CPU path =====
    State state_tmp;
    state_tmp.allocate(grid);

    PoissonGMG poisson_solver;
    poisson_solver.init(grid);

    FluxAccumulator acc;
    acc.allocate(grid.total_cells());

    std::vector<double> rho_cells;
    std::vector<double> poisson_rhs;

    std::timespec wall_start;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);

    while (t < cfg.t_end && !g_interrupted) {
        fill_ghost_cells(grid, state, cfg.gamma);

        double dt = compute_cfl_dt(grid, state, eos, cfg.cfl);
        if (t + dt > cfg.t_end) dt = cfg.t_end - t;

        state_tmp.copy_from(state);

        compute_rhs(grid, state, eos, acc, cfg.limiter);
        extract_density(grid, state, rho_cells);
        compute_poisson_rhs(grid, rho_cells, cfg.G, poisson_rhs);
        poisson_solver.solve(poisson_rhs.data(), state.phi.data());
        add_gravity_source(grid, state, acc);
        rk2_substep(grid, state, acc, dt);

        fill_ghost_cells(grid, state, cfg.gamma);
        compute_rhs(grid, state, eos, acc, cfg.limiter);
        extract_density(grid, state, rho_cells);
        compute_poisson_rhs(grid, rho_cells, cfg.G, poisson_rhs);
        poisson_solver.solve(poisson_rhs.data(), state.phi.data());
        add_gravity_source(grid, state, acc);
        rk2_substep(grid, state, acc, dt);

        int nr = grid.nr, nt = grid.ntheta;
        for (int i = 0; i < nr; ++i) {
            for (int j = 0; j < nt; ++j) {
                int k = grid.idx(i, j);
                state.rho[k] = 0.5 * (state_tmp.rho[k] + state.rho[k]);
                state.mr[k] = 0.5 * (state_tmp.mr[k] + state.mr[k]);
                state.mtheta[k] = 0.5 * (state_tmp.mtheta[k] + state.mtheta[k]);
                state.E[k] = 0.5 * (state_tmp.E[k] + state.E[k]);
            }
        }

        t += dt;
        step++;

        if (step % 200 == 0)
            print_progress(t, cfg.t_end, step, dt, wall_start);

        if (step % cfg.output_interval == 0) {
            std::fprintf(stderr, "\n");
            Diagnostics diag = compute_diagnostics(grid, state, cfg.gamma);
            std::printf("Step %6d  t = %.6e  dt = %.3e  M = %.10e  E = %.10e\n",
                        step, t, dt, diag.total_mass, diag.total_energy);

            char fname[512];
            std::snprintf(fname, sizeof(fname), "%s/output_%04d.vtk", run_dir.c_str(), step / cfg.output_interval);
            write_vtk(fname, grid, state, cfg.gamma);
        }
    }
    std::fprintf(stderr, "\n");
#endif

    Diagnostics diag = compute_diagnostics(grid, state, cfg.gamma);
    std::printf("Final: step %d  t = %.6e  M = %.10e  E = %.10e\n",
                step, t, diag.total_mass, diag.total_energy);
    {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/output_final.vtk", run_dir.c_str());
        write_vtk(path, grid, state, cfg.gamma);
    }

#ifdef USE_GPU
    ctx.destroy_tables();
#endif

    std::printf("Done.\n");
    return 0;
}
