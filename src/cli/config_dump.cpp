#include "cli/config_dump.h"
#include "cli/options.h"

#include <cstdio>
#include <string>

#ifndef STELLAR2D_GIT_HASH
#define STELLAR2D_GIT_HASH "unknown"
#endif

#ifndef STELLAR2D_BUILD_DATE
#define STELLAR2D_BUILD_DATE "unknown"
#endif

namespace {

const char* limiter_name(Limiter l) {
    switch (l) {
        case Limiter::MINMOD:   return "minmod";
        case Limiter::VAN_LEER: return "vanleer";
        case Limiter::MC:       return "mc";
    }
    return "?";
}

const char* yn(bool b) { return b ? "true" : "false"; }

} // namespace

void dump_resolved_cli(const SimConfig& cfg, const std::string& run_dir) {
    const std::string path = run_dir + "/config.dump.txt";
    FILE* fp = std::fopen(path.c_str(), "w");
    if (!fp) {
        std::fprintf(stderr,
                     "[config_dump] WARNING: cannot open %s for writing — "
                     "reproducibility dump skipped.\n", path.c_str());
        return;
    }

    // ---- Header --------------------------------------------------------
    std::fprintf(fp,
        "# stellar2d resolved CLI configuration\n"
        "# Produced by Tier A config_dump (plain key=value form).\n"
        "# Tier B will upgrade this to a TOML round-tripable form at\n"
        "# <run>/config.toml, reloadable with `stellar2d --config <path>`.\n"
        "# See docs/design/cli_unification_plan_2026-05-09.md.\n"
        "#\n"
        "# stellar2d version: %s\n"
        "# build date:        %s\n"
        "\n",
        STELLAR2D_GIT_HASH, STELLAR2D_BUILD_DATE);

    // ---- Core ----------------------------------------------------------
    std::fprintf(fp, "# --- core ---\n");
    std::fprintf(fp, "solver                   = %s\n",    cfg.solver_type.c_str());
    std::fprintf(fp, "test                     = %s\n",    cfg.test_case.c_str());
    std::fprintf(fp, "nr                       = %d\n",    cfg.nr);
    std::fprintf(fp, "ntheta                   = %d\n",    cfg.ntheta);
    std::fprintf(fp, "R_outer                  = %.17g\n", cfg.R_outer);
    std::fprintf(fp, "gamma                    = %.17g\n", cfg.gamma);
    std::fprintf(fp, "cfl                      = %.17g\n", cfg.cfl);
    std::fprintf(fp, "t_end                    = %.17g\n", cfg.t_end);
    std::fprintf(fp, "G                        = %.17g\n", cfg.G);
    std::fprintf(fp, "mesh                     = %s\n",    cfg.mesh_type.c_str());
    std::fprintf(fp, "limiter                  = %s\n",    limiter_name(cfg.limiter));
    std::fprintf(fp, "perturb                  = %.17g\n", cfg.perturb_amplitude);
    std::fprintf(fp, "run_base                 = %s\n",    cfg.run_base.c_str());

    // ---- IO ------------------------------------------------------------
    std::fprintf(fp, "\n# --- io ---\n");
    std::fprintf(fp, "output_interval          = %d\n",    cfg.output_interval);
    std::fprintf(fp, "vtk_interval             = %d\n",    cfg.vtk_interval);
    std::fprintf(fp, "vtk_dt                   = %.17g\n", cfg.vtk_dt);
    std::fprintf(fp, "diag_interval            = %d\n",    cfg.diag_interval);
    std::fprintf(fp, "frame_buffer             = %s\n",    yn(cfg.frame_buffer));
    std::fprintf(fp, "frame_headroom_mb        = %d\n",    cfg.frame_headroom_mb);
    std::fprintf(fp, "compute_error            = %s\n",    yn(cfg.compute_error));

    // ---- EOS / physics -------------------------------------------------
    std::fprintf(fp, "\n# --- eos / physics ---\n");
    std::fprintf(fp, "eos                      = %s\n",    cfg.eos_type.c_str());
    std::fprintf(fp, "eos_mu                   = %.17g\n", cfg.eos_mu);
    std::fprintf(fp, "eos_rad_a                = %.17g\n", cfg.eos_rad_a);
    std::fprintf(fp, "helm_table               = %s\n",    cfg.helm_table_path.c_str());
    std::fprintf(fp, "helm_Abar                = %.17g\n", cfg.helm_Abar);
    std::fprintf(fp, "helm_Zbar                = %.17g\n", cfg.helm_Zbar);
    std::fprintf(fp, "radiation                = %s\n",    yn(cfg.radiation_enabled));
    std::fprintf(fp, "rad_c_light              = %.17g\n", cfg.rad_c_light);
    std::fprintf(fp, "rad_T_phot_floor         = %.17g\n", cfg.rad_T_phot_floor);
    std::fprintf(fp, "nuclear                  = %s\n",    yn(cfg.nuclear_enabled));
    std::fprintf(fp, "nuc_X                    = %.17g\n", cfg.nuc_X);
    std::fprintf(fp, "nuc_Y                    = %.17g\n", cfg.nuc_Y);
    std::fprintf(fp, "nuc_epsilon_scale        = %.17g\n", cfg.nuc_epsilon_scale);
    std::fprintf(fp, "nuc_q_burn               = %.17g\n", cfg.nuc_q_burn);
    std::fprintf(fp, "nuc_T_floor              = %.17g\n", cfg.nuc_T_floor);
    std::fprintf(fp, "nuc_T_scale              = %.17g\n", cfg.nuc_T_scale);
    std::fprintf(fp, "nuc_compress_frac        = %.17g\n", cfg.nuc_compress_frac);
    std::fprintf(fp, "species                  = %s\n",    yn(cfg.species_enabled));
    std::fprintf(fp, "mlt                      = %s\n",    yn(cfg.mlt_enabled));
    std::fprintf(fp, "mlt_alpha                = %.17g\n", cfg.mlt_alpha);
    std::fprintf(fp, "kap_use_table            = %s\n",    yn(cfg.kap_use_table));
    std::fprintf(fp, "kap_highT_family         = %s\n",    cfg.kap_highT_family.c_str());
    std::fprintf(fp, "kap_lowT_family          = %s\n",    cfg.kap_lowT_family.c_str());
    std::fprintf(fp, "kap_table_Z              = %.17g\n", cfg.kap_table_Z);
    std::fprintf(fp, "kap_data_dir             = %s\n",    cfg.kap_data_dir.c_str());
    std::fprintf(fp, "kap_logT_lo_end          = %.17g\n", cfg.kap_logT_lo_end);
    std::fprintf(fp, "kap_logT_hi_start        = %.17g\n", cfg.kap_logT_hi_start);

    // ---- IC ------------------------------------------------------------
    std::fprintf(fp, "\n# --- initial conditions ---\n");
    std::fprintf(fp, "ic_solar                 = %s\n",    yn(cfg.ic_solar));
    std::fprintf(fp, "ic_rho_c                 = %.17g\n", cfg.ic_rho_c);
    std::fprintf(fp, "ic_R_star                = %.17g\n", cfg.ic_R_star);
    std::fprintf(fp, "ic_n_poly                = %.17g\n", cfg.ic_n_poly);
    std::fprintf(fp, "ic_mesa_path             = %s\n",    cfg.ic_mesa_path.c_str());
    std::fprintf(fp, "ic_mesa_seed_T           = %s\n",    yn(cfg.ic_mesa_seed_T));
    std::fprintf(fp, "ic_mesa_atm_zones        = %d\n",    cfg.ic_mesa_atm_zones);
    std::fprintf(fp, "atm_split                = %d\n",    cfg.atm_split);
    std::fprintf(fp, "rich_profile             = %s\n",    yn(cfg.rich_profile));

    // ---- bubble IC (shared by hse_bubble / cart_ale / cart_ale2) -------
    std::fprintf(fp, "\n# --- bubble IC ---\n");
    std::fprintf(fp, "bubble_mode              = %s\n",    cfg.bubble_mode.c_str());
    std::fprintf(fp, "bubble_xc                = %.17g\n", cfg.bubble_xc);
    std::fprintf(fp, "bubble_yc                = %.17g\n", cfg.bubble_yc);
    std::fprintf(fp, "bubble_rb                = %.17g\n", cfg.bubble_rb);
    std::fprintf(fp, "bubble_alpha             = %.17g\n", cfg.bubble_alpha);
    std::fprintf(fp, "bubble_beta              = %.17g\n", cfg.bubble_beta);
    for (size_t bi = 0; bi < cfg.bubbles.size(); ++bi) {
        const auto& b = cfg.bubbles[bi];
        std::fprintf(fp,
            "bubble_multi[%zu]          = %.17g, %.17g, %.17g, %.17g, %.17g\n",
            bi, b[0], b[1], b[2], b[3], b[4]);
    }

    // ---- BC ------------------------------------------------------------
    std::fprintf(fp, "\n# --- bc ---\n");
    std::fprintf(fp, "no_sponge                = %s\n",    yn(cfg.no_sponge));
    std::fprintf(fp, "radial_only              = %s\n",    yn(cfg.radial_only));
    std::fprintf(fp, "r_inner                  = %.17g\n", cfg.r_inner);
    std::fprintf(fp, "M_core                   = %.17g\n", cfg.M_core);

    // ---- solver.radial1d ----------------------------------------------
    std::fprintf(fp, "\n# --- solver.radial1d ---\n");
    std::fprintf(fp, "implicit                 = %s\n",    yn(cfg.implicit_mode));
    std::fprintf(fp, "dt_implicit              = %.17g\n", cfg.dt_implicit);
    std::fprintf(fp, "dt_implicit_scale        = %.17g\n", cfg.dt_implicit_scale);
    std::fprintf(fp, "no_viallet               = %s\n",    yn(cfg.no_viallet));
    std::fprintf(fp, "precond_tridiag          = %s\n",    yn(cfg.precond_tridiag));
    std::fprintf(fp, "jfnk_autodiff            = %s\n",    yn(cfg.jfnk_autodiff));
    std::fprintf(fp, "no_rhse_subtract         = %s\n",    yn(cfg.no_rhse_subtract));
    std::fprintf(fp, "newton_tol               = %.17g\n", cfg.newton_tol_override);
    std::fprintf(fp, "hse_resnap               = %d\n",    cfg.hse_resnap_interval);
    std::fprintf(fp, "dt_thermal_frac          = %.17g\n", cfg.dt_thermal_frac);
    std::fprintf(fp, "dt_mach_cap              = %.17g\n", cfg.dt_mach_cap);

    // ---- solver.lowmach ------------------------------------------------
    std::fprintf(fp, "\n# --- solver.lowmach ---\n");
    std::fprintf(fp, "precond                  = %s\n",    cfg.precond.c_str());
    std::fprintf(fp, "hllc_variant             = %d\n",    cfg.hllc_variant);

    // ---- solver.cart_ale / cart_ale2 -----------------------------------
    std::fprintf(fp, "\n# --- solver.cart_ale / cart_ale2 ---\n");
    std::fprintf(fp, "cart_ale_remap_order     = %d\n",    cfg.cart_ale_remap_order);
    std::fprintf(fp, "cart_ale_limiter         = %s\n",    cfg.cart_ale_limiter.c_str());
    std::fprintf(fp, "cart_ale_cq_lin          = %.17g\n", cfg.cart_ale_cq_lin);
    std::fprintf(fp, "cart_ale_cq_quad         = %.17g\n", cfg.cart_ale_cq_quad);
    std::fprintf(fp, "cart_ale_shear_aware     = %s\n",    yn(cfg.cart_ale_shear_aware));
    std::fprintf(fp, "cart_ale2_rebuild_order  = %d\n",    cfg.cart_ale2_rebuild_order);
    std::fprintf(fp, "cart_ale2_bc_x           = %s\n",    cfg.cart_ale2_bc_x.c_str());
    std::fprintf(fp, "cart_ale2_bc_y           = %s\n",    cfg.cart_ale2_bc_y.c_str());
    std::fprintf(fp, "cart_ale2_ppm            = %s\n",    yn(cfg.cart_ale2_ppm));
    std::fprintf(fp, "cart_ale2_ppm_limiter    = %s\n",    cfg.cart_ale2_ppm_limiter.c_str());
    std::fprintf(fp, "cart_ale2_ppm_space      = %s\n",    cfg.cart_ale2_ppm_space.c_str());
    std::fprintf(fp, "cart_ale2_ppm_char       = %s\n",    yn(cfg.cart_ale2_ppm_char));
    std::fprintf(fp, "cart_ale2_kh_k           = %d\n",    cfg.cart_ale2_kh_k);
    std::fprintf(fp, "cart_ale2_trace_cells    = %s\n",    cfg.cart_ale2_trace_cells.c_str());
    std::fprintf(fp, "cart_ale2_trace_step_cap = %d\n",    cfg.cart_ale2_trace_step_cap);
    std::fprintf(fp, "cart_ale2_slab_file      = %s\n",    cfg.cart_ale2_slab_file.c_str());
    std::fprintf(fp, "cart_ale2_slab_perturb   = %.17g\n", cfg.cart_ale2_slab_perturb);
    std::fprintf(fp, "cart_ale2_slab_seed_k    = %d\n",    cfg.cart_ale2_slab_seed_k);
    std::fprintf(fp, "cart_ale2_cool_tau       = %.17g\n", cfg.cart_ale2_cool_tau);
    std::fprintf(fp, "cart_ale2_heat_flux      = %.17g\n", cfg.cart_ale2_heat_flux);
    std::fprintf(fp, "cart_ale2_heat_lsun      = %.17g\n", cfg.cart_ale2_heat_lsun);
    std::fprintf(fp, "cart_ale2_heat_bot_R     = %.17g\n", cfg.cart_ale2_heat_bot_R);
    std::fprintf(fp, "cart_ale2_heat_bot_frac  = %.17g\n", cfg.cart_ale2_heat_bot_frac);
    std::fprintf(fp, "cart_ale2_cool_top_frac  = %.17g\n", cfg.cart_ale2_cool_top_frac);
    std::fprintf(fp, "cart_ale2_andrassy_amp   = %.17g\n", cfg.cart_ale2_andrassy_amp);
    std::fprintf(fp, "cart_ale2_andrassy_seed  = %d\n",    cfg.cart_ale2_andrassy_seed);
    std::fprintf(fp, "cart_ale2_andrassy_noise = %.17g\n", cfg.cart_ale2_andrassy_noise);

    // ---- solver.athena_vl2 --------------------------------------------
    std::fprintf(fp, "\n# --- solver.athena_vl2 ---\n");
    std::fprintf(fp, "athena_vl2_xorder        = %d\n",    cfg.athena_vl2_xorder);
    std::fprintf(fp, "athena_vl2_limiter       = %d\n",    cfg.athena_vl2_limiter);

    // ---- solver.pseudo_spectral ---------------------------------------
    std::fprintf(fp, "\n# --- solver.pseudo_spectral ---\n");
    std::fprintf(fp, "ps_nu                    = %.17g\n", cfg.ps_nu);
    std::fprintf(fp, "ps_Lx                    = %.17g\n", cfg.ps_Lx);
    std::fprintf(fp, "ps_Ly                    = %.17g\n", cfg.ps_Ly);
    std::fprintf(fp, "ps_vshear                = %.17g\n", cfg.ps_vshear);
    std::fprintf(fp, "ps_k                     = %d\n",    cfg.ps_k);
    std::fprintf(fp, "ps_explicit              = %s\n",    yn(cfg.ps_explicit));
    std::fprintf(fp, "ps_adv_only              = %s\n",    yn(cfg.ps_adv_only));
    std::fprintf(fp, "ps_forcing_eps           = %.17g\n", cfg.ps_forcing_eps);
    std::fprintf(fp, "ps_forcing_kf            = %d\n",    cfg.ps_forcing_kf);
    std::fprintf(fp, "ps_forcing_dk            = %d\n",    cfg.ps_forcing_dk);
    std::fprintf(fp, "ps_forcing_seed          = %llu\n",
                 static_cast<unsigned long long>(cfg.ps_forcing_seed));
    std::fprintf(fp, "ps_forcing_host_rng      = %s\n",    yn(cfg.ps_forcing_host_rng));
    std::fprintf(fp, "ps_drag_alpha            = %.17g\n", cfg.ps_drag_alpha);
    std::fprintf(fp, "ps_hyper_p               = %d\n",    cfg.ps_hyper_p);
    std::fprintf(fp, "ps_conservative          = %s\n",    yn(cfg.ps_conservative));
    std::fprintf(fp, "ps_batched_fft           = %s\n",    yn(cfg.ps_batched_fft));
    std::fprintf(fp, "ps_pi_dt                 = %s\n",    yn(cfg.ps_pi_dt));
    std::fprintf(fp, "ps_tg_k                  = %d\n",    cfg.ps_tg_k);
    std::fprintf(fp, "ps_vm_gamma              = %.17g\n", cfg.ps_vm_gamma);
    std::fprintf(fp, "ps_vm_sigma              = %.17g\n", cfg.ps_vm_sigma);
    std::fprintf(fp, "ps_vm_dist               = %.17g\n", cfg.ps_vm_dist);
    std::fprintf(fp, "ps_ckpt_every            = %d\n",    cfg.ps_ckpt_every);
    std::fprintf(fp, "ps_resume                = %s\n",    cfg.ps_resume.c_str());

    // ---- solver.sph2d_spectral ----------------------------------------
    std::fprintf(fp, "\n# --- solver.sph2d_spectral ---\n");
    std::fprintf(fp, "sph_R                    = %.17g\n", cfg.sph_R);
    std::fprintf(fp, "sph_Omega                = %.17g\n", cfg.sph_Omega);
    std::fprintf(fp, "sph_nu                   = %.17g\n", cfg.sph_nu);
    std::fprintf(fp, "sph_Lmax                 = %d\n",    cfg.sph_Lmax);
    std::fprintf(fp, "sph_drag                 = %.17g\n", cfg.sph_drag);
    std::fprintf(fp, "sph_hyper                = %d\n",    cfg.sph_hyper);
    std::fprintf(fp, "sph_pi_dt                = %s\n",    yn(cfg.sph_pi_dt));
    std::fprintf(fp, "sph_rossby_l             = %d\n",    cfg.sph_rossby_l);
    std::fprintf(fp, "sph_rossby_m             = %d\n",    cfg.sph_rossby_m);
    std::fprintf(fp, "sph_rossby_amp           = %.17g\n", cfg.sph_rossby_amp);
    std::fprintf(fp, "sph_forcing_eps          = %.17g\n", cfg.sph_forcing_eps);
    std::fprintf(fp, "sph_forcing_lmin         = %d\n",    cfg.sph_forcing_lmin);
    std::fprintf(fp, "sph_forcing_lmax         = %d\n",    cfg.sph_forcing_lmax);
    std::fprintf(fp, "sph_forcing_seed         = %llu\n",
                 static_cast<unsigned long long>(cfg.sph_forcing_seed));
    std::fprintf(fp, "sph_ckpt_every           = %d\n",    cfg.sph_ckpt_every);
    std::fprintf(fp, "sph_resume               = %s\n",    cfg.sph_resume.c_str());

    // ---- Test-case IC params ------------------------------------------
    std::fprintf(fp, "\n# --- test IC params ---\n");
    std::fprintf(fp, "shear_V0                 = %.17g\n", cfg.shear_V0);
    std::fprintf(fp, "shear_k                  = %d\n",    cfg.shear_k);
    std::fprintf(fp, "shear_rho                = %.17g\n", cfg.shear_rho);
    std::fprintf(fp, "shear_P                  = %.17g\n", cfg.shear_P);
    std::fprintf(fp, "ewave_rho0               = %.17g\n", cfg.ewave_rho0);
    std::fprintf(fp, "ewave_P0                 = %.17g\n", cfg.ewave_P0);
    std::fprintf(fp, "ewave_u0                 = %.17g\n", cfg.ewave_u0);
    std::fprintf(fp, "ewave_A                  = %.17g\n", cfg.ewave_A);
    std::fprintf(fp, "ewave_k                  = %d\n",    cfg.ewave_k);
    std::fprintf(fp, "ewave_periods            = %.17g\n", cfg.ewave_periods);
    std::fprintf(fp, "awave_rho0               = %.17g\n", cfg.awave_rho0);
    std::fprintf(fp, "awave_P0                 = %.17g\n", cfg.awave_P0);
    std::fprintf(fp, "awave_A                  = %.17g\n", cfg.awave_A);
    std::fprintf(fp, "awave_k                  = %d\n",    cfg.awave_k);
    std::fprintf(fp, "awave_periods            = %.17g\n", cfg.awave_periods);

    std::fclose(fp);
    std::printf("[config_dump] wrote %s\n", path.c_str());
}
