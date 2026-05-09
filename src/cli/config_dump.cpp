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

const char* bool_str(bool b) { return b ? "true" : "false"; }

// Escape a string for safe embedding in a TOML basic-string literal.
// We only need the characters that actually appear in stellar2d field
// values (paths, names, trace-cell specs): backslash, double quote, and
// the usual control chars.
std::string toml_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x",
                                  static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

} // namespace

void dump_resolved_cli(const SimConfig& cfg, const std::string& run_dir) {
    const std::string path = run_dir + "/config.toml";
    FILE* fp = std::fopen(path.c_str(), "w");
    if (!fp) {
        std::fprintf(stderr,
                     "[config_dump] WARNING: cannot open %s for writing - "
                     "reproducibility dump skipped.\n", path.c_str());
        return;
    }

    // File-scoped writer helpers.
    auto put_str = [&](const char* key, const std::string& v) {
        std::fprintf(fp, "%-24s = \"%s\"\n", key, toml_escape(v).c_str());
    };
    auto put_bool = [&](const char* key, bool v) {
        std::fprintf(fp, "%-24s = %s\n", key, bool_str(v));
    };
    auto put_int = [&](const char* key, int v) {
        std::fprintf(fp, "%-24s = %d\n", key, v);
    };
    auto put_u64 = [&](const char* key, unsigned long long v) {
        std::fprintf(fp, "%-24s = %llu\n", key, v);
    };
    auto put_double = [&](const char* key, double v) {
        std::fprintf(fp, "%-24s = %.17g\n", key, v);
    };

    // ---- Header --------------------------------------------------------
    std::fprintf(fp,
        "# stellar2d resolved CLI configuration (TOML).\n"
        "# Reload via:  stellar2d run --config %s\n"
        "# See docs/design/cli_unification_plan_2026-05-09.md §2d, §6.\n"
        "#\n"
        "# stellar2d version: %s\n"
        "# build date:        %s\n"
        "\n",
        path.c_str(), STELLAR2D_GIT_HASH, STELLAR2D_BUILD_DATE);

    // ---- Core ----------------------------------------------------------
    std::fprintf(fp, "# --- core ---\n");
    put_str  ("solver",       cfg.solver_type);
    put_str  ("test",         cfg.test_case);
    put_int  ("nr",           cfg.nr);
    put_int  ("ntheta",       cfg.ntheta);
    put_double("R_outer",     cfg.R_outer);
    put_double("gamma",       cfg.gamma);
    put_double("cfl",         cfg.cfl);
    put_double("t_end",       cfg.t_end);
    put_double("G",           cfg.G);
    put_str  ("mesh",         cfg.mesh_type);
    put_str  ("limiter",      limiter_name(cfg.limiter));
    put_double("perturb",     cfg.perturb_amplitude);
    put_str  ("run_base",     cfg.run_base);

    // ---- IO ------------------------------------------------------------
    std::fprintf(fp, "\n# --- io ---\n");
    put_int  ("output_interval",    cfg.output_interval);
    put_int  ("vtk_interval",       cfg.vtk_interval);
    put_double("vtk_dt",            cfg.vtk_dt);
    put_int  ("diag_interval",      cfg.diag_interval);
    put_bool ("frame_buffer",       cfg.frame_buffer);
    put_int  ("frame_headroom_mb",  cfg.frame_headroom_mb);
    put_bool ("compute_error",      cfg.compute_error);

    // ---- EOS / physics -------------------------------------------------
    std::fprintf(fp, "\n# --- eos / physics ---\n");
    put_str  ("eos",                cfg.eos_type);
    put_double("eos_mu",            cfg.eos_mu);
    put_double("eos_rad_a",         cfg.eos_rad_a);
    put_str  ("helm_table",         cfg.helm_table_path);
    put_double("helm_Abar",         cfg.helm_Abar);
    put_double("helm_Zbar",         cfg.helm_Zbar);
    put_bool ("radiation",          cfg.radiation_enabled);
    put_double("rad_c_light",       cfg.rad_c_light);
    put_double("rad_T_phot_floor",  cfg.rad_T_phot_floor);
    put_bool ("nuclear",            cfg.nuclear_enabled);
    put_double("nuc_X",             cfg.nuc_X);
    put_double("nuc_Y",             cfg.nuc_Y);
    put_double("nuc_epsilon_scale", cfg.nuc_epsilon_scale);
    put_double("nuc_q_burn",        cfg.nuc_q_burn);
    put_double("nuc_T_floor",       cfg.nuc_T_floor);
    put_double("nuc_T_scale",       cfg.nuc_T_scale);
    put_double("nuc_compress_frac", cfg.nuc_compress_frac);
    put_bool ("species",            cfg.species_enabled);
    put_bool ("mlt",                cfg.mlt_enabled);
    put_double("mlt_alpha",         cfg.mlt_alpha);
    put_bool ("kap_use_table",      cfg.kap_use_table);
    put_str  ("kap_highT_family",   cfg.kap_highT_family);
    put_str  ("kap_lowT_family",    cfg.kap_lowT_family);
    put_double("kap_table_Z",       cfg.kap_table_Z);
    put_str  ("kap_data_dir",       cfg.kap_data_dir);
    put_double("kap_logT_lo_end",   cfg.kap_logT_lo_end);
    put_double("kap_logT_hi_start", cfg.kap_logT_hi_start);

    // ---- IC ------------------------------------------------------------
    std::fprintf(fp, "\n# --- initial conditions ---\n");
    put_bool ("ic_solar",           cfg.ic_solar);
    put_double("ic_rho_c",          cfg.ic_rho_c);
    put_double("ic_R_star",         cfg.ic_R_star);
    put_double("ic_n_poly",         cfg.ic_n_poly);
    put_str  ("ic_mesa_path",       cfg.ic_mesa_path);
    put_bool ("ic_mesa_seed_T",     cfg.ic_mesa_seed_T);
    put_int  ("ic_mesa_atm_zones",  cfg.ic_mesa_atm_zones);
    put_int  ("atm_split",          cfg.atm_split);
    put_bool ("rich_profile",       cfg.rich_profile);

    // ---- bubble IC (single-bubble; multi-bubble array Tier C) ----------
    std::fprintf(fp, "\n# --- bubble IC ---\n");
    put_str  ("bubble_mode",        cfg.bubble_mode);
    put_double("bubble_xc",         cfg.bubble_xc);
    put_double("bubble_yc",         cfg.bubble_yc);
    put_double("bubble_rb",         cfg.bubble_rb);
    put_double("bubble_alpha",      cfg.bubble_alpha);
    put_double("bubble_beta",       cfg.bubble_beta);
    if (!cfg.bubbles.empty()) {
        std::fprintf(fp,
            "# NOTE: %zu bubble(s) from --bubble are not dumped in Tier B-3 TOML.\n"
            "# Tier C will add [[bubble_multi]] array-of-tables support.\n",
            cfg.bubbles.size());
    }

    // ---- BC ------------------------------------------------------------
    std::fprintf(fp, "\n# --- bc ---\n");
    put_bool ("no_sponge",          cfg.no_sponge);
    put_bool ("radial_only",        cfg.radial_only);
    put_double("r_inner",           cfg.r_inner);
    put_double("M_core",            cfg.M_core);

    // ---- solver.radial1d ----------------------------------------------
    std::fprintf(fp, "\n# --- solver.radial1d ---\n");
    put_bool ("implicit",           cfg.implicit_mode);
    put_double("dt_implicit",       cfg.dt_implicit);
    put_double("dt_implicit_scale", cfg.dt_implicit_scale);
    put_bool ("no_viallet",         cfg.no_viallet);
    put_bool ("precond_tridiag",    cfg.precond_tridiag);
    put_bool ("jfnk_autodiff",      cfg.jfnk_autodiff);
    put_bool ("no_rhse_subtract",   cfg.no_rhse_subtract);
    put_double("newton_tol",        cfg.newton_tol_override);
    put_int  ("hse_resnap",         cfg.hse_resnap_interval);
    put_double("dt_thermal_frac",   cfg.dt_thermal_frac);
    put_double("dt_mach_cap",       cfg.dt_mach_cap);

    // ---- solver.lowmach ------------------------------------------------
    std::fprintf(fp, "\n# --- solver.lowmach ---\n");
    put_str  ("precond",            cfg.precond);
    put_int  ("hllc_variant",       cfg.hllc_variant);

    // ---- solver.cart_ale / cart_ale2 -----------------------------------
    std::fprintf(fp, "\n# --- solver.cart_ale / cart_ale2 ---\n");
    put_int  ("cart_ale_remap_order",    cfg.cart_ale_remap_order);
    put_str  ("cart_ale_limiter",        cfg.cart_ale_limiter);
    put_double("cart_ale_cq_lin",        cfg.cart_ale_cq_lin);
    put_double("cart_ale_cq_quad",       cfg.cart_ale_cq_quad);
    put_bool ("cart_ale_shear_aware",    cfg.cart_ale_shear_aware);
    put_int  ("cart_ale2_rebuild_order", cfg.cart_ale2_rebuild_order);
    put_str  ("cart_ale2_bc_x",          cfg.cart_ale2_bc_x);
    put_str  ("cart_ale2_bc_y",          cfg.cart_ale2_bc_y);
    put_bool ("cart_ale2_ppm",           cfg.cart_ale2_ppm);
    put_str  ("cart_ale2_ppm_limiter",   cfg.cart_ale2_ppm_limiter);
    put_str  ("cart_ale2_ppm_space",     cfg.cart_ale2_ppm_space);
    put_bool ("cart_ale2_ppm_char",      cfg.cart_ale2_ppm_char);
    put_int  ("cart_ale2_kh_k",          cfg.cart_ale2_kh_k);
    put_str  ("cart_ale2_trace_cells",   cfg.cart_ale2_trace_cells);
    put_int  ("cart_ale2_trace_step_cap",cfg.cart_ale2_trace_step_cap);
    put_str  ("cart_ale2_slab_file",     cfg.cart_ale2_slab_file);
    put_double("cart_ale2_slab_perturb", cfg.cart_ale2_slab_perturb);
    put_int  ("cart_ale2_slab_seed_k",   cfg.cart_ale2_slab_seed_k);
    put_double("cart_ale2_cool_tau",     cfg.cart_ale2_cool_tau);
    put_double("cart_ale2_heat_flux",    cfg.cart_ale2_heat_flux);
    put_double("cart_ale2_heat_lsun",    cfg.cart_ale2_heat_lsun);
    put_double("cart_ale2_heat_bot_R",   cfg.cart_ale2_heat_bot_R);
    put_double("cart_ale2_heat_bot_frac",cfg.cart_ale2_heat_bot_frac);
    put_double("cart_ale2_cool_top_frac",cfg.cart_ale2_cool_top_frac);
    put_double("cart_ale2_andrassy_amp", cfg.cart_ale2_andrassy_amp);
    put_int  ("cart_ale2_andrassy_seed", cfg.cart_ale2_andrassy_seed);
    put_double("cart_ale2_andrassy_noise",cfg.cart_ale2_andrassy_noise);

    // ---- solver.athena_vl2 --------------------------------------------
    std::fprintf(fp, "\n# --- solver.athena_vl2 ---\n");
    put_int  ("athena_vl2_xorder",  cfg.athena_vl2_xorder);
    put_int  ("athena_vl2_limiter", cfg.athena_vl2_limiter);

    // ---- solver.pseudo_spectral ---------------------------------------
    std::fprintf(fp, "\n# --- solver.pseudo_spectral ---\n");
    put_double("ps_nu",              cfg.ps_nu);
    put_double("ps_Lx",              cfg.ps_Lx);
    put_double("ps_Ly",              cfg.ps_Ly);
    put_double("ps_vshear",          cfg.ps_vshear);
    put_int  ("ps_k",                cfg.ps_k);
    put_bool ("ps_explicit",         cfg.ps_explicit);
    put_bool ("ps_adv_only",         cfg.ps_adv_only);
    put_double("ps_forcing_eps",     cfg.ps_forcing_eps);
    put_int  ("ps_forcing_kf",       cfg.ps_forcing_kf);
    put_int  ("ps_forcing_dk",       cfg.ps_forcing_dk);
    put_u64  ("ps_forcing_seed",     static_cast<unsigned long long>(cfg.ps_forcing_seed));
    put_bool ("ps_forcing_host_rng", cfg.ps_forcing_host_rng);
    put_double("ps_drag_alpha",      cfg.ps_drag_alpha);
    put_int  ("ps_hyper_p",          cfg.ps_hyper_p);
    put_bool ("ps_conservative",     cfg.ps_conservative);
    put_bool ("ps_batched_fft",      cfg.ps_batched_fft);
    put_bool ("ps_pi_dt",            cfg.ps_pi_dt);
    put_int  ("ps_tg_k",             cfg.ps_tg_k);
    put_double("ps_vm_gamma",        cfg.ps_vm_gamma);
    put_double("ps_vm_sigma",        cfg.ps_vm_sigma);
    put_double("ps_vm_dist",         cfg.ps_vm_dist);
    put_int  ("ps_ckpt_every",       cfg.ps_ckpt_every);
    put_str  ("ps_resume",           cfg.ps_resume);

    // ---- solver.sph2d_spectral ----------------------------------------
    std::fprintf(fp, "\n# --- solver.sph2d_spectral ---\n");
    put_double("sph_R",              cfg.sph_R);
    put_double("sph_Omega",          cfg.sph_Omega);
    put_double("sph_nu",             cfg.sph_nu);
    put_int  ("sph_Lmax",            cfg.sph_Lmax);
    put_double("sph_drag",           cfg.sph_drag);
    put_int  ("sph_hyper",           cfg.sph_hyper);
    put_bool ("sph_pi_dt",           cfg.sph_pi_dt);
    put_int  ("sph_rossby_l",        cfg.sph_rossby_l);
    put_int  ("sph_rossby_m",        cfg.sph_rossby_m);
    put_double("sph_rossby_amp",     cfg.sph_rossby_amp);
    put_double("sph_forcing_eps",    cfg.sph_forcing_eps);
    put_int  ("sph_forcing_lmin",    cfg.sph_forcing_lmin);
    put_int  ("sph_forcing_lmax",    cfg.sph_forcing_lmax);
    put_u64  ("sph_forcing_seed",    static_cast<unsigned long long>(cfg.sph_forcing_seed));
    put_int  ("sph_ckpt_every",      cfg.sph_ckpt_every);
    put_str  ("sph_resume",          cfg.sph_resume);

    // ---- Test-case IC params ------------------------------------------
    std::fprintf(fp, "\n# --- test IC params ---\n");
    put_double("shear_V0",           cfg.shear_V0);
    put_int  ("shear_k",             cfg.shear_k);
    put_double("shear_rho",          cfg.shear_rho);
    put_double("shear_P",            cfg.shear_P);
    put_double("ewave_rho0",         cfg.ewave_rho0);
    put_double("ewave_P0",           cfg.ewave_P0);
    put_double("ewave_u0",           cfg.ewave_u0);
    put_double("ewave_A",            cfg.ewave_A);
    put_int  ("ewave_k",             cfg.ewave_k);
    put_double("ewave_periods",      cfg.ewave_periods);
    put_double("awave_rho0",         cfg.awave_rho0);
    put_double("awave_P0",           cfg.awave_P0);
    put_double("awave_A",            cfg.awave_A);
    put_int  ("awave_k",             cfg.awave_k);
    put_double("awave_periods",      cfg.awave_periods);

    std::fclose(fp);
    std::printf("[config_dump] wrote %s\n", path.c_str());
}
