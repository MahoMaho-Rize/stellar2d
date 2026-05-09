// Tier B-1 CLI — TOML -> SimConfig loader.
//
// Uses the vendored toml++ v3.4.0 single-header (third_party/tomlplusplus).
// We disable toml++ exceptions so the parser returns a parse_result with
// .failed()/.error() semantics, matching parse_cli's int-return style.
//
// Key list must stay synchronised with src/cli/config_dump.cpp.  Tier B-3
// will merge the two tables into a single field registry; for now we
// duplicate them and cross-check by running a round-trip regression.

#define TOML_EXCEPTIONS 0
#include <toml.hpp>

#include "cli/config_loader.h"
#include "cli/options.h"
#include "cli/suggest.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
//  Key whitelist — flat TOML form (Tier B-1).  Tier C will switch to
//  [grid]/[solver.<name>]/... sections and regenerate this from
//  SolverSpec definitions.
// ---------------------------------------------------------------------------
const std::vector<std::string>& known_toml_keys() {
    static const std::vector<std::string> k = {
        // core
        "solver", "test", "nr", "ntheta", "R_outer", "gamma", "cfl",
        "t_end", "G", "mesh", "limiter", "perturb", "run_base",
        // io
        "output_interval", "vtk_interval", "vtk_dt", "diag_interval",
        "frame_buffer", "frame_headroom_mb", "compute_error",
        // eos / physics
        "eos", "eos_mu", "eos_rad_a",
        "helm_table", "helm_Abar", "helm_Zbar",
        "radiation", "rad_c_light", "rad_T_phot_floor",
        "nuclear", "nuc_X", "nuc_Y", "nuc_epsilon_scale", "nuc_q_burn",
        "nuc_T_floor", "nuc_T_scale", "nuc_compress_frac",
        "species", "mlt", "mlt_alpha",
        "kap_use_table", "kap_highT_family", "kap_lowT_family",
        "kap_table_Z", "kap_data_dir", "kap_logT_lo_end", "kap_logT_hi_start",
        // ic
        "ic_solar", "ic_rho_c", "ic_R_star", "ic_n_poly",
        "ic_mesa_path", "ic_mesa_seed_T", "ic_mesa_atm_zones", "atm_split",
        "rich_profile",
        // bubble IC
        "bubble_mode", "bubble_xc", "bubble_yc", "bubble_rb",
        "bubble_alpha", "bubble_beta",
        // bc
        "no_sponge", "radial_only", "r_inner", "M_core",
        // solver.radial1d
        "implicit", "dt_implicit", "dt_implicit_scale",
        "no_viallet", "precond_tridiag", "jfnk_autodiff",
        "no_rhse_subtract", "newton_tol", "hse_resnap",
        "dt_thermal_frac", "dt_mach_cap",
        // solver.lowmach
        "precond", "hllc_variant",
        // solver.cart_ale / cart_ale2
        "cart_ale_remap_order", "cart_ale_limiter",
        "cart_ale_cq_lin", "cart_ale_cq_quad", "cart_ale_shear_aware",
        "cart_ale2_rebuild_order",
        "cart_ale2_bc_x", "cart_ale2_bc_y",
        "cart_ale2_ppm", "cart_ale2_ppm_limiter",
        "cart_ale2_ppm_space", "cart_ale2_ppm_char",
        "cart_ale2_kh_k", "cart_ale2_trace_cells", "cart_ale2_trace_step_cap",
        "cart_ale2_slab_file", "cart_ale2_slab_perturb", "cart_ale2_slab_seed_k",
        "cart_ale2_cool_tau",
        "cart_ale2_heat_flux", "cart_ale2_heat_lsun",
        "cart_ale2_heat_bot_R", "cart_ale2_heat_bot_frac",
        "cart_ale2_cool_top_frac",
        "cart_ale2_andrassy_amp", "cart_ale2_andrassy_seed",
        "cart_ale2_andrassy_noise",
        // athena_vl2
        "athena_vl2_xorder", "athena_vl2_limiter",
        // pseudo_spectral
        "ps_nu", "ps_Lx", "ps_Ly", "ps_vshear", "ps_k",
        "ps_explicit", "ps_adv_only",
        "ps_forcing_eps", "ps_forcing_kf", "ps_forcing_dk",
        "ps_forcing_seed", "ps_forcing_host_rng",
        "ps_drag_alpha", "ps_hyper_p",
        "ps_conservative", "ps_batched_fft", "ps_pi_dt",
        "ps_tg_k", "ps_vm_gamma", "ps_vm_sigma", "ps_vm_dist",
        "ps_ckpt_every", "ps_resume",
        // sph2d_spectral
        "sph_R", "sph_Omega", "sph_nu", "sph_Lmax",
        "sph_drag", "sph_hyper", "sph_pi_dt",
        "sph_rossby_l", "sph_rossby_m", "sph_rossby_amp",
        "sph_forcing_eps", "sph_forcing_lmin", "sph_forcing_lmax",
        "sph_forcing_seed",
        "sph_ckpt_every", "sph_resume",
        // test IC params
        "shear_V0", "shear_k", "shear_rho", "shear_P",
        "ewave_rho0", "ewave_P0", "ewave_u0",
        "ewave_A", "ewave_k", "ewave_periods",
        "awave_rho0", "awave_P0", "awave_A", "awave_k", "awave_periods",
    };
    return k;
}

// ---------------------------------------------------------------------------
//  Typed getters — each returns false only on type mismatch; missing keys
//  leave `out` untouched so SimConfig defaults remain.
// ---------------------------------------------------------------------------
bool get_bool(const toml::table& tbl, const char* key, bool& out, int& err) {
    auto node = tbl.get(key);
    if (!node) return true;
    if (auto v = node->value<bool>()) { out = *v; return true; }
    std::fprintf(stderr,
                 "[config] ERROR: key \"%s\" has wrong type (expected bool).\n", key);
    err = 1; return false;
}
bool get_int(const toml::table& tbl, const char* key, int& out, int& err) {
    auto node = tbl.get(key);
    if (!node) return true;
    if (auto v = node->value<int64_t>()) { out = static_cast<int>(*v); return true; }
    std::fprintf(stderr,
                 "[config] ERROR: key \"%s\" has wrong type (expected integer).\n", key);
    err = 1; return false;
}
bool get_u64(const toml::table& tbl, const char* key, uint64_t& out, int& err) {
    auto node = tbl.get(key);
    if (!node) return true;
    if (auto v = node->value<int64_t>()) {
        out = static_cast<uint64_t>(*v);
        return true;
    }
    std::fprintf(stderr,
                 "[config] ERROR: key \"%s\" has wrong type (expected integer).\n", key);
    err = 1; return false;
}
bool get_double(const toml::table& tbl, const char* key, double& out, int& err) {
    auto node = tbl.get(key);
    if (!node) return true;
    if (auto v = node->value<double>()) { out = *v; return true; }
    if (auto v = node->value<int64_t>()) { out = static_cast<double>(*v); return true; }
    std::fprintf(stderr,
                 "[config] ERROR: key \"%s\" has wrong type (expected float/int).\n", key);
    err = 1; return false;
}
bool get_string(const toml::table& tbl, const char* key,
                std::string& out, int& err) {
    auto node = tbl.get(key);
    if (!node) return true;
    if (auto v = node->value<std::string>()) { out = *v; return true; }
    std::fprintf(stderr,
                 "[config] ERROR: key \"%s\" has wrong type (expected string).\n", key);
    err = 1; return false;
}

// Map string limiter name to enum; leaves `out` untouched on missing key.
bool get_limiter(const toml::table& tbl, const char* key,
                 Limiter& out, int& err) {
    std::string s;
    if (!get_string(tbl, key, s, err)) return false;
    if (s.empty()) return true;     // key absent
    if (s == "minmod")       out = Limiter::MINMOD;
    else if (s == "vanleer") out = Limiter::VAN_LEER;
    else if (s == "mc")      out = Limiter::MC;
    else {
        std::fprintf(stderr,
                     "[config] ERROR: key \"%s\" has unknown limiter "
                     "\"%s\" (expected minmod | vanleer | mc).\n", key, s.c_str());
        err = 1; return false;
    }
    return true;
}

} // namespace

int load_toml_into_cfg(const std::string& path, SimConfig& cfg) {
    toml::parse_result res = toml::parse_file(path);
    if (!res) {
        const toml::parse_error& e = res.error();
        const toml::source_region& src = e.source();
        std::fprintf(stderr,
                     "[config] ERROR: cannot parse %s:\n  %.*s\n  at line %u, column %u\n",
                     path.c_str(),
                     static_cast<int>(e.description().length()),
                     e.description().data(),
                     static_cast<unsigned>(src.begin.line),
                     static_cast<unsigned>(src.begin.column));
        return 1;
    }
    const toml::table& tbl = res.table();

    // ---- Validation: every top-level key must be recognised --------------
    const auto& known = known_toml_keys();
    for (const auto& [k, _] : tbl) {
        std::string key(k.str());
        if (std::find(known.begin(), known.end(), key) == known.end()) {
            std::fprintf(stderr,
                         "[config] ERROR: unknown TOML key \"%s\" in %s\n",
                         key.c_str(), path.c_str());
            std::string s = suggest_closest(key, known, 3);
            if (!s.empty()) {
                std::fprintf(stderr, "  Did you mean %s?\n", s.c_str());
            }
            return 1;
        }
    }

    // ---- Apply -----------------------------------------------------------
    int err = 0;

    // core
    get_string(tbl, "solver",       cfg.solver_type,       err);
    get_string(tbl, "test",         cfg.test_case,         err);
    get_int   (tbl, "nr",           cfg.nr,                err);
    get_int   (tbl, "ntheta",       cfg.ntheta,            err);
    get_double(tbl, "R_outer",      cfg.R_outer,           err);
    get_double(tbl, "gamma",        cfg.gamma,             err);
    get_double(tbl, "cfl",          cfg.cfl,               err);
    get_double(tbl, "t_end",        cfg.t_end,             err);
    get_double(tbl, "G",            cfg.G,                 err);
    get_string(tbl, "mesh",         cfg.mesh_type,         err);
    get_limiter(tbl,"limiter",      cfg.limiter,           err);
    get_double(tbl, "perturb",      cfg.perturb_amplitude, err);
    get_string(tbl, "run_base",     cfg.run_base,          err);

    // io
    get_int   (tbl, "output_interval",    cfg.output_interval,    err);
    get_int   (tbl, "vtk_interval",       cfg.vtk_interval,       err);
    get_double(tbl, "vtk_dt",             cfg.vtk_dt,             err);
    get_int   (tbl, "diag_interval",      cfg.diag_interval,      err);
    get_bool  (tbl, "frame_buffer",       cfg.frame_buffer,       err);
    get_int   (tbl, "frame_headroom_mb",  cfg.frame_headroom_mb,  err);
    get_bool  (tbl, "compute_error",      cfg.compute_error,      err);

    // eos / physics
    get_string(tbl, "eos",                cfg.eos_type,           err);
    get_double(tbl, "eos_mu",             cfg.eos_mu,             err);
    get_double(tbl, "eos_rad_a",          cfg.eos_rad_a,          err);
    get_string(tbl, "helm_table",         cfg.helm_table_path,    err);
    get_double(tbl, "helm_Abar",          cfg.helm_Abar,          err);
    get_double(tbl, "helm_Zbar",          cfg.helm_Zbar,          err);
    get_bool  (tbl, "radiation",          cfg.radiation_enabled,  err);
    get_double(tbl, "rad_c_light",        cfg.rad_c_light,        err);
    get_double(tbl, "rad_T_phot_floor",   cfg.rad_T_phot_floor,   err);
    get_bool  (tbl, "nuclear",            cfg.nuclear_enabled,    err);
    get_double(tbl, "nuc_X",              cfg.nuc_X,              err);
    get_double(tbl, "nuc_Y",              cfg.nuc_Y,              err);
    get_double(tbl, "nuc_epsilon_scale",  cfg.nuc_epsilon_scale,  err);
    get_double(tbl, "nuc_q_burn",         cfg.nuc_q_burn,         err);
    get_double(tbl, "nuc_T_floor",        cfg.nuc_T_floor,        err);
    get_double(tbl, "nuc_T_scale",        cfg.nuc_T_scale,        err);
    get_double(tbl, "nuc_compress_frac",  cfg.nuc_compress_frac,  err);
    get_bool  (tbl, "species",            cfg.species_enabled,    err);
    get_bool  (tbl, "mlt",                cfg.mlt_enabled,        err);
    get_double(tbl, "mlt_alpha",          cfg.mlt_alpha,          err);
    get_bool  (tbl, "kap_use_table",      cfg.kap_use_table,      err);
    get_string(tbl, "kap_highT_family",   cfg.kap_highT_family,   err);
    get_string(tbl, "kap_lowT_family",    cfg.kap_lowT_family,    err);
    get_double(tbl, "kap_table_Z",        cfg.kap_table_Z,        err);
    get_string(tbl, "kap_data_dir",       cfg.kap_data_dir,       err);
    get_double(tbl, "kap_logT_lo_end",    cfg.kap_logT_lo_end,    err);
    get_double(tbl, "kap_logT_hi_start",  cfg.kap_logT_hi_start,  err);

    // ic
    get_bool  (tbl, "ic_solar",           cfg.ic_solar,           err);
    get_double(tbl, "ic_rho_c",           cfg.ic_rho_c,           err);
    get_double(tbl, "ic_R_star",          cfg.ic_R_star,          err);
    get_double(tbl, "ic_n_poly",          cfg.ic_n_poly,          err);
    get_string(tbl, "ic_mesa_path",       cfg.ic_mesa_path,       err);
    get_bool  (tbl, "ic_mesa_seed_T",     cfg.ic_mesa_seed_T,     err);
    get_int   (tbl, "ic_mesa_atm_zones",  cfg.ic_mesa_atm_zones,  err);
    get_int   (tbl, "atm_split",          cfg.atm_split,          err);
    get_bool  (tbl, "rich_profile",       cfg.rich_profile,       err);

    // bubble IC (single-bubble shortcut; multi-bubble array left for Tier B-2)
    get_string(tbl, "bubble_mode",        cfg.bubble_mode,        err);
    get_double(tbl, "bubble_xc",          cfg.bubble_xc,          err);
    get_double(tbl, "bubble_yc",          cfg.bubble_yc,          err);
    get_double(tbl, "bubble_rb",          cfg.bubble_rb,          err);
    get_double(tbl, "bubble_alpha",       cfg.bubble_alpha,       err);
    get_double(tbl, "bubble_beta",        cfg.bubble_beta,        err);

    // bc
    get_bool  (tbl, "no_sponge",          cfg.no_sponge,          err);
    get_bool  (tbl, "radial_only",        cfg.radial_only,        err);
    get_double(tbl, "r_inner",            cfg.r_inner,            err);
    get_double(tbl, "M_core",             cfg.M_core,             err);

    // solver.radial1d
    get_bool  (tbl, "implicit",           cfg.implicit_mode,      err);
    get_double(tbl, "dt_implicit",        cfg.dt_implicit,        err);
    get_double(tbl, "dt_implicit_scale",  cfg.dt_implicit_scale,  err);
    get_bool  (tbl, "no_viallet",         cfg.no_viallet,         err);
    get_bool  (tbl, "precond_tridiag",    cfg.precond_tridiag,    err);
    get_bool  (tbl, "jfnk_autodiff",      cfg.jfnk_autodiff,      err);
    get_bool  (tbl, "no_rhse_subtract",   cfg.no_rhse_subtract,   err);
    get_double(tbl, "newton_tol",         cfg.newton_tol_override,err);
    get_int   (tbl, "hse_resnap",         cfg.hse_resnap_interval,err);
    get_double(tbl, "dt_thermal_frac",    cfg.dt_thermal_frac,    err);
    get_double(tbl, "dt_mach_cap",        cfg.dt_mach_cap,        err);

    // solver.lowmach
    get_string(tbl, "precond",            cfg.precond,            err);
    get_int   (tbl, "hllc_variant",       cfg.hllc_variant,       err);

    // solver.cart_ale / cart_ale2
    get_int   (tbl, "cart_ale_remap_order",    cfg.cart_ale_remap_order,    err);
    get_string(tbl, "cart_ale_limiter",        cfg.cart_ale_limiter,        err);
    get_double(tbl, "cart_ale_cq_lin",         cfg.cart_ale_cq_lin,         err);
    get_double(tbl, "cart_ale_cq_quad",        cfg.cart_ale_cq_quad,        err);
    get_bool  (tbl, "cart_ale_shear_aware",    cfg.cart_ale_shear_aware,    err);
    get_int   (tbl, "cart_ale2_rebuild_order", cfg.cart_ale2_rebuild_order, err);
    get_string(tbl, "cart_ale2_bc_x",          cfg.cart_ale2_bc_x,          err);
    get_string(tbl, "cart_ale2_bc_y",          cfg.cart_ale2_bc_y,          err);
    get_bool  (tbl, "cart_ale2_ppm",           cfg.cart_ale2_ppm,           err);
    get_string(tbl, "cart_ale2_ppm_limiter",   cfg.cart_ale2_ppm_limiter,   err);
    get_string(tbl, "cart_ale2_ppm_space",     cfg.cart_ale2_ppm_space,     err);
    get_bool  (tbl, "cart_ale2_ppm_char",      cfg.cart_ale2_ppm_char,      err);
    get_int   (tbl, "cart_ale2_kh_k",          cfg.cart_ale2_kh_k,          err);
    get_string(tbl, "cart_ale2_trace_cells",   cfg.cart_ale2_trace_cells,   err);
    get_int   (tbl, "cart_ale2_trace_step_cap",cfg.cart_ale2_trace_step_cap,err);
    get_string(tbl, "cart_ale2_slab_file",     cfg.cart_ale2_slab_file,     err);
    get_double(tbl, "cart_ale2_slab_perturb",  cfg.cart_ale2_slab_perturb,  err);
    get_int   (tbl, "cart_ale2_slab_seed_k",   cfg.cart_ale2_slab_seed_k,   err);
    get_double(tbl, "cart_ale2_cool_tau",      cfg.cart_ale2_cool_tau,      err);
    get_double(tbl, "cart_ale2_heat_flux",     cfg.cart_ale2_heat_flux,     err);
    get_double(tbl, "cart_ale2_heat_lsun",     cfg.cart_ale2_heat_lsun,     err);
    get_double(tbl, "cart_ale2_heat_bot_R",    cfg.cart_ale2_heat_bot_R,    err);
    get_double(tbl, "cart_ale2_heat_bot_frac", cfg.cart_ale2_heat_bot_frac, err);
    get_double(tbl, "cart_ale2_cool_top_frac", cfg.cart_ale2_cool_top_frac, err);
    get_double(tbl, "cart_ale2_andrassy_amp",  cfg.cart_ale2_andrassy_amp,  err);
    get_int   (tbl, "cart_ale2_andrassy_seed", cfg.cart_ale2_andrassy_seed, err);
    get_double(tbl, "cart_ale2_andrassy_noise",cfg.cart_ale2_andrassy_noise,err);

    // athena_vl2
    get_int   (tbl, "athena_vl2_xorder",       cfg.athena_vl2_xorder,       err);
    get_int   (tbl, "athena_vl2_limiter",      cfg.athena_vl2_limiter,      err);

    // pseudo_spectral
    get_double(tbl, "ps_nu",                   cfg.ps_nu,                   err);
    get_double(tbl, "ps_Lx",                   cfg.ps_Lx,                   err);
    get_double(tbl, "ps_Ly",                   cfg.ps_Ly,                   err);
    get_double(tbl, "ps_vshear",               cfg.ps_vshear,               err);
    get_int   (tbl, "ps_k",                    cfg.ps_k,                    err);
    get_bool  (tbl, "ps_explicit",             cfg.ps_explicit,             err);
    get_bool  (tbl, "ps_adv_only",             cfg.ps_adv_only,             err);
    get_double(tbl, "ps_forcing_eps",          cfg.ps_forcing_eps,          err);
    get_int   (tbl, "ps_forcing_kf",           cfg.ps_forcing_kf,           err);
    get_int   (tbl, "ps_forcing_dk",           cfg.ps_forcing_dk,           err);
    get_u64   (tbl, "ps_forcing_seed",         cfg.ps_forcing_seed,         err);
    get_bool  (tbl, "ps_forcing_host_rng",     cfg.ps_forcing_host_rng,     err);
    get_double(tbl, "ps_drag_alpha",           cfg.ps_drag_alpha,           err);
    get_int   (tbl, "ps_hyper_p",              cfg.ps_hyper_p,              err);
    get_bool  (tbl, "ps_conservative",         cfg.ps_conservative,         err);
    get_bool  (tbl, "ps_batched_fft",          cfg.ps_batched_fft,          err);
    get_bool  (tbl, "ps_pi_dt",                cfg.ps_pi_dt,                err);
    get_int   (tbl, "ps_tg_k",                 cfg.ps_tg_k,                 err);
    get_double(tbl, "ps_vm_gamma",             cfg.ps_vm_gamma,             err);
    get_double(tbl, "ps_vm_sigma",             cfg.ps_vm_sigma,             err);
    get_double(tbl, "ps_vm_dist",              cfg.ps_vm_dist,              err);
    get_int   (tbl, "ps_ckpt_every",           cfg.ps_ckpt_every,           err);
    get_string(tbl, "ps_resume",               cfg.ps_resume,               err);

    // sph2d_spectral
    get_double(tbl, "sph_R",                   cfg.sph_R,                   err);
    get_double(tbl, "sph_Omega",               cfg.sph_Omega,               err);
    get_double(tbl, "sph_nu",                  cfg.sph_nu,                  err);
    get_int   (tbl, "sph_Lmax",                cfg.sph_Lmax,                err);
    get_double(tbl, "sph_drag",                cfg.sph_drag,                err);
    get_int   (tbl, "sph_hyper",               cfg.sph_hyper,               err);
    get_bool  (tbl, "sph_pi_dt",               cfg.sph_pi_dt,               err);
    get_int   (tbl, "sph_rossby_l",            cfg.sph_rossby_l,            err);
    get_int   (tbl, "sph_rossby_m",            cfg.sph_rossby_m,            err);
    get_double(tbl, "sph_rossby_amp",          cfg.sph_rossby_amp,          err);
    get_double(tbl, "sph_forcing_eps",         cfg.sph_forcing_eps,         err);
    get_int   (tbl, "sph_forcing_lmin",        cfg.sph_forcing_lmin,        err);
    get_int   (tbl, "sph_forcing_lmax",        cfg.sph_forcing_lmax,        err);
    get_u64   (tbl, "sph_forcing_seed",        cfg.sph_forcing_seed,        err);
    get_int   (tbl, "sph_ckpt_every",          cfg.sph_ckpt_every,          err);
    get_string(tbl, "sph_resume",              cfg.sph_resume,              err);

    // test IC params
    get_double(tbl, "shear_V0",                cfg.shear_V0,                err);
    get_int   (tbl, "shear_k",                 cfg.shear_k,                 err);
    get_double(tbl, "shear_rho",               cfg.shear_rho,               err);
    get_double(tbl, "shear_P",                 cfg.shear_P,                 err);
    get_double(tbl, "ewave_rho0",              cfg.ewave_rho0,              err);
    get_double(tbl, "ewave_P0",                cfg.ewave_P0,                err);
    get_double(tbl, "ewave_u0",                cfg.ewave_u0,                err);
    get_double(tbl, "ewave_A",                 cfg.ewave_A,                 err);
    get_int   (tbl, "ewave_k",                 cfg.ewave_k,                 err);
    get_double(tbl, "ewave_periods",           cfg.ewave_periods,           err);
    get_double(tbl, "awave_rho0",              cfg.awave_rho0,              err);
    get_double(tbl, "awave_P0",                cfg.awave_P0,                err);
    get_double(tbl, "awave_A",                 cfg.awave_A,                 err);
    get_int   (tbl, "awave_k",                 cfg.awave_k,                 err);
    get_double(tbl, "awave_periods",           cfg.awave_periods,           err);

    if (err) {
        std::fprintf(stderr,
                     "[config] aborting due to type errors while reading %s.\n",
                     path.c_str());
        return 1;
    }

    std::printf("[config] loaded %s\n", path.c_str());
    return 0;
}

// ---------------------------------------------------------------------------
//  Tier B-2: --profile NAME sugar.  Resolves NAME to
//  config/profiles/<NAME>.toml (relative to the current working directory)
//  and delegates to load_toml_into_cfg.  If NAME contains a slash or ends
//  in ".toml", it is treated as a path and used verbatim so that
//  `--profile /path/to/foo.toml` also works.
// ---------------------------------------------------------------------------

std::vector<std::string> available_profile_names() {
    std::vector<std::string> out;
    const std::filesystem::path dir = "config/profiles";
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec) ||
        !std::filesystem::is_directory(dir, ec)) {
        return out;
    }
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() == ".toml") {
            out.push_back(entry.path().stem().string());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

int load_profile_into_cfg(const std::string& name_or_path, SimConfig& cfg) {
    // Raw-path short-circuit so `--profile ./mine.toml` or absolute paths
    // behave identically to `--config`.
    bool is_path = (name_or_path.find('/') != std::string::npos) ||
                   (name_or_path.size() >= 5 &&
                    name_or_path.substr(name_or_path.size() - 5) == ".toml");

    std::string path;
    if (is_path) {
        path = name_or_path;
    } else {
        path = "config/profiles/" + name_or_path + ".toml";
    }

    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        std::fprintf(stderr,
                     "[profile] ERROR: profile \"%s\" not found (looked for %s).\n",
                     name_or_path.c_str(), path.c_str());

        const auto avail = available_profile_names();
        if (avail.empty()) {
            std::fprintf(stderr,
                         "  No profiles found under config/profiles/.\n"
                         "  Are you running from the repo root?  Pass an explicit\n"
                         "  path with --config <path> to point at a TOML elsewhere.\n");
        } else {
            std::string s = suggest_closest(name_or_path, avail, 3);
            if (!s.empty()) {
                std::fprintf(stderr, "  Did you mean --profile %s?\n", s.c_str());
            }
            std::fprintf(stderr, "  Available profiles:\n");
            for (const auto& n : avail) {
                std::fprintf(stderr, "    %s\n", n.c_str());
            }
        }
        return 1;
    }

    return load_toml_into_cfg(path, cfg);
}

// ---------------------------------------------------------------------------
//  Tier B-3: `stellar2d validate <path>` back-end.  Load the TOML into a
//  throwaway SimConfig — exercises the same code path as --config so any
//  unknown key / type mismatch / syntax error is surfaced.  Returns 0 on
//  success (valid), 1 on failure.
// ---------------------------------------------------------------------------
int validate_toml_file(const std::string& path) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        std::fprintf(stderr, "[validate] ERROR: file not found: %s\n",
                     path.c_str());
        return 1;
    }
    SimConfig scratch;
    int rc = load_toml_into_cfg(path, scratch);
    if (rc == 0) {
        std::printf("[validate] OK: %s is a valid stellar2d TOML config.\n",
                    path.c_str());
    }
    return rc;
}
