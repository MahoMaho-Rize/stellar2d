#include "sim/setup.h"

#include "sim/helpers.h"
#include "init/lane_emden.h"
#include "init/sedov.h"
#include "init/jeans.h"
#include "init/evrard.h"
#include "io/output.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#ifdef USE_GPU
void SimContext::destroy_tables() {
    if (helm_loaded) helm_tbl.destroy();
    if (kap_loaded) { kap_tbl_lowT.destroy(); kap_tbl_highT.destroy(); }
}
#endif

// Lane-Emden density ρ(r) callable built from a precomputed solution.
// Used by multiple mesh branches below; captured as a lambda to keep the
// call sites close to the boundary-condition logic they inform.
namespace {

struct LEDensity {
    LaneEmdenSolution sol;
    double alpha;
    double rho_c;
    double n_poly;

    double operator()(double r) const {
        double xi = r / alpha;
        if (xi >= sol.xi_1) return 1e-20;
        auto it = std::lower_bound(sol.xi.begin(), sol.xi.end(), xi);
        int idx = static_cast<int>(it - sol.xi.begin());
        if (idx <= 0) return rho_c;
        if (idx >= static_cast<int>(sol.xi.size())) return 1e-20;
        double x0 = sol.xi[idx - 1], x1 = sol.xi[idx];
        double t0 = sol.theta_le[idx - 1], t1 = sol.theta_le[idx];
        double frac = (xi - x0) / (x1 - x0);
        double theta_val = t0 + frac * (t1 - t0);
        return rho_c * std::pow(std::max(theta_val, 1e-15), n_poly);
    }
};

LEDensity make_le_density(double n_poly, double K_poly, double rho_c, double G) {
    LEDensity le;
    le.sol = solve_lane_emden(n_poly);
    double alpha2 = (n_poly + 1.0) * K_poly
                    * std::pow(rho_c, 1.0 / n_poly - 1.0) / (4.0 * M_PI * G);
    le.alpha = std::sqrt(alpha2);
    le.rho_c = rho_c;
    le.n_poly = n_poly;
    return le;
}

double integrate_core_mass(double r_inner, const LEDensity& rho_func) {
    const int nfine = 20000;
    double dr_f = r_inner / nfine;
    double m = 0.0;
    for (int ii = 0; ii < nfine; ++ii) {
        double r = (ii + 0.5) * dr_f;
        m += 4.0 * M_PI * rho_func(r) * r * r * dr_f;
    }
    return m;
}

bool is_lane_emden_family(const std::string& test_case) {
    return test_case == "lane_emden" || test_case == "lane_emden_perturbed"
        || test_case == "bubble";
}

} // namespace

static int setup_grid(SimConfig& cfg, Grid& grid) {
    if (cfg.mesh_type == "equimass" && is_lane_emden_family(cfg.test_case)) {
        auto rho_func = make_le_density(1.5, 1.0, 1.0, cfg.G);
        grid.init_equimass(cfg.nr, cfg.ntheta, cfg.R_outer, rho_func);
        std::printf("Using equimass radial mesh based on Lane-Emden density\n");
    } else if (cfg.mesh_type == "mass" && is_lane_emden_family(cfg.test_case)) {
        auto rho_func = make_le_density(1.5, 1.0, 1.0, cfg.G);
        double r_inner = (cfg.r_inner >= 0) ? cfg.r_inner : 0.0;
        cfg.r_inner = r_inner;
        if (r_inner > 0) cfg.M_core = integrate_core_mass(r_inner, rho_func);
        grid.init_mass_shell(cfg.nr, cfg.ntheta, cfg.R_outer, rho_func, r_inner);
        cfg.no_sponge = true;
        std::printf("Using hybrid mass-shell mesh (R_star=%.6f, r_inner=%.4f, M_core=%.5f, HSE outer BC)\n",
                    cfg.R_outer, r_inner, cfg.M_core);
    } else if (cfg.mesh_type == "uniform") {
        grid.init_uniform(cfg.nr, cfg.ntheta, cfg.R_outer);
        // TEMP (sphere_impl preview): if --r-inner > 0 on uniform mesh, compute
        // M_core the same way as mass mesh so pole_avg / core_excision wire up.
        if (cfg.r_inner > 0 && (cfg.test_case == "lane_emden" || cfg.test_case == "lane_emden_perturbed")) {
            auto rho_func = make_le_density(1.5, 1.0, 1.0, cfg.G);
            cfg.M_core = integrate_core_mass(cfg.r_inner, rho_func);
            std::printf("Using uniform radial mesh with r_inner=%.4f, M_core=%.5f (sphere_impl preview)\n",
                        cfg.r_inner, cfg.M_core);
        } else {
            std::printf("Using uniform radial mesh\n");
        }
    } else {
        grid.init(cfg.nr, cfg.ntheta, cfg.R_outer, cfg.log_alpha);
    }
    return 0;
}

static int setup_eos(const SimConfig& cfg, SimContext& ctx) {
    if (cfg.eos_type == "ideal_rad") {
        ctx.eos = EOS::ideal_rad(cfg.gamma, cfg.eos_mu, cfg.eos_rad_a);
        std::printf("EOS: ideal + radiation (γ=%.3f, μ=%.3f, a=%.3e)\n",
                    cfg.gamma, cfg.eos_mu, cfg.eos_rad_a);
    } else if (cfg.eos_type == "pre_ms") {
        PreMsParams p;
        // Use reasonable defaults; override via CLI later if needed.
        p.R_gas = 1.0 / cfg.eos_mu;  // inverse μ for code-unit consistency
        p.a_rad = cfg.eos_rad_a;
        ctx.eos = EOS::pre_ms(p);
        std::printf("EOS: pre-MS Chabrier-Baraffe (T_diss=%.0f, T_ion=%.0f, μ_cold=%.2f → μ_hot=%.2f)\n",
                    p.T_diss, p.T_ion, p.mu_cold, p.mu_hot);
    } else if (cfg.eos_type == "helmholtz") {
#ifdef USE_GPU
        if (ctx.helm_tbl.load(nullptr, cfg.helm_table_path.c_str(), 0) != 0) {
            std::fprintf(stderr,
                "ERROR: failed to load Helmholtz table at %s — run\n"
                "  tools/helm_convert <ascii> %s\n",
                cfg.helm_table_path.c_str(), cfg.helm_table_path.c_str());
            return 1;
        }
        ctx.helm_tbl.view.Abar = cfg.helm_Abar;
        ctx.helm_tbl.view.Zbar = cfg.helm_Zbar;
        ctx.eos = EOS::helmholtz(ctx.helm_tbl.view);
        ctx.helm_loaded = true;
        std::printf("EOS: Helmholtz (cococubed %dx%d, Abar=%.3f, Zbar=%.3f)\n",
                    HELM_IMAX, HELM_JMAX, cfg.helm_Abar, cfg.helm_Zbar);
#else
        std::fprintf(stderr, "ERROR: --eos helmholtz requires USE_GPU build\n");
        return 1;
#endif
    } else {
        ctx.eos = EOS::ideal(cfg.gamma, cfg.eos_mu);
    }
    return 0;
}

#ifdef USE_GPU
static int setup_kap_tables(const SimConfig& cfg, SimContext& ctx) {
    if (!cfg.kap_use_table) return 0;

    char z_buf[32];
    std::snprintf(z_buf, sizeof(z_buf), "%g", cfg.kap_table_Z);
    std::string z_tag = z_buf;
    std::string high_path = cfg.kap_data_dir + "/" + cfg.kap_highT_family
                          + "_z" + z_tag + ".kapbin";
    std::string low_path  = cfg.kap_data_dir + "/" + cfg.kap_lowT_family
                          + "_z" + z_tag + ".kapbin";

    int rc_hi = ctx.kap_tbl_highT.load(high_path.c_str(), 0);
    if (rc_hi != 0) {
        std::fprintf(stderr, "ERROR: failed to load high-T kap binary at %s (rc=%d)\n",
                     high_path.c_str(), rc_hi);
        return 1;
    }
    int rc_lo = ctx.kap_tbl_lowT.load(low_path.c_str(), 0);
    if (rc_lo != 0) {
        std::fprintf(stderr, "ERROR: failed to load low-T kap binary at %s (rc=%d)\n",
                     low_path.c_str(), rc_lo);
        return 1;
    }
    ctx.kap_loaded = true;
    std::printf("KAP: stitched {%s + %s} at Z=%s, seam logT ∈ [%.2f, %.2f]\n",
                cfg.kap_highT_family.c_str(), cfg.kap_lowT_family.c_str(),
                z_tag.c_str(), cfg.kap_logT_lo_end, cfg.kap_logT_hi_start);
    return 0;
}
#endif

static int setup_ic(const SimConfig& cfg, Grid& grid, State& state) {
    state.allocate(grid);

    if (cfg.test_case == "lane_emden") {
        LaneEmdenParams lep;
        lep.n_poly = 1.5; lep.rho_c = 1.0; lep.K_poly = 1.0; lep.G = cfg.G;
        init_lane_emden(grid, state, lep, cfg.gamma);
    } else if (cfg.test_case == "lane_emden_perturbed") {
        LaneEmdenParams lep;
        lep.n_poly = 1.5; lep.rho_c = 1.0; lep.K_poly = 1.0; lep.G = cfg.G;
        init_lane_emden_perturbed(grid, state, lep, cfg.gamma, cfg.perturb_amplitude);
    } else if (cfg.test_case == "bubble") {
        LaneEmdenParams lep;
        lep.n_poly = 1.5; lep.rho_c = 1.0; lep.K_poly = 1.0; lep.G = cfg.G;
        if (cfg.bubble_mode == "entropy") {
            init_lane_emden_bubble_entropy(grid, state, lep, cfg.gamma,
                                           0.5, M_PI/3.0, 0.15, 0.5);
            std::printf("Bubble mode: entropy (constant pressure, density perturbation)\n");
        } else {
            init_lane_emden_bubble(grid, state, lep, cfg.gamma,
                                   0.5, M_PI/3.0, 0.15, 0.5);
        }
    } else if (cfg.test_case == "sedov") {
        SedovParams sp;
        sp.rho_0 = 1.0; sp.E_blast = 1.0; sp.r_blast = 0.05;
        init_sedov(grid, state, sp, cfg.gamma);
    } else if (cfg.test_case == "jeans") {
        JeansParams jp;
        jp.rho_0 = 1.0; jp.cs = 1.0; jp.G = cfg.G; jp.epsilon = 1e-3;
        jp.k_r = 2.0 * M_PI / cfg.R_outer; jp.k_theta = 2.0;
        init_jeans(grid, state, jp, cfg.gamma);
    } else if (cfg.test_case == "evrard") {
        EvrardParams ep;
        ep.M = 1.0; ep.R = 1.0; ep.G = cfg.G;
        init_evrard(grid, state, ep, cfg.gamma);
    } else if (cfg.test_case == "hse" || cfg.test_case == "hse_perturbed"
               || cfg.test_case == "hse_bubble" || cfg.test_case == "sod"
               || cfg.test_case == "kh_shear" || cfg.test_case == "forced_turb"
               || cfg.test_case == "kh_lecoanet"
               || cfg.test_case == "taylor_green"
               || cfg.test_case == "double_shear_layer"
               || cfg.test_case == "vortex_merger"
               || cfg.test_case == "quad_vortex_merger"
               || cfg.test_case == "rossby_wave"
               || cfg.test_case == "jovian_bands"
               || cfg.test_case == "sl_basis_check"
               || cfg.test_case == "sl_poisson_test"
               || cfg.test_case == "sl_poisson_test_boussinesq"
               || cfg.test_case == "kh_shear_boussinesq"
               || cfg.test_case == "gmode_pulsation"
               || cfg.test_case == "gmode_2d_evp"
               || cfg.test_case == "gmode_eigenmode_td"
               || cfg.test_case == "gmode_exp_k"
               || cfg.test_case == "dns_triad"
               || cfg.test_case == "dns_triad_coupled"
               || cfg.test_case == "local_convection"
               || cfg.test_case == "andrassy2022"
               || cfg.test_case == "sedov2d"
               || cfg.test_case == "noh"
               || cfg.test_case == "gresho"
               || cfg.test_case == "yee_vortex") {
        // Grid-less test cases — each solver branch initialises its own IC.
    } else {
        std::fprintf(stderr, "Unknown test case: %s\n", cfg.test_case.c_str());
        return 1;
    }
    return 0;
}

int setup_simulation(SimConfig& cfg, SimContext& ctx) {
    // Derive R_outer for Lane-Emden family before the mesh is built.
    if (is_lane_emden_family(cfg.test_case)) {
        if (cfg.mesh_type == "mass")
            cfg.R_outer = compute_lane_emden_R_star(1.5, 1.0, 1.0, cfg.G);
        else
            cfg.R_outer = compute_lane_emden_R_outer(1.5, 1.0, 1.0, cfg.G);
    }

    std::printf("stellar2d - 2D Axisymmetric Euler + Self-Gravity\n");
    std::printf("Test case: %s, mesh: %s\n", cfg.test_case.c_str(), cfg.mesh_type.c_str());
    std::printf("Grid: %d x %d, R_outer = %.6f\n", cfg.nr, cfg.ntheta, cfg.R_outer);

    if (int rc = setup_grid(cfg, ctx.grid); rc != 0) return rc;
    if (int rc = setup_eos(cfg, ctx); rc != 0) return rc;
#ifdef USE_GPU
    if (int rc = setup_kap_tables(cfg, ctx); rc != 0) return rc;
#endif
    if (int rc = setup_ic(cfg, ctx.grid, ctx.state); rc != 0) return rc;

    ctx.run_dir = make_run_dir(cfg);
    std::printf("Output directory: %s/\n", ctx.run_dir.c_str());

    {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/output_0000.vtk", ctx.run_dir.c_str());
        write_vtk(path, ctx.grid, ctx.state, cfg.gamma);
    }

    return 0;
}
