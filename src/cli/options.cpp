#include "cli/options.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int parse_cli(int argc, char** argv, SimConfig& cfg) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--test") == 0 && i + 1 < argc)
            cfg.test_case = argv[++i];
        else if (std::strcmp(argv[i], "--nr") == 0 && i + 1 < argc)
            cfg.nr = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ntheta") == 0 && i + 1 < argc)
            cfg.ntheta = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--tend") == 0 && i + 1 < argc)
            cfg.t_end = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--cfl") == 0 && i + 1 < argc)
            cfg.cfl = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--output-interval") == 0 && i + 1 < argc)
            cfg.output_interval = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--mesh") == 0 && i + 1 < argc)
            cfg.mesh_type = argv[++i];
        else if (std::strcmp(argv[i], "--solver") == 0 && i + 1 < argc)
            cfg.solver_type = argv[++i];
        else if (std::strcmp(argv[i], "--precond") == 0 && i + 1 < argc)
            cfg.precond = argv[++i];
        else if (std::strcmp(argv[i], "--perturb") == 0 && i + 1 < argc)
            cfg.perturb_amplitude = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--eos") == 0 && i + 1 < argc)
            cfg.eos_type = argv[++i];
        else if (std::strcmp(argv[i], "--eos-mu") == 0 && i + 1 < argc)
            cfg.eos_mu = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--eos-rad-a") == 0 && i + 1 < argc)
            cfg.eos_rad_a = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--helm-table") == 0 && i + 1 < argc)
            cfg.helm_table_path = argv[++i];
        else if (std::strcmp(argv[i], "--helm-abar") == 0 && i + 1 < argc)
            cfg.helm_Abar = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--helm-zbar") == 0 && i + 1 < argc)
            cfg.helm_Zbar = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--kap") == 0)
            cfg.kap_use_table = true;
        else if (std::strcmp(argv[i], "--kap-highT") == 0 && i + 1 < argc)
            cfg.kap_highT_family = argv[++i];
        else if (std::strcmp(argv[i], "--kap-lowT") == 0 && i + 1 < argc)
            cfg.kap_lowT_family = argv[++i];
        else if (std::strcmp(argv[i], "--kap-Z") == 0 && i + 1 < argc)
            cfg.kap_table_Z = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--kap-dir") == 0 && i + 1 < argc)
            cfg.kap_data_dir = argv[++i];
        else if (std::strcmp(argv[i], "--kap-logT-lo-end") == 0 && i + 1 < argc)
            cfg.kap_logT_lo_end = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--kap-logT-hi-start") == 0 && i + 1 < argc)
            cfg.kap_logT_hi_start = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--radiation") == 0)
            cfg.radiation_enabled = true;
        else if (std::strcmp(argv[i], "--rad-c") == 0 && i + 1 < argc)
            cfg.rad_c_light = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--nuclear") == 0)
            cfg.nuclear_enabled = true;
        else if (std::strcmp(argv[i], "--nuc-x") == 0 && i + 1 < argc)
            cfg.nuc_X = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--nuc-scale") == 0 && i + 1 < argc)
            cfg.nuc_epsilon_scale = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--nuc-t-floor") == 0 && i + 1 < argc)
            cfg.nuc_T_floor = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--nuc-y") == 0 && i + 1 < argc)
            cfg.nuc_Y = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--nuc-q") == 0 && i + 1 < argc)
            cfg.nuc_q_burn = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--nuc-t-scale") == 0 && i + 1 < argc)
            cfg.nuc_T_scale = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--species") == 0)
            cfg.species_enabled = true;
        else if (std::strcmp(argv[i], "--ic-solar") == 0)
            cfg.ic_solar = true;
        else if (std::strcmp(argv[i], "--ic-rho-c") == 0 && i + 1 < argc)
            cfg.ic_rho_c = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ic-rstar") == 0 && i + 1 < argc)
            cfg.ic_R_star = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ic-n-poly") == 0 && i + 1 < argc)
            cfg.ic_n_poly = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ic-mesa") == 0 && i + 1 < argc)
            cfg.ic_mesa_path = argv[++i];
        else if (std::strcmp(argv[i], "--ic-mesa-seed-T") == 0)
            cfg.ic_mesa_seed_T = true;
        else if (std::strcmp(argv[i], "--ic-mesa-atm-zones") == 0 && i + 1 < argc)
            cfg.ic_mesa_atm_zones = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--atm-split") == 0 && i + 1 < argc)
            cfg.atm_split = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--rich-profile") == 0)
            cfg.rich_profile = true;
        else if (std::strcmp(argv[i], "--G") == 0 && i + 1 < argc)
            cfg.G = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--implicit") == 0)
            cfg.implicit_mode = true;
        else if (std::strcmp(argv[i], "--dt-implicit") == 0 && i + 1 < argc)
            cfg.dt_implicit = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--dt-implicit-scale") == 0 && i + 1 < argc)
            cfg.dt_implicit_scale = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--no-viallet") == 0)
            cfg.no_viallet = true;
        else if (std::strcmp(argv[i], "--precond-tridiag") == 0)
            cfg.precond_tridiag = true;
        else if (std::strcmp(argv[i], "--jfnk-autodiff") == 0)
            cfg.jfnk_autodiff = true;
        else if (std::strcmp(argv[i], "--no-rhse") == 0)
            cfg.no_rhse_subtract = true;
        else if (std::strcmp(argv[i], "--newton-tol") == 0 && i + 1 < argc)
            cfg.newton_tol_override = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--hse-resnap") == 0 && i + 1 < argc)
            cfg.hse_resnap_interval = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--dt-thermal-frac") == 0 && i + 1 < argc)
            cfg.dt_thermal_frac = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--dt-mach-cap") == 0 && i + 1 < argc)
            cfg.dt_mach_cap = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--rad-T-phot-floor") == 0 && i + 1 < argc)
            cfg.rad_T_phot_floor = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--nuc-compress") == 0 && i + 1 < argc)
            cfg.nuc_compress_frac = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--mlt") == 0)
            cfg.mlt_enabled = true;
        else if (std::strcmp(argv[i], "--mlt-alpha") == 0 && i + 1 < argc)
            cfg.mlt_alpha = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--bubble-xc") == 0 && i + 1 < argc)
            cfg.bubble_xc = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--bubble-yc") == 0 && i + 1 < argc)
            cfg.bubble_yc = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--bubble-rb") == 0 && i + 1 < argc)
            cfg.bubble_rb = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--bubble-alpha") == 0 && i + 1 < argc)
            cfg.bubble_alpha = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--bubble-beta") == 0 && i + 1 < argc)
            cfg.bubble_beta = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--bubble") == 0 && i + 1 < argc) {
            // --bubble "xc,yc,rb,alpha,beta"  — may be repeated
            std::array<double, 5> b{};
            int n = std::sscanf(argv[++i], "%lf,%lf,%lf,%lf,%lf",
                                &b[0], &b[1], &b[2], &b[3], &b[4]);
            if (n < 5) {
                std::fprintf(stderr, "ERROR: --bubble needs xc,yc,rb,alpha,beta\n");
                return 1;
            }
            cfg.bubbles.push_back(b);
        }
        else if (std::strcmp(argv[i], "--limiter") == 0 && i + 1 < argc) {
            std::string lim = argv[++i];
            if (lim == "vanleer") cfg.limiter = Limiter::VAN_LEER;
            else if (lim == "mc") cfg.limiter = Limiter::MC;
            else cfg.limiter = Limiter::MINMOD;
        }
        else if (std::strcmp(argv[i], "--bubble-mode") == 0 && i + 1 < argc)
            cfg.bubble_mode = argv[++i];
        else if (std::strcmp(argv[i], "--no-sponge") == 0)
            cfg.no_sponge = true;
        else if (std::strcmp(argv[i], "--lm-hllc") == 0)
            cfg.hllc_variant = 1;
        else if (std::strcmp(argv[i], "--lhllc") == 0)
            cfg.hllc_variant = 2;
        else if (std::strcmp(argv[i], "--hllc") == 0 && i + 1 < argc) {
            std::string v = argv[++i];
            if (v == "lhllc") cfg.hllc_variant = 2;
            else if (v == "lm") cfg.hllc_variant = 1;
            else cfg.hllc_variant = 0;
        }
        else if (std::strcmp(argv[i], "--radial-only") == 0)
            cfg.radial_only = true;
        else if (std::strcmp(argv[i], "--r-inner") == 0 && i + 1 < argc)
            cfg.r_inner = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--remap-order") == 0 && i + 1 < argc)
            cfg.cart_ale_remap_order = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--remap-limiter") == 0 && i + 1 < argc)
            cfg.cart_ale_limiter = argv[++i];
        else if (std::strcmp(argv[i], "--diag-interval") == 0 && i + 1 < argc)
            cfg.diag_interval = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--vtk-interval") == 0 && i + 1 < argc)
            cfg.vtk_interval = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--frame-buffer") == 0)
            cfg.frame_buffer = true;
        else if (std::strcmp(argv[i], "--frame-headroom-mb") == 0 && i + 1 < argc)
            cfg.frame_headroom_mb = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--vtk-dt") == 0 && i + 1 < argc)
            cfg.vtk_dt = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--cq-lin") == 0 && i + 1 < argc)
            cfg.cart_ale_cq_lin = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--cq-quad") == 0 && i + 1 < argc)
            cfg.cart_ale_cq_quad = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--shear-aware-av") == 0)
            cfg.cart_ale_shear_aware = true;
        else if (std::strcmp(argv[i], "--rebuild-order") == 0 && i + 1 < argc)
            cfg.cart_ale2_rebuild_order = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--bc-x") == 0 && i + 1 < argc)
            cfg.cart_ale2_bc_x = argv[++i];
        else if (std::strcmp(argv[i], "--bc-y") == 0 && i + 1 < argc)
            cfg.cart_ale2_bc_y = argv[++i];
        else if (std::strcmp(argv[i], "--ppm") == 0)
            cfg.cart_ale2_ppm = true;
        else if (std::strcmp(argv[i], "--ppm-limiter") == 0 && i + 1 < argc)
            cfg.cart_ale2_ppm_limiter = argv[++i];
        else if (std::strcmp(argv[i], "--ppm-space") == 0 && i + 1 < argc)
            cfg.cart_ale2_ppm_space = argv[++i];
        else if (std::strcmp(argv[i], "--no-ppm-char") == 0)
            cfg.cart_ale2_ppm_char = false;
        else if (std::strcmp(argv[i], "--ppm-char") == 0)
            cfg.cart_ale2_ppm_char = true;
        else if (std::strcmp(argv[i], "--cart-ale2-kh-k") == 0 && i + 1 < argc)
            cfg.cart_ale2_kh_k = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--trace-cells") == 0 && i + 1 < argc)
            cfg.cart_ale2_trace_cells = argv[++i];
        else if (std::strcmp(argv[i], "--trace-step-cap") == 0 && i + 1 < argc)
            cfg.cart_ale2_trace_step_cap = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ic-slab") == 0 && i + 1 < argc)
            cfg.cart_ale2_slab_file = argv[++i];
        else if (std::strcmp(argv[i], "--slab-perturb") == 0 && i + 1 < argc)
            cfg.cart_ale2_slab_perturb = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--slab-seed-k") == 0 && i + 1 < argc)
            cfg.cart_ale2_slab_seed_k = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--cool-tau") == 0 && i + 1 < argc)
            cfg.cart_ale2_cool_tau = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--heat-flux") == 0 && i + 1 < argc)
            cfg.cart_ale2_heat_flux = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--heat-lsun") == 0 && i + 1 < argc)
            cfg.cart_ale2_heat_lsun = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--heat-bot-R") == 0 && i + 1 < argc)
            cfg.cart_ale2_heat_bot_R = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--heat-bot-frac") == 0 && i + 1 < argc)
            cfg.cart_ale2_heat_bot_frac = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--cool-top-frac") == 0 && i + 1 < argc)
            cfg.cart_ale2_cool_top_frac = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--andrassy-amp") == 0 && i + 1 < argc)
            cfg.cart_ale2_andrassy_amp = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--andrassy-seed") == 0 && i + 1 < argc)
            cfg.cart_ale2_andrassy_seed = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--andrassy-noise") == 0 && i + 1 < argc)
            cfg.cart_ale2_andrassy_noise = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--athena-xorder") == 0 && i + 1 < argc)
            cfg.athena_vl2_xorder = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--athena-limiter") == 0 && i + 1 < argc) {
            std::string v = argv[++i];
            if      (v == "vanleer" || v == "vl") cfg.athena_vl2_limiter = 0;
            else if (v == "minmod")               cfg.athena_vl2_limiter = 1;
            else std::fprintf(stderr, "unknown --athena-limiter %s; using vanleer\n", v.c_str());
        }
        else if (std::strcmp(argv[i], "--ps-nu") == 0 && i + 1 < argc)
            cfg.ps_nu = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-Lx") == 0 && i + 1 < argc)
            cfg.ps_Lx = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-Ly") == 0 && i + 1 < argc)
            cfg.ps_Ly = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-vshear") == 0 && i + 1 < argc)
            cfg.ps_vshear = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-k") == 0 && i + 1 < argc)
            cfg.ps_k = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-explicit") == 0)
            cfg.ps_explicit = true;
        else if (std::strcmp(argv[i], "--ps-adv-only") == 0)
            cfg.ps_adv_only = true;
        else if (std::strcmp(argv[i], "--ps-forcing-eps") == 0 && i + 1 < argc)
            cfg.ps_forcing_eps = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-forcing-kf") == 0 && i + 1 < argc)
            cfg.ps_forcing_kf = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-forcing-dk") == 0 && i + 1 < argc)
            cfg.ps_forcing_dk = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-forcing-seed") == 0 && i + 1 < argc)
            cfg.ps_forcing_seed = std::strtoull(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--ps-forcing-host-rng") == 0)
            cfg.ps_forcing_host_rng = true;
        else if (std::strcmp(argv[i], "--ps-drag") == 0 && i + 1 < argc)
            cfg.ps_drag_alpha = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-hyper") == 0 && i + 1 < argc)
            cfg.ps_hyper_p = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-conservative") == 0)
            cfg.ps_conservative = true;
        else if (std::strcmp(argv[i], "--ps-batched-fft") == 0)
            cfg.ps_batched_fft = true;
        else if (std::strcmp(argv[i], "--ps-pi") == 0)
            cfg.ps_pi_dt = true;
        else if (std::strcmp(argv[i], "--ps-tg-k") == 0 && i + 1 < argc)
            cfg.ps_tg_k = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-vm-gamma") == 0 && i + 1 < argc)
            cfg.ps_vm_gamma = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-vm-sigma") == 0 && i + 1 < argc)
            cfg.ps_vm_sigma = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-vm-dist") == 0 && i + 1 < argc)
            cfg.ps_vm_dist = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-ckpt-every") == 0 && i + 1 < argc)
            cfg.ps_ckpt_every = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ps-resume") == 0 && i + 1 < argc)
            cfg.ps_resume = argv[++i];
        else if (std::strcmp(argv[i], "--run-base") == 0 && i + 1 < argc)
            cfg.run_base = argv[++i];
        // sph2d flags
        else if (std::strcmp(argv[i], "--sph-R") == 0 && i + 1 < argc)
            cfg.sph_R = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-Omega") == 0 && i + 1 < argc)
            cfg.sph_Omega = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-nu") == 0 && i + 1 < argc)
            cfg.sph_nu = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-Lmax") == 0 && i + 1 < argc)
            cfg.sph_Lmax = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-drag") == 0 && i + 1 < argc)
            cfg.sph_drag = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-hyper") == 0 && i + 1 < argc)
            cfg.sph_hyper = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-pi") == 0)
            cfg.sph_pi_dt = true;
        else if (std::strcmp(argv[i], "--sph-rossby-l") == 0 && i + 1 < argc)
            cfg.sph_rossby_l = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-rossby-m") == 0 && i + 1 < argc)
            cfg.sph_rossby_m = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-rossby-amp") == 0 && i + 1 < argc)
            cfg.sph_rossby_amp = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-forcing-eps") == 0 && i + 1 < argc)
            cfg.sph_forcing_eps = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-forcing-lmin") == 0 && i + 1 < argc)
            cfg.sph_forcing_lmin = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-forcing-lmax") == 0 && i + 1 < argc)
            cfg.sph_forcing_lmax = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-forcing-seed") == 0 && i + 1 < argc)
            cfg.sph_forcing_seed = std::strtoull(argv[++i], nullptr, 0);
        else if (std::strcmp(argv[i], "--sph-ckpt-every") == 0 && i + 1 < argc)
            cfg.sph_ckpt_every = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--sph-resume") == 0 && i + 1 < argc)
            cfg.sph_resume = argv[++i];
    }
    return 0;
}
