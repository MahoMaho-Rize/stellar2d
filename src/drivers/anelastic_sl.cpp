#include "drivers/drivers.h"

#ifdef USE_GPU
#include "anelastic_sl_solver.cuh"
#include "stellar_profile.h"
#include "gpu_common.cuh"  // CUDA_CHECK
#include "sim/helpers.h"
#include "sim/run_loop.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

int run_anelastic_sl(SimConfig& cfg, SimContext& ctx, double& t, int& step) {
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
                          ctx.run_dir.c_str());
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
                          ctx.run_dir.c_str());
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
                          ctx.run_dir.c_str());
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
                          "%s/gmode_eigenmode_td.csv", ctx.run_dir.c_str());
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
                          "%s/dns_triad.csv", ctx.run_dir.c_str());
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
                          "%s/dns_triad_coupled.csv", ctx.run_dir.c_str());
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
            // (u, v, b) float32 cubes to <ctx.run_dir>/snapshots/snap_NNNN.bin
            // after every N periods; 0 disables snapshotting.
            int snap_every = env_int("ANSL_DNS_SNAP_EVERY", 0);
            std::string snap_dir = ctx.run_dir + "/snapshots";
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
            std::snprintf(csv_path, sizeof(csv_path), "%s/kh_final.csv", ctx.run_dir.c_str());
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
        std::snprintf(basis_path, sizeof(basis_path), "%s/sl_basis.csv", ctx.run_dir.c_str());
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
    return 0;
}
#endif
