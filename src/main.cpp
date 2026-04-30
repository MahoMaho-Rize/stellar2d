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
#include "gpu/gpu_solver.h"
#endif
#include "gpu/lowmach_solver.h"
#include "gpu/fas_solver.cuh"
#include "gpu/simple_solver.cuh"
#include "gpu/projection_solver.cuh"
#include "gpu/radial1d_solver.cuh"
#include "gpu/wb2d_solver.cuh"
#include "gpu/ale2d_solver.cuh"
#include "gpu/cart_lag_solver.cuh"
#endif

#include <cstdio>
#include <cmath>
#include <string>
#include <cstring>
#include <ctime>
#include <csignal>
#include <sys/stat.h>

static volatile sig_atomic_t g_interrupted = 0;
static void handle_sigint(int) { g_interrupted = 1; }

struct SimConfig {
    int nr = 128;
    int ntheta = 64;
    double R_outer = 1.0;
    double log_alpha = 2.0;
    double gamma = 5.0 / 3.0;
    double cfl = 0.4;
    double t_end = 1.0;
    int output_interval = 100;
    double G = 1.0;
    std::string test_case = "lane_emden";
    std::string mesh_type = "log";
    std::string solver_type = "compressible"; // "compressible" or "lowmach"
    std::string precond = "line_jacobi";         // preconditioner for lowmach solver
    Limiter limiter = Limiter::MINMOD;
    double perturb_amplitude = 1e-3;  // density perturbation for lane_emden_perturbed
    std::string bubble_mode = "pressure"; // "pressure" or "entropy"
    bool no_sponge = false;
    bool lm_hllc = false;
    bool radial_only = false;  // enforce v_theta=0, skip theta-direction work (FAS/explicit only)
    double r_inner = -1.0;  // auto-set for mass mesh; override with --r-inner
    double M_core = 0.0;
};

static void extract_density(const Grid& grid, const State& state, std::vector<double>& rho_cells) {
    int nr = grid.nr, nt = grid.ntheta;
    rho_cells.resize(nr * nt);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nt; ++j)
            rho_cells[i * nt + j] = state.rho[grid.idx(i, j)];
}

static double compute_lane_emden_R_star(double n_poly, double K_poly, double rho_c, double G) {
    auto sol = solve_lane_emden(n_poly);
    double alpha2 = (n_poly + 1.0) * K_poly
                    * std::pow(rho_c, 1.0 / n_poly - 1.0)
                    / (4.0 * M_PI * G);
    double alpha = std::sqrt(alpha2);
    return alpha * sol.xi_1;
}

static double compute_lane_emden_R_outer(double n_poly, double K_poly, double rho_c, double G) {
    return compute_lane_emden_R_star(n_poly, K_poly, rho_c, G) * 1.1;
}

static void print_progress(double t, double t_end, int step, double dt,
                           std::timespec& t_start) {
    double frac = t / t_end;
    int pct = static_cast<int>(frac * 100.0);
    if (pct > 100) pct = 100;

    int bar_width = 30;
    int filled = static_cast<int>(frac * bar_width);

    std::timespec t_now;
    clock_gettime(CLOCK_MONOTONIC, &t_now);
    double elapsed = (t_now.tv_sec - t_start.tv_sec)
                   + (t_now.tv_nsec - t_start.tv_nsec) * 1e-9;

    double eta = (frac > 1e-6) ? elapsed / frac * (1.0 - frac) : 0.0;

    std::fprintf(stderr, "\r  [");
    for (int i = 0; i < bar_width; ++i)
        std::fputc(i < filled ? '#' : '.', stderr);
    std::fprintf(stderr, "] %3d%%  step %-8d  t=%.3e  dt=%.2e  elapsed %.0fs  ETA %.0fs  ",
                 pct, step, t, dt, elapsed, eta);
    std::fflush(stderr);
}

// Build a traceable run directory: runs/<test>_<nr>x<nt>_<timestamp>/
// Returns the path string (e.g. "runs/lane_emden_64x32_20260428_153012/").
static std::string make_run_dir(const SimConfig& cfg) {
    std::time_t now = std::time(nullptr);
    std::tm* lt = std::localtime(&now);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", lt);

    char dirname[512];
    std::snprintf(dirname, sizeof(dirname), "runs/%s_%dx%d_%s",
                  cfg.test_case.c_str(), cfg.nr, cfg.ntheta, ts);

    // mkdir -p runs/ and runs/<subdir>/
    mkdir("runs", 0755);
    mkdir(dirname, 0755);
    return std::string(dirname);
}

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);
    std::signal(SIGTERM, handle_sigint);

    SimConfig cfg;

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
            cfg.lm_hllc = true;
        else if (std::strcmp(argv[i], "--radial-only") == 0)
            cfg.radial_only = true;
        else if (std::strcmp(argv[i], "--r-inner") == 0 && i + 1 < argc)
            cfg.r_inner = std::atof(argv[++i]);
    }

    if (cfg.test_case == "lane_emden" || cfg.test_case == "lane_emden_perturbed"
        || cfg.test_case == "bubble") {
        if (cfg.mesh_type == "mass")
            cfg.R_outer = compute_lane_emden_R_star(1.5, 1.0, 1.0, cfg.G);
        else
            cfg.R_outer = compute_lane_emden_R_outer(1.5, 1.0, 1.0, cfg.G);
    }

    std::printf("stellar2d - 2D Axisymmetric Euler + Self-Gravity\n");
    std::printf("Test case: %s, mesh: %s\n", cfg.test_case.c_str(), cfg.mesh_type.c_str());
    std::printf("Grid: %d x %d, R_outer = %.6f\n", cfg.nr, cfg.ntheta, cfg.R_outer);

    Grid grid;
    if (cfg.mesh_type == "equimass" &&
        (cfg.test_case == "lane_emden" || cfg.test_case == "lane_emden_perturbed"
         || cfg.test_case == "bubble")) {
        double n_poly = 1.5, K_poly = 1.0, rho_c = 1.0, G = cfg.G;
        auto le_sol = solve_lane_emden(n_poly);
        double alpha2 = (n_poly + 1.0) * K_poly
                        * std::pow(rho_c, 1.0 / n_poly - 1.0) / (4.0 * M_PI * G);
        double alpha = std::sqrt(alpha2);

        auto rho_func = [&](double r) -> double {
            double xi = r / alpha;
            if (xi >= le_sol.xi_1) return 1e-20;
            // Interpolate Lane-Emden solution
            auto it = std::lower_bound(le_sol.xi.begin(), le_sol.xi.end(), xi);
            int idx = static_cast<int>(it - le_sol.xi.begin());
            if (idx <= 0) return rho_c;
            if (idx >= static_cast<int>(le_sol.xi.size())) return 1e-20;
            double x0 = le_sol.xi[idx - 1], x1 = le_sol.xi[idx];
            double t0 = le_sol.theta_le[idx - 1], t1 = le_sol.theta_le[idx];
            double frac = (xi - x0) / (x1 - x0);
            double theta_val = t0 + frac * (t1 - t0);
            return rho_c * std::pow(std::max(theta_val, 1e-15), n_poly);
        };

        grid.init_equimass(cfg.nr, cfg.ntheta, cfg.R_outer, rho_func);
        std::printf("Using equimass radial mesh based on Lane-Emden density\n");
    } else if (cfg.mesh_type == "mass" &&
               (cfg.test_case == "lane_emden" || cfg.test_case == "lane_emden_perturbed"
                || cfg.test_case == "bubble")) {
        double n_poly = 1.5, K_poly = 1.0, rho_c = 1.0, G = cfg.G;
        auto le_sol = solve_lane_emden(n_poly);
        double alpha2 = (n_poly + 1.0) * K_poly
                        * std::pow(rho_c, 1.0 / n_poly - 1.0) / (4.0 * M_PI * G);
        double alpha = std::sqrt(alpha2);

        auto rho_func = [&](double r) -> double {
            double xi = r / alpha;
            if (xi >= le_sol.xi_1) return 1e-20;
            auto it = std::lower_bound(le_sol.xi.begin(), le_sol.xi.end(), xi);
            int idx = static_cast<int>(it - le_sol.xi.begin());
            if (idx <= 0) return rho_c;
            if (idx >= static_cast<int>(le_sol.xi.size())) return 1e-20;
            double x0 = le_sol.xi[idx - 1], x1 = le_sol.xi[idx];
            double t0 = le_sol.theta_le[idx - 1], t1 = le_sol.theta_le[idx];
            double frac = (xi - x0) / (x1 - x0);
            double theta_val = t0 + frac * (t1 - t0);
            return rho_c * std::pow(std::max(theta_val, 1e-15), n_poly);
        };

        double r_inner = (cfg.r_inner >= 0) ? cfg.r_inner : 0.0;
        cfg.r_inner = r_inner;

        if (r_inner > 0) {
            const int nfine = 20000;
            double dr_f = r_inner / nfine;
            double m = 0.0;
            for (int ii = 0; ii < nfine; ++ii) {
                double r = (ii + 0.5) * dr_f;
                m += 4.0 * M_PI * rho_func(r) * r * r * dr_f;
            }
            cfg.M_core = m;
        }

        grid.init_mass_shell(cfg.nr, cfg.ntheta, cfg.R_outer, rho_func, r_inner);
        cfg.no_sponge = true;
        std::printf("Using hybrid mass-shell mesh (R_star=%.6f, r_inner=%.4f, M_core=%.5f, HSE outer BC)\n",
                    cfg.R_outer, r_inner, cfg.M_core);
    } else if (cfg.mesh_type == "uniform") {
        grid.init_uniform(cfg.nr, cfg.ntheta, cfg.R_outer);
        std::printf("Using uniform radial mesh\n");
    } else {
        grid.init(cfg.nr, cfg.ntheta, cfg.R_outer, cfg.log_alpha);
    }

    EOS eos(cfg.gamma);

    State state;
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
               || cfg.test_case == "sod") {
        // Cart-Lagrangian-only test cases — no Grid/State initialization needed;
        // cart_lag solver branch handles its own IC.
    } else {
        std::fprintf(stderr, "Unknown test case: %s\n", cfg.test_case.c_str());
        return 1;
    }

    std::string run_dir = make_run_dir(cfg);
    std::printf("Output directory: %s/\n", run_dir.c_str());

    {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/output_0000.vtk", run_dir.c_str());
        write_vtk(path, grid, state, cfg.gamma);
    }

    double t = 0.0;
    int step = 0;

    std::printf("Starting time integration...\n");

#ifdef USE_GPU
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
        r1d.init_lane_emden(1.0, 1.0, 1.5);          // ρ_c=1, K=1, n=1.5
        r1d.snapshot_hse();
        if (cfg.test_case == "lane_emden_perturbed")
            r1d.apply_perturbation(cfg.perturb_amplitude);

        std::timespec wall_start;
        clock_gettime(CLOCK_MONOTONIC, &wall_start);

        // Simple text output (CSV format) for radial1d mode
        char csv_path[512];
        std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", run_dir.c_str());
        std::FILE* csv = std::fopen(csv_path, "w");
        std::fprintf(csv, "step,t,dt,mass,KE,IE,PE,total_E,max_mach,max_vr\n");

        int frame = 0;
        while (t < cfg.t_end && !g_interrupted) {
            double dt = r1d.step(t, cfg.t_end);
            t += dt;
            step++;

            if (step % 200 == 0)
                print_progress(t, cfg.t_end, step, dt, wall_start);

            if (step % cfg.output_interval == 0) {
                auto d = r1d.compute_diagnostics();
                std::fprintf(stderr, "\n");
                std::printf("Step %6d  t=%.6e dt=%.3e M=%.10e E=%.10e |v|_max=%.3e Mach_max=%.3e\n",
                            step, t, dt, d.total_mass, d.total_E, d.max_vr, d.max_mach);
                std::fprintf(csv, "%d,%.10e,%.6e,%.10e,%.10e,%.10e,%.10e,%.10e,%.6e,%.6e\n",
                             step, t, dt, d.total_mass, d.total_KE, d.total_internal_E,
                             d.total_grav_E, d.total_E, d.max_mach, d.max_vr);
                std::fflush(csv);

                // Dump profile as simple text
                std::vector<double> r_face, v_face, rho_cell, P_cell, e_cell;
                r1d.download_profile(r_face, v_face, rho_cell, P_cell, e_cell);
                char path[512];
                std::snprintf(path, sizeof(path), "%s/profile_%04d.txt", run_dir.c_str(), ++frame);
                std::FILE* fp = std::fopen(path, "w");
                std::fprintf(fp, "# t = %.10e  step = %d\n# k r_face v_face rho P e_int\n", t, step);
                for (int k = 0; k < r1d.lev.nz; ++k) {
                    std::fprintf(fp, "%d %.10e %.10e %.10e %.10e %.10e\n",
                                 k, r_face[k], v_face[k], rho_cell[k], P_cell[k], e_cell[k]);
                }
                // last face
                std::fprintf(fp, "%d %.10e %.10e - - -\n", r1d.lev.nz, r_face[r1d.lev.nz], v_face[r1d.lev.nz]);
                std::fclose(fp);
            }
        }
        std::fclose(csv);
        std::fprintf(stderr, "\n");
        r1d.destroy();
    } else if (cfg.solver_type == "projection") {
        // ===== GPU semi-implicit pressure projection =====
        ProjSolver proj;
        proj.init(grid, eos, cfg.G, cfg.cfl);
        if (cfg.no_sponge) proj.sponge_kappa = 0.0;
        if (cfg.mesh_type == "mass") {
            proj.use_hse_outer_bc = true;
            proj.use_core_excision = (cfg.r_inner > 0);
            proj.M_core = cfg.M_core;
            proj.n_pole_avg = cfg.ntheta / 2;
            if (cfg.r_inner <= 0) {
                int n_uni = static_cast<int>(0.15 * cfg.R_outer / (cfg.R_outer / cfg.nr));
                proj.n_angular_avg = n_uni;
            }
        }

        if (cfg.test_case == "lane_emden_perturbed" || cfg.test_case == "bubble") {
            State state_hse;
            state_hse.allocate(grid);
            LaneEmdenParams lep;
            lep.n_poly = 1.5; lep.rho_c = 1.0; lep.K_poly = 1.0; lep.G = cfg.G;
            init_lane_emden(grid, state_hse, lep, cfg.gamma);
            proj.upload_state(grid, state_hse);
            proj.snapshot_hse();
        }

        proj.upload_state(grid, state);

        std::timespec wall_start;
        clock_gettime(CLOCK_MONOTONIC, &wall_start);

        while (t < cfg.t_end && !g_interrupted) {
            double dt = proj.step(t, cfg.t_end);
            t += dt;
            step++;

            if (step % 200 == 0)
                print_progress(t, cfg.t_end, step, dt, wall_start);

            if (step % cfg.output_interval == 0) {
                proj.download_state(grid, state);
                Diagnostics diag = compute_diagnostics(grid, state, cfg.gamma);
                std::fprintf(stderr, "\n");
                std::printf("Step %6d  t = %.6e  dt = %.3e  M = %.10e  E = %.10e\n",
                            step, t, dt, diag.total_mass, diag.total_energy);
                char fname[512];
                std::snprintf(fname, sizeof(fname), "%s/output_%04d.vtk", run_dir.c_str(), step / cfg.output_interval);
                write_vtk(fname, grid, state, cfg.gamma);
            }
        }
        std::fprintf(stderr, "\n");
        proj.download_state(grid, state);
        proj.destroy();
    } else if (cfg.solver_type == "simple") {
        // ===== GPU SIMPLE pressure-correction solver =====
        SimpleSolver sim;
        sim.init(grid, eos, cfg.G, cfg.cfl);
        if (cfg.no_sponge) sim.sponge_kappa = 0.0;
        if (cfg.mesh_type == "mass") {
            sim.use_hse_outer_bc = true;
            sim.use_core_excision = (cfg.r_inner > 0);
            sim.M_core = cfg.M_core;
            sim.n_pole_avg = cfg.ntheta / 2;
            if (cfg.r_inner <= 0) {
                int n_uni = static_cast<int>(0.15 * cfg.R_outer / (cfg.R_outer / cfg.nr));
                sim.n_angular_avg = n_uni;
            }
        }

        if (cfg.test_case == "lane_emden_perturbed" || cfg.test_case == "bubble") {
            State state_hse;
            state_hse.allocate(grid);
            LaneEmdenParams lep;
            lep.n_poly = 1.5; lep.rho_c = 1.0; lep.K_poly = 1.0; lep.G = cfg.G;
            init_lane_emden(grid, state_hse, lep, cfg.gamma);
            sim.upload_state(grid, state_hse);
            sim.snapshot_hse();
        }

        sim.upload_state(grid, state);

        std::timespec wall_start;
        clock_gettime(CLOCK_MONOTONIC, &wall_start);

        while (t < cfg.t_end && !g_interrupted) {
            double dt = sim.step(t, cfg.t_end);
            t += dt;
            step++;

            if (step % 200 == 0)
                print_progress(t, cfg.t_end, step, dt, wall_start);

            if (step % cfg.output_interval == 0) {
                sim.download_state(grid, state);
                Diagnostics diag = compute_diagnostics(grid, state, cfg.gamma);
                std::fprintf(stderr, "\n");
                std::printf("Step %6d  t = %.6e  dt = %.3e  M = %.10e  E = %.10e\n",
                            step, t, dt, diag.total_mass, diag.total_energy);
                char fname[512];
                std::snprintf(fname, sizeof(fname), "%s/output_%04d.vtk", run_dir.c_str(), step / cfg.output_interval);
                write_vtk(fname, grid, state, cfg.gamma);
            }
        }
        std::fprintf(stderr, "\n");
        sim.download_state(grid, state);
        sim.destroy();
    } else if (cfg.solver_type == "fas" || cfg.solver_type == "explicit") {
        bool use_explicit = (cfg.solver_type == "explicit");
        // ===== GPU FAS nonlinear multigrid path =====
        FasSolver fas;
        fas.use_simple_smoother = (cfg.precond != "block_jacobi");
        fas.limiter_type = static_cast<int>(cfg.limiter);
        fas.use_lm_hllc = cfg.lm_hllc;
        fas.radial_only = cfg.radial_only;
        fas.init(grid, eos, cfg.G, cfg.cfl);
        if (cfg.radial_only)
            std::printf("Radial-only mode: v_theta=0 enforced, theta fluxes and atm_reset skipped\n");
        if (cfg.no_sponge) fas.sponge_kappa = 0.0;
        if (cfg.mesh_type == "mass") {
            fas.use_hse_outer_bc = true;
            fas.use_core_excision = (cfg.r_inner > 0);
            fas.M_core = cfg.M_core;
            fas.n_pole_avg = cfg.ntheta / 2;
            if (cfg.r_inner <= 0) {
                int n_uni = static_cast<int>(0.15 * cfg.R_outer / (cfg.R_outer / cfg.nr));
                fas.n_angular_avg = n_uni;
                fas.central_damp_r = 0.15 * cfg.R_outer;
            }
        }

        if (cfg.test_case == "lane_emden_perturbed" || cfg.test_case == "bubble") {
            State state_hse;
            state_hse.allocate(grid);
            LaneEmdenParams lep;
            lep.n_poly = 1.5; lep.rho_c = 1.0; lep.K_poly = 1.0; lep.G = cfg.G;
            init_lane_emden(grid, state_hse, lep, cfg.gamma);
            fas.upload_state(grid, state_hse);
            fas.snapshot_hse();
        }

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

        std::timespec wall_start;
        clock_gettime(CLOCK_MONOTONIC, &wall_start);

        while (t < cfg.t_end && !g_interrupted) {
            double dt = use_explicit ? fas.step_explicit(t, cfg.t_end) : fas.step(t, cfg.t_end);
            t += dt;
            step++;

            if (step % cfg.output_interval == 0 && n_snaps < max_snaps) {
                // D2D snapshot: ~0.1ms vs D2H+VTK ~100ms
                long long off = (long long)n_snaps * 4 * snap_size;
                CUDA_CHECK(cudaMemcpy(d_snap_buf + off, fl.d_rho, snap_size*sizeof(double), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_snap_buf + off + snap_size, fl.d_mr, snap_size*sizeof(double), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_snap_buf + off + 2*snap_size, fl.d_mt, snap_size*sizeof(double), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_snap_buf + off + 3*snap_size, fl.d_rhoE, snap_size*sizeof(double), cudaMemcpyDeviceToDevice));
                snap_times.push_back(t);
                snap_dts.push_back(dt);
                snap_steps.push_back(step);
                n_snaps++;
            }

            if (step % 2000 == 0) print_progress(t, cfg.t_end, step, dt, wall_start);
        }
        std::fprintf(stderr, "\n");
        if (g_interrupted)
            std::printf("\nInterrupted at step %d, t=%.6e. ", step, t);
        std::printf("Writing %d snapshots from GPU...\n", n_snaps);

        // Bulk D2H + VTK write
        std::vector<double> h_buf(4LL * snap_size);
        for (int s = 0; s < n_snaps; ++s) {
            long long off = (long long)s * 4 * snap_size;
            CUDA_CHECK(cudaMemcpy(h_buf.data(), d_snap_buf + off, 4*snap_size*sizeof(double), cudaMemcpyDeviceToHost));
            // Unpack into state
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
    } else if (cfg.solver_type == "ale2d") {
        // ===== 2D axisymmetric Lagrangian (Caramana compatible) =====
        if (cfg.test_case != "lane_emden" && cfg.test_case != "lane_emden_perturbed") {
            std::fprintf(stderr, "ERROR: ale2d currently supports lane_emden / lane_emden_perturbed only\n");
            return 1;
        }
        Ale2DSolver ale;
        ale.init(grid, eos, cfg.G, cfg.cfl);
        ale.init_lane_emden(1.0, 1.0, 1.5);
        ale.snapshot_hse();
        if (cfg.test_case == "lane_emden_perturbed")
            ale.apply_perturbation(cfg.perturb_amplitude);

        std::timespec wall_start;
        clock_gettime(CLOCK_MONOTONIC, &wall_start);

        char csv_path[512];
        std::snprintf(csv_path, sizeof(csv_path), "%s/diagnostics.csv", run_dir.c_str());
        std::FILE* csv = std::fopen(csv_path, "w");
        std::fprintf(csv, "step,t,dt,mass,KE,IE,PE,total_E,max_mach,max_v\n");

        int frame = 0;
        while (t < cfg.t_end && !g_interrupted) {
            double dt = ale.step(t, cfg.t_end);
            t += dt;
            step++;

            if (step % 200 == 0) print_progress(t, cfg.t_end, step, dt, wall_start);

            if (step % cfg.output_interval == 0) {
                auto d = ale.compute_diagnostics();
                std::fprintf(stderr, "\n");
                std::printf("Step %6d  t=%.6e dt=%.3e M=%.10e E=%.10e |v|_max=%.3e Mach_max=%.3e\n",
                            step, t, dt, d.total_mass, d.total_E, d.max_v, d.max_mach);
                std::fprintf(csv, "%d,%.10e,%.6e,%.10e,%.10e,%.10e,%.10e,%.10e,%.6e,%.6e\n",
                             step, t, dt, d.total_mass, d.total_KE, d.total_internal_E,
                             d.total_grav_E, d.total_E, d.max_mach, d.max_v);
                std::fflush(csv);

                std::vector<double> rp, rhop, Pp, ep, vrp;
                ale.download_radial_profile(rp, rhop, Pp, ep, vrp);
                char path[512];
                std::snprintf(path, sizeof(path), "%s/profile_%04d.txt", run_dir.c_str(), ++frame);
                std::FILE* fp = std::fopen(path, "w");
                std::fprintf(fp, "# t = %.10e  step = %d\n# ic r rho P e_int v_r\n", t, step);
                for (int ic = 0; ic < ale.nr; ++ic)
                    std::fprintf(fp, "%d %.10e %.10e %.10e %.10e %.10e\n",
                                 ic, rp[ic], rhop[ic], Pp[ic], ep[ic], vrp[ic]);
                std::fclose(fp);
            }
        }
        std::fclose(csv);
        std::fprintf(stderr, "\n");
        ale.destroy();
    } else if (cfg.solver_type == "wb2d") {
        // ===== Well-Balanced 2D Eulerian (MESA-stabilized) =====
        Wb2DSolver wb;
        wb.limiter_type = static_cast<int>(cfg.limiter);
        wb.use_lm_hllc = cfg.lm_hllc ? 1 : 0;
        wb.init(grid, eos, cfg.G, cfg.cfl);
        if (cfg.mesh_type == "mass") {
            wb.n_pole_avg = cfg.ntheta / 2;
            if (cfg.r_inner <= 0) {
                int n_uni = static_cast<int>(0.15 * cfg.R_outer / (cfg.R_outer / cfg.nr));
                wb.n_angular_avg = n_uni;
                wb.central_damp_r = 0.15 * cfg.R_outer;
            }
        }
        if (!cfg.no_sponge) wb.sponge_kappa = 100.0;

        if (cfg.test_case == "lane_emden_perturbed" || cfg.test_case == "bubble") {
            State state_hse;
            state_hse.allocate(grid);
            LaneEmdenParams lep;
            lep.n_poly = 1.5; lep.rho_c = 1.0; lep.K_poly = 1.0; lep.G = cfg.G;
            init_lane_emden(grid, state_hse, lep, cfg.gamma);
            wb.upload_state(grid, state_hse);
            wb.snapshot_hse();
        }
        wb.upload_state(grid, state);
        if (!(cfg.test_case == "lane_emden_perturbed" || cfg.test_case == "bubble"))
            wb.snapshot_hse();

        std::timespec wall_start;
        clock_gettime(CLOCK_MONOTONIC, &wall_start);

        while (t < cfg.t_end && !g_interrupted) {
            double dt = wb.step(t, cfg.t_end);
            t += dt;
            step++;

            if (step % 200 == 0)
                print_progress(t, cfg.t_end, step, dt, wall_start);

            if (step % cfg.output_interval == 0) {
                wb.download_state(grid, state);
                Diagnostics diag = compute_diagnostics(grid, state, cfg.gamma);
                std::fprintf(stderr, "\n");
                std::printf("Step %6d  t = %.6e  dt = %.3e  M = %.10e  E = %.10e\n",
                            step, t, dt, diag.total_mass, diag.total_energy);
                char fname[512];
                std::snprintf(fname, sizeof(fname), "%s/output_%04d.vtk",
                              run_dir.c_str(), step / cfg.output_interval);
                write_vtk(fname, grid, state, cfg.gamma);
            }
        }
        std::fprintf(stderr, "\n");
        wb.download_state(grid, state);
        wb.destroy();
    } else if (cfg.solver_type == "lowmach") {
        // ===== GPU low-Mach path =====
        PrecondType pc = PrecondType::LINE_JACOBI;
        if (cfg.precond == "none")          pc = PrecondType::NONE;
        else if (cfg.precond == "block_jacobi") pc = PrecondType::BLOCK_JACOBI;
        else if (cfg.precond == "simple")   pc = PrecondType::SIMPLE;
        else if (cfg.precond == "line_jacobi") pc = PrecondType::LINE_JACOBI;
        else if (cfg.precond == "block_schur") pc = PrecondType::BLOCK_SCHUR;
        else if (cfg.precond == "combined") pc = PrecondType::COMBINED;
        else if (cfg.precond == "pbp")      pc = PrecondType::PBP;

        LowMachSolver lm;
        lm.init(grid, eos, cfg.G, cfg.cfl, pc);

        // For perturbed ICs: snapshot unperturbed HSE first, then load perturbation.
        if (cfg.test_case == "lane_emden_perturbed" || cfg.test_case == "bubble") {
            State state_hse;
            state_hse.allocate(grid);
            LaneEmdenParams lep;
            lep.n_poly = 1.5; lep.rho_c = 1.0; lep.K_poly = 1.0; lep.G = cfg.G;
            init_lane_emden(grid, state_hse, lep, cfg.gamma);
            lm.upload_state(grid, state_hse);
            lm.snapshot_hse();
        }

        lm.upload_state(grid, state);

        std::timespec wall_start;
        clock_gettime(CLOCK_MONOTONIC, &wall_start);

        while (t < cfg.t_end && !g_interrupted) {
            double dt = lm.step(t, cfg.t_end);
            t += dt;
            step++;

            if (step % 200 == 0)
                print_progress(t, cfg.t_end, step, dt, wall_start);

            if (step % cfg.output_interval == 0) {
                lm.download_state(grid, state);
                Diagnostics diag = compute_diagnostics(grid, state, cfg.gamma);

                double max_vr = 0, max_vt = 0;
                double rho_thresh = lm.atm_rho_thresh;
                for (int i = 0; i < grid.nr; i++)
                    for (int j = 0; j < grid.ntheta; j++) {
                        int k = grid.idx(i, j);
                        if (state.rho[k] < rho_thresh) continue;
                        double rho = state.rho[k];
                        max_vr = std::max(max_vr, std::fabs(state.mr[k] / rho));
                        max_vt = std::max(max_vt, std::fabs(state.mtheta[k] / rho));
                    }

                std::fprintf(stderr, "\n");
                std::printf("Step %6d  t = %.6e  dt = %.3e  M = %.10e  E = %.10e  |vr|=%.3e |vt|=%.3e\n",
                            step, t, dt, diag.total_mass, diag.total_energy, max_vr, max_vt);

                char fname[512];
                std::snprintf(fname, sizeof(fname), "%s/output_%04d.vtk", run_dir.c_str(), step / cfg.output_interval);
                write_vtk(fname, grid, state, cfg.gamma);
            }
        }
        std::fprintf(stderr, "\n");

        lm.download_state(grid, state);
        lm.destroy();
    } else {
#ifdef USE_AMGX
        // ===== GPU compressible path (HLLC + JFNK) =====
        GpuSolver gpu;
        gpu.init(grid, eos, cfg.G, cfg.cfl, cfg.limiter);
        gpu.upload_state(grid, state);

        std::timespec wall_start;
        clock_gettime(CLOCK_MONOTONIC, &wall_start);

        while (t < cfg.t_end && !g_interrupted) {
            double dt = gpu.step(t, cfg.t_end);
            t += dt;
            step++;

            if (step % 200 == 0)
                print_progress(t, cfg.t_end, step, dt, wall_start);

            if (step % cfg.output_interval == 0) {
                gpu.download_state(grid, state);
                Diagnostics diag = compute_diagnostics(grid, state, cfg.gamma);
                std::fprintf(stderr, "\n");
                std::printf("Step %6d  t = %.6e  dt = %.3e  M = %.10e  E = %.10e\n",
                            step, t, dt, diag.total_mass, diag.total_energy);

                char fname[512];
                std::snprintf(fname, sizeof(fname), "%s/output_%04d.vtk", run_dir.c_str(), step / cfg.output_interval);
                write_vtk(fname, grid, state, cfg.gamma);
            }
        }
        std::fprintf(stderr, "\n");

        gpu.download_state(grid, state);
        gpu.destroy();
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

    std::printf("Done.\n");
    return 0;
}
