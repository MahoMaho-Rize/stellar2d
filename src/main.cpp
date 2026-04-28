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
#ifdef USE_AMGX
#include "gpu/gpu_solver.h"
#endif
#include "gpu/lowmach_solver.h"
#include "gpu/fas_solver.cuh"
#endif

#include <cstdio>
#include <cmath>
#include <string>
#include <cstring>
#include <ctime>
#include <sys/stat.h>

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
};

static void extract_density(const Grid& grid, const State& state, std::vector<double>& rho_cells) {
    int nr = grid.nr, nt = grid.ntheta;
    rho_cells.resize(nr * nt);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nt; ++j)
            rho_cells[i * nt + j] = state.rho[grid.idx(i, j)];
}

static double compute_lane_emden_R_outer(double n_poly, double K_poly, double rho_c, double G) {
    auto sol = solve_lane_emden(n_poly);
    double alpha2 = (n_poly + 1.0) * K_poly
                    * std::pow(rho_c, 1.0 / n_poly - 1.0)
                    / (4.0 * M_PI * G);
    double alpha = std::sqrt(alpha2);
    double R_star = alpha * sol.xi_1;
    return R_star * 1.1;
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
    }

    if (cfg.test_case == "lane_emden" || cfg.test_case == "lane_emden_perturbed") {
        cfg.R_outer = compute_lane_emden_R_outer(1.5, 1.0, 1.0, cfg.G);
    }

    std::printf("stellar2d - 2D Axisymmetric Euler + Self-Gravity\n");
    std::printf("Test case: %s, mesh: %s\n", cfg.test_case.c_str(), cfg.mesh_type.c_str());
    std::printf("Grid: %d x %d, R_outer = %.6f\n", cfg.nr, cfg.ntheta, cfg.R_outer);

    Grid grid;
    if (cfg.mesh_type == "equimass" &&
        (cfg.test_case == "lane_emden" || cfg.test_case == "lane_emden_perturbed")) {
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
        init_lane_emden_perturbed(grid, state, lep, cfg.gamma, 1e-3);
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
    if (cfg.solver_type == "fas") {
        // ===== GPU FAS nonlinear multigrid path =====
        FasSolver fas;
        fas.init(grid, eos, cfg.G, cfg.cfl);

        if (cfg.test_case == "lane_emden_perturbed") {
            State state_hse;
            state_hse.allocate(grid);
            LaneEmdenParams lep;
            lep.n_poly = 1.5; lep.rho_c = 1.0; lep.K_poly = 1.0; lep.G = cfg.G;
            init_lane_emden(grid, state_hse, lep, cfg.gamma);
            fas.upload_state(grid, state_hse);
            fas.snapshot_hse();
        }

        fas.upload_state(grid, state);

        while (t < cfg.t_end) {
            double dt = fas.step(t, cfg.t_end);
            t += dt;
            step++;

            if (step % cfg.output_interval == 0) {
                fas.download_state(grid, state);
                Diagnostics diag = compute_diagnostics(grid, state, cfg.gamma);
                std::fprintf(stderr, "\n");
                std::printf("Step %6d  t = %.6e  dt = %.3e  M = %.10e  E = %.10e\n",
                            step, t, dt, diag.total_mass, diag.total_energy);
            }
        }
        fas.download_state(grid, state);
        fas.destroy();
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
        if (cfg.test_case == "lane_emden_perturbed") {
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

        while (t < cfg.t_end) {
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

        while (t < cfg.t_end) {
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

    while (t < cfg.t_end) {
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
