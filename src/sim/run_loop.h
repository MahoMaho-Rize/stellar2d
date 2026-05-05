#pragma once

#include "grid.h"
#include "state.h"
#include "cli/options.h"
#include "sim/setup.h"
#include "sim/helpers.h"
#include "init/lane_emden.h"
#include "io/output.h"

#include <csignal>
#include <cstdio>
#include <ctime>
#include <functional>
#include <string>

// SIGINT / SIGTERM set this so the time loop can bail out cleanly.
extern volatile std::sig_atomic_t g_interrupted;

#ifdef USE_GPU
// Type-erased callbacks the generic time loop invokes each step. Each solver
// driver fills these in before handing the struct to run_time_loop().
struct SolverOps {
    std::function<double(double, double)> step;     // (t, t_end) → dt_actual
    std::function<void(const Grid&, State&, double dt)> download;
    std::function<void()> destroy;
    int progress_interval = 200;
};

// Snapshot the HSE reference state into the solver for well-balancing.
// Only fires for lane_emden_perturbed / bubble; other cases are no-ops.
template <typename Solver>
inline void snapshot_hse_if_needed(const SimConfig& cfg, SimContext& ctx,
                                   Solver& solver) {
    if (cfg.test_case == "lane_emden_perturbed" || cfg.test_case == "bubble") {
        State state_hse;
        state_hse.allocate(ctx.grid);
        LaneEmdenParams lep;
        lep.n_poly = 1.5; lep.rho_c = 1.0; lep.K_poly = 1.0; lep.G = cfg.G;
        init_lane_emden(ctx.grid, state_hse, lep, cfg.gamma);
        solver.upload_state(ctx.grid, state_hse);
        solver.snapshot_hse();
    }
}

// Wire up HSE outer BC / core excision / pole averaging for mass-shell or
// uniform+r_inner meshes. No-op for log / equimass meshes.
template <typename Solver>
inline void configure_mass_mesh(const SimConfig& cfg, Solver& solver) {
    // TEMP: extend to uniform mesh with --r-inner > 0 (sphere_impl preview).
    bool is_mass = (cfg.mesh_type == "mass");
    bool is_uniform_with_rinner = (cfg.mesh_type == "uniform" && cfg.r_inner > 0);
    if (!is_mass && !is_uniform_with_rinner) return;
    solver.use_hse_outer_bc = true;
    solver.use_core_excision = (cfg.r_inner > 0);
    solver.M_core = cfg.M_core;
    solver.n_pole_avg = cfg.ntheta / 2;
    if (cfg.r_inner <= 0) {
        int n_uni = static_cast<int>(0.15 * cfg.R_outer / (cfg.R_outer / cfg.nr));
        solver.n_angular_avg = n_uni;
    }
}

// Generic explicit time-stepping loop driven by SolverOps callbacks.
// Uses `t` and `step` by reference so the caller can inspect the final state;
// handles periodic VTK dumps and stderr progress bar.
void run_time_loop(const SimConfig& cfg, SimContext& ctx,
                   double& t, int& step, SolverOps& ops);
#endif
