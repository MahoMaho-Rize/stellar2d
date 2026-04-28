// Preconditioner capability tests.
//
// These tests diagnose WHY dt degrades, not just WHETHER code is correct.
// For each preconditioner, sweep dt and measure:
//
//   1. GMRES iterations to reach tol=1e-3
//   2. Newton residual reduction ratio after one correction
//   3. Line search alpha accepted
//
// Output is a table that directly shows each preconditioner's dt ceiling.
//
// Usage:
//   ./test_precond_quality              (default: BLOCK_JACOBI)
//   ./test_precond_quality --all        (sweep all preconditioner types)

#include "gpu/lowmach_solver.h"
#include "init/lane_emden.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>

static std::vector<double> download(const double* d_ptr, int n) {
    std::vector<double> h(n);
    cudaMemcpy(h.data(), d_ptr, n * sizeof(double), cudaMemcpyDeviceToHost);
    return h;
}

static double l2_norm(const double* d_ptr, int n) {
    auto h = download(d_ptr, n);
    double s = 0;
    for (double x : h) s += x * x;
    return std::sqrt(s);
}

// ── Core diagnostic: measure preconditioner quality at a given dt ──

struct PrecondMetrics {
    double dt;
    int gmres_iters;
    double precond_ratio;   // ||M^{-1}F|| / ||F||
    double newton_reduction; // ||F_after|| / ||F_before|| (after one correction, alpha=1)
    double ls_alpha;         // line search alpha accepted (0 = failed)
    double F_before, F_after;
};

static PrecondMetrics measure_at_dt(LowMachSolver& lm, double dt) {
    int n = lm.nr * lm.nt;
    int N4 = 4 * n;
    PrecondMetrics m;
    m.dt = dt;

    // Pack state as U^n (Backward Euler reference)
    lm.pack_state(lm.d_Un);
    lm.compute_scaling();
    lm.assemble_block_jacobi(dt);

    // Compute F(U) at current state
    lm.compute_F(lm.d_Fk, dt);
    m.F_before = l2_norm(lm.d_Fk, N4);

    if (m.F_before < 1e-30) {
        m.gmres_iters = 0;
        m.precond_ratio = 0;
        m.newton_reduction = 1.0;
        m.ls_alpha = 1.0;
        m.F_after = 0;
        return m;
    }

    // Measure preconditioner scaling: ||M^{-1}F|| / ||F||
    double *d_MF;
    cudaMalloc(&d_MF, N4 * sizeof(double));
    lm.apply_preconditioner(lm.d_Fk, d_MF, dt);
    double norm_MF = l2_norm(d_MF, N4);
    m.precond_ratio = norm_MF / m.F_before;
    cudaFree(d_MF);

    // GMRES solve: count iterations
    cudaMemset(lm.d_gmres_w, 0, N4 * sizeof(double));
    // Zero the 5th component
    cudaMemset(lm.d_gmres_w + N4, 0, n * sizeof(double));
    m.gmres_iters = lm.gmres_solve(lm.d_gmres_w, lm.d_Fk, dt, 1e-3, 60);

    // Clamp correction (MESA-style)
    lm.clamp_correction(lm.d_gmres_w, 0.1);

    // Save state before correction
    double *d_Uk_save;
    cudaMalloc(&d_Uk_save, 5 * n * sizeof(double));
    lm.pack_state(d_Uk_save);

    // Check correction magnitude
    double norm_delta = l2_norm(lm.d_gmres_w, N4);

    // Apply full-step correction (alpha=1) to fluid, Φ frozen
    lm.unpack_delta(lm.d_gmres_w, 1.0);
    lm.apply_floor();
    // No solve_gravity — Φ stays frozen, consistent with JFNK matvec

    // Compute F at corrected state with frozen Φ
    lm.compute_F(lm.d_residual_ls, dt);
    m.F_after = l2_norm(lm.d_residual_ls, N4);
    m.newton_reduction = m.F_after / m.F_before;

    // Debug: for first couple entries
    static int dbg_count = 0;
    if (dbg_count < 2) {
        std::fprintf(stderr, "    [dbg] dt=%.2e: ||F||=%.3e ||delta||=%.3e ||F'||=%.3e\n",
                     dt, m.F_before, norm_delta, m.F_after);
        dbg_count++;
    }

    // Determine effective alpha via binary search
    if (m.F_after < m.F_before) {
        m.ls_alpha = 1.0;
    } else {
        // Backtrack to find what alpha works
        m.ls_alpha = 0.0;
        for (double alpha = 0.5; alpha >= 1.0/256; alpha *= 0.5) {
            // Restore state, apply correction, Φ frozen
            lm.unpack_set(d_Uk_save);
            lm.unpack_delta(lm.d_gmres_w, alpha);
            lm.apply_floor();

            lm.compute_F(lm.d_residual_ls, dt);
            double Fa = l2_norm(lm.d_residual_ls, N4);
            if (Fa < m.F_before) {
                m.ls_alpha = alpha;
                m.F_after = Fa;
                m.newton_reduction = Fa / m.F_before;
                break;
            }
        }
    }

    // Restore original state
    lm.unpack_set(d_Uk_save);
    cudaFree(d_Uk_save);
    return m;
}

// ── Run sweep: for each target dt, can step() succeed? ────────────
// Use step() directly with forced dt_current to measure the actual
// convergence behavior including Newton iterations and dt adaptation.

static void sweep_preconditioner(PrecondType pc, const char* name,
                                 const Grid& grid, const EOS& eos,
                                 const State& state_hse,
                                 const State& state_perturbed,
                                 double G) {
    std::fprintf(stderr, "\n── Preconditioner: %s (%dx%d) ──\n", name, grid.nr, grid.ntheta);
    std::fprintf(stderr, "  Single-step test: set dt_current, call step(), see what dt is accepted.\n");
    std::fprintf(stderr, "  %-10s  %12s  %8s\n", "dt_target", "dt_accepted", "ratio");
    std::fprintf(stderr, "  %-10s  %12s  %8s\n", "----------", "------------", "--------");

    double dt_values[] = {1e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 1e-4, 1e-3};
    int n_dt = sizeof(dt_values) / sizeof(dt_values[0]);

    int pass = 0;

    for (int i = 0; i < n_dt; ++i) {
        LowMachSolver lm;
        lm.init(grid, eos, G, 0.4, pc);
        lm.upload_state(grid, state_hse);
        lm.snapshot_hse();
        lm.upload_state(grid, state_perturbed);

        lm.dt_current = dt_values[i];
        double dt_accepted = lm.step(0.0, 1.0);
        double ratio = dt_accepted / dt_values[i];

        const char* status = (ratio > 0.5) ? "" : (ratio > 0.01) ? " DEGRADED" : " COLLAPSED";
        std::fprintf(stderr, "  %.3e  %12.3e  %8.3f%s\n",
                     dt_values[i], dt_accepted, ratio, status);

        if (ratio > 0.1) pass++;

        lm.destroy();
    }
    std::fprintf(stderr, "  Summary: %d/%d dt points where accepted dt > 10%% of target\n",
                 pass, n_dt);
}

// ── Multi-step stability test ─────────────────────────────────────
// Run N steps and track dt evolution to detect collapse.

static void test_dt_stability(PrecondType pc, const char* name,
                              const Grid& grid, const EOS& eos,
                              const State& state_hse,
                              const State& state_perturbed,
                              double G, int max_steps = 20) {
    LowMachSolver lm;
    lm.init(grid, eos, G, 0.4, pc);

    lm.upload_state(grid, state_hse);
    lm.snapshot_hse();
    lm.upload_state(grid, state_perturbed);

    std::fprintf(stderr, "\n── dt stability: %s (%d steps) ──\n", name, max_steps);
    std::fprintf(stderr, "  %4s  %12s  %12s\n", "step", "dt", "dt_ratio");
    std::fprintf(stderr, "  %4s  %12s  %12s\n", "----", "------------", "------------");

    double t = 0.0;
    double dt_prev = 0.0;
    double dt_min_seen = 1e30;
    double dt_max_seen = 0.0;
    int collapse_step = -1;

    for (int s = 0; s < max_steps; ++s) {
        double dt = lm.step(t, 1e10);
        t += dt;

        double ratio = (dt_prev > 0) ? dt / dt_prev : 0;
        std::fprintf(stderr, "  %4d  %12.3e  %12.3f\n", s, dt, ratio);

        if (dt < dt_min_seen) dt_min_seen = dt;
        if (dt > dt_max_seen) dt_max_seen = dt;

        // Detect collapse: dt drops by > 100x from peak
        if (dt_max_seen > 0 && dt < dt_max_seen * 0.01 && collapse_step < 0) {
            collapse_step = s;
        }

        dt_prev = dt;
    }

    std::fprintf(stderr, "  dt range: [%.3e, %.3e]  ratio: %.1fx\n",
                 dt_min_seen, dt_max_seen, dt_max_seen / dt_min_seen);
    if (collapse_step >= 0)
        std::fprintf(stderr, "  ** dt collapse detected at step %d **\n", collapse_step);
    else
        std::fprintf(stderr, "  No dt collapse detected in %d steps\n", max_steps);

    lm.destroy();
}

// ── main ──────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    bool all = false;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--all") == 0) all = true;

    int nr = 64, nt = 32;
    double gamma = 5.0 / 3.0;
    double G = 1.0;

    LaneEmdenParams lep;
    lep.n_poly = 1.5; lep.rho_c = 1.0; lep.K_poly = 1.0; lep.G = G;

    auto sol = solve_lane_emden(lep.n_poly);
    double alpha = std::sqrt((lep.n_poly + 1.0) * lep.K_poly
        * std::pow(lep.rho_c, 1.0 / lep.n_poly - 1.0)
        / (4.0 * M_PI * lep.G));
    double R_outer = alpha * sol.xi_1 * 1.1;

    Grid grid;
    grid.init(nr, nt, R_outer, 2.0);
    EOS eos(gamma);

    State state_hse;
    state_hse.allocate(grid);
    init_lane_emden(grid, state_hse, lep, gamma);

    State state_perturbed;
    state_perturbed.allocate(grid);
    init_lane_emden_perturbed(grid, state_perturbed, lep, gamma, 1e-3);

    std::fprintf(stderr, "=== Preconditioner capability diagnostic ===\n");
    std::fprintf(stderr, "Grid: %dx%d, Lane-Emden perturbed (eps=1e-3)\n\n", nr, nt);

    if (all) {
        sweep_preconditioner(PrecondType::BLOCK_JACOBI, "BLOCK_JACOBI",
                             grid, eos, state_hse, state_perturbed, G);
        sweep_preconditioner(PrecondType::LINE_JACOBI, "LINE_JACOBI",
                             grid, eos, state_hse, state_perturbed, G);

        std::fprintf(stderr, "\n=== dt stability over 20 steps ===\n");
        test_dt_stability(PrecondType::BLOCK_JACOBI, "BLOCK_JACOBI",
                          grid, eos, state_hse, state_perturbed, G, 20);
        test_dt_stability(PrecondType::LINE_JACOBI, "LINE_JACOBI",
                          grid, eos, state_hse, state_perturbed, G, 20);
    } else {
        sweep_preconditioner(PrecondType::BLOCK_JACOBI, "BLOCK_JACOBI",
                             grid, eos, state_hse, state_perturbed, G);
        sweep_preconditioner(PrecondType::LINE_JACOBI, "LINE_JACOBI",
                             grid, eos, state_hse, state_perturbed, G);
        test_dt_stability(PrecondType::LINE_JACOBI, "LINE_JACOBI",
                          grid, eos, state_hse, state_perturbed, G, 20);
    }

    std::fprintf(stderr, "\n=== Done ===\n");
    return 0;
}
