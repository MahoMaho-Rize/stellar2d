// Solver deep-diagnosis tests.
//
// Each test isolates ONE factor that could limit dt, measuring its
// contribution quantitatively. Tests are designed to answer specific
// questions, not just pass/fail.
//
// D1. Line search profile: ||F(U+αδU)|| vs α — is the direction descent?
// D2. GMRES residual: after solve, does J·x ≈ -F hold?
// D3. JFNK epsilon sensitivity: does the FD step size affect J·v quality?
// D4. Correction clamping: how much does clamping modify the GMRES output?
// D5. Block Jacobi vs no preconditioner: does BJ actually help GMRES?
// D6. Per-equation breakdown: which equation's residual dominates after correction?
// D7. Transition dt anatomy: compare state changes at dt=1e-7 vs dt=5e-7

#include "lowmach_solver.h"
#include "init/lane_emden.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

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

static double l2_norm_h(const std::vector<double>& v, int off, int n) {
    double s = 0;
    for (int i = off; i < off + n; ++i) s += v[i] * v[i];
    return std::sqrt(s);
}

struct Fixture {
    Grid grid;
    EOS eos;
    State state_hse, state_pert;
    LaneEmdenParams lep;
    int nr = 64, nt = 32;

    Fixture() : eos(5.0 / 3.0) {
        lep.n_poly = 1.5; lep.rho_c = 1.0; lep.K_poly = 1.0; lep.G = 1.0;
        auto sol = solve_lane_emden(lep.n_poly);
        double alpha = std::sqrt((lep.n_poly + 1.0) * lep.K_poly
            * std::pow(lep.rho_c, 1.0 / lep.n_poly - 1.0)
            / (4.0 * M_PI * lep.G));
        grid.init(nr, nt, alpha * sol.xi_1 * 1.1, 2.0);
        state_hse.allocate(grid);
        init_lane_emden(grid, state_hse, lep, eos.gamma);
        state_pert.allocate(grid);
        init_lane_emden_perturbed(grid, state_pert, lep, eos.gamma, 1e-3);
    }

    // Returns a fully initialized solver with Un = U_perturbed (ready for Newton)
    LowMachSolver make_solver(PrecondType pc = PrecondType::BLOCK_JACOBI) {
        LowMachSolver lm;
        lm.init(grid, eos, lep.G, 0.4, pc);
        lm.upload_state(grid, state_hse);
        lm.snapshot_hse();
        lm.upload_state(grid, state_pert);
        // Critical: pack current state as Un (Backward Euler reference)
        lm.pack_state(lm.d_Un);
        return lm;
    }
};

// ── D1: Line search profile ──────────────────────────────────────
// For a given dt, compute ||F(U + α·δU)||_fluid for α in [0..1].
// If the direction is correct, there should be a clear minimum at α>0.

static void test_d1_line_search_profile() {
    Fixture f;
    int n = f.nr * f.nt, N4 = 4 * n;

    double dt_values[] = {1e-7, 2e-7, 5e-7, 1e-6};
    int n_dt = 4;

    for (int d = 0; d < n_dt; ++d) {
        double dt = dt_values[d];
        LowMachSolver lm = f.make_solver();
        lm.pack_state(lm.d_Un);
        lm.compute_scaling();
        lm.assemble_block_jacobi(dt);
        lm.compute_F(lm.d_Fk, dt);

        double F0 = l2_norm(lm.d_Fk, N4);

        cudaMemset(lm.d_gmres_w, 0, N4 * sizeof(double));
        cudaMemset(lm.d_gmres_w + N4, 0, n * sizeof(double));
        int iters = lm.gmres_solve(lm.d_gmres_w, lm.d_Fk, dt, 1e-3, 60);

        // Don't clamp — test raw GMRES direction
        double *d_save;
        cudaMalloc(&d_save, 5 * n * sizeof(double));
        lm.pack_state(d_save);

        // Sanity: recompute F immediately and verify it matches F0
        lm.compute_F(lm.d_residual_ls, dt);
        double F_check = l2_norm(lm.d_residual_ls, N4);
        std::fprintf(stderr, "  D1 dt=%.0e (GMRES %d): F0=%.3e  F_recheck=%.3e  (ratio=%.3e)\n",
                     dt, iters, F0, F_check, F_check / F0);

        std::fprintf(stderr, "    alpha vs ||F||/||F0||:\n    ");

        // Verify round-trip: pack → unpack → compute_F should equal F0
        // Test the pack/unpack first
        double *d_save2;
        cudaMalloc(&d_save2, 5 * n * sizeof(double));
        lm.pack_state(d_save2);  // pack current state
        lm.unpack_set(d_save2);  // unpack it back
        lm.compute_F(lm.d_residual_ls, dt);
        double F_roundtrip = l2_norm(lm.d_residual_ls, N4);
        std::fprintf(stderr, "    F0=%.3e  F(roundtrip)=%.3e\n", F0, F_roundtrip);

        // Now test after GMRES
        lm.unpack_set(d_save);
        lm.compute_F(lm.d_residual_ls, dt);
        double F_after_gmres = l2_norm(lm.d_residual_ls, N4);
        std::fprintf(stderr, "    F(from d_save after GMRES)=%.3e\n", F_after_gmres);

        // Check d_save vs d_save2
        auto h_save = download(d_save, 5 * n);
        auto h_save2 = download(d_save2, 5 * n);
        double max_diff = 0;
        int max_diff_idx = 0;
        for (int i = 0; i < 5 * n; ++i) {
            double d = std::fabs(h_save[i] - h_save2[i]);
            if (d > max_diff) { max_diff = d; max_diff_idx = i; }
        }
        int eq = max_diff_idx / n;
        const char* eq_names[] = {"rho", "mr", "mt", "rhoE", "Phi"};
        std::fprintf(stderr, "    max|d_save - d_save2| = %.3e in %s[%d]\n",
                     max_diff, eq_names[eq], max_diff_idx % n);
        cudaFree(d_save2);

        double alpha_best = 0;
        double F_best = F0;
        for (int a = 0; a <= 20; ++a) {
            double alpha = a * 0.05;
            lm.unpack_set(d_save);
            lm.unpack_delta(lm.d_gmres_w, alpha);
            lm.apply_floor();
            lm.compute_F(lm.d_residual_ls, dt);
            double Fa = l2_norm(lm.d_residual_ls, N4);
            std::fprintf(stderr, "%.2f:%.2f ", alpha, Fa / F0);
            if (Fa < F_best) { F_best = Fa; alpha_best = alpha; }
        }
        std::fprintf(stderr, "\n    optimal: alpha=%.2f, ||F||/||F0||=%.3f\n",
                     alpha_best, F_best / F0);

        lm.unpack_set(d_save);
        cudaFree(d_save);
        lm.destroy();
    }
}

// ── D2: GMRES residual accuracy ──────────────────────────────────
// After GMRES solve, check ||J·x + F|| / ||F||.
// This tells us if GMRES actually found a good approximate solution.

static void test_d2_gmres_residual() {
    Fixture f;
    int n = f.nr * f.nt, N4 = 4 * n;

    double dt_values[] = {1e-8, 1e-7, 2e-7, 5e-7, 1e-6};

    std::fprintf(stderr, "  D2: GMRES residual ||Jx+F||/||F|| after solve:\n");
    std::fprintf(stderr, "    %-10s  %5s  %12s\n", "dt", "iters", "rel_residual");

    for (double dt : dt_values) {
        LowMachSolver lm = f.make_solver();
        lm.pack_state(lm.d_Un);
        lm.compute_scaling();
        lm.assemble_block_jacobi(dt);
        lm.compute_F(lm.d_Fk, dt);
        double F0 = l2_norm(lm.d_Fk, N4);

        cudaMemset(lm.d_gmres_w, 0, N4 * sizeof(double));
        cudaMemset(lm.d_gmres_w + N4, 0, n * sizeof(double));
        int iters = lm.gmres_solve(lm.d_gmres_w, lm.d_Fk, dt, 1e-3, 60);

        // Compute J·x
        double *d_Jx;
        cudaMalloc(&d_Jx, N4 * sizeof(double));
        lm.jfnk_matvec(lm.d_gmres_w, d_Jx, dt);

        // J·x + F
        auto Jx = download(d_Jx, N4);
        auto Fk = download(lm.d_Fk, N4);
        double res_sq = 0;
        for (int i = 0; i < N4; ++i) {
            double r = Jx[i] + Fk[i];
            res_sq += r * r;
        }
        double rel_res = std::sqrt(res_sq) / F0;
        std::fprintf(stderr, "    %.3e  %5d  %12.3e\n", dt, iters, rel_res);

        cudaFree(d_Jx);
        lm.destroy();
    }
}

// ── D3: JFNK epsilon sensitivity ─────────────────────────────────
// Compare J·v at different FD epsilon values.
// If results vary wildly with epsilon, the default Walker-Pernice
// formula may be giving a bad epsilon.

static void test_d3_epsilon_sensitivity() {
    Fixture f;
    int n = f.nr * f.nt, N4 = 4 * n;
    double dt = 1e-7;

    LowMachSolver lm = f.make_solver();
    lm.pack_state(lm.d_Un);
    lm.assemble_block_jacobi(dt);
    lm.compute_F(lm.d_Fk, dt);

    // Create a test vector
    std::vector<double> h_v(N4);
    for (int i = 0; i < N4; ++i) h_v[i] = std::sin(1.0 + i * 0.7) * 1e-3;
    double *d_v;
    cudaMalloc(&d_v, N4 * sizeof(double));
    cudaMemcpy(d_v, h_v.data(), N4 * sizeof(double), cudaMemcpyHostToDevice);

    // Default matvec (Walker-Pernice epsilon)
    double *d_Jv_ref;
    cudaMalloc(&d_Jv_ref, N4 * sizeof(double));
    lm.jfnk_matvec(d_v, d_Jv_ref, dt);
    auto Jv_ref = download(d_Jv_ref, N4);
    double norm_ref = l2_norm(d_Jv_ref, N4);

    // Manual matvec with explicit epsilon values
    std::fprintf(stderr, "  D3: JFNK J*v sensitivity to epsilon:\n");
    std::fprintf(stderr, "    Walker-Pernice: ||J*v|| = %.3e\n", norm_ref);

    double eps_values[] = {1e-12, 1e-10, 1e-8, 1e-6, 1e-4};
    for (double eps : eps_values) {
        // Save state
        double *d_save;
        cudaMalloc(&d_save, 5 * n * sizeof(double));
        lm.pack_state(d_save);

        // Perturb
        lm.unpack_delta(d_v, eps);
        lm.apply_floor();
        lm.compute_F(lm.d_residual_ls, dt);

        // (F(U+εv) - F(U)) / ε
        auto F_pert = download(lm.d_residual_ls, N4);
        auto F_base = download(lm.d_Fk, N4);
        double diff_sq = 0;
        for (int i = 0; i < N4; ++i) {
            double Jvi = (F_pert[i] - F_base[i]) / eps;
            double d = Jvi - Jv_ref[i];
            diff_sq += d * d;
        }
        double rel = std::sqrt(diff_sq) / norm_ref;
        std::fprintf(stderr, "    eps=%.0e: ||J_eps*v - J_wp*v|| / ||J_wp*v|| = %.3e\n", eps, rel);

        lm.unpack_set(d_save);
        cudaFree(d_save);
    }

    cudaFree(d_v);
    cudaFree(d_Jv_ref);
    lm.destroy();
}

// ── D4: Clamping impact ──────────────────────────────────────────
// How much does MESA-style clamping modify the GMRES output?

static void test_d4_clamping_impact() {
    Fixture f;
    int n = f.nr * f.nt, N4 = 4 * n;

    double dt_values[] = {1e-8, 1e-7, 5e-7, 1e-6};

    std::fprintf(stderr, "  D4: Correction clamping impact:\n");
    std::fprintf(stderr, "    %-10s  %12s  %12s  %8s\n",
                 "dt", "||delta_raw||", "||delta_clamp||", "ratio");

    for (double dt : dt_values) {
        LowMachSolver lm = f.make_solver();
        lm.pack_state(lm.d_Un);
        lm.compute_scaling();
        lm.assemble_block_jacobi(dt);
        lm.compute_F(lm.d_Fk, dt);

        cudaMemset(lm.d_gmres_w, 0, N4 * sizeof(double));
        cudaMemset(lm.d_gmres_w + N4, 0, n * sizeof(double));
        lm.gmres_solve(lm.d_gmres_w, lm.d_Fk, dt, 1e-3, 60);

        double norm_raw = l2_norm(lm.d_gmres_w, N4);

        lm.clamp_correction(lm.d_gmres_w, 0.1);
        double norm_clamp = l2_norm(lm.d_gmres_w, N4);

        std::fprintf(stderr, "    %.3e  %12.3e  %12.3e  %8.3f\n",
                     dt, norm_raw, norm_clamp, norm_clamp / norm_raw);
        lm.destroy();
    }
}

// ── D5: Block Jacobi vs Identity preconditioner ──────────────────
// Does BJ actually help? Compare GMRES convergence with and without.

static void test_d5_bj_vs_none() {
    Fixture f;
    int n = f.nr * f.nt, N4 = 4 * n;

    double dt_values[] = {1e-8, 1e-7, 5e-7, 1e-6};

    std::fprintf(stderr, "  D5: Block Jacobi vs no preconditioner:\n");
    std::fprintf(stderr, "    %-10s  %6s  %6s\n", "dt", "BJ", "NONE");

    for (double dt : dt_values) {
        int iters_bj, iters_none;

        {
            LowMachSolver lm = f.make_solver(PrecondType::BLOCK_JACOBI);
            lm.pack_state(lm.d_Un);
            lm.compute_scaling();
            lm.assemble_block_jacobi(dt);
            lm.compute_F(lm.d_Fk, dt);
            cudaMemset(lm.d_gmres_w, 0, N4 * sizeof(double));
            cudaMemset(lm.d_gmres_w + N4, 0, n * sizeof(double));
            iters_bj = lm.gmres_solve(lm.d_gmres_w, lm.d_Fk, dt, 1e-3, 60);
            lm.destroy();
        }
        {
            LowMachSolver lm = f.make_solver(PrecondType::NONE);
            lm.pack_state(lm.d_Un);
            lm.compute_scaling();
            lm.assemble_block_jacobi(dt);
            lm.compute_F(lm.d_Fk, dt);
            cudaMemset(lm.d_gmres_w, 0, N4 * sizeof(double));
            cudaMemset(lm.d_gmres_w + N4, 0, n * sizeof(double));
            iters_none = lm.gmres_solve(lm.d_gmres_w, lm.d_Fk, dt, 1e-3, 60);
            lm.destroy();
        }

        std::fprintf(stderr, "    %.3e  %6d  %6d\n", dt, iters_bj, iters_none);
    }
}

// ── D6: Per-equation residual after correction ───────────────────
// Which equation's residual explodes when the correction fails?

static void test_d6_per_equation_breakdown() {
    Fixture f;
    int n = f.nr * f.nt, N4 = 4 * n;

    double dt_values[] = {1e-7, 5e-7};

    for (double dt : dt_values) {
        LowMachSolver lm = f.make_solver();
        lm.pack_state(lm.d_Un);
        lm.compute_scaling();
        lm.assemble_block_jacobi(dt);
        lm.compute_F(lm.d_Fk, dt);

        auto F0 = download(lm.d_Fk, N4);
        double F0_rho = l2_norm_h(F0, 0, n);
        double F0_mr  = l2_norm_h(F0, n, n);
        double F0_mt  = l2_norm_h(F0, 2*n, n);
        double F0_rE  = l2_norm_h(F0, 3*n, n);

        cudaMemset(lm.d_gmres_w, 0, N4 * sizeof(double));
        cudaMemset(lm.d_gmres_w + N4, 0, n * sizeof(double));
        lm.gmres_solve(lm.d_gmres_w, lm.d_Fk, dt, 1e-3, 60);

        double *d_save;
        cudaMalloc(&d_save, 5 * n * sizeof(double));
        lm.pack_state(d_save);

        lm.unpack_delta(lm.d_gmres_w, 1.0);
        lm.apply_floor();
        lm.compute_F(lm.d_residual_ls, dt);
        auto F1 = download(lm.d_residual_ls, N4);
        double F1_rho = l2_norm_h(F1, 0, n);
        double F1_mr  = l2_norm_h(F1, n, n);
        double F1_mt  = l2_norm_h(F1, 2*n, n);
        double F1_rE  = l2_norm_h(F1, 3*n, n);

        std::fprintf(stderr, "  D6 dt=%.0e: per-equation ||F|| before -> after (ratio):\n", dt);
        std::fprintf(stderr, "    rho:  %10.3e -> %10.3e  (%.3f)\n", F0_rho, F1_rho, F1_rho/(F0_rho+1e-30));
        std::fprintf(stderr, "    mr:   %10.3e -> %10.3e  (%.3f)\n", F0_mr, F1_mr, F1_mr/(F0_mr+1e-30));
        std::fprintf(stderr, "    mt:   %10.3e -> %10.3e  (%.3f)\n", F0_mt, F1_mt, F1_mt/(F0_mt+1e-30));
        std::fprintf(stderr, "    rhoE: %10.3e -> %10.3e  (%.3f)\n", F0_rE, F1_rE, F1_rE/(F0_rE+1e-30));

        lm.unpack_set(d_save);
        cudaFree(d_save);
        lm.destroy();
    }
}

// ── D7: Transition anatomy ───────────────────────────────────────
// At dt=1e-7 (works) vs dt=5e-7 (fails), compare:
// - ||δU|| per variable
// - max |δU|/|U| per variable
// - which cells have the largest corrections

static void test_d7_transition_anatomy() {
    Fixture f;
    int n = f.nr * f.nt, N4 = 4 * n;

    double dt_values[] = {1e-7, 5e-7};

    for (double dt : dt_values) {
        LowMachSolver lm = f.make_solver();
        lm.pack_state(lm.d_Un);
        lm.compute_scaling();
        lm.assemble_block_jacobi(dt);
        lm.compute_F(lm.d_Fk, dt);

        cudaMemset(lm.d_gmres_w, 0, N4 * sizeof(double));
        cudaMemset(lm.d_gmres_w + N4, 0, n * sizeof(double));
        int iters = lm.gmres_solve(lm.d_gmres_w, lm.d_Fk, dt, 1e-3, 60);

        auto delta = download(lm.d_gmres_w, N4);
        auto Un = download(lm.d_Un, N4);

        double d_rho = l2_norm_h(delta, 0, n);
        double d_mr  = l2_norm_h(delta, n, n);
        double d_mt  = l2_norm_h(delta, 2*n, n);
        double d_rE  = l2_norm_h(delta, 3*n, n);

        // Max relative correction per variable
        double max_rel_rho = 0, max_rel_mr = 0, max_rel_rE = 0;
        int cell_rho = -1, cell_mr = -1, cell_rE = -1;
        for (int i = 0; i < n; ++i) {
            double s_rho = std::max(1.0, std::fabs(Un[i]));
            double r = std::fabs(delta[i]) / s_rho;
            if (r > max_rel_rho) { max_rel_rho = r; cell_rho = i; }

            double s_mr = std::max(1.0, std::fabs(Un[n+i]));
            r = std::fabs(delta[n+i]) / s_mr;
            if (r > max_rel_mr) { max_rel_mr = r; cell_mr = i; }

            double s_rE = std::max(1.0, std::fabs(Un[3*n+i]));
            r = std::fabs(delta[3*n+i]) / s_rE;
            if (r > max_rel_rE) { max_rel_rE = r; cell_rE = i; }
        }

        std::fprintf(stderr, "  D7 dt=%.0e (GMRES %d):\n", dt, iters);
        std::fprintf(stderr, "    ||δρ||=%.3e  ||δmr||=%.3e  ||δmt||=%.3e  ||δρe||=%.3e\n",
                     d_rho, d_mr, d_mt, d_rE);
        std::fprintf(stderr, "    max|δρ/s|=%.3e at cell(%d,%d)\n",
                     max_rel_rho, cell_rho/f.nt, cell_rho%f.nt);
        std::fprintf(stderr, "    max|δmr/s|=%.3e at cell(%d,%d)\n",
                     max_rel_mr, cell_mr/f.nt, cell_mr%f.nt);
        std::fprintf(stderr, "    max|δρe/s|=%.3e at cell(%d,%d)\n",
                     max_rel_rE, cell_rE/f.nt, cell_rE%f.nt);

        lm.destroy();
    }
}

int main() {
    std::fprintf(stderr, "=== Solver deep diagnosis ===\n\n");

    test_d1_line_search_profile();
    std::fprintf(stderr, "\n");
    test_d2_gmres_residual();
    std::fprintf(stderr, "\n");
    test_d3_epsilon_sensitivity();
    std::fprintf(stderr, "\n");
    test_d4_clamping_impact();
    std::fprintf(stderr, "\n");
    test_d5_bj_vs_none();
    std::fprintf(stderr, "\n");
    test_d6_per_equation_breakdown();
    std::fprintf(stderr, "\n");
    test_d7_transition_anatomy();

    std::fprintf(stderr, "\n=== Done ===\n");
    return 0;
}
