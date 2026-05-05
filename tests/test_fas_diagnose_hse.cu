// ============================================================
// test_fas_diagnose_hse.cu — Diagnose explicit solver HSE drift
//
// After ONE RK2 step of pure Lane-Emden HSE:
//   1. WHERE is max|vr|? (which radial shell, what ρ?)
//   2. What is the residual R(U_hse) BEFORE any time stepping?
//   3. Does the WB residual vanish? Or is there a systematic defect?
// ============================================================

#include "fas_solver.cuh"
#include "fas_hllc.cuh"
#include "init/lane_emden.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

static inline int h_idx(int i, int j, int nt, int ng) {
    return (i + ng) * (nt + 2*ng) + (j + ng);
}

int main()
{
    std::printf("=== Diagnose explicit HSE drift ===\n\n");
    double gamma = 5.0/3.0;

    int nr = 64, nt = 32;
    auto le = solve_lane_emden(1.5);
    double alpha2 = 2.5 * 1.0 * std::pow(1.0, -1.0/3.0) / (4*M_PI);
    double R_outer = std::sqrt(alpha2) * le.xi_1 * 1.1;

    Grid grid;
    grid.init(nr, nt, R_outer, 2.0);
    EOS eos(gamma);

    FasSolver fas;
    fas.init(grid, eos, 1.0, 0.4);

    State state, state_hse;
    state.allocate(grid); state_hse.allocate(grid);
    LaneEmdenParams lep; lep.n_poly=1.5; lep.rho_c=1.0; lep.K_poly=1.0; lep.G=1.0;
    init_lane_emden(grid, state, lep, gamma);
    init_lane_emden(grid, state_hse, lep, gamma);
    fas.upload_state(grid, state_hse);
    fas.snapshot_hse();
    fas.upload_state(grid, state);

    // ---- 1. Check initial state: vr should be exactly 0 ----
    std::printf("[1] Initial state check\n");
    double max_mr = 0;
    for (int i = 0; i < nr; i++)
        for (int j = 0; j < nt; j++) {
            int k = grid.idx(i, j);
            max_mr = std::fmax(max_mr, std::fabs(state.mr[k]));
        }
    std::printf("  max|mr| at t=0: %.4e (should be 0)\n\n", max_mr);

    // ---- 2. Take ONE explicit step ----
    double dt = fas.step_explicit(0.0, 100.0);
    fas.download_state(grid, state);
    std::printf("[2] After 1 step (dt = %.4e):\n", dt);

    // Find WHERE max|vr| occurs
    double max_vr = 0;
    int max_i = -1, max_j = -1;
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < nt; j++) {
            int k = grid.idx(i, j);
            double rho = std::fmax(state.rho[k], 1e-30);
            double vr = std::fabs(state.mr[k] / rho);
            if (vr > max_vr) {
                max_vr = vr; max_i = i; max_j = j;
            }
        }
    }
    int kmax = grid.idx(max_i, max_j);
    std::printf("  max|vr| = %.6e at cell (%d, %d)\n", max_vr, max_i, max_j);
    std::printf("  r = %.6f  θ = %.6f  ρ = %.6e  mr = %.6e\n",
                grid.r_center[max_i], grid.theta_center[max_j],
                state.rho[kmax], state.mr[kmax]);

    // ---- 3. Radial profile of vr at θ = π/2 (equator) ----
    std::printf("\n[3] Radial profile of vr at equator (j=%d):\n", nt/2);
    std::printf("  %4s  %10s  %10s  %12s  %12s  %12s\n",
                "i", "r", "rho", "mr", "vr", "|vr|/dt");
    for (int i = 0; i < nr; i += std::max(1, nr/20)) {
        int k = grid.idx(i, nt/2);
        double rho = std::fmax(state.rho[k], 1e-30);
        double vr = state.mr[k] / rho;
        std::printf("  %4d  %10.6f  %10.4e  %12.4e  %12.4e  %12.4e\n",
                    i, grid.r_center[i], state.rho[k], state.mr[k], vr, std::fabs(vr)/dt);
    }

    // ---- 4. Check if issue is at specific density threshold ----
    std::printf("\n[4] max|vr| by density bin:\n");
    double bins[] = {1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.1};
    for (int b = 0; b < 5; b++) {
        double rho_lo = bins[b], rho_hi = bins[b+1];
        double mv = 0;
        int cnt = 0;
        for (int i = 0; i < nr; i++)
            for (int j = 0; j < nt; j++) {
                int k = grid.idx(i, j);
                if (state.rho[k] >= rho_lo && state.rho[k] < rho_hi) {
                    double vr = std::fabs(state.mr[k] / state.rho[k]);
                    mv = std::fmax(mv, vr);
                    cnt++;
                }
            }
        std::printf("  ρ ∈ [%.0e, %.0e): %5d cells, max|vr| = %.4e, |vr|/dt = %.4e\n",
                    rho_lo, rho_hi, cnt, mv, mv/dt);
    }

    // ---- 5. Compare with Strang solver HSE ----
    std::printf("\n[5] Summary:\n");
    std::printf("  dt = %.4e\n", dt);
    std::printf("  max|vr| = %.4e  →  accel = %.4e  (g ≈ 1)\n",
                max_vr, max_vr/dt);
    std::printf("  If accel >> g, there's a WB/geometric source term bug.\n");

    fas.destroy();
    return 0;
}
