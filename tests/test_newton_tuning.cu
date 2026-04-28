// Automated Newton-Krylov parameter sweep.
//
// Systematically varies GMRES/Newton parameters and measures dt_accepted
// for a fixed target dt. Identifies which parameters actually matter.
//
// Parameters swept:
//   - GMRES tolerance (forcing term η): 1e-1, 1e-2, 1e-3, 1e-4
//   - GMRES max inner iterations: 5, 10, 20, 40, 60
//   - Clamp factor: 0.01, 0.05, 0.1, 0.3, 1.0
//   - Preconditioner: BLOCK_JACOBI, LINE_JACOBI

#include "gpu/lowmach_solver.h"
#include "init/lane_emden.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

struct TuningFixture {
    Grid grid;
    EOS eos;
    State state_hse, state_pert;
    LaneEmdenParams lep;
    int nr = 64, nt = 32;

    TuningFixture() : eos(5.0 / 3.0) {
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
};

// Modified step that exposes tuning parameters.
// Returns dt_accepted. Modifies solver internals directly.
static double try_step(LowMachSolver& lm, const Grid& grid, const EOS& eos,
                       const State& hse, const State& pert,
                       double G, PrecondType pc,
                       double target_dt, double gmres_tol, int gmres_max,
                       double clamp_factor) {
    // Fresh solver for each trial
    lm.init(grid, eos, G, 0.4, pc);
    lm.upload_state(grid, hse);
    lm.snapshot_hse();
    lm.upload_state(grid, pert);

    lm.dt_current = target_dt;

    // We need to reach into step() internals. Instead, just call step()
    // and parse the result. But step() uses hardcoded params.
    // Workaround: temporarily modify the solver's behavior via a custom step.

    int n = lm.nr * lm.nt, B = 256, N4 = 4*n;

    lm.apply_floor();
    lm.pack_state(lm.d_Un);
    lm.compute_scaling();

    double dt = target_dt;
    int max_newton = 20;
    int max_cuts = 8;
    bool converged = false;
    double dt_min = 1e-4 * dt;

    for (int cut = 0; cut < max_cuts && !converged; ++cut) {
        if (cut > 0) {
            dt *= 0.5;
            if (dt < dt_min) break;
            lm.unpack_set(lm.d_Un);
        }

        lm.assemble_block_jacobi(dt);

        double Fnorm0 = 0;
        bool diverged = false;

        for (int newton = 0; newton < max_newton; ++newton) {
            lm.apply_floor();
            lm.compute_F(lm.d_Fk, dt);

            // Scaled norm
            extern __global__ void k_lm_copy(double*, const double*, int);
            extern __global__ void k_lm_ediv(double*, const double*, int);
            extern __global__ void k_lm_zero(double*, int);

            k_lm_copy<<<(N4+B-1)/B,B>>>(lm.d_residual_ls, lm.d_Fk, N4);
            k_lm_ediv<<<(N4+B-1)/B,B>>>(lm.d_residual_ls, lm.d_scale_L, N4);

            // dot product via cublas-free reduction
            std::vector<double> h(N4);
            cudaMemcpy(h.data(), lm.d_residual_ls, N4*sizeof(double), cudaMemcpyDeviceToHost);
            double sq = 0;
            for (int i = 0; i < N4; i++) sq += h[i]*h[i];
            double Fnorm = std::sqrt(sq);

            if (newton == 0) {
                Fnorm0 = Fnorm;
                if (Fnorm < 1e-30) { converged = true; break; }
            }

            double Fnorm_pc = Fnorm / std::sqrt((double)N4);
            if (Fnorm < 1e-3 * Fnorm0 || Fnorm_pc < 1e-4) {
                converged = true; break;
            }
            if (Fnorm > 1e6 * Fnorm0 || std::isnan(Fnorm)) { diverged = true; break; }

            k_lm_zero<<<(n+B-1)/B,B>>>(lm.d_gmres_w + 4*n, n);
            int iters = lm.gmres_solve(lm.d_gmres_w, lm.d_Fk, dt, gmres_tol, gmres_max);

            lm.clamp_correction(lm.d_gmres_w, clamp_factor);

            extern __global__ void k_lm_pack(const double*, const double*, const double*,
                                              const double*, const double*, double*, int, int, int);
            extern __global__ void k_lm_unpack_set(double*, double*, double*, double*,
                                                    double*, const double*, int, int, int);
            extern __global__ void k_lm_unpack_add(double*, double*, double*, double*,
                                                    double*, const double*, double, int, int, int);

            k_lm_pack<<<(n+B-1)/B,B>>>(lm.d_rho,lm.d_mr,lm.d_mtheta,lm.d_rhoE,lm.d_phi,
                                         lm.d_gmres_Uk, lm.nr,lm.nt,lm.ng);

            double alpha = 1.0;
            double Fnorm_new = Fnorm;
            for (int ls = 0; ls < 8; ++ls) {
                k_lm_unpack_set<<<(n+B-1)/B,B>>>(lm.d_rho,lm.d_mr,lm.d_mtheta,lm.d_rhoE,lm.d_phi,
                                                   lm.d_gmres_Uk, lm.nr,lm.nt,lm.ng);
                k_lm_unpack_add<<<(n+B-1)/B,B>>>(lm.d_rho,lm.d_mr,lm.d_mtheta,lm.d_rhoE,lm.d_phi,
                                                   lm.d_gmres_w, alpha, lm.nr,lm.nt,lm.ng);
                lm.apply_floor();
                lm.compute_F(lm.d_residual_ls, dt);
                k_lm_ediv<<<(N4+B-1)/B,B>>>(lm.d_residual_ls, lm.d_scale_L, N4);
                cudaMemcpy(h.data(), lm.d_residual_ls, N4*sizeof(double), cudaMemcpyDeviceToHost);
                sq = 0; for (int i = 0; i < N4; i++) sq += h[i]*h[i];
                Fnorm_new = std::sqrt(sq);
                if (Fnorm_new < Fnorm) break;
                alpha *= 0.5;
            }

            if (Fnorm_new >= Fnorm) {
                k_lm_unpack_set<<<(n+B-1)/B,B>>>(lm.d_rho,lm.d_mr,lm.d_mtheta,lm.d_rhoE,lm.d_phi,
                                                   lm.d_gmres_Uk, lm.nr,lm.nt,lm.ng);
                diverged = true; break;
            }
        }

        if (diverged) lm.unpack_set(lm.d_Un);
    }

    lm.destroy();
    return converged ? dt : 0.0;
}

int main() {
    TuningFixture f;

    // Target dt values to test
    double dt_targets[] = {1e-7, 2.5e-7, 5e-7, 1e-6, 5e-6};
    int n_dt = 5;

    // Parameter grids
    double gmres_tols[] = {1e-1, 1e-2, 1e-3, 1e-4};
    int n_tol = 4;

    int gmres_maxs[] = {5, 10, 20, 40, 60};
    int n_max = 5;

    double clamp_factors[] = {0.01, 0.05, 0.1, 0.3, 1.0};
    int n_clamp = 5;

    PrecondType pcs[] = {PrecondType::BLOCK_JACOBI, PrecondType::LINE_JACOBI};
    const char* pc_names[] = {"BJ", "LJ"};
    int n_pc = 2;

    std::fprintf(stderr, "=== Newton-Krylov parameter sweep ===\n");
    std::fprintf(stderr, "Grid: %dx%d, Lane-Emden perturbed (eps=1e-3)\n\n", f.nr, f.nt);

    // Sweep 1: GMRES tolerance (fix others at default)
    std::fprintf(stderr, "── Sweep 1: GMRES tolerance (gmres_max=60, clamp=0.1) ──\n");
    std::fprintf(stderr, "  %-4s  %-8s", "PC", "tol");
    for (int d = 0; d < n_dt; d++) std::fprintf(stderr, "  dt=%.0e", dt_targets[d]);
    std::fprintf(stderr, "\n");

    for (int p = 0; p < n_pc; p++) {
        for (int t = 0; t < n_tol; t++) {
            std::fprintf(stderr, "  %-4s  %.0e  ", pc_names[p], gmres_tols[t]);
            for (int d = 0; d < n_dt; d++) {
                LowMachSolver lm;
                double acc = try_step(lm, f.grid, f.eos, f.state_hse, f.state_pert,
                                      f.lep.G, pcs[p], dt_targets[d],
                                      gmres_tols[t], 60, 0.1);
                double ratio = (acc > 0) ? acc / dt_targets[d] : 0.0;
                if (ratio >= 0.99) std::fprintf(stderr, "  %7s", "OK");
                else if (ratio > 0.1) std::fprintf(stderr, "  %6.0f%%", ratio*100);
                else std::fprintf(stderr, "  %7s", "FAIL");
            }
            std::fprintf(stderr, "\n");
        }
    }

    // Sweep 2: GMRES max iterations (fix tol=1e-3, clamp=0.1)
    std::fprintf(stderr, "\n── Sweep 2: GMRES max iters (tol=1e-3, clamp=0.1) ──\n");
    std::fprintf(stderr, "  %-4s  %-8s", "PC", "maxiter");
    for (int d = 0; d < n_dt; d++) std::fprintf(stderr, "  dt=%.0e", dt_targets[d]);
    std::fprintf(stderr, "\n");

    for (int p = 0; p < n_pc; p++) {
        for (int m = 0; m < n_max; m++) {
            std::fprintf(stderr, "  %-4s  %-8d", pc_names[p], gmres_maxs[m]);
            for (int d = 0; d < n_dt; d++) {
                LowMachSolver lm;
                double acc = try_step(lm, f.grid, f.eos, f.state_hse, f.state_pert,
                                      f.lep.G, pcs[p], dt_targets[d],
                                      1e-3, gmres_maxs[m], 0.1);
                double ratio = (acc > 0) ? acc / dt_targets[d] : 0.0;
                if (ratio >= 0.99) std::fprintf(stderr, "  %7s", "OK");
                else if (ratio > 0.1) std::fprintf(stderr, "  %6.0f%%", ratio*100);
                else std::fprintf(stderr, "  %7s", "FAIL");
            }
            std::fprintf(stderr, "\n");
        }
    }

    // Sweep 3: Clamp factor (fix tol=1e-3, max=60)
    std::fprintf(stderr, "\n── Sweep 3: Clamp factor (tol=1e-3, gmres_max=60) ──\n");
    std::fprintf(stderr, "  %-4s  %-8s", "PC", "clamp");
    for (int d = 0; d < n_dt; d++) std::fprintf(stderr, "  dt=%.0e", dt_targets[d]);
    std::fprintf(stderr, "\n");

    for (int p = 0; p < n_pc; p++) {
        for (int c = 0; c < n_clamp; c++) {
            std::fprintf(stderr, "  %-4s  %-8.2f", pc_names[p], clamp_factors[c]);
            for (int d = 0; d < n_dt; d++) {
                LowMachSolver lm;
                double acc = try_step(lm, f.grid, f.eos, f.state_hse, f.state_pert,
                                      f.lep.G, pcs[p], dt_targets[d],
                                      1e-3, 60, clamp_factors[c]);
                double ratio = (acc > 0) ? acc / dt_targets[d] : 0.0;
                if (ratio >= 0.99) std::fprintf(stderr, "  %7s", "OK");
                else if (ratio > 0.1) std::fprintf(stderr, "  %6.0f%%", ratio*100);
                else std::fprintf(stderr, "  %7s", "FAIL");
            }
            std::fprintf(stderr, "\n");
        }
    }

    // Sweep 4: Best combo search (tol × clamp for LJ only)
    std::fprintf(stderr, "\n── Sweep 4: tol × clamp for LINE_JACOBI (gmres_max=60) ──\n");
    std::fprintf(stderr, "  %-8s  %-8s", "tol", "clamp");
    for (int d = 0; d < n_dt; d++) std::fprintf(stderr, "  dt=%.0e", dt_targets[d]);
    std::fprintf(stderr, "\n");

    for (int t = 0; t < n_tol; t++) {
        for (int c = 0; c < n_clamp; c++) {
            std::fprintf(stderr, "  %.0e    %-8.2f", gmres_tols[t], clamp_factors[c]);
            for (int d = 0; d < n_dt; d++) {
                LowMachSolver lm;
                double acc = try_step(lm, f.grid, f.eos, f.state_hse, f.state_pert,
                                      f.lep.G, PrecondType::LINE_JACOBI, dt_targets[d],
                                      gmres_tols[t], 60, clamp_factors[c]);
                double ratio = (acc > 0) ? acc / dt_targets[d] : 0.0;
                if (ratio >= 0.99) std::fprintf(stderr, "  %7s", "OK");
                else if (ratio > 0.1) std::fprintf(stderr, "  %6.0f%%", ratio*100);
                else std::fprintf(stderr, "  %7s", "FAIL");
            }
            std::fprintf(stderr, "\n");
        }
    }

    std::fprintf(stderr, "\n=== Done ===\n");
    return 0;
}
