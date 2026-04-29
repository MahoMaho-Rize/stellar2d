// LowMach solver orchestration: init, destroy, step, upload/download, snapshot_hse

#include "lowmach_solver.h"
#include "lm_common.cuh"
#include <algorithm>

// ========================= Time step (Newton) =====================

double LowMachSolver::step(double t, double t_end) {
    int n = nr*nt, B = 256, N4 = 4*n;

    static bool hse_init = false;
    if (!hse_init) {
        if (!hse_set_externally)
            snapshot_hse();
        hse_init = true;
    }

    apply_floor();

    if (step_count == 0)
        diagnose_hse_residual();

    double dt_cap = 1.0;
    if (dt_current < 1e-30) dt_current = 1e-6;
    double dt_cfl = compute_cfl_dt();
    double cfl_max_factor = 20.0;
    double dt = std::min({dt_current, dt_cap, cfl_max_factor * dt_cfl, t_end - t});

    // Save U^{n-1} before overwriting d_Un
    int n5 = 5*nr*nt;
    CUDA_CHECK(cudaMemcpy(d_Un_prev, d_Un, n5*sizeof(double), cudaMemcpyDeviceToDevice));

    pack_state(d_Un);
    compute_scaling();

    // ===== Linear extrapolation initial guess =====
    // U⁰ = Uⁿ + (dt/dt_prev)·(Uⁿ - Uⁿ⁻¹)
    // This gives O(dt²) initial error instead of O(dt), dramatically
    // reducing Newton iterations at large dt.
    // Only activate after step 1 (need two states) and when dt_prev > 0.
    if (step_count > 0 && dt_prev > 1e-30) {
        double ratio = dt / dt_prev;
        // Cap extrapolation ratio to avoid overshooting when dt changes dramatically
        ratio = std::min(ratio, 2.0);
        // Compute delta = Uⁿ - Uⁿ⁻¹ in d_Fk (scratch), then unpack_add to live state
        k_lm_copy<<<(n5+B-1)/B,B>>>(d_Fk, d_Un, n5);
        k_lm_axpy<<<(n5+B-1)/B,B>>>(d_Fk, -1.0, d_Un_prev, n5);
        // Apply: state += ratio * delta
        k_lm_unpack_add<<<(n+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE,d_phi, d_Fk, ratio, nr,nt,ng);
        apply_floor();
    }

    int max_newton = 25;      // was 20 — more room before giving up
    int max_dt_cuts = 8;
    bool converged = false;
    int newton_iters_used = 0;

    double dt_min = 1e-4 * dt;

    for (int cut = 0; cut < max_dt_cuts && !converged; ++cut) {
        if (cut > 0) {
            dt *= 0.5;
            if (dt < dt_min) {
                std::fprintf(stderr, "FATAL: step %d dt=%.3e < dt_min=%.3e, solver diverged.\n",
                            step_count, dt, dt_min);
                std::exit(1);
            }
            unpack_set(d_Un);
            (void)0;
        }

        assemble_block_jacobi(dt);
        if (precond_type == PrecondType::SIMPLE || precond_type == PrecondType::COMBINED
            || precond_type == PrecondType::LINE_JACOBI || precond_type == PrecondType::PBP)
            assemble_simple(dt);

        double Fnorm0 = 0.0;
        double Fnorm_prev = 0.0;
        bool diverged = false;
        int ls_fails = 0;

        for (int newton = 0; newton < max_newton; ++newton) {
            apply_floor();

            compute_F(d_Fk, dt);
            // Scaled merit: ||L⁻¹·F||. Use d_residual_ls as scratch.
            k_lm_copy<<<(N4+B-1)/B,B>>>(d_residual_ls, d_Fk, N4);
            k_lm_ediv<<<(N4+B-1)/B,B>>>(d_residual_ls, d_scale_L, N4);
            double Fnorm = sqrt(gpu_dot(d_residual_ls, d_residual_ls, d_work_a, d_work_b, N4));

            if (newton == 0) {
                Fnorm0 = Fnorm;
                if (Fnorm < 1e-30) { converged = true; break; }
            }

            double Fnorm_per_cell = Fnorm / sqrt((double)N4);
            if (Fnorm < 1e-3 * Fnorm0 || Fnorm_per_cell < 1e-4) {
                converged = true;
                newton_iters_used = newton;
                (void)0;
                break;
            }
            if (Fnorm > 1e6 * Fnorm0 || std::isnan(Fnorm)) { diverged = true; break; }

            // Eisenstat-Walker forcing: loose at start, tighten as Newton converges.
            // Tuning sweep shows tol=1e-1 is optimal for large dt acceptance.
            double eta_gmres;
            if (newton == 0 || Fnorm_prev < 1e-30) {
                eta_gmres = 1e-2;
            } else {
                double ratio = Fnorm / Fnorm_prev;
                eta_gmres = 0.9 * ratio * ratio;
                eta_gmres = std::max(eta_gmres, 1e-4);
                eta_gmres = std::min(eta_gmres, 1e-2);
            }
            Fnorm_prev = Fnorm;

            // GMRES in physical space (d_Fk is unscaled)
            k_lm_zero<<<(n+B-1)/B,B>>>(d_gmres_w + 4*n, n);
            int gmres_iters = gmres_solve(d_gmres_w, d_Fk, dt, eta_gmres, GMRES_RESTART);

            clamp_correction(d_gmres_w, 0.1);

            k_lm_pack<<<(n+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE,d_phi, d_gmres_Uk, nr,nt,ng);
            double alpha = 1.0;
            double Fnorm_new = Fnorm;

            for (int ls = 0; ls < 8; ++ls) {
                k_lm_unpack_set<<<(n+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE,d_phi, d_gmres_Uk, nr,nt,ng);
                k_lm_unpack_add<<<(n+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE,d_phi, d_gmres_w, alpha, nr,nt,ng);
                apply_floor();
                compute_F(d_residual_ls, dt);
                k_lm_ediv<<<(N4+B-1)/B,B>>>(d_residual_ls, d_scale_L, N4);
                Fnorm_new = sqrt(gpu_dot(d_residual_ls, d_residual_ls, d_work_a, d_work_b, N4));
                if (Fnorm_new < Fnorm) break;
                alpha *= 0.5;
            }

            if (Fnorm_new >= Fnorm) {
                k_lm_unpack_set<<<(n+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE,d_phi, d_gmres_Uk, nr,nt,ng);
                ls_fails++;
                if (ls_fails >= 3) { diverged = true; break; }
                continue;
            }
            ls_fails = 0;

            (void)0;
        }

        if (diverged)
            unpack_set(d_Un);
    }

    // Diagnostic: on failure, identify WHERE the residual is largest
    if (!converged && step_count < 50) {
        unpack_set(d_Un);       // restore clean state for diagnostic
        apply_floor();
        compute_F(d_Fk, dt);
        std::vector<double> h_F(N4);
        CUDA_CHECK(cudaMemcpy(h_F.data(), d_Fk, N4*sizeof(double), cudaMemcpyDeviceToHost));
        // Find max |F| per equation
        double mx[4] = {}; int mi[4] = {};
        for (int q = 0; q < 4; ++q)
            for (int i = 0; i < n; ++i)
                if (std::abs(h_F[q*n+i]) > mx[q]) { mx[q] = std::abs(h_F[q*n+i]); mi[q] = i; }
        (void)mx; (void)mi;
    }

    // Update Φ for output/diagnostics (not used in Newton)
    if (converged) {
        apply_sponge(dt);
        solve_gravity();
    }

    // Simple dt adaptation: grow 1.2x on success, halved on failure (inside loop).
    if (converged) {
        dt_current = std::min(1.2 * dt, dt_cap);
    } else {
        dt_current = 0.5 * dt;
    }

    dt_prev = dt;
    step_count++;
    return dt;
}

// ========================= HSE diagnostic ==========================

void LowMachSolver::snapshot_hse() {
    int n = nr*nt, B = 256;
    apply_floor();
    compute_gravity_1d();
    CUDA_CHECK(cudaMemcpy(d_gr0, d_gr, nr*sizeof(double), cudaMemcpyDeviceToDevice));
    solve_gravity();
    k_lm_snapshot_hse<<<(n+B-1)/B,B>>>(d_rho,d_rhoE,d_phi,
        d_rho0,d_P0,d_phi0, nr,nt,ng,gamma);

    // Compute atmosphere threshold: 1e-6 * max(ρ₀)
    std::vector<double> h_rho0(n);
    CUDA_CHECK(cudaMemcpy(h_rho0.data(), d_rho0, n*sizeof(double), cudaMemcpyDeviceToHost));
    double rho_max = 0;
    for (int i = 0; i < n; ++i) rho_max = std::max(rho_max, h_rho0[i]);
    atm_rho_thresh = 1e-6 * rho_max;

    // Sponge layer: start where ρ₀ drops below 1% of max, end at outer boundary
    double sponge_density = 0.01 * rho_max;
    std::vector<double> h_rc(nr), h_rf(nr + 1);
    CUDA_CHECK(cudaMemcpy(h_rc.data(), d_r_center, nr*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rf.data(), d_r_face, (nr+1)*sizeof(double), cudaMemcpyDeviceToHost));
    sponge_r_top = h_rf[nr];
    sponge_r_start = sponge_r_top;  // default: no sponge
    for (int i = nr - 1; i >= 0; --i) {
        double rho_eq = h_rho0[i * nt + nt / 2];  // equatorial value
        if (rho_eq > sponge_density) {
            sponge_r_start = h_rc[i];
            break;
        }
    }

    std::fprintf(stderr, "  HSE snapshot: ρ_max=%.3e, atm_thresh=%.3e, sponge [%.3f, %.3f]\n",
                 rho_max, atm_rho_thresh, sponge_r_start, sponge_r_top);

    hse_set_externally = true;
}

void LowMachSolver::diagnose_hse_residual() {
    int n = nr*nt, N4 = 4*n;

    compute_residual(d_residual);

    std::vector<double> h_res(N4);
    CUDA_CHECK(cudaMemcpy(h_res.data(), d_residual, N4*sizeof(double), cudaMemcpyDeviceToHost));

    double max_rho=0, max_mr=0, max_mt=0, max_rhoE=0;
    for (int i = 0; i < n; ++i) {
        max_rho  = std::max(max_rho,  std::abs(h_res[i]));
        max_mr   = std::max(max_mr,   std::abs(h_res[n+i]));
        max_mt   = std::max(max_mt,   std::abs(h_res[2*n+i]));
        max_rhoE = std::max(max_rhoE, std::abs(h_res[3*n+i]));
    }

    double l2 = 0;
    for (int i = 0; i < N4; ++i) l2 += h_res[i]*h_res[i];
    l2 = std::sqrt(l2);

    (void)max_rho; (void)max_mr; (void)max_mt; (void)max_rhoE; (void)l2;
}

// ========================= Init ===================================

void LowMachSolver::init(const Grid& grid, const EOS& eos, double G, double cfl,
                         PrecondType pc) {
    precond_type = pc;
    nr = grid.nr; nt = grid.ntheta; ng = grid.ng;
    gamma = eos.gamma; G_const = G; cfl_num = cfl;
    total_phys = nr*nt;
    total_ghost = (nr+2*ng)*(nt+2*ng);
    dt_current = 0.0;

    int n = total_phys;

    // Grid arrays
    CUDA_CHECK(cudaMalloc(&d_r_face, (nr+1)*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r_center, nr*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dr, nr*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_theta_face, (nt+1)*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_theta_center, nt*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dtheta, nt*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cell_volume, n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_area_r, (nr+1)*nt*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_area_theta, nr*(nt+1)*sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_r_face, grid.r_face.data(), (nr+1)*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r_center, grid.r_center.data(), nr*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dr, grid.dr.data(), nr*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_theta_face, grid.theta_face.data(), (nt+1)*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_theta_center, grid.theta_center.data(), nt*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dtheta, grid.dtheta.data(), nt*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cell_volume, grid.cell_volume.data(), n*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_area_r, grid.area_r.data(), (nr+1)*nt*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_area_theta, grid.area_theta.data(), nr*(nt+1)*sizeof(double), cudaMemcpyHostToDevice));

    // sin(theta) arrays for Poisson residual (must match GMG stencil exactly)
    {
        std::vector<double> stf(nt+1), stc(nt);
        for (int j = 0; j <= nt; ++j) stf[j] = std::sin(grid.theta_face[j]);
        for (int j = 0; j < nt; ++j) stc[j] = std::sin(0.5*(grid.theta_face[j]+grid.theta_face[j+1]));
        CUDA_CHECK(cudaMalloc(&d_sin_theta_face, (nt+1)*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_sin_theta_center, nt*sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_sin_theta_face, stf.data(), (nt+1)*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sin_theta_center, stc.data(), nt*sizeof(double), cudaMemcpyHostToDevice));
    }

    // Precompute gradient stencil weights (eliminates branching in hot kernels)
    {
        std::vector<double> wr_m(n), wr_p(n), wt_m(n), wt_p(n);
        for (int i = 0; i < nr; ++i) {
            for (int j = 0; j < nt; ++j) {
                int flat = i*nt + j;
                double r = grid.r_center[i];
                // Radial gradient: dP/dr = wm*(P_{i-1}-Pc) + wp*(P_{i+1}-Pc)
                if (i > 0 && i < nr-1) {
                    double dl = grid.r_center[i] - grid.r_center[i-1];
                    double dh = grid.r_center[i+1] - grid.r_center[i];
                    wr_m[flat] = -dh / (dl*(dl+dh));
                    wr_p[flat] = dl / (dh*(dl+dh));
                } else if (i == 0 && nr > 1) {
                    wr_m[flat] = 0.0;
                    wr_p[flat] = 1.0 / (grid.r_center[1] - grid.r_center[0]);
                } else { // i == nr-1
                    wr_m[flat] = -1.0 / (grid.r_center[nr-1] - grid.r_center[nr-2]);
                    wr_p[flat] = 0.0;
                }
                // Theta gradient / r: (1/r)dP/dθ = wm*(P_{j-1}-Pc) + wp*(P_{j+1}-Pc)
                if (j > 0 && j < nt-1) {
                    double tc_m = 0.5*(grid.theta_face[j-1] + grid.theta_face[j]);
                    double tc_c = 0.5*(grid.theta_face[j] + grid.theta_face[j+1]);
                    double tc_p = 0.5*(grid.theta_face[j+1] + grid.theta_face[j+2]);
                    double dl = tc_c - tc_m, dh = tc_p - tc_c;
                    wt_m[flat] = -dh / (r * dl * (dl+dh));
                    wt_p[flat] = dl / (r * dh * (dl+dh));
                } else {
                    wt_m[flat] = 0.0;
                    wt_p[flat] = 0.0;
                }
            }
        }
        CUDA_CHECK(cudaMalloc(&d_grad_r_wm, n*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_grad_r_wp, n*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_grad_t_wm, n*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_grad_t_wp, n*sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_grad_r_wm, wr_m.data(), n*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_r_wp, wr_p.data(), n*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_t_wm, wt_m.data(), n*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_t_wp, wt_p.data(), n*sizeof(double), cudaMemcpyHostToDevice));
    }

    // State arrays (with ghosts)
    CUDA_CHECK(cudaMalloc(&d_rho, total_ghost*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mr, total_ghost*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mtheta, total_ghost*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rhoE, total_ghost*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_rho, 0, total_ghost*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mr, 0, total_ghost*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mtheta, 0, total_ghost*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_rhoE, 0, total_ghost*sizeof(double)));

    // Physics arrays (phys cells only)
    CUDA_CHECK(cudaMalloc(&d_phi, n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pi, n*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_phi, 0, n*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_pi, 0, n*sizeof(double)));

    CUDA_CHECK(cudaMalloc(&d_rho0, n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_P0, n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_phi0, n*sizeof(double)));

    // 1D radial gravity
    CUDA_CHECK(cudaMalloc(&d_gr, nr*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gr0, nr*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_shell_mass, nr*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gr, 0, nr*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gr0, 0, nr*sizeof(double)));

    // Newton vectors (5*n packed: ρ, mr, mt, ρe, Φ)
    CUDA_CHECK(cudaMalloc(&d_Un, 5*n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Un_prev, 5*n*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_Un_prev, 0, 5*n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Fk, 5*n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_residual, 5*n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_residual_ls, 5*n*sizeof(double)));

    // GMRES vectors (5N per vector)
    for (int i = 0; i <= GMRES_RESTART; ++i) {
        CUDA_CHECK(cudaMalloc(&d_gmres_V[i], 5*n*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_gmres_Z[i], 5*n*sizeof(double)));
    }
    CUDA_CHECK(cudaMalloc(&d_gmres_w, 5*n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gmres_Uk, 5*n*sizeof(double)));

    // Work buffers
    CUDA_CHECK(cudaMalloc(&d_work_a, n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_work_b, n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rhs_poisson, n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_inv_rho, n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_scale, 4*n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_scale_R, 4*n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_scale_L, 4*n*sizeof(double)));

    // Schur complement Helmholtz GMG
    CUDA_CHECK(cudaMalloc(&d_sigma_schur, n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_poisson_scale, n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_schur_rhs, n*sizeof(double)));
    gmg_schur.init(nr, nt, grid.r_face.data(), grid.theta_face.data());

    // Block-diagonal Jacobi preconditioner (always needed for Schur σ assembly)
    CUDA_CHECK(cudaMalloc(&d_blk_diag, n*16*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_blk_J, n*16*sizeof(double)));

    // SIMPLE preconditioner (also needed for COMBINED)
    d_Ap = nullptr; d_simple_p = nullptr; d_simple_div = nullptr;
    d_simple_vr_s = nullptr; d_simple_vt_s = nullptr;
    if (precond_type == PrecondType::SIMPLE || precond_type == PrecondType::COMBINED
        || precond_type == PrecondType::LINE_JACOBI || precond_type == PrecondType::PBP) {
        CUDA_CHECK(cudaMalloc(&d_Ap, n*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_simple_p, n*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_simple_div, n*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_simple_vr_s, n*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_simple_vt_s, n*sizeof(double)));
        gmg_pressure.init(nr, nt, grid.r_face.data(), grid.theta_face.data());
    }

    // GMG for gravity
    gmg.init(nr, nt, grid.r_face.data(), grid.theta_face.data());

    initialized = true;
    const char* pc_name = (precond_type == PrecondType::PBP) ? "PBP" :
                          (precond_type == PrecondType::BLOCK_SCHUR) ? "BlockSchur" :
                          (precond_type == PrecondType::LINE_JACOBI) ? "LineJacobi" :
                          (precond_type == PrecondType::COMBINED) ? "Combined" :
                          (precond_type == PrecondType::SIMPLE) ? "SIMPLE" :
                          (precond_type == PrecondType::BLOCK_JACOBI) ? "BlockJacobi" : "None";
    std::printf("Low-Mach GPU solver (JFNK+FGMRES+GMG, PC=%s): %dx%d (%d cells), %d MG levels\n",
                pc_name, nr, nt, n, gmg.n_levels);
    std::fflush(stdout);
}

// ========================= Upload / Download ======================

void LowMachSolver::upload_state(const Grid& grid, const State& state) {
    CUDA_CHECK(cudaMemcpy(d_rho, state.rho.data(), total_ghost*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mr, state.mr.data(), total_ghost*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mtheta, state.mtheta.data(), total_ghost*sizeof(double), cudaMemcpyHostToDevice));
    // Map total energy E = e + ke  →  internal energy ρe = ρE - 0.5*ρ(vr²+vθ²)
    std::vector<double> rhoE_host(total_ghost);
    for (int idx = 0; idx < total_ghost; ++idx) {
        double r = std::fmax(state.rho[idx], 1e-20);
        double vr = state.mr[idx] / r;
        double vt = state.mtheta[idx] / r;
        rhoE_host[idx] = state.E[idx] - 0.5 * r * (vr*vr + vt*vt);
        rhoE_host[idx] = std::fmax(rhoE_host[idx], 1e-30);
    }
    CUDA_CHECK(cudaMemcpy(d_rhoE, rhoE_host.data(), total_ghost*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_phi, state.phi.data(), total_phys*sizeof(double), cudaMemcpyHostToDevice));
}

void LowMachSolver::download_state(const Grid& grid, State& state) {
    CUDA_CHECK(cudaMemcpy(state.rho.data(), d_rho, total_ghost*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.mr.data(), d_mr, total_ghost*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.mtheta.data(), d_mtheta, total_ghost*sizeof(double), cudaMemcpyDeviceToHost));
    // Map internal energy ρe → total energy E = ρe + 0.5*ρ(vr²+vθ²)
    std::vector<double> rhoE_host(total_ghost);
    CUDA_CHECK(cudaMemcpy(rhoE_host.data(), d_rhoE, total_ghost*sizeof(double), cudaMemcpyDeviceToHost));
    for (int idx = 0; idx < total_ghost; ++idx) {
        double r = std::fmax(state.rho[idx], 1e-20);
        double vr = state.mr[idx] / r;
        double vt = state.mtheta[idx] / r;
        state.E[idx] = rhoE_host[idx] + 0.5 * r * (vr*vr + vt*vt);
    }
    CUDA_CHECK(cudaMemcpy(state.phi.data(), d_phi, total_phys*sizeof(double), cudaMemcpyDeviceToHost));
}

// ========================= Destroy ================================

void LowMachSolver::destroy() {
    if (!initialized) return;

    gmg.destroy();
    gmg_schur.destroy();
    cudaFree(d_sigma_schur);
    cudaFree(d_poisson_scale);
    cudaFree(d_schur_rhs);
    cudaFree(d_r_face); cudaFree(d_r_center); cudaFree(d_dr);
    cudaFree(d_theta_face); cudaFree(d_theta_center); cudaFree(d_dtheta);
    cudaFree(d_sin_theta_face); cudaFree(d_sin_theta_center);
    cudaFree(d_grad_r_wm); cudaFree(d_grad_r_wp);
    cudaFree(d_grad_t_wm); cudaFree(d_grad_t_wp);
    cudaFree(d_cell_volume); cudaFree(d_area_r); cudaFree(d_area_theta);
    cudaFree(d_rho); cudaFree(d_mr); cudaFree(d_mtheta); cudaFree(d_rhoE);
    cudaFree(d_phi); cudaFree(d_pi);
    cudaFree(d_rho0); cudaFree(d_P0); cudaFree(d_phi0);
    cudaFree(d_gr); cudaFree(d_gr0); cudaFree(d_shell_mass);
    cudaFree(d_Un); cudaFree(d_Un_prev); cudaFree(d_Fk); cudaFree(d_residual); cudaFree(d_residual_ls);
    for (int i = 0; i <= GMRES_RESTART; ++i) {
        cudaFree(d_gmres_V[i]);
        cudaFree(d_gmres_Z[i]);
    }
    cudaFree(d_gmres_w); cudaFree(d_gmres_Uk);
    cudaFree(d_work_a); cudaFree(d_work_b);
    cudaFree(d_rhs_poisson); cudaFree(d_inv_rho); cudaFree(d_scale);
    cudaFree(d_scale_R); cudaFree(d_scale_L);
    cudaFree(d_blk_diag);
    cudaFree(d_blk_J);
    if (d_Ap) cudaFree(d_Ap);
    if (d_simple_p) cudaFree(d_simple_p);
    if (d_simple_div) cudaFree(d_simple_div);
    if (d_simple_vr_s) cudaFree(d_simple_vr_s);
    if (d_simple_vt_s) cudaFree(d_simple_vt_s);
    if (precond_type == PrecondType::SIMPLE || precond_type == PrecondType::COMBINED
        || precond_type == PrecondType::LINE_JACOBI || precond_type == PrecondType::PBP)
        gmg_pressure.destroy();
    initialized = false;
}
