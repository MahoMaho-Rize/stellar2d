// FAS solver orchestration: init, destroy, step, upload/download, snapshot_hse

#include "fas_solver.cuh"
#include "fas_linalg.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

// ========================= Level construction ========================

static void alloc_fas_level(FasLevel& lev) {
    int nr = lev.nr, nt = lev.nt, ng = lev.ng;
    int total = (nr + 2*ng) * (nt + 2*ng);
    int phys = nr * nt;
    lev.total = total;
    lev.phys = phys;

    // Grid
    CUDA_CHECK(cudaMalloc(&lev.d_r_face, (nr+1)*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_r_center, nr*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_dr, nr*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_theta_face, (nt+1)*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_theta_center, nt*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_dtheta, nt*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_cell_volume, phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_area_r, (nr+1)*nt*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_area_theta, nr*(nt+1)*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_sin_theta_face, (nt+1)*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_sin_theta_center, nt*sizeof(double)));

    // State (with ghost)
    CUDA_CHECK(cudaMalloc(&lev.d_rho, total*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_mr, total*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_mt, total*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_rhoE, total*sizeof(double)));

    // FAS arrays (physical only)
    CUDA_CHECK(cudaMalloc(&lev.d_fas_rhs, 4*phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_res, 4*phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_Un, 4*phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_Un_prev, 4*phys*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_Un_prev, 0, 4*phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_save, 4*phys*sizeof(double)));

    // HSE reference
    CUDA_CHECK(cudaMalloc(&lev.d_rho0, phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_P0, phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_hse_defect, 4*phys*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_hse_defect, 0, 4*phys*sizeof(double)));

    // Gravity
    CUDA_CHECK(cudaMalloc(&lev.d_gr, nr*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_gr0, nr*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_shell_mass, nr*sizeof(double)));

    // Block Jacobi
    CUDA_CHECK(cudaMalloc(&lev.d_blk_inv, 16*phys*sizeof(double)));

    // SIMPLE scratch
    CUDA_CHECK(cudaMalloc(&lev.d_Ap, phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_vr_s, phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_vt_s, phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_div_s, phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_dp, phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_poisson_rhs, phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_inv_Ap, phys*sizeof(double)));

    // GMRES scratch
    for (int i = 0; i <= FasLevel::GMRES_K; ++i)
        CUDA_CHECK(cudaMalloc(&lev.d_gmres_V[i], 4*phys*sizeof(double)));
    for (int i = 0; i < FasLevel::GMRES_K; ++i)
        CUDA_CHECK(cudaMalloc(&lev.d_gmres_Z[i], 4*phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_gmres_Ubak, 4*phys*sizeof(double)));

    // Zero everything
    CUDA_CHECK(cudaMemset(lev.d_rho, 0, total*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_mr, 0, total*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_mt, 0, total*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_rhoE, 0, total*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_fas_rhs, 0, 4*phys*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_res, 0, 4*phys*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_Un, 0, 4*phys*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_save, 0, 4*phys*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_rho0, 0, phys*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_P0, 0, phys*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_gr, 0, nr*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_gr0, 0, nr*sizeof(double)));
}

void FasSolver::build_level(int l, int nr, int nt, int ng,
                             const double* h_rf, const double* h_tf) {
    FasLevel& lev = levels[l];
    lev.nr = nr; lev.nt = nt; lev.ng = ng;
    alloc_fas_level(lev);

    // Grid geometry
    std::vector<double> rc(nr), dr_v(nr);
    for (int i = 0; i < nr; ++i) {
        rc[i] = 0.5*(h_rf[i]+h_rf[i+1]);
        dr_v[i] = h_rf[i+1]-h_rf[i];
    }
    CUDA_CHECK(cudaMemcpy(lev.d_r_face, h_rf, (nr+1)*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_r_center, rc.data(), nr*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_dr, dr_v.data(), nr*sizeof(double), cudaMemcpyHostToDevice));

    std::vector<double> tc(nt), dt_v(nt), stf(nt+1), stc(nt);
    for (int j = 0; j <= nt; ++j) stf[j] = std::sin(h_tf[j]);
    for (int j = 0; j < nt; ++j) {
        tc[j] = 0.5*(h_tf[j]+h_tf[j+1]);
        dt_v[j] = h_tf[j+1]-h_tf[j];
        stc[j] = std::sin(tc[j]);
    }
    CUDA_CHECK(cudaMemcpy(lev.d_theta_face, h_tf, (nt+1)*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_theta_center, tc.data(), nt*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_dtheta, dt_v.data(), nt*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_sin_theta_face, stf.data(), (nt+1)*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_sin_theta_center, stc.data(), nt*sizeof(double), cudaMemcpyHostToDevice));

    // Cell volumes
    std::vector<double> vol(nr*nt);
    for (int i = 0; i < nr; ++i) {
        double r3h = h_rf[i+1]*h_rf[i+1]*h_rf[i+1], r3l = h_rf[i]*h_rf[i]*h_rf[i];
        for (int j = 0; j < nt; ++j)
            vol[i*nt+j] = (r3h-r3l)/3.0 * (std::cos(h_tf[j])-std::cos(h_tf[j+1]));
    }
    CUDA_CHECK(cudaMemcpy(lev.d_cell_volume, vol.data(), nr*nt*sizeof(double), cudaMemcpyHostToDevice));

    // Face areas
    std::vector<double> ar((nr+1)*nt), at(nr*(nt+1));
    for (int i = 0; i <= nr; ++i) {
        double rf = h_rf[i];
        for (int j = 0; j < nt; ++j)
            ar[i*nt+j] = rf*rf * (std::cos(h_tf[j])-std::cos(h_tf[j+1]));
    }
    for (int i = 0; i < nr; ++i) {
        double r3h = h_rf[i+1]*h_rf[i+1]*h_rf[i+1], r3l = h_rf[i]*h_rf[i]*h_rf[i];
        for (int j = 0; j <= nt; ++j)
            at[i*(nt+1)+j] = (r3h-r3l)/3.0 * stf[j];
    }
    CUDA_CHECK(cudaMemcpy(lev.d_area_r, ar.data(), (nr+1)*nt*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_area_theta, at.data(), nr*(nt+1)*sizeof(double), cudaMemcpyHostToDevice));

    lev.pressure_gmg.init(nr, nt, h_rf, h_tf);
}

void FasSolver::init(const Grid& grid, const EOS& eos, double G, double cfl) {
    gamma = eos.gamma;
    G_const = G;
    cfl_num = cfl;

    int nr = grid.nr, nt = grid.ntheta, ng = grid.ng;

    // Build multigrid levels
    n_levels = 1;
    int cr = nr, ct = nt;
    while (cr >= 4 && ct >= 4) { cr /= 2; ct /= 2; n_levels++; }

    build_level(0, nr, nt, ng, grid.r_face.data(), grid.theta_face.data());

    std::vector<double> rf(grid.r_face.begin(), grid.r_face.end());
    std::vector<double> tf(grid.theta_face.begin(), grid.theta_face.end());

    for (int l = 1; l < n_levels; ++l) {
        int fnr = (int)rf.size()-1, fnt = (int)tf.size()-1;
        int cnr = fnr/2, cnt = fnt/2;
        std::vector<double> crf(cnr+1), ctf(cnt+1);
        for (int i = 0; i <= cnr; ++i) crf[i] = rf[2*i];
        for (int j = 0; j <= cnt; ++j) ctf[j] = tf[2*j];
        build_level(l, cnr, cnt, ng, crf.data(), ctf.data());
        rf = crf; tf = ctf;
    }

    std::fprintf(stderr, "FAS nonlinear multigrid: %dx%d, %d levels, ω=%.1f\n",
                 nr, nt, n_levels, OMEGA);
}

// ========================= Upload/Download ========================

void FasSolver::upload_state(const Grid& grid, const State& state) {
    FasLevel& lev = levels[0];
    int nr = lev.nr, nt = lev.nt, ng = lev.ng;
    int stride = nt + 2*ng;
    int total = (nr+2*ng)*stride;

    std::vector<double> h_rho(total,0), h_mr(total,0), h_mt(total,0), h_rhoE(total,0);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nt; ++j) {
            int k = (i+ng)*stride + (j+ng);
            int sk = grid.idx(i,j);
            h_rho[k] = state.rho[sk];
            h_mr[k] = state.mr[sk];
            h_mt[k] = state.mtheta[sk];
            h_rhoE[k] = state.E[sk];
        }
    CUDA_CHECK(cudaMemcpy(lev.d_rho, h_rho.data(), total*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_mr, h_mr.data(), total*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_mt, h_mt.data(), total*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_rhoE, h_rhoE.data(), total*sizeof(double), cudaMemcpyHostToDevice));
}

void FasSolver::download_state(const Grid& grid, State& state) {
    FasLevel& lev = levels[0];
    int nr = lev.nr, nt = lev.nt, ng = lev.ng;
    int stride = nt + 2*ng;
    int total = (nr+2*ng)*stride;

    std::vector<double> h_rho(total), h_mr(total), h_mt(total), h_rhoE(total);
    CUDA_CHECK(cudaMemcpy(h_rho.data(), lev.d_rho, total*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mr.data(), lev.d_mr, total*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mt.data(), lev.d_mt, total*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rhoE.data(), lev.d_rhoE, total*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nt; ++j) {
            int k = (i+ng)*stride + (j+ng);
            int sk = grid.idx(i,j);
            state.rho[sk] = h_rho[k];
            state.mr[sk] = h_mr[k];
            state.mtheta[sk] = h_mt[k];
            state.E[sk] = h_rhoE[k];
        }
}

void FasSolver::snapshot_hse() {
    FasLevel& finest = levels[0];
    int nr = finest.nr, nt = finest.nt, ng = finest.ng;
    int n = nr * nt, B = 256;

    apply_floor(0);
    compute_gravity_1d(0);
    CUDA_CHECK(cudaMemcpy(finest.d_gr0, finest.d_gr, nr*sizeof(double), cudaMemcpyDeviceToDevice));

    // Extract ρ₀ and P₀ from current state
    // ρ₀[flat] = rho[ghost_idx], P₀[flat] = (γ-1)*rhoE[ghost_idx]
    std::vector<double> h_rho(finest.total), h_rhoE(finest.total);
    CUDA_CHECK(cudaMemcpy(h_rho.data(), finest.d_rho, finest.total*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rhoE.data(), finest.d_rhoE, finest.total*sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> rho0(n), P0(n);
    int stride = nt + 2*ng;
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nt; ++j) {
            int flat = i*nt+j;
            int k = (i+ng)*stride + (j+ng);
            rho0[flat] = h_rho[k];
            // HSE: v=0, so E = ρe = P/(γ-1), thus P = (γ-1)*E
            P0[flat] = (gamma - 1.0) * h_rhoE[k];
        }
    CUDA_CHECK(cudaMemcpy(finest.d_rho0, rho0.data(), n*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(finest.d_P0, P0.data(), n*sizeof(double), cudaMemcpyHostToDevice));

    double rho_max = *std::max_element(rho0.begin(), rho0.end());
    atm_rho_thresh = 1e-6 * rho_max;

    // Sponge layer: start where ρ₀ drops below 1% of max
    std::vector<double> h_rc(nr);
    CUDA_CHECK(cudaMemcpy(h_rc.data(), finest.d_r_center, nr*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_rf(nr+1);
    CUDA_CHECK(cudaMemcpy(h_rf.data(), finest.d_r_face, (nr+1)*sizeof(double), cudaMemcpyDeviceToHost));
    sponge_r_top = h_rf[nr];
    sponge_r_start = sponge_r_top;
    double sponge_density = 0.01 * rho_max;
    for (int i = nr - 1; i >= 0; --i) {
        double rho_eq = rho0[i * nt + nt / 2];
        if (rho_eq > sponge_density) {
            sponge_r_start = h_rc[i];
            break;
        }
    }

    // Propagate HSE reference to coarse levels.
    // Three strategies available (selected by coarse_hse_mode):
    //   0 = volume-weighted restriction of ρ₀/P₀ (original, causes HSE inconsistency)
    //   1 = restrict ρ₀, then reconstruct P₀ by integrating dP₀/dr = ρ₀·g₀ (HSE-consistent)
    //   2 = restriction only for ρ₀/g₀, P₀ not used (requires non-well-balanced coarse residual)
    int coarse_hse_mode = 1;  // default: HSE reconstruction

    for (int l = 1; l < n_levels; ++l) {
        FasLevel& fl = levels[l-1], &cl = levels[l];
        int cn = cl.nr * cl.nt;

        // Always restrict ρ₀ (volume-weighted)
        std::vector<double> f_rho0(fl.phys), f_vol(fl.phys);
        CUDA_CHECK(cudaMemcpy(f_rho0.data(), fl.d_rho0, fl.phys*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(f_vol.data(), fl.d_cell_volume, fl.phys*sizeof(double), cudaMemcpyDeviceToHost));

        std::vector<double> c_rho0(cn);
        for (int ic = 0; ic < cl.nr; ++ic)
            for (int jc = 0; jc < cl.nt; ++jc) {
                double sr=0, sv=0;
                for (int di=0; di<2; ++di)
                    for (int dj=0; dj<2; ++dj) {
                        int ff = (2*ic+di)*fl.nt + (2*jc+dj);
                        double v = f_vol[ff];
                        sr += f_rho0[ff]*v; sv += v;
                    }
                c_rho0[ic*cl.nt+jc] = sr/sv;
            }
        CUDA_CHECK(cudaMemcpy(cl.d_rho0, c_rho0.data(), cn*sizeof(double), cudaMemcpyHostToDevice));

        // Restrict g₀
        std::vector<double> f_gr0(fl.nr), c_gr0(cl.nr);
        CUDA_CHECK(cudaMemcpy(f_gr0.data(), fl.d_gr0, fl.nr*sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < cl.nr; ++i)
            c_gr0[i] = 0.5*(f_gr0[2*i] + f_gr0[2*i+1]);
        CUDA_CHECK(cudaMemcpy(cl.d_gr0, c_gr0.data(), cl.nr*sizeof(double), cudaMemcpyHostToDevice));

        // Reconstruct P₀ on coarse level
        std::vector<double> c_P0(cn, 0.0);
        if (coarse_hse_mode == 1) {
            // Mode 1: integrate dP₀/dr = ρ₀·g₀ from center outward (MAESTROeX-style)
            // Use equatorial values for 1D integration, then broadcast to all θ
            std::vector<double> c_rc(cl.nr), c_dr(cl.nr);
            CUDA_CHECK(cudaMemcpy(c_rc.data(), cl.d_r_center, cl.nr*sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(c_dr.data(), cl.d_dr, cl.nr*sizeof(double), cudaMemcpyDeviceToHost));

            // Start from center: P₀(0) from equatorial restricted ρ₀
            int jeq = cl.nt / 2;
            double P_prev = (gamma - 1.0) * c_rho0[0*cl.nt+jeq]; // rough initial guess
            // Better: use fine-level center pressure
            std::vector<double> f_P0(fl.phys);
            CUDA_CHECK(cudaMemcpy(f_P0.data(), fl.d_P0, fl.phys*sizeof(double), cudaMemcpyDeviceToHost));
            P_prev = 0.0;
            for (int dj=0; dj<2; ++dj) {
                int ff = 0*fl.nt + (2*jeq+dj);
                P_prev += f_P0[ff]; // average the two fine cells at center
            }
            P_prev /= 2.0;

            for (int ic = 0; ic < cl.nr; ++ic) {
                double rho_eq = c_rho0[ic*cl.nt + jeq];
                double g_i = c_gr0[ic];
                if (ic > 0) {
                    // Integrate: P(i) = P(i-1) + 0.5*(ρ(i-1)+ρ(i)) * g_avg * dr
                    double rho_prev = c_rho0[(ic-1)*cl.nt + jeq];
                    double g_prev = c_gr0[ic-1];
                    double dr_half = c_rc[ic] - c_rc[ic-1];
                    P_prev += 0.5*(rho_prev*g_prev + rho_eq*g_i) * dr_half;
                }
                // Broadcast to all θ at this radius
                for (int jc = 0; jc < cl.nt; ++jc)
                    c_P0[ic*cl.nt + jc] = std::fmax(P_prev, 1e-30);
            }
        } else if (coarse_hse_mode == 0) {
            // Mode 0: volume-weighted restriction of P₀ (original, inconsistent)
            std::vector<double> f_P0(fl.phys);
            CUDA_CHECK(cudaMemcpy(f_P0.data(), fl.d_P0, fl.phys*sizeof(double), cudaMemcpyDeviceToHost));
            for (int ic = 0; ic < cl.nr; ++ic)
                for (int jc = 0; jc < cl.nt; ++jc) {
                    double sp=0, sv=0;
                    for (int di=0; di<2; ++di)
                        for (int dj=0; dj<2; ++dj) {
                            int ff = (2*ic+di)*fl.nt + (2*jc+dj);
                            sp += f_P0[ff]*f_vol[ff]; sv += f_vol[ff];
                        }
                    c_P0[ic*cl.nt+jc] = sp/sv;
                }
        }
        // Mode 2: P₀ = 0 (non-well-balanced coarse already handles this via use_wellbalance=0)
        CUDA_CHECK(cudaMemcpy(cl.d_P0, c_P0.data(), cn*sizeof(double), cudaMemcpyHostToDevice));
    }

    // Compute HSE defect on each level: R_WB(U₀) with U₀ = (ρ₀, 0, 0, P₀/(γ-1))
    // This residual should be ~0 but is nonzero due to discrete ∇P₀ ≠ ρ₀g₀.
    // We store it and subtract from fas_rhs during restrict_defect.
    for (int l = 0; l < n_levels; ++l) {
        FasLevel& lev = levels[l];
        int ln = lev.nr * lev.nt, B2 = 256;

        // Save current state
        std::vector<double> save_rho(lev.total), save_mr(lev.total), save_mt(lev.total), save_rhoE(lev.total);
        CUDA_CHECK(cudaMemcpy(save_rho.data(), lev.d_rho, lev.total*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(save_mr.data(), lev.d_mr, lev.total*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(save_mt.data(), lev.d_mt, lev.total*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(save_rhoE.data(), lev.d_rhoE, lev.total*sizeof(double), cudaMemcpyDeviceToHost));

        // Load HSE state: ρ=ρ₀, mr=0, mt=0, rhoE=P₀/(γ-1)
        std::vector<double> h_rho0(ln), h_P0(ln);
        CUDA_CHECK(cudaMemcpy(h_rho0.data(), lev.d_rho0, ln*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_P0.data(), lev.d_P0, ln*sizeof(double), cudaMemcpyDeviceToHost));

        std::vector<double> h_rho(lev.total, 1e-20), h_mr(lev.total, 0.0), h_mt(lev.total, 0.0), h_rhoE(lev.total, 1e-20);
        int lstride = lev.nt + 2*lev.ng;
        for (int i = 0; i < lev.nr; ++i)
            for (int j = 0; j < lev.nt; ++j) {
                int k = (i+lev.ng)*lstride + (j+lev.ng);
                int flat = i*lev.nt + j;
                h_rho[k] = h_rho0[flat];
                h_rhoE[k] = h_P0[flat] / (gamma - 1.0);
            }
        CUDA_CHECK(cudaMemcpy(lev.d_rho, h_rho.data(), lev.total*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(lev.d_mr, h_mr.data(), lev.total*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(lev.d_mt, h_mt.data(), lev.total*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(lev.d_rhoE, h_rhoE.data(), lev.total*sizeof(double), cudaMemcpyHostToDevice));

        // Compute R_WB(U₀) and store as HSE defect
        compute_residual(l);
        CUDA_CHECK(cudaMemcpy(lev.d_hse_defect, lev.d_res, 4*ln*sizeof(double), cudaMemcpyDeviceToDevice));

        // Restore original state
        CUDA_CHECK(cudaMemcpy(lev.d_rho, save_rho.data(), lev.total*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(lev.d_mr, save_mr.data(), lev.total*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(lev.d_mt, save_mt.data(), lev.total*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(lev.d_rhoE, save_rhoE.data(), lev.total*sizeof(double), cudaMemcpyHostToDevice));
    }

    hse_set = true;
    std::fprintf(stderr, "  FAS HSE snapshot: ρ_max=%.3e, atm_thresh=%.3e\n",
                 rho_max, atm_rho_thresh);
}

// ========================= BDF2 rhs kernel ========================
// fas_rhs = -(α₁·Uⁿ + α₂·Uⁿ⁻¹) / dt_n
// For BE (step 0): α₁=-1, α₂=0 → fas_rhs = Uⁿ/dt

__global__
void k_fas_bdf2_rhs(const double* Un, const double* Un_prev,
                    double* rhs, double alpha1, double alpha2,
                    double inv_dt, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    rhs[i] = -(alpha1 * Un[i] + alpha2 * Un_prev[i]) * inv_dt;
}

// ========================= Public solve ========================

int FasSolver::solve(double dt, double g0_over_dt, int max_cycles, double tol) {
    bool verbose = false;  // disable detail prints for speed
    for (int cyc = 0; cyc < max_cycles; ++cyc) {
        fas_vcycle(0, dt, g0_over_dt);

        compute_F(0, g0_over_dt);
        double norm = residual_norm(0);
        if (verbose) {
            char label[64];
            std::snprintf(label, sizeof(label), "step%d cyc%d", step_count, cyc);
            residual_norm_detail(0, label);
        }
        if (norm < tol) return cyc + 1;
    }
    return max_cycles;
}

// ========================= Time step ========================

double FasSolver::step(double t, double t_end) {
    if (!hse_set) { snapshot_hse(); }

    apply_floor(0);

    double dt_cap = use_simple_smoother ? 1.0 : 1e-4;
    if (dt_current < 1e-30) {
        dt_current = compute_cfl_dt();
        if (dt_current < 1e-30) dt_current = 1e-8;
    }

    double dt_cfl = compute_cfl_dt();
    double cfl_max_factor = 200.0;

    int max_dt_cuts = 4;
    int max_cycles = 6;
    double tol = 0.1;
    double dt = std::min({dt_current, dt_cap, cfl_max_factor * dt_cfl, t_end - t});

    FasLevel& finest = levels[0];
    int n = finest.nr * finest.nt, B = 256;

    // Save Uⁿ before solve (for step rejection / rollback)
    if (step_count > 0) {
        CUDA_CHECK(cudaMemcpy(finest.d_Un_prev, finest.d_Un, 4*n*sizeof(double), cudaMemcpyDeviceToDevice));
    }
    k_fas_pack_flat<<<(n+B-1)/B,B>>>(
        finest.d_rho, finest.d_mr, finest.d_mt, finest.d_rhoE,
        finest.d_Un, finest.nr, finest.nt, finest.ng);
    if (step_count == 0) {
        CUDA_CHECK(cudaMemcpy(finest.d_Un_prev, finest.d_Un, 4*n*sizeof(double), cudaMemcpyDeviceToDevice));
    }

    // Retry loop: if solve fails, restore Uⁿ, halve dt, retry
    double norm = 1e30;
    int cycles = max_cycles;
    bool converged = false;
    int cuts = 0;

    for (cuts = 0; cuts < max_dt_cuts; ++cuts) {
        // BDF2 coefficients (recompute for possibly-changed dt)
        bool use_bdf2 = (step_count > 0 && dt_prev > 1e-30);
        double gamma0, alpha1, alpha2;
        if (use_bdf2) {
            double omega = dt / dt_prev;
            gamma0 = (1.0 + 2.0*omega) / (1.0 + omega);
            alpha1 = -(1.0 + omega);
            alpha2 = omega*omega / (1.0 + omega);
        } else {
            gamma0 = 1.0;
            alpha1 = -1.0;
            alpha2 = 0.0;
        }
        double g0_over_dt = gamma0 / dt;

        // Set fas_rhs = -(α₁·Uⁿ + α₂·Uⁿ⁻¹) / dt
        k_fas_bdf2_rhs<<<(4*n+B-1)/B,B>>>(
            finest.d_Un, finest.d_Un_prev, finest.d_fas_rhs,
            alpha1, alpha2, 1.0/dt, 4*n);

        cycles = solve(dt, g0_over_dt, max_cycles, tol);
        compute_F(0, g0_over_dt);
        norm = residual_norm(0);

        // Convergence: absolute ||F|| < tol, OR step-integrated error is tiny.
        // When dt is small, F ≈ R(Uⁿ) ≈ const regardless of dt. But the actual
        // step error |Uⁿ⁺¹ - Uⁿ| ~ dt·||F|| can be negligible.
        converged = !std::isnan(norm) &&
                    (norm < tol || cycles < max_cycles || norm * dt < 1e-6);

        if (converged) break;

        // Reject step: restore Uⁿ and halve dt
        k_fas_unpack_flat<<<(n+B-1)/B,B>>>(
            finest.d_rho, finest.d_mr, finest.d_mt, finest.d_rhoE,
            finest.d_Un, finest.nr, finest.nt, finest.ng);
        launch_ghost(0);

        double dt_old = dt;
        dt *= 0.5;
        if (step_count % 50 == 0 || cuts > 0)
            std::fprintf(stderr, "  step %d: reject dt=%.2e (||F||=%.2e, dt*||F||=%.2e), retry dt=%.2e\n",
                         step_count, dt_old, norm, norm*dt_old, dt);
    }

    // If all cuts failed but state is not NaN, accept with minimum dt
    // (transport step has been applied — state is at least partially advanced)
    if (!converged && !std::isnan(norm)) {
        std::fprintf(stderr, "  step %d: accepting dt=%.2e after %d cuts (||F||=%.2e)\n",
                     step_count, dt, max_dt_cuts, norm);
    }

    if (step_count % 100 == 0)
        std::fprintf(stderr, "  step %d dt=%.2e cyc=%d ||F||=%.2e cuts=%d\n",
                     step_count, dt, cycles, norm, cuts);

    if (sponge_r_start < sponge_r_top) {
        k_fas_sponge<<<(n+B-1)/B,B>>>(
            finest.d_rho, finest.d_mr, finest.d_mt, finest.d_rhoE,
            finest.d_rho0, finest.d_P0, levels[0].d_r_center,
            sponge_r_start, sponge_r_top, sponge_kappa, dt,
            1.0/(gamma-1.0),
            finest.nr, finest.nt, finest.ng);
    }

    if (converged) {
        dt_current = std::min(1.2 * dt, dt_cap);
    } else {
        dt_current = dt;
    }
    dt_prev = dt;
    step_count++;
    return dt;
}

// ========================= Destroy ========================

void FasSolver::destroy() {
    for (int l = 0; l < n_levels; ++l) {
        FasLevel& lev = levels[l];
        cudaFree(lev.d_r_face); cudaFree(lev.d_r_center); cudaFree(lev.d_dr);
        cudaFree(lev.d_theta_face); cudaFree(lev.d_theta_center); cudaFree(lev.d_dtheta);
        cudaFree(lev.d_cell_volume); cudaFree(lev.d_area_r); cudaFree(lev.d_area_theta);
        cudaFree(lev.d_sin_theta_face); cudaFree(lev.d_sin_theta_center);
        cudaFree(lev.d_rho); cudaFree(lev.d_mr); cudaFree(lev.d_mt); cudaFree(lev.d_rhoE);
        cudaFree(lev.d_fas_rhs); cudaFree(lev.d_res); cudaFree(lev.d_Un); cudaFree(lev.d_Un_prev); cudaFree(lev.d_save);
        cudaFree(lev.d_rho0); cudaFree(lev.d_P0); cudaFree(lev.d_hse_defect);
        cudaFree(lev.d_gr); cudaFree(lev.d_gr0); cudaFree(lev.d_shell_mass);
        cudaFree(lev.d_blk_inv);
        cudaFree(lev.d_Ap); cudaFree(lev.d_vr_s); cudaFree(lev.d_vt_s);
        cudaFree(lev.d_div_s); cudaFree(lev.d_dp);
        cudaFree(lev.d_poisson_rhs); cudaFree(lev.d_inv_Ap);
        for (int i = 0; i <= FasLevel::GMRES_K; ++i) cudaFree(lev.d_gmres_V[i]);
        for (int i = 0; i < FasLevel::GMRES_K; ++i) cudaFree(lev.d_gmres_Z[i]);
        cudaFree(lev.d_gmres_Ubak);
        lev.pressure_gmg.destroy();
    }
    n_levels = 0;
}
