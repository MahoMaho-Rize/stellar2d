// SIMPLE pressure-correction solver — standalone, no multigrid.
// Reuses FAS kernels for residual, ghost cells, gravity, floor, sponge, block-Jacobi, SIMPLE.

#include "simple_solver.cuh"
#include "fas_common.cuh"
#include "fas_hllc.cuh"
#include "fas_linalg.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

// ---- Forward declarations of kernels from fas_residual.cu / fas_smoothers.cu ----
// (These are compiled in other TUs; we just call them here.)

__global__ void k_fas_residual(
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*,
    const double*, const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    double*, int, int, int, double, double, int, int, int);
__global__ void k_fas_residual_origin(
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*,
    const double*, const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    double*, int, int, int, double, double, int, int, int);
__global__ void k_fas_compute_F(double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, double, int, int, int);
__global__ void k_fas_floor(double*, double*, double*, double*, int, int, int, double);
__global__ void k_fas_sponge(double*, double*, double*, double*,
    const double*, const double*, const double*,
    double, double, double, double, double, int, int, int);
__global__ void k_fas_ghost_r_in(double*, double*, double*, double*, int, int, int);
__global__ void k_fas_ghost_r_out(double*, double*, double*, double*, int, int, int);
__global__ void k_fas_ghost_t_n(double*, double*, double*, double*, int, int, int);
__global__ void k_fas_ghost_t_s(double*, double*, double*, double*, int, int, int);
__global__ void k_fas_pole_lock(double*, int, int, int);
__global__ void k_fas_shell_mass(const double*, const double*, double*, int, int, int);
__global__ void k_fas_gravity_from_shells(const double*, const double*, double*, int, double, double);
__global__ void k_fas_cfl(const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*, double*,
    int, int, int, double, double, int);
__global__ void k_fas_assemble_blkjac(
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*,
    const double*, const double*, const double*, const double*, const double*,
    const double*, double*, int, int, int, double, double);
__global__ void k_fas_mom_diag(const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    double*, int, int, int, double, double);
__global__ void k_fas_smooth_blkjac(double*, double*, double*, double*,
    const double*, const double*, const double*, double, double, int, int, int);
__global__ void k_fas_vstar(const double*, const double*, const double*, double, double*, double*, int);
__global__ void k_fas_div(const double*, const double*, const double*, const double*, const double*, double*, int, int);
__global__ void k_fas_prhs(const double*, double*, int, int);
__global__ void k_fas_inv_ap(const double*, double*, int);
__global__ void k_fas_simple_correct(double*, double*, double*, double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*, const double*,
    const double*, double, int, int, int);
__global__ void k_fas_bdf2_rhs(const double*, const double*, double*, double, double, double, int);
__global__ void k_fas_atm_reset(double*, double*, double*, double*,
    const double*, const double*, double, double, int, int, int);
__global__ void k_fas_ghost_r_out_hse(double*, double*, double*, double*,
    const double*, const double*, double, int, int, int);
__global__ void k_fas_angular_avg(double*, double*, double*, double*,
    const double*, int, int, int, int);
__global__ void k_fas_pole_avg(double*, double*, double*, double*,
    const double*, int, int, int, int);

// ========================= Init / Destroy ========================

static void alloc_simple_level(SimpleLevel& lev) {
    int nr = lev.nr, nt = lev.nt, ng = lev.ng;
    int total = (nr+2*ng)*(nt+2*ng);
    int phys = nr*nt;
    lev.total = total; lev.phys = phys;

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

    CUDA_CHECK(cudaMalloc(&lev.d_rho, total*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_mr, total*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_mt, total*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_rhoE, total*sizeof(double)));

    CUDA_CHECK(cudaMalloc(&lev.d_fas_rhs, 4*phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_res, 4*phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_Un, 4*phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_Un_prev, 4*phys*sizeof(double)));

    CUDA_CHECK(cudaMalloc(&lev.d_rho0, phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_P0, phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_hse_defect, 4*phys*sizeof(double)));

    CUDA_CHECK(cudaMalloc(&lev.d_gr, nr*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_gr0, nr*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_shell_mass, nr*sizeof(double)));

    CUDA_CHECK(cudaMalloc(&lev.d_blk_inv, 16*phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_Ap, phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_vr_s, phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_vt_s, phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_div_s, phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_dp, phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_poisson_rhs, phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_inv_Ap, phys*sizeof(double)));

    CUDA_CHECK(cudaMemset(lev.d_rho, 0, total*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_mr, 0, total*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_mt, 0, total*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_rhoE, 0, total*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_fas_rhs, 0, 4*phys*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_res, 0, 4*phys*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_Un, 0, 4*phys*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_Un_prev, 0, 4*phys*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_hse_defect, 0, 4*phys*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_rho0, 0, phys*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_P0, 0, phys*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_gr, 0, nr*sizeof(double)));
    CUDA_CHECK(cudaMemset(lev.d_gr0, 0, nr*sizeof(double)));
}

void SimpleSolver::init(const Grid& grid, const EOS& eos, double G, double cfl) {
    gamma = eos.gamma; G_const = G; cfl_num = cfl;
    int nr = grid.nr, nt = grid.ntheta, ng = grid.ng;
    lev.nr = nr; lev.nt = nt; lev.ng = ng;
    alloc_simple_level(lev);

    // Upload grid geometry (same as FAS build_level)
    std::vector<double> rc(nr), dr_v(nr);
    for (int i = 0; i < nr; i++) { rc[i] = 0.5*(grid.r_face[i]+grid.r_face[i+1]); dr_v[i] = grid.r_face[i+1]-grid.r_face[i]; }
    CUDA_CHECK(cudaMemcpy(lev.d_r_face, grid.r_face.data(), (nr+1)*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_r_center, rc.data(), nr*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_dr, dr_v.data(), nr*sizeof(double), cudaMemcpyHostToDevice));

    std::vector<double> tc(nt), dt_v(nt), stf(nt+1), stc(nt);
    for (int j = 0; j <= nt; j++) stf[j] = std::sin(grid.theta_face[j]);
    for (int j = 0; j < nt; j++) { tc[j] = 0.5*(grid.theta_face[j]+grid.theta_face[j+1]); dt_v[j] = grid.theta_face[j+1]-grid.theta_face[j]; stc[j] = std::sin(tc[j]); }
    CUDA_CHECK(cudaMemcpy(lev.d_theta_face, grid.theta_face.data(), (nt+1)*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_theta_center, tc.data(), nt*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_dtheta, dt_v.data(), nt*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_sin_theta_face, stf.data(), (nt+1)*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_sin_theta_center, stc.data(), nt*sizeof(double), cudaMemcpyHostToDevice));

    std::vector<double> vol(nr*nt), ar((nr+1)*nt), at(nr*(nt+1));
    for (int i = 0; i < nr; i++) {
        double r3h = grid.r_face[i+1]*grid.r_face[i+1]*grid.r_face[i+1];
        double r3l = grid.r_face[i]*grid.r_face[i]*grid.r_face[i];
        for (int j = 0; j < nt; j++) vol[i*nt+j] = (r3h-r3l)/3.0*(std::cos(grid.theta_face[j])-std::cos(grid.theta_face[j+1]));
    }
    for (int i = 0; i <= nr; i++) { double rf2 = grid.r_face[i]*grid.r_face[i]; for (int j = 0; j < nt; j++) ar[i*nt+j] = rf2*(std::cos(grid.theta_face[j])-std::cos(grid.theta_face[j+1])); }
    for (int i = 0; i < nr; i++) { double r3h = grid.r_face[i+1]*grid.r_face[i+1]*grid.r_face[i+1], r3l = grid.r_face[i]*grid.r_face[i]*grid.r_face[i]; for (int j = 0; j <= nt; j++) at[i*(nt+1)+j] = (r3h-r3l)/3.0*stf[j]; }
    CUDA_CHECK(cudaMemcpy(lev.d_cell_volume, vol.data(), nr*nt*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_area_r, ar.data(), (nr+1)*nt*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_area_theta, at.data(), nr*(nt+1)*sizeof(double), cudaMemcpyHostToDevice));

    lev.pressure_gmg.init(nr, nt, grid.r_face.data(), grid.theta_face.data());

    std::fprintf(stderr, "SIMPLE solver: %dx%d, ω=%.1f, %d inner iters\n", nr, nt, OMEGA, N_INNER);
}

// ========================= Upload / Download ========================

void SimpleSolver::upload_state(const Grid& grid, const State& state) {
    int nr = lev.nr, nt = lev.nt, ng = lev.ng, stride = nt+2*ng;
    std::vector<double> h_rho(lev.total,0), h_mr(lev.total,0), h_mt(lev.total,0), h_rhoE(lev.total,0);
    for (int i = 0; i < nr; i++) for (int j = 0; j < nt; j++) {
        int k = (i+ng)*stride+(j+ng), sk = grid.idx(i,j);
        h_rho[k]=state.rho[sk]; h_mr[k]=state.mr[sk]; h_mt[k]=state.mtheta[sk]; h_rhoE[k]=state.E[sk];
    }
    CUDA_CHECK(cudaMemcpy(lev.d_rho, h_rho.data(), lev.total*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_mr, h_mr.data(), lev.total*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_mt, h_mt.data(), lev.total*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_rhoE, h_rhoE.data(), lev.total*sizeof(double), cudaMemcpyHostToDevice));
}

void SimpleSolver::download_state(const Grid& grid, State& state) {
    int nr = lev.nr, nt = lev.nt, ng = lev.ng, stride = nt+2*ng;
    std::vector<double> h_rho(lev.total), h_mr(lev.total), h_mt(lev.total), h_rhoE(lev.total);
    CUDA_CHECK(cudaMemcpy(h_rho.data(), lev.d_rho, lev.total*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mr.data(), lev.d_mr, lev.total*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mt.data(), lev.d_mt, lev.total*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rhoE.data(), lev.d_rhoE, lev.total*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < nr; i++) for (int j = 0; j < nt; j++) {
        int k = (i+ng)*stride+(j+ng), sk = grid.idx(i,j);
        state.rho[sk]=h_rho[k]; state.mr[sk]=h_mr[k]; state.mtheta[sk]=h_mt[k]; state.E[sk]=h_rhoE[k];
    }
}

// ========================= HSE snapshot ========================

void SimpleSolver::snapshot_hse() {
    int nr = lev.nr, nt = lev.nt, ng = lev.ng, n = nr*nt;
    apply_floor();
    compute_gravity_1d();
    CUDA_CHECK(cudaMemcpy(lev.d_gr0, lev.d_gr, nr*sizeof(double), cudaMemcpyDeviceToDevice));

    std::vector<double> h_rho(lev.total), h_rhoE(lev.total);
    CUDA_CHECK(cudaMemcpy(h_rho.data(), lev.d_rho, lev.total*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rhoE.data(), lev.d_rhoE, lev.total*sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> rho0(n), P0(n);
    int stride = nt+2*ng;
    for (int i = 0; i < nr; i++) for (int j = 0; j < nt; j++) {
        int flat = i*nt+j, k = (i+ng)*stride+(j+ng);
        rho0[flat] = h_rho[k];
        P0[flat] = (gamma-1.0)*h_rhoE[k];
    }
    CUDA_CHECK(cudaMemcpy(lev.d_rho0, rho0.data(), n*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_P0, P0.data(), n*sizeof(double), cudaMemcpyHostToDevice));

    double rho_max = *std::max_element(rho0.begin(), rho0.end());
    atm_rho_thresh = 1e-3 * rho_max;

    {
        std::vector<double> h_vol(n);
        CUDA_CHECK(cudaMemcpy(h_vol.data(), lev.d_cell_volume, n*sizeof(double), cudaMemcpyDeviceToHost));
        interior_volume = 0.0;
        for (int flat = 0; flat < n; flat++)
            if (rho0[flat] >= atm_rho_thresh) interior_volume += h_vol[flat];
    }

    // Sponge
    std::vector<double> h_rc(nr), h_rf(nr+1);
    CUDA_CHECK(cudaMemcpy(h_rc.data(), lev.d_r_center, nr*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rf.data(), lev.d_r_face, (nr+1)*sizeof(double), cudaMemcpyDeviceToHost));
    sponge_r_top = h_rf[nr]; sponge_r_start = sponge_r_top;
    for (int i = nr-1; i >= 0; i--) { if (rho0[i*nt+nt/2] > 0.01*rho_max) { sponge_r_start = h_rc[i]; break; } }

    // HSE defect
    std::vector<double> save_rho(lev.total), save_mr(lev.total), save_mt(lev.total), save_rhoE2(lev.total);
    CUDA_CHECK(cudaMemcpy(save_rho.data(), lev.d_rho, lev.total*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(save_mr.data(), lev.d_mr, lev.total*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(save_mt.data(), lev.d_mt, lev.total*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(save_rhoE2.data(), lev.d_rhoE, lev.total*sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> h2_rho(lev.total, 1e-20), h2_mr(lev.total, 0), h2_mt(lev.total, 0), h2_rhoE(lev.total, 1e-20);
    for (int i = 0; i < nr; i++) for (int j = 0; j < nt; j++) {
        int k = (i+ng)*stride+(j+ng), flat = i*nt+j;
        h2_rho[k] = rho0[flat]; h2_rhoE[k] = P0[flat]/(gamma-1.0);
    }
    CUDA_CHECK(cudaMemcpy(lev.d_rho, h2_rho.data(), lev.total*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_mr, h2_mr.data(), lev.total*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_mt, h2_mt.data(), lev.total*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_rhoE, h2_rhoE.data(), lev.total*sizeof(double), cudaMemcpyHostToDevice));
    compute_residual();
    CUDA_CHECK(cudaMemcpy(lev.d_hse_defect, lev.d_res, 4*n*sizeof(double), cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaMemcpy(lev.d_rho, save_rho.data(), lev.total*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_mr, save_mr.data(), lev.total*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_mt, save_mt.data(), lev.total*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lev.d_rhoE, save_rhoE2.data(), lev.total*sizeof(double), cudaMemcpyHostToDevice));

    hse_set = true;
    std::fprintf(stderr, "  SIMPLE HSE snapshot: ρ_max=%.3e, atm=%.3e\n", rho_max, atm_rho_thresh);
}

// ========================= Building blocks ========================

void SimpleSolver::launch_ghost() {
    int B = 256;
    { dim3 g((lev.nt+B-1)/B, lev.ng); k_fas_ghost_r_in<<<g,B>>>(lev.d_rho,lev.d_mr,lev.d_mt,lev.d_rhoE,lev.nr,lev.nt,lev.ng); }
    if (use_hse_outer_bc && hse_set) {
        dim3 g((lev.nt+B-1)/B, lev.ng);
        k_fas_ghost_r_out_hse<<<g,B>>>(lev.d_rho,lev.d_mr,lev.d_mt,lev.d_rhoE,
            lev.d_rho0, lev.d_P0, 1.0/(gamma-1.0), lev.nr,lev.nt,lev.ng);
    } else {
        dim3 g((lev.nt+B-1)/B, lev.ng);
        k_fas_ghost_r_out<<<g,B>>>(lev.d_rho,lev.d_mr,lev.d_mt,lev.d_rhoE,lev.nr,lev.nt,lev.ng);
    }
    { dim3 g((lev.nr+2*lev.ng+B-1)/B, lev.ng); k_fas_ghost_t_n<<<g,B>>>(lev.d_rho,lev.d_mr,lev.d_mt,lev.d_rhoE,lev.nr,lev.nt,lev.ng); }
    { dim3 g((lev.nr+2*lev.ng+B-1)/B, lev.ng); k_fas_ghost_t_s<<<g,B>>>(lev.d_rho,lev.d_mr,lev.d_mt,lev.d_rhoE,lev.nr,lev.nt,lev.ng); }
    k_fas_pole_lock<<<(lev.nr+2*lev.ng+B-1)/B,B>>>(lev.d_mt, lev.nr, lev.nt, lev.ng);
}

void SimpleSolver::compute_gravity_1d() {
    int B = std::min(lev.nt, 256);
    k_fas_shell_mass<<<lev.nr, B, B*sizeof(double)>>>(lev.d_rho, lev.d_cell_volume, lev.d_shell_mass, lev.nr, lev.nt, lev.ng);
    int np2 = 1;
    while (np2 < lev.nr) np2 <<= 1;
    k_fas_gravity_from_shells<<<1, np2, np2*sizeof(double)>>>(lev.d_shell_mass, lev.d_r_center, lev.d_gr, lev.nr, G_const, M_core);
}

void SimpleSolver::apply_floor() {
    int n = lev.nr*lev.nt, B = 256;
    k_fas_floor<<<(n+B-1)/B,B>>>(lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE, lev.nr, lev.nt, lev.ng, gamma);
}

void SimpleSolver::compute_residual() {
    int n = lev.nr*lev.nt, B = 256;
    launch_ghost();
    compute_gravity_1d();
    k_fas_residual<<<(n+B-1)/B,B>>>(lev.d_rho,lev.d_mr,lev.d_mt,lev.d_rhoE,
        lev.d_cell_volume,lev.d_area_r,lev.d_area_theta,lev.d_r_center,lev.d_r_face,
        lev.d_theta_face,lev.d_dr,lev.d_dtheta,lev.d_gr,lev.d_gr0,lev.d_P0,lev.d_rho0,
        lev.d_res, lev.nr,lev.nt,lev.ng,gamma,atm_rho_thresh, 1, 0, 0);
    if (!use_core_excision) {
        k_fas_residual_origin<<<(lev.nt+B-1)/B,B>>>(lev.d_rho,lev.d_mr,lev.d_mt,lev.d_rhoE,
            lev.d_cell_volume,lev.d_area_r,lev.d_area_theta,lev.d_r_center,lev.d_r_face,
            lev.d_theta_face,lev.d_dr,lev.d_dtheta,lev.d_gr,lev.d_gr0,lev.d_P0,lev.d_rho0,
            lev.d_res, lev.nr,lev.nt,lev.ng,gamma,atm_rho_thresh, 1, 0, 0);
    }
    k_fas_axpy<<<(4*n+B-1)/B,B>>>(lev.d_res, -1.0, lev.d_hse_defect, 4*n);
}

void SimpleSolver::compute_F(double g0_over_dt) {
    int n = lev.nr*lev.nt, B = 256;
    compute_residual();
    k_fas_compute_F<<<(n+B-1)/B,B>>>(lev.d_res, lev.d_res, lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE,
        lev.d_fas_rhs, g0_over_dt, lev.nr, lev.nt, lev.ng);
}

void SimpleSolver::assemble_precond(double g0_over_dt) {
    int n = lev.nr*lev.nt, B = 256;
    k_fas_assemble_blkjac<<<(n+B-1)/B,B>>>(lev.d_rho,lev.d_mr,lev.d_mt,lev.d_rhoE,
        lev.d_cell_volume,lev.d_area_r,lev.d_area_theta,lev.d_r_center,lev.d_r_face,
        lev.d_theta_face,lev.d_dr,lev.d_dtheta,lev.d_gr0,
        lev.d_blk_inv, lev.nr,lev.nt,lev.ng,gamma,g0_over_dt);
    k_fas_mom_diag<<<(n+B-1)/B,B>>>(lev.d_rho,lev.d_mr,lev.d_mt,lev.d_rhoE,
        lev.d_dr,lev.d_r_center,lev.d_r_face,lev.d_dtheta,
        lev.d_Ap, lev.nr,lev.nt,lev.ng,g0_over_dt,gamma);
}

double SimpleSolver::residual_norm() {
    int n = lev.nr*lev.nt;
    std::vector<double> h(4*n), h_rho0(n);
    CUDA_CHECK(cudaMemcpy(h.data(), lev.d_res, 4*n*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rho0.data(), lev.d_rho0, n*sizeof(double), cudaMemcpyDeviceToHost));
    double mx = 0;
    for (int flat = 0; flat < n; flat++) {
        if (h_rho0[flat] < atm_rho_thresh) continue;
        for (int eq = 0; eq < 4; eq++) mx = std::max(mx, std::fabs(h[eq*n+flat]));
    }
    return mx;
}

double SimpleSolver::compute_cfl_dt() {
    int n = lev.nr*lev.nt, B = 256;
    k_fas_cfl<<<(n+B-1)/B,B>>>(lev.d_rho,lev.d_mr,lev.d_mt,lev.d_rhoE,
        lev.d_dr,lev.d_r_center,lev.d_dtheta,lev.d_rho0,lev.d_dp,
        lev.nr,lev.nt,lev.ng,gamma,atm_rho_thresh, n_angular_avg);
    double mn = gpu_reduce_min(lev.d_dp, lev.d_poisson_rhs, n);
    return cfl_num * mn;
}

// ========================= One SIMPLE iteration ========================

void SimpleSolver::simple_iteration(double dt, double g0_over_dt) {
    int n = lev.nr*lev.nt, B = 256;

    // 1. Block-Jacobi: U -= ω * J⁻¹_diag * F(U)
    compute_F(g0_over_dt);
    k_fas_smooth_blkjac<<<(n+B-1)/B,B>>>(lev.d_rho,lev.d_mr,lev.d_mt,lev.d_rhoE,
        lev.d_res, lev.d_blk_inv, lev.d_rho0, atm_rho_thresh,
        OMEGA, lev.nr, lev.nt, lev.ng);
    apply_floor();

    // 2. SIMPLE pressure correction
    compute_F(g0_over_dt);
    k_fas_vstar<<<(n+B-1)/B,B>>>(lev.d_res, lev.d_Ap, lev.d_rho0, atm_rho_thresh, lev.d_vr_s, lev.d_vt_s, n);
    k_fas_div<<<(n+B-1)/B,B>>>(lev.d_vr_s, lev.d_vt_s, lev.d_cell_volume, lev.d_area_r, lev.d_area_theta, lev.d_div_s, lev.nr, lev.nt);
    k_fas_inv_ap<<<(n+B-1)/B,B>>>(lev.d_Ap, lev.d_inv_Ap, n);
    k_fas_prhs<<<(n+B-1)/B,B>>>(lev.d_div_s, lev.d_poisson_rhs, lev.nr, lev.nt);
    CUDA_CHECK(cudaMemset(lev.d_dp, 0, n*sizeof(double)));
    lev.pressure_gmg.solve_varcoeff(lev.d_inv_Ap, lev.d_poisson_rhs, lev.d_dp, 2, 1e-2);
    k_fas_simple_correct<<<(n+B-1)/B,B>>>(lev.d_rho,lev.d_mr,lev.d_mt,lev.d_rhoE,
        lev.d_res, lev.d_blk_inv, lev.d_vr_s, lev.d_vt_s, lev.d_dp, lev.d_Ap,
        lev.d_r_center, lev.d_r_face, lev.d_theta_face,
        lev.d_rho0, atm_rho_thresh, lev.nr, lev.nt, lev.ng);
    apply_floor();

    // 3. Sponge
    if (sponge_r_start < sponge_r_top) {
        k_fas_sponge<<<(n+B-1)/B,B>>>(lev.d_rho,lev.d_mr,lev.d_mt,lev.d_rhoE,
            lev.d_rho0, lev.d_P0, lev.d_r_center,
            sponge_r_start, sponge_r_top, sponge_kappa, dt, 1.0/(gamma-1.0),
            lev.nr, lev.nt, lev.ng);
    }
}

// ========================= Time step ========================

double SimpleSolver::step(double t, double t_end) {
    if (!hse_set) snapshot_hse();
    apply_floor();

    int n = lev.nr*lev.nt, B = 256;

    if (dt_current < 1e-30) { dt_current = compute_cfl_dt(); if (dt_current < 1e-30) dt_current = 1e-8; }
    double dt_cfl = compute_cfl_dt();
    double dt = std::min({dt_current, 1.0, 200.0*dt_cfl, t_end - t});

    // Save Uⁿ
    if (step_count > 0)
        CUDA_CHECK(cudaMemcpy(lev.d_Un_prev, lev.d_Un, 4*n*sizeof(double), cudaMemcpyDeviceToDevice));
    k_fas_pack_flat<<<(n+B-1)/B,B>>>(lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE, lev.d_Un, lev.nr, lev.nt, lev.ng);
    if (step_count == 0)
        CUDA_CHECK(cudaMemcpy(lev.d_Un_prev, lev.d_Un, 4*n*sizeof(double), cudaMemcpyDeviceToDevice));

    // BDF2
    double norm = 1e30;
    int max_dt_cuts = 4;
    bool converged = false;
    int cuts = 0;

    for (cuts = 0; cuts < max_dt_cuts; cuts++) {
        bool use_bdf2 = (step_count > 0 && dt_prev > 1e-30);
        double gamma0, alpha1, alpha2;
        if (use_bdf2) {
            double w = dt/dt_prev;
            gamma0 = (1.0+2.0*w)/(1.0+w); alpha1 = -(1.0+w); alpha2 = w*w/(1.0+w);
        } else { gamma0 = 1.0; alpha1 = -1.0; alpha2 = 0.0; }
        double g0_over_dt = gamma0/dt;

        k_fas_bdf2_rhs<<<(4*n+B-1)/B,B>>>(lev.d_Un, lev.d_Un_prev, lev.d_fas_rhs, alpha1, alpha2, 1.0/dt, 4*n);

        assemble_precond(g0_over_dt);

        // SIMPLE iterations
        for (int iter = 0; iter < N_INNER; iter++)
            simple_iteration(dt, g0_over_dt);

        compute_F(g0_over_dt);
        norm = residual_norm();
        converged = !std::isnan(norm) && (norm < TOL || norm*dt < 1e-6);
        if (converged) break;

        // Reject: restore Uⁿ, halve dt
        k_fas_unpack_flat<<<(n+B-1)/B,B>>>(lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE, lev.d_Un, lev.nr, lev.nt, lev.ng);
        launch_ghost();
        double dt_old = dt; dt *= 0.5;
        if (step_count % 50 == 0 || cuts > 0)
            std::fprintf(stderr, "  step %d: reject dt=%.2e (||F||=%.2e), retry %.2e\n", step_count, dt_old, norm, dt);
    }

    if (!converged) {
        std::fprintf(stderr, "  step %d: rollback to Un (||F||=%.2e, dt=%.2e)\n", step_count, norm, dt);
        k_fas_unpack_flat<<<(n+B-1)/B,B>>>(lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE, lev.d_Un, lev.nr, lev.nt, lev.ng);
        launch_ghost();
        dt = dt_current * 0.25;
    }

    if (step_count % 100 == 0)
        std::fprintf(stderr, "  step %d dt=%.2e ||F||=%.2e cuts=%d\n", step_count, dt, norm, cuts);

    {
        double* d_rhoV = lev.d_res;
        double* d_EV   = lev.d_res + n;
        double* d_scr  = lev.d_res + 2*n;

        k_fas_rhoV_EV<<<(n+B-1)/B,B>>>(
            lev.d_rho, lev.d_rhoE, lev.d_cell_volume,
            d_rhoV, d_EV, lev.nr, lev.nt, lev.ng);
        double M_before = fas_reduce_sum(d_rhoV, d_scr, n);
        double E_before = fas_reduce_sum(d_EV, d_scr, n);

        k_fas_atm_reset<<<(n+B-1)/B,B>>>(lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE,
            lev.d_rho0, lev.d_P0, atm_rho_thresh, 1.0/(gamma-1.0), lev.nr, lev.nt, lev.ng);

        k_fas_rhoV_EV<<<(n+B-1)/B,B>>>(
            lev.d_rho, lev.d_rhoE, lev.d_cell_volume,
            d_rhoV, d_EV, lev.nr, lev.nt, lev.ng);
        double M_after = fas_reduce_sum(d_rhoV, d_scr, n);
        double E_after = fas_reduce_sum(d_EV, d_scr, n);

        if (interior_volume > 0) {
            double dM = (M_before - M_after) / interior_volume;
            double dE = (E_before - E_after) / interior_volume;
            if (fabs(dM) > 1e-30 || fabs(dE) > 1e-30)
                k_fas_conserve_correct<<<(n+B-1)/B,B>>>(
                    lev.d_rho, lev.d_rhoE, lev.d_rho0, atm_rho_thresh,
                    dM, dE, lev.nr, lev.nt, lev.ng);
        }
    }

    if (n_angular_avg > 0) {
        int B2 = std::min(lev.nt, 256);
        k_fas_angular_avg<<<n_angular_avg, B2, 5*B2*sizeof(double)>>>(
            lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE,
            lev.d_cell_volume,
            n_angular_avg, lev.nr, lev.nt, lev.ng);
    }
    if (n_pole_avg > 0) {
        k_fas_pole_avg<<<lev.nr, 1>>>(
            lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE,
            lev.d_cell_volume,
            n_pole_avg, lev.nr, lev.nt, lev.ng);
    }

    if (converged) dt_current = std::min(1.2*dt, 1.0); else dt_current = dt;
    dt_prev = dt;
    step_count++;
    return dt;
}

// ========================= Destroy ========================

void SimpleSolver::destroy() {
    cudaFree(lev.d_r_face); cudaFree(lev.d_r_center); cudaFree(lev.d_dr);
    cudaFree(lev.d_theta_face); cudaFree(lev.d_theta_center); cudaFree(lev.d_dtheta);
    cudaFree(lev.d_cell_volume); cudaFree(lev.d_area_r); cudaFree(lev.d_area_theta);
    cudaFree(lev.d_sin_theta_face); cudaFree(lev.d_sin_theta_center);
    cudaFree(lev.d_rho); cudaFree(lev.d_mr); cudaFree(lev.d_mt); cudaFree(lev.d_rhoE);
    cudaFree(lev.d_fas_rhs); cudaFree(lev.d_res); cudaFree(lev.d_Un); cudaFree(lev.d_Un_prev);
    cudaFree(lev.d_rho0); cudaFree(lev.d_P0); cudaFree(lev.d_hse_defect);
    cudaFree(lev.d_gr); cudaFree(lev.d_gr0); cudaFree(lev.d_shell_mass);
    cudaFree(lev.d_blk_inv);
    cudaFree(lev.d_Ap); cudaFree(lev.d_vr_s); cudaFree(lev.d_vt_s);
    cudaFree(lev.d_div_s); cudaFree(lev.d_dp); cudaFree(lev.d_poisson_rhs); cudaFree(lev.d_inv_Ap);
    lev.pressure_gmg.destroy();
}
