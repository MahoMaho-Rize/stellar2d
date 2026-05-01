// Well-Balanced 2D Eulerian solver — orchestration (init, destroy, step).
// Kernels live in wb2d_kernels.cu.

#include "wb2d_solver.cuh"
#include "fas_common.cuh"
#include "fas_linalg.cuh"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

// ======= Forward declarations of kernels in wb2d_kernels.cu =======
__global__ void k_wb2d_ghost_r_in(double*, double*, double*, double*, int, int, int);
__global__ void k_wb2d_ghost_r_out_hse(double*, double*, double*, double*,
    const double*, const double*, double, int, int, int);
__global__ void k_wb2d_ghost_t_n(double*, double*, double*, double*, int, int, int);
__global__ void k_wb2d_ghost_t_s(double*, double*, double*, double*, int, int, int);
__global__ void k_wb2d_pole_lock(double*, int, int, int);

__global__ void k_wb2d_shell_mass(const double*, const double*, double*, int, int, int);
__global__ void k_wb2d_gravity_from_shells(const double*, const double*, double*, int, double);

__global__ void k_wb2d_tw_viscosity(const double*, const double*, const double*,
    const double*, const double*, double*, double*, int, int, int, double, double, double);

__global__ void k_wb2d_floor(double*, double*, double*, double*,
    const double*, const double*, double, double, double, int, int, int);

__global__ void k_wb2d_cfl(const double*, const double*, const double*,
    const double*, const double*, const double*, const double*, const double*,
    double*, int, int, int, double, double, double);

__global__ void k_wb2d_residual(
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*,
    const double*, const double*,
    const double*, const double*,
    const double*, const double*,
    const double*, const double*,
    double*, int, int, int, double, int, int);

__global__ void k_wb2d_residual_origin(
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*,
    const double*, const double*,
    const double*, const double*,
    const double*,
    double*, int, int, int, double, int);

__global__ void k_wb2d_pack(const double*, const double*, const double*, const double*,
    double*, int, int, int);
__global__ void k_wb2d_unpack(double*, double*, double*, double*,
    const double*, int, int, int);
__global__ void k_wb2d_rk_update(double*, double*, double*, double*,
    const double*, double, int, int, int);
__global__ void k_wb2d_rk_average(double*, double*, double*, double*,
    const double*, int, int, int);
__global__ void k_wb2d_axpy(double*, double, const double*, int);
__global__ void k_wb2d_central_damp(double*, double*, const double*, const double*,
    double, double, int, int, int);
__global__ void k_wb2d_angular_avg(double*, double*, double*, double*,
    const double*, int, int, int, int);
__global__ void k_wb2d_pole_avg(double*, double*, double*, double*,
    const double*, int, int, int, int);
__global__ void k_wb2d_sponge(double*, double*, double*, double*,
    const double*, const double*, const double*,
    double, double, double, double, double, int, int, int);

// ============================================================
// init / destroy
// ============================================================
void Wb2DSolver::init(const Grid& grid, const EOS& eos, double G, double cfl) {
    gamma = eos.gamma;
    G_const = G;
    cfl_num = cfl;

    nr = grid.nr;
    nt = grid.ntheta;
    ng = grid.ng;
    total = (nr + 2*ng) * (nt + 2*ng);
    phys = nr * nt;

    auto mal = [](double** p, size_t nbytes) {
        CUDA_CHECK(cudaMalloc(p, nbytes));
        CUDA_CHECK(cudaMemset(*p, 0, nbytes));
    };

    // state
    mal(&d_rho,  total*sizeof(double));
    mal(&d_mr,   total*sizeof(double));
    mal(&d_mt,   total*sizeof(double));
    mal(&d_rhoE, total*sizeof(double));

    // HSE
    mal(&d_rho0,       phys*sizeof(double));
    mal(&d_P0,         phys*sizeof(double));
    mal(&d_hse_defect, 4*phys*sizeof(double));

    // TW viscosity
    mal(&d_Pvsc_r, phys*sizeof(double));
    mal(&d_Pvsc_t, phys*sizeof(double));

    // gravity
    mal(&d_gr,          nr*sizeof(double));
    mal(&d_gr0,         nr*sizeof(double));
    mal(&d_shell_mass,  nr*sizeof(double));

    // RK2 scratch
    mal(&d_Un,      4*phys*sizeof(double));
    mal(&d_res,     4*phys*sizeof(double));
    mal(&d_dt_cell, phys*sizeof(double));
    mal(&d_reduce_buf, phys*sizeof(double));
    mal(&d_reduce_out, phys*sizeof(double));

    // grid geometry
    mal(&d_r_face,      (nr+1)*sizeof(double));
    mal(&d_r_center,    nr*sizeof(double));
    mal(&d_dr,          nr*sizeof(double));
    mal(&d_theta_face,  (nt+1)*sizeof(double));
    mal(&d_theta_center,nt*sizeof(double));
    mal(&d_dtheta,      nt*sizeof(double));
    mal(&d_cell_volume, phys*sizeof(double));
    mal(&d_area_r,      (nr+1)*nt*sizeof(double));
    mal(&d_area_theta,  nr*(nt+1)*sizeof(double));

    // Upload geometry
    std::vector<double> h_rc(nr), h_dr(nr);
    for (int i = 0; i < nr; ++i) {
        h_rc[i] = 0.5*(grid.r_face[i] + grid.r_face[i+1]);
        h_dr[i] = grid.r_face[i+1] - grid.r_face[i];
    }
    std::vector<double> h_tc(nt), h_dt(nt);
    for (int j = 0; j < nt; ++j) {
        h_tc[j] = 0.5*(grid.theta_face[j] + grid.theta_face[j+1]);
        h_dt[j] = grid.theta_face[j+1] - grid.theta_face[j];
    }
    std::vector<double> h_vol(phys);
    for (int i = 0; i < nr; ++i) {
        double r3h = grid.r_face[i+1]*grid.r_face[i+1]*grid.r_face[i+1];
        double r3l = grid.r_face[i]*grid.r_face[i]*grid.r_face[i];
        for (int j = 0; j < nt; ++j)
            h_vol[i*nt + j] = (r3h - r3l)/3.0 *
                              (std::cos(grid.theta_face[j]) - std::cos(grid.theta_face[j+1]));
    }
    std::vector<double> h_ar((nr+1)*nt), h_at(nr*(nt+1));
    for (int i = 0; i <= nr; ++i) {
        double rf = grid.r_face[i];
        for (int j = 0; j < nt; ++j)
            h_ar[i*nt + j] = rf*rf *
                             (std::cos(grid.theta_face[j]) - std::cos(grid.theta_face[j+1]));
    }
    for (int i = 0; i < nr; ++i) {
        double r3h = grid.r_face[i+1]*grid.r_face[i+1]*grid.r_face[i+1];
        double r3l = grid.r_face[i]*grid.r_face[i]*grid.r_face[i];
        for (int j = 0; j <= nt; ++j)
            h_at[i*(nt+1) + j] = (r3h - r3l)/3.0 * std::sin(grid.theta_face[j]);
    }

    CUDA_CHECK(cudaMemcpy(d_r_face,   grid.r_face.data(),   (nr+1)*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r_center, h_rc.data(),          nr*sizeof(double),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dr,       h_dr.data(),          nr*sizeof(double),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_theta_face,  grid.theta_face.data(), (nt+1)*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_theta_center,h_tc.data(),       nt*sizeof(double),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dtheta,      h_dt.data(),       nt*sizeof(double),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cell_volume, h_vol.data(),      phys*sizeof(double),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_area_r,      h_ar.data(),       (nr+1)*nt*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_area_theta,  h_at.data(),       nr*(nt+1)*sizeof(double), cudaMemcpyHostToDevice));

    std::fprintf(stderr, "Wb2DSolver initialized: %dx%d, γ=%g, CFL=%g, CQ=%g, ZSH=%g\n",
                 nr, nt, gamma, cfl_num, CQ, ZSH);
}

void Wb2DSolver::destroy() {
    auto f = [](double* p) { if (p) cudaFree(p); };
    f(d_rho); f(d_mr); f(d_mt); f(d_rhoE);
    f(d_rho0); f(d_P0); f(d_hse_defect);
    f(d_Pvsc_r); f(d_Pvsc_t);
    f(d_gr); f(d_gr0); f(d_shell_mass);
    f(d_Un); f(d_res); f(d_dt_cell);
    f(d_reduce_buf); f(d_reduce_out);
    f(d_r_face); f(d_r_center); f(d_dr);
    f(d_theta_face); f(d_theta_center); f(d_dtheta);
    f(d_cell_volume); f(d_area_r); f(d_area_theta);
    std::memset(this, 0, sizeof(*this));
}

// ============================================================
// upload / download
// ============================================================
void Wb2DSolver::upload_state(const Grid& grid, const State& state) {
    int stride = nt + 2*ng;
    std::vector<double> h_rho(total,0), h_mr(total,0), h_mt(total,0), h_rhoE(total,0);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nt; ++j) {
            int k = (i+ng)*stride + (j+ng);
            int sk = grid.idx(i, j);
            h_rho[k]  = state.rho[sk];
            h_mr[k]   = state.mr[sk];
            h_mt[k]   = state.mtheta[sk];
            h_rhoE[k] = state.E[sk];
        }
    CUDA_CHECK(cudaMemcpy(d_rho,  h_rho.data(),  total*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mr,   h_mr.data(),   total*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mt,   h_mt.data(),   total*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhoE, h_rhoE.data(), total*sizeof(double), cudaMemcpyHostToDevice));
}

void Wb2DSolver::download_state(const Grid& grid, State& state) {
    int stride = nt + 2*ng;
    std::vector<double> h_rho(total), h_mr(total), h_mt(total), h_rhoE(total);
    CUDA_CHECK(cudaMemcpy(h_rho.data(),  d_rho,  total*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mr.data(),   d_mr,   total*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mt.data(),   d_mt,   total*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rhoE.data(), d_rhoE, total*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nt; ++j) {
            int k = (i+ng)*stride + (j+ng);
            int sk = grid.idx(i, j);
            state.rho[sk]    = h_rho[k];
            state.mr[sk]     = h_mr[k];
            state.mtheta[sk] = h_mt[k];
            state.E[sk]      = h_rhoE[k];
        }
}

// ============================================================
// Helpers used internally
// ============================================================
static void fill_ghost(Wb2DSolver& S) {
    int B = 256;
    { dim3 g((S.nt+B-1)/B, S.ng);
      k_wb2d_ghost_r_in<<<g, B>>>(S.d_rho, S.d_mr, S.d_mt, S.d_rhoE, S.nr, S.nt, S.ng); }
    { dim3 g((S.nt+B-1)/B, S.ng);
      k_wb2d_ghost_r_out_hse<<<g, B>>>(S.d_rho, S.d_mr, S.d_mt, S.d_rhoE,
          S.d_rho0, S.d_P0, 1.0/(S.gamma-1.0), S.nr, S.nt, S.ng); }
    { dim3 g((S.nr+2*S.ng+B-1)/B, S.ng);
      k_wb2d_ghost_t_n<<<g, B>>>(S.d_rho, S.d_mr, S.d_mt, S.d_rhoE, S.nr, S.nt, S.ng); }
    { dim3 g((S.nr+2*S.ng+B-1)/B, S.ng);
      k_wb2d_ghost_t_s<<<g, B>>>(S.d_rho, S.d_mr, S.d_mt, S.d_rhoE, S.nr, S.nt, S.ng); }
    int ntot = S.nr + 2*S.ng;
    k_wb2d_pole_lock<<<(ntot+B-1)/B, B>>>(S.d_mt, S.nr, S.nt, S.ng);
}

static void compute_gravity(Wb2DSolver& S) {
    int B = std::min(S.nt, 256);
    k_wb2d_shell_mass<<<S.nr, B, B*sizeof(double)>>>(
        S.d_rho, S.d_cell_volume, S.d_shell_mass, S.nr, S.nt, S.ng);
    int np2 = 1;
    while (np2 < S.nr) np2 <<= 1;
    k_wb2d_gravity_from_shells<<<1, np2, np2*sizeof(double)>>>(
        S.d_shell_mass, S.d_r_center, S.d_gr, S.nr, S.G_const);
}

static void compute_residual(Wb2DSolver& S) {
    int B = 256;
    fill_ghost(S);
    compute_gravity(S);
    // TW viscosity (cell-centered, uses ghost via fill_ghost)
    k_wb2d_tw_viscosity<<<(S.phys+B-1)/B, B>>>(
        S.d_rho, S.d_mr, S.d_mt, S.d_rhoE, S.d_cell_volume,
        S.d_Pvsc_r, S.d_Pvsc_t,
        S.nr, S.nt, S.ng, S.gamma, S.CQ, S.ZSH);
    // Residual
    k_wb2d_residual<<<(S.phys+B-1)/B, B>>>(
        S.d_rho, S.d_mr, S.d_mt, S.d_rhoE,
        S.d_cell_volume, S.d_area_r, S.d_area_theta,
        S.d_r_center, S.d_r_face,
        S.d_gr, S.d_gr0, S.d_P0, S.d_rho0,
        S.d_Pvsc_r, S.d_Pvsc_t,
        S.d_res, S.nr, S.nt, S.ng, S.gamma,
        S.limiter_type, S.use_lm_hllc);
    // Origin cell (i=0) — overwrites row j=0..nt-1 of residual
    k_wb2d_residual_origin<<<(S.nt+B-1)/B, B>>>(
        S.d_rho, S.d_mr, S.d_mt, S.d_rhoE,
        S.d_cell_volume, S.d_area_r, S.d_r_face,
        S.d_gr, S.d_gr0, S.d_P0, S.d_rho0,
        S.d_Pvsc_r,
        S.d_res, S.nr, S.nt, S.ng, S.gamma, S.use_lm_hllc);
    // Subtract discrete HSE defect: R_WB(U) - R_WB(U₀)
    k_wb2d_axpy<<<(4*S.phys+B-1)/B, B>>>(S.d_res, -1.0, S.d_hse_defect, 4*S.phys);
}

static void apply_floor(Wb2DSolver& S) {
    int B = 256;
    k_wb2d_floor<<<(S.phys+B-1)/B, B>>>(
        S.d_rho, S.d_mr, S.d_mt, S.d_rhoE,
        S.d_rho0, S.d_P0,
        S.rho_floor_frac, S.P_floor_frac, 1.0/(S.gamma-1.0),
        S.nr, S.nt, S.ng);
}

// ============================================================
// snapshot_hse: capture current state as HSE reference and
// pre-compute the discrete defect so R_WB(U₀) = 0 exactly.
// ============================================================
void Wb2DSolver::snapshot_hse() {
    int stride = nt + 2*ng;
    // First: fill ghost + gravity so g₀ is consistent
    fill_ghost(*this);
    compute_gravity(*this);
    CUDA_CHECK(cudaMemcpy(d_gr0, d_gr, nr*sizeof(double), cudaMemcpyDeviceToDevice));

    // Extract ρ₀, P₀ from current state (v=0 assumed, so P = (γ-1)·rhoE)
    std::vector<double> h_rho(total), h_rhoE(total);
    CUDA_CHECK(cudaMemcpy(h_rho.data(),  d_rho,  total*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rhoE.data(), d_rhoE, total*sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> h_rho0(phys), h_P0(phys);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nt; ++j) {
            int flat = i*nt + j;
            int k = (i+ng)*stride + (j+ng);
            h_rho0[flat] = h_rho[k];
            h_P0[flat]   = (gamma - 1.0) * h_rhoE[k];
        }
    CUDA_CHECK(cudaMemcpy(d_rho0, h_rho0.data(), phys*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_P0,   h_P0.data(),   phys*sizeof(double), cudaMemcpyHostToDevice));

    // Compute discrete HSE defect R_WB(U₀) ≡ defect.
    // Save real state, load HSE state, call compute_residual (with defect=0),
    // capture res into d_hse_defect, restore.
    std::vector<double> save_rho(total), save_mr(total), save_mt(total), save_rhoE(total);
    CUDA_CHECK(cudaMemcpy(save_rho.data(),  d_rho,  total*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(save_mr.data(),   d_mr,   total*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(save_mt.data(),   d_mt,   total*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(save_rhoE.data(), d_rhoE, total*sizeof(double), cudaMemcpyDeviceToHost));

    // Load HSE state: ρ=ρ₀, v=0, rhoE = P₀/(γ-1)
    std::vector<double> h_rho_hse(total, 1e-20), h_mr_hse(total, 0.0),
                        h_mt_hse(total, 0.0),  h_rhoE_hse(total, 1e-20);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nt; ++j) {
            int flat = i*nt + j;
            int k = (i+ng)*stride + (j+ng);
            h_rho_hse[k]  = h_rho0[flat];
            h_rhoE_hse[k] = h_P0[flat] / (gamma - 1.0);
        }
    CUDA_CHECK(cudaMemcpy(d_rho,  h_rho_hse.data(),  total*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mr,   h_mr_hse.data(),   total*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mt,   h_mt_hse.data(),   total*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhoE, h_rhoE_hse.data(), total*sizeof(double), cudaMemcpyHostToDevice));

    // Temporarily zero hse_defect so compute_residual returns raw R_WB(U₀)
    CUDA_CHECK(cudaMemset(d_hse_defect, 0, 4*phys*sizeof(double)));
    compute_residual(*this);
    CUDA_CHECK(cudaMemcpy(d_hse_defect, d_res, 4*phys*sizeof(double), cudaMemcpyDeviceToDevice));

    // Restore
    CUDA_CHECK(cudaMemcpy(d_rho,  save_rho.data(),  total*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mr,   save_mr.data(),   total*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mt,   save_mt.data(),   total*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhoE, save_rhoE.data(), total*sizeof(double), cudaMemcpyHostToDevice));

    double rho_max = *std::max_element(h_rho0.begin(), h_rho0.end());

    // Auto-configure sponge if user hasn't set sponge_kappa.
    // Start where ρ₀ drops below 1% of ρ_max (equatorial column).
    if (sponge_kappa > 0.0 && sponge_r_top <= 0.0) {
        std::vector<double> h_rc(nr), h_rf(nr+1);
        CUDA_CHECK(cudaMemcpy(h_rc.data(), d_r_center, nr*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_rf.data(), d_r_face,   (nr+1)*sizeof(double), cudaMemcpyDeviceToHost));
        sponge_r_top = h_rf[nr];
        sponge_r_start = sponge_r_top;
        double sponge_density = 0.01 * rho_max;
        for (int i = nr-1; i >= 0; --i) {
            double rho_eq = h_rho0[i*nt + nt/2];
            if (rho_eq > sponge_density) { sponge_r_start = h_rc[i]; break; }
        }
    }

    hse_set = true;
    std::fprintf(stderr, "  Wb2D HSE snapshot: ρ_max=%.3e, P_floor_frac=%.2g, "
                 "n_ang=%d n_pole=%d sponge=[%.3f,%.3f]\n",
                 rho_max, P_floor_frac, n_angular_avg, n_pole_avg,
                 sponge_r_start, sponge_r_top);
}

// ============================================================
// CFL
// ============================================================
double Wb2DSolver::compute_cfl_dt() {
    int B = 256;
    fill_ghost(*this);
    k_wb2d_cfl<<<(phys+B-1)/B, B>>>(
        d_rho, d_mr, d_mt, d_rhoE, d_rho0,
        d_dr, d_r_center, d_dtheta,
        d_dt_cell, nr, nt, ng, gamma, rho_floor_frac, comp_dt_frac);
    double dt_min = gpu_reduce_min(d_dt_cell, d_reduce_buf, phys);
    return cfl_num * dt_min;
}

// ============================================================
// step: RK2 (Heun) with floor before/after, central damp optional
// ============================================================
double Wb2DSolver::step(double t, double t_end) {
    if (!hse_set) snapshot_hse();

    int B = 256;

    apply_floor(*this);

    double dt = compute_cfl_dt();
    if (dt < 1e-30) dt = 1e-12;
    if (t + dt > t_end) dt = t_end - t;

    // Save Uⁿ
    k_wb2d_pack<<<(phys+B-1)/B, B>>>(d_rho, d_mr, d_mt, d_rhoE, d_Un, nr, nt, ng);

    // --- Stage 1: U* = Uⁿ + dt · R(Uⁿ) ---
    compute_residual(*this);
    k_wb2d_rk_update<<<(phys+B-1)/B, B>>>(d_rho, d_mr, d_mt, d_rhoE, d_res, dt, nr, nt, ng);
    apply_floor(*this);

    // --- Stage 2: compute R(U*) then accumulate ---
    compute_residual(*this);
    k_wb2d_rk_update<<<(phys+B-1)/B, B>>>(d_rho, d_mr, d_mt, d_rhoE, d_res, dt, nr, nt, ng);

    // Average: Uⁿ⁺¹ = ½(Uⁿ + U**)
    k_wb2d_rk_average<<<(phys+B-1)/B, B>>>(d_rho, d_mr, d_mt, d_rhoE, d_Un, nr, nt, ng);
    apply_floor(*this);

    // Central v_r damping (conservative, θ-symmetric)
    if (central_damp_r > 0.0) {
        k_wb2d_central_damp<<<(phys+B-1)/B, B>>>(
            d_mr, d_rhoE, d_rho, d_r_center,
            central_damp_r, central_damp_alpha, nr, nt, ng);
    }

    // Sponge layer: absorb outgoing waves into isothermal HSE buffer
    if (sponge_kappa > 0.0 && sponge_r_start < sponge_r_top) {
        k_wb2d_sponge<<<(phys+B-1)/B, B>>>(
            d_rho, d_mr, d_mt, d_rhoE, d_rho0, d_P0, d_r_center,
            sponge_r_start, sponge_r_top, sponge_kappa, dt,
            1.0/(gamma-1.0), nr, nt, ng);
    }

    // Angular averaging: treats the r→0 geometric focusing (θ-symmetric-safe)
    if (n_angular_avg > 0) {
        int B2 = std::min(nt, 256);
        k_wb2d_angular_avg<<<n_angular_avg, B2, 5*B2*sizeof(double)>>>(
            d_rho, d_mr, d_mt, d_rhoE, d_cell_volume,
            n_angular_avg, nr, nt, ng);
    }
    // Pole wedge averaging: treats sin θ→0 amplification at θ=0, π
    if (n_pole_avg > 0) {
        k_wb2d_pole_avg<<<nr, 1>>>(
            d_rho, d_mr, d_mt, d_rhoE, d_cell_volume,
            n_pole_avg, nr, nt, ng);
    }

    dt_current = dt;
    step_count++;
    if (step_count <= 10 || step_count % 1000 == 0)
        std::fprintf(stderr, "  [wb2d] step %d  t=%.4e  dt=%.3e\n", step_count, t+dt, dt);
    return dt;
}
