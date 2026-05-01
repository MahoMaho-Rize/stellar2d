// Well-Balanced 2D Eulerian solver kernels.
//
// All kernels are self-contained (only depend on fas_idx / CUDA_CHECK macro and
// inline HLLC/MUSCL helpers in fas_hllc.cuh). No new header is exposed.

#include "fas_common.cuh"
#include "fas_hllc.cuh"
#include <cmath>

// ===============================================================
// Ghost cells
// ===============================================================
__global__
void k_wb2d_ghost_r_in(double* rho, double* mr, double* mt, double* rhoE,
                        int nr, int nt, int ng) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y + 1;
    if (j >= nt || g > ng) return;
    int kg = fas_idx(-g, j, nt, ng), kp = fas_idx(g-1, j, nt, ng);
    rho[kg] = rho[kp]; mr[kg] = -mr[kp]; mt[kg] = mt[kp]; rhoE[kg] = rhoE[kp];
}

__global__
void k_wb2d_ghost_r_out_hse(double* rho, double* mr, double* mt, double* rhoE,
                             const double* rho0, const double* P0,
                             double gam_m1_inv, int nr, int nt, int ng) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y;
    if (j >= nt || g >= ng) return;
    int kg = fas_idx(nr+g, j, nt, ng);
    int flat_last = (nr-1)*nt + j;
    rho[kg]  = fmax(rho0[flat_last], 1e-20);
    mr[kg]   = 0.0;
    mt[kg]   = 0.0;
    rhoE[kg] = fmax(P0[flat_last] * gam_m1_inv, 1e-20);
}

__global__
void k_wb2d_ghost_t_n(double* rho, double* mr, double* mt, double* rhoE,
                      int nr, int nt, int ng) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y + 1;
    if (ii >= nr + 2*ng || g > ng) return;
    int i = ii - ng;
    int kg = fas_idx(i, -g, nt, ng), kp = fas_idx(i, g-1, nt, ng);
    rho[kg] = rho[kp]; mr[kg] = mr[kp]; mt[kg] = -mt[kp]; rhoE[kg] = rhoE[kp];
}

__global__
void k_wb2d_ghost_t_s(double* rho, double* mr, double* mt, double* rhoE,
                      int nr, int nt, int ng) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y;
    if (ii >= nr + 2*ng || g >= ng) return;
    int i = ii - ng;
    int kg = fas_idx(i, nt+g, nt, ng), kp = fas_idx(i, nt-1-g, nt, ng);
    rho[kg] = rho[kp]; mr[kg] = mr[kp]; mt[kg] = -mt[kp]; rhoE[kg] = rhoE[kp];
}

__global__
void k_wb2d_pole_lock(double* mt, int nr, int nt, int ng) {
    int i = blockIdx.x * blockDim.x + threadIdx.x - ng;
    if (i < -ng || i >= nr + ng) return;
    mt[fas_idx(i, 0, nt, ng)] = 0.0;
    mt[fas_idx(i, nt - 1, nt, ng)] = 0.0;
}

// ===============================================================
// 1D angle-averaged gravity (reused from FAS scheme)
// ===============================================================
__global__
void k_wb2d_shell_mass(const double* rho, const double* vol,
                       double* shell, int nr, int nt, int ng) {
    int i = blockIdx.x;
    if (i >= nr) return;
    extern __shared__ double smem[];
    int tid = threadIdx.x;
    double s = 0.0;
    for (int j = tid; j < nt; j += blockDim.x)
        s += rho[fas_idx(i, j, nt, ng)] * vol[i*nt + j];
    smem[tid] = s;
    __syncthreads();
    for (int st = blockDim.x/2; st > 0; st >>= 1) {
        if (tid < st) smem[tid] += smem[tid+st];
        __syncthreads();
    }
    if (tid == 0) shell[i] = smem[0] * 2.0 * M_PI;
}

__global__
void k_wb2d_gravity_from_shells(const double* shell_mass, const double* rc,
                                double* gr, int nr, double G) {
    extern __shared__ double smem[];
    int tid = threadIdx.x;
    smem[tid] = (tid < nr) ? shell_mass[tid] : 0.0;
    __syncthreads();
    int n = blockDim.x;
    // Up-sweep
    for (int d = 1; d < n; d <<= 1) {
        int idx = (tid + 1) * (d << 1) - 1;
        if (idx < n) smem[idx] += smem[idx - d];
        __syncthreads();
    }
    if (tid == n - 1) smem[tid] = 0.0;
    __syncthreads();
    for (int d = n >> 1; d >= 1; d >>= 1) {
        int idx = (tid + 1) * (d << 1) - 1;
        if (idx < n) {
            double tmp = smem[idx - d];
            smem[idx - d] = smem[idx];
            smem[idx] += tmp;
        }
        __syncthreads();
    }
    if (tid < nr) {
        double M_enc = smem[tid] + shell_mass[tid];
        gr[tid] = -G * M_enc / (rc[tid] * rc[tid]);
    }
}

// ===============================================================
// TW artificial viscosity (per cell, per direction)
//   Q = (CQ/V) · max(0, Δv − ZSH·√(PV))²
//   Δv_r > 0 means radial compression  (v_r decreases outward)
//   Δv_θ > 0 means polar compression
// ===============================================================
__global__
void k_wb2d_tw_viscosity(const double* rho, const double* mr, const double* mt,
                         const double* rhoE, const double* vol,
                         double* Pvsc_r, double* Pvsc_t,
                         int nr, int nt, int ng, double gam,
                         double CQ, double ZSH) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat / nt, j = flat % nt;
    int k = fas_idx(i, j, nt, ng);

    double rho_c = fmax(rho[k], 1e-20);
    double vr_c = mr[k] / rho_c, vt_c = mt[k] / rho_c;
    double KE = 0.5 * rho_c * (vr_c*vr_c + vt_c*vt_c);
    double P_c = fmax((gam - 1.0) * (rhoE[k] - KE), 1e-30);
    double V = vol[flat];

    // v_r neighbors (ghost cells are filled so i-1/i+1 are safe)
    int km = fas_idx(i-1, j, nt, ng);
    int kp = fas_idx(i+1, j, nt, ng);
    double vr_m = mr[km] / fmax(rho[km], 1e-20);
    double vr_p = mr[kp] / fmax(rho[kp], 1e-20);
    double dvr = 0.5 * (vr_m - vr_p);

    int ks = fas_idx(i, j-1, nt, ng);
    int kn = fas_idx(i, j+1, nt, ng);
    double vt_s = mt[ks] / fmax(rho[ks], 1e-20);
    double vt_n = mt[kn] / fmax(rho[kn], 1e-20);
    double dvt = 0.5 * (vt_s - vt_n);

    double thresh = ZSH * sqrt(P_c * V);
    double qr = dvr - thresh;
    double qt = dvt - thresh;
    Pvsc_r[flat] = (qr > 0.0) ? (CQ / V) * qr * qr : 0.0;
    Pvsc_t[flat] = (qt > 0.0) ? (CQ / V) * qt * qt : 0.0;
}

// ===============================================================
// Pressure / density floor (uniform, symmetry-preserving)
//   ρ ≥ rho_floor_frac · ρ₀
//   P ≥ P_floor_frac · P₀  (adjust rhoE to match)
// ===============================================================
__global__
void k_wb2d_floor(double* rho, double* mr, double* mt, double* rhoE,
                  const double* rho0, const double* P0,
                  double rho_floor_frac, double P_floor_frac,
                  double gam_m1_inv,
                  int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int k = fas_idx(flat/nt, flat%nt, nt, ng);

    double r = rho[k], E = rhoE[k];
    if (isnan(r) || isnan(E) || isnan(mr[k]) || isnan(mt[k])) {
        rho[k] = fmax(rho0[flat], 1e-20);
        mr[k] = 0.0; mt[k] = 0.0;
        rhoE[k] = fmax(P0[flat] * gam_m1_inv, 1e-20);
        return;
    }

    double rho_fl = rho_floor_frac * rho0[flat];
    if (r < rho_fl) r = rho_fl;
    rho[k] = r;

    double KE = 0.5 * (mr[k]*mr[k] + mt[k]*mt[k]) / fmax(r, 1e-20);
    double e_int = E - KE;
    double gam_m1 = 1.0 / gam_m1_inv;
    double P = gam_m1 * e_int;
    double P_fl = P_floor_frac * P0[flat];
    if (P < P_fl) {
        P = P_fl;
        e_int = P * gam_m1_inv;
        rhoE[k] = e_int + KE;
    }
}

// ===============================================================
// CFL dt: acoustic_r, acoustic_t, compression_r, compression_t
// ===============================================================
__global__
void k_wb2d_cfl(const double* rho, const double* mr, const double* mt,
                const double* rhoE, const double* rho0,
                const double* dr, const double* r_center, const double* dtheta,
                double* out, int nr, int nt, int ng,
                double gam, double rho_floor_frac, double comp_dt_frac) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat / nt, j = flat % nt;
    int k = fas_idx(i, j, nt, ng);

    double rho_c = fmax(rho[k], 1e-20);
    // Skip near-vacuum cells (below floor)
    if (rho_c < rho_floor_frac * rho0[flat] * 1.01) { out[flat] = 1e30; return; }

    double vr = mr[k] / rho_c, vt = mt[k] / rho_c;
    double KE = 0.5 * rho_c * (vr*vr + vt*vt);
    double P = fmax((gam - 1.0) * (rhoE[k] - KE), 1e-30);
    double cs = sqrt(gam * P / rho_c);

    double dt_r = dr[i] / (fabs(vr) + cs);
    double dt_t = r_center[i] * dtheta[j] / (fabs(vt) + cs);

    // compression limit (v_inner > v_outer means compression)
    double vr_m = mr[fas_idx(i-1,j,nt,ng)] / fmax(rho[fas_idx(i-1,j,nt,ng)], 1e-20);
    double vr_p = mr[fas_idx(i+1,j,nt,ng)] / fmax(rho[fas_idx(i+1,j,nt,ng)], 1e-20);
    double dvr = 0.5 * (vr_m - vr_p);

    double vt_s = mt[fas_idx(i,j-1,nt,ng)] / fmax(rho[fas_idx(i,j-1,nt,ng)], 1e-20);
    double vt_n = mt[fas_idx(i,j+1,nt,ng)] / fmax(rho[fas_idx(i,j+1,nt,ng)], 1e-20);
    double dvt = 0.5 * (vt_s - vt_n);

    double dt_cr = (dvr > 0.0) ? comp_dt_frac * dr[i] / dvr : 1e30;
    double dt_ct = (dvt > 0.0) ? comp_dt_frac * r_center[i] * dtheta[j] / dvt : 1e30;

    double d = fmin(fmin(dt_r, dt_t), fmin(dt_cr, dt_ct));
    out[flat] = d;
}

// ===============================================================
// Main residual kernel: HLLC + MUSCL-WB + TW viscosity (both directions)
// Follows design doc §"Residual Kernel Design, Option 1".
// ===============================================================
__global__
void k_wb2d_residual(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* vol, const double* ar, const double* at,
    const double* r_center, const double* r_face,
    const double* gr, const double* gr0,
    const double* P0, const double* rho0,
    const double* Pvsc_r, const double* Pvsc_t,
    double* res, int nr, int nt, int ng, double gam,
    int lim_type, int use_lm_hllc) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat / nt, j = flat % nt;
    int n = nr * nt;

    int k = fas_idx(i, j, nt, ng);
    double invV = 1.0 / vol[flat];

    double rho_c = fmax(rho[k], 1e-20);
    double vr_c = mr[k] / rho_c, vt_c = mt[k] / rho_c;
    double r = r_center[i];

    auto W = [&](int ii, int jj) -> FPrim {
        int kk = fas_idx(ii, jj, nt, ng);
        double r_ = fmax(rho[kk], 1e-20);
        double vr_ = mr[kk] / r_, vt_ = mt[kk] / r_;
        FPrim w;
        w.rho = r_; w.vr = vr_; w.vt = vt_;
        w.P = fmax((gam - 1.0) * (rhoE[kk] - 0.5 * r_ * (vr_*vr_ + vt_*vt_)), 1e-30);
        return w;
    };
    auto P0r_lam = [&](int ii) -> double { return P0[max(0,min(ii,nr-1))*nt + j]; };
    auto r0r_lam = [&](int ii) -> double { return rho0[max(0,min(ii,nr-1))*nt + j]; };
    auto P0t_lam = [&](int jj) -> double { return P0[i*nt + max(0,min(jj,nt-1))]; };
    auto r0t_lam = [&](int jj) -> double { return rho0[i*nt + max(0,min(jj,nt-1))]; };
    // Viscosity at a neighboring cell (zero outside physical range)
    auto Qr = [&](int ii, int jj) -> double {
        if (ii < 0 || ii >= nr || jj < 0 || jj >= nt) return 0.0;
        return Pvsc_r[ii*nt + jj];
    };
    auto Qt = [&](int ii, int jj) -> double {
        if (ii < 0 || ii >= nr || jj < 0 || jj >= nt) return 0.0;
        return Pvsc_t[ii*nt + jj];
    };

    // --- r-direction faces ---
    auto hllc_r_face = [&](int face_i) -> FFlux4 {
        FPrim wa = W(face_i-2, j), wb_ = W(face_i-1, j), wc = W(face_i, j), wd = W(face_i+1, j);
        double ppL, ppR, rpL, rpR;
        fas_recon(wa.P - P0r_lam(face_i-2), wb_.P - P0r_lam(face_i-1),
                  wc.P - P0r_lam(face_i),   wd.P - P0r_lam(face_i+1),
                  ppL, ppR, lim_type);
        fas_recon(wa.rho - r0r_lam(face_i-2), wb_.rho - r0r_lam(face_i-1),
                  wc.rho - r0r_lam(face_i),   wd.rho - r0r_lam(face_i+1),
                  rpL, rpR, lim_type);
        double P0f = 0.5 * (P0r_lam(face_i-1) + P0r_lam(face_i));
        double r0f = 0.5 * (r0r_lam(face_i-1) + r0r_lam(face_i));
        FPrim wl, wr;
        fas_recon(wa.vr, wb_.vr, wc.vr, wd.vr, wl.vr, wr.vr, lim_type);
        fas_recon(wa.vt, wb_.vt, wc.vt, wd.vt, wl.vt, wr.vt, lim_type);
        wl.rho = fmax(r0f + rpL, 1e-20); wr.rho = fmax(r0f + rpR, 1e-20);
        // TW viscosity adds to pressure: use max(Q_L, Q_R) to stay single-valued
        double qL = Qr(face_i-1, j), qR = Qr(face_i, j);
        double qf = fmax(qL, qR);
        wl.P = fmax(P0f + ppL + qf, 1e-30);
        wr.P = fmax(P0f + ppR + qf, 1e-30);
        return use_lm_hllc ? fas_hllc_lm(wl, wr, gam, true) : fas_hllc(wl, wr, gam, true);
    };

    FFlux4 fr_hi = hllc_r_face(i+1);
    FFlux4 fr_lo = hllc_r_face(i);

    double Ar_hi = ar[(i+1)*nt + j], Ar_lo = ar[i*nt + j];
    double P0f_rhi = 0.5 * (P0r_lam(i)   + P0r_lam(i+1));
    double P0f_rlo = 0.5 * (P0r_lam(i-1) + P0r_lam(i));

    double div_rho = -invV * (Ar_hi * fr_hi.f_rho - Ar_lo * fr_lo.f_rho);
    double div_mr  = -invV * (Ar_hi * (fr_hi.f_mr - P0f_rhi) - Ar_lo * (fr_lo.f_mr - P0f_rlo));
    double div_mt  = -invV * (Ar_hi * fr_hi.f_mt - Ar_lo * fr_lo.f_mt);
    double div_E   = -invV * (Ar_hi * fr_hi.f_E  - Ar_lo * fr_lo.f_E);

    // --- θ-direction faces ---
    auto hllc_t_face = [&](int face_j) -> FFlux4 {
        FPrim wa = W(i, face_j-2), wb_ = W(i, face_j-1), wc = W(i, face_j), wd = W(i, face_j+1);
        double ppL, ppR, rpL, rpR;
        fas_recon(wa.P - P0t_lam(face_j-2), wb_.P - P0t_lam(face_j-1),
                  wc.P - P0t_lam(face_j),   wd.P - P0t_lam(face_j+1),
                  ppL, ppR, lim_type);
        fas_recon(wa.rho - r0t_lam(face_j-2), wb_.rho - r0t_lam(face_j-1),
                  wc.rho - r0t_lam(face_j),   wd.rho - r0t_lam(face_j+1),
                  rpL, rpR, lim_type);
        double P0f = 0.5 * (P0t_lam(face_j-1) + P0t_lam(face_j));
        double r0f = 0.5 * (r0t_lam(face_j-1) + r0t_lam(face_j));
        FPrim wl, wr;
        fas_recon(wa.vr, wb_.vr, wc.vr, wd.vr, wl.vr, wr.vr, lim_type);
        fas_recon(wa.vt, wb_.vt, wc.vt, wd.vt, wl.vt, wr.vt, lim_type);
        wl.rho = fmax(r0f + rpL, 1e-20); wr.rho = fmax(r0f + rpR, 1e-20);
        double qL = Qt(i, face_j-1), qR = Qt(i, face_j);
        double qf = fmax(qL, qR);
        wl.P = fmax(P0f + ppL + qf, 1e-30);
        wr.P = fmax(P0f + ppR + qf, 1e-30);
        return use_lm_hllc ? fas_hllc_lm(wl, wr, gam, false) : fas_hllc(wl, wr, gam, false);
    };

    FFlux4 ft_hi = hllc_t_face(j+1);
    FFlux4 ft_lo = hllc_t_face(j);

    double At_hi = at[i*(nt+1) + j+1], At_lo = at[i*(nt+1) + j];
    double P0f_thi = 0.5 * (P0t_lam(j)   + P0t_lam(j+1));
    double P0f_tlo = 0.5 * (P0t_lam(j-1) + P0t_lam(j));

    div_rho += -invV * (At_hi * ft_hi.f_rho - At_lo * ft_lo.f_rho);
    div_mr  += -invV * (At_hi * ft_hi.f_mr  - At_lo * ft_lo.f_mr);
    div_mt  += -invV * (At_hi * (ft_hi.f_mt - P0f_thi) - At_lo * (ft_lo.f_mt - P0f_tlo));
    div_E   += -invV * (At_hi * ft_hi.f_E   - At_lo * ft_lo.f_E);

    // --- WB gravity source: ρ'·g₀ + ρ·g' ---
    double rho_ref = rho0[flat];
    double g0_r = gr0[i];
    double rhop_c = rho_c - rho_ref;
    double gp_r = gr[i] - g0_r;
    double gravity_r = rhop_c * g0_r + rho_c * gp_r;

    // --- geometric source (vanishes under θ-symmetry) ---
    double inv_r = 1.0 / r;
    double S_mr = rho_c * vt_c * vt_c * inv_r;
    double S_mt = -rho_c * vr_c * vt_c * inv_r;

    // --- total-energy source: ρ·v·g (gravity does work) ---
    double S_E = rho_c * vr_c * (g0_r + gp_r);

    res[flat]       = div_rho;
    res[n + flat]   = div_mr + gravity_r + S_mr;
    res[2*n + flat] = div_mt + S_mt;
    res[3*n + flat] = div_E + S_E;
}

// ===============================================================
// Origin wedge residual (i = 0): only outer radial face has area.
// No θ flux (all wedges share r=0 point). Uses first-order HLLC
// (MUSCL stencil too short).
// ===============================================================
__global__
void k_wb2d_residual_origin(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* vol, const double* ar,
    const double* r_face,
    const double* gr, const double* gr0,
    const double* P0, const double* rho0,
    const double* Pvsc_r,
    double* res, int nr, int nt, int ng, double gam, int use_lm_hllc) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nt) return;
    int flat = j;
    int n = nr * nt;
    int k = fas_idx(0, j, nt, ng);
    double invV = 1.0 / vol[flat];

    double rho_c = fmax(rho[k], 1e-20);
    double vr_c = mr[k] / rho_c;

    // Outer r face: first-order HLLC on perturbations + WB subtraction
    auto W = [&](int ii, int jj) -> FPrim {
        int kk = fas_idx(ii, jj, nt, ng);
        double r_ = fmax(rho[kk], 1e-20);
        double vr_ = mr[kk] / r_, vt_ = mt[kk] / r_;
        FPrim w;
        w.rho = r_; w.vr = vr_; w.vt = vt_;
        w.P = fmax((gam - 1.0) * (rhoE[kk] - 0.5*r_*(vr_*vr_ + vt_*vt_)), 1e-30);
        return w;
    };

    FPrim wl = W(0, j), wr = W(1, j);
    double P0_0 = P0[0*nt + j], P0_1 = P0[1*nt + j];
    double r0_0 = rho0[0*nt + j], r0_1 = rho0[1*nt + j];
    double P0f = 0.5 * (P0_0 + P0_1), r0f = 0.5 * (r0_0 + r0_1);
    double q0 = Pvsc_r[0*nt + j], q1 = Pvsc_r[1*nt + j];
    double qf = fmax(q0, q1);
    wl.P   = fmax(P0f + (wl.P - P0_0) + qf, 1e-30);
    wr.P   = fmax(P0f + (wr.P - P0_1) + qf, 1e-30);
    wl.rho = fmax(r0f + (wl.rho - r0_0), 1e-20);
    wr.rho = fmax(r0f + (wr.rho - r0_1), 1e-20);
    FFlux4 fr_hi = use_lm_hllc ? fas_hllc_lm(wl, wr, gam, true) : fas_hllc(wl, wr, gam, true);

    double Ar_hi = ar[1*nt + j];
    double P0f_rhi = 0.5 * (P0_0 + P0_1);

    double div_rho = -invV * (Ar_hi * fr_hi.f_rho);
    double div_mr  = -invV * (Ar_hi * (fr_hi.f_mr - P0f_rhi));
    double div_E   = -invV * (Ar_hi * fr_hi.f_E);

    double rho_ref = rho0[flat];
    double g0_r = gr0[0];
    double rhop_c = rho_c - rho_ref;
    double gp_r = gr[0] - g0_r;
    double gravity_r = rhop_c * g0_r + rho_c * gp_r;
    double S_E = rho_c * vr_c * (g0_r + gp_r);

    res[flat]       = div_rho;
    res[n + flat]   = div_mr + gravity_r;
    res[2*n + flat] = 0.0;
    res[3*n + flat] = div_E + S_E;
}

// ===============================================================
// RK2 helpers: pack/unpack from ghosted layout, update, average
// ===============================================================
__global__
void k_wb2d_pack(const double* rho, const double* mr, const double* mt,
                 const double* rhoE, double* out,
                 int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int k = fas_idx(flat/nt, flat%nt, nt, ng);
    int n = nr*nt;
    out[flat]       = rho[k];
    out[n + flat]   = mr[k];
    out[2*n + flat] = mt[k];
    out[3*n + flat] = rhoE[k];
}

__global__
void k_wb2d_unpack(double* rho, double* mr, double* mt, double* rhoE,
                   const double* in, int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int k = fas_idx(flat/nt, flat%nt, nt, ng);
    int n = nr*nt;
    rho[k]  = in[flat];
    mr[k]   = in[n + flat];
    mt[k]   = in[2*n + flat];
    rhoE[k] = in[3*n + flat];
}

__global__
void k_wb2d_rk_update(double* rho, double* mr, double* mt, double* rhoE,
                      const double* R, double dt_val,
                      int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int k = fas_idx(flat/nt, flat%nt, nt, ng);
    int n = nr*nt;
    rho[k]  += dt_val * R[flat];
    mr[k]   += dt_val * R[n + flat];
    mt[k]   += dt_val * R[2*n + flat];
    rhoE[k] += dt_val * R[3*n + flat];
}

__global__
void k_wb2d_rk_average(double* rho, double* mr, double* mt, double* rhoE,
                       const double* Un, int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int k = fas_idx(flat/nt, flat%nt, nt, ng);
    int n = nr*nt;
    rho[k]  = 0.5 * (Un[flat]       + rho[k]);
    mr[k]   = 0.5 * (Un[n + flat]   + mr[k]);
    mt[k]   = 0.5 * (Un[2*n + flat] + mt[k]);
    rhoE[k] = 0.5 * (Un[3*n + flat] + rhoE[k]);
}

__global__
void k_wb2d_axpy(double* y, double a, const double* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

// ===============================================================
// Angular average on innermost n_avg shells (θ-symmetric IC-safe).
// One block per shell, threads reduce over j.
// ===============================================================
__global__
void k_wb2d_angular_avg(double* rho, double* mr, double* mt, double* rhoE,
                        const double* vol,
                        int n_avg, int nr, int nt, int ng) {
    int i = blockIdx.x;
    if (i >= n_avg || i >= nr) return;

    extern __shared__ double smem[];
    double* s_rho  = smem;
    double* s_mr   = smem + blockDim.x;
    double* s_mt   = smem + 2 * blockDim.x;
    double* s_rhoE = smem + 3 * blockDim.x;
    double* s_vol  = smem + 4 * blockDim.x;
    int tid = threadIdx.x;

    double sum_rho = 0, sum_mr = 0, sum_mt = 0, sum_rhoE = 0, sum_vol = 0;
    for (int j = tid; j < nt; j += blockDim.x) {
        int k = fas_idx(i, j, nt, ng);
        double v = vol[i*nt + j];
        sum_rho  += rho[k] * v;
        sum_mr   += mr[k] * v;
        sum_mt   += mt[k] * v;
        sum_rhoE += rhoE[k] * v;
        sum_vol  += v;
    }
    s_rho[tid] = sum_rho; s_mr[tid] = sum_mr;
    s_mt[tid] = sum_mt; s_rhoE[tid] = sum_rhoE; s_vol[tid] = sum_vol;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            s_rho[tid]  += s_rho[tid+s];
            s_mr[tid]   += s_mr[tid+s];
            s_mt[tid]   += s_mt[tid+s];
            s_rhoE[tid] += s_rhoE[tid+s];
            s_vol[tid]  += s_vol[tid+s];
        }
        __syncthreads();
    }
    double inv_V = 1.0 / fmax(s_vol[0], 1e-30);
    double a_rho = s_rho[0]*inv_V, a_mr = s_mr[0]*inv_V;
    double a_mt = s_mt[0]*inv_V, a_rhoE = s_rhoE[0]*inv_V;
    for (int j = tid; j < nt; j += blockDim.x) {
        int k = fas_idx(i, j, nt, ng);
        rho[k] = a_rho; mr[k] = a_mr; mt[k] = 0.0; rhoE[k] = a_rhoE;
    }
}

// ===============================================================
// Pole wedge average: merge n_pole cells near each pole.
// One block per radial shell.
// ===============================================================
__global__
void k_wb2d_pole_avg(double* rho, double* mr, double* mt, double* rhoE,
                     const double* vol, int n_pole, int nr, int nt, int ng) {
    int i = blockIdx.x;
    if (i >= nr) return;

    // North pole
    {
        double sr = 0, sm = 0, se = 0, sv = 0;
        for (int j = 0; j < n_pole && j < nt; ++j) {
            int k = fas_idx(i, j, nt, ng);
            double v = vol[i*nt + j];
            sr += rho[k]*v; sm += mr[k]*v; se += rhoE[k]*v; sv += v;
        }
        double inv = 1.0/fmax(sv, 1e-30);
        double ar = sr*inv, am = sm*inv, ae = se*inv;
        for (int j = 0; j < n_pole && j < nt; ++j) {
            int k = fas_idx(i, j, nt, ng);
            rho[k] = ar; mr[k] = am; mt[k] = 0.0; rhoE[k] = ae;
        }
    }
    // South pole
    {
        double sr = 0, sm = 0, se = 0, sv = 0;
        for (int j = nt - n_pole; j < nt; ++j) {
            if (j < 0) continue;
            int k = fas_idx(i, j, nt, ng);
            double v = vol[i*nt + j];
            sr += rho[k]*v; sm += mr[k]*v; se += rhoE[k]*v; sv += v;
        }
        double inv = 1.0/fmax(sv, 1e-30);
        double ar = sr*inv, am = sm*inv, ae = se*inv;
        for (int j = nt - n_pole; j < nt; ++j) {
            if (j < 0) continue;
            int k = fas_idx(i, j, nt, ng);
            rho[k] = ar; mr[k] = am; mt[k] = 0.0; rhoE[k] = ae;
        }
    }
}

// ===============================================================
// Sponge layer: velocity damp + isothermal relaxation toward HSE.
// Active for r > r_sp. Cosine ramp from 0 at r_sp to kappa_max at r_tp.
// ===============================================================
__global__
void k_wb2d_sponge(double* rho, double* mr, double* mt, double* rhoE,
                   const double* rho0, const double* P0,
                   const double* r_center,
                   double r_sp, double r_tp, double kappa_max, double dt_s,
                   double gam_m1_inv, int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    double r = r_center[i];
    if (r <= r_sp) return;
    double xi = fmin((r - r_sp) / fmax(r_tp - r_sp, 1e-30), 1.0);
    double kappa = kappa_max * 0.5 * (1.0 - cos(M_PI * xi));
    double alpha = dt_s * kappa;
    double damp = 1.0 / (1.0 + alpha);
    int k = fas_idx(i, j, nt, ng);
    double mr_old = mr[k], mt_old = mt[k];
    mr[k] *= damp;
    mt[k] *= damp;
    double rho_c = fmax(rho[k], 1e-20);
    double KE_old = 0.5 * (mr_old*mr_old + mt_old*mt_old) / rho_c;
    double e_int = rhoE[k] - KE_old;
    double e_int_ref = P0[flat] * gam_m1_inv;
    double e_int_new = (e_int + alpha * e_int_ref) * damp;
    double KE_new = 0.5 * (mr[k]*mr[k] + mt[k]*mt[k]) / rho_c;
    rhoE[k] = e_int_new + KE_new;
}

// Conservative central v_r damping (same as FAS k_fas_central_damp).
__global__
void k_wb2d_central_damp(double* mr, double* rhoE,
                         const double* rho, const double* r_center,
                         double r_damp, double alpha,
                         int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat / nt, j = flat % nt;
    double r = r_center[i];
    if (r >= r_damp) return;
    double f = exp(-alpha * (1.0 - r / r_damp));
    int k = fas_idx(i, j, nt, ng);
    double rho_c = fmax(rho[k], 1e-20);
    double vr_old = mr[k] / rho_c;
    double vr_new = vr_old * f;
    double dKE = 0.5 * rho_c * (vr_old*vr_old - vr_new*vr_new);
    mr[k] = rho_c * vr_new;
    rhoE[k] -= dKE;
}
