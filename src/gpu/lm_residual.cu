// LowMach residual: ghost cells, gravity, upwind flux, floor, sponge, CFL

#include "lowmach_solver.h"
#include "lm_common.cuh"
#include <algorithm>

// ========================= Ghost cells ============================
// Reflecting at r=0 and θ poles, zero-gradient at outer r.

__global__ void k_lm_ghost_r_in(double* rho, double* mr, double* mt, double* rhoE,
                                 int nr, int nt, int ng) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y + 1;
    if (j >= nt || g > ng) return;
    int kg = d_idx(-g,j,nt,ng), kp = d_idx(g-1,j,nt,ng);
    rho[kg]=rho[kp]; mr[kg]=-mr[kp]; mt[kg]=mt[kp]; rhoE[kg]=rhoE[kp];
}

__global__ void k_lm_ghost_r_out(double* rho, double* mr, double* mt, double* rhoE,
                                  int nr, int nt, int ng) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y;
    if (j >= nt || g >= ng) return;
    int kg = d_idx(nr+g,j,nt,ng), kp = d_idx(nr-1,j,nt,ng);
    rho[kg]=rho[kp]; mr[kg]=mr[kp]; mt[kg]=mt[kp]; rhoE[kg]=rhoE[kp];
}

__global__ void k_lm_ghost_t_n(double* rho, double* mr, double* mt, double* rhoE,
                                int nr, int nt, int ng) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y + 1;
    if (ii >= nr+2*ng || g > ng) return;
    int i = ii - ng;
    int kg = d_idx(i,-g,nt,ng), kp = d_idx(i,g-1,nt,ng);
    rho[kg]=rho[kp]; mr[kg]=mr[kp]; mt[kg]=-mt[kp]; rhoE[kg]=rhoE[kp];
}

__global__ void k_lm_ghost_t_s(double* rho, double* mr, double* mt, double* rhoE,
                                int nr, int nt, int ng) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y;
    if (ii >= nr+2*ng || g >= ng) return;
    int i = ii - ng;
    int kg = d_idx(i,nt+g,nt,ng), kp = d_idx(i,nt-1-g,nt,ng);
    rho[kg]=rho[kp]; mr[kg]=mr[kp]; mt[kg]=-mt[kp]; rhoE[kg]=rhoE[kp];
}

__global__ void k_lm_pole_lock(double* mt, int nr, int nt, int ng) {
    int i = blockIdx.x * blockDim.x + threadIdx.x - ng;
    if (i < -ng || i >= nr + ng) return;
    mt[d_idx(i, 0, nt, ng)] = 0.0;
    mt[d_idx(i, nt - 1, nt, ng)] = 0.0;
}

void LowMachSolver::launch_ghost() {
    int B=256;
    { dim3 g((nt+B-1)/B, ng);       k_lm_ghost_r_in<<<g,B>>>(d_rho,d_mr,d_mtheta,d_rhoE,nr,nt,ng); }
    { dim3 g((nt+B-1)/B, ng);       k_lm_ghost_r_out<<<g,B>>>(d_rho,d_mr,d_mtheta,d_rhoE,nr,nt,ng); }
    { dim3 g((nr+2*ng+B-1)/B, ng);  k_lm_ghost_t_n<<<g,B>>>(d_rho,d_mr,d_mtheta,d_rhoE,nr,nt,ng); }
    { dim3 g((nr+2*ng+B-1)/B, ng);  k_lm_ghost_t_s<<<g,B>>>(d_rho,d_mr,d_mtheta,d_rhoE,nr,nt,ng); }
    int ntot = nr + 2*ng;
    k_lm_pole_lock<<<(ntot+B-1)/B,B>>>(d_mtheta, nr, nt, ng);
}

// ========================= Upwind advection =======================
// Donor-cell (1st-order upwind) flux: F(u, qL, qR) = u>0 ? u*qL : u*qR
// This replaces HLLC. No Riemann solver, no sound speed.
//
// Spatial residual R(U):
//   R_ρ   = -∇·(ρv)
//   R_mr  = -∇·(ρvr·v) - ∂P/∂r + ρgr + geometric
//   R_mt  = -∇·(ρvθ·v) - (1/r)∂P/∂θ + ρgθ + geometric
//   R_rhoE = -∇·(ρe·v) - P∇·v
//
// Gravity: full Φ (not perturbation).
// Pressure gradient: central difference (same stencil as Poisson).

// Well-balanced residual kernel.
//
// The momentum equation uses reference-state subtraction:
//   force = -(∇P - ∇P₀) - (ρ-ρ₀)∇Φ₀ - ρ∇Φ' + geom_source(P') + geom_source(P₀)
// where P₀, ρ₀, Φ₀ satisfy discrete HSE: -∇P₀ + geom(P₀) = ρ₀∇Φ₀ exactly.
// So the residual at HSE (ρ=ρ₀, P=P₀, Φ=Φ₀, v=0) is exactly zero.

// Helper: central-difference gradient with weighted averaging
__device__ double d_grad_r(const double* q, int i, int j, int nt,
                           const double* rc) {
    if (i <= 0 || i >= 1) {} // always use interior stencil from caller
    double gl = (q[i*nt+j] - q[(i-1)*nt+j]) / (rc[i] - rc[i-1]);
    double gr = (q[(i+1)*nt+j] - q[i*nt+j]) / (rc[i+1] - rc[i]);
    double dl = rc[i] - rc[i-1], dh = rc[i+1] - rc[i];
    return (dh*gl + dl*gr) / (dl+dh);
}

__global__
void k_lm_residual(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* vol, const double* ar, const double* at,
    const double* r_center, const double* r_face,
    const double* theta_face, const double* dr, const double* dtheta,
    const double* gr, const double* gr0,
    const double* P0, const double* rho0,
    const double* grad_r_wm, const double* grad_r_wp,
    const double* grad_t_wm, const double* grad_t_wp,
    double* res,
    int nr, int nt, int ng, double gamma, double atm_thresh) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    int n = nr*nt;

    // i==0 handled by k_lm_residual_origin
    if (i == 0) return;

    if (rho0[flat] < atm_thresh) {
        res[flat] = 0.0; res[n+flat] = 0.0;
        res[2*n+flat] = 0.0; res[3*n+flat] = 0.0;
        return;
    }

    int k = d_idx(i,j,nt,ng);
    double invV = 1.0 / vol[flat];

    double rho_c = fmax(rho[k], 1e-20);
    double vr_c = mr[k] / rho_c;
    double vt_c = mt[k] / rho_c;
    double e_c = rhoE[k] / rho_c;
    double P_c = fmax((gamma - 1.0) * rho_c * e_c, 1e-30);
    double r = r_center[i];

    // ===== Upwind advection (donor-cell) =====
    auto upwind_r = [&](int il, int ir, int jj, int c) -> double {
        int kl = d_idx(il,jj,nt,ng), kr_k = d_idx(ir,jj,nt,ng);
        double rl = fmax(rho[kl],1e-20), rr = fmax(rho[kr_k],1e-20);
        double vf = 0.5*(mr[kl]/rl + mr[kr_k]/rr);
        double ql, qr;
        if      (c==0) { ql=rho[kl]; qr=rho[kr_k]; }
        else if (c==1) { ql=mr[kl];  qr=mr[kr_k]; }
        else if (c==2) { ql=mt[kl];  qr=mt[kr_k]; }
        else           { ql=rhoE[kl]; qr=rhoE[kr_k]; }
        return vf * (vf >= 0.0 ? ql : qr);
    };
    auto upwind_t = [&](int ii, int jl, int jr, int c) -> double {
        int kl = d_idx(ii,jl,nt,ng), kr_k = d_idx(ii,jr,nt,ng);
        double rl = fmax(rho[kl],1e-20), rr = fmax(rho[kr_k],1e-20);
        double vf = 0.5*(mt[kl]/rl + mt[kr_k]/rr);
        double ql, qr;
        if      (c==0) { ql=rho[kl]; qr=rho[kr_k]; }
        else if (c==1) { ql=mr[kl];  qr=mr[kr_k]; }
        else if (c==2) { ql=mt[kl];  qr=mt[kr_k]; }
        else           { ql=rhoE[kl]; qr=rhoE[kr_k]; }
        return vf * (vf >= 0.0 ? ql : qr);
    };

    double Ar_hi=ar[(i+1)*nt+j], Ar_lo=ar[i*nt+j];
    double At_hi=at[i*(nt+1)+j+1], At_lo=at[i*(nt+1)+j];

    double div[4];
    for (int c = 0; c < 4; ++c) {
        double fr_hi = upwind_r(i,i+1,j,c), fr_lo = upwind_r(i-1,i,j,c);
        double ft_hi = upwind_t(i,j,j+1,c), ft_lo = upwind_t(i,j-1,j,c);
        div[c] = -invV*(Ar_hi*fr_hi - Ar_lo*fr_lo + At_hi*ft_hi - At_lo*ft_lo);
    }

    // ===== Well-balanced pressure + gravity (reference-state subtraction) =====
    // 1D radial gravity: g(r) from cumulative mass integral (no Poisson, no GMG noise).
    // Reference state satisfies: -∇P₀ + ρ₀·g₀(r) + geom(P₀) = 0 exactly.
    // Perturbation form:
    //   force_r = -(∇P - ∇P₀) + (ρ - ρ₀)·g₀(r) + ρ·(g(r) - g₀(r))
    //   force_θ = -(1/r)(∂P'/∂θ)  (no θ gravity component — 1D gravity is radial)

    // ===== Branchless pressure gradient using precomputed stencil weights =====
    // Clamped indices: at boundaries weight=0, so the clamped read is harmless.
    double Pp_c = P_c - P0[flat];

    int im = flat - nt * (int)(i > 0);
    int ip = flat + nt * (int)(i < nr-1);
    double Pp_m = fmax((gamma-1.0)*rhoE[d_idx(max(i-1,0),j,nt,ng)],1e-30) - P0[im];
    double Pp_p = fmax((gamma-1.0)*rhoE[d_idx(min(i+1,nr-1),j,nt,ng)],1e-30) - P0[ip];
    double dPp_dr = grad_r_wm[flat]*(Pp_m - Pp_c) + grad_r_wp[flat]*(Pp_p - Pp_c);

    int jm = flat - (int)(j > 0);
    int jp = flat + (int)(j < nt-1);
    double Pp_jm = fmax((gamma-1.0)*rhoE[d_idx(i,max(j-1,0),nt,ng)],1e-30) - P0[jm];
    double Pp_jp = fmax((gamma-1.0)*rhoE[d_idx(i,min(j+1,nt-1),nt,ng)],1e-30) - P0[jp];
    double dPp_dt_r = grad_t_wm[flat]*(Pp_jm - Pp_c) + grad_t_wp[flat]*(Pp_jp - Pp_c);

    double rhop_c = rho_c - rho0[flat];

    // 1D gravity: g_r(i) is radial only, same for all θ cells at this radius
    double g0_r = gr0[i];
    double gp_r = gr[i] - g0_r;

    // ===== Geometric source =====
    double inv_r = 1.0 / r;
    double S_mr = rho_c * vt_c * vt_c * inv_r;   // Eq. (5.1)
    double S_mt = -rho_c * vr_c * vt_c * inv_r;  // Eq. (5.2)

    // Eq. (10.5): well-balanced radial force — split as −∇P' + ρ'·g₀ + ρ·g'
    // so that the HSE component −∇P₀ + ρ₀·g₀ vanishes analytically.
    double force_r = -dPp_dr + rhop_c*g0_r + rho_c*gp_r;
    // Theta: only pressure (no gravity component)
    double force_t = -dPp_dt_r;

    // Gravity work on energy: ρ·v·g  (v=0 at HSE → zero, well-balanced)
    double S_E = rho_c * vr_c * (g0_r + gp_r);

    // ===== P∇·v (compression work in internal energy equation) =====
    auto vr_face_f = [&](int il, int ir, int jj) -> double {
        double rl = fmax(rho[d_idx(il,jj,nt,ng)],1e-20);
        double rr = fmax(rho[d_idx(ir,jj,nt,ng)],1e-20);
        return 0.5*(mr[d_idx(il,jj,nt,ng)]/rl + mr[d_idx(ir,jj,nt,ng)]/rr);
    };
    auto vt_face_f = [&](int ii, int jl, int jr) -> double {
        double rl = fmax(rho[d_idx(ii,jl,nt,ng)],1e-20);
        double rr = fmax(rho[d_idx(ii,jr,nt,ng)],1e-20);
        return 0.5*(mt[d_idx(ii,jl,nt,ng)]/rl + mt[d_idx(ii,jr,nt,ng)]/rr);
    };
    double div_v = invV*(
        Ar_hi*vr_face_f(i,i+1,j) - Ar_lo*vr_face_f(i-1,i,j) +
        At_hi*vt_face_f(i,j,j+1) - At_lo*vt_face_f(i,j-1,j));

    // ===== Assemble =====
    res[flat]       = div[0];
    res[n + flat]   = div[1] + force_r + S_mr;
    res[2*n + flat] = div[2] + force_t + S_mt;
    res[3*n + flat] = div[3] - P_c * div_v + S_E;
}

// ========================= Origin kernel (i=0, divergence theorem) ========
// Pie-slice cell at r=0: inner face has zero area.
// Volume-averaged <1/r> = 1.5/r_face[1].

__global__
void k_lm_residual_origin(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* vol, const double* ar, const double* at,
    const double* r_center, const double* r_face,
    const double* theta_face, const double* dr, const double* dtheta,
    const double* gr, const double* gr0,
    const double* P0, const double* rho0,
    double* res,
    int nr, int nt, int ng, double gamma, double atm_thresh) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nt) return;
    int flat = j;
    int n = nr*nt;

    if (rho0[flat] < atm_thresh) {
        res[flat] = 0.0; res[n+flat] = 0.0;
        res[2*n+flat] = 0.0; res[3*n+flat] = 0.0;
        return;
    }

    int k = d_idx(0, j, nt, ng);
    double invV = 1.0 / vol[flat];

    double rho_c = fmax(rho[k], 1e-20);
    double vr_c = mr[k] / rho_c;
    double vt_c = mt[k] / rho_c;
    double P_c = fmax((gamma - 1.0) * rhoE[k], 1e-30);

    double Ar_hi = ar[1*nt + j];
    double At_hi = at[0*(nt+1) + j+1], At_lo = at[0*(nt+1) + j];

    auto upwind_r_outer = [&](int jj, int c) -> double {
        int kl = d_idx(0,jj,nt,ng), kr_k = d_idx(1,jj,nt,ng);
        double rl = fmax(rho[kl],1e-20), rr = fmax(rho[kr_k],1e-20);
        double vf = 0.5*(mr[kl]/rl + mr[kr_k]/rr);
        double ql, qr;
        if      (c==0) { ql=rho[kl]; qr=rho[kr_k]; }
        else if (c==1) { ql=mr[kl];  qr=mr[kr_k]; }
        else if (c==2) { ql=mt[kl];  qr=mt[kr_k]; }
        else           { ql=rhoE[kl]; qr=rhoE[kr_k]; }
        return vf * (vf >= 0.0 ? ql : qr);
    };
    auto upwind_t = [&](int jl, int jr, int c) -> double {
        int kl = d_idx(0,jl,nt,ng), kr_k = d_idx(0,jr,nt,ng);
        double rl = fmax(rho[kl],1e-20), rr = fmax(rho[kr_k],1e-20);
        double vf = 0.5*(mt[kl]/rl + mt[kr_k]/rr);
        double ql, qr;
        if      (c==0) { ql=rho[kl]; qr=rho[kr_k]; }
        else if (c==1) { ql=mr[kl];  qr=mr[kr_k]; }
        else if (c==2) { ql=mt[kl];  qr=mt[kr_k]; }
        else           { ql=rhoE[kl]; qr=rhoE[kr_k]; }
        return vf * (vf >= 0.0 ? ql : qr);
    };

    double div[4];
    for (int c = 0; c < 4; ++c) {
        double fr_hi = upwind_r_outer(j, c);
        double ft_hi = upwind_t(j, j+1, c), ft_lo = upwind_t(j-1, j, c);
        div[c] = -invV*(Ar_hi*fr_hi + At_hi*ft_hi - At_lo*ft_lo);
    }

    double Pp_c = P_c - P0[flat];
    double rhop_c = rho_c - rho0[flat];
    double g0_r = gr0[0];

    double dh = r_center[1] - r_center[0];
    double Pp_p = fmax((gamma-1.0)*rhoE[d_idx(1,j,nt,ng)],1e-30) - P0[1*nt+j];
    double dPp_dr = (Pp_p - Pp_c) / dh;

    double r = r_center[0];
    double dPp_dt_r = 0;
    if (j > 0 && j < nt-1 && r > 1e-14) {
        double tc_m=0.5*(theta_face[j-1]+theta_face[j]);
        double tc_c=0.5*(theta_face[j]+theta_face[j+1]);
        double tc_p=0.5*(theta_face[j+1]+theta_face[j+2]);
        double dl=tc_c-tc_m, dhh=tc_p-tc_c;
        double Pp_m = fmax((gamma-1.0)*rhoE[d_idx(0,j-1,nt,ng)],1e-30) - P0[j-1];
        double Pp_pp = fmax((gamma-1.0)*rhoE[d_idx(0,j+1,nt,ng)],1e-30) - P0[j+1];
        dPp_dt_r = ((dhh*(Pp_c-Pp_m)/dl + dl*(Pp_pp-Pp_c)/dhh))/(r*(dl+dhh));
    }

    double gp_r = gr[0] - g0_r;

    double r_out = r_face[1];
    double inv_r_avg = (r_out > 1e-30) ? 1.5 / r_out : 0.0;
    double S_mr = rho_c * vt_c * vt_c * inv_r_avg;
    double S_mt = -rho_c * vr_c * vt_c * inv_r_avg;

    double force_r = -dPp_dr + rhop_c*g0_r + rho_c*gp_r;
    double force_t = -dPp_dt_r;

    double S_E = rho_c * vr_c * (g0_r + gp_r);

    double rl = fmax(rho[d_idx(0,j,nt,ng)],1e-20);
    double rr = fmax(rho[d_idx(1,j,nt,ng)],1e-20);
    double vr_outer = 0.5*(mr[d_idx(0,j,nt,ng)]/rl + mr[d_idx(1,j,nt,ng)]/rr);
    auto vt_face_f = [&](int jl, int jr) -> double {
        double rll = fmax(rho[d_idx(0,jl,nt,ng)],1e-20);
        double rrr = fmax(rho[d_idx(0,jr,nt,ng)],1e-20);
        return 0.5*(mt[d_idx(0,jl,nt,ng)]/rll + mt[d_idx(0,jr,nt,ng)]/rrr);
    };
    double div_v = invV*(Ar_hi*vr_outer
        + At_hi*vt_face_f(j,j+1) - At_lo*vt_face_f(j-1,j));

    res[flat]       = div[0];
    res[n + flat]   = div[1] + force_r + S_mr;
    res[2*n + flat] = div[2] + force_t + S_mt;
    res[3*n + flat] = div[3] - P_c * div_v + S_E;
}

// ========================= Pack / unpack (5-DOF: ρ, mr, mt, ρe, Φ) =

__global__
void k_lm_pack(const double* rho, const double* mr, const double* mt,
               const double* rhoE, const double* phi, double* packed,
               int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt, n = nr*nt;
    int k = d_idx(i,j,nt,ng);
    packed[flat] = rho[k]; packed[n+flat] = mr[k];
    packed[2*n+flat] = mt[k]; packed[3*n+flat] = rhoE[k];
    packed[4*n+flat] = phi[flat];
}

__global__
void k_lm_unpack_set(double* rho, double* mr, double* mt, double* rhoE,
                     double* phi, const double* packed,
                     int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt, n = nr*nt;
    int k = d_idx(i,j,nt,ng);
    rho[k] = packed[flat]; mr[k] = packed[n+flat];
    mt[k] = packed[2*n+flat]; rhoE[k] = packed[3*n+flat];
    phi[flat] = packed[4*n+flat];
}

__global__
void k_lm_unpack_add(double* rho, double* mr, double* mt, double* rhoE,
                     double* phi, const double* delta, double alpha,
                     int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt, n = nr*nt;
    int k = d_idx(i,j,nt,ng);
    rho[k] += alpha * delta[flat]; mr[k] += alpha * delta[n+flat];
    mt[k] += alpha * delta[2*n+flat]; rhoE[k] += alpha * delta[3*n+flat];
    phi[flat] += alpha * delta[4*n+flat];
}

// ========================= BLAS-like ops ==========================

__global__ void k_lm_axpy(double* y, double a, const double* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

__global__ void k_lm_scale(double* x, double a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= a;
}

__global__ void k_lm_copy(double* dst, const double* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

__global__ void k_lm_zero(double* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.0;
}

// ========================= Dot product ============================

__global__ void k_lm_reduce_sum(const double* in, double* out, int n);

__global__ void k_lm_dot(const double* a, const double* b, double* out, int n) {
    extern __shared__ double s[];
    int tid = threadIdx.x, idx = blockIdx.x*blockDim.x+tid;
    s[tid] = (idx < n) ? a[idx]*b[idx] : 0.0;
    __syncthreads();
    for (int st = blockDim.x/2; st > 0; st >>= 1) {
        if (tid < st) s[tid] += s[tid+st];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

// ========================= Floor ==================================

__global__
void k_lm_floor(double* rho, double* mr, double* mt, double* rhoE,
                int nr, int nt, int ng, double gamma,
                double rho_fl, double P_fl) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int k = d_idx(flat/nt, flat%nt, nt, ng);
    double r = rho[k];
    r = 0.5 * (r + sqrt(r*r + 4.0*rho_fl*rho_fl));
    rho[k] = fmax(r, rho_fl);
    double P = (gamma - 1.0) * rhoE[k];
    double P_s = 0.5 * (P + sqrt(P*P + 4.0*P_fl*P_fl));
    rhoE[k] = fmax(P_s, P_fl) / (gamma - 1.0);
}

// Freeze atmosphere cells to HSE reference state.
// Cells where rho0 < atm_thresh are vacuum — Newton corrections
// create spurious momentum (vr = mr/rho_floor → huge), crashing CFL.
__global__
void k_lm_freeze_atm(double* rho, double* mr, double* mt, double* rhoE,
                      const double* rho0, const double* P0,
                      int nr, int nt, int ng, double gamma, double atm_thresh) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    if (rho0[flat] >= atm_thresh) return;
    int k = d_idx(flat/nt, flat%nt, nt, ng);
    rho[k] = rho0[flat];
    mr[k] = 0.0;
    mt[k] = 0.0;
    rhoE[k] = P0[flat] / (gamma - 1.0);
}

// ========================= HSE snapshot ===========================

__global__
void k_lm_snapshot_hse(const double* rho, const double* rhoE, const double* phi,
                       double* rho0, double* P0, double* phi0,
                       int nr, int nt, int ng, double gamma) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int k = d_idx(flat/nt, flat%nt, nt, ng);
    double r = fmax(rho[k], 1e-20);
    rho0[flat] = r;
    P0[flat] = fmax((gamma - 1.0) * rhoE[k], 1e-30);
    phi0[flat] = phi[flat];
}

// ========================= Gravity ================================

__global__
void k_lm_rhovol(const double* rho, const double* vol, double* out,
                 int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    out[flat] = rho[d_idx(flat/nt,flat%nt,nt,ng)] * vol[flat];
}

__global__
void k_lm_reduce_sum(const double* in, double* out, int n) {
    extern __shared__ double s[];
    int tid = threadIdx.x, idx = blockIdx.x*blockDim.x+tid;
    s[tid] = (idx < n) ? in[idx] : 0.0;
    __syncthreads();
    for (int st = blockDim.x/2; st > 0; st >>= 1) {
        if (tid < st) s[tid] += s[tid+st];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

static double gpu_reduce_sum(double* src, double* dst, int n) {
    int B = 256;
    while (n > 1) {
        int nb = (n+B-1)/B;
        k_lm_reduce_sum<<<nb,B,B*sizeof(double)>>>(src, dst, n);
        n = nb; double* t = src; src = dst; dst = t;
    }
    double val;
    CUDA_CHECK(cudaMemcpy(&val, src, sizeof(double), cudaMemcpyDeviceToHost));
    return val;
}

__global__
void k_lm_grav_rhs(const double* rho, double* rhs, const double* rc,
                    int nr, int nt, int ng, double G, double M_total) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt;
    if (i == nr-1)
        rhs[flat] = -G * M_total / rc[nr-1];
    else
        rhs[flat] = 4.0 * M_PI * G * rho[d_idx(flat/nt,flat%nt,nt,ng)];
}

void LowMachSolver::solve_gravity() {
    int n = nr*nt, B = 256;
    k_lm_rhovol<<<(n+B-1)/B,B>>>(d_rho, d_cell_volume, d_work_a, nr, nt, ng);
    double M = gpu_reduce_sum(d_work_a, d_work_b, n) * 2.0 * M_PI;
    k_lm_grav_rhs<<<(n+B-1)/B,B>>>(d_rho, d_rhs_poisson, d_r_center, nr,nt,ng, G_const, M);
    gmg.solve(d_rhs_poisson, d_phi);
}

// ========================= 1D radial gravity ========================
// g_r(i) = -G·M(<r_face[i+1]) / r_center[i]²
// M(<r) = 2π ∫ Σ_j [ρ(i,j) · volume(i,j)] over shells.
// Computed via prefix sum over radial shells.

// Step 1: compute shell mass = 2π · Σ_j ρ(i,j)·vol(i,j) for each i
__global__
void k_lm_shell_mass(const double* rho, const double* vol,
                     double* shell, int nr, int nt, int ng) {
    int i = blockIdx.x;
    if (i >= nr) return;
    extern __shared__ double smem[];
    int tid = threadIdx.x;
    double s = 0.0;
    for (int j = tid; j < nt; j += blockDim.x)
        s += rho[d_idx(i,j,nt,ng)] * vol[i*nt+j];
    smem[tid] = s;
    __syncthreads();
    for (int st = blockDim.x/2; st > 0; st >>= 1) {
        if (tid < st) smem[tid] += smem[tid+st];
        __syncthreads();
    }
    if (tid == 0) shell[i] = smem[0] * 2.0 * M_PI;
}

// Step 2: prefix sum + g_r computation (single thread — nr is small)
__global__
void k_lm_gravity_from_shells(const double* shell_mass, const double* rc,
                               double* gr, int nr, double G) {
    if (threadIdx.x != 0) return;
    double M_enc = 0.0;
    for (int i = 0; i < nr; ++i) {
        M_enc += shell_mass[i];
        double r = rc[i];
        gr[i] = -G * M_enc / (r * r);
    }
}

void LowMachSolver::compute_gravity_1d() {
    int B = std::min(nt, 256);
    k_lm_shell_mass<<<nr, B, B*sizeof(double)>>>(
        d_rho, d_cell_volume, d_shell_mass, nr, nt, ng);
    k_lm_gravity_from_shells<<<1, 1>>>(
        d_shell_mass, d_r_center, d_gr, nr, G_const);
}

// ========================= Poisson residual (5th equation) ========
// F₅ = ∇²Φ - 4πGρ = 0
// Uses EXACTLY the same spherical Laplacian stencil as GMG to ensure
// that solve_gravity's output satisfies this equation to GMG tolerance.

__global__
void k_lm_poisson_residual(
    const double* phi, const double* rho,
    const double* r_face, const double* r_center, const double* dr,
    const double* dtheta,
    const double* sin_tf, const double* sin_tc,
    double* res5,
    int nr, int nt, int ng, double G, double M_total) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;

    // Dirichlet BC at outer boundary: Φ = -GM/R
    if (i == nr-1) {
        double phi_bc = -G * M_total / r_center[nr-1];
        res5[flat] = phi_bc - phi[flat];
        return;
    }

    double ri = r_center[i], ri2 = ri*ri, dri = dr[i];

    double Lphi = 0.0;
    double cC = 0.0;

    if (i > 0) {
        double rl = r_face[i];
        double cW = rl*rl / (ri2 * dri * (r_center[i] - r_center[i-1]));
        Lphi += cW * phi[(i-1)*nt+j];
        cC -= cW;
    }
    if (i < nr-1) {
        double rh = r_face[i+1];
        double cE = rh*rh / (ri2 * dri * (r_center[i+1] - r_center[i]));
        Lphi += cE * phi[(i+1)*nt+j];
        cC -= cE;
    }

    double sj = sin_tc[j], dtj = dtheta[j];
    if (j > 0) {
        double cS = sin_tf[j] / (ri2 * sj * dtj * 0.5 * (dtheta[j] + dtheta[j-1]));
        Lphi += cS * phi[i*nt+j-1];
        cC -= cS;
    }
    if (j < nt-1) {
        double cN = sin_tf[j+1] / (ri2 * sj * dtj * 0.5 * (dtheta[j] + dtheta[j+1]));
        Lphi += cN * phi[i*nt+j+1];
        cC -= cN;
    }
    Lphi += cC * phi[flat];

    double rho_c = fmax(rho[d_idx(i,j,nt,ng)], 1e-20);
    double raw = Lphi - 4.0 * M_PI * G * rho_c;
    double scale = fmax(fabs(cC), 1.0);
    res5[flat] = raw / scale;
}

// ========================= Compute F (Newton residual) ============

// F₁₋₄ = R(U,Φ) - (U-Uⁿ)/dt  (fluid, 4 equations)
// F₅ = ∇²Φ - 4πGρ             (Poisson, algebraic constraint, no time derivative)
__global__
void k_lm_compute_F(double* F, const double* R,
                    const double* rho, const double* mr,
                    const double* mt, const double* rhoE,
                    const double* Un, double inv_dt,
                    int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int n = nr*nt;
    int k = d_idx(flat/nt, flat%nt, nt, ng);
    F[flat]       = R[flat]       - inv_dt * (rho[k]  - Un[flat]);
    F[n + flat]   = R[n + flat]   - inv_dt * (mr[k]   - Un[n + flat]);
    F[2*n + flat] = R[2*n + flat] - inv_dt * (mt[k]   - Un[2*n + flat]);
    F[3*n + flat] = R[3*n + flat] - inv_dt * (rhoE[k] - Un[3*n + flat]);
    // F₅ is written separately by k_lm_poisson_residual
}

void LowMachSolver::compute_residual(double* d_res) {
    int n = nr*nt, B = 256;
    launch_ghost();
    compute_gravity_1d();
    k_lm_residual<<<(n+B-1)/B,B>>>(
        d_rho,d_mr,d_mtheta,d_rhoE,
        d_cell_volume,d_area_r,d_area_theta,
        d_r_center,d_r_face,d_theta_face,d_dr,d_dtheta,
        d_gr, d_gr0, d_P0, d_rho0,
        d_grad_r_wm, d_grad_r_wp, d_grad_t_wm, d_grad_t_wp,
        d_residual,
        nr,nt,ng,gamma, atm_rho_thresh);
    k_lm_residual_origin<<<(nt+B-1)/B,B>>>(
        d_rho,d_mr,d_mtheta,d_rhoE,
        d_cell_volume,d_area_r,d_area_theta,
        d_r_center,d_r_face,d_theta_face,d_dr,d_dtheta,
        d_gr, d_gr0, d_P0, d_rho0,
        d_residual,
        nr,nt,ng,gamma, atm_rho_thresh);
}

void LowMachSolver::compute_F(double* d_F, double dt) {
    int n = nr*nt, B = 256;
    compute_residual(d_residual);
    k_lm_compute_F<<<(n+B-1)/B,B>>>(d_F, d_residual,
        d_rho, d_mr, d_mtheta, d_rhoE, d_Un, 1.0/dt, nr, nt, ng);
}

void LowMachSolver::pack_state(double* d_packed) {
    int B = 256;
    k_lm_pack<<<(nr*nt+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE,d_phi, d_packed, nr,nt,ng);
}

void LowMachSolver::unpack_set(const double* d_packed) {
    int B = 256;
    k_lm_unpack_set<<<(nr*nt+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE,d_phi, d_packed, nr,nt,ng);
}

void LowMachSolver::unpack_delta(const double* d_delta, double alpha) {
    int B = 256;
    k_lm_unpack_add<<<(nr*nt+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE,d_phi, d_delta, alpha, nr,nt,ng);
}

void LowMachSolver::apply_floor() {
    int B = 256;
    k_lm_floor<<<(nr*nt+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE, nr,nt,ng,gamma, 1e-15, 1e-15);
}

void LowMachSolver::apply_sponge(double dt) {
    if (sponge_r_start <= 0 || sponge_r_start >= sponge_r_top) return;
    int n = nr*nt, B = 256;
    extern __global__ void k_lm_sponge(double*, double*, const double*, const double*,
                                        double, double, double, double, int, int, int);
    k_lm_sponge<<<(n+B-1)/B,B>>>(
        d_mr, d_mtheta, d_r_center, d_rho,
        sponge_r_start, sponge_r_top, sponge_kappa, dt,
        nr, nt, ng);
}

__global__
void k_lm_clamp(double* delta, const double* scale, double max_rel, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double lim = max_rel * scale[i];
    delta[i] = fmax(-lim, fmin(delta[i], lim));
}

// ========================= Sponge layer ============================
// Smooth velocity damping near the outer boundary to prevent acoustic
// reflections from corrupting the interior during long-time simulations.
//
// Profile: κ(r) = κ_max · ½(1 - cos(π·(r-r_sp)/(r_tp-r_sp)))
//   where r_sp = sponge start, r_tp = sponge top (outer boundary)
//   κ = 0 inside r_sp, ramps to κ_max at r_tp
//
// Damping: v *= 1/(1 + dt·κ)   (implicit damping, unconditionally stable)
//
// Ref: MAESTROeX (Almgren+2020), Eq. 41

__global__
void k_lm_sponge(double* mr, double* mt,
                 const double* r_center, const double* rho,
                 double r_sp, double r_tp, double kappa_max, double dt_sponge,
                 int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    double r = r_center[i];

    if (r <= r_sp) return;

    double xi = fmin((r - r_sp) / fmax(r_tp - r_sp, 1e-30), 1.0);
    double kappa = kappa_max * 0.5 * (1.0 - cos(M_PI * xi));
    double damp = 1.0 / (1.0 + dt_sponge * kappa);

    int k = (i + ng) * (nt + 2*ng) + (j + ng);
    mr[k] *= damp;
    mt[k] *= damp;
}

void LowMachSolver::compute_scaling() {
    int N4 = 4*nr*nt, n = nr*nt, B = 256;
    k_lm_compute_scale<<<(N4+B-1)/B,B>>>(d_Un, d_scale, N4);
    k_lm_compute_music_scale<<<(n+B-1)/B,B>>>(
        d_rho, d_mr, d_mtheta, d_rhoE,
        d_scale_R, d_scale_L, nr, nt, ng, gamma);
}

void LowMachSolver::clamp_correction(double* d_delta, double max_rel_change) {
    // Clamp only fluid components (4N). Φ correction is handled by Poisson solve.
    int N = 4*nr*nt, B = 256;
    k_lm_clamp<<<(N+B-1)/B,B>>>(d_delta, d_scale, max_rel_change, N);
}

// ========================= CFL (|v| + cs signal speed) ============

__global__
void k_lm_cfl(const double* rho, const double* mr, const double* mt,
              const double* rhoE,
              const double* dr, const double* rc, const double* dtheta,
              const double* rho0, double* out,
              int nr, int nt, int ng, double gam, double atm_thresh) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    if (rho0[flat] < atm_thresh) { out[flat] = 1e30; return; }
    int i = flat/nt, j = flat%nt;
    int k = d_idx(i,j,nt,ng);
    double rho_c = fmax(rho[k], 1e-20);
    double vr = fabs(mr[k] / rho_c);
    double vt = fabs(mt[k] / rho_c);
    double P = fmax((gam-1.0)*rhoE[k], 1e-30);
    double cs = sqrt(gam * P / rho_c);
    double dt_r = dr[i] / (vr + cs);
    double dt_t = rc[i] * dtheta[j] / (vt + cs);
    out[flat] = fmin(dt_r, dt_t);
}

__global__
void k_lm_reduce_min(const double* in, double* out, int n) {
    extern __shared__ double s[];
    int tid = threadIdx.x, idx = blockIdx.x*blockDim.x+tid;
    s[tid] = (idx < n) ? in[idx] : 1e30;
    __syncthreads();
    for (int st = blockDim.x/2; st > 0; st >>= 1) {
        if (tid < st) s[tid] = fmin(s[tid], s[tid+st]);
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

double LowMachSolver::compute_cfl_dt() {
    int n = nr*nt, B = 256;
    k_lm_cfl<<<(n+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE,
                                d_dr,d_r_center,d_dtheta,
                                d_rho0, d_work_a, nr,nt,ng, gamma, atm_rho_thresh);
    int cur = n;
    double *src = d_work_a, *dst = d_work_b;
    while (cur > 1) {
        int nb = (cur+B-1)/B;
        k_lm_reduce_min<<<nb,B,B*sizeof(double)>>>(src, dst, cur);
        cur = nb; double *t = src; src = dst; dst = t;
    }
    double val;
    CUDA_CHECK(cudaMemcpy(&val, src, sizeof(double), cudaMemcpyDeviceToHost));
    return cfl_num * val;
}

