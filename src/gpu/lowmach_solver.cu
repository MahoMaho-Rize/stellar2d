// Low-Mach fully-implicit GPU solver.
//
// Phase 1: adiabatic Euler + self-gravity, no nuclear reactions.
// Spatial: 1st-order upwind advection + central pressure/gravity gradient.
// Temporal: Backward Euler + Newton-GMRES (JFNK).
// Preconditioner: block-diagonal scaling (phase 1), GMG Schur complement (future).
//
// State vector per cell: (ρ, ρvr, ρvθ, ρe)  where e = internal energy per mass.
// Pressure: P = (γ-1)ρe.  Sound speed not used for CFL — advection CFL only.

#include "lowmach_solver.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cfloat>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(err)); std::exit(1); \
    } \
} while(0)

// ========================= Device helpers =========================

__device__ __forceinline__
int d_idx(int i, int j, int nt, int ng) {
    return (i + ng) * (nt + 2 * ng) + (j + ng);
}

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

void LowMachSolver::launch_ghost() {
    int B=256;
    { dim3 g((nt+B-1)/B, ng);       k_lm_ghost_r_in<<<g,B>>>(d_rho,d_mr,d_mtheta,d_rhoE,nr,nt,ng); }
    { dim3 g((nt+B-1)/B, ng);       k_lm_ghost_r_out<<<g,B>>>(d_rho,d_mr,d_mtheta,d_rhoE,nr,nt,ng); }
    { dim3 g((nr+2*ng+B-1)/B, ng);  k_lm_ghost_t_n<<<g,B>>>(d_rho,d_mr,d_mtheta,d_rhoE,nr,nt,ng); }
    { dim3 g((nr+2*ng+B-1)/B, ng);  k_lm_ghost_t_s<<<g,B>>>(d_rho,d_mr,d_mtheta,d_rhoE,nr,nt,ng); }
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
    const double* phi,
    const double* P0, const double* rho0, const double* phi0,
    double* res,
    int nr, int nt, int ng, double gamma) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    int n = nr*nt;
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
    // Compute ∇P', ∇Φ' using same central-diff stencil as the reference state.
    // The reference state P₀,ρ₀,Φ₀ is built to satisfy the SAME discrete stencil,
    // so the difference exactly cancels at HSE to machine precision.

    double Pp_c = P_c - P0[flat];
    double rhop_c = rho_c - rho0[flat];
    double phip_c = phi[flat] - phi0[flat];

    // Radial perturbation gradients
    double dPp_dr=0, dphip_dr=0, dphi0_dr=0;
    if (i > 0 && i < nr-1) {
        double Pp_m = fmax((gamma-1.0)*rhoE[d_idx(i-1,j,nt,ng)],1e-30) - P0[(i-1)*nt+j];
        double Pp_p = fmax((gamma-1.0)*rhoE[d_idx(i+1,j,nt,ng)],1e-30) - P0[(i+1)*nt+j];
        double dl = r_center[i]-r_center[i-1], dh = r_center[i+1]-r_center[i];
        double gl, gr;
        gl = (Pp_c-Pp_m)/dl; gr = (Pp_p-Pp_c)/dh;
        dPp_dr = (dh*gl+dl*gr)/(dl+dh);

        double ph_m = phi[(i-1)*nt+j]-phi0[(i-1)*nt+j];
        double ph_p = phi[(i+1)*nt+j]-phi0[(i+1)*nt+j];
        gl = (phip_c-ph_m)/dl; gr = (ph_p-phip_c)/dh;
        dphip_dr = (dh*gl+dl*gr)/(dl+dh);

        gl = (phi0[flat]-phi0[(i-1)*nt+j])/dl;
        gr = (phi0[(i+1)*nt+j]-phi0[flat])/dh;
        dphi0_dr = (dh*gl+dl*gr)/(dl+dh);
    } else if (i == 0 && nr > 1) {
        double dh = r_center[1]-r_center[0];
        dPp_dr = (fmax((gamma-1.0)*rhoE[d_idx(1,j,nt,ng)],1e-30)-P0[1*nt+j] - Pp_c)/dh;
        dphip_dr = (phi[1*nt+j]-phi0[1*nt+j] - phip_c)/dh;
        dphi0_dr = (phi0[1*nt+j]-phi0[0*nt+j])/dh;
    } else if (i == nr-1 && nr >= 2) {
        double dl = r_center[nr-1]-r_center[nr-2];
        dPp_dr = (Pp_c - (fmax((gamma-1.0)*rhoE[d_idx(nr-2,j,nt,ng)],1e-30)-P0[(nr-2)*nt+j]))/dl;
        dphip_dr = (phip_c - (phi[(nr-2)*nt+j]-phi0[(nr-2)*nt+j]))/dl;
        dphi0_dr = (phi0[(nr-1)*nt+j]-phi0[(nr-2)*nt+j])/dl;
    }

    // Theta perturbation gradients (divided by r)
    double dPp_dt_r=0, dphip_dt_r=0, dphi0_dt_r=0;
    if (j > 0 && j < nt-1) {
        double tc_m=0.5*(theta_face[j-1]+theta_face[j]);
        double tc_c=0.5*(theta_face[j]+theta_face[j+1]);
        double tc_p=0.5*(theta_face[j+1]+theta_face[j+2]);
        double dl=tc_c-tc_m, dh=tc_p-tc_c;

        double Pp_m = fmax((gamma-1.0)*rhoE[d_idx(i,j-1,nt,ng)],1e-30) - P0[i*nt+j-1];
        double Pp_p = fmax((gamma-1.0)*rhoE[d_idx(i,j+1,nt,ng)],1e-30) - P0[i*nt+j+1];
        dPp_dt_r = ((dh*(Pp_c-Pp_m)/dl + dl*(Pp_p-Pp_c)/dh))/(r*(dl+dh));

        double ph_m = phi[i*nt+j-1]-phi0[i*nt+j-1];
        double ph_p = phi[i*nt+j+1]-phi0[i*nt+j+1];
        dphip_dt_r = ((dh*(phip_c-ph_m)/dl + dl*(ph_p-phip_c)/dh))/(r*(dl+dh));

        dphi0_dt_r = ((dh*(phi0[flat]-phi0[i*nt+j-1])/dl +
                        dl*(phi0[i*nt+j+1]-phi0[flat])/dh))/(r*(dl+dh));
    }

    // ===== Geometric source (Eq. 5.1-5.4) =====
    double inv_r = 1.0 / r;
    double r2h = r_face[i+1]*r_face[i+1], r2l = r_face[i]*r_face[i];
    double dcos = cos(theta_face[j]) - cos(theta_face[j+1]);
    double dsin = sin(theta_face[j]) - sin(theta_face[j+1]);

    // Velocity-dependent geometric terms
    double S_mr = rho_c * vt_c * vt_c * inv_r;
    double S_mt = -rho_c * vr_c * vt_c * inv_r;

    // Well-balanced momentum force (perturbation only):
    //   -∇P' - ρ'∇Φ₀ - ρ∇Φ'
    // NO FV geometric source on P' — the central-diff ∇P' is the complete
    // spherical gradient. FV geom source on P₀ cancels exactly with
    // ∇P₀ + ρ₀∇Φ₀ by construction of the reference state.
    double force_r = -dPp_dr - rhop_c*dphi0_dr - rho_c*dphip_dr;
    double force_t = -dPp_dt_r - rhop_c*dphi0_dt_r - rho_c*dphip_dt_r;

    // Gravity work on energy (full, not perturbation — v=0 at HSE so this is zero)
    double S_E = -rho_c * (vr_c*(dphi0_dr+dphip_dr) + vt_c*(dphi0_dt_r+dphip_dt_r));

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

// ========================= Pack / unpack ==========================

__global__
void k_lm_pack(const double* rho, const double* mr, const double* mt,
               const double* rhoE, double* packed,
               int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt, n = nr*nt;
    int k = d_idx(i,j,nt,ng);
    packed[flat] = rho[k]; packed[n+flat] = mr[k];
    packed[2*n+flat] = mt[k]; packed[3*n+flat] = rhoE[k];
}

__global__
void k_lm_unpack_set(double* rho, double* mr, double* mt, double* rhoE,
                     const double* packed,
                     int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt, n = nr*nt;
    int k = d_idx(i,j,nt,ng);
    rho[k] = packed[flat]; mr[k] = packed[n+flat];
    mt[k] = packed[2*n+flat]; rhoE[k] = packed[3*n+flat];
}

__global__
void k_lm_unpack_add(double* rho, double* mr, double* mt, double* rhoE,
                     const double* delta, double alpha,
                     int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt, n = nr*nt;
    int k = d_idx(i,j,nt,ng);
    rho[k] += alpha * delta[flat]; mr[k] += alpha * delta[n+flat];
    mt[k] += alpha * delta[2*n+flat]; rhoE[k] += alpha * delta[3*n+flat];
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

static double gpu_dot(const double* a, const double* b, double* wa, int n) {
    int B = 256, nb = (n+B-1)/B;
    k_lm_dot<<<nb,B,B*sizeof(double)>>>(a, b, wa, n);
    std::vector<double> h(nb);
    CUDA_CHECK(cudaMemcpy(h.data(), wa, nb*sizeof(double), cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (int i = 0; i < nb; ++i) sum += h[i];
    return sum;
}

// ========================= Floor ==================================

__global__
void k_lm_floor(double* rho, double* mr, double* mt, double* rhoE,
                int nr, int nt, int ng, double gamma,
                double rho_fl, double P_fl) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int k = d_idx(flat/nt, flat%nt, nt, ng);
    if (rho[k] < rho_fl) {
        rho[k] = rho_fl; mr[k] = 0.0; mt[k] = 0.0;
        rhoE[k] = P_fl / (gamma - 1.0);
    }
    double e = rhoE[k] / fmax(rho[k], 1e-20);
    if (e < P_fl / ((gamma-1.0)*rho_fl))
        rhoE[k] = P_fl / (gamma - 1.0);
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

// ========================= Compute F (Newton residual) ============

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
}

void LowMachSolver::compute_residual(double* d_res) {
    int n = nr*nt, B = 256;
    launch_ghost();
    k_lm_residual<<<(n+B-1)/B,B>>>(
        d_rho,d_mr,d_mtheta,d_rhoE,
        d_cell_volume,d_area_r,d_area_theta,
        d_r_center,d_r_face,d_theta_face,d_dr,d_dtheta,
        d_phi, d_P0, d_rho0, d_phi0,
        d_residual,
        nr,nt,ng,gamma);
}

void LowMachSolver::compute_F(double* d_F, double dt) {
    int n = nr*nt, B = 256;
    compute_residual(d_residual);
    k_lm_compute_F<<<(n+B-1)/B,B>>>(d_F, d_residual,
        d_rho, d_mr, d_mtheta, d_rhoE, d_Un, 1.0/dt, nr, nt, ng);
}

void LowMachSolver::pack_state(double* d_packed) {
    int B = 256;
    k_lm_pack<<<(nr*nt+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE, d_packed, nr,nt,ng);
}

void LowMachSolver::unpack_set(const double* d_packed) {
    int B = 256;
    k_lm_unpack_set<<<(nr*nt+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE, d_packed, nr,nt,ng);
}

void LowMachSolver::unpack_delta(const double* d_delta, double alpha) {
    int B = 256;
    k_lm_unpack_add<<<(nr*nt+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE, d_delta, alpha, nr,nt,ng);
}

void LowMachSolver::apply_floor() {
    int B = 256;
    k_lm_floor<<<(nr*nt+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE, nr,nt,ng,gamma, 1e-15, 1e-15);
}

// ========================= Variable scaling (MESA-inspired) ========
// scale[i] = max(1, |Un[i]|) — normalizes Newton corrections so that
// GMRES works in a space where all components have comparable magnitude.

__global__
void k_lm_compute_scale(const double* Un, double* scale, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) scale[i] = fmax(1.0, fabs(Un[i]));
}

// Clamp correction: |delta[i]| <= max_rel * scale[i]
__global__
void k_lm_clamp(double* delta, const double* scale, double max_rel, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double lim = max_rel * scale[i];
    if (delta[i] > lim) delta[i] = lim;
    else if (delta[i] < -lim) delta[i] = -lim;
}

void LowMachSolver::compute_scaling() {
    int N = 4*nr*nt, B = 256;
    k_lm_compute_scale<<<(N+B-1)/B,B>>>(d_Un, d_scale, N);
}

void LowMachSolver::clamp_correction(double* d_delta, double max_rel_change) {
    int N = 4*nr*nt, B = 256;
    k_lm_clamp<<<(N+B-1)/B,B>>>(d_delta, d_scale, max_rel_change, N);
}

// ========================= Block-diagonal Jacobi preconditioner ====
// Approximate diagonal of J = dF/dU ≈ diag(-1/dt·I + dR/dU_diag).
// For each cell, the 4×4 diagonal block is:
//   J_ii ≈ -1/dt·I + diag(upwind_coeff)
// We store and invert these 4×4 blocks analytically (diagonal-dominant).

__global__
void k_lm_assemble_blkjac(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* dr, const double* rc, const double* dtheta, const double* vol,
    double* blk, int nr, int nt, int ng, double gamma, double inv_dt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    int k = d_idx(i,j,nt,ng);

    double rho_c = fmax(rho[k], 1e-20);
    double vr = fabs(mr[k] / rho_c);
    double vt = fabs(mt[k] / rho_c);

    // Upwind advection spectral radius per direction
    double sr_r = vr / dr[i];
    double sr_t = vt / (rc[i] * dtheta[j]);
    double sr = sr_r + sr_t;

    // Diagonal dominance: 1/dt + advection rate
    double d0 = inv_dt + sr;              // ρ equation
    double d1 = inv_dt + sr;              // ρvr equation
    double d2 = inv_dt + sr;              // ρvθ equation
    double d3 = inv_dt + sr;              // ρe equation

    // Store inverse of diagonal 4×4 block (since it's diagonal, inv = 1/d)
    double* B = &blk[flat * 16];
    for (int q = 0; q < 16; ++q) B[q] = 0.0;
    B[0]  = 1.0 / d0;
    B[5]  = 1.0 / d1;
    B[10] = 1.0 / d2;
    B[15] = 1.0 / d3;
}

// Apply block-diagonal preconditioner: Mv[cell] = B[cell] * v[cell]
__global__
void k_lm_apply_blkjac(const double* blk, const double* v, double* Mv,
                        int n) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n) return;
    const double* B = &blk[cell * 16];
    double v0 = v[cell], v1 = v[n+cell], v2 = v[2*n+cell], v3 = v[3*n+cell];
    Mv[cell]     = B[0]*v0 + B[1]*v1 + B[2]*v2 + B[3]*v3;
    Mv[n+cell]   = B[4]*v0 + B[5]*v1 + B[6]*v2 + B[7]*v3;
    Mv[2*n+cell] = B[8]*v0 + B[9]*v1 + B[10]*v2 + B[11]*v3;
    Mv[3*n+cell] = B[12]*v0 + B[13]*v1 + B[14]*v2 + B[15]*v3;
}

void LowMachSolver::assemble_block_jacobi(double dt) {
    int n = nr*nt, B = 256;
    k_lm_assemble_blkjac<<<(n+B-1)/B,B>>>(
        d_rho, d_mr, d_mtheta, d_rhoE,
        d_dr, d_r_center, d_dtheta, d_cell_volume,
        d_blk_diag, nr, nt, ng, gamma, 1.0/dt);
}

void LowMachSolver::apply_preconditioner(const double* d_v, double* d_Mv) {
    int n = nr*nt, N = 4*n, B = 256;
    if (precond_type == PrecondType::BLOCK_JACOBI) {
        k_lm_apply_blkjac<<<(n+B-1)/B,B>>>(d_blk_diag, d_v, d_Mv, n);
    } else {
        k_lm_copy<<<(N+B-1)/B,B>>>(d_Mv, d_v, N);
    }
}

// ========================= JFNK matvec ============================
// J·v ≈ [F(U+εv) - F(U)] / ε, gravity frozen.

void LowMachSolver::jfnk_matvec(const double* d_v, double* d_Jv, double dt) {
    int N = 4*nr*nt, B = 256;

    double norm_v = sqrt(gpu_dot(d_v, d_v, d_work_a, N));
    if (norm_v < 1e-30) { k_lm_zero<<<(N+B-1)/B,B>>>(d_Jv, N); return; }

    // Walker-Pernice epsilon (from MOOSE/PETSc)
    double norm_U = sqrt(gpu_dot(d_Un, d_Un, d_work_a, N));
    double eps_fd = sqrt(1e-15) * (1.0 + norm_U) / norm_v;

    // Save current state
    k_lm_pack<<<(nr*nt+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE, d_gmres_Uk, nr,nt,ng);

    // Perturb: U += ε·v
    k_lm_unpack_add<<<(nr*nt+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE, d_v, eps_fd, nr,nt,ng);
    apply_floor();

    // F(U+εv) with frozen gravity
    compute_F(d_Jv, dt);

    // Jv = (F(U+εv) - Fk) / ε
    k_lm_axpy<<<(N+B-1)/B,B>>>(d_Jv, -1.0, d_Fk, N);
    k_lm_scale<<<(N+B-1)/B,B>>>(d_Jv, 1.0/eps_fd, N);

    // Restore state
    k_lm_unpack_set<<<(nr*nt+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE, d_gmres_Uk, nr,nt,ng);
}

// ========================= FGMRES ================================
// Flexible GMRES: Z[j] = M⁻¹·V[j], solution built from Z vectors.
// This allows the preconditioner to vary (needed for future SIMPLE/GMG).

int LowMachSolver::gmres_solve(double* d_x, const double* d_b, double dt,
                                double tol, int max_iter) {
    int N = 4*nr*nt, B = 256;
    int m = std::min(max_iter, (int)GMRES_RESTART);

    std::vector<double> H((m+1)*m, 0.0);
    std::vector<double> cs(m), sn(m), g(m+1, 0.0);

    // r0 = -F (solve J·x = -F)
    k_lm_copy<<<(N+B-1)/B,B>>>(d_gmres_V[0], d_b, N);
    k_lm_scale<<<(N+B-1)/B,B>>>(d_gmres_V[0], -1.0, N);

    double beta = sqrt(gpu_dot(d_gmres_V[0], d_gmres_V[0], d_work_a, N));
    if (beta < 1e-30) return 0;
    k_lm_scale<<<(N+B-1)/B,B>>>(d_gmres_V[0], 1.0/beta, N);
    g[0] = beta;

    int j;
    for (j = 0; j < m; ++j) {
        // Z[j] = M⁻¹ · V[j]
        apply_preconditioner(d_gmres_V[j], d_gmres_Z[j]);

        // w = J · Z[j]
        jfnk_matvec(d_gmres_Z[j], d_gmres_w, dt);

        // Arnoldi
        for (int i = 0; i <= j; ++i) {
            H[i*m+j] = gpu_dot(d_gmres_w, d_gmres_V[i], d_work_a, N);
            k_lm_axpy<<<(N+B-1)/B,B>>>(d_gmres_w, -H[i*m+j], d_gmres_V[i], N);
        }
        H[(j+1)*m+j] = sqrt(gpu_dot(d_gmres_w, d_gmres_w, d_work_a, N));

        if (H[(j+1)*m+j] < 1e-30) { j++; break; }
        k_lm_copy<<<(N+B-1)/B,B>>>(d_gmres_V[j+1], d_gmres_w, N);
        k_lm_scale<<<(N+B-1)/B,B>>>(d_gmres_V[j+1], 1.0/H[(j+1)*m+j], N);

        // Givens rotations
        for (int i = 0; i < j; ++i) {
            double h1 = H[i*m+j], h2 = H[(i+1)*m+j];
            H[i*m+j]     =  cs[i]*h1 + sn[i]*h2;
            H[(i+1)*m+j] = -sn[i]*h1 + cs[i]*h2;
        }
        double h1 = H[j*m+j], h2 = H[(j+1)*m+j];
        double t = sqrt(h1*h1 + h2*h2);
        cs[j] = h1/t; sn[j] = h2/t;
        H[j*m+j] = t; H[(j+1)*m+j] = 0.0;
        g[j+1] = -sn[j]*g[j]; g[j] = cs[j]*g[j];

        if (fabs(g[j+1]) < tol * beta) { j++; break; }
    }

    // Back-substitution: x = Σ y[i] · Z[i]  (Z, not V!)
    std::vector<double> y(j);
    for (int i = j-1; i >= 0; --i) {
        y[i] = g[i];
        for (int kk = i+1; kk < j; ++kk)
            y[i] -= H[i*m+kk] * y[kk];
        y[i] /= H[i*m+i];
    }

    k_lm_zero<<<(N+B-1)/B,B>>>(d_x, N);
    for (int i = 0; i < j; ++i)
        k_lm_axpy<<<(N+B-1)/B,B>>>(d_x, y[i], d_gmres_Z[i], N);

    return j;
}

// ========================= CFL (advection only) ===================

__global__
void k_lm_cfl(const double* rho, const double* mr, const double* mt,
              const double* dr, const double* rc, const double* dtheta,
              double* out, int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    int k = d_idx(i,j,nt,ng);
    double r = fmax(rho[k], 1e-20);
    double vr = fabs(mr[k] / r);
    double vt = fabs(mt[k] / r);
    double dt_r = dr[i] / fmax(vr, 1e-20);
    double dt_t = rc[i] * dtheta[j] / fmax(vt, 1e-20);
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
    k_lm_cfl<<<(n+B-1)/B,B>>>(d_rho,d_mr,d_mtheta, d_dr,d_r_center,d_dtheta,
                                d_work_a, nr,nt,ng);
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

// ========================= Time step (Newton) =====================

double LowMachSolver::step(double t, double t_end) {
    int n = nr*nt, B = 256, N = 4*n;

    static int step_count = 0;

    // HSE snapshot + diagnostic on first call.
    // If snapshot_hse() was already called externally (e.g. for perturbed IC),
    // skip re-snapshotting to preserve the unperturbed reference.
    static bool hse_init = false;
    if (!hse_init) {
        if (!hse_set_externally) {
            snapshot_hse();
        }
        diagnose_hse_residual();
        hse_init = true;
    }

    // Ensure clean state before packing Un.
    // Do NOT re-solve gravity here — Φ is already consistent with ρ
    // from the previous step (or from diagnose_hse_residual at init).
    // Re-solving would introduce O(tol) Φ perturbation that breaks
    // the well-balanced reference state.
    apply_floor();

    // Advection CFL (no sound speed!)
    double dt_cfl = compute_cfl_dt();
    // For near-static initial conditions, dt_cfl can be huge — cap it
    double dt_cap = 0.1;
    if (dt_current < 1e-30) dt_current = std::min(dt_cfl, dt_cap);
    double dt = std::min({dt_current, dt_cfl, t_end - t});

    pack_state(d_Un);
    compute_scaling();

    int max_newton = 20;
    int max_dt_cuts = 8;
    bool converged = false;

    double dt_min = 1e-8 * dt;

    for (int cut = 0; cut < max_dt_cuts && !converged; ++cut) {
        if (cut > 0) {
            dt *= 0.5;
            if (dt < dt_min) {
                std::fprintf(stderr, "FATAL: step %d dt=%.3e < dt_min=%.3e, solver diverged.\n",
                            step_count, dt, dt_min);
                std::exit(1);
            }
            unpack_set(d_Un);
            if (step_count < 30)
                std::fprintf(stderr, "  step %d: Newton failed, cutting dt -> %.3e\n",
                            step_count, dt);
        }

        if (precond_type == PrecondType::BLOCK_JACOBI)
            assemble_block_jacobi(dt);

        double Fnorm0 = 0.0;
        bool diverged = false;

        for (int newton = 0; newton < max_newton; ++newton) {
            apply_floor();
            if (newton > 0)
                solve_gravity();

            compute_F(d_Fk, dt);
            double Fnorm = sqrt(gpu_dot(d_Fk, d_Fk, d_work_a, N));

            if (newton == 0) {
                Fnorm0 = Fnorm;
                if (Fnorm < 1e-30) { converged = true; break; }
            }

            // Convergence: relative OR per-cell absolute
            double Fnorm_per_cell = Fnorm / sqrt((double)N);
            if (Fnorm < 1e-4 * Fnorm0 || Fnorm_per_cell < 1e-6) {
                converged = true;
                if (step_count < 30)
                    std::fprintf(stderr, "  step %d converged at newton %d: ||F||=%.3e per-cell=%.3e (dt=%.3e)\n",
                                step_count, newton, Fnorm, Fnorm_per_cell, dt);
                break;
            }
            if (Fnorm > 1e6 * Fnorm0 || std::isnan(Fnorm)) { diverged = true; break; }

            int gmres_iters = gmres_solve(d_gmres_w, d_Fk, dt, 1e-3, GMRES_RESTART);

            // MESA-style correction clamping: |δU_i| ≤ 0.1 * scale_i
            clamp_correction(d_gmres_w, 0.1);

            // Line search with backtracking
            k_lm_pack<<<(n+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE, d_gmres_Uk, nr,nt,ng);
            double alpha = 1.0;
            double Fnorm_new = Fnorm;
            for (int ls = 0; ls < 8; ++ls) {
                k_lm_unpack_set<<<(n+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE, d_gmres_Uk, nr,nt,ng);
                unpack_delta(d_gmres_w, alpha);
                apply_floor();
                solve_gravity();
                compute_F(d_residual_ls, dt);
                Fnorm_new = sqrt(gpu_dot(d_residual_ls, d_residual_ls, d_work_a, N));
                if (Fnorm_new < Fnorm) break;
                alpha *= 0.5;
            }

            if (Fnorm_new >= Fnorm) {
                k_lm_unpack_set<<<(n+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE, d_gmres_Uk, nr,nt,ng);
                diverged = true;
                if (step_count < 30)
                    std::fprintf(stderr, "  step %d newton %d: line search failed, ||F||=%.3e\n",
                                step_count, newton, Fnorm);
                break;
            }

            if (step_count < 20)
                std::fprintf(stderr, "  step %d newton %d: ||F|| %.3e -> %.3e  a=%.2f  GMRES %d  dt=%.3e\n",
                            step_count, newton, Fnorm, Fnorm_new, alpha, gmres_iters, dt);
        }

        if (diverged)
            unpack_set(d_Un);
    }

    if (converged)
        dt_current = std::min(1.5 * dt, std::min(dt_cfl, dt_cap));
    else
        dt_current = 0.25 * dt;

    step_count++;
    return dt;
}

// ========================= HSE diagnostic ==========================

void LowMachSolver::snapshot_hse() {
    int n = nr*nt, B = 256;
    apply_floor();
    solve_gravity();
    k_lm_snapshot_hse<<<(n+B-1)/B,B>>>(d_rho,d_rhoE,d_phi,
        d_rho0,d_P0,d_phi0, nr,nt,ng,gamma);
    hse_set_externally = true;
}

void LowMachSolver::diagnose_hse_residual() {
    int n = nr*nt, N = 4*n;

    compute_residual(d_residual);

    // Download and find max per equation
    std::vector<double> h_res(N);
    CUDA_CHECK(cudaMemcpy(h_res.data(), d_residual, N*sizeof(double), cudaMemcpyDeviceToHost));

    double max_rho=0, max_mr=0, max_mt=0, max_rhoE=0;
    for (int i = 0; i < n; ++i) {
        max_rho  = std::max(max_rho,  std::abs(h_res[i]));
        max_mr   = std::max(max_mr,   std::abs(h_res[n+i]));
        max_mt   = std::max(max_mt,   std::abs(h_res[2*n+i]));
        max_rhoE = std::max(max_rhoE, std::abs(h_res[3*n+i]));
    }

    double l2 = 0;
    for (int i = 0; i < N; ++i) l2 += h_res[i]*h_res[i];
    l2 = std::sqrt(l2);

    std::fprintf(stderr, "=== HSE Diagnostic (initial residual R(U₀)) ===\n");
    std::fprintf(stderr, "  max|R_ρ|    = %.6e\n", max_rho);
    std::fprintf(stderr, "  max|R_ρvr|  = %.6e\n", max_mr);
    std::fprintf(stderr, "  max|R_ρvθ|  = %.6e\n", max_mt);
    std::fprintf(stderr, "  max|R_ρe|   = %.6e\n", max_rhoE);
    std::fprintf(stderr, "  ||R||₂      = %.6e\n", l2);
    std::fprintf(stderr, "  Target: all < 1e-10 for well-balanced scheme\n");
    std::fprintf(stderr, "================================================\n");
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

    // Newton vectors (4*n packed)
    CUDA_CHECK(cudaMalloc(&d_Un, 4*n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Fk, 4*n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_residual, 4*n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_residual_ls, 4*n*sizeof(double)));

    // GMRES vectors
    for (int i = 0; i <= GMRES_RESTART; ++i) {
        CUDA_CHECK(cudaMalloc(&d_gmres_V[i], 4*n*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_gmres_Z[i], 4*n*sizeof(double)));
    }
    CUDA_CHECK(cudaMalloc(&d_gmres_w, 4*n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gmres_Uk, 4*n*sizeof(double)));

    // Work buffers
    CUDA_CHECK(cudaMalloc(&d_work_a, n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_work_b, n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rhs_poisson, n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_inv_rho, n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_scale, 4*n*sizeof(double)));

    // Block-diagonal Jacobi preconditioner
    d_blk_diag = nullptr;
    if (precond_type == PrecondType::BLOCK_JACOBI) {
        CUDA_CHECK(cudaMalloc(&d_blk_diag, n*16*sizeof(double)));
    }

    // GMG for gravity
    gmg.init(nr, nt, grid.r_face.data(), grid.theta_face.data());

    initialized = true;
    const char* pc_name = (precond_type == PrecondType::BLOCK_JACOBI) ? "BlockJacobi" : "None";
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
    cudaFree(d_r_face); cudaFree(d_r_center); cudaFree(d_dr);
    cudaFree(d_theta_face); cudaFree(d_theta_center); cudaFree(d_dtheta);
    cudaFree(d_cell_volume); cudaFree(d_area_r); cudaFree(d_area_theta);
    cudaFree(d_rho); cudaFree(d_mr); cudaFree(d_mtheta); cudaFree(d_rhoE);
    cudaFree(d_phi); cudaFree(d_pi);
    cudaFree(d_rho0); cudaFree(d_P0); cudaFree(d_phi0);
    cudaFree(d_Un); cudaFree(d_Fk); cudaFree(d_residual); cudaFree(d_residual_ls);
    for (int i = 0; i <= GMRES_RESTART; ++i) {
        cudaFree(d_gmres_V[i]);
        cudaFree(d_gmres_Z[i]);
    }
    cudaFree(d_gmres_w); cudaFree(d_gmres_Uk);
    cudaFree(d_work_a); cudaFree(d_work_b);
    cudaFree(d_rhs_poisson); cudaFree(d_inv_rho); cudaFree(d_scale);
    if (d_blk_diag) cudaFree(d_blk_diag);
    initialized = false;
}
