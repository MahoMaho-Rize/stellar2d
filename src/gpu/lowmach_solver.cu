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
    const double* gr, const double* gr0,
    const double* P0, const double* rho0,
    double* res,
    int nr, int nt, int ng, double gamma, double atm_thresh) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    int n = nr*nt;

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

    double Pp_c = P_c - P0[flat];
    double rhop_c = rho_c - rho0[flat];

    // Radial pressure perturbation gradient
    double dPp_dr = 0;
    if (i > 0 && i < nr-1) {
        double Pp_m = fmax((gamma-1.0)*rhoE[d_idx(i-1,j,nt,ng)],1e-30) - P0[(i-1)*nt+j];
        double Pp_p = fmax((gamma-1.0)*rhoE[d_idx(i+1,j,nt,ng)],1e-30) - P0[(i+1)*nt+j];
        double dl = r_center[i]-r_center[i-1], dh = r_center[i+1]-r_center[i];
        dPp_dr = (dh*(Pp_c-Pp_m)/dl + dl*(Pp_p-Pp_c)/dh) / (dl+dh);
    } else if (i == 0 && nr > 1) {
        double dh = r_center[1]-r_center[0];
        dPp_dr = (fmax((gamma-1.0)*rhoE[d_idx(1,j,nt,ng)],1e-30)-P0[1*nt+j] - Pp_c)/dh;
    } else if (i == nr-1 && nr >= 2) {
        double dl = r_center[nr-1]-r_center[nr-2];
        dPp_dr = (Pp_c - (fmax((gamma-1.0)*rhoE[d_idx(nr-2,j,nt,ng)],1e-30)-P0[(nr-2)*nt+j]))/dl;
    }

    // Theta pressure perturbation gradient (divided by r)
    double dPp_dt_r = 0;
    if (j > 0 && j < nt-1) {
        double tc_m=0.5*(theta_face[j-1]+theta_face[j]);
        double tc_c=0.5*(theta_face[j]+theta_face[j+1]);
        double tc_p=0.5*(theta_face[j+1]+theta_face[j+2]);
        double dl=tc_c-tc_m, dh=tc_p-tc_c;
        double Pp_m = fmax((gamma-1.0)*rhoE[d_idx(i,j-1,nt,ng)],1e-30) - P0[i*nt+j-1];
        double Pp_p = fmax((gamma-1.0)*rhoE[d_idx(i,j+1,nt,ng)],1e-30) - P0[i*nt+j+1];
        dPp_dt_r = ((dh*(Pp_c-Pp_m)/dl + dl*(Pp_p-Pp_c)/dh))/(r*(dl+dh));
    }

    // 1D gravity: g_r(i) is radial only, same for all θ cells at this radius
    double g0_r = gr0[i];
    double gp_r = gr[i] - g0_r;

    // ===== Geometric source =====
    double inv_r = 1.0 / r;
    double S_mr = rho_c * vt_c * vt_c * inv_r;
    double S_mt = -rho_c * vr_c * vt_c * inv_r;

    // Well-balanced radial force: -∇P' + ρ'·g₀ + ρ·g'
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
    double P = (gamma - 1.0) * rhoE[k];
    if (P < P_fl)
        rhoE[k] = P_fl / (gamma - 1.0);
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

// ========================= Variable scaling =============================
// Two separate scaling systems:
//
// 1. Clamp scale (MESA): max(1, |U|) — limits Newton correction magnitude
__global__
void k_lm_compute_scale(const double* Un, double* scale, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) scale[i] = fmax(1.0, fabs(Un[i]));
}

// 2. MUSIC L/R scaling (Viallet 2016): balances GMRES residual/unknown norms
//    R (unknowns): what "1 unit" of correction means
//    L (residuals): what "1 unit" of residual means
//    GMRES solves (L⁻¹·J·R)·x̃ = -L⁻¹·F, then δU = R·x̃
__global__
void k_lm_compute_music_scale(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    double* scR, double* scL,
    int nr, int nt, int ng, double gamma) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int n = nr*nt;
    int i = flat/nt, j = flat%nt;
    int k = (i + ng) * (nt + 2*ng) + (j + ng);

    double rho_c = fmax(rho[k], 1e-20);
    double e_c = rhoE[k] / rho_c;
    double P = fmax((gamma - 1.0) * rho_c * e_c, 1e-30);
    double cs = sqrt(gamma * P / rho_c);
    double vr = mr[k] / rho_c, vt = mt[k] / rho_c;

    double a1 = 1e-5, a2 = 1.0;

    // Right scaling (unknowns)
    scR[flat]       = fmax(rho_c, 1.0);
    scR[n + flat]   = fmax(fabs(mr[k]),  rho_c * a1 * cs);
    scR[2*n + flat] = fmax(fabs(mt[k]),  rho_c * a1 * cs);
    scR[3*n + flat] = fmax(fabs(rhoE[k]), 1.0);

    // Left scaling (residuals) — floor at 1.0 to avoid division by tiny atmosphere values
    scL[flat]       = fmax(rho_c, 1.0);
    scL[n + flat]   = fmax(rho_c * fmax(fabs(vr), a2 * cs), 1.0);
    scL[2*n + flat] = fmax(rho_c * fmax(fabs(vt), a2 * cs), 1.0);
    scL[3*n + flat] = fmax(fabs(rhoE[k]), 1.0);
}

// Elementwise v[i] *= s[i]
__global__ void k_lm_emul(double* v, const double* s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] *= s[i];
}
// Elementwise v[i] /= s[i]
__global__ void k_lm_ediv(double* v, const double* s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] /= s[i];
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

// ========================= Block-diagonal Jacobi preconditioner ====
// Numerically assemble the TRUE 4×4 diagonal block of J = dF/dU for
// each cell by column-wise finite differencing of the residual.
// This captures ALL intra-cell coupling: pressure-energy, gravity-density, etc.
// Then LU-invert each 4×4 block on the GPU.

// Assemble 4×4 Jacobian diagonal block analytically.
// J = dF/dU where F = R(U) - (U-Un)/dt.
// The diagonal block captures how each cell's F depends on its OWN state.
//
// Key couplings:
//   dF_mr/d(rhoE) = -dP'/dr·d(P)/d(rhoE) = -(γ-1)/Δr  (pressure-energy)
//   dF_rhoE/d(mr) = -P·d(div_v)/d(mr)                   (compression work)
//   dF_mr/d(rho)  = -ρ'·dΦ₀/dr → captured via ρ dependence
// Plus the time derivative: -1/dt on diagonal.
//
// We compute this analytically instead of numerically to avoid
// the race condition of in-place FD on global arrays.

__global__
void k_lm_assemble_blkjac(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* vol, const double* ar, const double* at,
    const double* r_center, const double* r_face, const double* theta_face,
    const double* dr, const double* dtheta,
    const double* gr0,
    double* blk, double* blk_J,
    int nr, int nt, int ng, double gamma, double inv_dt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    int k = d_idx(i,j,nt,ng);

    double rho_c = fmax(rho[k], 1e-20);
    double vr_c = mr[k] / rho_c;
    double vt_c = mt[k] / rho_c;
    double e_c = rhoE[k] / rho_c;
    double P_c = fmax((gamma-1.0)*rho_c*e_c, 1e-30);
    double r = r_center[i];
    double invV = 1.0 / vol[flat];

    // Advection rate (upwind diagonal contribution)
    double sr_r = fabs(vr_c) / dr[i];
    double sr_t = fabs(vt_c) / (r * dtheta[j]);
    double sr = sr_r + sr_t;

    // Pressure gradient coupling: dP/d(rhoE) = (γ-1)
    // Central diff stencil: dP_dr involves P at i-1,i,i+1
    // The diagonal part (dependence on P_i) has coefficient:
    //   for weighted central diff: -(dh/(dl*(dl+dh)) + dl/(dh*(dl+dh)))
    double dPdr_coeff = 0.0; // d(dP/dr)/dP_c
    if (i > 0 && i < nr-1) {
        double dl = r_center[i]-r_center[i-1], dh = r_center[i+1]-r_center[i];
        dPdr_coeff = -(dh/(dl*(dl+dh)) + dl/(dh*(dl+dh)));
    } else if (i == 0 && nr > 1) {
        dPdr_coeff = -1.0/(r_center[1]-r_center[0]);
    } else if (i == nr-1 && nr >= 2) {
        dPdr_coeff = 1.0/(r_center[nr-1]-r_center[nr-2]);
    }

    double dPdt_coeff = 0.0; // d((1/r)dP/dθ)/dP_c
    if (j > 0 && j < nt-1) {
        double tc_m=0.5*(theta_face[j-1]+theta_face[j]);
        double tc_c=0.5*(theta_face[j]+theta_face[j+1]);
        double tc_p=0.5*(theta_face[j+1]+theta_face[j+2]);
        double dl=tc_c-tc_m, dh=tc_p-tc_c;
        dPdt_coeff = -(dh/(dl*(dl+dh)) + dl/(dh*(dl+dh))) / r;
    }

    // dP/d(rhoE) = (γ-1)  (since P = (γ-1)*rhoE for our state variable)
    double dP_drhoE = gamma - 1.0;

    // 1D gravity coupling: dF_mr/d(rho) += g₀(r) (from ρ'·g₀ term)
    double g0_val = gr0[i];

    // div_v diagonal contribution: d(div_v)/d(mr_c) and d(div_v)/d(mt_c)
    // From FV divergence: div_v = (1/V)(Ar_hi*vr_hi - Ar_lo*vr_lo + ...)
    // vr at face (i,i+1) = 0.5*(mr[k]/ρ + mr[k_right]/ρ_right)
    // d(vr_face)/d(mr[k]) = 0.5/ρ for both faces touching cell k
    double Ar_hi=ar[(i+1)*nt+j], Ar_lo=ar[i*nt+j];
    double At_hi=at[i*(nt+1)+j+1], At_lo=at[i*(nt+1)+j];
    double ddivv_dmr = invV * 0.5/rho_c * (Ar_hi + Ar_lo);
    double ddivv_dmt = invV * 0.5/rho_c * (At_hi + At_lo);

    // Build 4×4 J block (row-major)
    double J[16];
    for (int q=0; q<16; q++) J[q] = 0.0;

    // Diagonal: -1/dt - advection_rate
    J[0]  = -(inv_dt + sr);  // dF_rho/d(rho)
    J[5]  = -(inv_dt + sr);  // dF_mr/d(mr)
    J[10] = -(inv_dt + sr);  // dF_mt/d(mt)
    J[15] = -(inv_dt + sr);  // dF_rhoE/d(rhoE)

    // Pressure-energy coupling: dF_mr/d(rhoE) = -d(dP'/dr)/d(rhoE)
    // = -dPdr_coeff * dP/d(rhoE) = -dPdr_coeff * (γ-1)
    J[1*4+3] = -dPdr_coeff * dP_drhoE;  // dF_mr/d(rhoE)
    J[2*4+3] = -dPdt_coeff * dP_drhoE;  // dF_mt/d(rhoE)

    // 1D gravity-density coupling: dF_mr/d(rho) += g₀(r) (from +ρ'·g₀ term)
    J[1*4+0] += g0_val;   // dF_mr/d(rho)
    // No θ gravity component with 1D gravity

    // Compression work coupling: dF_rhoE/d(mr) = -P * d(div_v)/d(mr)
    J[3*4+1] = -P_c * ddivv_dmr;  // dF_rhoE/d(mr)
    J[3*4+2] = -P_c * ddivv_dmt;  // dF_rhoE/d(mt)

    // 4×4 LU with partial pivoting → invert
    double A[16]; for(int q=0;q<16;q++) A[q]=J[q];
    double inv_m[16]; for(int q=0;q<16;q++) inv_m[q]=(q/4==q%4)?1.0:0.0;

    for(int col=0;col<4;col++){
        double mx=fabs(A[col*4+col]); int mi=col;
        for(int row=col+1;row<4;row++){if(fabs(A[row*4+col])>mx){mx=fabs(A[row*4+col]);mi=row;}}
        if(mi!=col){
            for(int q=0;q<4;q++){double t=A[col*4+q];A[col*4+q]=A[mi*4+q];A[mi*4+q]=t;}
            for(int q=0;q<4;q++){double t=inv_m[col*4+q];inv_m[col*4+q]=inv_m[mi*4+q];inv_m[mi*4+q]=t;}
        }
        double d=A[col*4+col];
        if(fabs(d)<1e-30) d=(d>=0?1e-30:-1e-30);
        for(int row=col+1;row<4;row++){
            double m=A[row*4+col]/d;
            for(int q=col;q<4;q++) A[row*4+q]-=m*A[col*4+q];
            for(int q=0;q<4;q++) inv_m[row*4+q]-=m*inv_m[col*4+q];
        }
    }
    for(int c=0;c<4;c++){
        for(int row=3;row>=0;row--){
            double s=inv_m[row*4+c];
            for(int q=row+1;q<4;q++) s-=A[row*4+q]*inv_m[q*4+c];
            inv_m[row*4+c]=s/A[row*4+row];
        }
    }

    double* BJ = &blk_J[flat*16];
    for(int q=0;q<16;q++) BJ[q]=J[q];

    double* B = &blk[flat*16];
    for(int q=0;q<16;q++) B[q]=inv_m[q];
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
        d_cell_volume, d_area_r, d_area_theta,
        d_r_center, d_r_face, d_theta_face,
        d_dr, d_dtheta,
        d_gr0,
        d_blk_diag, d_blk_J, nr, nt, ng, gamma, 1.0/dt);
}

// ========================= SIMPLE preconditioner ===================
// Given residual vector r = (r_ρ, r_mr, r_mt, r_ρe):
//
// Step 1: Approximate momentum solve (diagonal):
//   δvr* = r_mr / Ap,  δvt* = r_mt / Ap
//   where Ap = 1/dt + upwind_rate (per cell)
//
// Step 2: Pressure Poisson (GMG):
//   ∇·(dt/ρ · ∇δp) = ∇·(δv*)    [constant-coeff approx: dt/ρ ≈ dt/ρ₀]
//   Actually use: ∇²δp = (ρ/dt) · ∇·(δv*)  with same Laplacian as gravity
//
// Step 3: Velocity correction:
//   δvr = δvr* - (dt/ρ) ∂δp/∂r
//   δvt = δvt* - (dt/ρ) (1/r)∂δp/∂θ
//
// Step 4: Pass through density and energy:
//   δρ  = r_ρ / (1/dt)  = dt * r_ρ
//   δρe = r_ρe / (1/dt) = dt * r_ρe

__global__
void k_lm_simple_mom_diag(const double* rho, const double* mr, const double* mt,
                          const double* dr, const double* rc, const double* dtheta,
                          double* Ap, int nr, int nt, int ng, double inv_dt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    int k = d_idx(i,j,nt,ng);
    double rho_c = fmax(rho[k], 1e-20);
    double vr = fabs(mr[k]/rho_c), vt = fabs(mt[k]/rho_c);
    Ap[flat] = inv_dt + vr/dr[i] + vt/(rc[i]*dtheta[j]);
}

// Step 1: δv* = -r_momentum / Ap  (negative because J_diag = -Ap)
__global__
void k_lm_simple_vstar(const double* r_mr, const double* r_mt,
                       const double* Ap, double* vr_s, double* vt_s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double inv_ap = -1.0 / Ap[i];
    vr_s[i] = r_mr[i] * inv_ap;
    vt_s[i] = r_mt[i] * inv_ap;
}

// Step 2: compute divergence of δv* using FV divergence theorem
// div = (1/V) Σ(A_face · v_face), where v_face = average of neighbors
__global__
void k_lm_simple_div(const double* vr_s, const double* vt_s,
                     const double* vol, const double* ar, const double* at,
                     double* div_out, int nr, int nt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    double invV = 1.0 / vol[flat];

    // Radial face velocities (average of neighbors, zero at boundaries)
    double vr_lo = (i > 0)    ? 0.5*(vr_s[(i-1)*nt+j] + vr_s[i*nt+j]) : 0.0;
    double vr_hi = (i < nr-1) ? 0.5*(vr_s[i*nt+j] + vr_s[(i+1)*nt+j]) : 0.0;

    // Theta face velocities
    double vt_lo = (j > 0)    ? 0.5*(vt_s[i*nt+j-1] + vt_s[i*nt+j]) : 0.0;
    double vt_hi = (j < nt-1) ? 0.5*(vt_s[i*nt+j] + vt_s[i*nt+j+1]) : 0.0;

    double Ar_hi = ar[(i+1)*nt+j], Ar_lo = ar[i*nt+j];
    double At_hi = at[i*(nt+1)+j+1], At_lo = at[i*(nt+1)+j];

    div_out[flat] = invV*(Ar_hi*vr_hi - Ar_lo*vr_lo + At_hi*vt_hi - At_lo*vt_lo);
}

// Step 2b: compute α = 1/Ap for variable-coefficient Poisson
__global__
void k_lm_simple_alpha(const double* Ap, double* alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) alpha[i] = 1.0 / Ap[i];
}

// Poisson RHS = div(v*), with Dirichlet δp=0 at outer boundary
__global__
void k_lm_simple_prhs(const double* div_v, double* rhs, int nr, int nt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    rhs[flat] = div_v[flat];
    if (flat/nt == nr-1) rhs[flat] = 0.0;
}

// Step 3: correct velocity and assemble output
// out_ρ  = -dt * r_ρ
// out_mr = δvr* - (1/Ap) * ∂δp/∂r
// out_mt = δvt* - (1/Ap) * (1/r)∂δp/∂θ
// out_ρe = -dt * r_ρe
__global__
void k_lm_simple_correct(
    const double* r_rho, const double* r_mr, const double* r_mt, const double* r_rhoE,
    const double* Ap,
    const double* vr_s, const double* vt_s,
    const double* dp,
    const double* rc, const double* theta_face,
    double* out,
    int nr, int nt, double dt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt, n = nr*nt;

    // Pressure gradient of correction δp
    double dp_dr = 0.0;
    if (i > 0 && i < nr-1) {
        double dl = rc[i]-rc[i-1], dh = rc[i+1]-rc[i];
        double gl = (dp[flat]-dp[(i-1)*nt+j])/dl;
        double gr = (dp[(i+1)*nt+j]-dp[flat])/dh;
        dp_dr = (dh*gl+dl*gr)/(dl+dh);
    } else if (i == 0 && nr > 1)
        dp_dr = (dp[1*nt+j]-dp[0])/( rc[1]-rc[0]);
    else if (i == nr-1 && nr >= 2)
        dp_dr = (dp[(nr-1)*nt+j]-dp[(nr-2)*nt+j])/(rc[nr-1]-rc[nr-2]);

    double dp_dt_r = 0.0;
    if (j > 0 && j < nt-1) {
        double tc_m = 0.5*(theta_face[j-1]+theta_face[j]);
        double tc_c = 0.5*(theta_face[j]+theta_face[j+1]);
        double tc_p = 0.5*(theta_face[j+1]+theta_face[j+2]);
        double dl = tc_c-tc_m, dh = tc_p-tc_c;
        dp_dt_r = ((dh*(dp[flat]-dp[i*nt+j-1])/dl + dl*(dp[i*nt+j+1]-dp[flat])/dh))
                  / (rc[i]*(dl+dh));
    }

    double inv_ap = 1.0 / Ap[flat];
    out[flat]       = -dt * r_rho[flat];
    out[n + flat]   = vr_s[flat] - inv_ap * dp_dr;
    out[2*n + flat] = vt_s[flat] - inv_ap * dp_dt_r;
    out[3*n + flat] = -dt * r_rhoE[flat];
}

// ========================= Line-Jacobi preconditioner ==============
// For each θ-line (fixed j), solve the block-tridiagonal system along r.
// Each cell has a 4×4 diagonal block (same as block Jacobi) plus
// 4×4 off-diagonal blocks from the radial upwind + pressure stencil.
// One GPU block per θ-line; sequential Thomas algorithm within the block.

__global__
void k_lm_line_solve(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* vol, const double* ar, const double* at,
    const double* r_center, const double* r_face, const double* theta_face,
    const double* dr, const double* dtheta,
    const double* gr0,
    const double* v_in, double* Mv_out,
    int nr, int nt, int ng, double gamma, double inv_dt) {
    int j = blockIdx.x;  // one block per θ-line
    if (j >= nt) return;

    // Shared memory for the Thomas algorithm: L[nr], D[nr], U[nr], rhs[nr]
    // Each is a 4×4 matrix stored as 16 doubles (row-major).
    extern __shared__ double smem[];
    double* L = smem;                      // nr * 16
    double* D = L + nr * 16;              // nr * 16
    double* U = D + nr * 16;              // nr * 16
    double* rhs_s = U + nr * 16;          // nr * 4
    double* x_s = rhs_s + nr * 4;         // nr * 4

    int tid = threadIdx.x;
    int n = nr * nt;

    // Step 1: Each thread assembles its row's L, D, U blocks
    for (int i = tid; i < nr; i += blockDim.x) {
        int flat = i * nt + j;
        int k = d_idx(i, j, nt, ng);

        double rho_c = fmax(rho[k], 1e-20);
        double vr_c = mr[k] / rho_c;
        double vt_c = mt[k] / rho_c;
        double P_c = fmax((gamma - 1.0) * rhoE[k], 1e-30);
        double r = r_center[i];
        double invV = 1.0 / vol[flat];
        double sr_t = fabs(vt_c) / (r * dtheta[j]);

        double Ar_hi = ar[(i+1)*nt+j], Ar_lo = ar[i*nt+j];

        // Diagonal: -(1/dt + sr) with sr including theta part
        double sr = fabs(vr_c)/dr[i] + sr_t;
        double diag_val = -(inv_dt + sr);

        // Off-diagonal from upwind advection in r:
        // Flux at face (i-1,i): vr_face * q_upwind. If vr>0, q comes from i-1.
        // dF/d(q_{i-1}) = vr_face * A_lo / V when upwinding from left.
        // dF/d(q_i) = -vr_face * A_hi / V when upwinding from right.
        double vf_lo = 0.0, vf_hi = 0.0;
        if (i > 0) {
            double rl = fmax(rho[d_idx(i-1,j,nt,ng)], 1e-20);
            vf_lo = 0.5 * (mr[d_idx(i-1,j,nt,ng)] / rl + vr_c);
        }
        if (i < nr - 1) {
            double rr = fmax(rho[d_idx(i+1,j,nt,ng)], 1e-20);
            vf_hi = 0.5 * (vr_c + mr[d_idx(i+1,j,nt,ng)] / rr);
        }

        // Lower block (coupling to i-1): from upwind flux at face (i-1,i)
        double* Lb = &L[i * 16];
        for (int q = 0; q < 16; q++) Lb[q] = 0.0;
        if (i > 0) {
            double coeff = (vf_lo >= 0.0 ? vf_lo : 0.0) * Ar_lo * invV;
            // Advection: dR_c/dU_{i-1} contributes +coeff to diagonal entries
            // (upwind from left means flux carries q_{i-1})
            Lb[0] = coeff;   // dR_rho/d(rho_{i-1})
            Lb[5] = coeff;   // dR_mr/d(mr_{i-1})
            Lb[10] = coeff;  // dR_mt/d(mt_{i-1})
            Lb[15] = coeff;  // dR_rhoE/d(rhoE_{i-1})
        }

        // Upper block (coupling to i+1)
        double* Ub = &U[i * 16];
        for (int q = 0; q < 16; q++) Ub[q] = 0.0;
        if (i < nr - 1) {
            double coeff = (vf_hi < 0.0 ? -vf_hi : 0.0) * Ar_hi * invV;
            Ub[0] = coeff;
            Ub[5] = coeff;
            Ub[10] = coeff;
            Ub[15] = coeff;

            // Pressure coupling to i+1: dR_mr/d(rhoE_{i+1}) from ∇P
            double dh = r_center[i+1] - r_center[i];
            double dl = (i > 0) ? r_center[i] - r_center[i-1] : dh;
            double pcoeff = -(gamma - 1.0) * dl / (dh * (dl + dh));
            Ub[1*4+3] += pcoeff;  // dF_mr/d(rhoE_{i+1})
        }

        // Diagonal block: full 4×4 from block Jacobi + corrections
        double* Db = &D[i * 16];
        // Start from the block Jacobi diagonal already computed in d_blk_diag
        // BUT d_blk_diag stores J⁻¹, not J. We need J itself here.
        // Recompute J diagonal inline (same as k_lm_assemble_blkjac but not inverted).

        for (int q = 0; q < 16; q++) Db[q] = 0.0;
        Db[0] = diag_val;   // dF_rho/d(rho)
        Db[5] = diag_val;   // dF_mr/d(mr)
        Db[10] = diag_val;  // dF_mt/d(mt)
        Db[15] = diag_val;  // dF_rhoE/d(rhoE)

        // Pressure-energy coupling
        if (i > 0 && i < nr-1) {
            double dl = r_center[i]-r_center[i-1], dh = r_center[i+1]-r_center[i];
            double pcoeff = -(gamma-1.0) * (-(dh/(dl*(dl+dh)) + dl/(dh*(dl+dh))));
            Db[1*4+3] = pcoeff;
        } else if (i == 0 && nr > 1) {
            Db[1*4+3] = (gamma-1.0) / (r_center[1]-r_center[0]);
        } else if (i == nr-1 && nr >= 2) {
            Db[1*4+3] = -(gamma-1.0) / (r_center[nr-1]-r_center[nr-2]);
        }

        // 1D gravity-density coupling: dF_mr/d(rho) += g₀(r)
        Db[1*4+0] += gr0[i];

        // Lower block pressure coupling to i-1
        if (i > 0) {
            double dl = r_center[i] - r_center[i-1];
            double dh = (i < nr-1) ? r_center[i+1] - r_center[i] : dl;
            double pcoeff = -(gamma - 1.0) * dh / (dl * (dl + dh));
            Lb[1*4+3] += pcoeff;  // dF_mr/d(rhoE_{i-1})
        }

        // Advection correction to diagonal from self-flux
        double coeff_lo_self = (vf_lo < 0.0 ? -vf_lo : 0.0) * Ar_lo * invV;
        double coeff_hi_self = (vf_hi >= 0.0 ? vf_hi : 0.0) * Ar_hi * invV;
        double adv_self = -(coeff_lo_self + coeff_hi_self);
        Db[0] += adv_self;
        Db[5] += adv_self;
        Db[10] += adv_self;
        Db[15] += adv_self;

        // Compression work coupling: dF_rhoE/d(mr), dF_rhoE/d(mt)
        double ddivv_dmr = invV * 0.5/rho_c * (Ar_hi + Ar_lo);
        double At_hi_l = at[i*(nt+1)+j+1], At_lo_l = at[i*(nt+1)+j];
        double ddivv_dmt = invV * 0.5/rho_c * (At_hi_l + At_lo_l);
        Db[3*4+1] = -P_c * ddivv_dmr;
        Db[3*4+2] = -P_c * ddivv_dmt;

        // RHS: input vector
        rhs_s[i*4+0] = v_in[flat];
        rhs_s[i*4+1] = v_in[n + flat];
        rhs_s[i*4+2] = v_in[2*n + flat];
        rhs_s[i*4+3] = v_in[3*n + flat];
    }
    __syncthreads();

    // Step 2: Block Thomas forward sweep (sequential, thread 0 only)
    if (tid == 0) {
        // Forward elimination
        for (int i = 0; i < nr; i++) {
            double* Db = &D[i*16];
            double* rb = &rhs_s[i*4];

            if (i > 0) {
                double* Lb = &L[i*16];
                double* Dp = &D[(i-1)*16]; // D_{i-1} after elimination = upper triangular

                // Compute multiplier M = L_i * D_{i-1}^{-1}
                // Then D_i -= M * U_{i-1}, rhs_i -= M * rhs_{i-1}
                // For simplicity, solve D_{i-1} * X = L_i^T column by column
                // This is expensive but nr is small (~64)

                double* Up = &U[(i-1)*16];

                // M = L_i * inv(D_{i-1}) — compute row by row
                // Since D_{i-1} has been modified by forward elim, it's upper triangular
                // Use back-substitution to get inv(D)*L^T columns

                // Actually for a 4×4 system, just do the standard block Thomas:
                // Solve D_{i-1} * tmp = U_{i-1} for each column → already done
                // Instead, compute M = L_i * D_{i-1}^{-1} by solving D_{i-1}^T * M^T = L_i^T

                // Simplest approach: invert D_{i-1} explicitly (4×4 Gauss-Jordan)
                double inv_D[16];
                {
                    double A[16];
                    for(int q=0;q<16;q++) A[q]=Dp[q];
                    for(int q=0;q<16;q++) inv_D[q]=(q/4==q%4)?1.0:0.0;
                    for(int col=0;col<4;col++){
                        double mx=fabs(A[col*4+col]); int mi=col;
                        for(int row=col+1;row<4;row++){if(fabs(A[row*4+col])>mx){mx=fabs(A[row*4+col]);mi=row;}}
                        if(mi!=col){
                            for(int q=0;q<4;q++){double t=A[col*4+q];A[col*4+q]=A[mi*4+q];A[mi*4+q]=t;}
                            for(int q=0;q<4;q++){double t=inv_D[col*4+q];inv_D[col*4+q]=inv_D[mi*4+q];inv_D[mi*4+q]=t;}
                        }
                        double d=A[col*4+col]; if(fabs(d)<1e-30) d=1e-30;
                        for(int row=col+1;row<4;row++){
                            double m=A[row*4+col]/d;
                            for(int q=col;q<4;q++) A[row*4+q]-=m*A[col*4+q];
                            for(int q=0;q<4;q++) inv_D[row*4+q]-=m*inv_D[col*4+q];
                        }
                    }
                    for(int c=0;c<4;c++){
                        for(int row=3;row>=0;row--){
                            double s=inv_D[row*4+c];
                            for(int q=row+1;q<4;q++) s-=A[row*4+q]*inv_D[q*4+c];
                            inv_D[row*4+c]=s/A[row*4+row];
                        }
                    }
                }

                // M = L_i * inv(D_{i-1})
                double M[16];
                for(int r=0;r<4;r++)
                    for(int c=0;c<4;c++){
                        double s=0;
                        for(int q=0;q<4;q++) s+=Lb[r*4+q]*inv_D[q*4+c];
                        M[r*4+c]=s;
                    }

                // D_i -= M * U_{i-1}
                for(int r=0;r<4;r++)
                    for(int c=0;c<4;c++){
                        double s=0;
                        for(int q=0;q<4;q++) s+=M[r*4+q]*Up[q*4+c];
                        Db[r*4+c]-=s;
                    }

                // rhs_i -= M * rhs_{i-1}
                double* rp = &rhs_s[(i-1)*4];
                for(int r=0;r<4;r++){
                    double s=0;
                    for(int q=0;q<4;q++) s+=M[r*4+q]*rp[q];
                    rb[r]-=s;
                }
            }
        }

        // Back-substitution
        for (int i = nr-1; i >= 0; i--) {
            double* Db = &D[i*16];
            double* rb = &rhs_s[i*4];
            double* xb = &x_s[i*4];

            // rb -= U_i * x_{i+1}
            if (i < nr-1) {
                double* Ub = &U[i*16];
                double* xp = &x_s[(i+1)*4];
                for(int r=0;r<4;r++){
                    double s=0;
                    for(int q=0;q<4;q++) s+=Ub[r*4+q]*xp[q];
                    rb[r]-=s;
                }
            }

            // Solve D_i * x_i = rb (D_i is modified from forward sweep)
            // Inline 4×4 solve
            double A[16]; for(int q=0;q<16;q++) A[q]=Db[q];
            double b4[4]; for(int q=0;q<4;q++) b4[q]=rb[q];
            for(int col=0;col<4;col++){
                double mx=fabs(A[col*4+col]); int mi=col;
                for(int row=col+1;row<4;row++){if(fabs(A[row*4+col])>mx){mx=fabs(A[row*4+col]);mi=row;}}
                if(mi!=col){
                    for(int q=0;q<4;q++){double t=A[col*4+q];A[col*4+q]=A[mi*4+q];A[mi*4+q]=t;}
                    {double t=b4[col];b4[col]=b4[mi];b4[mi]=t;}
                }
                double d=A[col*4+col]; if(fabs(d)<1e-30) d=1e-30;
                for(int row=col+1;row<4;row++){
                    double m=A[row*4+col]/d;
                    for(int q=col;q<4;q++) A[row*4+q]-=m*A[col*4+q];
                    b4[row]-=m*b4[col];
                }
            }
            for(int row=3;row>=0;row--){
                double s=b4[row];
                for(int q=row+1;q<4;q++) s-=A[row*4+q]*b4[q];
                xb[row]=s/A[row*4+row];
            }
        }
    }
    __syncthreads();

    // Step 3: Write output
    for (int i = tid; i < nr; i += blockDim.x) {
        int flat = i * nt + j;
        Mv_out[flat]       = x_s[i*4+0];
        Mv_out[n + flat]   = x_s[i*4+1];
        Mv_out[2*n + flat] = x_s[i*4+2];
        Mv_out[3*n + flat] = x_s[i*4+3];
    }
}

// θ-direction block-tridiagonal line solve: one block per radial line.
// Mirror of k_lm_line_solve but sweeps along θ at fixed r.
__global__
void k_lm_line_solve_theta(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* vol, const double* ar, const double* at,
    const double* r_center, const double* r_face, const double* theta_face,
    const double* dr, const double* dtheta,
    const double* v_in, double* Mv_out,
    int nr, int nt, int ng, double gamma, double inv_dt) {
    int i = blockIdx.x;  // one block per r-line
    if (i >= nr) return;

    extern __shared__ double smem[];
    double* L = smem;                 // nt * 16
    double* D = L + nt * 16;         // nt * 16
    double* U = D + nt * 16;         // nt * 16
    double* rhs_s = U + nt * 16;     // nt * 4
    double* x_s = rhs_s + nt * 4;    // nt * 4

    int tid = threadIdx.x;
    int n = nr * nt;
    double r = r_center[i];

    for (int j = tid; j < nt; j += blockDim.x) {
        int flat = i * nt + j;
        int k = d_idx(i, j, nt, ng);

        double rho_c = fmax(rho[k], 1e-20);
        double vr_c = mr[k] / rho_c;
        double vt_c = mt[k] / rho_c;
        double P_c = fmax((gamma - 1.0) * rhoE[k], 1e-30);
        double invV = 1.0 / vol[flat];
        double sr_r = fabs(vr_c) / dr[i];

        double At_hi = at[i*(nt+1)+j+1], At_lo = at[i*(nt+1)+j];

        double sr = sr_r + fabs(vt_c) / (r * dtheta[j]);
        double diag_val = -(inv_dt + sr);

        // θ-direction face velocities for upwind
        double vf_lo = 0.0, vf_hi = 0.0;
        if (j > 0) {
            double rl = fmax(rho[d_idx(i,j-1,nt,ng)], 1e-20);
            vf_lo = 0.5 * (mt[d_idx(i,j-1,nt,ng)] / rl + vt_c);
        }
        if (j < nt - 1) {
            double rr = fmax(rho[d_idx(i,j+1,nt,ng)], 1e-20);
            vf_hi = 0.5 * (vt_c + mt[d_idx(i,j+1,nt,ng)] / rr);
        }

        // Lower block (j-1)
        double* Lb = &L[j * 16];
        for (int q = 0; q < 16; q++) Lb[q] = 0.0;
        if (j > 0) {
            double coeff = (vf_lo >= 0.0 ? vf_lo : 0.0) * At_lo * invV;
            Lb[0] = coeff; Lb[5] = coeff; Lb[10] = coeff; Lb[15] = coeff;

            // Pressure coupling: dF_mt/d(rhoE_{j-1}) from (1/r)∂P'/∂θ
            double tc_m = 0.5*(theta_face[j-1]+theta_face[j]);
            double tc_c = 0.5*(theta_face[j]+theta_face[j+1]);
            double dl = tc_c - tc_m;
            double dh = (j < nt-1) ? 0.5*(theta_face[j+1]+theta_face[j+2]) - tc_c : dl;
            double pcoeff = -(gamma - 1.0) * dh / (r * dl * (dl + dh));
            Lb[2*4+3] += pcoeff;
        }

        // Upper block (j+1)
        double* Ub = &U[j * 16];
        for (int q = 0; q < 16; q++) Ub[q] = 0.0;
        if (j < nt - 1) {
            double coeff = (vf_hi < 0.0 ? -vf_hi : 0.0) * At_hi * invV;
            Ub[0] = coeff; Ub[5] = coeff; Ub[10] = coeff; Ub[15] = coeff;

            double tc_c = 0.5*(theta_face[j]+theta_face[j+1]);
            double tc_p = 0.5*(theta_face[j+1]+theta_face[j+2]);
            double dh = tc_p - tc_c;
            double dl = (j > 0) ? tc_c - 0.5*(theta_face[j-1]+theta_face[j]) : dh;
            double pcoeff = -(gamma - 1.0) * dl / (r * dh * (dl + dh));
            Ub[2*4+3] += pcoeff;
        }

        // Diagonal block
        double* Db = &D[j * 16];
        for (int q = 0; q < 16; q++) Db[q] = 0.0;
        Db[0] = diag_val; Db[5] = diag_val; Db[10] = diag_val; Db[15] = diag_val;

        // θ pressure gradient diagonal: d(∂P/∂θ)/dP_j
        if (j > 0 && j < nt-1) {
            double tc_m = 0.5*(theta_face[j-1]+theta_face[j]);
            double tc_c = 0.5*(theta_face[j]+theta_face[j+1]);
            double tc_p = 0.5*(theta_face[j+1]+theta_face[j+2]);
            double dl = tc_c-tc_m, dh = tc_p-tc_c;
            double pcoeff = -(gamma-1.0) * (-(dh/(dl*(dl+dh)) + dl/(dh*(dl+dh)))) / r;
            Db[2*4+3] = pcoeff;
        }

        // Advection self-coupling
        double coeff_lo_self = (vf_lo < 0.0 ? -vf_lo : 0.0) * At_lo * invV;
        double coeff_hi_self = (vf_hi >= 0.0 ? vf_hi : 0.0) * At_hi * invV;
        double adv_self = -(coeff_lo_self + coeff_hi_self);
        Db[0] += adv_self; Db[5] += adv_self; Db[10] += adv_self; Db[15] += adv_self;

        // Compression work coupling
        double Ar_hi_l = ar[(i+1)*nt+j], Ar_lo_l = ar[i*nt+j];
        double ddivv_dmr = invV * 0.5/rho_c * (Ar_hi_l + Ar_lo_l);
        double ddivv_dmt = invV * 0.5/rho_c * (At_hi + At_lo);
        Db[3*4+1] = -P_c * ddivv_dmr;
        Db[3*4+2] = -P_c * ddivv_dmt;

        rhs_s[j*4+0] = v_in[flat];
        rhs_s[j*4+1] = v_in[n + flat];
        rhs_s[j*4+2] = v_in[2*n + flat];
        rhs_s[j*4+3] = v_in[3*n + flat];
    }
    __syncthreads();

    // Block Thomas: forward sweep + back-substitution (thread 0)
    if (tid == 0) {
        for (int j = 0; j < nt; j++) {
            double* Db = &D[j*16];
            double* rb = &rhs_s[j*4];
            if (j > 0) {
                double* Lb = &L[j*16];
                double* Dp = &D[(j-1)*16];
                double* Up = &U[(j-1)*16];
                // inv(D_{j-1})
                double inv_D[16], A[16];
                for(int q=0;q<16;q++) A[q]=Dp[q];
                for(int q=0;q<16;q++) inv_D[q]=(q/4==q%4)?1.0:0.0;
                for(int col=0;col<4;col++){
                    double mx=fabs(A[col*4+col]); int mi=col;
                    for(int row=col+1;row<4;row++){if(fabs(A[row*4+col])>mx){mx=fabs(A[row*4+col]);mi=row;}}
                    if(mi!=col){
                        for(int q=0;q<4;q++){double t=A[col*4+q];A[col*4+q]=A[mi*4+q];A[mi*4+q]=t;}
                        for(int q=0;q<4;q++){double t=inv_D[col*4+q];inv_D[col*4+q]=inv_D[mi*4+q];inv_D[mi*4+q]=t;}
                    }
                    double d=A[col*4+col]; if(fabs(d)<1e-30) d=1e-30;
                    for(int row=col+1;row<4;row++){
                        double m=A[row*4+col]/d;
                        for(int q=col;q<4;q++) A[row*4+q]-=m*A[col*4+q];
                        for(int q=0;q<4;q++) inv_D[row*4+q]-=m*inv_D[col*4+q];
                    }
                }
                for(int c=0;c<4;c++){
                    for(int row=3;row>=0;row--){
                        double s=inv_D[row*4+c];
                        for(int q=row+1;q<4;q++) s-=A[row*4+q]*inv_D[q*4+c];
                        inv_D[row*4+c]=s/A[row*4+row];
                    }
                }
                double M[16];
                for(int rr=0;rr<4;rr++)
                    for(int c=0;c<4;c++){
                        double s=0;
                        for(int q=0;q<4;q++) s+=Lb[rr*4+q]*inv_D[q*4+c];
                        M[rr*4+c]=s;
                    }
                for(int rr=0;rr<4;rr++)
                    for(int c=0;c<4;c++){
                        double s=0;
                        for(int q=0;q<4;q++) s+=M[rr*4+q]*Up[q*4+c];
                        Db[rr*4+c]-=s;
                    }
                double* rp = &rhs_s[(j-1)*4];
                for(int rr=0;rr<4;rr++){
                    double s=0;
                    for(int q=0;q<4;q++) s+=M[rr*4+q]*rp[q];
                    rb[rr]-=s;
                }
            }
        }
        for (int j = nt-1; j >= 0; j--) {
            double* Db = &D[j*16];
            double* rb = &rhs_s[j*4];
            double* xb = &x_s[j*4];
            if (j < nt-1) {
                double* Ub = &U[j*16];
                double* xp = &x_s[(j+1)*4];
                for(int rr=0;rr<4;rr++){
                    double s=0;
                    for(int q=0;q<4;q++) s+=Ub[rr*4+q]*xp[q];
                    rb[rr]-=s;
                }
            }
            double AA[16]; for(int q=0;q<16;q++) AA[q]=Db[q];
            double b4[4]; for(int q=0;q<4;q++) b4[q]=rb[q];
            for(int col=0;col<4;col++){
                double mx=fabs(AA[col*4+col]); int mi=col;
                for(int row=col+1;row<4;row++){if(fabs(AA[row*4+col])>mx){mx=fabs(AA[row*4+col]);mi=row;}}
                if(mi!=col){
                    for(int q=0;q<4;q++){double t=AA[col*4+q];AA[col*4+q]=AA[mi*4+q];AA[mi*4+q]=t;}
                    {double t=b4[col];b4[col]=b4[mi];b4[mi]=t;}
                }
                double d=AA[col*4+col]; if(fabs(d)<1e-30) d=1e-30;
                for(int row=col+1;row<4;row++){
                    double m=AA[row*4+col]/d;
                    for(int q=col;q<4;q++) AA[row*4+q]-=m*AA[col*4+q];
                    b4[row]-=m*b4[col];
                }
            }
            for(int row=3;row>=0;row--){
                double s=b4[row];
                for(int q=row+1;q<4;q++) s-=AA[row*4+q]*b4[q];
                xb[row]=s/AA[row*4+row];
            }
        }
    }
    __syncthreads();

    for (int j = tid; j < nt; j += blockDim.x) {
        int flat = i * nt + j;
        Mv_out[flat]       = x_s[j*4+0];
        Mv_out[n + flat]   = x_s[j*4+1];
        Mv_out[2*n + flat] = x_s[j*4+2];
        Mv_out[3*n + flat] = x_s[j*4+3];
    }
}

// ========================= 2-DOF momentum-only r-line solve ===========
// Solves only the (mr, mt) block along r, WITHOUT pressure coupling.
// 2×2 blocks: diagonal = -(1/dt + sr), off-diagonal = upwind advection.
// Pressure gradient is excluded — it enters through the Schur complement.
// One GPU block per θ-line.
__global__
void k_lm_mom_line_solve(
    const double* rho, const double* mr, const double* mt,
    const double* vol, const double* ar,
    const double* r_center, const double* dr, const double* dtheta,
    const double* v_in_mr, const double* v_in_mt,
    double* out_vr, double* out_vt,
    int nr, int nt, int ng, double inv_dt) {
    int j = blockIdx.x;
    if (j >= nt) return;

    extern __shared__ double smem[];
    // 2×2 blocks: L[nr*4], D[nr*4], U[nr*4], rhs[nr*2], x[nr*2]
    double* L = smem;
    double* D = L + nr * 4;
    double* U = D + nr * 4;
    double* rhs_s = U + nr * 4;
    double* x_s = rhs_s + nr * 2;

    int tid = threadIdx.x;
    int n = nr * nt;

    for (int i = tid; i < nr; i += blockDim.x) {
        int flat = i * nt + j;
        int k = d_idx(i, j, nt, ng);
        double rho_c = fmax(rho[k], 1e-20);
        double vr_c = mr[k] / rho_c;
        double vt_c = mt[k] / rho_c;
        double r = r_center[i];
        double invV = 1.0 / vol[flat];

        double sr = fabs(vr_c)/dr[i] + fabs(vt_c)/(r*dtheta[j]);
        double Ar_hi = ar[(i+1)*nt+j], Ar_lo = ar[i*nt+j];

        double vf_lo = 0.0, vf_hi = 0.0;
        if (i > 0) {
            double rl = fmax(rho[d_idx(i-1,j,nt,ng)], 1e-20);
            vf_lo = 0.5 * (mr[d_idx(i-1,j,nt,ng)] / rl + vr_c);
        }
        if (i < nr - 1) {
            double rr = fmax(rho[d_idx(i+1,j,nt,ng)], 1e-20);
            vf_hi = 0.5 * (vr_c + mr[d_idx(i+1,j,nt,ng)] / rr);
        }

        // Lower 2×2
        double* Lb = &L[i * 4];
        Lb[0] = Lb[1] = Lb[2] = Lb[3] = 0.0;
        if (i > 0) {
            double coeff = (vf_lo >= 0.0 ? vf_lo : 0.0) * Ar_lo * invV;
            Lb[0] = coeff; Lb[3] = coeff;
        }

        // Upper 2×2
        double* Ub = &U[i * 4];
        Ub[0] = Ub[1] = Ub[2] = Ub[3] = 0.0;
        if (i < nr - 1) {
            double coeff = (vf_hi < 0.0 ? -vf_hi : 0.0) * Ar_hi * invV;
            Ub[0] = coeff; Ub[3] = coeff;
        }

        // Diagonal 2×2
        double* Db = &D[i * 4];
        double diag_val = -(inv_dt + sr);
        double coeff_lo_self = (vf_lo < 0.0 ? -vf_lo : 0.0) * Ar_lo * invV;
        double coeff_hi_self = (vf_hi >= 0.0 ? vf_hi : 0.0) * Ar_hi * invV;
        double adv_self = -(coeff_lo_self + coeff_hi_self);
        Db[0] = diag_val + adv_self;
        Db[1] = 0.0;
        Db[2] = 0.0;
        Db[3] = diag_val + adv_self;

        // RHS: input residual for mr, mt
        rhs_s[i*2+0] = v_in_mr[flat];
        rhs_s[i*2+1] = v_in_mt[flat];
    }
    __syncthreads();

    // 2×2 block Thomas (thread 0)
    if (tid == 0) {
        for (int i = 0; i < nr; i++) {
            double* Db = &D[i*4];
            double* rb = &rhs_s[i*2];
            if (i > 0) {
                double* Lb = &L[i*4];
                double* Dp = &D[(i-1)*4];
                double* Up = &U[(i-1)*4];
                // inv(D_{i-1}) for 2×2
                double det = Dp[0]*Dp[3] - Dp[1]*Dp[2];
                if (fabs(det) < 1e-30) det = 1e-30;
                double id = 1.0/det;
                double inv_D[4] = {Dp[3]*id, -Dp[1]*id, -Dp[2]*id, Dp[0]*id};
                // M = L * inv(D)
                double M[4];
                M[0] = Lb[0]*inv_D[0] + Lb[1]*inv_D[2];
                M[1] = Lb[0]*inv_D[1] + Lb[1]*inv_D[3];
                M[2] = Lb[2]*inv_D[0] + Lb[3]*inv_D[2];
                M[3] = Lb[2]*inv_D[1] + Lb[3]*inv_D[3];
                // D -= M * U
                Db[0] -= M[0]*Up[0] + M[1]*Up[2];
                Db[1] -= M[0]*Up[1] + M[1]*Up[3];
                Db[2] -= M[2]*Up[0] + M[3]*Up[2];
                Db[3] -= M[2]*Up[1] + M[3]*Up[3];
                // rhs -= M * rhs_{i-1}
                double* rp = &rhs_s[(i-1)*2];
                rb[0] -= M[0]*rp[0] + M[1]*rp[1];
                rb[1] -= M[2]*rp[0] + M[3]*rp[1];
            }
        }
        for (int i = nr-1; i >= 0; i--) {
            double* Db = &D[i*4];
            double* rb = &rhs_s[i*2];
            double* xb = &x_s[i*2];
            if (i < nr-1) {
                double* Ub = &U[i*4];
                double* xp = &x_s[(i+1)*2];
                rb[0] -= Ub[0]*xp[0] + Ub[1]*xp[1];
                rb[1] -= Ub[2]*xp[0] + Ub[3]*xp[1];
            }
            double det = Db[0]*Db[3] - Db[1]*Db[2];
            if (fabs(det) < 1e-30) det = 1e-30;
            double id = 1.0/det;
            xb[0] = (Db[3]*rb[0] - Db[1]*rb[1]) * id;
            xb[1] = (Db[0]*rb[1] - Db[2]*rb[0]) * id;
        }
    }
    __syncthreads();

    // Output: predicted velocity δvr, δvt (divide momentum correction by ρ)
    for (int i = tid; i < nr; i += blockDim.x) {
        int flat = i * nt + j;
        int k = d_idx(i, j, nt, ng);
        double rho_c = fmax(rho[k], 1e-20);
        out_vr[flat] = x_s[i*2+0] / rho_c;
        out_vt[flat] = x_s[i*2+1] / rho_c;
    }
}

// ========================= PBP assembly kernel =======================
// Assemble full 4-DOF output from PBP components:
// out[ρ]  = r_ρ / A_ρ  (point Jacobi on continuity)
// out[mr] = ρ·(ṽr - (1/Ap)·∂δp/∂r)
// out[mt] = ρ·(ṽt - (1/Ap)·(1/r)·∂δp/∂θ)
// out[E]  = r_E / A_E  (point Jacobi on energy)
__global__
void k_lm_pbp_assemble(
    const double* rho_state, const double* blk_diag,
    const double* v_in,      // original 4n RHS
    const double* vr_pred, const double* vt_pred,  // predicted velocities
    const double* dp, const double* Ap,
    const double* r_center, const double* theta_face,
    double* Mv_out,
    int nr, int nt, int ng, double gamma) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int n = nr*nt;
    int i = flat/nt, j = flat%nt;
    int k = d_idx(i, j, nt, ng);
    double rho_c = fmax(rho_state[k], 1e-20);
    double r = r_center[i];

    // ρ: point Jacobi — use (0,0) element of block inverse
    const double* B = &blk_diag[flat*16];
    Mv_out[flat] = B[0] * v_in[flat];

    // Pressure gradient correction
    double dp_dr = 0.0;
    if (i > 0 && i < nr-1) {
        double dl=r_center[i]-r_center[i-1], dh=r_center[i+1]-r_center[i];
        dp_dr = (dh*(dp[flat]-dp[(i-1)*nt+j])/dl + dl*(dp[(i+1)*nt+j]-dp[flat])/dh)/(dl+dh);
    } else if (i==0 && nr>1)
        dp_dr = (dp[1*nt+j]-dp[0])/(r_center[1]-r_center[0]);
    else if (i==nr-1 && nr>=2)
        dp_dr = (dp[(nr-1)*nt+j]-dp[(nr-2)*nt+j])/(r_center[nr-1]-r_center[nr-2]);

    double dp_dt_r = 0.0;
    if (j > 0 && j < nt-1) {
        double tc_m=0.5*(theta_face[j-1]+theta_face[j]), tc_c=0.5*(theta_face[j]+theta_face[j+1]);
        double tc_p=0.5*(theta_face[j+1]+theta_face[j+2]);
        double dl=tc_c-tc_m, dh=tc_p-tc_c;
        dp_dt_r = (dh*(dp[flat]-dp[i*nt+j-1])/dl + dl*(dp[i*nt+j+1]-dp[flat])/dh)/(r*(dl+dh));
    }

    double inv_ap = 1.0 / Ap[flat];
    double dvr = vr_pred[flat] - inv_ap * dp_dr;
    double dvt = vt_pred[flat] - inv_ap * dp_dt_r;

    // Convert corrected velocity back to momentum
    Mv_out[n + flat]   = rho_c * dvr;
    Mv_out[2*n + flat] = rho_c * dvt;

    // Energy: point Jacobi — use (3,3) element of block inverse
    Mv_out[3*n + flat] = B[15] * v_in[3*n + flat];
}

// Velocity-only pressure correction: vr -= (1/Ap)*∂δp/∂r, vt -= (1/Ap)*(1/r)*∂δp/∂θ
__global__
void k_lm_simple_vcorr(double* vr_io, double* vt_io,
                       const double* dp, const double* Ap,
                       const double* rc, const double* theta_face,
                       int nr, int nt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;

    double dp_dr = 0.0;
    if (i > 0 && i < nr-1) {
        double dl=rc[i]-rc[i-1], dh=rc[i+1]-rc[i];
        dp_dr = (dh*(dp[flat]-dp[(i-1)*nt+j])/dl + dl*(dp[(i+1)*nt+j]-dp[flat])/dh)/(dl+dh);
    } else if (i==0 && nr>1)
        dp_dr = (dp[1*nt+j]-dp[0])/(rc[1]-rc[0]);
    else if (i==nr-1 && nr>=2)
        dp_dr = (dp[(nr-1)*nt+j]-dp[(nr-2)*nt+j])/(rc[nr-1]-rc[nr-2]);

    double dp_dt_r = 0.0;
    if (j > 0 && j < nt-1) {
        double tc_m=0.5*(theta_face[j-1]+theta_face[j]), tc_c=0.5*(theta_face[j]+theta_face[j+1]);
        double tc_p=0.5*(theta_face[j+1]+theta_face[j+2]);
        double dl=tc_c-tc_m, dh=tc_p-tc_c;
        dp_dt_r = (dh*(dp[flat]-dp[i*nt+j-1])/dl + dl*(dp[i*nt+j+1]-dp[flat])/dh)/(rc[i]*(dl+dh));
    }

    double inv_ap = 1.0 / Ap[flat];
    vr_io[flat] -= inv_ap * dp_dr;
    vt_io[flat] -= inv_ap * dp_dt_r;
}

void LowMachSolver::assemble_simple(double dt) {
    int n = nr*nt, B = 256;
    k_lm_simple_mom_diag<<<(n+B-1)/B,B>>>(
        d_rho, d_mr, d_mtheta, d_dr, d_r_center, d_dtheta,
        d_Ap, nr, nt, ng, 1.0/dt);
}

void LowMachSolver::apply_simple(const double* d_v, double* d_Mv, double dt) {
    int n = nr*nt, B = 256;

    // Step 1: v* = r_momentum / Ap
    k_lm_simple_vstar<<<(n+B-1)/B,B>>>(
        d_v + n, d_v + 2*n, d_Ap, d_simple_vr_s, d_simple_vt_s, n);

    // Step 2: div(v*)
    k_lm_simple_div<<<(n+B-1)/B,B>>>(
        d_simple_vr_s, d_simple_vt_s,
        d_cell_volume, d_area_r, d_area_theta,
        d_simple_div, nr, nt);

    // Compute α = 1/Ap for variable-coefficient Poisson
    k_lm_simple_alpha<<<(n+B-1)/B,B>>>(d_Ap, d_inv_rho, n);

    // Poisson RHS = div(v*), Dirichlet δp=0 at boundary
    k_lm_simple_prhs<<<(n+B-1)/B,B>>>(d_simple_div, d_rhs_poisson, nr, nt);

    // Solve ∇·(α∇δp) = div(v*), where α = 1/Ap
    CUDA_CHECK(cudaMemset(d_simple_p, 0, n*sizeof(double)));
    gmg_pressure.solve_varcoeff(d_inv_rho, d_rhs_poisson, d_simple_p, 10, 1e-3);

    // Step 3: correct velocities with (1/Ap)∇δp, assemble full output
    k_lm_simple_correct<<<(n+B-1)/B,B>>>(
        d_v, d_v + n, d_v + 2*n, d_v + 3*n,
        d_Ap,
        d_simple_vr_s, d_simple_vt_s,
        d_simple_p,
        d_r_center, d_theta_face,
        d_Mv,
        nr, nt, dt);
}

// Assemble Schur complement σ(x) = 4πGρ / Ap
// where Ap = 1/dt + upwind_rate (from block Jacobi diagonal)
__global__
void k_lm_assemble_sigma(const double* rho, const double* blk_diag,
                         double* sigma, int n, int ng, int nt, double G) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= n) return;
    double rho_c = fmax(rho[d_idx(flat/nt, flat%nt, nt, ng)], 1e-20);
    // Ap ≈ |J_{rho,rho}| = |blk_diag inverse's (0,0) entry|
    // But we need J_diag not J⁻¹. The block Jacobi diagonal value was -(1/dt+sr).
    // We stored the INVERSE. Extract Ap from the (0,0) element of blk inverse:
    //   blk_inv[0][0] ≈ -1/Ap for diagonally-dominant system.
    // So Ap ≈ -1/blk_inv[0][0].
    double inv00 = blk_diag[flat * 16]; // (0,0) element of J⁻¹
    double Ap = (fabs(inv00) > 1e-30) ? fabs(1.0 / inv00) : 1e30;
    sigma[flat] = 4.0 * M_PI * G * rho_c / Ap;
}

// Precompute S(x) = max(|cC|, 1) for Poisson residual scaling.
// Matches the stencil in k_lm_poisson_residual exactly.
__global__
void k_lm_poisson_scale(
    const double* r_face, const double* r_center, const double* dr,
    const double* dtheta,
    const double* sin_tf, const double* sin_tc,
    double* scale_out,
    int nr, int nt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;

    if (i == nr-1) { scale_out[flat] = 1.0; return; }

    double ri = r_center[i], ri2 = ri*ri, dri = dr[i];
    double cC = 0.0;
    if (i > 0) {
        double rl = r_face[i];
        cC -= rl*rl / (ri2 * dri * (r_center[i] - r_center[i-1]));
    }
    if (i < nr-1) {
        double rh = r_face[i+1];
        cC -= rh*rh / (ri2 * dri * (r_center[i+1] - r_center[i]));
    }
    double sj = sin_tc[j], dtj = dtheta[j];
    if (j > 0) cC -= sin_tf[j] / (ri2 * sj * dtj * 0.5 * (dtheta[j] + dtheta[j-1]));
    if (j < nt-1) cC -= sin_tf[j+1] / (ri2 * sj * dtj * 0.5 * (dtheta[j] + dtheta[j+1]));

    scale_out[flat] = fmax(fabs(cC), 1.0);
}

// Elementwise: out[i] = a[i] * b[i]
__global__
void k_lm_mul(const double* a, const double* b, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

// Elementwise: out[i] = a[i] / b[i]
__global__
void k_lm_div(const double* a, const double* b, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] / b[i];
}

void LowMachSolver::assemble_schur_sigma(double dt) {
    int n = nr*nt, B = 256;
    k_lm_assemble_sigma<<<(n+B-1)/B,B>>>(
        d_rho, d_blk_diag, d_sigma_schur, n, ng, nt, G_const);
    k_lm_poisson_scale<<<(n+B-1)/B,B>>>(
        d_r_face, d_r_center, d_dr, d_dtheta,
        d_sin_theta_face, d_sin_theta_center,
        d_poisson_scale, nr, nt);
}

void LowMachSolver::apply_block_schur(const double* d_v, double* d_Mv, double dt) {
    int n = nr*nt, B = 256;

    // Step 1: Fluid part — block Jacobi on first 4N
    k_lm_apply_blkjac<<<(n+B-1)/B,B>>>(d_blk_diag, d_v, d_Mv, n);

    // Step 2: Build Schur RHS in physical (unscaled) space.
    // v₅ is scaled: v₅_scaled = (∇²Φ - 4πGρ)/S
    // Need physical: v₅_phys = v₅_scaled * S
    // J_Φρ = -4πG, so coupling term = (-4πG)·δρ
    // But δρ comes from block Jacobi applied to *scaled* input, so it's physical.
    // RHS = v₅_phys - J_Φρ·δρ = S·v₅ + 4πG·δρ
    k_lm_mul<<<(n+B-1)/B,B>>>(d_v + 4*n, d_poisson_scale, d_schur_rhs, n);
    k_lm_axpy<<<(n+B-1)/B,B>>>(d_schur_rhs, 4.0*M_PI*G_const, d_Mv, n);

    // Step 3: Solve (∇² - σ)δΦ = RHS with Helmholtz GMG
    CUDA_CHECK(cudaMemset(d_Mv + 4*n, 0, n*sizeof(double)));
    gmg_schur.solve_helmholtz(d_sigma_schur, d_schur_rhs, d_Mv + 4*n, 10, 1e-4);
}

void LowMachSolver::apply_preconditioner(const double* d_v, double* d_Mv, double dt) {
    int n = nr*nt, B = 256;

    if (precond_type == PrecondType::PBP) {
        // Physics-Based Preconditioning (PBP):
        //   1. Momentum predict: 2-DOF r-line solve (no pressure) → ṽr, ṽt
        //   2. Pressure Poisson: ∇·(α∇δp) = div(ṽ), α=(γ-1)/Ap, via GMG
        //   3. Velocity correct + assemble all 4 DOFs
        double inv_dt = 1.0 / dt;

        // Step 1: Solve A_v · δv = r_v (momentum-only, 2-DOF per cell)
        // Input: d_v[n..2n] = r_mr, d_v[2n..3n] = r_mt (momentum residual)
        // Output: d_simple_vr_s, d_simple_vt_s = predicted velocity corrections
        {
            int smem = nr * (3*4 + 2*2) * sizeof(double);
            int threads = std::min(nr, 256);
            k_lm_mom_line_solve<<<nt, threads, smem>>>(
                d_rho, d_mr, d_mtheta,
                d_cell_volume, d_area_r,
                d_r_center, d_dr, d_dtheta,
                d_v + n, d_v + 2*n,
                d_simple_vr_s, d_simple_vt_s,
                nr, nt, ng, inv_dt);
        }

        // Step 2: Divergence of predicted velocity
        k_lm_simple_div<<<(n+B-1)/B,B>>>(
            d_simple_vr_s, d_simple_vt_s,
            d_cell_volume, d_area_r, d_area_theta,
            d_simple_div, nr, nt);

        // α = (γ-1)/Ap for Schur complement ≈ ∇·(α∇)
        k_lm_simple_alpha<<<(n+B-1)/B,B>>>(d_Ap, d_inv_rho, n);

        k_lm_simple_prhs<<<(n+B-1)/B,B>>>(d_simple_div, d_rhs_poisson, nr, nt);

        // Solve pressure Poisson (3 V-cycles, approximate is fine for preconditioner)
        CUDA_CHECK(cudaMemset(d_simple_p, 0, n*sizeof(double)));
        gmg_pressure.solve_varcoeff(d_inv_rho, d_rhs_poisson, d_simple_p, 3, 1e-2);

        // Step 3: Assemble all 4 DOFs
        k_lm_pbp_assemble<<<(n+B-1)/B,B>>>(
            d_rho, d_blk_diag,
            d_v,
            d_simple_vr_s, d_simple_vt_s,
            d_simple_p, d_Ap,
            d_r_center, d_theta_face,
            d_Mv,
            nr, nt, ng, gamma);
    } else if (precond_type == PrecondType::BLOCK_SCHUR) {
        apply_block_schur(d_v, d_Mv, dt);
    } else if (precond_type == PrecondType::LINE_JACOBI) {
        int smem = nr * (3*16 + 2*4) * sizeof(double);
        int threads = std::min(nr, 256);
        k_lm_line_solve<<<nt, threads, smem>>>(
            d_rho, d_mr, d_mtheta, d_rhoE,
            d_cell_volume, d_area_r, d_area_theta,
            d_r_center, d_r_face, d_theta_face,
            d_dr, d_dtheta, d_gr0,
            d_v, d_Mv,
            nr, nt, ng, gamma, 1.0/dt);
    } else if (precond_type == PrecondType::SIMPLE) {
        apply_simple(d_v, d_Mv, dt);
    } else if (precond_type == PrecondType::BLOCK_JACOBI) {
        k_lm_apply_blkjac<<<(n+B-1)/B,B>>>(d_blk_diag, d_v, d_Mv, n);
    } else {
        k_lm_copy<<<(4*n+B-1)/B,B>>>(d_Mv, d_v, 4*n);
    }
}

// ========================= JFNK matvec ============================
// Pure 4-DOF: GMRES on fluid (ρ, ρvr, ρvθ, ρe).
// Gravity g(r) is recomputed from ρ in compute_residual, so the
// Jacobian-vector product J·v naturally includes ∂g/∂ρ coupling.

void LowMachSolver::jfnk_matvec(const double* d_v, double* d_Jv, double dt) {
    int n = nr*nt, N4 = 4*n, B = 256;

    double norm_v = sqrt(gpu_dot(d_v, d_v, d_work_a, N4));
    if (norm_v < 1e-30) { k_lm_zero<<<(N4+B-1)/B,B>>>(d_Jv, N4); return; }

    double norm_U = sqrt(gpu_dot(d_Un, d_Un, d_work_a, N4));
    double eps_fd = sqrt(1e-15) * (1.0 + norm_U) / norm_v;

    k_lm_pack<<<(n+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE,d_phi, d_gmres_Uk, nr,nt,ng);

    k_lm_unpack_add<<<(n+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE,d_phi, d_v, eps_fd, nr,nt,ng);
    apply_floor();

    // compute_residual recomputes g(r) from perturbed ρ, so ∂g/∂ρ is captured
    compute_residual(d_residual);
    k_lm_compute_F<<<(n+B-1)/B,B>>>(d_Jv, d_residual,
        d_rho, d_mr, d_mtheta, d_rhoE, d_Un, 1.0/dt, nr, nt, ng);

    k_lm_axpy<<<(N4+B-1)/B,B>>>(d_Jv, -1.0, d_Fk, N4);
    k_lm_scale<<<(N4+B-1)/B,B>>>(d_Jv, 1.0/eps_fd, N4);

    k_lm_unpack_set<<<(n+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_rhoE,d_phi, d_gmres_Uk, nr,nt,ng);
}

// ========================= FGMRES ================================
// Right-preconditioned FGMRES in physical space.
// GMRES minimizes ||F + J·δU||₂ directly.
// L scaling is used only for convergence check and Newton merit function,
// not inside the Krylov basis — this keeps the preconditioner consistent.

int LowMachSolver::gmres_solve(double* d_x, const double* d_b, double dt,
                                double tol, int max_iter) {
    int n = nr*nt, N = 4*n, B = 256;
    int m = std::min(max_iter, (int)GMRES_RESTART);

    std::vector<double> H((m+1)*m, 0.0);
    std::vector<double> cs(m), sn(m), g(m+1, 0.0);

    k_lm_copy<<<(N+B-1)/B,B>>>(d_gmres_V[0], d_b, N);
    k_lm_scale<<<(N+B-1)/B,B>>>(d_gmres_V[0], -1.0, N);
    k_lm_zero<<<(n+B-1)/B,B>>>(d_gmres_V[0] + N, n);

    double beta = sqrt(gpu_dot(d_gmres_V[0], d_gmres_V[0], d_work_a, N));
    if (beta < 1e-30) return 0;
    k_lm_scale<<<(N+B-1)/B,B>>>(d_gmres_V[0], 1.0/beta, N);
    g[0] = beta;

    int j;
    for (j = 0; j < m; ++j) {
        apply_preconditioner(d_gmres_V[j], d_gmres_Z[j], dt);
        k_lm_zero<<<(n+B-1)/B,B>>>(d_gmres_Z[j] + N, n);

        jfnk_matvec(d_gmres_Z[j], d_gmres_w, dt);
        k_lm_zero<<<(n+B-1)/B,B>>>(d_gmres_w + N, n);

        for (int i = 0; i <= j; ++i) {
            H[i*m+j] = gpu_dot(d_gmres_w, d_gmres_V[i], d_work_a, N);
            k_lm_axpy<<<(N+B-1)/B,B>>>(d_gmres_w, -H[i*m+j], d_gmres_V[i], N);
        }
        H[(j+1)*m+j] = sqrt(gpu_dot(d_gmres_w, d_gmres_w, d_work_a, N));

        if (H[(j+1)*m+j] < 1e-30) { j++; break; }
        k_lm_copy<<<(N+B-1)/B,B>>>(d_gmres_V[j+1], d_gmres_w, N);
        k_lm_scale<<<(N+B-1)/B,B>>>(d_gmres_V[j+1], 1.0/H[(j+1)*m+j], N);

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
              const double* rho0, double* out,
              int nr, int nt, int ng, double atm_thresh) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    // Skip atmosphere cells — they're frozen, their velocity is meaningless
    if (rho0[flat] < atm_thresh) { out[flat] = 1e30; return; }
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
                                d_rho0, d_work_a, nr,nt,ng, atm_rho_thresh);
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
    double dt = std::min({dt_current, dt_cap, t_end - t});

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
            if (step_count < 30 || (step_count < 500 && step_count % 50 == 0))
                std::fprintf(stderr, "  step %d: Newton failed (cut %d), dt -> %.3e\n",
                            step_count, cut, dt);
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
            double Fnorm = sqrt(gpu_dot(d_residual_ls, d_residual_ls, d_work_a, N4));

            if (newton == 0) {
                Fnorm0 = Fnorm;
                if (Fnorm < 1e-30) { converged = true; break; }
            }

            double Fnorm_per_cell = Fnorm / sqrt((double)N4);
            if (Fnorm < 1e-3 * Fnorm0 || Fnorm_per_cell < 1e-4) {
                converged = true;
                newton_iters_used = newton;
                if (step_count < 30 || (step_count < 500 && step_count % 50 == 0))
                    std::fprintf(stderr, "  step %d converged at newton %d: ||F_s||=%.3e per-cell=%.3e (dt=%.3e)\n",
                                step_count, newton, Fnorm, Fnorm_per_cell, dt);
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
                Fnorm_new = sqrt(gpu_dot(d_residual_ls, d_residual_ls, d_work_a, N4));
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

            if (step_count < 30 || (step_count < 500 && step_count % 50 == 0))
                std::fprintf(stderr, "  step %d newton %d: ||F|| %.3e -> %.3e  a=%.2f  eta=%.1e  GMRES %d  dt=%.3e\n",
                            step_count, newton, Fnorm, Fnorm_new, alpha, eta_gmres, gmres_iters, dt);
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
        std::fprintf(stderr,
            "  DIAG step %d (dt=%.3e, dt_good=%.3e): max|F| by eq:\n"
            "    rho: %.3e @cell %d (i=%d,j=%d)\n"
            "    mr:  %.3e @cell %d (i=%d,j=%d)\n"
            "    mt:  %.3e @cell %d (i=%d,j=%d)\n"
            "    rhoE:%.3e @cell %d (i=%d,j=%d)\n",
            step_count, dt, dt_good,
            mx[0], mi[0], mi[0]/nt, mi[0]%nt,
            mx[1], mi[1], mi[1]/nt, mi[1]%nt,
            mx[2], mi[2], mi[2]/nt, mi[2]%nt,
            mx[3], mi[3], mi[3]/nt, mi[3]%nt);
    }

    // Update Φ for output/diagnostics (not used in Newton)
    if (converged)
        solve_gravity();

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
    std::fprintf(stderr, "  HSE snapshot: ρ_max=%.3e, atm_thresh=%.3e\n", rho_max, atm_rho_thresh);

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

    std::fprintf(stderr, "=== HSE Diagnostic (initial residual, 4-DOF + 1D gravity) ===\n");
    std::fprintf(stderr, "  max|R_ρ|    = %.6e\n", max_rho);
    std::fprintf(stderr, "  max|R_ρvr|  = %.6e\n", max_mr);
    std::fprintf(stderr, "  max|R_ρvθ|  = %.6e\n", max_mt);
    std::fprintf(stderr, "  max|R_ρe|   = %.6e\n", max_rhoE);
    std::fprintf(stderr, "  ||R||₂      = %.6e\n", l2);
    std::fprintf(stderr, "  Target: all < 1e-10\n");
    std::fprintf(stderr, "=============================================================\n");
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
