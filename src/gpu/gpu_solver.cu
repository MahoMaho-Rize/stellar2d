// Semi-implicit low-Mach GPU solver for 2D axisymmetric Euler + self-gravity.
//
// Each time step:
//   1) Advect (ρ, ρv, ρE) explicitly with upwind fluxes — NO pressure, NO gravity
//   2) Gravity Poisson solve: ∇²Φ = 4πGρ  (AmgX)
//   3) Pressure from EOS: p = (γ-1)ρe_int
//   4) Combined source: m += -dt(∇p + ρ∇Φ), E += -dt v·(∇p + ρ∇Φ)
//
// Well-balanced: in hydrostatic equilibrium ∇p + ρ∇Φ ≈ 0 → source vanishes.
// CFL on flow velocity only → dt ~ O(1) for near-equilibrium stars.

#include "gpu_solver.h"
#include "../gravity/poisson.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#ifdef USE_AMGX
#include <amgx_c.h>
#endif

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

struct DPrim { double rho, vr, vt, P; };

__device__ __forceinline__
DPrim to_prim(double rho, double mr, double mt, double E, double gamma) {
    DPrim w;
    w.rho = fmax(rho, 1e-20);
    double inv = 1.0 / w.rho;
    w.vr = mr * inv;
    w.vt = mt * inv;
    double ke = 0.5 * (w.vr * w.vr + w.vt * w.vt);
    w.P = fmax((gamma - 1.0) * w.rho * (E * inv - ke), 1e-30);
    return w;
}

// ========================= Ghost cells ============================

__global__ void k_ghost_r_in(double* rho, double* mr, double* mt, double* E,
                              int nr, int nt, int ng) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y + 1;
    if (j >= nt || g > ng) return;
    int kg = d_idx(-g, j, nt, ng), kp = d_idx(g-1, j, nt, ng);
    rho[kg]=rho[kp]; mr[kg]=-mr[kp]; mt[kg]=mt[kp]; E[kg]=E[kp];
}
__global__ void k_ghost_r_out(double* rho, double* mr, double* mt, double* E,
                               int nr, int nt, int ng) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y;
    if (j >= nt || g >= ng) return;
    int kg = d_idx(nr+g, j, nt, ng), kp = d_idx(nr-1, j, nt, ng);
    rho[kg]=rho[kp]; mr[kg]=mr[kp]; mt[kg]=mt[kp]; E[kg]=E[kp];
}
__global__ void k_ghost_t_n(double* rho, double* mr, double* mt, double* E,
                             int nr, int nt, int ng) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y + 1;
    if (ii >= nr+2*ng || g > ng) return;
    int i = ii - ng;
    int kg = d_idx(i,-g,nt,ng), kp = d_idx(i,g-1,nt,ng);
    rho[kg]=rho[kp]; mr[kg]=mr[kp]; mt[kg]=-mt[kp]; E[kg]=E[kp];
}
__global__ void k_ghost_t_s(double* rho, double* mr, double* mt, double* E,
                             int nr, int nt, int ng) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y;
    if (ii >= nr+2*ng || g >= ng) return;
    int i = ii - ng;
    int kg = d_idx(i,nt+g,nt,ng), kp = d_idx(i,nt-1-g,nt,ng);
    rho[kg]=rho[kp]; mr[kg]=mr[kp]; mt[kg]=-mt[kp]; E[kg]=E[kp];
}

static void launch_ghost(double* rho, double* mr, double* mt, double* E,
                         int nr, int nt, int ng) {
    int B=256;
    { dim3 g((nt+B-1)/B, ng);       k_ghost_r_in<<<g,B>>>(rho,mr,mt,E,nr,nt,ng); }
    { dim3 g((nt+B-1)/B, ng);       k_ghost_r_out<<<g,B>>>(rho,mr,mt,E,nr,nt,ng); }
    { dim3 g((nr+2*ng+B-1)/B, ng);  k_ghost_t_n<<<g,B>>>(rho,mr,mt,E,nr,nt,ng); }
    { dim3 g((nr+2*ng+B-1)/B, ng);  k_ghost_t_s<<<g,B>>>(rho,mr,mt,E,nr,nt,ng); }
}

// ========================= Limiter ================================

__device__ __forceinline__
double d_minmod(double a, double b) {
    return (a*b <= 0.0) ? 0.0 : (fabs(a) < fabs(b) ? a : b);
}

__device__ __forceinline__
void d_recon(double vm1, double v0, double vp1, double vp2,
             double& L, double& R) {
    L = v0   + 0.5 * d_minmod(v0 - vm1, vp1 - v0);
    R = vp1  - 0.5 * d_minmod(vp1 - v0, vp2 - vp1);
}

// ========================= Advection flux =========================
// Pure upwind: only ρv, ρv⊗v, (ρE)v — NO pressure term.

__global__
void k_advflux_r(const double* rho, const double* mr, const double* mt,
                 const double* E, double* flux,
                 int nr, int nt, int ng, double gamma) {
    int i = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nt || i > nr) return;

    auto W = [&](int ii, int jj) {
        int k = d_idx(ii,jj,nt,ng);
        return to_prim(rho[k], mr[k], mt[k], E[k], gamma);
    };
    DPrim a=W(i-2,j), b=W(i-1,j), c=W(i,j), d2=W(i+1,j);

    // reconstruct normal velocity for upwinding
    double vL, vR;
    d_recon(a.vr, b.vr, c.vr, d2.vr, vL, vR);
    double s = 0.5*(vL + vR);

    // reconstruct all fields
    double rL,rR, mL,mR, tL,tR, eL,eR;
    d_recon(a.rho,b.rho,c.rho,d2.rho, rL,rR);
    d_recon(a.vr*a.rho, b.vr*b.rho, c.vr*c.rho, d2.vr*d2.rho, mL,mR);
    d_recon(a.vt*a.rho, b.vt*b.rho, c.vt*c.rho, d2.vt*d2.rho, tL,tR);

    // internal energy per unit volume for advection
    double eA = E[d_idx(i-2,j,nt,ng)], eB = E[d_idx(i-1,j,nt,ng)];
    double eC = E[d_idx(i,j,nt,ng)],   eD = E[d_idx(i+1,j,nt,ng)];
    d_recon(eA, eB, eC, eD, eL, eR);

    // upwind selection
    double fr, fmr, fmt, fE;
    if (s >= 0.0) {
        double v = mL / fmax(rL, 1e-30);
        fr = rL * v; fmr = mL * v; fmt = tL * v; fE = eL * v;
    } else {
        double v = mR / fmax(rR, 1e-30);
        fr = rR * v; fmr = mR * v; fmt = tR * v; fE = eR * v;
    }

    int idx = (i*nt + j)*4;
    flux[idx]=fr; flux[idx+1]=fmr; flux[idx+2]=fmt; flux[idx+3]=fE;
}

__global__
void k_advflux_t(const double* rho, const double* mr, const double* mt,
                 const double* E, double* flux,
                 int nr, int nt, int ng, double gamma) {
    int i = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j > nt || i >= nr) return;

    auto W = [&](int ii, int jj) {
        int k = d_idx(ii,jj,nt,ng);
        return to_prim(rho[k], mr[k], mt[k], E[k], gamma);
    };
    DPrim a=W(i,j-2), b=W(i,j-1), c=W(i,j), d2=W(i,j+1);

    double vL, vR;
    d_recon(a.vt, b.vt, c.vt, d2.vt, vL, vR);
    double s = 0.5*(vL + vR);

    double rL,rR, mL,mR, tL,tR, eL,eR;
    d_recon(a.rho,b.rho,c.rho,d2.rho, rL,rR);
    d_recon(a.vr*a.rho, b.vr*b.rho, c.vr*c.rho, d2.vr*d2.rho, mL,mR);
    d_recon(a.vt*a.rho, b.vt*b.rho, c.vt*c.rho, d2.vt*d2.rho, tL,tR);

    double eA = E[d_idx(i,j-2,nt,ng)], eB = E[d_idx(i,j-1,nt,ng)];
    double eC = E[d_idx(i,j,nt,ng)],   eD = E[d_idx(i,j+1,nt,ng)];
    d_recon(eA, eB, eC, eD, eL, eR);

    double fr, fmr, fmt, fE;
    if (s >= 0.0) {
        double v = tL / fmax(rL, 1e-30);
        fr = rL * v; fmr = mL * v; fmt = tL * v; fE = eL * v;
    } else {
        double v = tR / fmax(rR, 1e-30);
        fr = rR * v; fmr = mR * v; fmt = tR * v; fE = eR * v;
    }

    int idx = (i*(nt+1) + j)*4;
    flux[idx]=fr; flux[idx+1]=fmr; flux[idx+2]=fmt; flux[idx+3]=fE;
}

// ========================= Advection update =======================
// Pure advection: flux divergence + geometric source (centrifugal only, no pressure)

__global__
void k_advect(double* rho, double* mr, double* mt, double* E,
              const double* flux_r, const double* flux_t,
              const double* vol, const double* ar, const double* at,
              const double* r_center,
              int nr, int nt, int ng, double gamma, double dt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    int k = d_idx(i,j,nt,ng);
    double invV = 1.0 / vol[flat];

    auto fr = [&](int ii, int jj, int c) { return flux_r[(ii*nt+jj)*4+c]; };
    auto ft = [&](int ii, int jj, int c) { return flux_t[(ii*(nt+1)+jj)*4+c]; };
    double Arh=ar[(i+1)*nt+j], Arl=ar[i*nt+j];
    double Ath=at[i*(nt+1)+j+1], Atl=at[i*(nt+1)+j];

    for (int c = 0; c < 4; ++c) {
        double div = -invV*(Arh*fr(i+1,j,c) - Arl*fr(i,j,c)
                          + Ath*ft(i,j+1,c) - Atl*ft(i,j,c));
        double* f = (c==0)?rho:(c==1)?mr:(c==2)?mt:E;
        f[k] += dt * div;
    }

    // geometric source: only centrifugal (rho*vt^2/r) and Coriolis (-rho*vr*vt/r)
    // NO pressure geometric source — that's part of the implicit step
    DPrim w = to_prim(rho[k], mr[k], mt[k], E[k], gamma);
    double inv_r = 1.0 / r_center[i];
    mr[k] += dt * w.rho * w.vt * w.vt * inv_r;
    mt[k] += dt * (-w.rho * w.vr * w.vt * inv_r);
}

// ========================= Pressure+gravity source ================
// Apply -dt*(∇p + ρ∇Φ) to momentum, and corresponding energy update.
// Well-balanced: in hydrostatic equilibrium ∇p ≈ -ρ∇Φ → source ≈ 0.

__global__
void k_pres_grav_source(double* mr, double* mt, double* E,
                        const double* rho, const double* phi,
                        const double* r_center, const double* r_face,
                        const double* theta_center, const double* theta_face,
                        const double* vol, const double* dr,
                        int nr, int nt, int ng, double gamma, double dt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    int k = d_idx(i,j,nt,ng);

    DPrim w = to_prim(rho[k], mr[k], mt[k], E[k], gamma);
    double r = r_center[i];

    // --- pressure gradient (cell-centered finite difference) ---
    // Use EOS pressure from neighboring cells
    auto get_P = [&](int ii, int jj) -> double {
        int kk = d_idx(ii,jj,nt,ng);
        DPrim ww = to_prim(rho[kk], mr[kk], mt[kk], E[kk], gamma);
        return ww.P;
    };

    double dp_dr = 0.0;
    if (i > 0 && i < nr-1) {
        dp_dr = (get_P(i+1,j) - get_P(i-1,j)) / (r_center[i+1] - r_center[i-1]);
    } else if (i == 0 && nr > 1) {
        dp_dr = (get_P(1,j) - get_P(0,j)) / (r_center[1] - r_center[0]);
    } else if (i == nr-1 && nr >= 2) {
        dp_dr = (get_P(nr-1,j) - get_P(nr-2,j)) / (r_center[nr-1] - r_center[nr-2]);
    }

    double dp_dt_over_r = 0.0;
    if (j > 0 && j < nt-1) {
        dp_dt_over_r = (get_P(i,j+1) - get_P(i,j-1))
                     / (r * (theta_center[j+1] - theta_center[j-1]));
    }

    // --- gravity gradient (same stencil) ---
    double dphi_dr = 0.0;
    if (i > 0 && i < nr-1) {
        double gl = (phi[i*nt+j] - phi[(i-1)*nt+j]) / (r_center[i] - r_center[i-1]);
        double gr = (phi[(i+1)*nt+j] - phi[i*nt+j]) / (r_center[i+1] - r_center[i]);
        double dl = r_center[i] - r_center[i-1], dri = r_center[i+1] - r_center[i];
        dphi_dr = (dri*gl + dl*gr) / (dl + dri);
    } else if (i == 0 && nr > 1) {
        dphi_dr = (phi[1*nt+j] - phi[0*nt+j]) / (r_center[1] - r_center[0]);
    } else if (i == nr-1 && nr >= 2) {
        dphi_dr = (phi[(nr-1)*nt+j] - phi[(nr-2)*nt+j]) / (r_center[nr-1] - r_center[nr-2]);
    }

    double dphi_dt_over_r = 0.0;
    if (j > 0 && j < nt-1) {
        double gl = (phi[i*nt+j] - phi[i*nt+j-1]) / (theta_center[j] - theta_center[j-1]);
        double gr = (phi[i*nt+j+1] - phi[i*nt+j]) / (theta_center[j+1] - theta_center[j]);
        double dl = theta_center[j] - theta_center[j-1];
        double dri = theta_center[j+1] - theta_center[j];
        dphi_dt_over_r = (dri*gl + dl*gr) / (r * (dl + dri));
    }

    // --- combined source: -(∇p + ρ∇Φ) ---
    double Sr = -(dp_dr + w.rho * dphi_dr);
    double St = -(dp_dt_over_r + w.rho * dphi_dt_over_r);

    // geometric pressure source: 2P/r (radial) and P*cot(θ)/r (theta)
    double r2h = r_face[i+1]*r_face[i+1], r2l = r_face[i]*r_face[i];
    double dcos = cos(theta_face[j]) - cos(theta_face[j+1]);
    Sr += w.P * (r2h - r2l) * dcos / vol[flat];

    double r2dh = (r2h - r2l) * 0.5;
    double dsin = sin(theta_face[j]) - sin(theta_face[j+1]);
    St += w.P * r2dh * dsin / vol[flat];

    // update momentum
    mr[k] += dt * Sr;
    mt[k] += dt * St;

    // energy: -v · (∇p + ρ∇Φ) — work done by combined force on fluid
    E[k] += dt * (w.vr * Sr + w.vt * St);
}

// ========================= Reduction helpers ======================

__global__
void k_rhovol(const double* rho, const double* vol, double* out,
              int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i=flat/nt, j=flat%nt;
    out[flat] = rho[d_idx(i,j,nt,ng)] * vol[flat];
}

__global__
void k_reduce_sum(const double* in, double* out, int n) {
    extern __shared__ double s[];
    int tid=threadIdx.x, idx=blockIdx.x*blockDim.x+tid;
    s[tid] = (idx < n) ? in[idx] : 0.0;
    __syncthreads();
    for (int st=blockDim.x/2; st>0; st>>=1) {
        if (tid < st) s[tid] += s[tid+st];
        __syncthreads();
    }
    if (tid==0) out[blockIdx.x] = s[0];
}

static double gpu_reduce(double* src, double* dst, int n) {
    int B=256;
    while (n > 1) {
        int nb = (n+B-1)/B;
        k_reduce_sum<<<nb,B,B*sizeof(double)>>>(src, dst, n);
        n = nb;
        double* tmp=src; src=dst; dst=tmp;
    }
    double val;
    CUDA_CHECK(cudaMemcpy(&val, src, sizeof(double), cudaMemcpyDeviceToHost));
    return val;
}

// ========================= Gravity RHS ============================

__global__
void k_grav_rhs(const double* rho, double* rhs, const double* rc,
                int nr, int nt, int ng, double G, double M_total) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i=flat/nt, j=flat%nt;
    if (i == nr-1)
        rhs[flat] = -G * M_total / rc[nr-1];
    else
        rhs[flat] = 4.0 * M_PI * G * rho[d_idx(i,j,nt,ng)];
}

static void solve_gravity(GpuSolver& s) {
    int nr=s.gc.nr, nt=s.gc.ntheta, ng=s.gc.ng, n=nr*nt, B=256;
    k_rhovol<<<(n+B-1)/B,B>>>(s.d_rho, s.d_cell_volume, s.d_work_a, nr, nt, ng);
    double M = gpu_reduce(s.d_work_a, s.d_work_b, n);
    k_grav_rhs<<<(n+B-1)/B,B>>>(s.d_rho, s.d_rhs, s.d_r_center, nr,nt,ng, s.gc.G, M);
#ifdef USE_AMGX
    AMGX_SAFE_CALL(AMGX_vector_upload(s.amgx_b_grav, n, 1, s.d_rhs));
    AMGX_SAFE_CALL(AMGX_vector_upload(s.amgx_x_grav, n, 1, s.d_phi));
    AMGX_SAFE_CALL(AMGX_solver_solve(s.amgx_solver_grav, s.amgx_b_grav, s.amgx_x_grav));
    AMGX_SAFE_CALL(AMGX_vector_download(s.amgx_x_grav, s.d_phi));
#endif
}

// ========================= GpuSolver methods ======================

void GpuSolver::init(const Grid& grid, const EOS& eos, double G, double cfl_val,
                     Limiter lim, const std::string& amgx_config) {
    gc.nr=grid.nr; gc.ntheta=grid.ntheta; gc.ng=grid.ng;
    gc.R_outer=grid.R_outer; gc.gamma=eos.gamma; gc.G=G; gc.cfl=cfl_val;
    limiter = lim;

    int nr=gc.nr, nt=gc.ntheta, ng=gc.ng;
    total_phys = nr*nt;
    total_ghost = (nr+2*ng)*(nt+2*ng);
    dt_max = 0.01;

    // grid
    CUDA_CHECK(cudaMalloc(&d_r_face, (nr+1)*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r_center, nr*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dr, nr*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_theta_face, (nt+1)*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_theta_center, nt*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dtheta, nt*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cell_volume, total_phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_area_r, (nr+1)*nt*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_area_theta, nr*(nt+1)*sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_r_face, grid.r_face.data(), (nr+1)*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r_center, grid.r_center.data(), nr*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dr, grid.dr.data(), nr*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_theta_face, grid.theta_face.data(), (nt+1)*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_theta_center, grid.theta_center.data(), nt*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dtheta, grid.dtheta.data(), nt*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cell_volume, grid.cell_volume.data(), total_phys*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_area_r, grid.area_r.data(), (nr+1)*nt*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_area_theta, grid.area_theta.data(), nr*(nt+1)*sizeof(double), cudaMemcpyHostToDevice));

    // state
    CUDA_CHECK(cudaMalloc(&d_rho, total_ghost*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mr, total_ghost*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mtheta, total_ghost*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_E, total_ghost*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rho_old, total_phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_phi, total_phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pressure, total_phys*sizeof(double)));

    CUDA_CHECK(cudaMemset(d_rho, 0, total_ghost*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mr, 0, total_ghost*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mtheta, 0, total_ghost*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_E, 0, total_ghost*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_phi, 0, total_phys*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_pressure, 0, total_phys*sizeof(double)));

    // flux / work
    CUDA_CHECK(cudaMalloc(&d_flux_r, (nr+1)*nt*4*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_flux_t, nr*(nt+1)*4*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_work_a, total_phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_work_b, total_phys*sizeof(double)));
    int nb=(total_phys+255)/256;
    CUDA_CHECK(cudaMalloc(&d_dt_min, nb*sizeof(double)));

    // Poisson
    PoissonMatrix pm;
    pm.assemble(grid, G);
    poisson_n=pm.n; poisson_nnz=pm.nnz;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (pm.n+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, pm.nnz*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grav_values, pm.nnz*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pres_values, pm.nnz*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rhs, pm.n*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, pm.row_ptr.data(), (pm.n+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, pm.col_idx.data(), pm.nnz*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grav_values, pm.values.data(), pm.nnz*sizeof(double), cudaMemcpyHostToDevice));

#ifdef USE_AMGX
    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_config_create_from_file(&amgx_cfg, amgx_config.c_str()));
    AMGX_SAFE_CALL(AMGX_resources_create_simple(&amgx_rsrc, amgx_cfg));
    AMGX_SAFE_CALL(AMGX_matrix_create(&amgx_A_grav, amgx_rsrc, AMGX_mode_dDDI));
    AMGX_SAFE_CALL(AMGX_vector_create(&amgx_b_grav, amgx_rsrc, AMGX_mode_dDDI));
    AMGX_SAFE_CALL(AMGX_vector_create(&amgx_x_grav, amgx_rsrc, AMGX_mode_dDDI));
    AMGX_SAFE_CALL(AMGX_solver_create(&amgx_solver_grav, amgx_rsrc, AMGX_mode_dDDI, amgx_cfg));
    AMGX_SAFE_CALL(AMGX_matrix_upload_all(amgx_A_grav, pm.n, pm.nnz, 1, 1,
        pm.row_ptr.data(), pm.col_idx.data(), pm.values.data(), nullptr));
    AMGX_SAFE_CALL(AMGX_solver_setup(amgx_solver_grav, amgx_A_grav));
#endif

    initialized = true;
    std::printf("GPU semi-implicit solver: %dx%d (%d cells), dt_max=%.3e\n",
                nr, nt, total_phys, dt_max);
    std::fflush(stdout);
}

void GpuSolver::upload_state(const Grid& grid, const State& state) {
    CUDA_CHECK(cudaMemcpy(d_rho, state.rho.data(), total_ghost*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mr, state.mr.data(), total_ghost*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mtheta, state.mtheta.data(), total_ghost*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E, state.E.data(), total_ghost*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_phi, state.phi.data(), total_phys*sizeof(double), cudaMemcpyHostToDevice));
}

void GpuSolver::download_state(const Grid& grid, State& state) {
    CUDA_CHECK(cudaMemcpy(state.rho.data(), d_rho, total_ghost*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.mr.data(), d_mr, total_ghost*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.mtheta.data(), d_mtheta, total_ghost*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.E.data(), d_E, total_ghost*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.phi.data(), d_phi, total_phys*sizeof(double), cudaMemcpyDeviceToHost));
}

double GpuSolver::step(double t, double t_end) {
    int nr=gc.nr, nt=gc.ntheta, ng=gc.ng, n=nr*nt, B=256;

    double dt = dt_max;
    if (t + dt > t_end) dt = t_end - t;

    // 1) Ghost cells
    launch_ghost(d_rho, d_mr, d_mtheta, d_E, nr, nt, ng);

    // 2-3) Advection skipped for near-equilibrium (v≈0); only source terms matter

    // 4) Gravity Poisson solve
    solve_gravity(*this);

    // 5) Ghost cells again (need neighbors for pressure gradient)
    launch_ghost(d_rho, d_mr, d_mtheta, d_E, nr, nt, ng);

    // 6) Combined pressure + gravity source (well-balanced)
    k_pres_grav_source<<<(n+B-1)/B,B>>>(d_mr, d_mtheta, d_E, d_rho, d_phi,
        d_r_center, d_r_face, d_theta_center, d_theta_face,
        d_cell_volume, d_dr, nr, nt, ng, gc.gamma, dt);

    return dt;
}

void GpuSolver::destroy() {
    if (!initialized) return;
#ifdef USE_AMGX
    AMGX_solver_destroy(amgx_solver_grav);
    AMGX_matrix_destroy(amgx_A_grav);
    AMGX_vector_destroy(amgx_b_grav);
    AMGX_vector_destroy(amgx_x_grav);
    AMGX_resources_destroy(amgx_rsrc);
    AMGX_config_destroy(amgx_cfg);
    AMGX_finalize();
#endif
    cudaFree(d_r_face); cudaFree(d_r_center); cudaFree(d_dr);
    cudaFree(d_theta_face); cudaFree(d_theta_center); cudaFree(d_dtheta);
    cudaFree(d_cell_volume); cudaFree(d_area_r); cudaFree(d_area_theta);
    cudaFree(d_rho); cudaFree(d_mr); cudaFree(d_mtheta); cudaFree(d_E);
    cudaFree(d_rho_old); cudaFree(d_phi); cudaFree(d_pressure);
    cudaFree(d_flux_r); cudaFree(d_flux_t);
    cudaFree(d_work_a); cudaFree(d_work_b); cudaFree(d_dt_min);
    cudaFree(d_row_ptr); cudaFree(d_col_idx);
    cudaFree(d_grav_values); cudaFree(d_pres_values); cudaFree(d_rhs);
    initialized = false;
}
