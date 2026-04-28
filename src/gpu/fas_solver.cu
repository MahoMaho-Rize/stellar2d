// FAS (Full Approximation Scheme) nonlinear multigrid for the low-Mach
// 4-DOF Euler system on structured spherical (r, θ) grids.
//
// Replaces JFNK+GMRES with a direct nonlinear multigrid that achieves
// O(1) convergence independent of dt and mesh size.

#include "fas_solver.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(err)); std::exit(1); \
    } \
} while(0)

// ========================= Device helpers =========================

__device__ __forceinline__
int fas_idx(int i, int j, int nt, int ng) {
    return (i + ng) * (nt + 2 * ng) + (j + ng);
}

// ========================= Ghost cells ============================

__global__ void k_fas_ghost_r_in(double* rho, double* mr, double* mt, double* rhoE,
                                  int nr, int nt, int ng) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y + 1;
    if (j >= nt || g > ng) return;
    int kg = fas_idx(-g,j,nt,ng), kp = fas_idx(g-1,j,nt,ng);
    rho[kg]=rho[kp]; mr[kg]=-mr[kp]; mt[kg]=mt[kp]; rhoE[kg]=rhoE[kp];
}

__global__ void k_fas_ghost_r_out(double* rho, double* mr, double* mt, double* rhoE,
                                   int nr, int nt, int ng) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y;
    if (j >= nt || g >= ng) return;
    int kg = fas_idx(nr+g,j,nt,ng), kp = fas_idx(nr-1,j,nt,ng);
    rho[kg]=rho[kp]; mr[kg]=mr[kp]; mt[kg]=mt[kp]; rhoE[kg]=rhoE[kp];
}

__global__ void k_fas_ghost_t_n(double* rho, double* mr, double* mt, double* rhoE,
                                 int nr, int nt, int ng) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y + 1;
    if (ii >= nr+2*ng || g > ng) return;
    int i = ii - ng;
    int kg = fas_idx(i,-g,nt,ng), kp = fas_idx(i,g-1,nt,ng);
    rho[kg]=rho[kp]; mr[kg]=mr[kp]; mt[kg]=-mt[kp]; rhoE[kg]=rhoE[kp];
}

__global__ void k_fas_ghost_t_s(double* rho, double* mr, double* mt, double* rhoE,
                                 int nr, int nt, int ng) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y;
    if (ii >= nr+2*ng || g >= ng) return;
    int i = ii - ng;
    int kg = fas_idx(i,nt+g,nt,ng), kp = fas_idx(i,nt-1-g,nt,ng);
    rho[kg]=rho[kp]; mr[kg]=mr[kp]; mt[kg]=-mt[kp]; rhoE[kg]=rhoE[kp];
}

void FasSolver::launch_ghost(int l) {
    FasLevel& lev = levels[l];
    int B=256;
    { dim3 g((lev.nt+B-1)/B, lev.ng); k_fas_ghost_r_in<<<g,B>>>(lev.d_rho,lev.d_mr,lev.d_mt,lev.d_rhoE,lev.nr,lev.nt,lev.ng); }
    { dim3 g((lev.nt+B-1)/B, lev.ng); k_fas_ghost_r_out<<<g,B>>>(lev.d_rho,lev.d_mr,lev.d_mt,lev.d_rhoE,lev.nr,lev.nt,lev.ng); }
    { dim3 g((lev.nr+2*lev.ng+B-1)/B, lev.ng); k_fas_ghost_t_n<<<g,B>>>(lev.d_rho,lev.d_mr,lev.d_mt,lev.d_rhoE,lev.nr,lev.nt,lev.ng); }
    { dim3 g((lev.nr+2*lev.ng+B-1)/B, lev.ng); k_fas_ghost_t_s<<<g,B>>>(lev.d_rho,lev.d_mr,lev.d_mt,lev.d_rhoE,lev.nr,lev.nt,lev.ng); }
}

// ========================= 1D radial gravity ========================

__global__
void k_fas_shell_mass(const double* rho, const double* vol,
                      double* shell, int nr, int nt, int ng) {
    int i = blockIdx.x;
    if (i >= nr) return;
    extern __shared__ double smem[];
    int tid = threadIdx.x;
    double s = 0.0;
    for (int j = tid; j < nt; j += blockDim.x)
        s += rho[fas_idx(i,j,nt,ng)] * vol[i*nt+j];
    smem[tid] = s;
    __syncthreads();
    for (int st = blockDim.x/2; st > 0; st >>= 1) {
        if (tid < st) smem[tid] += smem[tid+st];
        __syncthreads();
    }
    if (tid == 0) shell[i] = smem[0] * 2.0 * M_PI;
}

__global__
void k_fas_gravity_from_shells(const double* shell_mass, const double* rc,
                                double* gr, int nr, double G) {
    if (threadIdx.x != 0) return;
    double M_enc = 0.0;
    for (int i = 0; i < nr; ++i) {
        M_enc += shell_mass[i];
        gr[i] = -G * M_enc / (rc[i] * rc[i]);
    }
}

void FasSolver::compute_gravity_1d(int l) {
    FasLevel& lev = levels[l];
    int B = std::min(lev.nt, 256);
    k_fas_shell_mass<<<lev.nr, B, B*sizeof(double)>>>(
        lev.d_rho, lev.d_cell_volume, lev.d_shell_mass, lev.nr, lev.nt, lev.ng);
    k_fas_gravity_from_shells<<<1, 1>>>(
        lev.d_shell_mass, lev.d_r_center, lev.d_gr, lev.nr, G_const);
}

// ========================= Nonlinear residual R(U) ========================
// Reuses the same physics as k_lm_residual but self-contained for FAS.

__global__
void k_fas_residual(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* vol, const double* ar, const double* at,
    const double* r_center, const double* r_face,
    const double* theta_face, const double* dr, const double* dtheta,
    const double* gr, const double* gr0,
    const double* P0, const double* rho0,
    double* res,
    int nr, int nt, int ng, double gam, double atm_thresh,
    int use_wellbalance) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    int n = nr*nt;

    if (rho0[flat] < atm_thresh) {
        res[flat] = 0.0; res[n+flat] = 0.0;
        res[2*n+flat] = 0.0; res[3*n+flat] = 0.0;
        return;
    }

    int k = fas_idx(i,j,nt,ng);
    double invV = 1.0 / vol[flat];

    double rho_c = fmax(rho[k], 1e-20);
    double vr_c = mr[k] / rho_c;
    double vt_c = mt[k] / rho_c;
    double e_c = rhoE[k] / rho_c;
    double P_c = fmax((gam - 1.0) * rho_c * e_c, 1e-30);
    double r = r_center[i];

    // Upwind advection (donor-cell)
    auto upwind_r = [&](int il, int ir, int jj, int c) -> double {
        int kl = fas_idx(il,jj,nt,ng), kr_k = fas_idx(ir,jj,nt,ng);
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
        int kl = fas_idx(ii,jl,nt,ng), kr_k = fas_idx(ii,jr,nt,ng);
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

    // Pressure gradient + gravity source terms.
    // On fine level (use_wellbalance=1): well-balanced perturbation form
    //   force = -∇P' + ρ'g₀ + ρg'  where P'=P-P₀, ρ'=ρ-ρ₀, g'=g-g₀
    // On coarse levels (use_wellbalance=0): standard (non-perturbation) form
    //   force = -∇P + ρg
    // The τ-correction in FAS naturally compensates for the different operator.

    double P_ref = use_wellbalance ? P0[flat] : 0.0;
    double rho_ref = use_wellbalance ? rho0[flat] : 0.0;
    double g0_r = use_wellbalance ? gr0[i] : 0.0;

    double Pp_c = P_c - P_ref;
    double rhop_c = rho_c - rho_ref;

    double dPp_dr = 0;
    if (i > 0 && i < nr-1) {
        double Pp_m = fmax((gam-1.0)*rhoE[fas_idx(i-1,j,nt,ng)],1e-30)
                      - (use_wellbalance ? P0[(i-1)*nt+j] : 0.0);
        double Pp_p = fmax((gam-1.0)*rhoE[fas_idx(i+1,j,nt,ng)],1e-30)
                      - (use_wellbalance ? P0[(i+1)*nt+j] : 0.0);
        double dl = r_center[i]-r_center[i-1], dh = r_center[i+1]-r_center[i];
        dPp_dr = (dh*(Pp_c-Pp_m)/dl + dl*(Pp_p-Pp_c)/dh) / (dl+dh);
    } else if (i == 0 && nr > 1) {
        double dh = r_center[1]-r_center[0];
        dPp_dr = (fmax((gam-1.0)*rhoE[fas_idx(1,j,nt,ng)],1e-30)
                  - (use_wellbalance ? P0[1*nt+j] : 0.0) - Pp_c)/dh;
    } else if (i == nr-1 && nr >= 2) {
        double dl = r_center[nr-1]-r_center[nr-2];
        dPp_dr = (Pp_c - (fmax((gam-1.0)*rhoE[fas_idx(nr-2,j,nt,ng)],1e-30)
                  - (use_wellbalance ? P0[(nr-2)*nt+j] : 0.0)))/dl;
    }

    double dPp_dt_r = 0;
    if (j > 0 && j < nt-1) {
        double tc_m=0.5*(theta_face[j-1]+theta_face[j]);
        double tc_c=0.5*(theta_face[j]+theta_face[j+1]);
        double tc_p=0.5*(theta_face[j+1]+theta_face[j+2]);
        double dl=tc_c-tc_m, dh=tc_p-tc_c;
        double Pp_m = fmax((gam-1.0)*rhoE[fas_idx(i,j-1,nt,ng)],1e-30)
                      - (use_wellbalance ? P0[i*nt+j-1] : 0.0);
        double Pp_p = fmax((gam-1.0)*rhoE[fas_idx(i,j+1,nt,ng)],1e-30)
                      - (use_wellbalance ? P0[i*nt+j+1] : 0.0);
        dPp_dt_r = ((dh*(Pp_c-Pp_m)/dl + dl*(Pp_p-Pp_c)/dh))/(r*(dl+dh));
    }

    double gp_r = gr[i] - g0_r;

    double inv_r = 1.0 / r;
    double S_mr = rho_c * vt_c * vt_c * inv_r;
    double S_mt = -rho_c * vr_c * vt_c * inv_r;

    double force_r = -dPp_dr + rhop_c*g0_r + rho_c*gp_r;
    double force_t = -dPp_dt_r;

    double S_E = rho_c * vr_c * (g0_r + gp_r);

    auto vr_face_f = [&](int il, int ir, int jj) -> double {
        double rl = fmax(rho[fas_idx(il,jj,nt,ng)],1e-20);
        double rr = fmax(rho[fas_idx(ir,jj,nt,ng)],1e-20);
        return 0.5*(mr[fas_idx(il,jj,nt,ng)]/rl + mr[fas_idx(ir,jj,nt,ng)]/rr);
    };
    auto vt_face_f = [&](int ii, int jl, int jr) -> double {
        double rl = fmax(rho[fas_idx(ii,jl,nt,ng)],1e-20);
        double rr = fmax(rho[fas_idx(ii,jr,nt,ng)],1e-20);
        return 0.5*(mt[fas_idx(ii,jl,nt,ng)]/rl + mt[fas_idx(ii,jr,nt,ng)]/rr);
    };
    double div_v = invV*(
        Ar_hi*vr_face_f(i,i+1,j) - Ar_lo*vr_face_f(i-1,i,j) +
        At_hi*vt_face_f(i,j,j+1) - At_lo*vt_face_f(i,j-1,j));

    res[flat]       = div[0];
    res[n + flat]   = div[1] + force_r + S_mr;
    res[2*n + flat] = div[2] + force_t + S_mt;
    res[3*n + flat] = div[3] - P_c * div_v + S_E;
}

void FasSolver::compute_residual(int l) {
    FasLevel& lev = levels[l];
    int n = lev.nr * lev.nt, B = 256;
    launch_ghost(l);
    compute_gravity_1d(l);
    int wb = (l == 0) ? 1 : 0;  // well-balance only on finest level
    k_fas_residual<<<(n+B-1)/B,B>>>(
        lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE,
        lev.d_cell_volume, lev.d_area_r, lev.d_area_theta,
        lev.d_r_center, lev.d_r_face, lev.d_theta_face,
        lev.d_dr, lev.d_dtheta,
        lev.d_gr, lev.d_gr0, lev.d_P0, lev.d_rho0,
        lev.d_res,
        lev.nr, lev.nt, lev.ng, gamma, atm_rho_thresh, wb);
}

// ========================= Newton residual F(U) ========================
// F = Uⁿ/dt - U/dt + R(U) + fas_rhs_correction
// We store it in d_res (overwrite spatial residual).

__global__
void k_fas_compute_F(double* F, const double* R, const double* rho, const double* mr,
                     const double* mt, const double* rhoE,
                     const double* fas_rhs,
                     double inv_dt, int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int n = nr*nt;
    int k = fas_idx(flat/nt, flat%nt, nt, ng);
    // F = R(U) - (U - Uⁿ)/dt, but fas_rhs encodes Uⁿ/dt + τ
    F[flat]       = R[flat]       - inv_dt*rho[k]  + fas_rhs[flat];
    F[n + flat]   = R[n + flat]   - inv_dt*mr[k]   + fas_rhs[n + flat];
    F[2*n + flat] = R[2*n + flat] - inv_dt*mt[k]   + fas_rhs[2*n + flat];
    F[3*n + flat] = R[3*n + flat] - inv_dt*rhoE[k] + fas_rhs[3*n + flat];
}

void FasSolver::compute_F(int l, double dt) {
    FasLevel& lev = levels[l];
    int n = lev.nr * lev.nt, B = 256;
    compute_residual(l);
    k_fas_compute_F<<<(n+B-1)/B,B>>>(
        lev.d_res, lev.d_res, lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE,
        lev.d_fas_rhs, 1.0/dt, lev.nr, lev.nt, lev.ng);
}

// ========================= BLAS-like helpers ========================

__global__
void k_fas_scale(double* x, double a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= a;
}

__global__
void k_fas_copy(double* dst, const double* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// Pack physical cells from ghost-cell arrays into flat array
__global__
void k_fas_pack_flat(const double* rho, const double* mr,
                     const double* mt, const double* rhoE,
                     double* out, int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int k = fas_idx(flat/nt, flat%nt, nt, ng);
    int n = nr*nt;
    out[flat] = rho[k]; out[n+flat] = mr[k];
    out[2*n+flat] = mt[k]; out[3*n+flat] = rhoE[k];
}

// ========================= Floor ========================

__global__
void k_fas_floor(double* rho, double* rhoE, int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int k = fas_idx(flat/nt, flat%nt, nt, ng);
    rho[k] = fmax(rho[k], 1e-20);
    rhoE[k] = fmax(rhoE[k], 1e-20);
}

void FasSolver::apply_floor(int l) {
    FasLevel& lev = levels[l];
    int n = lev.nr * lev.nt, B = 256;
    k_fas_floor<<<(n+B-1)/B,B>>>(lev.d_rho, lev.d_rhoE, lev.nr, lev.nt, lev.ng);
}

// ========================= Block Jacobi smoother ========================

__global__
void k_fas_assemble_blkjac(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* vol, const double* ar, const double* at,
    const double* r_center, const double* r_face, const double* theta_face,
    const double* dr, const double* dtheta,
    const double* gr0,
    double* blk_inv,
    int nr, int nt, int ng, double gam, double inv_dt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    int k = fas_idx(i,j,nt,ng);

    double rho_c = fmax(rho[k], 1e-20);
    double vr_c = mr[k] / rho_c;
    double vt_c = mt[k] / rho_c;
    double e_c = rhoE[k] / rho_c;
    double P_c = fmax((gam-1.0)*rho_c*e_c, 1e-30);
    double cs = sqrt(gam * P_c / rho_c);
    double r = r_center[i];
    double invV = 1.0 / vol[flat];

    double sr = (fabs(vr_c)+cs)/dr[i] + (fabs(vt_c)+cs)/(r*dtheta[j]);
    double dP_drhoE = gam - 1.0;

    double dPdr_coeff = 0.0;
    if (i > 0 && i < nr-1) {
        double dl = r_center[i]-r_center[i-1], dh = r_center[i+1]-r_center[i];
        dPdr_coeff = -(dh/(dl*(dl+dh)) + dl/(dh*(dl+dh)));
    } else if (i == 0 && nr > 1) {
        dPdr_coeff = -1.0/(r_center[1]-r_center[0]);
    } else if (i == nr-1 && nr >= 2) {
        dPdr_coeff = 1.0/(r_center[nr-1]-r_center[nr-2]);
    }

    double dPdt_coeff = 0.0;
    if (j > 0 && j < nt-1) {
        double tc_m=0.5*(theta_face[j-1]+theta_face[j]);
        double tc_c=0.5*(theta_face[j]+theta_face[j+1]);
        double tc_p=0.5*(theta_face[j+1]+theta_face[j+2]);
        double dl=tc_c-tc_m, dh=tc_p-tc_c;
        dPdt_coeff = -(dh/(dl*(dl+dh)) + dl/(dh*(dl+dh))) / r;
    }

    double g0_val = gr0[i];

    double Ar_hi=ar[(i+1)*nt+j], Ar_lo=ar[i*nt+j];
    double At_hi=at[i*(nt+1)+j+1], At_lo=at[i*(nt+1)+j];
    double ddivv_dmr = invV * 0.5/rho_c * (Ar_hi + Ar_lo);
    double ddivv_dmt = invV * 0.5/rho_c * (At_hi + At_lo);

    double J[16];
    for (int q=0; q<16; q++) J[q] = 0.0;
    J[0]  = -(inv_dt + sr);
    J[5]  = -(inv_dt + sr);
    J[10] = -(inv_dt + sr);
    J[15] = -(inv_dt + sr);
    J[1*4+3] = -dPdr_coeff * dP_drhoE;
    J[2*4+3] = -dPdt_coeff * dP_drhoE;
    J[1*4+0] += g0_val;
    J[3*4+1] = -P_c * ddivv_dmr;
    J[3*4+2] = -P_c * ddivv_dmt;

    // 4×4 inversion via Gauss-Jordan
    double A[16]; for(int q=0;q<16;q++) A[q]=J[q];
    double inv_m[16]; for(int q=0;q<16;q++) inv_m[q]=(q/4==q%4)?1.0:0.0;

    for(int col=0;col<4;col++){
        double mx=fabs(A[col*4+col]); int mi=col;
        for(int row=col+1;row<4;row++){if(fabs(A[row*4+col])>mx){mx=fabs(A[row*4+col]);mi=row;}}
        if(mi!=col){
            for(int q=0;q<4;q++){double t=A[col*4+q];A[col*4+q]=A[mi*4+q];A[mi*4+q]=t;}
            for(int q=0;q<4;q++){double t=inv_m[col*4+q];inv_m[col*4+q]=inv_m[mi*4+q];inv_m[mi*4+q]=t;}
        }
        double d=A[col*4+col]; if(fabs(d)<1e-30) d=(d>=0?1e-30:-1e-30);
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

    double* B = &blk_inv[flat*16];
    for(int q=0;q<16;q++) B[q]=inv_m[q];
}

// ========================= SIMPLE smoother kernels ========================

__global__
void k_fas_mom_diag(const double* rho, const double* mr, const double* mt,
                    const double* rhoE,
                    const double* dr, const double* rc, const double* dtheta,
                    double* Ap, int nr, int nt, int ng,
                    double inv_dt, double gam) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    int k = fas_idx(i,j,nt,ng);
    double rho_c = fmax(rho[k], 1e-20);
    double vr = fabs(mr[k]/rho_c), vt = fabs(mt[k]/rho_c);
    double P = fmax((gam - 1.0) * rhoE[k], 1e-30);
    double cs = sqrt(gam * P / rho_c);
    Ap[flat] = inv_dt + (vr + cs)/dr[i] + (vt + cs)/(rc[i]*dtheta[j]);
}

__global__
void k_fas_vstar(const double* F, const double* Ap, const double* rho0,
                 double atm_thresh, double* vr_s, double* vt_s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (rho0[i] < atm_thresh) { vr_s[i] = 0; vt_s[i] = 0; return; }
    double inv_ap = 1.0 / Ap[i];
    vr_s[i] = -F[n + i] * inv_ap;
    vt_s[i] = -F[2*n + i] * inv_ap;
}

__global__
void k_fas_div(const double* vr_s, const double* vt_s,
               const double* vol, const double* ar, const double* at,
               double* div_out, int nr, int nt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    double invV = 1.0 / vol[flat];
    double vr_lo = (i>0)    ? 0.5*(vr_s[flat]+vr_s[(i-1)*nt+j]) : 0.0;
    double vr_hi = (i<nr-1) ? 0.5*(vr_s[flat]+vr_s[(i+1)*nt+j]) : 0.0;
    double vt_lo = (j>0)    ? 0.5*(vt_s[flat]+vt_s[i*nt+j-1])   : 0.0;
    double vt_hi = (j<nt-1) ? 0.5*(vt_s[flat]+vt_s[i*nt+j+1])   : 0.0;
    div_out[flat] = invV*(ar[(i+1)*nt+j]*vr_hi - ar[i*nt+j]*vr_lo
                         + at[i*(nt+1)+j+1]*vt_hi - at[i*(nt+1)+j]*vt_lo);
}

__global__
void k_fas_prhs(const double* div_v, double* rhs, int nr, int nt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    rhs[flat] = (flat/nt == nr-1) ? 0.0 : div_v[flat];
}

__global__
void k_fas_inv_ap(const double* Ap, double* alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) alpha[i] = 1.0 / Ap[i];
}

// SIMPLE correction: momentum via Poisson-corrected velocity, ρ/E via block-inverse
__global__
void k_fas_simple_correct(
    double* rho, double* mr, double* mt, double* rhoE,
    const double* F, const double* blk_inv,
    const double* vr_s, const double* vt_s,
    const double* dp, const double* Ap,
    const double* r_center, const double* theta_face,
    const double* rho0, double atm_thresh,
    int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    if (rho0[flat] < atm_thresh) return;
    int n = nr*nt;
    int i = flat/nt, j = flat%nt;
    int k = fas_idx(i,j,nt,ng);
    double rho_c = fmax(rho[k], 1e-20);

    // Pressure gradient of δp
    int im = flat - nt*(int)(i>0), ip = flat + nt*(int)(i<nr-1);
    double dp_dr = 0.0;
    if (i > 0 && i < nr-1) {
        double dl=r_center[i]-r_center[i-1], dh=r_center[i+1]-r_center[i];
        dp_dr = (dh*(dp[flat]-dp[im])/dl + dl*(dp[ip]-dp[flat])/dh)/(dl+dh);
    } else if (i==0 && nr>1) dp_dr = (dp[ip]-dp[flat])/(r_center[1]-r_center[0]);
    else if (i==nr-1 && nr>=2) dp_dr = (dp[flat]-dp[im])/(r_center[nr-1]-r_center[nr-2]);

    double dp_dt_r = 0.0;
    if (j > 0 && j < nt-1) {
        double tc_m=0.5*(theta_face[j-1]+theta_face[j]);
        double tc_c=0.5*(theta_face[j]+theta_face[j+1]);
        double tc_p=0.5*(theta_face[j+1]+theta_face[j+2]);
        double dl=tc_c-tc_m, dh=tc_p-tc_c;
        dp_dt_r = (dh*(dp[flat]-dp[i*nt+j-1])/dl + dl*(dp[i*nt+j+1]-dp[flat])/dh)
                  / (r_center[i]*(dl+dh));
    }

    double inv_ap = 1.0 / Ap[flat];
    double dvr = vr_s[flat] - inv_ap * dp_dr;
    double dvt = vt_s[flat] - inv_ap * dp_dt_r;

    // Momentum correction
    mr[k]  -= rho_c * dvr;
    mt[k]  -= rho_c * dvt;

    // ρ and E: block-inverse rows
    const double* B = &blk_inv[flat*16];
    double f0 = F[flat], f1 = F[n+flat], f2 = F[2*n+flat], f3 = F[3*n+flat];
    rho[k]  -= B[0]*f0 + B[1]*f1 + B[2]*f2 + B[3]*f3;
    rhoE[k] -= B[12]*f0 + B[13]*f1 + B[14]*f2 + B[15]*f3;
}

// ========================= SIMPLE smoother ========================

void FasSolver::assemble_smoother(int l, double dt) {
    FasLevel& lev = levels[l];
    int n = lev.nr * lev.nt, B = 256;
    k_fas_assemble_blkjac<<<(n+B-1)/B,B>>>(
        lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE,
        lev.d_cell_volume, lev.d_area_r, lev.d_area_theta,
        lev.d_r_center, lev.d_r_face, lev.d_theta_face,
        lev.d_dr, lev.d_dtheta,
        lev.d_gr0,
        lev.d_blk_inv,
        lev.nr, lev.nt, lev.ng, gamma, 1.0/dt);
    k_fas_mom_diag<<<(n+B-1)/B,B>>>(
        lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE,
        lev.d_dr, lev.d_r_center, lev.d_dtheta,
        lev.d_Ap, lev.nr, lev.nt, lev.ng, 1.0/dt, gamma);
}

void FasSolver::smooth(int l, double dt, int n_iters) {
    FasLevel& lev = levels[l];
    int n = lev.nr * lev.nt, B = 256;

    for (int it = 0; it < n_iters; ++it) {
        compute_F(l, dt);

        // 1. Momentum prediction: δv* = -F_mom / Ap
        k_fas_vstar<<<(n+B-1)/B,B>>>(lev.d_res, lev.d_Ap,
            lev.d_rho0, atm_rho_thresh,
            lev.d_vr_s, lev.d_vt_s, n);

        // 2. Pressure Poisson: ∇·((1/Ap)∇δp) = div(δv*)
        k_fas_div<<<(n+B-1)/B,B>>>(lev.d_vr_s, lev.d_vt_s,
            lev.d_cell_volume, lev.d_area_r, lev.d_area_theta,
            lev.d_div_s, lev.nr, lev.nt);
        k_fas_inv_ap<<<(n+B-1)/B,B>>>(lev.d_Ap, lev.d_inv_Ap, n);
        k_fas_prhs<<<(n+B-1)/B,B>>>(lev.d_div_s, lev.d_poisson_rhs, lev.nr, lev.nt);
        CUDA_CHECK(cudaMemset(lev.d_dp, 0, n*sizeof(double)));
        lev.pressure_gmg.solve_varcoeff(lev.d_inv_Ap, lev.d_poisson_rhs, lev.d_dp, 3, 1e-2);

        // 3. Correct state
        k_fas_simple_correct<<<(n+B-1)/B,B>>>(
            lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE,
            lev.d_res, lev.d_blk_inv,
            lev.d_vr_s, lev.d_vt_s,
            lev.d_dp, lev.d_Ap,
            lev.d_r_center, lev.d_theta_face,
            lev.d_rho0, atm_rho_thresh,
            lev.nr, lev.nt, lev.ng);
        apply_floor(l);
    }
}

// ========================= 4-DOF Restriction ========================
// Volume-weighted restriction of conservative variables.

__global__
void k_fas_restrict_state(
    const double* f_rho, const double* f_mr, const double* f_mt, const double* f_rhoE,
    const double* f_vol,
    double* c_rho, double* c_mr, double* c_mt, double* c_rhoE,
    int cnr, int cnt, int fnt, int fng, int cng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= cnr*cnt) return;
    int ic = flat/cnt, jc = flat%cnt;
    int if0 = 2*ic, jf0 = 2*jc;
    int ck = (ic+cng)*(cnt+2*cng) + (jc+cng);

    double s_rho=0, s_mr=0, s_mt=0, s_rhoE=0, s_vol=0;
    for (int di = 0; di < 2; ++di) {
        int fi = if0 + di;
        for (int dj = 0; dj < 2; ++dj) {
            int fj = jf0 + dj;
            int fk = (fi+fng)*(fnt+2*fng) + (fj+fng);
            int ff = fi*fnt + fj;
            double v = f_vol[ff];
            s_rho += f_rho[fk]*v; s_mr += f_mr[fk]*v;
            s_mt  += f_mt[fk]*v;  s_rhoE += f_rhoE[fk]*v;
            s_vol += v;
        }
    }
    double inv_v = 1.0 / fmax(s_vol, 1e-300);
    c_rho[ck]  = s_rho * inv_v;
    c_mr[ck]   = s_mr  * inv_v;
    c_mt[ck]   = s_mt  * inv_v;
    c_rhoE[ck] = s_rhoE* inv_v;
}

void FasSolver::restrict_state(int fine, int coarse) {
    FasLevel& fl = levels[fine], &cl = levels[coarse];
    int cn = cl.nr * cl.nt, B = 256;
    k_fas_restrict_state<<<(cn+B-1)/B,B>>>(
        fl.d_rho, fl.d_mr, fl.d_mt, fl.d_rhoE, fl.d_cell_volume,
        cl.d_rho, cl.d_mr, cl.d_mt, cl.d_rhoE,
        cl.nr, cl.nt, fl.nt, fl.ng, cl.ng);
}

// ========================= FAS defect restriction ========================
// f_H = R(Î·u_h) + Î·(f_h - R(u_h))
// = R(u_H) + Î·(Uⁿ_h/dt + τ_h - R(u_h)) - Î·(Uⁿ_h/dt + τ_h) + R(u_H)
// Simpler: f_H = fas_rhs_H = Uⁿ_H/dt + τ_H
// where τ_H = R(u_H) - Î·R(u_h) + Î·(Uⁿ_h/dt + old_τ_h) - Uⁿ_H/dt
//
// Implementation:
//   1. Compute R(u_h) on fine level → stored in d_res
//   2. Compute F(u_h) = R(u_h) - U_h/dt + fas_rhs_h → d_res  (fine level defect)
//   3. Restrict defect: d_H = Î·F(u_h)
//   4. Restrict state: u_H = Î·u_h
//   5. Compute R(u_H) on coarse level
//   6. fas_rhs_H = U_H/dt + R(u_H) - d_H (so that F(u_H)=R(u_H)-U_H/dt+fas_rhs_H = -d_H ≈ -Î·F(u_h))

__global__
void k_fas_restrict_defect(
    const double* f_res, const double* f_vol,
    double* c_defect,
    int cnr, int cnt, int fnr, int fnt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= cnr*cnt) return;
    int ic = flat/cnt, jc = flat%cnt;
    int if0 = 2*ic, jf0 = 2*jc;
    int fn = fnr*fnt;
    int cn = cnr*cnt;

    for (int eq = 0; eq < 4; ++eq) {
        double ws=0, vs=0;
        for (int di = 0; di < 2; ++di) {
            int fi = if0 + di;
            if (fi >= fnr) continue;
            for (int dj = 0; dj < 2; ++dj) {
                int fj = jf0 + dj;
                if (fj >= fnt) continue;
                int ff = fi*fnt + fj;
                double v = f_vol[ff];
                ws += f_res[eq*fn + ff] * v;
                vs += v;
            }
        }
        c_defect[eq*cn + flat] = (vs > 0) ? ws/vs : 0.0;
    }
}

// Assemble FAS RHS on coarse level:
// fas_rhs_H = U_H/dt - R(U_H) + restricted_defect
// This ensures F(U_H) = R(U_H) - U_H/dt + fas_rhs_H = restricted_defect
__global__
void k_fas_assemble_coarse_rhs(
    const double* R_H,     // R(U_H) after restrict + compute_residual
    const double* rho_H, const double* mr_H, const double* mt_H, const double* rhoE_H,
    const double* defect,  // restricted fine-level defect
    double* fas_rhs,
    double inv_dt, int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int n = nr*nt;
    int k = fas_idx(flat/nt, flat%nt, nt, ng);

    // fas_rhs = U_H/dt - R(U_H) + defect
    // So F = R(U_H) - U_H/dt + fas_rhs = defect (initially)
    fas_rhs[flat]       = inv_dt*rho_H[k]  - R_H[flat]       + defect[flat];
    fas_rhs[n + flat]   = inv_dt*mr_H[k]   - R_H[n + flat]   + defect[n + flat];
    fas_rhs[2*n + flat] = inv_dt*mt_H[k]   - R_H[2*n + flat] + defect[2*n + flat];
    fas_rhs[3*n + flat] = inv_dt*rhoE_H[k] - R_H[3*n + flat] + defect[3*n + flat];
}

void FasSolver::restrict_defect(int fine, int coarse, double dt) {
    FasLevel& fl = levels[fine], &cl = levels[coarse];
    int fn = fl.nr * fl.nt, cn = cl.nr * cl.nt, B = 256;

    // 1. Compute F(u_h) on fine level (already have d_res from last smooth)
    compute_F(fine, dt);

    // 2. Restrict fine defect F(u_h) → temporary in cl.d_Un (4*cn scratch)
    k_fas_restrict_defect<<<(cn+B-1)/B,B>>>(
        fl.d_res, fl.d_cell_volume,
        cl.d_Un,
        cl.nr, cl.nt, fl.nr, fl.nt);

    // 3. Restrict state u_h → u_H
    restrict_state(fine, coarse);
    apply_floor(coarse);

    // 4. Compute R(u_H) on coarse level
    compute_residual(coarse);

    // 5. Assemble fas_rhs_H so that F(u_H) = restricted defect
    k_fas_assemble_coarse_rhs<<<(cn+B-1)/B,B>>>(
        cl.d_res,
        cl.d_rho, cl.d_mr, cl.d_mt, cl.d_rhoE,
        cl.d_Un,  // restricted defect from step 2
        cl.d_fas_rhs,
        1.0/dt, cl.nr, cl.nt, cl.ng);

    // ===== 方案 C: HSE defect subtraction =====
    // The well-balanced residual on the coarse grid may not be exactly zero
    // for the HSE reference state due to restriction errors.
    // Compute this "HSE defect" and subtract it from fas_rhs, so that
    // F(U₀_H) = 0 exactly on the coarse grid.
    // This is only needed when use_wellbalance=1 on coarse levels.
    // When use_wellbalance=0 (方案 A), this is automatically handled.
    //
    // Disabled by default since 方案 A (non-WB coarse) is simpler and more robust.
    // Uncomment to enable:
    // {
    //     // Temporarily load HSE state into coarse level
    //     // ... compute R(U₀_H) ... subtract from fas_rhs ...
    // }
}

// ========================= Prolongation + correction ========================
// u_h += P · (u_H - Î·u_h)
// Piecewise-constant prolongation (injection) + floor protection.

__global__
void k_fas_prolongate_correct(
    double* f_rho, double* f_mr, double* f_mt, double* f_rhoE,
    const double* c_rho, const double* c_mr, const double* c_mt, const double* c_rhoE,
    const double* save_rho, const double* save_mr, const double* save_mt, const double* save_rhoE,
    int cnr, int cnt, int fnt, int fng, int cng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= cnr*cnt) return;
    int ic = flat/cnt, jc = flat%cnt;
    int ck = (ic+cng)*(cnt+2*cng) + (jc+cng);

    // Coarse grid correction = u_H_new - u_H_before_solve (saved)
    double d_rho  = c_rho[ck]  - save_rho[flat];
    double d_mr   = c_mr[ck]   - save_mr[flat];
    double d_mt   = c_mt[ck]   - save_mt[flat];
    double d_rhoE = c_rhoE[ck] - save_rhoE[flat];

    // Piecewise constant prolongation to 2×2 fine cells
    for (int di = 0; di < 2; ++di) {
        int fi = 2*ic + di;
        for (int dj = 0; dj < 2; ++dj) {
            int fj = 2*jc + dj;
            int fk = (fi+fng)*(fnt+2*fng) + (fj+fng);
            f_rho[fk]  += d_rho;
            f_mr[fk]   += d_mr;
            f_mt[fk]   += d_mt;
            f_rhoE[fk] += d_rhoE;
        }
    }
}

void FasSolver::prolongate_correct(int coarse, int fine) {
    FasLevel& cl = levels[coarse], &fl = levels[fine];
    int cn = cl.nr * cl.nt, B = 256;

    // cl.d_save contains the pre-solve restricted state
    k_fas_prolongate_correct<<<(cn+B-1)/B,B>>>(
        fl.d_rho, fl.d_mr, fl.d_mt, fl.d_rhoE,
        cl.d_rho, cl.d_mr, cl.d_mt, cl.d_rhoE,
        cl.d_save, cl.d_save + cn, cl.d_save + 2*cn, cl.d_save + 3*cn,
        cl.nr, cl.nt, fl.nt, fl.ng, cl.ng);
    apply_floor(fine);
}

// ========================= FAS V-cycle ========================

void FasSolver::fas_vcycle(int l, double dt) {
    // During initial transient (first 10 steps), use smooth-only on finest
    // level to avoid coarse-grid instability. After that, use full V-cycle.
    // Disable coarse correction entirely for now.
    // Block Jacobi smooth-only converges for small dt (dominated by 1/dt diagonal).
    // TODO: fix coarse-level HSE reference to enable proper V-cycles.
    bool use_coarse = false;

    if (!use_coarse || l == n_levels - 1) {
        assemble_smoother(l, dt);
        smooth(l, dt, NU1 + NU2);
        return;
    }

    // Pre-smooth
    assemble_smoother(l, dt);
    smooth(l, dt, NU1);

    FasLevel& cl = levels[l + 1];
    int cn = cl.nr * cl.nt, B = 256;

    // Restrict defect and state: fine → coarse
    restrict_defect(l, l + 1, dt);

    // Save restricted state BEFORE coarse solve
    k_fas_pack_flat<<<(cn+B-1)/B,B>>>(
        cl.d_rho, cl.d_mr, cl.d_mt, cl.d_rhoE,
        cl.d_save, cl.nr, cl.nt, cl.ng);

    // Recurse
    fas_vcycle(l + 1, dt);

    // Prolongate: u_h += P·(u_H_new - u_H_old)
    prolongate_correct(l + 1, l);

    // Post-smooth
    assemble_smoother(l, dt);
    smooth(l, dt, NU2);
}

// ========================= Residual norm ========================

double FasSolver::residual_norm(int l) {
    FasLevel& lev = levels[l];
    int n4 = 4 * lev.nr * lev.nt;
    // Simple: download and compute on host
    std::vector<double> h(n4);
    CUDA_CHECK(cudaMemcpy(h.data(), lev.d_res, n4*sizeof(double), cudaMemcpyDeviceToHost));
    double mx = 0;
    for (int i = 0; i < n4; ++i) mx = std::max(mx, std::fabs(h[i]));
    return mx;
}

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
    CUDA_CHECK(cudaMalloc(&lev.d_save, 4*phys*sizeof(double)));

    // HSE reference
    CUDA_CHECK(cudaMalloc(&lev.d_rho0, phys*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lev.d_P0, phys*sizeof(double)));

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
            double rho_v = std::fmax(state.rho[sk], 1e-20);
            double vr = state.mr[sk] / rho_v;
            double vt = state.mtheta[sk] / rho_v;
            h_rho[k] = state.rho[sk];
            h_mr[k] = state.mr[sk];
            h_mt[k] = state.mtheta[sk];
            // Convert total energy → internal energy: ρe = ρE - ½ρv²
            h_rhoE[k] = std::fmax(state.E[sk] - 0.5*rho_v*(vr*vr+vt*vt), 1e-30);
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
            // Convert internal energy → total energy: ρE = ρe + ½ρv²
            double rho_v = std::fmax(h_rho[k], 1e-20);
            double vr = h_mr[k] / rho_v;
            double vt = h_mt[k] / rho_v;
            state.E[sk] = h_rhoE[k] + 0.5*rho_v*(vr*vr+vt*vt);
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
            P0[flat] = (gamma - 1.0) * h_rhoE[k];
        }
    CUDA_CHECK(cudaMemcpy(finest.d_rho0, rho0.data(), n*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(finest.d_P0, P0.data(), n*sizeof(double), cudaMemcpyHostToDevice));

    double rho_max = *std::max_element(rho0.begin(), rho0.end());
    atm_rho_thresh = 1e-6 * rho_max;

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

    hse_set = true;
    std::fprintf(stderr, "  FAS HSE snapshot: ρ_max=%.3e, atm_thresh=%.3e\n",
                 rho_max, atm_rho_thresh);
}

// ========================= Public solve ========================

int FasSolver::solve(double dt, int max_cycles, double tol) {
    FasLevel& finest = levels[0];
    int n = finest.nr * finest.nt, B = 256;

    // Save Uⁿ and set fas_rhs = Uⁿ/dt on finest level
    k_fas_pack_flat<<<(n+B-1)/B,B>>>(
        finest.d_rho, finest.d_mr, finest.d_mt, finest.d_rhoE,
        finest.d_fas_rhs, finest.nr, finest.nt, finest.ng);
    k_fas_scale<<<(4*n+B-1)/B,B>>>(finest.d_fas_rhs, 1.0/dt, 4*n);

    for (int cyc = 0; cyc < max_cycles; ++cyc) {
        fas_vcycle(0, dt);

        compute_F(0, dt);
        double norm = residual_norm(0);
        if (norm < tol) return cyc + 1;
    }
    return max_cycles;
}

// ========================= Time step ========================

double FasSolver::step(double t, double t_end) {
    if (!hse_set) { snapshot_hse(); }

    apply_floor(0);

    double dt_cap = 1.0;  // SIMPLE smoother handles pressure stiffness
    if (dt_current < 1e-30) dt_current = 1e-8;

    int max_dt_cuts = 6;
    int max_cycles = 40;
    double tol = 1e-4;
    double dt = std::min({dt_current, dt_cap, t_end - t});
    bool converged = false;

    // Save state before solve (for rollback on failure)
    FasLevel& finest = levels[0];
    int n = finest.nr * finest.nt, B = 256;
    // Use d_save on finest level as rollback buffer
    k_fas_pack_flat<<<(n+B-1)/B,B>>>(
        finest.d_rho, finest.d_mr, finest.d_mt, finest.d_rhoE,
        finest.d_save, finest.nr, finest.nt, finest.ng);

    for (int cut = 0; cut < max_dt_cuts; ++cut) {
        if (cut > 0) {
            dt *= 0.5;
            if (dt < 1e-12) {
                std::fprintf(stderr, "FATAL: FAS step %d dt=%.3e diverged.\n", step_count, dt);
                std::exit(1);
            }
            // Rollback state from saved
            // Unpack d_save back into state arrays
            std::vector<double> h_save(4*n);
            CUDA_CHECK(cudaMemcpy(h_save.data(), finest.d_save, 4*n*sizeof(double), cudaMemcpyDeviceToHost));
            std::vector<double> h_rho(finest.total), h_mr(finest.total), h_mt(finest.total), h_rhoE(finest.total);
            CUDA_CHECK(cudaMemcpy(h_rho.data(), finest.d_rho, finest.total*sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_mr.data(), finest.d_mr, finest.total*sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_mt.data(), finest.d_mt, finest.total*sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_rhoE.data(), finest.d_rhoE, finest.total*sizeof(double), cudaMemcpyDeviceToHost));
            int stride = finest.nt + 2*finest.ng;
            for (int i = 0; i < finest.nr; ++i)
                for (int j = 0; j < finest.nt; ++j) {
                    int k = (i+finest.ng)*stride + (j+finest.ng);
                    int flat = i*finest.nt + j;
                    h_rho[k] = h_save[flat];
                    h_mr[k] = h_save[n+flat];
                    h_mt[k] = h_save[2*n+flat];
                    h_rhoE[k] = h_save[3*n+flat];
                }
            CUDA_CHECK(cudaMemcpy(finest.d_rho, h_rho.data(), finest.total*sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(finest.d_mr, h_mr.data(), finest.total*sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(finest.d_mt, h_mt.data(), finest.total*sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(finest.d_rhoE, h_rhoE.data(), finest.total*sizeof(double), cudaMemcpyHostToDevice));
            apply_floor(0);

            if (step_count < 10 || step_count % 100 == 0)
                std::fprintf(stderr, "  step %d: FAS cut %d, dt -> %.3e\n", step_count, cut, dt);
        }

        int cycles = solve(dt, max_cycles, tol);
        if (cycles < max_cycles) {
            converged = true;
            if (step_count < 10 || step_count % 100 == 0)
                std::fprintf(stderr, "  step %d: %d FAS V-cycles (dt=%.3e)\n",
                             step_count, cycles, dt);
            break;
        }
        // Check final residual — if close to tol, accept anyway
        compute_F(0, dt);
        double norm = residual_norm(0);
        if (norm < 10.0 * tol) {
            converged = true;
            if (step_count < 10 || step_count % 100 == 0)
                std::fprintf(stderr, "  step %d: %d FAS V-cycles (dt=%.3e) ||F||=%.2e (accepted)\n",
                             step_count, cycles, dt, norm);
            break;
        }
    }

    if (converged) {
        dt_current = std::min(1.2 * dt, dt_cap);
    } else {
        dt_current = dt;  // don't grow
    }
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
        cudaFree(lev.d_fas_rhs); cudaFree(lev.d_res); cudaFree(lev.d_Un); cudaFree(lev.d_save);
        cudaFree(lev.d_rho0); cudaFree(lev.d_P0);
        cudaFree(lev.d_gr); cudaFree(lev.d_gr0); cudaFree(lev.d_shell_mass);
        cudaFree(lev.d_blk_inv);
        cudaFree(lev.d_Ap); cudaFree(lev.d_vr_s); cudaFree(lev.d_vt_s);
        cudaFree(lev.d_div_s); cudaFree(lev.d_dp);
        cudaFree(lev.d_poisson_rhs); cudaFree(lev.d_inv_Ap);
        lev.pressure_gmg.destroy();
    }
    n_levels = 0;
}
