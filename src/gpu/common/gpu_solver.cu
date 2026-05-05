// Fully-implicit GPU solver: Backward Euler + JFNK + GMRES.
//
// Each time step solves: (U^{n+1} - U^n)/dt = R(U^{n+1})
// via Newton: [I/dt - dR/dU] δU = R(U^k) - (U^k - U^n)/dt
// via GMRES:  J·v computed Jacobian-free as [F(U+εv)-F(U)]/ε
//
// R(U) = well-balanced HLLC flux divergence + geometric source + gravity.

#include "gpu_solver.h"
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
    int kg = d_idx(-g,j,nt,ng), kp = d_idx(g-1,j,nt,ng);
    rho[kg]=rho[kp]; mr[kg]=-mr[kp]; mt[kg]=mt[kp]; E[kg]=E[kp];
}
__global__ void k_ghost_r_out(double* rho, double* mr, double* mt, double* E,
                               int nr, int nt, int ng) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y;
    if (j >= nt || g >= ng) return;
    int kg = d_idx(nr+g,j,nt,ng), kp = d_idx(nr-1,j,nt,ng);
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

// ========================= MUSCL + HLLC ===========================

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

struct DFlux4 { double f_rho, f_mr, f_mt, f_E; };

__device__
DFlux4 d_hllc(DPrim wl, DPrim wr, double gamma, bool radial) {
    double rhol=wl.rho, rhor=wr.rho, pl=wl.P, pr=wr.P;
    double ul = radial ? wl.vr : wl.vt;
    double ur = radial ? wr.vr : wr.vt;
    double vtl = radial ? wl.vt : wl.vr;
    double vtr = radial ? wr.vt : wr.vr;

    double cl = sqrt(gamma*pl/rhol), cr = sqrt(gamma*pr/rhor);
    double sl = fmin(ul-cl, ur-cr), sr = fmax(ul+cl, ur+cr);
    double denom = rhol*(sl-ul) - rhor*(sr-ur);
    if (fabs(denom) < 1e-300) denom = -1e-300;
    double s_star = (pr-pl + rhol*ul*(sl-ul) - rhor*ur*(sr-ur)) / denom;

    auto phys_flux = [&](double rho, double u, double vt, double p) -> DFlux4 {
        double E = p/(gamma-1.0) + 0.5*rho*(u*u + vt*vt);
        DFlux4 f;
        f.f_rho = rho*u;
        if (radial) { f.f_mr = rho*u*u+p; f.f_mt = rho*u*vt; }
        else        { f.f_mr = rho*u*vt;  f.f_mt = rho*u*u+p; }
        f.f_E = (E+p)*u;
        return f;
    };

    auto star_flux = [&](double rho, double u, double vt, double p,
                         double sk, DFlux4 fk) -> DFlux4 {
        double E = p/(gamma-1.0) + 0.5*rho*(u*u+vt*vt);
        double ratio = rho*(sk-u)/(sk-s_star);
        double E_star = ratio*(E/rho + (s_star-u)*(s_star + p/(rho*(sk-u))));
        DFlux4 f;
        f.f_rho = fk.f_rho + sk*(ratio-rho);
        if (radial) {
            f.f_mr = fk.f_mr + sk*(ratio*s_star - rho*u);
            f.f_mt = fk.f_mt + sk*(ratio*vt - rho*vt);
        } else {
            f.f_mr = fk.f_mr + sk*(ratio*vt - rho*vt);
            f.f_mt = fk.f_mt + sk*(ratio*s_star - rho*u);
        }
        f.f_E = fk.f_E + sk*(E_star-E);
        return f;
    };

    if (sl >= 0.0) return phys_flux(rhol,ul,vtl,pl);
    else if (s_star >= 0.0) { auto fl=phys_flux(rhol,ul,vtl,pl); return star_flux(rhol,ul,vtl,pl,sl,fl); }
    else if (sr >= 0.0) { auto fr=phys_flux(rhor,ur,vtr,pr); return star_flux(rhor,ur,vtr,pr,sr,fr); }
    else return phys_flux(rhor,ur,vtr,pr);
}

// Well-balanced HLLC: reconstruct perturbations p'=P-P₀, ρ'=ρ-ρ₀
__global__
void k_hllc_r(const double* rho, const double* mr, const double* mt,
              const double* E, const double* P0, const double* rho0,
              double* flux, int nr, int nt, int ng, double gamma) {
    int i = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nt || i > nr) return;

    auto W = [&](int ii, int jj) {
        int k = d_idx(ii,jj,nt,ng);
        return to_prim(rho[k], mr[k], mt[k], E[k], gamma);
    };
    auto P0r = [&](int ii) { return P0[max(0,min(ii,nr-1))*nt+j]; };
    auto r0r = [&](int ii) { return rho0[max(0,min(ii,nr-1))*nt+j]; };

    DPrim wa=W(i-2,j), wb=W(i-1,j), wc=W(i,j), wd=W(i+1,j);

    double ppL,ppR,rpL,rpR;
    d_recon(wa.P-P0r(i-2), wb.P-P0r(i-1), wc.P-P0r(i), wd.P-P0r(i+1), ppL, ppR);
    d_recon(wa.rho-r0r(i-2), wb.rho-r0r(i-1), wc.rho-r0r(i), wd.rho-r0r(i+1), rpL, rpR);

    double P0f = 0.5*(P0r(i-1)+P0r(i)), r0f = 0.5*(r0r(i-1)+r0r(i));

    DPrim wl, wr;
    d_recon(wa.vr, wb.vr, wc.vr, wd.vr, wl.vr, wr.vr);
    d_recon(wa.vt, wb.vt, wc.vt, wd.vt, wl.vt, wr.vt);
    wl.rho = fmax(r0f+rpL, 1e-20); wr.rho = fmax(r0f+rpR, 1e-20);
    wl.P = fmax(P0f+ppL, 1e-30); wr.P = fmax(P0f+ppR, 1e-30);

    DFlux4 f = d_hllc(wl, wr, gamma, true);
    int idx = (i*nt+j)*4;
    flux[idx]=f.f_rho; flux[idx+1]=f.f_mr; flux[idx+2]=f.f_mt; flux[idx+3]=f.f_E;
}

__global__
void k_hllc_t(const double* rho, const double* mr, const double* mt,
              const double* E, const double* P0, const double* rho0,
              double* flux, int nr, int nt, int ng, double gamma) {
    int i = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j > nt || i >= nr) return;

    auto W = [&](int ii, int jj) {
        int k = d_idx(ii,jj,nt,ng);
        return to_prim(rho[k], mr[k], mt[k], E[k], gamma);
    };
    auto P0t = [&](int jj) { return P0[i*nt+max(0,min(jj,nt-1))]; };
    auto r0t = [&](int jj) { return rho0[i*nt+max(0,min(jj,nt-1))]; };

    DPrim wa=W(i,j-2), wb=W(i,j-1), wc=W(i,j), wd=W(i,j+1);

    double ppL,ppR,rpL,rpR;
    d_recon(wa.P-P0t(j-2), wb.P-P0t(j-1), wc.P-P0t(j), wd.P-P0t(j+1), ppL, ppR);
    d_recon(wa.rho-r0t(j-2), wb.rho-r0t(j-1), wc.rho-r0t(j), wd.rho-r0t(j+1), rpL, rpR);

    double P0f = 0.5*(P0t(j-1)+P0t(j)), r0f = 0.5*(r0t(j-1)+r0t(j));

    DPrim wl, wr;
    d_recon(wa.vr, wb.vr, wc.vr, wd.vr, wl.vr, wr.vr);
    d_recon(wa.vt, wb.vt, wc.vt, wd.vt, wl.vt, wr.vt);
    wl.rho = fmax(r0f+rpL, 1e-20); wr.rho = fmax(r0f+rpR, 1e-20);
    wl.P = fmax(P0f+ppL, 1e-30); wr.P = fmax(P0f+ppR, 1e-30);

    DFlux4 f = d_hllc(wl, wr, gamma, false);
    int idx = (i*(nt+1)+j)*4;
    flux[idx]=f.f_rho; flux[idx+1]=f.f_mr; flux[idx+2]=f.f_mt; flux[idx+3]=f.f_E;
}

// ========================= Spatial residual R(U) ==================
// Writes R into d_res (4*n packed: [rho-part | mr-part | mt-part | E-part])

__global__
void k_spatial_residual(
    const double* rho, const double* mr, const double* mt, const double* E,
    const double* flux_r, const double* flux_t,
    const double* vol, const double* ar, const double* at,
    const double* r_center, const double* r_face,
    const double* theta_face,
    const double* phi, const double* phi0,
    double* res,
    int nr, int nt, int ng, double gamma) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    int k = d_idx(i,j,nt,ng);
    int n = nr*nt;
    double invV = 1.0 / vol[flat];

    auto fr = [&](int ii, int jj, int c) { return flux_r[(ii*nt+jj)*4+c]; };
    auto ft = [&](int ii, int jj, int c) { return flux_t[(ii*(nt+1)+jj)*4+c]; };
    double Arh=ar[(i+1)*nt+j], Arl=ar[i*nt+j];
    double Ath=at[i*(nt+1)+j+1], Atl=at[i*(nt+1)+j];

    double div[4];
    for (int c = 0; c < 4; ++c)
        div[c] = -invV*(Arh*fr(i+1,j,c) - Arl*fr(i,j,c)
                       + Ath*ft(i,j+1,c) - Atl*ft(i,j,c));

    DPrim w = to_prim(rho[k], mr[k], mt[k], E[k], gamma);
    double r = r_center[i], inv_r = 1.0 / r;

    // Geometric source (Eq. 5.1-5.4)
    double S_mr = w.rho * w.vt * w.vt * inv_r;
    double S_mt = -w.rho * w.vr * w.vt * inv_r;

    double r2h = r_face[i+1]*r_face[i+1], r2l = r_face[i]*r_face[i];
    double dcos = cos(theta_face[j]) - cos(theta_face[j+1]);
    S_mr += w.P * (r2h - r2l) * dcos * invV;

    double dsin = sin(theta_face[j]) - sin(theta_face[j+1]);
    S_mt += w.P * 0.5*(r2h - r2l) * dsin * invV;

    // Gravity: perturbation potential Φ' = Φ - Φ₀
    auto phip = [&](int ii, int jj) { return phi[ii*nt+jj] - phi0[ii*nt+jj]; };

    double dphip_dr = 0.0;
    if (i > 0 && i < nr-1) {
        double gl = (phip(i,j)-phip(i-1,j)) / (r_center[i]-r_center[i-1]);
        double gr = (phip(i+1,j)-phip(i,j)) / (r_center[i+1]-r_center[i]);
        double dl = r_center[i]-r_center[i-1], dh = r_center[i+1]-r_center[i];
        dphip_dr = (dh*gl + dl*gr) / (dl+dh);
    } else if (i == 0 && nr > 1)
        dphip_dr = (phip(1,j)-phip(0,j)) / (r_center[1]-r_center[0]);
    else if (i == nr-1 && nr >= 2)
        dphip_dr = (phip(nr-1,j)-phip(nr-2,j)) / (r_center[nr-1]-r_center[nr-2]);

    double dphip_dt_r = 0.0;
    if (j > 0 && j < nt-1) {
        double gl = (phip(i,j)-phip(i,j-1)) / (r_center[i]*(0.5*(theta_face[j]+theta_face[j+1]) - 0.5*(theta_face[j-1]+theta_face[j])));
        double gr = (phip(i,j+1)-phip(i,j)) / (r_center[i]*(0.5*(theta_face[j+1]+theta_face[j+2]) - 0.5*(theta_face[j]+theta_face[j+1])));
        double dl = 0.5*(theta_face[j]+theta_face[j+1]) - 0.5*(theta_face[j-1]+theta_face[j]);
        double dh = 0.5*(theta_face[j+1]+theta_face[j+2]) - 0.5*(theta_face[j]+theta_face[j+1]);
        dphip_dt_r = (dh*gl + dl*gr) / (dl+dh);
    }

    double rho_v = rho[k];
    double vr_v = w.vr, vt_v = w.vt;

    S_mr -= rho_v * dphip_dr;
    S_mt -= rho_v * dphip_dt_r;
    double S_E = -rho_v * (vr_v*dphip_dr + vt_v*dphip_dt_r);

    res[flat]       = div[0];           // dρ/dt
    res[n + flat]   = div[1] + S_mr;    // d(ρvr)/dt
    res[2*n + flat] = div[2] + S_mt;    // d(ρvθ)/dt
    res[3*n + flat] = div[3] + S_E;     // d(ρE)/dt
}

// ========================= Pack / unpack ==========================
// Convert between ghost-cell state arrays and flat 4*n vectors.

__global__
void k_pack(const double* rho, const double* mr, const double* mt,
            const double* E, double* packed,
            int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt, n = nr*nt;
    int k = d_idx(i,j,nt,ng);
    packed[flat] = rho[k]; packed[n+flat] = mr[k];
    packed[2*n+flat] = mt[k]; packed[3*n+flat] = E[k];
}

__global__
void k_unpack_add(double* rho, double* mr, double* mt, double* E,
                  const double* delta, double sign,
                  int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt, n = nr*nt;
    int k = d_idx(i,j,nt,ng);
    rho[k] += sign * delta[flat]; mr[k] += sign * delta[n+flat];
    mt[k] += sign * delta[2*n+flat]; E[k] += sign * delta[3*n+flat];
}

__global__
void k_unpack_set(double* rho, double* mr, double* mt, double* E,
                  const double* packed,
                  int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt, n = nr*nt;
    int k = d_idx(i,j,nt,ng);
    rho[k] = packed[flat]; mr[k] = packed[n+flat];
    mt[k] = packed[2*n+flat]; E[k] = packed[3*n+flat];
}

// ========================= Vector operations ======================
// BLAS-like on 4*n vectors, all device.

__global__ void k_axpy(double* y, double a, const double* x, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

__global__ void k_scale(double* x, double a, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) x[i] *= a;
}

__global__ void k_copy(double* dst, const double* src, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

__global__ void k_set_zero(double* x, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.0;
}

// F(U) = R(U) - (U - Un)/dt
// Reads current state directly from ghost-cell arrays to avoid buffer conflicts.
__global__ void k_compute_F(double* F, const double* R,
                            const double* rho, const double* mr_a,
                            const double* mt_a, const double* E_a,
                            const double* Un, double inv_dt,
                            int nr_g, int nt_g, int ng_g) {
    int flat = blockIdx.x*blockDim.x + threadIdx.x;
    int n = nr_g * nt_g;
    if (flat >= n) return;
    int i = flat / nt_g, j = flat % nt_g;
    int k = (i + ng_g) * (nt_g + 2*ng_g) + (j + ng_g);

    double U_rho = rho[k], U_mr = mr_a[k], U_mt = mt_a[k], U_E = E_a[k];

    F[flat]       = R[flat]       - inv_dt * (U_rho - Un[flat]);
    F[n + flat]   = R[n + flat]   - inv_dt * (U_mr  - Un[n + flat]);
    F[2*n + flat] = R[2*n + flat] - inv_dt * (U_mt  - Un[2*n + flat]);
    F[3*n + flat] = R[3*n + flat] - inv_dt * (U_E   - Un[3*n + flat]);
}

// dot product (partial block sums)
__global__ void k_dot(const double* a, const double* b, double* out, int n) {
    extern __shared__ double s[];
    int tid = threadIdx.x, idx = blockIdx.x*blockDim.x+tid;
    s[tid] = (idx < n) ? a[idx]*b[idx] : 0.0;
    __syncthreads();
    for (int st=blockDim.x/2; st>0; st>>=1) {
        if (tid < st) s[tid] += s[tid+st];
        __syncthreads();
    }
    if (tid==0) out[blockIdx.x] = s[0];
}

static double gpu_dot(const double* a, const double* b, double* wa, double* wb, int n) {
    int B=256, nb=(n+B-1)/B;
    k_dot<<<nb,B,B*sizeof(double)>>>(a, b, wa, n);
    int cur = nb;
    double* src = wa; double* dst = wb;
    while (cur > 1) {
        int nb2 = (cur+B-1)/B;
        // sum reduction
        k_dot<<<nb2,B,B*sizeof(double)>>>(src, src, dst, cur);
        // Actually this is wrong - k_dot computes a*b, for sum we need a special kernel.
        // Let me just do it on host for correctness.
        break;
    }
    // Pull to host for final reduction
    std::vector<double> h(nb);
    CUDA_CHECK(cudaMemcpy(h.data(), wa, nb*sizeof(double), cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (int i = 0; i < nb; ++i) sum += h[i];
    return sum;
}

// ========================= Atmosphere floor =======================

__global__
void k_floor(double* rho, double* mr, double* mt, double* E,
             int nr, int nt, int ng, double gamma, double rho_fl, double P_fl) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int k = d_idx(flat/nt, flat%nt, nt, ng);
    if (rho[k] < rho_fl) {
        rho[k] = rho_fl; mr[k] = 0.0; mt[k] = 0.0;
        E[k] = P_fl / ((gamma-1.0)*rho_fl) * rho_fl;
    }
}

// ========================= HSE snapshot ===========================

__global__
void k_snapshot_hse(const double* rho, const double* mr, const double* mt,
                    const double* E, const double* phi,
                    double* rho0, double* P0, double* phi0,
                    int nr, int nt, int ng, double gamma) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int k = d_idx(flat/nt, flat%nt, nt, ng);
    DPrim w = to_prim(rho[k], mr[k], mt[k], E[k], gamma);
    rho0[flat] = w.rho; P0[flat] = w.P; phi0[flat] = phi[flat];
}

// ========================= CFL dt ================================

__global__
void k_cfl_dt(const double* rho, const double* mr, const double* mt, const double* E,
              const double* dr, const double* rc, const double* dtheta,
              double* out, int nr, int nt, int ng, double gamma) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    int k = d_idx(i,j,nt,ng);
    DPrim w = to_prim(rho[k], mr[k], mt[k], E[k], gamma);
    double cs = sqrt(gamma * w.P / w.rho);
    double dt_r = dr[i] / (fabs(w.vr) + cs);
    double dt_t = rc[i] * dtheta[j] / (fabs(w.vt) + cs);
    out[flat] = fmin(dt_r, dt_t);
}

__global__
void k_reduce_min(const double* in, double* out, int n) {
    extern __shared__ double s[];
    int tid=threadIdx.x, idx=blockIdx.x*blockDim.x+tid;
    s[tid] = (idx < n) ? in[idx] : 1e30;
    __syncthreads();
    for (int st=blockDim.x/2; st>0; st>>=1) {
        if (tid < st) s[tid] = fmin(s[tid], s[tid+st]);
        __syncthreads();
    }
    if (tid==0) out[blockIdx.x] = s[0];
}

// ========================= Gravity ================================

__global__
void k_rhovol(const double* rho, const double* vol, double* out,
              int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    out[flat] = rho[d_idx(flat/nt,flat%nt,nt,ng)] * vol[flat];
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

static double gpu_reduce_sum(double* src, double* dst, int n) {
    int B=256;
    while (n > 1) {
        int nb = (n+B-1)/B;
        k_reduce_sum<<<nb,B,B*sizeof(double)>>>(src, dst, n);
        n = nb; double* t=src; src=dst; dst=t;
    }
    double val;
    CUDA_CHECK(cudaMemcpy(&val, src, sizeof(double), cudaMemcpyDeviceToHost));
    return val;
}

__global__
void k_grav_rhs(const double* rho, double* rhs, const double* rc,
                int nr, int nt, int ng, double G, double M_total) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i=flat/nt;
    if (i == nr-1)
        rhs[flat] = -G * M_total / rc[nr-1];
    else
        rhs[flat] = 4.0 * M_PI * G * rho[d_idx(flat/nt,flat%nt,nt,ng)];
}

void GpuSolver::solve_gravity() {
    int n = nr*nt, B = 256;
    k_rhovol<<<(n+B-1)/B,B>>>(d_rho, d_cell_volume, d_work_grav_a, nr, nt, ng);
    double M = gpu_reduce_sum(d_work_grav_a, d_work_grav_b, n) * 2.0 * M_PI;
    k_grav_rhs<<<(n+B-1)/B,B>>>(d_rho, d_rhs_poisson, d_r_center, nr,nt,ng, G_const, M);
    gmg.solve(d_rhs_poisson, d_phi);
}

// ========================= GpuSolver methods ======================

void GpuSolver::compute_residual(double* d_res) {
    int n = nr*nt, B = 256;

    launch_ghost(d_rho, d_mr, d_mtheta, d_E, nr, nt, ng);

    { dim3 g((nt+B-1)/B, nr+1); k_hllc_r<<<g,B>>>(d_rho,d_mr,d_mtheta,d_E,d_P0,d_rho0,d_flux_r,nr,nt,ng,gamma); }
    { dim3 g((nt+1+B-1)/B, nr); k_hllc_t<<<g,B>>>(d_rho,d_mr,d_mtheta,d_E,d_P0,d_rho0,d_flux_t,nr,nt,ng,gamma); }

    k_spatial_residual<<<(n+B-1)/B,B>>>(
        d_rho,d_mr,d_mtheta,d_E,
        d_flux_r,d_flux_t,
        d_cell_volume,d_area_r,d_area_theta,
        d_r_center,d_r_face,d_theta_face,
        d_phi,d_phi0,
        d_res, nr,nt,ng,gamma);
}

void GpuSolver::compute_F(double* d_F, double dt) {
    int n = nr*nt, B = 256;
    compute_residual(d_residual);
    k_compute_F<<<(n+B-1)/B,B>>>(d_F, d_residual,
        d_rho, d_mr, d_mtheta, d_E, d_Un, 1.0/dt, nr, nt, ng);
}

void GpuSolver::pack_state(double* d_packed) {
    int B = 256;
    k_pack<<<(nr*nt+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_E, d_packed, nr,nt,ng);
}

void GpuSolver::unpack_delta(const double* d_delta, double sign) {
    int B = 256;
    k_unpack_add<<<(nr*nt+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_E, d_delta, sign, nr,nt,ng);
}

void GpuSolver::apply_floor() {
    int B = 256;
    k_floor<<<(nr*nt+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_E, nr,nt,ng,gamma, 1e-15, 1e-15);
}

void GpuSolver::jfnk_matvec(const double* d_v, double* d_Jv, double dt) {
    int N = 4*nr*nt, B = 256;

    double norm_v = sqrt(gpu_dot(d_v, d_v, d_work_dot_a, d_work_dot_b, N));
    if (norm_v < 1e-30) { k_set_zero<<<(N+B-1)/B,B>>>(d_Jv, N); return; }
    double eps_fd = 1e-7 / norm_v;

    // Save current state
    k_pack<<<(nr*nt+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_E, d_gmres_Uk, nr,nt,ng);

    // Perturb: U += ε·v
    k_unpack_add<<<(nr*nt+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_E, d_v, eps_fd, nr,nt,ng);
    apply_floor();

    // Gravity is FROZEN during GMRES — Φ stays at Newton-level value.
    // This is the standard inexact Newton approach: dΦ/dU is lagged,
    // Newton outer loop re-solves Φ each iteration to maintain accuracy.

    // Compute F(U+εv) with frozen Φ
    compute_F(d_Jv, dt);

    // Jv = (F(U+εv) - Fk) / ε
    k_axpy<<<(N+B-1)/B,B>>>(d_Jv, -1.0, d_Fk, N);
    k_scale<<<(N+B-1)/B,B>>>(d_Jv, 1.0/eps_fd, N);

    // Restore state (Φ is already at correct Newton-level value)
    k_unpack_set<<<(nr*nt+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_E, d_gmres_Uk, nr,nt,ng);
}

// ========================= Jacobian assembly ======================
// Approximate Jacobian: A ≈ I/dt - dR/dU
// Uses 1st-order Rusanov flux Jacobian for the diagonal block.
// Each cell (i,j) contributes a 4×4 block to the diagonal:
//   A_diag = I/dt + (λ_r/Δr + λ_θ/(rΔθ)) · I₄
// where λ = |v| + c_s is the Rusanov wave speed.
// Off-diagonal blocks couple to 4 neighbors with -λ_face/(2·Δx) · I₄.
// This captures the correct spectral radius of the spatial operator.

__global__
void k_assemble_jacobian(
    const double* rho, const double* mr, const double* mt, const double* E,
    const double* dr, const double* rc, const double* dtheta,
    const double* vol,
    int* row_ptr, int* col_idx, double* values,
    int nr_g, int nt_g, int ng_g, double gamma, double inv_dt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr_g * nt_g) return;
    int i = flat / nt_g, j = flat % nt_g;
    int k = d_idx(i, j, nt_g, ng_g);

    DPrim w = to_prim(rho[k], mr[k], mt[k], E[k], gamma);
    double cs = sqrt(gamma * w.P / w.rho);

    // Count neighbors (interior cells have 4, boundary cells fewer)
    // But sparsity is pre-built with max 5 entries per row.
    // We just fill the values. row_ptr/col_idx are set at init.

    int base = row_ptr[flat];
    int nnz_row = row_ptr[flat + 1] - base;

    // Iterate through the pre-built columns for this row
    for (int idx = base; idx < base + nnz_row; ++idx) {
        int col = col_idx[idx];
        double* blk = &values[idx * 16]; // 4×4 block, row-major

        // Zero the block
        for (int b = 0; b < 16; ++b) blk[b] = 0.0;

        if (col == flat) {
            // Diagonal block: I/dt + sum of Rusanov spectral radii
            double sr = 0.0;
            if (i > 0) sr += (fabs(w.vr)+cs) / (rc[i]-rc[i-1]);
            if (i < nr_g-1) sr += (fabs(w.vr)+cs) / (rc[i+1]-rc[i]);
            if (j > 0) sr += (fabs(w.vt)+cs) / (rc[i]*(dtheta[j]+dtheta[j-1])*0.5);
            if (j < nt_g-1) sr += (fabs(w.vt)+cs) / (rc[i]*(dtheta[j]+dtheta[j+1])*0.5);
            // Store -J ≈ I/dt + spectral_radius (positive definite M-matrix for AMG)
            double d = inv_dt + sr;
            blk[0] = d; blk[5] = d; blk[10] = d; blk[15] = d;
        } else {
            // Off-diagonal of -J: -λ/(2·Δx) (negative, standard M-matrix pattern)
            int ci = col / nt_g, cj = col % nt_g;
            int kn = d_idx(ci, cj, nt_g, ng_g);
            DPrim wn = to_prim(rho[kn], mr[kn], mt[kn], E[kn], gamma);
            double cs_n = sqrt(gamma * wn.P / wn.rho);
            double dist, lambda;
            if (ci != i) {
                dist = fabs(rc[ci] - rc[i]);
                lambda = fmax(fabs(w.vr)+cs, fabs(wn.vr)+cs_n);
            } else {
                dist = rc[i] * fabs(dtheta[j]+dtheta[cj]) * 0.5;
                lambda = fmax(fabs(w.vt)+cs, fabs(wn.vt)+cs_n);
            }
            double d = -0.5 * lambda / fmax(dist, 1e-30);
            blk[0] = d; blk[5] = d; blk[10] = d; blk[15] = d;
        }
    }
}

void GpuSolver::assemble_jacobian(double dt) {
    int n = nr*nt, B = 256;
    k_assemble_jacobian<<<(n+B-1)/B, B>>>(
        d_rho, d_mr, d_mtheta, d_E,
        d_dr, d_r_center, d_dtheta, d_cell_volume,
        d_jac_row_ptr, d_jac_col_idx, d_jac_values,
        nr, nt, ng, gamma, 1.0/dt);

    CUDA_CHECK(cudaDeviceSynchronize());

    AMGX_RC rc;
    if (!amgx_setup_done) {
        rc = AMGX_matrix_upload_all(amgx_A, n, jac_nnz, 4, 4,
            d_jac_row_ptr, d_jac_col_idx, d_jac_values, nullptr);
        if (rc != AMGX_RC_OK)
            std::fprintf(stderr, "AMGX matrix_upload_all failed: %d\n", rc);
        rc = AMGX_solver_setup(amgx_solver, amgx_A);
        if (rc != AMGX_RC_OK)
            std::fprintf(stderr, "AMGX solver_setup failed: %d\n", rc);
        else
            std::fprintf(stderr, "AmgX setup OK: n=%d nnz=%d block=4x4\n", n, jac_nnz);
        amgx_setup_done = true;
    } else {
        rc = AMGX_matrix_replace_coefficients(amgx_A, n, jac_nnz, d_jac_values, nullptr);
        if (rc != AMGX_RC_OK)
            std::fprintf(stderr, "AMGX replace_coefficients failed: %d\n", rc);
    }
}

void GpuSolver::apply_preconditioner(const double* d_v, double* d_Mv) {
    int n = nr*nt, N = 4*n, B = 256;
    AMGX_vector_upload(amgx_b, n, 4, d_v);
    AMGX_vector_set_zero(amgx_x, n, 4);
    AMGX_solver_solve(amgx_solver, amgx_b, amgx_x);
    AMGX_vector_download(amgx_x, d_Mv);
    k_scale<<<(N+B-1)/B,B>>>(d_Mv, -1.0, N);
}

// ========================= FGMRES =================================
// Flexible GMRES: stores Z[j] = M^{-1}·V[j] separately so the
// preconditioner can vary between iterations (though we freeze it).

int GpuSolver::fgmres_solve(double* d_x, const double* d_b, double dt,
                            double tol, int max_iter) {
    int N = 4*nr*nt, B = 256;
    int m = std::min(max_iter, (int)GMRES_RESTART);

    std::vector<double> H((m+1)*m, 0.0);
    std::vector<double> cs(m), sn(m), g(m+1, 0.0);

    // r0 = -F (since we solve Jx = -F)
    k_copy<<<(N+B-1)/B,B>>>(d_gmres_V[0], d_b, N);
    k_scale<<<(N+B-1)/B,B>>>(d_gmres_V[0], -1.0, N);

    double beta = sqrt(gpu_dot(d_gmres_V[0], d_gmres_V[0], d_work_dot_a, d_work_dot_b, N));
    if (beta < 1e-30) return 0;
    k_scale<<<(N+B-1)/B,B>>>(d_gmres_V[0], 1.0/beta, N);
    g[0] = beta;

    int j;
    for (j = 0; j < m; ++j) {
        // Z[j] = M^{-1} · V[j]
        apply_preconditioner(d_gmres_V[j], d_gmres_Z[j]);

        // w = J · Z[j]
        jfnk_matvec(d_gmres_Z[j], d_gmres_w, dt);

        // Arnoldi
        for (int i = 0; i <= j; ++i) {
            H[i*m+j] = gpu_dot(d_gmres_w, d_gmres_V[i], d_work_dot_a, d_work_dot_b, N);
            k_axpy<<<(N+B-1)/B,B>>>(d_gmres_w, -H[i*m+j], d_gmres_V[i], N);
        }
        H[(j+1)*m+j] = sqrt(gpu_dot(d_gmres_w, d_gmres_w, d_work_dot_a, d_work_dot_b, N));

        if (H[(j+1)*m+j] < 1e-30) { j++; break; }
        k_copy<<<(N+B-1)/B,B>>>(d_gmres_V[j+1], d_gmres_w, N);
        k_scale<<<(N+B-1)/B,B>>>(d_gmres_V[j+1], 1.0/H[(j+1)*m+j], N);

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

    // Back-substitution
    std::vector<double> y(j);
    for (int i = j-1; i >= 0; --i) {
        y[i] = g[i];
        for (int kk = i+1; kk < j; ++kk)
            y[i] -= H[i*m+kk] * y[kk];
        y[i] /= H[i*m+i];
    }

    // x = sum y[i] · Z[i]  (note: Z, not V — this is the FGMRES difference)
    k_set_zero<<<(N+B-1)/B,B>>>(d_x, N);
    for (int i = 0; i < j; ++i)
        k_axpy<<<(N+B-1)/B,B>>>(d_x, y[i], d_gmres_Z[i], N);

    return j;
}

// ========================= Init / Upload / Download ===============

void GpuSolver::init(const Grid& grid, const EOS& eos, double G, double cfl,
                     Limiter lim) {
    nr = grid.nr; nt = grid.ntheta; ng = grid.ng;
    gamma = eos.gamma; G_const = G; cfl_num = cfl;
    limiter = lim;
    total_phys = nr*nt;
    total_ghost = (nr+2*ng)*(nt+2*ng);
    dt_max = 0.01;

    int n = total_phys, B = 256;

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

    CUDA_CHECK(cudaMalloc(&d_rho, total_ghost*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mr, total_ghost*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mtheta, total_ghost*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_E, total_ghost*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_rho, 0, total_ghost*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mr, 0, total_ghost*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mtheta, 0, total_ghost*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_E, 0, total_ghost*sizeof(double)));

    CUDA_CHECK(cudaMalloc(&d_Un, 4*n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_phi, n*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_phi, 0, n*sizeof(double)));

    CUDA_CHECK(cudaMalloc(&d_rho0, n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_P0, n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_phi0, n*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_rho0, 0, n*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_P0, 0, n*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_phi0, 0, n*sizeof(double)));

    CUDA_CHECK(cudaMalloc(&d_flux_r, (nr+1)*nt*4*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_flux_t, nr*(nt+1)*4*sizeof(double)));

    CUDA_CHECK(cudaMalloc(&d_residual, 4*n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Fk, 4*n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gmres_w, 4*n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gmres_Uk, 4*n*sizeof(double)));
    for (int i = 0; i <= GMRES_RESTART; ++i) {
        CUDA_CHECK(cudaMalloc(&d_gmres_V[i], 4*n*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_gmres_Z[i], 4*n*sizeof(double)));
    }

    int nb = (4*n + B - 1) / B;
    int buf_sz = std::max(4*n, nb);
    CUDA_CHECK(cudaMalloc(&d_work_dot_a, buf_sz*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_work_dot_b, buf_sz*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_work_grav_a, n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_work_grav_b, n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rhs_poisson, n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_residual_ls, 4*n*sizeof(double)));

    gmg.init(nr, nt, grid.r_face.data(), grid.theta_face.data());

    // ===== Build block-CSR sparsity pattern (5-point stencil) =====
    {
        std::vector<int> h_row(n + 1);
        std::vector<int> h_col;
        h_col.reserve(5 * n);
        h_row[0] = 0;
        for (int ii = 0; ii < nr; ++ii) {
            for (int jj = 0; jj < nt; ++jj) {
                int row = ii * nt + jj;
                std::vector<int> cols;
                if (ii > 0)     cols.push_back((ii-1)*nt + jj);
                if (jj > 0)     cols.push_back(ii*nt + (jj-1));
                cols.push_back(row); // diagonal
                if (jj < nt-1)  cols.push_back(ii*nt + (jj+1));
                if (ii < nr-1)  cols.push_back((ii+1)*nt + jj);
                std::sort(cols.begin(), cols.end());
                for (int c : cols) h_col.push_back(c);
                h_row[row + 1] = (int)h_col.size();
            }
        }
        jac_nnz = (int)h_col.size();

        CUDA_CHECK(cudaMalloc(&d_jac_row_ptr, (n+1)*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_jac_col_idx, jac_nnz*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_jac_values, jac_nnz*16*sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_jac_row_ptr, h_row.data(), (n+1)*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_jac_col_idx, h_col.data(), jac_nnz*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_jac_values, 0, jac_nnz*16*sizeof(double)));
    }

    // ===== AmgX initialization (persistent) =====
    {
        AMGX_initialize();

        // Inline config: AGGREGATION AMG, 1 V-cycle, DILU smoother, block_size=4
        const char* amgx_config_str =
            "config_version=2,"
            "solver(s)=AMG,"
            "s:algorithm=AGGREGATION,"
            "s:selector=SIZE_2,"
            "s:max_levels=20,"
            "s:cycle=V,"
            "s:max_iters=1,"
            "s:convergence=ABSOLUTE,"
            "s:tolerance=0.0,"
            "s:print_solve_stats=0,"
            "s:monitor_residual=0,"
            "s:smoother(sm)=MULTICOLOR_DILU,"
            "sm:max_iters=2,"
            "s:coarse_solver(cs)=DENSE_LU_SOLVER";
        AMGX_config_create(&amgx_cfg, amgx_config_str);
        AMGX_resources_create_simple(&amgx_rsrc, amgx_cfg);
        AMGX_matrix_create(&amgx_A, amgx_rsrc, AMGX_mode_dDDI);
        AMGX_vector_create(&amgx_b, amgx_rsrc, AMGX_mode_dDDI);
        AMGX_vector_create(&amgx_x, amgx_rsrc, AMGX_mode_dDDI);
        AMGX_solver_create(&amgx_solver, amgx_rsrc, AMGX_mode_dDDI, amgx_cfg);
        amgx_setup_done = false;
    }

    initialized = true;
    std::printf("GPU fully-implicit solver (JFNK+FGMRES+AmgX): %dx%d (%d cells), %d MG levels, nnz=%d\n",
                nr, nt, n, gmg.n_levels, jac_nnz);
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

// ========================= Time step ==============================

double GpuSolver::compute_cfl_dt() {
    int n = nr*nt, B = 256;
    k_cfl_dt<<<(n+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_E,
        d_dr,d_r_center,d_dtheta, d_work_grav_a, nr,nt,ng,gamma);

    int cur = n;
    double *src = d_work_grav_a, *dst = d_work_grav_b;
    while (cur > 1) {
        int nb = (cur+B-1)/B;
        k_reduce_min<<<nb,B,B*sizeof(double)>>>(src, dst, cur);
        cur = nb; double *t = src; src = dst; dst = t;
    }
    double val;
    CUDA_CHECK(cudaMemcpy(&val, src, sizeof(double), cudaMemcpyDeviceToHost));
    return cfl_num * val;
}

double GpuSolver::step(double t, double t_end) {
    int n = nr*nt, B = 256, N = 4*n;

    // Snapshot HSE on first call
    static bool hse_init = false;
    if (!hse_init) {
        solve_gravity();
        k_snapshot_hse<<<(n+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_E,d_phi,
            d_rho0,d_P0,d_phi0, nr,nt,ng,gamma);
        hse_init = true;
    }

    double dt_cfl = compute_cfl_dt();
    if (dt_adapt < 1e-30) dt_adapt = dt_cfl;
    double dt = std::min(dt_adapt, dt_cfl);
    if (t + dt > t_end) dt = t_end - t;

    // Save U^n
    pack_state(d_Un);

    int max_newton = 20;      // was 15 — allow more iterations before giving up
    int max_dt_cuts = 6;
    bool converged = false;
    int newton_iters_used = 0;
    int ls_fails_total = 0;   // track consecutive line search failures

    for (int cut = 0; cut < max_dt_cuts && !converged; ++cut) {
        if (cut > 0) {
            dt *= 0.5;
            // Restore U^n
            k_unpack_set<<<(n+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_E, d_Un, nr,nt,ng);
            if (step_count < 20)
                std::fprintf(stderr, "  step %d: Newton failed, cutting dt -> %.3e\n",
                            step_count, dt);
        }

        assemble_jacobian(dt);

        double Fnorm0 = 0.0;
        double Fnorm_prev = 0.0;
        bool diverged = false;
        ls_fails_total = 0;

        for (int newton = 0; newton < max_newton; ++newton) {
            solve_gravity();
            apply_floor();

            compute_F(d_Fk, dt);
            double Fnorm = sqrt(gpu_dot(d_Fk, d_Fk, d_work_dot_a, d_work_dot_b, N));

            if (newton == 0) {
                Fnorm0 = Fnorm;
                if (Fnorm < 1e-30) { converged = true; break; }
            }
            // Convergence: relative OR absolute per-cell tolerance
            double Fnorm_per_cell = Fnorm / sqrt((double)N);
            if (Fnorm < 1e-4 * Fnorm0 || Fnorm_per_cell < 1e-6) {
                converged = true;
                newton_iters_used = newton;
                if (step_count < 20)
                    std::fprintf(stderr, "  step %d converged at newton %d: ||F||=%.3e per-cell=%.3e (dt=%.3e)\n",
                                step_count, newton, Fnorm, Fnorm_per_cell, dt);
                break;
            }
            if (Fnorm > 1e6 * Fnorm0 || std::isnan(Fnorm)) { diverged = true; break; }

            // Eisenstat-Walker forcing: adapt GMRES tolerance based on Newton progress
            double eta_gmres;
            if (newton == 0 || Fnorm_prev < 1e-30) {
                eta_gmres = 1e-1;  // loose on first Newton iteration
            } else {
                // EW Choice 2: η_k = γ·(||F_k||/||F_{k-1}||)^α
                double ratio = Fnorm / Fnorm_prev;
                eta_gmres = 0.9 * ratio * ratio;   // γ=0.9, α=2
                eta_gmres = std::max(eta_gmres, 1e-4);  // don't over-solve
                eta_gmres = std::min(eta_gmres, 1e-1);  // don't under-solve
            }
            Fnorm_prev = Fnorm;

            int fgmres_iters = fgmres_solve(d_gmres_w, d_Fk, dt, eta_gmres, GMRES_RESTART);

            // Line search with softer failure handling
            k_pack<<<(n+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_E, d_gmres_Uk, nr,nt,ng);
            double alpha = 1.0;
            double Fnorm_new = Fnorm;
            for (int ls = 0; ls < 6; ++ls) {
                k_unpack_set<<<(n+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_E, d_gmres_Uk, nr,nt,ng);
                unpack_delta(d_gmres_w, alpha);
                apply_floor();
                solve_gravity();
                compute_F(d_residual_ls, dt);
                Fnorm_new = sqrt(gpu_dot(d_residual_ls, d_residual_ls, d_work_dot_a, d_work_dot_b, N));
                if (Fnorm_new < Fnorm) break;
                alpha *= 0.5;
            }

            if (Fnorm_new >= Fnorm) {
                // Soft failure: restore state but DON'T immediately break.
                // Allow a few consecutive line search failures before declaring divergence.
                k_unpack_set<<<(n+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_E, d_gmres_Uk, nr,nt,ng);
                ls_fails_total++;
                if (step_count < 20)
                    std::fprintf(stderr, "  step %d newton %d: line search stalled (fail %d), ||F||=%.3e\n",
                                step_count, newton, ls_fails_total, Fnorm);
                // Diverge only after 3 consecutive line search failures
                if (ls_fails_total >= 3) { diverged = true; break; }
                continue;
            }
            ls_fails_total = 0;  // reset on success

            if (step_count < 10)
                std::fprintf(stderr, "  step %d newton %d: ||F|| %.3e -> %.3e  a=%.2f  eta=%.1e  FGMRES %d  dt=%.3e\n",
                            step_count, newton, Fnorm, Fnorm_new, alpha, eta_gmres, fgmres_iters, dt);
        }

        if (diverged) {
            k_unpack_set<<<(n+B-1)/B,B>>>(d_rho,d_mr,d_mtheta,d_E, d_Un, nr,nt,ng);
        }
    }

    // ===== SER-style dt adaptation with dt_good memory =====
    if (converged) {
        // Track the best dt that worked
        if (dt > dt_good) dt_good = dt;

        // Growth factor based on Newton iteration count (SER-like)
        double growth;
        if (newton_iters_used <= 3)       growth = 2.0;   // easy convergence → aggressive growth
        else if (newton_iters_used <= 6)  growth = 1.5;   // moderate
        else if (newton_iters_used <= 10) growth = 1.2;   // hard convergence → cautious
        else                              growth = 1.05;  // barely converged → almost flat

        double dt_next = growth * dt;

        // Fast recovery: if we're well below dt_good, jump toward it.
        // Use conservative threshold (4x gap) to avoid overaggressive recovery.
        if (dt_good > 0.0 && dt < 0.25 * dt_good) {
            // Geometric mean between growth*dt and dt_good for faster recovery
            double dt_recover = sqrt(dt_next * dt_good);
            dt_next = std::max(dt_next, dt_recover);
        }

        dt_adapt = std::min(dt_next, dt_cfl);
    } else {
        // Failure: shrink dt AND decay dt_good toward current reality.
        // Without decay, dt_good becomes stale after physics changes and
        // the recovery mechanism creates a destructive propose-fail-crash cycle.
        dt_adapt = 0.5 * dt;
        if (dt_good > 0.0)
            dt_good = std::max(dt, 0.5 * dt_good);
    }

    step_count++;
    return dt;
}

// ========================= Destroy ================================

void GpuSolver::destroy() {
    if (!initialized) return;

    // AmgX cleanup
    AMGX_solver_destroy(amgx_solver);
    AMGX_matrix_destroy(amgx_A);
    AMGX_vector_destroy(amgx_b);
    AMGX_vector_destroy(amgx_x);
    AMGX_resources_destroy(amgx_rsrc);
    AMGX_config_destroy(amgx_cfg);
    AMGX_finalize();

    gmg.destroy();
    cudaFree(d_r_face); cudaFree(d_r_center); cudaFree(d_dr);
    cudaFree(d_theta_face); cudaFree(d_theta_center); cudaFree(d_dtheta);
    cudaFree(d_cell_volume); cudaFree(d_area_r); cudaFree(d_area_theta);
    cudaFree(d_rho); cudaFree(d_mr); cudaFree(d_mtheta); cudaFree(d_E);
    cudaFree(d_Un); cudaFree(d_phi);
    cudaFree(d_rho0); cudaFree(d_P0); cudaFree(d_phi0);
    cudaFree(d_flux_r); cudaFree(d_flux_t);
    cudaFree(d_residual); cudaFree(d_Fk);
    cudaFree(d_gmres_w); cudaFree(d_gmres_Uk);
    for (int i = 0; i <= GMRES_RESTART; ++i) {
        cudaFree(d_gmres_V[i]);
        cudaFree(d_gmres_Z[i]);
    }
    cudaFree(d_work_dot_a); cudaFree(d_work_dot_b);
    cudaFree(d_work_grav_a); cudaFree(d_work_grav_b);
    cudaFree(d_rhs_poisson); cudaFree(d_residual_ls);
    cudaFree(d_jac_row_ptr); cudaFree(d_jac_col_idx); cudaFree(d_jac_values);
    initialized = false;
}
