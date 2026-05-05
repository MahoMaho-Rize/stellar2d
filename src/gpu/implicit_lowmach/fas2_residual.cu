// FAS residual evaluation: ghost cells, gravity, HLLC flux divergence, floor, sponge, CFL

#include "fas2_solver.cuh"
#include "fas_hllc.cuh"
#include "fas_linalg.cuh"
#include "eos.h"
#include <cmath>
#include <vector>
#include <algorithm>

// ========================= Ghost cells ============================

__global__ void k_fas2_ghost_r_in(double* rho, double* mr, double* mt, double* rhoE,
                                  int nr, int nt, int ng) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y + 1;
    if (j >= nt || g > ng) return;
    int kg = fas_idx(-g,j,nt,ng), kp = fas_idx(g-1,j,nt,ng);
    rho[kg]=rho[kp]; mr[kg]=-mr[kp]; mt[kg]=mt[kp]; rhoE[kg]=rhoE[kp];
}

__global__ void k_fas2_ghost_r_out(double* rho, double* mr, double* mt, double* rhoE,
                                   int nr, int nt, int ng) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y;
    if (j >= nt || g >= ng) return;
    int kg = fas_idx(nr+g,j,nt,ng), kp = fas_idx(nr-1,j,nt,ng);
    rho[kg]=rho[kp]; mr[kg]=mr[kp]; mt[kg]=mt[kp]; rhoE[kg]=rhoE[kp];
}

__global__ void k_fas2_ghost_r_out_hse(double* rho, double* mr, double* mt, double* rhoE,
                                       const double* rho0, const double* P0,
                                       EOS eos,
                                       int nr, int nt, int ng) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y;
    if (j >= nt || g >= ng) return;
    int kg = fas_idx(nr+g, j, nt, ng);
    int flat_last = (nr-1)*nt + j;
    double rho_r = fmax(rho0[flat_last], 1e-20);
    rho[kg]  = rho_r;
    mr[kg]   = 0.0;
    mt[kg]   = 0.0;
    rhoE[kg] = fmax(rho_r * eos.internal_energy(rho_r, P0[flat_last]), 1e-20);
}

__global__ void k_fas2_ghost_t_n(double* rho, double* mr, double* mt, double* rhoE,
                                 int nr, int nt, int ng) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y + 1;
    if (ii >= nr+2*ng || g > ng) return;
    int i = ii - ng;
    int kg = fas_idx(i,-g,nt,ng), kp = fas_idx(i,g-1,nt,ng);
    rho[kg]=rho[kp]; mr[kg]=mr[kp]; mt[kg]=-mt[kp]; rhoE[kg]=rhoE[kp];
}

__global__ void k_fas2_ghost_t_s(double* rho, double* mr, double* mt, double* rhoE,
                                 int nr, int nt, int ng) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y;
    if (ii >= nr+2*ng || g >= ng) return;
    int i = ii - ng;
    int kg = fas_idx(i,nt+g,nt,ng), kp = fas_idx(i,nt-1-g,nt,ng);
    rho[kg]=rho[kp]; mr[kg]=mr[kp]; mt[kg]=-mt[kp]; rhoE[kg]=rhoE[kp];
}

__global__ void k_fas2_pole_lock(double* mt, int nr, int nt, int ng) {
    int i = blockIdx.x * blockDim.x + threadIdx.x - ng;
    if (i < -ng || i >= nr + ng) return;
    mt[fas_idx(i, 0, nt, ng)] = 0.0;
    mt[fas_idx(i, nt - 1, nt, ng)] = 0.0;
}

void FasSolver2::launch_ghost(int l) {
    FasLevel2& lev = levels[l];
    int B=256;
    { dim3 g((lev.nt+B-1)/B, lev.ng); k_fas2_ghost_r_in<<<g,B>>>(lev.d_rho,lev.d_mr,lev.d_mt,lev.d_rhoE,lev.nr,lev.nt,lev.ng); }
    if (use_hse_outer_bc && hse_set) {
        dim3 g((lev.nt+B-1)/B, lev.ng);
        k_fas2_ghost_r_out_hse<<<g,B>>>(lev.d_rho,lev.d_mr,lev.d_mt,lev.d_rhoE,
            lev.d_rho0, lev.d_P0, eos, lev.nr,lev.nt,lev.ng);
    } else {
        dim3 g((lev.nt+B-1)/B, lev.ng);
        k_fas2_ghost_r_out<<<g,B>>>(lev.d_rho,lev.d_mr,lev.d_mt,lev.d_rhoE,lev.nr,lev.nt,lev.ng);
    }
    { dim3 g((lev.nr+2*lev.ng+B-1)/B, lev.ng); k_fas2_ghost_t_n<<<g,B>>>(lev.d_rho,lev.d_mr,lev.d_mt,lev.d_rhoE,lev.nr,lev.nt,lev.ng); }
    { dim3 g((lev.nr+2*lev.ng+B-1)/B, lev.ng); k_fas2_ghost_t_s<<<g,B>>>(lev.d_rho,lev.d_mr,lev.d_mt,lev.d_rhoE,lev.nr,lev.nt,lev.ng); }
    int ntot = lev.nr + 2*lev.ng;
    k_fas2_pole_lock<<<(ntot+B-1)/B,B>>>(lev.d_mt, lev.nr, lev.nt, lev.ng);
}

// ========================= 1D radial gravity ========================

__global__
void k_fas2_shell_mass(const double* rho, const double* vol,
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
void k_fas2_gravity_from_shells(const double* shell_mass, const double* rc,
                                double* gr, int nr, double G,
                                double M_core) {
    extern __shared__ double smem[];
    int tid = threadIdx.x;
    smem[tid] = (tid < nr) ? shell_mass[tid] : 0.0;
    __syncthreads();

    // Inclusive prefix sum (Blelloch-style up-sweep + down-sweep)
    int n = blockDim.x;
    // Up-sweep
    for (int d = 1; d < n; d <<= 1) {
        int idx = (tid + 1) * (d << 1) - 1;
        if (idx < n) smem[idx] += smem[idx - d];
        __syncthreads();
    }
    // Down-sweep for inclusive scan
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
    // smem[tid] is now exclusive prefix sum; add shell_mass[tid] for inclusive
    if (tid < nr) {
        double M_enc = M_core + smem[tid] + shell_mass[tid];
        gr[tid] = -G * M_enc / (rc[tid] * rc[tid]);
    }
}

void FasSolver2::compute_gravity_1d(int l) {
    FasLevel2& lev = levels[l];
    int B = std::min(lev.nt, 256);
    k_fas2_shell_mass<<<lev.nr, B, B*sizeof(double)>>>(
        lev.d_rho, lev.d_cell_volume, lev.d_shell_mass, lev.nr, lev.nt, lev.ng);
    // Round up to power of 2 for prefix scan
    int np2 = 1;
    while (np2 < lev.nr) np2 <<= 1;
    k_fas2_gravity_from_shells<<<1, np2, np2*sizeof(double)>>>(
        lev.d_shell_mass, lev.d_r_center, lev.d_gr, lev.nr, G_const, M_core);
}

// ========================= Nonlinear residual R(U) ========================
// Standard kernel for i >= 1 (all non-singular cells).

__global__
void k_fas2_residual(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* vol, const double* ar, const double* at,
    const double* r_center, const double* r_face,
    const double* theta_face, const double* dr, const double* dtheta,
    const double* gr, const double* gr0,
    const double* P0, const double* rho0,
    double* res,
    int nr, int nt, int ng, EOS eos, double atm_thresh,
    int use_wellbalance, int lim_type, int hllc_variant, int radial_only) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    int n = nr*nt;

    // i==0 handled by k_fas2_residual_origin (unless core excision: r_face[0]>0)
    if (i == 0 && r_face[0] < 1e-30) return;

    // All cells use WB — prevents O(h²/ρ) spurious acceleration in low-density surface
    int wb_local = use_wellbalance;

    int k = fas_idx(i,j,nt,ng);
    double invV = 1.0 / vol[flat];

    double rho_c = fmax(rho[k], 1e-20);
    double vr_c = mr[k] / rho_c;
    double vt_c = mt[k] / rho_c;
    double KE_c = 0.5 * rho_c * (vr_c*vr_c + vt_c*vt_c);
    double e_c = fmax((rhoE[k] - KE_c) / rho_c, 1e-30);
    double P_c = fmax(eos.pressure(rho_c, e_c), 1e-30);
    double r = r_center[i];

    // ===== Flux-level WB-MUSCL+HLLC =====
    // Helper: primitives from FAS state (rhoE = total energy E = ρe + ½ρv²)
    auto W = [&](int ii, int jj) -> FPrim {
        int kk = fas_idx(ii,jj,nt,ng);
        double r_ = fmax(rho[kk], 1e-20);
        double vr_ = mr[kk] / r_, vt_ = mt[kk] / r_;
        double e_ = fmax((rhoE[kk] - 0.5*r_*(vr_*vr_ + vt_*vt_)) / r_, 1e-30);
        FPrim w;
        w.rho = r_;
        w.vr = vr_;
        w.vt = vt_;
        w.P = fmax(eos.pressure(r_, e_), 1e-30);
        return w;
    };
    int wb = use_wellbalance;
    auto P0r = [&](int ii) -> double { return wb ? P0[max(0,min(ii,nr-1))*nt+j] : 0.0; };
    auto r0r = [&](int ii) -> double { return wb ? rho0[max(0,min(ii,nr-1))*nt+j] : 0.0; };
    auto P0t = [&](int jj) -> double { return wb ? P0[i*nt+max(0,min(jj,nt-1))] : 0.0; };
    auto r0t = [&](int jj) -> double { return wb ? rho0[i*nt+max(0,min(jj,nt-1))] : 0.0; };

    // --- r-direction faces ---
    auto hllc_r_face = [&](int face_i) -> FFlux4 {
        FPrim wa=W(face_i-2,j), wb_=W(face_i-1,j), wc=W(face_i,j), wd=W(face_i+1,j);
        double ppL,ppR,rpL,rpR;
        fas_recon(wa.P-P0r(face_i-2), wb_.P-P0r(face_i-1), wc.P-P0r(face_i), wd.P-P0r(face_i+1), ppL, ppR, lim_type);
        fas_recon(wa.rho-r0r(face_i-2), wb_.rho-r0r(face_i-1), wc.rho-r0r(face_i), wd.rho-r0r(face_i+1), rpL, rpR, lim_type);
        double P0f = 0.5*(P0r(face_i-1)+P0r(face_i)), r0f = 0.5*(r0r(face_i-1)+r0r(face_i));
        FPrim wl, wr;
        fas_recon(wa.vr, wb_.vr, wc.vr, wd.vr, wl.vr, wr.vr, lim_type);
        fas_recon(wa.vt, wb_.vt, wc.vt, wd.vt, wl.vt, wr.vt, lim_type);
        wl.rho = fmax(r0f+rpL, 1e-20); wr.rho = fmax(r0f+rpR, 1e-20);
        wl.P = fmax(P0f+ppL, 1e-30); wr.P = fmax(P0f+ppR, 1e-30);
        return fas_hllc_dispatch(wl, wr, eos, true, hllc_variant);
    };
    FFlux4 fr_hi = hllc_r_face(i+1);
    FFlux4 fr_lo = hllc_r_face(i);

    double Ar_hi=ar[(i+1)*nt+j], Ar_lo=ar[i*nt+j];

    // Flux-level WB: subtract P₀_face from radial momentum flux
    double P0f_rhi = 0.5*(P0r(i) + P0r(i+1));
    double P0f_rlo = 0.5*(P0r(i-1) + P0r(i));

    // Radial divergence
    double div_rho = -invV*(Ar_hi*fr_hi.f_rho - Ar_lo*fr_lo.f_rho);
    double div_mr  = -invV*(Ar_hi*(fr_hi.f_mr - P0f_rhi) - Ar_lo*(fr_lo.f_mr - P0f_rlo));
    double div_mt  = -invV*(Ar_hi*fr_hi.f_mt - Ar_lo*fr_lo.f_mt);
    double div_E   = -invV*(Ar_hi*fr_hi.f_E - Ar_lo*fr_lo.f_E);

    // --- θ-direction faces (skipped when radial_only) ---
    if (!radial_only) {
        auto hllc_t_face = [&](int face_j) -> FFlux4 {
            FPrim wa=W(i,face_j-2), wb_=W(i,face_j-1), wc=W(i,face_j), wd=W(i,face_j+1);
            double ppL,ppR,rpL,rpR;
            fas_recon(wa.P-P0t(face_j-2), wb_.P-P0t(face_j-1), wc.P-P0t(face_j), wd.P-P0t(face_j+1), ppL, ppR, lim_type);
            fas_recon(wa.rho-r0t(face_j-2), wb_.rho-r0t(face_j-1), wc.rho-r0t(face_j), wd.rho-r0t(face_j+1), rpL, rpR, lim_type);
            double P0f = 0.5*(P0t(face_j-1)+P0t(face_j)), r0f = 0.5*(r0t(face_j-1)+r0t(face_j));
            FPrim wl, wr;
            fas_recon(wa.vr, wb_.vr, wc.vr, wd.vr, wl.vr, wr.vr, lim_type);
            fas_recon(wa.vt, wb_.vt, wc.vt, wd.vt, wl.vt, wr.vt, lim_type);
            wl.rho = fmax(r0f+rpL, 1e-20); wr.rho = fmax(r0f+rpR, 1e-20);
            wl.P = fmax(P0f+ppL, 1e-30); wr.P = fmax(P0f+ppR, 1e-30);
            return fas_hllc_dispatch(wl, wr, eos, false, hllc_variant);
        };
        FFlux4 ft_hi = hllc_t_face(j+1);
        FFlux4 ft_lo = hllc_t_face(j);

        double At_hi=at[i*(nt+1)+j+1], At_lo=at[i*(nt+1)+j];
        double P0f_thi = 0.5*(P0t(j) + P0t(j+1));
        double P0f_tlo = 0.5*(P0t(j-1) + P0t(j));

        div_rho += -invV*(At_hi*ft_hi.f_rho - At_lo*ft_lo.f_rho);
        div_mr  += -invV*(At_hi*ft_hi.f_mr  - At_lo*ft_lo.f_mr);
        div_mt  += -invV*(At_hi*(ft_hi.f_mt - P0f_thi) - At_lo*(ft_lo.f_mt - P0f_tlo));
        div_E   += -invV*(At_hi*ft_hi.f_E   - At_lo*ft_lo.f_E);
    }

    // WB gravity source: ρ'g₀ + ρg'
    double rho_ref = wb ? rho0[flat] : 0.0;
    double g0_r = wb ? gr0[i] : 0.0;
    double rhop_c = rho_c - rho_ref;
    double gp_r = gr[i] - g0_r;
    double gravity_r = rhop_c*g0_r + rho_c*gp_r;

    // Geometric source (zero under radial_only since v_theta=0)
    double inv_r = 1.0 / r;
    double S_mr = radial_only ? 0.0 : rho_c * vt_c * vt_c * inv_r;
    double S_mt = radial_only ? 0.0 : -rho_c * vr_c * vt_c * inv_r;

    // Total energy source: ρv·g (gravity does work)
    double S_E = rho_c * vr_c * (g0_r + gp_r);

    res[flat]       = div_rho;
    res[n + flat]   = div_mr + gravity_r + S_mr;
    res[2*n + flat] = radial_only ? 0.0 : (div_mt + S_mt);
    res[3*n + flat] = div_E + S_E;
}

// ========================= Origin kernel (i=0, divergence theorem) ========
// The innermost cell is a pie-slice wedge touching r=0.
// Inner face has zero area (point). Only the outer radial face carries flux.
// Volume = (1/3)*r_face[1]^3 * (cos(θ_lo) - cos(θ_hi))
// Outer area = r_face[1]^2 * (cos(θ_lo) - cos(θ_hi))
// No 1/r source terms — use exact integral: ∫ρv²/r dV = ρv² * 0.5*r_face[1]^2*(cosθ_lo-cosθ_hi)
// which divided by V gives 1.5/r_face[1] — the volume-averaged <1/r>.

__global__
void k_fas2_residual_origin(
    const double* rho, const double* mr, const double* mt, const double* rhoE,
    const double* vol, const double* ar, const double* at,
    const double* r_center, const double* r_face,
    const double* theta_face, const double* dr, const double* dtheta,
    const double* gr, const double* gr0,
    const double* P0, const double* rho0,
    double* res,
    int nr, int nt, int ng, EOS eos, double atm_thresh,
    int use_wellbalance, int lim_type, int hllc_variant, int radial_only) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nt) return;
    int i = 0;
    int flat = j;
    int n = nr*nt;

    int k = fas_idx(0, j, nt, ng);
    double invV = 1.0 / vol[flat];

    double rho_c = fmax(rho[k], 1e-20);
    double vr_c = mr[k] / rho_c;
    double vt_c = mt[k] / rho_c;
    double KE_c = 0.5 * rho_c * (vr_c*vr_c + vt_c*vt_c);
    double e_c = fmax((rhoE[k] - KE_c) / rho_c, 1e-30);
    double P_c = fmax(eos.pressure(rho_c, e_c), 1e-30);

    double Ar_hi = ar[1*nt + j];
    double At_hi = at[0*(nt+1) + j+1], At_lo = at[0*(nt+1) + j];

    // Helper to get primitives (rhoE = total energy)
    auto W = [&](int ii, int jj) -> FPrim {
        int kk = fas_idx(ii,jj,nt,ng);
        double r_ = fmax(rho[kk], 1e-20);
        double vr_ = mr[kk]/r_, vt_ = mt[kk]/r_;
        double e_ = fmax((rhoE[kk] - 0.5*r_*(vr_*vr_+vt_*vt_)) / r_, 1e-30);
        FPrim w; w.rho = r_; w.vr = vr_; w.vt = vt_;
        w.P = fmax(eos.pressure(r_, e_), 1e-30); return w;
    };
    int wb = use_wellbalance;
    auto P0r = [&](int ii) -> double { return wb ? P0[max(0,min(ii,nr-1))*nt+j] : 0.0; };
    auto r0r = [&](int ii) -> double { return wb ? rho0[max(0,min(ii,nr-1))*nt+j] : 0.0; };
    auto P0t = [&](int jj) -> double { return wb ? P0[max(0,min(jj,nt-1))] : 0.0; };
    auto r0t = [&](int jj) -> double { return wb ? rho0[max(0,min(jj,nt-1))] : 0.0; };

    // Outer r-face: first-order HLLC (no MUSCL, stencil too short at origin)
    {
        FPrim wl = W(0,j), wr = W(1,j);
        double P0f = 0.5*(P0r(0)+P0r(1)), r0f = 0.5*(r0r(0)+r0r(1));
        wl.P = fmax(P0f + (wl.P - P0r(0)), 1e-30);
        wr.P = fmax(P0f + (wr.P - P0r(1)), 1e-30);
        wl.rho = fmax(r0f + (wl.rho - r0r(0)), 1e-20);
        wr.rho = fmax(r0f + (wr.rho - r0r(1)), 1e-20);
        FFlux4 fr_hi = fas_hllc_dispatch(wl, wr, eos, true, hllc_variant);

        // θ-face fluxes: computed for symmetry with k_fas2_residual but unused in origin cell
        // (origin cell is a wedge touching r=0; all θ-cells share the same point).
        // Skip entirely under radial_only.
        if (!radial_only) {
            auto hllc_t_face = [&](int face_j) -> FFlux4 {
                FPrim wa_=W(0,face_j-2), wb_=W(0,face_j-1), wc_=W(0,face_j), wd_=W(0,face_j+1);
                double ppL,ppR,rpL,rpR;
                fas_recon(wa_.P-P0t(face_j-2), wb_.P-P0t(face_j-1), wc_.P-P0t(face_j), wd_.P-P0t(face_j+1), ppL, ppR, lim_type);
                fas_recon(wa_.rho-r0t(face_j-2), wb_.rho-r0t(face_j-1), wc_.rho-r0t(face_j), wd_.rho-r0t(face_j+1), rpL, rpR, lim_type);
                double P0ff = 0.5*(P0t(face_j-1)+P0t(face_j)), r0ff = 0.5*(r0t(face_j-1)+r0t(face_j));
                FPrim wll, wrr;
                fas_recon(wa_.vr, wb_.vr, wc_.vr, wd_.vr, wll.vr, wrr.vr, lim_type);
                fas_recon(wa_.vt, wb_.vt, wc_.vt, wd_.vt, wll.vt, wrr.vt, lim_type);
                wll.rho = fmax(r0ff+rpL, 1e-20); wrr.rho = fmax(r0ff+rpR, 1e-20);
                wll.P = fmax(P0ff+ppL, 1e-30); wrr.P = fmax(P0ff+ppR, 1e-30);
                return fas_hllc_dispatch(wll, wrr, eos, false, hllc_variant);
            };
            (void)hllc_t_face(j+1);
            (void)hllc_t_face(j);
        }

        // Flux-level WB: subtract HSE background pressure from radial momentum flux
        double P0f_rhi = 0.5*(P0r(0) + P0r(1));

        // Origin cell: only outer radial face has nonzero area (inner face at r=0 has A=0).
        // No theta flux (all j-cells share the same point r=0).
        // The reflecting inner BC means the inner face contributes a pressure force
        // P_center * A_inner, but A_inner = 0, so no mass/energy flux from inner face.
        // However, the momentum flux needs the pressure at r=0 pushing outward:
        // physically, the center is a reflecting wall with P = P_cell.
        // For the divergence theorem on a wedge touching r=0:
        //   ∫∇·F dV = F_outer·A_outer  (no inner face contribution for mass/energy)
        //   For momentum: add P_center * A_inner = 0 (area is zero at r=0)
        // So the formulas are correct as written — single-sided flux.
        double div_rho = -invV*(Ar_hi*fr_hi.f_rho);
        double div_mr  = -invV*(Ar_hi*(fr_hi.f_mr - P0f_rhi));
        double div_mt  = 0.0;
        double div_E   = -invV*(Ar_hi*fr_hi.f_E);

        double rho_ref = wb ? rho0[flat] : 0.0;
        double g0_r = wb ? gr0[0] : 0.0;
        double rhop_c = rho_c - rho_ref;
        double gp_r = gr[0] - g0_r;
        double gravity_r = rhop_c*g0_r + rho_c*gp_r;

        double S_E = rho_c * vr_c * (g0_r + gp_r);

        res[flat]       = div_rho;
        res[n + flat]   = div_mr + gravity_r;
        res[2*n + flat] = 0.0;
        res[3*n + flat] = div_E + S_E;
        return;
    }
}

// Transport step: U += ω_cfl · dt_cfl_local · R(U)
//
// R(U) is the spatial residual (HLLC flux divergence + gravity sources).
// dU/dt = R(U), so this is a forward-Euler pseudo-time step at the local
// signal speed.  Each sweep propagates wave errors ~ω_cfl grid cells.
// Uses R (not F) because F contains a 1/dt time term that would blow up.
__global__
void k_fas2_transport_step(
    double* rho, double* mr, double* mt, double* rhoE,
    const double* R,
    const double* dr, const double* r_center, const double* r_face,
    const double* dtheta,
    const double* rho0, double atm_thresh,
    int nr, int nt, int ng, EOS eos, double omega_cfl) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    if (rho0[flat] < atm_thresh) return;
    int i = flat/nt, j = flat%nt;
    int k = fas_idx(i,j,nt,ng);
    int n = nr*nt;

    double rho_c = fmax(rho[k], 1e-20);
    double vr = fabs(mr[k] / rho_c);
    double vt = fabs(mt[k] / rho_c);
    double KE = 0.5 * rho_c * (vr*vr + vt*vt);
    double e_c = fmax((rhoE[k] - KE) / rho_c, 1e-30);
    double P = fmax(eos.pressure(rho_c, e_c), 1e-30);
    double cs = eos.sound_speed(rho_c, P);
    double r_eff = (i == 0 && r_face[1] > 1e-30) ? (2.0/3.0)*r_face[1] : r_center[i];
    double dt_r = dr[i] / (vr + cs);
    double dt_t = r_eff * dtheta[j] / (vt + cs);
    double dtau = omega_cfl * fmin(dt_r, dt_t);

    rho[k]  += dtau * R[flat];
    mr[k]   += dtau * R[n + flat];
    mt[k]   += dtau * R[2*n + flat];
    rhoE[k] += dtau * R[3*n + flat];
}

void FasSolver2::compute_residual(int l) {
    FasLevel2& lev = levels[l];
    int n = lev.nr * lev.nt, B = 256;
    launch_ghost(l);
    compute_gravity_1d(l);
    int wb = 1;
    k_fas2_residual<<<(n+B-1)/B,B>>>(
        lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE,
        lev.d_cell_volume, lev.d_area_r, lev.d_area_theta,
        lev.d_r_center, lev.d_r_face, lev.d_theta_face,
        lev.d_dr, lev.d_dtheta,
        lev.d_gr, lev.d_gr0, lev.d_P0, lev.d_rho0,
        lev.d_res,
        lev.nr, lev.nt, lev.ng, eos, atm_rho_thresh, wb, limiter_type, hllc_variant,
        (int)radial_only);
    if (!use_core_excision) {
        k_fas2_residual_origin<<<(lev.nt+B-1)/B,B>>>(
            lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE,
            lev.d_cell_volume, lev.d_area_r, lev.d_area_theta,
            lev.d_r_center, lev.d_r_face, lev.d_theta_face,
            lev.d_dr, lev.d_dtheta,
            lev.d_gr, lev.d_gr0, lev.d_P0, lev.d_rho0,
            lev.d_res,
            lev.nr, lev.nt, lev.ng, eos, atm_rho_thresh, wb, limiter_type, hllc_variant,
            (int)radial_only);
    }
    // Subtract pre-computed HSE defect: R_corrected(U) = R(U) - R(U₀)
    // This ensures R(U₀) = 0 exactly, fixing WB inconsistency on coarse grids.
    k_fas_axpy<<<(4*n+B-1)/B,B>>>(lev.d_res, -1.0, lev.d_hse_defect, 4*n);
}

// ========================= Newton residual F(U) ========================
// F = Uⁿ/dt - U/dt + R(U) + fas_rhs_correction
// We store it in d_res (overwrite spatial residual).

__global__
void k_fas2_compute_F(double* F, const double* R, const double* rho, const double* mr,
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

void FasSolver2::compute_F(int l, double g0_over_dt) {
    FasLevel2& lev = levels[l];
    int n = lev.nr * lev.nt, B = 256;
    compute_residual(l);
    k_fas2_compute_F<<<(n+B-1)/B,B>>>(
        lev.d_res, lev.d_res, lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE,
        lev.d_fas_rhs, g0_over_dt, lev.nr, lev.nt, lev.ng);
}
// ========================= Floor ========================

__global__
void k_fas2_floor(double* rho, double* mr, double* mt, double* rhoE,
                 int nr, int nt, int ng, double gam) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int k = fas_idx(flat/nt, flat%nt, nt, ng);
    double r = rho[k], E = rhoE[k];
    if (isnan(r) || isnan(E) || isnan(mr[k]) || isnan(mt[k])) {
        rho[k] = 1e-20; mr[k] = 0.0; mt[k] = 0.0; rhoE[k] = 1e-20;
        return;
    }
    const double rho_fl = 1e-20;
    r = 0.5 * (r + sqrt(r*r + 4.0*rho_fl*rho_fl));
    r = fmax(r, rho_fl);
    rho[k] = r;
    double KE = 0.5 * (mr[k]*mr[k] + mt[k]*mt[k]) / r;
    double e_int = E - KE;
    const double e_fl = 1e-20;
    e_int = 0.5 * (e_int + sqrt(e_int*e_int + 4.0*e_fl*e_fl));
    e_int = fmax(e_int, e_fl);
    rhoE[k] = e_int + KE;
}

void FasSolver2::apply_floor(int l) {
    FasLevel2& lev = levels[l];
    int n = lev.nr * lev.nt, B = 256;
    k_fas2_floor<<<(n+B-1)/B,B>>>(lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE,
                                   lev.nr, lev.nt, lev.ng, gamma);
}

// Zero theta-momentum everywhere (incl. ghosts) — enforces v_theta=0 invariant
__global__
void k_fas2_zero_mt(double* mt, int nr, int nt, int ng) {
    int total = (nr + 2*ng) * (nt + 2*ng);
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= total) return;
    mt[k] = 0.0;
}

// ========================= Atmosphere reset (MUSIC-style) ====================
// Force cells with ρ₀ < atm_rho_thresh to exact HSE state: ρ=ρ₀, v=0, E=P₀/(γ-1).
// Prevents numerical noise in near-vacuum from poisoning the implicit solve.

__global__
void k_fas2_atm_reset(double* rho, double* mr, double* mt, double* rhoE,
                     const double* rho0, const double* P0,
                     double atm_thresh, EOS eos,
                     int nr, int nt, int ng, int strict_atm_only) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int k = fas_idx(flat/nt, flat%nt, nt, ng);

    bool is_atm = rho0[flat] < atm_thresh;
    // evacuated trigger: injects mass when rho drops below 1% of rho0 — disabled in strict mode
    // (non-conservative; main source of baseline mass drift).
    bool evacuated = (!strict_atm_only) && rho[k] < 0.01 * fmax(rho0[flat], 1e-20);

    // Hot vacuum (T/T₀ > 100 or T/T₀ < 1e-4): numerical safety valve against pressure
    // divergence. Only triggers in pathological states; essential for stability near boundaries.
    // Low-T branch catches the case where rhoE - KE has collapsed to near zero due to roundoff
    // (e.g. outer cells where internal energy << kinetic energy).
    bool hot_vacuum = false;
    if (!is_atm && !evacuated) {
        double rho_c = fmax(rho[k], 1e-20);
        double vr = mr[k] / rho_c, vt = mt[k] / rho_c;
        double e_c = fmax((rhoE[k] - 0.5*rho_c*(vr*vr+vt*vt)) / rho_c, 1e-30);
        double P_c = fmax(eos.pressure(rho_c, e_c), 0.0);
        double T_ratio = (P_c * rho0[flat]) / fmax(P0[flat] * rho_c, 1e-30);
        hot_vacuum = (T_ratio > 100.0) || (T_ratio < 1e-4);
    }

    if (!is_atm && !evacuated && !hot_vacuum) return;
    rho[k]  = fmax(rho0[flat], 1e-20);
    mr[k]   = 0.0;
    mt[k]   = 0.0;
    double rho_r = fmax(rho0[flat], 1e-20);
    rhoE[k] = fmax(rho_r * eos.internal_energy(rho_r, P0[flat]), 1e-20);
}

// ========================= Conservation helpers ==============================
// Compute ρ·V and E·V per cell (for global reduce before/after atm_reset)
__global__
void k_fas2_rhoV_EV(const double* rho, const double* rhoE, const double* vol,
                   double* rhoV_out, double* EV_out,
                   int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int k = fas_idx(flat/nt, flat%nt, nt, ng);
    double V = vol[flat];
    rhoV_out[flat] = rho[k] * V;
    EV_out[flat]   = rhoE[k] * V;
}

// Distribute mass/energy correction over interior cells, proportional to cell volume.
// delta_rho_per_vol = ΔM / V_interior  (density correction, uniform)
// delta_E_per_vol   = ΔE / V_interior  (energy density correction, uniform)
__global__
void k_fas2_conserve_correct(double* rho, double* rhoE,
                            const double* rho0,
                            double atm_thresh,
                            double delta_rho_per_vol, double delta_E_per_vol,
                            int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    if (rho0[flat] < atm_thresh) return;
    int k = fas_idx(flat/nt, flat%nt, nt, ng);
    rho[k]  += delta_rho_per_vol;
    rhoE[k] += delta_E_per_vol;
}

// ========================= Angular averaging near origin ======================
// Volume-weighted θ-average for shells i < n_avg.
// Eliminates geometric focusing artifacts at r→0 (PLUTO-style axis smoothing).
// One block per shell (i), threads reduce over j.

__global__
void k_fas2_angular_avg(double* rho, double* mr, double* mt, double* rhoE,
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
        double v = vol[i * nt + j];
        sum_rho  += rho[k] * v;
        sum_mr   += mr[k] * v;
        sum_mt   += mt[k] * v;
        sum_rhoE += rhoE[k] * v;
        sum_vol  += v;
    }
    s_rho[tid] = sum_rho; s_mr[tid] = sum_mr;
    s_mt[tid] = sum_mt; s_rhoE[tid] = sum_rhoE;
    s_vol[tid] = sum_vol;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_rho[tid] += s_rho[tid + s];
            s_mr[tid]  += s_mr[tid + s];
            s_mt[tid]  += s_mt[tid + s];
            s_rhoE[tid]+= s_rhoE[tid + s];
            s_vol[tid] += s_vol[tid + s];
        }
        __syncthreads();
    }

    double inv_V = 1.0 / fmax(s_vol[0], 1e-30);
    double avg_rho  = s_rho[0] * inv_V;
    double avg_mr   = s_mr[0] * inv_V;
    double avg_mt   = s_mt[0] * inv_V;
    double avg_rhoE = s_rhoE[0] * inv_V;

    for (int j = tid; j < nt; j += blockDim.x) {
        int k = fas_idx(i, j, nt, ng);
        rho[k]  = avg_rho;
        mr[k]   = avg_mr;
        mt[k]   = 0.0;  // no θ-momentum at origin
        rhoE[k] = avg_rhoE;
    }
}

// ========================= Pole wedge averaging ==============================
// Merge n_pole cells near each pole (j=0..n_pole-1 and j=nt-n_pole..nt-1)
// into volume-weighted averages. Fixes polar axis numerical focusing where
// At/V ~ 1/dθ amplifies any asymmetry near θ=0,π.
// One block per radial shell, threads reduce over the n_pole cells.

__global__
void k_fas2_pole_avg(double* rho, double* mr, double* mt, double* rhoE,
                    const double* vol,
                    int n_pole, int nr, int nt, int ng) {
    int i = blockIdx.x;
    if (i >= nr) return;

    // North pole: j = 0 .. n_pole-1
    {
        double sum_rho = 0, sum_mr = 0, sum_rhoE = 0, sum_vol = 0;
        for (int j = 0; j < n_pole && j < nt; ++j) {
            int k = fas_idx(i, j, nt, ng);
            double v = vol[i * nt + j];
            sum_rho  += rho[k] * v;
            sum_mr   += mr[k] * v;
            sum_rhoE += rhoE[k] * v;
            sum_vol  += v;
        }
        double inv_V = 1.0 / fmax(sum_vol, 1e-30);
        double avg_rho  = sum_rho * inv_V;
        double avg_mr   = sum_mr * inv_V;
        double avg_rhoE = sum_rhoE * inv_V;
        for (int j = 0; j < n_pole && j < nt; ++j) {
            int k = fas_idx(i, j, nt, ng);
            rho[k]  = avg_rho;
            mr[k]   = avg_mr;
            mt[k]   = 0.0;
            rhoE[k] = avg_rhoE;
        }
    }

    // South pole: j = nt-n_pole .. nt-1
    {
        double sum_rho = 0, sum_mr = 0, sum_rhoE = 0, sum_vol = 0;
        for (int j = nt - n_pole; j < nt; ++j) {
            if (j < 0) continue;
            int k = fas_idx(i, j, nt, ng);
            double v = vol[i * nt + j];
            sum_rho  += rho[k] * v;
            sum_mr   += mr[k] * v;
            sum_rhoE += rhoE[k] * v;
            sum_vol  += v;
        }
        double inv_V = 1.0 / fmax(sum_vol, 1e-30);
        double avg_rho  = sum_rho * inv_V;
        double avg_mr   = sum_mr * inv_V;
        double avg_rhoE = sum_rhoE * inv_V;
        for (int j = nt - n_pole; j < nt; ++j) {
            if (j < 0) continue;
            int k = fas_idx(i, j, nt, ng);
            rho[k]  = avg_rho;
            mr[k]   = avg_mr;
            mt[k]   = 0.0;
            rhoE[k] = avg_rhoE;
        }
    }
}

// ========================= Central radial damping ============================
// In the angular-averaged zone (i < n_damp), damp radial velocity to prevent
// geometric convergence runaway at r→0. Removes kinetic energy from v_r
// while preserving density and pressure (no mass injection).
// Damping factor: v_r *= exp(-alpha * (1 - r/r_damp)) for r < r_damp

__global__
void k_fas2_central_damp(double* mr, double* rhoE,
                        const double* rho, const double* r_center,
                        double r_damp, double alpha,
                        int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr * nt) return;
    int i = flat / nt, j = flat % nt;
    double r = r_center[i];
    if (r >= r_damp) return;

    double f = exp(-alpha * (1.0 - r / r_damp));
    int k = fas_idx(i, j, nt, ng);
    double rho_c = fmax(rho[k], 1e-20);
    double vr_old = mr[k] / rho_c;
    double vr_new = vr_old * f;
    double dKE = 0.5 * rho_c * (vr_old * vr_old - vr_new * vr_new);
    mr[k] = rho_c * vr_new;
    rhoE[k] -= dKE;
}

// ========================= Sponge + isothermal buffer ========================
// Velocity damping: v *= 1/(1 + dt·κ)
// Thermal relaxation: rhoE → rhoE₀ with same timescale (MUSIC-style)
//   rhoE_new = (rhoE + dt·κ·rhoE₀) / (1 + dt·κ)
// This creates a stably stratified isothermal layer that prevents
// convective instability from reaching the boundary.

__global__
void k_fas2_sponge(double* rho, double* mr, double* mt, double* rhoE,
                  const double* rho0, const double* P0,
                  const double* r_center,
                  double r_sp, double r_tp, double kappa_max, double dt_s,
                  double gam_minus1_inv,
                  int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int i = flat/nt, j = flat%nt;
    double r = r_center[i];
    if (r <= r_sp) return;
    double xi = fmin((r - r_sp) / fmax(r_tp - r_sp, 1e-30), 1.0);
    double kappa = kappa_max * 0.5 * (1.0 - cos(M_PI * xi));
    double alpha = dt_s * kappa;
    double damp = 1.0 / (1.0 + alpha);
    int k = (i + ng) * (nt + 2*ng) + (j + ng);
    double mr_old = mr[k], mt_old = mt[k];
    mr[k] *= damp;
    mt[k] *= damp;
    // Relax total energy: damp KE (via momentum damping) and relax internal energy toward HSE
    // E_new = ½ρv_new² + e_int_relaxed
    // where e_int_relaxed = (e_int + α·P₀/(γ-1)) / (1+α)
    double rho_c = fmax(rho[k], 1e-20);
    double KE_old = 0.5 * (mr_old*mr_old + mt_old*mt_old) / rho_c;
    double e_int = rhoE[k] - KE_old;
    double e_int_ref = P0[flat] * gam_minus1_inv;
    double e_int_new = (e_int + alpha * e_int_ref) * damp;
    double KE_new = 0.5 * (mr[k]*mr[k] + mt[k]*mt[k]) / rho_c;
    rhoE[k] = e_int_new + KE_new;
}

// ========================= Residual norm ========================

double FasSolver2::residual_norm(int l) {
    FasLevel2& lev = levels[l];
    int n = lev.nr * lev.nt;
    int n4 = 4 * n;
    std::vector<double> h(n4);
    CUDA_CHECK(cudaMemcpy(h.data(), lev.d_res, n4*sizeof(double), cudaMemcpyDeviceToHost));

    // Only count interior cells (ρ₀ >= atm_rho_thresh)
    std::vector<double> h_rho0(n);
    CUDA_CHECK(cudaMemcpy(h_rho0.data(), lev.d_rho0, n*sizeof(double), cudaMemcpyDeviceToHost));

    double mx = 0;
    for (int flat = 0; flat < n; ++flat) {
        if (h_rho0[flat] < atm_rho_thresh) continue;
        for (int eq = 0; eq < 4; ++eq)
            mx = std::max(mx, std::fabs(h[eq*n + flat]));
    }
    return mx;
}

// Detailed residual norm with per-equation and per-region breakdown
void FasSolver2::residual_norm_detail(int l, const char* label) {
    FasLevel2& lev = levels[l];
    int n = lev.nr * lev.nt;
    int n4 = 4 * n;
    std::vector<double> h(n4);
    CUDA_CHECK(cudaMemcpy(h.data(), lev.d_res, n4*sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> h_rho0(n);
    CUDA_CHECK(cudaMemcpy(h_rho0.data(), lev.d_rho0, n*sizeof(double), cudaMemcpyDeviceToHost));

    double eq_max[4] = {0,0,0,0};        // per-equation L∞
    double interior_max[4] = {0,0,0,0};   // interior cells only
    double atm_max[4] = {0,0,0,0};        // atmosphere cells only
    int n_interior = 0, n_atm = 0;

    for (int flat = 0; flat < n; ++flat) {
        bool is_interior = h_rho0[flat] >= atm_rho_thresh;
        if (is_interior) n_interior++; else n_atm++;
        for (int eq = 0; eq < 4; ++eq) {
            double val = std::fabs(h[eq*n + flat]);
            eq_max[eq] = std::max(eq_max[eq], val);
            if (is_interior)
                interior_max[eq] = std::max(interior_max[eq], val);
            else
                atm_max[eq] = std::max(atm_max[eq], val);
        }
    }

    std::fprintf(stderr, "  [%s] L%d  ||F||: rho=%.2e mr=%.2e mt=%.2e E=%.2e\n",
                 label, l, eq_max[0], eq_max[1], eq_max[2], eq_max[3]);
    std::fprintf(stderr, "         interior(%d): rho=%.2e mr=%.2e mt=%.2e E=%.2e\n",
                 n_interior, interior_max[0], interior_max[1], interior_max[2], interior_max[3]);
    std::fprintf(stderr, "         atm(%d):      rho=%.2e mr=%.2e mt=%.2e E=%.2e\n",
                 n_atm, atm_max[0], atm_max[1], atm_max[2], atm_max[3]);
}

// ========================= CFL dt ========================

__global__
void k_fas2_cfl(const double* rho, const double* mr, const double* mt, const double* rhoE,
               const double* dr, const double* r_center, const double* dtheta,
               const double* rho0, double* out,
               int nr, int nt, int ng, EOS eos, double atm_thresh,
               int n_angular_avg, int radial_only) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    if (rho0[flat] < atm_thresh) { out[flat] = 1e30; return; }
    int i = flat/nt, j = flat%nt;
    int k = fas_idx(i,j,nt,ng);
    double rho_c = fmax(rho[k], 1e-20);
    if (rho_c < 0.01 * rho0[flat]) { out[flat] = 1e30; return; }
    double vr = fabs(mr[k] / rho_c);
    double vt = fabs(mt[k] / rho_c);
    double KE = 0.5 * rho_c * (vr*vr + vt*vt);
    double e_c = fmax((rhoE[k] - KE) / rho_c, 1e-30);
    double P = fmax(eos.pressure(rho_c, e_c), 1e-30);
    double cs = eos.sound_speed(rho_c, P);
    double dt_r = dr[i] / (vr + cs);
    // Radial-only mode or angular-averaged shells: skip theta CFL
    if (radial_only || i < n_angular_avg) {
        out[flat] = dt_r;
    } else {
        double dt_t = r_center[i] * dtheta[j] / (vt + cs);
        out[flat] = fmin(dt_r, dt_t);
    }
}

double FasSolver2::compute_cfl_dt() {
    FasLevel2& lev = levels[0];
    int n = lev.nr * lev.nt, B = 256;
    k_fas2_cfl<<<(n+B-1)/B,B>>>(
        lev.d_rho, lev.d_mr, lev.d_mt, lev.d_rhoE,
        lev.d_dr, lev.d_r_center, lev.d_dtheta,
        lev.d_rho0, lev.d_res,
        lev.nr, lev.nt, lev.ng, eos, atm_rho_thresh, n_angular_avg, (int)radial_only);
    return cfl_num * gpu_reduce_min(lev.d_res, lev.d_poisson_rhs, n);
}

