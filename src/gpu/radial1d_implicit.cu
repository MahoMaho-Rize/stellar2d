// Phase 4: radial1d implicit Backward-Euler + JFNK Newton-Krylov.
// Ports the cart_impl BE + JFNK framework to 1D Lagrangian radial.
//
// State layout (packed, total N_dof = 3·nz):
//   d_U[0         .. nz-1  ] = v_face[k=1..nz]
//   d_U[nz        .. 2·nz-1] = r_face[k=1..nz]
//   d_U[2·nz      .. 3·nz-1] = e_zone[k=0..nz-1]
// (face k=0 is pinned v=0, r=0; not part of U.)
//
// Residual R(U) (dU/dt = R(U)):
//   R_v[k] = -A_k · (XP_k − XP_{k-1}) / dm̄_k − g_k       (face k=1..nz)
//   R_r[k] = v_k                                          (face)
//   R_e[k] = -(P_k + Pvsc_k)·(dV/dt)_k / dm_k + ε_pp,k    (zone)
//
//   XP_k = P(ρ_k, e_k) + Pvsc_k   for zone k
//   XP_{nz} = P_surf_floor        (ghost above surface)
//   dm̄_k = 0.5·(dm_{k-1} + dm_k)  for k=1..nz-1
//   dm̄_nz = 0.5·dm_{nz-1}
//   dV/dt = 4π·(r_{k+1}²·v_{k+1} − r_k²·v_k)
//   A_k = 4π r_k²;  g_k = G·M_k/r_k²   (g_0 = 0)

#include "radial1d_solver.cuh"
#include "fas_common.cuh"
#include "../physics/nuclear_pp.h"
#include "radial1d_residual_dual.cuh"

// Forward declare kernels from radial1d_kernels.cuh (defined in radial1d_solver.cu TU).
static constexpr double PI4_IMPL = 12.566370614359172;
#define PI4 PI4_IMPL

extern __global__ void k_rad1d_zone_primitives(
    const double*, const double*, const double*,
    double*, double*, double*, int, double);
extern __global__ void k_rad1d_zone_primitives_eos(
    const double*, const double*, const double*,
    double*, double*, double*, int, EOS);
extern __global__ void k_rad1d_enclosed_mass(const double*, double*, int);
extern __global__ void k_rad1d_gravity(const double*, const double*, double*, int, double);
extern __global__ void k_rad1d_artificial_viscosity(
    const double*, const double*, const double*, double*, int, double, double);
extern __global__ void k_rad1d_T_from_rho_e(
    const double*, const double*, double*, int, EOS);

// Diagnostic: L_surf with τ=2/3 photospheric BC. Walks optical depth from
// outer boundary inward, identifies photosphere zone, uses T there in Stefan.
// Also writes the photosphere index + T into out[1], out[2] for debugging.
__global__ static void k_r1di_diag_L_surf(
    const double* rho, const double* e_int, const double* r,
    double* out, int nz, EOS eos, OpacityParams opa,
    double sigma_sb)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    // Mirror the residual BC: grey Eddington 1-zone atmosphere.
    //   T_eff⁴ = (4/3) T_k⁴ / (τ_k + 2/3)
    int k = nz - 1;
    double rho_k = fmax(rho[k], 1e-30);
    double e_k   = fmax(e_int[k], 1e-30);
    double T_helm = eos.temperature_from_rho_e(rho_k, e_k);
    double T_ideal = e_k / 9.6e7;
    if (T_ideal < 1.0)  T_ideal = 1.0;
    if (T_ideal > 1e8) T_ideal = 1e8;
    double T_k = (T_helm > 10.0 && T_helm < 3.0 * T_ideal) ? T_helm : T_ideal;
    double dr_k = r[k+1] - r[k];
    if (dr_k < 0.0) dr_k = 0.0;
    double kap_k = grey_opacity(rho_k, T_k > 1.0 ? T_k : 1.0, opa);
    if (!(kap_k > 1e-30)) kap_k = 1e-30;
    double tau_k = kap_k * rho_k * dr_k;
    double Tk2 = T_k * T_k;
    double Tk4 = Tk2 * Tk2;
    double T_eff4 = (4.0 / 3.0) * Tk4 / (tau_k + 2.0 / 3.0);
    double A = 4.0 * 3.14159265358979323846 * r[nz] * r[nz];
    double T_eff = sqrt(sqrt(T_eff4));
    out[0] = A * sigma_sb * T_eff4;
    out[1] = (double)k;
    out[2] = T_eff;
    out[3] = tau_k;
}

// Species-only burn kernel: advance X, Y by dt using current ρ, T. Does NOT
// modify e_int — that's done implicitly by the Newton solver via R_e.
static __global__ void k_r1di_nuclear_pp_species(
    double* e_int, double* X, double* Y, const double* rho,
    int nz, EOS eos, NuclearPPParams pars, double dt)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nz) return;
    double rho_k = fmax(rho[k], 1e-30);
    double T_k = eos.temperature_from_rho_e(rho_k, fmax(e_int[k], 1e-30));
    double Xk = fmax(X[k], 0.0);
    double eps = nuclear_pp_epsilon_X(rho_k, T_k, Xk, pars);
    // Only species update (e_int untouched — handled implicitly).
    double dX = -dt * eps / fmax(pars.q_burn, 1e-30);
    double Xnew = Xk + dX;
    if (Xnew < 0.0) { dX = -Xk; Xnew = 0.0; }
    X[k] = Xnew;
    Y[k] = fmin(1.0, Y[k] + (-dX));
}
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------
// Simple reductions (copied pattern from cart_impl_solver.cu)
// ---------------------------------------------------------------------
static __global__ void k_r1di_dot_partial(const double* a, const double* b, double* part, int N) {
    extern __shared__ double sd[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double s = 0.0;
    while (i < N) { s += a[i]*b[i]; i += blockDim.x * gridDim.x; }
    sd[tid] = s; __syncthreads();
    for (int off = blockDim.x/2; off > 0; off >>= 1) {
        if (tid < off) sd[tid] += sd[tid+off];
        __syncthreads();
    }
    if (tid == 0) part[blockIdx.x] = sd[0];
}

static double gpu_dot_r1di(const double* a, const double* b, int N) {
    int B = 256, nb = (N+B-1)/B;
    if (nb > 256) nb = 256;
    static double* d_p = nullptr;
    static int cap = 0;
    if (cap < nb) {
        if (d_p) cudaFree(d_p);
        cudaMalloc(&d_p, nb*sizeof(double));
        cap = nb;
    }
    k_r1di_dot_partial<<<nb, B, B*sizeof(double)>>>(a, b, d_p, N);
    std::vector<double> h(nb);
    cudaMemcpy(h.data(), d_p, nb*sizeof(double), cudaMemcpyDeviceToHost);
    double s = 0;
    for (double v : h) s += v;
    return s;
}
static double gpu_norm_r1di(const double* a, int N) { return std::sqrt(gpu_dot_r1di(a, a, N)); }

// Row-scaled RMS: sqrt( Σ (F_i · invL_i)² / N ) — the natural Newton metric
// when Viallet row-scaling is active. Each component is weighed by its
// reciprocal natural magnitude, so F_v, F_r, F_e enter the norm with
// comparable weight even when their raw magnitudes span 10 orders.
__global__ static void k_r1di_mul_sq_partial(const double* F, const double* invL,
                                             double* out, int N) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    double v = 0.0;
    if (i < N) {
        double w = F[i] * invL[i];
        v = w * w;
    }
    sdata[tid] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}
static double gpu_norm_scaled_r1di(const double* F, const double* invL, int N) {
    int B = 256, nb = (N + B - 1) / B;
    if (nb > 256) nb = 256;
    static double* d_p = nullptr;
    static int cap = 0;
    if (cap < nb) {
        if (d_p) cudaFree(d_p);
        cudaMalloc(&d_p, nb * sizeof(double));
        cap = nb;
    }
    k_r1di_mul_sq_partial<<<nb, B, B * sizeof(double)>>>(F, invL, d_p, N);
    std::vector<double> h(nb);
    cudaMemcpy(h.data(), d_p, nb * sizeof(double), cudaMemcpyDeviceToHost);
    double s = 0.0;
    for (double v : h) s += v;
    return std::sqrt(s / (double)N);
}

// ---------------------------------------------------------------------
// Generic utility kernels
// ---------------------------------------------------------------------
__global__ static void k_r1di_copy(double* dst, const double* src, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) dst[i] = src[i];
}
__global__ static void k_r1di_axpy(double* y, double a, const double* x, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) y[i] += a*x[i];
}
__global__ static void k_r1di_scale(double* x, double a, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) x[i] *= a;
}
__global__ static void k_r1di_mul_diag(double* x, const double* D, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) x[i] *= D[i];
}

// ---------------------------------------------------------------------
// Pack / unpack (face k=1..nz, zones k=0..nz-1)
// ---------------------------------------------------------------------
__global__ static void k_r1di_pack(
    double* U,
    const double* v_face,     // (nz+1) — index 1..nz used
    const double* r_face,
    const double* e_zone,     // (nz)
    int nz)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nz) return;
    int kf = i + 1;                       // face index (1..nz)
    U[i]            = v_face[kf];
    U[nz + i]       = r_face[kf];
    U[2*nz + i]     = e_zone[i];
}

__global__ static void k_r1di_unpack(
    const double* U,
    double* v_face,           // (nz+1) — writes 1..nz; v_face[0] pinned to 0 elsewhere
    double* r_face,
    double* e_zone,
    int nz)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nz) return;
    int kf = i + 1;
    v_face[kf] = U[i];
    r_face[kf] = U[nz + i];
    e_zone[i]  = fmax(U[2*nz + i], 1e-30);
}

// r_face[0] = 0, v_face[0] = 0 enforced on the device
__global__ static void k_r1di_pin_center(double* v_face, double* r_face) {
    if (threadIdx.x == 0 && blockIdx.x == 0) { v_face[0] = 0.0; r_face[0] = 0.0; }
}

// ---------------------------------------------------------------------
// Build the primitives (rho, P, Pvsc) from current (r, e) state.
// We reuse existing kernels via helpers but tag with different dispatch.
// ---------------------------------------------------------------------
// forward decl from solver.cu
static inline void prims_and_visc(Radial1DSolver& S) {
    int nz = S.lev.nz, B = 256;
    if (S.use_eos) {
        k_rad1d_zone_primitives_eos<<<(nz+B-1)/B, B>>>(
            S.lev.d_r, S.lev.d_dm, S.lev.d_e_int,
            S.lev.d_Vol, S.lev.d_rho, S.lev.d_P, nz, S.eos);
    } else {
        k_rad1d_zone_primitives<<<(nz+B-1)/B, B>>>(
            S.lev.d_r, S.lev.d_dm, S.lev.d_e_int,
            S.lev.d_Vol, S.lev.d_rho, S.lev.d_P, nz, S.gamma);
    }
    k_rad1d_artificial_viscosity<<<(nz+B-1)/B, B>>>(
        S.lev.d_v, S.lev.d_Vol, S.lev.d_P, S.lev.d_Pvsc, nz, S.CQ, S.ZSH);
}

// ---------------------------------------------------------------------
// R(U) kernel: write R_v, R_r, R_e into packed residual.
//
// Inputs (already up to date):
//   r[k] (nz+1), v[k] (nz+1), dm[k] (nz), M[k] (nz+1), g[k] (nz+1),
//   Vol[k] (nz), rho[k] (nz), P[k] (nz), Pvsc[k] (nz)
//
// Nuclear pp source folded into R_e when nuc params supplied.
// ---------------------------------------------------------------------
__global__ static void k_r1di_residual(
    double* R,                                // packed residual (size 3·nz)
    const double* r_face,                     // nz+1
    const double* v_face,                     // nz+1
    const double* dm,                         // nz
    const double* g_face,                     // nz+1
    const double* Vol,                        // nz
    const double* rho,                        // nz
    const double* P,                          // nz
    const double* Pvsc,                       // nz
    const double* e_int,                      // nz
    int nz,
    EOS eos,
    bool use_eos,
    double P_surf_floor,
    NuclearPPParams npars,
    int nuclear_on,
    dualR::RadParams rad,
    int nz_atm_split)
{
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nz) return;
    // Split mode: atm zones see hydro only. Rad + nuclear source terms
    // are handled by the operator-split BE-rad post-Newton.
    int k_split = nz - nz_atm_split;
    bool in_atm = (nz_atm_split > 0) && (k >= k_split);
    // face-based index kf = k+1 (face 1..nz)
    int kf = k + 1;

    // ----- R_v (face kf) -----
    double rk   = r_face[kf];
    double Ak   = PI4 * rk * rk;
    double XP_above, XP_below, dm_bar;
    if (kf == nz) {
        // ghost above surface
        XP_above = P_surf_floor;
        XP_below = P[kf-1] + Pvsc[kf-1];
        dm_bar   = 0.5 * dm[kf-1];
    } else {
        XP_above = P[kf]   + Pvsc[kf];
        XP_below = P[kf-1] + Pvsc[kf-1];
        dm_bar   = 0.5 * (dm[kf-1] + dm[kf]);
    }
    dm_bar = fmax(dm_bar, 1e-30);
    double gk = g_face[kf];
    double R_v = -Ak * (XP_above - XP_below) / dm_bar - gk;

    // ----- R_r (face kf) -----
    double R_r = v_face[kf];

    // ----- R_e (zone k) -----
    // dV/dt = 4π · (r_{k+1}² v_{k+1} − r_k² v_k)
    double rL = r_face[k];
    double rR = r_face[k+1];
    double vL = v_face[k];
    double vR = v_face[k+1];
    double dVdt = PI4 * (rR*rR*vR - rL*rL*vL);
    double dm_k = fmax(dm[k], 1e-30);
    double Pt   = P[k] + Pvsc[k];
    double R_e = -Pt * dVdt / dm_k;

    // Nuclear source (in residual so it's implicit)
    if (nuclear_on && !in_atm) {
        double rho_k = fmax(rho[k], 1e-30);
        double T_k;
        if (use_eos) T_k = eos.temperature_from_rho_e(rho_k, fmax(e_int[k], 1e-30));
        else         T_k = fmax(e_int[k], 1e-30) / (1.0/(eos.gamma - 1.0));  // fallback, unused
        double eps = nuclear_pp_epsilon(rho_k, T_k, npars);
        R_e += eps;
    }

    // Radiation source (same formula as rad_face_L_dual, scalar path).
    // This couples hydro and rad in the same Newton solve — replaces the
    // operator-split BE-rad, which left transient momentum imbalances that
    // stalled Newton at Mach > 1 in quasi-static KH contraction.
    if (rad.enabled && use_eos && !in_atm) {
        double rho_k = fmax(rho[k], 1e-30);
        double e_k   = fmax(e_int[k], 1e-30);
        double T_k   = eos.temperature_from_rho_e(rho_k, e_k);
        double L_in = 0.0, L_out = 0.0;

        // Inner face: between zones k-1 and k
        if (k >= 1) {
            double rho_m = fmax(rho[k-1], 1e-30);
            double e_m   = fmax(e_int[k-1], 1e-30);
            double T_m   = eos.temperature_from_rho_e(rho_m, e_m);
            double rho_face = 0.5 * (rho_m + rho_k);
            double T_face   = 0.5 * (T_m + T_k);
            double kap = grey_opacity(rho_face, T_face > 1.0 ? T_face : 1.0, rad.opa);
            if (!(kap > 1e-30)) kap = 1e-30;
            double D = rad.c_light / (3.0 * kap * rho_face);
            double A_face = PI4 * r_face[k] * r_face[k];
            double rc_lo = 0.5 * (r_face[k-1] + r_face[k]);
            double rc_hi = 0.5 * (r_face[k]   + r_face[k+1]);
            double dr_zc = fmax(rc_hi - rc_lo, 1e-30);
            double Tm4 = T_m*T_m; Tm4 *= Tm4;
            double Tk4 = T_k*T_k; Tk4 *= Tk4;
            L_in = A_face * D * rad.a_rad * (Tm4 - Tk4) / dr_zc;
            if (rad.K_conv != nullptr) {
                double K_face = 0.5 * (rad.K_conv[k-1] + rad.K_conv[k]);
                if (K_face > 0.0)
                    L_in += A_face * K_face * (T_m - T_k) / dr_zc;
            }
        }

        // Outer face: between zones k and k+1, or surface
        if (k < nz - 1) {
            double rho_p = fmax(rho[k+1], 1e-30);
            double e_p   = fmax(e_int[k+1], 1e-30);
            double T_p   = eos.temperature_from_rho_e(rho_p, e_p);
            double rho_face = 0.5 * (rho_k + rho_p);
            double T_face   = 0.5 * (T_k + T_p);
            double kap = grey_opacity(rho_face, T_face > 1.0 ? T_face : 1.0, rad.opa);
            if (!(kap > 1e-30)) kap = 1e-30;
            double D = rad.c_light / (3.0 * kap * rho_face);
            double A_face = PI4 * r_face[k+1] * r_face[k+1];
            double rc_lo = 0.5 * (r_face[k]   + r_face[k+1]);
            double rc_hi = 0.5 * (r_face[k+1] + r_face[k+2]);
            double dr_zc = fmax(rc_hi - rc_lo, 1e-30);
            double Tk4 = T_k*T_k; Tk4 *= Tk4;
            double Tp4 = T_p*T_p; Tp4 *= Tp4;
            L_out = A_face * D * rad.a_rad * (Tk4 - Tp4) / dr_zc;
            if (rad.K_conv != nullptr) {
                double K_face = 0.5 * (rad.K_conv[k] + rad.K_conv[k+1]);
                if (K_face > 0.0)
                    L_out += A_face * K_face * (T_k - T_p) / dr_zc;
            }
        } else {
            // Grey Eddington 1-zone atmosphere BC:
            //   T⁴(τ) = (3/4) T_eff⁴ (τ + 2/3)
            //   ⇒ T_eff⁴ = (4/3) T_k⁴ / (τ_k + 2/3)
            // where τ_k = κ(ρ_k,T_k)·ρ_k·Δr_k is the optical depth of the
            // outermost zone treated as a single atmosphere layer. This
            // replaces the earlier τ-scan which collapsed onto zone k for
            // MESA-seeded IC (Phase 4 Δr-floor makes τ_k ≫ 2/3 in one zone).
            double dr_k = r_face[k+1] - r_face[k];
            if (dr_k < 0.0) dr_k = 0.0;
            double kap_k = grey_opacity(rho_k, T_k > 1.0 ? T_k : 1.0, rad.opa);
            if (!(kap_k > 1e-30)) kap_k = 1e-30;
            double tau_k = kap_k * rho_k * dr_k;
            double Tk2 = T_k * T_k;
            double Tk4 = Tk2 * Tk2;
            double T_eff4 = (4.0 / 3.0) * Tk4 / (tau_k + 2.0 / 3.0);
            if (rad.T_phot_floor > 0.0) {
                double f2 = rad.T_phot_floor * rad.T_phot_floor;
                T_eff4 += f2 * f2;
            }
            double A_surf = PI4 * r_face[k+1] * r_face[k+1];
            L_out = A_surf * rad.sigma_sb * T_eff4;
        }

        R_e += (L_in - L_out) / dm_k;
    }

    // Write packed
    R[k]          = R_v;
    R[nz + k]     = R_r;
    R[2*nz + k]   = R_e;
}

__global__ static void k_r1di_compute_F(
    double* F,
    const double* U,
    const double* Un,
    const double* R,
    const double* R_hse,
    double inv_dt,
    int N,
    double rhse_scale)  // 1.0 = well-balanced, 0.0 = --no-rhse diagnostic
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    F[i] = (U[i] - Un[i]) * inv_dt - (R[i] - rhse_scale * R_hse[i]);
}

// ---------------------------------------------------------------------
// Viallet eq-72 scaling — diagonal, per-field.
//   face v: ρ_bar · max(|v|, α1·cs)     ; R = max(|v|, α2·cs)    ; invL = 1/L
//   face r: ρ_bar · cs                   ; R = cs                 ; invL = 1/L
//   zone e: ρ · cs²                      ; R = cs²/ρ              ; invL = 1/L
// where ρ_bar is face-averaged density. For HSE we use the reference (ρ0).
// ---------------------------------------------------------------------
// Per-field dimensional scaling for 1D Lagrangian BE + JFNK.
//
// Dimensional analysis of each residual row, with local ρ, cs at HSE:
//   R_v (face momentum)     ~ g ~ cs²/R_star    dim [L/T²]
//   R_r (face kinematic)    ~ cs                dim [L/T]
//   R_e (zone energy)       ~ cs³/R_star        dim [L²/T³]
//
// Natural scales for δU:
//   δv ~ cs,   δr ~ R_star,   δe ~ cs²
// Natural scales for F (= (U−Uⁿ)/dt − R) when dt ≫ τ_acoustic:
//   F_v ~ cs²/R_star,  F_r ~ cs,  F_e ~ cs³/R_star
//
// So: R_diag  = diag(cs,          R_star,     cs²)
//     L_diag  = diag(cs²/R_star,  cs,         cs³/R_star)
//     invL    = 1 / L_diag
// This is dimensionally consistent per row and independent of ρ — very
// different from Viallet's cell-centered (ρ, ρcs², ρcs², ρcs²) form.
__global__ static void k_r1di_build_scaling(
    const double* rho,        // nz (current state — NOT HSE, so rad-driven
                              //     evolution sees its own scale)
    const double* P,          // nz (current state)
    const double* e_int,      // nz (current state; needed for T via EOS)
    const double* r_face,     // nz+1 (current state; for Δr per cell)
    const double* v_face,     // nz+1 (unused; kept for signature)
    double* L, double* R_col, double* invL,
    EOS eos, bool use_eos, double alpha1, double alpha2,
    int nz,
    double R_star,
    int rad_on, double a_rad, double c_light, OpacityParams opa,
    int nuc_on, NuclearPPParams npars)
{
    (void)v_face; (void)alpha1; (void)alpha2;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nz) return;
    int kf = i + 1;

    // Derive T(ρ, e) for this zone and its neighbour (face average).
    auto T_of = [&](int j) -> double {
        double rj = fmax(rho[j], 1e-30);
        double ej = fmax(e_int[j], 1e-30);
        if (use_eos) return fmax(eos.temperature_from_rho_e(rj, ej), 1.0);
        // ideal fallback
        double cv = (1.0 / eos.mu) / (eos.gamma - 1.0);
        return fmax(ej / fmax(cv, 1e-30), 1.0);
    };
    double T_i = T_of(i);

    // Current-state face-averaged ρ, P, and T (for rad).
    double rho_bar, P_bar, T_bar;
    if (kf == nz) {
        rho_bar = fmax(rho[kf-1], 1e-30);
        P_bar   = fmax(P[kf-1], 1e-30);
        T_bar   = T_i;
    } else {
        rho_bar = 0.5*(rho[kf-1] + rho[kf]);
        P_bar   = 0.5*(P[kf-1]   + P[kf]);
        T_bar   = 0.5*(T_i + T_of(kf));
    }
    rho_bar = fmax(rho_bar, 1e-30);
    P_bar   = fmax(P_bar,   1e-30);
    double cs_f = sqrt(fmax(eos.gamma * P_bar / rho_bar, 1e-30));
    double R_s = fmax(R_star, 1e-30);

    // Column scales R_col (O(U_typ)) — set from current state so FD probes
    // stay linear as state evolves. Unchanged structure vs. HSE version.
    //   R_v = cs_f, R_r = R_star, R_e = cs²
    // Row scales L — typical |F_i| at current state. Rad addition matters
    // at high nr, where dm shrinks and L/dm swamps PdV/dm.

    // ---- face v ----
    double Lv_hydro = cs_f * cs_f / R_s;
    L[i]       = Lv_hydro;
    R_col[i]   = cs_f;
    invL[i]    = 1.0 / fmax(Lv_hydro, 1e-30);

    // ---- face r ----
    double Lr  = cs_f;
    L[nz + i]   = Lr;
    R_col[nz + i] = R_s;
    invL[nz + i]= 1.0 / fmax(Lr, 1e-30);

    // ---- zone e ----
    // Hydro part: F_e^hydro ~ P·v / (ρ·R) ~ cs³/R
    double rho_c = fmax(rho[i], 1e-30);
    double P_c   = fmax(P[i], 1e-30);
    double cs_c  = sqrt(fmax(eos.gamma * P_c / rho_c, 1e-30));
    double Le_hydro = cs_c * cs_c * cs_c / R_s;

    // Rad part: F_e^rad ~ (L_out - L_in) / dm
    //   L ≈ 4π r² · D a T⁴ / Δr_zc, D = c / (3κρ)
    //   dm ≈ ρ · 4π r² Δr
    //   so L/dm ≈ c a T⁴ / (3 κ ρ² · Δr²)
    // Use local r_face[i+1] and Δr ≈ r[i+1] - r[i] for the cell.
    double Le_rad = 0.0;
    if (rad_on) {
        double r_hi = r_face[i+1];
        double r_lo = r_face[i];
        double dr = fmax(r_hi - r_lo, 1e-30);
        double T4 = T_bar*T_bar; T4 *= T4;
        double kap = grey_opacity(rho_bar, T_bar, opa);
        if (!(kap > 1e-30)) kap = 1e-30;
        // c a T⁴ / (3 κ ρ² Δr²) — same dimensional form as
        // (L/dm), where L ~ 4π r² · (c a T⁴)/(3 κ ρ) / Δr and dm ~ ρ V.
        Le_rad = c_light * a_rad * T4 / (3.0 * kap * rho_bar * rho_bar * dr * dr);
    }

    // Nuclear part: F_e^nuc ~ ε_pp. At pre-MS T=7e6, ρ=20 this is ~5e3
    // erg/g/s — tiny vs cs³/R ~ 1e14 at the same state. Without adding
    // it to the row floor, Newton tol=1e-8 in scaled space (|F|/Le < tol)
    // admits ε_pp as noise and the solver false-converges with ΔU=0.
    double Le_nuc = 0.0;
    if (nuc_on) {
        Le_nuc = nuclear_pp_epsilon(rho_c, T_i, npars);
    }

    // Row-scale is the LARGER of hydro / rad, but floored by nuc so
    // ε-sized sources drive Newton instead of being scaled into noise.
    // In a near-HSE state the true |F_e| ≈ ε_nuc + rad_imbalance, so
    // setting Le to that magnitude gives the right sensitivity.
    //
    // Cap Le from above by ε_nuc when nuclear burns (sensitive to small
    // sources). Cap Le from below by e_k/R_s·cs_c (typical residual
    // magnitude |F_e| = Δe/dt where dt ~ acoustic transit = R_s/cs; this
    // prevents Le_rad's "max possible L / dm" overestimate from making
    // Newton over-aggressive in atmospheric zones, where ρ² and Δr² are
    // small so Le_rad shoots to 10¹² while actual imbalance is 10⁵).
    double e_k = fmax(e_int[i], 1e-30);
    double Le_e_typ = e_k * cs_c / R_s;
    double Le = fmax(Le_hydro, Le_e_typ);
    if (Le_rad > Le) Le = fmin(Le_rad, 100.0 * Le_e_typ);  // allow rad up to 100×
    if (Le_nuc > 0.0 && Le_nuc < Le) Le = Le_nuc;
    L[2*nz + i]    = Le;
    R_col[2*nz + i] = cs_c * cs_c;
    invL[2*nz + i] = 1.0 / fmax(Le, 1e-30);
}

// ---------------------------------------------------------------------
// Allocation / teardown
// ---------------------------------------------------------------------
void Radial1DSolver::init_implicit() {
    int nz = lev.nz;
    N_dof = n_fields * nz;
    size_t nb = N_dof * sizeof(double);
    auto mal = [&](double** p) { CUDA_CHECK(cudaMalloc(p, nb)); CUDA_CHECK(cudaMemset(*p, 0, nb)); };
    mal(&d_U); mal(&d_Un); mal(&d_Ubak);
    mal(&d_R); mal(&d_R_hse);
    mal(&d_F); mal(&d_Fk);
    mal(&d_scale_L); mal(&d_scale_R); mal(&d_scale_invL);
    for (int k = 0; k <= GMRES_K; ++k) mal(&d_V[k]);
    for (int k = 0; k <  GMRES_K; ++k) mal(&d_Z[k]);
    mal(&d_gmres_w);
    // Block-tridiag PC scratch: 9 doubles per zone × 3 block arrays.
    size_t nb_blk = nz * 9 * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_A_diag,  nb_blk));
    CUDA_CHECK(cudaMalloc(&d_A_lower, nb_blk));
    CUDA_CHECK(cudaMalloc(&d_A_upper, nb_blk));
    CUDA_CHECK(cudaMemset(d_A_diag,  0, nb_blk));
    CUDA_CHECK(cudaMemset(d_A_lower, 0, nb_blk));
    CUDA_CHECK(cudaMemset(d_A_upper, 0, nb_blk));
    CUDA_CHECK(cudaMalloc(&d_thomas_y, nz * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_thomas_rhs, nz * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_matvec_scratch, nb));
    std::fprintf(stderr, "  radial1d implicit: allocated (N_dof=%d, Viallet=%s, block-tridiag PC=%s)\n",
                 N_dof, use_viallet_scaling ? "ON" : "OFF",
                 precond_tridiag ? "ON" : "OFF");
}

void Radial1DSolver::destroy_implicit() {
    auto f = [](double* p) { if (p) cudaFree(p); };
    f(d_U); f(d_Un); f(d_Ubak); f(d_R); f(d_R_hse); f(d_F); f(d_Fk);
    f(d_scale_L); f(d_scale_R); f(d_scale_invL);
    for (int k = 0; k <= GMRES_K; ++k) f(d_V[k]);
    for (int k = 0; k <  GMRES_K; ++k) f(d_Z[k]);
    f(d_gmres_w);
    d_U = d_Un = d_Ubak = d_R = d_R_hse = d_F = d_Fk = nullptr;
    d_scale_L = d_scale_R = d_scale_invL = nullptr;
    d_gmres_w = nullptr;
    for (int k = 0; k <= GMRES_K; ++k) d_V[k] = nullptr;
    for (int k = 0; k <  GMRES_K; ++k) d_Z[k] = nullptr;
    auto f2 = [](double** p) { if (*p) { cudaFree(*p); *p = nullptr; } };
    f2(&d_A_diag); f2(&d_A_lower); f2(&d_A_upper);
    f2(&d_thomas_y); f2(&d_thomas_rhs); f2(&d_matvec_scratch);
}

// ---------------------------------------------------------------------
// Pack / unpack (device-side wrappers)
// ---------------------------------------------------------------------
void Radial1DSolver::pack_state_from_device() {
    int nz = lev.nz, B = 256;
    k_r1di_pack<<<(nz+B-1)/B, B>>>(d_U, lev.d_v, lev.d_r, lev.d_e_int, nz);
}

void Radial1DSolver::unpack_state_to_device() {
    int nz = lev.nz, B = 256;
    k_r1di_unpack<<<(nz+B-1)/B, B>>>(d_U, lev.d_v, lev.d_r, lev.d_e_int, nz);
    k_r1di_pin_center<<<1, 1>>>(lev.d_v, lev.d_r);
    prims_and_visc(*this);
}

// ---------------------------------------------------------------------
// R(U) — uses current lev.d_{v,r,e_int,Vol,rho,P,Pvsc,M,g}, NuclearPP if on
// ---------------------------------------------------------------------
void Radial1DSolver::compute_R_implicit() {
    int nz = lev.nz, B = 256, nf = nz + 1;
    // Primitives + visc from current state
    prims_and_visc(*this);
    // Gravity + enclosed mass from current r
    launch_enclosed_mass();
    k_rad1d_gravity<<<(nf+B-1)/B, B>>>(lev.d_r, lev.d_M, lev.d_gr, nz, G_const);

    NuclearPPParams npars;
    npars.X_hydrogen = nuc_X; npars.T_floor = nuc_T_floor;
    npars.T_scale = nuc_T_scale; npars.epsilon_scale = nuc_epsilon_scale;
    npars.q_burn = nuc_q_burn;

    // Dynamic ε scale: let physics compress time by picking
    //   scale = compress_frac · (core e) / (ε_phys · dt_current)
    // so Δe per step ≤ compress_frac · e. Stores back to npars.epsilon_scale.
    if (nuclear_enabled && use_eos && nuc_compress_frac > 0.0 && dt_current > 0.0) {
        // Peek core state on host once per R(U) evaluation (fast; nz small).
        double h_rho_c, h_e_c;
        cudaMemcpy(&h_rho_c, lev.d_rho, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_e_c,   lev.d_e_int, sizeof(double), cudaMemcpyDeviceToHost);
        NuclearPPParams np_base = npars;
        np_base.epsilon_scale = 1.0;  // physical
        double T_c = eos.temperature_from_rho_e(fmax(h_rho_c, 1e-30), fmax(h_e_c, 1e-30));
        double eps_phys = nuclear_pp_epsilon_X(h_rho_c, T_c, nuc_X, np_base);
        if (eps_phys > 0.0) {
            double scale_target = nuc_compress_frac * h_e_c / (eps_phys * dt_current);
            // Clamp to [1, 1e12] and apply user's manual scale as a ceiling.
            if (scale_target < 1.0) scale_target = 1.0;
            if (scale_target > nuc_epsilon_scale && nuc_epsilon_scale > 1.0)
                scale_target = nuc_epsilon_scale;
            npars.epsilon_scale = scale_target;
        }
    }

    dualR::RadParams rad;
    rad.enabled  = (radiation_enabled && use_eos) ? 1 : 0;
    rad.a_rad    = rad_a_rad;
    rad.c_light  = rad_c_light;
    rad.sigma_sb = rad_c_light * rad_a_rad / 4.0;
    rad.T_phot_floor = rad_T_phot_floor;
    fill_opacity_params(rad.opa);
    rad.K_conv = (mlt_enabled && d_K_conv != nullptr) ? d_K_conv : nullptr;

    k_r1di_residual<<<(nz+B-1)/B, B>>>(
        d_R,
        lev.d_r, lev.d_v, lev.d_dm, lev.d_gr,
        lev.d_Vol, lev.d_rho, lev.d_P, lev.d_Pvsc, lev.d_e_int,
        nz, eos, use_eos, P_surf_floor,
        npars, nuclear_enabled ? 1 : 0, rad, nz_atm_split);
}

void Radial1DSolver::compute_F_implicit(double inv_dt) {
    // Always refresh the legacy (lev.d_v, d_r, d_e_int) fields + primitives
    // from the packed d_U before evaluating R. Skipping this leaves R stale
    // against the latest Newton iterate.
    unpack_state_to_device();
    compute_R_implicit();
    int N = N_dof, B = 256;
    double rhse_scale = no_rhse_subtract ? 0.0 : 1.0;
    k_r1di_compute_F<<<(N+B-1)/B, B>>>(d_F, d_U, d_Un, d_R, d_R_hse, inv_dt, N, rhse_scale);
}

double Radial1DSolver::residual_norm_implicit() {
    int N = N_dof;
    if (use_viallet_scaling && d_scale_invL != nullptr) {
        // Row-scaled: ‖invL · F‖_RMS — each field weighed by its natural
        // magnitude so v/r/e components contribute comparably.
        return gpu_norm_scaled_r1di(d_F, d_scale_invL, N);
    }
    return std::sqrt(gpu_dot_r1di(d_F, d_F, N) / (double)N);
}

void Radial1DSolver::build_scaling_implicit() {
    if (!use_viallet_scaling) return;
    int nz = lev.nz, B = 256;
    // Read R_star (= r_face[nz]) from device once.
    double R_star = 0.0;
    CUDA_CHECK(cudaMemcpy(&R_star, lev.d_r + nz, sizeof(double), cudaMemcpyDeviceToHost));
    if (!(R_star > 0.0)) R_star = 1.0;
    OpacityParams opa;
    if (radiation_enabled && use_eos) fill_opacity_params(opa);
    int rad_on = (radiation_enabled && use_eos) ? 1 : 0;
    NuclearPPParams npars;
    npars.X_hydrogen = nuc_X;
    npars.epsilon_scale = nuc_epsilon_scale;
    npars.T_floor = nuc_T_floor;
    npars.T_scale = nuc_T_scale;
    npars.q_burn = nuc_q_burn;
    int nuc_on = (nuclear_enabled && use_eos) ? 1 : 0;
    k_r1di_build_scaling<<<(nz+B-1)/B, B>>>(
        lev.d_rho, lev.d_P, lev.d_e_int, lev.d_r, lev.d_v,
        d_scale_L, d_scale_R, d_scale_invL,
        eos, use_eos, viallet_alpha1, viallet_alpha2, nz, R_star,
        rad_on, rad_a_rad, rad_c_light, opa,
        nuc_on, npars);
}

// ---------------------------------------------------------------------
// snapshot_hse_implicit — compute R(U_hse) once the HSE state is set
// ---------------------------------------------------------------------
void Radial1DSolver::snapshot_hse_implicit() {
    // Assumes current device state is the HSE. Pack and compute R.
    // Nuclear and radiation are source terms, NOT part of the hydrostatic
    // balance — they are physical energy injection. Temporarily disable
    // so R_hse captures only (advection + gravity) which is what HSE
    // balances.
    bool saved_nuclear   = nuclear_enabled;
    bool saved_radiation = radiation_enabled;
    nuclear_enabled   = false;
    radiation_enabled = false;
    pack_state_from_device();
    compute_R_implicit();
    nuclear_enabled   = saved_nuclear;
    radiation_enabled = saved_radiation;

    int N = N_dof, B = 256;
    k_r1di_copy<<<(N+B-1)/B, B>>>(d_R_hse, d_R, N);
    double nrm = gpu_norm_r1di(d_R_hse, N);
    std::fprintf(stderr, "  radial1d implicit HSE defect: ||R_hse||=%.3e  (N=%d)\n", nrm, N);
}

// ---------------------------------------------------------------------
// JFNK matvec (Viallet-scaled, unit-normalize v̂)
// ---------------------------------------------------------------------
void Radial1DSolver::jfnk_matvec_implicit(const double* d_v_in, double* d_Jv, double inv_dt) {
    int N = N_dof, B = 256;

    // Step 1: w = R · v_in if scaling on
    const double* v_scaled = d_v_in;
    if (use_viallet_scaling) {
        k_r1di_copy<<<(N+B-1)/B, B>>>(d_gmres_w, d_v_in, N);
        k_r1di_mul_diag<<<(N+B-1)/B, B>>>(d_gmres_w, d_scale_R, N);
        v_scaled = d_gmres_w;
    }

    // FD step. The GMRES basis d_V[j] is in **scaled (δX)** space with
    // ||d_V[j]|| = 1 (unit vector). After Right-scaling v_scaled = R · d_V[j]
    // the perturb magnitude per component is α · R_i. Dimensionally-tuned
    // R_i is already O(U_typ_i), so we want α ≈ √εₘ — a fixed small
    // number, NOT ||U||/||v||. The old Knoll-Keyes ||U||/||v|| form gives
    // α ~ √εₘ · ||U||/||R|| ≫ √εₘ in cgs (||U|| ≫ ||R||), over-perturbing
    // thin shells and breaking the FD-linearity assumption.
    double norm_v = gpu_norm_r1di(v_scaled, N);
    if (norm_v < 1e-30) {
        CUDA_CHECK(cudaMemset(d_Jv, 0, N*sizeof(double)));
        return;
    }
    // FD step. For Viallet-scaled matvec, v_scaled = R·d_v already carries
    // physical magnitudes (R_i ≈ U_typ_i for row i), so α = √εₘ gives
    // ~ppm relative perturb per component.  For the unscaled path, v_in
    // is a unit basis vector with no physical units, so we rescale via
    // the full norm (Knoll-Keyes ||U||/||v|| form).
    double alpha;
    if (use_viallet_scaling) {
        alpha = std::sqrt(1e-15);  // ~3e-8
    } else {
        double norm_U = gpu_norm_r1di(d_Un, N);
        double u_ref = (norm_U > 1.0) ? norm_U : 1.0;
        alpha = std::sqrt(1e-15) * u_ref / norm_v;
    }
    double eps_hat = alpha * norm_v;  // kept for back-scaling: eps_hat = α·||v||

    // Save U, perturb
    k_r1di_copy<<<(N+B-1)/B, B>>>(d_Ubak, d_U, N);
    k_r1di_axpy<<<(N+B-1)/B, B>>>(d_U, alpha, v_scaled, N);
    unpack_state_to_device();

    // F(U + ε·v̂_scaled)
    compute_F_implicit(inv_dt);
    // J·(R·v) = (F − F_k) · ‖R·v‖ / ε
    k_r1di_copy<<<(N+B-1)/B, B>>>(d_Jv, d_F, N);
    k_r1di_axpy<<<(N+B-1)/B, B>>>(d_Jv, -1.0, d_Fk, N);
    double sc = norm_v / eps_hat;
    k_r1di_scale<<<(N+B-1)/B, B>>>(d_Jv, sc, N);

    // Apply invL
    if (use_viallet_scaling) {
        k_r1di_mul_diag<<<(N+B-1)/B, B>>>(d_Jv, d_scale_invL, N);
    }

    // Restore U
    k_r1di_copy<<<(N+B-1)/B, B>>>(d_U, d_Ubak, N);
    unpack_state_to_device();
}

// ---------------------------------------------------------------------
// AD J·v matvec via Dual<1>. One residual eval, exact derivatives, no
// finite-difference noise floor. Same column/row scaling as the FD path.
// ---------------------------------------------------------------------
__global__ static void k_r1di_ad_seed(
    dual::Dual<1>* U_d, const double* U, const double* v_seed, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dual::Dual<1> u;
    u.v = U[i];
    u.g[0] = v_seed[i];
    U_d[i] = u;
}

__global__ static void k_r1di_ad_residual(
    dual::Dual<1>* R_d, const dual::Dual<1>* U_d,
    const double* dm, int nz,
    double G_const, double P_surf_floor,
    double CQ, double ZSH,
    EOS eos, NuclearPPParams npars, int nuclear_on,
    dualR::RadParams rad, int nz_atm_split)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nz) return;
    dual::Dual<1> Rv, Rr, Re;
    dualR::residual_zone_dual<1>(k, nz, U_d, dm,
                                  G_const, P_surf_floor, CQ, ZSH,
                                  eos, npars, nuclear_on, rad, nz_atm_split,
                                  Rv, Rr, Re);
    R_d[k]          = Rv;
    R_d[nz + k]     = Rr;
    R_d[2*nz + k]   = Re;
}

// Build F = (U - Un)/dt - (R - R_hse) with Dual. Since U appears linearly
// in (U-Un)/dt its contribution to J·v is just v_seed/dt — so F_grad =
// inv_dt · v_seed - R_grad. R_hse is constant so doesn't affect gradient.
__global__ static void k_r1di_ad_compute_Jv(
    double* Jv,
    const dual::Dual<1>* R_d,          // ∂R/∂(seed direction) in .g[0]
    const double* v_seed,               // direction
    double inv_dt, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // F = (U-Un)/dt - (R - R_hse), so J·v = v/dt - (∂R/∂U) · v
    // Dual gave us ∂R/∂U · v_seed in R_d[i].g[0]
    Jv[i] = inv_dt * v_seed[i] - R_d[i].g[0];
}

// Dispatch: AD if flag set, else FD. Internal use only.
static inline void r1di_matvec(Radial1DSolver& S, const double* v_in,
                               double* Jv, double inv_dt) {
    if (S.jfnk_autodiff) S.jfnk_matvec_ad(v_in, Jv, inv_dt);
    else                 S.jfnk_matvec_implicit(v_in, Jv, inv_dt);
}

void Radial1DSolver::jfnk_matvec_ad(const double* d_v_in, double* d_Jv, double inv_dt) {
    int N = N_dof, B = 256;
    int nz = lev.nz;

    // Step 1: apply column scale v_scaled = R · v_in (if Viallet on)
    const double* v_scaled = d_v_in;
    if (use_viallet_scaling) {
        k_r1di_copy<<<(N+B-1)/B, B>>>(d_gmres_w, d_v_in, N);
        k_r1di_mul_diag<<<(N+B-1)/B, B>>>(d_gmres_w, d_scale_R, N);
        v_scaled = d_gmres_w;
    }

    // Lazy-allocate Dual buffers on first call.
    static dual::Dual<1>* s_d_U_d = nullptr;
    static dual::Dual<1>* s_d_R_d = nullptr;
    static int s_N_cached = 0;
    if (s_N_cached != N) {
        if (s_d_U_d) cudaFree(s_d_U_d);
        if (s_d_R_d) cudaFree(s_d_R_d);
        CUDA_CHECK(cudaMalloc(&s_d_U_d, N * sizeof(dual::Dual<1>)));
        CUDA_CHECK(cudaMalloc(&s_d_R_d, N * sizeof(dual::Dual<1>)));
        s_N_cached = N;
    }

    // Seed Dual<1>: value = U, gradient = v_scaled.
    k_r1di_ad_seed<<<(N+B-1)/B, B>>>(s_d_U_d, d_U, v_scaled, N);

    // Evaluate Dual residual. Uses residual_zone_dual which reconstructs
    // r, v, e from U internally — no need to unpack the scalar path.
    NuclearPPParams npars;
    npars.X_hydrogen   = nuc_X;
    npars.T_floor      = nuc_T_floor;
    npars.T_scale      = nuc_T_scale;
    npars.epsilon_scale= nuc_epsilon_scale;
    npars.q_burn       = nuc_q_burn;
    dualR::RadParams rad;
    rad.enabled  = (radiation_enabled && use_eos) ? 1 : 0;
    rad.a_rad    = rad_a_rad;
    rad.c_light  = rad_c_light;
    rad.sigma_sb = rad_c_light * rad_a_rad / 4.0;
    rad.T_phot_floor = rad_T_phot_floor;
    fill_opacity_params(rad.opa);
    rad.K_conv = (mlt_enabled && d_K_conv != nullptr) ? d_K_conv : nullptr;
    k_r1di_ad_residual<<<(nz+B-1)/B, B>>>(
        s_d_R_d, s_d_U_d, lev.d_dm, nz,
        G_const, P_surf_floor, CQ, ZSH,
        eos, npars, nuclear_enabled ? 1 : 0, rad, nz_atm_split);

    // J·v_scaled = inv_dt · v_scaled - ∂R/∂U · v_scaled
    k_r1di_ad_compute_Jv<<<(N+B-1)/B, B>>>(d_Jv, s_d_R_d, v_scaled, inv_dt, N);

    // Apply row scale invL (matches FD path postconditioning)
    if (use_viallet_scaling) {
        k_r1di_mul_diag<<<(N+B-1)/B, B>>>(d_Jv, d_scale_invL, N);
    }
}

void Radial1DSolver::apply_precond_implicit(const double* d_v_in, double* d_Mv, double inv_dt) {
    if (precond_tridiag && d_A_diag != nullptr) {
        apply_precond_tridiag(d_v_in, d_Mv);
        return;
    }
    int N = N_dof, B = 256;
    k_r1di_copy<<<(N+B-1)/B, B>>>(d_Mv, d_v_in, N);
    (void)inv_dt;
}

// ---------------------------------------------------------------------
// Block-tridiag preconditioner: build + apply.
//
// We assemble an approximation to A = invL · J · R (the scaled operator
// that jfnk_matvec_implicit computes). A is structurally block-tridiag
// with 3×3 blocks (per zone × 3 fields). To extract the entries with
// minimum matvec count we use a 3-coloring in the zone axis:
//   color c ∈ {0,1,2}: e_i = 1 for all zones i where i%3 == c
// For a given field f ∈ {0,1,2}, probing e_i produces J · e_i whose
// components at zone i contribute to the (f,f) diagonal and whose
// components at zone i±1 contribute to upper/lower blocks. Since zones
// with the same color are >= 3 apart, there is no overlap.
//
// Total: 3 colors × 3 fields = 9 matvecs ≈ 18 F evals per Newton step.
// Compare: a 30-iter GMRES cycle today costs ≈ 60 F evals.
// ---------------------------------------------------------------------
__global__ static void k_r1di_fill_color_probe(
    double* d_v_in, int n_per_field, int color, int field)
{
    // Set d_v_in = e-vector for (color, field). Zone i, field f writes
    // v_in[field*n_per_field + i] = 1 if i%3 == color.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_per_field) return;
    for (int ff = 0; ff < 3; ++ff) {
        int idx = ff * n_per_field + i;
        d_v_in[idx] = (ff == field && (i % 3 == color)) ? 1.0 : 0.0;
    }
}

// Given the Jv output of a colored probe with (color, field) = (c, f),
// extract the column entries into the correct block slots:
//   Jv[row g, zone j] where j%3 == c  → A_diag[j].row_g.col_f
//   Jv[row g, zone j] where j%3 == c-1 → A_upper[j].row_g.col_f   (j is below a probed cell)
//   Jv[row g, zone j] where j%3 == c+1 → A_lower[j].row_g.col_f   (j is above a probed cell)
// Block layout: A_* array has nz·9 doubles, block i row-major: 3 rows × 3 cols.
__global__ static void k_r1di_extract_block_column(
    const double* d_Jv, double* d_A_diag, double* d_A_lower, double* d_A_upper,
    int n_per_field, int color, int field)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_per_field) return;
    int nz = n_per_field;
    for (int g = 0; g < 3; ++g) {
        double val = d_Jv[g * nz + j];
        int blk_offset = j * 9 + g * 3 + field;  // row g, col field, in 3×3 block
        int jmod = j % 3;
        if (jmod == color) {
            d_A_diag[blk_offset] = val;
        } else if (jmod == (color + 1) % 3) {
            // zone j sits "above" a probed zone (probed at j-1)
            d_A_lower[blk_offset] = val;
        } else if (jmod == (color + 2) % 3) {
            // zone j sits "below" a probed zone (probed at j+1)
            d_A_upper[blk_offset] = val;
        }
    }
}

void Radial1DSolver::build_precond_tridiag(double inv_dt) {
    int nz = lev.nz, B = 256;
    int N = N_dof;
    CUDA_CHECK(cudaMemset(d_A_diag,  0, nz * 9 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_A_lower, 0, nz * 9 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_A_upper, 0, nz * 9 * sizeof(double)));

    // jfnk_matvec_implicit reads state from d_U / primitives. Cache F_k since
    // jfnk_matvec perturbs U and restores via d_Ubak. Existing Newton caller
    // already cached d_Fk before calling us; we don't need to re-evaluate F.
    // The matvec itself uses d_Fk (assumed current) for the finite-difference
    // baseline — which is exactly what the Newton outer loop guarantees.

    // We need a dedicated output buffer distinct from both d_matvec_scratch
    // (the probe input) and d_gmres_w (used as workspace by jfnk_matvec).
    // Allocate on demand (tiny overhead since done once per build call).
    double* d_probe_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_probe_out, N * sizeof(double)));

    for (int color = 0; color < 3; ++color) {
        for (int field = 0; field < 3; ++field) {
            k_r1di_fill_color_probe<<<(nz+B-1)/B, B>>>(
                d_matvec_scratch, nz, color, field);
            r1di_matvec(*this, d_matvec_scratch, d_probe_out, inv_dt);
            k_r1di_extract_block_column<<<(nz+B-1)/B, B>>>(
                d_probe_out, d_A_diag, d_A_lower, d_A_upper,
                nz, color, field);
        }
    }

    CUDA_CHECK(cudaFree(d_probe_out));
}

// Block-Thomas sweep on the CPU. nz is small (128–512), the 3×3 per-block
// LU is trivial, no need to run this on GPU. Costs one D→H of three
// nz·9-double arrays + one H→D of the result.
void Radial1DSolver::apply_precond_tridiag(const double* d_v_in, double* d_Mv)
{
    int nz = lev.nz, N = N_dof;
    std::vector<double> h_D(nz*9), h_L(nz*9), h_U(nz*9);
    std::vector<double> h_rhs(N), h_y(N);

    CUDA_CHECK(cudaMemcpy(h_D.data(), d_A_diag,  nz*9*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_L.data(), d_A_lower, nz*9*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_U.data(), d_A_upper, nz*9*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rhs.data(), d_v_in, N*sizeof(double), cudaMemcpyDeviceToHost));

    // Reorganise rhs from field-major (packed) to zone-major (3 per zone).
    // rhs_zone[i*3 + g] = h_rhs[g*nz + i]
    auto rhs_z = [&](int i, int g) -> double& {
        static thread_local std::vector<double> z(nz*3);
        (void)z;  // placeholder
        return h_rhs[g*nz + i];  // we'll index packed directly below
    };
    (void)rhs_z;

    // Local helpers: 3x3 matrix ops. A is row-major 9 doubles.
    auto idx3 = [](int r, int c) { return r*3 + c; };

    auto mat_add_to = [&](double* A, const double* B_minus_3x3) {
        for (int k = 0; k < 9; ++k) A[k] += B_minus_3x3[k];
    };
    (void)mat_add_to;

    auto mat3_mul = [&](const double* A, const double* B, double* C) {
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) {
                double s = 0;
                for (int k = 0; k < 3; ++k) s += A[idx3(r,k)] * B[idx3(k,c)];
                C[idx3(r,c)] = s;
            }
    };
    auto mat3_sub = [&](double* A, const double* B) {
        for (int k = 0; k < 9; ++k) A[k] -= B[k];
    };
    auto mat3_vec = [&](const double* A, const double* x, double* y) {
        for (int r = 0; r < 3; ++r) {
            double s = 0;
            for (int c = 0; c < 3; ++c) s += A[idx3(r,c)] * x[c];
            y[r] = s;
        }
    };
    // Solve 3x3 A·x = b using partial-pivot LU on a 3x4 augmented matrix.
    auto solve3x3 = [&](const double* A_in, const double* b_in, double* x_out) -> bool {
        double A[3][4];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) A[r][c] = A_in[idx3(r,c)];
            A[r][3] = b_in[r];
        }
        for (int k = 0; k < 3; ++k) {
            int piv = k;
            double pivmax = std::fabs(A[k][k]);
            for (int r = k+1; r < 3; ++r) {
                double a = std::fabs(A[r][k]);
                if (a > pivmax) { pivmax = a; piv = r; }
            }
            if (pivmax < 1e-30) return false;
            if (piv != k) {
                for (int c = 0; c < 4; ++c) std::swap(A[k][c], A[piv][c]);
            }
            double inv_p = 1.0 / A[k][k];
            for (int r = k+1; r < 3; ++r) {
                double m = A[r][k] * inv_p;
                for (int c = k; c < 4; ++c) A[r][c] -= m * A[k][c];
            }
        }
        for (int r = 2; r >= 0; --r) {
            double s = A[r][3];
            for (int c = r+1; c < 3; ++c) s -= A[r][c] * x_out[c];
            x_out[r] = s / A[r][r];
        }
        return true;
    };

    // Thomas forward: modify D_i, y_i in place.
    //   For i = 0:      D₀' = D₀,         y₀ = D₀⁻¹ rhs₀
    //   For i > 0:      D_i' = D_i − L_i D_{i-1}'⁻¹ U_{i-1}
    //                   y_i  = D_i'⁻¹ (rhs_i − L_i y_{i-1})
    // We avoid inverting D by solving (D_i') · y_i = rhs_i − L_i y_{i-1}.
    //
    // Since D_{i-1}⁻¹ · U_{i-1} is needed to form D_i', we store it as "Uhat".
    std::vector<double> Uhat_store(nz*9, 0.0);  // Uhat_i = (D'_i)⁻¹ · U_i

    // Working copies of D along the sweep
    std::vector<double> Dp(9);
    std::vector<double> rhs_i(3), y_i(3), tmp3(3), tmp33(9);

    // i = 0
    for (int k = 0; k < 9; ++k) Dp[k] = h_D[k];       // D'_0 = D_0
    // Solve D'_0 · Uhat_0 = U_0 (column-wise)
    for (int c = 0; c < 3; ++c) {
        double col_U[3] = { h_U[idx3(0,c)], h_U[idx3(1,c)], h_U[idx3(2,c)] };
        double col_Uhat[3];
        if (!solve3x3(Dp.data(), col_U, col_Uhat)) {
            std::fprintf(stderr, "  [tridiag PC] singular D'_0 col %d\n", c);
            cudaMemcpy(d_Mv, d_v_in, N*sizeof(double), cudaMemcpyDeviceToDevice);
            return;
        }
        for (int r = 0; r < 3; ++r) Uhat_store[idx3(r,c)] = col_Uhat[r];
    }
    // Solve D'_0 · y_0 = rhs_0  (rhs_0 is field-major packed rhs)
    for (int g = 0; g < 3; ++g) rhs_i[g] = h_rhs[g*nz + 0];
    if (!solve3x3(Dp.data(), rhs_i.data(), y_i.data())) {
        std::fprintf(stderr, "  [tridiag PC] singular D'_0 for y\n");
        cudaMemcpy(d_Mv, d_v_in, N*sizeof(double), cudaMemcpyDeviceToDevice);
        return;
    }
    for (int g = 0; g < 3; ++g) h_y[g*nz + 0] = y_i[g];

    // i = 1 .. nz-1
    for (int i = 1; i < nz; ++i) {
        // D'_i = D_i − L_i · Uhat_{i-1}
        double* Li = &h_L[i*9];
        double* Di = &h_D[i*9];
        double* Uhat_prev = &Uhat_store[(i-1)*9];
        mat3_mul(Li, Uhat_prev, tmp33.data());
        for (int k = 0; k < 9; ++k) Dp[k] = Di[k] - tmp33[k];
        // rhs_i = rhs_i − L_i · y_{i-1}
        double y_prev[3] = { h_y[0*nz + (i-1)], h_y[1*nz + (i-1)], h_y[2*nz + (i-1)] };
        mat3_vec(Li, y_prev, tmp3.data());
        for (int g = 0; g < 3; ++g) rhs_i[g] = h_rhs[g*nz + i] - tmp3[g];
        // y_i = D'_i⁻¹ rhs_i
        if (!solve3x3(Dp.data(), rhs_i.data(), y_i.data())) {
            std::fprintf(stderr, "  [tridiag PC] singular D'_%d\n", i);
            cudaMemcpy(d_Mv, d_v_in, N*sizeof(double), cudaMemcpyDeviceToDevice);
            return;
        }
        for (int g = 0; g < 3; ++g) h_y[g*nz + i] = y_i[g];
        // Uhat_i = D'_i⁻¹ U_i (only if i < nz-1; at i=nz-1, U_i is unused)
        if (i < nz - 1) {
            double* Ui = &h_U[i*9];
            for (int c = 0; c < 3; ++c) {
                double col_U[3] = { Ui[idx3(0,c)], Ui[idx3(1,c)], Ui[idx3(2,c)] };
                double col_Uhat[3];
                if (!solve3x3(Dp.data(), col_U, col_Uhat)) {
                    std::fprintf(stderr, "  [tridiag PC] singular D'_%d for Uhat\n", i);
                    cudaMemcpy(d_Mv, d_v_in, N*sizeof(double), cudaMemcpyDeviceToDevice);
                    return;
                }
                for (int r = 0; r < 3; ++r) Uhat_store[i*9 + idx3(r,c)] = col_Uhat[r];
            }
        }
    }

    // Backward substitution: x_{nz-1} = y_{nz-1}; x_i = y_i − Uhat_i · x_{i+1}
    std::vector<double> h_x(N, 0.0);
    for (int g = 0; g < 3; ++g) h_x[g*nz + (nz-1)] = h_y[g*nz + (nz-1)];
    for (int i = nz - 2; i >= 0; --i) {
        double x_next[3] = { h_x[0*nz + (i+1)], h_x[1*nz + (i+1)], h_x[2*nz + (i+1)] };
        mat3_vec(&Uhat_store[i*9], x_next, tmp3.data());
        for (int g = 0; g < 3; ++g)
            h_x[g*nz + i] = h_y[g*nz + i] - tmp3[g];
    }

    CUDA_CHECK(cudaMemcpy(d_Mv, h_x.data(), N*sizeof(double), cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------------
// Diagnostic: FD Jacobian vs. JFNK matvec consistency check.
//
// For a list of "probe" DOF indices (0..N_dof-1), perturb U by a unit vector
// e_i scaled by ε (absolute FD), recompute F, and compare (F(U+ε·e_i) −
// F_k)/ε  —  that's column i of J. Then run jfnk_matvec with v = e_i and see
// if the result matches. If the two disagree by much, something in the matvec
// path (Viallet scaling order, ε_hat unit-normalize, or primitives refresh)
// is wrong.
//
// Called by set_implicit_diag_probe = true in main; prints to stderr once.
// ---------------------------------------------------------------------
void radial1d_implicit_diag_probe(Radial1DSolver& S, double inv_dt,
                                  const std::vector<int>& probes,
                                  double eps_fd)
{
    int N = S.N_dof, B = 256;
    double* d_ei;
    CUDA_CHECK(cudaMalloc(&d_ei, N*sizeof(double)));
    std::vector<double> h_ei(N, 0.0);
    std::vector<double> h_Fk(N), h_Fpert(N), h_col_fd(N), h_col_mv(N), h_Ubak(N);

    // Cache F_k from current state (Newton outer already did, but re-do locally)
    S.compute_F_implicit(inv_dt);
    cudaMemcpy(h_Fk.data(), S.d_F, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Ubak.data(), S.d_U, N*sizeof(double), cudaMemcpyDeviceToHost);

    // Save Fk into S.d_Fk for matvec
    k_r1di_copy<<<(N+B-1)/B, B>>>(S.d_Fk, S.d_F, N);

    std::fprintf(stderr,
        "  ---- FD-vs-matvec Jacobian probe (eps_fd=%.1e, Viallet=%s) ----\n",
        eps_fd, S.use_viallet_scaling ? "on" : "off");

    // Read scaling diag on host for fair comparison when Viallet is on.
    // The matvec returns (invL · J · R) · e_i; to compare against the bare
    // FD column (J · e_i in physical space), we must either scale the FD
    // column by invL · J · R · invR (identity, no change) OR multiply the
    // MV column back by L · invR on rows... simpler: force the test in
    // physical space by temporarily disabling scaling for the probe.
    bool saved_viallet = S.use_viallet_scaling;
    S.use_viallet_scaling = false;
    for (int probe : probes) {
        if (probe < 0 || probe >= N) continue;
        std::fill(h_ei.begin(), h_ei.end(), 0.0);
        h_ei[probe] = 1.0;
        cudaMemcpy(d_ei, h_ei.data(), N*sizeof(double), cudaMemcpyHostToDevice);

        // Scale the FD step so the perturbation is relative (safe for
        // cgs where ||U|| ~ 1e15). Use |U_i| if non-trivial, else a
        // per-field typical: cs for v, R_star for r, cs² for e.
        double u_i = std::fabs(h_Ubak[probe]);
        double fd_step = eps_fd * (u_i > 1.0 ? u_i : 1.0);

        cudaMemcpy(S.d_U, h_Ubak.data(), N*sizeof(double), cudaMemcpyHostToDevice);
        k_r1di_axpy<<<(N+B-1)/B, B>>>(S.d_U, fd_step, d_ei, N);
        S.unpack_state_to_device();
        S.compute_F_implicit(inv_dt);
        cudaMemcpy(h_Fpert.data(), S.d_F, N*sizeof(double), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; ++i) {
            h_col_fd[i] = (h_Fpert[i] - h_Fk[i]) / fd_step;
        }

        cudaMemcpy(S.d_U, h_Ubak.data(), N*sizeof(double), cudaMemcpyHostToDevice);
        S.unpack_state_to_device();
        S.jfnk_matvec_implicit(d_ei, S.d_gmres_w, inv_dt);
        cudaMemcpy(h_col_mv.data(), S.d_gmres_w, N*sizeof(double), cudaMemcpyDeviceToHost);

        // Restore U one more time
        cudaMemcpy(S.d_U, h_Ubak.data(), N*sizeof(double), cudaMemcpyHostToDevice);
        S.unpack_state_to_device();

        // Compare: norm of diff, max elementwise diff, location of max
        double n_fd = 0, n_mv = 0, n_df = 0, max_abs = 0;
        int max_i = -1;
        for (int i = 0; i < N; ++i) {
            n_fd += h_col_fd[i]*h_col_fd[i];
            n_mv += h_col_mv[i]*h_col_mv[i];
            double d = h_col_fd[i] - h_col_mv[i];
            n_df += d*d;
            double ad = std::fabs(d);
            if (ad > max_abs) { max_abs = ad; max_i = i; }
        }
        n_fd = std::sqrt(n_fd); n_mv = std::sqrt(n_mv); n_df = std::sqrt(n_df);
        double rel = (n_fd > 1e-30) ? n_df / n_fd : 0.0;
        // Decode probe: field (0=v, 1=r, 2=e), zone index in that field
        int nz = S.lev.nz;
        int field = probe / nz, zidx = probe % nz;
        const char* fname[] = {"v_face", "r_face", "e_zone"};
        std::fprintf(stderr,
            "   probe[%d] = %s[%d]: ||col_FD||=%.3e, ||col_MV||=%.3e, ||diff||=%.3e (rel=%.2e), max|Δ|=%.3e at row %d\n",
            probe, fname[field], zidx, n_fd, n_mv, n_df, rel, max_abs, max_i);
    }

    cudaFree(d_ei);
    S.use_viallet_scaling = saved_viallet;
    std::fprintf(stderr, "  ---- end probe ----\n");
}

// ---------------------------------------------------------------------
// FGMRES (direct port of cart_impl's gmres_solve, with r1di scratch)
// ---------------------------------------------------------------------
int Radial1DSolver::gmres_solve_implicit(double* d_x, const double* d_b,
                                          double inv_dt, double tol, int max_iter)
{
    int N = N_dof, B = 256;
    int m = std::min(max_iter, (int)GMRES_K);
    std::vector<double> H((m+1)*m, 0.0), cs_arr(m), sn(m), g(m+1, 0.0);

    k_r1di_copy<<<(N+B-1)/B, B>>>(d_V[0], d_b, N);
    k_r1di_scale<<<(N+B-1)/B, B>>>(d_V[0], -1.0, N);
    double beta = gpu_norm_r1di(d_V[0], N);
    if (beta < 1e-30) return 0;
    k_r1di_scale<<<(N+B-1)/B, B>>>(d_V[0], 1.0/beta, N);
    g[0] = beta;

    int j;
    for (j = 0; j < m; ++j) {
        apply_precond_implicit(d_V[j], d_Z[j], inv_dt);
        r1di_matvec(*this, d_Z[j], d_gmres_w, inv_dt);

        // CGS2
        for (int i = 0; i <= j; ++i) {
            H[i*m+j] = gpu_dot_r1di(d_gmres_w, d_V[i], N);
            k_r1di_axpy<<<(N+B-1)/B, B>>>(d_gmres_w, -H[i*m+j], d_V[i], N);
        }
        for (int i = 0; i <= j; ++i) {
            double hc = gpu_dot_r1di(d_gmres_w, d_V[i], N);
            H[i*m+j] += hc;
            k_r1di_axpy<<<(N+B-1)/B, B>>>(d_gmres_w, -hc, d_V[i], N);
        }
        H[(j+1)*m+j] = gpu_norm_r1di(d_gmres_w, N);
        if (H[(j+1)*m+j] < 1e-30) { j++; break; }
        k_r1di_copy<<<(N+B-1)/B, B>>>(d_V[j+1], d_gmres_w, N);
        k_r1di_scale<<<(N+B-1)/B, B>>>(d_V[j+1], 1.0 / H[(j+1)*m+j], N);

        for (int i = 0; i < j; ++i) {
            double h1 = H[i*m+j], h2 = H[(i+1)*m+j];
            H[i*m+j]     =  cs_arr[i]*h1 + sn[i]*h2;
            H[(i+1)*m+j] = -sn[i]*h1 + cs_arr[i]*h2;
        }
        double h1 = H[j*m+j], h2 = H[(j+1)*m+j];
        double tt = std::sqrt(h1*h1 + h2*h2);
        cs_arr[j] = h1/tt; sn[j] = h2/tt;
        H[j*m+j] = tt; H[(j+1)*m+j] = 0.0;
        g[j+1] = -sn[j]*g[j]; g[j] = cs_arr[j]*g[j];

        if (std::fabs(g[j+1]) < tol * beta) { j++; break; }
    }
    if (step_count < 2) {
        std::fprintf(stderr, "    GMRES: β=%.3e final|g|=%.3e (j=%d)\n",
                     beta, std::fabs(g[std::min(j,m)]), j);
    }

    std::vector<double> y(j);
    for (int i = j-1; i >= 0; --i) {
        y[i] = g[i];
        for (int kk = i+1; kk < j; ++kk) y[i] -= H[i*m+kk] * y[kk];
        y[i] /= H[i*m+i];
    }

    CUDA_CHECK(cudaMemset(d_x, 0, N*sizeof(double)));
    for (int i = 0; i < j; ++i) {
        k_r1di_axpy<<<(N+B-1)/B, B>>>(d_x, y[i], d_Z[i], N);
    }
    return j;
}

// ---------------------------------------------------------------------
// Newton outer loop
// ---------------------------------------------------------------------
extern void radial1d_implicit_diag_probe(Radial1DSolver& S, double inv_dt,
                                         const std::vector<int>& probes,
                                         double eps_fd);

int Radial1DSolver::newton_solve_implicit(double dt) {
    int N = N_dof, B = 256;
    double inv_dt = 1.0 / dt;
    double init_res = -1.0;
    // Refresh K_conv ONCE per timestep (not per Newton iter). Oscillating K
    // between iters — because K depends on super-adiabaticity which is very
    // sensitive to T — destabilises Newton at high resolution. Freezing K at
    // the start-of-step state mirrors what BE-split does (fully lagged).
    if (mlt_enabled && radiation_enabled && use_eos) {
        unpack_state_to_device();
        refresh_K_conv_implicit();
    }
    for (int it = 0; it < newton_max_iter; ++it) {
        // Refresh primitives first so build_scaling sees current ρ/P/e.
        // compute_F_implicit also unpacks, but the scaling kernel needs
        // current-state inputs to set row-scales correctly.
        unpack_state_to_device();
        if (use_viallet_scaling) build_scaling_implicit();

        compute_F_implicit(inv_dt);
        k_r1di_copy<<<(N+B-1)/B, B>>>(d_Fk, d_F, N);
        double res_norm = residual_norm_implicit();
        if (it == 0) init_res = res_norm;

        // Run Jacobian probe once at step 0 iter 0 to diagnose matvec.
        if (false && step_count == 0 && it == 0) {
            int nz = lev.nz;
            // Probe middle-of-star DOFs in each field
            std::vector<int> probes = {
                nz/2,              // v_face mid
                nz + nz/2,         // r_face mid
                2*nz + nz/2,       // e_zone mid
                nz/4,              // v_face quarter
                2*nz + 1,          // e_zone near center
                2*nz + nz - 2      // e_zone near surface
            };
            radial1d_implicit_diag_probe(*this, inv_dt, probes, 1e-4);
            // Re-compute F after probe (probe restores U but we want fresh F_k)
            compute_F_implicit(inv_dt);
            k_r1di_copy<<<(N+B-1)/B, B>>>(d_Fk, d_F, N);
        }
        if (step_count < 2) {
            std::fprintf(stderr, "  r1di_newton ENTRY: step=%d iter=%d dt=%.2e ||F||=%.3e tol=%.1e\n",
                         step_count, it, dt, res_norm, newton_tol);
            if (it == 0) {
                int nz = lev.nz;
                std::vector<double> h_F(N);
                cudaMemcpy(h_F.data(), d_F, N*sizeof(double), cudaMemcpyDeviceToHost);
                double f_v=0, f_r=0, f_e=0, f_v_max=0, f_r_max=0, f_e_max=0;
                for (int i = 0; i < nz; ++i) {
                    f_v += h_F[i]*h_F[i];
                    if (std::fabs(h_F[i]) > f_v_max) f_v_max = std::fabs(h_F[i]);
                }
                for (int i = 0; i < nz; ++i) {
                    f_r += h_F[nz+i]*h_F[nz+i];
                    if (std::fabs(h_F[nz+i]) > f_r_max) f_r_max = std::fabs(h_F[nz+i]);
                }
                for (int i = 0; i < nz; ++i) {
                    f_e += h_F[2*nz+i]*h_F[2*nz+i];
                    if (std::fabs(h_F[2*nz+i]) > f_e_max) f_e_max = std::fabs(h_F[2*nz+i]);
                }
                std::vector<double> h_U(N);
                cudaMemcpy(h_U.data(), d_U, N*sizeof(double), cudaMemcpyDeviceToHost);
                double u_v=0, u_r=0, u_e=0;
                for (int i = 0; i < nz; ++i) u_v += h_U[i]*h_U[i];
                for (int i = 0; i < nz; ++i) u_r += h_U[nz+i]*h_U[nz+i];
                for (int i = 0; i < nz; ++i) u_e += h_U[2*nz+i]*h_U[2*nz+i];
                std::fprintf(stderr,
                    "    ||F|| breakdown: v=%.2e(max=%.2e) r=%.2e(max=%.2e) e=%.2e(max=%.2e)\n",
                    std::sqrt(f_v), f_v_max,
                    std::sqrt(f_r), f_r_max,
                    std::sqrt(f_e), f_e_max);
                std::fprintf(stderr,
                    "    ||U|| breakdown: v=%.2e r=%.2e e=%.2e\n",
                    std::sqrt(u_v), std::sqrt(u_r), std::sqrt(u_e));
            }
        }
        if (it == 0 && res_norm < newton_tol) return 0;
        if (!std::isfinite(res_norm)) {
            std::fprintf(stderr, "  r1di_newton: NaN residual at iter %d\n", it);
            return -1;
        }

        // Build block-tridiag preconditioner BEFORE scaling d_F into the
        // GMRES RHS — jfnk_matvec_implicit overwrites d_F via its internal
        // compute_F calls, so we must not pre-populate d_F with invL·F_k
        // before the PC build. Refreshed every Newton iter (9 matvecs).
        if (precond_tridiag) {
            build_precond_tridiag(inv_dt);
            // Re-evaluate d_F at U (PC build's last matvec restored U → d_Ubak
            // but d_F still holds F(U_perturbed)). Restore d_F = d_Fk.
            k_r1di_copy<<<(N+B-1)/B, B>>>(d_F, d_Fk, N);
        }

        // Scale RHS: F ← invL · F  (d_F contains F_k; use d_F as d_x-RHS temp)
        if (use_viallet_scaling) {
            k_r1di_mul_diag<<<(N+B-1)/B, B>>>(d_F, d_scale_invL, N);
        }

        int gm = gmres_solve_implicit(d_gmres_w, d_F, inv_dt, gmres_tol, gmres_max_iter);

        // Back-scale: δU = R · δX
        if (use_viallet_scaling) {
            k_r1di_mul_diag<<<(N+B-1)/B, B>>>(d_gmres_w, d_scale_R, N);
        }
        if (step_count < 2) {
            // Dump per-field ||δU||
            int nz = lev.nz;
            std::vector<double> h_dU(N);
            cudaMemcpy(h_dU.data(), d_gmres_w, N*sizeof(double), cudaMemcpyDeviceToHost);
            double n_v=0, n_r=0, n_e=0;
            for (int i = 0; i < nz; ++i) n_v += h_dU[i]*h_dU[i];
            for (int i = 0; i < nz; ++i) n_r += h_dU[nz+i]*h_dU[nz+i];
            for (int i = 0; i < nz; ++i) n_e += h_dU[2*nz+i]*h_dU[2*nz+i];
            // Also compare to |U| per field
            std::vector<double> h_U(N);
            cudaMemcpy(h_U.data(), d_U, N*sizeof(double), cudaMemcpyDeviceToHost);
            double u_v=0, u_r=0, u_e=0;
            for (int i = 0; i < nz; ++i) u_v += h_U[i]*h_U[i];
            for (int i = 0; i < nz; ++i) u_r += h_U[nz+i]*h_U[nz+i];
            for (int i = 0; i < nz; ++i) u_e += h_U[2*nz+i]*h_U[2*nz+i];
            std::fprintf(stderr,
                "    δU: ||v||=%.2e (rel %.2e) ||r||=%.2e (rel %.2e) ||e||=%.2e (rel %.2e)\n",
                std::sqrt(n_v), std::sqrt(n_v/std::max(u_v,1e-60)),
                std::sqrt(n_r), std::sqrt(n_r/std::max(u_r,1e-60)),
                std::sqrt(n_e), std::sqrt(n_e/std::max(u_e,1e-60)));
        }

        // ---- Armijo backtracking line search (ported from MESA mod_newton.f90 ----
        // Keep U_k in d_Ubak so we can restore; trial α starting at 1.
        // If every α ∈ {1, 0.5, 0.25, …, 0.001} produces a worse residual,
        // we must NOT leave the state at the last-tried (worst) α.
        //
        // Also: pre-shrink α so no zone ends up with e < 0.5·e_old — this
        // prevents outer-atmosphere zones from being driven into the 1e-30
        // floor on a single Newton step, which creates a T=0 artefact that
        // the radiation BC then "fixes" by emitting nothing.
        k_r1di_copy<<<(N+B-1)/B, B>>>(d_Ubak, d_U, N);
        double alpha = 1.0;
        // Scan (on host) for maximum α that keeps U_e positive-of-half-old.
        // d_U layout: [v..., r..., e...]. Read the e block + proposed δe.
        {
            std::vector<double> h_U_old(N), h_dU(N);
            CUDA_CHECK(cudaMemcpy(h_U_old.data(), d_Ubak, N*sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_dU.data(),    d_gmres_w, N*sizeof(double), cudaMemcpyDeviceToHost));
            double alpha_max = 1.0;
            int nz_ = lev.nz;
            for (int k = 0; k < nz_; ++k) {
                double e_old = h_U_old[2*nz_ + k];
                double de    = h_dU[2*nz_ + k];
                if (de < 0.0 && e_old > 0.0) {
                    // α·|de| ≤ 0.1·e_old  ⇒  α ≤ 0.1·e_old/|de|
                    // 0.1 factor (not 0.5) because Newton can take 5-10
                    // iterations per step, cumulatively 0.5^10 = 1e-3.
                    double a_lim = 0.1 * e_old / (-de);
                    if (a_lim < alpha_max) alpha_max = a_lim;
                }
            }
            if (alpha_max < 1.0) alpha = alpha_max;
        }
        double new_res = res_norm;  // placeholder
        bool ls_ok = false;
        int ls_tries = 0;
        const int ls_max = 10;
        const double ls_shrink = 0.5;
        const double ls_accept = 0.99;   // accept if ||F(U+αδ)|| < 0.99 · ||F(U)||
        for (ls_tries = 0; ls_tries < ls_max; ++ls_tries) {
            k_r1di_copy<<<(N+B-1)/B, B>>>(d_U, d_Ubak, N);
            k_r1di_axpy<<<(N+B-1)/B, B>>>(d_U, alpha, d_gmres_w, N);
            unpack_state_to_device();
            compute_F_implicit(inv_dt);
            new_res = residual_norm_implicit();
            if (std::isfinite(new_res) && new_res < ls_accept * res_norm) {
                ls_ok = true;
                break;
            }
            alpha *= ls_shrink;
        }
        if (!ls_ok) {
            // Restore U_k (reject the step) and bail out so outer step_implicit
            // can halve dt.
            k_r1di_copy<<<(N+B-1)/B, B>>>(d_U, d_Ubak, N);
            unpack_state_to_device();
            if (step_count < 3) {
                std::fprintf(stderr,
                    "  r1di_newton iter %d: line search FAILED (α shrank to %.1e, still worse); bail\n",
                    it, alpha);
            }
            return -1;
        }
        if (step_count < 3 && it < 4) {
            std::fprintf(stderr,
                "  r1di_newton iter %d: ||F||=%.3e → %.3e, GMRES=%d, α=%.3e (ls=%d)\n",
                it, res_norm, new_res, gm, alpha, ls_tries);
        }
        bool abs_ok = (new_res < newton_tol);
        // Relative convergence: require 4 orders drop from init. A mere 2×
        // cut (the old 0.5 factor) let Newton stop after 2 iters at a
        // nonphysical state in nr=1024, where init_res can be O(1e3) and
        // 0.5·init_res = 250 is still far from satisfied.
        bool rel_ok = (init_res > 0 && new_res < newton_rel_tol * init_res);
        // Stall: 5% flat over 3 consecutive iters. Bail to dt-cut.
        bool stall  = (it >= 2 && res_norm > 0 &&
                       std::fabs(new_res - res_norm) < 0.05 * res_norm);
        if (abs_ok || rel_ok || stall) {
            if (step_count < 3 || step_count % 500 == 0) {
                std::fprintf(stderr,
                    "  r1di_newton converged in %d iters (||F||: %.3e → %.3e, GMRES=%d)\n",
                    it+1, init_res, new_res, gm);
            }
            return it + 1;
        }
    }
    std::fprintf(stderr, "  r1di_newton: max_iter %d reached (||F|| high)\n", newton_max_iter);
    return -1;
}

// ---------------------------------------------------------------------
// step_implicit — save Uⁿ, run Newton with dt-cut fallback
// ---------------------------------------------------------------------
double Radial1DSolver::step_implicit(double t, double t_end, double dt_try) {
    int N = N_dof, B = 256;
    (void)t;
    if (!hse_set) snapshot_hse();           // legacy state snapshot (d_rho0, d_P0)
    // Ensure implicit scratch is allocated
    if (d_U == nullptr) init_implicit();
    // R_hse is captured by the caller BEFORE any perturbation is applied.
    // Do NOT auto-snapshot here — it would freeze the perturbed state as
    // the HSE reference, making F(U)=0 for all perturbations.

    // Guard dt and save Uⁿ from current device state
    if (dt_try <= 0.0) dt_try = 1.0;
    if (t + dt_try > t_end) dt_try = t_end - t;

    pack_state_from_device();
    k_r1di_copy<<<(N+B-1)/B, B>>>(d_Un, d_U, N);

    // Periodically re-snapshot R_hse to track the evolving quasi-static
    // reference (KH contraction makes initial HSE invalid after ~100 steps).
    if (hse_resnap_interval > 0 && step_count > 0
        && (step_count % hse_resnap_interval) == 0) {
        snapshot_hse_implicit();
    }
    if (step_count < 3 || step_count % 100 == 0) {
        std::fprintf(stderr, "  step_implicit ENTER: step=%d dt_try=%.3e\n", step_count, dt_try);
    }

    int max_cuts = 5;
    double dt = dt_try;
    for (int cut = 0; cut <= max_cuts; ++cut) {
        if (cut > 0) {
            // Restore U ← Un and push back to device
            k_r1di_copy<<<(N+B-1)/B, B>>>(d_U, d_Un, N);
            unpack_state_to_device();
        }
        int iters = newton_solve_implicit(dt);
        if (iters >= 0) {
            // Radiation diffusion is now inside the implicit Newton residual
            // R(U) (dualR::residual_zone_dual + k_r1di_residual), so hydro and
            // rad couple in one solve — the operator-split path is retained
            // only as a fallback, gated on rad_be_split. Default OFF.
            // Operator-split atmosphere: after Newton converges on the
            // inner-core subsystem (atm zones excluded from residual source
            // terms), evolve the atm rad+MLT via BE-rad over [k_split, nz).
            // Inner face Dirichlet T comes from the Newton-converged inner
            // outermost zone. Surface BC remains Eddington T(τ).
            if (nz_atm_split > 0 && radiation_enabled && use_eos) {
                int k_split = lev.nz - nz_atm_split;
                apply_radiation_diffusion_implicit(dt, k_split);
                prims_and_visc(*this);
            }

            if (rad_be_split && radiation_enabled && use_eos) {
                apply_radiation_diffusion_implicit(dt);
                prims_and_visc(*this);
            } else if (radiation_enabled && use_eos) {
                // Diagnostic-only L_surf via device kernel (τ=2/3 scan).
                static double* s_d_Lout = nullptr;
                if (!s_d_Lout) CUDA_CHECK(cudaMalloc(&s_d_Lout, 4 * sizeof(double)));
                OpacityParams opa;
                fill_opacity_params(opa);
                double sigma_sb = rad_c_light * rad_a_rad / 4.0;
                k_r1di_diag_L_surf<<<1, 1>>>(lev.d_rho, lev.d_e_int, lev.d_r,
                                              s_d_Lout, lev.nz, eos, opa, sigma_sb);
                double h_diag[4] = {0, 0, 0, 0};
                CUDA_CHECK(cudaMemcpy(h_diag, s_d_Lout, 4 * sizeof(double),
                                      cudaMemcpyDeviceToHost));
                rad_impl_L_surf = h_diag[0];
                rad_impl_phot_zone = (int)h_diag[1];
                rad_impl_T_phot = h_diag[2];
                rad_impl_tau_surf = h_diag[3];
            }

            // Species burn-up: Newton has updated e_int implicitly (R_e
            // contains ε_pp source). Now update X, Y explicitly from the
            // converged ρ, T of this step. This is operator-split *only
            // for species*, not for the energy source, which is fully
            // implicit.
            if (nuclear_enabled && use_eos && species_enabled) {
                int nz = lev.nz;
                NuclearPPParams npars;
                npars.X_hydrogen = nuc_X;
                npars.T_floor = nuc_T_floor;
                npars.T_scale = nuc_T_scale;
                npars.epsilon_scale = nuc_epsilon_scale;
                npars.q_burn = nuc_q_burn;

                k_r1di_nuclear_pp_species<<<(nz+B-1)/B, B>>>(
                    lev.d_e_int, d_X, d_Y, lev.d_rho, nz,
                    eos, npars, dt);
            }

            dt_current = dt;
            step_count++;
            return dt;
        }
        dt *= 0.5;
    }
    // All cuts failed: rollback
    k_r1di_copy<<<(N+B-1)/B, B>>>(d_U, d_Un, N);
    unpack_state_to_device();
    std::fprintf(stderr, "  r1di step %d: rollback to Un (all cuts failed)\n", step_count);
    step_count++;
    return 0.0;
}
