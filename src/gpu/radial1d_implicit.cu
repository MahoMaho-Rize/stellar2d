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
    int nuclear_on)
{
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nz) return;
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
    if (nuclear_on) {
        double rho_k = fmax(rho[k], 1e-30);
        double T_k;
        if (use_eos) T_k = eos.temperature_from_rho_e(rho_k, fmax(e_int[k], 1e-30));
        else         T_k = fmax(e_int[k], 1e-30) / (1.0/(eos.gamma - 1.0));  // fallback, unused
        double eps = nuclear_pp_epsilon(rho_k, T_k, npars);
        R_e += eps;
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
    int N)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    F[i] = (U[i] - Un[i]) * inv_dt - (R[i] - R_hse[i]);
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
    const double* rho0,       // nz (HSE reference)
    const double* P0,         // nz (HSE reference)
    const double* v_face,     // nz+1 (unused in this formulation but kept for signature)
    double* L, double* R, double* invL,
    EOS eos, double alpha1, double alpha2,
    int nz,
    double R_star)
{
    (void)v_face; (void)alpha1; (void)alpha2;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nz) return;
    int kf = i + 1;

    // face-averaged ρ and P
    double rho_bar, P_bar;
    if (kf == nz) {
        rho_bar = fmax(rho0[kf-1], 1e-30);
        P_bar   = fmax(P0[kf-1], 1e-30);
    } else {
        rho_bar = 0.5*(rho0[kf-1] + rho0[kf]);
        P_bar   = 0.5*(P0[kf-1]   + P0[kf]);
    }
    rho_bar = fmax(rho_bar, 1e-30);
    P_bar   = fmax(P_bar,   1e-30);
    double cs_f = sqrt(fmax(eos.gamma * P_bar / rho_bar, 1e-30));
    double R_s = fmax(R_star, 1e-30);

    // ---- Left scaling only (row normalization); R = identity ----
    // Why identity R: forward-scaling δX = R⁻¹ δU and back-scaling
    // δU = R δX requires R to be balanced against the column magnitudes of
    // J. A "typical-U" R overshoots because ∂R_e/∂e is O(cs²), not O(1/dt).
    // Instead of trying to guess column scales, only normalize rows so F
    // components are O(1), and let GMRES work in δU space directly.
    //
    // Row scale L_i = typical |F_i|:
    //   F_v ~ g ~ cs²/R_star     (momentum imbalance)
    //   F_r ~ cs                  (kinematic, R_r = v)
    //   F_e ~ P·v/ρ ~ cs² · (cs/R) = cs³/R_star
    // Use HSE reference cs (per zone) so scaling is state-independent.

    // ---- face v ----
    double Lv  = cs_f * cs_f / R_s;
    L[i]       = Lv;
    R[i]       = 1.0;
    invL[i]    = 1.0 / fmax(Lv, 1e-30);

    // ---- face r ----
    double Lr  = cs_f;
    L[nz + i]   = Lr;
    R[nz + i]   = 1.0;
    invL[nz + i]= 1.0 / fmax(Lr, 1e-30);

    // ---- zone e ----
    double rho_c = fmax(rho0[i], 1e-30);
    double P_c   = fmax(P0[i], 1e-30);
    double cs_c  = sqrt(fmax(eos.gamma * P_c / rho_c, 1e-30));
    double Le  = cs_c * cs_c * cs_c / R_s;
    L[2*nz + i]    = Le;
    R[2*nz + i]    = 1.0;
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
    std::fprintf(stderr, "  radial1d implicit: allocated (N_dof=%d, Viallet=%s)\n",
                 N_dof, use_viallet_scaling ? "ON" : "OFF");
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
    k_rad1d_enclosed_mass<<<1, 1>>>(lev.d_dm, lev.d_M, nz);
    k_rad1d_gravity<<<(nf+B-1)/B, B>>>(lev.d_r, lev.d_M, lev.d_gr, nz, G_const);

    NuclearPPParams npars;
    npars.X_hydrogen = nuc_X; npars.T_floor = nuc_T_floor;
    npars.T_scale = nuc_T_scale; npars.epsilon_scale = nuc_epsilon_scale;
    npars.q_burn = nuc_q_burn;

    k_r1di_residual<<<(nz+B-1)/B, B>>>(
        d_R,
        lev.d_r, lev.d_v, lev.d_dm, lev.d_gr,
        lev.d_Vol, lev.d_rho, lev.d_P, lev.d_Pvsc, lev.d_e_int,
        nz, eos, use_eos, P_surf_floor,
        npars, nuclear_enabled ? 1 : 0);
}

void Radial1DSolver::compute_F_implicit(double inv_dt) {
    // Always refresh the legacy (lev.d_v, d_r, d_e_int) fields + primitives
    // from the packed d_U before evaluating R. Skipping this leaves R stale
    // against the latest Newton iterate.
    unpack_state_to_device();
    compute_R_implicit();
    int N = N_dof, B = 256;
    k_r1di_compute_F<<<(N+B-1)/B, B>>>(d_F, d_U, d_Un, d_R, d_R_hse, inv_dt, N);
}

double Radial1DSolver::residual_norm_implicit() {
    int N = N_dof;
    return std::sqrt(gpu_dot_r1di(d_F, d_F, N) / (double)N);
}

void Radial1DSolver::build_scaling_implicit() {
    if (!use_viallet_scaling) return;
    int nz = lev.nz, B = 256;
    // Read R_star (= r_face[nz]) from device once.
    double R_star = 0.0;
    CUDA_CHECK(cudaMemcpy(&R_star, lev.d_r + nz, sizeof(double), cudaMemcpyDeviceToHost));
    if (!(R_star > 0.0)) R_star = 1.0;
    k_r1di_build_scaling<<<(nz+B-1)/B, B>>>(
        lev.d_rho0, lev.d_P0, lev.d_v,
        d_scale_L, d_scale_R, d_scale_invL,
        eos, viallet_alpha1, viallet_alpha2, nz, R_star);
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
    bool saved_nuclear = nuclear_enabled;
    nuclear_enabled = false;
    pack_state_from_device();
    compute_R_implicit();
    nuclear_enabled = saved_nuclear;

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

void Radial1DSolver::apply_precond_implicit(const double* d_v_in, double* d_Mv, double /*inv_dt*/) {
    int N = N_dof, B = 256;
    k_r1di_copy<<<(N+B-1)/B, B>>>(d_Mv, d_v_in, N);
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
        jfnk_matvec_implicit(d_Z[j], d_gmres_w, inv_dt);

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
    for (int it = 0; it < newton_max_iter; ++it) {
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
        }
        if (it == 0 && res_norm < newton_tol) return 0;
        if (!std::isfinite(res_norm)) {
            std::fprintf(stderr, "  r1di_newton: NaN residual at iter %d\n", it);
            return -1;
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
        k_r1di_copy<<<(N+B-1)/B, B>>>(d_Ubak, d_U, N);
        double alpha = 1.0;
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
        bool rel_ok = (init_res > 0 && new_res < 0.5 * init_res);
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

    std::fprintf(stderr, "  step_implicit ENTER: step=%d dt_try=%.3e\n", step_count, dt_try);

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
