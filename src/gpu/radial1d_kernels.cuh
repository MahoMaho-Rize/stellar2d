#pragma once
#include <cmath>
#include "../eos.h"

// Device kernels for 1D Lagrangian radial stellar hydrodynamics.
// All kernels operate on Radial1DLevel's device arrays.
//
// Staggering:
//   Faces: index k = 0..nz (size nz+1).  k=0 is center (r=0), k=nz is surface.
//   Zones: index k = 0..nz-1 (size nz).  Zone k sits between face k and face k+1.
//
// Invariants:
//   v[0] = 0 (reflective BC at origin)
//   r[0] = 0
//   dm[k] is constant

static constexpr double PI4 = 12.566370614359172;    // 4π
static constexpr double PI43 = 4.188790204786391;    // 4π/3

// ========================================================================
// Zone-centered primitives (compute Vol, rho, P from r[], dm[], e_int[])
// ========================================================================

__global__
void k_rad1d_zone_primitives(
    const double* r,        // (nz+1) face radii
    const double* dm,       // (nz) zone masses
    const double* e_int,    // (nz) specific internal energy
    double* Vol,            // (nz) zone volumes (output)
    double* rho,            // (nz) zone densities (output)
    double* P,              // (nz) zone pressure (output)
    int nz, double gam)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nz) return;
    double rL = r[k], rR = r[k+1];
    double Vk = PI43 * (rR*rR*rR - rL*rL*rL);
    Vk = fmax(Vk, 1e-30);
    Vol[k] = Vk;
    double rho_k = dm[k] / Vk;
    rho[k] = rho_k;
    double e_k = fmax(e_int[k], 1e-30);
    P[k] = (gam - 1.0) * rho_k * e_k;
}

// EOS-aware primitives kernel (overload). Uses EOS.pressure(ρ, e).
__global__
void k_rad1d_zone_primitives_eos(
    const double* r,
    const double* dm,
    const double* e_int,
    double* Vol,
    double* rho,
    double* P,
    int nz, EOS eos)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nz) return;
    double rL = r[k], rR = r[k+1];
    double Vk = PI43 * (rR*rR*rR - rL*rL*rL);
    Vk = fmax(Vk, 1e-30);
    Vol[k] = Vk;
    double rho_k = dm[k] / Vk;
    rho[k] = rho_k;
    double e_k = fmax(e_int[k], 1e-30);
    P[k] = eos.pressure(rho_k, e_k);
}

// ========================================================================
// Enclosed mass at faces: M[0] = 0, M[k] = Σ dm[0..k-1]
// Performed via prefix scan on nz+1 elements. Simple serial scan in 1 block.
// ========================================================================

__global__
void k_rad1d_enclosed_mass(
    const double* dm,       // (nz) zone masses
    double* M,              // (nz+1) enclosed mass at face k
    int nz)
{
    // single-threaded prefix sum (nz is small, ~128-2048)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    double acc = 0.0;
    M[0] = 0.0;
    for (int k = 0; k < nz; ++k) {
        acc += dm[k];
        M[k+1] = acc;
    }
}

// ========================================================================
// Gravity at faces: g[k] = G * M[k] / r[k]²  (g[0] = 0 by L'Hopital)
// ========================================================================

__global__
void k_rad1d_gravity(
    const double* r,        // (nz+1)
    const double* M,        // (nz+1)
    double* g,              // (nz+1) output face gravity
    int nz, double G_const)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k > nz) return;
    double rk = r[k];
    if (k == 0 || rk < 1e-20) {
        g[k] = 0.0;
    } else {
        g[k] = G_const * M[k] / (rk * rk);
    }
}

// ========================================================================
// Tscharnuter-Winkler artificial viscosity.
// Q[k] = 0 if Δv_k ≤ ZSH√(P_k V_k)
//      else (CQ / V_k) * (Δv_k - ZSH√(P_k V_k))²
// where Δv_k = v[k] - v[k+1]  (positive = compression of zone k)
// ========================================================================

__global__
void k_rad1d_artificial_viscosity(
    const double* v,        // (nz+1)
    const double* Vol,      // (nz)
    const double* P,        // (nz)
    double* Pvsc,           // (nz) output
    int nz, double CQ, double ZSH)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nz) return;
    double dv = v[k] - v[k+1];   // >0 means zone is being compressed
    double Pk = fmax(P[k], 1e-30);
    double Vk = fmax(Vol[k], 1e-30);
    double sqrt_PV = sqrt(Pk * Vk);
    double thresh = ZSH * sqrt_PV;
    if (dv <= thresh) {
        Pvsc[k] = 0.0;
    } else {
        double excess = dv - thresh;
        Pvsc[k] = (CQ / Vk) * excess * excess;
    }
}

// ========================================================================
// Save previous state (needed for RK2 and energy update).
// Copies r -> r_prev, v -> v_prev, Vol -> Vol_prev, e_int -> e_prev
// ========================================================================

__global__
void k_rad1d_save_state(
    const double* r, double* r_prev,
    const double* v, double* v_prev,
    int n_faces)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_faces) return;
    r_prev[k] = r[k];
    v_prev[k] = v[k];
}

__global__
void k_rad1d_save_cells(
    const double* Vol, double* Vol_prev,
    const double* e_int, double* e_prev,
    int nz)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nz) return;
    Vol_prev[k] = Vol[k];
    e_prev[k] = e_int[k];
}

// ========================================================================
// Momentum update (face-centered).
// dv/dt = -A_k * (XP_k - XP_{k-1}) / dm_bar_k  -  g_k
//   A_k       = 4π r_k²
//   XP_k      = P[k] + Pvsc[k]  (zone k, which is *above* face k)
//   XP_{k-1}  = P[k-1] + Pvsc[k-1]
//   dm_bar_k  = 0.5 * (dm[k-1] + dm[k])
//
// Boundary handling:
//   Face 0 (center): v[0] = 0 enforced (do not update)
//   Face nz (surface): XP above = P_surf_floor (ghost), no Pvsc
//
// v_new = v_old + dt * (RHS)
// ========================================================================

__global__
void k_rad1d_momentum_update(
    const double* r,        // (nz+1)
    const double* dm,       // (nz)
    const double* P,        // (nz)
    const double* Pvsc,     // (nz)
    const double* g_face,   // (nz+1)
    double* v,              // (nz+1) in/out
    int nz, double P_surf_floor, double dt)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k > nz) return;
    if (k == 0) { v[0] = 0.0; return; }   // pinned center

    double rk = r[k];
    double Ak = PI4 * rk * rk;

    // Zone above face k is zone k;  zone below is zone k-1.
    double XP_above, XP_below;
    double dm_bar;
    if (k == nz) {
        // surface face: ghost zone above has P = P_surf_floor, no Pvsc
        XP_above = P_surf_floor;
        XP_below = P[k-1] + Pvsc[k-1];
        // for dm_bar at surface, use half of last zone mass (no zone above)
        dm_bar = 0.5 * dm[k-1];
    } else {
        XP_above = P[k] + Pvsc[k];
        XP_below = P[k-1] + Pvsc[k-1];
        dm_bar = 0.5 * (dm[k-1] + dm[k]);
    }
    dm_bar = fmax(dm_bar, 1e-30);

    double dpdm = (XP_above - XP_below) / dm_bar;
    double gk = g_face[k];
    double accel = -Ak * dpdm - gk;

    v[k] += dt * accel;
}

// ========================================================================
// Position update: r_new = r_old + dt * v
// Applied to faces 1..nz (face 0 stays at origin).
// ========================================================================

__global__
void k_rad1d_position_update(
    const double* v,        // (nz+1)
    double* r,              // (nz+1) in/out
    int nz, double dt)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k > nz) return;
    if (k == 0) { r[0] = 0.0; return; }
    r[k] += dt * v[k];
    // monotonicity safety: never let r[k] cross r[k-1] (handled by CFL)
}

// ========================================================================
// Energy update (adiabatic + viscous work):
//   d(e)/dt = -(P + Q) * dV/dt / dm  (per-mass specific internal energy)
// Discretized:
//   e_new = e_old - (P + Q) * (V_new - V_old) / dm
// (V_new already computed via r_new, Vol is recomputed by zone_primitives)
// ========================================================================

__global__
void k_rad1d_energy_update(
    const double* Vol,      // (nz) current volumes (after position update)
    const double* Vol_prev, // (nz) previous volumes
    const double* P,        // (nz) current gas pressure (from previous state)
    const double* Pvsc,     // (nz) current Pvsc (from previous state)
    const double* dm,       // (nz)
    double* e_int,          // (nz) in/out
    int nz, double dt)      // dt unused — absorbed into V - V_prev
{
    (void)dt;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nz) return;
    double dV = Vol[k] - Vol_prev[k];
    double Pt = P[k] + Pvsc[k];
    double dm_k = fmax(dm[k], 1e-30);
    // e_new - e_old = -Pt * dV / dm   (specific internal energy change)
    double de = -Pt * dV / dm_k;
    double e_new = e_int[k] + de;
    e_int[k] = fmax(e_new, 1e-30);
}

// ========================================================================
// RK2 average for faces: v = 0.5*(v_prev + v), r = 0.5*(r_prev + r)
// ========================================================================

__global__
void k_rad1d_rk_average_faces(
    const double* r_prev, double* r,
    const double* v_prev, double* v,
    int nz)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k > nz) return;
    if (k == 0) { r[0] = 0.0; v[0] = 0.0; return; }
    r[k] = 0.5 * (r_prev[k] + r[k]);
    v[k] = 0.5 * (v_prev[k] + v[k]);
}

__global__
void k_rad1d_rk_average_cells(
    const double* e_prev, double* e_int,
    int nz)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nz) return;
    e_int[k] = fmax(0.5 * (e_prev[k] + e_int[k]), 1e-30);
}

// ========================================================================
// CFL / compression-limited dt per zone.
// dt_k = min(
//    acoustic:    dr_k / (|v_k| + c_s_k)
//    compression: fraction * dr_k / max(Δv_k, small)   (if Δv > 0)
// )
// ========================================================================

__global__
void k_rad1d_cfl(
    const double* r,        // (nz+1)
    const double* v,        // (nz+1)
    const double* rho,      // (nz)
    const double* P,        // (nz)
    double* dt_cell,        // (nz) output
    int nz, double gam, double comp_fraction)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nz) return;
    double dr = r[k+1] - r[k];
    dr = fmax(dr, 1e-30);
    double Pk = fmax(P[k], 1e-30);
    double rho_k = fmax(rho[k], 1e-30);
    double cs = sqrt(gam * Pk / rho_k);
    double vmax = fmax(fabs(v[k]), fabs(v[k+1]));
    double dt_acoustic = dr / (vmax + cs);

    double dv_compress = v[k] - v[k+1];  // >0 if compressing
    double dt_comp = 1e30;
    if (dv_compress > 1e-20) {
        dt_comp = comp_fraction * dr / dv_compress;
    }
    dt_cell[k] = fmin(dt_acoustic, dt_comp);
}

__global__
void k_rad1d_cfl_eos(
    const double* r, const double* v, const double* rho, const double* P,
    double* dt_cell,
    int nz, EOS eos, double comp_fraction)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nz) return;
    double dr = r[k+1] - r[k];
    dr = fmax(dr, 1e-30);
    double Pk = fmax(P[k], 1e-30);
    double rho_k = fmax(rho[k], 1e-30);
    double cs = eos.sound_speed(rho_k, Pk);
    double vmax = fmax(fabs(v[k]), fabs(v[k+1]));
    double dt_acoustic = dr / (vmax + cs);
    double dv_compress = v[k] - v[k+1];
    double dt_comp = 1e30;
    if (dv_compress > 1e-20) dt_comp = comp_fraction * dr / dv_compress;
    dt_cell[k] = fmin(dt_acoustic, dt_comp);
}

// ========================================================================
// Diagnostics:
//   total_mass = Σ dm[k]                         (constant, sanity check)
//   total_KE   = Σ 0.5 * m_face * v²             (where m_face = 0.5*(dm[k-1]+dm[k]))
//                Simplified: KE of zone = 0.5 * dm[k] * ((v[k]² + v[k+1]²)/2)
//   total_IE   = Σ dm[k] * e_int[k]
//   total_PE   = -Σ dm[k] * G*M_avg/r_avg        (using shell-averaged M,r)
// ========================================================================

__global__
void k_rad1d_diag_per_zone(
    const double* dm,
    const double* e_int,
    const double* rho,
    const double* P,
    const double* v,       // (nz+1)
    const double* M,       // (nz+1) enclosed mass
    const double* r,       // (nz+1)
    double* out_mass,      // (nz)
    double* out_KE,
    double* out_IE,
    double* out_PE,
    double* out_mach,      // per-zone Mach
    double* out_vmax,      // per-zone max |v|
    int nz, double gam, double G_const)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nz) return;
    double dm_k = dm[k];
    out_mass[k] = dm_k;
    double vk = 0.5 * (v[k] + v[k+1]);
    out_KE[k] = 0.5 * dm_k * vk * vk;
    out_IE[k] = dm_k * e_int[k];
    // grav PE approximation: -G * M_cell * dm / r_cell where cell center
    double r_cell = 0.5 * (r[k] + r[k+1]);
    double M_cell = 0.5 * (M[k] + M[k+1]);
    out_PE[k] = (r_cell > 1e-20) ? -G_const * M_cell * dm_k / r_cell : 0.0;
    // Mach
    double Pk = fmax(P[k], 1e-30);
    double rho_k = fmax(rho[k], 1e-30);
    double cs = sqrt(gam * Pk / rho_k);
    double vmag = fmax(fabs(v[k]), fabs(v[k+1]));
    out_mach[k] = vmag / fmax(cs, 1e-30);
    out_vmax[k] = vmag;
}

__global__
void k_rad1d_diag_per_zone_eos(
    const double* dm, const double* e_int, const double* rho, const double* P,
    const double* v, const double* M, const double* r,
    double* out_mass, double* out_KE, double* out_IE, double* out_PE,
    double* out_mach, double* out_vmax,
    int nz, EOS eos, double G_const)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nz) return;
    double dm_k = dm[k];
    out_mass[k] = dm_k;
    double vk = 0.5 * (v[k] + v[k+1]);
    out_KE[k] = 0.5 * dm_k * vk * vk;
    out_IE[k] = dm_k * e_int[k];
    double r_cell = 0.5 * (r[k] + r[k+1]);
    double M_cell = 0.5 * (M[k] + M[k+1]);
    out_PE[k] = (r_cell > 1e-20) ? -G_const * M_cell * dm_k / r_cell : 0.0;
    double Pk = fmax(P[k], 1e-30);
    double rho_k = fmax(rho[k], 1e-30);
    double cs = eos.sound_speed(rho_k, Pk);
    double vmag = fmax(fabs(v[k]), fabs(v[k+1]));
    out_mach[k] = vmag / fmax(cs, 1e-30);
    out_vmax[k] = vmag;
}
