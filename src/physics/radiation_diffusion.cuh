#pragma once

// 1D radial grey radiation diffusion.
//
// Solves ∂(ρ e)/∂t = -div F_rad / ρ with
//   F_rad = -D · ∂(a T⁴) / ∂r
//   D     = c · λ / (κ ρ)      [optically thick limit: λ = 1/3]
// in spherical geometry:
//   div F = (1/r²) ∂(r² F_rad) / ∂r
//
// Explicit forward Euler subcycle:
//   e_new[k] = e[k] - dt · div_F[k] / ρ[k]
// where div_F uses face-centered F_rad.
//
// dt_rad ≤ 0.5 · dr² · κρ / c   (parabolic CFL; highly restrictive)
//
// Reusable by Path B — kernel signature takes raw arrays,無 geometry assumption.

#include <cmath>
#include "opacity.h"
#include "eos_pre_ms.h"

#ifdef __CUDACC__

// Radiation diffusion params (code units)
// c_light: speed of light in code units
// a_rad: radiation constant (cgs: 7.566e-15; in code units depends on scaling)
struct RadDiffParams {
    double c_light = 1.0;   // speed of light (code units)
    double a_rad   = 1.0;   // radiation constant (code units)
    OpacityParams opacity;
};

// One explicit forward-Euler radiation diffusion step.
// Signature matches radial1d Lagrangian staggering:
//   e_int[nz], rho[nz], dm[nz] (zone-centered, constant mass)
//   r[nz+1] (face radii)
//   T_cache[nz] (optional; if NULL we recompute from e and EOS)
__global__
inline void k_rad_diffusion_1d(
    double* e_int,          // (nz)  specific internal energy (updated)
    const double* rho,      // (nz)
    const double* r,        // (nz+1) face radii
    const double* dm,       // (nz) zone masses (for flux -> energy conversion)
    double* T_work,         // (nz) scratch: temperature
    double* F_work,         // (nz+1) scratch: radiation flux at faces
    int nz,
    EOS eos,
    RadDiffParams pars,
    double dt_sub)
{
    // Phase 1: each thread computes T[k] from (ρ, e) via EOS.
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < nz) {
        double r_k = fmax(rho[k], 1e-30);
        double e_k = fmax(e_int[k], 1e-30);
        T_work[k] = eos.temperature_from_rho_e(r_k, e_k);
    }
    // Need global sync between phases — this kernel only does Phase 1.
}

// Phase 2 kernel: compute F_rad at faces (needs T_work and ρ ready).
// Must be launched AFTER Phase 1 with cudaDeviceSynchronize or stream fence.
__global__
inline void k_rad_diffusion_1d_flux(
    const double* T_work,   // (nz)
    const double* rho,      // (nz)
    const double* r,        // (nz+1)
    double* F_work,         // (nz+1)
    int nz,
    EOS eos,
    RadDiffParams pars)
{
    int kf = blockIdx.x * blockDim.x + threadIdx.x;
    if (kf > nz) return;
    if (kf == 0 || kf == nz) {
        // Reflective at center (symmetric T → zero flux).
        // At surface: radiate to space,approximated 0 flux for now
        // (more rigorous: F = σ T_surf⁴; skip here, add later).
        F_work[kf] = 0.0;
        return;
    }
    // Face between zones kf-1 and kf.
    double rho_f = 0.5 * (rho[kf-1] + rho[kf]);
    double T_f   = 0.5 * (T_work[kf-1] + T_work[kf]);
    double kap_f = grey_opacity(rho_f, T_f, pars.opacity);
    double D_f   = pars.c_light / (3.0 * kap_f * rho_f);

    // Face area 4π r²
    double r_f = r[kf];
    double A_f = 4.0 * 3.14159265358979323846 * r_f * r_f;

    // Grad of aT⁴: central diff across face
    // dr between zone centers
    double rc_lo = 0.5 * (r[kf-1] + r[kf]);
    double rc_hi = 0.5 * (r[kf]   + r[kf+1]);
    double dr_zc = rc_hi - rc_lo;
    if (dr_zc < 1e-30) dr_zc = 1e-30;
    double aT4_lo = pars.a_rad * pow(T_work[kf-1], 4.0);
    double aT4_hi = pars.a_rad * pow(T_work[kf],   4.0);
    double grad_aT4 = (aT4_hi - aT4_lo) / dr_zc;

    F_work[kf] = -D_f * grad_aT4 * A_f;  // total luminosity thru face
}

// Phase 3 kernel: update e_int from div F.
// ∂(ρ e)/∂t · V = -div F · V = -(F_out - F_in)
// Lagrangian: dm = ρ V constant,so d(e)/dt = -(F_out - F_in) / dm
__global__
inline void k_rad_diffusion_1d_update(
    double* e_int,
    const double* F_work,
    const double* dm,
    int nz,
    double dt_sub)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nz) return;
    double dE = F_work[k+1] - F_work[k];   // positive = energy flowing out
    e_int[k] -= dt_sub * dE / fmax(dm[k], 1e-30);
    // Floor
    if (e_int[k] < 1e-30) e_int[k] = 1e-30;
}

// Compute safe dt for explicit radiation diffusion (parabolic CFL):
//   dt_rad ≈ 0.25 · min_k (dr_k² · κ_k · ρ_k / c)
__global__
inline void k_rad_diffusion_dt(
    const double* rho,
    const double* T_work,
    const double* r,
    double* dt_out,   // (nz) per-zone dt
    int nz,
    RadDiffParams pars)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nz) return;
    double dr = r[k+1] - r[k];
    double r_k = fmax(rho[k], 1e-30);
    double T_k = fmax(T_work[k], 1.0);
    double kap = grey_opacity(r_k, T_k, pars.opacity);
    // Parabolic: dt ≤ C · dr² · (3κρ)/c  (coefficient 0.25 for safety)
    dt_out[k] = 0.25 * dr * dr * 3.0 * kap * r_k / pars.c_light;
}

#endif  // __CUDACC__
