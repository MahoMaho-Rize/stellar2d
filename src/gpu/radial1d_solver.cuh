#pragma once

#include "../grid.h"
#include "../state.h"
#include "../eos.h"

// 1D radial Lagrangian stellar hydrodynamics solver (MESA RSP-inspired).
//
// Primary design goals:
//   - Exact mass conservation (Lagrangian mass zones, dm_k fixed in time)
//   - Well-posed origin: v_center = 0 pinned at r = 0 (no wedge cell pathology)
//   - Stable surface: P_surf floor prevents rhoE collapse to KE
//   - Tscharnuter-Winkler artificial viscosity: shock-triggered only
//   - Compression-based dt limiter (not just acoustic CFL)
//
// Staggering:
//   - Faces (interfaces): r[k], v[k], k = 0..nz
//     r[0]    = 0 (center, pinned)
//     r[nz]   = R_star (surface)
//     v[0]    = 0 (reflective BC at center)
//   - Zones (cells):    Vol[k], e_int[k], rho[k], P[k], Pvsc[k], dm[k], k = 0..nz-1
//     Zone k sits between face k and face k+1
//     dm[k] = rho[k] * Vol_3d[k] is zone mass (constant over time)
//
// Equations (Lagrangian):
//   dr_k/dt        = v_k                           (face motion)
//   dv_k/dt        = -A_k * (XP_k - XP_{k-1}) / dm_bar_k - g_k   (momentum)
//   d(rho e)_k/dt  = -P_tot_k * dV_k/dt / V_k      (adiabatic)
// where
//   A_k       = 4π r_k²                                          (face area)
//   dm_bar_k  = 0.5 * (dm_{k-1} + dm_k)                          (mass around face)
//   XP_k      = P_k + Pvsc_k                                      (total pressure)
//   g_k       = G * M_k / r_k²                                    (gravity, M_k = Σ dm_j, j<k)
//   Pvsc_k    = (CQ/V_k) * max(0, Δv_k - ZSH√(PV))²              (artif viscosity)
//   Δv_k      = v_k - v_{k+1}  > 0 means compression
//
// Boundary conditions:
//   v[0] = 0 (pinned, reflective at center)
//   At the surface face k = nz: use P_surf_floor (ghost zone pressure) when computing XP gradient

struct Radial1DLevel {
    int nz;  // number of zones

    // Face-centered (size nz+1)
    double* d_r;        // face radius
    double* d_v;        // face velocity
    double* d_M;        // enclosed mass at face k (M[0] = 0, M[nz] = total)
    double* d_gr;       // face gravity g = G*M/r² (g[0] = 0 by L'Hopital)
    double* d_r_prev;   // previous face radius (for RK2)
    double* d_v_prev;   // previous face velocity

    // Zone-centered (size nz)
    double* d_dm;       // zone mass (constant)
    double* d_Vol;      // zone volume V = (4π/3)(r[k+1]³ - r[k]³)
    double* d_rho;      // zone density = dm / Vol
    double* d_e_int;    // zone specific internal energy
    double* d_P;        // zone gas pressure (from EOS)
    double* d_Pvsc;     // zone artificial viscosity
    double* d_rhoE;     // zone total energy density (for diagnostics / save)
    double* d_Vol_prev; // previous volume (for energy update)
    double* d_e_prev;   // previous specific internal energy

    // HSE reference (for diagnostics and surface floor)
    double* d_rho0;     // initial density profile (zone-centered)
    double* d_P0;       // initial pressure profile
    double* d_r0;       // initial face radii

    // Scratch
    double* d_dt_cell;  // per-zone dt (for reduction)
    double* d_scratch;  // generic scratch for reductions
};

struct Radial1DSolver {
    // Physical / numerical parameters
    double gamma = 5.0/3.0;
    double G_const = 1.0;
    double cfl = 0.4;

    // Artificial viscosity (Tscharnuter-Winkler)
    double CQ = 2.0;      // viscosity coefficient
    double ZSH = 0.1;     // shock-detection threshold (fraction of sound crossing speed)

    // Surface pressure floor
    double P_surf_floor = 0.0;   // P at ghost zone beyond nz-1; set to initial P0[nz-1]

    // Compression dt limit (MESA: dt ≤ fraction * Δr / Δv)
    double comp_dt_fraction = 0.1;

    // State
    Radial1DLevel lev;
    double dt_current = 0.0;
    int step_count = 0;
    bool hse_set = false;

    // Lifecycle
    void init(int nz, double gamma, double G, double cfl);
    void destroy();

    // Initialize from a Lane-Emden polytrope (equal-mass shells).
    // Central density rho_c, polytropic constant K, index n.
    void init_lane_emden(double rho_c, double K_poly, double n_poly);

    // Apply a radial-only pressure perturbation: P *= (1 + amp*sin(π r/R))
    // Also adjust density self-consistently for adiabatic perturbation:
    // rho *= (1 + amp*sin(πr/R))^(1/γ)  (isentropic along the perturbation)
    // Note: consistency with the 2D test requires a *density* perturbation
    //       ρ' = ρ₀ (1 + amp sin(πr/R)),
    //       P' = P₀ (1 + γ amp sin(πr/R))   (adiabatic).
    // See init_lane_emden_perturbed() in src/init/lane_emden.cpp.
    void apply_perturbation(double amplitude);

    // Time step (explicit RK2 with compression-limited dt)
    double step(double t, double t_end);

    // Compute dt from both acoustic CFL and compression limit
    double compute_dt();

    // Diagnostics: download profile to host (size nz zones, nz+1 faces)
    void download_profile(std::vector<double>& r_face,
                          std::vector<double>& v_face,
                          std::vector<double>& rho_cell,
                          std::vector<double>& P_cell,
                          std::vector<double>& e_cell);

    // Compute conservation diagnostics (mass, total energy) on GPU
    struct Diagnostics {
        double total_mass;
        double total_KE;
        double total_internal_E;
        double total_grav_E;
        double total_E;
        double max_mach;
        double max_vr;
    };
    Diagnostics compute_diagnostics();

    // Snapshot current state as HSE reference (for floors)
    void snapshot_hse();
};
