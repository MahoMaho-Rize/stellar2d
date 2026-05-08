#pragma once

#include <vector>
#include <string>

// ============================================================
// StrangSolver — Cartesian 2D Strang-splitting Euler solver
//
// Grid:  uniform Cartesian [x_lo,x_hi] × [y_lo,y_hi]
// BCs:   x-periodic, y-bottom reflective, y-top outflow
// Flux:  MUSCL-Hancock (MC limiter) + LM-HLLC
// Split: X(dt/2) → Y(dt/2) → Y(dt/2) → X(dt/2)
//
// Storage (perturbation form):
//   d_rho  = ρ  − ρ̄(y)           density perturbation
//   d_mx   = ρu                   total x-momentum  (background = 0)
//   d_my   = ρv                   total y-momentum  (background = 0)
//   d_E    = E  − p̄(y)/(γ−1)     energy perturbation
//
// 1-D background arrays (ny elements, device):
//   d_rho_bar[j], d_p_bar[j]  — isentropic HSE
// ============================================================

struct StrangSolver {
    // ---- grid ------------------------------------------------
    int    nx, ny, ng;          // physical cells, ghost width (=2)
    double dx, dy;
    double x_lo, y_lo, x_hi, y_hi;
    double Lx, Ly;

    // ---- physics ---------------------------------------------
    double gamma;
    double g_grav;              // constant downward gravity (>0)
    double cfl_number;
    double K_poly;              // isentropic constant  p = K ρ^γ
    double rho0_bottom;         // ρ at y = y_lo

    // ---- state (device) --------------------------------------
    double *d_rho, *d_mx, *d_my, *d_E;   // perturbation conserved

    // ---- 1-D HSE background (device, ny elements each) -------
    double *d_rho_bar, *d_p_bar;

    // ---- host copies of background ---------------------------
    std::vector<double> h_rho_bar, h_p_bar;

    // ---- scratch (device, 4 * total_cells each) ----
    //  face states from MUSCL-Hancock: (ρ, u, v, P) per cell
    double *d_wL, *d_wR;

    // ---- CFL scratch (device, nx*ny) ----
    double *d_cfl_buf;

    // ---- bookkeeping -----------------------------------------
    int    step_count;

    // ---- scheme toggles --------------------------------------
    // Rieper-style low-Mach correction on the HLLC contact-wave S*
    // pressure jump.  Default true (required by Andrassy-style low-M
    // stratified convection, which needs pressure-dissipation
    // suppression at M→0).  Flip to false for acoustic / linwave
    // tests where the pressure jump IS the physics.
    bool   use_lm_fix = true;

    // ---- public API ------------------------------------------
    void   init(int nx, int ny,
                double Lx, double Ly,
                double gamma, double g, double cfl,
                double K = 1.0, double rho0 = 1.0);

    void   init_bubble(double x0, double y0, double R0,
                       double dS,        // entropy boost factor
                       double eps,       // perturbation amplitude
                       int    k_mode,    // azimuthal wave number
                       double dr_smooth); // smoothing width (≈3Δx)

    double step(double t, double t_end);

    // Write Cartesian RECTILINEAR_GRID VTK
    void   write_vtk(const char* filename);

    // Diagnostics (computed on host after download)
    double total_mass();
    double total_energy();
    double max_velocity();

    void   destroy();

    // ---- internal helpers ------------------------------------
    int  stride()      const { return nx + 2 * ng; }
    int  total_cells() const { return stride() * (ny + 2 * ng); }

    void   fill_ghost_x();
    void   fill_ghost_y();
    void   sweep_x(double dt);
    void   sweep_y(double dt);
    double compute_dt();
};
