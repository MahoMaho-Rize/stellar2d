#pragma once

#include <vector>

// Cartesian 2D staggered quad Lagrangian (Caramana compatible).
//
// Uniform (nx, ny) cell grid in (x, y) ∈ [0, Lx] × [0, Ly].
// Node positions free to move. No axisymmetric corrections, no gravity (A1).
//
// Primary target: verify Caramana compatible hydro works on simple planar
// problems (Sod shock tube, Noh) before tackling curvilinear / axisymmetric.
//
// References (~/ale-refs/):
//   - Matterflow: triangle Caramana-compatible (Main.c, MeshObject.c)
//   - HYDRO-ALE:  subzone Green's theorem force layout
//   - hal3d:      compatible energy Σ(F_subcell · v_node)

struct CartLagSolver {
    // ---- Size info ----
    int nx = 0, ny = 0;            // physical cells
    int nnode_x = 0, nnode_y = 0;  // nx+1, ny+1
    int nnode = 0;
    int ncell = 0;
    int nsub = 0;                   // 4 * ncell

    // ---- Node arrays (nnode) ----
    double *d_X, *d_Y;              // positions
    double *d_vX, *d_vY;             // velocities
    double *d_FX, *d_FY;             // net forces
    double *d_mnode;                 // node mass (constant in A1)
    double *d_dX, *d_dY;             // displacement (used by compatible energy)

    // ---- Cell arrays (ncell) ----
    double *d_dm;                    // cell mass (constant in A1)
    double *d_Vol;                   // cell area (planar 2D "volume")
    double *d_Area0;                 // initial area — reference length² for Q
    double *d_rho;
    double *d_e_int;
    double *d_P, *d_Q, *d_cs;
    double *d_minheight;
    double *d_strain_rate;

    // ---- Subcell arrays (nsub) ----
    double *d_FSX, *d_FSY;           // subcell force on adjacent node

    // ---- Scratch ----
    double *d_dt_cell;
    double *d_reduce_buf;

    // ---- Parameters ----
    double gamma = 1.4;
    double cfl = 0.25;
    double CQ_lin = 0.5;
    double CQ_quad = 2.0;
    double comp_dt_frac = 0.25;

    // Boundary type: 0 = reflective walls on all 4 sides
    int bc_type = 0;

    // Downward gravity (positive = pointing in -y direction). 0 = off.
    double g_y = 0.0;

    // ---- Bookkeeping ----
    double dt_current = 0.0;
    int step_count = 0;

    // ---- Lifecycle ----
    void init(int nx, int ny, double Lx, double Ly, double gamma, double cfl);
    void destroy();

    // Sod shock tube IC: x < Lx/2 → (ρ_L, P_L), x ≥ Lx/2 → (ρ_R, P_R); v = 0.
    void init_sod();
    // Uniform IC: ρ, P, v = (vx, vy) constant.
    void init_uniform(double rho, double P, double vx, double vy);

    // Polytropic HSE slab: column in hydrostatic equilibrium under constant gravity g_y.
    //   ρ(y) = ρ_base · (1 − y/y_top)^(1/(γ−1))            [polytrope, θ=1−y/y_top]
    //   P(y) = K · ρ(y)^γ,  with K chosen so dP/dy = -ρ g.
    // Sets g_y on the solver. Uses full box height Ly as y_top.
    // Optional sinusoidal perturbation on e_int (for pulsation tests).
    void init_hse_polytrope(double rho_base, double g_val,
                             double perturb_amp = 0.0);

    // One kick-drift-kick step with Caramana compatible energy update.
    double step(double t, double t_end);

    struct Diagnostics {
        double total_mass;
        double total_KE;
        double total_internal_E;
        double total_PE;         // Σ m_node · g · Y_node
        double total_E;
        double max_v;
        double max_mach;
    };
    Diagnostics compute_diagnostics();

    // Dump 1D slice along y = Ly/2 (for Sod comparison).
    void download_xslice(std::vector<double>& x,
                         std::vector<double>& rho,
                         std::vector<double>& P,
                         std::vector<double>& vx,
                         std::vector<double>& e_int);
};
