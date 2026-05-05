#pragma once

#include "grid.h"
#include "state.h"
#include "eos.h"

// 2D axisymmetric Lagrangian hydro solver (ALE, Phase A1+A2).
//
// References (cloned to ~/ale-refs/):
//   - Matterflow (Caramana compatible, triangles)
//   - HYDRO-ALE (subzone layout + Green's-theorem forces)
//   - hal3d (full ALE cycle, 3D unstructured)
//
// Data model:
//   - Structured (i,j) topology retained (nr radial × nt angular cells).
//   - Nodes at ((nr+1) × (nt+1)) vertices can move freely in (R, Z) plane.
//     R = cylindrical radius, Z = symmetry axis.
//   - Each cell has 4 corners; a "subcell" is the (cell, corner) pairing.
//   - dm[cell], m_node[node] are constant in A1/A2 (no remap).
//   - Compatible energy update: e_int -= Σ(F_subcell · v_node) · dt / m_cell.
//
// Geometry (axisymmetric revolved around Z):
//   Cell volume V = (π/3) Σ_edges (R_a² + R_a R_b + R_b²)(Z_b - Z_a)     [Pappus]
//   Edge "area vector" = 2π R_mid · (dZ, -dR)  [outward annular normal]
//
// BCs:
//   - i=0 nodes: pinned at origin (position + velocity = 0)
//   - j=0 and j=nt nodes: constrained to R=0 (velocity_R = 0, pos_R = 0)
//   - i=nr (outer) nodes: free surface, ghost P = P_surf_floor
//
// Gravity:
//   1D shell-angle-averaged. shell_mass[i] = Σ_j dm[i,j] is invariant.
//   g_node at radial index ir: magnitude = -G·M_enc(ir)/r² · r_hat.

struct Ale2DSolver {
    // ---- Size info ----
    int nr = 0, nt = 0;
    int nnode_r = 0, nnode_t = 0;   // nr+1, nt+1
    int nnode = 0;                  // nnode_r * nnode_t
    int ncell = 0;                  // nr * nt
    int nsub = 0;                   // 4 * ncell

    // ---- Node arrays (size nnode) ----
    double *d_R, *d_Z;              // position
    double *d_R_prev, *d_Z_prev;    // previous (RK2 / backup)
    double *d_vR, *d_vZ;            // velocity
    double *d_FR, *d_FZ;            // net force
    double *d_mnode;                // node mass (constant)

    // ---- Cell arrays (size ncell) ----
    double *d_dm;                   // cell mass (constant in A1/A2)
    double *d_Vol;                  // revolved volume
    double *d_Vol_prev;
    double *d_Area;                 // 2D cross-section area (for Q length scale)
    double *d_Area0;                // initial Area — constant reference length²
    double *d_rho;                  // = dm/Vol
    double *d_e_int;                // specific internal energy
    double *d_e_prev;
    double *d_P, *d_Q;              // gas pressure, artificial viscosity
    double *d_cs;                   // sound speed
    double *d_minheight;            // minimum edge-perpendicular distance (CFL)
    double *d_strain_rate;          // -dV/dt / V (compression rate)

    // ---- HSE reference (cell-centered, constant after snapshot_hse) ----
    double *d_rho0, *d_P0, *d_e0;

    // ---- HSE-defect force at nodes, captured during snapshot_hse.
    //       Subtracted from node force every step so that HSE is exactly stationary.
    double *d_FR_hse, *d_FZ_hse;

    // ---- Subcell arrays (size nsub = 4*ncell) ----
    double *d_FSR, *d_FSZ;          // subcell force on the adjacent node

    // ---- 1D angle-averaged gravity ----
    double *d_shell_mass;           // size nr, constant after init
    double *d_M_enc;                // size nr+1, prefix sum of shell_mass

    // ---- Scratch ----
    double *d_dt_cell;              // size ncell
    double *d_reduce_buf, *d_reduce_out;

    // ---- Parameters ----
    double gamma = 5.0/3.0;
    double G_const = 1.0;
    double cfl = 0.4;
    double CQ_lin = 0.5;            // linear viscosity coefficient
    double CQ_quad = 2.0;           // quadratic viscosity coefficient
    double comp_dt_frac = 0.25;
    double P_surf_floor = 0.0;

    // ---- Bookkeeping ----
    double dt_current = 0.0;
    int step_count = 0;
    bool hse_set = false;

    // ---- Lifecycle ----
    void init(const Grid& grid, const EOS& eos, double G, double cfl);
    void destroy();

    // Load Lane-Emden HSE onto the initial Lagrangian mesh (re-uses Grid's r_face, theta_face).
    // Call AFTER init.
    void init_lane_emden(double rho_c, double K_poly, double n_poly);

    // Apply radial perturbation matching init_lane_emden_perturbed.
    void apply_perturbation(double amplitude);

    // Store current state as HSE reference (d_rho0/P0/e0).
    void snapshot_hse();

    // One explicit kick-drift-kick step with Caramana compatible energy update.
    // Returns dt actually taken.
    double step(double t, double t_end);

    // Diagnostic reduction (host-side, similar to Radial1DSolver::Diagnostics).
    struct Diagnostics {
        double total_mass;
        double total_KE;
        double total_internal_E;
        double total_grav_E;
        double total_E;
        double max_mach;
        double max_v;
    };
    Diagnostics compute_diagnostics();

    // Download a radial profile (volume-weighted θ-average per radial shell).
    // Sizes: r_cell size nr; rho, P, e, vr size nr.
    void download_radial_profile(std::vector<double>& r_cell,
                                 std::vector<double>& rho_cell,
                                 std::vector<double>& P_cell,
                                 std::vector<double>& e_cell,
                                 std::vector<double>& vr_cell);
};
