#pragma once

#include <string>
#include <vector>

// ============================================================
// AthenaMHDSolver — 2D Eulerian ideal MHD on a Yee-staggered grid
// (vl2 integrator + HLLD Riemann + constrained transport for
// divB=0 preservation).
//
// Derivation dossier:  docs/derivations/mhd/manuscript.pdf
//   §A1-A5  ideal MHD conservation laws + HLLD + CT
//   §A6-A8  PLM reconstruction + VL2 predictor-corrector + CFL
//   §A9     HLLD degeneracy branches (B_x=0, Alfvén locus, HLL fallback)
//   §A10    Powell 8-wave source ≡ 0 under CT
//   §A11    Stone+08 Table-1 linear-wave IC
//
// Sketch (one step):
//   stage 1 (predictor):  order=1 donor-cell reconstruction of w^n
//                         HLLD → F^(1/2);  GS05-averaged corner EMF^(1/2)
//                         u*    = u^n − (dt/2) [∂_x F + ∂_y G]
//                         B_f*  = B_f^n − (dt/2) curl_f(EMF^(1/2))
//   stage 2 (corrector):  order=xorder PLM on w*
//                         HLLD → F^(n+1/2);  GS05 corner EMF^(n+1/2)
//                         u^{n+1}   = u^n − dt [∂_x F + ∂_y G]
//                         B_f^{n+1} = B_f^n − dt curl_f(EMF^(n+1/2))
//
// Storage — 2D MHD, no z-coupling in variables, z-momentum / B_z retained
// because §A3 eigensystem has them (out-of-plane Alfvén / slow waves).
//   cell-centred conserved:   ρ, m_x, m_y, m_z, E   (5 doubles per cell)
//   face-centred B:           B_x at i+½ faces,  B_y at j+½ faces
//                             (B_z stays cell-centred for 2D reduction)
//   corner-centred EMF:       E_z at (i+½, j+½) corners
//
// Cell-centred B (diagnostic only, recomputed from faces each cons→prim):
//   B_x_cc_{i,j} = ½(B_x_face_{i-½,j} + B_x_face_{i+½,j})
//   B_y_cc_{i,j} = ½(B_y_face_{i,j-½} + B_y_face_{i,j+½})
//
// Ghost width ng = 3 (PLM needs 2, keep 3 for future PPM and corner EMF).
// ============================================================

struct AthenaMHDSolver {
    // ---- grid --------------------------------------------------
    int nx = 0, ny = 0;
    int ng = 3;
    double dx = 0.0, dy = 0.0;
    double x_lo = 0.0, x_hi = 0.0;
    double y_lo = 0.0, y_hi = 0.0;
    double Lx = 0.0, Ly = 0.0;

    // ---- physics -----------------------------------------------
    double gamma = 5.0 / 3.0;
    double cfl = 0.4;
    double cfl_limit = 0.5;

    // ---- scheme toggles ----------------------------------------
    int  limiter = 0;   // 0 = van Leer harmonic, 1 = minmod
    int  xorder  = 2;   // 1 donor-cell, 2 PLM. (3 PPM reserved for future)

    // BC flags (match athena_vl2 convention):
    //   x_bc:  0 periodic, 1 reflecting (m_x,B_x flip), 2 outflow
    //   y_bc:  0 periodic, 1 reflecting (m_y,B_y flip), 2 outflow
    int x_bc = 0;
    int y_bc = 1;

    // ---- state (device) ----------------------------------------
    // Cell-centred conserved (5 fields):
    double *d_rho = nullptr, *d_mx = nullptr, *d_my = nullptr, *d_mz = nullptr, *d_E = nullptr;
    // Stage-1 scratch:
    double *d_rho1 = nullptr, *d_mx1 = nullptr, *d_my1 = nullptr, *d_mz1 = nullptr, *d_E1 = nullptr;
    // Cell-centred B (reconstructed from faces in cons→prim; also stores B_z directly):
    double *d_Bx_cc = nullptr, *d_By_cc = nullptr, *d_Bz_cc = nullptr;

    // Face-centred B (Yee grid):
    //   Bxf sized (nx+1+2g) × (ny+2g)      — x-normal faces  at i+½
    //   Byf sized (nx+2g)   × (ny+1+2g)    — y-normal faces  at j+½
    double *d_Bxf = nullptr, *d_Byf = nullptr;
    double *d_Bxf1 = nullptr, *d_Byf1 = nullptr;   // stage-1 scratch
    // B_z cell-centred (stage-1 scratch):
    double *d_Bz1 = nullptr;

    // Corner-centred E_z (GS05 average):   (nx+1+2g) × (ny+1+2g)
    double *d_Ez_corner = nullptr;
    // Face-centred "partial" EMF contributions from HLLD x/y fluxes:
    //   Ezx_face[i+½, j]  = −F^x_{By}  (from the x-sweep)
    //   Ezy_face[i,   j+½] =  F^y_{Bx} (from the y-sweep)
    double *d_Ezx_face = nullptr;   // lives on x-faces, same shape as d_Bxf
    double *d_Ezy_face = nullptr;   // lives on y-faces, same shape as d_Byf

    // ---- primitive scratch (cell-centred, 7 fields) -----------
    //   w = (ρ, u, v, w, B_x_cc, B_y_cc, B_z_cc, P)
    double *d_w_rho = nullptr, *d_w_u = nullptr, *d_w_v = nullptr, *d_w_w = nullptr;
    double *d_w_Bx = nullptr, *d_w_By = nullptr, *d_w_Bz = nullptr;
    double *d_w_P  = nullptr;

    // ---- face-flux buffers (7 conservatives each) --------------
    // x-faces (i+½):  F(ρ, m_x, m_y, m_z, B_y_f, B_z_cc, E)
    //                 (F^x_{B_x} = 0 by convention — normal B doesn't advect itself)
    double *d_Fx_rho=nullptr, *d_Fx_mx=nullptr, *d_Fx_my=nullptr, *d_Fx_mz=nullptr;
    double *d_Fx_By =nullptr, *d_Fx_Bz=nullptr, *d_Fx_E=nullptr;
    // y-faces (j+½):  G(ρ, m_x, m_y, m_z, B_x_f, B_z_cc, E)
    //                 (G^y_{B_y} = 0)
    double *d_Gy_rho=nullptr, *d_Gy_mx=nullptr, *d_Gy_my=nullptr, *d_Gy_mz=nullptr;
    double *d_Gy_Bx =nullptr, *d_Gy_Bz=nullptr, *d_Gy_E=nullptr;

    // ---- source-term tables (optional gravity) ----------------
    double *d_g_row = nullptr;
    std::vector<double> h_g_row;
    std::vector<double> h_phi_row;

    // ---- CFL scratch -------------------------------------------
    double *d_cfl_buf = nullptr;

    // ---- bookkeeping -------------------------------------------
    int step_count = 0;
    double dt_current = 0.0;

    // =================== public API ============================
    void init(int nx, int ny, double Lx, double Ly,
              double gamma, double cfl);
    void destroy();

    // ----- initial conditions (§A11) ---------------------------
    // Brio-Wu 1988 MHD shock tube: canonical 1D MHD benchmark in x.
    //   x<Lx/2: ρ=1.0,  P=1.0, B_y=+1.0
    //   x>=Lx/2: ρ=0.125, P=0.1, B_y=−1.0
    //   γ=2, B_x=0.75, v=0, B_z=0.  y=periodic or reflect (1D).
    void init_brio_wu();

    // Stone+08 Table-1 linear MHD wave (ρ₀=1, p₀=1/γ, v₀=0,
    // B₀=(1, √2, ½), γ=5/3 ⇒ c_f=2, c_Ax=1, c_s=½, c_s0=1).
    // `mode` ∈ {FAST=0, ALFVEN=1, SLOW=2, ENTROPY=3} and k is the
    // number of wavelengths across Lx.  Amplitude A defaults to 1e-6.
    enum LinearWaveMode { FAST_M=0, ALFVEN=1, SLOW=2, ENTROPY=3 };
    void init_linear_wave(LinearWaveMode mode, int k, double A = 1e-6);

    // Oblique linear wave (§F1 rotated-eigenvector IC, Stone+08 §6.2).
    // Wave vector k = 2π(kx_int/Lx, ky_int/Ly); vector components of
    // the eigenvector rotated by θ = atan2(ky/Ly, kx/Lx).
    // Background same as init_linear_wave.  Amplitude A = 1e-6 default.
    void init_linear_wave_oblique(LinearWaveMode mode,
                                  int kx_int, int ky_int,
                                  double A = 1e-6);

    // Field-loop (Gardiner-Stone 2005 Fig 3): circular B_z-polarised
    // flux tube of radius R=0.3 advected diagonally with v=(1, 0.5).
    // Locks div·B = 0 across 10 crossings.
    void init_field_loop(double v_adv_x = 1.0, double v_adv_y = 0.5,
                         double R = 0.3, double A0 = 1e-3);

    // Orszag-Tang vortex (Orszag-Tang 1979): canonical 2D MHD test.
    //   ρ = 25/36π,  P = 5/12π,  γ=5/3
    //   v = (−sin y, sin x),  B = (−sin y, sin 2x)
    //   L = 2π in both directions, fully periodic.
    void init_orszag_tang();

    // Ryu-Jones 1995 RJ2a shock tube — the "all 7 waves" tube
    // (Stone+08 Fig 14).  1D along x,  γ=5/3.
    //   L: ρ=1.08, v=(1.2,0.01,0.5),   P=0.95, Bx=2,   By=3.6,   Bz=2
    //   R: ρ=1.0,  v=(0,0,0),          P=1.0,  Bx=2,   By=4.0,   Bz=2
    // all divided by √(4π).  Outflow in x.  t_f = 0.2.
    void init_rj2a();

    // Ryu-Jones RJ4d switch-on slow rarefaction (Stone+08 Fig 15).
    //   L: ρ=1, v=0,   P=1,   Bx=0.7, By=+1,        Bz=0
    //   R: ρ=0.3, v=0, P=0.2, Bx=0.7, By=cos(3),    Bz=sin(3)
    // t_f = 0.16.
    void init_rj4d();

    // Circularly polarized (CP) Alfvén wave, 1D (Tóth 2000 §3.2.2).
    //   Background ρ=1, P=0.1, Bx=1;
    //   δB⊥ = 0.1 (sin 2πx, cos 2πx),  δv⊥ parallel to δB⊥ for traveling.
    //   Periodic.  t_f = 1 (one period).  γ=5/3.
    // `traveling = true`: v_x = 0; `false`: v_x = −c_A = −1 (standing).
    void init_cpaw_1d(bool traveling = true);

    // CP Alfvén wave, 2D (GS05 §5.2).
    //   Wave axis tilted θ=tan⁻¹(2) ≈ 63.4° to x-axis,  λ = 1 along rotated
    //   x₁.  Domain √5 × √5/2 periodic.  t_f = 1 (1 period).
    void init_cpaw_2d(bool traveling = true);

    // MHD blast wave (Balsara-Spicer 1999 / Stone+08 Fig 28).
    //   ρ=1 uniform;  P=10 inside r<0.1, P=0.1 outside.
    //   B = (1,1,0)/√2 (45° inclined),  v=0.
    //   Domain [-0.5,0.5] × [-0.75,0.75] periodic.  t_f = 0.2.
    void init_blast();

    // MHD rotor (Tóth 2000 Rotor Test 1, Stone+08 Fig 25).
    //   Inside r<0.1: ρ=10, rigid-body rotation at ω=200
    //                   (peak v_tang = 20 at r=0.1).
    //   Outside:     ρ=1, v=0.
    //   P=1 uniform.  B = (5/√4π, 0, 0).  γ=1.4.
    //   Outflow BC on all 4 sides.  t_f = 0.15.
    void init_rotor();

    // One vl2 predictor-corrector step. Returns dt actually used.
    double step(double t, double t_end);

    // ---- diagnostics ------------------------------------------
    struct Diagnostics {
        double total_mass;
        double total_KE;
        double total_IE;
        double total_ME;          // magnetic energy
        double total_E;           // KE + IE + ME
        double max_divB;          // cell-centred |∇·B|
        double max_v;
        double max_fast_mach;
    };
    Diagnostics compute_diagnostics();

    // ---- VTK writer -------------------------------------------
    void write_vtk_2d(const char* filename, double Lx_in, double Ly_in);

    // ---- internals --------------------------------------------
    int stride_x() const { return nx + 2 * ng; }
    int stride_y() const { return ny + 2 * ng; }
    int total_cells() const { return stride_x() * stride_y(); }
    // x-face array: (nx+1+2g) * (ny+2g)   — stride sy
    int total_fx() const { return (nx + 1 + 2 * ng) * (ny + 2 * ng); }
    // y-face array: (nx+2g) * (ny+1+2g)   — stride sy+1
    int total_fy() const { return (nx + 2 * ng) * (ny + 1 + 2 * ng); }
    // corner array: (nx+1+2g) * (ny+1+2g) — stride sy+1
    int total_corner() const { return (nx + 1 + 2 * ng) * (ny + 1 + 2 * ng); }

    // Host helpers used by tests and drivers.
    void fill_ghost();
    void cons_to_prim();
    void calc_hydro_flux_and_emf(int order);
    void apply_flux_divergence_and_ct(int stage, double dt);
    double compute_dt();
    void stage_advance(int stage, double dt);
};
