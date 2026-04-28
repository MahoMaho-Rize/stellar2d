#pragma once

#include "../grid.h"
#include "../state.h"
#include "../eos.h"
#include "gmg_gpu.cuh"

// Low-Mach fully-implicit GPU solver for stellar evolution.
//
// Equations (primitive variables ρ, vr, vθ, T on cell centers):
//   ∂ρ/∂t + ∇·(ρv) = 0                                (continuity)
//   ρ(∂v/∂t + v·∇v) = -∇π + ρg                        (momentum)
//   ρcₚ(∂T/∂t + v·∇T) = ∇·(κ∇T) + εnuc + (dP₀/dt)   (energy)
//   ∇·(β₀ v) = S                                       (constraint)
//
// Phase 1 simplification: adiabatic, no nuclear reactions, no conduction.
//   ∂ρ/∂t + ∇·(ρv) = 0
//   ∂(ρv)/∂t + ∇·(ρvv) + ∇π = ρg + geometric source
//   ∂(ρe)/∂t + ∇·(ρev) + P₀ ∇·v = 0
//   P = (γ-1)ρe   (equation of state, determines P₀)
//
// Time integration: Backward Euler, Newton-GMRES (JFNK).
// Spatial: upwind advection (donor cell) + central difference pressure/gravity.
// Linear solver: GMRES with block-diagonal scaling preconditioner.
// Gravity: GMG constant-coefficient Poisson.

enum class PrecondType { NONE, BLOCK_JACOBI, SIMPLE, COMBINED, LINE_JACOBI, BLOCK_SCHUR, PBP };

struct LowMachSolver {
    void init(const Grid& grid, const EOS& eos, double G, double cfl,
              PrecondType pc = PrecondType::LINE_JACOBI);
    void upload_state(const Grid& grid, const State& state);
    void download_state(const Grid& grid, State& state);
    double step(double t, double t_end);
    void destroy();

    int nr, nt, ng;
    double gamma, G_const, cfl_num;
    int total_phys, total_ghost;
    bool initialized = false;
    bool hse_set_externally = false;

    // Device grid arrays
    double *d_r_face, *d_r_center, *d_dr;
    double *d_theta_face, *d_theta_center, *d_dtheta;
    double *d_cell_volume, *d_area_r, *d_area_theta;
    double *d_sin_theta_face, *d_sin_theta_center;

    // Precomputed gradient stencil weights (branchless pressure gradient).
    // dP/dr = wr_m*(P_{i-1}-P_c) + wr_p*(P_{i+1}-P_c)
    // (1/r)dP/dθ = wt_m*(P_{j-1}-P_c) + wt_p*(P_{j+1}-P_c)
    // Boundary cells have zero weights → clamped neighbor reads are harmless.
    double *d_grad_r_wm, *d_grad_r_wp;
    double *d_grad_t_wm, *d_grad_t_wp;

    // State: ρ, ρvr, ρvθ, ρe (with ghost cells)
    double *d_rho, *d_mr, *d_mtheta, *d_rhoE;

    // Gravity potential Φ (physical cells only) — part of state vector for 5-DOF
    double *d_phi;

    // Perturbation pressure π (physical cells only)
    double *d_pi;

    // Background hydrostatic equilibrium
    double *d_rho0, *d_P0, *d_phi0;

    // 1D radial gravity: g_r(i) = -G·M(<r_i)/r_i², angle-averaged
    double *d_gr;       // current gravity (size nr)
    double *d_gr0;      // reference gravity at HSE (size nr)
    double *d_shell_mass; // scratch for shell mass computation (size nr)

    // Packed state vectors for Newton (5*n: ρ, ρvr, ρvθ, ρe, Φ)
    double *d_Un;      // U^n saved state
    double *d_Un_prev; // U^{n-1} for extrapolation initial guess
    double dt_prev = 0.0;  // dt from previous step (for extrapolation ratio)
    double *d_Fk;      // F(U^k) residual
    double *d_residual; // scratch for residual computation

    // GMRES arrays
    static constexpr int GMRES_RESTART = 120;
    double *d_gmres_V[GMRES_RESTART + 1];
    double *d_gmres_Z[GMRES_RESTART + 1]; // preconditioned
    double *d_gmres_w;
    double *d_gmres_Uk; // state save for matvec & line search

    // Work buffers
    double *d_work_a, *d_work_b;
    double *d_rhs_poisson;
    double *d_inv_rho; // 1/ρ for variable-coefficient Poisson (future)
    double *d_residual_ls; // line search F output
    double *d_scale;   // clamp scaling: max(1, |U|) per DOF (4*n)
    double *d_scale_R; // MUSIC right scaling (unknowns), 4*n
    double *d_scale_L; // MUSIC left scaling (residuals), 4*n

    // Block-diagonal Jacobi preconditioner: 4×4 block inverse per cell
    double *d_blk_diag;
    // Un-inverted 4×4 block diagonal (for ADI middle step: D·v)
    double *d_blk_J;

    // SIMPLE preconditioner scratch (n = nr*nt each)
    double *d_Ap;           // momentum diagonal: 1/dt + upwind_coeff (per cell)
    double *d_simple_p;     // pressure correction from GMG
    double *d_simple_div;   // divergence RHS for pressure Poisson
    double *d_simple_vr_s, *d_simple_vt_s; // intermediate velocity (star)

    // Second GMG instance for pressure Poisson (separate from gravity)
    GmgGpu gmg_pressure;

    // Third GMG instance for Schur complement Helmholtz (∇² - σ)
    GmgGpu gmg_schur;
    double *d_sigma_schur; // σ(x) = 4πGρ/Ap per cell
    double *d_poisson_scale; // S(x) = max(|cC|, 1) per cell — Poisson residual scaling
    double *d_schur_rhs; // scratch for Schur RHS assembly

    PrecondType precond_type;

    // GMG for gravity
    GmgGpu gmg;

    // Adaptive dt
    double dt_current;
    double dt_good = 0.0;
    int step_count = 0;

    // Atmosphere threshold: cells with ρ₀ < atm_rho_thresh are frozen
    double atm_rho_thresh = 0.0;

    // Sponge layer: smooth velocity damping near outer boundary
    double sponge_r_start = 0.0;  // inner edge (auto-set from ρ₀)
    double sponge_r_top = 0.0;    // outer edge (= R_outer)
    double sponge_kappa = 10.0;   // damping strength

    // Internal methods
    void compute_residual(double* d_res);
    void compute_F(double* d_F, double dt);
    void apply_sponge(double dt);
    void jfnk_matvec(const double* d_v, double* d_Jv, double dt);
    int gmres_solve(double* d_x, const double* d_b, double dt,
                    double tol, int max_iter);
    void pack_state(double* d_packed);
    void unpack_set(const double* d_packed);
    void unpack_delta(const double* d_delta, double alpha);
    void apply_floor();
    void solve_gravity();
    void compute_gravity_1d();
    void launch_ghost();
    double compute_cfl_dt();
    void compute_scaling();
    void clamp_correction(double* d_delta, double max_rel_change);
    void assemble_block_jacobi(double dt);
    void assemble_simple(double dt);
    void apply_preconditioner(const double* d_v, double* d_Mv, double dt);
    void apply_simple(const double* d_v, double* d_Mv, double dt);
    void apply_block_schur(const double* d_v, double* d_Mv, double dt);
    void assemble_schur_sigma(double dt);

    // Snapshot current state as HSE reference (call before adding perturbations)
    void snapshot_hse();
    // Diagnostics: compute R(U) on initial state, print max residual per equation
    void diagnose_hse_residual();
};
