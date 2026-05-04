#pragma once

#include "../grid.h"
#include "../state.h"
#include "../eos.h"
#include "fas2_common.cuh"
#include "gmg_gpu.cuh"

// FAS (Full Approximation Scheme) nonlinear multigrid for the low-Mach
// 4-DOF Euler system: (ρ, ρvr, ρvθ, E).
//
// Backward Euler: F(U) = (U - Uⁿ)/dt - R(U) = 0
//
// FAS V-cycle:
//   1. Pre-smooth: damped block Jacobi on F(u) = 0
//   2. Restrict: u_H = Î·u_h, f_H = R(Î·u_h) + Î·(f_h - R(u_h))
//      where f_h = Uⁿ/dt (source from time derivative)
//   3. Recurse on coarse level
//   4. Prolongate: u_h += P·(u_H - Î·u_h)
//   5. Post-smooth

struct FasLevel2 {
    int nr, nt, ng;
    int total;  // (nr + 2*ng) * (nt + 2*ng)
    int phys;   // nr * nt

    // Grid geometry (device)
    double *d_r_face, *d_r_center, *d_dr;
    double *d_theta_face, *d_theta_center, *d_dtheta;
    double *d_cell_volume, *d_area_r, *d_area_theta;
    double *d_sin_theta_face, *d_sin_theta_center;

    // State arrays with ghost cells: ρ, ρvr, ρvθ, ρe
    double *d_rho, *d_mr, *d_mt, *d_rhoE;

    // RHS for FAS (includes τ-correction): size 4*phys
    double *d_fas_rhs;

    // Residual scratch: size 4*phys
    double *d_res;

    // Saved Uⁿ and Uⁿ⁻¹ for time derivative: size 4*phys each
    double *d_Un;
    double *d_Un_prev;

    // Saved restricted state before coarse solve (for prolongation correction): size 4*phys
    double *d_save;

    // HSE reference for well-balanced residual
    double *d_rho0, *d_P0;

    // Pre-computed WB residual of HSE state: R_WB(U₀) on this level (4*phys)
    // Should be ~0 but nonzero due to discrete HSE inconsistency.
    // Subtracted from fas_rhs after restrict_defect to ensure F(U₀)=0 exactly.
    double *d_hse_defect;

    // 1D gravity arrays
    double *d_gr, *d_gr0;
    double *d_shell_mass;

    // Block Jacobian diagonal (4×4 per cell, 16*phys)
    double *d_blk_inv;

    // SIMPLE smoother scratch (per level)
    double *d_Ap;           // momentum diagonal (phys)
    double *d_vr_s, *d_vt_s; // predicted velocity (phys each)
    double *d_div_s;        // divergence scratch (phys)
    double *d_dp;           // pressure correction (phys)
    double *d_poisson_rhs;  // Poisson RHS (phys)
    double *d_inv_Ap;       // 1/Ap (phys)

    // GMRES(k) scratch (per level): k=3, right-preconditioned
    // V[0..k]: Krylov basis vectors, each 4*phys
    // Z[0..k-1]: preconditioned search directions, each 4*phys
    static constexpr int GMRES_K = 30;
    double *d_gmres_V[GMRES_K + 1];   // Krylov basis vectors, each 4*phys
    double *d_gmres_Z[GMRES_K];       // preconditioned directions, each 4*phys
    double *d_gmres_Ubak;             // state backup for matvec, size 4*phys
    double *d_Fk;                     // saved F(U) for JFNK matvec, size 4*phys
    double *d_gmres_w;                // matvec scratch, size 4*phys

    // Viallet 2016 eq 72 per-cell asymmetric scaling (fas2 fix 3/4).
    // Size 4*phys each. Rebuilt each Newton iter from (ρ₀, P₀, current state).
    //   L[eq,cell]: left scale  (applied to residual F before Krylov)
    //   R[eq,cell]: right scale (applied to δV → δU after Krylov)
    //   invL[eq,cell] = 1/L: cached for matvec output scaling
    //
    // L_ρ = ρ₀                             R_ρ = ρ₀
    // L_m = ρ₀·max(|v|, α₁·c_s)  α₁=1e-5   R_m = max(|v|, α₂·c_s)  α₂=1
    // L_E = ρ₀·c_s²                        R_E = c_s²
    double *d_scale_L;
    double *d_scale_R;
    double *d_scale_invL;
    GmgGpu pressure_gmg;   // per-level pressure Poisson solver
};

struct FasSolver2 {
    void init(const Grid& grid, const EOS& eos, double G, double cfl);
    void destroy();

    // Solve F(U) = (U-Uⁿ)/dt - R(U) = 0 using FAS V-cycles.
    // State is in d_rho/d_mr/d_mt/d_rhoE on finest level.
    // Returns number of V-cycles used.
    int solve(double dt, double g0_over_dt, int max_cycles = 20, double tol = 1e-4);

    // Upload / download state from host
    void upload_state(const Grid& grid, const State& state);
    void download_state(const Grid& grid, State& state);

    // Snapshot HSE reference (call once on equilibrium state)
    void snapshot_hse();

    // Full time step (manages dt, calls solve)
    double step(double t, double t_end);

    // Explicit RK2 time step (CFL-limited, for reference runs)
    double step_explicit(double t, double t_end);

    // CFL-based dt estimate (|v|+cs signal speed)
    double compute_cfl_dt();

    int n_levels = 0;
    FasLevel2 levels[12];

    double gamma, G_const, cfl_num;
    EOS eos;
    double dt_current = 0.0;
    double dt_prev = 0.0;
    double atm_rho_thresh = 0.0;
    int step_count = 0;
    bool hse_set = false;
    bool use_simple_smoother = true; // true=SIMPLE (Poisson-based), false=block Jacobi
    bool use_line_jacobi = true;     // Line-Jacobi preconditioner for GMRES (replaces block-Jacobi)
    int limiter_type = 0;            // 0=minmod, 1=van_leer, 2=MC
    int hllc_variant = 0;            // 0=standard, 1=Rieper LM-HLLC, 2=Minoshima LHLLC
    bool use_hse_outer_bc = false;   // HSE Dirichlet at outer radial boundary
    bool use_core_excision = false;  // r_face[0] > 0, skip origin kernel
    bool radial_only = false;        // enforce v_theta=0 and skip theta-direction work
    double M_core = 0.0;            // enclosed mass inside r_inner
    int n_angular_avg = 0;           // angular-average this many inner shells per step
    int n_pole_avg = 0;              // wedge-average this many cells near each pole
    double central_damp_r = 0.0;     // damp v_r for r < this radius (0 = off)

    double sponge_r_start = 0.0;
    double sponge_r_top = 0.0;
    double sponge_kappa = 100.0;
    double interior_volume = 0.0;  // total volume of cells with ρ₀ >= atm_thresh

    // Viallet 2016 eq 72 scaling parameters.
    // α₁=1e-5 (L, residual side) - small floor keeps low-Mach momentum signal
    // α₂=1    (R, correction side) - large floor prevents tiny δU from passing tol
    double music_alpha1 = 1e-5;
    double music_alpha2 = 1.0;
    bool use_music_scaling = true;  // set false to mimic pre-fix FAS behavior

    // Line-implicit-in-r preconditioner (fas2 fix 4/4). When true, replaces
    // point-block-Jacobi M⁻¹ with a block-tri-diagonal solve along each
    // θ column. Captures the radial stiff eigendirection exactly — essential
    // on log-spaced radial meshes where innermost cells drive spectral radius.
    bool use_line_precond_r = false;  // set true to enable

    static constexpr int NU1 = 4;     // pre-smooth iterations
    static constexpr int NU2 = 4;     // post-smooth iterations
    static constexpr double OMEGA = 0.8;  // damping factor

    // Public for testing access
    void launch_ghost(int l);
    void apply_floor(int l);
    void restrict_state_pub(int fine, int coarse) { restrict_state(fine, coarse); }

private:
    void build_level(int l, int nr, int nt, int ng,
                     const double* h_rf, const double* h_tf);
    void build_coarse_geometry(int fine, int coarse);

    // FAS operations
    void fas_vcycle(int l, double dt, double g0_over_dt);
    void smooth(int l, double dt, double g0_over_dt, int n_iters);
    void compute_residual(int l);
    void compute_F(int l, double g0_over_dt);
    void compute_gravity_1d(int l);

    void restrict_state(int fine, int coarse);
    void restrict_defect(int fine, int coarse, double g0_over_dt);
    void prolongate_correct(int coarse, int fine);

    void assemble_smoother(int l, double g0_over_dt);

    // JFNK outer solver
    void jfnk_matvec(const double* d_v, double* d_Jv, double dt, double g0_over_dt);
    void apply_preconditioner(const double* d_v, double* d_Mv, double dt, double g0_over_dt);
    int gmres_solve(double* d_x, const double* d_b, double dt, double g0_over_dt,
                    double tol, int max_iter);

    // Diagnostics
    double residual_norm(int l);
    void residual_norm_detail(int l, const char* label);
};
