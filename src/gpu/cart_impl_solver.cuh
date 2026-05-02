#pragma once

#include "../grid.h"
#include "../state.h"
#include "../eos.h"
#include "fas_common.cuh"

// Cartesian 2D implicit Backward-Euler low-Mach solver.
//
// Motivation:球極 log mesh + JFNK 在 256x256 條件數 ~10^7 導致 Newton
// 直接 bypass(cyc=0,state 凍結);fas2/projection 在同一 mesh 都失敗。
// MUSIC/SLH/MAESTROeX 文獻一致:所有生產級低馬赫恆星代碼都用 Cartesian。
//
// 這個求解器在 Cartesian 均勻網格上做 BE + JFNK,借用 fas2 的四個 fix:
//   - CGS2 Gram-Schmidt (Knoll-Keyes 2004)
//   - Unit-normalize v̂ in JFNK matvec (Trilinos NOX)
//   - Viallet 2016 eq 72 asymmetric L/R scaling (α₁=1e-5, α₂=1)
//   - Line-implicit-in-y preconditioner(重力方向)
//
// 物理:2D Cartesian,常重力 g_y,無自引力。HSE:dP/dy = -ρ·g。
// 狀態:(ρ, ρvx, ρvy, E) cell-centered + ng=2 ghost cells。
// BC:x 方向 periodic,y 方向 HSE Dirichlet (top/bottom)。
//
// Not a low-Mach-model scheme (不是 MAESTROeX 的 p = p0+π split);
// 這是 full compressible BE + Newton-Krylov,適用 Ms >= 1e-3 左右。

struct CartImplSolver {
    // ---- Geometry ----
    int nx = 0, ny = 0;              // physical cell counts
    int ng = 2;                      // ghost cells (for MUSCL recon)
    int stride = 0;                  // nx + 2*ng
    int total = 0;                   // (nx+2*ng)*(ny+2*ng)
    int phys = 0;                    // nx*ny
    double Lx = 0.0, Ly = 0.0;
    double dx = 0.0, dy = 0.0;

    // ---- State (total-sized, ghost included) ----
    double *d_rho = nullptr, *d_mx = nullptr, *d_my = nullptr, *d_rhoE = nullptr;
    double *d_Un = nullptr;   // saved Uⁿ for BE time derivative, size 4*phys
    double *d_Ubak = nullptr; // state backup for JFNK matvec, size 4*phys

    // ---- HSE reference (phys-sized) ----
    double *d_rho0 = nullptr, *d_P0 = nullptr;
    double *d_hse_defect = nullptr;  // R_wb(U_hse) precomputed, size 4*phys

    // ---- Residual + Newton scratch ----
    double *d_res = nullptr;     // R(U) or F(U), size 4*phys
    double *d_Fk = nullptr;      // F(Uk) cached for matvec, size 4*phys

    // ---- Viallet 2016 eq 72 scaling (fix 3) ----
    double *d_scale_L = nullptr, *d_scale_R = nullptr, *d_scale_invL = nullptr;  // 4*phys each

    // ---- GMRES(K) scratch ----
    static constexpr int GMRES_K = 30;
    double *d_V[GMRES_K + 1] = {};   // Krylov basis, each 4*phys
    double *d_Z[GMRES_K] = {};       // preconditioned directions, each 4*phys
    double *d_gmres_w = nullptr;     // matvec scratch, 4*phys

    // ---- Parameters ----
    EOS eos;
    double gamma = 5.0/3.0;
    double g_y = 1.0;                // gravity magnitude, pulls -y
    double cfl_num = 0.5;
    double dt_current = 0.0;
    int step_count = 0;
    bool hse_set = false;
    double atm_rho_thresh = 0.0;

    // Fix 3: Viallet eq 72 asymmetric scaling
    double music_alpha1 = 1e-5;
    double music_alpha2 = 1.0;
    bool use_music_scaling = true;

    // Fix 4: line-implicit-in-y preconditioner (default on)
    bool use_line_precond_y = true;

    // HLLC variant: 0=standard, 1=Rieper LM-HLLC, 2=Minoshima LHLLC
    int hllc_variant = 0;
    int limiter_type = 1;            // 0=minmod, 1=van_leer, 2=MC

    // Newton tolerance
    double newton_tol = 1e-8;
    int newton_max_iter = 15;
    double gmres_tol = 1e-3;
    int gmres_max_iter = GMRES_K;

    // ---- Lifecycle ----
    void init(int nx, int ny, double Lx, double Ly, const EOS& eos_in,
              double gamma_in, double g_y_in, double cfl);
    void destroy();

    // Initial conditions
    void init_hse_polytrope(double rho_base, double perturb_amp = 0.0);

    // Capture current state as HSE reference (call after init, before perturb)
    void snapshot_hse();

    // One BE step, returns dt taken
    double step(double t, double t_end);

    // CFL-based dt estimate
    double compute_cfl_dt();

    // Host <-> device (uses a simple Grid-less interface since we don't use Grid)
    // State order: rho, mx, my, E, each phys-sized, row-major (i*ny + j)
    void download_host(std::vector<double>& rho, std::vector<double>& mx,
                       std::vector<double>& my, std::vector<double>& E);

    // Cartesian VTK writer (STRUCTURED_GRID, cell-centered data)
    void write_vtk(const char* filename);

    // Simple diagnostics (total mass, total energy, max |v|, max Mach)
    struct Diag { double mass, energy, max_v, max_mach; };
    Diag compute_diagnostics();

private:
    void alloc_all();
    void apply_bc();
    void apply_floor();
    void compute_F(double inv_dt);   // writes d_res = F(U) = (U-Un)/dt - R(U)
    double residual_norm();

    // JFNK
    void jfnk_matvec(const double* d_v, double* d_Jv, double inv_dt);
    void apply_precond(const double* d_v, double* d_Mv, double inv_dt);
    int gmres_solve(double* d_x, const double* d_b, double inv_dt, double tol, int max_iter);
    int newton_solve(double dt);
};
