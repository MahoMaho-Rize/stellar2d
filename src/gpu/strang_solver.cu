// ============================================================
// strang_solver.cu — Cartesian 2D Strang-splitting Euler solver
//
// Part 1: Grid init + HSE background + bubble init + VTK I/O
// Part 2: MC limiter + MUSCL-Hancock predictor + ghost cells
// ============================================================

#include "strang_solver.cuh"
#include "strang_device.cuh"    // MC limiter, HSE, cons2prim, Euler flux
#include "fas_common.cuh"       // CUDA_CHECK macro

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <vector>

// ============================================================
//  Index helper — 2D with ghost cells
// ============================================================
static __device__ __forceinline__
int sidx(int i, int j, int stride) {
    return j * stride + i;
}

// ============================================================
//  GHOST CELL KERNELS
// ============================================================

// ---- X-direction: periodic ----
// Each thread handles one ghost cell (left or right) for one row j
__global__
void k_ghost_x(double* d_rho, double* d_mx, double* d_my, double* d_E,
               int nx, int ny, int ng, int str)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Total ghost cells: 2 * ng * (ny + 2*ng)  [left + right, all rows including ghost rows]
    int ny_g = ny + 2 * ng;
    int n_ghost = 2 * ng * ny_g;
    if (tid >= n_ghost) return;

    int side = tid / (ng * ny_g);      // 0 = left, 1 = right
    int rem  = tid % (ng * ny_g);
    int g    = rem / ny_g;             // ghost layer index (0..ng-1)
    int jg   = rem % ny_g;            // row index (including ghost)

    int ig_ghost, ig_src;
    if (side == 0) {
        // Left ghost: ig = g → source from ig = ng + nx - ng + g = nx + g
        ig_ghost = g;
        ig_src   = ng + nx - ng + g;   // wrap from physical right
    } else {
        // Right ghost: ig = ng + nx + g → source from ig = ng + g
        ig_ghost = ng + nx + g;
        ig_src   = ng + g;             // wrap from physical left
    }

    int k_dst = jg * str + ig_ghost;
    int k_src = jg * str + ig_src;

    d_rho[k_dst] = d_rho[k_src];
    d_mx [k_dst] = d_mx [k_src];
    d_my [k_dst] = d_my [k_src];
    d_E  [k_dst] = d_E  [k_src];
}

// ---- Y-direction: bottom reflective, top outflow ----
// Mirror perturbation values for WB consistency
__global__
void k_ghost_y(double* d_rho, double* d_mx, double* d_my, double* d_E,
               int nx, int ny, int ng, int str)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nx_g = nx + 2 * ng;
    int n_ghost = 2 * ng * nx_g;   // bottom + top, full row width
    if (tid >= n_ghost) return;

    int side = tid / (ng * nx_g);   // 0 = bottom, 1 = top
    int rem  = tid % (ng * nx_g);
    int g    = rem / nx_g;          // ghost layer (0..ng-1)
    int ig   = rem % nx_g;          // column (including ghost)

    int jg_ghost, jg_src;

    if (side == 0) {
        // Bottom reflective: ghost jg = ng-1-g → mirror from jg = ng+g
        jg_ghost = ng - 1 - g;
        jg_src   = ng + g;
    } else {
        // Top outflow (zero-gradient): ghost jg = ng+ny+g → copy from jg = ng+ny-1
        jg_ghost = ng + ny + g;
        jg_src   = ng + ny - 1;
    }

    int k_dst = jg_ghost * str + ig;
    int k_src = jg_src   * str + ig;

    d_rho[k_dst] =  d_rho[k_src];
    d_mx [k_dst] =  d_mx [k_src];
    d_E  [k_dst] =  d_E  [k_src];

    if (side == 0) {
        // Reflective: negate y-momentum (normal velocity)
        d_my[k_dst] = -d_my[k_src];
    } else {
        d_my[k_dst] =  d_my[k_src];
    }
}

// ============================================================
//  MUSCL-HANCOCK PREDICTOR — X-SWEEP
//
//  For each physical cell (ig, jg), computes half-step predicted
//  primitive states at its left and right faces.
//  Slopes computed on perturbation quantities (WB).
//  Since ρ̄(y) is constant along x-rows, WB is trivial here.
//
//  Output layout: d_wL[k*4+c], d_wR[k*4+c], c=0..3 → (ρ,u,v,P)
//  Convention:
//    d_wL[k] = left face of cell k  = RIGHT state at face i-1/2
//    d_wR[k] = right face of cell k = LEFT state at face i+1/2
// ============================================================
__global__
void k_muscl_hancock_x(
    const double* __restrict__ d_rho,
    const double* __restrict__ d_mx,
    const double* __restrict__ d_my,
    const double* __restrict__ d_E,
    const double* __restrict__ d_rho_bar,
    const double* __restrict__ d_p_bar,
    double* __restrict__ d_wL,
    double* __restrict__ d_wR,
    double dt, double dx, double gamma,
    int nx, int ny, int ng, int str)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny) return;

    int i_phys = tid % nx;
    int j_phys = tid / nx;
    int ig = i_phys + ng;
    int jg = j_phys + ng;

    double rho_bg = d_rho_bar[j_phys];
    double p_bg   = d_p_bar[j_phys];
    double gm1    = gamma - 1.0;
    double E_bg   = p_bg / gm1;

    // Read 3 neighboring cells: (ig-1), (ig), (ig+1) at same jg
    int km = jg * str + (ig - 1);
    int k0 = jg * str + ig;
    int kp = jg * str + (ig + 1);

    // Recover total conserved → total primitive for each neighbor
    // Cell i-1
    double rho_m = d_rho[km] + rho_bg;
    double E_m   = d_E[km]   + E_bg;
    double um, vm, Pm;
    d_cons2prim(rho_m, d_mx[km], d_my[km], E_m, gm1, um, vm, Pm);

    // Cell i
    double rho_0 = d_rho[k0] + rho_bg;
    double E_0   = d_E[k0]   + E_bg;
    double u0, v0, P0;
    d_cons2prim(rho_0, d_mx[k0], d_my[k0], E_0, gm1, u0, v0, P0);

    // Cell i+1
    double rho_p = d_rho[kp] + rho_bg;
    double E_p   = d_E[kp]   + E_bg;
    double up, vp, Pp;
    d_cons2prim(rho_p, d_mx[kp], d_my[kp], E_p, gm1, up, vp, Pp);

    // Perturbation differences for MC limiter (ρ' and P' = P - p̄)
    double drho_L = d_rho[k0] - d_rho[km];   // ρ'_i - ρ'_{i-1}
    double drho_R = d_rho[kp] - d_rho[k0];
    double du_L   = u0 - um;
    double du_R   = up - u0;
    double dv_L   = v0 - vm;
    double dv_R   = vp - v0;
    double dP_L   = (P0 - p_bg) - (Pm - p_bg);  // = P0 - Pm
    double dP_R   = (Pp - p_bg) - (P0 - p_bg);  // = Pp - P0

    // MC-limited slopes
    double s_rho = d_mc_limit(drho_L, drho_R);
    double s_u   = d_mc_limit(du_L,   du_R);
    double s_v   = d_mc_limit(dv_L,   dv_R);
    double s_P   = d_mc_limit(dP_L,   dP_R);

    // Extrapolate to left and right faces (total primitive)
    // Left face of cell: face at i - 1/2
    double rL = rho_0 - 0.5 * s_rho;
    double uL = u0    - 0.5 * s_u;
    double vL = v0    - 0.5 * s_v;
    double PL = P0    - 0.5 * s_P;

    // Right face of cell: face at i + 1/2
    double rR = rho_0 + 0.5 * s_rho;
    double uR = u0    + 0.5 * s_u;
    double vR = v0    + 0.5 * s_v;
    double PR = P0    + 0.5 * s_P;

    // Positivity enforcement
    rL = fmax(rL, 1e-20);  rR = fmax(rR, 1e-20);
    PL = fmax(PL, 1e-20);  PR = fmax(PR, 1e-20);

    // ---- Hancock half-step: evolve face values by dt/2 ----
    // dU/dt = -(F_R - F_L)/dx
    // U^{n+1/2} = U + 0.5*(dt/dx)*(F_L - F_R)
    double fL0, fL1, fL2, fL3;  // Euler flux at left face
    double fR0, fR1, fR2, fR3;  // Euler flux at right face

    d_euler_flux_x(rL, uL, vL, PL, gm1, fL0, fL1, fL2, fL3);
    d_euler_flux_x(rR, uR, vR, PR, gm1, fR0, fR1, fR2, fR3);

    double coeff = 0.5 * dt / dx;

    // Left face conserved
    double EL = PL / gm1 + 0.5 * rL * (uL*uL + vL*vL);
    double cL_rho = rL      + coeff * (fL0 - fR0);
    double cL_mx  = rL * uL + coeff * (fL1 - fR1);
    double cL_my  = rL * vL + coeff * (fL2 - fR2);
    double cL_E   = EL      + coeff * (fL3 - fR3);

    // Right face conserved
    double ER = PR / gm1 + 0.5 * rR * (uR*uR + vR*vR);
    double cR_rho = rR      + coeff * (fL0 - fR0);
    double cR_mx  = rR * uR + coeff * (fL1 - fR1);
    double cR_my  = rR * vR + coeff * (fL2 - fR2);
    double cR_E   = ER      + coeff * (fL3 - fR3);

    // Convert back to primitive
    cL_rho = fmax(cL_rho, 1e-20);
    cR_rho = fmax(cR_rho, 1e-20);
    double uL2, vL2, PL2, uR2, vR2, PR2;
    d_cons2prim(cL_rho, cL_mx, cL_my, cL_E, gm1, uL2, vL2, PL2);
    d_cons2prim(cR_rho, cR_mx, cR_my, cR_E, gm1, uR2, vR2, PR2);

    // Store half-step predicted face states (total primitive)
    d_wL[k0*4+0] = cL_rho;  d_wL[k0*4+1] = uL2;
    d_wL[k0*4+2] = vL2;     d_wL[k0*4+3] = PL2;

    d_wR[k0*4+0] = cR_rho;  d_wR[k0*4+1] = uR2;
    d_wR[k0*4+2] = vR2;     d_wR[k0*4+3] = PR2;
}

// ============================================================
//  MUSCL-HANCOCK PREDICTOR — Y-SWEEP (Well-Balanced)
//
//  Critical WB treatment: slopes computed on perturbation (ρ', P').
//  Face reconstruction uses ρ̄(y_face) and p̄(y_face) so that
//  in HSE both sides of every face see identical states.
// ============================================================
__global__
void k_muscl_hancock_y(
    const double* __restrict__ d_rho,
    const double* __restrict__ d_mx,
    const double* __restrict__ d_my,
    const double* __restrict__ d_E,
    const double* __restrict__ d_rho_bar,
    const double* __restrict__ d_p_bar,
    double* __restrict__ d_wL,
    double* __restrict__ d_wR,
    double dt, double dy, double gamma,
    double y_lo, double rho0_bottom, double g_grav, double K_poly,
    int nx, int ny, int ng, int str)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny) return;

    int i_phys = tid % nx;
    int j_phys = tid / nx;
    int ig = i_phys + ng;
    int jg = j_phys + ng;

    double gm1     = gamma - 1.0;
    double inv_gm1 = 1.0 / gm1;

    // HSE precomputed coefficients
    double rho0_gm1 = pow(rho0_bottom, gm1);
    double hse_coeff = gm1 * g_grav / (gamma * K_poly);

    // Background at cell centers j-1, j, j+1
    // Clamp j indices to valid range for background lookup
    int jm = max(j_phys - 1, 0);
    int jp = min(j_phys + 1, ny - 1);

    double rho_bg_m = d_rho_bar[jm];
    double p_bg_m   = d_p_bar[jm];
    double E_bg_m   = p_bg_m / gm1;

    double rho_bg_0 = d_rho_bar[j_phys];
    double p_bg_0   = d_p_bar[j_phys];
    double E_bg_0   = p_bg_0 / gm1;

    double rho_bg_p = d_rho_bar[jp];
    double p_bg_p   = d_p_bar[jp];
    double E_bg_p   = p_bg_p / gm1;

    // Read 3 neighbors in y-direction
    int km = (jg - 1) * str + ig;
    int k0 = jg       * str + ig;
    int kp = (jg + 1) * str + ig;

    // Total conserved → total primitive
    double rho_m = d_rho[km] + rho_bg_m;
    double E_m   = d_E[km]   + E_bg_m;
    double um_t, vm_t, Pm_t;
    d_cons2prim(rho_m, d_mx[km], d_my[km], E_m, gm1, um_t, vm_t, Pm_t);

    double rho_0 = d_rho[k0] + rho_bg_0;
    double E_0   = d_E[k0]   + E_bg_0;
    double u0, v0, P0;
    d_cons2prim(rho_0, d_mx[k0], d_my[k0], E_0, gm1, u0, v0, P0);

    double rho_p = d_rho[kp] + rho_bg_p;
    double E_p   = d_E[kp]   + E_bg_p;
    double up_t, vp_t, Pp_t;
    d_cons2prim(rho_p, d_mx[kp], d_my[kp], E_p, gm1, up_t, vp_t, Pp_t);

    // PERTURBATION differences for MC slopes (WB!)
    double drho_L = d_rho[k0] - d_rho[km];     // ρ'_j - ρ'_{j-1}
    double drho_R = d_rho[kp] - d_rho[k0];
    double dv_L   = v0 - vm_t;
    double dv_R   = vp_t - v0;
    double du_L   = u0 - um_t;
    double du_R   = up_t - u0;
    // Pressure perturbation: P' = P - p̄(y_cell)
    double Pp_m   = Pm_t - p_bg_m;
    double Pp_0   = P0   - p_bg_0;
    double Pp_p   = Pp_t - p_bg_p;
    double dP_L   = Pp_0 - Pp_m;
    double dP_R   = Pp_p - Pp_0;

    // MC-limited slopes (on perturbation)
    double s_rho = d_mc_limit(drho_L, drho_R);
    double s_u   = d_mc_limit(du_L,   du_R);
    double s_v   = d_mc_limit(dv_L,   dv_R);
    double s_P   = d_mc_limit(dP_L,   dP_R);

    // --- Face reconstruction with WB ---
    // Bottom face (j-1/2): this is the LEFT face of cell j
    double y_bot = y_lo + j_phys * dy;          // face y-coordinate
    double rho_bar_bot = d_hse_rho(y_bot, rho0_gm1, hse_coeff, inv_gm1);
    double p_bar_bot   = d_hse_p(rho_bar_bot, K_poly, gamma);

    // Perturbation extrapolated to bottom face
    double rhoP_bot = d_rho[k0] - 0.5 * s_rho;   // ρ'
    double PP_bot   = Pp_0      - 0.5 * s_P;      // P'
    double u_bot    = u0        - 0.5 * s_u;
    double v_bot    = v0        - 0.5 * s_v;

    // Total at bottom face
    double rL = rho_bar_bot + rhoP_bot;
    double PL = p_bar_bot   + PP_bot;
    double uL = u_bot;
    double vL = v_bot;

    // Top face (j+1/2): this is the RIGHT face of cell j
    double y_top = y_lo + (j_phys + 1) * dy;
    double rho_bar_top = d_hse_rho(y_top, rho0_gm1, hse_coeff, inv_gm1);
    double p_bar_top   = d_hse_p(rho_bar_top, K_poly, gamma);

    double rhoP_top = d_rho[k0] + 0.5 * s_rho;
    double PP_top   = Pp_0      + 0.5 * s_P;
    double u_top    = u0        + 0.5 * s_u;
    double v_top    = v0        + 0.5 * s_v;

    double rR = rho_bar_top + rhoP_top;
    double PR = p_bar_top   + PP_top;
    double uR = u_top;
    double vR = v_top;

    // Positivity
    rL = fmax(rL, 1e-20);  rR = fmax(rR, 1e-20);
    PL = fmax(PL, 1e-20);  PR = fmax(PR, 1e-20);

    // ---- Hancock half-step (y-direction flux + gravity source) ----
    // The full gravity −ρg must be included to balance the background
    // pressure gradient dp̄/dy = −ρ̄g in HSE.
    double gL0, gL1, gL2, gL3;
    double gR0, gR1, gR2, gR3;
    d_euler_flux_y(rL, uL, vL, PL, gm1, gL0, gL1, gL2, gL3);
    d_euler_flux_y(rR, uR, vR, PR, gm1, gR0, gR1, gR2, gR3);

    double coeff = 0.5 * dt / dy;

    // Bottom face conserved (NO gravity in Hancock — only in update kernel)
    double EL = PL / gm1 + 0.5 * rL * (uL*uL + vL*vL);
    double cL_rho = rL      + coeff * (gL0 - gR0);
    double cL_mx  = rL * uL + coeff * (gL1 - gR1);
    double cL_my  = rL * vL + coeff * (gL2 - gR2);
    double cL_E   = EL      + coeff * (gL3 - gR3);

    // Top face conserved
    double ER = PR / gm1 + 0.5 * rR * (uR*uR + vR*vR);
    double cR_rho = rR      + coeff * (gL0 - gR0);
    double cR_mx  = rR * uR + coeff * (gL1 - gR1);
    double cR_my  = rR * vR + coeff * (gL2 - gR2);
    double cR_E   = ER      + coeff * (gL3 - gR3);

    // Convert back to primitive
    cL_rho = fmax(cL_rho, 1e-20);
    cR_rho = fmax(cR_rho, 1e-20);
    double uL2, vL2, PL2, uR2, vR2, PR2;
    d_cons2prim(cL_rho, cL_mx, cL_my, cL_E, gm1, uL2, vL2, PL2);
    d_cons2prim(cR_rho, cR_mx, cR_my, cR_E, gm1, uR2, vR2, PR2);

    // Store: wL = bottom face (LEFT), wR = top face (RIGHT)
    d_wL[k0*4+0] = cL_rho;  d_wL[k0*4+1] = uL2;
    d_wL[k0*4+2] = vL2;     d_wL[k0*4+3] = PL2;

    d_wR[k0*4+0] = cR_rho;  d_wR[k0*4+1] = uR2;
    d_wR[k0*4+2] = vR2;     d_wR[k0*4+3] = PR2;
}

// ============================================================
//  HLLC + UPDATE KERNEL — X-SWEEP
//
//  For each physical cell: compute HLLC at both faces, update.
//  No gravity source in x-sweep.
// ============================================================
__global__
void k_hllc_update_x(
    double* __restrict__ d_rho,
    double* __restrict__ d_mx,
    double* __restrict__ d_my,
    double* __restrict__ d_E,
    const double* __restrict__ d_p_bar,
    const double* __restrict__ d_wL,
    const double* __restrict__ d_wR,
    double dt, double dx, double gamma,
    int nx, int ny, int ng, int str)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny) return;

    int i_phys = tid % nx;
    int j_phys = tid / nx;
    int ig = i_phys + ng;
    int jg = j_phys + ng;
    int k0 = jg * str + ig;

    // Right face i+1/2: left=wR[k0], right=wL[k0+1]
    int kR = jg * str + (ig + 1);
    double FR0, FR1, FR2, FR3;
    d_lmhllc(d_wR[k0*4+0], d_wR[k0*4+1], d_wR[k0*4+2], d_wR[k0*4+3],
             d_wL[kR*4+0], d_wL[kR*4+1], d_wL[kR*4+2], d_wL[kR*4+3],
             gamma, FR0, FR1, FR2, FR3);

    // Left face i-1/2: left=wR[k0-1], right=wL[k0]
    int kL = jg * str + (ig - 1);
    double FL0, FL1, FL2, FL3;
    d_lmhllc(d_wR[kL*4+0], d_wR[kL*4+1], d_wR[kL*4+2], d_wR[kL*4+3],
             d_wL[k0*4+0], d_wL[k0*4+1], d_wL[k0*4+2], d_wL[k0*4+3],
             gamma, FL0, FL1, FL2, FL3);

    double dtdx = dt / dx;
    d_rho[k0] -= dtdx * (FR0 - FL0);
    d_mx [k0] -= dtdx * (FR1 - FL1);
    d_my [k0] -= dtdx * (FR2 - FL2);
    d_E  [k0] -= dtdx * (FR3 - FL3);
}

// ============================================================
//  HLLC + UPDATE KERNEL — Y-SWEEP (with full gravity source)
//
//  HLLC in y: normal=v, tangential=u.
//  Gravity source: S_my = -ρ_total * g,  S_E = -my * g
// ============================================================
__global__
void k_hllc_update_y(
    double* __restrict__ d_rho,
    double* __restrict__ d_mx,
    double* __restrict__ d_my,
    double* __restrict__ d_E,
    const double* __restrict__ d_rho_bar,
    const double* __restrict__ d_p_bar,
    const double* __restrict__ d_wL,
    const double* __restrict__ d_wR,
    double dt, double dy, double gamma, double g_grav,
    int nx, int ny, int ng, int str)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny) return;

    int i_phys = tid % nx;
    int j_phys = tid / nx;
    int ig = i_phys + ng;
    int jg = j_phys + ng;
    int k0 = jg * str + ig;

    double gm1    = gamma - 1.0;
    double rho_bg = d_rho_bar[j_phys];

    // Top face j+1/2: left=wR[k0], right=wL[k0+str]
    // HLLC y-direction: pass (ρ, v, u, P) as (ρ, un, ut, P)
    int kT = (jg + 1) * str + ig;
    double GT0, GT_mn, GT_mt, GT3;
    d_lmhllc(d_wR[k0*4+0], d_wR[k0*4+2], d_wR[k0*4+1], d_wR[k0*4+3],
             d_wL[kT*4+0], d_wL[kT*4+2], d_wL[kT*4+1], d_wL[kT*4+3],
             gamma, GT0, GT_mn, GT_mt, GT3);

    // Bottom face j-1/2: left=wR[k0-str], right=wL[k0]
    int kB = (jg - 1) * str + ig;
    double GB0, GB_mn, GB_mt, GB3;
    d_lmhllc(d_wR[kB*4+0], d_wR[kB*4+2], d_wR[kB*4+1], d_wR[kB*4+3],
             d_wL[k0*4+0], d_wL[k0*4+2], d_wL[k0*4+1], d_wL[k0*4+3],
             gamma, GB0, GB_mn, GB_mt, GB3);

    // Gravity source — re-enabled with correct sign
    double rho_total = fmax(d_rho[k0] + rho_bg, 1e-20);
    double S_my = -rho_total * g_grav;
    double S_E  = -d_my[k0] * g_grav;

    double dtdy = dt / dy;
    d_rho[k0] -= dtdy * (GT0    - GB0);
    d_mx [k0] -= dtdy * (GT_mt  - GB_mt);
    // IMPORTANT: separate -= and += to avoid operator precedence bug
    d_my [k0]  = d_my[k0] - dtdy * (GT_mn - GB_mn) + dt * S_my;
    d_E  [k0]  = d_E[k0]  - dtdy * (GT3   - GB3)   + dt * S_E;
}

// ============================================================
//  CFL KERNEL — compute max signal speed / cell size
// ============================================================
__global__
void k_strang_cfl(const double* __restrict__ d_rho,
                  const double* __restrict__ d_mx,
                  const double* __restrict__ d_my,
                  const double* __restrict__ d_E,
                  const double* __restrict__ d_rho_bar,
                  const double* __restrict__ d_p_bar,
                  double* __restrict__ d_buf,
                  double dx, double dy, double gamma,
                  int nx, int ny, int ng, int str)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny) return;

    int i_phys = tid % nx;
    int j_phys = tid / nx;
    int k = (j_phys + ng) * str + (i_phys + ng);

    double gm1 = gamma - 1.0;
    double rho = fmax(d_rho[k] + d_rho_bar[j_phys], 1e-30);
    double E_t = d_E[k] + d_p_bar[j_phys] / gm1;
    double u = d_mx[k] / rho;
    double v = d_my[k] / rho;
    double P = fmax(gm1 * (E_t - 0.5 * rho * (u*u + v*v)), 1e-30);
    double cs = sqrt(gamma * P / rho);

    d_buf[tid] = (fabs(u) + cs) / dx + (fabs(v) + cs) / dy;
}

// ============================================================
//  FACE-STATE GHOST FILL — X (periodic)
//  After MUSCL-Hancock, copy face states to ghost cells so
//  HLLC at boundary physical cells has valid neighbor data.
// ============================================================
__global__
void k_ghost_face_x(double* d_wL, double* d_wR,
                    int nx, int ny, int ng, int str)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ny_g = ny + 2 * ng;
    if (tid >= ny_g) return;
    int jg = tid;

    // Left ghost: ig = ng-1  ←  periodic partner ig = ng+nx-1
    int kd = jg * str + (ng - 1);
    int ks = jg * str + (ng + nx - 1);
    for (int c = 0; c < 4; c++) {
        d_wL[kd*4+c] = d_wL[ks*4+c];
        d_wR[kd*4+c] = d_wR[ks*4+c];
    }

    // Right ghost: ig = ng+nx  ←  periodic partner ig = ng
    kd = jg * str + (ng + nx);
    ks = jg * str + ng;
    for (int c = 0; c < 4; c++) {
        d_wL[kd*4+c] = d_wL[ks*4+c];
        d_wR[kd*4+c] = d_wR[ks*4+c];
    }
}

// ============================================================
//  FACE-STATE GHOST FILL — Y (bottom reflective, top outflow)
// ============================================================
__global__
void k_ghost_face_y(double* d_wL, double* d_wR,
                    int nx, int ny, int ng, int str)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nx_g = nx + 2 * ng;
    if (tid >= nx_g) return;
    int ig = tid;

    // Bottom reflective: ghost jg=ng-1, mirror from first physical jg=ng
    // Ghost's TOP face (wR) at boundary = reflected BOTTOM face (wL) of cell j=0
    // Same (ρ,u,P), negate v (index 2)
    {
        int kd = (ng - 1) * str + ig;
        int ks = ng * str + ig;
        d_wR[kd*4+0] =  d_wL[ks*4+0];   // ρ from wL of cell 0
        d_wR[kd*4+1] =  d_wL[ks*4+1];   // u
        d_wR[kd*4+2] = -d_wL[ks*4+2];   // -v (reflect normal)
        d_wR[kd*4+3] =  d_wL[ks*4+3];   // P
        // Also fill wL for completeness (ghost bottom face = reflected top face)
        d_wL[kd*4+0] =  d_wR[ks*4+0];
        d_wL[kd*4+1] =  d_wR[ks*4+1];
        d_wL[kd*4+2] = -d_wR[ks*4+2];
        d_wL[kd*4+3] =  d_wR[ks*4+3];
    }

    // Top outflow: ghost jg=ng+ny, copy from last physical jg=ng+ny-1
    // Ghost's BOTTOM face (wL) = TOP face (wR) of last cell (zero gradient)
    {
        int kd = (ng + ny) * str + ig;
        int ks = (ng + ny - 1) * str + ig;
        for (int c = 0; c < 4; c++) {
            d_wL[kd*4+c] = d_wR[ks*4+c];   // ghost bottom = last cell top
            d_wR[kd*4+c] = d_wR[ks*4+c];   // ghost top = same (outflow)
        }
    }
}

// ============================================================
//  1. INIT: allocate GPU memory, build HSE background
// ============================================================

void StrangSolver::init(int nx_in, int ny_in,
                        double Lx_in, double Ly_in,
                        double gamma_in, double g_in, double cfl_in,
                        double K_in, double rho0_in)
{
    nx = nx_in;  ny = ny_in;  ng = 2;
    Lx = Lx_in;  Ly = Ly_in;
    x_lo = 0.0;  x_hi = Lx;
    y_lo = 0.0;  y_hi = Ly;
    dx = Lx / nx;
    dy = Ly / ny;
    gamma      = gamma_in;
    g_grav     = g_in;
    cfl_number = cfl_in;
    K_poly     = K_in;
    rho0_bottom = rho0_in;
    step_count = 0;

    int N = total_cells();

    // Allocate state arrays (perturbation storage)
    CUDA_CHECK(cudaMalloc(&d_rho, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mx,  N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_my,  N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_E,   N * sizeof(double)));

    // Zero everything
    CUDA_CHECK(cudaMemset(d_rho, 0, N * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mx,  0, N * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_my,  0, N * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_E,   0, N * sizeof(double)));

    // Background HSE arrays (1-D, ny elements)
    CUDA_CHECK(cudaMalloc(&d_rho_bar, ny * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p_bar,   ny * sizeof(double)));

    // Scratch for MUSCL-Hancock face states (4 primitives per cell)
    CUDA_CHECK(cudaMalloc(&d_wL, 4 * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_wR, 4 * N * sizeof(double)));

    // CFL scratch buffer (one double per physical cell)
    CUDA_CHECK(cudaMalloc(&d_cfl_buf, nx * ny * sizeof(double)));

    // ---- Build isentropic HSE background on host ----
    // p̄ = K ρ̄^γ,  dp̄/dy = −ρ̄ g
    // Analytical:  ρ̄(y) = [ ρ₀^{γ−1} − (γ−1)g y / (γ K) ]_+^{1/(γ−1)}
    h_rho_bar.resize(ny);
    h_p_bar.resize(ny);

    double gm1 = gamma - 1.0;
    double inv_gm1 = 1.0 / gm1;

    for (int j = 0; j < ny; ++j) {
        double y = y_lo + (j + 0.5) * dy;      // cell center
        double arg = std::pow(rho0_bottom, gm1) - gm1 * g_grav * y / (gamma * K_poly);
        if (arg < 1e-20) arg = 1e-20;           // atmosphere floor
        h_rho_bar[j] = std::pow(arg, inv_gm1);
        h_p_bar[j]   = K_poly * std::pow(h_rho_bar[j], gamma);
    }

    // Upload to device
    CUDA_CHECK(cudaMemcpy(d_rho_bar, h_rho_bar.data(), ny * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p_bar,   h_p_bar.data(),   ny * sizeof(double), cudaMemcpyHostToDevice));

    std::printf("Strang solver: %d × %d  Lx=%.3f Ly=%.3f  dx=%.4e dy=%.4e\n",
                nx, ny, Lx, Ly, dx, dy);
    std::printf("  HSE background: ρ_bot=%.4f  ρ_top=%.6f  p_bot=%.4f  p_top=%.6f\n",
                h_rho_bar[0], h_rho_bar[ny-1], h_p_bar[0], h_p_bar[ny-1]);
}

// ============================================================
//  2. BUBBLE INIT KERNEL
// ============================================================

__global__
void k_strang_init_bubble(double* d_rho, double* d_mx, double* d_my, double* d_E,
                          const double* d_rho_bar, const double* d_p_bar,
                          int nx, int ny, int ng, int stride,
                          double x_lo, double y_lo, double dx, double dy,
                          double gamma,
                          double x0, double y0, double R0,
                          double dS, double eps, int k_mode, double dr_smooth)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny) return;

    int i_phys = tid % nx;           // physical index [0, nx)
    int j_phys = tid / nx;           // physical index [0, ny)
    int ig = i_phys + ng;            // index with ghost offset
    int jg = j_phys + ng;
    int k  = jg * stride + ig;       // flat index in ghost-padded array

    double x = x_lo + (i_phys + 0.5) * dx;
    double y = y_lo + (j_phys + 0.5) * dy;

    // Background
    double rho_bg = d_rho_bar[j_phys];
    double p_bg   = d_p_bar[j_phys];
    double gm1    = gamma - 1.0;
    double E_bg   = p_bg / gm1;     // background internal energy (v=0)

    // Distance to bubble center
    double ddx = x - x0;
    double ddy = y - y0;
    double r   = sqrt(ddx * ddx + ddy * ddy);

    // Azimuthal angle for perturbation seeds
    double theta = atan2(ddy, ddx);

    // Perturbed effective radius
    double R_eff = R0 * (1.0 + eps * cos((double)k_mode * theta));

    // Smooth blending function: 1 inside bubble, 0 outside
    double alpha = 0.5 * (1.0 - tanh((r - R_eff) / dr_smooth));

    // Entropy perturbation: S = K (1 + dS * alpha)
    //   At same pressure p_bg:  ρ = (p / S)^{1/γ}
    double S_local = 1.0 * (1.0 + dS * alpha);   // K=1 absorbed
    double rho_total = pow(p_bg / S_local, 1.0 / gamma);

    // Pressure stays at background (pressure equilibrium)
    double p_total = p_bg;
    double E_total = p_total / gm1;   // no KE (v = 0)

    // Store perturbations
    d_rho[k] = rho_total - rho_bg;
    d_mx[k]  = 0.0;
    d_my[k]  = 0.0;
    d_E[k]   = E_total - E_bg;     // = (p_bg - p_bg)/(γ-1) = 0
}

void StrangSolver::init_bubble(double x0, double y0, double R0,
                               double dS, double eps, int k_mode,
                               double dr_smooth)
{
    int N_phys = nx * ny;
    int B = 256;
    int G = (N_phys + B - 1) / B;

    k_strang_init_bubble<<<G, B>>>(
        d_rho, d_mx, d_my, d_E,
        d_rho_bar, d_p_bar,
        nx, ny, ng, stride(),
        x_lo, y_lo, dx, dy, gamma,
        x0, y0, R0, dS, eps, k_mode, dr_smooth);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("  Bubble: center=(%.3f,%.3f) R0=%.3f dS=%.2f eps=%.3f k=%d dr=%.4f\n",
                x0, y0, R0, dS, eps, k_mode, dr_smooth);
}

// ============================================================
//  3. VTK OUTPUT (Cartesian RECTILINEAR_GRID)
// ============================================================

void StrangSolver::write_vtk(const char* filename)
{
    int N = total_cells();
    std::vector<double> h_rho(N), h_mx(N), h_my(N), h_E(N);

    CUDA_CHECK(cudaMemcpy(h_rho.data(), d_rho, N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mx.data(),  d_mx,  N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_my.data(),  d_my,  N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_E.data(),   d_E,   N * sizeof(double), cudaMemcpyDeviceToHost));

    int str = stride();
    double gm1 = gamma - 1.0;

    std::ofstream out(filename);
    out << "# vtk DataFile Version 3.0\n";
    out << "Strang Cartesian Bubble\n";
    out << "ASCII\n";
    out << "DATASET RECTILINEAR_GRID\n";
    out << "DIMENSIONS " << (nx + 1) << " " << (ny + 1) << " 1\n";

    // X coordinates (face positions)
    out << "X_COORDINATES " << (nx + 1) << " double\n";
    for (int i = 0; i <= nx; ++i)
        out << x_lo + i * dx << "\n";

    // Y coordinates (face positions)
    out << "Y_COORDINATES " << (ny + 1) << " double\n";
    for (int j = 0; j <= ny; ++j)
        out << y_lo + j * dy << "\n";

    out << "Z_COORDINATES 1 double\n0.0\n";

    int ncells = nx * ny;
    out << "CELL_DATA " << ncells << "\n";

    // --- Density (total = perturbation + background) ---
    out << "SCALARS density double 1\nLOOKUP_TABLE default\n";
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int k = (j + ng) * str + (i + ng);
            double rho = h_rho[k] + h_rho_bar[j];
            out << rho << "\n";
        }
    }

    // --- Pressure ---
    out << "SCALARS pressure double 1\nLOOKUP_TABLE default\n";
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int k = (j + ng) * str + (i + ng);
            double rho = std::fmax(h_rho[k] + h_rho_bar[j], 1e-30);
            double u   = h_mx[k] / rho;
            double v   = h_my[k] / rho;
            double KE  = 0.5 * rho * (u*u + v*v);
            double E_tot = h_E[k] + h_p_bar[j] / gm1;
            double P   = gm1 * (E_tot - KE);
            out << P << "\n";
        }
    }

    // --- Density perturbation ---
    out << "SCALARS density_perturbation double 1\nLOOKUP_TABLE default\n";
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            out << h_rho[(j + ng) * str + (i + ng)] << "\n";

    // --- Velocity vector ---
    out << "VECTORS velocity double\n";
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int k = (j + ng) * str + (i + ng);
            double rho = std::fmax(h_rho[k] + h_rho_bar[j], 1e-30);
            double u = h_mx[k] / rho;
            double v = h_my[k] / rho;
            out << u << " " << v << " 0\n";
        }
    }

    // --- Mach number ---
    out << "SCALARS mach double 1\nLOOKUP_TABLE default\n";
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int k = (j + ng) * str + (i + ng);
            double rho = std::fmax(h_rho[k] + h_rho_bar[j], 1e-30);
            double u   = h_mx[k] / rho;
            double v   = h_my[k] / rho;
            double spd = sqrt(u*u + v*v);
            double KE  = 0.5 * rho * (u*u + v*v);
            double E_tot = h_E[k] + h_p_bar[j] / gm1;
            double P   = std::fmax(gm1 * (E_tot - KE), 1e-30);
            double cs  = sqrt(gamma * P / rho);
            out << spd / cs << "\n";
        }
    }

    // --- Entropy (S = P / ρ^γ) ---
    out << "SCALARS entropy double 1\nLOOKUP_TABLE default\n";
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int k = (j + ng) * str + (i + ng);
            double rho = std::fmax(h_rho[k] + h_rho_bar[j], 1e-30);
            double u   = h_mx[k] / rho;
            double v   = h_my[k] / rho;
            double KE  = 0.5 * rho * (u*u + v*v);
            double E_tot = h_E[k] + h_p_bar[j] / gm1;
            double P   = std::fmax(gm1 * (E_tot - KE), 1e-30);
            out << P / pow(rho, gamma) << "\n";
        }
    }
}

// ============================================================
//  4. DIAGNOSTICS
// ============================================================

double StrangSolver::total_mass()
{
    int N = total_cells();
    std::vector<double> h(N);
    CUDA_CHECK(cudaMemcpy(h.data(), d_rho, N * sizeof(double), cudaMemcpyDeviceToHost));

    int str = stride();
    double sum = 0.0;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            sum += (h[(j + ng) * str + (i + ng)] + h_rho_bar[j]) * dx * dy;
    return sum;
}

double StrangSolver::total_energy()
{
    int N = total_cells();
    std::vector<double> h(N);
    CUDA_CHECK(cudaMemcpy(h.data(), d_E, N * sizeof(double), cudaMemcpyDeviceToHost));

    int str = stride();
    double gm1 = gamma - 1.0;
    double sum = 0.0;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            sum += (h[(j + ng) * str + (i + ng)] + h_p_bar[j] / gm1) * dx * dy;
    return sum;
}

double StrangSolver::max_velocity()
{
    int N = total_cells();
    std::vector<double> hx(N), hy(N), hr(N);
    CUDA_CHECK(cudaMemcpy(hx.data(), d_mx,  N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hy.data(), d_my,  N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hr.data(), d_rho, N * sizeof(double), cudaMemcpyDeviceToHost));

    int str = stride();
    double vmax = 0.0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int k = (j + ng) * str + (i + ng);
            double rho = std::fmax(hr[k] + h_rho_bar[j], 1e-30);
            double u = hx[k] / rho;
            double v = hy[k] / rho;
            vmax = std::fmax(vmax, std::sqrt(u*u + v*v));
        }
    }
    return vmax;
}

void StrangSolver::fill_ghost_x()
{
    int ny_g = ny + 2 * ng;
    int n = 2 * ng * ny_g;
    int B = 256;
    k_ghost_x<<<(n+B-1)/B, B>>>(d_rho, d_mx, d_my, d_E,
                                  nx, ny, ng, stride());
}

void StrangSolver::fill_ghost_y()
{
    int nx_g = nx + 2 * ng;
    int n = 2 * ng * nx_g;
    int B = 256;
    k_ghost_y<<<(n+B-1)/B, B>>>(d_rho, d_mx, d_my, d_E,
                                  nx, ny, ng, stride());
}
void StrangSolver::sweep_x(double dt)
{
    int N = nx * ny, B = 256, G = (N + B - 1) / B;
    int str = stride();

    // 1. MUSCL-Hancock predictor → d_wL, d_wR
    k_muscl_hancock_x<<<G, B>>>(d_rho, d_mx, d_my, d_E,
        d_rho_bar, d_p_bar, d_wL, d_wR,
        dt, dx, gamma, nx, ny, ng, str);

    // 2. Fill face-state ghosts (periodic)
    int ny_g = ny + 2 * ng;
    k_ghost_face_x<<<(ny_g+B-1)/B, B>>>(d_wL, d_wR, nx, ny, ng, str);

    // 3. HLLC flux + conservative update
    k_hllc_update_x<<<G, B>>>(d_rho, d_mx, d_my, d_E,
        d_p_bar, d_wL, d_wR,
        dt, dx, gamma, nx, ny, ng, str);
}

void StrangSolver::sweep_y(double dt)
{
    int N = nx * ny, B = 256, G = (N + B - 1) / B;
    int str = stride();

    // 1. MUSCL-Hancock predictor (WB) → d_wL, d_wR
    k_muscl_hancock_y<<<G, B>>>(d_rho, d_mx, d_my, d_E,
        d_rho_bar, d_p_bar, d_wL, d_wR,
        dt, dy, gamma,
        y_lo, rho0_bottom, g_grav, K_poly,
        nx, ny, ng, str);

    // 2. Fill face-state ghosts (reflective bottom, outflow top)
    int nx_g = nx + 2 * ng;
    k_ghost_face_y<<<(nx_g+B-1)/B, B>>>(d_wL, d_wR, nx, ny, ng, str);

    // 3. HLLC flux + update + gravity
    k_hllc_update_y<<<G, B>>>(d_rho, d_mx, d_my, d_E,
        d_rho_bar, d_p_bar, d_wL, d_wR,
        dt, dy, gamma, g_grav,
        nx, ny, ng, str);
}

double StrangSolver::compute_dt()
{
    int N = nx * ny, B = 256;
    k_strang_cfl<<<(N+B-1)/B, B>>>(d_rho, d_mx, d_my, d_E,
        d_rho_bar, d_p_bar, d_cfl_buf,
        dx, dy, gamma, nx, ny, ng, stride());

    // Download and find max (simple; could use thrust::reduce for speed)
    std::vector<double> h(N);
    CUDA_CHECK(cudaMemcpy(h.data(), d_cfl_buf, N * sizeof(double), cudaMemcpyDeviceToHost));

    double max_inv_dt = 0.0;
    for (int i = 0; i < N; i++)
        max_inv_dt = std::fmax(max_inv_dt, h[i]);

    if (max_inv_dt < 1e-30) max_inv_dt = 1e-30;
    return cfl_number / max_inv_dt;
}

double StrangSolver::step(double t, double t_end)
{
    double dt = compute_dt();
    if (t + dt > t_end) dt = t_end - t;

    // Strang splitting: X(dt/2) → Y(dt/2) → Y(dt/2) → X(dt/2)
    double half = 0.5 * dt;

    fill_ghost_x();   sweep_x(half);
    fill_ghost_y();   sweep_y(half);
    fill_ghost_y();   sweep_y(half);
    fill_ghost_x();   sweep_x(half);

    step_count++;
    return dt;
}

// ============================================================
//  6. CLEANUP
// ============================================================

void StrangSolver::destroy()
{
    cudaFree(d_rho);  cudaFree(d_mx);  cudaFree(d_my);  cudaFree(d_E);
    cudaFree(d_rho_bar); cudaFree(d_p_bar);
    cudaFree(d_wL);   cudaFree(d_wR);
    cudaFree(d_cfl_buf);
}
