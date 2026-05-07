// ============================================================
// athena_vl2_kernels.cu — CUDA kernels for the AthenaVL2Solver port
//
// Porting notes:
//  - Indexing convention: same layout as cart_ale2 (ic*ny+jc for cell arrays,
//    ic*nnode_y+jc for node arrays).  Here we keep a ghost-inclusive layout
//    where i ∈ [0, nx+2g), j ∈ [0, ny+2g), and the valid interior runs
//    i ∈ [ng, ng+nx), j ∈ [ng, ng+ny).
//    flat(i, j) = i*sy + j, with sy = ny + 2g.   (column-major in x.)
//    This matches cart_ale2's ic*ny+jc pattern so VTK output paths can share
//    "cell_scalar" formatting with interior-only iteration.
//
//  - Face-flux arrays:  Fx is sized (nx+1+2g)*(ny+2g), indexed [i*sy + j]
//                       where i ∈ [0, nx+1+2g) and j same as cell.
//                       The face at cell-interface between cells (i-1, j) and
//                       (i, j) has i-index = i (same as its right neighbour).
//                       Fy is sized (nx+2g)*(ny+1+2g), Fy flat(i, j) = i*(sy+1)+j.
//
//  - PLM limiter: Athena default (minmod_=false) uses the "simplified van Leer"
//    harmonic limiter on uniform meshes:
//       dqm = 2·dqL·dqR / (dqL + dqR), zero if dqL·dqR ≤ 0.
//    PLM slope = dqm (slope per cell), L/R states at face i+1/2:
//       qL(i+1/2) = q(i)   + 0.5·dqm(i)
//       qR(i+1/2) = q(i+1) − 0.5·dqm(i+1)
//    (Athena+++: plm_simple.cpp lines ~100 for the VL limiter path.)
// ============================================================

#include <cstdio>
#include <cmath>

#include <cuda_runtime.h>

#include "athena_vl2_solver.cuh"
#include "gpu/common/gpu_hllc.cuh"
#include "gpu/common/gpu_common.cuh"

namespace {

// Unified ghost width and strides (device-side copies embedded in kernels).
__device__ __forceinline__ int cflat(int i, int j, int sy) { return i * sy + j; }

// flat index for x-face array: (nx+1+2g) × (ny+2g) with sy = ny+2g.
__device__ __forceinline__ int fx_flat(int i, int j, int sy) { return i * sy + j; }

// flat index for y-face array: (nx+2g) × (ny+1+2g), using stride syp1 = ny+1+2g.
__device__ __forceinline__ int fy_flat(int i, int j, int syp1) { return i * syp1 + j; }

// Simplified VL harmonic limiter (matches Athena minmod_=false path).
__device__ __forceinline__ double vl_limit(double dqL, double dqR) {
    double dq2 = dqL * dqR;
    if (dq2 <= 0.0) return 0.0;
    return 2.0 * dq2 / (dqL + dqR);
}

__device__ __forceinline__ double mm_limit(double dqL, double dqR) {
    if (dqL * dqR <= 0.0) return 0.0;
    return (dqL >= 0.0) ? fmin(dqL, dqR) : fmax(dqL, dqR);
}

__device__ __forceinline__ double lim_dispatch(double dqL, double dqR, int lim) {
    return (lim == 1) ? mm_limit(dqL, dqR) : vl_limit(dqL, dqR);
}

} // namespace

// ============================================================
// cons → prim (ghost-inclusive, uses floor on P)
// ============================================================
__global__ void k_athvl2_cons_to_prim(
    const double* __restrict__ rho,
    const double* __restrict__ mx,
    const double* __restrict__ my,
    const double* __restrict__ E,
    const double* __restrict__ s,    // nullable (scalar)
    double* __restrict__ w_rho,
    double* __restrict__ w_u,
    double* __restrict__ w_v,
    double* __restrict__ w_P,
    double* __restrict__ w_X,        // nullable
    int sx, int sy, double gm1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= sx || j >= sy) return;
    int c = cflat(i, j, sy);
    double r  = fmax(rho[c], 1e-30);
    double u  = mx[c] / r;
    double v  = my[c] / r;
    double ke = 0.5 * r * (u*u + v*v);
    double P  = gm1 * (E[c] - ke);
    P = fmax(P, 1e-30);
    w_rho[c] = r;
    w_u[c]   = u;
    w_v[c]   = v;
    w_P[c]   = P;
    if (w_X != nullptr && s != nullptr) {
        w_X[c] = s[c] / r;
    }
}

// ============================================================
// Ghost fill: x = periodic, y = reflecting (symmetric ρ, P, IM1; antisymmetric IM2)
// We fill the ghost cells of all 4 fields + optional scalar.  Operates on
// conserved arrays so that cons_to_prim+reconstruction see consistent ghosts.
// ============================================================
__global__ void k_athvl2_fill_ghost_x_periodic(
    double* rho, double* mx, double* my, double* E, double* s,
    int nx, int ng, int sx, int sy)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= sy || g >= ng) return;
    // left ghost:   i = g is copy of i = ng + nx − ng + g  = nx + g
    // right ghost:  i = ng + nx + g is copy of i = ng + g
    int iL_dst = g;
    int iL_src = nx + g;
    int iR_dst = ng + nx + g;
    int iR_src = ng + g;
    int cL_dst = cflat(iL_dst, j, sy);
    int cL_src = cflat(iL_src, j, sy);
    int cR_dst = cflat(iR_dst, j, sy);
    int cR_src = cflat(iR_src, j, sy);
    rho[cL_dst] = rho[cL_src];
    mx [cL_dst] = mx [cL_src];
    my [cL_dst] = my [cL_src];
    E  [cL_dst] = E  [cL_src];
    rho[cR_dst] = rho[cR_src];
    mx [cR_dst] = mx [cR_src];
    my [cR_dst] = my [cR_src];
    E  [cR_dst] = E  [cR_src];
    if (s != nullptr) { s[cL_dst] = s[cL_src]; s[cR_dst] = s[cR_src]; }
}

__global__ void k_athvl2_fill_ghost_y_reflect(
    double* rho, double* mx, double* my, double* E, double* s,
    int ny, int ng, int sx, int sy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= sx || g >= ng) return;
    // bottom ghost: j = ng − 1 − g ←mirror→ j = ng + g
    int jB_dst = ng - 1 - g;
    int jB_src = ng + g;
    int jT_dst = ng + ny + g;
    int jT_src = ng + ny - 1 - g;
    int cBd = cflat(i, jB_dst, sy);
    int cBs = cflat(i, jB_src, sy);
    int cTd = cflat(i, jT_dst, sy);
    int cTs = cflat(i, jT_src, sy);
    rho[cBd] = rho[cBs];
    mx [cBd] = mx [cBs];
    my [cBd] = -my[cBs];   // antisymmetric normal momentum
    E  [cBd] = E  [cBs];
    rho[cTd] = rho[cTs];
    mx [cTd] = mx [cTs];
    my [cTd] = -my[cTs];
    E  [cTd] = E  [cTs];
    if (s != nullptr) { s[cBd] = s[cBs]; s[cTd] = s[cTs]; }
}

// ============================================================
// x-direction flux via PLM (primitive) + HLLC
// order = 1 → donor-cell (pure upwind Toro-style);
// order = 2 → PLM with VL/minmod limiter.
// Writes Fx_{rho,mx,my,E,s} at face i+1/2 for i ∈ [ng-1, ng+nx].
// ============================================================
__global__ void k_athvl2_flux_x(
    const double* __restrict__ w_rho,
    const double* __restrict__ w_u,
    const double* __restrict__ w_v,
    const double* __restrict__ w_P,
    const double* __restrict__ w_X,       // nullable
    double* __restrict__ Fx_rho,
    double* __restrict__ Fx_mx,
    double* __restrict__ Fx_my,
    double* __restrict__ Fx_E,
    double* __restrict__ Fx_s,            // nullable
    int nx, int ng, int sx, int sy,
    int order, int limiter, double gamma)
{
    int i_face = blockIdx.x * blockDim.x + threadIdx.x;
    int j      = blockIdx.y * blockDim.y + threadIdx.y;
    // Need faces at i_face ∈ [ng-1, ng+nx] (inclusive, so nx+2 faces total for this
    // correction-phase stencil). PLM needs cells i_face-1 and i_face and their
    // neighbours i_face-2, i_face+1.
    if (i_face < ng - 1 || i_face > ng + nx) return;
    if (j < 1 || j >= sy - 1) return;   // no flux computation in the outer-edge ghost row

    int iL = i_face - 1;   // left  cell  (centre of donor for +u)
    int iR = i_face;       // right cell
    int cL  = cflat(iL,   j, sy);
    int cR  = cflat(iR,   j, sy);

    double rL = w_rho[cL], uL = w_u[cL], vL = w_v[cL], PL = w_P[cL];
    double rR = w_rho[cR], uR = w_u[cR], vR = w_v[cR], PR = w_P[cR];

    if (order >= 2) {
        // Need neighbours (iL-1) and (iR+1)
        int iLL = iL - 1;
        int iRR = iR + 1;
        int cLL = cflat(iLL, j, sy);
        int cRR = cflat(iRR, j, sy);
        double drL = lim_dispatch(w_rho[cL] - w_rho[cLL], w_rho[cR] - w_rho[cL], limiter);
        double duL = lim_dispatch(w_u  [cL] - w_u  [cLL], w_u  [cR] - w_u  [cL], limiter);
        double dvL = lim_dispatch(w_v  [cL] - w_v  [cLL], w_v  [cR] - w_v  [cL], limiter);
        double dPL = lim_dispatch(w_P  [cL] - w_P  [cLL], w_P  [cR] - w_P  [cL], limiter);

        double drR = lim_dispatch(w_rho[cR] - w_rho[cL], w_rho[cRR] - w_rho[cR], limiter);
        double duR = lim_dispatch(w_u  [cR] - w_u  [cL], w_u  [cRR] - w_u  [cR], limiter);
        double dvR = lim_dispatch(w_v  [cR] - w_v  [cL], w_v  [cRR] - w_v  [cR], limiter);
        double dPR = lim_dispatch(w_P  [cR] - w_P  [cL], w_P  [cRR] - w_P  [cR], limiter);

        rL += 0.5 * drL;  uL += 0.5 * duL;  vL += 0.5 * dvL;  PL += 0.5 * dPL;
        rR -= 0.5 * drR;  uR -= 0.5 * duR;  vR -= 0.5 * dvR;  PR -= 0.5 * dPR;
        rL = fmax(rL, 1e-30);  rR = fmax(rR, 1e-30);
        PL = fmax(PL, 1e-30);  PR = fmax(PR, 1e-30);
    }

    FPrim wlL, wrR;
    wlL.rho = rL; wlL.vt = uL; wlL.vr = vL; wlL.P = PL;
    wrR.rho = rR; wrR.vt = uR; wrR.vr = vR; wrR.P = PR;
    // gpu_hllc(radial=false): normal is "vt" direction, tangential is "vr".
    // Here we set normal = u (x), tangential = v. radial=false => normal=vt.
    FFlux4 ff = gpu_hllc(wlL, wrR, gamma, /*radial=*/false);

    int f = fx_flat(i_face, j, sy);
    Fx_rho[f] = ff.f_rho;
    // gpu_hllc (radial=false): f_mr = ρ·u·v  (tangential), f_mt = ρ·u²+P (normal)
    Fx_mx [f] = ff.f_mt;   // normal (x)
    Fx_my [f] = ff.f_mr;   // tangential (y)
    Fx_E  [f] = ff.f_E;
    if (Fx_s != nullptr && w_X != nullptr) {
        double X_up = (ff.f_rho >= 0.0) ? w_X[cL] : w_X[cR];
        Fx_s[f] = ff.f_rho * X_up;
    }
}

// ============================================================
// y-direction flux (same story, roles of normal/tangential swapped).
// ============================================================
__global__ void k_athvl2_flux_y(
    const double* __restrict__ w_rho,
    const double* __restrict__ w_u,
    const double* __restrict__ w_v,
    const double* __restrict__ w_P,
    const double* __restrict__ w_X,       // nullable
    double* __restrict__ Fy_rho,
    double* __restrict__ Fy_mx,
    double* __restrict__ Fy_my,
    double* __restrict__ Fy_E,
    double* __restrict__ Fy_s,            // nullable
    int ny, int ng, int sx, int sy,
    int order, int limiter, double gamma)
{
    int i      = blockIdx.x * blockDim.x + threadIdx.x;
    int j_face = blockIdx.y * blockDim.y + threadIdx.y;
    int syp1 = sy + 1;                   // y-face stride (one extra per column)
    if (i < 1 || i >= sx - 1) return;
    if (j_face < ng - 1 || j_face > ng + ny) return;

    int jL = j_face - 1;
    int jR = j_face;
    int cL = cflat(i, jL, sy);
    int cR = cflat(i, jR, sy);

    double rL = w_rho[cL], uL = w_u[cL], vL = w_v[cL], PL = w_P[cL];
    double rR = w_rho[cR], uR = w_u[cR], vR = w_v[cR], PR = w_P[cR];

    if (order >= 2) {
        int jLL = jL - 1;
        int jRR = jR + 1;
        int cLL = cflat(i, jLL, sy);
        int cRR = cflat(i, jRR, sy);
        double drL = lim_dispatch(w_rho[cL] - w_rho[cLL], w_rho[cR] - w_rho[cL], limiter);
        double duL = lim_dispatch(w_u  [cL] - w_u  [cLL], w_u  [cR] - w_u  [cL], limiter);
        double dvL = lim_dispatch(w_v  [cL] - w_v  [cLL], w_v  [cR] - w_v  [cL], limiter);
        double dPL = lim_dispatch(w_P  [cL] - w_P  [cLL], w_P  [cR] - w_P  [cL], limiter);

        double drR = lim_dispatch(w_rho[cR] - w_rho[cL], w_rho[cRR] - w_rho[cR], limiter);
        double duR = lim_dispatch(w_u  [cR] - w_u  [cL], w_u  [cRR] - w_u  [cR], limiter);
        double dvR = lim_dispatch(w_v  [cR] - w_v  [cL], w_v  [cRR] - w_v  [cR], limiter);
        double dPR = lim_dispatch(w_P  [cR] - w_P  [cL], w_P  [cRR] - w_P  [cR], limiter);

        rL += 0.5 * drL;  uL += 0.5 * duL;  vL += 0.5 * dvL;  PL += 0.5 * dPL;
        rR -= 0.5 * drR;  uR -= 0.5 * duR;  vR -= 0.5 * dvR;  PR -= 0.5 * dPR;
        rL = fmax(rL, 1e-30);  rR = fmax(rR, 1e-30);
        PL = fmax(PL, 1e-30);  PR = fmax(PR, 1e-30);
    }

    // For y-direction: normal is v, tangential is u.
    FPrim wlL, wrR;
    wlL.rho = rL; wlL.vt = vL; wlL.vr = uL; wlL.P = PL;
    wrR.rho = rR; wrR.vt = vR; wrR.vr = uR; wrR.P = PR;
    FFlux4 ff = gpu_hllc(wlL, wrR, gamma, /*radial=*/false);

    int f = fy_flat(i, j_face, syp1);
    Fy_rho[f] = ff.f_rho;
    // radial=false: f_mt = ρ·v²+P (normal = y), f_mr = ρ·v·u (tangential = x)
    Fy_my [f] = ff.f_mt;
    Fy_mx [f] = ff.f_mr;
    Fy_E  [f] = ff.f_E;
    if (Fy_s != nullptr && w_X != nullptr) {
        double X_up = (ff.f_rho >= 0.0) ? w_X[cL] : w_X[cR];
        Fy_s[f] = ff.f_rho * X_up;
    }
}

// ============================================================
// Flux-divergence update:  u_new = u_old − dt·(∂_x Fx + ∂_y Fy)
// where u_old lives in d_rho/... (THE CONSERVED AT START OF STAGE, i.e.
// always u^n during vl2 stage 2) and u_new goes into d_rho_dst/....
// The source-term piece is applied in a separate kernel afterwards.
// ============================================================
__global__ void k_athvl2_flux_divergence(
    const double* u_rho, const double* u_mx, const double* u_my, const double* u_E,
    const double* u_s,     // nullable
    double* u_rho_dst, double* u_mx_dst, double* u_my_dst, double* u_E_dst,
    double* u_s_dst,       // nullable
    const double* Fx_rho, const double* Fx_mx, const double* Fx_my, const double* Fx_E,
    const double* Fx_s,    // nullable
    const double* Fy_rho, const double* Fy_mx, const double* Fy_my, const double* Fy_E,
    const double* Fy_s,    // nullable
    int nx, int ny, int ng, int sx, int sy,
    double dx, double dy, double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + ng;
    int j = blockIdx.y * blockDim.y + threadIdx.y + ng;
    if (i >= ng + nx || j >= ng + ny) return;
    int c  = cflat(i, j, sy);
    int fxL = fx_flat(i,     j, sy);
    int fxR = fx_flat(i + 1, j, sy);
    int syp1 = sy + 1;
    int fyB = fy_flat(i, j,     syp1);
    int fyT = fy_flat(i, j + 1, syp1);

    double inv_dx = dt / dx;
    double inv_dy = dt / dy;

    u_rho_dst[c] = u_rho[c] - inv_dx * (Fx_rho[fxR] - Fx_rho[fxL])
                            - inv_dy * (Fy_rho[fyT] - Fy_rho[fyB]);
    u_mx_dst [c] = u_mx [c] - inv_dx * (Fx_mx [fxR] - Fx_mx [fxL])
                            - inv_dy * (Fy_mx [fyT] - Fy_mx [fyB]);
    u_my_dst [c] = u_my [c] - inv_dx * (Fx_my [fxR] - Fx_my [fxL])
                            - inv_dy * (Fy_my [fyT] - Fy_my [fyB]);
    u_E_dst  [c] = u_E  [c] - inv_dx * (Fx_E  [fxR] - Fx_E  [fxL])
                            - inv_dy * (Fy_E  [fyT] - Fy_E  [fyB]);
    if (u_s != nullptr && u_s_dst != nullptr) {
        u_s_dst[c] = u_s[c] - inv_dx * (Fx_s[fxR] - Fx_s[fxL])
                            - inv_dy * (Fy_s[fyT] - Fy_s[fyB]);
    }
}

// ============================================================
// Source terms (variable g(y) gravity + volumetric heating q̇(y))
// Applied as Athena:
//   src_m2 = dt · ρ · (−g)         [ g > 0 means gravity pulls −y ]
//   src_E  = dt · (ρ·v_y·(−g) + q̇)
//   ρ, v_y are the "prim at start of stage" — passed in via w_rho, w_v.
// In Athena that's phydro->w at stage-1 start (n state) or stage-2 start (u*).
// We use the primitive arrays that were populated before the stage's flux calc.
// ============================================================
__global__ void k_athvl2_source_terms(
    const double* w_rho,
    const double* w_v,      // v_y
    double* u_mx,  // inplace cons update (only IM2 and IEN modified)
    double* u_my,
    double* u_E,
    const double* g_row,    // ny values, indexed by j-ng
    const double* q_row,    // ny values
    int nx, int ny, int ng, int sx, int sy,
    double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + ng;
    int j = blockIdx.y * blockDim.y + threadIdx.y + ng;
    if (i >= ng + nx || j >= ng + ny) return;
    int c = cflat(i, j, sy);
    int jr = j - ng;
    double g = g_row[jr];
    double q = q_row[jr];
    double rho = w_rho[c];
    double vy  = w_v  [c];
    // Gravity is along −ê_y:  du_my = −ρ·g·dt,   dE = −ρ·v_y·g·dt
    double sm = -dt * rho * g;
    u_my[c] += sm;
    u_E [c] += sm * vy;
    // Volumetric heating  dE += q̇·dt
    u_E [c] += dt * q;
}

// ============================================================
// Per-cell dt estimate  Δt_cell = min(dx/(|u|+c), dy/(|v|+c))
// ============================================================
__global__ void k_athvl2_cfl(
    const double* __restrict__ w_rho,
    const double* __restrict__ w_u,
    const double* __restrict__ w_v,
    const double* __restrict__ w_P,
    double* __restrict__ dt_buf,
    int nx, int ny, int ng, int sx, int sy,
    double dx, double dy, double gamma)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + ng;
    int j = blockIdx.y * blockDim.y + threadIdx.y + ng;
    int flat = (i - ng) * ny + (j - ng);
    if (i >= ng + nx || j >= ng + ny) {
        return;
    }
    int c = cflat(i, j, sy);
    double r = fmax(w_rho[c], 1e-30);
    double cs = sqrt(gamma * fmax(w_P[c], 1e-30) / r);
    double dtx = dx / (fabs(w_u[c]) + cs);
    double dty = dy / (fabs(w_v[c]) + cs);
    dt_buf[flat] = fmin(dtx, dty);
}
