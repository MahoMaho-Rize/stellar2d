// ============================================================
// athena_mhd_kernels.cu — CUDA kernels for the AthenaMHDSolver
//
// Derivation dossier:  docs/mhd_derivations/manuscript.pdf
//   §A1   ideal MHD conservation laws + Lorentz-force identity
//   §A3   7-wave flux Jacobian eigensystem
//   §A4   HLLD intermediate states
//   §A5   Constrained Transport preservation of ∇·B = 0
//   §A6   PLM reconstruction + TVD limiters
//   §A8   fast-magnetosonic CFL
//   §A9   HLLD degeneracy branches (see gpu_hlld.cuh)
//
// Grid (Yee-staggered):
//   cell-centred:  ρ, m_n, B_z_cc  (B_z is 1-D along z, cell-centred in xy)
//   x-faces (i+½): B_x
//   y-faces (j+½): B_y
//   corners (i+½, j+½): E_z   (composed from 4 neighbouring x/y faces)
//
// Indexing:
//   cflat  (i, j)  = i · sy + j,            sy = ny + 2g
//   fx_flat(i, j)  = i · sy + j,            (nx+1+2g) x (ny+2g)
//   fy_flat(i, j)  = i · syp1 + j,          (nx+2g) x (ny+1+2g)
//   cn_flat(i, j)  = i · syp1 + j,          (nx+1+2g) x (ny+1+2g)
// ============================================================

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#include "athena_mhd_solver.cuh"
#include "gpu/common/gpu_hlld.cuh"
#include "gpu/common/gpu_common.cuh"

namespace {

__device__ __forceinline__ int cflat (int i, int j, int sy)   { return i * sy   + j; }
__device__ __forceinline__ int fx_flat(int i, int j, int sy)   { return i * sy   + j; }
__device__ __forceinline__ int fy_flat(int i, int j, int syp1) { return i * syp1 + j; }
__device__ __forceinline__ int cn_flat(int i, int j, int syp1) { return i * syp1 + j; }

__device__ __forceinline__ double vl_limit(double a, double b) {
    double ab = a * b;
    if (ab <= 0.0) return 0.0;
    return 2.0 * ab / (a + b);
}
__device__ __forceinline__ double mm_limit(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return (a >= 0.0) ? fmin(a, b) : fmax(a, b);
}
__device__ __forceinline__ double lim_dispatch(double a, double b, int lim) {
    return (lim == 1) ? mm_limit(a, b) : vl_limit(a, b);
}

} // namespace

// ============================================================
// cons → prim  (cell-centred; reads face-averaged B to build B_cc)
// ============================================================
__global__ void k_athmhd_cons_to_prim(
    const double* __restrict__ rho,
    const double* __restrict__ mx,
    const double* __restrict__ my,
    const double* __restrict__ mz,
    const double* __restrict__ E,
    const double* __restrict__ Bxf,       // size (nx+1+2g) x (ny+2g)
    const double* __restrict__ Byf,       // size (nx+2g)   x (ny+1+2g)
    const double* __restrict__ Bz_cc,     // cell-centred (Bz stays cc in 2D)
    double* __restrict__ w_rho,
    double* __restrict__ w_u,
    double* __restrict__ w_v,
    double* __restrict__ w_w,
    double* __restrict__ w_Bx,
    double* __restrict__ w_By,
    double* __restrict__ w_Bz,
    double* __restrict__ w_P,
    double* __restrict__ Bx_cc_out,       // diagnostic, same as w_Bx
    double* __restrict__ By_cc_out,       // diagnostic, same as w_By
    int sx, int sy, double gm1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= sx || j >= sy) return;
    int c = cflat(i, j, sy);
    int syp1 = sy + 1;
    // B_cc from face averages (§A5 footnote; arithmetic average is
    // the standard Yee → cell-centre reconstruction).
    double Bxc = 0.5 * (Bxf[fx_flat(i,   j, sy)]
                      + Bxf[fx_flat(i+1, j, sy)]);
    double Byc = 0.5 * (Byf[fy_flat(i, j,   syp1)]
                      + Byf[fy_flat(i, j+1, syp1)]);
    double Bzc = Bz_cc[c];
    double r  = fmax(rho[c], GPU_MHD_FLOOR);
    double u  = mx[c] / r;
    double v  = my[c] / r;
    double w_ = mz[c] / r;
    double ke = 0.5 * r * (u*u + v*v + w_*w_);
    double me = 0.5 * (Bxc*Bxc + Byc*Byc + Bzc*Bzc);
    double P  = gm1 * (E[c] - ke - me);
    P = fmax(P, GPU_MHD_FLOOR);
    w_rho[c] = r;
    w_u[c]   = u;
    w_v[c]   = v;
    w_w[c]   = w_;
    w_Bx[c]  = Bxc;
    w_By[c]  = Byc;
    w_Bz[c]  = Bzc;
    w_P[c]   = P;
    Bx_cc_out[c] = Bxc;
    By_cc_out[c] = Byc;
}

// ============================================================
// Ghost fills — same layout as athena_vl2, but extended to ALL
// seven conservative fields plus face B.
//
// For face-normal B under reflecting BC:
//   x-reflect wall at i=ng:   B_x_face is ANTISYMMETRIC
//                              (B_x flips, because it's normal to wall)
//     → B_x_face[ng-1-g, j] = -B_x_face[ng+g, j]
//   y-reflect wall at j=ng:   B_y_face is ANTISYMMETRIC
// Tangential face B is SYMMETRIC at reflecting walls.
// ============================================================

__global__ void k_athmhd_ghost_x_periodic_cc(
    double* rho, double* mx, double* my, double* mz, double* E, double* Bz,
    int nx, int ng, int sx, int sy)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= sy || g >= ng) return;
    int iL_dst = g;
    int iL_src = nx + g;
    int iR_dst = ng + nx + g;
    int iR_src = ng + g;
    int cLd = cflat(iL_dst, j, sy), cLs = cflat(iL_src, j, sy);
    int cRd = cflat(iR_dst, j, sy), cRs = cflat(iR_src, j, sy);
    rho[cLd] = rho[cLs]; mx[cLd] = mx[cLs]; my[cLd] = my[cLs];
    mz [cLd] = mz [cLs]; E [cLd] = E [cLs]; Bz[cLd] = Bz[cLs];
    rho[cRd] = rho[cRs]; mx[cRd] = mx[cRs]; my[cRd] = my[cRs];
    mz [cRd] = mz [cRs]; E [cRd] = E [cRs]; Bz[cRd] = Bz[cRs];
}

__global__ void k_athmhd_ghost_y_periodic_cc(
    double* rho, double* mx, double* my, double* mz, double* E, double* Bz,
    int ny, int ng, int sx, int sy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= sx || g >= ng) return;
    int jB_dst = g, jB_src = ny + g;
    int jT_dst = ng + ny + g, jT_src = ng + g;
    int cBd = cflat(i, jB_dst, sy), cBs = cflat(i, jB_src, sy);
    int cTd = cflat(i, jT_dst, sy), cTs = cflat(i, jT_src, sy);
    rho[cBd] = rho[cBs]; mx[cBd] = mx[cBs]; my[cBd] = my[cBs];
    mz [cBd] = mz [cBs]; E [cBd] = E [cBs]; Bz[cBd] = Bz[cBs];
    rho[cTd] = rho[cTs]; mx[cTd] = mx[cTs]; my[cTd] = my[cTs];
    mz [cTd] = mz [cTs]; E [cTd] = E [cTs]; Bz[cTd] = Bz[cTs];
}

__global__ void k_athmhd_ghost_y_reflect_cc(
    double* rho, double* mx, double* my, double* mz, double* E, double* Bz,
    int ny, int ng, int sx, int sy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= sx || g >= ng) return;
    int jBd = ng - 1 - g, jBs = ng + g;
    int jTd = ng + ny + g, jTs = ng + ny - 1 - g;
    int cBd = cflat(i, jBd, sy), cBs = cflat(i, jBs, sy);
    int cTd = cflat(i, jTd, sy), cTs = cflat(i, jTs, sy);
    rho[cBd] =  rho[cBs]; mx[cBd] =  mx[cBs]; my[cBd] = -my[cBs];
    mz [cBd] =  mz [cBs]; E [cBd] =  E [cBs]; Bz[cBd] =  Bz[cBs];  // Bz tangential
    rho[cTd] =  rho[cTs]; mx[cTd] =  mx[cTs]; my[cTd] = -my[cTs];
    mz [cTd] =  mz [cTs]; E [cTd] =  E [cTs]; Bz[cTd] =  Bz[cTs];
}

// Face-B periodic: both Bxf (at x-faces) and Byf (at y-faces) wrap.
__global__ void k_athmhd_ghost_x_periodic_face(
    double* Bxf, double* Byf,
    int nx, int ny, int ng, int sy)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    int syp1 = sy + 1;
    int syfx = sy;                                 // Bxf column stride
    if (g < ng) {
        // Bxf: columns live at x-faces (nx+1+2g).  Periodic wrap:
        //   left ghost i = g  ←  interior i = nx + g
        //   right ghost i = ng + nx + 1 + g   ←  interior i = ng + 1 + g
        if (j < sy) {
            int iLd = g;
            int iLs = nx + g;
            int iRd = ng + nx + 1 + g;
            int iRs = ng + 1 + g;
            Bxf[fx_flat(iLd, j, syfx)] = Bxf[fx_flat(iLs, j, syfx)];
            Bxf[fx_flat(iRd, j, syfx)] = Bxf[fx_flat(iRs, j, syfx)];
        }
        // Byf: columns live at cell centres (nx+2g).  Periodic wrap
        //   identical to cell-centred fields.
        if (j < syp1) {
            int iLd = g;
            int iLs = nx + g;
            int iRd = ng + nx + g;
            int iRs = ng + g;
            Byf[fy_flat(iLd, j, syp1)] = Byf[fy_flat(iLs, j, syp1)];
            Byf[fy_flat(iRd, j, syp1)] = Byf[fy_flat(iRs, j, syp1)];
        }
    }
}

__global__ void k_athmhd_ghost_y_periodic_face(
    double* Bxf, double* Byf,
    int nx, int ny, int ng, int sy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    int syp1 = sy + 1;
    int syfx = sy;
    if (g < ng) {
        // Byf wraps in j like cell-centred
        if (i < nx + 2 * ng) {
            int jBd = g;
            int jBs = ny + g;
            int jTd = ng + ny + 1 + g;
            int jTs = ng + 1 + g;
            Byf[fy_flat(i, jBd, syp1)] = Byf[fy_flat(i, jBs, syp1)];
            Byf[fy_flat(i, jTd, syp1)] = Byf[fy_flat(i, jTs, syp1)];
        }
        // Bxf: its column has stride sy=ny+2g (same as cc in y).
        if (i < nx + 1 + 2 * ng) {
            int jBd = g;
            int jBs = ny + g;
            int jTd = ng + ny + g;
            int jTs = ng + g;
            Bxf[fx_flat(i, jBd, syfx)] = Bxf[fx_flat(i, jBs, syfx)];
            Bxf[fx_flat(i, jTd, syfx)] = Bxf[fx_flat(i, jTs, syfx)];
        }
    }
}

// Outflow (zero-gradient) in y: cell-centred copy.
__global__ void k_athmhd_ghost_y_outflow_cc(
    double* rho, double* mx, double* my, double* mz, double* E, double* Bz,
    int ny, int ng, int sx, int sy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= sx || g >= ng) return;
    int jBd = ng - 1 - g;
    int jBs = ng;                    // bottom interior edge
    int jTd = ng + ny + g;
    int jTs = ng + ny - 1;           // top interior edge
    int cBd = cflat(i, jBd, sy), cBs = cflat(i, jBs, sy);
    int cTd = cflat(i, jTd, sy), cTs = cflat(i, jTs, sy);
    rho[cBd] = rho[cBs]; mx[cBd] = mx[cBs]; my[cBd] = my[cBs];
    mz [cBd] = mz [cBs]; E [cBd] = E [cBs]; Bz[cBd] = Bz[cBs];
    rho[cTd] = rho[cTs]; mx[cTd] = mx[cTs]; my[cTd] = my[cTs];
    mz [cTd] = mz [cTs]; E [cTd] = E [cTs]; Bz[cTd] = Bz[cTs];
}

// Outflow in x for cell-centred (zero-gradient copy).
__global__ void k_athmhd_ghost_x_outflow_cc(
    double* rho, double* mx, double* my, double* mz, double* E, double* Bz,
    int nx, int ng, int sx, int sy)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= sy || g >= ng) return;
    int iL_dst = ng - 1 - g;
    int iL_src = ng;
    int iR_dst = ng + nx + g;
    int iR_src = ng + nx - 1;
    int cLd = cflat(iL_dst, j, sy), cLs = cflat(iL_src, j, sy);
    int cRd = cflat(iR_dst, j, sy), cRs = cflat(iR_src, j, sy);
    rho[cLd] = rho[cLs]; mx[cLd] = mx[cLs]; my[cLd] = my[cLs];
    mz [cLd] = mz [cLs]; E [cLd] = E [cLs]; Bz[cLd] = Bz[cLs];
    rho[cRd] = rho[cRs]; mx[cRd] = mx[cRs]; my[cRd] = my[cRs];
    mz [cRd] = mz [cRs]; E [cRd] = E [cRs]; Bz[cRd] = Bz[cRs];
}

// Outflow in x for face-B (zero-gradient copy of both face arrays).
// Safe enough when the flow at the boundary is genuinely outflowing;
// will slowly violate divB by O(ULP × t) if a wave re-enters, but for
// the standard rotor / Brio-Wu / RJ tests the wave reaches the
// boundary only at the end of the run.
__global__ void k_athmhd_ghost_x_outflow_face(
    double* Bxf, double* Byf,
    int nx, int ny, int ng, int sy)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    int syp1 = sy + 1;
    int syfx = sy;
    if (g < ng) {
        if (j < sy) {
            int iLd = ng - 1 - g;
            int iLs = ng;                     // first interior x-face
            int iRd = ng + nx + 1 + g;
            int iRs = ng + nx;                // last interior x-face
            Bxf[fx_flat(iLd, j, syfx)] = Bxf[fx_flat(iLs, j, syfx)];
            Bxf[fx_flat(iRd, j, syfx)] = Bxf[fx_flat(iRs, j, syfx)];
        }
        if (j < syp1) {
            int iLd = ng - 1 - g;
            int iLs = ng;
            int iRd = ng + nx + g;
            int iRs = ng + nx - 1;
            Byf[fy_flat(iLd, j, syp1)] = Byf[fy_flat(iLs, j, syp1)];
            Byf[fy_flat(iRd, j, syp1)] = Byf[fy_flat(iRs, j, syp1)];
        }
    }
}

// Outflow in y for face-B.
__global__ void k_athmhd_ghost_y_outflow_face(
    double* Bxf, double* Byf,
    int nx, int ny, int ng, int sy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    int syp1 = sy + 1;
    int syfx = sy;
    if (g < ng) {
        if (i < nx + 2 * ng) {
            int jBd = ng - 1 - g;
            int jBs = ng;                     // first interior y-face
            int jTd = ng + ny + 1 + g;
            int jTs = ng + ny;                // last interior y-face
            Byf[fy_flat(i, jBd, syp1)] = Byf[fy_flat(i, jBs, syp1)];
            Byf[fy_flat(i, jTd, syp1)] = Byf[fy_flat(i, jTs, syp1)];
        }
        if (i < nx + 1 + 2 * ng) {
            int jBd = ng - 1 - g;
            int jBs = ng;
            int jTd = ng + ny + g;
            int jTs = ng + ny - 1;
            Bxf[fx_flat(i, jBd, syfx)] = Bxf[fx_flat(i, jBs, syfx)];
            Bxf[fx_flat(i, jTd, syfx)] = Bxf[fx_flat(i, jTs, syfx)];
        }
    }
}

// Reflecting y-wall:  B_y_face antisymmetric, B_x_face symmetric.
__global__ void k_athmhd_ghost_y_reflect_face(
    double* Bxf, double* Byf,
    int nx, int ny, int ng, int sy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    int syp1 = sy + 1;
    int syfx = sy;
    if (g < ng) {
        if (i < nx + 2 * ng) {
            // Byf normal to y-wall:  antisymmetric reflection
            //   bottom wall face at j = ng
            //   ghost at j = ng − 1 − g  ←  -Byf at j = ng + 1 + g
            //   (Byf[ng] is ON the wall = 0 enforced separately if needed)
            int jBd = ng - 1 - g;
            int jBs = ng + 1 + g;
            int jTd = ng + ny + 1 + g;
            int jTs = ng + ny - 1 - g;
            Byf[fy_flat(i, jBd, syp1)] = -Byf[fy_flat(i, jBs, syp1)];
            Byf[fy_flat(i, jTd, syp1)] = -Byf[fy_flat(i, jTs, syp1)];
        }
        if (i < nx + 1 + 2 * ng) {
            // Bxf tangential to y-wall:  symmetric reflection
            int jBd = ng - 1 - g;
            int jBs = ng + g;
            int jTd = ng + ny + g;
            int jTs = ng + ny - 1 - g;
            Bxf[fx_flat(i, jBd, syfx)] =  Bxf[fx_flat(i, jBs, syfx)];
            Bxf[fx_flat(i, jTd, syfx)] =  Bxf[fx_flat(i, jTs, syfx)];
        }
    }
}

// ============================================================
// Reconstruction helper — limit a 5-tuple of primitives (rho, u, v, w, P)
// and a 2-tuple of transverse B (Bt, Bs) using van Leer / minmod on each.
// Returns L/R states at the face.
// ============================================================
__device__ __forceinline__
void recon_mhd_prim(const double qLL[8], const double qL [8],
                    const double qR [8], const double qRR[8],
                    double qL_face[8], double qR_face[8],
                    int order, int limiter)
{
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        qL_face[k] = qL[k];
        qR_face[k] = qR[k];
    }
    if (order >= 2) {
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            double dL = lim_dispatch(qL[k] - qLL[k], qR[k] - qL[k], limiter);
            double dR = lim_dispatch(qR[k] - qL[k], qRR[k] - qR[k], limiter);
            qL_face[k] = qL[k] + 0.5 * dL;
            qR_face[k] = qR[k] - 0.5 * dR;
        }
    }
    // floor density and pressure
    if (qL_face[0] < GPU_MHD_FLOOR) qL_face[0] = GPU_MHD_FLOOR;
    if (qR_face[0] < GPU_MHD_FLOOR) qR_face[0] = GPU_MHD_FLOOR;
    if (qL_face[7] < GPU_MHD_FLOOR) qL_face[7] = GPU_MHD_FLOOR;
    if (qR_face[7] < GPU_MHD_FLOOR) qR_face[7] = GPU_MHD_FLOOR;
}

// ============================================================
// x-direction MHD flux via PLM + HLLD.
//
// At face i+½ (between cells i and i+1):
//   Normal direction = x.  The RIEMANN state needs:
//     prim = (ρ, v_x, v_y, v_z, B_y_cc, B_z_cc, P)
//     + constant B_x on the face = Bxf[i+1, j].
//   HLLD returns flux with 7 components mapped to cons order.
//   We also emit E_zx_face[i+½, j] = − F^x_{B_y}  (for CT, §A5).
// ============================================================
__global__ void k_athmhd_flux_x(
    const double* __restrict__ w_rho,
    const double* __restrict__ w_u,
    const double* __restrict__ w_v,
    const double* __restrict__ w_w,
    const double* __restrict__ w_Bx,
    const double* __restrict__ w_By,
    const double* __restrict__ w_Bz,
    const double* __restrict__ w_P,
    const double* __restrict__ Bxf,
    double* __restrict__ Fx_rho, double* __restrict__ Fx_mx,
    double* __restrict__ Fx_my,  double* __restrict__ Fx_mz,
    double* __restrict__ Fx_By,  double* __restrict__ Fx_Bz,
    double* __restrict__ Fx_E,
    double* __restrict__ Ezx_face,
    int nx, int ng, int sx, int sy,
    int order, int limiter, double gamma)
{
    int i_face = blockIdx.x * blockDim.x + threadIdx.x;
    int j      = blockIdx.y * blockDim.y + threadIdx.y;
    if (i_face < ng - 1 || i_face > ng + nx) return;
    if (j < 1 || j >= sy - 1) return;

    int iL = i_face - 1;
    int iR = i_face;
    int cLL = cflat(iL - 1, j, sy);
    int cL  = cflat(iL,     j, sy);
    int cR  = cflat(iR,     j, sy);
    int cRR = cflat(iR + 1, j, sy);

    double qLL[8], qL[8], qR[8], qRR[8];
    qLL[0]=w_rho[cLL]; qLL[1]=w_u[cLL]; qLL[2]=w_v[cLL]; qLL[3]=w_w[cLL];
    qLL[4]=w_By[cLL];  qLL[5]=w_Bz[cLL]; qLL[6]=w_Bx[cLL]; qLL[7]=w_P[cLL];
    qL [0]=w_rho[cL ]; qL [1]=w_u[cL ]; qL [2]=w_v[cL ]; qL [3]=w_w[cL ];
    qL [4]=w_By[cL ];  qL [5]=w_Bz[cL ]; qL [6]=w_Bx[cL ]; qL [7]=w_P[cL ];
    qR [0]=w_rho[cR ]; qR [1]=w_u[cR ]; qR [2]=w_v[cR ]; qR [3]=w_w[cR ];
    qR [4]=w_By[cR ];  qR [5]=w_Bz[cR ]; qR [6]=w_Bx[cR ]; qR [7]=w_P[cR ];
    qRR[0]=w_rho[cRR]; qRR[1]=w_u[cRR]; qRR[2]=w_v[cRR]; qRR[3]=w_w[cRR];
    qRR[4]=w_By[cRR];  qRR[5]=w_Bz[cRR]; qRR[6]=w_Bx[cRR]; qRR[7]=w_P[cRR];

    double qLf[8], qRf[8];
    recon_mhd_prim(qLL, qL, qR, qRR, qLf, qRf, order, limiter);

    // Longitudinal (normal) B at the x-face — FACE-CENTRED, not
    // reconstructed from cell centres (that would break ∇·B = 0).
    double Bn = Bxf[fx_flat(i_face, j, sy)];

    MHDPrim wl, wr;
    wl.rho = qLf[0]; wl.vn = qLf[1]; wl.vt = qLf[2]; wl.vs = qLf[3];
    wl.Bt  = qLf[4]; wl.Bs  = qLf[5];                   wl.P  = qLf[7];
    wr.rho = qRf[0]; wr.vn = qRf[1]; wr.vt = qRf[2]; wr.vs = qRf[3];
    wr.Bt  = qRf[4]; wr.Bs  = qRf[5];                   wr.P  = qRf[7];

    FluxMHD7 f = gpu_hlld(wl, wr, Bn, gamma);

    int fx = fx_flat(i_face, j, sy);
    Fx_rho[fx] = f.f_rho;
    Fx_mx [fx] = f.f_mn;
    Fx_my [fx] = f.f_mt;
    Fx_mz [fx] = f.f_ms;
    Fx_By [fx] = f.f_Bt;                // flux of B_y across an x-face
    Fx_Bz [fx] = f.f_Bs;                // flux of B_z across an x-face
    Fx_E  [fx] = f.f_E;

    // E_z from x-sweep.  At an x-face, the MHD B_y flux is
    //   F^x_{B_y} = v_x B_y − v_y B_x  =  −E_z  (right-hand convention)
    // so  E_z-contribution-from-x-sweep  =  − F^x_{B_y}.
    Ezx_face[fx] = -f.f_Bt;
}

// ============================================================
// y-direction MHD flux via PLM + HLLD.  Same structure as flux_x
// with coordinate swap  (x↔y), i.e. normal = y, tangential-1 = z (held),
// tangential-2 = x.  Order of MHDPrim members becomes:
//   vn = v_y, vt = v_z, vs = v_x  (keeps right-handed)
//   Bt = B_z, Bs = B_x
// We pack carefully to preserve sign conventions.
// ============================================================
__global__ void k_athmhd_flux_y(
    const double* __restrict__ w_rho,
    const double* __restrict__ w_u,      // v_x
    const double* __restrict__ w_v,      // v_y
    const double* __restrict__ w_w,      // v_z
    const double* __restrict__ w_Bx,
    const double* __restrict__ w_By,
    const double* __restrict__ w_Bz,
    const double* __restrict__ w_P,
    const double* __restrict__ Byf,
    double* __restrict__ Gy_rho, double* __restrict__ Gy_mx,
    double* __restrict__ Gy_my,  double* __restrict__ Gy_mz,
    double* __restrict__ Gy_Bx,  double* __restrict__ Gy_Bz,
    double* __restrict__ Gy_E,
    double* __restrict__ Ezy_face,
    int ny, int ng, int sx, int sy,
    int order, int limiter, double gamma)
{
    int i      = blockIdx.x * blockDim.x + threadIdx.x;
    int j_face = blockIdx.y * blockDim.y + threadIdx.y;
    int syp1   = sy + 1;
    if (i < 1 || i >= sx - 1) return;
    if (j_face < ng - 1 || j_face > ng + ny) return;

    int jL = j_face - 1;
    int jR = j_face;
    int cLL = cflat(i, jL - 1, sy);
    int cL  = cflat(i, jL,     sy);
    int cR  = cflat(i, jR,     sy);
    int cRR = cflat(i, jR + 1, sy);

    // Use the NORMAL = y coordinate convention for HLLD:
    //   vn = v_y,   vt = v_z,   vs = v_x
    //   Bt = B_z,   Bs = B_x
    // Pack the 8-vector same-order as flux_x but with these indices:
    //   [rho, vn, vt, vs, Bt, Bs, Bn(unused), P]
    auto pack = [&](int c, double q[8]) {
        q[0] = w_rho[c];
        q[1] = w_v  [c];    // vn = v_y
        q[2] = w_w  [c];    // vt = v_z
        q[3] = w_u  [c];    // vs = v_x
        q[4] = w_Bz [c];    // Bt = B_z
        q[5] = w_Bx [c];    // Bs = B_x
        q[6] = w_By [c];    // Bn (unused in reconstruction, kept for completeness)
        q[7] = w_P  [c];
    };
    double qLL[8], qL[8], qR[8], qRR[8];
    pack(cLL, qLL); pack(cL, qL); pack(cR, qR); pack(cRR, qRR);

    double qLf[8], qRf[8];
    recon_mhd_prim(qLL, qL, qR, qRR, qLf, qRf, order, limiter);

    double Bn = Byf[fy_flat(i, j_face, syp1)];

    MHDPrim wl, wr;
    wl.rho = qLf[0]; wl.vn = qLf[1]; wl.vt = qLf[2]; wl.vs = qLf[3];
    wl.Bt  = qLf[4]; wl.Bs  = qLf[5];                   wl.P  = qLf[7];
    wr.rho = qRf[0]; wr.vn = qRf[1]; wr.vt = qRf[2]; wr.vs = qRf[3];
    wr.Bt  = qRf[4]; wr.Bs  = qRf[5];                   wr.P  = qRf[7];

    FluxMHD7 f = gpu_hlld(wl, wr, Bn, gamma);

    // HLLD returned (n=y, t=z, s=x).  Remap to physical (x, y, z):
    //   f.f_mn  → flux of m_y (G_y^{m_y})
    //   f.f_mt  → flux of m_z (G_y^{m_z})
    //   f.f_ms  → flux of m_x (G_y^{m_x})
    //   f.f_Bt  → flux of B_z (G_y^{B_z})   (tangent-1 in rotated frame)
    //   f.f_Bs  → flux of B_x (G_y^{B_x})   (tangent-2 in rotated frame)
    int fy = fy_flat(i, j_face, syp1);
    Gy_rho[fy] = f.f_rho;
    Gy_my [fy] = f.f_mn;
    Gy_mz [fy] = f.f_mt;
    Gy_mx [fy] = f.f_ms;
    Gy_Bz [fy] = f.f_Bt;
    Gy_Bx [fy] = f.f_Bs;
    Gy_E  [fy] = f.f_E;

    // CT: y-face EMF contribution.
    //   F^y_{B_x} = v_y B_x − v_x B_y = +E_z  (right-hand rule)
    //   So E_z-contribution-from-y-sweep = + F^y_{B_x} = + f.f_Bs.
    Ezy_face[fy] = f.f_Bs;
}

// ============================================================
// Corner E_z via Gardiner-Stone 2005 average (§A5).
//
//   E_z^{i+½, j+½} = ¼ [ Ezx_face^{i+½, j}  + Ezx_face^{i+½, j+1}
//                      + Ezy_face^{i, j+½}  + Ezy_face^{i+1, j+½} ]
//
// The Ezx_face array lives on x-faces (i+½, j cell-row); Ezy_face on
// y-faces (i cell-col, j+½).  We sample 4 neighbours for each
// interior corner.
// ============================================================
__global__ void k_athmhd_corner_emf(
    const double* __restrict__ Ezx_face,    // x-face EMF
    const double* __restrict__ Ezy_face,    // y-face EMF
    double* __restrict__ Ez_corner,
    int nx, int ny, int ng, int sy)
{
    int syp1 = sy + 1;
    int i_corner = blockIdx.x * blockDim.x + threadIdx.x;
    int j_corner = blockIdx.y * blockDim.y + threadIdx.y;
    // Corners at (i+½, j+½) map to (i_corner, j_corner) index with
    // i_corner ∈ [ng, ng+nx], j_corner ∈ [ng, ng+ny] (nx+1 × ny+1
    // active corners).
    if (i_corner < ng || i_corner > ng + nx) return;
    if (j_corner < ng || j_corner > ng + ny) return;

    // Ezx_face indices:
    //   above corner (i+½, j+½) :  x-face at (i_corner, j_corner)
    //   below corner (i+½, j-½) :  x-face at (i_corner, j_corner - 1)
    double ex_above = Ezx_face[fx_flat(i_corner, j_corner,     sy)];
    double ex_below = Ezx_face[fx_flat(i_corner, j_corner - 1, sy)];

    // Ezy_face indices:
    //   left  of corner (i-½, j+½) : y-face at (i_corner - 1, j_corner)
    //   right of corner (i+½, j+½) : y-face at (i_corner,     j_corner)
    double ey_left  = Ezy_face[fy_flat(i_corner - 1, j_corner, syp1)];
    double ey_right = Ezy_face[fy_flat(i_corner,     j_corner, syp1)];

    Ez_corner[cn_flat(i_corner, j_corner, syp1)] =
        0.25 * (ex_above + ex_below + ey_left + ey_right);
}

// ============================================================
// Flux-divergence update for the cell-centred conservatives.
//   U_dst = U_src − dt · (∂_x Fx + ∂_y Gy)
// This handles  (ρ, m_x, m_y, m_z, B_z_cc, E) — the 6 cell-centred
// evolved quantities (B_x, B_y on faces updated separately by CT).
// ============================================================
__global__ void k_athmhd_flux_divergence(
    const double* u_rho, const double* u_mx, const double* u_my,
    const double* u_mz, const double* u_Bz, const double* u_E,
    double* u_rho_dst, double* u_mx_dst, double* u_my_dst,
    double* u_mz_dst, double* u_Bz_dst, double* u_E_dst,
    const double* Fx_rho, const double* Fx_mx, const double* Fx_my,
    const double* Fx_mz, const double* Fx_Bz, const double* Fx_E,
    const double* Gy_rho, const double* Gy_mx, const double* Gy_my,
    const double* Gy_mz, const double* Gy_Bz, const double* Gy_E,
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
    double a = dt / dx, b = dt / dy;
    u_rho_dst[c] = u_rho[c] - a*(Fx_rho[fxR]-Fx_rho[fxL]) - b*(Gy_rho[fyT]-Gy_rho[fyB]);
    u_mx_dst [c] = u_mx [c] - a*(Fx_mx [fxR]-Fx_mx [fxL]) - b*(Gy_mx [fyT]-Gy_mx [fyB]);
    u_my_dst [c] = u_my [c] - a*(Fx_my [fxR]-Fx_my [fxL]) - b*(Gy_my [fyT]-Gy_my [fyB]);
    u_mz_dst [c] = u_mz [c] - a*(Fx_mz [fxR]-Fx_mz [fxL]) - b*(Gy_mz [fyT]-Gy_mz [fyB]);
    u_Bz_dst [c] = u_Bz [c] - a*(Fx_Bz [fxR]-Fx_Bz [fxL]) - b*(Gy_Bz [fyT]-Gy_Bz [fyB]);
    u_E_dst  [c] = u_E  [c] - a*(Fx_E  [fxR]-Fx_E  [fxL]) - b*(Gy_E  [fyT]-Gy_E  [fyB]);
}

// ============================================================
// Face-B update via CT (§A5 Evans-Hawley 1988):
//   B_x^{i+½,j, n+1} = B_x^{i+½,j, n} − (Δt/Δy)(E_z^{i+½,j+½} − E_z^{i+½,j-½})
//   B_y^{i,j+½, n+1} = B_y^{i,j+½, n} + (Δt/Δx)(E_z^{i+½,j+½} − E_z^{i-½,j+½})
// ============================================================
__global__ void k_athmhd_update_face_B(
    const double* Bxf_src, const double* Byf_src,
    double* Bxf_dst, double* Byf_dst,
    const double* Ez_corner,
    int nx, int ny, int ng, int sx, int sy,
    double dx, double dy, double dt)
{
    int syp1 = sy + 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    // Bxf: update at x-faces in interior + one-column cushion
    //   active range i ∈ [ng, ng+nx],  j ∈ [ng, ng+ny)
    if (i >= ng && i <= ng + nx && j >= ng && j < ng + ny) {
        int fx = fx_flat(i, j, sy);
        double Ez_top = Ez_corner[cn_flat(i, j + 1, syp1)];
        double Ez_bot = Ez_corner[cn_flat(i, j,     syp1)];
        Bxf_dst[fx] = Bxf_src[fx] - (dt / dy) * (Ez_top - Ez_bot);
    }
    // Byf: active range i ∈ [ng, ng+nx), j ∈ [ng, ng+ny]
    if (i >= ng && i < ng + nx && j >= ng && j <= ng + ny) {
        int fy = fy_flat(i, j, syp1);
        double Ez_right = Ez_corner[cn_flat(i + 1, j, syp1)];
        double Ez_left  = Ez_corner[cn_flat(i,     j, syp1)];
        Byf_dst[fy] = Byf_src[fy] + (dt / dx) * (Ez_right - Ez_left);
    }
}

// ============================================================
// Copy face-B arrays (used in stage swaps).
// ============================================================
__global__ void k_athmhd_copy_face_B(
    const double* src_Bxf, const double* src_Byf,
    double* dst_Bxf, double* dst_Byf,
    int sx, int sy)
{
    int syp1 = sy + 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < sx + 1 && j < sy) dst_Bxf[i * sy + j] = src_Bxf[i * sy + j];
    if (i < sx && j < sy + 1) dst_Byf[i * syp1 + j] = src_Byf[i * syp1 + j];
}

// ============================================================
// Source term: gravity along −ê_y (optional for MHSE wind runs;
// not used in Brio-Wu / field-loop / linear-wave / OT).
// ============================================================
__global__ void k_athmhd_source_gravity(
    const double* w_rho, const double* w_v,
    double* u_my, double* u_E,
    const double* g_row,
    int nx, int ny, int ng, int sx, int sy, double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + ng;
    int j = blockIdx.y * blockDim.y + threadIdx.y + ng;
    if (i >= ng + nx || j >= ng + ny) return;
    int c = cflat(i, j, sy);
    int jr = j - ng;
    double g = g_row[jr];
    double sm = -dt * w_rho[c] * g;
    u_my[c] += sm;
    u_E [c] += sm * w_v[c];
}

// ============================================================
// Well-balanced MHSE defect subtraction (§B4):
//   U_dst -= dt_stage · rhs_hse_*
// where rhs_hse_* = R(U_hse) + gravity(U_hse) was captured once at
// snapshot_hse().  For an atmosphere at MHSE this makes the total
// residual exactly zero (machine precision), so an unperturbed HSE
// state stays put forever regardless of grid spacing.
// Applied to all 6 cell-centred conservatives so the ρ and Bz flux
// residuals at reflective walls also cancel.
// ============================================================
__global__ void k_athmhd_source_wb_subtract(
    double* u_rho, double* u_mx, double* u_my, double* u_mz,
    double* u_E, double* u_Bz,
    const double* rhs_hse_rho, const double* rhs_hse_mx,
    const double* rhs_hse_my,  const double* rhs_hse_mz,
    const double* rhs_hse_E,   const double* rhs_hse_Bz,
    int nx, int ny, int ng, int sx, int sy, double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + ng;
    int j = blockIdx.y * blockDim.y + threadIdx.y + ng;
    if (i >= ng + nx || j >= ng + ny) return;
    int c = cflat(i, j, sy);
    u_rho[c] -= dt * rhs_hse_rho[c];
    u_mx [c] -= dt * rhs_hse_mx [c];
    u_my [c] -= dt * rhs_hse_my [c];
    u_mz [c] -= dt * rhs_hse_mz [c];
    u_E  [c] -= dt * rhs_hse_E  [c];
    u_Bz [c] -= dt * rhs_hse_Bz [c];
}

// ============================================================
// CFL:  Δt_cell = min( Δx / (|u|+c_f),  Δy / (|v|+c_f) )
// with c_f the fast-magnetosonic speed in the respective direction.
// (§A8: unsplit rule; here we use the per-direction max.)
// ============================================================
__global__ void k_athmhd_cfl(
    const double* __restrict__ w_rho,
    const double* __restrict__ w_u,
    const double* __restrict__ w_v,
    const double* __restrict__ w_w,
    const double* __restrict__ w_Bx,
    const double* __restrict__ w_By,
    const double* __restrict__ w_Bz,
    const double* __restrict__ w_P,
    double* __restrict__ dt_buf,
    int nx, int ny, int ng, int sx, int sy,
    double dx, double dy, double gamma)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + ng;
    int j = blockIdx.y * blockDim.y + threadIdx.y + ng;
    int flat = (i - ng) * ny + (j - ng);
    if (i >= ng + nx || j >= ng + ny) return;
    int c = cflat(i, j, sy);
    double r = fmax(w_rho[c], GPU_MHD_FLOOR);
    double P = fmax(w_P[c], GPU_MHD_FLOOR);
    // c_f along x: normal = B_x
    double cfx = sqrt(mhd_fast_speed_sq(r, P, w_Bx[c], w_By[c], w_Bz[c], gamma));
    // c_f along y: normal = B_y (swap transverse labels; same formula)
    double cfy = sqrt(mhd_fast_speed_sq(r, P, w_By[c], w_Bx[c], w_Bz[c], gamma));
    double dtx = dx / (fabs(w_u[c]) + cfx);
    double dty = dy / (fabs(w_v[c]) + cfy);
    dt_buf[flat] = fmin(dtx, dty);
}

// ============================================================
// Cell-centred div·B diagnostic (§A5 check).
//   (∇·B)^n_{i,j} = (B_x^{i+½} − B_x^{i−½})/Δx
//                 + (B_y^{j+½} − B_y^{j−½})/Δy
// Must remain at machine precision under CT.
// ============================================================
__global__ void k_athmhd_divB_cc(
    const double* Bxf, const double* Byf,
    double* divB_cc,
    int nx, int ny, int ng, int sy,
    double dx, double dy)
{
    int syp1 = sy + 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x + ng;
    int j = blockIdx.y * blockDim.y + threadIdx.y + ng;
    if (i >= ng + nx || j >= ng + ny) return;
    double dBx = Bxf[fx_flat(i + 1, j, sy)]
               - Bxf[fx_flat(i,     j, sy)];
    double dBy = Byf[fy_flat(i, j + 1, syp1)]
               - Byf[fy_flat(i, j,     syp1)];
    divB_cc[cflat(i, j, sy)] = dBx / dx + dBy / dy;
}

// ============================================================
// §C6 Spitzer anisotropic conduction kernels.
//
// Step 1 — k_athmhd_compute_T: cell-centred T = P/ρ (code units, μ=1).
//          Written to d_T_cc buffer; used by flux kernels.
// Step 2 — k_athmhd_conduction_flux_x / _y: face-centred F_c normal
//          component  F_n = -κ₀ T^(5/2) (b̂·∇T) b̂_n
//          with b̂ = B/|B| face-averaged from neighbouring cells, and
//          ∇T reconstructed by centred difference + transverse 4-point
//          average of the tangential component.
// Step 3 — k_athmhd_apply_conduction: E -= dt · ∇·F_c (the divergence
//          of the face-flux array, same formula as hydro flux update).
//
// The CFL bound (§C6-CFL) is computed on the fly:
//   Δt_cond = ½ · min_cell [ ρ·c_v·min(Δx,Δy)² / κ_∥(T) ]
// where κ_∥(T) = κ₀ T^{5/2}.
// ============================================================

__global__ void k_athmhd_compute_T(
    const double* __restrict__ w_rho,
    const double* __restrict__ w_P,
    double* __restrict__ T_cc,
    int sx, int sy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= sx || j >= sy) return;
    int c = i * sy + j;
    double r = fmax(w_rho[c], 1e-30);
    T_cc[c] = w_P[c] / r;   // code units (μ=1, k_B=1)
}

// x-face: flux F_n = F·x̂ = -κ∥(T_f) (b̂·∇T)_f · b̂_x
__global__ void k_athmhd_conduction_flux_x(
    const double* __restrict__ T_cc,
    const double* __restrict__ w_Bx,   // B_x_cc (recomputed in cons_to_prim)
    const double* __restrict__ w_By,
    const double* __restrict__ w_Bz,
    double* __restrict__ Fx_cond,
    int nx, int ny, int ng, int sx, int sy,
    double dx, double dy, double kappa0)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + ng;
    int j = blockIdx.y * blockDim.y + threadIdx.y + ng;
    if (i > ng + nx || j >= ng + ny) return;   // i+½ faces: i ∈ [ng, ng+nx]
    int fx = i * sy + j;
    int cL = (i - 1) * sy + j;
    int cR = i * sy + j;
    // Face-averaged B (arithmetic mean of left/right cell-centred B).
    double Bx_f = 0.5 * (w_Bx[cL] + w_Bx[cR]);
    double By_f = 0.5 * (w_By[cL] + w_By[cR]);
    double Bz_f = 0.5 * (w_Bz[cL] + w_Bz[cR]);
    double Bmag = sqrt(Bx_f*Bx_f + By_f*By_f + Bz_f*Bz_f);
    if (Bmag < 1e-30) { Fx_cond[fx] = 0.0; return; }
    double bx = Bx_f / Bmag;
    double by = By_f / Bmag;
    // ∂T/∂x at face: centred difference.
    double dTdx = (T_cc[cR] - T_cc[cL]) / dx;
    // ∂T/∂y at face: 4-point transverse average (upper/lower, left/right).
    int cLu = cL + 1, cLd = cL - 1;    // (i-1, j±1)
    int cRu = cR + 1, cRd = cR - 1;    // (i,   j±1)
    double dTdy_L = (T_cc[cLu] - T_cc[cLd]) * 0.5 / dy;
    double dTdy_R = (T_cc[cRu] - T_cc[cRd]) * 0.5 / dy;
    double dTdy_f = 0.5 * (dTdy_L + dTdy_R);
    double bdotgradT = bx * dTdx + by * dTdy_f;
    // Face-centred T: arithmetic avg (for T^{5/2}).
    double T_f = 0.5 * (T_cc[cL] + T_cc[cR]);
    double kappa_par = kappa0 * pow(fmax(T_f, 1e-30), 2.5);
    Fx_cond[fx] = -kappa_par * bdotgradT * bx;
}

__global__ void k_athmhd_conduction_flux_y(
    const double* __restrict__ T_cc,
    const double* __restrict__ w_Bx,
    const double* __restrict__ w_By,
    const double* __restrict__ w_Bz,
    double* __restrict__ Gy_cond,
    int nx, int ny, int ng, int sx, int sy,
    double dx, double dy, double kappa0)
{
    int syp1 = sy + 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x + ng;
    int j = blockIdx.y * blockDim.y + threadIdx.y + ng;
    if (i >= ng + nx || j > ng + ny) return;   // j+½ faces
    int fy = i * syp1 + j;
    int cB = i * sy + (j - 1);
    int cT = i * sy + j;
    double Bx_f = 0.5 * (w_Bx[cB] + w_Bx[cT]);
    double By_f = 0.5 * (w_By[cB] + w_By[cT]);
    double Bz_f = 0.5 * (w_Bz[cB] + w_Bz[cT]);
    double Bmag = sqrt(Bx_f*Bx_f + By_f*By_f + Bz_f*Bz_f);
    if (Bmag < 1e-30) { Gy_cond[fy] = 0.0; return; }
    double bx = Bx_f / Bmag;
    double by = By_f / Bmag;
    // ∂T/∂y at face: centred.
    double dTdy = (T_cc[cT] - T_cc[cB]) / dy;
    // ∂T/∂x at face: 4-point transverse average.
    int cBl = cB - sy, cBr = cB + sy;     // (i±1, j-1)
    int cTl = cT - sy, cTr = cT + sy;     // (i±1, j)
    double dTdx_B = (T_cc[cBr] - T_cc[cBl]) * 0.5 / dx;
    double dTdx_T = (T_cc[cTr] - T_cc[cTl]) * 0.5 / dx;
    double dTdx_f = 0.5 * (dTdx_B + dTdx_T);
    double bdotgradT = bx * dTdx_f + by * dTdy;
    double T_f = 0.5 * (T_cc[cB] + T_cc[cT]);
    double kappa_par = kappa0 * pow(fmax(T_f, 1e-30), 2.5);
    Gy_cond[fy] = -kappa_par * bdotgradT * by;
}

// Apply -∇·F_c · dt to E: E[c] -= dt · [(Fx[i+1]-Fx[i])/dx + (Gy[j+1]-Gy[j])/dy]
__global__ void k_athmhd_apply_conduction(
    double* __restrict__ E,
    const double* __restrict__ Fx_cond,
    const double* __restrict__ Gy_cond,
    int nx, int ny, int ng, int sx, int sy,
    double dx, double dy, double dt)
{
    int syp1 = sy + 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x + ng;
    int j = blockIdx.y * blockDim.y + threadIdx.y + ng;
    if (i >= ng + nx || j >= ng + ny) return;
    int c = i * sy + j;
    double dFx = Fx_cond[(i + 1) * sy + j] - Fx_cond[i * sy + j];
    double dGy = Gy_cond[i * syp1 + (j + 1)] - Gy_cond[i * syp1 + j];
    E[c] -= dt * (dFx / dx + dGy / dy);
}

// CFL buffer: per-cell Δt_cond = ½·ρ·c_v·min(Δx,Δy)²/κ_∥(T)
// with c_v = 1/(γ-1) in code units (k_B=μ=1).
__global__ void k_athmhd_conduction_cfl(
    const double* __restrict__ w_rho,
    const double* __restrict__ T_cc,
    double* __restrict__ dt_buf,
    int nx, int ny, int ng, int sy,
    double dx, double dy, double kappa0, double gm1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + ng;
    int j = blockIdx.y * blockDim.y + threadIdx.y + ng;
    if (i >= ng + nx || j >= ng + ny) return;
    int c = i * sy + j;
    int cell = (i - ng) * ny + (j - ng);
    double T = fmax(T_cc[c], 1e-30);
    double kpar = kappa0 * pow(T, 2.5);
    if (kpar < 1e-30) { dt_buf[cell] = 1e30; return; }
    double hmin = fmin(dx, dy);
    double cv = 1.0 / gm1;
    dt_buf[cell] = 0.5 * w_rho[c] * cv * hmin * hmin / kpar;
}

// ============================================================
// §C7 Townsend 2009 closed-form optically-thin cooling.
//   Single-segment power-law Λ(T) = Λ₀ (T/T_ref)^α.
//   ODE:  ρ c_v dT/dt = -ρ² Λ / (μ_e m_u)²  → in code units
//         dT/dt = -(γ-1) ρ Λ = -C · T^α  with
//           C = (γ-1) · ρ · Λ₀ / T_ref^α
//   Closed form (α ≠ 1): T = [ T₀^{1-α} - C(1-α) dt ]^{1/(1-α)}
//   α = 1 degenerate:    T = T₀ · exp(-C dt)
//   Floor T at Tfloor (prevents T → 0 on runaway cooling).
//   Only the thermal energy is updated:
//       ΔE = ρ c_v (T_new - T_old)          (T_new ≤ T_old always)
// ============================================================
__global__ void k_athmhd_cool_townsend(
    const double* __restrict__ w_rho,
    const double* __restrict__ w_P,
    double* __restrict__ E,
    int nx, int ny, int ng, int sy,
    double dt, double gm1,
    double Lambda0, double Tref, double alpha, double Tfloor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + ng;
    int j = blockIdx.y * blockDim.y + threadIdx.y + ng;
    if (i >= ng + nx || j >= ng + ny) return;
    int c = i * sy + j;
    double rho = fmax(w_rho[c], 1e-30);
    double T0  = fmax(w_P[c] / rho, Tfloor);
    // dT/dt = -gm1·ρ·Λ(T) = -C · T^α,  C = gm1·ρ·Λ₀/Tref^α
    double C = gm1 * rho * Lambda0 * pow(Tref, -alpha);
    double Tnew;
    const double one_minus_a = 1.0 - alpha;
    if (fabs(one_minus_a) < 1e-12) {
        Tnew = T0 * exp(-C * dt);
    } else {
        double base = pow(T0, one_minus_a) - C * one_minus_a * dt;
        if (base <= 0.0) {
            Tnew = Tfloor;
        } else {
            Tnew = pow(base, 1.0 / one_minus_a);
        }
    }
    if (Tnew < Tfloor) Tnew = Tfloor;
    if (Tnew > T0)     Tnew = T0;
    double cv = 1.0 / gm1;
    double dE = rho * cv * (Tnew - T0);
    E[c] += dE;
}
