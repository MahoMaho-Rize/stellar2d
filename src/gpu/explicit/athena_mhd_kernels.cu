// ============================================================
// athena_mhd_kernels.cu — CUDA kernels for the AthenaMHDSolver
//
// Derivation dossier:  docs/derivations/mhd/manuscript.pdf
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

// T is a scalar — ghost fill is a straight mirror for any BC (reflect
// / periodic / outflow).  This kernel OVERWRITES the ghost layer of T
// after compute_T.  Needed because compute_T uses ghost ρ and B_cc,
// which under reflect-face-B give a DIFFERENT p = (γ-1)(E - KE - ME)
// than the interior neighbour (normal-B flips on face reflection,
// cell-centred B at the ghost row ends up ≠ mirrored interior B_cc).
// That mismatch spuriously makes T jump at the wall and the Spitzer
// flux becomes nonzero even on a ∇T=0 isothermal atmosphere.
__global__ void k_athmhd_ghost_T_y_reflect(
    double* T_cc, int ny, int ng, int sx, int sy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= sx || g >= ng) return;
    int jBd = ng - 1 - g, jBs = ng + g;
    int jTd = ng + ny + g, jTs = ng + ny - 1 - g;
    T_cc[cflat(i, jBd, sy)] = T_cc[cflat(i, jBs, sy)];
    T_cc[cflat(i, jTd, sy)] = T_cc[cflat(i, jTs, sy)];
}

__global__ void k_athmhd_ghost_T_y_periodic(
    double* T_cc, int ny, int ng, int sx, int sy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= sx || g >= ng) return;
    int jBd = g, jBs = ny + g;
    int jTd = ng + ny + g, jTs = ng + g;
    T_cc[cflat(i, jBd, sy)] = T_cc[cflat(i, jBs, sy)];
    T_cc[cflat(i, jTd, sy)] = T_cc[cflat(i, jTs, sy)];
}

__global__ void k_athmhd_ghost_T_y_outflow(
    double* T_cc, int ny, int ng, int sx, int sy)
{
    // Zero-gradient extrapolation (same as ρ outflow ghost).
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= sx || g >= ng) return;
    int jBd = ng - 1 - g, jBs = ng;
    int jTd = ng + ny + g, jTs = ng + ny - 1;
    T_cc[cflat(i, jBd, sy)] = T_cc[cflat(i, jBs, sy)];
    T_cc[cflat(i, jTd, sy)] = T_cc[cflat(i, jTs, sy)];
}

__global__ void k_athmhd_ghost_T_x_periodic(
    double* T_cc, int nx, int ng, int sx, int sy)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= sy || g >= ng) return;
    int iLd = g, iLs = nx + g;
    int iRd = ng + nx + g, iRs = ng + g;
    T_cc[cflat(iLd, j, sy)] = T_cc[cflat(iLs, j, sy)];
    T_cc[cflat(iRd, j, sy)] = T_cc[cflat(iRs, j, sy)];
}

__global__ void k_athmhd_ghost_T_x_outflow(
    double* T_cc, int nx, int ng, int sx, int sy)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= sy || g >= ng) return;
    int iLd = ng - 1 - g, iLs = ng;
    int iRd = ng + nx + g, iRs = ng + nx - 1;
    T_cc[cflat(iLd, j, sy)] = T_cc[cflat(iLs, j, sy)];
    T_cc[cflat(iRd, j, sy)] = T_cc[cflat(iRs, j, sy)];
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

// ============================================================
// §C8 chromospheric blended cooling (Shimizu+22 / Suzuki+25).
//   Q_R = ξ Q_thck + (1-ξ) Q_thin,  ξ = max(0, 1 - p_chr/p)
//   thick: Newton relaxation ∂T/∂t = -(T - T_ref_thck)/τ_thck
//   thin:  Townsend §C7 power-law Λ(T) = Λ₀ (T/T_ref_thin)^α
//   Operator-split per cell: thick exponential step first (weighted
//   by ξ in the exponent), then Townsend closed-form thin step on
//   the result (thin rate scaled by (1-ξ) via the C coefficient).
//   O(dt²) splitting error; matches B-M3 single-segment closed form
//   in each pure limit.  Only thermal energy touched:
//     ΔE = ρ c_v (T_new - T_old)         (T_new ≤ T_old always)
// ============================================================
__global__ void k_athmhd_cool_chromo(
    const double* __restrict__ w_rho,
    const double* __restrict__ w_P,
    double* __restrict__ E,
    int nx, int ny, int ng, int sy,
    double dt, double gm1,
    double p_chr, double T_ref_thck, double tau_thck,
    double Lambda0, double T_ref_thin, double alpha,
    double Tfloor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + ng;
    int j = blockIdx.y * blockDim.y + threadIdx.y + ng;
    if (i >= ng + nx || j >= ng + ny) return;
    int c = i * sy + j;
    double rho = fmax(w_rho[c], 1e-30);
    double p   = fmax(w_P[c], 0.0);
    double T0  = fmax(p / rho, Tfloor);
    // Blend weight.
    double xi;
    if (p <= p_chr) {
        xi = 0.0;
    } else {
        xi = 1.0 - p_chr / p;
    }
    // Thick step: T_a = T_ref + (T0 - T_ref) · exp(-xi · dt / τ).
    double T_a;
    if (tau_thck > 0.0) {
        T_a = T_ref_thck + (T0 - T_ref_thck) * exp(-xi * dt / tau_thck);
    } else {
        T_a = T0;
    }
    // Thin step on T_a with weight (1-xi) baked into C.
    double one_minus_xi = 1.0 - xi;
    double T_b;
    if (one_minus_xi > 0.0) {
        double C = one_minus_xi * gm1 * rho * Lambda0 * pow(T_ref_thin, -alpha);
        const double one_minus_a = 1.0 - alpha;
        if (fabs(one_minus_a) < 1e-12) {
            T_b = T_a * exp(-C * dt);
        } else {
            double base = pow(T_a, one_minus_a) - C * one_minus_a * dt;
            if (base <= 0.0) {
                T_b = Tfloor;
            } else {
                T_b = pow(base, 1.0 / one_minus_a);
            }
        }
    } else {
        T_b = T_a;
    }
    if (T_b < Tfloor) T_b = Tfloor;
    if (T_b > T0)     T_b = T0;     // cooling only
    double cv = 1.0 / gm1;
    double dE = rho * cv * (T_b - T0);
    E[c] += dE;
}

// ============================================================
// §E2 Characteristic inner BC for 2D MHD (replaces interior-SET driver).
//
// Ghost-fill formulas (derived sympy-verified in §E2):
//   Let  v_x^drv(t) = Σ_N A_N · sin(2π f_N t + φ_N)      (§E1 waveform)
//   Alfvén Riemann invariants (B_0 = B_y0 ŷ):
//       z̃⁺ = -v_x + B_x/√ρ_0  propagates at +v_A (incoming into domain)
//       z̃⁻ = +v_x + B_x/√ρ_0  propagates at −v_A (outgoing to bottom)
//   BC:  z̃⁺|ghost = −2 v_x^drv(t)  (prescribed incoming)
//        z̃⁻|ghost = z̃⁻|interior    (absorbing outgoing)
//   Invert:
//       v_x|ghost = v_x^drv + ½(v_x^int + B_x^int/√ρ_0)
//       B_x|ghost = √ρ_0 · [−v_x^drv + ½(v_x^int + B_x^int/√ρ_0)]
//
// Applied only at bottom (j < ng) in a y-reflect run.  Top boundary
// stays reflective (handled by k_athmhd_ghost_y_reflect_*).  Other
// fields (ρ, v_y, v_z, B_z, P for E) mirror the reflective recipe so
// HSE is preserved.  E is reconstructed from the new (v_x, B_x, v_y,
// v_z, B_z, ρ, P) on the ghost.
//
// Interior side of the Alfvén invariant uses the **first interior row**
// (j = ng), which is the closest neighbour to the ghost for Riemann
// solving — same location the old interior-SET kernel used to overwrite.
//
// z-polarised Alfvén channel: same formula with v_z^drv = 0.
// ============================================================
__global__ void k_athmhd_ghost_y_characteristic_cc(
    double* __restrict__ rho,
    double* __restrict__ mx,
    double* __restrict__ my,
    double* __restrict__ mz,
    double* __restrict__ E,
    double* __restrict__ Bx_cc,
    double* __restrict__ By_cc,
    double* __restrict__ Bz_cc,
    const double* __restrict__ d_f,
    const double* __restrict__ d_amp,
    const double* __restrict__ d_phi,
    int N_modes, double t_now,
    int nx, int ny, int ng, int sx, int sy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= sx || g >= ng) return;

    // Driver waveform v_x^drv (shared across all x, as per §E1 simplification
    // "1D-photospheric-driver-in-2D").  Re-evaluated per thread; fine since
    // N_modes is small (≤ 256 in practice).
    double two_pi = 6.283185307179586;
    double vx_drv = 0.0;
    for (int n = 0; n < N_modes; ++n) {
        vx_drv += d_amp[n] * sin(two_pi * d_f[n] * t_now + d_phi[n]);
    }

    // ---- Bottom ghost row g ∈ [0, ng), jBd = ng - 1 - g ----
    // Mirror-interior row for BOTH the Alfvén invariant and the
    // non-Alfvén reflective fields: jBs = ng + g.  Using j=ng for all
    // ghost layers flattened the PLM slope in the y-reconstruction and
    // produced an O(A_rms²) spurious mass flux at the j=ng-½ face.
    int jBd = ng - 1 - g;
    int jBs = ng + g;                       // mirror interior row
    int cBd = cflat(i, jBd, sy);
    int cBint = cflat(i, jBs, sy);          // interior reference follows g

    // Read interior primitives needed for z̃⁻ and polarisation.
    double r_int   = fmax(rho[cBint], 1e-30);
    double vx_int  = mx[cBint] / r_int;
    double Bx_int  = Bx_cc[cBint];

    // Characteristic ghost v_x and B_x.  Uses local ρ_int as √ρ_0 reference —
    // consistent with the linearisation used in §E2 (locally constant ρ_0).
    double sqrt_r0 = sqrt(r_int);
    double half_zm = 0.5 * (vx_int + Bx_int / sqrt_r0);
    double vx_gh   = vx_drv + half_zm;
    double Bx_gh   = sqrt_r0 * (-vx_drv + half_zm);

    // Non-Alfvén fields: reflect mirror from the symmetric interior cell
    //   j = ng + g  (standard reflective-y recipe for ρ, tangential mom,
    //   tangential B; v_y antisymmetric).  jBs already computed above.
    int cBs = cflat(i, jBs, sy);
    double r_gh  = rho[cBs];                 // symmetric ρ (HSE)
    double my_gh = -my[cBs];                 // antisymmetric v_y
    double mz_gh =  mz[cBs];                 // symmetric v_z (z-Alfvén does
                                             //   not currently have a driver,
                                             //   but a pure-absorber fill is
                                             //   needed for correctness; see
                                             //   block below)
    double Bz_gh =  Bz_cc[cBs];              // tangential B, symmetric
    double By_gh =  By_cc[cBs];              // not used for Riemann but keep
                                             //   cc consistent with face

    // Apply the characteristic formula to the z-polarised Alfvén channel
    //   as a pure absorber (v_z^drv = 0):
    //   v_z|ghost = ½(v_z^int + B_z^int/√ρ_0)
    //   B_z|ghost = √ρ_0 · ½(v_z^int + B_z^int/√ρ_0)
    // Only do this if the z-channel is non-trivial (|v_z|, |B_z| > tiny),
    // otherwise the reflect-mirror path already gives 0.  This branchless
    // version always computes both and blends is OK since the absorber
    // formula reduces to 0 in the quiescent z limit anyway.  To keep the
    // implementation simple we always apply it.
    // z-polarised Alfvén: same linear-extrapolation fix.  v_z^drv = 0.
    double vz_i0   = mz[cBint0] / r_int0;
    double Bz_i0   = Bz_cc[cBint0];
    double vz_i1   = mz[cBint1] / r_int1;
    double Bz_i1   = Bz_cc[cBint1];
    double zmz_i0  = vz_i0 + Bz_i0 / sqrt_r0;
    double zmz_i1  = vz_i1 + Bz_i1 / sqrt_r0;
    double zmz_extrap = (double)(g + 2) * zmz_i0
                      - (double)(g + 1) * zmz_i1;
    double half_zmz = 0.5 * zmz_extrap;
    double vz_gh_ch = half_zmz;
    double Bz_gh_ch = sqrt_r0 * half_zmz;

    // Write ghost primitives back as conservatives.  E ghost is
    // reconstructed from (ρ, v, B_cc, P) with P taken as the symmetric
    // mirror's P (HSE pressure), per reflective-y thermodynamics.
    //
    // Extract P_mirror from E[cBs] with the MIRROR v and B (same as what
    // reflect_cc would have produced before our override).  This is
    // bit-identical to reading E[cBs] and rewriting KE/ME with the new
    // ghost fields.
    //
    // E = P/(γ-1) + ½ρ|v|² + ½|B|²   (gas + KE + ME)
    // Compute P_mirror = P at the symmetric interior cell from its E.
    // We don't have γ in the kernel signature; extract via a secondary
    // expression: P_mirror = E[cBs] - KE_mirror - ME_mirror, multiplied
    // by (γ-1).  Actually P_mirror = (γ-1)*(E - KE - ME), so we can
    // write back:
    //   E_gh = (γ-1)·P_mirror/(γ-1) + KE_gh + ME_gh
    //        = (E[cBs] - KE_mirror - ME_mirror) + KE_gh + ME_gh
    // (the (γ-1) cancels).  This keeps P_ghost = P_mirror = HSE pressure.
    double r_mirror  = rho[cBs];
    double vx_mirror = mx[cBs] / fmax(r_mirror, 1e-30);
    double vy_mirror = -my[cBs] / fmax(r_mirror, 1e-30);   // antisym
    double vz_mirror = mz[cBs] / fmax(r_mirror, 1e-30);
    double KE_mirror = 0.5 * r_mirror * (vx_mirror*vx_mirror
                                        + vy_mirror*vy_mirror
                                        + vz_mirror*vz_mirror);
    double Bx_mirror = Bx_cc[cBs];
    double By_mirror = By_cc[cBs];
    double Bz_mirror = Bz_cc[cBs];
    double ME_mirror = 0.5 * (Bx_mirror*Bx_mirror
                             + By_mirror*By_mirror
                             + Bz_mirror*Bz_mirror);
    double P_over_gm1 = E[cBs] - KE_mirror - ME_mirror;   // = P/(γ-1)

    // Ghost KE + ME with characteristic (v_x, B_x) and pure-absorber (v_z, B_z).
    double vy_gh_val = my_gh / fmax(r_gh, 1e-30);
    double KE_gh = 0.5 * r_gh * (vx_gh*vx_gh
                                + vy_gh_val*vy_gh_val
                                + vz_gh_ch*vz_gh_ch);
    double ME_gh = 0.5 * (Bx_gh*Bx_gh
                         + By_gh*By_gh
                         + Bz_gh_ch*Bz_gh_ch);

    rho [cBd] = r_gh;
    mx  [cBd] = r_gh * vx_gh;
    my  [cBd] = my_gh;
    mz  [cBd] = r_gh * vz_gh_ch;
    E   [cBd] = P_over_gm1 + KE_gh + ME_gh;
    Bx_cc[cBd] = Bx_gh;
    By_cc[cBd] = By_gh;                     // unchanged from mirror
    Bz_cc[cBd] = Bz_gh_ch;
    (void)mz_gh; (void)Bz_gh;               // absorber values override mirror

    // ---- Top ghost row: still reflective (unchanged) ----
    // Handled separately by k_athmhd_ghost_y_reflect_cc on top, but
    // since that kernel also fills the bottom, we must NOT double-fill.
    // Plan: fill_ghost host dispatch calls reflect_cc for the TOP only
    // when driver_on (by re-using reflect_cc but on a top-only path),
    // OR calls a split "top-reflect" kernel.  For simplicity we'll do
    // the full top-reflect here as well:
    int jTd = ng + ny + g;
    int jTs = ng + ny - 1 - g;
    int cTd = cflat(i, jTd, sy);
    int cTs = cflat(i, jTs, sy);
    rho[cTd] =  rho[cTs]; mx[cTd] =  mx[cTs]; my[cTd] = -my[cTs];
    mz [cTd] =  mz [cTs]; E [cTd] =  E [cTs];
    Bz_cc[cTd] = Bz_cc[cTs];
    Bx_cc[cTd] = Bx_cc[cTs];
    By_cc[cTd] = By_cc[cTs];
}

// Face-B characteristic fill for the bottom ghost row(s).  Top ghost
// is done by the reflect_face kernel (the caller invokes both).
//
// Key identity (§E2):  B_x^face,i±½,j_g = B_x^cc,ghost  gives
//   ½(B_x^face,i-½ + B_x^face,i+½) = B_x^cc,ghost  (exact).
// B_y face at j = ng (the wall) is kept by the reflect rule; the
// ghost B_y face at j = ng - 1 - g mirrors antisymmetrically from
// j = ng + 1 + g — same as k_athmhd_ghost_y_reflect_face.
__global__ void k_athmhd_ghost_y_characteristic_face(
    double* __restrict__ Bxf,
    double* __restrict__ Byf,
    const double* __restrict__ rho,
    const double* __restrict__ mx,
    const double* __restrict__ Bx_cc,
    const double* __restrict__ d_f,
    const double* __restrict__ d_amp,
    const double* __restrict__ d_phi,
    int N_modes, double t_now,
    int nx, int ny, int ng, int sy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    int syp1 = sy + 1;
    int syfx = sy;
    if (g >= ng) return;

    // Driver waveform (same as cc kernel).
    double two_pi = 6.283185307179586;
    double vx_drv = 0.0;
    for (int n = 0; n < N_modes; ++n) {
        vx_drv += d_amp[n] * sin(two_pi * d_f[n] * t_now + d_phi[n]);
    }

    // --- Byf antisymmetric mirror (bottom + top), same as reflect_face ---
    if (i < nx + 2 * ng) {
        int jBd = ng - 1 - g;
        int jBs = ng + 1 + g;
        int jTd = ng + ny + 1 + g;
        int jTs = ng + ny - 1 - g;
        Byf[fy_flat(i, jBd, syp1)] = -Byf[fy_flat(i, jBs, syp1)];
        Byf[fy_flat(i, jTd, syp1)] = -Byf[fy_flat(i, jTs, syp1)];
    }

    // --- Bxf characteristic bottom, symmetric top ---
    if (i < nx + 1 + 2 * ng) {
        int jBd = ng - 1 - g;
        int jBs = ng + g;
        int jTd = ng + ny + g;
        int jTs = ng + ny - 1 - g;
        // Top: symmetric mirror (unchanged)
        Bxf[fx_flat(i, jTd, syfx)] = Bxf[fx_flat(i, jTs, syfx)];
        // Bottom: characteristic.  The interior-side reference row is
        // the mirror row j = ng + g, matching the cell-centred kernel.
        // (A previous version used j=ng for all ghost layers, which
        // flattened the y-slope seen by PLM reconstruction and produced
        // an O(A_rms²) spurious mass flux at j = ng - ½.)
        int ic_ref = i;
        if (ic_ref >= nx + 2 * ng) ic_ref = nx + 2 * ng - 1;
        // §E2-v2: linear extrapolation of z^- from the two lowest
        // interior rows (j=ng, j=ng+1) for the same reason as the cc
        // kernel — avoids the O(kΔy) phase-slip reflection that the
        // nearest-mirror gave (empirically ~22% evanescent excess at
        // the bottom cell, documented in e1_t7_analytic_on_mesh_decomposition.md).
        int cRef0 = cflat(ic_ref, ng,     sy);
        int cRef1 = cflat(ic_ref, ng + 1, sy);
        double r_int0 = fmax(rho[cRef0], 1e-30);
        double sqrt_r0 = sqrt(r_int0);
        double zm_i0  = mx[cRef0] / r_int0 + Bx_cc[cRef0] / sqrt_r0;
        double r_int1 = fmax(rho[cRef1], 1e-30);
        double zm_i1  = mx[cRef1] / r_int1 + Bx_cc[cRef1] / sqrt_r0;
        double zm_extrap = (double)(g + 2) * zm_i0
                         - (double)(g + 1) * zm_i1;
        double half_zm = 0.5 * zm_extrap;
        double Bx_gh   = sqrt_r0 * (-vx_drv + half_zm);
        Bxf[fx_flat(i, jBd, syfx)] = Bx_gh;
        (void)jBs;
    }
}

// ============================================================
// Top-only outflow — used in combination with §E2 bottom driver +
// §E4 PML sponge.  Writes zero-gradient extrapolation on the top ghost
// rows only; bottom ghost is left untouched (filled by §E2).
// ============================================================
__global__ void k_athmhd_ghost_y_top_outflow_cc(
    double* rho, double* mx, double* my, double* mz, double* E,
    double* Bx_cc, double* By_cc, double* Bz_cc,
    int nx, int ny, int ng, int sx, int sy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= sx || g >= ng) return;
    int jTd = ng + ny + g;
    int jTs = ng + ny - 1;
    int cTd = cflat(i, jTd, sy), cTs = cflat(i, jTs, sy);
    rho  [cTd] = rho  [cTs];
    mx   [cTd] = mx   [cTs];
    my   [cTd] = my   [cTs];
    mz   [cTd] = mz   [cTs];
    E    [cTd] = E    [cTs];
    Bx_cc[cTd] = Bx_cc[cTs];
    By_cc[cTd] = By_cc[cTs];
    Bz_cc[cTd] = Bz_cc[cTs];
}

__global__ void k_athmhd_ghost_y_top_outflow_face(
    double* Bxf, double* Byf,
    int nx, int ny, int ng, int sy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    int syp1 = sy + 1;
    int syfx = sy;
    if (g >= ng) return;
    // Byf top: zero-gradient from the last interior face j = ng + ny.
    if (i < nx + 2 * ng) {
        int jTd = ng + ny + 1 + g;
        int jTs = ng + ny;
        Byf[fy_flat(i, jTd, syp1)] = Byf[fy_flat(i, jTs, syp1)];
    }
    // Bxf top: zero-gradient from the last interior row.
    if (i < nx + 1 + 2 * ng) {
        int jTd = ng + ny + g;
        int jTs = ng + ny - 1;
        Bxf[fx_flat(i, jTd, syfx)] = Bxf[fx_flat(i, jTs, syfx)];
    }
}

// ============================================================
// §E3 — Top outgoing characteristic BC (cell-centred).
//
// Bottom is handled by `k_athmhd_ghost_y_characteristic_cc` (§E2).  This
// kernel overrides ONLY the top ghost rows so the full ghost-fill
// pipeline is: bottom via §E2 characteristic OR reflect_cc (depending
// on driver_on), then top via §E3 outgoing OR reflect (depending on
// top_outgoing).  Called from fill_ghost when top_outgoing == true.
//
// Algebra (see docs/derivations/mhd/sections/e3_top_outgoing_bc.md):
//   Interior reference row (mirror-index convention) jTs = ng + ny - 1 - g.
//   z^+|_{int}   = -v_x^{int} + B_x^{int}/√ρ0       (outgoing)
//   z^-|_{ghost} = 0                                 (no incoming)
//   ⇒  v_x|_{top_ghost} = -(z^+_{int})/2 = ½(v_x^{int} - B_x^{int}/√ρ0)
//      B_x|_{top_ghost} = √ρ0 (z^+_{int})/2 = ½(-√ρ0 v_x^{int} + B_x^{int})
// The z-polarised Alfvén channel is absorbed identically by symmetry.
// Non-Alfvén fields (rho, m_y, E) use outflow / mirror rules — pressure
// and density take the interior mirror (HSE), v_y antisymmetric.
// ============================================================
__global__ void k_athmhd_ghost_y_top_outgoing_cc(
    double* __restrict__ rho,
    double* __restrict__ mx,
    double* __restrict__ my,
    double* __restrict__ mz,
    double* __restrict__ E,
    double* __restrict__ Bx_cc,
    double* __restrict__ By_cc,
    double* __restrict__ Bz_cc,
    int nx, int ny, int ng, int sx, int sy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= sx || g >= ng) return;

    int jTd = ng + ny + g;              // top ghost row (destination)
    int jTs = ng + ny - 1 - g;          // interior mirror reference
    int cTd = cflat(i, jTd, sy);
    int cTs = cflat(i, jTs, sy);

    double r_int = fmax(rho[cTs], 1e-30);
    double sqrt_r0 = sqrt(r_int);
    double vx_int = mx[cTs] / r_int;
    double vz_int = mz[cTs] / r_int;
    double Bx_int = Bx_cc[cTs];
    double Bz_int = Bz_cc[cTs];

    // §E3 ghost-fill closure — Alfvén-x channel
    double vx_gh = 0.5 * (vx_int - Bx_int / sqrt_r0);
    double Bx_gh = 0.5 * (-sqrt_r0 * vx_int + Bx_int);
    // z-polarised Alfvén (pure absorber, same formula)
    double vz_gh = 0.5 * (vz_int - Bz_int / sqrt_r0);
    double Bz_gh = 0.5 * (-sqrt_r0 * vz_int + Bz_int);

    // Non-Alfvén fields:
    //   ρ, By_cc       -- mirror (HSE, symmetric)
    //   v_y (my)       -- antisymmetric mirror (wall-normal)
    //   p (via E)      -- mirror P, recompute E = P/(γ-1) + KE_gh + ME_gh
    double r_gh  = rho[cTs];
    double By_gh = By_cc[cTs];
    double my_gh = -my[cTs];

    // Extract P_mirror via bit-identical trick (same pattern as §E2):
    //   E = P/(γ-1) + KE + ME.  Write E_gh = (E_mirror - KE_mirror - ME_mirror)
    //                                        + KE_gh + ME_gh.
    double r_mirror  = rho[cTs];
    double vx_mirror = mx[cTs] / fmax(r_mirror, 1e-30);
    double vy_mirror = -my[cTs] / fmax(r_mirror, 1e-30);
    double vz_mirror = mz[cTs] / fmax(r_mirror, 1e-30);
    double Bx_mirror = Bx_cc[cTs];
    double By_mirror = By_cc[cTs];
    double Bz_mirror = Bz_cc[cTs];
    double KE_mirror = 0.5 * r_mirror * (vx_mirror*vx_mirror
                                        + vy_mirror*vy_mirror
                                        + vz_mirror*vz_mirror);
    double ME_mirror = 0.5 * (Bx_mirror*Bx_mirror
                             + By_mirror*By_mirror
                             + Bz_mirror*Bz_mirror);
    double P_over_gm1 = E[cTs] - KE_mirror - ME_mirror;

    double vy_gh = my_gh / fmax(r_gh, 1e-30);
    double KE_gh = 0.5 * r_gh * (vx_gh*vx_gh + vy_gh*vy_gh + vz_gh*vz_gh);
    double ME_gh = 0.5 * (Bx_gh*Bx_gh + By_gh*By_gh + Bz_gh*Bz_gh);

    rho  [cTd] = r_gh;
    mx   [cTd] = r_gh * vx_gh;
    my   [cTd] = my_gh;
    mz   [cTd] = r_gh * vz_gh;
    E    [cTd] = P_over_gm1 + KE_gh + ME_gh;
    Bx_cc[cTd] = Bx_gh;
    By_cc[cTd] = By_gh;
    Bz_cc[cTd] = Bz_gh;
}

// ============================================================
// §E3.5 — Stone-1999 radiation BC (cell-centred) for top outgoing
// characteristic wave.  Applies the recursion
//   z^+|_{ghost,g+1} = 2 cos(kΔy) z^+|_{ghost,g} - z^+|_{ghost,g-1}
// seeded by the top two interior cells' z^+ values, then inverts to
// primitives via
//   v_x = -z^+/2,   B_x = (√ρ_0/2) z^+
// z^- is held identically zero (no incoming), and the z-polarised
// Alfvén channel is treated analogously.  Non-Alfvén fields use
// symmetric mirror (same as outflow).
//
// Seed: g = -1  →  interior row y_{top-1}:  jTs2 = ng + ny - 2
//       g =  0  →  interior row y_{top}:    jTs1 = ng + ny - 1
// Destination: g = 0 → jTd = ng + ny;   g = 1 → jTd = ng + ny + 1; …
// ============================================================
__global__ void k_athmhd_ghost_y_top_radiation_cc(
    double* __restrict__ rho,
    double* __restrict__ mx,
    double* __restrict__ my,
    double* __restrict__ mz,
    double* __restrict__ E,
    double* __restrict__ Bx_cc,
    double* __restrict__ By_cc,
    double* __restrict__ Bz_cc,
    double cos_kdy,        // 2cos(kΔy) = 2·cos(2π f Δy / v_A(y_top))
    int nx, int ny, int ng, int sx, int sy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sx) return;

    // Seed positions — top two interior cells.
    int jSeed1 = ng + ny - 1;       // top interior
    int jSeed2 = ng + ny - 2;       // one below
    int cS1 = cflat(i, jSeed1, sy);
    int cS2 = cflat(i, jSeed2, sy);

    double r_top = fmax(rho[cS1], 1e-30);
    double sqrt_r0 = sqrt(r_top);    // use ρ at the top interior cell

    // z^+ = -v_x + B_x/√ρ_0  from each seed cell.
    double vx_s1 = mx[cS1] / r_top;
    double Bx_s1 = Bx_cc[cS1];
    double zp_s1 = -vx_s1 + Bx_s1 / sqrt_r0;
    double vz_s1 = mz[cS1] / r_top;
    double Bz_s1 = Bz_cc[cS1];
    double zpZ_s1 = -vz_s1 + Bz_s1 / sqrt_r0;

    double r2 = fmax(rho[cS2], 1e-30);
    double vx_s2 = mx[cS2] / r2;
    double Bx_s2 = Bx_cc[cS2];
    double zp_s2 = -vx_s2 + Bx_s2 / sqrt(r2);
    double vz_s2 = mz[cS2] / r2;
    double Bz_s2 = Bz_cc[cS2];
    double zpZ_s2 = -vz_s2 + Bz_s2 / sqrt(r2);

    // Iterate recursion  z+_{g+1} = 2cos(kΔy) z+_g - z+_{g-1}  for
    // g = 0..ng-1.  Initial state:
    //   "prev" = z+_{n-1} = zp_s2
    //   "curr" = z+_n     = zp_s1
    // After one step we get z+_{n+1} (ghost layer 0 at jTd=ng+ny).
    double zp_prev  = zp_s2;
    double zp_curr  = zp_s1;
    double zpZ_prev = zpZ_s2;
    double zpZ_curr = zpZ_s1;

    // HSE mirror source for non-Alfvén fields: symmetric about wall, so
    // ghost layer g uses interior row jMirror = ng + ny - 1 - g.
    // P, ρ, By, v_y take the mirror value (v_y antisymmetric).
    for (int g = 0; g < ng; ++g) {
        double zp_next  = cos_kdy * zp_curr  - zp_prev;
        double zpZ_next = cos_kdy * zpZ_curr - zpZ_prev;

        int jTd = ng + ny + g;
        int jMirror = ng + ny - 1 - g;        // symmetric HSE mirror
        int cDst = cflat(i, jTd, sy);
        int cMir = cflat(i, jMirror, sy);

        // Alfvén fields: primitive closure  v_x = -z^+/2, B_x = (√ρ/2) z^+
        double vx_gh = -0.5 * zp_next;
        double Bx_gh =  0.5 * sqrt_r0 * zp_next;
        double vz_gh = -0.5 * zpZ_next;
        double Bz_gh =  0.5 * sqrt_r0 * zpZ_next;

        // Non-Alfvén:
        double r_mir  = rho[cMir];
        double By_gh  = By_cc[cMir];
        double my_gh  = -my[cMir];                 // antisymmetric v_y

        // Recompute E with ghost fields and mirror P.  E = P/(γ-1) + KE + ME.
        double r_mirror  = r_mir;
        double vx_mirror = mx[cMir] / fmax(r_mirror, 1e-30);
        double vy_mirror = -my[cMir] / fmax(r_mirror, 1e-30);
        double vz_mirror = mz[cMir] / fmax(r_mirror, 1e-30);
        double Bx_mirror = Bx_cc[cMir];
        double By_mirror = By_cc[cMir];
        double Bz_mirror = Bz_cc[cMir];
        double KE_mirror = 0.5 * r_mirror * (vx_mirror*vx_mirror
                                            + vy_mirror*vy_mirror
                                            + vz_mirror*vz_mirror);
        double ME_mirror = 0.5 * (Bx_mirror*Bx_mirror
                                 + By_mirror*By_mirror
                                 + Bz_mirror*Bz_mirror);
        double P_over_gm1 = E[cMir] - KE_mirror - ME_mirror;

        double vy_gh = my_gh / fmax(r_mir, 1e-30);
        double KE_gh = 0.5 * r_mir * (vx_gh*vx_gh + vy_gh*vy_gh + vz_gh*vz_gh);
        double ME_gh = 0.5 * (Bx_gh*Bx_gh + By_gh*By_gh + Bz_gh*Bz_gh);

        rho  [cDst] = r_mir;
        mx   [cDst] = r_mir * vx_gh;
        my   [cDst] = my_gh;
        mz   [cDst] = r_mir * vz_gh;
        E    [cDst] = P_over_gm1 + KE_gh + ME_gh;
        Bx_cc[cDst] = Bx_gh;
        By_cc[cDst] = By_gh;
        Bz_cc[cDst] = Bz_gh;

        // Advance recursion state
        zp_prev  = zp_curr;  zp_curr  = zp_next;
        zpZ_prev = zpZ_curr; zpZ_curr = zpZ_next;
    }
}

// ============================================================
// §E3.5 face-B: Bxf top ghost built from the same cc values.
// Byf top uses symmetric mirror (HSE).
// ============================================================
__global__ void k_athmhd_ghost_y_top_radiation_face(
    double* __restrict__ Bxf,
    double* __restrict__ Byf,
    const double* __restrict__ rho,
    const double* __restrict__ mx,
    const double* __restrict__ Bx_cc,
    double cos_kdy,
    int nx, int ny, int ng, int sy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx + 1 + 2 * ng) return;
    int syp1 = sy + 1;
    int syfx = sy;

    // Byf top: symmetric mirror (matches outflow / §E3).
    if (i < nx + 2 * ng) {
        for (int g = 0; g < ng; ++g) {
            int jTd = ng + ny + 1 + g;
            int jTs = ng + ny - g;
            Byf[fy_flat(i, jTd, syp1)] = Byf[fy_flat(i, jTs, syp1)];
        }
    }

    // Bxf top: apply radiation recursion to z^+ at x-faces, seeded from
    // the top two interior face values reconstructed from cc.
    int ic_ref = i;
    if (ic_ref >= nx + 2 * ng) ic_ref = nx + 2 * ng - 1;
    int jSeed1 = ng + ny - 1;
    int jSeed2 = ng + ny - 2;
    int cS1 = cflat(ic_ref, jSeed1, sy);
    int cS2 = cflat(ic_ref, jSeed2, sy);

    double r1 = fmax(rho[cS1], 1e-30);
    double r2 = fmax(rho[cS2], 1e-30);
    double sqrt_r0 = sqrt(r1);
    double vx1 = mx[cS1] / r1;
    double vx2 = mx[cS2] / r2;
    double Bx1 = Bx_cc[cS1];
    double Bx2 = Bx_cc[cS2];
    // Face B_x is x-face; for the top ghost row we use the §E3 identity
    // B_x^face,i±½ = B_x^cc,ghost.  So we just broadcast the radiation-BC
    // cc value to both faces of the ghost cell.
    double zp_s1 = -vx1 + Bx1 / sqrt_r0;
    double zp_s2 = -vx2 + Bx2 / sqrt(r2);

    double zp_prev = zp_s2;
    double zp_curr = zp_s1;
    for (int g = 0; g < ng; ++g) {
        double zp_next = cos_kdy * zp_curr - zp_prev;
        int jTd = ng + ny + g;
        double Bx_gh = 0.5 * sqrt_r0 * zp_next;
        Bxf[fx_flat(i, jTd, syfx)] = Bx_gh;
        zp_prev = zp_curr;
        zp_curr = zp_next;
    }
}

// ============================================================
// §E3 — Top outgoing characteristic BC (face-B).
//
// Byf on the top normal face uses the standard symmetric mirror (same
// as outflow-face and reflect-face for tangential → symmetric, but the
// normal wall face itself is not touched since it sits at j = ng + ny
// and is interior-adjacent; we only fill ghost faces beyond that).
// To stay consistent with the cc kernel we mirror Byf symmetrically.
//
// Bxf on the top gets §E3 characteristic cell-centred value, with both
// x-faces of the top ghost cell equal to the cc ghost value
// (average == cc, see §E3 Identity 6).
// ============================================================
__global__ void k_athmhd_ghost_y_top_outgoing_face(
    double* __restrict__ Bxf,
    double* __restrict__ Byf,
    const double* __restrict__ rho,
    const double* __restrict__ mx,
    const double* __restrict__ Bx_cc,
    int nx, int ny, int ng, int sy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    int syp1 = sy + 1;
    int syfx = sy;
    if (g >= ng) return;

    // Byf top: symmetric mirror (matches outflow, preserves ∇·B under
    // our zero-gradient assumption on the non-Alfvén channel).
    if (i < nx + 2 * ng) {
        int jTd = ng + ny + 1 + g;
        int jTs = ng + ny - g;          // symmetric interior face
        Byf[fy_flat(i, jTd, syp1)] = Byf[fy_flat(i, jTs, syp1)];
    }

    // Bxf top: characteristic (match cc rule).
    if (i < nx + 1 + 2 * ng) {
        int jTd = ng + ny + g;
        int jTs = ng + ny - 1 - g;      // interior reference row
        int ic_ref = i;
        if (ic_ref >= nx + 2 * ng) ic_ref = nx + 2 * ng - 1;
        int cRef = cflat(ic_ref, jTs, sy);
        double r_int = fmax(rho[cRef], 1e-30);
        double vx_int = mx[cRef] / r_int;
        double Bx_int = Bx_cc[cRef];
        double sqrt_r0 = sqrt(r_int);
        // B_x|_{top_ghost} = ½(-√ρ0 v_x^{int} + B_x^{int})
        double Bx_gh = 0.5 * (-sqrt_r0 * vx_int + Bx_int);
        Bxf[fx_flat(i, jTd, syfx)] = Bx_gh;
    }
}

// ============================================================
// §E4 PML absorbing sponge (cell-centred part).  Damps v_x, v_z, and
// Bz_cc via implicit-Euler on the Alfvén characteristic.  Bx_cc is
// NOT touched here — it's regenerated from Bxf by cons_to_prim each
// step, so damping Bx_cc directly would be overwritten.  The face
// Bxf damping is done by a separate face-side kernel (below).
//
// Closed-form implicit-Euler inverse (from sympy §E4 Identity 5):
//    Let τ = dt·σ, √ρ = sqrt(ρ_local).  Apply
//       v_x^new  = [(τ+2) v_x  +  (τ/√ρ) B_x_cc ] / (2(τ+1))
//       v_z^new  = [(τ+2) v_z  +  (τ/√ρ) B_z_cc ] / (2(τ+1))
// Bx_cc is taken as the READ value for the RHS but is not written
// back; only v_x and momentum/energy are modified here.  Bz_cc
// is damped because it's a pure cc storage (no face version in 2D).
// Energy is recomputed to match the new (v, B).
// ============================================================
__global__ void k_athmhd_apply_pml(
    double* __restrict__ rho,
    double* __restrict__ mx,
    double* __restrict__ my,
    double* __restrict__ mz,
    double* __restrict__ E,
    double* __restrict__ Bxf,
    double* __restrict__ Bx_cc,
    double* __restrict__ By_cc,
    double* __restrict__ Bz_cc,
    double dt,
    double y_start,    // lo-edge of PML
    double y_end,      // hi-edge (top of domain)
    double sigma0,     // peak damping rate at y_end
    double y_lo,       // domain y_lo
    double dy,
    int nx, int ny, int ng, int sx, int sy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + ng;
    int j = blockIdx.y * blockDim.y + threadIdx.y + ng;
    if (i >= ng + nx || j >= ng + ny) return;

    double yc = y_lo + ((j - ng) + 0.5) * dy;
    if (yc < y_start) return;   // outside absorber

    // Quadratic profile: σ(y) = σ_0 · ((y - y_start)/Δ)²
    double Delta = y_end - y_start;
    double s = (yc - y_start) / fmax(Delta, 1e-30);
    double sigma = sigma0 * s * s;
    double tau = dt * sigma;
    if (!(tau > 0.0)) return;

    int c = cflat(i, j, sy);
    double r = fmax(rho[c], 1e-30);
    double sqrt_r = sqrt(r);

    // Read current primitives + energy split.
    double vx = mx[c] / r;
    double vy = my[c] / r;
    double vz = mz[c] / r;
    double Bx = Bx_cc[c];
    double By = By_cc[c];
    double Bz = Bz_cc[c];

    double KE_old = 0.5 * r * (vx*vx + vy*vy + vz*vz);
    double ME_old = 0.5 * (Bx*Bx + By*By + Bz*Bz);
    double P_over_gm1 = E[c] - KE_old - ME_old;

    // Implicit-Euler PML — closed form (§E4 Identity 5 + 7).  Write NEW
    // B_x_cc with the SAME closed form the face kernel uses on B_xf.
    // For an x-uniform Alfvén wave the face average equals the cc
    // value exactly, so cons_to_prim's reconstruction B_x_cc =
    // ½(B_xf^L + B_xf^R) gives the same NEW value we write here.
    // Writing it explicitly is what lets the total-energy update below
    // use the NEW ME self-consistently — otherwise the damped magnetic
    // energy leaks into thermal pressure (Identity 7) and CFL dt
    // collapses.
    //
    // This kernel assumes the face kernel has ALREADY run (ordering
    // invariance, Identity 8) so Bxf is not needed here as input, only
    // as a consistency reference: the face kernel reads the OLD mx
    // (which is still OLD because this cc kernel has NOT yet written
    // to mx), and we read OLD Bx_cc (still intact because the face
    // kernel did not touch Bx_cc).  Both passes therefore see the OLD
    // state of the OTHER variable.
    double inv = 1.0 / (2.0 * (tau + 1.0));
    double vx_new = ((tau + 2.0) * vx + (tau / sqrt_r) * Bx) * inv;
    double vz_new = ((tau + 2.0) * vz + (tau / sqrt_r) * Bz) * inv;
    double Bx_new = ((tau * sqrt_r) * vx + (tau + 2.0) * Bx) * inv;
    double Bz_new = ((tau * sqrt_r) * vz + (tau + 2.0) * Bz) * inv;

    double KE_new = 0.5 * r * (vx_new*vx_new + vy*vy + vz_new*vz_new);
    double ME_new = 0.5 * (Bx_new*Bx_new + By*By + Bz_new*Bz_new);

    mx   [c] = r * vx_new;
    mz   [c] = r * vz_new;
    Bx_cc[c] = Bx_new;     // self-consistent with face-averaged Bxf update
    Bz_cc[c] = Bz_new;
    E    [c] = P_over_gm1 + KE_new + ME_new;
}

// ============================================================
// §E4 PML face-B damping (x-uniform wave assumption).
//
// On the face at (i+½, j) inside the PML region, apply the full
// implicit-Euler §E4 formula using the v_x of the LEFT cell (i, j)
// as the coupling reference (Alfvén waves in this setup are x-uniform
// so left and right cells have identical primitives).  Per face,
// Bxf^new = [(τ/√ρ) v_x + (τ+2) Bxf] / (2(τ+1)).
//
// This is NOT a pure scaling — it explicitly mixes v_x into Bxf,
// matching the characteristic invariant z^+ = -v_x + Bx/√ρ damping.
// Each face is touched by exactly one thread (no race).
// ============================================================
__global__ void k_athmhd_apply_pml_face(
    double* __restrict__ Bxf,
    const double* __restrict__ rho,
    const double* __restrict__ mx,
    double dt,
    double y_start,
    double y_end,
    double sigma0,
    double y_lo,
    double dy,
    int nx, int ny, int ng, int sx, int sy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;     // x-face index (0..nx+2ng)
    int j = blockIdx.y * blockDim.y + threadIdx.y + ng;
    if (i > nx + 2 * ng || j >= ng + ny) return;

    double yc = y_lo + ((j - ng) + 0.5) * dy;
    if (yc < y_start) return;

    double Delta = y_end - y_start;
    double s = (yc - y_start) / fmax(Delta, 1e-30);
    double sigma = sigma0 * s * s;
    double tau = dt * sigma;
    if (!(tau > 0.0)) return;

    // Use the LEFT cell (i-1, j) for rho, v_x reference.  At i=0 there's
    // no left cell — clamp to i=ng (interior) which is the rightmost
    // "real" cell in the ghost region.  We actually want a cell whose
    // primitives match the face; since the physical grid is x-uniform
    // the choice of i_ref doesn't matter.  Use the cell on the right
    // (i, j) if i < nx + 2ng, else (i-1, j).
    int i_ref = (i < nx + 2 * ng) ? i : (nx + 2 * ng - 1);
    int c = cflat(i_ref, j, sy);
    double r = fmax(rho[c], 1e-30);
    double sqrt_r = sqrt(r);
    double vx = mx[c] / r;

    int f = fx_flat(i, j, sy);
    double Bxf_old = Bxf[f];
    double inv = 1.0 / (2.0 * (tau + 1.0));
    double Bxf_new = ((tau * sqrt_r) * vx + (tau + 2.0) * Bxf_old) * inv;
    Bxf[f] = Bxf_new;
}
