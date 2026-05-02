// Kernels for the anelastic SL solver.
//
// Memory layout convention (ALL buffers):
//   Physical space:  d_*[jy * nx + ix]     (y is slow, x is fast — ny × nx row-major)
//   cuFFT R2C in x:  d_*hat[jy * nh + kx]  (ny × nh row-major complex)
//   SL spectral:     d_*hat_sl[n * nh + kx] (n_modes × nh row-major complex)
//
// For cuBLAS DGEMM (column-major), interpretation:
//   d_fhat is treated as a matrix with nh rows (leading) × ny cols (trailing).
//   That is: entry (kx, y) at offset y*nh + kx.  In col-major terms,
//   lda = nh, rows = nh, cols = ny.
//   Similarly d_Psi (host row-major, ny × n_modes) equals a col-major
//   (n_modes rows, ny cols) matrix with lda = n_modes.

#include <cuda_runtime.h>
#include <cufft.h>

extern "C" {

// ── weight_fhat_inplace ─────────────────────────────────────────────────
// ghat[jy, kx] = w[jy] * fhat[jy, kx]   (ghat and fhat can alias)
__global__ void k_weight_fhat_inplace(
    cufftDoubleComplex* ghat,
    const cufftDoubleComplex* fhat,
    const double* weight,  // (ny,)
    int ny, int nh)
{
    int jy = blockIdx.y * blockDim.y + threadIdx.y;
    int kx = blockIdx.x * blockDim.x + threadIdx.x;
    if (jy >= ny || kx >= nh) return;
    double w = weight[jy];
    cufftDoubleComplex in = fhat[jy * nh + kx];
    cufftDoubleComplex out;
    out.x = w * in.x;
    out.y = w * in.y;
    ghat[jy * nh + kx] = out;
}

// ── apply_cc_weight ─────────────────────────────────────────────────────
// For the forward SL transform we compute G_n(kx) = Σ_jy w_cc[jy] * ψ_n(y_jy) * g(jy, kx)
// = (Ψ^T · (W_cc * g))(n, kx).
// So we pre-multiply g by w_cc[jy] along the y axis before GEMM.
// (Identical shape to k_weight_fhat_inplace; kept separate for clarity.)

// ── diag_divide_SL ──────────────────────────────────────────────────────
// Qhat[n, kx] = -Ghat[n, kx] / (mu[n] + kx^2)
__global__ void k_diag_divide_sl(
    cufftDoubleComplex* Qhat,
    const cufftDoubleComplex* Ghat,
    const double* mu,    // (n_modes,)
    const double* kx,    // (nh,)
    int n_modes, int nh)
{
    int n  = blockIdx.y * blockDim.y + threadIdx.y;
    int kk = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= n_modes || kk >= nh) return;
    double denom = mu[n] + kx[kk] * kx[kk];
    // Guard DC mode of constant-coeff Laplacian (mu_0 may be tiny if Boussinesq
    // is exact; harmless because source has zero mean).
    if (denom < 1e-30 && denom > -1e-30) denom = 1e-30;
    cufftDoubleComplex g = Ghat[n * nh + kk];
    cufftDoubleComplex q;
    q.x = -g.x / denom;
    q.y = -g.y / denom;
    Qhat[n * nh + kk] = q;
}

// ── normalize_ifft ──────────────────────────────────────────────────────
// cuFFT C2R does not normalise; we scale by 1/nx after IFFT.
__global__ void k_normalize(
    double* d, int n, double s)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] *= s;
}


}  // extern "C"
