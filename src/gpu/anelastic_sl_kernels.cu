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


// ── 2/3 dealias mask in x: zero modes with |kx| > (2/3) kx_max. ─────────
__global__ void k_dealias_x_inplace(
    cufftDoubleComplex* fhat,
    int ny, int nh, int kx_cut)
{
    int jy = blockIdx.y * blockDim.y + threadIdx.y;
    int kk = blockIdx.x * blockDim.x + threadIdx.x;
    if (jy >= ny || kk >= nh) return;
    if (kk > kx_cut) {
        int off = jy * nh + kk;
        fhat[off].x = 0.0;
        fhat[off].y = 0.0;
    }
}

// ── spectral-space ∂_x via i·kx multiply ─────────────────────────────────
// dst[jy, kx] = (i · kx_vec[kx]) · src[jy, kx]   (complex multiply by i·kx)
// Also applies 1/nx normalisation factor `inv_nx` so the subsequent IFFT
// output is the physical derivative without a second normalise pass.
__global__ void k_mult_ikx_out(
    cufftDoubleComplex* dst,
    const cufftDoubleComplex* src,
    const double* kx_vec,   // (nh,)
    double inv_nx,
    int ny, int nh)
{
    int jy = blockIdx.y * blockDim.y + threadIdx.y;
    int kk = blockIdx.x * blockDim.x + threadIdx.x;
    if (jy >= ny || kk >= nh) return;
    int off = jy * nh + kk;
    double kx = kx_vec[kk] * inv_nx;
    cufftDoubleComplex in = src[off];
    cufftDoubleComplex o;
    o.x = -kx * in.y;   // Re(i·kx · (a+ib)) = -kx·b
    o.y =  kx * in.x;   // Im(i·kx · (a+ib)) = +kx·a
    dst[off] = o;
}

// ── spectral-space ∂_xx via -kx² multiply ────────────────────────────────
__global__ void k_mult_mkx2_out(
    cufftDoubleComplex* dst,
    const cufftDoubleComplex* src,
    const double* kx_vec,
    double inv_nx,
    int ny, int nh)
{
    int jy = blockIdx.y * blockDim.y + threadIdx.y;
    int kk = blockIdx.x * blockDim.x + threadIdx.x;
    if (jy >= ny || kk >= nh) return;
    int off = jy * nh + kk;
    double kx = kx_vec[kk];
    double fac = -kx * kx * inv_nx;
    cufftDoubleComplex in = src[off];
    cufftDoubleComplex o;
    o.x = fac * in.x;
    o.y = fac * in.y;
    dst[off] = o;
}

// ── accumulate:  acc[i] += fac · a[i] · b[i] ────────────────────────────
__global__ void k_fma_product(
    double* acc, double fac,
    const double* a, const double* b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += fac * a[i] * b[i];
}

// ── accumulate:  acc[i] += fac · a[i] ───────────────────────────────────
__global__ void k_fma_scalar(
    double* acc, double fac, const double* a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += fac * a[i];
}

// ── Shu-Osher RK3 combine:  out[i] = ca·a[i] + cb·(b[i] + dt·r[i]) ───────
__global__ void k_rk3_combine(
    double* out,
    const double* a, const double* b, const double* r,
    double ca, double cb, double dt, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ca * a[i] + cb * (b[i] + dt * r[i]);
}

// ── in-place subtraction:  a[i] -= b[i]  ────────────────────────────────
__global__ void k_sub_inplace(double* a, const double* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] -= b[i];
}

// ── pointwise store: c[i] = a[i] + b[i]  ───────────────────────────────
__global__ void k_add_out(double* c, const double* a, const double* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// ── Anelastic buoyancy: dRV[j] += b[j] (hydrostatic buoyancy, g=1) ──────
__global__ void k_add_buoyancy(double* dRV, const double* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dRV[i] += b[i];
}

// ── Anelastic N² forcing: dRB[j] -= N²(y_j) · v[j] ──────────────────────
// For row-major (ny × nx), y is the slow axis → index j = idx / nx.
__global__ void k_sub_N2_v(double* dRB, const double* v, const double* N2,
                           int nx, int ny) {
    int jy = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (jy >= ny || ix >= nx) return;
    int k = jy * nx + ix;
    dRB[k] -= N2[jy] * v[k];
}

// ── Row-multiply: out[j, i] += fac · a[j, i] · rho[j] ──────────────────
// Adds fac·ρ(y)·src to acc (used to build anelastic projection RHS).
__global__ void k_fma_row(double* acc, double fac, const double* src,
                          const double* row_scale, int nx, int ny) {
    int jy = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (jy >= ny || ix >= nx) return;
    int k = jy * nx + ix;
    acc[k] += fac * src[k] * row_scale[jy];
}

// ── ω = ∂v/∂x − ∂u/∂y ──────────────────────────────────────────────────
__global__ void k_compute_omega(
    double* omega, const double* dvdx, const double* dudy, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) omega[i] = dvdx[i] - dudy[i];
}

// ── zero the Dirichlet y-boundary rows of a physical field ──────────────
// Row-major (ny × nx) layout: rows jy=0 and jy=ny-1 become 0.
__global__ void k_zero_y_boundary(double* f, int nx, int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= nx) return;
    f[0 * nx + ix]          = 0.0;
    f[(ny - 1) * nx + ix]   = 0.0;
}

// ── Zero the kx=0 (x-mean) column of a cuFFT R2C spectrum ──────────
// Layout: d_fhat[jy * nh + 0] = DC coefficient for row jy.
// Used to project out x-mean flow produced by the nonlinear block.
// Anelastic continuity ∇·(ρ₀u)=0 plus Dirichlet walls v(0)=v(Ly)=0
// forbid a nonzero ⟨v⟩_x: any such mean would need ⟨u⟩_x with
// ∂_y(ρ⟨u⟩)=0, which has only the trivial solution under v-walls →
// k=0 component is an unphysical Reynolds-stress-driven DC sink.
__global__ void k_zero_kx0_column(cufftDoubleComplex* d_fhat,
                                  int ny, int nh) {
    int jy = blockIdx.x * blockDim.x + threadIdx.x;
    if (jy >= ny) return;
    size_t off = (size_t)jy * nh + 0;
    d_fhat[off].x = 0.0;
    d_fhat[off].y = 0.0;
}

// ── store: dst[i] = -N²(y_i) · v[i]  (row-scaled product, overwrite) ───
__global__ void k_neg_N2_v_out(double* dst, const double* v, const double* N2,
                               int nx, int ny) {
    int jy = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (jy >= ny || ix >= nx) return;
    int k = jy * nx + ix;
    dst[k] = -N2[jy] * v[k];
}

// ── Row-multiply store: dst[j, i] = rho(y_j) · src[j, i]  (overwrite) ──
__global__ void k_row_mul_out(double* dst, const double* src,
                              const double* row_scale, int nx, int ny) {
    int jy = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (jy >= ny || ix >= nx) return;
    int k = jy * nx + ix;
    dst[k] = src[k] * row_scale[jy];
}

// ── û(kx, y) = -(1/(i·kx·ρ(y))) · ĝ(kx, y)  for kx≠0; û(0,·) = 0 ──────
// Acts on cuFFT R2C output layout: d_out[jy*nh + kx_idx] = complex sample.
// Computes û = (i / (kx · ρ(y))) · ĝ  (since -1/i = i).  Skips kx=0 column.
// Also normalises by (1/nx) so the subsequent C2R produces properly-scaled
// physical values (cuFFT C2R is unnormalised).
__global__ void k_u_from_div_v(cufftDoubleComplex* d_out,
                               const cufftDoubleComplex* d_g_hat,
                               const double* d_kx, const double* d_rho,
                               double inv_nx, int ny, int nh) {
    int kx_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int jy     = blockIdx.y * blockDim.y + threadIdx.y;
    if (kx_idx >= nh || jy >= ny) return;
    int off = (size_t)jy * nh + kx_idx;
    double kx = d_kx[kx_idx];
    if (kx_idx == 0 || fabs(kx) < 1e-30) {
        d_out[off].x = 0.0; d_out[off].y = 0.0;
        return;
    }
    double inv = inv_nx / (kx * d_rho[jy]);
    cufftDoubleComplex g = d_g_hat[off];
    // û = (i / (kx·ρ)) · ĝ = inv · (i · (gr + i·gi)) = inv · (-gi + i·gr)
    d_out[off].x = -inv * g.y;
    d_out[off].y =  inv * g.x;
}

// ── max absolute value reduction (two-pass: blocks → scratch → host) ────
__global__ void k_max_abs_pass1(
    const double* a, int n, double* out_blocks)
{
    __shared__ double s[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    double v = 0.0;
    if (i < n) v = fabs(a[i]);
    s[tid] = v;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) {
            double o = s[tid + off];
            if (o > s[tid]) s[tid] = o;
        }
        __syncthreads();
    }
    if (tid == 0) out_blocks[blockIdx.x] = s[0];
}

// ── Pack (u, v, b) double → float32 frame slot for VRAM ring buffer ────
// out[0..ncell)        = u (float)
// out[ncell..2*ncell)  = v (float)
// out[2*ncell..3*ncell)= b (float)
__global__ void k_ansl_pack_snap(const double* u, const double* v,
                                 const double* b, float* out, int ncell) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncell) return;
    out[i]             = (float)u[i];
    out[i + ncell]     = (float)v[i];
    out[i + 2 * ncell] = (float)b[i];
}

}  // extern "C"
