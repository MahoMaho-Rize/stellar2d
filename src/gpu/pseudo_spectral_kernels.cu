// 偽譜法 2D 不可壓 NS (渦度-流函數) 的 device kernels。
//
// 譜空間 array 維度: nx × nh,row-major,index = ic * nh + jc。
// ic 對應 kx (可正可負,R2C 約定的 frequency-shifted 排列),
// jc 對應 ky (R2C 只保留非負頻率)。
// 物理 array 維度: nx × ny, flat index = ic * ny + jc (row-major, ic 慢)。

#include <cuda_runtime.h>
#include <cufft.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================
// 初始化 kx, ky 與 2/3 dealias mask
//   kx[ic]:  ic ≤ nx/2 → ic;  ic > nx/2 → ic - nx   (× 2π/Lx)
//   ky[jc]:  jc × 2π/Ly  (非負)
//   dealias[ic, jc] = 1 當 |shift(ic)| ≤ nx/3 且 jc ≤ ny/3,否則 0
// ============================================================
__global__ void k_init_wavenumbers(double* kx, double* ky, double* dealias,
                                   int nx, int ny, int nh,
                                   double Lx, double Ly) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * nh;
    if (gid >= total) return;

    int ic = gid / nh;
    int jc = gid - ic * nh;

    int i_shift = (ic <= nx / 2) ? ic : ic - nx;
    double kxv = (2.0 * M_PI / Lx) * (double)i_shift;
    double kyv = (2.0 * M_PI / Ly) * (double)jc;

    if (jc == 0) kx[ic] = kxv;
    if (ic == 0) ky[jc] = kyv;

    int cut_x = nx / 3;
    int cut_y = ny / 3;
    int i_mag = (i_shift < 0) ? -i_shift : i_shift;
    dealias[gid] = (i_mag <= cut_x && jc <= cut_y) ? 1.0 : 0.0;
}

// ============================================================
// 從 ω̂ 求 û, v̂:  ψ̂ = ω̂ / |k|²,  û = i ky ψ̂,  v̂ = -i kx ψ̂
// 同時套 dealias mask。
// ============================================================
__global__ void k_spec_uv(const cufftDoubleComplex* omega_hat,
                          const double* kx, const double* ky,
                          const double* dealias,
                          cufftDoubleComplex* u_hat,
                          cufftDoubleComplex* v_hat,
                          int nx, int nh) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * nh;
    if (gid >= total) return;

    int ic = gid / nh;
    int jc = gid - ic * nh;
    double kxv = kx[ic];
    double kyv = ky[jc];
    double k2  = kxv * kxv + kyv * kyv;
    double inv_k2 = (k2 > 0.0) ? 1.0 / k2 : 0.0;
    double mask = dealias[gid];

    cufftDoubleComplex w = omega_hat[gid];
    // ψ̂ = ω̂ / k²
    double psi_r = w.x * inv_k2;
    double psi_i = w.y * inv_k2;

    // û = i ky · ψ̂ = (-ky·ψ_i, ky·ψ_r)
    u_hat[gid].x = -kyv * psi_i * mask;
    u_hat[gid].y =  kyv * psi_r * mask;

    // v̂ = -i kx · ψ̂ = (kx·ψ_i, -kx·ψ_r)
    v_hat[gid].x =  kxv * psi_i * mask;
    v_hat[gid].y = -kxv * psi_r * mask;
}

// ============================================================
// ∂ω/∂x 譜 = i kx ω̂,  ∂ω/∂y 譜 = i ky ω̂。同樣套 dealias。
// ============================================================
__global__ void k_spec_grad_omega(const cufftDoubleComplex* omega_hat,
                                  const double* kx, const double* ky,
                                  const double* dealias,
                                  cufftDoubleComplex* dwx_hat,
                                  cufftDoubleComplex* dwy_hat,
                                  int nx, int nh) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * nh;
    if (gid >= total) return;

    int ic = gid / nh;
    int jc = gid - ic * nh;
    double kxv = kx[ic];
    double kyv = ky[jc];
    double mask = dealias[gid];

    cufftDoubleComplex w = omega_hat[gid];
    dwx_hat[gid].x = -kxv * w.y * mask;
    dwx_hat[gid].y =  kxv * w.x * mask;
    dwy_hat[gid].x = -kyv * w.y * mask;
    dwy_hat[gid].y =  kyv * w.x * mask;
}

// ============================================================
// 物理空間對流項  N(x,y) = u·∂ω/∂x + v·∂ω/∂y
// ============================================================
__global__ void k_compute_convection(const double* u, const double* v,
                                     const double* dwx, const double* dwy,
                                     double* Nphys, int ncell) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= ncell) return;
    Nphys[gid] = u[gid] * dwx[gid] + v[gid] * dwy[gid];
}

// cuFFT C2R 輸出需除以 N = nx·ny
__global__ void k_scale_inplace(double* x, double s, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    x[gid] *= s;
}

// ============================================================
// rhs_hat = -N̂ - ν |k|² ω̂   (並套 2/3 dealias mask)
// ============================================================
__global__ void k_form_rhs_hat(const cufftDoubleComplex* N_hat,
                               const cufftDoubleComplex* omega_hat,
                               const double* kx, const double* ky,
                               const double* dealias,
                               cufftDoubleComplex* rhs_hat,
                               double nu, int nx, int nh) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * nh;
    if (gid >= total) return;
    int ic = gid / nh;
    int jc = gid - ic * nh;
    double kxv = kx[ic], kyv = ky[jc];
    double k2  = kxv * kxv + kyv * kyv;
    double mask = dealias[gid];

    cufftDoubleComplex n = N_hat[gid];
    cufftDoubleComplex w = omega_hat[gid];
    rhs_hat[gid].x = (-n.x - nu * k2 * w.x) * mask;
    rhs_hat[gid].y = (-n.y - nu * k2 * w.y) * mask;
}

// ============================================================
// Shu-Osher RK3 低存儲級更新:
//   y_out = a · y_orig + b · y_curr + c_dt · rhs
// ============================================================
__global__ void k_rk_combine(cufftDoubleComplex* y_out,
                             const cufftDoubleComplex* y_orig,
                             const cufftDoubleComplex* y_curr,
                             const cufftDoubleComplex* rhs,
                             double a, double b, double c_dt,
                             int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    y_out[gid].x = a * y_orig[gid].x + b * y_curr[gid].x + c_dt * rhs[gid].x;
    y_out[gid].y = a * y_orig[gid].y + b * y_curr[gid].y + c_dt * rhs[gid].y;
}

// ============================================================
// 診斷: block-wise reduction
//   out[4·blockIdx] = {max|v|, max|ω|, Σ½(u²+v²)·dA, Σ½ω²·dA}
// ============================================================
__global__ void k_reduce_diag(const double* u, const double* v,
                              const double* omega,
                              double* out, int ncell, double cell_area) {
    extern __shared__ double shm[];
    double* s_max_v    = shm;
    double* s_max_w    = shm + blockDim.x;
    double* s_sum_ke   = shm + 2 * blockDim.x;
    double* s_sum_enst = shm + 3 * blockDim.x;

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    double mv = 0.0, mw = 0.0, sk = 0.0, se = 0.0;
    if (gid < ncell) {
        double uu = u[gid], vv = v[gid], ww = omega[gid];
        double speed2 = uu * uu + vv * vv;
        mv = sqrt(speed2);
        mw = fabs(ww);
        sk = 0.5 * speed2 * cell_area;
        se = 0.5 * ww * ww * cell_area;
    }
    s_max_v[tid]    = mv;
    s_max_w[tid]    = mw;
    s_sum_ke[tid]   = sk;
    s_sum_enst[tid] = se;
    __syncthreads();

    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) {
            s_max_v[tid]     = fmax(s_max_v[tid], s_max_v[tid + off]);
            s_max_w[tid]     = fmax(s_max_w[tid], s_max_w[tid + off]);
            s_sum_ke[tid]   += s_sum_ke[tid + off];
            s_sum_enst[tid] += s_sum_enst[tid + off];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[4 * blockIdx.x + 0] = s_max_v[0];
        out[4 * blockIdx.x + 1] = s_max_w[0];
        out[4 * blockIdx.x + 2] = s_sum_ke[0];
        out[4 * blockIdx.x + 3] = s_sum_enst[0];
    }
}

// ============================================================
// Frame pool snapshot — layout per frame: [ω | u | v] × ncell
// ============================================================
__global__ void k_snapshot_frame(const double* omega, const double* u,
                                 const double* v,
                                 double* slot, int ncell) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= ncell) return;
    slot[0 * ncell + gid] = omega[gid];
    slot[1 * ncell + gid] = u[gid];
    slot[2 * ncell + gid] = v[gid];
}
