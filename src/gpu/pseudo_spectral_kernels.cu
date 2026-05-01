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

    // 圓形 2/3 dealias: |k|² ≤ (N/3·2π/L)²  各向同性、無 corner 偽跡
    // (原方形 dealias: max(|kx|,|ky|) ≤ N/3 會偏向座標軸方向)
    double k_cut = 2.0 * M_PI * (double)(nx < ny ? nx : ny) / 3.0
                 / (Lx < Ly ? Lx : Ly);
    double k_mag2 = kxv * kxv + kyv * kyv;
    dealias[gid] = (k_mag2 <= k_cut * k_cut) ? 1.0 : 0.0;
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
// 物理空間對流項 (advective form)  N_A(x,y) = u·∂ω/∂x + v·∂ω/∂y
// ============================================================
__global__ void k_compute_convection(const double* u, const double* v,
                                     const double* dwx, const double* dwy,
                                     double* Nphys, int ncell) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= ncell) return;
    Nphys[gid] = u[gid] * dwx[gid] + v[gid] * dwy[gid];
}

// ============================================================
// 物理空間流積 (for conservative form): uω(x,y), vω(x,y)
// 寫回 u_out, v_out 空間 (呼叫者必須確保 u_out, v_out 可以被覆寫;
// 通常就是 d_u, d_v 本身 —— 因為 advective N_A 已算完,u/v 本輪
// 不再需要保留)。
// ============================================================
__global__ void k_compute_u_omega(const double* u, const double* v,
                                  const double* omega,
                                  double* uomega, double* vomega, int ncell) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= ncell) return;
    double w = omega[gid];
    uomega[gid] = u[gid] * w;
    vomega[gid] = v[gid] * w;
}

// ============================================================
// 譜空間組 skew-symmetric convection:
//   N̂_S = ½ (N̂_A + N̂_C)
//   其中 N̂_C = i·kx·(uω)^ + i·ky·(vω)^
// 輸入: N̂_A (已在 N_A_hat 裡), (uω)^, (vω)^
// 輸出: rhs_hat 以 -N̂_S 形式 (配合 k_form_rhs_adv_only 的慣例不同,這裡
//      直接寫 rhs = -N̂_S (dealiased);粘性由 IFRK 處理)
// ============================================================
__global__ void k_form_rhs_skew(const cufftDoubleComplex* N_A_hat,
                                const cufftDoubleComplex* uw_hat,
                                const cufftDoubleComplex* vw_hat,
                                const double* kx, const double* ky,
                                const double* dealias,
                                cufftDoubleComplex* rhs_hat,
                                int nx, int nh) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * nh;
    if (gid >= total) return;
    int ic = gid / nh;
    int jc = gid - ic * nh;
    double kxv = kx[ic], kyv = ky[jc];
    double m = dealias[gid];

    // N̂_C = i·kx·(uω)^ + i·ky·(vω)^
    //     = (-kx·uw.y - ky·vw.y, kx·uw.x + ky·vw.x)
    cufftDoubleComplex uw = uw_hat[gid];
    cufftDoubleComplex vw = vw_hat[gid];
    double NC_r = -(kxv * uw.y + kyv * vw.y);
    double NC_i =  (kxv * uw.x + kyv * vw.x);

    cufftDoubleComplex NA = N_A_hat[gid];
    // N̂_S = ½ (N̂_A + N̂_C) , rhs = -N̂_S
    rhs_hat[gid].x = -0.5 * (NA.x + NC_r) * m;
    rhs_hat[gid].y = -0.5 * (NA.y + NC_i) * m;
}

// cuFFT C2R 輸出需除以 N = nx·ny
__global__ void k_scale_inplace(double* x, double s, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    x[gid] *= s;
}

// 套 dealias mask (用於 IC 後清除高 k Gibbs 殘餘 / 強制 k≥kmax 為 0)
__global__ void k_apply_dealias(cufftDoubleComplex* hat,
                                const double* dealias, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    double m = dealias[gid];
    hat[gid].x *= m;
    hat[gid].y *= m;
}

// ============================================================
// rhs_hat = -N̂ (只含對流, 套 dealias mask) — IFRK3 用
//   粘性項在 k_ifrk_combine 裡經積分因子 exp(-νk²Δt·λ) 精確處理。
// ============================================================
__global__ void k_form_rhs_adv_only(const cufftDoubleComplex* N_hat,
                                    const double* dealias,
                                    cufftDoubleComplex* rhs_hat,
                                    int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    double m = dealias[gid];
    rhs_hat[gid].x = -N_hat[gid].x * m;
    rhs_hat[gid].y = -N_hat[gid].y * m;
}

// ============================================================
// rhs_hat = -N̂ - ν |k|² ω̂   (並套 2/3 dealias mask) — 顯式版
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
// IFRK3 級更新 (積分因子):
//   y_out = E_a·(a·y_orig) + E_b·(b·y_curr + c_dt·rhs)
// 其中 E_a = exp(-νk²Δt·expo_a),E_b = exp(-νk²Δt·expo_b)。
//
// 對應 IFRK3 三級:
//   stage 1 (y1): a=1, b=0, c=Δt, expo_a=expo_b=1       → E_a=E_b=E^1
//   stage 2 (y2): a=¾, b=¼, c=¼Δt, expo_a=½, expo_b=-½  → y_orig 帶 E^½,
//                 y_curr 帶 E^{-½} (y1 已是 E^1 倍,與 E^{-½} 合成 E^½)
//                 但更直觀:我們用正係數版本 —— 見下方 step() 註解。
//
// 實作採直接係數版:
//   y_out = (a·E_a)·y_orig + (b·E_b)·y_curr + (c_dt·E_b)·rhs
// stage 1: E_a = E_b = exp(-νk²Δt)
// stage 2: E_a = exp(-νk²Δt·½), E_b = exp(-νk²Δt·(-½))
// stage 3: E_a = exp(-νk²Δt),   E_b = exp(-νk²Δt·½)
//
// 注意 E_b 在 stage 2 指數為負 (放大),但 y_curr 本身帶 E^1,合成後 ≤ 1 不放大。
// 公式驗證:令 N=0,IFRK 應化簡為 y_{n+1} = E·yₙ (精確擴散)。
//   stage 1: y1 = E·ωₙ
//   stage 2: y2 = ¾·E^½·ωₙ + ¼·E^{-½}·y1 = ¾E^½·ωₙ + ¼E^{½}·ωₙ = E^½·ωₙ
//   stage 3: y3 = ⅓·E·ωₙ + ⅔·E^½·y2 = ⅓E·ωₙ + ⅔E^½·E^½·ωₙ = E·ωₙ  ✓
// ============================================================
__global__ void k_ifrk_combine(cufftDoubleComplex* y_out,
                               const cufftDoubleComplex* y_orig,
                               const cufftDoubleComplex* y_curr,
                               const cufftDoubleComplex* rhs,
                               const double* kx, const double* ky,
                               double a, double b, double c_dt,
                               double expo_a, double expo_b,
                               double nu_dt,
                               int nx, int nh) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * nh;
    if (gid >= total) return;
    int ic = gid / nh;
    int jc = gid - ic * nh;
    double k2 = kx[ic] * kx[ic] + ky[jc] * ky[jc];
    double Ea = exp(-nu_dt * k2 * expo_a);
    double Eb = exp(-nu_dt * k2 * expo_b);
    double ca = a * Ea;
    double cb = b * Eb;
    double cc = c_dt * Eb;
    y_out[gid].x = ca * y_orig[gid].x + cb * y_curr[gid].x + cc * rhs[gid].x;
    y_out[gid].y = ca * y_orig[gid].y + cb * y_curr[gid].y + cc * rhs[gid].y;
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
// 譜耗散率 reduction:   ε_spec = 2ν · Σ_k k² |ω̂(k)|²  / N²
//   對 R2C 佈局,jc=0 與 jc=nh-1 (若 ny 偶) 模只出現一次,其餘模要 ×2。
//   輸出 out[blockIdx] 為 block 內加和,host 再求總和。
// 這裡我們忽略 jc=0/末 的半計數 (差 O(1/N)),實用足夠。
// ============================================================
__global__ void k_reduce_k2E(const cufftDoubleComplex* omega_hat,
                             const double* kx, const double* ky,
                             double* out, int ncplx, int nx, int nh,
                             double scale /* = 2/N² for Hermitian doubling + normalization */) {
    extern __shared__ double shm_k2[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    double s = 0.0;
    if (gid < ncplx) {
        int ic = gid / nh;
        int jc = gid - ic * nh;
        double kxv = kx[ic], kyv = ky[jc];
        double k2 = kxv * kxv + kyv * kyv;
        cufftDoubleComplex w = omega_hat[gid];
        double mag2 = w.x * w.x + w.y * w.y;
        // jc = 0 (和 ny 偶時 jc = nh-1) 不重複,其他 ×2 (Hermitian 對稱)
        double weight = (jc == 0 || jc == nh - 1) ? 1.0 : 2.0;
        s = scale * weight * k2 * mag2;
    }
    shm_k2[tid] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) shm_k2[tid] += shm_k2[tid + off];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = shm_k2[0];
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
