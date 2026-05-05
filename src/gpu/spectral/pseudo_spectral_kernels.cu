// 偽譜法 2D 不可壓 NS (渦度-流函數) 的 device kernels。
//
// 譜空間 array 維度: nx × nh,row-major,index = ic * nh + jc。
// ic 對應 kx (可正可負,R2C 約定的 frequency-shifted 排列),
// jc 對應 ky (R2C 只保留非負頻率)。
// 物理 array 維度: nx × ny, flat index = ic * ny + jc (row-major, ic 慢)。

#include <cuda_runtime.h>
#include <cufft.h>
#include <curand_kernel.h>
#include <cstdint>

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
// Fused:從 ω̂ 一次生成 5 個譜 field 到連續 batch buffer
//   out[0] = û    = i·ky·ω̂ / |k|²
//   out[1] = v̂    = -i·kx·ω̂ / |k|²
//   out[2] = (∂ω/∂x)^  = i·kx·ω̂
//   out[3] = (∂ω/∂y)^  = i·ky·ω̂
//   out[4] = ω̂  (dealiased copy)
// dealias mask apply 在所有 5 個輸出上(ω̂ 輸入保持未污染 state)
// 每個 thread 處理 1 個 (ic, jc),讀一次 ω̂ 寫 5 個 complex → 大幅減少
// memory bandwidth 壓力(舊版 2 個獨立 kernel 各讀一次 ω̂ + 寫 2 個,共 4 讀 4 寫)。
// ============================================================
__global__ void k_build_5fields_spec(const cufftDoubleComplex* omega_hat,
                                     const double* kx, const double* ky,
                                     const double* dealias,
                                     cufftDoubleComplex* out,    // 5 * ncplx 連續
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
    double wr = w.x, wi = w.y;

    // ψ̂ = ω̂ / k²
    double psi_r = wr * inv_k2;
    double psi_i = wi * inv_k2;

    // û = i·ky·ψ̂
    out[0 * total + gid].x = -kyv * psi_i * mask;
    out[0 * total + gid].y =  kyv * psi_r * mask;
    // v̂ = -i·kx·ψ̂
    out[1 * total + gid].x =  kxv * psi_i * mask;
    out[1 * total + gid].y = -kxv * psi_r * mask;
    // (∂ω/∂x)^ = i·kx·ω̂
    out[2 * total + gid].x = -kxv * wi * mask;
    out[2 * total + gid].y =  kxv * wr * mask;
    // (∂ω/∂y)^ = i·ky·ω̂
    out[3 * total + gid].x = -kyv * wi * mask;
    out[3 * total + gid].y =  kyv * wr * mask;
    // ω̂ (dealiased)
    out[4 * total + gid].x = wr * mask;
    out[4 * total + gid].y = wi * mask;
}

// Fused 4-field variant(advective/conservative 不需要 ω 物理空間):
//   out[0..3] = û, v̂, (∂ω/∂x)^, (∂ω/∂y)^
__global__ void k_build_4fields_spec(const cufftDoubleComplex* omega_hat,
                                     const double* kx, const double* ky,
                                     const double* dealias,
                                     cufftDoubleComplex* out,
                                     int nx, int nh) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * nh;
    if (gid >= total) return;
    int ic = gid / nh;
    int jc = gid - ic * nh;
    double kxv = kx[ic], kyv = ky[jc];
    double k2 = kxv * kxv + kyv * kyv;
    double inv_k2 = (k2 > 0.0) ? 1.0 / k2 : 0.0;
    double mask = dealias[gid];
    cufftDoubleComplex w = omega_hat[gid];
    double wr = w.x, wi = w.y;
    double psi_r = wr * inv_k2;
    double psi_i = wi * inv_k2;
    out[0 * total + gid].x = -kyv * psi_i * mask;
    out[0 * total + gid].y =  kyv * psi_r * mask;
    out[1 * total + gid].x =  kxv * psi_i * mask;
    out[1 * total + gid].y = -kxv * psi_r * mask;
    out[2 * total + gid].x = -kxv * wi * mask;
    out[2 * total + gid].y =  kxv * wr * mask;
    out[3 * total + gid].x = -kyv * wi * mask;
    out[3 * total + gid].y =  kyv * wr * mask;
}

// Fused 物理空間對流 + skew 輔助:一次讀 5 場(u, v, dwx, dwy, ω),寫 3 場(N_A, uω, vω)
// batched R2C 友好 layout:output 在連續 3-batch buffer [N_A | uω | vω]
__global__ void k_compute_skew_nonlinear(const double* phys5,    // (u, v, dwx, dwy, ω) × ncell
                                         double* phys3,          // (N_A, uω, vω) × ncell
                                         int ncell) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= ncell) return;
    double u  = phys5[0 * ncell + gid];
    double v  = phys5[1 * ncell + gid];
    double dx = phys5[2 * ncell + gid];
    double dy = phys5[3 * ncell + gid];
    double w  = phys5[4 * ncell + gid];
    phys3[0 * ncell + gid] = u * dx + v * dy;   // N_A
    phys3[1 * ncell + gid] = u * w;             // uω
    phys3[2 * ncell + gid] = v * w;             // vω
}

// Advective variant (無 skew):4 場進 → 1 場出(N_A)
__global__ void k_compute_adv_nonlinear(const double* phys4,
                                        double* N_A, int ncell) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= ncell) return;
    double u  = phys4[0 * ncell + gid];
    double v  = phys4[1 * ncell + gid];
    double dx = phys4[2 * ncell + gid];
    double dy = phys4[3 * ncell + gid];
    N_A[gid] = u * dx + v * dy;
}

// Conservative variant:4 場進(u, v, dwx, dwy)+ ω 單獨 → 2 場出(uω, vω)
// 為了和 4-batch pipeline 配合,ω 由 caller 另算一次 IFFT 存入 phys5[4] 之類位置。
// 不過 conservative 路徑我們選 5-batch IFFT(同 skew),只差最後 compute 不同。
__global__ void k_compute_conservative_nonlinear(const double* phys5,
                                                 double* phys2,   // (uω, vω)
                                                 int ncell) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= ncell) return;
    double u = phys5[0 * ncell + gid];
    double v = phys5[1 * ncell + gid];
    double w = phys5[4 * ncell + gid];
    phys2[0 * ncell + gid] = u * w;
    phys2[1 * ncell + gid] = v * w;
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
// Conservative/rotational form (Basdevant 1986):
//   N_C = ∂(uω)/∂x + ∂(vω)/∂y
// ∇·u=0 下連續方程 N_A ≡ N_C;離散下也守 enstrophy,只要 2 次 R2C FFT
// (vs skew 的 3 次)。輸入 (uω)^ 在 uw_hat,(vω)^ 在 vw_hat。
// 輸出 rhs = -N̂_C (套 dealias)。
// ============================================================
__global__ void k_form_rhs_conservative(const cufftDoubleComplex* uw_hat,
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
    cufftDoubleComplex uw = uw_hat[gid];
    cufftDoubleComplex vw = vw_hat[gid];
    // N̂_C = i·kx·(uω)^ + i·ky·(vω)^
    double NC_r = -(kxv * uw.y + kyv * vw.y);
    double NC_i =  (kxv * uw.x + kyv * vw.x);
    rhs_hat[gid].x = -NC_r * m;
    rhs_hat[gid].y = -NC_i * m;
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
// rhs_hat = -N̂ - (ν·|k|^(2p) + α)·ω̂   (並套 2/3 dealias mask) — 顯式版
//   hyper_p: 1 為標準 Laplacian;>1 為 hyperviscosity -ν·(-Δ)^p。
//   drag_alpha: linear drag -α·ω,破壞 condensate (Kraichnan regime)。
// ============================================================
__global__ void k_form_rhs_hat(const cufftDoubleComplex* N_hat,
                               const cufftDoubleComplex* omega_hat,
                               const double* kx, const double* ky,
                               const double* dealias,
                               cufftDoubleComplex* rhs_hat,
                               double nu, double drag_alpha,
                               int hyper_p, int nx, int nh) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * nh;
    if (gid >= total) return;
    int ic = gid / nh;
    int jc = gid - ic * nh;
    double kxv = kx[ic], kyv = ky[jc];
    double k2  = kxv * kxv + kyv * kyv;
    double k2p = k2;
    for (int q = 1; q < hyper_p; ++q) k2p *= k2;   // |k|^(2p)
    double mask = dealias[gid];
    double diss = nu * k2p + drag_alpha;

    cufftDoubleComplex n = N_hat[gid];
    cufftDoubleComplex w = omega_hat[gid];
    rhs_hat[gid].x = (-n.x - diss * w.x) * mask;
    rhs_hat[gid].y = (-n.y - diss * w.y) * mask;
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
                               double nu_dt, double drag_dt,
                               int hyper_p,
                               int nx, int nh) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * nh;
    if (gid >= total) return;
    int ic = gid / nh;
    int jc = gid - ic * nh;
    double k2 = kx[ic] * kx[ic] + ky[jc] * ky[jc];
    double k2p = k2;
    for (int q = 1; q < hyper_p; ++q) k2p *= k2;
    // 總耗散指數 L·dt = (ν·|k|^(2p) + α)·dt
    double Ldt = nu_dt * k2p + drag_dt;
    // Clamp 防下溢產生 denormal * inf = NaN;Ldt·expo 極大時直接歸 0(exp(-∞)=0)。
    double Ea = (Ldt * expo_a >  700.0) ? 0.0
              : (Ldt * expo_a < -700.0) ? __longlong_as_double(0x7FF0000000000000LL)
              : exp(-Ldt * expo_a);
    double Eb = (Ldt * expo_b >  700.0) ? 0.0
              : (Ldt * expo_b < -700.0) ? __longlong_as_double(0x7FF0000000000000LL)
              : exp(-Ldt * expo_b);
    // 若 Ea/Eb 達到 0 (強耗散),強制 ω̂(k) → 0(避免 0·NaN 污染)
    double y_orig_x = y_orig[gid].x, y_orig_y = y_orig[gid].y;
    double y_curr_x = y_curr[gid].x, y_curr_y = y_curr[gid].y;
    double rhs_x    = rhs[gid].x,    rhs_y    = rhs[gid].y;
    double ca = a * Ea;
    double cb = b * Eb;
    double cc = c_dt * Eb;
    double out_x = ca * y_orig_x + cb * y_curr_x + cc * rhs_x;
    double out_y = ca * y_orig_y + cb * y_curr_y + cc * rhs_y;
    if (!isfinite(out_x)) out_x = 0.0;
    if (!isfinite(out_y)) out_y = 0.0;
    y_out[gid].x = out_x;
    y_out[gid].y = out_y;
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
// 通用 Σ k^(2·p_eff)·|ω̂|²·weight·scale reduction。
//   p_eff = 0  → 純 Σ|ω̂|² (能量代理)
//   p_eff = 1  → Σk²|ω̂|² = ∫|∇ω|²
//   p_eff = p-1 (hyperviscosity p):  對應 eps_KE hyper 修正項
//   p_eff = p   (hyperviscosity p):  對應 eps_enstrophy
__global__ void k_reduce_k2E(const cufftDoubleComplex* omega_hat,
                             const double* kx, const double* ky,
                             double* out, int ncplx, int nx, int nh,
                             double scale /* = (Lx·Ly)/N² 量級 */,
                             int p_eff /* 0,1,… exponent of k^2 */) {
    extern __shared__ double shm_k2[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    double s = 0.0;
    if (gid < ncplx) {
        int ic = gid / nh;
        int jc = gid - ic * nh;
        double kxv = kx[ic], kyv = ky[jc];
        double k2 = kxv * kxv + kyv * kyv;
        double k2p = 1.0;
        for (int q = 0; q < p_eff; ++q) k2p *= k2;
        cufftDoubleComplex w = omega_hat[gid];
        double mag2 = w.x * w.x + w.y * w.y;
        // jc = 0 (和 ny 偶時 jc = nh-1) 不重複,其他 ×2 (Hermitian 對稱)
        double weight = (jc == 0 || jc == nh - 1) ? 1.0 : 2.0;
        s = scale * weight * k2p * mag2;
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
// Ring-integrated energy spectrum E(k) (譜密度,Σ E·dk = KE_total)
//
// 對每個 R2C mode (ic, jc) 做:
//   E_mode = (Lx·Ly / N⁴) · w(k) · ½·|ω̂|² / k²      (k>0;k=0 丟棄)
//   bin    = round(|k|/dk),  dk = 2π/max(Lx,Ly)
//   atomicAdd 進 E_bins[bin]
//
// w(k): R2C Hermitian 冗餘因子 = 2 (一般 jc≥1) 或 1 (jc=0 與 Nyquist)。
//
// 輸出 E_bins 是「每 bin 內的總能量 (未除 dk)」;
// host 端讀回後再 / dk 得到譜密度。
// ============================================================
__global__ void k_reduce_spectrum_bins(const cufftDoubleComplex* omega_hat,
                                       const double* kx, const double* ky,
                                       double* E_bins,
                                       int nx, int nh, int nbins,
                                       double dk_inv,         // 1 / dk
                                       double mode_scale,     // Lx·Ly / N⁴ · ½
                                       int    ny_even) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * nh;
    if (gid >= total) return;

    int ic = gid / nh;
    int jc = gid - ic * nh;
    double kxv = kx[ic];
    double kyv = ky[jc];
    double k2  = kxv * kxv + kyv * kyv;
    if (k2 <= 0.0) return;                       // 跳過 DC

    cufftDoubleComplex w = omega_hat[gid];
    double mag2 = w.x * w.x + w.y * w.y;
    // Hermitian 重數:jc=0 與 (ny 偶時) jc=nh-1 不雙計
    double weight = (jc == 0 || (ny_even && jc == nh - 1)) ? 1.0 : 2.0;
    double E_mode = mode_scale * weight * mag2 / k2;

    double kmag = sqrt(k2);
    int b = (int)(kmag * dk_inv + 0.5);          // round
    if (b < 0)       b = 0;
    if (b >= nbins)  b = nbins - 1;
    atomicAdd(&E_bins[b], E_mode);
}

// ============================================================
// 隨機相位強迫 (white-noise forcing in vorticity, thin shell)
//   Per step:  Δω̂(k) = √dt · σ(k) · e^{iφ}
//   idx[i]    = 被強迫的 ncplx 索引
//   sigma[i]  = 對應 σ 值
//   phase_cos[i], phase_sin[i]: 每步由 host mt19937 生成
// σ 的挑選由 host 端在 init_forcing 做,滿足
//   ε_inj = ½·(Lx·Ly/N^4)·Σ w(k)·σ²/|k|²
// ============================================================
__global__ void k_apply_forcing(cufftDoubleComplex* omega_hat,
                                const int* idx, const double* sigma,
                                const double* phase_cos,
                                const double* phase_sin,
                                double sqrt_dt, int n_forcing) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_forcing) return;
    int k_idx = idx[gid];
    double s = sigma[gid] * sqrt_dt;
    omega_hat[k_idx].x += s * phase_cos[gid];
    omega_hat[k_idx].y += s * phase_sin[gid];
}

// ============================================================
// cuRAND-based device-side forcing:per-mode Philox 生成 phase,省 D2H。
//   每 mode 有自己的 curandStatePhilox4_32_10_t。init 一次,後續重複用。
// ============================================================
__global__ void k_init_curand_states(curandStatePhilox4_32_10_t* states,
                                     uint64_t seed, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    // 每個 mode 不同 sequence,保證獨立
    curand_init(seed, (unsigned long long)gid, 0, &states[gid]);
}

__global__ void k_apply_forcing_curand(cufftDoubleComplex* omega_hat,
                                       const int* idx, const double* sigma,
                                       curandStatePhilox4_32_10_t* states,
                                       double sqrt_dt, int n_forcing) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_forcing) return;
    curandStatePhilox4_32_10_t st = states[gid];
    double u = curand_uniform_double(&st);
    double phi = 2.0 * M_PI * u;
    states[gid] = st;
    double cphi, sphi;
    sincos(phi, &sphi, &cphi);
    int k_idx = idx[gid];
    double s = sigma[gid] * sqrt_dt;
    omega_hat[k_idx].x += s * cphi;
    omega_hat[k_idx].y += s * sphi;
}

// ============================================================
// DC mode 清零 (R2C 約定下 gid=0 即 (ic=0, jc=0))。
// 防止浮點誤差 + forcing 殘影累積成 spurious mean vorticity。
// ============================================================
__global__ void k_clear_dc(cufftDoubleComplex* hat) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        hat[0].x = 0.0;
        hat[0].y = 0.0;
    }
}

// ============================================================
// Taylor-Green 解析解誤差(譜空間;避開 IFFT 數值噪聲):
//   ω_exact_hat(t) = ω_hat_ic · exp(-2·ν·k_f²·t)     (k_f: 對應的 TG 波數物理值²)
//   err_L2² = (Lx·Ly/N⁴) · Σ w(k)·|ω̂(t) - ω̂_exact(t)|²
// 注:TG 在連續下對流項 ≡ 0,故數值誤差純來自 IFRK/hyper 處理。
// 對每個 mode 算 |diff|²,weight 同 R2C Hermitian。
// ============================================================
__global__ void k_reduce_tg_err(const cufftDoubleComplex* omega_hat,
                                const cufftDoubleComplex* omega_hat_ic,
                                double decay,   // exp(-2·ν·k_f²·t)
                                double* out, int ncplx, int nh,
                                double scale) {
    extern __shared__ double shm_tg[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    double s = 0.0;
    if (gid < ncplx) {
        int ic = gid / nh;
        int jc = gid - ic * nh;
        (void)ic;
        double weight = (jc == 0 || jc == nh - 1) ? 1.0 : 2.0;
        cufftDoubleComplex w  = omega_hat[gid];
        cufftDoubleComplex w0 = omega_hat_ic[gid];
        double ex_r = decay * w0.x;
        double ex_i = decay * w0.y;
        double dr = w.x - ex_r;
        double di = w.y - ex_i;
        s = scale * weight * (dr * dr + di * di);
    }
    shm_tg[tid] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) shm_tg[tid] += shm_tg[tid + off];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = shm_tg[0];
}

// ============================================================
// 譜空間動能耗散率交叉檢查:
//   eps_KE_spec = 2·(ν·Σk²p·|ψ̂|²·k² + α·Σ|ψ̂|²·k²) ·scale
// 當前做法更簡:eps_KE = 2·(ν + α·某種表示)?
// 其實 d(KE)/dt = -2·(ν·∫|∇ω|² + α·∫|∇ψ|²? no: α 在 ω 方程)
// d(KE)/dt(drag) = ∫u · ∂u/∂t dA;∂u/∂t 含 -α·u(由 ω 的 -α·ω 投影回 u)
//   → drag 對 KE 貢獻 = -2α·KE   (見文獻 Boffetta-Ecke 2012 eq 7)
// Laplacian/hyper 對 KE 貢獻 = -2ν·Σk^(2p-2)·|ω̂|²·factor
//   p=1: -2ν·Ω
//   p>1: -2ν·∫ω·(-Δ)^(p-1)·ω dA
// 我們直接在 host 側用 total_KE 和 譜 Σk^(2p-2)|ω̂|² 合成,不需新 kernel。
// ============================================================

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
