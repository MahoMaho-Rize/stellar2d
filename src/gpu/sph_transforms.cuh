#pragma once

// 球面譜變換 (θ, φ):φ 用 Fourier,θ 用 Associated Legendre。
//
// 實 field f(θ,φ) 的球諧展開:
//   f(θ,φ) = Σ_{l=0}^{L} Σ_{m=-l}^{l} f̂_l^m · Y_l^m(θ,φ)
// 其中 Y_l^m = N_l^m · P_l^m(cos θ) · exp(imφ),
//   N_l^m = √((2l+1)/(4π) · (l-m)!/(l+m)!)
// (4π-normalized:  ∫_{S²} Y_l^m · Y_l'^m'* dΩ = δ_{ll'} δ_{mm'}。)
//
// 對實 f,存 m ≥ 0 半平面(f̂_l^{-m} = (-1)^m · conj(f̂_l^m)),總 DOF (L+1)(L+2)/2 複數。
//
// Pipeline:
//   SpatialToSpectral(f):
//     1. φ 方向 R2C FFT,逐緯度得到 f_m(θ_j),j=0..N_θ-1  (複數)
//     2. θ 方向 Gauss-Legendre quadrature:
//        f̂_l^m = Σ_j w_j · f_m(θ_j) · P_l^m(cos θ_j) · N_l^m
//   SpectralToSpatial(f̂):
//     1. θ 方向逆變換:f_m(θ_j) = Σ_{l≥m} f̂_l^m · P_l^m(cos θ_j) · N_l^m
//     2. φ 方向 C2R FFT 得物理 f(θ_j, φ_i)
//
// 節點:θ 用 Gauss-Legendre (cos θ_j 為 Legendre roots),φ 用等距 N_φ 點 (cuFFT 標準)。
//
// 關鍵演算法選擇:
//   - P_l^m(x) 在初始化時 precompute 並存 (N_θ × DOF_spec) 的大表(CPU 版 naive 作法),
//     device 端 kernel 做 O(L²·N_θ·N_φ) 的 reduce。
//   - 產生代表的量級:L=127 (N_θ≈192, N_φ=384) 下 P 表 ~ 192·128·129/2·8B = 12.7 MB,OK。
//   - O(N²) LT 不符合工業代碼(SHTns 做 O(N² log N)),但先有能跑的版本,perf 不夠再上 SHTns。

#include <cufft.h>
#include <cstdint>
#include <vector>
#include <string>

struct SphTransform {
    // ---- 網格 ----
    int N_theta = 0;   // Gauss-Legendre 緯度節點數
    int N_phi   = 0;   // 經度等距節點數
    int N_mphi  = 0;   // = N_phi/2 + 1,R2C 複數樣本數
    int L_max   = 0;   // 球諧最高階,通常取 N_theta/3*2 或 min(N_theta-1, N_mphi-1)
    int n_lm    = 0;   // = (L_max+1)(L_max+2)/2,上三角 DOF
    int n_phys  = 0;   // = N_theta * N_phi
    int n_fper  = 0;   // = N_theta * N_mphi   (Fourier per-latitude complex field)

    double R = 1.0;    // 球半徑(影響 ∇² 特徵值 = -l(l+1)/R²)

    // ---- Gauss-Legendre 緯度節點/權重(host + device) ----
    std::vector<double> h_cos_theta;   // x_j = cos θ_j, size N_theta
    std::vector<double> h_weight;      // w_j,          size N_theta
    double* d_cos_theta = nullptr;
    double* d_weight    = nullptr;

    // ---- Precomputed associated Legendre table ----
    //   P_table[j, l, m] = P_l^m(x_j) · N_l^m  (已乘上 normalization)
    //   layout: row-major (j, idx_lm),idx_lm = l*(l+1)/2 + m,大小 N_theta × n_lm
    // 另存分離 m 的扁平:d_P_jm[j][idx_lm] 物理尺寸。
    double* d_P_table = nullptr;       // size N_theta * n_lm

    // ---- cuFFT plans ----
    // Batched R2C along φ:一次變換 N_theta 條緯度
    cufftHandle plan_r2c = 0;   // real(N_theta, N_phi) → complex(N_theta, N_mphi)
    cufftHandle plan_c2r = 0;

    // ---- scratch buffer ----
    // Fourier-per-latitude 複數 buffer,forward/backward 中間狀態使用
    cufftDoubleComplex* d_fmphi_tmp = nullptr;  // size n_fper

    // ---- 生命週期 ----
    void init(int N_theta_, int N_phi_, int L_max_, double R_);
    void destroy();

    // ---- 前向:物理 → 譜 ----
    // in_phys : (N_theta, N_phi) row-major double
    // out_lm  : n_lm 複數,idx_lm = l*(l+1)/2 + m 對應 (l, m)
    void spatial_to_spectral(const double* in_phys,
                             cufftDoubleComplex* out_lm);

    // ---- 逆向:譜 → 物理 ----
    // in_lm   : n_lm 複數
    // out_phys: (N_theta, N_phi) row-major double
    void spectral_to_spatial(const cufftDoubleComplex* in_lm,
                             double* out_phys);

    // 索引工具:(l, m) → flat idx (要求 m ≥ 0 且 m ≤ l)
    static inline int lm_to_idx(int l, int m) { return l * (l + 1) / 2 + m; }
};
