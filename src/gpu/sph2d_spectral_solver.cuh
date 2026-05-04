#pragma once

// 2D 薄球殼 barotropic 不可壓 Navier-Stokes 偽譜求解器(渦度-流函數)。
//
// 方程(絕對渦度形式):
//   ∂_t ζ + u·∇ζ + (2Ω/R²)·∂_φ ψ = ν·∇²_h ζ - α·ζ + forcing
//   ∇²_h ψ = -ζ   →  ψ̂_l^m = R²/(l(l+1)) · ζ̂_l^m
//   u_θ = -(1/(R·sinθ)) · ∂_φ ψ,   u_φ = (1/R) · ∂_θ ψ
// 其中 Ω 是球面自轉角速度,R 是球半徑,(2Ω/R²)·∂_φψ 來自 planetary vorticity 對流。
//
// 基底:球諧 Y_l^m(θ,φ),只存 m ≥ 0(Hermitian 對稱 for real field)。
// 時間積分:IFRK3 積分因子(Shu-Osher 低存儲),黏性/drag/hyper 在 (l,m) 各自對角積分。
// 對流:Jacobian J(ψ,ζ) 在物理空間算(θ 方向 central FD,φ 方向譜 ∂_φ)。
//
// 不完美處(標記為 Phase A 後續 iteration):
//   1. θ 方向 Jacobian 的 ∂_θ 用 2 階 central FD,不是全譜精度。
//      — 對 Rossby 波色散驗證(線性)無影響,對高 Re 湍流 cascade 準確度受限。
//      — 改進:用向量球諧展開速度場,完全保持譜精度。
//   2. Forward Legendre 是 O(N²) naive。L_max=127 可忍,L_max>255 時需要 SHTns。

#include "sph_transforms.cuh"
#include <curand_kernel.h>
#include <cufft.h>
#include <cstdint>
#include <string>
#include <vector>

struct Sph2DSpectralSolver {
    // ---- 網格 / 球諧基底 ----
    SphTransform T;
    int N_theta = 0, N_phi = 0, L_max = 0, n_lm = 0, n_phys = 0, n_fper = 0;
    double R = 1.0;

    // ---- 物理參數 ----
    double Omega = 0.0;         // 自轉角速度 (2Ω 是 Coriolis parameter)
    double nu    = 1e-4;        // 運動黏度
    double cfl   = 0.4;
    double dt_max = 5e-3;
    double dt_min = 1e-12;
    double drag_alpha = 0.0;
    int    hyper_p    = 1;
    bool   use_pi_dt  = false;
    double dt_prev    = 0.0;

    // ---- 譜空間狀態 ----
    cufftDoubleComplex* d_zeta_hat   = nullptr;   // ζ̂ 當前
    cufftDoubleComplex* d_psi_hat    = nullptr;   // ψ̂
    cufftDoubleComplex* d_rhs_hat    = nullptr;   // 當前級 rhs
    cufftDoubleComplex* d_k1_hat     = nullptr;   // RK3 起點拷貝
    cufftDoubleComplex* d_tmp_hat    = nullptr;   // scratch
    cufftDoubleComplex* d_zeta_hat_ic = nullptr;  // Rossby analytic 比對用

    // ---- 物理空間工作場 ----
    double* d_zeta       = nullptr;   // ζ(θ,φ)
    double* d_psi        = nullptr;   // ψ(θ,φ)
    double* d_dpsi_dphi  = nullptr;
    double* d_dzeta_dphi = nullptr;
    double* d_dpsi_dth   = nullptr;
    double* d_dzeta_dth  = nullptr;
    double* d_rhs_phys   = nullptr;

    // ---- 輔助 ----
    double* d_sin_theta  = nullptr;   // N_theta
    double* d_theta      = nullptr;   // N_theta
    std::vector<double> h_sin_theta;
    std::vector<double> h_theta;

    // ---- 診斷 ----
    double* d_reduce     = nullptr;
    int reduce_blocks    = 0;
    double* d_E_bins     = nullptr;   // 球諧能譜 E(l),大小 L_max+1

    // ---- forcing ----
    bool forcing_enabled = false;
    int    forcing_n     = 0;
    double forcing_eps   = 0.0;
    int    forcing_l_min = 0, forcing_l_max = 0;
    uint64_t forcing_seed = 0x5a5a5a5aULL;
    int*    d_forcing_idx   = nullptr;
    double* d_forcing_sigma = nullptr;
    curandStatePhilox4_32_10_t* d_forcing_states = nullptr;

    // ---- 書記 ----
    double dt_current = 0.0;
    int step_count = 0;

    // ---- Rossby analytic benchmark ----
    // 記錄 IC 的 (l, m) mode,analytic 傳播 ω_rw·t,
    // ∂_t ζ = -(2Ω·m)/(l(l+1)) · ζ (線性近似)
    // 即 ζ̂_l^m(t) = ζ̂_l^m(0)·exp(-i·ω_rw·t),ω_rw = -2Ω·m/(l(l+1))
    bool   has_rossby_ic = false;
    int    rossby_l = 0;
    int    rossby_m = 0;

    // ---- VRAM frame buffer ----
    double* d_frame_pool = nullptr;
    int frame_capacity = 0;
    int frame_count = 0;
    int total_frames = 0;
    std::vector<double> frame_times;
    std::vector<int>    frame_steps;
    std::string frame_out_dir;

    // ---- 生命週期 ----
    void init(int N_theta_, int N_phi_, int L_max_,
              double R_, double Omega_, double nu_, double cfl_);
    void destroy();

    // Rossby wave 單模 IC: ζ = amp · Re[Y_l^m(θ,φ)]
    void init_rossby_wave(int l, int m, double amp);

    // Jovian bands: 0 IC + 稀疏 narrow-band forcing (l ≈ l_forcing,m 任意)
    void init_zero();
    void init_forcing(int l_min, int l_max, double eps,
                      uint64_t seed = 0x5a5a5a5aULL);
    void apply_forcing(double dt);

    // IFRK3 one step
    double step();

    // 診斷結構
    struct Diagnostics {
        double total_KE;
        double total_enstrophy;
        double max_v;
        double max_zeta;
        double err_L2;        // Rossby analytic,否則 NaN
        double t_eval;
    };
    Diagnostics compute_diagnostics(double t_eval = 0.0);
    void compute_spectrum(std::vector<double>& out);    // E(l) 大小 L_max+1

    // I/O
    void download_zeta(std::vector<double>& h);
    void download_uv(std::vector<double>& h_ut, std::vector<double>& h_uphi);
    void write_vtk_2d(const char* filename);

    // VRAM frame pool
    void alloc_frame_buffer(int headroom_mb);
    void capture_frame(double t, int step);
    void flush_frames_to_disk(const std::string& run_dir);
    void free_frame_buffer();

    // Checkpoint
    void save_checkpoint(const std::string& path, double t);
    bool load_checkpoint(const std::string& path, double& t_out);
};
