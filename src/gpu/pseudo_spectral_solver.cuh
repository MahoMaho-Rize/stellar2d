#pragma once

#include <cufft.h>
#include <string>
#include <vector>

// 2D 不可壓縮 Navier-Stokes 偽譜法求解器 (GPU, cuFFT)。
//
// 渦度-流函數形式 (vorticity-streamfunction):
//   ∂ω/∂t + (u·∇)ω = ν ∇²ω
//   ∇²ψ = -ω,  u = ∂ψ/∂y,  v = -∂ψ/∂x
//
// 雙週期域 [0, Lx] × [0, Ly],Fourier spectral。
// 時間推進: 低存儲 Shu-Osher RK3 顯式。
// 反混疊 (dealiasing): 2/3 rule。
// 只為 KH 不穩定性演示,與 cart_ale/cart_ale2 完全獨立。
//
// 複用的 cart_ale2 設計模式:
//   - VRAM frame pool + binary VTK flush (高頻輸出不阻塞 GPU)
//   - Cartesian STRUCTURED_GRID VTK (兼容 render_cart_ale.py 風格)
//   - --test kh_shear IC 參數體系 (雙 tanh 剪切層 + 正弦擾動)

struct PseudoSpectralSolver {
    // ---- 網格 ----
    int nx = 0, ny = 0;          // 物理格點數 (row-major, flat = ic*ny + jc 以對齊 cart_ale2)
    int nh = 0;                  // (ny/2 + 1)  — R2C 後 y 方向複數樣本數
    int ncell = 0;               // nx * ny
    int ncplx = 0;               // nx * nh

    double Lx = 1.0, Ly = 1.0;
    double dx = 0.0, dy = 0.0;

    // ---- 物理參數 ----
    double nu  = 1e-4;           // 運動黏度 (實測 KH: 1e-4 ~ 1e-5)
    double cfl = 0.5;            // 對流 CFL 上限
    double dt_visc_factor = 0.5; // 顯式擴散 CFL 安全係數 (ν·dt·|k|²max ≤ factor)
    double dt_max = 5e-3;
    // 積分因子 RK3 (IFRK3):黏性在譜空間解析積分為 exp(-νk²Δt)。
    // 啟用後 dt 只受對流 CFL 限制,可大幅放寬。預設 true。
    bool   use_ifrk = true;

    // ---- 物理空間 (double, ncell) ----
    double* d_omega = nullptr;   // ω
    double* d_u     = nullptr;   // u = +∂ψ/∂y
    double* d_v     = nullptr;   // v = -∂ψ/∂x
    double* d_dwx   = nullptr;   // ∂ω/∂x
    double* d_dwy   = nullptr;   // ∂ω/∂y
    double* d_Nphys = nullptr;   // 對流項 N(ω) = u·∂ω/∂x + v·∂ω/∂y

    // ---- 譜空間 (cufftDoubleComplex, ncplx) ----
    cufftDoubleComplex* d_omega_hat = nullptr;
    cufftDoubleComplex* d_rhs_hat   = nullptr;    // 當前級 rhs
    cufftDoubleComplex* d_tmp_hat   = nullptr;    // scratch (ψ, du, dv 等)
    cufftDoubleComplex* d_k1_hat    = nullptr;    // RK3 中間暫存

    // ---- 波數 / dealias 遮罩 ----
    //  kx 大小 nx (含負頻率, frequency-shifted 的 R2C 約定: 0,1,..,nx/2,-nx/2+1,..,-1)
    //  ky 大小 nh (非負)
    double* d_kx      = nullptr;
    double* d_ky      = nullptr;
    double* d_dealias = nullptr;    // ncplx, 0/1 mask (2/3 rule)

    // ---- Reduction scratch (max|u|, max|v|, 動能) ----
    double* d_reduce = nullptr;
    int reduce_blocks = 0;

    // ---- cuFFT plans ----
    cufftHandle plan_r2c = 0;    // d_omega  → d_omega_hat
    cufftHandle plan_c2r = 0;    // hat      → physical

    // ---- 診斷/書記 ----
    double dt_current = 0.0;
    int step_count = 0;

    // ---- VRAM-buffered frame dump (複用 cart_ale2 設計) ----
    // 每幀儲存 3 個 cell-centered 場: ω, u, v  (3 × ncell × double)
    double* d_frame_pool = nullptr;
    int frame_capacity = 0;
    int frame_count = 0;
    int total_frames = 0;
    std::vector<double> frame_times;
    std::vector<int>    frame_steps;
    std::string frame_out_dir;

    // ---- 生命週期 ----
    void init(int nx_, int ny_, double Lx_, double Ly_, double nu_, double cfl_);
    void destroy();

    // Kelvin-Helmholtz: 雙 tanh 剪切層 (對齊 cart_ale2::init_kh_shear 幾何)
    //   y ∈ (0.25 Ly, 0.75 Ly): vx = +vshear
    //   else                   : vx = -vshear
    //   vy seed: amp · sin(k·2π·x/Lx) × Gaussian bump (σ = 0.05 Ly) on each interface
    //   ω0 = ∂vy/∂x - ∂vx/∂y  由解析公式設置 (避免 FFT 微分損失精度)
    // 不可壓假設下 ρ=1,P 不參與演化(無熱力學)。
    void init_kh_shear(double vshear, double amp, int k);

    // RK3 Shu-Osher 低存儲推進。回傳 dt。
    double step();

    // 下載到 host (向量大小將自動調整為 ncell)
    void download_omega(std::vector<double>& h);
    void download_uv(std::vector<double>& h_u, std::vector<double>& h_v);

    // ASCII Cartesian STRUCTURED_GRID VTK (debug / 小網格用)
    void write_vtk_2d(const char* filename);

    // ---- VRAM frame buffer API (介面鏡像 cart_ale2) ----
    void alloc_frame_buffer(int headroom_mb);
    void capture_frame(double t, int step);
    void flush_frames_to_disk(const std::string& run_dir);
    void free_frame_buffer();

    // 診斷量
    struct Diagnostics {
        double total_KE;        // ∫½(u²+v²) dA
        double total_enstrophy; // ∫½ω² dA
        double max_v;
        double max_omega;
    };
    Diagnostics compute_diagnostics();
};
