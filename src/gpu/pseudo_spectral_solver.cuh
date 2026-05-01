#pragma once

#include <cufft.h>
#include <curand_kernel.h>
#include <cstdint>
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
    double dt_min = 1e-12;
    // Linear drag 係數 α:∂ω/∂t 加 -α·ω (破壞 condensate,允許 forced turb 穩態)。
    // 0 = 停用。啟用後 IFRK 積分因子變 exp(-(ν·k^(2p) + α)·dt·expo)。
    double drag_alpha = 0.0;
    // Hyperviscosity 冪次:-(-Δ)^p·ν·ω。p=1 為標準 Laplacian;p>1 把耗散壓到最高 k。
    // 注意:dt_visc_factor 下顯式擴散 CFL ∝ ν·|k|^(2p)·dt,p 大時 dt_max 要手動調。
    int    hyper_p = 1;
    // 積分因子 RK3 (IFRK3):黏性在譜空間解析積分為 exp(-νk²Δt)。
    // 啟用後 dt 只受對流 CFL 限制,可大幅放寬。預設 true。
    bool   use_ifrk = true;
    // Skew-symmetric 對流形式 N_S = ½(N_A + N_C) 取代 advective N_A。
    // 離散下嚴格保持能量+enstrophy (Orszag 1971)。代價:每級多 1 次 R2C FFT。
    // 預設 true;--ps-adv-only 強制回到 advective。
    bool   use_skew = true;
    // Conservative / rotational form (Basdevant 1986): N_C = ∂(uω)/∂x + ∂(vω)/∂y
    //   連續條件下 N_C ≡ N_A (∇·u=0 時);離散下也守 enstrophy,但只要 2 次 R2C。
    //   use_conservative 與 use_skew 互斥 (後者優先);兩者都 false 回到 advective。
    bool   use_conservative = false;
    // Batched FFT pipeline(cufftPlanMany + 連續 batch buffer + fused kernels)。
    // 實測 @ 4070 消費卡:1024² 慢 16%,2048² 慢 13%(cuFFT planMany 對消費卡無
    // 額外調度優化 + 連續 5/3 batch buffer 超過 L2 48MB → TLB 壓力上升)。
    // 保留作為 opt-in,將來換 A100/H100 datacenter 卡可能成為 winner。
    // 預設 false,--ps-batched-fft 啟用。
    bool   use_batched_fft = false;
    // PI controller (Söderlind 2003):dt 跟隨 max|v| 變化平滑調節,避免 CFL overshoot。
    // 預設 false (保當前可重現性);啟用後 dt_prev / err_prev 被維護。
    bool   use_pi_dt = false;
    double dt_prev = 0.0;
    double pi_cfl_target = 0.0;   // 啟用時自動設為 cfl;<=0 走預設。

    // ---- 物理空間 (double, ncell) ----
    // d_physbatch:連續 5-batch buffer,layout = [u | v | dwx | dwy | ω] × ncell
    //   offset: u=0, v=ncell, dwx=2*ncell, dwy=3*ncell, ω_copy=4*ncell
    // d_physbatch2:連續 3-batch buffer for skew,layout = [N_A | uω | vω] × ncell
    // 原始 d_omega 保留(物理 ω 的權威拷貝,供 diagnostic / checkpoint / VTK)
    double* d_physbatch  = nullptr;   // 5 * ncell
    double* d_physbatch2 = nullptr;   // 3 * ncell
    double* d_omega = nullptr;        // ω(權威)

    // 兼容舊 API 的指標別名(指向 d_physbatch 內偏移,不另外分配)
    double* d_u     = nullptr;   // = d_physbatch + 0*ncell
    double* d_v     = nullptr;   // = d_physbatch + 1*ncell
    double* d_dwx   = nullptr;   // = d_physbatch + 2*ncell
    double* d_dwy   = nullptr;   // = d_physbatch + 3*ncell
    double* d_Nphys = nullptr;   // = d_physbatch2 + 0*ncell (fused skew 時)

    // ---- 譜空間 (cufftDoubleComplex, ncplx) ----
    // d_specbatch:連續 5-batch buffer,layout = [û | v̂ | (dω/dx)^ | (dω/dy)^ | ω̂_derived] × ncplx
    //   對應物理 d_physbatch 的 5 個 field(batched IFFT)
    // d_specbatch2:連續 3-batch buffer,layout = [N̂_A | (uω)^ | (vω)^]  × ncplx
    //   對應物理 d_physbatch2(batched FFT)
    cufftDoubleComplex* d_specbatch  = nullptr;   // 5 * ncplx
    cufftDoubleComplex* d_specbatch2 = nullptr;   // 3 * ncplx
    cufftDoubleComplex* d_omega_hat = nullptr;    // ω̂ 權威
    cufftDoubleComplex* d_rhs_hat   = nullptr;    // 當前級 rhs
    cufftDoubleComplex* d_tmp_hat   = nullptr;    // scratch
    cufftDoubleComplex* d_k1_hat    = nullptr;    // RK3 y_orig 存檔
    cufftDoubleComplex* d_skew_hat  = nullptr;    // 舊名保留,現不再使用(等價 d_specbatch2+2·ncplx)

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
    cufftHandle plan_r2c = 0;    // single field (IC upload / forcing injection diag)
    cufftHandle plan_c2r = 0;    // single field (同上)
    // Batched plans(核心性能路徑):
    //   plan_c2r_b5  一次做 5 個 C2R (u, v, dwx, dwy, ω)
    //   plan_r2c_b3  一次做 3 個 R2C (N_A, uω, vω)        (skew path)
    //   plan_c2r_b4  一次做 4 個 C2R (u, v, dwx, dwy)      (advective/conservative)
    //   plan_r2c_b1  = plan_r2c(為了 symmetry)
    //   plan_c2r_b1_omega  單 ω̂ IFFT (conservative 要 ω 做 uω/vω, 物理空間)
    //   plan_r2c_b2  一次做 2 個 R2C (uω, vω)              (conservative path)
    cufftHandle plan_c2r_b5 = 0;
    cufftHandle plan_r2c_b3 = 0;
    cufftHandle plan_c2r_b4 = 0;
    cufftHandle plan_r2c_b2 = 0;

    // ---- 隨機相位強迫 (Lilly 1969 / Boffetta-Ecke 2012 風格) ----
    //   在薄殼 |k| ∈ [(kf-dk)·2π/L, (kf+dk)·2π/L] 注入 white-noise 相位,
    //   控制每步能量注入率 = ε_inj。
    //   Δω̂(k) = √dt · σ(k) · e^{iφ_rand}  (φ 每步重擲)
    //   σ 由 host 端在 init_forcing 算好,滿足 ε_inj = (Lx·Ly/N²)·σ²·Σ_shell 1/k²
    //   為保 Hermitian 對稱,只動 jc ∈ [1, nh-2] 的模 (conjugate 在 array 外)。
    bool    forcing_enabled = false;
    double  forcing_eps     = 0.0;   // 目標能量注入率
    int     forcing_kf      = 0;     // 中心波數 (mode 整數, 物理 k = 2π·kf/L)
    int     forcing_dk      = 1;     // 殼厚 (半寬 mode)
    int     forcing_n       = 0;     // 殼內模個數
    int*    d_forcing_idx   = nullptr;  // ncplx 索引 (size forcing_n)
    double* d_forcing_sigma = nullptr;  // σ 值    (size forcing_n)
    double* d_forcing_cos   = nullptr;  // host-path: 每步新 phase (size forcing_n)
    double* d_forcing_sin   = nullptr;
    uint64_t forcing_seed   = 0x5a5a5a5aULL;
    // cuRAND device-side path(預設 ON;避免 host mt19937+D2H 每步同步)
    bool    forcing_use_curand = true;
    curandStatePhilox4_32_10_t* d_forcing_states = nullptr;

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

    // ---- 譜 CSV (per frame;與 VTK 同步採樣) ----
    //   bin 數 nbins = min(nx,ny)/2 + 1;dk = 2π/max(Lx,Ly)
    //   每 capture_frame 呼叫時做一次 GPU ring-integrate,結果 push 進 frame_spectra
    //   flush 時連同 frames.csv 一起寫 spectrum.csv
    int nbins = 0;
    double dk_bin = 0.0;
    double* d_E_bins = nullptr;                    // GPU scratch (nbins doubles)
    std::vector<std::vector<double>> frame_spectra;

    // ---- 生命週期 ----
    void init(int nx_, int ny_, double Lx_, double Ly_, double nu_, double cfl_);
    void destroy();

    // 零場初始化 (所有 ω̂=0;通常配合 forcing 使用)
    void init_zero();

    // 設置隨機相位強迫:
    //   k_f:中心波數 (mode 整數,物理 |k| = 2π·k_f/L)
    //   dk :殼半寬 (mode)
    //   eps:目標能量注入率 (per unit area)
    //   seed:mt19937 種子 (同 seed 可重現)
    void init_forcing(int kf, int dk, double eps, uint64_t seed = 0x5a5a5a5aULL);

    // 生成新一組 phase (host mt19937) 並 add √dt·σ·e^{iφ} 到 ω̂
    void apply_forcing(double dt);

    // Kelvin-Helmholtz: 雙 tanh 剪切層 (對齊 cart_ale2::init_kh_shear 幾何)
    //   y ∈ (0.25 Ly, 0.75 Ly): vx = +vshear
    //   else                   : vx = -vshear
    //   vy seed: amp · sin(k·2π·x/Lx) × Gaussian bump (σ = 0.05 Ly) on each interface
    //   ω0 = ∂vy/∂x - ∂vx/∂y  由解析公式設置 (避免 FFT 微分損失精度)
    // 不可壓假設下 ρ=1,P 不參與演化(無熱力學)。
    void init_kh_shear(double vshear, double amp, int k);

    // Taylor-Green vortex (純擴散解析解):
    //   ω(x,y,0) = 2k·cos(k·2π·x/Lx)·cos(k·2π·y/Ly)
    //   ω(x,y,t) = ω(x,y,0)·exp(-2ν·k_phys²·t),  k_phys² = (k·2π/L)²·2
    // 對流項應嚴格為 0(u·∇ω 在 Taylor-Green 幾何上精確消去),只測 IFRK3 積分因子與空間階。
    // 啟用後會存 ω̂₀ 到 d_omega_hat_ic,compute_diagnostics 可額外算 err_L2。
    void init_taylor_green(int k);
    bool has_analytic_ic = false;
    int  analytic_k = 0;     // Taylor-Green 波數 mode
    cufftDoubleComplex* d_omega_hat_ic = nullptr;

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

    // 診斷量。2D 不可壓 NS 下:
    //   d(KE)/dt = -2ν·Ω               (動能耗散率 = 2ν × enstrophy)
    //   d(Ω)/dt  = -ν·∫|∇ω|² dA        (enstrophy 耗散率,對譜 2ν·∫k²·½|ω̂|²)
    // 後者可診斷 ν_eff:比較 d(Ω)/dt 的實測值 (由 Ω 數值微分) 和 ν·∫|∇ω|² dA,
    //   若兩者接近 → 數值粘性低;若實測遠大 → 有顯著數值耗散。
    struct Diagnostics {
        double total_KE;        // ∫½(u²+v²) dA
        double total_enstrophy; // ∫½ω² dA
        double max_v;
        double max_omega;
        double eps_KE;          // 動能耗散率 = 2ν·Ω + 2α·KE (drag 貢獻)
        double eps_enstrophy;   // enstrophy 耗散率 = ν·∫|∇ω|²p 項 (譜空間算)
        double eps_KE_spec;     // 譜空間重算的 eps_KE,用於交叉檢查
        double err_L2;          // |ω_num - ω_exact| L2 (解析解 IC 才有意義,否則 NaN)
        double t_eval;          // 傳入 compute_diagnostics 的當前 t (err_L2 需要)
    };
    Diagnostics compute_diagnostics(double t_eval = 0.0);

    // GPU ring-integrate: 算 E(k) per bin (長度 nbins),寫入 out。
    // 規則 (與 scripts/spectrum_pseudo_spectral.py 一致):
    //   Σ_k E(k)·dk = KE_total
    void compute_spectrum_bins(std::vector<double>& out);

    // ---- Checkpoint / Restart ----
    // 二進位 dump:header(magic,nx,ny,Lx,Ly,nu,step,t,forcing_seed_state) + ω̂ (ncplx·16B)。
    // save: 呼叫端提供當前 t;load: 回傳 t (解析器再 restore)。
    void save_checkpoint(const std::string& path, double t);
    bool load_checkpoint(const std::string& path, double& t_out);
};
