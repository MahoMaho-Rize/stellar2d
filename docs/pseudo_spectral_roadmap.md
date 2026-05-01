# pseudo_spectral 求解器完善路線圖與 MHD 擴展計劃

**日期**: 2026-05-01
**分支**: `pseudo-spectral` (from `cart_ale2-ppm-periodic`)
**目的**: context compact 前記錄當前狀態、未了事項、中長期走向;最終目標是擴展到天體物理 2D/3D MHD (吸積盤 MRI / ISM 湍流 / 太陽磁對流)

---

## Part 1. 當前狀態速查

### 1.1 已實現功能

**核心 2D 不可壓 NS 偽譜求解器** (渦度-流函數形式 ω-ψ):
- ✅ Fourier spectral in 雙週期 `[0, Lx] × [0, Ly]`
- ✅ cuFFT R2C/C2R (row-major: ic*ny+jc 物理, ic*nh+jc 譜)
- ✅ **IFRK3** (積分因子 RK3, Shu-Osher 低存儲) — 粘性 exp(-νk²Δt) 解析積分
- ✅ **Skew-symmetric convection** N_S = ½(N_A + N_C) — Orszag 1971 能量/enstrophy 嚴格守恆
- ✅ **圓形 2/3 dealias** `|k|² ≤ (N/3·2π/L)²` — 各向同性
- ✅ **IC Gibbs clean**: δ = max(4·dy, 0.01·Ly) + 初始譜套 dealias mask
- ✅ **ν_eff 診斷**: `eps_KE = 2ν·Ω` 對比實測 `-dKE/dt`,1024² Re=2×10⁵ 實測比值 **1.000** → 數值粘性 ≈ 0%
- ✅ **隨機相位 forcing** (Lilly/Alvelius):薄殼 √dt·σ·e^{iφ} 白噪聲注入,ε_inj 實測完美匹配
- ✅ VRAM frame pool + binary VTK (複用 cart_ale2 pattern)
- ✅ **譜 CSV 解耦 VTK**:per VTK frame GPU ring-integrate E(k) 寫 `spectrum.csv`;跨 run 秒級分析
- ✅ CLI 完整:`--ps-{nu,Lx,Ly,vshear,k,explicit,adv-only,forcing-{eps,kf,dk,seed}}`
- ✅ 渲染腳本 `scripts/render_pseudo_spectral.py` 雙 panel (ω, |v|)
- ✅ 譜分析腳本 `scripts/spectrum_pseudo_spectral.py` (多時間切片疊圖)
- ✅ 譜斜率擬合腳本 `scripts/spectrum_fit_pseudo_spectral.py` (帶 k_mode 參數,1.96σ CI 陰影)
- ✅ 純 CSV 分析腳本 `scripts/spectrum_from_csv.py` (fit/heatmap/overlay 三合一,auto-split k_η_ens 邊界檢查)
- ✅ VTK → CSV 回補腳本 `scripts/backfill_spectrum_csv.py` (舊 run 永久保留譜資訊)

### 1.2 檔案清單

```
src/gpu/
├── pseudo_spectral_solver.cuh      ~180 lines — struct + API + Diagnostics + forcing + spectrum bins
├── pseudo_spectral_solver.cu       ~900 lines — init/step/IC/diag/VTK/frame pool/forcing/spectrum.csv
└── pseudo_spectral_kernels.cu      ~380 lines — 14 CUDA kernels (新增 forcing + spectrum reduce)

scripts/
├── render_pseudo_spectral.py          ~305 lines — 1080p/1152p 渲染 (h264_nvenc, yuv420p high profile)
├── spectrum_pseudo_spectral.py        ~155 lines — 環積分能譜 (VTK-based,legacy)
├── spectrum_fit_pseudo_spectral.py    ~230 lines — VTK-based fit + Kraichnan CI (legacy)
├── spectrum_from_csv.py               ~280 lines — **純 CSV 分析** (fit/heatmap/overlay 三合一)
└── backfill_spectrum_csv.py           ~90  lines — 從既有 VTK 回補 spectrum.csv

docs/
├── pseudo_spectral_design_2026-05-01.md   完整設計/方法學
└── pseudo_spectral_roadmap.md             本文檔

CMakeLists.txt: 連結 CUDA::cufft,src/gpu/pseudo_spectral_*.cu 已納入 SOURCES
src/main.cpp:   --solver pseudo_spectral --test {kh_shear,forced_turb} 分支 (line ~1000)
```

### 1.3 已產出視頻/圖

`videos/`:
- `ps_kh_1024_tend100.mp4` — Re=10⁵ t=100 (舊版顯式)
- `ps_kh_1024_Re1e6_tend40.mp4` — Re=10⁶ t=40 (borderline DNS)
- `ps_kh_1024_Re2e5_newstack.mp4` — **新版** (IFRK+skew+圓 dealias+Gibbs) Re=2×10⁵ k=7 t=40
- `ps_kh_1024_Re2e5_newstack_spectrum.png` — 10 時間切片疊圖
- `ps_kh_1024_Re2e5_fit_t15_k7.png` — **decaying Kraichnan k^{-3} 驗證** (slope −3.45 ± 0.31, 1.4σ ✓)
- `ps_forced1024_kf8_tend30.mp4` — **Forced turb** 1024² k_f=8 ν=5e-5 tend=30 (39 MB, 10s, 302 frames)
- `ps_forced1024_heatmap.png` — **E(k,t) 熱圖**(condensate 建立過程清晰可見)
- `ps_forced1024_enstrophy_k4.png` — **Condensate regime k^{-4} 驗證** (雙 panel: 時間疊圖 + 補償譜)

### 1.4 數值方法驗證結果

**Re=2×10⁵ 1024² k=7 t=40 (新版全 stack)**:
| 量 | 值 |
|---|---|
| 步數 | 92254 (IFRK3, dt~4.3e-4) |
| KE 耗散 | 4.82% (40 時標) |
| **ν_eff / ν_code** | **1.000** (數值粘性貢獻 0%) |
| **dx/η** | **1.00** (完美 well-resolved DNS) |
| t=15 enstrophy slope | **−3.45 ± 0.31** (理論 −3, 1.4σ ✓) |
| Inverse cascade slope | −2.4 ~ −2.8 (decaying 常態,比 −5/3 陡) |

### 1.5 Forced turbulence 首輪驗證 (1024² k_f=8 ν=5e-5 ε=0.1 tend=30)

**參數**:N=1024²,k_f=8 mode,dk=1,ν=5e-5,ε_inj=0.1,tend=30 (231924 steps, ~24 分鐘)

| 診斷 | 實測 | 理論/解釋 |
|---|---|---|
| ε_KE(t=30) | 0.031 | 比 ε_inj 低 — 注入 > 粘性耗散,系統**未達穩態** |
| KE(t=30) | 2.30(仍線性增長) | condensate 填充中,dKE/dt ≈ 0.075 = ε_inj − ε_KE ✓ |
| Ω(t=30) | 313 | 小尺度穩態 (~250 持續 t=5 起) |
| **ε_Ω 穩定性** | 240 持續 t=5→30 | ✓ **小尺度 cascade 真正穩態** |
| k_η_ens | 356 rad/m | (ε_Ω/ν³)^{1/6} |

**Enstrophy cascade 斜率** (k ∈ [65, 300],4 時刻):

| t | slope | SE | R² |
|---|---|---|---|
| 5 | -4.14 | 0.06 | 0.993 |
| 10 | -4.15 | 0.07 | 0.991 |
| 20 | -4.08 | 0.07 | 0.990 |
| 30 | -4.04 | 0.06 | 0.992 |

**時間上完全穩定 → 真實譜律**

**補償譜檢驗** `E·k^p` 平整度(variability = max/min):
- `E·k³` (純 Kraichnan 1967):4.9×(失敗)
- `E·k³·ln^{1/3}` (Kraichnan 1971 log 修正):3.3×(略改善,仍不平)
- **`E·k⁴`:1.78×(最平 ✓)**

### 1.6 物理詮釋

我們的 run **不在** Kraichnan k^{-3} regime,而在 **condensate-modified enstrophy cascade** (Borue 1994, Bracco-McWilliams 2010):
- 沒有 large-scale drag → inverse cascade 能量堆在最低幾個 mode 形成 condensate vortex
- condensate 產生的強 strain 陡化小尺度譜形 → **k^{-4}** (而非 -3)
- 文獻報告值 -4.0 到 -4.5,我們的 -4.08 ± 0.06 **完全符合**

**結論**:
- 求解器小尺度物理 ✓ 正確(cascade 穩定性 + 譜律符合文獻)
- 求解器大尺度物理 ✓ 正確(condensate 形成符合無 drag 預期)
- 要拿 Kraichnan 經典雙段 (-5/3 + -3) 必須加 **linear drag -α·ω** 打破 condensate(下一階段)

---

## Part 2. 待完善的項目(在擴 MHD 之前)

按優先級排:

### 2.1 [✓ 部分完成] Forced turbulence — 現有 stochastic forcing, 下一步加 linear drag

**已完成** (commit `dac5b9d`):
- ✅ `k_apply_forcing` kernel: 薄殼 √dt·σ·e^{iφ} 白噪聲(host mt19937 per-step)
- ✅ σ 由 init_forcing 從 ε_inj 反解 (Alvelius 1999 約定)
- ✅ CLI `--ps-forcing-{eps,kf,dk,seed}`
- ✅ `--test forced_turb` (零 IC + forcing) 加入 test whitelist
- ✅ 驗證:實測 ε_KE 完美匹配 ε_inj(512² run ε_KE=0.100 = ε_inj=0.1)

**現狀限制**:
- 沒有 large-scale drag → 系統永不達到完全穩態 (condensate 無限成長)
- 當前得到 -4.08 譜律(condensate regime),不是 Kraichnan -3
- 跑到 tend=30 KE 仍線性增長(9% of 飽和值),tend 需 > 200 才接近飽和

**下一步:加 linear drag -α·ω**(~20 LOC):
- IFRK3 積分因子擴展:exp(-νk²Δt) → exp(-(νk² + α)·Δt)
- CLI `--ps-drag <α>`,預設 0(不啟用,保當前行為可重現)
- 建議 α = 0.1:τ_drag = 10,配合 tend=30 達真正穩態
- 預期結果:
  - k < k_f:**Kraichnan k^{-5/3}** (inverse cascade to drag sink)
  - k > k_f:**Kraichnan k^{-3}** (enstrophy cascade, drag 不影響小尺度)
  - 兩段同時達穩態(小尺度 cascade + inverse cascade 被 drag sink 吸收)

### 2.2 [中] Taylor-Green 收斂測試

**動機**: 目前只有 end-to-end smoke test,沒有**解析解驗證**。Taylor-Green vortex 是 2D NS 有精確解析解 `ω(t) = ω₀·exp(-2νt)` 的最經典 benchmark。

**實作** (~50 行,新增 `init_taylor_green`):
```
ω(x,y,0) = 2k·cos(k·x)·cos(k·y)
ω(x,y,t) = 2k·cos(k·x)·cos(k·y)·exp(-2νk²t)
```
純 diffusion,對流為零。**測 IFRK3 積分因子的絕對正確性**。預期誤差 < 1e-10 (double precision floor)。

**新 CLI**: `--test taylor_green`, `--ps-tg-k <int>` (預設 2)。
**新輸出**: `diagnostics.csv` 多一欄 `err_L2` = |ω_num - ω_exact| L2 範數。

### 2.3 [中] 收斂階測試 (spatial + temporal)

**動機**: 聲稱 IFRK3 是 3 階,SSP-RK3 也是 3 階,光譜法 N 變大應得**指數收斂**。現在沒數字支持。

**實作** (~30 行腳本):
- 固定 t_end,掃 N = {64, 128, 256, 512, 1024},算 err_L2 → 畫 log-log → 斜率應 -∞(exp 收斂)
- 固定 N,掃 dt_factor = {1, 0.5, 0.25, 0.125},算 err_L2 → 畫 log-log → 斜率應 -3

**檔案**: `scripts/convergence_pseudo_spectral.py`。搭配 Taylor-Green。

### 2.4 [中] 能譜時間演化熱圖 (k–t 圖)

**動機**: 單個時刻的譜看不出 cascade 動態。經典呈現是 `E(k, t)` 作為 k-t 平面的 colormap,能一眼看出能量如何在 k 空間移動。

**實作** (~50 行,新 script):
- 所有幀提取 E(k),拼成 2D (nk, nt) 矩陣
- 畫 log₁₀ E 的 pcolormesh,疊 k_inj 和 k_η 曲線
- **cascade 動態可視化的標準呈現**

### 2.5 [低] Hyperviscosity 選項

**動機**: 高 Re 下解析度不足時常用 `-ν_p·(-Δ)^p·ω` (p=4~8) 取代 Laplacian,把耗散壓到最高 k 幾個 bin。IFRK3 框架可直接擴展:只改 `k_ifrk_combine` 的指數因子 `exp(-ν_p·k^{2p}·Δt·expo)`。

**CLI**: `--ps-hyper <p>` (預設 1 = 標準 Laplacian)。

### 2.6 [低] 自適應 dt(PI controller)

**動機**: 目前 dt = CFL·dx/max(|v|) 估計一次,若 max|v| 突變可能 overshoot。工業代碼常用 Söderlind 2003 PI controller。

**影響有限** (現有 runs 穩定),**最低優先**。

---

## Part 3. MHD 擴展的詳細規劃

### 3.1 物理目標與科學問題

目標對象(按天體物理重要性):

| 問題 | 參數 | 意義 |
|---|---|---|
| **2D MHD 湍流** (Biskamp 2003) | decaying,Pm = ν/η = 1 | 湍流-磁場相互作用的最基本 benchmark |
| **ISM 2D MHD turbulence** | forced, Pm~1, 強 guide field | 星際介質分子雲,恆星形成 |
| **MRI 剪切盒** | shearing box + rotation + guide B | **吸積盤角動量傳輸** — MHD 的最大應用 |
| **太陽/恆星磁對流** | + 溫度 (Boussinesq) + 重力 | 表面對流 dynamo |
| **磁重聯** | 反向 B 場 IC | 耀斑、地磁尾、脈衝星磁層 |

最務實起點:**2D 不可壓 MHD decaying turbulence** (3.2 節)。

### 3.2 2D 不可壓 MHD 的數學

**狀態變量**:渦度 ω + 磁向量勢 ψ_B(2D 磁場可用標量勢)
- B = ∇ × (ψ_B · ẑ) = (∂ψ_B/∂y, -∂ψ_B/∂x)
- j_z = (∇ × B)_z = -∇²ψ_B (電流密度)

**支配方程** (Elsässer 形式的簡化):
```
∂ω/∂t + u·∇ω = ν∇²ω + (B·∇)j_z                    (Lorentz 力 vorticity 源)
∂ψ_B/∂t + u·∇ψ_B = η∇²ψ_B                          (磁感應方程)
∇²ψ = -ω,  u = (∂ψ/∂y, -∂ψ/∂x)                    (流函數)
```

兩個 evolved 場:ω (渦度) + ψ_B (磁勢)。每步要算 **兩套 rhs**,共用 u 但 B 的梯度也要 FFT。

**譜空間**:
```
∂ω̂/∂t = -N̂_ω(ω̂, ψ̂_B) - ν|k|²·ω̂
∂ψ̂_B/∂t = -N̂_B(ω̂, ψ̂_B) - η|k|²·ψ̂_B

N_ω = u·∇ω - (B·∇)j_z         (渦度 = 流對流 + Lorentz)
N_B = u·∇ψ_B                   (磁勢 = 被動平流 by u)
j_z 譜 = |k|²·ψ̂_B
```

### 3.3 IFRK3 在 MHD 的使用

**黏性與磁擴散都是 Laplacian** → 各自用積分因子:
- ω 分支用 `exp(-ν|k|²Δt)`
- ψ_B 分支用 `exp(-η|k|²Δt)`

兩個獨立 RK3 級耦合在同一 timestep(不能拆分,因為 rhs 互相依賴 via B/u)。

### 3.4 新求解器檔案結構(遵循不覆蓋原則)

```
src/gpu/
├── mhd2d_solver.cuh          新增 (fork pseudo_spectral)
├── mhd2d_solver.cu           新增
└── mhd2d_kernels.cu          新增
```

**關鍵:不在 pseudo_spectral_* 內加磁場**,開新 solver。`pseudo_spectral` 保留為「最小 2D 不可壓 NS」基線。

新 struct 欄位(相比 PseudoSpectralSolver 新增):
- `d_psiB`, `d_psiB_hat`, `d_psiB_rhs_hat`, `d_psiB_k1_hat` (4 buffer)
- `d_jz` (物理電流)
- `d_Bx, d_By` (物理磁場;或動態從 ψ_B 算)
- `d_skewB_hat` (skew for N_B)
- `eta` (磁擴散率)

新 Diagnostics:
- total_ME (磁能 = ½∫|B|² dA)
- total_total_E = KE + ME
- eps_ME = 2η·∫|j_z|² dA/2 (磁耗散率)
- cross_helicity = ∫u·B dA

### 3.5 新 CLI

```
--solver mhd2d
--test orszag_tang        # 經典 OT vortex IC (Orszag & Tang 1979)
--test current_sheet      # 磁重聯
--test decaying_mhd_turb  # 隨機 IC + 衰減

--mhd-nu <ν>         運動黏度
--mhd-eta <η>        磁擴散率
--mhd-Pm <Pm>        Prandtl magnetic (η = ν/Pm, 覆蓋 --mhd-eta)
--mhd-B0 <B>         guide field (加均勻 B_x 基底)
```

### 3.6 標杆 IC: Orszag-Tang vortex

文獻最常 cite 的 2D MHD benchmark:
```
ω₀ = 2(sin(x) + sin(y))
ψ_B₀ = cos(x) + cos(2y)/2
```

跑 t ~ 0.5–2,看**電流片 (current sheet)** 形成、**磁重聯**、磁能-動能轉換。
和 Dahlburg-Picone 1989、Strauss-Monticello 1981 結果對比。

### 3.7 驗證指標

| 指標 | 預期 |
|---|---|
| 總能量守恆 | t=0 → 終止:<2% 耗散 (Pm=1, Re=Rm=10⁴) |
| Cross helicity | Ideal MHD 不變量,有 η/ν 時緩慢衰減 |
| 電流片寬度 | ~ η^{1/2} scaling |
| 磁重聯率 | Sweet-Parker:~ S^{-1/2},Petschek:~ ln(S)^{-1} |

### 3.8 實作工作量估計

| 階段 | 新增 LOC | 時間估計 |
|---|---|---|
| fork pseudo_spectral → mhd2d (框架) | ~400 | 半天 |
| 雙 field IFRK3 coupling | ~150 | 半天 |
| 新 kernels (N_ω with Lorentz, j_z 算、skew for B) | ~250 | 1 天 |
| IC: Orszag-Tang + current sheet | ~100 | 半天 |
| Diagnostics (ME, cross helicity, eps_ME) | ~100 | 半天 |
| CLI + main dispatch | ~80 | 半天 |
| Render script (3/4 panel: ω, j_z, B 向量線) | ~200 | 半天 |
| Verification with OT | - | 1 天 |
| **合計** | **~1300 LOC** | **~5 天** |

### 3.9 可能的坑 (預先踩)

1. **skew-symmetric for Lorentz**: `(B·∇)j_z` 也要用對稱形式嗎? 答:MHD 文獻傳統用 **Elsässer** 變量 z± = u ± B 把方程對稱化,skew 的關鍵是**對 u 和 B 分別保結構**。最保險先做 advective 驗證,再加 skew。

2. **Initial ∇·B = 0 自動保證**:2D 用向量勢時 `B = ∇ × (ψẑ)` 自動無散度。3D 要 projection 清除 ∇·B(成本高很多)。

3. **Magnetic CFL**: Alfvén 速度 c_A = B/√ρ,對流 CFL 要改成 `dt = cfl·dx/max(|v|+c_A)`。否則快磁聲波會 blow up。

4. **兩個 IFRK3 level 一致性**: ω 和 ψ_B 共用同一個 dt(不能分 multirate,因為 rhs 耦合)。但 ν 和 η 不同 → 用各自指數因子。

### 3.10 後續擴展路徑(MHD 之後)

有了 2D MHD 基礎後可做:

- **3D MHD** (~2 週):cuFFT 3D plan,VRAM 吃緊,256³ 勉強 1024³ 要 H100
- **剪切盒 / MRI** (~3–5 天):加 Coriolis 項 + 線性剪切源項 (Lesur-Longaretti 2007)
- **Boussinesq MHD**: +溫度場 → 太陽磁對流模型
- **Relativistic MHD**: 要可壓縮 + 完整 stress tensor,偽譜不是最佳選擇(有限體積 + HLLD 更自然)

---

## Part 4. 工作順序建議

### Phase A: 完善現有偽譜法(1–2 週)

1. **Forced 2D NS turbulence**(2.1)
   - ✓ Stochastic forcing 實作 (commit `dac5b9d`)
   - ✓ spectrum.csv 解耦 VTK
   - ✓ 1024² first run:condensate-regime k^{-4} 驗證(符合 Borue 1994)
   - **下一步**:加 linear drag → Kraichnan k^{-5/3} + k^{-3} 經典雙段
2. **Taylor-Green convergence test**(2.2 + 2.3) → 定量誤差數字,spectral convergence 曲線
3. **E(k,t) 熱圖**(2.4) → ✓ **已免費送達** via spectrum_from_csv.py heatmap mode
4. **整理 docs/**:把當前這份 roadmap 之外再補一份 `pseudo_spectral_validation.md` 收錄所有 verification 數字

### Phase B: 2D MHD 基礎(~5 天)

5. **fork mhd2d solver**(3.4)
6. **Orszag-Tang benchmark**(3.6)—— 跑出來和文獻對比
7. **Decaying MHD turbulence**(1024² Pm=1 Rm=10⁴)
8. **Forced MHD turbulence**(複用 Phase A 的 forcing 機制)

完成後**有磁流體演示**,可對標 Biskamp 2003、Politano-Pouquet 1995。

### Phase C: 天體物理特化(~1 週+)

9. **Shearing box**(加 Coriolis + 剪切源項)
10. **2D MRI 截斷版**(Hawley-Balbus 1992 的 2D pre-cursor)
11. **或者 3D 不可壓 MHD**(若硬體允許)

完成後**有一個能跑 MRI 的 mini-SNOOPY**(對比 Lesur 2007)。

---

## Part 5. 重要設計決策記錄

為防 compact 後丟失 context,記錄關鍵決策及其**理由**:

### 5.1 為什麼用 ω-ψ 而非 primitive (u, v, p)?

- **省一個場**:ω-ψ 只有 1 個標量 (ω),primitive 需 3 個 (u, v, p) + 每步解 pressure Poisson
- **消壓力**:2D 不可壓壓力完全由 velocity 決定,沒獨立演化;解 Poisson 只是為了 enforce ∇·u=0,ω-ψ 自動滿足
- **weakness**:擴 3D 不自然(ω 變 3 分量 vector),MHD 擴展需加 ψ_B 對應結構

### 5.2 為什麼用積分因子(IFRK)而非 IMEX?

- **對純 Fourier + 各向同性 Laplacian,IFRK 是對角積分,逐模精確**
- IMEX (ARS343, CNAB2) 要解線性系統,對於 Fourier Laplacian 也是對角的,所以**理論上等價**
- IFRK 實作更簡單,只需 `exp(-νk²Δt)` 指數因子
- **局限**:若加 hyperviscosity `-ν_p(-Δ)^p` 也 OK (只改指數);若 operator 非對角(如 variable ν),IMEX 更自然

### 5.3 為什麼 skew-symmetric 而非 rotational form?

SpectralDNS (Mortensen 2016) 用 rotational `u × ω + ∇(½u²)`,對 primitive velocity eq 最自然。
我們用 ω-ψ,對流項是 `u·∇ω`,skew-symmetric 是 ω 方程的經典對應 (Orszag 1971)。

### 5.4 為什麼 R2C 而非 C2C FFT?

- **記憶體省一半** (實信號對稱,只存非負 ky)
- cuFFT R2C 路徑成熟,性能和 C2C 相當
- **代價**: index 要小心 `nh = ny/2+1`

### 5.5 為什麼不追 3D?

- VRAM 爆炸:1024³ = 10⁹ cells × 8 bytes × 多 buffer → 80GB+,消費卡不夠
- 現有 4070 (12–16 GB) 最多 256³ × ~10 buffer 就快滿
- **3D 是 H100/A100 等級硬體的遊戲**,單卡玩不動
- 所以本專案定位:**minimal pedagogical 2D spectral solver + MHD 擴展**

---

## Part 6. 關鍵檔案索引(給未來自己 / 給 agent)

### 源碼
- `src/gpu/pseudo_spectral_solver.cuh` — struct API,所有 flag 解釋在此
- `src/gpu/pseudo_spectral_solver.cu:228` — `compute_rhs_adv_only` (核心 rhs,skew 邏輯)
- `src/gpu/pseudo_spectral_solver.cu:370` — `step()` (IFRK3 / 顯式分支)
- `src/gpu/pseudo_spectral_solver.cu:440` — `compute_diagnostics` (eps_KE, eps_enstrophy)
- `src/gpu/pseudo_spectral_kernels.cu:k_ifrk_combine` — IFRK3 線性組合(關鍵:expo_a, expo_b 參數)

### 文檔
- `CLAUDE.md` — 資產保留原則,**千萬不要覆蓋已有 solver**
- `docs/pseudo_spectral_design_2026-05-01.md` — 第一版設計快照
- `docs/pseudo_spectral_roadmap.md` — **本文檔**
- `memory/feedback_preserve_solvers.md` — 保留求解器資產的 feedback memory
- `memory/feedback_cleanup_runs.md` — 清理 runs/ 只刪 VTK 不刪 CSV

### 腳本
- `scripts/render_pseudo_spectral.py` — VTK → MP4(h264_nvenc, yuv420p high)
- `scripts/spectrum_pseudo_spectral.py` — E(k) 多時間切片
- `scripts/spectrum_fit_pseudo_spectral.py` — **記得傳 k_mode 參數**,否則 auto 會誤判

### Commit 鏈(本輪工作)
```
c83f30f  DOC: pseudo_spectral design & progress note
9eaa30e  DOC: CLAUDE.md 补列 cart_ale2 + pseudo_spectral 求解器资产
74275a4  ADD: 譜斜率定量回歸腳本 + Kraichnan 擬合驗證
60785a6  FIX: spectrum_fit 腳本用 IC 擾動模數 k_mode 而非譜峰作 k_inj
```

---

## Part 7. 快速上手指令

**編譯**:
```bash
pixi run build-gpu
```

**跑目前最好看的 KH 算例 (Re=2×10⁵, k=7, IFRK+skew)**:
```bash
./build/stellar2d --solver pseudo_spectral --test kh_shear \
  --nr 1024 --ntheta 1024 --tend 40 \
  --ps-nu 5e-6 --ps-vshear 0.5 --ps-k 7 --perturb 0.1 \
  --output-interval 2000 --vtk-dt 0.02 \
  --frame-buffer --frame-headroom-mb 2048
# ~15 分鐘
```

**跑 forced turbulence (1024² condensate regime, -4 enstrophy cascade)**:
```bash
./build/stellar2d --solver pseudo_spectral --test forced_turb \
  --nr 1024 --ntheta 1024 --tend 30 \
  --ps-nu 5e-5 --ps-forcing-eps 0.1 --ps-forcing-kf 8 --ps-forcing-dk 1 \
  --output-interval 2000 --vtk-dt 0.1 \
  --frame-buffer --frame-headroom-mb 2048
# ~25 分鐘,302 VTK frames + spectrum.csv
```

**渲染**:
```bash
pixi run python scripts/render_pseudo_spectral.py \
  runs/kh_shear_1024x1024_<ts> \
  videos/out.mp4 30 12
```

**譜分析**:
```bash
# 疊圖
pixi run python scripts/spectrum_pseudo_spectral.py runs/<ts> videos/spec.png 10

# 擬合 (注意傳 k_mode!)
pixi run python scripts/spectrum_fit_pseudo_spectral.py runs/<ts> videos/fit.png 15 7
# 15 = t_snap, 7 = k_mode (對應 --ps-k)
```

**清理** (feedback memory:只刪 VTK 保 CSV):
```bash
find runs/ -name "*.vtk" -delete
```

---

## Part 8. 給 future session 的 context pointer

若 compact 後新 session 問「這個專案在做什麼」:
1. 讀 `CLAUDE.md` 了解求解器資產保留原則
2. 讀本文檔(`docs/pseudo_spectral_roadmap.md`)了解當前階段
3. 讀 `docs/pseudo_spectral_design_2026-05-01.md` 了解方法學細節
4. 用 `git log --oneline pseudo-spectral -20` 看最近 commit
5. 用者偏好**簡體中文註解、繁體中文對話**,commit 用 `ADD:/FIX:/DOC:/OPT:` 前綴

**當前最大方向**:Phase A(Forced turb + Taylor-Green 驗證)→ Phase B(2D MHD Orszag-Tang)→ Phase C(MRI / 3D)。

**千萬不要做的事**:
- 在 pseudo_spectral_*.cu 內塞磁場/重力/溫度(要開新 solver)
- 修改任何既有 solver(strang, cart_ale*, fas, lowmach 等)
- `rm -rf runs/*`(要只刪 *.vtk,保 CSV)
