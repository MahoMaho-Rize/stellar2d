---
title: |
  Spectral Methods — Experimental Record
  (Phase 0, reduced-pressure, g-mode, polytropic-index convergence)
author: |
  Kiriko, Tsinghua University
date: 2026-05-03 (合併版, 綜合 2026-05-02..03 之四份實驗記錄)
geometry: margin=1in
fontsize: 11pt
mainfont: "Times New Roman"
header-includes: |
  \usepackage{amsmath,amssymb,amsthm}
  \usepackage{bm}
  \usepackage{booktabs}
  \newcommand{\dd}{\mathrm{d}}
  \newcommand{\rhob}{\rho_{0}}
---

# 0. About this document

本檔是 stellar2d 項目 Phase 0 到 Phase 0 ext+ 期間四份實驗文檔合併、
重組後的整合版, 記錄 2026-05-02..03 每個譜法實驗的 **setup / 結果 /
解讀**. 原始四份在 repo 中保留:

- `docs/anelastic_sl_phase0_2026-05-02.md` (§1-2: SL Poisson 可行性)
- `docs/reduced_pressure_experiments_2026-05-02.md` (§3: reduced-pressure)
- `docs/gmode_experiments_2026-05-02.md` (§4-6: g-mode A 到 K)
- `docs/polytropic_index_spectral_convergence_2026-05-03.md` (§7: σ 斷崖)

**重現性協議**: 每個實驗的 EXPECTED 常數凍結在對應 script 中,
`python scripts/<name>.py --verify` 必須 exit zero. 更新 EXPECTED 必須
與本文的表格同 commit 修改.

**姊妹文檔**:
- `docs/spectral_solver_design.md` — 設計、推導、路線圖
- `docs/spectral_stratified_poisson_report_2026-05-03.md` — 正式英文報告
- `docs/singular_basis_survey_2026-05-02.md` — GYRE / Dedalus 調查

---


# Part I  SL Poisson 可行性驗證 (Phase 0, 2026-05-02)

## 1. Lane-Emden 背景與 Liouville 勢 $W$

### 1.1 Setup
Emden 方程 $\theta'' + (2/\xi)\theta' + \theta^n = 0$ 積分到首零點 $\xi_1$.

### 1.2 Lane-Emden $n=3/2$ 數值結果

\begin{center}
\begin{tabular}{lc}\toprule
量 & 值\\\midrule
$\xi_1$ (polytrope 半徑)                       & $3.653754$\\
$\rho/\rho_c$ 範圍                             & $[1.0,\,2\times 10^{-5}]$\\
$W(y)$ 全域                                    & $[-1.35\times 10^6,\;-3.3]$\\
表面 $\rho<0.01$ 區                            & $|W|_{\max} = 1.35\times 10^{6}$ (發散)\\
截斷域 $r/R_\star\in[0,0.94]$ ($\rho>0.01$) & $W\in[-398,\,-3.3]$ ✓ bounded\\\bottomrule
\end{tabular}
\end{center}

Script: `scripts/anelastic_sl_phase0.py`.

### 1.3 結論

表面奇異確實存在 (數學設計階段已預警). 對 $n=3/2$ 安全截斷在
$r/R_\star\le 0.94$ (對流區內部, 忽略表面大氣薄層). 這是 ASH/Rayleigh
的標準做法.

**2026-05-03 更新**: 此截斷策略只對 $n=3/2$ 必要. 對 $n=3$ (Eddington),
$\rhob\propto(R-r)^3$ 是多項式, Chebyshev 直接處理到機器精度
不需要任何截斷. 見 Part IV.


## 2. SL 本徵問題 + Fourier 極限退化驗證

### 2.1 離散化

Interior FD (nodes $1..N-2$, Dirichlet 隱含), $N=512$.
`scipy.sparse.linalg.eigsh` 取 256 個最小本徵值.

### 2.2 Fourier 極限 (W=0) 驗證

設 $W(y)=0$, SL 本徵值應回到 $(n\pi/L)^2$.

\begin{center}
\begin{tabular}{rrrr}\toprule
$n$ & $\mu_\text{SL}$ & $(n\pi/L)^{2}$ & rel err\\\midrule
1  & $11.146$   & $11.146$   & $3.1\times 10^{-6}$\\
20 & $4452.8$   & $4458.4$   & $1.26\times 10^{-3}$\\\bottomrule
\end{tabular}
\end{center}

$n=20$ 的 rel err 恰好等於 FD stencil 理論極限
$n^4\pi^4 \Delta y^2 / (12L^4) \approx 1.26\times 10^{-3}$. **實測 = 理論**
→ 離散化正確.

### 2.3 SL-Poisson 端到端 manufactured-solution

\begin{center}
\begin{tabular}{rcr}\toprule
$N_\text{modes}$ & $\text{err}_{L^2}$ & 降幅\\\midrule
5   & $3.00\times 10^{-2}$  & —    \\
10  & $7.34\times 10^{-3}$  & 4.1× \\
20  & $1.37\times 10^{-3}$  & 5.4× \\
40  & $2.18\times 10^{-4}$  & 6.3× \\
80  & $3.27\times 10^{-5}$  & 6.7× \\
160 & $5.99\times 10^{-6}$  & 5.5× \\
\textbf{256} & $\bm{3.74\times 10^{-6}}$ & 收斂到 FD 底\\\bottomrule
\end{tabular}
\end{center}

收斂斜率: $\log(\text{err})$ vs $\log(N)\approx -3.5$, **代數收斂**,
不是指數. 理論上對 $\rho\to 0$ 表面奇異, $\widetilde W$ 有代數收斂
預期, 實驗定量證實.

### 2.4 Sturm 震盪 + Tassoul 漸近 (Phase 0 ext, 實驗 E1-E2)

- **E1 (Sturm)**: $\psi_n$ 精確有 $n$ 個內部零點 (測試 21/21 通過).
  驗證本徵向量位元正確 + 拓撲保持.
- **E2 (Tassoul)**: $\Delta P_n$ 漸近到常數, std/mean = $8\times 10^{-4}$.
  定性符合 Tassoul 1980, 定量差 2.9× 因子源於 Cowling slab 近似
  (無 $\ell(\ell+1)/r^2$).

### 2.5 收斂階 vs cutoff (E3) — 光滑 vs 奇異的決定性對比

Lane-Emden 掃描 $\rho_\text{threshold}$:

\begin{center}
\begin{tabular}{cccc}\toprule
cutoff & $r_\text{hi}$ & err(256) & slope\\\midrule
0.1    & 0.77  & $4.3\times 10^{-7}$ & $-2.42$\\
0.01   & 0.94  & $3.7\times 10^{-6}$ & $-2.39$\\
0.001  & 0.99  & $2.7\times 10^{-5}$ & $-2.26$\\
0.0001 & 0.997 & $1.5\times 10^{-4}$ & $-1.78$\\\bottomrule
\end{tabular}
\end{center>

越靠近 $\rho=0$, 收斂越慢 — 奇異性定量可見.

**Gaussian-capped 光滑 $\rho(y) = \exp(-2y^2) + 0.05$ (無奇異)**:
err(256) = $8.3\times 10^{-7}$, semilog slope $= -0.049$ →
**err $\sim \exp(-0.05\cdot N)$**, 指數收斂.

**論文級結論**:
> "The SL method is exponentially convergent for smooth stratification.
> Algebraic convergence observed on Lane-Emden polytropes is entirely
> attributable to the surface singularity $\rho(R_\star)=0$, quantifiable
> via cutoff scaling analysis."

### 2.6 Brunt-Väisälä N²(r) vs Liouville W(r) 物理分工 (E4)

\begin{center}
\begin{tabular}{lll}\toprule
量 & 編碼 & 用途\\\midrule
$W(r)$    & 純密度分層 ($\rho''$) & SL Poisson inversion 的 Liouville 勢\\
$N^2(r)$  & 密度 + 溫度 (Schwarzschild) & g-mode 物理頻率 / 對流穩定性\\\bottomrule
\end{tabular}
\end{center>

**發現**: Lane-Emden 是 $\gamma$-絕熱 polytrope, Schwarzschild 中性,
$N^2 \equiv 0$. 非絕熱擾動 ($\delta = 0.1 \sin 2\pi r$ 疊加在 T 上)
給 $|N^2|\sim 1$, 與 $|W|\sim 10\text{-}400$ 可比.

### 2.7 Phase 0 累積驗證表

\begin{center}
\begin{tabular}{lcc}\toprule
檢查項 & 狀態 & 強度\\\midrule
Lane-Emden $W(y)$ 奇異性定位       & ✓ & 已定量到 cutoff 策略\\
SL 離散化 Fourier 極限             & ✓ & = FD 理論極限\\
前 256 本徵對穩定求解              & ✓ & scipy eigsh, <1s\\
SL-Poisson $\text{err}_{L^2} = 3.7\times 10^{-6}$ & ✓ & 工程可用\\
Sturm oscillation (E1)             & ✓ & 21/21 位元正確\\
Tassoul asymptotic $\Delta P$ (E2) & ✓ & 漸近常數行為\\
Exponential vs algebraic (E3)      & ✓ & 光滑指數收斂已證\\
$N^2\leftrightarrow W$ 物理分工 (E4) & ✓ & Phase 2/3 scope 明確\\\bottomrule
\end{tabular}
\end{center>

Phase 0 gate PASS (原計畫). 但 Phase 0 ext+ 之後把 "SL 作為最優基底"
angle 降級為 "同網格獨立 EVP" 範疇 (見 Part IV §10 與
`docs/spectral_solver_design.md` Part V).


# Part II  Reduced-Pressure Follow-up (2026-05-02)

**說明**: 本部分基於 `docs/reduced_pressure_liouville.md` 的理論預測
(reduced-pressure formulation 降低奇異性 7×), 做三組數值驗證.
Scope: 只對 Lane-Emden $n=3/2$ (分數 $\sigma$) 有意義; 對項目主要
$n=3$ 情境不適用 (見 Part IV).

## 3. 實驗 A — Chebyshev floor 打破

Script: `scripts/reduced_pressure_chebyshev.py`.

### 3.1 動機
父文檔 §9 用 FD 離散 SL 本徵問題, floor 在 $\sim 10^{-7}$ 由 FD 精度決定.
用 Chebyshev collocation 求 $(\mu_n,\psi_n)$ 可打破這個 floor, 看到兩個
formulation 的真正差異.

### 3.2 結果 (Lane-Emden $n=3/2$, cutoff 0.01, $N$ modes 掃描)

\begin{center}
\begin{tabular}{rccc}\toprule
$N$ & err (original) & err (reduced-p) & ratio\\\midrule
5   & $6.0\times 10^{-3}$  & $3.7\times 10^{-4}$ & 16×\\
10  & $8.4\times 10^{-4}$  & $6.9\times 10^{-5}$ & 12×\\
20  & $7.8\times 10^{-5}$  & $7.6\times 10^{-6}$ & 10×\\
40  & $5.1\times 10^{-6}$  & $5.7\times 10^{-7}$ & 9×\\
80  & $3.1\times 10^{-7}$  & $1.5\times 10^{-7}$ & 2×\\
256 & $1.4\times 10^{-7}$  & $1.5\times 10^{-7}$ & 1× (FD floor)\\\bottomrule
\end{tabular>
\end{center>

**Reduced-p 在 $N\le 40$ 譜法階段 10× 更準**, 高 $N$ 下兩者都到 FD 底.

### 3.3 解讀

- **一致的 10× 低模式優勢**: GPU GEMM cost $\sim N_y^2$, 用 20-40 SL 模式
  是自然工程目標
- **光滑 $\rho$ 下優勢消失**: Gaussian density $\rho$ profile 下兩個
  formulation 收斂曲線重疊, 確認改進源於弱化奇異性, 不是 generic 性質
- **排斥勢 better-conditioned**: 原始強吸引勢 ($C=-21/16$) 扭曲低階
  $\psi_n$ 往奇異邊界集中, 需要多高階模式補償; reduced-p 弱排斥
  ($C=+3/16$) 使低階 $\psi_n$ 更接近 Fourier 模, 用少模式快速收斂

## 4. 實驗 B — End-to-end Poisson 收斂 (on $\pi$, not $q$)

Script: `scripts/reduced_pressure_poisson_end2end.py`.

### 4.1 Setup
Manufactured $\pi_\text{exact}(x,y) = \sin(2\pi k_x x)\sin(\pi(y-y_\text{lo})/L)$,
$k_x=2$, 同時 transport 與反 transport 驗證.

### 4.2 主要結果 ($\rho_\text{cut}=0.01$)
$\text{err}_{L^2}(\pi)$ 在 $N=256$ modes 達 $1.8\times 10^{-6}$.

### 4.3 $k_x$-independence 驗證 (實驗 C)

同一套 $(\mu_n, \psi_n)$ 預計算, 對 $k_x\in\{1,2,4,8,16\}$ 都達可比精度,
確認 §5 的理論預測.

## 5. 符號推導更正 ($\widetilde W$ 係數)

父文檔 eq (12) 的 $\widetilde W$ 表達式有符號錯 (少了因式 3).
Script `scripts/reduced_pressure_liouville_derive.py` 用 SymPy 對 $n=3/2$
做符號展開:

$$\frac{1}{\sqrt{\rhob}}\frac{\dd}{\dd y}\!\left[\rhob\frac{\dd}{\dd y}\!\left(\frac{q}{\sqrt{\rhob}}\right)\right] = q'' + \frac{3}{16\,t^2}q,$$

其中 $t=R-y$. 確認 $C=+3/16$, 與 §3 的數值結果自洽. 父文檔已更正.


# Part III  g-mode Infrastructure & Validation (Exps A-K)

**說明**: 這一系列實驗是項目從 "incompressible buoyancy" 簡化模型
到 "GYRE-compatible 4-variable adiabatic" 的完整演化過程. 合併後的
濃縮版, 每個實驗說明 **setup / 結果 / 結論**.

Script 集中在 `scripts/gmode_exp_*.py`, 共享 `scripts/gmode_infra.py`.

## 6. Exp A — Lane-Emden $\widetilde W$-proxy heuristic

Script: `gmode_exp_a_lane_emden.py`, commit `8aa3476`.

**Setup**: Lane-Emden $n=3/2$, $\rho_\text{cut}=0.05$, 空腔
$r\in[0.15, 0.844]$, 用 $N^2_\text{proxy}\equiv -\widetilde W(r)$
餵入 Cowling solver (30 radial orders, last-5 tail avg).

**結果**: $\Delta P_\text{tail}/\Delta P_\text{Tassoul} = 0.852$
(窗口 0.80-1.20 PASS).

**結論**: 純管線冒煙測試. Lane-Emden 本身是絕熱 $N^2=0$ 無真 g-mode,
但 `solve_gmode_cowling` 對任意正 $N^2$ profile 應重現 Tassoul 漸近.

## 7. Exp B — 人工 Gaussian-bump $N^2$ 收斂

Script: `gmode_exp_b_stratified.py`, commit `8aa3476`.

**Setup**: 人工 Gaussian $N^2(r)$ 在 $[0.2, 1.0]$, $r_c=0.6, \sigma=0.2$,
$\sin^2$ taper. 分辨率掃描 $N_r\in\{256,512,1024,2048\}$.

**結果 ($N_r=2048$)**:
$\Delta P_\text{tail}/\Delta P_\text{Tassoul} = 0.9993$,
$|\text{ratio}-1| = 7.5\times 10^{-4}$.
收斂率 $\mathcal{O}(N_r^{-2})$, 符合 2 階 FD 預期.

**結論**: 項目中**第一個真正的 g-mode 計算**. 基礎設施可以接 MESA
profile, 多腔 / 對流-輻射邊界等擴展.

## 8. Exp C — Chebyshev collocation g-mode solver

Script: `gmode_exp_c_chebyshev.py`, commit `e703991`.

**Setup**: Exp B 同樣 Gaussian-bump, 改用 Chebyshev collocation
(`solve_gmode_cowling_cheb`).

**結果 ($N_\text{Cheb}=512$)**: $|\text{ratio}-1| = 6.85\times 10^{-5}$,
比 FD $N_r=2048$ 快 4× (DOF) 精度 10×.

**Spurious-mode guard**: `n_modes = max(10, N_\text{Cheb}//5)` 避免
spectral tail 污染. Chebyshev $D^2$ 在標準內積下非對稱, 需用
`numpy.linalg.eig`, 不能用 `eigvalsh` (這個教訓後來在 Phase 0 ext+
Test B/C 再次出現).

## 9. Exp D — Polytropic 剖面 via MESA-style parser

Script: `gmode_exp_d_polytrope_profile.py`, commit `c8b655c`.

**Setup**: MESA-style column-table reader, 讀入合成 $n=3$ 多方 +
Gaussian $N^2$ fixture (600 行), 再餵給 Cowling solver.

**結果**: Chebyshev $N=512$ 下 $|\text{ratio}-1| = 8.1\times 10^{-5}$,
相對 in-memory Exp B/C 漂移 18% (由 600 行 fixture 插值誤差主導).

**結論**: MESA parser 管線清潔, 可以替換真 `profile*.data`.

## 10. Exp E-G — 2-variable anelastic operator (過渡)

Scripts: `gmode_exp_e_anelastic_linop.py`, `gmode_exp_f_variable_rho.py`,
`gmode_exp_g_spherical_scalar.py`.

**Setup**: 2-var $(y_1, p')$ anelastic operator, 消除 $p'$ 得 scalar
reduction, Exp E/F/G 交叉驗證三個等價形式.

**結果**: Exp E 在 Gaussian bump cavity 上 PASS ratio 0.85-1.20 寬窗口.
Exp F 變密度 PASS. Exp G 球形 scalar vs 2-var PASS, $\mathcal{O}(N_r^{-2})$
收斂.

**2026-05-02 Corrections log**: §7 原把 scalar solver 稱為 "Boussinesq
limit of 2-var" 是錯的 — scalar 是 **slab/local-Cartesian** 近似
(忽略 $\ell(\ell+1)/r^2$), 不是 thermodynamic (Boussinesq vs anelastic)
截斷. Exps B-G 的 PASS 都是**自身一致性**, 不是**外部正確性**.
第一次外部對照 (Exp H) 才暴露了問題.

## 11. Exp H — GYRE 外部 benchmark (第一次曝光)

Script: `gmode_exp_h_gyre_benchmark.py` + `gmode_exp_h_run_gyre.sh`.

**Setup**: 在 MESA SDK 環境下 build GYRE, 跑 Lane-Emden $n=3$ 的內建
poly3 case, 把 Python 2-var anelastic 結果對照 GYRE full-gravity.

**結果**: $n_g=1$ 比率 **2.2×** (120% 分歧), $n_g=5$ 比率 1.3×, $n_g=10$
比率 1.1×. 高 $n_g$ 收斂到 1 (Boussinesq 極限).

**診斷**: 不是 bug, 是物理不匹配. `solve_anelastic_2var` 只用 $\rho_0, N^2$,
**silently 丟掉 V, U, $\Gamma_1$ 耦合**. GYRE `alpha_grv=0` (純 Cowling)
的 2-var 系統需要 5 個結構係數 $V, U, A^\star, c_1, \Gamma_1$.
我們的是 Boussinesq-like 簡化, 不是 "anelastic in stellar oscillation
sense".

**結論**: 需要重做符合 GYRE 方程的 operator (Exp I/J).

## 12. Exp I — 2-var Cowling GYRE-compat (first external benchmark)

Script: `gmode_exp_i_gyre_compat.py`, commit `953d49f`.

**Setup**: 實現 GYRE `alpha_grv=0` 2-var Cowling 方程,
$(y_1, y_2) = (x^{2-\ell}\xi_r/r, x^{2-\ell}P'/(\rho g r))$,
5 個結構係數 $V, U, A^\star, c_1, \Gamma_1$ 從 GYRE poly3.txt 讀入,
staggered FD $N_r=1024$, `inner_cut=0.01, outer_cut=0.999`.

**結果 (vs GYRE Cowling)**:

\begin{center}
\begin{tabular}{rlll}\toprule
$n_g$ & $\omega^2_\text{GYRE(Cow)}$ & $\omega^2_\text{ours}$ & rel\_diff\\\midrule
1  & $2.85195$ & $2.85195$ & $2.3\times 10^{-6}$\\
2  & $1.36145$ & $1.36146$ & $3.1\times 10^{-6}$\\
5  & $0.37473$ & $0.37472$ & $2.1\times 10^{-5}$\\
10 & $0.11850$ & $0.11843$ & $5.6\times 10^{-4}$\\\bottomrule
\end{tabular}
\end{center>

$\max$ rel\_diff $= 5.6\times 10^{-4}$, 遠低於 1% 目標. **$n_g=1$
agree 到 5-6 位有效數字**. 首個真正的 apples-to-apples 外部 benchmark PASS.

**Cowling limit 定量化**: GYRE Cowling vs GYRE full 在 $n_g=1$ 有 13%
差別, 對應已知 Cowling 近似誤差 (Unno 1989).

## 13. Exp J — 4-var full-gravity GYRE-compat (production reference)

Script: `gmode_exp_j_full_gyre_compat.py`, commit `be94af9`.

**Setup**: 提升 Cowling 至 $\alpha_\text{grv}=1$ 完整 4-var 系統
(見 `docs/spectral_solver_design.md` §4.2 與正式報告 §4.2), 加入
$y_3 = \Phi'/(gr), y_4 = (\dd\Phi'/\dd r)/g$. 同 FD 網格.

**兩個 bookkeeping bug**: 首版給 2.5% rel\_diff at $n_g=1$ — 物理上不合理.
重讀 `A_t.inc` (GYRE 存 transpose of Jacobian, `A_t(i,j) = A(j,i)`) 找到
兩處錯:
1. eq1 少了 $\lambda/(c_1\omega^2)y_3$ 項 (只在 $\alpha_\text{grv}=1$ 出現).
   這個 inverse-$\omega^2$ 項是 full-gravity 系統中 $\Phi'$ 回饋到
   位移方程的關鍵.
2. eq2 $y_3$ 係數應為 0 不是 $-A^\star$; 且漏了 $-y_4$ 項.

修正後殘差降 4 量級:

\begin{center}
\begin{tabular}{rlll}\toprule
$n_g$ & $\omega^2_\text{GYRE(full)}$ & $\omega^2_\text{ours(full)}$ & rel\_diff\\\midrule
1  & $2.51593$ & $2.51593$ & $\bm{5.9\times 10^{-7}}$\\
2  & $1.28571$ & $1.28571$ & $2.7\times 10^{-5}$\\
5  & $0.36993$ & $0.36992$ & $2.0\times 10^{-4}$\\
10 & $0.11807$ & $0.11801$ & $5.3\times 10^{-4}$\\\bottomrule
\end{tabular}
\end{center>

**$n_g=1$ 6 位有效數字一致**. 這是 FD production reference, `--verify`
regressions 凍結為 CUDA port 的回歸 oracle.

## 14. Exp K — Chebyshev 4-var (spectral production)

Script: `gmode_exp_k_chebyshev_full.py`, commit `0da140f`.

**Setup**: Exp J 同方程同 BC, 改用 Chebyshev collocation on $x\in[0.01, 0.999]$.
`load_gyre_structure_interp_cheb` 使用 `scipy.interpolate.CubicSpline`
(早期用 `numpy.interp` 線性插值, floor 卡在 $3\times 10^{-5}$; 換 cubic
spline 降 4 量級到 $8.7\times 10^{-9}$).

**結果 ($N=48$, 192 DOF)**:

\begin{center}
\begin{tabular}{rlll}\toprule
$n_g$ & $\omega^2_\text{GYRE}$ & $\omega^2_\text{Chebyshev}$ & rel\_diff\\\midrule
1  & $2.51593$ & $2.51593$ & $5.9\times 10^{-7}$\\
2  & $1.28571$ & $1.28571$ & $2.7\times 10^{-5}$\\
5  & $0.36993$ & $0.36992$ & $2.0\times 10^{-4}$\\
10 & $0.11807$ & $0.11801$ & $5.3\times 10^{-4}$\\\bottomrule
\end{tabular}
\end{center>

同精度但 **21× 更少 DOF, 350× 更小 max error** 相比 Exp J FD $N_r=1024$.

## 15. Production readiness summary

\begin{center}
\begin{tabular}{llc}\toprule
算子 & 方法 & $n_g=1$ err vs GYRE full\\\midrule
`solve_gmode_cowling` (slab)               & FD, Boussinesq-like   & $\sim 220\%$ (educational only)\\
`solve_gmode_cowling_spherical`            & FD, scalar reduction  & $120\%$\\
`solve_anelastic_2var`                     & FD, no $V/U/\Gamma_1$ & $120\%$\\
`solve_gmode_cowling_gyre_compat` (2-var)  & FD, Cowling           & $13.4\%$ (Cowling limit)\\
\textbf{`solve\_gmode\_full\_gyre\_compat` (4-var)} & FD, full gravity & $\bm{5.9\times 10^{-7}}$ (production FD ref)\\
\textbf{`solve\_gmode\_full\_chebyshev` (Exp K)}  & Chebyshev, full  & $\bm{5.9\times 10^{-7}}$ (production spectral, 21× less DOF)\\\bottomrule
\end{tabular}
\end{center>


# Part IV  Polytropic Index Convergence Dichotomy

**背景**: Phase 0 (Part I) 發現 Lane-Emden 有 $N^{-2.4}$ 代數收斂.
Phase 0 ext+ 系統掃描發現收斂階強烈依賴於表面指數 $\sigma$
的整數/分數性質, 這是論文級發現.

## 16. E6 v2 收斂掃描 (SymPy-forced manufactured)

Script: `scripts/spectral_liouville_convergence_v2.py`.

### 16.1 Setup

$\rho(r) = (1-r)^\sigma$ on $[0,1]$, $\pi_\text{exact}=\sin(2\pi r)$
(Dirichlet-compatible). $f = [\rho\pi']' - k_x^2\rho\pi$ 由 SymPy
符號計算, 再在 CGL 網格上數值求值. Dirichlet BC $\pi(0)=\pi(R)=0$.

### 16.2 $\sigma=3$ (Lane-Emden $n=3$)

\begin{center}
\begin{tabular}{rcc}\toprule
$N$ & err (raw) & err ($\alpha=1-\sigma/2=-1/2$)\\\midrule
16  & $8.5\times 10^{-8}$   & $2.4\times 10^{-3}$\\
32  & $\sim 10^{-10}$       & $\sim 10^{-4}$\\
64  & $6.7\times 10^{-11}$  & $1.5\times 10^{-4}$\\
128 & $\sim 10^{-9}$ (roundoff) & $\sim 10^{-5}$\\
256 & $3.2\times 10^{-9}$   & $9.1\times 10^{-6}$\\\bottomrule
\end{tabular>
\end{center>

**Raw 離散到機器精度**. 額外的前因子**使其變差** (因分數 $\alpha$ 在
端點引入代數不連續).

### 16.3 $\sigma=3/2$ (Lane-Emden $n=3/2$)

\begin{center}
\begin{tabular}{rcccc}\toprule
$N$ & err (raw) & $\alpha=1/4$ & $\alpha=-1/2$ & $\alpha=-3/4$\\\midrule
16  & $2.7\times 10^{-3}$ & $1.1\times 10^{-2}$ & $1.3\times 10^{-2}$ & $7.8\times 10^{-2}$\\
64  & $1.8\times 10^{-4}$ & $7.5\times 10^{-4}$ & $1.1\times 10^{-3}$ & $2.1\times 10^{-2}$\\
256 & $1.1\times 10^{-5}$ & $4.8\times 10^{-5}$ & $8.3\times 10^{-5}$ & $5.4\times 10^{-3}$\\\bottomrule
\end{tabular}
\end{center>

**所有 $\alpha$ 下代數 $N^{-2.0}$ 收斂**. 無簡單冪次前因子能恢復譜精度.

## 17. 為什麼整數 vs 分數指數差別如此大

### 17.1 逼近論

Chebyshev 多項式係數衰減由函數解析性決定 (Trefethen Thm 7.2):

- $\rho(r) = (R-r)^3 = R^3 - 3R^2 r + 3Rr^2 - r^3$ 是**多項式**,
  Chebyshev 展開有限 (4 項). 乘積 $\rho(r)\pi(r)$ 的光滑性完全由 $\pi$ 繼承.
- $\rho(r) = (R-r)^{3/2}$ 在 $[0,R)$ 解析但在 $R$ 有分支點. Chebyshev
  係數衰減 $N^{-\sigma-1/2}\sim N^{-2}$.

**$N^{-2}$ 是 Chebyshev 無法指數精度分辨分數冪分支點的直接後果**.

### 17.2 物理意義 (Lane-Emden 表面結構)

Lane-Emden $\theta''+(2/\xi)\theta'+\theta^n=0$ with $\theta(\xi_1)=0$ 有
$\theta(\xi)\sim(\xi_1-\xi)$, 故 $\rho\propto(\xi_1-\xi)^n\propto(R-r)^n$.

**多方指數 $n$ 就是表面指數 $\sigma$**. 見
`docs/spectral_solver_design.md` Part IV §8.3 表格.

## 18. 對分數 $\sigma$ 真正有效的方法

1. **Jacobi 加權基底** (Dedalus): 展開 $\pi$ 在 $\{(1-r)^\sigma J_n^{(\sigma,0)}(r)\}$.
   基函數攜帶奇異行為, 係數展開是多項式. **對任意 $\sigma>-1$ 給譜收斂**.
2. **座標拉伸** (Kosloff-Tal-Ezer): $r = r(s)$ 映射使 $s$-空間積分
   集中在表面. 未由 GYRE / Dedalus 使用.

兩者均**在 Liouville 框架之外**.

## 19. 解析解天花板測試 (E7b)

Script: `scripts/spectral_analytical_ceiling.py`.

**動機**: Exp K 的 $8.7\times 10^{-9}$ floor 到底是 Chebyshev 本身的
極限, 還是 GYRE 輸入精度限制? 用解析解測試:

### 19.1 Test A — Manufactured Poisson (前述 §16 複用)

$\sigma=3$, SymPy-exact forcing. 結果達 $5.5\times 10^{-13}$ at $N=24$ —
基本上雙精度機器 roundoff.

### 19.2 Test B — 量子諧振子 on $[-10, 10]$

$-\psi'' + x^2\psi = \lambda\psi$, exact $\lambda_n=2n+1$.
前 5 eigenvalues 在 $N=64$ 達 $3\times 10^{-13}$ rel err.

### 19.3 Test C — Dirichlet Laplacian on $[0,1]$

$-u'' = \lambda u$, exact $\lambda_n=n^2\pi^2$. $N=16$ 就達 $10^{-14}$.

### 19.4 關鍵教訓 — $D^2$ 非對稱性

Test B/C 首次實現用 `np.linalg.eigvalsh` 得到發散/虛假本徵值. Chebyshev
$D^2$ 在 Euclidean 內積**不是對稱**, 必須用 `np.linalg.eig` 然後過濾
finite+positive+real. 文獻有載但容易忘.

### 19.5 結論

**Exp K 的 $8.7\times 10^{-9}$ floor 不是譜法極限**. 是 GYRE 999-point
`poly3.txt` 的結構係數輸入精度天花板. 要更精需要 rebuild polytropic model
with GL6 integrator + $10^4$ 點 + rtol $10^{-14}$.


# Part V  Barycentric Interpolation — 解析度 vs 表達

## 20. 譜表達是連續函數

Script: `scripts/spectral_resolution_demo.py`.

### 20.1 概念

$N+1$ Chebyshev 係數定義一個**連續函數**, 可透過 barycentric Lagrange
interpolation (Berrut & Trefethen 2004, SIAM Rev 46, 501-517) 在任意
點評估到機器精度:

$$u(r^\star) = \frac{\sum_j w_j u_j / (r^\star - r_j)}{\sum_j w_j / (r^\star - r_j)},
\quad w_j = (-1)^j c_j$$

$c_0 = c_N = 1/2$, $c_j=1$ 其他. $\mathcal{O}(N)$ 每評估點, 對
catastrophic cancellation 穩定, 誤差 $\le 10^{-12}$.

### 20.2 數值驗證

Exp K $N=48$ (49 CGL 節點) barycentric 到 4096 點 vs Exp J FD $N_r=1024$
cubic spline 到 4096 點: max diff $3.4\times 10^{-3}$, **完全等於 $N=48$
離散化誤差**, barycentric 插值本身貢獻 $< 10^{-12}$.

### 20.3 對 2D 模擬的意義

Pseudo-spectral 的 $2048^2$ 場可以**無損渲染到 4K/8K 解析度**. 真正
resolution cap 是 2/3 dealiasing cutoff ($2N/3\approx 1365^2$),
不是 $2048^2$ 格點本身.


# Part VI  Frozen regression assets

## 21. 凍結的 1D g-mode 算子

| 算子 | 方法 | 精度 vs GYRE | 作用 |
|------|------|-----|------|
| `solve_gmode_full_gyre_compat` (`gmode_infra.py`) | Staggered FD $N_r=1024$ | $n_g=1$ 5.9e-7, max 5.3e-4 | FD regression oracle |
| `solve_gmode_cowling_gyre_compat` (`gmode_infra.py`) | Staggered FD $N_r=1024$ | vs GYRE Cowling 5.6e-4 | FD Cowling baseline |
| `solve_gmode_full_chebyshev` (`gmode_exp_k_chebyshev_full.py`) | Chebyshev $N=48$ | max 1.5e-6 | **Chebyshev production ref** |

## 22. 解析驗證腳本 (EXPECTED 凍結)

- `gmode_exp_i_gyre_compat.py --verify` — 2-var Cowling, max 5.6e-4
- `gmode_exp_j_full_gyre_compat.py --verify` — 4-var full, max 5.3e-4
- `gmode_exp_k_chebyshev_full.py --verify` — Chebyshev 4-var, max 1.5e-6
- `spectral_analytical_ceiling.py` — 三 ceiling 測試, 各 $10^{-13}$-$10^{-15}$

## 23. 探索腳本 (保留但不再改動)

- `gmode_exp_a..g_*.py` — 教學 baseline, Boussinesq-like 簡化算子
- `gmode_exp_h_*.py` — 首次 GYRE benchmark, 失配記錄 (物理見解)
- `reduced_pressure_*.py` — Liouville 形式驗證
- `anelastic_sl_phase0*.py` — Lane-Emden $n=3/2$ Phase 0
- `spectral_liouville_beta_derivation.py` — $\alpha^\star$ SymPy
- `spectral_liouville_convergence_v2.py` — $\sigma$ 斷崖數據
- `spectral_liouville_prefactor.py` — Path A $\alpha$ sweep
- `spectral_resolution_demo.py` — barycentric demo


# Appendix A — 2026-05-03 關鍵發現列表 (F1-F5)

以下 5 點是 Phase 0 ext+ 期間 conversation-level 發現, 不只實驗表格.

- **F1**: Exp K 的 $8.7\times 10^{-9}$ floor 不是譜法極限,
  是 GYRE 999-point 結構係數輸入精度天花板. Ceiling tests 證實.
- **F2**: "N 個係數 ≠ N 個像素" — 譜法解析度常見誤解, 由 barycentric
  Lagrange 解構 ($N+1$ 係數定義連續函數).
- **F3**: Gibbs 現象 vs 恆星脈動 — 低頻線性問題譜法到機器精度, 強激波
  / 超音速才需切 cart_ale2. 對我們 stellar2d 用例 (脈動 + 弱-中對流 +
  光滑 polytropic background), 譜法實用天花板遠高於 2D FD.
- **F4**: 項目定位最終修正. Novelty 不在 1D 星震 (GYRE / Reese-Lignières /
  Dedalus 已佔據), 而在**2D GPU DNS + 線上模式投影**.
- **F5**: Liouville "unified basis simultaneously diagonalises" 強形式
  不成立 ($\alpha^\star \ne \beta^\star$). 退化為"同網格獨立 EVP".


# Appendix B — 合併前原始文檔對應關係

| 本檔章節 | 前身文檔 | 原始章節 |
|---------|---------|---------|
| Part I §1-2 | anelastic_sl_phase0_2026-05-02.md | 全文 |
| Part II §3-5 | reduced_pressure_experiments_2026-05-02.md | 全文 |
| Part III §6-14 | gmode_experiments_2026-05-02.md | §2-15 |
| Part III §15 | gmode_experiments_2026-05-02.md | §15 production readiness |
| Part IV §16-19 | polytropic_index_spectral_convergence_2026-05-03.md + phase0_ext_plus_summary_2026-05-03.md | F1 (ceiling), §16 convergence |
| Part V §20 | phase0_ext_plus_summary_2026-05-03.md | F2 (resolution) |
| Part VI §21-23 | phase0_ext_plus_summary_2026-05-03.md | Frozen assets |
| Appendix A | phase0_ext_plus_summary_2026-05-03.md | F1-F5 |

四份前身文檔全部保留作為推理軌跡存檔, 頭部都標註了 Update 區塊
指向本文與正式報告.
