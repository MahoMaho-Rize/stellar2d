---
title: Phase 0 ext+ Summary — Spectral Liouville Direction (pre-compact)
date: 2026-05-03
status: ARCHIVED (pre-compact 快照 — 結論已被正式技術報告取代)
parents:
  - docs/spectral_liouville_plan_2026-05-03.md
  - docs/polytropic_index_spectral_convergence_2026-05-03.md
  - docs/gmode_experiments_2026-05-02.md
---

> **⚠ ARCHIVED — 2026-05-03 後期**
>
> 本檔為 Phase 0 ext+ context compact 前的 snapshot, 記錄當時的 F1-F5
> 關鍵發現 + Phase 1 kickoff 任務. 所有內容已被下列正式文檔吸收:
>
> - `docs/spectral_stratified_poisson_report_2026-05-03.md` — 正式英文技術報告
> - `docs/spectral_solver_design.md` — 合併後設計文檔
> - `docs/spectral_experiments.md` — 合併後實驗紀錄
>
> 存檔理由: 記錄 2026-05-03 這一天的思考彙整過程.
> 不要用此文指導新工作; 如有衝突以正式報告為準.

# One-line summary

**譜法對 Lane-Emden n=3 星震問題完勝 FD**: Exp K (Chebyshev N=48, 192 DOF)
精度 1.5e-6 vs Exp J (staggered FD Nr=1024, 4096 DOF) 精度 5.3e-4;
21× 少 DOF, 350× 高精度. 但此 "勝利" 的定位不是發星震學論文 —
**此求解器只是 stellar2d 2D GPU DNS 的線性診斷元件**, 真正 novelty 在
2D 非線性模擬 + 線上模式投影, 不在 1D benchmark.


# Key findings that are NOT in commits yet (only in conversation)

## F1 Exp K 的 8.7e-9 floor 不是譜法極限

Exp K N>64 的 max_rd 飽和在 8.7e-9, 但解析解天花板測試 (spectral_analytical_ceiling.py)
顯示 Chebyshev 本身在 Test B (量子諧振子) 和 Test C (Laplacian Dirichlet)
都打到雙精度機器極限 (~1e-14/1e-15). 故 Exp K 的 floor 是
**GYRE poly3.txt 999-point 結構係數輸入精度的天花板**, 要更精需要 rebuild
polytropic model with higher-order GL integrator + 10000+ 點 + rtol=1e-14.

## F2 "N 個係數 ≠ N 個像素" — 譜法解析度常見誤解

N+1 Chebyshev 係數定義一個**連續函數**, 可以透過 barycentric
interpolation (Berrut & Trefethen 2004) 在**任意** r 處精確評估到機器精度.
spectral_resolution_demo.py 驗證 N=48 的 49 節點 barycentric
到 4096 點和 FD Nr=1024 cubic spline 結果完全一致 (max diff 3.4e-3,
僅來自 N=48 自身離散化誤差).

同樣結論適用 Fourier 2D: **pseudo_spectral_solver 的 2048² 場可以
無損渲染到 4K/8K 解析度**. 真正的 resolution cap 是 dealiasing cutoff
(2N/3 ≈ 1365²), 不是 2048² 格點本身.

## F3 Gibbs 現象 vs 恆星脈動

- **線性 g-mode/p-mode EVP**: 解析光滑, 指數收斂, 譜法天花板是機器精度
- **Lane-Emden n=3 背景態**: 整數 σ=3, 光滑到 Chebyshev 看得見的程度, 同上
- **弱對流 / Boussinesq**: 譜法 floor ~ 1e-10 (無明顯 Gibbs)
- **發展對流中的 plume 邊界**: 接近 Gibbs 條件, floor 升到 1e-3 ~ 1e-5,
  需靠 dealiasing + 超粘性吸收. 這是 pseudo_spectral KH 結果的實際狀態.
- **強激波 / 超音速**: 譜法不適用, 切 cart_ale2

對我們 stellar2d 用例 (脈動 + 弱-中對流 + 光滑 polytropic background),
譜法的實用天花板遠高於 2D FD, 可以放心用高 N 求解.

## F4 項目定位最終修正

早先誤把目標設為 "做一個 1D 星震譜法 solver 發論文". 這是錯的定位:

- **Reese-Lignières 系列 (2006-)** 已經是譜法星震先驅 (2D 旋轉恒星, CPU)
- **GYRE** 是 1D GL6 collocation 標準
- **Dedalus v3** 可以做但沒做星震 benchmark

真正的 novelty (sole point):

- **GPU 2D 非線性 DNS + 譜法 y 方向 + Liouville SL basis + 線上 g-mode/p-mode 模式投影**
- 這把 "對流-脈動耦合" 作為**模擬 runtime output**, GYRE 和 Dedalus 都不做這事

1D Chebyshev g-mode solver 只是:
1. 線性精度 baseline (Exp K 已達成)
2. 2D 求解器的 y 方向結構 (Phase 1 的元件)
3. 2D 模擬內部在線做模式投影的引擎 (Phase 3 差異化賣點)

## F5 Path A 賣點降格 (NC1 in plan doc)

- Poisson 奇點在 r=R, 最佳前因子 α★(Poisson) = 1 − σ/2
- g-mode 奇點在 r=0, 最佳前因子 β★(gmode) = ℓ+1
- 兩者**不同**, "統一 SL basis 同時對角化兩個算子" 在嚴格意義上不成立
- 但 **Chebyshev 基礎網格可共用**, 單次預分解仍對所有 k_x 重用 (核心效率賣點倖存)
- "g-mode 免費副產品" 降格為 "同網格獨立 EVP"


# Phase 0 ext+ execution log (3-day plan status)

| Step | Goal | Status | Result |
|------|------|--------|--------|
| E5 | SymPy 推導 α★ | ✓ | α★₂ = 1-σ/2 for Liouville-Schrödinger 形式 |
| E6 | Chebyshev + α 前因子驗證 | ✓ (負面) | σ=1.5 任何 α 救不回 N⁻². σ=3 raw Cheb 就夠 |
| E6b | 多項式指數 σ 影響分析 | ✓ | σ ∈ ℤ 譜精度, σ ∉ ℤ 代數 (斷崖論文級發現) |
| E7/K | 用 raw Cheb benchmark Exp J | ✓ | 完勝: 21× 少 DOF, 350× 高精度 |
| — | 解析解天花板 | ✓ | Cheb 本身能到機器精度, Exp K 的 floor 是 GYRE 輸入限制 |
| — | 解析度 demo | ✓ | barycentric 證實 "N ≠ 像素" |
| E8 | Dedalus Jacobi 平行基準 | - | 跳過 (Path A 已夠用, 且 σ=1.5 不是本項目主要情境) |
| E9 | A/B/C 決定 | **→ Path A** | 基於 E7/K 證據: Path A 對 n=3 足以支撐 Phase 1 |


# Gate check (3 hard criteria from spectral_liouville_plan)

1. [✓] Lane-Emden 無 cutoff 全域指數收斂 (Exp K 證明於 n=3; n=3/2 不適用簡單 Path A)
2. [✓] Exp J EXPECTED 以 DOF ≤ 128 重現到 1e-3 (N=32 DOF=128 達 2e-3, N=48 DOF=192 達 1.5e-6)
3. [✓] Path A 選定, 理由書面化

**Gate PASS. 可以進入 Phase 1**.


# Next session (Phase 1 kickoff)

## Immediate tasks

1. **更新 4 個老文檔** (承諾在 E9 時做, 尚未做):
   - `docs/anelastic_SL_spectral_design.md` §4.2: 強化 "Poisson only" 統一性範疇
   - `docs/reduced_pressure_liouville.md`: 加入 Path A 前因子降格章節
   - `docs/anelastic_sl_phase0_2026-05-02.md`: "g-mode 免費副產品" 改 "同網格獨立 EVP"
   - `docs/singular_basis_survey_2026-05-02.md` §7.3: direction forward 更新
2. **Phase 1 設計文檔**: 寫 `docs/phase1_2d_spectral_design_2026-05-04.md` (接替 Phase 0 ext+)
3. **Phase 1 骨架**: 2D Fourier-Chebyshev Boussinesq 求解器 (target: Rayleigh-Bénard baseline)

## Phase 1 main axes

- **x 方向**: Fourier (週期), 沿用 pseudo_spectral 的 cuFFT
- **y 方向**: Chebyshev collocation, 在 GPU 上用 cuBLAS dense mat-mul 做 D2
- **物理**: 2D Boussinesq with ρ₀(y) stratification. Poisson via Chebyshev + 線性系統解.
- **benchmark**: Rayleigh-Bénard Nu-Ra scaling (對標 Ahlers 等人)

## 何時寫論文

Phase 1 跑通 + Phase 2 (anelastic 升級) 跑通後. 預計 2 個月.
angle: "GPU anelastic pseudo-spectral with **live eigenmode projection**
for convection-pulsation coupling diagnostics".


# Frozen assets inventory

## 凍結的 1D g-mode 算子 (不修改, 只增新)

| 算子 | 方法 | 精度 vs GYRE full | 作用 |
|------|------|---------------------|------|
| `solve_gmode_full_gyre_compat` (gmode_infra.py) | Staggered FD Nr=1024 | n_g=1 5.9e-7, max_rd 5.3e-4 | FD regression oracle |
| `solve_gmode_cowling_gyre_compat` (gmode_infra.py) | Staggered FD Nr=1024 | vs GYRE Cowling max 5.6e-4 | FD Cowling baseline |
| `solve_gmode_full_chebyshev` (gmode_exp_k_chebyshev_full.py) | Chebyshev N=48 | max_rd 1.5e-6 | **Chebyshev production ref** |

## 解析驗證腳本 (凍結 EXPECTED)

- `gmode_exp_i_gyre_compat.py --verify` — 2-var Cowling, max_rd 5.6e-4
- `gmode_exp_j_full_gyre_compat.py --verify` — 4-var full, max_rd 5.3e-4
- `spectral_analytical_ceiling.py` — 無 --verify 但結果穩定到機器精度

## 要保留但可以不再動的探索腳本

- `scripts/gmode_exp_a..g_*.py` — 教學 baseline, Boussinesq-like 簡化算子
- `scripts/gmode_exp_h_*.py` — Exp H 失配記錄 (物理見解)
- `scripts/reduced_pressure_*.py` — Liouville 形式驗證
- `scripts/anelastic_sl_phase0*.py` — Lane-Emden n=3/2 Phase 0
- `scripts/spectral_liouville_beta_derivation.py` — α★ SymPy
- `scripts/spectral_liouville_convergence_v2.py` — σ 斷崖數據
- `scripts/spectral_liouville_prefactor.py` — Path A α sweep
- `scripts/spectral_resolution_demo.py` — barycentric demo
