# 下一步工作計劃 — 2026-05-03 封板

> 今日把 block-tridiag PC 打通,dt 天花板升 5 個數量級,但 **T_c 還是凍結**。
> 這份文件拆清楚接下來該做什麼,按**先解結構阻塞、再擴物理、再上 2D**
> 的順序。

---

## 立即優先:解開 BE-rad / R_hse 耦合(T_c 凍結修復)

### 症狀

用 `--precond-tridiag` 跑 pre-MS IC 到 10¹² s(32 千年,34 步,0 失敗):
- dt 成長 88 s → 5×10¹¹ s ✓
- 能量守到 10⁻¹⁰ ✓
- **T_c 固定在 4.1415350000e+06 K 不動** ✗
- **IE_total 變化 < 10⁻⁷ 跨整個積分** ✗

### 診斷

三個觀察疊起來鎖定結構問題:

1. **Newton 短路.** `newton_solve_implicit` 開頭 `if (it == 0 && res_norm < newton_tol) return 0;`
   殺掉了 Newton 更新,因為初始 ‖F‖ = 3×10⁻⁹ < tol 1×10⁻⁸。**Newton 什麼事都沒做。**

2. **BE-rad 算了 ΔT 卻看不到效果.** `apply_radiation_diffusion_implicit` 回報
   `L_surf = 2.5×10³⁷ erg/s`,但每步 `k_rad1d_apply_dT` 的 ΔT 透過 `eos.cv()`
   換算回 `e_int += cv·ΔT` → 對整個 nz·e_int 積分看不到變化。

3. **tighten tol 讓事情崩潰.** `--newton-tol 1e-15` 逼 Newton 去解,但 ‖F‖ 永遠達不到 tol,
   dt 被連續 halved 到 0。

### 根因

**well-balanced `R_hse` 把時間演化鎖死在 HSE 附近**。Newton 解的是 `F(U) = (U−Uⁿ)/dt − (R(U) − R_hse)`,
當 U ≈ U_hse 時 `R(U) ≈ R_hse → F ≈ 0`。radiation 的 energy loss 只進 `apply_radiation_diffusion_implicit`
(operator-split),不回到 Newton 的 R 裡。所以:

- hydro step(Newton):HSE 不動,F 幾乎零
- rad step:e_int 小幅下降
- 下一輪 Newton:新的 R_hse 還是原來那個,所以 F 還是幾乎零

**結構問題:rad 應該進入 R,不該 operator-split。**

### 解法(三條路選一)

#### 路 A:把 BE-rad 融進 Newton 的 F(推薦,但工作量大)

現在 F 只含 hydro + pp_source。加入 `R_rad = -∇·F_rad / dm`(發散形式),Newton 就會同時
解 hydro + rad coupling。BE-rad 就不再是 operator-split,它變成 implicit F 的一部分。

**工作量估計:**
- F kernel 加 radiation divergence 項(使用 current T via EOS):1 天
- colored FD matvec 自動包含 radiation 耦合(sparsity pattern 不變):0 天(PC 自動適配)
- 測試 pre-MS staircase 能時間演化:0.5 天
- **合計:1.5 天**

**風險:** radiation 和 hydro 的 stiffness scale 差很多,可能要調 Viallet scaling
或 GMRES 容錯。但 block-tridiag PC 吸收大部分條件數問題。

#### 路 B:運算子拆分但 HSE 週期性 resnap

現在 `hse_resnap_interval = 0` 預設,測試 `=1` 沒用(因為 F 同樣小)。真正要做的是:

- rad step 後,**重新 snapshot R_hse 到新的熱力狀態**,再跑 Newton 解 hydro response
- Newton 就會看到 "HSE 變了,要調整 (v, r, e) 來維持"

**工作量估計:** 0.5 天(只改 step_implicit)。

**風險:** 每步 resnap 可能把 rad 的效應吃回去。需要實驗看 T_c 有沒有真的跑。

#### 路 C:直接去掉 well-balancing

`F = (U−Uⁿ)/dt − R(U)`,不減 R_hse。Newton 直接解絕對方程。失去 well-balanced 的 roundoff 優勢,
但 rad 進入 F 不是問題。

**工作量估計:** 0.25 天(改一行)。

**風險:** 在深內部 cs² 能量項 10¹⁶,Newton 算相對收斂會損失精度。

### 建議

**先走路 B(0.5 天)做 proof-of-concept**,確認 T_c 開始動。如果動得正確,升級到路 A 做生產版本。
路 C 當對照 benchmark。

---

## 第二優先:autodiff PC(觸發條件可能已到)

### 觸發條件檢查

`radial1d_autodiff_jacobian_plan.md` 列了三個觸發:

- [x] 要把 dt 從 1e6 推到 ≥ 1e10 → **已達**(PC 讓 dt 到 5×10¹¹)
- [ ] 引入 aprox13 核網路 → 沒做
- [ ] cart_ale2 要寫 implicit Newton → 沒做

**第一條已觸發**,第二條間接相關:如果我們要上 weak screening / ppII/III / CNO,
核反應變 stiff,FD matvec noise 可能變主要。

### 當前 FD noise 估算

JFNK FD 步長 `α ≈ √ε_m ≈ 3×10⁻⁸`,在 cgs 下 ‖U‖ ≈ 10¹⁶ → 相對 FD 噪音 `~10⁻⁸`,
跟 Newton tol 1e-8 剛好同量級。**GMRES 看不到更精細的結構,這就是為什麼 tol 收不緊**。

autodiff 消除 FD 噪音,GMRES 可以推 tol 到 1e-12。**這是解 T_c 凍結後**可能需要的。

### 建議

**先別做**。等路 A/B/C 跑通,觀察新的 GMRES 收斂曲線。如果 tol 卡在 1e-8,autodiff 上。
如果 1e-8 已經 fine,autodiff 留到 cart_ale2 implicit Newton 再做。

工作量估計(當觸發):**3 天**(Dual\<T\> POD + Helm 相容 + F kernel)。

---

## 第三優先:補物理 — pp-chain 品質

### 當前狀態(staircase 驗證)

| | pp-only | 加 weak screening | 加 ppII/III | 加 CNO |
|--|--------|-------------------|-------------|--------|
| Pre-MS 誤差 | ×1.9 | ×1.3 | ×1.1 | ×1.05 |
| ZAMS 誤差 | -8% | -7% | -5% | -2% |

工作量估計:
- weak screening(Salpeter 1954):0.25 天
- ppII/III branch corrections:0.5 天
- CNO 旁路:0.5 天(只需要個 ε_cno(T, X, Z) 解析公式)

**合計 1.25 天**,誤差 ×1.9 → ×1.05,跟 MESA 等量級。

### 建議

**時間演化跑通後再做**。現在加物理補丁,每次還是 snapshot staircase,加減驗證不出演化一致性。
等路 A/B/C 解開 T_c 凍結,就可以用 *軌跡差異* 驗證每個物理補丁(比 staircase 資訊量大得多)。

---

## 第四優先:回 cart_ale2 2D(戰略重點)

### 當前狀態

- §9.1 bottom heating 實裝 ✓
- §9.1a / 9.1b 2D→3D flux 重整未做:需要把 Lz 的有效深度計入 F_bot 標稱化,否則 τ_eq ≈ 2×10⁵ τ_dyn

### 為什麼要回這邊

**最終目標**按照 `radial1d_session_2026-05-02_03_journal.md` 結論,是把 radial1d 一個收斂的
ZAMS state map 到 cart_ale2 當 Cartesian IC,跑 2D 對流 cell。**MESA 做不到這件事,我們可以**。

radial1d 1D 只是橋樑,不是獨立 deliverable。

### 建議

**T_c 凍結解開 → 跑一個自洽 ZAMS 軌跡 → 用這個 ZAMS 去做 2D 橋**。前面兩步做完之後
(估 ~3 天),cart_ale2 §9.1b 就變成 "把 radial1d 生成的 ZAMS state 映出去",而不是
"用 MESA profile 映出去"。差別:radial1d 的 ZAMS 是 radial1d 自己解的,邊界條件、EOS、opacity 全一致。

工作量估計(pipeline 已經到位):**1 天**。

---

## 第五優先:Path B 2D sphere wedge(長程目標)

不在本 session scope。等 Path A 1D 軌跡 + cart_ale2 2D box 都穩,再啟動。

`docs/path_a_status_2026-05-02.md` 描述的 1-2 月 MVP 路線不變。

---

## 整體時序圖

```
今日(封板)
  │
  ▼
路 A/B/C 解 T_c 凍結 ─── 0.5 – 1.5 天 ─── **關鍵階段**
  │
  ▼
radial1d 自洽 pre-MS → ZAMS 軌跡 ─── 0.5 天 ─── HR 圖、T_c(t)、L_nuc(t)
  │
  ├── (如果 FD noise 變瓶頸) autodiff PC ─── 3 天 ── 光環
  │
  ▼
cart_ale2 2D 橋接(radial1d ZAMS → Cartesian IC) ─── 1 天 ── 獨立價值點
  │
  ▼
pp-chain 物理補丁(screening + ppII/III + CNO) ─── 1.25 天 ── 收尾精度
  │
  ▼
Path B sphere wedge 啟動 ─── 1-2 月
```

總時程到真正「1D → 2D 閉環 demo」:**~3-5 天**(不含 Path B)。

---

## 風險與不確定性

- **路 A 把 rad 進 F 可能破壞 Newton 收斂性.** BE-rad 有 Picard 迭代 + T^4 線性化,直接塞進
  outer JFNK matvec 可能 FD 跨物理 scale 的耦合處理不當。fallback 是路 B / C。
- **T_c 凍結可能還有第二層原因.** 即使 F 耦合進 rad,如果 BE-rad 本身的 ΔT 計算有 bug
  (比如 cv 用錯、T 反解不收斂),仍然看不到 evolution。實驗時加 `apply_radiation_diffusion_implicit`
  開頭印 `Σ ρ cv ΔT · Vol` 驗證實際能量損失。
- **pp-chain 的 ×1.9 不是物理必然.** weak screening 實測可能改 20% 不是 40%;staircase 對不準 
  還有 Abar/Zbar 設定的小差異要排查。

---

## 改動/新增文件索引

### Source
- `src/gpu/radial1d_solver.{cuh,cu}` — `Diagnostics` + L_nuc kernels + block-tridiag PC buffers
- `src/gpu/radial1d_kernels.cuh` — 3 個新 kernel(`k_rad1d_nuclear_L[_species]`, `k_rad1d_T_from_rho_e`)
- `src/gpu/radial1d_implicit.cu` — `build_precond_tridiag` / `apply_precond_tridiag` + Newton hook
- `src/gpu/cart_ale2_solver.{cuh,cu}` — `k_cale2_thermal_step` 融合 cooling + heating
- `src/main.cpp` — `--heat-{flux,lsun,bot-R,bot-frac}`, `--cool-top-frac`, `--precond-tridiag`, `--newton-tol`

### Docs
- `docs/radial1d_ignition_endtoend_2026-05-03.md` — 8-snapshot staircase
- `docs/radial1d_precond_tridiag_2026-05-03.md` — PC 報告
- `docs/session_journal_2026-05-03.md` — 今日跑完整敘事
- `docs/next_steps_2026-05-03.md` — 本文件
- `docs/cart_ale2_local_convection_2026-05-03.md` — §9.1 heating 更新

### Scripts + Images
- `scripts/plot_ignition_trajectory.py`
- `scripts/plot_precond_speedup.py`
- `docs/images/ignition_trajectory.png`
- `docs/images/precond_tridiag_speedup.png`

### 沒做到的文件(留給後續)
- `docs/radial1d_time_integration_2026-05-XX.md` — 等 T_c 凍結解開後的成果
- `docs/radial1d_ppchain_upgrades_2026-05-XX.md` — screening + CNO 補丁後
- `docs/cart_ale2_bridge_2026-05-XX.md` — radial1d ZAMS → 2D box 橋接
