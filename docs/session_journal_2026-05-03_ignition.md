# 2026-05-03 開發 journal — pre-MS 1 M☉ 點火 (nr=1024)

> 起點:上次 session (`f7f6f7f`) 跑出了 nr=1024 的 HR / T_c-ρ_c 圖,但 T_c
> **凍結**在 IC 4.18×10⁶ K 跨 10¹⁶ s,整個 dashboard 平線。用戶精準指出
> 這不是 real pre-MS,是模擬失敗。本次 session 拆解原因並修復。

## 本次 commit 時間線

| commit | 內容 |
|---|---|
| `04ecc92` | 寫計畫 — 假設根因 A(R_hse 吸掉 rad drive)/ B(Stefan BC 失效)/ C(MLT 缺失) |
| `b29a37c` | 診斷 A 驗證 + **真正修復**:Newton/GMRES 假收斂問題 |
| `db0222d` | MLT 接回 rad-in-F 殘差 + K_conv cap |

## 今天做了什麼

### Phase 1:診斷實驗

加了 `--no-rhse` flag,在 nr=128 和 nr=1024 比較 R_hse 減 / 不減:

- **nr=128** 兩路都正常演化 (T_c 4.2→8.5×10⁶ K) → **原根因 A 不成立**
- **nr=1024** 兩路都不穩,但症狀不同 → 真正根因是 **Newton 在高解析度下假收斂**

### Phase 2:Newton/GMRES 真收斂 (`b29a37c`)

三管齊下:

1. **當前態 Viallet scaling**:`k_r1di_build_scaling` 從 HSE snapshot
   (`d_rho0/d_P0`)改成讀當前態(`d_rho/d_P/d_e_int`)。row/col scales 跟著
   state 演化,不再鎖死在 IC。
2. **rad-aware Le row-scale**:在 e-方程行尺度加入 rad 貢獻
     `Le = max(cs³/R_star, c·a·T⁴/(3κρ²Δr²))`
   nr=1024 外層 dm 縮小 8×,rad per-mass term 大爆炸,原 hydro scale 會
   低估 ||F_e|| → Newton 假收斂。
3. **tol 收緊**:
   - `gmres_tol` 1e-3 → 1e-6(原本 GMRES j=1 就 |g|<tol·β 退出)
   - `newton_rel_tol` 0.5 → 1e-3(原本 2 iter 就「收斂」)

**效果 (nr=1024 無 MLT)**:
- step 0 ||F||: 553 → 5e-6 (8 decades) 用 11 Newton iter
- step 1..4600 零失敗,dt 穩在 1.6×10¹⁰ s
- T_c 4.18 → 5.33×10⁶ K 跨 2.2 Myr — 真實 KH 速率

### Phase 3:MLT 接到 rad-in-F (`db0222d`)

把 Böhm-Vitense MLT 從 operator-split BE-rad 路徑接到全耦合 Newton。

實作:
1. `RadParams.K_conv`:Picard-lagged 標量 conductivity 陣列
2. `rad_face_L_dual<N>` 加 convective flux term
     `F_conv = A · K_face · (T_L − T_R) / Δr_zc`
   對 Dual<N> T 線性(AD 自然流通),K 為標量(每步 lag)
3. scalar path `k_r1di_residual` 同樣加
4. `refresh_K_conv_implicit()` 公開方法包住 `k_rad1d_mlt_cond`
5. Newton outer loop **每步開頭**刷一次 K(不是每 iter),避免 K-swing

**K_conv cap** 防 Schwarzschild 邊界爆:
  `K_conv ≤ 10⁴ · ρ·cp·c/(3κρ)`
pre-MS 包膜 K 從 ~10¹⁸ 降到 ~10¹⁴,BE 不再爆。

**效果 (nr=1024 帶 MLT)**:
- 前 700 步零失敗,dt 穩在 1.5×10¹⁰ s
- T_c **4.18 → 6.68×10⁶ K** (+60%)
- ρ_c **1.4 → 27.9 g/cc** (+20×)
- **conv_mass_frac = 0.61**,n_conv = 625 zones — MLT 真的啟動了
- t = 8.1×10¹² s ≈ 260 kyr

## 成果對照

| 指標 | IC | session 前(frozen) | session 後 | pp 點火目標 |
|---|---|---|---|---|
| T_c (K) | 4.18e6 | 4.18e6 (完全不動) | **6.68e6** | 1.5e7 |
| ρ_c (g/cc) | 1.39 | 1.43 | **27.9** | ~100 |
| t 走到 | 0 | 10¹⁶ s | 8.1×10¹² s | 3×10¹⁵ s (30 Myr) |
| conv_mass_frac | 0 | 0.16 (無實際運作) | **0.61** | ~0.8 (Hayashi) |
| Newton 失敗 | — | cascade | **0 (前 700 步)** | 0 |

## 還沒點火

T_c 離 1.5×10⁷ K 還差 2.2×,需要再 30 Myr。當前 solver 撐不過去:

1. **step ~740 後 dt collapse**(仍在調查)— 可能:
   - 核心過度收縮產生 subsonic shock
   - MLT K 在某些 zone 觸到上限,造成局部硬切換
   - Newton 在深部對流帶找到多解
2. **表面 BC 仍 τ_sum=0, phot_zone=1023** —(根因 B)L_surf 只 0.23 L☉,
   真實 pre-MS 應 1-3 L☉,能量出不去也拖慢收縮
3. **T_eff 仍 ~1000K** — Helm 表底限,沒法讓光譜正確 descend Hayashi

## 保留資產

本次 session 不會改的:

- `src/physics/dual.cuh` + `helmholtz_eos_dual.cuh`(forward AD 完整測試通過)
- `src/gpu/radial1d_residual_dual.cuh` Dual 殘差(MLT 加進來了,結構 clean)
- `--no-rhse` CLI flag(保留作未來診斷工具)
- Newton 真收斂配方(current-state scaling + rad-aware Le + tol 收緊)

## Phase 4:繼續排查 (commit `6333017`)

### 診斷 step 700 dt collapse 的根因

抓出 profile_0030 (step 600) 外層:
- 正常 zones 1019-1022: ρ ~ 1e-3 → 3.7e-4 平滑遞減
- **zone 1023 異常:ρ = 0.022 (50× spike), e_int = 1e-30 (literal FLOOR)**
- Δr_1023 = 2e8 cm(鄰近 zone 的 1/60)

### 根因 E:MESA IC 外層 Δr 擠壓 + ρ_geom 不一致

MESA profile.data 在外層大氣**超采樣** — 300+ MESA zones 佔 <10⁻⁵ 質量
(H, He 光球精細結構)。radial1d 1024-zone equal-mass shell 把這些
全塞到 1 個外層 zone,結果:

1. Δr_1023 僅 5e8 cm(相鄰 zone 的 1/10 之內)
2. ρ_geom = dm/Vol 暴增 10⁴×(從 MESA 1e-7 → 1.77e-3)
3. Helm seed e(ρ_mesa, T_mesa) 用 MESA 的 ρ=1e-7,但 runtime ρ=dm/Vol
   → 不一致 → Helm 返回 e=1e-30(floor)
4. 演化到 step 600 外層「突然甦醒」,e 從 1e-30 跳到 1e12,速度衝刺,
   Newton 跟不上 → dt collapse

### 雙管修復 (commit `6333017`)

1. **外殼 Δr floor**:若 outermost Δr < 0.7·內部 Δr,把 face 往內移讓
   殼間距至少達 0.7×。質量 dm 不變,只平滑幾何。
2. **ρ_geom 一致性**:`h_rho = max(rho_mesa_interp, dm/Vol_k)`。
   外層用 ρ_geom,核心 zones 仍用 MESA ρ。

### 結果

| 指標 | 修前 (db0222d) | 修後 (6333017) |
|---|---|---|
| step 700 前失敗 | 多 | **0** |
| 總失敗(到 t=1.3e13s) | 101+ | **7** |
| IC profile 外層 e_int | 1e-30 (floor) | 2e13 (physical) |
| 最終 T_c 飽和點 | 6.68×10⁶ | 6.67×10⁶(相同) |

dt collapse 被推後,但**同一 T_c 飽和點未變**。

### Phase 5:Helm T_inv 微調(失敗回滾)

嘗試:改 `helm_T_from_rho_e` 初始 T_guess 從固定 10⁴ 改成 ideal-ion
估計 `T ≈ e/9.6e7`。目的:讓外殼 Helm 不要 converge 到 1280K noise。

結果:T_phot 略升(1280K → 1388K),L_surf 略增(0.53 → 0.72 L☉),
但 dt collapse 時間點從 t=1.3e13 s 前移到 t=3e12 s,失敗數增加。
Newton 因 T_inv 變化路徑而變激進,反而更快卡住。已 revert。

## 真正的點火障礙(本 session 未解)

系統在 T_c ≈ 6.7×10⁶ K, ρ_c ≈ 28 g/cc 附近 **物理飽和**,不管怎麼
調 Newton / IC 都到不了更深。推測是:

1. **L_surf 上限**:當前 ~0.5-0.7 L☉,真實 pre-MS 1 M☉ Hayashi 應 1.5-3 L☉
2. 太陽熱不出去 → 核心收縮被阻止
3. 解開需要修 τ=2/3 photospheric 掃描(用 sub-zone 內插 / Eddington 大氣)
   或放棄 grey atmosphere 用 MESA-style 外部 T(τ) 表

## 下次 session 要解的

**優先級 1**:step 700 之後 dt collapse 的根因
- 看 Mach 分布 profile(不是 max),找哪幾個 zone 失控
- 看 K_conv 是否卡在 cap 上,cap 太嚴可能就是瓶頸

**優先級 2**:表面 BC(根因 B)
- τ=2/3 掃描用內插而不是整個 zone 的 T
- 或放棄 grey atmosphere,用 MESA-style 外部 atmospheric table
- 解了 L_surf 就能合理流出能量 → KH contraction 加速

**優先級 3**:ZAMS demo
- Phase 2 過後推到 T_c=1.5×10⁷ K
- L_nuc/L_surf 從 0 → 1
- X(hydrogen) 開始下降

## 關鍵的設計決策(別忘了)

- R_hse 減法**繼續用**(原本以為是問題,diagnostic 證實不是)
- MLT K **每步刷一次**,不是 per Newton iter(避免震盪)
- K cap 固定 10⁴× rad(用戶可 later 暴露成 CLI flag,現在 hard-coded)
- Newton `rel_tol` 從 0.5 → 1e-3 是**所有解析度共用的預設**
  (nr=128 也要更嚴的 Newton,只是影響不明顯)

## Phase 6:Eddington BC + Viallet Le 修 + IC 結構性挖穿

### 6.1 Eddington 1-zone atmosphere BC (commit 14f2e38)

用 `T_eff⁴ = (4/3) T_k⁴ / (τ_k + 2/3)` 取代 τ-scan BC。三處同步改:
- Dual<N> residual (`rad_face_L_dual`)
- scalar `k_r1di_residual` 外層分支
- diagnostic `k_r1di_diag_L_surf`

結果:
- dt 從 ~10³ 崩潰 → 穩定 10¹⁰ s (7 量級)
- T_phot: 1280 K noise → 物理 790-1248 K
- L_surf: 10⁴⁰ noise → 10³⁰-10³¹ well-defined

但系統仍**完全凍結**:KE=0, v_max=0, T_c 8 位精確不變。

### 6.2 Viallet scaling e-row 被太大 hydro scale 吞 (commit 8912326)

verbose log 揭示:`||F||=1.03e-11 < tol=1e-8` → Newton 直接退出,
ΔU=0 永久鎖。但 `||F_e||_unscaled = 51` 非零。

原因:Le 用 `max(cs³/R, c·a·T⁴/(3κρ²Δr²))` ≈ 10¹⁴-10¹⁶,
而真 F_e ≈ ε_pp ≈ 5000。scaled F_e ≈ 5000/10¹⁴ = 5e-11,< tol。

修法:`Le = min(Le, Le_nuc)` 當 Le_nuc > 0。讓 Newton 真正對
ε 級 source 敏感。

效果:||F|| 1e-11 → 7.85e+12,GMRES 跑 5 iter 每步,系統開始
演化。但 T_c **仍然卡死在 7.353e6**。

### 6.3 挖到真正根因:MESA IC 外層 T 結構性錯誤

`Helm(ρ=0.032, ?)` 返回 e=8.98×10¹³ 對應 **T ≈ 7×10⁶ K**,
而非 MESA 光球的 4500 K。

分析 `/tmp/mesa_1Msol_late_preMS.ic`:MESA 原始 data 732 行,
外層 300+ 行覆蓋 <1e⁻⁵ 質量。radial1d 128 equal-mass zones:
每 zone dm=1.55e31 g,外層 zone 127 的 m_enc 中心約 1.55e31 g,
這對應 MESA **row 722** 的 T=7.3×10⁶ K。

**光球 4500 K data 被當 noise 壓掉**。

Eddington BC 雖數學對(稀釋 τ_k=10¹¹ 給 T_eff≈630 K),但
T_k=7×10⁶ 是深處 T 不是光球底溫,結果 T_eff 不等於 4500 K
的真 T_eff。

### 6.4 反證:ZAMS IC demo 成功

`--ic-mesa /tmp/mesa_1Msol_zams.ic --ic-mesa-seed-T`:

| 指標 | 值 |
|---|---|
| T_c | 1.339×10⁷ K |
| ρ_c | 75.8 g/cc |
| L_nuc | 2.455×10³³ erg/s ≈ 0.6 L☉ |

**真點火**。這證明 solver 全套(Newton + Viallet + Eddington +
MLT + pp)完全正確,只是 pre-MS → ZAMS 的 KH 收縮被外層 zone
結構性問題阻斷。

## 最終根因(本 session 挖到底)

T_c 在 7.35×10⁶ K 飽和的鏈條:

1. equal-mass Lagrangian zoning 把 MESA 光球(300+ rows 佔
   <10⁻⁵ M) 塞進 1 個 radial1d 外層 zone
2. 該 zone 的 h_T[k] 取自 m_enc 中點的 MESA 內插 → 得到 **深處 T
   7×10⁶ K**,不是光球 4500 K
3. Eddington 1-zone BC 假設 T_k 是大氣底部 T,但 T_k 其實是深處 T
   → T_eff 計算意義不對
4. L_surf 被推到 10⁻⁴ L☉(應 1-3 L☉)
5. 核心能量無法流出 → KH 收縮停滯 → T_c 不爬

## 真解(下次 session)

**選項 A — log-P outer zoning**:重寫 `init_from_mesa` 讓外層
10-20 zones 在 log P 或 log τ 上等分,專門解析光球結構。外層
zone dm 會非常不均,但能正確表達 T(τ)。工作量半天。

**選項 B — outer atmosphere sub-grid**:在 BC kernel 內部對
最外 zone 做 sub-step Eddington 積分,找 τ=2/3 對應的 T_ph,
用它當 T_eff。不改 grid,只改 BC。工作量~1 天。

**選項 C — 繞過**:直接 ZAMS IC 做 pp 點火 demo(已驗證成功)。
pre-MS 演化變 2D convection 的後續任務。

## 保留資產(不要回滾)

- commit 14f2e38 — Eddington 1-zone BC(dt 穩定性必備)
- commit 8912326 — nuc-aware Le scaling(Newton 能看到 ε source)

這兩個都是 **正確的** 修法,只是上面 IC 問題比它們更根本。
未來任何 outer-atm 修法都要搭這兩個。
