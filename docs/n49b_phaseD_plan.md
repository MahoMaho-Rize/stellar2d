# N49B Phase D 計劃 — radial1d 正式 1D hydro 爆炸

**日期**: 2026-05-04
**分支**: `n49b-replication`(後續新 feature 分支)
**目標**: 把 Phase C 的 12.75 M☉ post-SN Mg/Ne 從 **1.35 降到 0.75 ± 20%**
**根因診斷**: Fryxell parametric + 單調冷卻把 O-rich zone 卡在 T₉ ≈ 1-2.5 窗口
(Ne 燒動、Mg 燒不動),ratio 反向漂移。真正需要:**每個 zone 的自洽 T(t)/ρ(t)**
由激波傳播 + reverse shock + 自由膨脹給出,這個只有 full 1D hydro 能做。

---

## Plan A(基礎計畫):radial1d 正式爆炸 — 6 個工作日

### Day 1 — Species infrastructure 擴展 2 → 13

**目標**: radial1d device state 從 `(X, Y)` 擴成 `X[13]`,CLI/I/O/init loader 全通。

**任務**:
- 在 `src/gpu/radial1d_solver.cuh` 把 `d_X, d_Y` 換成 `d_X_spec[N_SPEC]`(`N_SPEC = alpha_net::N_SPEC = 13`),舊 pp-chain 路徑保留為 `species_mode = pp` 向後相容
- 新 `species_mode = alpha13` 走 13-slot 路徑
- `init_species_alpha()` 把 host 的 `double X[nz][13]` memcpy 到 device
- `download_species_alpha()` 輸出為 npz / text
- `--species-mode {pp,alpha13}` CLI flag
- 測試:初始化 homogeneous X = (0.5 He, 0.5 O), 0 dt → download 驗證一致

**風險**: 低,純 bookkeeping。
**Deliverable**: `src/gpu/radial1d_solver.{cu,cuh}` 新增 ~150 行,編譯通過。

### Day 2 — Sukhbold IC 載入 + Bomb 注入

**目標**: `--test sukhbold_bomb --ic-sukhbold 15.90.dat --mass-cut 1.6 --E-SN 1.0e51`
能把 Sukhbold profile 灌進 radial1d 的 Lagrangian grid,內核 excise,外層疊一個熱爆能。

**任務**:
- `scripts/n49b/convert_sukhbold_ic.py`: Sukhbold .dat → radial1d IC(格式擴展,
  加 `# n_species 13` 和 `X_He X_C X_O X_Ne X_Mg X_Si X_S X_Ar X_Ca X_Ti X_Cr X_Fe X_Ni` 欄)
- `src/gpu/radial1d_solver.cu::init_from_sukhbold()`: 擴展 `init_from_mesa()`
  插值到 radial1d equal-mass shells,同時插 13 個 X_i
- **Mass cut 處理**: 內層 m < M_cut 的 zone 設為 inactive (d_active[k]=0),
  Newton 跳過;或簡單做法 —— 直接 drop,把 M_cut 當 inner boundary
- **Bomb**: 在 M_cut 之上 0.1 M☉ 厚度內,對 `d_e_int` 疊加 `E_SN / (0.1 M☉)` erg/g
- Dump IC 驗證圖:對比 Sukhbold profile vs radial1d 插值結果

**風險**: 中。inner boundary 處理是 radial1d 沒做過的(之前一直是 r_in=0 的
球心邊界)。可能需要新加 `--ic-inner-boundary mass_cut` 模式。
**Deliverable**: 單 run 能從 Sukhbold 15.90 IC 跑出一個爆炸初態,t=0 的 diagnostic 正確。

### Day 3 — α-network 作為 Newton 外的 operator split 源

**目標**: 每個 Newton step 收斂後,對每個 zone 用收斂的 `(ρ, T, dt)` 跑
`alpha_net::advance_substep()`,更新 X[13] 和釋放的核能回注 e_int。

**任務**:
- 在 `radial1d_solver.cu::step()` 的 Newton 收斂後加 `nuclear_operator_split()`
- 該 kernel:對每個 zone 調 `alpha_net::advance_substep()`,返回 ε_total,
  加到 e_int
- **不** 把 α-net 塞進 Newton 殘差 — stiff 會爆
- CLI:`--alpha13-source` 開關,與 pp-chain 互斥

**風險**: 中。stiff nuclear source 和 implicit hydro 的 dt 控制要對齊:
- 如果核反應 timescale << hydro dt,α-net 內部 substep 已經處理
- 如果核反應釋能 >> e_int/dt,可能要反過來壓 hydro dt(`--nuc-compress` 機制
  已經有,改成走 α-net 能量)

**Deliverable**: 單 zone bench:從 (X_O=0.5, X_He=0.5, ρ=1e7, T=2e9) 開始,
跑 10 個 hydro step,α-net 結果和 standalone advance 一致到 1%。

### Day 4 — HSE well-balance vs 激波傳播

**核心矛盾**:
```
F = (U − Uⁿ)/dt − (R(U) − R_hse)
```
`R_hse` 是 IC 時的 HSE reference。爆炸後星不再 HSE。激波通過時 zone 的 R 和 R_hse 差一大截,Newton 會 fight 這個差。

**三種處理(測試決定哪個工作)**:

1. **完全關掉 well-balance** — `--no-rhse-subtract` 已有,直接用
   - 對爆炸 timescale(10s of s << τ_dyn_star ≈ 10³ s)可能沒問題
   - 靜水壓平衡由 Newton 本身處理,不需要 R_hse trick

2. **每步 re-snap R_hse** — `--hse-resnap 1` 已有
   - 但 R_hse 是 HSE reference,爆炸狀態下本身就**沒有** HSE,強行 re-snap 會 snap 到非 HSE 的 R,等於關 well-balance 但多算一次

3. **激波面的 local well-balance**: 太複雜,跳過

**測試**:
- 15.90 M☉ bomb IC,跑 10 s,檢查 shock 能否從 r=10⁹ cm 傳到 r=10¹¹ cm
- Track dt evolution(別像 pre-MS KH 那樣 collapse)
- Track Newton 收斂(rel_tol 1e-9 已緊,可能要放鬆到 1e-5)

**Deliverable**: 一份 `docs/n49b_phaseD_shock_stability.md` 記錄三種處理結果,
選一個走下去。

**風險**: **高**。memory feedback_radial1d_not_stellar_evolution.md 記錄:
HSE well-balanced + Newton 在 v=0 背景下是穩定的,一旦大 v 出現可能有 Newton
條件數問題。可能需要 Day 4.5 debug。

### Day 5 — 4 個 progenitor 完整 run

**任務**:
- 對 12.02 / 12.75 / 15.28 / 15.90 跑 `radial1d --test sukhbold_bomb --alpha13-source`
- 每 run ~10 分鐘(nz=256, 10 s)? 或 30 分鐘?根據 Day 4 結果估計
- Dump post-SN X_i(m_enc) 到 npz
- 敏感性:mass cut ∈ {1.4, 1.6, 1.8},E_SN ∈ {0.5, 1.0, 2.0, 4.0} × 10⁵¹ erg
  (N49B 觀測 2-4 × 10⁵¹ erg,覆蓋論文區間)

**Deliverable**: 4 × {3 mass cuts} × {4 E_SN} = 48 個 npz。總 run 時間 ~12 小時 wall clock(overnight)。

### Day 6 — Fig 7 + report

**任務**:
- 更新 `scripts/n49b/fig7_pre_vs_post.py` 讀 radial1d npz(不是 post-processing 的)
- 繪製 Fig 7 4×2 面板,和論文直接對比
- 寫 `docs/n49b_replication_stage2_phaseD.md`:
  - 列出所有 Mg/Ne 數字和論文精確差距
  - mass cut / E_SN 敏感性表
  - 如果 12.75 Mg/Ne 在 [0.6, 0.9] 範圍 → **完全對上論文**,發 Suwa
  - 如果超出 → debug trajectory(hydro 本身問題還是 α-net 耦合)

**Deliverable**: 可發 PR,可寄 email。

### 總工作量估計

| 階段 | 最樂觀 | 現實 | 悲觀 |
|---|---|---|---|
| Day 1 species infra | 3 小時 | 1 天 | 2 天(測試 corner case) |
| Day 2 IC + bomb | 半天 | 1 天 | 2 天(inner boundary debug) |
| Day 3 α-net hook | 半天 | 1 天 | 1 天 |
| Day 4 HSE + shock | 1 天 | 1.5 天 | **3 天** (架構級調試) |
| Day 5 runs | 半天 | 1 天 | 1 天 |
| Day 6 report | 半天 | 半天 | 1 天 |
| **合計** | **3.5 天** | **6 天** | **10 天** |

**Blocker 信號**:Day 4 如果兩種 HSE 處理都給不穩定 Newton → 切 Plan B。

---

## Plan B(備案):Magkotsios+2010 trajectory family — 1-2 個工作日

只動 Python,不碰 radial1d。核心:替換 `scripts/n49b/explosive_nucleo.py`
裡 `evolve_zone()` 的 trajectory 函數。

### 工作分解

**Step B1**: 實作 Magkotsios+2010 Eq 1 trajectory (半天)
- T_peak 從 Woosley & Weaver 1995 風格的 piston energy 計算,**不用** Fryxell
- 冷卻:**指數** 而非 power-law 單調:
  ```
  T(t) = T_peak · exp(-t / (3 τ_HD))
  ρ(t) = ρ_peak · exp(-t / τ_HD)
  τ_HD = 446 / √(ρ_peak / 1 g cc⁻¹)  s
  ```
- 這比當前 `T ∝ 1/t` 更物理,冷卻更快,zone 脫離 Ne(α,γ)Mg 窗口更快

**Step B2**: 加 reverse shock 階段(半天)
- Magkotsios+2010 Table 1 給的是 (T_peak, τ) 雙參數 trajectory
- reverse shock 在 shock 到達 outer layer 後回傳,**二次加熱** inner layer 到 ~0.7 × T_peak
- 這是 12.75 的 Mg 終於被燒掉的關鍵 —— 二次加熱把 zone 短暫推進 T₉ > 3

**Step B3**: Piston 能量定位(半天)
- 不用 Fryxell 解析公式,用 **shock jump condition**:
  ```
  T_peak(m) = [E_SN / (internal energy ahead) · γ factor] · T_ahead(m)
  ```
- 即激波壓縮比 × 當地音速² 在能量沉積後的放大

**Step B4**: 跑 4 個 progenitor + 繪圖(半天)

### 總工作量

| 階段 | 時間 |
|---|---|
| B1 trajectory | 半天 |
| B2 reverse shock | 半天 |
| B3 piston placement | 半天 |
| B4 runs + fig | 半天 |
| **合計** | **1-2 天** |

### 精度期望

Magkotsios+2010 原 paper 聲稱他們的 trajectory family 和 full 1D hydro
對後處理 X_i 精度 < 10%。我們複刻 Sato+2024 的 12.75 post Mg/Ne 應能到
**0.7-0.9 範圍**(論文 0.75)。15.90 本來就對,不變。

---

## 決策樹

```
                             Start Phase D
                                  │
                                  ▼
                       執行 Plan A Day 1-3
                          (species + IC + α-net hook)
                                  │
                                  ▼
                       Day 4 HSE/shock stability
                          │           │
                       穩定          不穩定
                          │           │
                          ▼           ▼
                     繼續 A         切 Plan B
                          │           │
                          ▼           ▼
                     Day 5-6        1-2 天完成
                          │           │
                          └─────┬─────┘
                                ▼
                            驗收標準:
                       12.75 post Mg/Ne ∈ [0.6, 0.9]
                       15.90 post Mg/Ne ∈ [1.2, 1.4]
                                │
                                ▼
                        發 Suwa / 開 PR
```

## 驗收標準

**必須**:
- 12.75 M☉ post Mg/Ne ∈ **[0.6, 0.9]**(論文 0.75 ± 20%)
- 15.90 M☉ post Mg/Ne ≥ 1.0(shell merger 簽名保留)
- 4 個 progenitor 的 Fig 7 面板和論文視覺對應

**加分**:
- 不同 E_SN (0.5-4 × 10⁵¹ erg) 的敏感性表 — 論文沒做,我們加一個 bonus analysis
- Mass cut (1.4-1.8 M☉) 敏感性表
- 12.02 M☉ post Mg/Ne < 0.5(無 shell merger baseline)

## Fallback-of-fallback

如果 Plan B 也給不到 Mg/Ne < 1,說明:
- Sato+2024 的 0.75 數字依賴了我們沒 port 的 (α,p) / (p,γ) 副反應
- 或 explosive nuclear burning regime 比我們模型更複雜
- 這時**誠實對 Suwa 說**:"我們 Mg/Ne 趨勢在 shell-merger 和 no-merger 組別定性正確,
  12.75 intrusion case 定量差 30% 來自 (α,p) 副路徑缺失,在本項目 scope 外"

這不丟人 —— 對標的是 1D 爆炸核合成的**方法學完整性**,不是重跑一個 MESA。

---

## 後續 Stage 3 關聯

Phase D 如果完成,**12.75 的 pre-SN IC + 自洽 post-SN 結果** 可以直接當
Stage 3 的 cart_ale2 2D 對流實驗的 baseline:
- 1D 告訴我們 Mg/Ne 在 shell merger 後的分布
- 2D 告訴我們這個分布是 **radial** (1D 正確) 還是 **bubble-structured**(2D 新物理)
- 這才是真正 Suwa 組沒有的結果
