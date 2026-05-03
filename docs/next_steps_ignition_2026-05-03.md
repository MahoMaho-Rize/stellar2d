# 下一步:修復 1 M☉ pre-MS 點火演化凍結 — 2026-05-03

> 前置 context:`docs/next_steps_2026-05-03.md`(cv 修 + autodiff) 已完成,
> 基礎設施全部就位。然後加 rad-in-F + τ=2/3 photospheric BC,跑 nr=1024
> 生產圖(`docs/images/ignition_1024_1Msol_*.png`)— **圖展示了 frozen
> evolution,不是 real pre-MS**。本文件拆清後續修復路徑。

---

## 現況診斷(用戶精準指出的異常)

### nr=1024 生產跑(commit `f7f6f7f`)的病徵

1. **T_c 不動**:10¹⁶ s(≈ 300 Myr)跨度,T_c 從 4.1796e6 → 4.1785e6,
   僅 0.002 dex。真實 1 M☉ pre-MS 應到 1.5×10⁷ K 才點燃 pp-chain。
2. **ρ_c 不動**:1.4271 → 1.4283 g/cc。真實 ZAMS 核心應 ~100 g/cc。
3. **L_surf 瞬間暴跌**:10³⁹ → 10⁻² L☉ 在 2-3 步內,這是 IC relaxation
   artifact 不是物理演化。
4. **T_eff 瞬間暴跌**:7×10⁴ K → 1000 K(Helm table floor)。Hayashi
   track 應穩在 4000 K。
5. **L_nuc < 10⁻⁴ L☉ 恆定**:pp-chain 沒被點燃 = T_c 太低。
6. **整個 dashboard 是平線** — classic "frozen state" 模擬失敗症狀。

### 兩個對照比較

| 版本 | Operator split BE-rad | T_c 演化 | 最終結果 |
|---|---|---|---|
| nr=128, commit 早於 `c80d51f` | ON | **會動**(4.14e6 → 3.5e6 K, ρ 1.4 → 2) | step ~20 Mach 爆炸 |
| nr=1024, commit `f7f6f7f` | OFF (rad-in-F) | **完全不動** | 穩但 frozen |

**結論**:rad-in-F 把演化驅動力幹掉了。operator-split 那版至少數值上有
物理驅動(只是沒有 quasi-static balance,所以 Mach 爆),rad-in-F 版穩
卻沒有驅動。

---

## 三個可能的根因(優先級降序)

### 根因 A:R_hse 減法吃掉 rad drive(最可能)

現在 F:
```
F(U) = (U - Uⁿ)/dt - (R(U) - R_hse)
```

`snapshot_hse_implicit` 在拍 R_hse 時:
- 暫時 disable nuclear_enabled (OK,nuclear 是源項不應進 HSE)
- **我在 f7f6f7f 加了** 暫時 disable radiation_enabled (**也許錯了?**)

邏輯:HSE 的定義是 ∇P = -ρg,純重力 + 壓力。但我們現在 R 裡**包含** rad
divergence 項,R_hse 裡**沒有** rad 項。所以 `R - R_hse` 仍然有 rad 貢獻
— 應該是對的。

但 Newton 每步找到的 U 會使 R(U) ≈ R_hse(因為 (U-Uⁿ)/dt 在大 dt 下 ≈ 0)。
也就是 Newton 把 U 調到「hydro balance + rad = 0」的狀態。**rad = 0 就是
凍結的原因** — Newton 把 U 調到讓 rad divergence 自抵消的狀態,而不是讓
rad driving contraction。

**驗證辦法**:
1. print 每步 ||R_rad term only||,看是不是很小
2. 完全去掉 `R_hse` 減法,跑裸 F = (U-Uⁿ)/dt - R(U),看 Newton 會不會
   真的抓到 rad drive

### 根因 B:rad BC 在光學厚 zone 裡用 Stefan 不對

nr=1024 最外 zone τ_z = κ·ρ·Δr 能達 10^10 以上。Stefan 只在**真空界面**
(τ≪1)才是對的 BC。在光學厚極限,F_rad = (c/3κ)·∇(aT⁴),是內部 flux,
surface BC 是 Eddington → T(τ=2/3) = T_eff。

我現在 τ=2/3 scan 只找到 T_phot,仍用 σ·T_phot⁴ 當 outward flux。但這個
T_phot 是 optical τ=2/3 的 **zone 中心 T**,不是 surface T_eff。Eddington
近似:T(τ)⁴ = (3/4)(τ + 2/3)·T_eff⁴,代入 τ=2/3 → T_phot⁴ = T_eff⁴。
所以**如果** τ=2/3 正好在 zone boundary,公式對。但我們 τ 躍變在 zone
內,掃描找到的 zone 可能 τ 已 ≫ 2/3,T_phot 就比 T_eff 低太多。

**驗證辦法**:
1. 掃描內插:`T_phot⁴ = T_inner⁴ + (T_outer⁴ - T_inner⁴)·(τ_target - τ_acc)/dτ_zone`
2. 比 MESA 在同 state 的 T_eff / T_phot,看我們差多少

### 根因 C:MESA IC 的 MLT 帶沒有被我們解析 / 處理

pre-MS 1 M☉ 大量 convective envelope(80% 質量對流)。我現在 **operator-
split BE-rad 路徑有 MLT**,但 **rad-in-F 路徑沒有**。沒有 MLT 時,輻射
解不動對流區 → 光學厚 envelope 鎖死 → 能量沒法運出核心 → T_c 動不了。

**驗證辦法**:
1. 看 nr=1024 跑的 `conv_mfrac` 是否合理(應 ~0.8)
2. 對流帶有沒有實際運作(Lagrangian 1D MLT 是 conductivity-like diffusion,
   應讓 convective 帶 T 梯度 ≈ ∇_ad)

---

## 修復路徑(按優先級)

### Phase 1:診斷實驗(0.5 天)

**目標**:定位是 A/B/C 哪個,用最少改動:

1. **A 診斷**:
   - 加 `--no-rhse`:F = (U-Uⁿ)/dt - R(U) 不減 R_hse
   - 跑一段 1e12 s 看 T_c, ρ_c 有沒有動
   - 如果動了:**根因 A 確認**,rad-in-F 正確,R_hse 是 culprit
   - 如果還不動:轉 B

2. **B 診斷**:
   - 加 `--rad-surf-bc stefan|eddington|diffusive`
   - 切回 operator-split BE-rad(用 `--rad-be-split` flag 已經有)
   - 對比兩路 1e10 s 演化曲線

3. **C 診斷**:
   - grep `nr=1024` log 看 `conv_mfrac`
   - 如果 < 0.1 就是對流 detection 失效

### Phase 2:根因修復(1-1.5 天)

**case A — R_hse 移除或重新設計**:
- 簡單版:直接砍 R_hse 減法,用 scaled F 讓 Newton 看到真實 drive
- 代價:失去 well-balanced HSE roundoff 優勢。在 cgs 下 well-balanced
  是為了 e cores have e ≈ 10¹⁶ 但 F 要算到 1e-8 量級精度的 reason。
  沒 well-balanced 要重新調 tol。
- 進階版:snapshot R_hse 用 **lagged** state(N 步前的 HSE),不追快照

**case B — 改 Eddington atmosphere BC**:
- τ=2/3 掃描加內插,T_phot 取插值而不是整個 zone
- 或放棄 grey atmosphere,用 MESA-style 外部 atmospheric table

**case C — 把 MLT 接回 rad-in-F**:
- rad-in-F 的 residual kernel 加 convective flux term
- Picard-lag K_conv 同樣 structure(跟 rad 一樣處理)
- 算是 rad-in-F 的自然擴展

### Phase 3:驗證標準(2 天)

完全通過的判定:

1. **HR 圖**顯示真實 Hayashi descent:
   - 起點在 log T_eff ≈ 3.7, log L ≈ 1
   - 垂直下降 ~1-2 decade in L(Hayashi 幾乎固定 T_eff)
   - 曲線在 ~30 Myr 尺度真實演化(不是 IC transient)

2. **T_c - ρ_c** 圖顯示 KH contraction:
   - 從 (4.14e6, 1.4) 朝 (1.5e7, 100) 方向移動
   - 穿過 pp ignition 線

3. **守恆**:L_surf ≈ -dIE/dt(virial 扣除 PE),在每個 t 都成立

4. **Newton 穩定**:Mach max < 0.01 全程

5. **與 MESA 同齡 snapshot 比**:
   - T_c, ρ_c, L, T_eff 在 10% 以內
   - ρ(r), T(r), L(r) profile 誤差 < 20%

### Phase 4:完整 ZAMS 到達(1 天)

一旦 Phase 3 過,推到 T_c = 1.5×10⁷ K 看 pp-chain 點燃:
- L_nuc 從 10⁻⁸ L☉ 漲到 1 L☉
- L_nuc / L_surf 從 0 漸趨 1
- 核心 X(hydrogen) 開始下降(species on)

---

## 風險與不確定性

- **R_hse 移除可能暴露新 stiff problem**:Newton 以前靠 well-balanced
  把大量 roundoff 吃掉,不減後每步 ||F||_raw 從 1e-8 跳到 1e16,Newton
  tol 要相應改。
- **Eddington BC 在 sub-zone 分辨率下不 universal**:表面分辨率依賴性
  強,可能要 nr=2048 或 non-uniform zoning 才穩。
- **rad + MLT 一起 implicit**:數學上多出一組耦合,block-tridiag PC
  可能退化到 block-pentadiag(MLT 有 zone ±2 耦合),PC 結構要重做。

---

## 時序估計

| 階段 | 估時 | 里程碑 |
|------|------|-------|
| Phase 1 診斷 | 0.5 天 | 鎖定 A/B/C |
| Phase 2 修復 | 1-1.5 天 | T_c 開始真實演化 |
| Phase 3 驗證 | 2 天 | 可重現圖 + MESA 10% 一致 |
| Phase 4 ZAMS | 1 天 | pp-chain ignition demo |

**合計:4.5-5 天**到真正能發表的 1D ignition demo。

---

## 保留資產

本次工作產生的正確基礎設施(**不會改**):

- `src/physics/dual.cuh` — forward-mode AD(驗證通過)
- `src/physics/helmholtz_eos_dual.cuh` — Helm AD 相容
- `src/gpu/radial1d_residual_dual.cuh` — Dual-templated F kernel(結構對)
- `src/gpu/radial1d_implicit.cu` `jfnk_matvec_ad` — exact J·v matvec
- block-tridiag PC assembly(同樣 AD / FD 都能用)
- `helm_T_from_rho_e` analytic cv + grid clamp fix(commit `b8bb90d`)
- `scripts/plot_hr_tc_rhoc.py` — 診斷 / 發表圖 pipeline
- `docs/images/ignition_1024_1Msol_*.png` — **保留作為 bug 症狀快照**,
  修復後可以對比

---

## 引用 / 對比標準

- MESA r26 pre-MS → ZAMS 軌跡(我們 `run_mesa_1Msol.sh` 可重現)
- Kippenhahn, Weigert & Weiss 2012 §23 "Pre-main-sequence evolution"
- Hayashi 1961 forbidden zone
- Tout+ 1996 ZAMS analytic fits
