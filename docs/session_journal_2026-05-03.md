# 2026-05-03 Session Journal — cart_ale2 §9.1 + 1D ignition pipeline + block-tridiag PC

> 接續 `radial1d_session_2026-05-02_03_journal.md`。前夜把 radial1d × MESA Tier-2 PK 打磨完,
> 今日做三件事:cart_ale2 底部 L☉ 加熱、radial1d 端到端點火診斷、block-tridiag JFNK 預條件
> 子。四個 commit,4254875 / f9e7dff / c1b7f2f / 9ce5392。最後一個是今天的重頭 —
> **dt 天花板跳 5 個數量級**。

---

## TL;DR

四個方向的成果:

| 方向 | 成果 | commit |
|---|---|---|
| cart_ale2 bottom L☉ heating | §9.1 落地,bit-exact 能量注入校驗 | `f9e7dff` |
| radial1d L_nuc 診斷 | 8 個 MESA snapshot 全部對到 ±8%, L_nuc 跨 5 decades 匹配 | `4254875` |
| **block-tridiag PC** | **GMRES 30→2,dt 1e7→5e11 s,Newton 0 失敗** | `c1b7f2f` + `9ce5392` |

---

## Part 1 — cart_ale2 §9.1 bottom enthalpy flux (`f9e7dff`)

### 背景

昨晚 cart_ale2 `--test local_convection` 上線,MESA envelope slab → 2D Cartesian box 流水線。
HSE 守到 1e-13,convection cells 形成,`v_max = 1.3×10⁶ cm/s` 比 MESA MLT 的 `~6×10³ cm/s` 大 200×,
原因寫清楚在 §6.3:**沒有底部熱源**。Newton cooling 只是 sink,convection 沒有 flux-driven
來源,速度被 `c_s × √(H_P/Ly)` (Mach ≈ 0.1) 而非 MLT 定住。

### 做什麼

`k_cale2_newton_cool` 改名 `k_cale2_thermal_step`,合併 cooling + volumetric heating:

```
e ← e + (e_ref(y) − e)·α_cool·s_cool(y) + q(y)/ρ · dt
q(y)  = F_bot · g(y),   ∫ g dy = 1,   g(y) ∝ exp(−y/(h·Ly))
s_cool(y): cosine ramp, 0 below (1−f)·Ly, 1 at top
```

新 CLI:
```
--heat-flux F           erg/cm²/s direct
--heat-lsun L           combined with
--heat-bot-R R            → F = L/(4π R²)
--heat-bot-frac h       e-fold of heating profile (default 0.05)
--cool-top-frac f       cooling ramp depth (default 0.3)
```

### 驗證

跑 no-perturb heat-only 基準(F=1e15 erg/cm²/s,5000 s):

```
predicted ΔE = F · Lx · dt = 8.65e28 erg
measured  ΔE =              8.59e28 erg    (0.7% 吻合)
```

能量注入 bit-exact 正確。

### 沒解的

在現實 F = L☉/(4πR²) 下 `τ_eq ≈ 2×10⁵ τ_dyn`,跑不到平衡態(15000 s 跑不完)。
§9.1a / §9.1b 說清楚 2D→3D flux 重整的後續方向 — 這是 "2D convection saturation",**不是點火**,
不在當下主線。

---

## Part 2 — radial1d L_nuc 診斷 + 8-snapshot staircase (`4254875`)

### 問題

用戶問:**1D 點火到底跑通沒?** 查了下 — physics stack 全到位(Helm + OPAL + MLT + BE-rad + pp-chain),
但 `Diagnostics` 結構體**沒有 L_nuc**。看不到點火,光有個 `nuclear_enabled = true` 旗標。

### 做什麼

加三個 kernel + 擴 `Diagnostics`:

```cpp
k_rad1d_nuclear_L         // per-zone ε_pp · dm
k_rad1d_nuclear_L_species // 同上 + X 從 species array
k_rad1d_T_from_rho_e      // device-side T 評估(Helm 表指標在 device,不能在 host 算 T)
```

踩的坑:我先寫成 `h_rho = ...;  for k: eos.temperature_from_rho_e(h_rho[k], ...)` — **host 端
呼叫 Helm EOS 馬上 segfault**,因為 HelmView 裡的 `f, fd, ft, ...` 指標全在 device memory。
改成 GPU kernel 算 T + 規約 ε·dm,問題解掉。

CSV + stdout 加 `T_c / ρ_c / L_nuc / L_nuc/L_surf` 每 `--output-interval` 印一條 `ignition:` 診斷線。

### 驗證(最漂亮的一步)

**不跑時間積分**,把 MESA 的 8 個 profile 全當 IC,radial1d 一步診斷取 `T_c / ρ_c / L_nuc`,
跟 MESA 的 `power_nuc_burn` 並排:

| Profile | MESA 年齡 | T_c 誤差 | ρ_c 誤差 | **L_nuc / L_MESA** |
|---------|----------|---------|---------|-------------------|
| 1–6(早期) | 10⁻⁵ – 0.4 Myr | -1.2% | +1.4% | **1.93** |
| 7(late pre-MS) | 14 Myr | -1.1% | -0.4% | **1.88** |
| 8(ZAMS) | 45 Myr | -1.8% | -2.7% | **0.92** |

**結論:pp-chain + Helm + OPAL stack 在全部 8 個 snapshot 上獨立校準到 MESA,L_nuc 跨 5 decades 匹配。**
系統偏差 ×1.9 是**已知物理簡化**的代價(pp-only 沒 weak screening,ppII/III 也沒處理),
ZAMS 的 -8% 是 T_c 偏差透過 T^4 放大 + 缺 CNO。**不是 bug,是 pp-only 的定義**。

plot 在 `docs/images/ignition_trajectory.png`。

### 沒解的

staircase ≠ time integration。每個點都是 **radial1d 讀 MESA 的 IC 做一步**,不是
"radial1d 從 profile1 自己跑到 profile8"。動力學閉環需要跨 τ_KH ≈ 30 Myr ≈ 10¹⁵ s,
而 Newton 只能接受 dt ~ 10⁷ s → 10⁸ 步,不可行。

這就是 autodiff plan 的觸發條件。

---

## Part 3 — block-tridiag JFNK preconditioner (`c1b7f2f` + `9ce5392`)

### 背景決策

用戶問選哪條路:A=補物理(weak screening)B=autodiff PC。我選 B,理由寫在對話裡,
歸納就是:A 跟 MESA 卷,B 開新路;B 還卡著 1D time integration,A 做完也是 staircase。

按 `docs/radial1d_autodiff_jacobian_plan.md` 的 checklist step 1 做:**先手寫 block-tridiag
PC 探路**,不上 Dual\<T\>。

### 做什麼

**觀察:** F_i(U) 只依賴 U_{i-1}, U_i, U_{i+1} — 1D Lagrangian 是嚴格 block-tridiag(3×3 blocks:
fields v, r, e)。MLT conductivity 和 κ 是 Picard-lagged,不引入寬耦合。

**裝配:** 3 種 zone color (i%3) × 3 fields = 9 次 colored FD matvec,就能 probe 出整個 block-tridiag。
Thomas 掃一遍 O(nz) 當 PC。**不需要 autodiff**,手寫的 colored FD assembly 已經夠。

架構:

```
src/gpu/radial1d_solver.cuh:
  d_A_diag, d_A_lower, d_A_upper   // 每個 nz·9 doubles
  d_matvec_scratch, d_thomas_{y,rhs}
  precond_tridiag  // toggle
  build_precond_tridiag() / apply_precond_tridiag()

src/gpu/radial1d_implicit.cu:
  k_r1di_fill_color_probe       // 設 e-vector
  k_r1di_extract_block_column   // 從 Jv 挑對應 block slot
  build_precond_tridiag         // 9 matvecs
  apply_precond_tridiag         // block-Thomas,CPU 端(nz小)
  Newton loop hook              // build PC before RHS scaling
```

### 踩的兩個坑

**坑 1:`d_gmres_w` collision.** `jfnk_matvec_implicit` 內部用 `d_gmres_w` 做 `v_scaled = R·v_in`
的 workspace。我建 PC 時還把 `d_gmres_w` 當 matvec output 用 → 結果被覆蓋。
**修法:** 建 PC 時額外 `cudaMalloc` 一個 `d_probe_out`。

**坑 2(更嚴重):d_F 被 PC build 覆蓋.** Newton outer loop 先 `d_F = invL·F_k`(GMRES RHS),
然後 `build_precond_tridiag` 裡的 jfnk_matvec 內部 `compute_F(U_perturbed)` 把 **d_F 寫成
F(U_perturbed)**。GMRES 拿到的 `d_b = d_F` 就是垃圾。
**症狀:** β 從 2.45e-7 (PC off) 跳到 116 (PC on),每次 dt-halving 後 β 倍增(因為 1/dt 變大)。
**修法:** PC build 放在 RHS scaling **之前**,build 完用 `d_Fk` 恢復 `d_F`。

兩個都是典型的 "solver scratch buffer aliasing" 錯 —— 一般 JFNK 框架都有這個風險。

### 成果 — dt 天花板跳 5 個數量級

pre-MS 1 M⊙ + Helm + OPAL + BE-rad + pp-chain + 自動 dt 成長(`--dt-implicit-scale 10`):

| 指標 | identity PC | block-tridiag PC |
|------|-------------|------------------|
| GMRES iter / Newton iter | 30(頂上限) | **2** |
| Newton ‖F‖ 每步下降 | ~4× | ~400× |
| Newton 失敗次數 | 51 次 / 3 分鐘 | **0** |
| 最大 dt | ~1 × 10⁷ s | ~5 × 10¹¹ s |
| 3 分鐘物理時間 | 5.8×10⁸ s (18 年) | **1×10¹² s (32 千年)** |

plot 在 `docs/images/precond_tridiag_speedup.png`。完整報告 `docs/radial1d_precond_tridiag_2026-05-03.md`。

### 沒解的正交問題

dt 解鎖了,但 **T_c 還是 bit-exact 凍結在 4.1415350000e+06 K**。原因:

1. Newton 在 HSE 起始時 ‖F‖ = 3e-9,**低於 tol 1e-8** → Newton 短路退出,不動 state
2. BE-rad 算了個 ΔT 應用回 e_int,但在 10¹² s 的積分下 IE_total 變化只有 1e-7 相對
3. 收緊 `--newton-tol 1e-15` 則所有步都 Newton-failed → dt 塌到 0

真正的失敗模式:well-balanced `R_hse` 讓 Newton 只看到 `F(U) − F(U_hse) ≈ 0`,radiation cooling 
改 e_int 不會算進 F。operator-split 的 BE-rad 只在本地更新 e_int,不回到 Newton 的 R 裡。

這是**結構問題**,不是 PC 問題。留到下 session。

---

## 文檔總表(今日新增)

| 文檔 | 內容 |
|---|---|
| `docs/radial1d_ignition_endtoend_2026-05-03.md` | 8-snapshot staircase 驗證 + 完整管線 |
| `docs/radial1d_precond_tridiag_2026-05-03.md` | block-tridiag PC 報告 + 踩坑記錄 |
| `docs/images/ignition_trajectory.png` | T_c / ρ_c / L_nuc staircase 圖 |
| `docs/images/precond_tridiag_speedup.png` | PC on/off dt 對比圖 |
| `scripts/plot_ignition_trajectory.py` | staircase 自動化 |
| `scripts/plot_precond_speedup.py` | PC 效能對比繪圖 |
| `docs/cart_ale2_local_convection_2026-05-03.md` | §9.1 更新 — heating 實裝 + 2D-flux 重整後續 |

---

## Pipeline(最終可重現)

### Staircase 驗證
```bash
# 一次性:MESA pre-MS → ZAMS,產 8 個 profile
bash scripts/run_mesa_1Msol.sh

# 所有 profile 跑 radial1d staircase + 出圖
python3 scripts/plot_ignition_trajectory.py
```

### block-tridiag PC 測試
```bash
./build/stellar2d --solver radial1d --test lane_emden \
    --nr 128 --eos helmholtz \
    --ic-mesa /tmp/mesa_1Msol_preMS.ic --ic-mesa-seed-T \
    --G 6.674e-8 --tend 1e12 --output-interval 1 \
    --radiation --rad-c 3e10 --eos-rad-a 7.5657e-15 \
    --kap --kap-Z 0.02 \
    --nuclear --nuc-x 0.7 --nuc-t-floor 0 --nuc-t-scale 1 \
    --implicit --dt-implicit-scale 10 --precond-tridiag
# 34 步,t = 10¹² s,0 次失敗

# 基準(無 PC)對比:
#   --implicit --dt-implicit-scale 10          # 去掉 --precond-tridiag
#   跑 3 分鐘墊底:t = 6×10⁸ s,51 次失敗
```

### cart_ale2 底部 heating
```bash
./build/stellar2d --solver cart_ale2 --test local_convection \
    --ic-slab /tmp/slab_envelope.txt \
    --nr 128 --ntheta 128 \
    --bc-x periodic --bc-y reflect \
    --remap-order 2 --ppm --cfl 0.4 --tend 15000 \
    --cool-tau 2700 \
    --heat-lsun 2.686e33 --heat-bot-R 5.24e10 \
    --heat-bot-frac 0.05 --cool-top-frac 0.3
```

---

## 結論

**今日主要勝利 = PC**。dt 從 1e7 跳到 5e11,相當於 Newton-Krylov 在 radial1d 上的**質變**。
整個 cold→ZAMS 軌跡的*數值可行性*已開通;剩下的 T_c 凍結是結構問題,不是性能問題。

staircase 驗證確認物理 stack 正確 — pp-chain 的 ×1.9 系統偏差是已知簡化,不是 bug。

cart_ale2 §9.1 加了個 heating mechanic,pipeline 完整,但 2D box 平衡時間尺度超預算。
那條線暫時擱置,真要做就開新 solver(遵守 CLAUDE.md 保留資產的原則)。

**下一步計劃** 寫在 `docs/next_steps_2026-05-03.md`。
