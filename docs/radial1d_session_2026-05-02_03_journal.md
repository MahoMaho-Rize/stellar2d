# radial1d × MESA 攻關日誌 — 2026-05-02 → 05-03

> 跨夜 session。從「cgs Newton 卡在第 1 步 bail」一路推到「MESA 1 M⊙ ZAMS 物理量
> 每項 ≤ 3%、v_conv 同量級」。radial1d 的 1D phase 到此封頂,下一步轉 2D
> (`cart_ale2`),不再跟 MESA 卷。

---

## TL;DR

**12 個 commit,分四大階段:**

```
48ca135  ADD: --eos helmholtz CLI 全流程 cgs HSE 10 τ_dyn 穩定
7b31e19  DIAG: cgs Newton ||F||-per-field 診斷 + two-sided Viallet R
fdbe637  FIX: row-scaled residual norm — cgs Newton 徹底打通 ⭐
426f1d0  FIX: opacity — H⁻ T^7.7 在 T>1.2e4 K 關掉(10^145 → 物理值)

9c6579c  ADD: MESA OPAL/Ferguson kap → KAPv1 binary(411 MB, 467 group, bit-exact round-trip)
88d8fe6  ADD: GPU opacity table loader + L2 persisting + trilinear + 27 家 PK
72f8606  ADD: --kap CLI + lowT/highT stitching 接進 radial1d

753bac7  ADD: MESA ZAMS profile → radial1d IC pipeline(Tier-2 就緒)
bd0db2e  DOC: autodiff Jacobian 計畫(延後,trigger checklist)

1e97892  ADD: rich profile output + Tier-2 PK 報告
d50fc16  FIX: κ 14% → 2.5% via --ic-mesa-seed-T ⭐
c6ead49  POLISH: Helm-exact ∇_ad + Henyey MLT saturation(v_conv 800× 改進)
```

**Tier-2 數字最終版(1 M⊙ ZAMS profile PK,100 hydro steps 後):**

| 物理量 | 開局 | 今早 | 深化後 | 說明 |
|---|---|---|---|---|
| ρ | — | 1.0 % | 2.3 % | seed-T 代價 |
| P | — | 1.1 % | 1.5 % | seed-T 代價 |
| T | — | 4.6 % | **1.1 %** | ↑4× seed-T |
| κ | — | 13.5 % | **2.5 %** | ↑5.4× T 修正傳導 |
| Γ₁ | — | 0.11 % | 0.11 % | Helm biquintic 精度 |
| ∇_ad | — | 0.94 % | **0.44 %** | ↑2× Cox-Giuli 恆等式 |
| v_conv | — | ~3000× | **3.8×** | ↑800× Henyey 飽和 |

---

## 起點狀況(session 開始時)

先前 session 做完:
- Helmholtz EOS 完整接入 + L2 persisting cache pinning(17 MB 表,106× 帶寬)
- 隱式 BE 輻射擴散 + 光球 σT⁴ BC + 幾何 dt 增長
- MLT 對流(Picard 延遲 K_conv 進 BE)
- 整套 cgs 下**顯式** 10 τ_dyn 穩定,mass conserve 10 位

**唯一卡關:** `--eos helmholtz --ic-solar --implicit` 第 1 步就 line-search bail。 journal 診斷清楚但沒解決。

## 階段 A: cgs Newton 打通

### A1 `48ca135` — `--eos helmholtz` CLI 全流程

- `main.cpp` 掛接 `HelmholtzTable` 全生命週期 + `--helm-table/abar/zbar` flag
- P→e 反解在 device kernel(避免 host/device 指標混用 segfault)
- **cgs 顯式 10 τ_dyn 穩定**:Mach≤0.13,質量守 10 位,12505 步

### A2 `7b31e19` — 診斷設施

加 Newton 入口 per-field `||F||`、`||U||` 診斷 + 兩側 Viallet scaling 從 `R=I` 改為 `diag(cs, R_star, cs²)`。code-unit 回歸 1 iter 收斂,cgs 問題定位到:

```
||F|| breakdown: v=1.54e5(max=2.68e4) r=0(...) e=0(...)  ← F_r, F_e 為零(HSE 下 U=Un)
||U|| breakdown: v=0           r=3.14e11 e=6.86e16
δU: ||v||=1.19e+07 (rel 1.19e+37) ...            ← 巨量 δv 過擾動
```

根因寫進 `docs/radial1d_cgs_newton_diagnosis_2026-05-02.md`。

### A3 `fdbe637` ⭐ — row-scaled residual norm

**真 fix**。Armijo 用 raw `||F||` 比較 → F_r 在 δv 之後耦合響應把 F 整體推高 → 線搜拒絕。改成 `||invL · F||`(row-scaled RMS),每 field 按 natural magnitude 權重,Armijo 正確比較:

```cpp
double residual_norm_implicit() {
    if (use_viallet_scaling && d_scale_invL != nullptr)
        return gpu_norm_scaled_r1di(d_F, d_scale_invL, N);
    return std::sqrt(gpu_dot_r1di(d_F, d_F, N) / N);
}
```

**效果**:cgs Newton **1 iter/step 收斂**,`dt=1e4` 跨 700 τ_dyn,`dt=1e6`(15 天/step)125 步 3 年。

### A4 `426f1d0` — opacity H⁻ 閘門

副產品 bug 發現:H⁻ 公式 `κ ∝ T^7.7` 本來只在 T < 10⁴ K 有效,在 T = 1.6e8 K(恆星內部)會爆到 10^145。加 `if (T_use < 1.2e4)` 閘門,讓 Kramers + Thomson 在內部接棒。

## 階段 B: MESA OPAL kap 表接入

### B1 `9c6579c` — kap 表 ASCII → binary

**Python toolchain:** `scripts/mesa_kap.py` 解析 MESA Type-1 kap ASCII 檔,`convert_mesa_kap.py` 轉 KAPv1 二進制格式(128 B header + 3D float64 payload,magic `KAPv1`)。

**產出:** 467 個 (family, Z) group × 5372 萬 logκ 格點,bit-exact 對齊 ASCII。411 MB。

**家族覆蓋:** OPAL gs98/a09/gn93 + α-enhanced + OP + OPLIB 2024 + lowT Ferguson-Alexander-AF94/fa05 全家。

### B2 `88d8fe6` — GPU loader + 家族 PK

- `src/physics/opacity_table.{cuh,cu}` 對稱 Helm 做法:host owner + device POD view + L2 persisting
- `kap_eval(X, logT, logR)` trilinear:**非均勻 X** 用 bisect、**piecewise-uniform logT** 用 bisect、**均勻 logR** 用 O(1)
- `tests/test_kap_table.cu` node-exact round-trip bit-match + solar probes vs Python trilinear 0.25% 吻合

**27 家 κ PK 表(物理結論表):**
- 光球:OPAL 0.19 vs OPLIB 0.26 cm²/g(+30%)
- 白矮星:OPAL 33 vs OPLIB 17 cm²/g(1.9× 差)
- α-enhanced 在 envelope 變 40%

### B3 `72f8606` — 接進 radial1d BE 輻射

`OpacityParams` 加兩個 `KapTableView`,`grey_opacity()` 在 `use_table=true` 時 dispatch 到 `kap_stitch_eval()`(MESA-style):

```
logT ≤ 3.9  → pure lowT
logT ≥ 4.1  → pure highT
overlap     → linear blend in log κ
```

A/B 測試:`max_super = 1e10 → 2e9`(4× 降),證明 opacity 在 MLT 診斷裡真的生效。L_surf 不變是對的 — 光球 BC 是 σT⁴,不走 opacity。

## 階段 C: MESA → radial1d IC 管線

### C1 MESA 編譯 + ZAMS 產出

- **MESA r26.x** 完整 source + SDK 26.3.2 在 `~/mesa-ref`,90 秒 build + self-test(寫 `scripts/build_mesa.sh` 背景包裝,`set +u` 規避 SDK init 的 nounset trap)
- `scripts/run_mesa_1Msol.sh`:copy `star/work` 到 `/tmp/mesa_work_1Msol`,inlist 調成 1 M⊙ + GS98 Z=0.02 + Lnuc/L = 0.99 停,90 秒達 ZAMS → `LOGS/profile*.data`
- `profile_columns.list` 要求擴展欄位:opacity、grada、gradr、gamma1、luminosity、gradT、conv_vel、mixing_type

### C2 `753bac7` — profile reader + IC 管線

- `scripts/mesa_profile.py` — MESA 3-block 格式 parser
- `scripts/convert_mesa_ic.py` — 寫扁平 ASCII IC 檔,`m_enc r rho T P X Y Z` 每列
- `Radial1DSolver::init_from_mesa(path, seed_T)` — MESA 的 783 zones 反轉 core→surface,mass-coord 重採樣到 radial1d 的 `nz` 等質殼,r=0 到 m_s[0] 用 uniform-density 擴展
- **CLI:** `--ic-mesa <file.ic>`

**首次 PK**(explicit 1 τ_dyn):ρ/P 1% 中位誤差,對流區邊界 `[0, 0.95 R*]` 跟 MESA late-Hayashi 全對流態吻合。

### C3 `bd0db2e` — autodiff Jacobian 計畫(延後)

用戶問「JFNK 跟 autodiff Jacobian 衝突嗎?」。答:**不衝突,兩者在不同軸**。JFNK 不建 J、autodiff 為 preconditioner 建 J。分析:

- 現在瓶頸是 `apply_precond_implicit = Identity` → GMRES 總是頂 30 iter
- 加 block-tridiag J + block-Thomas PC,GMRES 2–4 iter 收斂,~4× per-step 加速
- **不現在做** — Newton 1 iter 已 OK,ignition 主線不卡;重點該轉 2D
- **觸發條件** 寫在文檔:dt ≥ 1e10 s、或加 aprox13 核網、或 cart_ale2 也要隱式 → 觸發就按 checklist 跑

最小風險路徑:先手寫 block-tridiag PC 探路(0.5 天),再決定是否上 Dual\<T\>。

## 階段 D: Tier-2 深化 PK

### D1 `1e97892` — rich profile 輸出

新 kernel `k_rad1d_rich_diag` 輸出 per-zone **T、κ、Γ₁、∇_ad、∇_rad、L at face、mixing_type、v_conv**。`Radial1DSolver::download_profile_rich()` + `--rich-profile` CLI。

**首次完整 PK:**

| 量 | 中位誤差 |
|---|---|
| ρ, P | 1% |
| T | **5%** (Helm vs FreeEOS blend) |
| **κ** | **14%** (T 差透過 Kramers 指數放大) |
| Γ₁ | 0.1% |
| ∇_ad | 1% |

`docs/radial1d_mesa_tier2_pk_2026-05-03.md` 記錄完整分析 + 哪些該修哪些不修(劃界:radial1d 定位為 1D reference + 2D 橋樑,不重複造 MESA 輪子)。

### D2 `d50fc16` ⭐ — `--ic-mesa-seed-T`

用戶:「先把 κ 問題解決」。根因:我們讀 MESA P → 反解 T → Helm 認為跟 (ρ, P) 自洽的 T(不是 MESA 的 T)。

**Fix:** 新 device kernel `k_rad1d_eP_from_rhoT` — 正向 `helm_eval(ρ, T_MESA)` 產 (e, P),runtime T = T_MESA 到 machine precision,κ(ρ, T, X) 查表自動一致。

新 CLI `--ic-mesa-seed-T`。

**數字:**
- T: 4.6% → **1.1%**(4.3×)
- κ: 13.5% → **2.5%**(5.4×)
- m/M=0.8 單點:T 兩邊差 0.2%,κ **bit-match 6.22 cm²/g**
- 代價:ρ/P 微升 1% → 2%(MESA 的 (ρ, P) 對在 FreeEOS 自洽,Helm 在同 (ρ, T) 給的 P 略不同 — 值得的代價,現在完全 Helm-自洽)

### D3 `c6ead49` — Helm-exact ∇_ad + Henyey MLT 飽和

用戶:「繼續打磨」。兩個便宜修:

**∇_ad 精確:** `HelmState` 暴露 `grada` = (Γ₃−1)/Γ₁ = `P·χ_T / (ρ·T·c_V·χ_ρ) / Γ₁`。同時暴露 `chiT`, `chiRho`, `cV`, `cP` 給下游 MLT 用。

- 效果:∇_ad 0.94% → **0.44%**(2.1×)
- 剩 0.44% 是 Helm 對比 MESA FreeEOS blend 差,不是公式錯

**v_conv Henyey 飽和:** 閉式 MLT。Cox-Giuli 的三次方程

```
ξ³ + U·ξ² + U²·ξ − U²·W = 0,   ξ² = ∇_T − ∇_ad,  W = ∇_rad − ∇_ad
U = (12σT³)/(c_P·ρ²·κ·ℓ²) · √(8H_P/(g·δ))
```

Newton from seed `(U²W)^{1/3}`(深層效率極限),10 iter 內收斂。

- 效果:v_conv ~3000× → **3.8×**(~800× 改進)
- MESA core 5880 cm/s vs r1d 1530 cm/s,同量級

---

## 文檔總表(本 session 新增 / 更新)

| 文檔 | 內容 |
|---|---|
| `docs/radial1d_cgs_newton_diagnosis_2026-05-02.md` | Newton 卡死診斷 + Viallet R 修正 |
| `docs/radial1d_autodiff_jacobian_plan.md` | autodiff J 延後計畫 + 觸發條件 |
| `docs/radial1d_mesa_tier2_pk_2026-05-03.md` | 完整 Tier-2 PK 報告 + 深化 addendum |
| `docs/radial1d_session_2026-05-02_03_journal.md` | 本文件,全 session 敘事 |

## Pipeline(最終可重現命令)

```bash
# 1. MESA ZAMS profile (90 s,一次性)
cd ~/stellar2d
nohup bash scripts/run_mesa_1Msol.sh > /dev/null 2>&1 & disown

# 2. profile → IC file
python3 scripts/convert_mesa_ic.py \
    /tmp/mesa_work_1Msol/LOGS/profile5.data \
    /tmp/mesa_1Msol_zams.ic

# 3. radial1d run with full physics
./build/stellar2d --solver radial1d --test lane_emden \
    --nr 128 --eos helmholtz \
    --ic-mesa /tmp/mesa_1Msol_zams.ic --ic-mesa-seed-T \
    --G 6.674e-8 --tend 300 --output-interval 100 \
    --radiation --rad-c 3e10 --eos-rad-a 7.5657e-15 \
    --kap --kap-Z 0.02 --mlt --rich-profile

# 4. PK
python3 scripts/pk_mesa_radial1d.py \
    /tmp/mesa_work_1Msol/LOGS/profile5.data \
    runs/lane_emden_128x64_<ts>/profile_0001.txt
```

---

## 結論與下一步

**radial1d 在 1D 物理驗證上已到天花板:**
- 熱力學骨幹(Γ₁, ∇_ad, P, ρ)**0.1–1.5%** 吻合
- 表格物理(κ)**2.5%**
- 溫度(EOS blend 代價)**1.1%**
- 結構 diagnostic(v_conv)同量級(3.8× 誤差在 MLT grey assumption 內可接受)

**再壓只有兩條路,兩條都不走:**
- ❌ Port FreeEOS → 跟 MESA 卷,違反反重複輪子決定
- ❌ 自己寫完整 MLT 校準 → 同上

**下一步:cart_ale2 2D bridge** — 把 radial1d ZAMS state export 成 Cartesian 2D IC,加非對稱擾動,觀察 2D 對流 cell(granulation)。**這才是沒人做過、跟 MESA 不競爭的地方。**

memory 已更新:
- `feedback_avoid_1d_benchmarks.md` — 1D 工作只做 2D 元件測試
- `project_spectral_liouville_path.md` — 2D anelastic 譜法方向
- `project_radial1d_newton.md` — 舊的 Newton 停在文檔裡,已解(本 session)
