# radial1d cgs Newton 穩定性診斷 — 2026-05-02 晚

## TL;DR

成功把 `--eos helmholtz` 接通 main.cpp + radial1d,並在 cgs 下跑了 10 τ_dyn 的 **顯式** HSE 算例(Mach ≤ 0.13,質量守 10 位)。切到 **隱式 Newton + Viallet scaling** 後,第一步就線搜 bail。

今天的進展定位了根因,但 cgs Newton 完整修復 需要新的一輪工程(估 2–3 天)。診斷結果、半解決的改進、以及待辦路線圖都在此。

---

## 做到的事

1. `--eos helmholtz` CLI + HelmholtzTable 全生命週期掛接 main.cpp。
2. P→e 反解在 GPU 上做(新 kernel `k_rad1d_e_from_rhoP`),避開 host/device 指標混用的 segfault。
3. 顯式 RK2 + Helm 在 cgs 下通過 10 τ_dyn HSE 穩定性測試。
4. Newton 入口加 per-field `||F||`、`||U||` 診斷,方便以後回歸。
5. 修正 Viallet `R` = identity → `R` = diag(cs, R_star, cs²) 的結構性錯誤(原設計假設 `U_typ = 1`,對 cgs 完全不成立)。

## 核心病灶

IC 有 `v = 0`(整顆星靜止),但 `r ~ 7e10 cm`、`e ~ 7e16 erg/g`。三個場的典型量級跨 10 個數量級。

**單一全域 α = √ε_m ≈ 3e-8 的 JFNK finite-difference**,無論是否 Viallet,都會:
- v 方向:`α · v_basis` 乘到 `U_v = 0` 上,perturbation 變成 *絕對* 位移,perturb 過大,非線性響應把 F 推到比原本大一兩個數量級。
- r, e 方向:`α · U_typ_i` 產生 ppm 級相對位移,OK。

換成 **兩側 Viallet**(R 帶 U_typ 信息),FD linearity 對 r, e 是恢復了,但 v 仍然是 U_v=0 → **R_v = cs_f** 會強迫 v 在每次 FD 都被擾動到 `α · cs` ≈ 1 cm/s。這在 iter 0 還可以,但 GMRES 解出的 δv 是 *絕對* 的 1e7 cm/s 尺度,丟回 F:
- F_r_new = δr/dt - (v_n + δv) 中的 −δv 項佔 1e7,把 F_r 從 0 推到 1e7 等級
- 全域 `||F||_new` 因此比 pre 大 → line search 拒絕任何 α ∈ [1e-3, 1]

線性化其實 *是* 對的(GMRES 給的 δU 在線性意義上把 F 解為零);是 *Newton 的全域 ||F|| 比較* 在 U_v 為 0 的初始態下有盲點 —— 就算 Newton 步是最佳的,**執行之後** v 從 0 跳到 1e7,F_r 的 `R_r(U) = v` 項在全域 norm 裡就被放大成新的主項,overshoots pre-step norm。

## 半解決的改進(已 commit)

把 Viallet `R` 改成 per-row 的 cs / R_star / cs² 量級,擋住 FD 過擾動。在 code-unit 單位下仍然 1 iter 收斂(回歸測過)。cgs 下,線搜還是失敗,但 δr、δe 的相對數字已經從 1e-7 級恢復成 1e-3、1e-2 級,**GMRES 內部的 β、final|g| 比例關係正確** —— 線搜失敗的是 norm 比較規則,不再是 FD 失真。

## 真修復的方向(3 選 1)

### A. **按 F-field 分別比較 line search**
- 現在 Armijo 用單一 `||F||_new < 0.99 · ||F||_old`。
- 改成 `per-field` 比較:v / r / e 各自要降,或至少不能任何一項爆 10×。
- 優點:幾行代碼,不動 GMRES;同時保留對 KH contraction(v 會長出來)的兼容。
- 缺點:可能需要每步重新歸一化,MESA 做類似的事但有 dampening term。

### B. **把 U normalize 到 dimensionless space(「全域 reference scaling」)**
- 在 `init_implicit` 裡 precompute `U_typ = [cs_face, R_star, cs_cell²]`。
- 把 `d_U`、`d_Un` 都 / U_typ 存,Newton 全在 O(1) 空間做。
- R_hse 也同步 scale。
- 優點:概念最乾淨,跟 MESA 的做法對齊。
- 缺點:kernel 要同步改(primitives、diagnostics、R 評估都要知道 U_typ),工程量 ~2 天。

### C. **繼續 --no-viallet + 加入 damping factor on δv**
- 保留絕對 U,在 line search 前先 `δv_clip = min(δv, α_damp · cs)`。
- 本質是手動的 trust-region。
- 優點:不動線搜邏輯。
- 缺點:物理不乾淨,ignition 階段 v 會長到 cs 以上,damping 會拖慢。

**推薦 B** —— 根本做對,不是 patch。估 2 天。

## 今日 commit

```
48ca135  ADD: --eos helmholtz CLI end-to-end — cgs HSE 10 τ_dyn stable
<next>   FIX: Viallet R-scaling per-field + Newton entry diagnostics
```

## 延伸觀察(給未來自己)

1. `radial1d` implicit 的 `||F||=0` at iter 0 when no perturbation + HSE snapshot match → Newton 正確地什麼都不做。這是 well-balanced reference 生效的證據。
2. HSE defect `||R_hse|| = 2.69e+05`(N=192)在 cgs 下 *看起來很大*,但對 face r 來說是 `∂r/∂t - v ≈ 0` 的 discretization error,per-face 量級 `2.7e5/sqrt(192) ≈ 2e4` vs r ~ 7e10 → 相對 3e-7,**完全合理**。
3. 顯式模式下 MLT 把整顆 n=1.5 多方星標為對流(conv_mass_frac=1.0, max_super=5e34),這是對的 —— γ=5/3 多方星本身就全對流不穩。
4. 下輪攻擊:重做 U normalization(方向 B),先在 ideal_rad 測通,再接 Helmholtz。

---

*續自 `docs/radial1d_ignition_journey_2026-05-02.md` ——
今日結論:Helm 接通 ✅,Newton 仍卡 ❌,下次攻 U-normalize。*
