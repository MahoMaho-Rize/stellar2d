# 球極座標奇點處理:文獻標準做法(2026-05-02)

> 在投入 sphere_impl 實作前,調研 MUSIC / SLH / Athena++ / PLUTO / ZEUS 等代碼如何處理 r=0 和 θ=0/π 奇點。結論很出乎預期:**沒人真的「處理」奇點,大家都是繞過去**。

## TL;DR

| 代碼 | r=0 | θ=0/π | 本質 |
|---|---|---|---|
| **MUSIC** (Viallet 2011) | 切除 `r_inner=0.2 R★` | **切除** `θ∈[π/4, 3π/4]` 90° wedge + periodic | 全繞開 |
| **SLH** (Edelmann 2021) | 對球極:切除。全球:**cubed-sphere**(不用球極) | 同上 | 全繞開 |
| **Horst 2020** | `r_min=0.007 R★` | wedge | 切除 |
| **Athena++** | 球極模式只支持 wedge | `polar_wedge` BC(sign-flip reflect) | 對顯式夠 |
| **PLUTO** | `axisymmetric` BC(v_φ, B_φ 翻號) | 同 | 對顯式夠 |

**關鍵洞察**:
- 「用球極但經過極軸」這種配置在文獻裡**不存在**生產級實作
- 做整顆恆星球對稱必須選 **cubed-sphere**(SLH 做法),球極是死路
- 隱式代碼(MUSIC、SLH)**明確避開極軸**,因為 JFNK 對 metric singularity 極敏感

## 具體 MUSIC 做法(我們的 primary reference)

**A-type star 實際跑的配置**(Viallet 2011 Table row at line 861):

```
(r, θ) ∈ [0.2, 1.0] R★ × [π/4, 3π/4]   ← 90° 赤道 wedge
Radial BC:
  r=r_inner: non-penetrative (v_r=0) + stress-free (∂_r(v_θ/r)=0)
             + fixed energy flux F★(from 1D MESA)
  r=r_outer: same + fixed T_out
Angular BC:
  θ=π/4, 3π/4: periodic(這是關鍵,把 wedge 變成 torus-like)
Gravity:
  g(r) = -G·M(r)/r², precomputed
  M(r) = M_core + ∫ ρ_1D(r') r'² dr', M_core = 1D model 內核質量
  fixed in time,不做 self-gravity Poisson
```

**核心:不做 self-gravity,g(r) 從 1D MESA model 預計算,時間上固定。**

## r=0 奇點的標準處理(Viallet 2011 §3.2)

**4 個 radial BC 條件**:
1. `u_r = 0` at `r = r_in, r_out`(non-penetrative wall)
2. `∂_r(u_θ) − u_θ/r = 0` at boundaries(stress-free)
3. `F★` fixed energy flux 進 `r_in`(constant luminosity)
4. `T_out` Dirichlet at `r_out`

**Ghost cell 構造**(關鍵細節,Goffrey 2017 + Pratt 2016 Eq. 5):

```
ρ_ghost: discrete HSE extrapolation
  p_ghost = p_interior + ρ̄·|g(r)|·Δr   (ρ̄ 是 interface-averaged ρ)
  ρ_ghost 由 EOS(p_ghost, T_fixed) 反推

v_r_ghost = -v_r_interior   (reflect,for non-penetrative)
v_θ_ghost = +v_θ_interior   (extrapolate,for stress-free)
e_ghost: 由 fixed F★ 和 1D luminosity 反推
```

**我們之前 ||F||→∞ 的 bug 很可能就是 ghost 密度用 zeroth-order extrapolation 而不是 discrete HSE,Goffrey 2017 明確警告這會導致長期 HSE drift**。

## JFNK 的特殊要求(對我們最重要)

Viallet 2016 §3 明示 JFNK 做法:

**Rule 1**:**Ghost cells 不能是獨立 DOF**。必須是 interior state 的 algebraic function,在 F(X) 裡面計算。這樣 J·v = [F(X+εv) - F(X)] / ε 會**自動**正確穿過 BC — 不用特別處理邊界行。

**我們 fas2 ghost 處理可能違反這個**,需要查。

**Rule 2**:FD Jacobian 在邊界附近容易 cancellation。MUSIC 2016 用 **physics-based preconditioner**(PBP)避開,我們先用 identity precond,不行再加。

**Rule 3**:**implicit ≠ 無條件穩**。MUSIC 2011 時間步依然受 **最小 cell** 限制(CFL)。如果我們把 axis 包進 domain,axis cell `Δr · r·sin(θ)·Δθ → 0`,dt 還是塌陷 — 所以必須 excise。

## 已知失效模式(gotchas,有文獻引用)

1. **Zero-area θ-face flux**:`r²·sin(θ)·Δr·Δφ → 0` 導致 `0×NaN = NaN`。**解法**:periodic BC 在 wedge 內,根本不 evaluate 極軸 face。
2. **Angular momentum 保持**(Mignone 2007):ρ·v_φ·r·sin(θ) 形式才守恆,不是 ρ·v_φ。我們 2D axisymmetric 不涉及 φ,但若加旋轉要注意。
3. **Ghost ρ 0 階外插破壞 HSE**(Pratt 2016 Eq. 5 vs 0-order):很可能是我們現在的 bug。
4. **JFNK ε 在邊界 cell 的 cancellation**:用 `ε = sqrt(ulp)·(1+‖X‖)/‖v‖` 且 unit-normalize v̂(Trilinos NOX)可緩解。fas2/cart_impl 已做此 fix。
5. **Implicit CFL 限制**:wedge excision 後,最小 cell 由 `Δr` 或 `r_inner·Δθ` 決定,都是 O(1/N),不再 O(1/N²)。

## 對我們 sphere_impl 計劃的修正

### 修正前的 Phase 0(錯誤)
> 「加 r_inner excision,加極軸處理,驗證 HSE」

這個假設「存在一個正確的極軸處理能讓 Newton 穩」。**文獻告訴我們沒有這種東西**。

### 修正後的 Phase 0(正確)

**核心設計決策:用 MUSIC 的 wedge 配置**
- 徑向:`[0.2 R★, 1.0 R★]`(切除內核)
- 角向:**`[π/4, 3π/4]`(90° 赤道 wedge)+ periodic BC**(完全避開極軸)
- 自引力:**不做 Poisson**,g(r) 從 Lane-Emden 1D model 預計算,固定

**這直接消除兩個奇點,無需任何特殊處理**。剩下的就是做好 HSE-aware ghost extrapolation(Pratt 2016 Eq. 5)。

### 具體實作清單(1 週內可完成)

**Phase 0A(2 天):幾何 + HSE**
1. clone cart_impl → sphere_impl,幾何改 `(r, θ)`,metric 用 `r²·sin(θ)·Δr·Δθ`
2. wedge mesh:`r ∈ [0.2 R★, R★]`, `θ ∈ [π/4, 3π/4]`,angular BC periodic
3. 徑向 BC:HSE ghost extrapolation (Pratt 2016),non-penetrative
4. Gravity:預計算 `g[i] = -G·M(r_i)/r_i²`,M_core 用 Lane-Emden 1D integral
5. 驗證:`perturb=0` 能 HSE 機器精度穩定到 `t = τ_acoustic = R★/c_s` 不炸

**Phase 0B(2 天):擾動演化**
1. `perturb=1e-4` 看擾動起步
2. 128² 驗證 scaling
3. 和 cart_impl 同樣跑 Ma=1e-4 → 1e-2 indicator

**Phase 0C(2 天):GPU 硬化 + VTK + diagnostics**
1. 確保所有熱路徑 GPU-native,0 host roundtrip
2. Spherical-aware VTK writer
3. M/E/AM 守恆 diag

**Gate 條件**:Phase 0A 必須通過(HSE 穩)才進 0B。

## 為什麼 MUSIC 不做整顆星?

關鍵觀察:**MUSIC 從來不號稱做「整顆恆星」**。Viallet 2011 明確說是「the radiative envelope」(輻射外層)。對**內核**他們用 1D MESA,對**外層**用 2D MUSIC(wedge),**不自引力耦合**。

這代表:**我們對「做 protostellar 整顆星」的目標,球極 wedge 也做不到**。只能做「外層對流 wedge」。

**如果真要做整顆球對稱恆星(包括核心)**,唯一路線是:
- **cubed-sphere**(SLH 做法,Calhoun 2008 JCP)— 完全避免球極座標,用 6 個 logically-Cartesian patches 拼成球面
- 工作量:2-3 個月,遠超 sphere_impl wedge 的 1-2 週

## 對老闆 / NAOJ cluster 任務的對齊

**如果目標是做恆星對流(主序內部、紅巨星外層等)**:
- sphere_impl wedge + 1D MESA gravity + 2 週完成 → 能在 NAOJ cluster 跑 256²+,和 MUSIC 是 direct competitor
- **這就是生產級 path**

**如果目標是從 pre-MS 到主序的整個 protostellar 演化**:
- 需要整顆星 + self-gravity,sphere_impl wedge 不夠
- cubed-sphere 2-3 個月,或者放棄隱式改 explicit
- 應該先和老闆確認目標

## 推薦實作順序

1. **今天**:和老闆對齊目標(wedge 對流 vs full 整顆星)
2. **本週**:sphere_impl Phase 0A(2 天)— HSE wedge 穩定
3. **下週**:Phase 0B + 0C(4 天)— 擾動演化 + 硬化
4. **第 3 週**:benchmark vs MUSIC paper,寫 paper or internal report
5. **第 4+ 週**:根據需要加輻射、真實 EOS、核反應

## 附錄:文獻定位

| 來源 | arXiv | 用途 |
|---|---|---|
| Viallet 2011 A&A 531 A86 | 1103.1524 | §3.1 mesh, §3.2 BC 4 條, wedge 配置 |
| Viallet 2016 A&A 586 A153 | 1512.03662 | JFNK 算法 §3, physics-based preconditioner |
| Goffrey 2017 A&A 600 A7 | 1610.10053 | HSE benchmark BC,Pratt 2016 Eq. 5 引用 |
| Edelmann 2021 A&A 652 A53 | 2102.13111 | SLH cubed-sphere |
| Horst 2020 | 2006.11286 | r_min=0.007 R★,solid-wall |
| Stone 2020 Athena++ | 2005.06651 | `polar_wedge` BC 源碼 |
| Mignone 2007 PLUTO | astro-ph/0701854 | axisymmetric BC,angular momentum form |
