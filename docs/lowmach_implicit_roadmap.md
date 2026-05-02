# 低馬赫隱式求解器路線圖

> 2026-05-02. 基於 deep-research 對 Viallet 2011/2016、Miczek 2015、Edelmann 2021、Goffrey 2017、Andrassy 2022、MAESTROeX (Fan 2019) 全文 PDF 解析的綜合。取代早先 `lowmach_music_plan.md`(2026-05-01)那份基於 agent summary 的推論 —— 那份的某些假設(「MESA velocity-floor 風格就夠」)**與我們的棧不匹配**,實驗已驗證失敗。

## 現狀 (2026-05-02)

- `radiation-eos-fas` 主線:Ideal + radiation EOS 接通 FAS/SIMPLE/projection,64×32 bit-identical,256×128 dt→2e-5 退化
- `radiation-eos-fas + LHLLC commit`:Minoshima LHLLC 港入 `fas_hllc.cuh`,64×32 OK,256² 步驟 0 `||F||=0.126`(比 standard HLLC 的 2e7 降 8 個量級)**但** Newton step 4+ 炸到 10^50 —— **原因不是 LHLLC 本身錯,是我們 implicit 棧在 log mesh 高分辨率上有更底層的 bug**
- `music-phase2-scaling` branch:加了 `residual_norm_scaled`、Viallet ε(錯了,已 revert)、大量 probe kernel。Probe 確認 **Newton δU 在 cell (5,0) — log mesh 內層 + 極軸 j=0 — 產生 NaN,Gram-Schmidt 沿著 NaN 傳播**。根因不是 flux 耗散,也不是 scaling

## 三個代碼家族 — 實打實的差異

Deep research 拿 PDF 全文核對,三個**實際工作**的低馬赫恆星代碼在三個維度上全部**都不是**我們的棧:

| 代碼 | Flux | 時間方法 | Preconditioner | 極軸 / r=0 處理 |
|---|---|---|---|---|
| **MUSIC** (Viallet 2011/2016) | **van Leer scalar upwind**(非 Godunov) | Crank-Nicolson(+ BE fallback) | 2011: 解析 Jacobian + MUMPS direct LU。<br>2016: **物理 preconditioner(Park-style SI),AMG 解 Helmholtz-like δp 系統** | **切掉 r < 0.2 R★**,不穿過奇點 |
| **SLH** (Miczek 2013/2015, Edelmann 2021) | **AUSM+-up 或 Miczek Roe-preconditioned** | ESDIRK23/34 (L-stable SDIRK) | **期刊論文未公開,只在 Miczek 2013 TUM 博論** | Curvilinear cell-centered,Cartesian topology,**不經過 axis** |
| **MAESTROeX** (Fan 2019) | Explicit PPM + CTU | **低馬赫模型方程 + 投影法**(非 Newton-Krylov) | AMReX geometric MG 解 variable-coefficient Poisson | **Cartesian AMR**,1D radial base state 映射上去 |

**關鍵文獻 pinpoints**(每條都 PDF 全文驗證過):

- Viallet 2011 A&A 531 A86 §3.1, §3.1.4, §5.2:staggered spherical FV,**uniform radial mesh**,切掉內邊界於 0.2 R★,van Leer upwind(§3.1.2 eq 17)
- Viallet 2016 A&A 586 A153 §3–§4:Crank-Nicolson(§2 eqs 7-10),JFNK + Park PBP(§3)。關鍵公式:
  - **eq 47–49**:Helmholtz-like δp 方程(PBP 的核心)
  - **eq 72**:左右不對稱 scaling,`L_u = ρ·max(|u|, α₁·c_s)`, `R_u = max(|u|, α₂·c_s)`
  - **§5.1.1 + footnote**:**`α₁ = 10⁻⁵, α₂ = 1`** —— 這個不對稱性是低馬赫運作的關鍵,不是 MESA 的 `frac=0.1`(我之前抄錯了)
  - **eq 77**:δ = λ·(λ + ‖U‖/‖v‖),**λ=10⁻⁷** 為低馬赫 (§5.1.1)。**但這是給 scaled U**,我們存 raw U 所以直接抄 λ=10⁻⁷ 會讓 ε 過大
  - **§3.7**:SI preconditioner 在 CFL_adv > 0.2 開始不穩,CFL_adv > 1 無用。不是萬靈丹
  - **§5.1.2 + Tab 1**:Ms=10⁻⁶, CFL_adv=0.5,PBP 崩潰,Newton iter 2→4.5,GMRES iter 16→291,wall-clock ×50 慢。修法:**clamp CFL_adv ≤ 0.05**
- Miczek 2015 A&A 576 A50 §7 eq 32:**HLLC/Roe 的耗散在 M→0 極限下 O(1/M),是低馬赫的根本問題**。eq 35-37:Roe preconditioned flux `F = (1/2)[F^L + F^R − P⁻¹|{PA}_roe|(U^R−U^L)]`,P_V 5×5 矩陣構造
- Goffrey 2017 A&A 600 A7 Tab 1:MUSIC 所有 benchmark **都在 Cartesian grid 上跑**,從不碰球極座標。這證實球極 + log mesh + implicit 是**文獻未解問題**

## 我們的 bug 的文獻定位

Deep research 的結論 —— **我們踩到的雷在恆星流體文獻裡沒有直接討論,因為沒有人用這個棧**。但在**一般 CFD Newton-Krylov 文獻**和**輻射輸運 / 中子學**文獻裡是典型的失效模式:

- **Knoll & Keyes 2004** JCP 193, 357,§4:明確警告 "algebraic point-block Jacobi scales poorly with grid anisotropy and mesh stretching,line-Jacobi or ADI-type preconditioners are required whenever one direction dominates the transport timescale"
- **Chacón et al.** JCP 2008/2014/2019:"cell-local approximations are at best nonuniformly bad preconditioners for hyperbolic systems with order-of-magnitude disparate wave speeds"
- **Mavriplis** 1998+:aspect ratio > 10³ 時必須 line-implicit,不能 point-Jacobi

我們觀察到的失敗鏈(probe 驗證):
```
R(U₀) @ (0, 27) = 6.79e-3              — spatial residual OK
δU                                     — 應該 ~1e-10
[gmres j=1] ||J·Z|| = 1.85e+5          — JFNK matvec 爆
[gmres] H non-finite at j=17           — Gram-Schmidt NaN
post-perturb(5, 0) ρ = nan             — δU 在 axis 小 cell 含 NaN
floor clip to 1e-20                    — F 下次 evaluation 讀 1e-20 鄰居
F step0 nw1 @ (5, 0) = 1.92e+7          — 幾何爆炸
```

**位置 (5, 0)** = log mesh 第 6 徑向層(非常小 cell)+ 極軸 j=0。Jacobian spectral radius `sr = 2·((|v|+c)/dr + (|v|+c)/(r·dθ))` 在這些 cell ~10⁷。block-Jacobi 的 4×4 diagonal inverse 實質為零 → 前 precondition 方向幾乎正交於真的 Krylov 方向 → Gram-Schmidt 失穩。

## 路線圖:三條獨立的求解器實驗

**不覆蓋現有求解器**。每條路線開新求解器 struct + 新文件,保留 `fas_solver` 作為對比 baseline。這是這個 repo 的一貫原則(見 CLAUDE.md「不可覆蓋的求解器資產」)。

---

### 選項 A — `fas2_solver`(修現有 implicit)

**賭注**:Newton/JFNK/block-Jacobi 的失敗是**工程 bug**而非**架構錯誤**,四個針對性 fix 就能救 256² log mesh。

**文獻支持**:Knoll-Keyes 2004 §4(line-implicit),Viallet 2016 eq 72(scaling),Trilinos NOX internals(CGS2 + unit-normalize v)。**每項都有文獻 pinpoint**,不是我瞎想。

**四個 fix 順序**:

1. **Gram-Schmidt → CGS2**(Classical GS twice,或 MGS + 再正交化)
   - 位置:`src/gpu/fas_solver.cu` 的 `gmres_solve()` Gram-Schmidt 循環
   - 當前單次 MGS 在 ill-conditioned JM⁻¹ 下第一輪留 `O(‖J‖·ε_mach)` 殘差,第二輪拉回 `ε_mach`
   - **10 行改動**,最安全的單點修復,Knoll-Keyes 2004 明示

2. **JFNK matvec v 的 unit-L2 normalize**
   - 位置:`src/gpu/fas_solver.cu::jfnk_matvec`
   - 當前:`ε = sqrt(ulp)·(1+‖U‖)/‖v‖`,當 `‖v‖→0`(後期 Arnoldi 向量)→ `ε→∞`,狀態推到非物理
   - 正確做法:**先 `v_hat = v/‖v‖`,對 `v_hat` 做 FD matvec,結果乘 `‖v‖`**。Trilinos NOX 這麼做
   - 同時 clamp `‖v‖²` floor at `10⁻³⁰·‖u‖²` 避免除零

3. **Viallet 2016 eq 72 asymmetric L/R scaling**
   - 位置:`residual_norm_scaled`(已有)+ **new**:左 scaling 進 GMRES 的 b,右 scaling 進 δU 後 unscale
   - 關鍵參數:**`α₁ = 10⁻⁵`(L)**,**`α₂ = 1`(R)**,不是 MESA 的 0.1
   - 這是讓 Ms=10⁻⁶ 收斂的核心,Viallet 2016 §5.1.1 明言

4. **Block-Jacobi → Line-implicit-in-r**
   - 位置:`src/gpu/fas_smoothers.cu` — 新 kernel `k_fas_line_jac_r`
   - 每個 (j, θ column),組裝沿 i=0..Nr-1 的 block-tri-diagonal Jacobian,**直接 banded LU 求解**
   - CUDA 每 block 負責一個 column,shared memory 裝 Jacobian blocks
   - 成本:O(Nr · block³) per column,trivially parallel over Nt — GPU 友善
   - 文獻:Knoll-Keyes 2004 §4.2,Mavriplis 1998(RANS line-implicit)

**額外**(低風險,必做):
- `r_inner` 從 `r_face[0]` 拉到 `0.05 R★`(MUSIC 做 0.2 R★),直接繞過最病的 cells。用 `--r-inner 0.05` 即可(已有 flag)
- Crank-Nicolson 做對照(Viallet 2016 是 CN,不是 BE):BE 過於 dissipative,可能把真物理 damp 沒了

**驗證 criteria**:
- 64×32 `--perturb 1e-3`:保持 bit-identical(或退化可忽略)
- 256×128 `--perturb 1e-3`:dt 穩定在 ~1e-3,E 守恆 10⁻¹⁰
- 256×256 `--perturb 1e-3`:至少跑到 t=0.1,dt > 1e-5
- Ideal + radiation EOS 在上述網格都過關

**工作量**:1-2 週,以 fix 4 為主風險(其他三個 20 行以內)

**檔案計劃**:
- `src/gpu/fas2_solver.{cuh,cu}` —— clone `fas_solver.*`,改 Gram-Schmidt / matvec / 新 line-jacobi
- `src/gpu/fas2_kernels.cu` —— line_jac_r kernel,residual scaling kernel
- `--solver fas2` dispatch
- CMakeLists.txt 新增

**失敗判據**:如果 4 個 fix 全上,256² 仍 NaN / dt 塌,證據指向 **flux 本身** 是元兇(O(1/M) 耗散),進路線 C

---

### 選項 C — `slh_solver`(低馬赫 flux,換一條路由)

**賭注**:HLLC 在 implicit 低馬赫下**結構性錯誤**(Miczek 2015 §7 eq 32:O(1/M) 耗散不可消),換 AUSM+-up 或 Miczek Roe-preconditioned flux 就不需要選項 A 的 scaling 技巧

**文獻支持**:Miczek 2015 A&A 576 A50 全文,PDF eq 37 完整 P_V 矩陣。Edelmann 2021 A&A 652 A53:SLH 生產級恆星代碼用這個方法,ESDIRK23 時間積分

**實作**:

1. **Flux**:Miczek-Roe preconditioned 或 AUSM+-up(Liou 2006)
   - P_V 5×5 矩陣(Miczek 2015 eq 37):`δ = 1/μ - 1`,`μ = min[1, max(M_local, M_cut)]`
   - 耗散項 `|PA|_roe` 用 P⁻¹ scale,保證 M→0 時所有 5 個守恆量耗散 O(1)
   - 或 AUSM+-up:無矩陣,壓力/速度 splitting,同樣 M→0 耗散 O(1)

2. **時間積分**:ESDIRK23(Hosea-Shampine 1996)或保留 BE(先易後難)
   - L-stable,3 階段,2 階
   - Butcher tableau 寫死,每階段一個 Newton-Krylov solve

3. **Preconditioner**:
   - 先試 point-block-Jacobi(與現有 FAS 類似)
   - 若發現同樣 NaN,上選項 A 的 line-implicit(兩者正交)

**驗證 criteria**:
- Sod shock tube(驗證 flux 正確性)
- Lane-Emden perturbed 256× 系列(與路線 A 對照)
- Low-Mach 極限 M=10⁻³,10⁻⁴,10⁻⁵ 降冪測試 —— Miczek 2015 Fig 9-12 有 reference 值

**工作量**:1 週(單 flux 替換)+ 若加 ESDIRK 再 1 週

**檔案計劃**:
- `src/gpu/slh_solver.{cuh,cu}` —— 新求解器
- `src/gpu/slh_flux.cuh` —— Miczek-Roe 或 AUSM+-up 實作,P_V 矩陣
- `src/gpu/slh_esdirk.cu` —— 時間積分(optional 初期)
- `--solver slh` dispatch

**失敗判據**:Sod / Lane-Emden 都過但 256² log mesh 仍 NaN → 即使 flux 對,implicit 本身就是 ill-posed(證據指向 B 是必需)

---

### 選項 B — `pbp_solver`(Viallet PBP 物理 preconditioner)

**賭注**:選項 A 和 C 都不夠(或效率太差),必須上 Viallet 2016 的物理 preconditioner —— **AMG 解 Helmholtz-like δp 方程**作為 M⁻¹

**文獻支持**:Viallet 2016 A&A 586 A153 §3-§4,PDF 的 eq 47-49 + 79-80 + §3.5 AMG 參數

**實作**:

1. **SI scheme 推導**:
   - 把 stiff 項(sound wave + thermal diffusion)隱式,其他顯式
   - 壓力方程 BE,Picard-linearize
   - δu/Δt + (1/ρⁿ)∂_x δp = −F_u(eq 49)
   - 消去 δu,得 **Helmholtz δp 方程** (eq 47):
     ```
     δp/Δt − a²·Δt·∂²ₓ δp − (Γ₃−1)·∂ₓ(χⁿ ∂ₓ δT) = −F̃_p
     ```

2. **SI 矩陣作 preconditioner M**:
   - Right-preconditioned GMRES:`J·M⁻¹ δX' = −F_U`,然後 `M δX = δX'`
   - Matvec on Krylov vector v:
     1. Solve `M·w = v`(只需解 Helmholtz + triangular back-subs for δu, δe)
     2. Transform `δV → δX` via `∂X/∂V`(Viallet 2016 eq 20-23,trivial)
     3. JFNK matvec on w 得 `J·M⁻¹·v`

3. **AMG 解 Helmholtz**:
   - GPU 上用 **AmgX** (NVIDIA) 或 **Ginkgo / hypre-GPU**
   - Smoothed aggregation,damping 1.2
   - Inner tolerance η' = 10⁻⁴

**驗證 criteria**:
- Viallet 2016 Tab 1 的 Ms=10⁻², 10⁻⁴, 10⁻⁶ 降冪測試,Newton/GMRES 迭代數與其對比
- 256×256 CFL_adv=0.1 能跑到 t=τ_conv

**工作量**:**1 個月**(Viallet 自己說的 5 年 → 我們單人 1 月)

**檔案計劃**:
- `src/gpu/pbp_solver.{cuh,cu}`
- `src/gpu/pbp_si.cu` —— SI scheme 組裝,Helmholtz operator
- `src/gpu/pbp_amg.cu` —— AmgX wrapper(或自寫 SA)
- `--solver pbp` dispatch

**失敗判據**:CFL_adv=0.05 都無法跑到 Ms=10⁻⁴ → 需要換更強 preconditioner(Chacón Schur-complement)或走選項 C 的路線整個換 flux + 時間

---

## 共同的不動工作

三條路線都適用,先做,獨立於選擇:

1. **`r_inner = 0.05 R★`** 在所有 perturbed Lane-Emden 測試(`--r-inner` 已有 flag,但 init 是否接?需要查)
2. **修 Gram-Schmidt CGS2**(選項 A 的 fix 1),這個是**純粹數值 bug**,任何路線都要
3. **JFNK matvec 加 v 的 unit-normalize + ‖v‖ clamp**(選項 A 的 fix 2),同上
4. **`residual_norm_scaled` 的 α 參數**從 0.1 改成 **`α₁=1e-5, α₂=1` 不對稱**(`fas_residual.cu:838+`),同樣路線無關

這 4 項可以現在就直接做進 `fas_solver`,作為 baseline 改進,不影響對三個新求解器的對比。

---

## 推薦起點

**從選項 A 開始**,理由:
- 4 個 fix 都有文獻 pinpoint,不是空中樓閣
- 1-2 週工作量最小,風險最可控
- 如果失敗,證據明確指向「必須換 flux / 換 preconditioner」,**縮小路線選擇**
- 不影響現有 `fas_solver`(新開 `fas2_solver`),隨時可回退

選項 B 和 C 是選項 A **成功後**決定下一步,或 **失敗後**基於失敗信號選走哪一條。

**現在就做的共同工作**(上節第 1-4 條)已經包含在選項 A 的範圍內 —— 等於兩件事合在一起做。

## 參考文獻

| Ref | 引用 | PDF 位置 |
|---|---|---|
| Viallet 2011 | A&A 531 A86, arXiv:1103.1524 | - |
| Viallet 2016 | A&A 586 A153, arXiv:1512.03662 | - |
| Goffrey 2017 | A&A 600 A7, arXiv:1610.10053 | - |
| Miczek 2015 | A&A 576 A50, arXiv:1409.8289 | - |
| Edelmann 2021 | A&A 652 A53, arXiv:2102.13111 | - |
| Horst 2020 | A&A 641 A18, arXiv:2006.03011 | - |
| Andrassy 2022 | arXiv:2111.01165 | - |
| Fan 2019 MAESTROeX | ApJ, arXiv:1908.03634 | - |
| Knoll-Keyes 2004 | JCP 193, 357, DOI 10.1016/j.jcp.2003.08.010 | - |
| Park 2009 | JCP 228, 7427, DOI 10.1016/j.jcp.2009.05.002 | - |
| Miczek 2013 PhD | TUM | **未取得,需 TUM library** |
| Edelmann 2014 PhD | TUM | **未取得,需 TUM library** |

**文獻空缺**:Miczek 2013 和 Edelmann 2014 兩個 TUM 博論是 SLH Newton-Krylov / preconditioner 的**唯一**權威來源,期刊論文全部 defer 到它們。有需要時通過 TUM library / mediatum@ub.tum.de 取得。
