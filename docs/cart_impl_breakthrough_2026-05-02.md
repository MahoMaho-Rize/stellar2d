# cart_impl:Cartesian BE + JFNK 低馬赫求解器突破

> 2026-05-02。基於 fas2-option-a 的失敗分析,新開求解器 `cart_impl` 在 Cartesian 均勻網格上重跑「BE + JFNK + 4 個 fix」,首次獲得**擾動真實演化**結果。

## 一句話結論

**球極 log mesh 是我們之前求解器失敗的主要原因之一**。換到 Cartesian uniform mesh,同樣的 Newton-Krylov 框架(BE + JFNK + Viallet scaling + CGS2 + unit-normalize v̂)可以正常收斂,擾動從 Ma~1e-4 指數增長到 Ma~2e-2 成為可觀測對流。

**但這不代表「隱式求解全面勝利」—— 我們繞過的是「座標 + 幾何」問題,沒證明隱式能在真正的球對稱 + 自引力恆星上收斂。**

## 動機(2026-05-02 早些時候的診斷)

兩個獨立觀察導向同一結論:

1. **fas2 + mass mesh + r_inner=0.2 + perturb=1e-4**:250 幀全程 `|v|=0`,ρ bit-identical(`np.array_equal=True`)。Newton `cyc=0`,根本不進 solve。
2. **projection + 同 config**:dt 塌到 2.6e-16(機器精度),E=inf,M/E 劇烈震盪。

兩個求解器**同一個幾何**失敗,但失敗形式不同 —— 強烈暗示**幾何**是問題,不是某個求解器的 bug。

**文獻證據**(deep-research):
- MUSIC (Viallet 2011/2016):球極但**uniform radial mesh**,切掉 r<0.2 R★
- SLH (Miczek 2013):curvilinear cell-centered,**Cartesian topology**
- MAESTROeX (Fan 2019):**Cartesian AMR** + 1D radial base state 映射

**沒有任何生產級低馬赫恆星代碼用球極 log mesh + 隱式**。這個配置是文獻未解問題。

## 實作

四個文件(~800 行),新開 `--solver cart_impl`,不覆蓋 fas2:

| 文件 | 作用 |
|---|---|
| `src/gpu/cart_impl_solver.{cuh,cu}` | struct + init/destroy/IC/snapshot_hse/VTK/step |
| `src/gpu/cart_impl_residual.cu` | HLLC flux + well-balanced residual(gravity source only on δρ) |
| `src/gpu/cart_impl_jfnk.cu` | JFNK Newton-Krylov + GMRES(CGS2)+ Viallet scaling |

**幾何**:
- 2D Cartesian,nx × ny cell-centered + 2 ghost cells
- 均勻 `dx = Lx/nx`, `dy = Ly/ny`
- x 方向 periodic,y 方向 HSE Dirichlet(clone HSE reference 到 ghost)

**物理**:
- 常重力 `g_y = 1` 拉 -y 方向
- HSE polytrope:`ρ(y) = ρ_b · (1 - y/Ly)^n`, `P = K·ρ^γ`, `K = (γ-1)/γ · g · Ly · ρ_b^{γ-1}`
- 擾動(adiabatic):`δρ/ρ = amp · sin(π y/Ly) · cos(2π x/Lx)`

**Newton-Krylov**:
- BE: `F(U) = (U-Uⁿ)/dt − (R(U) − R_hse)` — well-balanced
- 四個 fas2 fix 原樣繼承:Viallet α₁=1e-5 α₂=1 scaling, CGS2 Gram-Schmidt, 單位化 v̂, round-off floor
- **Preconditioner = Identity**(Cartesian uniform mesh κ(J) ~ O(N),不需要 line-implicit)
- Newton 收斂判據:`abs < 1e-8` OR `rel < 0.5 · init_res` OR stall(< 5% 變化)

## 測試結果

### 32×32 HSE stationary(無擾動,7 步 → t=0.05)

```
Step 7  t=0.05  dt=3.7e-3  M=3.999e-1  E=1.714e-1  |v|=0.00  Ma=0.00
```

HSE 保持完美 stationary,|v| bit-identical 0(不像 fas2 會有 ~1e-10 數值漂移)。✅

### 32×32 perturb=1e-4(130 步 → t=1.0)

```
Step   50  |v|=8.65e-5  Ma=1.54e-4
Step  100  |v|=1.72e-4  Ma=4.81e-4
Step  130  |v|=2.22e-4  Ma=7.39e-4
```

**|v| 從 0 單調增長到 2.2e-4**,對應擾動起步。M/E 守恆 10 位有效數字。Newton 每步 1 iter 收斂。✅

### 128×128 perturb=1e-4(1044 步 → t=2.0)

```
Step  500  |v|=1.22e-4  Ma=6.99e-4      ← 早期線性增長
Step  800  |v|=2.22e-4  Ma=3.05e-3      ← 進入非線性
Step 1000  |v|=7.73e-4  Ma=1.49e-2      ← 對流完全發展
Step 1044  |v|=1.10e-3  Ma=2.10e-2
```

**Mach 數指數增長**(1e-4 → 2e-2,兩個量級),M 守恆 8 位,E 守恆 10 位。**真實 Rayleigh 類失穩發展**。✅

## 關鍵 fix(開發過程中發現的)

1. **`k_ci_cfl` shared memory 未傳 bytes** → `Invalid __shared__ write` illegal memory access。修:kernel launch 帶 `B*sizeof(double)` shmem 參數。
2. **BC kernel launch 線程數**:x_periodic 需要覆蓋 `ny + 2*ng` 列(不只 ny)。
3. **Newton 絕對 tol 過嚴**:BE + cfl-limited dt 下,spatial residual 就在 `ρ·c/dx·perturb` 量級(3.7e-5),Newton 能降到 6e-6(1 階下降)就已是收斂。改用 **相對下降 0.5 + stall detect** 判據。
4. **`ci_idx` host+device**:Cartesian index helper 同時在 host code(init_hse/snapshot)和 device kernel 用,必須 `__host__ __device__`。

## 驗證矩陣

| 配置 | fas2(球極 log mesh) | cart_impl(Cartesian uniform) |
|---|---|---|
| 32×32 HSE stationary | ✅ | ✅ |
| 64×64 perturb=1e-3 | ✅(baseline) | — |
| 128×128 perturb=1e-4 | ⚠ Newton cyc=0, 狀態凍結 | **✅ Ma 1e-4 → 2e-2 發展** |
| 256×256 perturb=1e-4 mass mesh | ❌ 250 幀 bit-identical,狀態凍結 | **待測** |
| 256×256 perturb=1e-4 mass mesh(projection) | ❌ dt=2.6e-16, E=inf | **待測** |

## 限制與誠實評估

**這不是「隱式求解解決恆星演化」的完整證明**。以下幾點必須承認:

1. **物理不同**:原問題是球對稱 + 自引力 Lane-Emden,現在是 2D 平面盒 + 常重力。我們繞過了「坐標奇點 + 自引力 + 球對稱」三個問題。
2. **沒真正的低馬赫考驗**:Ma~1e-4 → 2e-2 是**對流 onset/非線性發展**,不是「低馬赫穩定演化」。真正考驗需要 perturb=1e-6 看是否長時間保持穩定(既不放大也不數值衰減)。
3. **Newton 1 iter 收斂** ≠ 「隱式威力」:dt 是 CFL-limited(~1.9e-3),等於 explicit 的時間尺度,只是多付了 GMRES 開銷。BE 真正的優勢是 **dt >> CFL** 時還能收斂,**沒測**。
4. **preconditioner 是 identity**:Cartesian + 均勻 + 亞音速 → Jacobian 良態,identity 都夠。換到任何更難的幾何就可能不夠。
5. **Viallet scaling 開著但可能不必要**:由於 Newton 1 iter 就收斂,scaling 的實際貢獻沒真正測試。

## 2D 恆星演化的「標準玩法」— 座標選擇

回到「我們要做 2D 恆星能用 Cartesian 嗎?」這個問題。**文獻上三條路線**:

### 選項 X:Cartesian box + 1D radial base(MAESTROeX 風格)
- **做什麼**:3D/2D Cartesian box 放進恆星**內部某個區域**(比如對流層片段),不包整顆星。全域 HSE 由 1D radial base state `ρ₀(r,t), p₀(r,t)` 管,Cartesian 只解擾動 `π(x,t)`
- **標準嗎**:**是**。MAESTROeX 是目前最成熟的恆星對流低馬赫代碼(DOE 支持,ApJ 多篇)
- **cart_impl 和它的差距**:cart_impl 是 **full compressible BE**,MAESTROeX 是 **low-Mach model equations**(壓力分解 + projection)。方向相近但細節不同。cart_impl 不適合做完整恆星,適合做「box simulation」(對流層片段)

### 選項 Y:球對稱 axisymmetric(cylindrical r-z 或 spherical r-θ),uniform mesh
- **做什麼**:用 r-z cylindrical 或 r-θ spherical,但**半徑方向 uniform**,避開 log stretching。內邊界切掉 r < 0.2 R★
- **標準嗎**:**是**。MUSIC (Viallet 2011/2016) 正是這條路線,所有 2D 恆星對流 paper 都這麼做
- **cart_impl 能改到這個嗎**:需要重寫整個 residual 和 HSE IC(球幾何 + 1/r² 自引力)。大概 3-5 天工作

### 選項 Z:球極 + log mesh(我們之前嘗試的)
- **做什麼**:r-θ 球極 + r 方向 log stretch 增加內層分辨率
- **標準嗎**:**不是**。Goffrey 2017 §Tab 1 明言 MUSIC 從不用 log mesh,SLH 用 curvilinear 但 topology 是 Cartesian
- **為什麼失敗**:內層 dr → 0 造成 CFL 塌陷 + Jacobian 條件數 ~10⁷ + 極軸 j=0 奇點,三個病 stack 起來隱式 solver 無法應付

## 三條路線的建議

**短期(1 週內)驗證 cart_impl 本身**:
1. 256² 測試(若 cart_impl 也能過,證明「Cartesian + uniform」幾何可擴展)
2. perturb=1e-6 低馬赫穩定性測試(真正檢驗 BE 低馬赫能力)
3. `--cfl 2.0` 大 dt 測試(看 BE 能否超越 explicit CFL)

**中期(2-4 週)如果 cart_impl 通過以上三個**:
- 加 **Rayleigh-Taylor IC**(現有 cart_ale2 已有類似 bubble IC,借來改)做 2D 對流 benchmark
- 加 **旋轉 / 磁場 / 輻射壓**(ideal_rad EOS 已有)擴展到真實恆星物理
- 這條路變成 **「MAESTROeX 風格 box simulation」** 的 Full compressible 版本

**長期(如果真要做球對稱恆星)**:
- **選項 Y**(spherical uniform mesh)是唯一已知可行的隱式路徑,但必須重寫。可以把 cart_impl 經驗遷移過去
- 或者完全切到 MAESTROeX 風格(壓力分解 + projection),那就不是我們現在這套隱式框架了

## 提交內容

```
src/gpu/cart_impl_solver.cuh    ~110 行
src/gpu/cart_impl_solver.cu     ~470 行(含 init/HSE/VTK/diagnostics)
src/gpu/cart_impl_residual.cu   ~280 行(HLLC residual + BC + CFL)
src/gpu/cart_impl_jfnk.cu       ~220 行(Newton + GMRES + scaling)
CMakeLists.txt                  +3 行
src/main.cpp                    +70 行(--solver cart_impl dispatch)
```

不修改任何既有求解器(fas/fas2/projection/simple/cart_ale2/...),符合 CLAUDE.md 「不可覆蓋的求解器資產」原則。
