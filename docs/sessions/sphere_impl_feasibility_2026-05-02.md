# sphere_impl 可行性驗證:fas2 + uniform + 極軸處理 + 中心切除

> 2026-05-02。在投入 2-3 週寫新 `sphere_impl` 之前,做 60 分鐘 controlled experiment 驗證 repo 裡現有的「球極 uniform + pole_avg + core_excision」 kernels 是否可工作。

## 實驗設置

**核心問題**:deep-research 文獻(MUSIC 2011/2016, SLH 2021)指出生產級球極隱式求解器必須:
1. 球極 **uniform** radial mesh(非 log)
2. 特殊 **極軸處理**(angular averaging 或 wedge excision)
3. **中心切除**(r<0.2 R★)避開 r=0 奇點

`fas_solver.cuh` 裡有 `n_pole_avg`, `n_angular_avg`, `use_core_excision`, `M_core` 這些參數,`fas_smoothers.cu` 有 `k_fas2_pole_avg`,`fas2_residual.cu` 在 `use_core_excision=true` 時 skip `k_fas2_residual_origin`。這些資產**存在 repo 裡**,說明前人嘗試過此路線,但狀態未知。

**實驗目的**:把這些資產 wire 起來,看是否能穩定跑 HSE lane_emden。

## 做了什麼

`src/main.cpp` 臨時改動(兩處):

1. `configure_mass_mesh` lambda:原只對 `mesh_type=mass` 生效,extend 為 **mass OR (uniform+r_inner>0)**。
2. uniform mesh 初始化分支:當 `--r-inner > 0` 時計算 `M_core`(和 mass mesh 同邏輯)。

同時帶上前一輪的 `fas2 tol 0.1 → 1e-6` 修正(避免 Newton 靜默 bypass)。

## 控制實驗(全部超時 30s,4 個配置)

### 實驗 A:64² uniform,**無 r_inner**,perturb=0

```bash
./build/stellar2d --solver fas2 --test lane_emden_perturbed --perturb 0 \
  --nr 64 --ntheta 64 --mesh uniform --precond line_r --tend 1.0
```

結果:
```
step 0 dt=2.13e-04 cyc=0 ||F||=4.52e-15 cuts=0
Step 48  t=1.0  M=4.82e-01  E=3.83e-01
```

✅ **HSE 完美保持** 到 t=1,||F||=4.5e-15(機器精度)。cyc=0 正確(HSE 穩態無需 Newton)。

---

### 實驗 B:64² uniform + **r_inner=0.2**,perturb=0

```bash
./build/stellar2d --solver fas2 --test lane_emden_perturbed --perturb 0 \
  --nr 64 --ntheta 64 --mesh uniform --r-inner 0.2 \
  --tend 1.0
```

結果:
```
Interrupted at step 28, t=7.10e-05. dt = 0.00e+00
```

❌ **加 r_inner 立即炸**。沒擾動、純 HSE 都不穩。step 28 dt 已塌為 0。

---

### 實驗 C:64² uniform + r_inner=0.2 + **line_r precond**,perturb=0

```bash
./build/stellar2d --solver fas2 --test lane_emden_perturbed --perturb 0 \
  --nr 64 --ntheta 64 --mesh uniform --r-inner 0.2 --precond line_r \
  --tend 1.0
```

結果:
```
step 16: reject dt=3.10e-15 (||F||=inf, dt*||F||=inf), retry dt=3.10e-15
step 16: rollback to Un (||F||=inf, dt=3.10e-15)
```

❌ **更快炸**(step 16,||F||=inf)。加上 line precond 更不穩。

---

### 實驗 D:128² / 256² 同樣配置,perturb=1e-4

- 128²:step 200 dt=7.4e-22,step 338 dt=0
- 256²:step 107 ||F||=1.82e+234(!),dt=1.18e-20

❌ **全炸**,和 grid size 無關。

---

## 結論

**fas2 repo 裡現有的 core_excision / pole_avg kernels 不能直接用**,有 bug:

| 組合 | 結果 |
|---|---|
| fas2 球極 uniform,**無** r_inner | ✅ HSE 機器精度穩定 |
| fas2 球極 uniform + **有** r_inner=0.2 + pole_avg | ❌ 純 HSE 都炸到 inf |

根因定位(推測,需進一步 probe):
1. **r_inner boundary treatment bug**:core_excision 模式下,`k_fas2_residual` 依然處理 `r=r_inner` 那層 cell,但其 ghost cell(應為 excised core)沒被正確設置,拉出 non-physical flux
2. 或 `k_fas2_gravity_from_shells` 算 `g(r)` 時,M_core + shell_mass 的合成在 r=r_inner+ 有跳變,residual 炸
3. 或 pole_avg 和 BC 組合破壞 WB(well-balanced)對消,讓 R(U_hse) ≠ 0

## 對 sphere_impl 計劃的影響

**不能簡單從 fas2 clone-and-rename 得到 sphere_impl**。repo 裡已有的 core_excision / pole_avg 資產**不生產 grade**,必須**重新實作核心**。

### 修正計劃

**Phase 0(新增,1 週)**:**核心修復 & 最小原型**
- 新 `sphere_impl` 求解器,從 cart_impl clone 幾何 → 改球極 r-θ uniform
- **從零做 r_inner excision**:r=r_inner 處設 reflective inner wall OR 用 HSE Dirichlet ghost(看哪個穩),不重用 fas2 的實作
- **從零做極軸**:兩種方案並列測,用哪個走哪個:
  - (a) `sin(θ) = 0` 處 cell 直接 skip(wedge excision,j<j_cut 和 j>nt-j_cut 不算 residual)
  - (b) angular average(重寫 fas_pole_avg,但先驗證正確性)
- HSE 驗證:無擾動 128² 能穩定到 t=τ_acoustic = R★/c_s 不炸 ← 這是 Phase 0 的 gate

**Phase 1(2-3 週)**:一旦 HSE 穩定,才加:
- 擾動 → 對流演化(像 cart_impl 的流程)
- GPU 原生 + 長時間演化
- 1D shell 自引力(這個可以從 fas 港,相對簡單)
- diagnostics + VTK

**Phase 2+**(後續):EOS、輻射、BC sponge、conservation correction 等生產硬化

### 時間估計修正

原估 3-5 天 MVP,改為 **1 週 MVP**(加上「自己寫 r_inner 和 極軸處理」的時間),畢竟這兩個是最關鍵也最坑的點。

### 好消息

**實驗 A 明確證明**:球極 uniform mesh + tol=1e-6 + **無特殊幾何處理**,HSE 本身就是穩的(||F||=4.5e-15)。這是 sphere_impl 的 **可用起點**。上面幾個幾何處理(r_inner, pole_avg)是「加進去之後炸」的,說明**幾何處理本身的實作有 bug,不是球極座標內在不可解**。

## 下一步建議

1. **立即**:commit 這份實驗記錄 + 臨時改動(M_core 在 uniform mesh 也算),作為 baseline 證據
2. **1-2 天**:從 cart_impl clone → `sphere_impl`,只做「球極 uniform + **無** r_inner + **無** pole_avg」,驗證擾動能演化
3. **3-5 天**:加回 r_inner(自己寫,不用 fas2 的),驗證小半徑穩定
4. **1 週**:加極軸處理,256² 衝刺

**老闆的 NAOJ cluster 任務**:最遲 2 週交一個經過實驗驗證的 sphere_impl MVP,第 3 週開始做生產硬化和 benchmark。比直接上手寫兩週再發現幾何處理炸再 debug 一週好得多。
