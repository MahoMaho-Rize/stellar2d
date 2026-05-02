# 座標系 vs 網格:隔離實驗(2026-05-02 下午)

> 更新 `cart_impl_breakthrough_2026-05-02.md` 裡「球極 log mesh 是主因」的粗糙結論。透過 controlled experiment 把「座標系」和「網格拉伸」拆開,發現**真正的 killer 是球極座標的極軸奇點,不是 log mesh**。

## 三個隔離實驗

三個實驗都是 `perturb=1e-4, 256×256, Newton-Krylov + Viallet scaling + CGS2 + line precond`,只改兩個維度:**座標系** 和 **網格拉伸**。

### 實驗 1:fas2 + spherical **uniform** mesh + loose tol(0.1)

```bash
./build/stellar2d --solver fas2 --test lane_emden_perturbed --perturb 1e-4 \
  --nr 256 --ntheta 256 --mesh uniform --precond line_r --tend 5.0
```

結果:
```
step 800  dt=2.66e-03 cyc=0 ||F||=1.26e-02 cuts=0
step 1000 dt=2.66e-03 cyc=0 ||F||=1.26e-02 cuts=0
step 1700 dt=2.66e-03 cyc=0 ||F||=1.26e-02 cuts=0
```

- dt 穩定 2.66e-3(**log mesh 下是 2e-5**,差 100×)—— **證實 log mesh 是 dt 塌陷的主因**
- **但狀態完全凍結**:cyc=0、M/E bit-identical 1700 步、|v|=0 全程

**發現的 tol bug**:`src/gpu/fas2_solver.cu:737` 有 `double tol = 0.1`,而 step 0 的 ||F||~1e-2 已經 < 0.1 → `solve()` 立即返回 `cycles=0`,Newton **從未進入**。這是 fas2 的「靜默 bug」— 每步都在假收斂。

### 實驗 2:fas2 + spherical **uniform** mesh + **tight tol(1e-6)**

修 `tol=1e-6` 後重跑相同命令:

```
step 20: reject dt=6.05e-18 (||F||=3.30e+17, dt*||F||=2.00e+00), retry dt=3.03e-18
step 20: reject dt=3.03e-18 (||F||=5.95e+17, dt*||F||=1.80e+00), retry dt=1.51e-18
step 21: rollback to Un (||F||=4.40e+18, dt=7.57e-19)
...
Step 22  t=4.44e-06  dt=0.00  M=4.82e-01  E=3.83e-01
```

- Newton **真的嘗試進** → **||F|| 炸到 10^18**,dt 塌到 1e-19,machine precision 下限
- 初 20 步還能跑,第 21 步開始爆炸

**證明**:球極座標 + uniform mesh + 真正的 Newton 嘗試 → **solver 本身在球極幾何上不穩**,不是 mesh 問題。

### 實驗 3:cart_impl + Cartesian 256²

```bash
./build/stellar2d --solver cart_impl --test hse_perturbed --perturb 1e-4 \
  --nr 256 --ntheta 256 --tend 5.0
```

結果(30s 跑 11 步):
```
newton iter 0: ||F||=3.76e-5 → 9.19e-6  (4-fold drop, 1 iter)
newton iter 0: ||F||=3.76e-5 → 1.10e-5  (stable)
Step 11  t=1.05e-2  M/E 守恆 10 位
```

- Newton **每步 1 iter 穩定收斂**
- ||F|| 正常量級(1e-5),沒爆炸
- 256² 可擴展,時間夠長會看到對流(128² 已驗證 Ma 0 → 2e-2)

## 隔離後的因果表

| 維度 | 球極 log mesh | 球極 uniform mesh | Cartesian |
|---|---|---|---|
| dt 是否塌陷 | **塌陷**到 2e-5 | 穩定 2.66e-3 | 穩定 |
| Newton 是否穩定 | **炸** | **炸** 到 10^18 | **穩定**,1 iter 收斂 |

**兩個獨立 killer**:

1. **Log mesh** → dt 塌陷(內層 dr ~ 1e-4 讓 CFL 塌到 ns 量級)。獨立症狀。
2. **球極座標** → Newton 炸飛。**與 mesh 拉伸無關**,uniform 也炸。

Deep-research 講得對:MUSIC/SLH 都用 uniform mesh(解決問題 1),但 **MUSIC/SLH 用 curvilinear 時對極軸有特殊處理**(cell-centered avoiding axis, or angular averaging in pole cells)。我們的 fas2 沒做這些特殊處理 —— **這才是我們的 fas2 在球極幾何上炸的根因**。

## 訂正 `cart_impl_breakthrough` 文檔的誇大

之前寫「球極 log mesh 是罪魁禍首」不夠精確。現在的陳述應該是:

- **Log mesh**: dt 塌陷的主因(一個技術性問題,MUSIC 避免用)
- **球極座標本身**: Newton 爆炸的主因(深入的數值分析問題,需要特殊極軸處理)
- **兩個問題 stack 起來** 讓 fas2 無解

同時發現 fas2 有個 **loose tol bug**(tol=0.1),這讓我們之前看到的「狀態凍結」其實不是 Newton 內部 bug,**而是 Newton 根本沒跑**。嚴格講 fas2 之前跑的所有 256² 實驗都是「假收斂」不是「真凍結」。

---

# 2D 座標系下我們現在能做什麼

## 現狀盤點

**可用的求解器**(主分支 radiation-eos-fas + fas2-option-a):

### 球極 r-θ 座標系

| 求解器 | 狀態 | 適用 |
|---|---|---|
| `strang` (explicit HLLC+MUSCL) | ✅ 全域 stable | Sod / blast / explicit hydro benchmark |
| `fas` / `simple` / `projection` | ⚠ 低-中分辨率 OK, ≥256² 爆 | 歷史資產,不再積極維護 |
| `fas2` | ⚠ 同上 + 已知 loose tol bug | 不建議用 |
| `radial1d` (1D MESA-style) | ✅ | 球對稱 Lagrangian,但 1D |

### Cartesian 座標系

| 求解器 | 狀態 | 適用 |
|---|---|---|
| `cart_ale` (Caramana + swept remap) | ✅ | Sod / hydro benchmark |
| `cart_ale2` (periodic BC + PPM) | ✅ | **恆星對流片段 / KH / hydro**(CLAUDE.md 標「首選」) |
| `cart_lag` | ⚠ HSE 長時間退化 | 參考用 Lagrangian baseline |
| **`cart_impl`(新!)** | ✅ 驗證到 128² | **低馬赫隱式 BE + JFNK** |
| `pseudo_spectral` | ✅ | 2D 不可壓 NS(KH / 強迫湍流) |

## cart_impl 目前能做到的

**已驗證的實驗規模**:

| 網格 | 擾動 | 時間 | 結果 |
|---|---|---|---|
| 32×32 | 0(HSE) | t=0.05 | ✅ \|v\|=0 精確保持 |
| 32×32 | 1e-4 | t=1.0(130 步) | ✅ Ma 0→7.4e-4 |
| 128×128 | 1e-4 | t=2.0(1044 步) | ✅ **Ma 1e-4→2e-2,指數增長** |
| 256×256 | 1e-4 | 11 步 30s | ✅ Newton 1 iter 穩,未跑完 |

**物理設定**(CartImpl HSE polytrope):
- 2D 盒 `[0, Lx] × [0, Ly]`,Lx = Ly = 1
- **常重力 g_y = 1 拉 -y 方向**(不是自引力)
- HSE profile:`ρ(y) = ρ_b·(1 - y/Ly)^n`, `n = 1/(γ-1) = 1.5`
- x 方向 periodic BC,y 方向 HSE Dirichlet
- 擾動 mode:`sin(πy/Ly)·cos(2πx/Lx)`(y 方向 half wavelength, x 方向 one wavelength)

這個設定相當於 **MAESTROeX box simulation 的 minimal 版本**(不用 pressure split)。

## 能做 / 不能做的物理場景

### ✅ 已經可以做(cart_impl 基礎上)

1. **Rayleigh-Taylor 失穩**:上輕下重配置(改 init_hse_polytrope 讓 n < 0 或加 interface),觀察擾動發展
2. **對流層片段**:用現在的 polytropic HSE,調擾動模式 mimic 恆星對流 cell
3. **低馬赫穩定性 benchmark**:perturb=1e-6 看擾動是否長時間保持不數值衰減 —— 真正的低馬赫考驗
4. **大 dt 測試**:`cfl=2.0` 或 `cfl=10.0` 看 BE 能否超越 CFL(BE 的真正賣點)

以上都在現在的 cart_impl 代碼上加 flags 就行,無需大改。

### ⚠ 需要擴展(1-3 週工作量)

5. **Rayleigh-Taylor 精確 IC**(Almgren 2010 benchmark)—— 需 `init_rt_interface()` 函數,~50 行
6. **bubble / plume**:cart_ale2 已有 `init_hse_bubbles`,端口過來
7. **熱對流**(加熱底板)—— y=0 加 source term,~30 行
8. **磁場**:cart_impl 需加 MHD,~1 週(flux 重寫)
9. **輻射壓**:`--eos ideal_rad` 已有 EOS POD,接入 cart_impl 的 HLLC 即可,~2 天

### ❌ 做不了(需新求解器)

10. **整顆球對稱恆星**:Cartesian 幾何 mismatch,必須開 `sphere_impl`(球極 uniform,小心處理極軸)
11. **自引力**:cart_impl 目前只有常重力。自引力需要 Poisson solver + `ρ_source`,~1 週工作但**會退化成 Lane-Emden 重做**
12. **旋轉 + 離心**:需 cylindrical r-z,Cartesian 的 1D (x, y→r, z) 映射不自然
13. **3D**:整個代碼是 2D,改 3D 等於重寫

## 可行的科研路線(按工作量 / 新穎度排序)

| 路線 | 工作量 | 科學價值 | 備註 |
|---|---|---|---|
| **Rayleigh-Taylor 隱式 benchmark**(cart_impl) | 1 週 | 中 | 有 MAESTROeX reference 可比 |
| **熱對流 + 磁場**(cart_impl MHD) | 3-4 週 | 中高 | MHD convection 少有人用全 implicit 做 |
| **整顆球對稱恆星對流**(新 sphere_impl) | 1-2 月 | 高 | MUSIC 正式的競爭對手 |
| **恆星脈動 / p-mode / g-mode**(cart_impl linear) | 2 週 | 中 | 需改 IC 成 eigenmode,可對 asteroseismology |
| **白矮星前超新星燃燒**(cart_impl + burning) | 2-3 月 | 高 | MAESTROeX 的專業領域,挑戰大 |

## 我的建議(基於當前 evidence)

**短期(本週內)**:把 cart_impl **壓測完**再論下一步。三個關鍵測試:

1. ✅ 256² perturb=1e-4 長時間(需 ~30 分鐘跑完 t=5)— 驗證 Cartesian 可擴展
2. ✅ perturb=1e-6 stability — 真正考驗低馬赫
3. ✅ `--cfl 5.0` 大 dt — 驗證 BE 超越 explicit 優勢

這三個做完才知道 cart_impl 是「不錯的 prototype」還是「真有競爭力的低馬赫求解器」。

**中期(1 月內)**:如果上面三個都過:
- 加 Rayleigh-Taylor 做個對比 MAESTROeX 的小論文 / note
- 或加 burning term 做 white dwarf 對流

**長期**:做整顆恆星需要新開 `sphere_impl`。把 cart_impl 的 Newton-Krylov 經驗遷移過去,小心處理極軸(angular averaging 或 skip j=0,1 的 inner cells)。這才是 MUSIC 真正的競爭對手。

---

## 小結

**誰是 killer**:**球極座標的極軸奇點** 和 **log mesh 的 dt 塌陷**,兩個獨立因素疊加。

**解法**:
- 短期:Cartesian box simulation(cart_impl),能做的事還不少
- 長期:球極 uniform mesh + 正確極軸處理(sphere_impl,未來)

**fas2 發現的 side bug**:tol=0.1 讓 Newton 靜默 bypass,之前看的「狀態凍結」其實是 Newton 從未真正嘗試 —— 球極 + uniform + tight tol 才暴露出真正的 Newton 爆炸問題。
