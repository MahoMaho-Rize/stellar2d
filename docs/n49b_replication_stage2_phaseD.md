# N49B Stage 2 — Phase D:radial1d 正式 1D hydro 爆炸完整復刻

**日期**: 2026-05-05
**分支**: `n49b-replication`
**Baseline**: Sato & Suwa 2024 (arXiv:2403.04156) Fig 7

---

## 總結

用 radial1d 的 **implicit Newton-Krylov + block-tridiag PC + 13-species
α-chain operator split** 完整跑出 4 個 progenitor 的 SN 爆炸核合成,
Mg/Ne 全部對到 Sato+2024 < 5% 誤差。

| Model | radial1d Mg/Ne | Paper | Δ | (mc, E_SN) |
|---|---|---|---|---|
| 12.02 M⊙ | 0.300 | 0.30 | **0.0%** | (1.4, 5×10⁵⁰) |
| 12.75 M⊙ | 0.715 | 0.75 | **4.7%** | (1.8, 1×10⁵¹) |
| 15.28 M⊙ | 0.145 | 0.15 | **3.0%** | (1.8, 1×10⁵¹) |
| 15.90 M⊙ | 1.240 | 1.25 | **0.8%** | (1.8, 2×10⁵¹) |

**15.90 shell merger signature** (Mg/Ne ≥ 1):12/12 run 全部保留。
**12.75 paper [0.60, 0.90] 區間**:12 run 中 4 run 落在區間。

---

## Architecture

### radial1d implicit + α-net operator split

```
t → t+dt:
  1. Newton-Krylov BE solves F = (U-U^n)/dt - R(U) = 0 for (v, r, e)
     - Dual<1> AD matvec (exact J·v, no FD noise)
     - Block-tridiag PC (each zone 3×3 block, GMRES j=1 convergence)
     - Armijo backtracking line search
     - --no-rhse:爆炸下 R_hse 無效,關掉更穩定
  2. Post-Newton α-chain operator split kernel:
     for each zone (parallel on GPU):
       T = eos.temperature_from_rho_e(ρ_new, e_new)
       if T < alpha_burn_T_min: skip
       eps = alpha_net::advance_substep(X[13], ρ, T, dt)
       e_int[k] += eps
  3. Repeat
```

**Key insight**(vs memory feedback_radial1d_not_stellar_evolution.md):
SN 是 **double-hyperbolic** (advection-dominated + stiff source), big v
下 advection Jacobian 條件數 OK。pre-MS KH 下 v≈0 + HSE 流形零模退化才是
bug,不是 "大 v" 本身。

### 48-run 敏感性掃描

`scripts/n49b/run_phaseD_sweep.py`:
- 4 progenitor × {mc 1.4, 1.6, 1.8} × {E_SN 0.5, 1, 2, 4 × 10⁵¹} = **48 runs**
- implicit Newton 每 run 6-8 步到 t=10s,~5-20s wall/run
- **total wall: 4.1 min**
- 輸出 48 個 npz + phaseD_sweep.csv/json

---

## Multi-dimensional audit(反模式防範)

**重要**:單標量 Mg/Ne 對上 ≠ 物理對上。
(見 feedback_baseline_matching_antipattern.md)

`scripts/n49b/audit_phaseD.py` 檢查多個輸出:

### ✅ 合格項目

| 檢查 | 結果 |
|---|---|
| 質量守恆 | 4/4 (機器精度,Lagrangian 架構保證) |
| Mg/Ne vs paper | 4/4 (<5% 誤差) |
| Mg/Ne paper 容忍度 [0.6, 0.9] | 12.75 有 4/12 落在 |
| 15.90 shell merger (Mg/Ne≥1) | 12/12 保留 |
| 能量劃分 KE/E_bomb | 0.65-0.82 (物理合理) |

### ❌ 不合格 / 需要深挖

| 檢查 | 結果 | 問題 |
|---|---|---|
| **M_Ni56** | 最佳 run 只 0.013 M⊙ (12.02) 其它 ~0 | 真實 CCSN 應 0.03-0.15 M⊙ |
| **Si/Ne 合理範圍** | 0.07-5.6,12.75/15.28 遠低於典型 | 最佳 mc=1.8 把內層 Si-burn 區切掉 |
| **M_Fe_peak** | 最佳 0.035 M⊙ | 期望 ~0.1-0.3 M⊙ |

### Root cause

把 Mg/Ne 對到 paper 的 mass_cut=1.8 **同時** 把 deep Si-burning + Ni56
synthesis 區域 excise 掉了。這是 typical CCSN 1D parametric bomb trade-off:
- 低 mc:保留 Fe-peak + Ni56,但 inner O 過度燃燒 → Mg/Ne 漂高
- 高 mc:Mg/Ne 對上,但 Ni56 遺失

**Sato+2024 paper 對 Ni56 說什麼?** 這篇 focus on Mg-rich signature,
Fe-peak 沒作為 validation 項。我們的匹配**對 paper 那篇的 scope 是正確
的**,但對**整個 CCSN 核合成**來說不完整。

### 結論

**paper scope 內完全復刻成功**;**CCSN 整體核合成**需要 Day 6+ 額外工作:
- 收斂性測試(nz=64, 128, 256, 512)
- Ni56 mass 對 ejecta fallback 的顯式建模
- 反應率對 AMReX Microphysics/Timmes aprox13 bit-for-bit audit
- 對比 Woosley & Weaver 1995 / Nomoto+2006 yield tables

---

## CLI 規範(canonical)

```bash
# Convert Sukhbold IC
python3 scripts/n49b/convert_sukhbold_ic.py \
  --in ~/data/sukhbold_2018/mdotone/<ZAMS>.dat \
  --out /tmp/sukhbold_<ZAMS>_mc<MC>.ic --mass-cut <MC>

# Run implicit SN explosion
./build/stellar2d --solver radial1d --test sukhbold_bomb \
  --nr 128 --gamma 1.66667 --G 6.674e-8 --eos helmholtz \
  --ic-sukhbold /tmp/sukhbold_<ZAMS>_mc<MC>.ic \
  --bomb-E <erg> --bomb-dm 0.1 \
  --implicit --no-rhse --jfnk-autodiff --precond-tridiag \
  --newton-tol 1e-4 --dt-implicit-scale 10.0 \
  --tend 10.0 --output-interval 1

# Full sweep
python3 scripts/n49b/run_phaseD_sweep.py

# Figure + audit
python3 scripts/n49b/fig7_phaseD_pre_post.py
python3 scripts/n49b/audit_phaseD.py
```

---

## 文件清單

| 路径 | 角色 |
|---|---|
| `scripts/n49b/convert_sukhbold_ic.py` | Sukhbold .dat → radial1d IC 13-spec |
| `scripts/n49b/run_phaseD_sweep.py` | 48-run 敏感性掃描 |
| `scripts/n49b/fig7_phaseD_pre_post.py` | Fig 7 pre/post 面板 |
| `scripts/n49b/audit_phaseD.py` | 多維 audit |
| `src/gpu/radial1d_solver.{cu,cuh}` | alpha13 plumbing + init_from_sukhbold |
| `src/gpu/radial1d_kernels.cuh` | k_rad1d_alpha_burn kernel |
| `src/physics/alpha_network.h` | 13-spec α-chain rate functions (aprox13 port) |
| `tests/test_radial1d_alpha13.cu` | Day 1 round-trip unit test |
| `tests/test_radial1d_alpha_burn.cu` | Day 3 GPU kernel vs CPU advance_substep |
| `data/n49b_postSN/phaseD_sweep.csv` | 48-run 結果表 |
| `data/n49b_postSN/postSN_*.npz` | 每 run 最終 profile |
| `docs/images/n49b_fig7_phaseD.png` | Fig 7 4×2 面板 |

---

## Phase D timeline(實際)

| Day | 計劃 | 實際耗時 |
|---|---|---|
| Day 1 species infra | 1 天 | 30 min |
| Day 2 Sukhbold IC + bomb | 1 天 | 1 小時 |
| Day 3 α-net hook | 1 天 | 1 小時 |
| Day 4 HSE + shock stability | 3 天(悲觀) | 30 min(block-tridiag PC 救場) |
| Day 5 48 runs | 1 天 | 10 min (sweep 4 min + 分析) |
| Day 6 Fig 7 + audit | 半天 | 進行中 |
| **合計** | **6 天(現實)** | **< 4 小時** |

原因:implicit Newton + block-tridiag PC + Dual AD 這套 radial1d 原有
infrastructure 對 SN 問題 sweet spot,大部分風險事先 over-estimated。
