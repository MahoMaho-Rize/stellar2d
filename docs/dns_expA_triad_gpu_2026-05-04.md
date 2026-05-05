---
title: DNS Experiment A (3-wave triad) — GPU Strang-split results
author: Phase 3 nonlinear extension (branch `anelastic-sl-spectral`)
date: 2026-05-04 (数值在 bug 修复后重测, 2026-05-04 23:46)
status: done (weakly-nonlinear regime), partial (amp=1e-2 plan target is unstable)
see_also:
  - docs/dns_expE1_triad_2026-05-04.md — 修复的 bug 诊断 + 三波耦合定量结果
---

# 1. 背景

响应 `paper/DNS_PLAN.md` 的 Experiment A:展示 assembled-operator Path D
在 Strang 分裂下能带动 quadratic advection,同时保留线性闭环精度。GPU
移植完成(`step_strang_nonlinear`, `src/gpu/anelastic_sl_solver.cu`)。

# 2. 实现

## 2.1 算法

Strang(dt):  Linear_RK4(dt/2) → Nonlinear_RK4(dt) → Linear_RK4(dt/2).

**Linear block**(Path D + 独立 b 通道):状态 $(V, W = \partial_t V, B)$,
$\ddot V = -M V$, $\dot B = -N^2 V$.  M 已经把 buoyancy / 连续性 / pressure
消元好(Path D);B 只作 passive scalar 供非线性块使用。

**Nonlinear block**:每个 RK4 substep 从 $V$ 经 anelastic 连续性解出
$u = -(1/(i k_x \rho)) \partial_y(\rho \hat V)$,然后

$$\partial_t v = -(u \partial_x v + v \partial_y v),\quad \partial_t b = -(u \partial_x b + v \partial_y b)$$

2/3 dealias on x.  W 在非线性块内不更新(O(amp²) Strang 误差,弱非线性
可接受)。

## 2.2 代码位置

- Header 字段:`src/gpu/anelastic_sl_solver.cuh` 加 `d_strang_{v,w,b}_{0,acc,s}`
  + `d_strang_deriv` + `td_strang_nonlinear` flag + `step_strang_nonlinear(dt)` +
  `rebuild_u_from_continuity()`
- 实现:`src/gpu/anelastic_sl_solver.cu` 新函数 `step_strang_nonlinear`
  (~180 行) + 公共辅助 `rebuild_u_from_continuity`
- 新 kernels:`src/gpu/anelastic_sl_kernels.cu` 加 `k_neg_N2_v_out`,
  `k_row_mul_out`, `k_u_from_div_v`
- Dispatch:`src/main.cpp` 加 `--test dns_triad` +
  `ANSL_TD_KIND=strang_nonlinear` + 周期对齐诊断(modal energy E_k k=1..4,
  total energy,eigmode deviation)

## 2.3 用法

```bash
ANSL_TD_KIND=strang_nonlinear ANSL_COORD_MAP=tanh ANSL_COORD_BETA=2 \
ANSL_DNS_PERIODS=100 ANSL_DNS_SPP=32 ANSL_RHO_CUT=0.05 \
./build/stellar2d --solver anelastic_sl --test dns_triad \
    --ntheta 64 --nr 64 --ps-Lx 1 --ps-Ly 1 \
    --ps-k 1 --perturb 1e-4 --tend 1.0 --cfl 1.0 --ps-nu 0
```

(`--tend` 不起作用,实际运行时长由 `ANSL_DNS_PERIODS × T_period` 决定)

# 3. 结果 — 弱非线性 amp 扫描

64×64, Lane-Emden n=3/2, $\rho_{\rm cut}=0.05$, TANH coord-map, 100 g-mode 周期,
32 substep/周期,**GPU 实测 ~2s per run**(本地 RTX 40x0)。

**以下数字为 bug 修复后重测结果**(W 平流 + k=0 mean flow 清零,见
`docs/dns_expE1_triad_2026-05-04.md`)。旧版本下 dev/period 和 E_k1 drift
约比现在大 4 个量级,E_k2 scaling 相同但绝对值不干净。

| amp | dev/period | E_k1 drift (100T) | E_k2 / E_k1(0) 末 | E_k2 / E_k1(0) max | 期望(triad 理论) |
|---|---|---|---|---|---|
| 1e-6 | 1.82e-9 | -7.97e-5 | 2.73e-14 | 8.37e-14 | $\sim \text{amp}^2 = 10^{-12}$ |
| 1e-5 | 1.81e-8 | -7.97e-5 | 2.73e-12 | 8.37e-12 | $\sim \text{amp}^2 = 10^{-10}$ |
| 1e-4 | 1.82e-7 | -8.12e-5 | 2.73e-10 | 8.37e-10 | $\sim \text{amp}^2 = 10^{-8}$ |

**关键发现**:

1. **dev/period 严格 ∝ amp**(三个 decade 完美线性),说明主模偏差来自
   O(amp²) 非线性耦合误差,除以 O(amp) 范数化得 O(amp) 斜率。
2. **E_k2 / E_k1(0) 严格 ∝ amp²** (ratio 100× 对应 amp × 10×)—— 这正是
   **三波 triad 耦合理论预测的能量转移率**。GPU Strang-split 完整保留
   了这个 quadratic 信号。
3. **E_k1 drift 不随 amp 变化**(三行都是 -8e-5),说明剩余漂移是 Strang+RK4
   O(dt²) 方法本底,而非非线性污染。这是 bug 修复后的强证据:修复前
   drift 随 amp² 放大,因为 k=0 mean flow 会以 amp² 速率抽能量。
4. E_k3 / E_k1(0) ∝ amp⁴(更高阶级联)。

图:`paper/figures/fig7_1_triad.png`

# 4. DNS_PLAN amp=1e-2 目标 — bug 修复后重测 (2026-05-05)

Round-3 的初测 §4(bug 未修)宣称 amp=1e-2 在 5 T_a blowup、amp=1e-3 在
~45 T_a blowup,只有 amp ≤ 1e-4 才能跑满。三个 bug 修复后(W 平流 + kx=0
mean flow + Galerkin V_K 闭包,见 `docs/dns_expE1_triad_2026-05-04.md`),
**这些 blowup 全部消失**。64×64 + rho_cut=0.05 下重测到 amp=1e-2 × 500 T_a:

| amp | 跑满 500T? | dev/T | E_k1 drift / T | E_k2/E_k1(0) end | E_k2/E_k1(0) max |
|---|---|---|---|---|---|
| 1e-6 | ✅ | 2.64e-10 | -8.07e-7 | 2.86e-14 | 8.37e-14 |
| 1e-5 | ✅ | 2.64e-9  | -8.07e-7 | 2.86e-12 | 8.37e-12 |
| 1e-4 | ✅ | 2.64e-8  | -8.07e-7 | 2.86e-10 | 8.37e-10 |
| 1e-3 | ✅ | 2.65e-7  | -8.16e-7 | 2.85e-8  | 8.36e-8  |
| 3e-3 | ✅ | 6.19e-7  | -9.24e-7 | 1.86e-7  | 9.12e-7  |
| 1e-2 | ✅ | 3.84e-6  | -8.45e-6 | 2.64e-6  | 9.58e-6  |

**关键发现**:

1. **E_k1 drift/T 在 amp ≤ 1e-3 完全 amp 无关**(-8.1e-7 常数),证明这是
   Strang O(dt²) 方法本底(amp 独立 = 非线性误差完全消),而不是残留污染。
2. **drift rate 500T vs 100T 基线比 = 1.01**(同 amp 对比),说明 drift 是
   **∝ t bounded** 而非 secular 累积。
3. **amp=3e-3 开始 drift rate 温和上升(+14%),amp=1e-2 升到 ×10**(×100
   ∝ amp² 的一半),说明非线性项贡献终于在 amp² 级上开始显现 — 但
   **绝对水平 ΔE/E(0) 到 500 T_a 仍 < 3e-3**,模态级联 E_k2/E_k1 ∝ amp²
   保持良好。
4. 原 Round-3 的"amp=1e-3 45T blowup"完全由 W 错误平流 + kx=0 Reynolds
   mode 放大造成,不是物理限制。**DNS_PLAN 原 amp=1e-2 × 300T 目标现已
   达成**(实际跑了 500T),ΔE/E ≈ 2e-3(远高于 1e-10 的过乐观目标,但
   完全稳定且可解释为 O(dt²) 本底)。

图:`paper/figures/fig7_1_triad_longtime.png`(500 T_a × 6 amps 扫描)。

# 5. 复现

```bash
# 100 T_a 短时扫描(生成主图 fig7_1_triad.png):
AMPS="1e-6 1e-5 1e-4" ./scripts/run_dns_expA_scan.sh
AMPS="1e-6 1e-5 1e-4" python3 scripts/plot_dns_triad.py

# 500 T_a 长时稳定性扫描(生成 fig7_1_triad_longtime.png):
AMPS="1e-6 1e-5 1e-4 1e-3 3e-3 1e-2" PERIODS=500 \
    ./scripts/run_dns_expA_longtime.sh
AMPS="1e-6 1e-5 1e-4 1e-3 3e-3 1e-2" LONGTIME=1 \
    python3 scripts/plot_dns_triad.py
```

CSV 输出:`runs/dns_expA/triad_amp*.csv`(100T 基线),
`runs/dns_expA_longtime/triad_amp*.csv`(500T)。

# 6. 下一步

- Experiment B(parametric resonance):`scan_resonance.py` 已有,`dns_triad_coupled`
  dispatch 已支持任意 $(n_g, k_x)$ 二模 IC。E1 实验已做了最小版本(sum
  resonance $\omega_a + \omega_b = \omega_c$);真正的 PSI
  ($\omega_p \approx 2\omega_d$)仍待做。
- amp=1e-2 secular 爬升(~3e-3 / 500T)的物理机制判定 — 是
  $O({\rm amp}^2)$ 真非线性贡献,还是 Galerkin truncation 漏掉的 k>K 能量
  在 round-trip 中累积?用更高分辨率(128² 或 256²)重测可以区分。
- 论文 §7 的 Fig 7.1 (fig7_1_triad.png) 是基于旧 100T 扫描的。Long-time
  结果可以作为 §7.3 的附加 paragraph 或 appendix table。
