---
title: DNS Experiment A (3-wave triad) — GPU Strang-split results
author: Phase 3 nonlinear extension (branch `anelastic-sl-spectral`)
date: 2026-05-04
status: done (weakly-nonlinear regime), partial (amp=1e-2 plan target is unstable)
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

| amp | dev/period | E_k1 drift (100T) | E_k2 / E_k1(0) 末 | 期望(triad 理论) |
|---|---|---|---|---|
| 1e-6 | 1.53e-5 | -8.0e-5 | 2.73e-14 | $\sim \text{amp}^2 = 10^{-12}$ |
| 1e-5 | 1.53e-4 | -1.2e-4 | 2.73e-12 | $\sim \text{amp}^2 = 10^{-10}$ |
| 1e-4 | 1.54e-3 | -3.9e-3 | 2.78e-10 | $\sim \text{amp}^2 = 10^{-8}$ |

**关键发现**:

1. **dev/period 严格 ∝ amp**(三个 decade 完美线性),说明主模偏差来自
   O(amp²) 非线性耦合误差,除以 O(amp) 范数化得 O(amp) 斜率。
2. **E_k2 / E_k1(0) 严格 ∝ amp²** (ratio 100× 对应 amp × 10×)—— 这正是
   **三波 triad 耦合理论预测的能量转移率**。GPU Strang-split 完整保留
   了这个 quadratic 信号。
3. E_k3 / E_k1(0) ∝ amp⁴(更高阶级联)。
4. 主模 E_k1 drift 随 amp 放大,但在 amp=1e-4 仍 < 1%。

图:`paper/figures/fig7_1_triad.png`

# 4. DNS_PLAN amp=1e-2 目标的局限

DNS_PLAN §A 原计划跑 amp=1e-2 × 300 周期,并宣称 ΔE/E < 1e-10.  实际:

- **Python prototype `scripts/nonlinear_path1_opsplit.py`**(laptop side 原始)
  amp=1e-2 @ 64x64 @ Lane-Emden (rho_cut=0.05):**第 5 周期 blowup**
  ($|v| \sim 10^{16}$)
- **GPU port** 完全复现这个行为(第 4-5 周期 NaN),确认 **不是移植 bug**
- amp=1e-3 也 blowup 在 ~45 周期
- 只有 amp ≤ 1e-4 才能跑满 100 周期

原因:
1. $\rho_{\rm cut}=0.05$ 逼近表面 $N^2$ 奇点,壁面附近解有陡梯度
2. $64^2$ 无粘性 + 2/3 dealias 仅能处理 $O(\text{amp}^2/\text{res})$ 级别的非线性 aliasing
3. Strang 分裂中 $B$ 通道独立积分 $\dot B = -N^2 V$ 没有回馈到 $V$ 的闭合动力学
   (M 已经消除 $B$),两条路径 $B$ 的累积导致 $\int b^2/N^2$ (表面发散)长时漂移

DNS_PLAN 的 ΔE/E < 1e-10 需要一个 **mass-energy-consistent split** (不是当前
的 Strang 形式)或者 $\rho_{\rm cut} \ge 0.2$ 的较保守恒星截断。这已超出
原 DNS_PLAN 的预期工作量,应写入 review response 而非执行。

# 5. 复现

```bash
# 三个 amp,64×64,100 周期:
for amp in 1e-6 1e-5 1e-4; do
  ANSL_TD_KIND=strang_nonlinear ANSL_COORD_MAP=tanh ANSL_COORD_BETA=2 \
  ANSL_DNS_PERIODS=100 ANSL_DNS_SPP=32 ANSL_RHO_CUT=0.05 \
  ./build/stellar2d --solver anelastic_sl --test dns_triad \
      --ntheta 64 --nr 64 --ps-Lx 1 --ps-Ly 1 \
      --ps-k 1 --perturb $amp --tend 1.0 --cfl 1.0 --ps-nu 0
done
python3 scripts/plot_dns_triad.py   # 生成 fig7_1_triad.png
```

CSV 输出保存到 `runs/dns_expA/triad_amp{1e-6,1e-5,1e-4}.csv`.

# 6. 下一步

- 做 $\rho_{\rm cut}=0.3$ + amp=1e-3 + 300 周期实验(稳定性应会改善)
- 或者实现 energy-consistent Strang 变体(需要与 Python prototype 同步)
- Experiment B(parametric resonance)需要先做 `scan_resonance.py` 找
  $\omega_a \approx 2\omega_b$ 的 mode 对;GPU 移植已支持任意 $(n_g, k_x)$
  IC,基础设施齐全
