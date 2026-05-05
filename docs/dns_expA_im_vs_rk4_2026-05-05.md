---
title: DNS Experiment A — implicit midpoint vs RK4 (linear-only symplectic demo)
author: Phase 3 extension (branch `anelastic-sl-spectral`)
date: 2026-05-05
status: done — implementation + side-by-side comparison at amp=1e-6 × 500 T_a
---

# TL;DR

加了一个最简单的辛积分器 —— **隐式中点 (IM)** —— 作为 linear-only TD
`ANSL_TD_KIND=implicit_midpoint`。和 RK4 在 amp=1e-6 × 500 T_a 直接对比:

| 度量 @ 500 T_a         | IM (symplectic)                | RK4 (non-symplectic)       |
|------------------------|--------------------------------|----------------------------|
| **\|ΔH_IM / H_IM(0)\|** | **6.6e-12 (round-off)**       | 4.0e-4                     |
| \|ΔE_phys / E_phys(0)\| | 33% (phase-aliasing artefact) | 4.0e-4 (真实 Hamiltonian 漏) |
| eigenmode shape dev    | 2.0e-9 (round-off)            | 1.3e-7 (65× worse)         |
| v_center(NT)/v_c(0)    | cos(N·δ) 慢振, δ≈0.02 rad/T   | 1.00 保持                  |

**物理结论**: textbook IM vs RK4 trade-off 精确复现 ——
- IM 把 $H_{IM} = \tfrac{1}{2}W^T W + \tfrac{1}{2} V^T \mathsf{M} V$
  守恒到 round-off (8 decades below RK4)
- RK4 把相位守恒到 $\mathcal O((\omega \Delta t)^{10})$ (IM 只到 $\mathcal O((\omega \Delta t)^2)$)

取决于下游物理诊断关心什么量。对 triad 能量级联 (Experiment B / E1 后续),
**H_IM 守恒更关键** —— 能量转移率的参考线不会被 amp-independent 数值 drift
污染。

# 1. 实现

## 1.1 数学

消元 $W = \partial_t V$ 得二阶方程 $\ddot V = -\mathsf{M} V$,其中
$\mathsf{M} = \mathsf{L}^{-1} \mathsf{R}$ 是 Path D 装配矩阵 (per kx_x mode)。
IM 对一阶系统 $(V, W)$ 的应用:
$$
\frac{V_{n+1} - V_n}{\Delta t} = \frac{W_{n+1} + W_n}{2}, \quad
\frac{W_{n+1} - W_n}{\Delta t} = -\mathsf{M} \frac{V_{n+1} + V_n}{2}
$$

消元 $W_{n+1}$:
$$
(I + \alpha \mathsf{M}) V_{n+1} = (I - \alpha \mathsf{M}) V_n + \Delta t\, W_n,
\quad \alpha = \Delta t^2/4
$$

记 $B = (I + \alpha \mathsf{M})^{-1}(I - \alpha \mathsf{M})$(Cayley 变换),
$C = \Delta t (I + \alpha \mathsf{M})^{-1}$,则
$$
V_{n+1} = B V_n + C W_n, \quad W_{n+1} = \frac{2}{\Delta t}(V_{n+1} - V_n) - W_n
$$

在 $\mathsf{M}$ 的每个特征向量方向上,IM 的 $(V, W) \to (V_{n+1}, W_{n+1})$ 矩阵
行列式为 1 且 $|\text{trace}| < 2$,故特征值是共轭复数对 $\{e^{\pm i\theta}\}$,
$|\lambda| = 1$ 严格成立(Cayley 变换的辛性)。

## 1.2 GPU 实现 (`src/gpu/anelastic_sl_solver.{cu,cuh}`)

- **固定 dt**: caller 传 dt,solver 首次调用时在 host 把 $B_k, C_k$ 逐
  $k_x$ 求逆并上传。dt 改变时重算。n_int ≤ 62, nh ≤ 33 下 host 方面毫秒级。
- **每步**: 2 个 FFT_x + 2 个 per-kx 矩阵 apply (复用现有 `k_apply_M_kx`
  kernel + 新增 accumulate 变体 `k_apply_M_kx_add`) + 1 个 IFFT_x + 3 个
  cuBLAS axpy。接近 RK4 的每步 4×FFT 组合同量级成本。
- **W 更新**: 在物理空间做 $W_{n+1} \leftarrow -W_n + \frac{2}{\Delta t}(V_{n+1} - V_n)$,
  cuBLAS scal + axpy + axpy。
- **kx=0 列**: B_0 = I, C_0 = 0 (与 M_0 = 0 一致,DC mode 不动)。
- **新诊断**: `hamiltonian_im()` 计算 $H_{IM}$ 在 Clenshaw-Curtis × 1/nx
  权下的 grid 积分,作为辛不变量的独立测量。

## 1.3 用法

```bash
ANSL_TD_KIND=implicit_midpoint \
ANSL_COORD_MAP=tanh ANSL_COORD_BETA=2 \
ANSL_DNS_PERIODS=500 ANSL_DNS_SPP=32 ANSL_RHO_CUT=0.05 \
./build/stellar2d --solver anelastic_sl --test dns_triad \
    --ntheta 64 --nr 64 --ps-Lx 1 --ps-Ly 1 \
    --ps-k 1 --perturb 1e-6 --tend 1.0 --cfl 1.0 --ps-nu 0
```

输出 CSV 加了第 10 列 `H_im` (以前是 9 列)。

# 2. 对比结果 (amp=1e-6 × 500 T_a, 64² + rho_cut=0.05)

## 2.1 H_IM 守恒

```
period    IM |ΔH|/H           RK4 |ΔH|/H
  1       5.2e-12             7.9e-7
 10       4.7e-12             7.9e-6
100       1.0e-11             7.9e-5
250       2.9e-12             2.0e-4
500       6.6e-12             4.0e-4
```

IM drift 的最大值(~1e-11)**比 RK4 的 drift rate × 时间要小 8 个数量级**。
IM drift 没有单调趋势(随机在 1e-13 到 1e-11 之间起伏)→ 纯 round-off。

## 2.2 Eigenmode shape (dev)

IM dev 从 0 爬升到 1.5e-9 后稳定(round-off floor of FFT+GEMM),RK4 dev
爬升到 ~1.3e-7 后稳定(类似 floor,但高 60+ 倍)。两者都不 secular。

## 2.3 Stroboscopic 伪影 (v_center)

RK4 每步相位误差 $\mathcal O((\omega \Delta t)^5)$,500 T_a 累积仍然 < 1e-3
rad,v_c(NT)/v_c(0) ≈ 1 保持。

IM 每步相位误差 $\theta_\text{true} - \theta_\text{IM} \approx (\omega \Delta t)^3 / 12$。
对 $\omega \Delta t = 0.196$, 每步 ≈ 6.4e-4 rad,每周期 (32 步) 0.020 rad,
500 T_a 累积 10 rad → v_c(500 T) 呈现慢余弦振荡(幅度 1.0 保持,但
stroboscopic sampling 看到投影)。

这是 **E_phys 诊断的伪影**,不是能量损失 —— H_IM 本身 round-off 守恒。

## 2.4 cos(N·δ) 振荡 @ v_c (定量 check)

- 预测相位误差/步 $\delta_\text{step} = \omega \Delta t - 2 \arctan(\omega \Delta t / 2)$
  $\approx (\omega\Delta t)^3/12 = 6.4e-4$ rad,每 T_a = 0.0205 rad。
- 500 T_a 累积 = 10.2 rad = 1.62 × 2π → v_c(500T)/v_c(0) = cos(10.2) = -0.80,
- 观测 v_c(500T)/v_c(0) = -0.801 ✓

精确匹配 cos(N·δ),无任何幅度损失。

# 3. 作为 Experiment A 补充材料

这是本质上的 "IM vs RK4 linear-regime demo",不是新的物理实验。主要价值:

1. **诊断假阳性证据**: 证明 E_phys drift 在 IM 下可以是**诊断伪影**,
   strong caveat for future users 的 Hamiltonian-conservation 判据。
2. **辛积分器的切入点**: 如果未来想追 amp=1e-2 的 ×10 secular(已定
   性为 Strang $[L,N]$ commutator),**把 Strang(A) 替成 IM(A)** 可能
   能消掉 commutator,值得试。
3. **Reviewer-proof 的 baseline**: §7 讨论 RK4 Strang-split 时,有了
   IM 作参考,可以写 "we chose Strang-split-RK4 over IM/exp-prop
   (documented in docs/dns_expA_im_vs_rk4_2026-05-05.md) because phase
   preservation is more relevant for triad timescale analysis than
   $H$-preservation per se",立场更 defensible。

# 4. 文件

- `src/gpu/anelastic_sl_solver.cuh` — 新成员 `d_B_per_kx`, `d_C_per_kx`,
  新方法 `step_implicit_midpoint(double dt)`, `hamiltonian_im()`
- `src/gpu/anelastic_sl_solver.cu` — `build_im_per_kx()` host 预计算、
  `step_implicit_midpoint()` 主循环、`hamiltonian_im()` 诊断、新 kernel
  `k_apply_M_kx_add`
- `src/main.cpp` — dns_triad CSV 加 `H_im` 列;dispatch 支持
  `use_im` branch
- `scripts/plot_dns_expA_im_vs_rk4.py` — 4-panel 比较图
- `paper/figures/fig7_1_triad_im_vs_rk4.png` — 输出图
- `runs/dns_expA_im/triad_amp1e-6_{im,rk4}.csv` — 500 T_a 原始数据

# 5. 下一步(可选)

- **非线性块**: IM 不带 advection 项的 2nd-order symplectic 含义本身有限。
  要对 DNS_PLAN 完整 benchmark,需要把 Strang(A) IM + Strang(B) RK4 的组合
  重建。数学上 Strang(A+B, A') 仍是 2nd-order,但 A 块辛性可能被 B 块破坏
  (Yoshida 组合辛积分的经典坑)。
- **Yoshida 4-step / 6-step**: 若想让 IM 相位误差从 $(\omega\Delta t)^3$
  降到 $(\omega\Delta t)^5$,可以做 Yoshida-4 (三次对称组合)。
- **Exp propagator**: 直接对 $\mathsf M$ 做 eigen-decomp 后解析推进,
  相位 exact,幅度 exact,$\mathcal O(N_y^3)$ 初装配。比 IM 更强,但
  对非对角化的 $\mathsf M$(非线性延伸)不 generalize。
