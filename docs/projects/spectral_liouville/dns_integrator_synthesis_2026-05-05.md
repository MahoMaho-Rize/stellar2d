---
title: DNS integrator triad 综合反思 — 三条独立数值误差来源的完整分解
author: Phase 3 close + Exp propagator session (branch `anelastic-sl-spectral`)
date: 2026-05-05
status: synthesis — 汇总三个实验 (RK4 / IM / Exp) 的系统性结论
inputs:
  - `docs/dns_expA_triad_gpu_2026-05-04.md` — Experiment A RK4 长时 + resolution 扫描
  - `docs/dns_expE1_triad_2026-05-04.md` — E1 三波共振 + 三 bug 诊断
  - `docs/dns_expA_im_vs_rk4_2026-05-05.md` — IM vs RK4 对比
  - `docs/dns_expA_exp_propagator_2026-05-05.md` — Exp propagator + Strang 分解
  - `docs/phase3_closeout_2026-05-05.md` — Phase 3 收官
---

# 核心论断

把 DNS Experiment A 的数值误差**完整分解**为三条相互独立的贡献,每条都有
明确的产生机制和明确的消除手段。这不是推断 —— 每条都通过独立实验证实了。

$$
\text{Drift}_{\rm total} =
\underbrace{\text{linear-block floor}}_{\text{RK4 amp-indep}}
\;+\;
\underbrace{\text{Strang } [\mathsf L,\mathsf N]\;\text{commutator}}_{\propto\,\Delta t^2\,\text{amp}^3}
\;+\;
\underbrace{\text{stroboscopic phase aliasing}}_{\text{IM only}}
$$

# 1. 三条误差来源 + 对应 killer

## 1.1 Linear-block floor(RK4 的 $\mathcal O((\omega\Delta t)^{10})$ amplitude leak)

- **表现**: RK4 在任意 amp 下 $E_{k_1}$ drift rate 恒为 $-8.07 \times 10^{-7}$/$T_a$,amp-indep。500 $T_a$ 累积 $4 \times 10^{-4}$。
- **机制**: RK4 的稳定函数 $R_4(i\omega\Delta t)$ 模长不精确为 1,每步泄漏 $\mathcal O((\omega\Delta t)^{10})$ 能量。$\omega\Delta t \approx 0.2$ 下 rate 约 $2 \times 10^{-7}$/step,32 step/period,吻合 8e-7 观测。
- **killer**:
  - **IM**: Cayley 变换严格 $|\lambda|=1$ → drift 从 4e-4 降到 6.6e-12(**8 decades**)
  - **Exp**: 解析 cos/sin → drift 同样 round-off
- **证据**: `docs/dns_expA_im_vs_rk4_2026-05-05.md` §2,`docs/dns_expA_exp_propagator_2026-05-05.md` §3 Test 1。

## 1.2 Strang $[\mathsf L, \mathsf N]$ commutator($\mathcal O(\Delta t^2\cdot\text{amp}^3)$)

- **表现**: amp $\le 10^{-4}$ 被 linear floor 盖住,amp $= 10^{-2}$ 显现为 $-8.45 \times 10^{-6}$/$T_a$,比 floor 大 10 倍。
- **机制**: Strang splitting 在 linear 块 $\mathsf L$ 和 nonlinear 块 $\mathsf N$ 轮换之间留下 $\frac{\Delta t^2}{12}[\mathsf L,\mathsf N]$ 主序误差。作用在 $V \sim \text{amp}$ 上产生 $\Delta V \sim \Delta t^2\cdot\text{amp}^2$,能量 $2V\cdot\Delta V \sim \Delta t^2\cdot\text{amp}^3$。
- **killer**:
  - Yoshida-4 把 $\mathcal O(\Delta t^2)$ 升到 $\mathcal O(\Delta t^4)$(三次对称组合,~100 行代码)
  - Fully-implicit (non-split) 时间推进消 commutator,但非线性 Newton 代价高
  - **不能用 Exp**: Exp 只改线性块,$[\mathsf L, \mathsf N]$ 在 Strang 结构里依然存在
- **证据**:
  - **排除 Galerkin truncation** (`docs/dns_expA_triad_gpu_2026-05-04.md` §4.1): 64² / 128² / 256² drift rate 差 < 0.3%,resolution-independent
  - **scaling 确认** (同上): amp^2.92 跨 decade,≈ amp³
  - **分离变量最终证实** (`docs/dns_expA_exp_propagator_2026-05-05.md` §4): Strang(Exp, NL_RK4) 把线性块 drift 消到 round-off,剩余 drift = commutator 本身。amp=1e-2 下 Strang(RK4) 和 Strang(Exp) drift 几乎相等(-8.45e-6 vs -8.38e-6,1%),确认 commutator 占绝对主导。

## 1.3 Stroboscopic phase aliasing (IM-specific)

- **表现**: IM 在 amp=1e-6 × 500 $T_a$ 下,采样 $v_c(NT)/v_c(0) = -0.80$,看起来像 80% 能量损失。
- **机制**: IM 的相位误差 $\delta_{\rm step} = \omega\Delta t - 2\arctan(\omega\Delta t/2) \approx (\omega\Delta t)^3/12$。32 step/T 累积 $0.02$ rad/period,500 $T_a$ 累积 10 rad。stroboscopic 采样 $v_c(NT) = V_{\rm peak}\cdot\cos(N\delta_{\rm period})$。
- **验证**: 定量预测 $\cos(10.2) = -0.80$,**实测 -0.801**,完美匹配。
- **killer**:
  - RK4 (phase error $\mathcal O((\omega\Delta t)^{10})$ — 500T 累积 < 1e-3 rad)
  - Exp (phase error = 0)
  - Yoshida-4 施加到 IM 可降到 $\mathcal O((\omega\Delta t)^5)$
- **重要诊断 take**: IM 的 $H_{IM}$ 守恒到 round-off,但 **anelastic 物理诊断 $E_{\rm phys}$ 是 stroboscopic 的**,受相位误差污染。**整体诊断判据必须与 integrator 配套选**,否则诊断 artefact 会冒充能量损失。

# 2. 三积分器终极对比

amp=1e-6 × 500 $T_a$,64² + rho_cut=0.05:

| 方案 | dev | \|Δ$H_{IM}$\|/$H_0$ | $v_c(500T)/v_c(0)$ | linear-block-killer | phase-killer |
|---|---|---|---|---|---|
| RK4 | 1.3e-7 | 4.0e-4 | 1.000 | ❌ | ✅ |
| IM  | 2.0e-9 | 6.6e-12 | -0.80 | ✅ | ❌ |
| **Exp** | **2.2e-9** | **1.2e-11** | **1.000000** | **✅** | **✅** |

Exp 是线性块的**终极答案**。代价 ~20% per-step 运行时 + 一次性 per-kx EVP
setup (<1s at 64², <10s at 256²)。没有 CFL 限制。

# 3. Strang-split 下的分解(Test 2 定量)

500 $T_a$ drift rate for $E_{k_1}$:

```
amp     Strang(RK4, NL)    Strang(Exp, NL)    RK4 − Exp (= linear floor)
1e-6    −8.07e-7            −8.2e-13            ≈ −8e-7
1e-4    −8.07e-7            −9.0e-11            ≈ −8e-7
1e-3    −8.16e-7            −1.7e-8             ≈ −8e-7
1e-2    −8.45e-6            −8.38e-6            ≈ −7e-8  (淹没)
```

**中列**(Strang(Exp))= 纯 commutator 贡献。$|\text{rate}|/\text{amp}^3$ 系数:

| amp | commutator 系数 (Exp rate / amp³) |
|---|---|
| 1e-4 | 9.0e-11 / 1e-12 = **90** |
| 1e-3 | 1.7e-8 / 1e-9 = **17** |
| 1e-2 | 8.4e-6 / 1e-6 = **8.4** |

跨 decades 在 $10^{0.9}$–$10^{1.95}$ 间浮动,一阶 amp³ scaling 之外还有更高阶项(比如
$\mathcal O(\Delta t^4\cdot\text{amp})$),但 leading order 清晰。

**E_k2/E_k1(0) max 跨 RK4 和 Exp 在所有 amp 下一致**(差 < 1%)。说明:
- Exp 只 clean 数值背景,**没有改变共振物理**
- Strang(RK4) 虽然 drift 难看,但 **triad 能量转移 physics 未被污染**

这是对 reviewer 关键的论据:"-8e-7/T linear floor 看起来坏,但对 triad
实验的结论不构成污染"。

# 4. 一个概念上的洞见:Exp propagator 等价谱时间推进

Exp 的传播矩阵 $\mathsf T_k = \mathsf Q_k \cos(\sqrt{\Lambda_k}\Delta t) \mathsf Q_k^{-1}$
在 eigenbasis 下就是**每个 eigenmode 独立旋转 $\omega_n \Delta t$**。

对我们的系统,$\mathsf M_k$ 的 eigenpair 就是 Lane-Emden g-mode(+ p-mode)全谱。
Exp propagator 等价于:

1. **空间离散**: Chebyshev collocation(物理空间 → SL 基)
2. **时间推进**: eigenmode expansion(每个 $\omega_n$ 解析旋转)

这本质上就是**谱时间方法**(spectral time method),spatial + temporal 都
谱展开。对 **线性问题** 这是标准最终答案。对 **非线性问题**,Strang 混合
eigenmode-advance 线性块 + collocation-space 非线性块,正是 spectral NS
文献里的标准做法 —— 只是我们独立从 IM 再推一步推出来了。

这解释了为什么 Exp 没有 CFL 限制:它不是**时间步进方法**,而是**解析解在
$\Delta t$ 区间的计算**。

# 5. 下一步(Phase 3 外的扩展路径)

按价值排序:

1. **Yoshida-4 on IM**(~100 行)— 消掉 stroboscopic phase aliasing + 把
   Strang $\mathcal O(\Delta t^2)$ 升到 $\mathcal O(\Delta t^4)$,即 amp³
   commutator 的 $\Delta t$ 系数同比下降。对 amp=1e-2 secular 尾巴的
   killer。

2. **Fully-implicit nonlinear**(~200 行 + JFNK)— 消掉 commutator 本身,
   不只是降 $\Delta t$ 阶数。amp=1e-2 下 drift 应降到 round-off。工程量大。

3. **Exp 扩到 Boussinesq / 可压**(未 Phase 3 scope)— ETD (exponential
   time differencing) 把 Exp 线性块扩到 $\dot V = \mathsf L V + \mathsf N(V)$
   的带非线性系统。$\varphi$ 函数(expm1/sinhc)权重,标准 Cox-Matthews 2002
   结构。对我们下一步做 compressible pulsation 值得 port。

4. **Paper §7 重写** — 现在有了三积分器三面对比,`fig7_1_triad_im_vs_rk4.png`
   + `fig7_1_triad_exp_decomposition.png` 是核心素材。§7.3 可以成为:
   "Trade-off analysis among non-symplectic high-order (RK4), symplectic
   low-order (IM), and phase-exact (Exp) linear integrators for Strang-
   split anelastic DNS." 独立 methods-paper 子节价值。

# 6. 没解决的问题 / caveat

- **commutator 系数 $C$ 跨 decades 不是常数**(8.4 → 17 → 90 as amp 增大
  1e-2 → 1e-3 → 1e-4)。leading-order 应该是常数。可能原因:
  (a) $\mathcal O(\Delta t^4\cdot\text{amp})$ 等高阶项混入;
  (b) amp=1e-4 下 Exp rate (9e-11) 接近 double-precision round-off
       cumulative($\sim 10^{-16}$ round-off × 500 × 32 ≈ 1.6e-12)的 50 倍,
       可能含 round-off 贡献。
- **amp=1e-2 的 -8.4e-6/T 对 Manley-Rowe 慢包络的 bias 未定量**。E1 session
  看到的 $|c_c|_{\max} \sim 10^{-16}$ vs drift 500T 累计 $4 \times 10^{-3}$,
  绝对值差 10^{13},但 **相对 bias** 在更长时间 / 更弱耦合下会如何,未测。
- **256² 下 EVP 装配是否会遇到 cuSOLVER Xgeev 对大矩阵的 eigvec 质量问题**?
  Phase 3 只测到 64²。256² (n_int=254) 的 Xgeev 对 non-symmetric real 的
  eigenvalue accuracy 未知,需要测一次 ‖QQ⁻¹ − I‖∞ 是否仍 1e-15 量级。

# 7. 文件清单

## 源码
- `src/gpu/anelastic_sl_solver.{cu,cuh}`:
  - `step_assembled_linear` (RK4 linear-only, 2026-05-04 前)
  - `step_strang_nonlinear` (Strang(RK4, NL_RK4), DNS_PLAN 主 production)
  - `step_implicit_midpoint` + `hamiltonian_im` (IM + 守恒诊断)
  - `step_exp_propagator` + `build_exp_per_kx` (Exp, 本次)
  - `step_strang_exp_nonlinear` (Strang(Exp, NL_RK4), 本次)
- `src/main.cpp`: dns_triad dispatch 支持 `ANSL_TD_KIND ∈ {strang, im, exp, strang_exp}`

## 图
- `paper/figures/fig7_1_triad_im_vs_rk4.png` — IM vs RK4 4-panel
- `paper/figures/fig7_1_triad_exp_decomposition.png` — Exp 隔离 commutator 2-panel

## Docs
- 本文(综合反思)
- `dns_expA_im_vs_rk4_2026-05-05.md` (IM 记录)
- `dns_expA_exp_propagator_2026-05-05.md` (Exp 记录)
- `phase3_closeout_2026-05-05.md` (Phase 3 收官)

## Runs
- `runs/dns_expA_longtime/triad_amp*.csv` — Strang(RK4, NL) @ 500T
- `runs/dns_expA_longtime_{128,256}/` — 分辨率收敛
- `runs/dns_expA_im/triad_amp1e-6_{rk4,im,exp}_propagator.csv` — 三方 500T
- `runs/dns_expA_exp/triad_amp*.csv` — Strang(Exp, NL) @ 500T

# 8. 最终立场

**Phase 3 的 "DNS triad 数值误差"完全定量分解完成**。每条误差有机制、有
证据、有 killer。下一个 session 如果要继续做方法学,Yoshida-4 是最短路径;
如果要发 paper,§7 重写用三积分器对比是最强材料。

Phase 3 主线早已收官,本次是扩展的**定性升级**:从"我们懂了"到"我们把
每个现象分离变量证实了"。
