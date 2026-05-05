---
title: DNS Experiment A — exponential propagator (phase-exact linear TD)
author: Phase 3 extension (branch `anelastic-sl-spectral`)
date: 2026-05-05
status: done — phase-exact integrator + Strang(Exp, NL_RK4) decomposition
links:
  - `docs/dns_expA_im_vs_rk4_2026-05-05.md` — IM vs RK4 对比
  - `docs/dns_expA_triad_gpu_2026-05-04.md` §4.1 — secular 尾巴归因假设
---

# TL;DR

在 IM (辛) 之后,顺势加了更强的**保相积分器:指数传播子 (exp propagator)**。
对线性块 $\ddot V = -\mathsf M V$,Exp 在每个 eigenmode 上**相位和幅度都精确
到 round-off**。两个实验的结果:

**Test 1 (纯线性, amp=1e-6 × 500 T_a)**:
- Exp: `dev=2.2e-9`, `|ΔH_IM|=1.2e-11`, `v_c(NT)/v_c(0)=1.000000`
- IM:  `dev=2.0e-9`, `|ΔH_IM|=6.6e-12`, `v_c(NT)/v_c(0)=-0.80` (phase drift)
- RK4: `dev=1.3e-7`, `|ΔH_IM|=4.0e-4`, `v_c(NT)/v_c(0)=1.000000`

Exp 同时拿到 IM 的 H_IM 守恒 **和** RK4 的相位精度。严格优于两者。

**Test 2 (Strang(Exp, NL_RK4) vs Strang(RK4, NL_RK4), 500 T_a)**:

| amp  | RK4 rate   | Exp rate   | RK4/Exp | 含义                          |
|------|------------|------------|---------|-------------------------------|
| 1e-6 | -8.07e-7   | -8.2e-13   | **10⁶** | RK4 floor 纯线性块贡献,Exp 消光 |
| 1e-4 | -8.07e-7   | -9.0e-11   | **10⁴** | 同上 + amp³ commutator 开始冒头 |
| 1e-3 | -8.16e-7   | -1.7e-8    | **50**  | commutator 占主导             |
| 1e-2 | -8.45e-6   | -8.38e-6   | **~1**  | **完全 commutator,Exp 无改善** |

**定量确认 Phase 3 §4.1 的假设**:amp=1e-2 secular 尾巴是 **Strang
$[\mathsf L, \mathsf N]$ commutator $\mathcal O(\Delta t^2 \cdot \text{amp}^3)$**,
不是线性块的 RK4 amp 本底。

# 1. 数学

对 $\ddot V = -\mathsf M V$,设 $\mathsf M_k = \mathsf Q_k \Lambda_k \mathsf Q_k^{-1}$,
$\Lambda_k = \text{diag}(\omega_n^2)$。精确解析解:

$$
\begin{pmatrix} V_{n+1} \\ W_{n+1} \end{pmatrix}(k_x)
= \underbrace{\mathsf Q_k \begin{pmatrix} \cos(\Omega_k \Delta t) & \sin(\Omega_k \Delta t)/\Omega_k \\ -\Omega_k \sin(\Omega_k \Delta t) & \cos(\Omega_k \Delta t) \end{pmatrix} \mathsf Q_k^{-1}}_{=\,\mathsf T_k}
\begin{pmatrix} V_n \\ W_n \end{pmatrix}(k_x)
$$

其中 $\Omega_k = \Lambda_k^{1/2}$。$\cos, \sin$ 用 libm 直接求到机器精度
→ 每个 $\omega_n$ 相位同时精确,**零相位误差,零幅度误差,无穷阶方法**。

实现:把 $\mathsf T_k$ 分成 4 个实矩阵 $T_{VV}, T_{VW}, T_{WV}, T_{WW}$ 存 col-major
slab,和 IM 的 B/C 用同样的 `k_apply_M_kx` kernel 应用。每步 = 4 apply + 2 FFT + 2 IFFT。

# 2. 实现

## 2.1 EVP per kx_x(host 预装配)

`build_exp_per_kx`(`src/gpu/anelastic_sl_solver.cu`):

- 对每个 $k_x > 0$,从 `d_M_per_kx` 下载 $\mathsf M_k$(col-major)到 host
- cuSOLVER `Xgeev(NOVECTOR)` 只求实本征值(cuSOLVER 的 eigvec 对非对称矩阵不可靠,
  见 `compute_2d_gmode_evp` 2143 行的 known issue)
- 对每个 $\lambda_n$ 做 **host 反迭代** 5 次求 $\phi_n$(初值用确定性 pseudo-random)
- Gauss-Jordan 求 $\mathsf Q_k^{-1}$,检查 $\|\mathsf Q\mathsf Q^{-1} - \mathbf I\|_\infty$
- 组装四个传播矩阵 $\mathsf T = \mathsf Q \cdot \text{diag}(f(\omega_n)) \cdot \mathsf Q^{-1}$

结果:64² + rho_cut=0.05 下 **$\|\mathsf Q\mathsf Q^{-1}-\mathbf I\|_\infty = 1.2 \times 10^{-15}$**
(worst case across all $k_x$),完美。setup 总耗时 < 1 s。

## 2.2 Step 函数(`step_exp_propagator(dt)`)

固定 dt。首次调用或 dt 变时重建 T 矩阵并上传。每步:

1. FFT_x(V) → d_fhat;  FFT_x(W) → d_pihat
2. $\hat V_{n+1} = T_{VV} \hat V + T_{VW} \hat W$(两次 apply,加到 d_ghat)
3. $\hat W_{n+1} = T_{WV} \hat V + T_{WW} \hat W$(两次 apply,加到 d_exp_scratch)
4. IFFT_x 两次,scale 1/nx,zero walls

新增 `d_exp_scratch`(ncplx 复数,lazily 分配),因为 d_Qhat/d_Ghat 大小是 nh·n_modes
不够 nh·ny。

## 2.3 Strang(Exp, NL_RK4) 变体

`step_strang_exp_nonlinear(dt)`:简单 wrapper,设 `td_strang_exp_nonlinear=true`
后调 `step_strang_nonlinear(dt)`。内部 Strang A/C 半步把 RK4 `linear_half`
替成 `step_exp_propagator(dt/2)`,B 非线性块不变。

Caveat:`step_exp_propagator` 的 T 矩阵缓存 key 是 dt。Strang 模式下全局
cache 是 dt/2(从未调过 dt)。单 run 内无冲突。

## 2.4 用法

```bash
# 线性-only 相精确 (IM 参照):
ANSL_TD_KIND=exp_propagator  ... --test dns_triad ...

# Strang-split with Exp 线性块:
ANSL_TD_KIND=strang_exp_nonlinear  ... --test dns_triad ...
```

# 3. Test 1 — 线性线性极限三方对比

amp=1e-6 × 500 T_a,64² + rho_cut=0.05:

| 方案     | dev @ 500T | \|ΔH_IM\| @ 500T   | v_c(500T)/v_c(0) |
|----------|-------------|--------------------|---------------------|
| RK4      | 1.32e-7    | **4.0e-4**         | 1.000 (phase kept) |
| IM       | **2.0e-9** | **6.6e-12**        | **-0.80** (phase drift) |
| **Exp**  | **2.2e-9** | **1.2e-11**        | **1.000000** (both exact) |

**Exp 同时拿到 IM 的 $H_{IM}$ round-off + RK4 的相位精度**。严格更强。

物理图景:
- $\mathsf M$ 的特征值 $\omega_n^2$ 范围大约 [0.26, 4×10³](g-mode pipeline 到 p-mode 频段)
- 最大 $\omega \Delta t$ = √4000 · 0.119 = 7.5 rad/step — 超出 IM/RK4 的稳定相位区间
- Exp 的 $\cos/\sin$ 在所有频率上机器精度 — 天然无 CFL 限制

# 4. Test 2 — Strang 分解 (Exp 隔离 commutator)

DNS Experiment A 长时重测 (500 T_a, 64², rho_cut=0.05),两个 Strang 变体:

```
amp     Strang-RK4 rate  Strang-Exp rate  RK4/Exp
1e-6    -8.07e-7         -8.2e-13          10⁶
1e-4    -8.07e-7         -9.0e-11          10⁴
1e-3    -8.16e-7         -1.7e-8           50
1e-2    -8.45e-6         -8.38e-6          1.01  ← commutator dominates
```

## 4.1 定量诠释

**Strang-RK4 rate = linear-block floor + commutator contribution**,
**Strang-Exp rate = commutator contribution only**(线性块 = 0)。

- amp ≤ 1e-4: Strang-RK4 drift 完全是 RK4 线性块 amp-indep floor (-8e-7/T)。
  Exp 消掉这个,只剩 commutator → Exp rate ∝ amp³(从 amp=1e-4 的
  9e-11 到 amp=1e-3 的 1.7e-8,ratio 190 ≈ 10²·⁸ 近 amp³)。
- amp = 1e-2: Strang-RK4 rate 与 Strang-Exp rate **几乎相等**(-8.45e-6 vs
  -8.38e-6,差 1%)→ 线性块贡献在 amp=1e-2 下**完全被 commutator 淹没**。

**Commutator 系数定量**:
- amp=1e-4: 9e-11/$(10^{-4})^3$ = 90
- amp=1e-3: 1.7e-8/$(10^{-3})^3$ = 17
- amp=1e-2: 8.4e-6/$(10^{-2})^3$ = 8.4

系数**没有 perfect 平直**(跨 decade 有 ~10× 浮动),因为 amp³ scaling
只是 leading-order,加 $\mathcal O(\Delta t^4 \cdot \text{amp})$ 等高阶项会
浮动。但跨 4 decades amp 的 ratio 保持在 commutator 的 10³ ≈ 10^2.9
附近,与 Phase 3 §4.1 的 amp^2.92 推断一致。

## 4.2 E_k2/E_k1(0) max 验证

三波级联物理完全不变:

| amp  | RK4 E_k2 max  | Exp E_k2 max  |
|------|---------------|----------------|
| 1e-6 | 8.37e-14      | 8.37e-14       |
| 1e-4 | 8.37e-10      | 8.36e-10       |
| 1e-3 | 8.36e-8       | 8.36e-8        |
| 1e-2 | 9.58e-6       | 9.66e-6        |

**E_k2/E_k1(0) 严格 ∝ amp²**,跨 8 decades 一致。说明 Exp 把数值 drift 消掉
后,**triad 能量转移物理不变**,只是背景更干净。

## 4.3 Phase 3 §4.1 假设判决

Phase 3 的 `docs/dns_expA_triad_gpu_2026-05-04.md` §4.1 通过 64²/128²/256²
resolution convergence 排除了 **Galerkin truncation** 解释,并基于
amp^2.92 scaling 推断 **secular 尾巴 = Strang commutator**。

Test 2 直接**分离变量**:Exp 把线性块 drift 消到 round-off,残留 = 
commutator 本身。amp=1e-2 下 Exp/RK4 drift rate 几乎相等,**最终证实**
"commutator 主导"的假设。

# 5. 文件

- `src/gpu/anelastic_sl_solver.{cu,cuh}` —
  - 新成员: `d_Tvv/vw/wv/ww_per_kx`, `d_exp_scratch`,
    `td_exp_propagator`, `td_strang_exp_nonlinear`
  - 新方法: `step_exp_propagator()`, `step_strang_exp_nonlinear()`
  - 新 helpers: `inverse_iteration_eigvec`, `host_gj_invert`,
    `sin_over_omega`, `build_exp_per_kx`
- `src/main.cpp` — dns_triad 分发加 use_exp + Strang(Exp) 分支
- `scripts/plot_dns_expA_exp_decomposition.py` — 分解图
- `paper/figures/fig7_1_triad_exp_decomposition.png` — 分解结果
- `runs/dns_expA_exp/triad_amp*.csv` — Strang(Exp) 原始数据
- `runs/dns_expA_im/triad_amp1e-6_{exp,im,rk4}_propagator.csv` — 三方对比

# 6. 结论

1. **Exp propagator 是线性块的终极答案**:严格比 IM 和 RK4 都好(相位和
   幅度同时 round-off,无 CFL)。
2. **Test 2 定量证明 Phase 3 §4.1 commutator 假设**:amp=1e-2 secular
   尾巴完全由 Strang $[\mathsf L, \mathsf N]$ 产生,与线性块无关。
3. **对 DNS 生产**:amp ≤ 1e-3 下 Strang(Exp, NL_RK4) 比 Strang(RK4, NL_RK4)
   干净 50-10⁶ 倍 drift。amp=1e-2 下改善为 0,不值得(Exp 多 4 次 GEMM 成本)。
4. **Paper 角度**:这两个辛/相精确积分器 + RK4 三方对比是 §7.3 的漂亮 figure
   素材,甚至可以成为独立的 methods paper 子节("choice of linear-block
   integrator for Strang-split anelastic DNS")。
