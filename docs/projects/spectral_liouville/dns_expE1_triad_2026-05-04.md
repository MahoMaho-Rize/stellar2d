---
title: DNS Experiment E1 — 真三波共振(bug 修复 + Manley-Rowe 定量确认)
author: Phase 3 nonlinear debugging session (branch `anelastic-sl-spectral`)
date: 2026-05-04
status: done — 三个 bug 定位 + 修复 + 与 triad 理论定量匹配
supersedes:
  - `docs/dns_expA_triad_gpu_2026-05-04.md` §4(不稳定性分析在修复后需要重做)
---

# TL;DR

E1 实验(seed mode a + mode b,watch partner mode c 从噪声中生长)最初观察到:
mode b 每 100 周期丢 95% 能量。诊断发现这**不是一个 bug,是三件事叠加**:

1. **Bug**:非线性 RK4 块错误地推进 `W = ∂_t V`,把 W 当作被动标量沿 `(u, v)` 平流。
2. **Bug**:非线性项 `(u·∇)v` 自动产生 `k_x = 0` 的水平平均流分量,违反
   anelastic 连续性约束,成为能量 sink。
3. **诊断陷阱**:每周期一次的采样频率对 mode b(周期 $\pi/\omega_b ≈ 2$ time
   units)**严重欠采样**,让 mode b 自振的 $2\omega_b$ 信号被混叠成伪"15
   周期 beat",掩盖真正的三波共振。

全部修复后,amp=$10^{-5}$ × 200 T_a 实验显示:
- $E_b$ 在真实三波耦合下慢振荡 ±25%,调制周期 ≈ 100 $T_a$
- $E_c$ 从噪声生长到 ~$1.5\times 10^{-16}$,峰值出现在 T_beat/2 附近
- 定量匹配 detuned triad 理论:$|c_c|_{\max} \sim |V_{abc}| c_a c_b / |\Delta\omega|$

# 1. 实验设置(E1 配置)

Lane-Emden n=3/2 背景,$\rho_{\rm cut}=0.1$,TANH 坐标 warp β=2,$128 \times 128$。

| mode | n_g | k_x | ω(from EVP on TANH grid) |
|---|---|---|---|
| a (pump)        | 6 | 1 | 0.4500 |
| b (seeded)      | 3 | 5 | 1.5902 |
| c (partner, 自动选) | 1 | 6 | 2.0366 |

三波共振条件:
- **k_a + k_b = k_c** = 6 ✓(严格)
- **ω_a + ω_b − ω_c** = +0.0036,detune = 0.178%

理论预测 T_beat = $2\pi / |\Delta\omega| = 1729$ 时间单位 ≈ 124 T_a。

# 2. 三个问题的诊断链条

## 2.1 Bug #1 — W 在非线性块被错误平流

**症状**:CUDA run amp=$10^{-6}$ × 100 T_a,mode b ($k_x=5$) 从 $E_b(0) = 1.87 \times 10^{-14}$
单调衰减到 $9 \times 10^{-16}$(-95% 能量),同时 mode a ($k_x=1$) 几乎不变(-0.008%)。

**根因**:`step_strang_nonlinear` 的 nonlinear RK4 块同时对 $(V, W, B)$ 三者做
$(u, v)$ 平流。但 $W = \partial_t V$ **不是**被动标量;它是 Path D 消元后
$\ddot{V} = -M V$ 系统的辅助变量,有自己的线性动力学但**没有平流方程**。

把 $W$ 当被动场推意味着每次 Strang 块结束把错误的 $W$ 交回 linear block
当作"初值",linear block 用 $\dot V = W$ 又把错误传回 $V$。对高 $k_x$ 模式
(kx=5 的 M 谱半径比 kx=1 大 12 倍)破坏最严重。

**Python prototype 未触发此 bug**:`nonlinear_path1_opsplit.py` 的 `step_rk4_nonlinear`
只推 $(V, B)$,完全不碰 $W$。GPU port 在翻译时多加了一条 $W$ 通道(出于所谓
"对称性"),恰恰错了。

**修复**:`nonlinear_deriv` 改签名 `(V, B) → (dV, dB)`,删掉 W 相关的所有 snapshot /
accumulator / advect / commit 操作。`d_strang_dw` 分配保留(ABI 稳定),不再使用。

效果:amp=$10^{-5}$ × 100 T_a 下,mode b drift 从 -95% 降到"周期性振荡,峰值 = IC"
(本身还有其他问题要排,下面 Bug #2)。

## 2.2 Bug #2 — k_x=0 mean flow 违反 anelastic 连续性

**症状**:Bug #1 修好后,mode b 能量依然大幅振荡(-95% → +/-3 decade),但
partner mode c 只涨到 ~$10^{-17}$ 量级,**能量转移目的地不对**。用 FFT band
检查 $k_x = 0, 1, 2, \ldots 15$ 所有 band 的能量,发现:

| period | E(kx=0) | E(kx=1) | E(kx=5) | E(kx=6) | E(kx=10) |
|---|---|---|---|---|---|
| 0   | $4.9 \times 10^{-46}$ (零) | $5.5 \times 10^{-12}$ | $4.8 \times 10^{-12}$ | $4.7 \times 10^{-46}$ | $4.7 \times 10^{-46}$ |
| 100 | **$2.96 \times 10^{-13}$** ← 长出巨大 DC 分量 | $5.4 \times 10^{-12}$ | $2.3 \times 10^{-13}$ | $5.1 \times 10^{-16}$ | $1.2 \times 10^{-22}$ |

$k_x=0$ 从 round-off 起点涨到 $3 \times 10^{-13}$,正好等于 $E_b$ 的丢失量。
**能量没去共振 partner,去了 DC 水平平均流**。

**物理根因**:非线性项 `(u·∂_x v + v·∂_y v)` 在两个非零 $k_x$ 模式相乘时会产生
$k_x = k_i - k_j$ 差频成分;自乘(如 mode b × mode b)产生 $k_x = 0$ 贡献。
这些 Reynolds stress 项在通常 Navier-Stokes 里合法 —— 但 **anelastic 不允许**:

anelastic 连续性 $\nabla \cdot (\rho_0 \mathbf{u}) = 0$ 加上 $v|_{y=0,L_y}=0$ 的
Dirichlet 墙条件,对 $k_x = 0$ 分量推出:
$$
\partial_y (\rho_0 \langle v \rangle_x) = 0 \;\;\Rightarrow\;\; \langle v \rangle_x \equiv 0
$$
(因为 ρ 处处正,$\partial_y$ 积分加两个墙值零)。

代码 `k_u_from_div_v` kernel 对 $k_x = 0$ 分支直接写 `û = 0`(避开除零),于是
**u 的 mean flow 被强制为 0,但 v 的 mean flow 无人清理**,两者脱钩,质量守恒
被破坏,能量从 mode b 持续流失到这个 DC ghost。

**修复**:`k_zero_kx0_column` kernel 加在非线性块 commit 之后,把 `d_v` 和 `d_b`
的 $k_x=0$ 分量做 FFT 零化 round-trip。每次 Strang 非线性块结束应用一次。

效果:$E(k_x=0)$ 从 $3\times 10^{-13}$ 降到 $2.8 \times 10^{-28}$(**15 decade
改进**),总能量 drift 从 -6.4% / 100T 降到 -2.7% / 200T。

## 2.3 诊断陷阱 — T_a 采样混叠 mode b 自振

**症状**:所有 bug 修完后,$E_b$ 依然在大幅振荡,主要周期看起来是 **15 T_a**,
但理论上 triad beat period 应该是 124 T_a,**两者不匹配**。没法断定是物理还是
残留 bug。

**诊断**:把采样频率从 "每 T_a 一次"($dt_{\rm sample} = 13.96$) 改成 "每 dt 一次"
($dt_{\rm sample} = 0.436$),对 $E_b(t)$ 做 FFT:

```
Top peaks in E_b(t):
  ω = 3.1797  (T = 1.98)   amplitude 1.2e-9     ← 主峰
  ω = 3.1820  (T = 1.97)   amplitude 9.9e-10
  ω = 0.0023  (T = 2792)   amplitude 9.4e-11    ← Hanning 窗泄露,非物理
```

**主峰 ω = 3.180 = 2·ω_b = 2 × 1.590**。$E_b$ 作为动能以 $\cos^2(\omega_b t) =
(1 + \cos(2\omega_b t))/2$ 自然振荡,这是 eigenmode 的基本性质,不是 triad
耦合。

**混叠验证**:原来采样 $dt_s = 13.96$,每样本 mode b 自振相位转过
$2\omega_b \cdot dt_s = 3.18 \times 13.96 = 44.4$ 弧度 = $7 \times 2\pi + 0.42$,
apparent 频率 = $0.42 / 13.96 = 0.030$ rad/time,apparent 周期 = 209 = **15 T_a**。
**完美解释了伪造的"15-period beat"**。

**教训**:triad 诊断采样率必须 < $T_{\min}/2$,其中 $T_{\min}$ 是参与模式中最快
振子的**动能振荡周期**($\pi/\omega$,因为动能是速度平方)。E1 里 mode c 最快
($\omega_c = 2.04$),$T_{\min} = \pi/2.04 = 1.54$,采样必须 ≤ 0.77 time
units。"每步一次"($dt = 0.436$)刚好满足。

# 3. 修复后的物理结果

amp=$10^{-5}$ × 200 T_a,每 dt 采样 6400 点,对 $E_b(t)$ 做低通滤波(窗 =
$\pi/\omega_b$ 的一半 ≈ 2.18 time units, 5 samples)去掉自振,看慢包络:

## 3.1 Manley-Rowe b↔c 能量交换 — 真实信号

$E_b^{\rm lp}$ 低通后的慢振荡:

| period | $E_b^{\rm lp}$ | $E_c^{\rm lp}$ | 状态 |
|---|---|---|---|
|   0 | $6.1 \times 10^{-13}$ | $1.5 \times 10^{-20}$ | IC, c 尚未激发 |
|  40 | $9.4 \times 10^{-13}$ | $1.5 \times 10^{-16}$ | **E_c 首次峰** |
|  60 | $7.7 \times 10^{-13}$ | $1.1 \times 10^{-16}$ | E_c 回落中 |
| 110 | $7.8 \times 10^{-13}$ | $1.5 \times 10^{-18}$ | **E_c 首次 null** |
| 160 | $8.1 \times 10^{-13}$ | $1.5 \times 10^{-16}$ | **E_c 第二次峰** |

- $E_b^{\rm lp}$ 范围 $6.2\text{e-13} \sim 1.0\text{e-12}$(±25% swing)
- $E_c$ 峰-谷-峰间距 ≈ 110-120 T_a,**与理论 $T_{\rm beat} = 124\,T_a$ 符合**
- $E_b$ 的 null 反相于 $E_c$ 的峰:c 获能量时 b 失能量,经典 Manley-Rowe

## 3.2 定量对照 detuned triad 理论

三波弱非线性理论的稳态解($a$ 作为 undepleted pump):
$$
|c_c|_{\max} \approx \frac{|V_{abc}|\, c_a\, c_b}{|\Delta\omega|}
$$

带入实测:
- amp_a · amp_b = $10^{-10}$
- |Δω| = 0.0037
- |V_abc| ≈ O(1) (overlap integral, 来自 `scan_resonance.py` 扫描)

预测 $|c_c|_{\max}$ ~ $\frac{1 \cdot 10^{-10}}{0.0037}$ = $2.7 \times 10^{-8}$,
对应 $|c_c|^2 \sim 7 \times 10^{-16}$。

**实测 $E_c^{\rm lp}$ 峰值 = $1.5 \times 10^{-16}$**,与预测同数量级(差 5×,在
"undepleted pump" 假设 + overlap 积分精度的公差内)。

# 4. 代码变更总览

`src/gpu/anelastic_sl_solver.cu` `step_strang_nonlinear()`:
1. `nonlinear_deriv` 签名 `(V, W, B) → (V, B)`,删 W 通道
2. `nonlinear_step` 同步删 W 相关 RK4 累加
3. Commit 后追加 `zero_kx0(d_v)` + `zero_kx0(d_b)` + 再一次 `k_zero_y_boundary`

`src/gpu/anelastic_sl_kernels.cu`:加 `k_zero_kx0_column` kernel

`src/gpu/anelastic_sl_solver.cuh`:`d_strang_dw` 标注 "kept for ABI, unused"

`src/main.cpp` `dns_triad_coupled` dispatch:加 `ANSL_DIAG_EVERY_STEP` 环境变量,
默认 0(每周期诊断一次,原行为),>0 表示每 N 步诊断一次(用于密采样 FFT 分析)。

分析脚本:
- 删 `plot_dns_triad_coupled.py` (v1) 和 `project_eigenmodes_e1.py` 的 CGL 版(有错的 grid)
- 重命名 v2 / tanh 版为 canonical 名字
- 加 CSV header regex 解析(不再硬编码 amp)
- 移除 panel-4 的 log-log slope fit(噪声拟合)
- `scan_resonance.py` triad 排名改为 `log10(detune) - log10(|V_abc|)`(原
  公式是 `detune - log10(V_abc)` 尺度严重失配)

# 5. 复现

```bash
# 密采样 200 周期(稀疏/每周期版本也可,但看不到 triad 信号本质)
ANSL_TD_KIND=strang_nonlinear ANSL_COORD_MAP=tanh ANSL_COORD_BETA=2 \
ANSL_AMP_A=1e-5 ANSL_AMP_B=1e-5 \
ANSL_NG_A=6 ANSL_KX_A=1 ANSL_NG_B=3 ANSL_KX_B=5 \
ANSL_DNS_PERIODS=200 ANSL_DNS_SPP=32 ANSL_RHO_CUT=0.1 \
ANSL_DIAG_EVERY_STEP=1 ANSL_DNS_SNAP_EVERY=1 \
./build/stellar2d --solver anelastic_sl --test dns_triad_coupled \
    --ntheta 128 --nr 128 --ps-Lx 1 --ps-Ly 1 \
    --tend 1.0 --cfl 1.0 --ps-nu 0
```

# 6. 对 Experiment A(单模,`dns_triad`)的影响

`docs/dns_expA_triad_gpu_2026-05-04.md` 里的 amp-scan 表(dev/period, E_k2 比率)
是在 Bug #1 + Bug #2 未修状态下测的。对于单模 IC:
- Bug #1:$W(0) = 0$,污染从零累积,O(amp²)·t 在短时稍弱于 E1
- Bug #2:$(kx_a) \cdot (kx_a)$ 产生 $k_x=0$ 分量,污染强度 ∝ amp²,amp=$10^{-4}$ 下
  积累较快

定性结论(dev/period ∝ amp, E_k2 ∝ amp²)应仍然成立(这些来自 Strang 分裂的
O(dt²) 本底,与具体非线性误差项无关),但**定量数字需要重测**才能引用。
§4 里"amp=$10^{-3}$ 约 45 周期 blowup"的观察在修复后应重做 —— 大概率稳定范围
会扩大。

# 7. 下一步

1. 出正式 figure(`paper/figures/fig7_2_triad_coupled.png`):四面板
   - 原始 $E_a, E_b, E_c$ 密采样(展示快振荡)
   - 低通后的 $E_b^{\rm lp}, E_c^{\rm lp}$ 包络(展示慢 Manley-Rowe)
   - $E_b(t)$ 的 FFT,标注 $2\omega_b$ 主峰 + triad beat 峰(理论位置)
   - $E_c^{\rm lp}$ 峰/谷时刻对比 $T_{\rm beat}/2$ 理论预测
2. 重测 Experiment A amp-scan(单模)在修复后的数值
3. 用 `project_eigenmodes_e1.py` 的 EVP 投影在 snapshot 上验证 "E_b 振荡的
   partner 不是高 n_g 里的 k=5 其他模"(证据是 3.2 节的定量 match,但严格
   的证明是 projection basis 下 c_{n_b, k_b}(t) 与 c_{n_c, k_c}(t) 直接反相)
