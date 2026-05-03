---
title: SL spectral — 坐标变换 stretch 实验(TANH 几何 / LOGRHO 物理)
author: Phase 1e 续作(branch `anelastic-sl-spectral`)
date: 2026-05-03
---

# 动机

`docs/eigenmode_td_filter_basis_2026-05-03.md` 结尾留下一个开放问题:

> Reduced-pressure 变换让 $1/\rho_0$ 不进矩阵系数,但势函数
> $\tilde W \sim 1/\rho_0^2$ 在 $\rho_0 \to 0$ 处仍然发散。当前靠 `rho_cut=0.01`
> 把域截到 $[0, 0.99R]$ 在截断端点硬加 Dirichlet BC 让问题 well-posed。

合作者的直觉:**是否可以通过"非常好的密度势拟合"把 $\tilde W$ 压下去?**

分析了两条路径:
- 纯 $\rho_0(y)$ 的解析重拟合 —— 不行。$\tilde W$ 的代数发散是
  reduced-pressure 变换本身的后果,$\rho_0 \sim (y_s - y)^3$ 是 Lane-Emden
  解在表面的渐近行为,与拟合精度无关。
- **改变自变量坐标** —— 可以。把 CGL 节点布在某个 stretched 坐标 $s$ 上,
  使 $y(s)$ 在边界附近采样密集;物理微分 $d/dy = (1/y'(s))\, d/ds$。
  这正是 GYRE 内部 $(y_1, y_2, y_3, y_4)$ 变量的设计思路。

本文档记录两个坐标变换 `TANH`(纯几何)和 `LOGRHO`(物理驱动)的实现 +
测试 + 发现。**核心结论:TANH 有效,LOGRHO 需要更深入的重构才有效。**

# 1. 实现总览

## 1.1 架构改动

`AnelasticSLSolver::set_background` 的前半段重构成:

1. **$s$ 网格:** CGL 节点 `h_s_cgl` 布在 $s \in [0, L_y]$(原 `y` 的位置)
2. **Map $y(s)$:** 按 `coord_map` 构造
   - `IDENTITY`:$y = s$
   - `TANH`:$y(s) = \frac{L_y}{2}\!\left(1 + \tanh(\beta\xi)/\tanh(\beta)\right)$, $\xi = 2s/L_y - 1$
   - `LOGRHO`:$y(s)$ 由反查 $s = -\ln(\rho_0/\rho_c) \cdot L_y/u_{\max}$ 得到
3. **$y'(s)$:** 要么闭式(TANH),要么 $D_s \cdot y(s)$ 数值微分(LOGRHO)
4. **$D$ 重缩:** 先建 `D_scaled` 在 $s$ 上(标准 Chebyshev),然后每行
   乘 $1/y'(s_i)$ → 现在 $D$ 作用在 $y$ 空间值上 $= d/dy$
5. **CC 权重:** 物理积分是 $\int f(y)\,dy = \int f(y(s))\,y'(s)\,ds$,所以
   `h_cc_weights[k] = w_s[k] · Ly/2 · y'(s_k)`。**漏这一步 SL 正交性崩,
   Boussinesq manufactured test 能给出 15% 错误(我第一版就漏了,调试时发现)**

## 1.2 API

- `enum class CoordMap { IDENTITY, TANH, LOGRHO }`
- 成员 `coord_map`, `coord_beta`, `h_s_cgl`, `h_dy_ds`
- Env vars:
  - `ANSL_COORD_MAP=identity|tanh|logrho`
  - `ANSL_COORD_BETA`(仅 TANH,默认 2.0)

下游代码(SL 本征求解、Poisson pipeline、W̃ 计算、时域)**完全不感知**
坐标变换,因为 $D$ 已经被重缩成物理 $d/dy$ 算子,其他地方都以 $y$ 值
和 $y$-空间权重工作。

# 2. 调试过程中发现的 bugs

## 2.1 TANH $y'$ 缺 $\beta$ 因子

$$y'(s) = \frac{dy}{d\xi}\frac{d\xi}{ds} = \frac{L_y}{2}\cdot\frac{\beta\,\text{sech}^2(\beta\xi)}{\tanh\beta}\cdot\frac{2}{L_y}
     = \beta \cdot \frac{\text{sech}^2(\beta\xi)}{\tanh\beta}$$

我第一版写了 `sech²/tanh(β)`,漏了 $\beta$ 因子。表现:极限 $\beta\to 0$
本应还原 identity(aspect=1),但 μ_0 错了 8 个数量级。修复后 identity 极限
干净回到机器精度。

## 2.2 CC 权重漏 $y'$ 因子

$$\int_0^{L_y} f(y)\,g(y)\,dy = \int_0^{L_y} f(y(s))\,g(y(s))\,y'(s)\,ds
                              \approx \sum_i w^{(s)}_i\,y'(s_i)\,f(y_i)\,g(y_i)$$

所以 y-空间的 CC 权重必须是 `w_s · y'(s)`。漏这一步,SL 本征函数在 y-空间
不再正交,Poisson 解给出 15% L2 误差(β=2,Boussinesq)。修复后该 case
回到 2.3e-7(机器精度附近)。

# 3. 数值结果

## 3.1 Boussinesq 回归测试(manufactured test)

32×32,$L_x = L_y = 1$:

| map | aspect | $\mu_0$ | Poisson rel err |
|---|---|---|---|
| identity | 1.0 | 9.8696 | 1.01e-8 |
| TANH β=0.0001 | 1.0 | 9.8696 | 1.01e-8 |
| TANH β=1 | 2.4 | 9.8696 | 1.63e-8 |
| TANH β=2 | 14 | 9.8696 | 2.33e-7 |
| TANH β=4 | 720 | 9.87 | 2.86e-4 |
| LOGRHO | fallback → identity | 9.8696 | 1.01e-8 |

结论:**TANH 在 Boussinesq(ρ₀=1, W̃=0)下不破坏任何东西**,机器精度
保留直到 β=2(aspect 14)。β=4 时 aspect 720 → D² 条件数爆 → 开始退化。
LOGRHO 正确地 fall back 到 identity(ρ₀ 非单调-下降)。

## 3.2 Lane-Emden n=3/2 manufactured test

32×32,rho_cut=0.01:

| map | `|W̃|_max` | Poisson rel err_L2 |
|---|---|---|
| identity | **30.3** | 1.4e-3 |
| **TANH β=2** | 28.8 | **3.8e-4** (**3.6× better**) |
| LOGRHO (as-implemented) | **1.47e4** | **5.2e-3 (worse!)** |

**TANH 成功**:3.6× 精度提升,W̃ 几乎不变(纯几何 clustering 对 W̃ 的绝对值
不造作,只是让 CGL 节点在边界附近更密 → 更高分辨率 + 更好条件数)。

**LOGRHO 失败**(但失败的方式揭示了理论):实现只改了节点 placement,**没有
重做 reduced-pressure 代数**。W̃ 公式仍然是 $(\rho')^2/(4\rho^2) - \rho''/(2\rho)$
在 y-空间下的直接表达;当 LOGRHO 把节点推到 $\rho \to 0.01$ 的区域密集时,
**$1/\rho^2 \propto 10^4$ 的惩罚被更多的节点同时吃到**,数值上比 identity
更糟。

## 3.3 LOGRHO 节点分布诊断

ny=16,Lane-Emden,LOGRHO:

```
y nodes near 0:   [0.000, 0.126, 0.257, 0.379, 0.496, ...]
y nodes near Ly:  [..., 0.962, 0.979, 0.992, 0.998, 1.000]
Δy near 0:   [0.126, 0.132, 0.122]
Δy near Ly:  [0.0124, 0.00633, 0.00214]   ← 58× 密集
```

**节点 clustering 本身是正确的** —— 表面 58× 密集。但 W̃ 在最后一个 CGL 节点
上仍然是 -5778(而 identity 下只是 30),因为代数奇点 $1/\rho^2$ 没被削。

# 4. 理论解释 — 为什么 LOGRHO 没奏效

## 4.1 Reduced-pressure 代数是针对 y-坐标写的

原始推导(`docs/reduced_pressure_chebyshev.*`):令 $\pi = p/\rho_0$,原方程
$\nabla\!\cdot\!(1/\rho_0 \nabla p) = f$ 变换为
$$-\pi''(y) + \tilde W(y)\,\pi(y) + k_x^2\,\pi(y) = f'(y)$$
$\tilde W = (\rho_0')^2/(4\rho_0^2) - \rho_0''/(2\rho_0)$ 以 **y-空间导数**出现。

## 4.2 LOGRHO 真想做的事

如果把坐标直接切到 $s = -\ln \rho_0$,**重新在 $s$-coord 下推导 reduced-pressure 方程**,
Liouville 势能就变成
$$\tilde W_s(s) = \frac{(\rho_0'(s))^2}{4 \rho_0(s)^2} \cdot y'(s)^2 - \frac{\rho_0''(s)}{2\rho_0(s)} \cdot y'(s)^2 + (\text{metric terms from chain rule})$$

对 Lane-Emden $\rho \sim \exp(-s)$,$\rho'/\rho = -1$ 常数,$\tilde W_s \sim O(1)$
在整个域上 —— **真的有界**。

## 4.3 为什么我当前实现只做了一半

我的代码:
- 节点布在 $s$:✓
- $D_s$ 重缩成 $d/dy$:✓
- 用这个 $d/dy$ 计算 $\rho'(y) = D\,\rho$:✓(就是原公式)
- 把这个 $\rho'(y), \rho''(y)$ 代进 **y-空间的** $\tilde W$ 公式:✓
- 但 **$\tilde W$ 公式本身是 y-空间的** —— $1/\rho^2$ 在 $\rho \to 0$ 仍然爆

要真正 regularize 需要**完整重新推导 $s$-coord 下的 SL 谱框架**:
- 新坐标下的 $d/ds$ 为基础算子(不是物理 $d/dy$ 的 chain rule)
- 新的 $\tilde W_s$ 公式
- 新的测试函数 $\pi(s)$ 而非 $\pi(y)$
- manufactured test 也要在 $s$-coord 下构造

这是 1-2 天的 Phase 2 重构,不在本次 commit 范围。

# 5. 使用建议

**当前代码里 TANH 是推荐的 stretch**。对 Lane-Emden 类表面奇异问题:

```bash
ANSL_COORD_MAP=tanh ANSL_COORD_BETA=2 ./build/stellar2d \
    --solver anelastic_sl --test sl_poisson_test \
    --ntheta 64 --nr 64 --ps-Lx 1 --ps-Ly 1
```

预期:Poisson L2 误差降低 ~3.6×。不改变 Boussinesq 行为(ρ₀=1 下 $\tilde W \equiv 0$,
TANH 纯几何)。

`LOGRHO` 保留但**标记为实验性** —— 当前实现只是节点 clustering,不 regularize
$\tilde W$;要真正工作需要 $s$-coord 重构代数。

# 6. 代码变更

- `src/gpu/anelastic_sl_solver.cuh`:
  - `enum class CoordMap { IDENTITY, TANH, LOGRHO }`
  - 成员 `coord_map`, `coord_beta`, `h_s_cgl`, `h_dy_ds`

- `src/gpu/anelastic_sl_solver.cu::set_background`:
  - 拆分原本合在一起的 CGL + ρ 采样 + D scale 步骤
  - 插入 coord-map 生成 $y(s), y'(s)$
  - $D$ 按 $1/y'(s_i)$ 行缩放
  - CC 权重乘 $y'(s)$
  - LOGRHO post-pass(Lane-Emden only):取 identity-grid ρ 反查得到 y(s)
  - Boussinesq/stratified_n2 下 LOGRHO 会 fall back 到 identity 并警告

# 7. 后续

1. **s-coord 下完整重写 reduced-pressure 代数**(Phase 2 主战场,工程量 1-2 天)
2. **TANH clustering + SL filter 联合测试**:目前 filter 实验都在 Boussinesq 做,
   换成 Lane-Emden + TANH + SL filter 看能否跑到谱精度
3. **时域 Lane-Emden g-mode 验证**:现在有 TANH 有 SL filter 有 EVP IC,
   可以做真实变密度 g-mode 的时域频率精度测试(对应 Phase 1e 文档
   §10 的 item 1)
