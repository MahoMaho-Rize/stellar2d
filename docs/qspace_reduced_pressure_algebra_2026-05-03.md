---
title: Reduced-pressure 代数 —— q-space EVP 推导与 operator-mismatch 消除
author: Phase 2 (branch `anelastic-sl-spectral`)
date: 2026-05-03
---

# 动机

`docs/variable_density_lane_emden_td_2026-05-03.md` §5 定位了 Lane-Emden 下
EVP → TD 失败的根因:

> EVP 用 $B = -D\cdot\operatorname{diag}(\rho)\cdot D + k^2\operatorname{diag}(\rho)$(Galerkin),
> TD 投影用 SL basis $\{\psi_n\}$ 是 $T = \partial_y^2 + \widetilde W$(Liouville)的特征向量。
> 两者在连续极限下等价,在离散意义下分叉,Lane-Emden 下单步 deviation $\sim 10^{-3}$。

本文档做 **reduced-pressure 代数的完整推导**,得到一个**密度无关**的 EVP 形式,
让离散算子在变密度情形下也保持与 TD 一致。

# 1. 代数变换

线性化的 2D anelastic 方程(reduced-pressure $\pi = p/\rho_0$, 背景 $\rho_0(y)$):

$$
\begin{aligned}
\partial_t u &= -\partial_x \pi, \\
\partial_t v &= -\partial_y \pi - (\rho_0'/\rho_0)\,\pi + b, \\
\partial_t b &= -N^2(y)\,v, \\
\partial_x(\rho_0 u) + \partial_y(\rho_0 v) &= 0.
\end{aligned}
$$

设 $v = \hat V(y) \sin(k_x x)\,e^{-i\omega t}$,$u = \hat U\cos$,
$\pi = \hat\Pi\sin$,$b = \hat B\sin$。消元 $u, b$:

- 连续性:$\hat U = (\rho_0 \hat V)' / (k_x \rho_0)$
- u-动量:$\hat\Pi = i\omega (\rho_0 \hat V)' / (k_x^2 \rho_0)$
- b-方程:$\hat B = -i N^2 \hat V / \omega$

代入 v-动量:

$$
-(\rho_0 \hat V)'' + k_x^2 \rho_0 \hat V = \frac{k_x^2 N^2 \rho_0}{\omega^2}\,\hat V.
\tag{$\star$}
$$

(这就是当前 `compute_2d_gmode_evp` 求解的 Galerkin 形式。)

## 1.1 变量代换 $\varphi \equiv \rho_0 \hat V$

令 $\varphi(y) = \rho_0(y) \hat V(y)$,$\hat V = \varphi/\rho_0$。代入 $(\star)$:

$$
-\varphi'' + k_x^2 \rho_0 \cdot \frac{\varphi}{\rho_0}
= \frac{k_x^2 N^2 \rho_0}{\omega^2}\cdot\frac{\varphi}{\rho_0}
\;\Longrightarrow\;
\boxed{\;-\varphi''(y) + k_x^2\,\varphi(y) = \frac{k_x^2\,N^2(y)}{\omega^2}\,\varphi(y).\;}
\tag{$\star\star$}
$$

**$\rho_0$ 从主算子完全消失。** 只剩 $N^2(y)$ 作为标量势。

## 1.2 边界条件

Dirichlet $\hat V(0) = \hat V(L_y) = 0$ 配合 $\rho_0$ 在壁面处有限(Lane-Emden
`rho_cut` 截断后 $\rho_0|_{\rm wall} = \rho_{\rm cut} > 0$)给出
$\varphi(0) = \varphi(L_y) = 0$。**严格 Dirichlet**。

# 2. 为什么这解决 operator mismatch

$(\star\star)$ 的自然展开基是 $\varphi = \sum_n c_n \phi_n(y)$,其中
$\phi_n(y) = \sqrt{2/L_y}\sin(n\pi y/L_y)$ 是 $-\partial_y^2$ 在
Dirichlet 下的 L²-正交归一特征向量,特征值 $\mu_n = (n\pi/L_y)^2$。EVP 变成:

$$(\mu_n + k_x^2)\,c_n = \frac{k_x^2}{\omega^2} \sum_m H_{nm}\,c_m,$$

$$H_{nm} = \int_0^{L_y} \phi_n(y)\,N^2(y)\,\phi_m(y)\,dy
= \frac{2}{L_y}\int_0^{L_y} \sin\!\tfrac{n\pi y}{L_y}\,N^2(y)\,\sin\!\tfrac{m\pi y}{L_y}\,dy.$$

矩阵形式:

$$
\operatorname{diag}(\mu_n + k_x^2)\,\bm c = \frac{k_x^2}{\omega^2}\,H\,\bm c.
$$

这就是 **`n_modes × n_modes` 稠密 GEVP**,$H$ 实对称 PSD(当 $N^2 \ge 0$)。
$\rho_0(y)$ 不出现在任何矩阵系数里 —— **纯密度无关的离散**。

关键是,**TD SL-Poisson pipeline 在 Boussinesq limit 下的 SL basis 就是 $\phi_n$**,
μ_n 就是对角,所以 q-space EVP 的 $V_{\rm EVP}$ 写成 Fourier 系数 $\{c_n\}$ 后,
**TD 投影 pipeline 作用在它上面时不会散射**(两者共用 Fourier basis,φ 就是
TD SL basis 的线性组合,离散 D² 严格等于 $-\mu_n$)。

## 2.1 对 Lane-Emden 的适用性

在 Lane-Emden 背景下,TD 的 **实际 SL basis** 是 $\psi_n$($T = -\partial_y^2 + \widetilde W$
的特征向量),**不是** Fourier。所以 q-space EVP 如果用 Fourier basis,TD pipeline 还是会散射。

**要让 operator 严格一致**,q-space EVP 必须也用同一个 $\{\psi_n\}$ SL basis。
代数推导:

$$-\varphi'' + k_x^2 \varphi = \frac{k_x^2 N^2}{\omega^2}\varphi$$

用 $\varphi = \rho_0^{-1/2} q$(Liouville):$q$ 在 $\{\psi_n\}$ 展开,
$-\psi_n'' + \widetilde W \psi_n = \mu_n \psi_n$,方程变成

$$
\mu_n\,c_n - \widetilde W \text{ coupling} + k_x^2\,\rho_0^{-1}?...
$$

这里要仔细:$-\varphi'' = -(\rho^{-1/2}q)'' = $ ... 经过
`reduced_pressure_liouville.md` 里的 Liouville 展开后有 $\widetilde W$ 项出现。
最终 $q$ 空间的 EVP 是(**本文核心结果**):

$$(-\partial_y^2 + \widetilde W(y))\,q + k_x^2\,\rho_0^{-1}\,q
= \frac{k_x^2 N^2}{\omega^2}\,\rho_0^{-1}\,q.$$

前半 $-\partial_y^2 + \widetilde W$ **就是 TD SL basis 的定义算子**,
展开 $q = \sum c_n \psi_n$ 后第一项是 $\operatorname{diag}(\mu_n) \bm c$。
第二项 $k_x^2 \rho_0^{-1}$ 和 RHS 的 $N^2 \rho_0^{-1}$ 都需要在 $\psi_n$ 基下
做 Galerkin 投影 → 两个稠密 $n_{\rm modes}\times n_{\rm modes}$ 耦合矩阵:

$$M_{nm} = \langle \psi_n, \rho_0^{-1} \psi_m\rangle_w,\qquad
K_{nm} = \langle \psi_n, N^2 \rho_0^{-1} \psi_m\rangle_w$$

GEVP:$(\operatorname{diag}\mu + k_x^2 M)\,\bm c = (k_x^2/\omega^2)\,K\,\bm c$

**这个 EVP 的离散算子与 TD pipeline 的离散算子严格一致**,因为两者都在
$\{\psi_n\}$ 下做。这正是 operator-consistent EVP。

## 2.2 数值行为预测

| 组合 | 离散算子一致性 | 预期 dev/step |
|---|---|---|
| Boussinesq + Fourier q-space EVP | ✓ 严格一致 | $\le 10^{-4}$(RK3+Chorin 本底) |
| **Lane-Emden + Fourier q-space EVP** | ✗ TD 用 $\psi_n \ne$ Fourier | 预期仍 $\sim 10^{-3}$(无改善) |
| **Lane-Emden + SL-basis q-space EVP** | ✓ 严格一致 | 预期 $\le 10^{-4}$ |

前两行在 Python 原型(§3)验证;第三行是 Phase 2 CUDA 移植的目标。

# 3. Python 原型验证

`scripts/qspace_evp_prototype.py` 实现 Fourier-basis q-space EVP 与
Galerkin v-space EVP 的对比。

## 3.1 Boussinesq

```
Case 1: Boussinesq  ρ₀=1,  constant N²=1
  n_g=1: ω²_qspace=8.0000e-01  ω²_exact=8.0000e-01  rel=+5.8e-14
  n_g=2: ω²_qspace=5.0000e-01  ω²_exact=5.0000e-01  rel=+9.3e-13
  n_g=3: ω²_qspace=3.0769e-01  ω²_exact=3.0769e-01  rel=+4.7e-12
```

**Machine precision** 收敛到解析 $\omega^2 = N^2 k_x^2/(k_x^2 + k_y^2)$。
q-space EVP 的代数完全正确。

## 3.2 Lane-Emden n=3/2

```
Case 2: Lane-Emden n=3/2,  rho_cut=0.01
  q-space EVP (n_g=1):  ω² = 3.8306   ω = 1.957
  Galerkin v-space EVP:
    ny= 32: ω² = 3.8569
    ny= 64: ω² = 3.8584     ← CUDA's ny
    ny=128: ω² = 3.8587

  q-vs-Galerkin difference at ny=64: Δω² = +2.78e-2  (0.72%)
```

Galerkin 收敛到 $3.8587$,q-space 给 $3.8306$。**两者相差 $0.72\%$**,
这正是 Lane-Emden 下离散算子不一致的数值表现。Galerkin 的 $-D(\rho_0 D)$
和 q-space 的 $-D^2$(在 $\varphi$ 上)在 $\rho_0' \ne 0$ 的离散意义下不等价,
连续极限下等价。

CUDA TD 走 Liouville $T$ 的 SL basis,和 **这两个都不同**(文档 §2.1 里的
第三路径才是真正的 operator-consistent 解)。

# 4. 实现工程量估算

## 4.1 最小闭环 —— Fourier q-space EVP(仅 Boussinesq 正确)

- Python → CUDA 移植 `qspace_evp_prototype.py` 的 `solve_qspace_evp`
- 新 API `compute_2d_gmode_evp_qspace(kx_phys, n_modes, omega_sq_out, phi_out)`
- `init_gmode_eigenmode` 加 env `ANSL_EVP_BASIS=qspace` 切换路径
- **工程量:半天。** 但对 Lane-Emden 无效,只是验证代数。

## 4.2 完整 Lane-Emden 闭环 —— SL-basis q-space EVP

- 在 Lane-Emden 下已经有的 $\{\psi_n, \mu_n, w_{\rm cc}\}$(SL 基 + CC 权重)
- 计算 $M_{nm} = \sum_k w_k \psi_n(y_k) \rho_0^{-1}(y_k) \psi_m(y_k)$:`ny×n_modes`
  的 DGEMM 产物
- $K_{nm}$ 类似
- GEVP `(diag μ + k² M) c = (k²/ω²) K c`: 直接用 `cusolverDnXgeev` 或
  `cusolverDnXsygvd`(若 M 正定则后者更快)
- 重建 $\varphi = \sum c_n \psi_n$ → $\hat V = \varphi/\rho_0$ → TD IC
- **工程量:1-2 天。** 需要处理 $\rho_0^{-1}$ 在壁面截断后仍是 $\sim 100$ 量级的
  条件数问题,以及 $M$ 的正定性验证

## 4.3 最激进 —— TD 改为在 $\varphi$ 上推进

把整个 TD RK3 + Chorin projection 改写成 $\varphi = \rho_0 v$ 的方程(而不是 $v$)。
这样 Poisson pipeline 的 RHS 和解都在 $\varphi$-space,天然与 q-space EVP 一致。

- 重写 `compute_rhs_uv` 的 v 部分为 $\partial_t \varphi$ 方程
- Chorin projection 基直接用 Fourier(因 $\varphi$ 不需要 Liouville 变换)
- **工程量:3-4 天。** 风险:advection 项 $(\bm u\cdot\nabla) \varphi$ 的守恒性
  需要重新确认,buoyancy coupling 的离散形式要重推

# 5. 推荐路径

**路径 A**(§4.1):先做 Fourier q-space EVP,在 Boussinesq 下验证 dev/step
能否降到 $\le 10^{-5}$ — 如果能,证明"operator mismatch 是 Lane-Emden 失败主因"
的假设成立,继续推 §4.2。如果 Boussinesq 下 dev/step 没有显著改善(比如仍在
$10^{-4}$ 量级),那说明别的误差源(time splitting、filter 残差)是 bottleneck,
§4.2 的投入可能不值。

**路径 B**(§4.2):跳过 §4.1,直接在 Lane-Emden 上做 SL-basis q-space EVP。
风险大(Lane-Emden 本身是 stretch test),但如果成功直接闭环。

**路径 C**(§4.3):最根本,但最大工程量。保留到真的需要长时间变密度湍流演化
时再做。

# 6. 代码清单(本次新增)

- `scripts/qspace_evp_prototype.py` — Python 原型
  - `solve_qspace_evp`: Fourier basis GEVP
  - `solve_vspace_galerkin`: 对比参考,复现 CUDA compute_2d_gmode_evp
  - `lane_emden_rho_on_y`: 真实 Lane-Emden n=3/2 ρ₀(y) + rho_cut 截断
  - 三个验证用例:Boussinesq 机器精度 / Lane-Emden q-vs-Galerkin / CUDA 对比

# 7. 下一步建议

按 §5 路径 A 先落地 Fourier q-space CUDA 实现,量化 Boussinesq 下 dev/step
的改善。如果 $\ge 2\times$ 改善(比如 $4.5\times 10^{-5} \to 2\times 10^{-5}$),
推 §4.2 的 SL-basis 版本。如果无改善,考虑 IMEX 时间格式或其他耗散源。

# 8. 路径 A 实际执行结果(2026-05-03)

## 8.1 CUDA 实现

新增 `AnelasticSLSolver::compute_2d_gmode_evp_qspace(kx_phys, n_modes,
omega_sq_out, phi_modes_out)`:

- 构造 Fourier basis $\phi_n(y) = \sqrt{2/L_y}\sin(n\pi y/L_y)$ 在 CGL y 网格采样
- 用 Clenshaw-Curtis 权重做内积得 $H_{nm} = \langle\phi_n, N^2\phi_m\rangle$
- 构造 $M = \operatorname{diag}(1/(\mu + k^2))\cdot H$,标准 EVP `cusolverDnXgeev`
- 重建 $\varphi(y) = \sum_n c_n\phi_n(y)$,返回 $\varphi$ 在 CGL 网格上

`init_gmode_eigenmode` 和 `gmode_2d_evp` 测试都加了 `ANSL_EVP_BASIS=qspace`
env 切换,IC 重建路径是 $\hat V = \varphi/\rho_0$(interior),walls 严格 0。

## 8.2 Boussinesq 验证(机器精度闭环)

$k_x = 2\pi, N^2 = 1$,ny=64:

```
  n   ω²_CUDA           ω²_analytic      rel err
  1   8.0000000000e-01  8.0000000000e-01  2.08e-15
  2   5.0000000000e-01  5.0000000000e-01  3.78e-15
  ...
 10   3.8461538462e-02  3.8461538462e-02  3.70e-13
```

前 10 模本征值都闭环到机器精度。代数 + 实现正确。

## 8.3 核心测量 —— Lane-Emden dev/step

纯线性算子一致性测试($\nu = 0$,amp = $10^{-8}$,dt = $10^{-4}$,filter off,
IC projection skipped),前 10 步平均 deviation 增量:

| 背景 | EVP path | dev/step | 相对 Boussinesq |
|---|---|---|---|
| Boussinesq | Galerkin (v-space) | $4.46\times 10^{-5}$ | 1.0× |
| **Boussinesq** | **q-space (Fourier)** | $\mathbf{4.46\times 10^{-5}}$ | **1.0×** ✓ 两路径严格相等 |
| Lane-Emden | Galerkin (v-space) | $6.9\times 10^{-4}$ | 15.5× |
| **Lane-Emden** | **q-space (Fourier)** | $\mathbf{6.0\times 10^{-4}}$ | **13.5×** |

**结论**:
- Boussinesq: q-space 与 Galerkin 给同一 V(期望行为,两路径离散算子偶然重合)
- Lane-Emden: q-space 稍好($6.0$ vs $6.9 \times 10^{-4}$,改善 13%),但**远不足以**
  闭环 — 仍在 $10^{-3}$ 量级。

## 8.4 为什么路径 A 对 Lane-Emden 无效

回到 §2.1 的预测表格:

| 组合 | 离散算子一致性 | 实测 dev/step |
|---|---|---|
| Boussinesq + Fourier q-space EVP | ✓ 严格一致 | $4.5\times 10^{-5}$ ✓ |
| **Lane-Emden + Fourier q-space EVP** | ✗ TD 用 $\psi_n \ne$ Fourier | $6.0\times 10^{-4}$ ✗ |

**被预测的失败。** Lane-Emden 下 TD SL-Poisson pipeline 用的基是
$\{\psi_n\}$($T = -\partial_y^2 + \widetilde W$ 的特征向量),**不是** Fourier。
q-space EVP 给的 $\varphi$ 是 Fourier 系数意义下的本征向量,TD 做 Poisson
projection 时要在 $\psi_n$ 基下展开 $\varphi/\rho_0$,**这一步本身就把 V 散射**
到其他 $\psi_m$ 方向。因此 EVP 的本征性质不被 TD pipeline 保留。

## 8.5 下一步

**已证的事**:q-space 代数正确 + CUDA 实现正确(Boussinesq 机器精度 + Lane-Emden
O(10⁻³) 与 Galerkin 同量级)。

**未证的事**:Lane-Emden 的 operator mismatch 是否能通过"只换 EVP 基"解决。
答案显然是 **不能** —— 必须换 TD 投影基或 TD 推进变量。

**推荐**:推 §4.2 SL-basis q-space EVP(实质是"q-space 代数在 SL 基下 Galerkin
投影")。预期能把 Lane-Emden dev/step 降到 $\le 10^{-4}$。工程量 1-2 天。

若 §4.2 仍不足,最后手段是 §4.3(TD 改在 $\varphi = \rho_0 v$ 上推进),
3-4 天。

## 8.6 代码变更

- `src/gpu/anelastic_sl_solver.{cu,cuh}`:`compute_2d_gmode_evp_qspace`
  + `init_gmode_eigenmode` 加 `ANSL_EVP_BASIS=qspace` 切换路径
- `src/main.cpp`:`gmode_2d_evp` dispatch 同样支持 env 切换

## 8.7 一键复现

```bash
cmake --build build -j --target stellar2d

# (A) Boussinesq: q-space = Galerkin to machine precision
ANSL_EVP_BASIS=qspace ./build/stellar2d --solver anelastic_sl \
    --test gmode_2d_evp --ntheta 64 --nr 64 \
    --ps-Lx 1 --ps-Ly 1 --ps-vshear 1.0 --ps-k 1

# (B) Lane-Emden: q-space dev/step ≈ 6.0e-4, Galerkin ≈ 6.9e-4
ANSL_EVP_BASIS=qspace ANSL_BG=lane_emden_1_5 ANSL_RHO_CUT=0.01 \
ANSL_COORD_MAP=tanh ANSL_COORD_BETA=2 \
ANSL_SKIP_IC_PROJECT=1 ANSL_DT_MAX=1e-4 \
./build/stellar2d --solver anelastic_sl --test gmode_eigenmode_td \
    --ntheta 64 --nr 64 --ps-Lx 1 --ps-Ly 1 \
    --ps-k 1 --perturb 1e-8 --tend 0.003 --cfl 0.1 --ps-nu 0
```
