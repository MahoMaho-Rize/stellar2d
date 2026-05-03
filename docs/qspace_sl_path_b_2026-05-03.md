---
title: 路径 B (SL-basis q-space EVP) 实施 + 验证结果
author: Phase 2 (branch `anelastic-sl-spectral`)
date: 2026-05-03
---

# 目的

`docs/qspace_reduced_pressure_algebra_2026-05-03.md` §4.2 定义的路径 B:
把 reduced-pressure q-space EVP 的 Galerkin 投影从 Fourier 基换到 TD 真实用的
SL 基 $\{\psi_n\}$,期望 TD 线性算子的 eigenmode 被严格保留。

# 1. 代数推导

原方程(见 §2 of algebra doc):$-\varphi''(y) + k_x^2\varphi = (k_x^2 N^2(y)/\omega^2)\varphi$,
$\varphi = \rho_0 \hat V$,Dirichlet $\varphi|_{\rm wall} = 0$。

用 TD SL 基 $\{\psi_n\}$ 展开 $\varphi = \sum_n c_n \psi_n$。由 SL 定义
$-\psi_n'' = \mu_n \psi_n - \widetilde W \psi_n$,代入并对 $\psi_m$ 做 CC 内积:

$$
\boxed{\;(L - \widetilde W_{\rm mat})\,\bm c \;=\; \frac{k_x^2}{\omega^2}\,H\,\bm c\;}
$$

矩阵定义:
- $L = \operatorname{diag}(\mu_n + k_x^2)$(对角,$n_{\rm modes}$ 维)
- $\widetilde W_{nm} = \langle\psi_n, \widetilde W\psi_m\rangle_{\rm cc} = \sum_k w_k\,\psi_n(y_k)\,\widetilde W(y_k)\,\psi_m(y_k)$
- $H_{nm} = \langle\psi_n, N^2 \psi_m\rangle_{\rm cc}$

两边的离散算子(D, w_cc, ψ_n, μ_n)都**来自 `set_background` 存下的 `h_Psi/h_mu/h_cc_weights`**,
与 TD SL-Poisson pipeline 严格共用。

实现方式:M = (L - W̃_mat)⁻¹ · H,cusolverDnDgetrf/getrs 再 cusolverDnXgeev。
eigenvalue λ = ω²/k² → ω² = k² · λ。

# 2. 实现

`AnelasticSLSolver::compute_2d_gmode_evp_qspace_sl(kx_phys, n_modes, omega_sq, v_modes)`:
- 输入:已完成 `set_background` 的 solver
- 输出:ω² 降序排列;v_modes(ny, n_kept) 行主序,已除以 ρ₀(即 V̂ = φ/ρ₀),壁面 0
- 主要成本:两个 $n_{\rm modes}^2 \cdot n_y$ 的双循环组装 H / W̃_mat(微不足道,n_modes ≤ 32)
- 可用 env `ANSL_EVP_BASIS=qspace_sl` 在 `init_gmode_eigenmode` 和 `gmode_2d_evp` 中切换

EVP 总路径现有三个:

| `ANSL_EVP_BASIS=` | 方法 | 离散基 | Boussinesq 精度 |
|---|---|---|---|
| 不设 / `galerkin`/`v` | v-space Galerkin | $-D\rho D$ 作用在 v 上 | ω²≤1e-14 |
| `qspace` / `q` / `phi` | q-space Fourier (路径 A) | $-D^2$ 作用在 φ 上,展开 Fourier | ω²≤2e-15 |
| **`qspace_sl` / `sl`** | **q-space SL (路径 B)** | **$-D^2 + \widetilde W$,展开 ψ_n** | **ω²≤3e-14** |

# 3. Boussinesq 验证

$k_x = 2\pi, L=1, N^2 = 1$,ny=64:

```
n    ω²_CUDA            ω²_analytic        rel err
1    8.0000000000e-01   8.0000000000e-01   2.83e-14
2    5.0000000000e-01   5.0000000000e-01   2.20e-14
...
10   3.8461538462e-02   3.8461538462e-02   3.88e-14
```

**机器精度闭环。** Boussinesq 下 $\widetilde W \equiv 0$,$\psi_n = \sqrt{2/L}\sin(n\pi y/L)$,
$H_{nm} = N^2 \delta_{nm}$,EVP 退化为 $(\mu_n + k^2)c_n = (k^2/\omega^2)c_n$,
即 $\omega^2 = k^2 N^2/(\mu_n + k^2)$。CUDA 实际算出的就是这个。

# 4. Lane-Emden 验证

## 4.1 谱值 ω²

```
n_g   Galerkin (v)    qspace Fourier   qspace SL
1     3.858814e+00    3.830583e+00    3.888985e+00
2     1.963080e+00    1.930252e+00    1.998203e+00
3     1.157733e+00    1.141258e+00    1.174928e+00
```

三路径在 Lane-Emden 下相互漂移 $\sim 0.7\%$(vs Python 原型 $\sim 0.7\%$ 之差),
全部在 ny=64 离散化误差带内。

## 4.2 核心测量 —— dev/step

纯线性算子一致性测试(同 path A §8.3 设置,$\nu = 0$,amp = $10^{-8}$,
dt = $10^{-4}$,filter off,skip IC project):

| Background | EVP path | dev/step | rel to Boussinesq baseline |
|---|---|---|---|
| Boussinesq | Galerkin | $4.46 \times 10^{-5}$ | 1.0× |
| Boussinesq | q-space Fourier | $4.46 \times 10^{-5}$ | 1.0× |
| **Boussinesq** | **q-space SL** | $4.46 \times 10^{-5}$ | **1.0×** ✓ 预期 |
| Lane-Emden | Galerkin | $6.9 \times 10^{-4}$ | 15.5× |
| Lane-Emden | q-space Fourier | $6.0 \times 10^{-4}$ | 13.5× |
| **Lane-Emden** | **q-space SL** | $\mathbf{5.9 \times 10^{-4}}$ | **13.2×** |

**路径 B 实测只改善 4%**(6.9 → 5.9,同 qspace Fourier 的 13% 改善在误差内)。

**完全没有闭合到 Boussinesq 水平。**

## 4.3 为什么路径 B 失败(意外的经验事实)

路径 A 失败是预期的:Fourier 基和 TD SL 基在 Lane-Emden 下不同。
**路径 B 被设计成与 TD SL 基一致,但仍然不够。**

重新分析 TD 动力学里所有使用 y-方向离散算子 D 的点:

| 位置 | 算子 | 在 SL/ψ_n 基下"天然"吗? |
|---|---|---|
| `apply_dy` in `compute_rhs_uv`(u, v advection) | D | ✗ 不是 |
| `apply_dy` for ∂y b in advection | D | ✗ |
| `apply_dy` for ∂y π in projection | D | ✗ |
| `apply_dy` for ∂y v in divergence RHS | D | ✗ |
| `sl_poisson_solve` 里 Ψ^T, diag(1/(μ+k²)), Ψ | ψ_n ✓ | ✓ |

只有 SL-Poisson 这一步和 $\{\psi_n\}$ 算子严格一致。剩下所有 $\partial_y$ 都用
**物理空间 D**,它的离散本征向量是 Chebyshev 模,**不是** $\psi_n$。

EVP 保证:在**连续极限**下 $V_{\rm EVP}$ 是完整 TD 线性算子的特征向量。
EVP 本身无论用 Galerkin、Fourier q-space 还是 SL q-space,都是**同一个连续算子
的不同离散化**,ny→∞ 下三者收敛到同一 $\omega^2$。

但 TD 动力学是**多个离散算子拼接**:SL-Poisson(ψ_n)+ advection D(Chebyshev)+
buoyancy 的 y-逐点乘法。EVP 给的本征向量在离散意义下**不是每个算子的不变方向**,
每步都以 $O(1/n_y^2)$ 泄漏。

**根因定位**:
1. $\partial_y$ 算子在 `apply_dy` 里是 Chebyshev D,不是 SL-basis-consistent 的 $D_\psi$(如果存在)
2. $\rho_0'/\rho_0$ 在 buoyancy / continuity 里通过逐点乘法作用,不通过 $\psi_n$ 投影
3. EVP 换基只改变"$V_{\rm EVP}$ 用什么离散表示",不改变 TD 里这些 $O(1)$ 量级的操作

## 4.4 dev/step 的"不可降低"本底

Boussinesq $4.5 \times 10^{-5}$ 来自 RK3 + Chorin splitting 的耗散误差(时间一阶
splitting,但单 step 表现为 $\partial_y$ 投影的数值残差)。Lane-Emden 的
$\sim 6 \times 10^{-4}$ 是 $15 \times$ 本底。

**经验观察**:$6.9/6.0/5.9 \times 10^{-4}$ 的三路径几乎一致。这说明 Lane-Emden
dev/step 本底**不取决于 EVP 基选择** —— 是 TD 的 apply_dy / 连续性 / buoyancy 在
$\rho_0' \ne 0$ 下的**离散算子误差**决定的。

换言之,**EVP 给的 V 在离散 TD 线性算子下根本不是特征向量**,不管你用哪套基做 EVP。

# 5. 结论

## 5.1 已证的事实

1. 路径 A(Fourier)和路径 B(SL basis)各自是自洽、收敛的 EVP 离散化
2. Boussinesq 下三路径给同样的 V(机器精度),dev/step 相同
3. Lane-Emden 下三路径给不同的 V($\sim 0.7\%$ 差),dev/step 仍在同一量级 $6 \times 10^{-4}$

## 5.2 被否证的假设

**"换 EVP 基能让 Lane-Emden TD 闭环"** 不成立。TD 在 $\rho_0' \ne 0$ 时,$\partial_y$
的离散 Chebyshev D 与 $\psi_n$ 的协作关系**不**被 EVP 单方面决定。

## 5.3 剩下的路径

唯一能真正消除 Lane-Emden operator mismatch 的是:**让 TD 本身在 $\psi_n$ 基下推进**,
或者等价地,**让 TD 在 $\varphi = \rho_0 v$ 变量上推进**(§4.3 of algebra doc,路径 C)。

路径 C 工程量 3-4 天,涉及:
- `compute_rhs_uv` 改写为 $\partial_t \varphi$ 方程
- `project_div_free` 改为解 $\nabla\cdot\nabla\pi = \text{RHS}$(Fourier Poisson,
  因为 $\rho_0$ 消失 → 简化)
- advection $(\bm u\cdot\nabla)\varphi$ 离散守恒形式
- buoyancy coupling $b, N^2$ 在 $\varphi$ 空间的新离散形式

## 5.4 替代方案(不是闭环,是解决实际应用的路径)

如果目标是"跑 g-mode 时域",而不是"单 eigenmode 完美保真":

1. **走 Exp K 频率谱**(已验证 $3.6 \times 10^{-5}$ vs GYRE),对**静态结构**做分析
2. **Boussinesq + 伪密度轮廓**做时域,用 $N^2(y)$ profile 模拟真实恒星
3. **接受 $6 \times 10^{-4}$ 量级的 dev/step**,但加足够耗散(SL filter)让泄漏被耗散掉,
   换取频率**相对**准确($\pm 10\%$,不是 $\pm 0.25\%$)

# 6. 代码变更清单

- `src/gpu/anelastic_sl_solver.cuh`:API `compute_2d_gmode_evp_qspace_sl`
- `src/gpu/anelastic_sl_solver.cu`:
  - 实现 `compute_2d_gmode_evp_qspace_sl`(188 行)
  - `init_gmode_eigenmode`:三路径 switch(Galerkin / Fourier / SL qspace)
- `src/main.cpp`:`gmode_2d_evp` dispatch 同样三路径

# 7. 一键复现

```bash
cmake --build build -j --target stellar2d

# Boussinesq qspace-SL: machine precision (rel_err ~1e-14)
ANSL_EVP_BASIS=qspace_sl ./build/stellar2d --solver anelastic_sl \
    --test gmode_2d_evp --ntheta 64 --nr 64 \
    --ps-Lx 1 --ps-Ly 1 --ps-vshear 1.0 --ps-k 1

# Lane-Emden qspace-SL TD dev/step: still ~6e-4 (no improvement vs Galerkin)
ANSL_EVP_BASIS=qspace_sl ANSL_BG=lane_emden_1_5 ANSL_RHO_CUT=0.01 \
ANSL_COORD_MAP=tanh ANSL_COORD_BETA=2 \
ANSL_SKIP_IC_PROJECT=1 ANSL_DT_MAX=1e-4 \
./build/stellar2d --solver anelastic_sl --test gmode_eigenmode_td \
    --ntheta 64 --nr 64 --ps-Lx 1 --ps-Ly 1 \
    --ps-k 1 --perturb 1e-8 --tend 0.003 --cfl 0.1 --ps-nu 0
```
