---
title: 2D 极坐标 anelastic SL 求解器 — 设计方案
author: Phase 3 kick-off(branch `anelastic-sl-spectral` 或新分支 `polar-anelastic`)
date: 2026-05-03
status: 设计中
---

# 1. 动机

当前 Cartesian box solver(`AnelasticSLSolver`)在 Path D 下变密度 Lane-Emden
g-mode 时域闭环到机器精度。但 Cartesian 几何**无法做恒星扇区 / annulus 几何** —
要做"恒星内部真实 2D 流体演化"必须上曲线坐标。

**2D 极坐标 $(r, \phi)$** 是最经济的一步:
- 对应恒星赤道截面 / 子午面二维剖面
- $\phi$ 周期 → Fourier 分解(和当前 $x$ 一样)
- $r$ 是径向 → Chebyshev 在 $[r_{\rm in}, R]$(和当前 $y$ 一样)
- **Path D 的矩阵结构直接移植**(per-$m$ 矩阵代替 per-$k_x$ 矩阵)
- 能对接 MESA profile(ρ(r), N²(r) 原生径向)

对标:MagIC(2D/3D annulus + spectral),ASH(full 3D),Rayleigh(3D)。
我们目标是 **2D 极坐标 CUDA GPU 实现 + Path D 机器精度线性闭环**。

# 2. 几何与域

## 2.1 域

$$(r, \phi) \in [r_{\rm in}, r_{\rm out}] \times [0, 2\pi)$$

**径向**:恒星中心不必包含(shell/annulus 几何)。
- $r_{\rm in} = 0.1 R$(或更小):避开几何奇点 $1/r$
- $r_{\rm out} = 0.999 R$:避开 $\rho \to 0$ 表面奇点

**方位**:完全周期,**自然 Fourier** $e^{im\phi}$, $m \in \mathbb{Z}$。

## 2.2 边界条件

径向内外壁:
- **Impermeable**:$u_r|_{r_{\rm in}} = u_r|_{r_{\rm out}} = 0$(Dirichlet on $u_r$)
- **Free-slip(stress-free)**:$u_\phi$ 不受限,切向自由
- **压力**:Neumann(已由动量方程隐含)

**BC 与 Path D 匹配**:和 Cartesian box 的 $v|_{\rm wall}=0$ 完全同构,
只是字母换成 $u_r$。

# 3. 控制方程

线性 anelastic(已足够 Phase 1 闭环,非线性后续增量):

$$
\begin{aligned}
\partial_t u_r &= -\frac{1}{\rho_0}\partial_r p + b_r \\
\partial_t u_\phi &= -\frac{1}{\rho_0 r}\partial_\phi p \\
\partial_t b &= -N^2(r)\, u_r \\
\frac{1}{r}\partial_r(r\rho_0 u_r) + \frac{\rho_0}{r}\partial_\phi u_\phi &= 0
\end{aligned}
$$

Reduced-pressure 变换 $\pi = p/\rho_0$:

$$
\begin{aligned}
\partial_t u_r &= -\partial_r \pi - \frac{\rho_0'}{\rho_0}\pi + b \\
\partial_t u_\phi &= -\frac{1}{r}\partial_\phi \pi \\
\partial_t b &= -N^2 u_r \\
\frac{1}{r}(r\rho_0 u_r)' + \frac{\rho_0}{r}\partial_\phi u_\phi &= 0
\end{aligned}
$$

(**$'$ 代表 $\partial_r$**)

## 3.1 Fourier-$\phi$ 分解

设 $u_r(r, \phi, t) = \hat U_r(r, t) e^{im\phi}$,其他字段同样。代入:

$$
\begin{aligned}
\partial_t \hat U_r &= -\partial_r \hat \pi - \frac{\rho_0'}{\rho_0}\hat \pi + \hat b \\
\partial_t \hat U_\phi &= -\frac{im}{r}\hat \pi \\
\partial_t \hat b &= -N^2 \hat U_r \\
\frac{1}{r}(r\rho_0 \hat U_r)' + \frac{i m \rho_0}{r}\hat U_\phi &= 0
\end{aligned}
$$

## 3.2 消元得 $\ddot U_r$ 方程(Path D 的基础)

从 $\partial_\phi$-动量:$\hat U_\phi = -\dfrac{m}{\omega r}\hat\pi$(在 $e^{-i\omega t}$ 假设下,$\partial_t \to -i\omega$;二阶推导中用 $\ddot{} = -\omega^2$)。

从连续性:$\hat U_\phi = \dfrac{i (r\rho_0 \hat U_r)'}{m\rho_0}$。

联立消 $\hat U_\phi$:$\hat\pi = \dfrac{\omega}{m^2}\cdot \dfrac{i(r\rho_0 \hat U_r)'}{\rho_0}\cdot(-r\omega)$... 需要仔细推:

从连续性 $\hat U_\phi = -\dfrac{(r\rho_0 \hat U_r)'}{im\rho_0}$(带入 $1/r$ 因子)。
从 φ-动量 $-i\omega\hat U_\phi = -\dfrac{im}{r}\hat\pi$,所以 $\hat\pi = \dfrac{\omega r}{m}\hat U_\phi = -\dfrac{\omega r}{m}\cdot\dfrac{(r\rho_0\hat U_r)'}{im\rho_0} = \dfrac{i\omega r (r\rho_0\hat U_r)'}{m^2 \rho_0}$。

代入 $r$-动量(时间简谐,b = $-iN^2\hat U_r/\omega$):

$$
-\omega^2\hat U_r = \omega^2\hat U_r\cdot\frac{N^2}{\omega^2} - \partial_r\!\left[\frac{i\omega r (r\rho_0\hat U_r)'}{m^2\rho_0}\right]\cdot i - \frac{\rho_0'}{\rho_0}\cdot(i\omega)\cdot\ldots
$$

(完整推导放到实现前的 self-check Python script,此处仅展示结构。)

**最终形式(结构保证,系数待验证)**:

$$
\boxed{\; L_m \ddot{\hat U}_r = -R_m \hat U_r, \;}
$$

其中
- $L_m = -\dfrac{1}{m^2}\partial_r\!\left[\dfrac{r}{\rho_0}\partial_r(r\rho_0 \cdot)\right] + I$(双 $r$ 算子,**线性代数严格自伴随**)
- $R_m = N^2(r) \cdot I$

**这就是 Cartesian 版 $L = -D\rho D + k^2\rho$ 的极坐标同构**。只是:
- $k_x^2 \to m^2$
- $D\rho D \to \dfrac{1}{r}\partial_r(r\rho_0\cdot)\cdot\dfrac{1}{\rho_0}\partial_r(\cdot)\cdot$ 的组合(因 divergence in polar 有 $1/r$ 几何因子)

**Path D 直接套**:per-$m$ assemble 一个 $n_r \times n_r$ 的 $M_m = L_m^{-1}R_m$。

# 4. 离散化

## 4.1 径向基

选择 **Chebyshev CGL on $[r_{\rm in}, r_{\rm out}]$**:
- 现有基础设施:`anelastic_sl_solver.cu` 的 `cheb_D1_interval`、`cc_weights` 直接复用
- TANH coord-map 直接复用(已在 `set_background` 内,切换到 r-grid 只要把 `y_asc` 改成 `r_asc`)

避免 $r=0$ 几何奇点:**内边界 $r_{\rm in} > 0$**,做 shell/annulus。对全盘需要 Zernike 或 Jacobi $(1-r^2)^{|m|/2}$,那是 Phase 2+ 工作。

## 4.2 方位基

cuFFT R2C in $\phi$(batch size $n_r$)。**直接继承 Cartesian 实现**,把 `nx` 重命名为 `n_phi`,`Lx` 重命名为 $2\pi$。$k_\phi$ 是整数 $m$,物理波数 = $m/r$(注意这是**空间变化**,但 per-mode EVP 里 $m$ 是常数)。

## 4.3 Path D 矩阵 assemble

对每个 $m \in \{0, 1, \ldots, n_h-1\}$:

1. 构造 $L_m = -\dfrac{1}{m^2}\hat L + I$,其中 $\hat L$ 是复合算子 $\partial_r [r/\rho_0 \partial_r (r\rho_0\cdot)]$ 的 row-major 矩阵(用 $D$, diag($r$), diag($\rho_0$), diag($1/\rho_0$)复合)
2. 构造 $R_m = \operatorname{diag}(N^2)$(与 $m$ 无关,可预存一次)
3. Gauss-Jordan inversion → $M_m = L_m^{-1}R_m$
4. Interior slice to $(n_r-2) \times (n_r-2)$ (Dirichlet $\hat U_r|_{\rm wall}=0$)
5. 上传 device

**$m = 0$ 特殊处理**:$L_0 = \infty \cdot \hat L + I$ 退化(对应轴对称模)。物理上 $m=0$ 对应径向脉动(p-mode 或纯浮力),$N^2 = 0$ 时无振荡。**写 $M_0 = 0$**,等价于 $m=0$ 不演化(初始零就保持零)。

## 4.4 时间格式

**RK4**,与 Cartesian Path D 完全一致,直接复用 `step_assembled_linear` 的架构:
- 状态 $(U_r, W_r = \partial_t U_r)$ 在物理空间
- 每 RK4 substep:FFT_φ → apply $-M_m$ per m → IFFT_φ → 累加

# 5. CUDA 实现规划

## 5.1 新文件

**不要**直接改 `anelastic_sl_solver`(项目规矩:保留阶段性资产)。开新的:

```
src/gpu/polar_anelastic_solver.cuh
src/gpu/polar_anelastic_solver.cu
src/gpu/polar_anelastic_kernels.cu
```

## 5.2 结构骨架

```cpp
struct PolarAnelasticSolver {
    // Grid
    int n_r = 0, n_phi = 0, n_h = 0;
    double r_in, r_out;

    // Background (radial)
    std::vector<double> h_r_cgl, h_rho, h_rho_prime, h_N2;
    std::vector<double> h_Dr_row;  // Chebyshev D on [r_in, r_out]

    // Device buffers (mirror of AnelasticSLSolver)
    double* d_ur = nullptr;       // u_r(r, phi) row-major
    double* d_uphi = nullptr;
    double* d_b = nullptr;
    double* d_w_r = nullptr;      // ∂_t u_r (Path D state)
    // ... plus RK4 scratch

    // Path D assembled operators
    double* d_M_per_m = nullptr;  // (n_h * n_int²)
    int n_int = 0;

    // cuFFT plans (R2C in phi, batch n_r)
    cufftHandle plan_r2c_phi, plan_c2r_phi;

    void init(int n_r, int n_phi, double r_in, double r_out);
    void set_background(const StellarProfile& prof);  // or "lane_emden_n3"
    void init_gmode_eigenmode(int m_mode, int n_g, double amp);
    double step_assembled_linear();     // Path D RK4
    // ...
};
```

## 5.3 关键复用

- **`StellarProfile`**(已有):MESA/GYRE profile 输入
- **`lane_emden_solve`**(需参数化 n):背景 ρ/N² 测试用
- **Chebyshev D + CC weights + TANH coord-map**:从 `anelastic_sl_solver.cu` 抽取成 static helpers
- **cuFFT R2C 1D**:同 Cartesian
- **Path D 的 `k_apply_M_kx` kernel**:重命名 `k_apply_M_m`,逻辑完全一致
- **RK4 时间格式代码**:`step_assembled_linear` 直接复制

## 5.4 **不需要**的基础设施(这阶段)

- 球谐 `SphTransform`:Phase 3 才需要(上到真正球壳)
- 非线性 advection:Phase 4 增量
- ARS/IMEX 时间格式:显式 RK4 已闭环

# 6. 验证计划

## 6.1 空间离散 sanity

- Boussinesq($\rho$=const, $N^2$=const)在 annulus 上有解析 g-mode(参考 Bessel 函数),EVP 对比到机器精度
- Lane-Emden n=3(整数 σ)ny 扫描显示**指数收敛**
- Lane-Emden n=3/2 应看到 $N^{-2}$ 代数收敛(和 Cartesian 结果一致)

## 6.2 Path D 闭环

同 Cartesian 的 dev/step 测试:
- 线性算子一致性:dev/step ≤ 1e-14(机器精度)
- 长时 FFT:主峰 rel_err ≤ 1e-4
- 变密度 Lane-Emden vs Boussinesq 都机器精度

## 6.3 GYRE cross-check

- 取 GYRE poly3.txt,在极坐标 annulus 中 $m$ 取整数(球谐 $\ell \to$ 2D 极坐标 $m$ 的映射是**近似**的,严格对不到 — 这是 2D 极坐标 vs 3D 球壳的区别)
- 做 Exp K 风格的本征频率比较(不是逐模 matchmaking,是 spectrum 量级级匹配)

# 7. 工程量估算

| 任务 | 天 | 风险 |
|---|---|---|
| Python 原型:2D 极坐标 EVP + Path D TD(annulus + Bessel 回归) | 1 | 低(照着 `full_galerkin_closure_test.py` 改) |
| 新文件骨架 + CMakeLists.txt 条目 | 0.5 | 低 |
| `set_background`:Lane-Emden n(任意)→ r/ρ/N² | 0.5 | 低(复用 `stellar_profile.cpp`) |
| Path D assemble operators 极坐标版本 | 1 | **中**(几何因子 $1/r$ 和 $r\rho_0$ 的离散形式要验证) |
| `step_assembled_linear` 复制 + 适配 | 0.5 | 低 |
| init_gmode_eigenmode 极坐标版本 | 0.5 | 低 |
| 验证:dev/step、FFT、ny 扫描 | 1 | 低 |
| Bessel 解析对比 + Lane-Emden n=3 收敛阶 | 0.5 | 低 |
| GYRE 对比 | 0.5 | 中(物理 mismatch 解释) |

**总计:5-6 天**

风险主要在 §3.2 代数推导 + 几何因子离散化(§5.3 的 `assemble`),这两步需要
Python 原型先验证代数完全正确,然后 CUDA 移植是复制粘贴。

# 8. 里程碑

**M1**(2 天):Python 2D 极坐标 Path D 原型闭环 to machine precision on
Lane-Emden(任意 n)。

**M2**(3-4 天):CUDA 2D 极坐标 solver 通过 Boussinesq 回归 + Lane-Emden n=3
收敛阶测试。

**M3**(5-6 天):GYRE 频率对比文档,PolarAnelasticSolver 落库。

**M4+**(非线性):加 advection → 对流模拟,与 MagIC / ASH 对标。

# 9. 替代方案考虑(均否决)

- **3D Cartesian 双周期水平**(2-3 天)—— 物理价值高但仍然不是恒星几何
- **球壳 2D axisymmetric(r, θ)**(6-8 天)—— 正确但 $\sin\theta \to 0$ 极轴奇点需要 associated Legendre,工程量是极坐标 annulus 的 1.5 倍
- **全 3D 球壳 (r, θ, φ)**(10-14 天)—— 终极目标,分阶段走,先 2D 极坐标验证架构

# 10. 开始前的 checklist

- [ ] 代数推导的 Python self-check(3.2 节结构)
- [ ] Bessel 函数解析 g-mode 公式(Boussinesq annulus benchmark)
- [ ] `lane_emden_solve` 参数化 n(目前只 n=3/2)
- [ ] 选择内外边界 $r_{\rm in}, r_{\rm out}$ 默认值
- [ ] 确定 m=0 模式处理方式(当前文档给 $M_0 = 0$,可能需要单独分支做 axisymmetric pulsation)

# 11. 小结

**2D 极坐标 = Cartesian box 的直接同构重映射**:
- $(x, y) \to (\phi, r)$
- $k_x \to m$
- $v \to u_r$
- Dirichlet box → impermeable annulus

**Path D 数学保留**,唯一新工作是 **几何因子的正确离散**(div/grad 在极坐标有 $1/r$, $r$),这是 5-6 天工程量的核心。其他都是复用 + 移植。

设计评审过后,建议开新分支 `polar-anelastic` 开始实施。
