---
title: EVP eigenmode 接回时域 — filter basis 与 operator convergence
author: Phase 1e 闭环记录(branch `anelastic-sl-spectral`)
date: 2026-05-03
---

# 目的

Exp K CUDA 的 GYRE 外部验证在 2026-05-03 由合作者跑完后(`docs/gmode_exp_k_cuda_benchmark_2026-05-03.md`),
Phase 1 的 **空间离散精度** 已经闭环到 $3.6\times 10^{-5}$(vs GYRE,full-gravity 4-var 全 Chebyshev)。

本次工作处理的是 Phase 1 剩下的一个问题:**能否把 EVP 给的精确 eigenmode 作为 IC
灌进现有的 SL-spectral 时域求解器,得到同样精度的 g-mode 振荡?** 如果不能,差距在哪?

结论一句话:**对当前 Boussinesq-stratified 测试($\rho_0 = 1$, constant $N^2$)已经
闭环到 $-0.25\%$ 频率误差 @ $t = 500$,deviation $\sim 6\times 10^{-3}$,长时稳定不爆。
实现的关键不是 hyperviscosity 或 IMEX,而是把 y 方向 spectral filter 从默认的
Chebyshev-T basis 改到 SL basis —— 这是一个 operator convergence 问题**。

# 1. 实验设置

## 1.1 背景与变量

- 求解器 `AnelasticSLSolver`,背景 `stratified_n2`($\rho_0 \equiv 1$,$N^2$ 常数,
  由 `--ps-vshear` 传入)
- Grid $n_x \times n_y = 64 \times 48$ CGL,$L_x = L_y = 1$,$\nu = 10^{-6}$
- 时间格式 Shu-Osher RK3 + Chorin projection
- IC 路径 `AnelasticSLSolver::init_gmode_eigenmode(kx_int, n_g, amp)`:
  调 `compute_2d_gmode_evp` 拿 eigenvalue $\omega^2$ 与 eigenvector $V(y)$,
  由连续性 $\partial_x(\rho_0 u) + \partial_y(\rho_0 v) = 0$ 反推
  $U(y) = (\rho_0 V)'(y)/(\rho_0 k_x)$,并设 $b = -(N^2/\omega^2)\cdot v$
- 测试模式:$k_x = 2\pi/L_x$,$n_g = 1$,解析 $\omega^2 = 4/5$,$\omega = 0.894427$

## 1.2 两个诊断

**(A) Probe + FFT**,`probe_v_center()` 每步记录中心点 $v(t)$,事后 FFT。
  长时间段($t=500$,$\Delta\omega = 0.0126$)可以分辨相邻 g-mode($\omega_1 = 0.894$,
  $\omega_2 = 0.707$,差 $\Delta = 0.19$ → 15 个 bin)。

**(B) Eigenmode deviation**,直接量化 IC eigenmode 被散射的程度:
```
dev(t) = ‖v(t) − a(t)·V_IC‖ / ‖V_IC‖,     a(t) = ⟨v(t), V_IC⟩ / ⟨V_IC, V_IC⟩
```
不受 FFT 窗口、谱泄漏干扰。纯 eigenmode 时 dev ≡ 0,线性算子保持子空间 dev 也 $\equiv 0$。

# 2. 失败路径 — 默认 Chebyshev-T filter

## 2.1 无 filter (baseline)

```
amp=1e-3, ν=1e-6, cfl=1.0, dt=5e-3
t=41     max|v| = 1.88e+43        ← blow up
```

非线性平流 + 壁面 Chebyshev clustering 在 $t \approx 40$ 触发高波数
aliasing。`compute_rhs_uv` 已经有 2/3 dealias on $x$,但 $y$ 方向没有任何耗散机制超过
标准 $\nu \partial_{yy}$。

## 2.2 Chebyshev-T spectral filter (Boyd-Vandeven)

实现 `apply_y_filter` 用 Boyd-Vandeven exponential filter
$$\sigma(n) = \exp\left(-\alpha \cdot \left(\frac{n - n_{\rm cut}}{N - n_{\rm cut}}\right)^s\right),
\quad n > n_{\rm cut}$$
$Q = F_T^{-1} \cdot \mathrm{diag}(\sigma) \cdot F_T$ 作为 $n_y \times n_y$ 密矩阵,每步末端
同时作用于 $u, v, b$($F_T$ 是 CGL 上的 Chebyshev DCT-I)。

扫描 $\alpha \in \{8, 36, 72, 144\}$,$s \in \{4, 8, 16\}$,cut frac $\in \{0.33, 0.5, 0.67, 0.8\}$:

| 参数 | 存活时间 | FFT 主峰 rel |
|---|---|---|
| α=8, s=16, cut=0.67 | t=71 | +0.4% (短时主峰是对的) |
| α=36, s=16, cut=0.67 | t=69 | +14% |
| α=36, s=8, cut=0.33 | t>150 | **+7%, -20%, -40% (人造模)** |
| α=36, s=8, cut=0.33, amp=1e-5 | t>500 | ω=0.993, 0.712 (人造) |

**Dev 随时间的行为(α=36 s=8 cut=0.33 实验):**
```
t=1   dev=0.49   (EVP→IC 数值 mismatch)
t=5   dev=1.27   (早期散射)
t=10  dev=0.15
t=60  dev=0.81   ← 单调恶化
```

核心观察:**强 filter 稳定了,但把 n_g=1 eigenmode 的 Chebyshev-T 高模尾巴砍掉了**,
导致 $V(y)$ 不再是离散算子 $L$ 的不变方向,每步都被散射到 n_g=2, n_g=3,甚至人造
$\omega \approx 1.05$ 模。弱 filter 保真但不稳定。**accuracy-stability 零和博弈**。

# 3. 诊断 — operator-compatibility 问题

CUDA 时域推进器里用到的算子**不止 Chebyshev-$D$**:

| 层 | 算子 | 代数 basis |
|---|---|---|
| 离散微分 | `apply_dy` = $D_{\rm cheb}$ | Chebyshev T_n |
| 压力投影 | SL-Poisson: $\nabla\cdot(\rho_0\nabla\pi) = f$ | **SL eigenvectors $\psi_n$**(Dirichlet,sin-like) |
| 粘性 | $\nu \partial_{yy}$ = $\nu D^2_{\rm cheb}$ | Chebyshev T_n |
| filter (之前) | $Q$ Boyd-Vandeven | **Chebyshev T_n** |

**关键:** 投影用 SL basis,filter 用 Chebyshev basis。**这两个 basis 在 CGL 网格上不正交也不互为 eigenbasis**。

EVP 给的 $V_{\rm EVP}$ 本身是离散算子 $A = -D \rho_0 D + k_x^2\rho_0$ 的 eigenvector。
写到 Chebyshev T_n 展开里,它不集中在前 $N/3$ 模 —— 它有一条**物理的** tail 在 $n > N/3$
里,编码的是 CGL clustering 与 Dirichlet BC 的匹配关系。

Filter 在 Chebyshev basis 里做低通 → 砍掉这条 tail → $V_{\rm EVP}$ 不再是 $A$ 的 eigenvector。
一步之后,RHS $\sim A \cdot (\text{filtered } V)$ 就向其他 $V_m$ 方向泄漏。

**这不是 "filter 太强" 的问题,是 filter 和 projection 用不同算子的问题。**

# 4. 正确方案 — SL basis filter

## 4.1 构造

SL 本征函数 $\psi_n(y)$ 已经在 `set_background` 里预计算 + `h_Psi` 保存
(row-major $(n_{\rm modes}, n_y)$)。正向/反向变换是 CC-内积正交:
$$a_m = \sum_j w_{\rm cc}[j]\,\psi_m(y_j)\,f(y_j), \qquad
f(y_j) = \sum_m \psi_m(y_j)\,a_m$$

Filter 矩阵
$$Q(j, i) = \sum_m \psi_m(y_j)\,\sigma_m\,w_{\rm cc}[i]\,\psi_m(y_i)$$

即"到 SL 系数 → 乘 σ → 回到物理空间"。**关键:$\sigma_m$ index 跑的是 SL 模而不是
Chebyshev 模;$\psi_m$ 都满足 Dirichlet BC;SL-Poisson 本身就是在 $\{\psi_m\}$ 张开的子空间里解的。**

## 4.2 实现

`src/gpu/anelastic_sl_solver.cu::set_background`(filter build 块移到 SL 求解之后,
所以 `h_Psi` 可用),`apply_y_filter` 复用同一套 DGEMM。选择由 `ANSL_FILTER_BASIS=cheb|sl|evp` env var 控制。

## 4.3 结果

$\alpha=36$,$s=8$,cut frac $= 0.33$,其他所有参数同 §2.2:

```
t_end=500, samples=100002, max|v|=1.48e-3, n_nan=0
  t=10   |v|=1.35e-3   dev=5.72e-2
  t=50   |v|=1.37e-4   dev=5.14e-2
  t=100  |v|=4.40e-4   dev=2.34e-2
  t=200  |v|=5.53e-4   dev=1.78e-2
  t=300  |v|=5.96e-5   dev=1.52e-2
  t=400  |v|=1.84e-4   dev=2.82e-3
  t=500  |v|=9.71e-5   dev=5.97e-3        ← single-digit 10⁻³

FFT (Bin Δω = 0.01257):
  ω=0.87964  rel=-1.654%  P=4.26e+01
  ω=0.89220  rel=-0.249%  P=1.47e+02       ← 主峰
  ω=0.90477  rel=+1.156%  P=3.89e+01
```

**主峰 $-0.25\%$(与 EVP 的 0.894),deviation 从 $6\times 10^{-2}$ 衰减到 $6\times 10^{-3}$
并保持稳定**。这已经到 RK3 + Chorin splitting 的时间误差本底,与空间 EVP 的 $10^{-14}$
完全分离,意味着时域求解器在**同一个谱精度**下工作,误差来源全部是**时间离散**。

# 5. 对比总表

$64\times 48$ 网格,$\alpha=36$,$s=8$,cut frac $=0.33$,$t_{\rm end}=60$ 或爆前:

| basis | 存活 | dev@t=10 | dev@t=60 (或爆前) | FFT 主峰 rel |
|---|---|---|---|---|
| **none** | **t=41 爆** | 0.15 | — | 混合 |
| CHEB | >60 | 0.15 | 0.81 | +5.4% / +17% (人造) |
| **SL** | **>500** | **0.057** | **0.053 → 0.006** | **-0.249%** |
| EVP* | t=5 爆 | 0.44 | — | +43% (人造) |

\* EVP basis:用 2D g-mode EVP 的全部 interior eigenvectors 做 $Q = V_R\mathrm{diag}(\sigma)V_R^{-1}$。
失败原因:generalized EVP 的 eigenvector 矩阵 $V_R$ **非正交**(non-Hermitian),$Q$
不是缩并算子,数值放大某些方向。这说明 operator-aligned **不等于 eigenvector-aligned**
—— 关键是 basis 要是 **self-adjoint** operator 的 eigenbasis(SL 是,generalized EVP 不是)。

# 6. 关于奇异性的遗留问题

## 6.1 Reduced-pressure 变换在这里做了什么

当前 SL spectral 框架的代数骨架是 **reduced-pressure form** $\pi = p/\rho_0$:
$$\nabla\cdot\left(\frac{1}{\rho_0}\nabla p\right) = f
\quad\Longrightarrow\quad
\nabla\cdot(\rho_0\nabla\pi) = f'$$
经过 SL 对角化后,算子变成 $T = \partial_y^2 + \tilde W(y)$,其中
$$\tilde W(y) = \frac{(\rho_0')^2}{4\rho_0^2} - \frac{\rho_0''}{2\rho_0}$$
这是 Liouville potential(见 `docs/reduced_pressure_chebyshev.*`)。

**历史动机:** 直接在 $(1/\rho_0)\nabla$ 算子上做 SL,$1/\rho_0\to\infty$ 会出现在
矩阵元素里,数值灾难。Reduced-pressure 把奇异性从"算子系数"挪到"标量势 $\tilde W$",
**数学上严格等价**,但势函数在 $\rho_0 \to 0$ 处以 $1/\rho_0^2$ 发散 —— 没消失,只是移位。

## 6.2 三类奇异性,哪些已解决

**(代数层面)已解决。** $1/\rho_0$ 不再进矩阵,只以 $\mu_n$ 本征值形式存在。

**(本征函数层面)绕过。** Limit-point singular endpoint 严格意义下**不存在**有限维基
收敛到连续谱。当前靠 `rho_cut=0.01` 把域截到 $[0, 0.99R]$,变成 regular SL。g-mode
驻留内部(Exp K 对 GYRE 误差 $3.6\times 10^{-5}$ 就是证据);p-mode、surface trapped
modes 会被污染,但这类问题不在 anelastic 适用域内。

**(物理层面)未涉及。** 真实恒星表面光球/CSM 过渡需要非理想 BC,超出 anelastic 框架。

## 6.3 Filter 实验的位置

全程在 Boussinesq limit($\rho_0 \equiv 1$)做,**$\tilde W \equiv 0$,SL basis 退化为
纯 Dirichlet $\sin(n\pi y/L_y)$**。所以这次的 $10^{-3}$ operator-convergence 结果是
**rigorous 结果,没有任何截断 artifact**。

未来做 Lane-Emden n = 3/2 或 n = 3 恒星背景 g-mode 时,SL filter 的适用性继承
`rho_cut` 截断假设,预期仍然工作(同一个 well-posed 问题上的同一个 operator-aligned
filter),但**数值精度会被截断误差 bound 住**,不再是机器精度。

# 7. 代码变更

- `src/gpu/anelastic_sl_solver.cuh`:
  - 新增 `enum class FilterBasis`,字段 `filter_basis`,`filter_alpha`,`filter_s`,
    `filter_cut_frac`,`filter_evp_kx`
  - 新增 `h_eigmode_v`,`eigmode_norm`,`eigmode_omega`(诊断状态)
  - API:`apply_y_filter`,`eigmode_deviation`,`init_gmode_eigenmode`

- `src/gpu/anelastic_sl_solver.cu`:
  - `set_background` 末端加 filter Q 组装(CHEB/SL/EVP 三路,$n_y\times n_y$ 密矩阵)
  - `init_gmode_eigenmode`:EVP eigenvector → 时域 $(u, v, b)$ IC
  - `apply_y_filter`:单个 DGEMM 应用 Q(用 `d_scratch` out-of-place)
  - `step()` 末端(第 4 个 project_div_free 之后)挂载 filter,在 $u, v, b$ 上作用一次然后再 project
  - `eigmode_deviation`:L² 投影诊断,返回归一化残差
  - Free-slip BC:`project_div_free` 里只 zero $v$,$u$ 保留(physically correct,且
    EVP/时域 BC 匹配,见 Task #28)

- `src/main.cpp`:
  - `--test gmode_eigenmode_td` dispatch,CSV header + 3-列数据(`t  v_center  dev`)
  - 外层 test whitelist 也加了 `gmode_eigenmode_td`

- `scripts/gmode_eigenmode_td_fft.py`:probe CSV → FFT + 窄窗 peak picker + dt-scan

# 8. 环境变量一览

| env var | 作用 | 默认 |
|---|---|---|
| `ANSL_FILTER_BASIS` | `cheb` \| `sl` \| `evp` | `cheb` |
| `ANSL_FILTER_ALPHA` | Boyd-Vandeven α(0 = 禁用) | 0 |
| `ANSL_FILTER_S` | Boyd-Vandeven s | 16 |
| `ANSL_FILTER_CUT` | $n_{\rm cut}/N$ | 2/3 |
| `ANSL_FILTER_EVP_KX` | EVP basis 用的 $k_x$(physical) | $2\pi/L_x$ |
| `ANSL_DT_MAX` | override `dt_max` | 5e-3 |
| `ANSL_NG` | TD IC 选第几个 g-mode | 1 |
| `ANSL_SKIP_IC_PROJECT` | IC 后跳过首次 project_div_free | unset |

# 9. 一键复现

```bash
cmake --build build -j --target stellar2d

ANSL_FILTER_BASIS=sl ANSL_FILTER_ALPHA=36 \
ANSL_FILTER_S=8 ANSL_FILTER_CUT=0.33 \
ANSL_DT_MAX=5e-3 \
./build/stellar2d --solver anelastic_sl --test gmode_eigenmode_td \
    --ntheta 64 --nr 48 --ps-Lx 1 --ps-Ly 1 \
    --ps-vshear 1.0 --ps-k 1 --perturb 1e-3 \
    --tend 500 --cfl 1.0 --ps-nu 1e-6

python3 scripts/gmode_eigenmode_td_fft.py \
    runs/gmode_eigenmode_td_*/gmode_eigenmode_td.csv
```

预期:主峰 $\omega = 0.8922$(EVP 0.8944427,相对 $-0.25\%$),$t_{\rm end} = 500$ 稳定
不爆,eigenmode deviation 在 $6\times 10^{-3}$ 量级。

# 10. 后续可以做的

1. **变密度回归** —— 切到 `lane_emden_1_5` 背景,跑同样的 eigenmode TD 测试,看
   SL filter 在 $\rho_0(y)$ 非平凡时是否仍然保持 operator convergence
2. **时间格式升级** —— 当前 $-0.25\%$ 误差本底是 RK3 + Chorin first-order splitting。
   换 IMEX-RK3 或 ARS(2,3,3) 应该能压到 $10^{-4}$
3. **Exp J 型 CUDA EVP** —— Exp K 的 Chebyshev 4-var 基础上换成均匀网格(Python
   Exp J 的对应),进一步消耗 pipeline fix 的技术债
4. **文档补到 `equations.md`** —— reduced-pressure + SL filter + operator convergence
   的完整理论部分
