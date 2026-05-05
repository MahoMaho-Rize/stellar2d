---
title: SL spectral — 变密度 Lane-Emden n=3/2 g-mode 时域实验
author: Phase 1e 续作(branch `anelastic-sl-spectral`)
date: 2026-05-03
---

# 目的

`docs/eigenmode_td_filter_basis_2026-05-03.md` 最后的 §5 遗留问题:

> 全程在 Boussinesq limit($\rho_0 \equiv 1$)做,**$\tilde W \equiv 0$,SL basis
> 退化为纯 Dirichlet $\sin(n\pi y/L_y)$**。未来做 Lane-Emden n = 3/2 恒星背景
> g-mode 时,SL filter 的适用性继承 `rho_cut` 截断假设,**预期仍然工作**,但
> 数值精度会被截断误差 bound 住。

这次的工作就是验证"预期仍然工作"。**结论:预期错了。** SL filter + TANH + EVP IC
在 Lane-Emden 下**不保持 operator convergence**;FFT 主峰 rel err 从 Boussinesq 的
$-0.25\%$ 退化到 $-9.4\%$,单步 eigenmode deviation 增长率增大 $15\times$。

关键发现是一个**结构性不匹配**:2D EVP 和 TD 投影用的是**两个不同的离散自伴随算子**,
在 Boussinesq 下因 $\rho_0 \equiv 1$ 偶然重合,在 Lane-Emden 下分叉。这不是 SL
filter 的锅,不是 TANH 的锅,也不是代码 bug — 是谱法离散化本身的问题。

# 1. 实验设置

- 求解器 `AnelasticSLSolver`,背景 `lane_emden_1_5`($n = 3/2$ Lane-Emden),`rho_cut=0.01`
- Grid $n_x \times n_y = 64 \times 64$,$L_x = L_y = 1$,$\nu = 10^{-6}$,CGL + TANH $\beta=2$
- EVP:`init_gmode_eigenmode(kx_int=1, n_g=1, amp=1e-3)` → $\omega^2 = 3.859$, $\omega = 1.964$, $T = 3.20$
- SL filter:$\alpha = 36$, $s = 8$, cut frac $= 0.33$,basis = SL(和 Boussinesq 同参数)

**环境变量**(新增):
| env | 作用 | 默认 |
|---|---|---|
| `ANSL_BG=lane_emden_1_5` | 把 `gmode_eigenmode_td` 的背景从 `stratified_n2` 切到变密度 Lane-Emden | unset(Boussinesq) |
| `ANSL_RHO_CUT` | Lane-Emden 表面截断 | 0.01 |
| `ANSL_SKIP_IC_PROJECT=1` | 跳过 IC 后的 `project_div_free` | unset |

# 2. 空间离散闭环(Step 1 + 2)

## 2.1 SL-Poisson manufactured test

```
ANSL_COORD_MAP=tanh ANSL_COORD_BETA=2 ./build/stellar2d \
    --solver anelastic_sl --test sl_poisson_test \
    --ntheta 64 --nr 64 --ps-Lx 1 --ps-Ly 1
```

Lane-Emden + TANH,$n_y = 64$:**Poisson rel err $= 9.4\times 10^{-5}$**。
Space OK,下面的问题不是 Poisson 求解。

## 2.2 2D EVP

```
ANSL_BG=lane_emden_1_5 ANSL_COORD_MAP=tanh ANSL_COORD_BETA=2 \
./build/stellar2d --solver anelastic_sl --test gmode_2d_evp ...
```

| $n_g$ | $\omega^2$_CUDA | $\omega$ |
|---|---|---|
| 1 | 3.8588 | 1.964 |
| 2 | 1.9631 | 1.401 |
| 3 | 1.1577 | 1.076 |

机器精度 EVP(`[2D EVP] max_imag_ratio=0`,所有 62 个 mode 全 real positive)。

# 3. 时域 — 失败路径

## 3.1 完整 pipeline 跑 $t=300$

```bash
ANSL_BG=lane_emden_1_5 ANSL_RHO_CUT=0.01 \
ANSL_COORD_MAP=tanh ANSL_COORD_BETA=2 \
ANSL_FILTER_BASIS=sl ANSL_FILTER_ALPHA=36 ANSL_FILTER_S=8 ANSL_FILTER_CUT=0.33 \
ANSL_DT_MAX=2e-3 \
./build/stellar2d --solver anelastic_sl --test gmode_eigenmode_td \
    --ntheta 64 --nr 64 --ps-Lx 1 --ps-Ly 1 \
    --ps-k 1 --perturb 1e-3 --tend 300 --cfl 1.0 --ps-nu 1e-6
```

```
t_end=300, samples=150001, max|v|=3.69e-04, n_nan=0
  t=6     |v|=1.52e-6   dev=2.27e-1
  t=30    |v|=1.53e-6   dev=1.64e-1
  t=90    |v|=5.26e-7   dev=7.92e-2
  t=150   |v|=1.63e-7   dev=2.51e-2
  t=300   |v|=1.40e-8   dev=1.00e-3        ← |v| 衰减 5 个数量级

FFT 主峰:ω = 1.780, rel err = -9.4%, leakage = 1.0
```

长时稳定不爆,但 $|v|$ 从 $10^{-3}$ 单调衰减到 $10^{-8}$ —— SL filter 吃掉了
g-mode 本身。FFT 在末端只看到衰减趋势里的伪峰。

## 3.2 弱 filter → 线性失稳

试 $\alpha = 8, s = 16$ 或 $\alpha = 16, s = 8, {\rm cut} = 0.5$ 等四个弱化参数:
**全部 $t < 15$ 内就 NaN**,即使用 amp=1e-5 也一样。strong filter
衰减 / weak filter 爆炸,**两头都不工作**,与 Boussinesq 下 $\alpha=36$ 能稳住到
$t=500$ 形成鲜明对比。

# 4. 诊断 — operator consistency test

## 4.1 设置

跳过 IC projection($\rho=\text{Lane-Emden}$ 下 EVP 的 $U = (\rho V)'/(\rho k_x)$
解析上已经严格满足 $\nabla\cdot(\rho u) = 0$,只有离散残差);关 filter;关粘性;
$dt = 10^{-4}$;amp = $10^{-8}$(纯线性)。measure 每步 dev 增量作为
"EVP 特征向量 in TD linear operator 的一阶泄漏率":

```bash
ANSL_SKIP_IC_PROJECT=1 ANSL_DT_MAX=1e-4 ANSL_FILTER_ALPHA=0 \
./build/stellar2d ... --perturb 1e-8 --tend 0.01 --cfl 0.1 --ps-nu 0
```

## 4.2 结果

| 背景 | dev/step | dev@step-1 | 解释 |
|---|---|---|---|
| **Boussinesq** ($\rho_0 = 1$) | $4.5\times 10^{-5}$ | $4.5\times 10^{-5}$ | $V_{\rm EVP}$ 接近 $T$ 的特征向量 |
| **Lane-Emden** ($n = 3/2$) | $\mathbf{6.9\times 10^{-4}}$ | $6.9\times 10^{-4}$ | **$15\times$ 更差** |

在 Boussinesq 下 EVP 的 V 每步只泄漏 $4.5\times 10^{-5}$ 到其他方向,
这是 RK3 + Chorin splitting 的本底。**Lane-Emden 下每步泄漏 $6.9\times 10^{-4}$,
比 Boussinesq 差 $15\times$** —— 这就是 §3.1 里 FFT 主峰从 $-0.25\%$ 退化到
$-9.4\%$ 的根因。

# 5. 根因分析 — 两套离散算子

## 5.1 EVP 用的算子

`compute_2d_gmode_evp` 直接从 primitive anelastic 方程消元:

$$B \hat v = \omega^2 A \hat v,\qquad
A = \operatorname{diag}(k_x^2 N^2 \rho_0),\qquad
B = -D\cdot\operatorname{diag}(\rho_0)\cdot D + k_x^2 \operatorname{diag}(\rho_0)$$

这里 $D$ 是 CGL Chebyshev 微分矩阵(TANH coord-map 下已重缩成 $d/dy$)。
$B$ 是**Galerkin 离散的自伴随算子** $-\partial_y(\rho_0 \partial_y\cdot) + k_x^2 \rho_0$。

## 5.2 TD 投影用的算子

`project_div_free` 解 $\nabla\cdot(\rho_0\nabla\pi) = \operatorname{RHS}$。
`sl_poisson_solve` 的谱展开 basis 是**另一个离散算子** $T = \partial_y^2 + \widetilde W$
的特征向量 $\{\psi_n\}$(Liouville 变换 $\hat\pi = \rho_0^{-1/2} q$ 下的 SL 算子):

$$T q = -\mu_n q,\qquad \widetilde W(y) = \frac{(\rho_0')^2}{4\rho_0^2} - \frac{\rho_0''}{2\rho_0}$$

$T$ 和 $-D(\rho_0 D) + k^2\rho_0$ **在连续极限下等价**(那是 Liouville 变换的
定义),但**在 ny 有限的离散意义下不等价** —— Galerkin 项
$-D\cdot\operatorname{diag}(\rho_0)\cdot D$ 的矩阵元素与 $\partial_y^2 + \widetilde W$
的对应矩阵元素差 $O(1/n_y^2)$,正是 D 的离散 commutator 残差。

## 5.3 为什么 Boussinesq 不受影响

$\rho_0 \equiv 1$ 时 $\widetilde W \equiv 0$,SL 算子退化为 $T = -D^2$;
同时 EVP 的 $B = -D^2 + k^2 I$。**两套矩阵数值上严格相等**(除对角加 $k^2$),
所以 $V_{\rm EVP}$ 就是 $T$ 特征向量,TD 投影不散射它。

Lane-Emden 下 $\rho_0'$ 非零,$-D\rho_0 D \neq T$ 数值上差 $O(10^{-3})$ 量级的
矩阵元素,这正好对应 dev/step $\sim 10^{-3}$ 的泄漏率。

## 5.4 这不是 bug

两套算子各自都是 **一致(consistent)**、**稳定**、**收敛到相同连续极限**的 ny×ny 离散。
EVP 算子是 Galerkin 分部积分后的自然形式;TD 算子是 Liouville 变换后解
常系数 Helmholtz 的自然形式。他们数学上等价,**离散化选择不同**。

非线性求解域常见的坑:两种"等价"的离散 Poisson 算子在变系数下分叉。

# 6. 尝试过的其他修正路径(全部无效)

## 6.1 ~~v 动量加 $-(\rho'/\rho)\pi$~~

误判:以为 Chorin projection 里 `v -= ∂y π` 漏了 `-(ρ'/ρ)π`。
重新推导后发现当前代码的 RHS $= \rho\cdot(\partial_x u + \partial_y v) + \rho' v$
配合 `v -= ∂y π` 是**自洽的**(代入验证 $\nabla\cdot(\rho u_{\rm new}) = 0$ ✓)。
这个改动会打破 Chorin 的守恒性。已撤销。

## 6.2 ~~Weak filter 保留 g-mode tail~~

$\alpha = 8$, $s = 16$, cut frac $= 0.67$:$t < 10$ 就 NaN(非线性稳定性
对 Lane-Emden 要求的耗散下限远高于 Boussinesq)。

## 6.3 ~~amp = $10^{-5}$ 压制非线性~~

同样在 $t < 8$ NaN —— 确认线性阶段就不稳,不是平流 aliasing。

# 7. 当前状态

**保留**:$ANSL_BG$ env var 路径、$d\_rho\_prime\_over\_rho$ 字段(Phase 2
s-coord 重构会用到)、`init_gmode_eigenmode` 里 IC 峰值 + $\rho$ 范围诊断。

**未解决**:Lane-Emden 下 EVP → TD eigenmode 保真。

# 8. 结论 — 何时 SL filter + EVP IC 可信

| 条件 | 结论 |
|---|---|
| Boussinesq / $\rho_0 = $ const | ✓ 全保真,$-0.25\%$ 频率 |
| stratified_n2(常密度,强 $N^2$) | ✓ 同 Boussinesq |
| 轻度变密度($\max\rho_0'/\rho_0 \ll 1$) | ⚠ 未测,预期按 $\rho_0'/\rho_0$ 线性恶化 |
| **Lane-Emden n = 3/2** | ✗ $-9.4\%$ 频率,$|v|$ 衰减 5 个数量级 |

阈值经验规则:**dev/step(线性算子泄漏率)应该 $\lesssim 10^{-4}$** 才能用
当前 filter-based 工程方案。Lane-Emden n=3/2 是 $\sim 10^{-3}$,超阈值 $10\times$。

# 9. 下一步选择

按工程量 + 成功概率排:

1. **TD projection 换基到 EVP $B$ 算子**(最大工程量,最可靠)
   - 把 `sl_poisson_solve` 的 SL basis 换成 2D EVP 的 $V_{\rm EVP}$(每 $k_x$ 独立)
   - 但 EVP eigenvectors **非正交**(generalized EVP 的 V_R),要做 Gram-Schmidt
     或 bi-orthogonal expansion
   - 参考 `docs/eigenmode_td_filter_basis_2026-05-03.md` §5 里 EVP-basis filter
     失败的原因,这次要在 projection 级做,比 filter 级更根本

2. **s-coord 下重写 reduced-pressure 代数**
   `docs/coord_map_stretch_2026-05-03.md` §7 的 Phase 2,1-2 天工程量。在 $s = -\ln\rho$
   坐标下 $\widetilde W$ 保持 $O(1)$ bounded,$-D\rho D$ 和 $T$ 的数值差距可能大幅收窄

3. **IMEX 时间格式**
   Chorin splitting 的一阶 splitting error 叠加 operator mismatch;
   IMEX-RK3 可能把一阶 splitting 项吃掉,让 operator 差异纯净暴露(也可能反而
   让失稳时机提前)。诊断性,不是修复。

4. **接受 Lane-Emden 在当前架构下不支持 g-mode 时域**
   Document 此限制,恒星 g-mode 走 EVP 频率谱路径(Exp K CUDA 已验证 $3.6\times 10^{-5}$
   vs GYRE),时域保留给 Boussinesq / stratified_n2 / 简化 $N^2$ profile

# 10. 一键复现(失败路径)

```bash
cmake --build build -j --target stellar2d

# Step 1: space discretisation OK (rel err ~1e-4)
ANSL_COORD_MAP=tanh ANSL_COORD_BETA=2 ./build/stellar2d \
    --solver anelastic_sl --test sl_poisson_test \
    --ntheta 64 --nr 64 --ps-Lx 1 --ps-Ly 1

# Step 2: EVP gives machine-precision omega^2
ANSL_BG=lane_emden_1_5 ANSL_COORD_MAP=tanh ANSL_COORD_BETA=2 \
./build/stellar2d --solver anelastic_sl --test gmode_2d_evp \
    --ntheta 64 --nr 64 --ps-Lx 1 --ps-Ly 1 --ps-k 1

# Step 3: TD blows up or decays — see §3.1
ANSL_BG=lane_emden_1_5 ANSL_RHO_CUT=0.01 \
ANSL_COORD_MAP=tanh ANSL_COORD_BETA=2 \
ANSL_FILTER_BASIS=sl ANSL_FILTER_ALPHA=36 ANSL_FILTER_S=8 ANSL_FILTER_CUT=0.33 \
ANSL_DT_MAX=2e-3 \
./build/stellar2d --solver anelastic_sl --test gmode_eigenmode_td \
    --ntheta 64 --nr 64 --ps-Lx 1 --ps-Ly 1 \
    --ps-k 1 --perturb 1e-3 --tend 300 --cfl 1.0 --ps-nu 1e-6

# Step 4: operator consistency diagnostic — 15× worse than Boussinesq
ANSL_SKIP_IC_PROJECT=1 ANSL_DT_MAX=1e-4 \
./build/stellar2d --solver anelastic_sl --test gmode_eigenmode_td \
    --ntheta 64 --nr 64 --ps-Lx 1 --ps-Ly 1 \
    --ps-k 1 --perturb 1e-8 --tend 0.01 --cfl 0.1 --ps-nu 0

# Repeat step 4 with ANSL_BG unset for Boussinesq baseline:
# dev/step_ratio = (6.9e-4) / (4.5e-5) ≈ 15
```

# 11. 代码变更清单

- `src/main.cpp::main`:`gmode_eigenmode_td` dispatch 加 `ANSL_BG=lane_emden_1_5` /
  `ANSL_RHO_CUT` 覆盖,切换到变密度背景
- `src/gpu/anelastic_sl_solver.cuh`:成员 `d_rho_prime_over_rho`(Phase 2 预备)
- `src/gpu/anelastic_sl_solver.cu::set_background`:计算 $\rho'/\rho$ 上传到 device
- `src/gpu/anelastic_sl_solver.cu::init_gmode_eigenmode`:诊断打印
  IC 峰值 `|V|, |U|, |b|` 和 $\rho\in[\min, \max]$,用于排除 EVP 给的 U 在壁面
  $\rho \to 0$ 处是否放大(实测 $|U| / |V| \approx 4\times$,正常)
