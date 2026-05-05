---
title: Path C Python benchmark — φ=ρv TD 原型,否决 3-4 天 CUDA 投入
author: Phase 2 (branch `anelastic-sl-spectral`)
date: 2026-05-03
---

# 目的

`docs/qspace_sl_path_b_2026-05-03.md` §5 定位到 Lane-Emden operator mismatch 只能
通过路径 C(TD 改在 $\varphi = \rho_0 v$ 变量上推进)消除。路径 C 的 CUDA 移植
预计 3-4 天工程量,在投入前先写 Python benchmark 验证其可行性。

**核心问题**:φ-space TD 的 eigenvector 在完整离散 TD 线性算子下是否是不变方向?
如果是 → dev/step $\le 10^{-5}$,可以投入。如果不是 → dev/step 仍在 $10^{-4}$ 量级,
3-4 天工程量不值。

# 1. Python 测试设置

`scripts/path_c_td_benchmark.py` 实现:

- **CGL 网格 + Chebyshev D 矩阵 + Clenshaw-Curtis 权重**(匹配 CUDA 离散化)
- 两个背景:Boussinesq($\rho_0 = 1, N^2 = 1$)和 Lane-Emden($\rho_0$ 真实求 LE,
  `rho_cut = 0.01`,$N^2 = \max(0, -\rho'/\rho)$,匹配 CUDA)
- **两个 EVP**:
  - v-space: $(-D\rho D + k^2\rho)V = (k^2N^2\rho/\omega^2)V$,Dirichlet
  - φ-space: $(-D^2 + k^2)\phi = (k^2 N^2/\omega^2)\phi$,Dirichlet
- **两个 toy TD**:都是 $\partial_t X = Y,\ \partial_t Y = -N^2 X$,forward Euler
  带 Dirichlet BC。区别是 X = V(v-space)vs X = φ(φ-space)。**故意不做 Chorin
  projection**,只测线性 oscillator 耦合下 eigenvector 的不变性。

测量指标:**dev/step = (dev[15] − dev[1]) / 14** ,dev = ‖X − a(t)·X_IC‖/‖X_IC‖,
$a(t)$ 是 L²(w_cc) 投影系数。

# 2. 结果

## 2.1 Boussinesq(控制组)

```
v-space TD:  ω²=0.8000  dev/step ≈ 1e-17   (machine precision)
φ-space TD:  ω²=0.8000  dev/step ≈ 1e-17   (machine precision)
```

两路径都在机器精度 — toy TD 设置是正确的。

## 2.2 Lane-Emden —— 核心测量

```
  ny     v-space dev/step      φ-space dev/step      改善
  32     8.876e-05             6.811e-05             23%
  64     8.867e-05             6.796e-05             23%
  128    8.865e-05             6.792e-05             23%
  256    8.864e-05             6.791e-05             23%
```

**三个决定性观察:**

1. **不收敛到 0**:dev/step 随 ny 根本不下降 — 这不是离散化误差,是算子内在属性
2. **φ-space 只改善 23%**,远低于预期的 10× 改善
3. Boussinesq 同一 toy 给机器精度,证明代码是对的

## 2.3 物理解释 —— 为什么 φ-space 也救不了

buoyancy 方程 $\partial_t B = -N^2(y) V$。在 φ-space($\phi = \rho V$):
$$\partial_t (\rho B) = \rho \partial_t B = -\rho N^2(y) V = -N^2(y) \cdot \phi$$

即 **φ-space 里 $N^2(y)$ 仍然以 pointwise 乘法出现**。这个 pointwise 乘法在
CGL 网格上对 eigenvector 不是 Galerkin-consistent 操作 —— 它把 $\phi$ 映射
到物理空间,逐点乘 $N^2$,再映射回来。连续极限等价,离散不等价。

**这就是 Lane-Emden dev/step $\sim 10^{-4}$ 的真正本底**:不是 $-D\rho D$ 的
discrete Leibniz 失效(路径 B 诊断),而是 **$N^2(y)$ 逐点乘法**(buoyancy)
和 **$\rho'/\rho$ 逐点乘法**(continuity)两个 pointwise-product 操作的
discrete mismatch。

## 2.4 真正要闭环需要做什么

消除 pointwise product 散射的**唯一**方式是把所有 $N^2, \rho, \rho'/\rho$
相关操作都在 SL 基(或其他合适谱基)下做 Galerkin 投影:

- buoyancy $N^2 V$ → $(N^2_{nm}) \cdot c$,$N^2_{nm} = \langle \psi_n, N^2 \psi_m\rangle$
- advection $(\bm u \cdot \nabla)$ → 三元积 tensor $T_{nml}$
- continuity $\rho' v$ → $(\rho'_{nm}) \cdot c$

这等于**把整个 TD solver 换成 spectral-Galerkin-in-y**:每步 $O(n_{\rm modes}^2 n_x)$
而不是 $O(n_y n_x)$,VRAM 从 $n_y n_x$ 变成 $n_{\rm modes}^2 n_x$(二次),内存代价
$\sim 2 \times$,且 advection 的三元积需要 de-aliasing 和守恒性证明,**工程量 $\ge$ 7-10 天**。

# 3. 决策 —— 否决路径 C,选择替代方案

| 选项 | 工程量 | 预期 dev/step | 结果 |
|---|---|---|---|
| 路径 A(Fourier q-space EVP) | 0.5 天 | 6.0e-4 | ✓ 已做 |
| 路径 B(SL q-space EVP) | 1 天 | 5.9e-4 | ✓ 已做 |
| **路径 C(TD 改 φ 推进)** | **3-4 天** | **预估 7e-5(按 Python toy)** | **✗ 否决** |
| 全 spectral-Galerkin 重写 | 7-10 天 | 理论 1e-15 | 风险高,长期目标 |

**Path C 被 Python benchmark 否决**的理由:
- 改善只有 23%,不改变数量级
- 3-4 天 CUDA 移植风险(重写 `compute_rhs_uv` + `project_div_free` +
  continuity + IC reconstruction)换不到数量级提升
- Buoyancy 的 $N^2$ pointwise product 是实际 bottleneck,路径 C 不解决

# 4. 推荐路径

## 4.1 短期(用户当前工作流)

接受 Lane-Emden TD 的 $\sim 6 \times 10^{-4}$ dev/step 量级。记录此限制。
恒星 g-mode **频率分析**走 Exp K(已验证 $3.6 \times 10^{-5}$ vs GYRE);
**时域演化**走 Boussinesq + 自定义 $N^2(y)$ profile(已闭环到 $-0.25\%$)。

## 4.2 中期(如果要真正 Lane-Emden 时域)

直接跳过路径 C,评估:
- **全 spectral-Galerkin-in-y 重写**(7-10 天,彻底闭环)
- **IMEX-RK3 + Lagrangian buoyancy**(时间格式升级 + 物理平流与 buoyancy
  分离处理,可能压到 $10^{-5}$ 但无理论保证)
- **接受 Lane-Emden 下 $\pm 10\%$ 频率误差**,把算力投到更高 ny 或更小 dt

# 5. 代码变更

- `scripts/path_c_td_benchmark.py` — Python 原型(ny 可扫,两背景 + 两路径
  + 两测量,自动对比)

# 6. 一键复现

```bash
# Boussinesq + Lane-Emden,ny=64(同 CUDA)
python3 scripts/path_c_td_benchmark.py

# ny 收敛扫描
for ny in 32 64 128 256; do
    python3 scripts/path_c_td_benchmark.py --ny $ny | grep "dev/step"
done
```

预期:v-space dev/step ≈ 8.86e-5,φ-space dev/step ≈ 6.79e-5,**不随 ny 下降**。

# 7. 小结

这是一个经典的"先写原型验证再投资源"案例成功执行的例子 —— **半小时的 Python 脚本
省下 3-4 天的 CUDA 工作量**。路径 C 的代数推导是正确的,实现是可做的,但**其带来
的改善不够大,因为 operator mismatch 的真正大头(buoyancy $N^2$ pointwise product)
不在 $\rho$ 变换的照顾范围内**。

决定:**否决路径 C**,将 Lane-Emden TD 精度限制归档(docs/variable_density_lane_emden_td_2026-05-03.md
§5.4 的替代方案),Phase 2 暂告一段落。
