# ALE2D 轴对称离散修正:Hoop Stress 项

**Status**: Design (pre-implementation)
**Date**: 2026-05-01
**Branch**: radial-only-mode
**Scope**: 单 kernel 修改,补齐 `k_ale_node_forces` 缺失的 hoop-stress 源项
**Related**: `docs/ale_design.md`

---

## 0. 摘要

当前 ALE2D 的 Green's-theorem 离散力只写了 `-∮ 2πR P n̂ dℓ`(表面项),
漏掉了轴对称坐标特有的 `+2π ∫∫ P dR dZ · R̂`(hoop 源项)。
这导致:

1. 均匀 P 下每 cell 有 spurious 净力 `-2π P A_2D · R̂`(指向 Z 轴);
2. 球对称 Lane-Emden HSE 上 `max|F|/m ≠ 0`(就是 `snapshot_hse` 打印的那个 defect);
3. Caramana compatible-energy 恒等式 `Σ F_sub·v = -(P+Q)·V̇` 在 O(1) 尺度上不闭合,
   而不是设计文档里原本以为的 O(h²) staggering 误差。

修复范围:一个 kernel 三行代码 + 一个签名参数。不涉及几何、CFL、能量、BC、重力。

---

## 1. 判定两个怀疑点

| # | 怀疑点 | 结论 |
|---|---|---|
| ① | Hoop stress 缺失 | **确实是 bug** |
| ② | `2π R_mid` 权重混淆量纲 | **不是 bug**,current code 在 true-area convention 下自洽 |

### 1.1 关于怀疑 ②(`2π R_mid` 权重)

量纲自检:

| 量 | 当前代码 | 维度 |
|---|---|---|
| `F_edge = P · 2π R_mid · (dZ, -dR)` | [压强]·[长度²] | [力] ✓ |
| `m_node = 0.25 Σ dm`,`dm = ρ · V_Pappus`(含 2π) | | [质量] ✓ |
| `F/m` | | [加速度] ✓ |

这是 Caramana 的 **true-area convention**:力与质量都含真实的 `2π R` 加权,是物理量纲。
Maire-Loubère 的 **per-radian convention** 是 `F = P·R_mid·(dZ,-dR)`、`m = ρ V/(2π)`,
结果等价但不能与 true-area 混用。

**结论**:`2π R_mid` 权重正确,不需要改。只是在 true-area 下**漏了 hoop**。

---

## 2. 数学推导

### 2.1 连续形式

轴对称 (R, Z) 坐标,scalar pressure P。对一个 cross-section cell Ω(2D 截面积 `A_2D`,
revolved volume `V = 2π ∫_Ω R dR dZ`)做 R 方向动量积分:

$$
\int_\Omega \rho \frac{Dv_R}{Dt}\, 2\pi R\, dR\, dZ
= -\int_\Omega \frac{\partial P}{\partial R}\, 2\pi R\, dR\, dZ
$$

右边关键一步 —— product rule:

$$
R\,\frac{\partial P}{\partial R}
= \frac{\partial (RP)}{\partial R} - P
$$

代回:

$$
F^R_\text{cell}
= -2\pi\int_\Omega \frac{\partial(RP)}{\partial R}\,dR\,dZ
\;+\; \underbrace{2\pi \int_\Omega P\,dR\,dZ}_{\text{hoop}}
$$

第一项用散度定理变成边界通量:

$$
\boxed{\;F^R_\text{cell}
= -\oint_{\partial\Omega} 2\pi R\, P\, n_R\, d\ell
\;+\; 2\pi\, A_{2D}\,\bar P\;}
$$

Z 方向类似推导**不产生 hoop**,因为 Z 向方程两边乘 R 后 `R ∂P/∂Z = ∂(RP)/∂Z` 没有余项:

$$
F^Z_\text{cell}
= -\oint_{\partial\Omega} 2\pi R\, P\, n_Z\, d\ell
$$

### 2.2 几何一致性自检(P ≡ 常数)

真实物理受力必须为 0。

- 表面项: `-P · ∮ 2πR n_R dℓ = -P · 2π · ∂/∂R(∫ R dR dZ) = -P · 2π · A_2D`
- Hoop 项: `+P · 2π · A_2D`
- 合计:0 ✓

只写表面项时,**均匀 P 下每 cell 残留净力 `-2π P A_2D · R̂`**,指向 -R(向轴)。
球对称 HSE 上会把所有物质往 Z 轴挤。

### 2.3 离散对应

当前 `k_ale_node_forces` 每条 edge 算:

$$
\vec F_\text{edge} = (P+Q) \cdot 2\pi R_\text{mid} \cdot (dZ,\, -dR)
$$

4 条 edge 相加正是离散的 `-∮ 2πR P (n_R, n_Z) dℓ`。**这部分是对的**,缺的只是每 cell 的:

$$
\vec F^\text{hoop}_\text{cell} = \bigl(2\pi (P+Q)\, A_{2D},\; 0\bigr)
$$

均分到 4 个 corner,每个 corner 的 subcell 增量:

$$
\Delta \vec F_\text{sub,corner} = \Bigl(\tfrac{\pi}{2} (P+Q)\, A_{2D},\; 0\Bigr)
$$

---

## 3. 兼容能量(compatible-hydro)是否仍然成立?

加 hoop 之前必须验证。Caramana compatible-hydro 要求:

$$
\frac{d e_\text{int}}{dt} = -\frac{1}{m}\sum_\text{corners} \vec F_\text{sub}\cdot \vec v_\text{node}
\quad\Longleftrightarrow\quad -(P+Q)\,\frac{\dot V}{m}
$$

Pappus 体积的时间导数(对 CCW 四边形严格):

$$
\dot V = 2\pi \oint R\,(\vec v\cdot \hat n)\,d\ell
\;+\; 2\pi\, A_{2D}\,\bar v_R
$$

第一项来自边界顶点移动改变边界通量;第二项是 `V = 2π ∫R dR dZ` 对积分域的
Reynolds-transport 项(离散上等于 4 corner 的 `v_R` 平均)。

做功分项:

| 力 | 做功 | 匹配哪一项 |
|---|---|---|
| 表面项 `F_surf` | `-(P+Q) · 2π ∮ R (v·n) dℓ` | `-(P+Q) × 第一项` ✓ |
| Hoop 项 `F_hoop = 2π(P+Q)A_2D · R̂` 均分 4 corner | `2π(P+Q)A_2D · (1/4)Σ v_R^corner` = `2π(P+Q)A_2D · ⟨v_R⟩` | `-(P+Q) × 第二项` 的反号 ✓ |

**加 hoop 后 `Σ F_sub · v = -(P+Q) V̇` 严格闭合**。不加 hoop 才是有 O(A_2D·P·v) 的系统性 bug。
设计文档里 A1 条目写的 "energy 有 O(h²) staggering 误差" 被这个一阶漏项掩盖了。

---

## 4. 具体代码修改方案

### 4.1 修改 1(核心):`k_ale_node_forces` 加 hoop

**文件**:`src/gpu/ale2d_kernels.cu`
**函数**:`k_ale_node_forces`(line 169-209)
**新签名**:增加一个 `const double* Area` 参数,避免重算 `A_2D`

```cpp
__global__
void k_ale_node_forces(const double* R, const double* Z,
                       const double* P, const double* Q,
                       const double* Area,        // <-- NEW: 传入 d_Area
                       double* FR, double* FZ,
                       double* FSR, double* FSZ,
                       int nr, int nt);
```

**逻辑改动**:在现有角循环(line 199-208)里,给 `sx` 加 hoop 贡献,
并把这个贡献一并塞进 `FSR`(这样 compatible-energy kernel 自动吃到):

```cpp
double PQ     = P[flat] + Q[flat];
double A2D    = Area[flat];                  // 已由 k_ale_geometry 算好
double F_hoop_per_corner = 0.5 * M_PI * PQ * A2D;  // = 2π(P+Q)·A_2D / 4

for (int k = 0; k < 4; ++k) {
    int km = (k + 3) & 3;
    double sx = 0.5 * (aR[km] + aR[k]) + F_hoop_per_corner;  // R 分量加 hoop
    double sz = 0.5 * (aZ[km] + aZ[k]);                       // Z 分量不动
    FSR[flat*4 + k] = sx;
    FSZ[flat*4 + k] = sz;
    atomicAdd(&FR[I[k]], sx);
    atomicAdd(&FZ[I[k]], sz);
}
```

要点:

- **只加 R 分量**,Z 方向没有 hoop;
- 使用 `Area[flat]`(geometry kernel 已存的 `fabs` 形式)—— 正常 CCW mesh 下 A_2D > 0;
- **hoop 必须写进 `FSR`**,这样 `k_ale_energy_update` 才能自动得到正确的 `-(P+Q) V̇`;
- 不要改 Z 分量、不要改 `aR[k]`/`aZ[k]` 的 edge 公式。

### 4.2 修改 2(管道):调用点传 `d_Area`

**文件**:`src/gpu/ale2d_solver.cu`

两处调用需要加 `d_Area`:

- `snapshot_hse` 里(约 line 304):
  ```cpp
  k_ale_node_forces<<<BCell, B>>>(d_R, d_Z, d_P, d_Q, d_Area,
                                  d_FR, d_FZ, d_FSR, d_FSZ, nr, nt);
  ```
- `step` 里(约 line 369):同上

**顺序约束**:`k_ale_geometry` 必须在 `k_ale_node_forces` 之前执行
(这样 `d_Area` 是当前 mesh 的值)。当前代码在 `step()` 里是先 geometry 再 force,
在 `snapshot_hse` 里也是先 geometry 再 force —— 已经正确,无需调整。

### 4.3 前向声明同步

`src/gpu/ale2d_solver.cu` line 18 的 forward declaration 需要加上新的 `const double*` 参数。

### 4.4 不需要改的东西

- ❌ `k_ale_eos_and_q` 里的 `dVdt`:它用 Pappus flux form,和 hoop 是一致的
  (dV/dt 整体被 hoop+surface 之和正确匹配)
- ❌ `k_ale_energy_update`:它对 subcell·velocity 求内积,只要 `FSR` 里含 hoop
  就自动正确
- ❌ `2π R_mid` 权重:不是 bug
- ❌ gravity kernel、CFL、BC(axis/origin)、node_update、node_mass
- ❌ 质量 / dm / 节点质量:全部不变

---

## 5. 验证清单

改完跑:

### 5.1 HSE static test(首要)

```
./stellar2d run --solver ale2d --test lane_emden --nr 128 --ntheta 32 --tend 100
```

**期望**:
- `snapshot_hse` 打印的 `max|F|/m` 从 O(g) 量级下降到 O(几何离散误差)——
  典型 `~1e-3 · g` 到 `~1e-5 · g` 甚至更低,取决于 grid resolution;
- 节点静止到相对 1e-8 以内(nr=128);
- `ΔM = 0` 严格;
- `ΔE/|E| < 1e-6` 在 t=100。

### 5.2 均匀 P 单元测试(可选,最干净的判据)

在 HSE 流程之外,写一个 smoke test:P[cell] ≡ 常数,gravity = 0,
期望所有节点合力 = 0(在边界之外的内部节点)。当前代码**不可能**通过,
改完后**必须**通过(到机器精度,除了浮点原子加的舍入误差)。

### 5.3 径向扰动 vs radial1d

```
./stellar2d run --solver ale2d --test lane_emden_perturbed --nr 256 --ntheta 32 --tend 10
```

**期望**:
- 径向剖面(`download_radial_profile`)和 radial1d 在 t=10 达到 within 2× 精度;
- `ΔE/|E|` 在 `~2e-3`(Caramana compatible 的典型水平);
- θ 对称到 1e-12。

### 5.4 残留 HSE defect 的解读

如果改完后 `max|F|/m` 仍不为零,那就不是 hoop 问题,而是其他离散不一致,
候选原因:

1. cell centroid 的 `r = sqrt(R_c² + Z_c²)` 与 `M_enc[in]`(按 node 径向
   index 查表)插值不一致 —— 重力点(node)与压强积分点(cell face)的 r 不同;
2. Lane-Emden 初值在 cell 内是 **centroid 评估** 而非 mass-weighted,
   `ρ = dm/V` 与"真实" ρ 有 O(h²) 偏差;
3. 这些是真正的 O(h²) 离散误差,应该随 nr 提高而下降 —— 用两种 resolution
   可以验证 scaling。

---

## 6. 参考文献定位

- **Caramana, Shashkov, Whalen 1998** (JCP 144, 70) Appendix A
  "Axisymmetric Coordinates":给出的公式即 `F = -∮P dA + ∫∫ P dA / R · R̂`,
  使用 true-area convention。
- **Whalen 1996** (LA-UR-96-1953):更详细的 1D cylindrical vs 2D axisymmetric
  类比,明确指出 hoop 是 axisymmetric 相对 planar-2D 的唯一离散差异。
- **Maire et al. 2008** (JCP 228, 2391):用 per-radian convention,公式表面
  无 `2π`,**不要与 true-area 混用**。

用户原问题里 "Matterflow / HYDRO-ALE 是笛卡尔所以不需要 hoop" 完全正确 ——
这也正是照抄平面 2D 的 `F = P·(dz,-dr)` 在轴对称下漏项的根本原因。

---

## 7. 实施预算

| 任务 | 预计耗时 |
|---|---|
| kernel 签名 + 3 行改动 | 10 分钟 |
| 两个调用点同步 | 5 分钟 |
| forward declaration 同步 | 2 分钟 |
| 编译 + HSE 验证 | 20 分钟 |
| 径向扰动回归 | 30 分钟 |

**总计:~1 小时**,外加可能的 HSE 残差调查(若有,属 §5.4 范畴)。
