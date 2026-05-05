# Cart-Lag HSE 退化诊断 + ALE Rezone/Remap 方案

**Status**: Design (pre-implementation)
**Date**: 2026-05-01
**Branch**: ale2d-cartesian
**Scope**: 记录 cart_lag 在 HSE 下 dt 自发退化的现象,分析猜想,提出
           Eulerian Rezone+Remap(hal3d 风格三相 ALE)方案。

---

## 0. 背景(必读)

本分支 `ale2d-cartesian` 在上一个里程碑实现了笛卡尔 2D Caramana-compatible
Lagrangian 求解器(`src/gpu/cart_lag_{solver,kernels}.{cuh,cu}`),commit 8037725。
三个测试:
- `--test sod`:Sod 激波管,t=0.25 前激波未到墙时质量/能量机器精度守恒,
  激波反射后 mesh tangle → 纯 Lagrangian 的 Dukowicz-Meltz 预期行为。
- `--test hse`:等熵多方 HSE,无扰动。低分辨率/短时间看似稳。
- `--test hse_perturbed`:同上+2% 正弦扰动。

后续验证发现**即使没有扰动的 HSE,随着分辨率提高和步数增加,mesh 仍然
单调退化到 dt=1e-13**。这是本文档要记录的核心问题。

---

## 1. 观察到的现象

### 1.1 64×64 HSE 无扰动 tend=1
- 131 步完成,|v|_max = 3.5e-4,看似稳定。

### 1.2 256×256 HSE 无扰动(用户实测,Ctrl+C 终止)
```
Step   5000  t=2.859  dt=1e-12   |v|=3.55e-2
Step  10000  t=2.865  dt=3.3e-9  |v|=3.93e-2
Step  20000  t=2.885  dt=1.7e-8  |v|=5.21e-2
Step  40000  t=2.919  dt=1.7e-6  |v|=8.02e-2
Step  80000  t=2.942  dt=8.9e-11 |v|=1.02e-1
Step  85000  t=2.944  dt=3.8e-12 |v|=1.04e-1
```
- M/E 全程 10 位机器精度守恒(这是 Caramana compatible 的保证)。
- |v| 从 0 单调增长到 ~0.1(声速 0.8 量级的 12%),**没有衰减趋势**。
- dt 从 2e-3 崩到 1e-12,11 秒实际跑了 85000 步只推进到 t=2.94,
  即**dt 和进度成指数负相关**。
- 推算:按 dt~1e-12 跑到 t=1000 约需要 1e15 步,不可能完成。

### 1.3 HSE-force subtraction 尝试(已试,失败)
在 `snapshot_hse_force()` 里捕获 HSE 状态下的残差力 F_HSE,之后每步
`F ← F - F_HSE`。
- 理论上静止态抵消 O(h) 残差。
- 实测:**dt 退化模式完全没改变**,256² 还是 t=2.85 死。
- 原因:snapshot 仅对 snapshot 那一刻的节点位置精确,mesh 动 ε 后
  真实 F(x+ε) ≠ F_HSE(x),差值 ~ O(ε·|∂F/∂x|),这个差值继续驱动节点
  进一步偏移 → 正反馈。

---

## 2. 诊断和猜想

### 2.1 事实确认
- 初始力残差 `max|F|/m = 2.898e+02`(256² 网格,前几步观测)。
  这是一个 **O(1) 量级的离散 HSE defect**,不是机器 roundoff。
- Caramana compatible 保证 E 守恒,**不保证**节点速度的离散演化稳定。
- M、E、动量(应该也是,未打印)都精确守恒,但 **熵可以在 cell 间 scrambled
  并不违反守恒律**——典型"静态不稳"(spurious hourglass)的特征。

### 2.2 猜想 A(主要猜想):Spurious Hourglass / Zero-Energy Mode
四边形拉格朗日在 staggered 离散下有**零能模式**(zero-energy mode,
在工程文献里叫 hourglass mode):一些节点位移组合会让 cell 面积不变
但节点位置变化,因此力做的功为零,**这些模式不被恢复力抑制**。

- 经典例子:对角线节点同向平移,cell 面积(Shoelace)不变,P 不做功。
- 数值实验里 hourglass 模式会在任何小扰动(包括 O(h) HSE defect)下
  持续增长,直到非线性耦合把它们饱和在某个噪声水平。

**支持这个猜想的证据**:
- |v| 增长但总 E 不变 → 能量在某个 "0-energy" 通道上流动。
- 高分辨率更早发作:h 越小,hourglass 波长越小,但增长率 ~ 1/h。
- 无扰动 HSE 也发作 → 说明 driver 是 roundoff + 离散 defect 本身,不是扰动。

Caramana 1998 附录 B 专门讨论了这个,"Sub-zonal pressure / hourglass
viscosity" 就是解决这个的。Matterflow 引入的 "MatterFlow" 补丁也正是
hourglass 控制(`FunctionsMaterial.c::CalculateMatterFlowAcc`)。

**我们没实现 hourglass 控制**,所以在原始 Caramana 上所有已知的 hourglass
坑都会踩。

### 2.3 猜想 B(次要):人工粘性耦合与几何触发
- 我们的 von Neumann-Richtmyer Q 只在 strain_rate > 0(压缩)时激活。
- HSE 下 strain_rate = 0,Q = 0,**没有耗散**。
- roundoff 产生的速度扰动没有 damping 机制,自由漂移。

### 2.4 猜想 C(边界效应)
- 反射 BC `k_clag_bc_reflective` 只零化**节点的法向速度/力**,
  但没有对边界 cell 做特殊处理。
- 边界 cell 的切向压力梯度在 staggered 格式下会产生 spurious 切向力。
- 高分辨率边界 cell 占比小,但残余力的绝对量对单个节点来讲是一样的。

### 2.5 哪个是主因
**猜想 A 最符合观察。** 三个证据:
1. |v| 增长但能量守恒 → 能量走零能通道
2. 高分辨率越惨 → hourglass 波长 ~ h 越小越容易激发
3. 无扰动也发作 → 不需要外部驱动,模式本身不稳

---

## 3. 方案:Eulerian Rezone + Conservative Remap(ALE 三相)

选择**完全 Eulerian rezone**(hal3d 的 `eulerian_rezone`),即每步后
把节点拉回初始位置。这样 mesh **物理上不动**,hourglass、tangle、
dt 退化一并消失。

物理上这等价于 Eulerian,但数值路径完全不同:
- Lagrangian phase:节点跟流体走(当前 cart_lag 的 step())
- Rezone:节点 **强制回到初始位置**
- Remap:把物理量(ρ, v, e)从 "lagrangian-推进后的网格" 守恒映射到
  "原始网格"

### 3.1 数据流(每步)

```
current state (u_n on mesh_n):
  mesh_n    = original positions X0, Y0  [永远不变]
  state_n   = {dm, vX, vY, e_int}  defined on mesh_n

Phase 1 - Lagrangian (currently cart_lag::step):
  mesh_L    = deformed mesh (nodes moved by v·dt)
  state_L   = state after Caramana compatible update

Phase 2 - Rezone:
  mesh_{n+1} = mesh_n  (即 X0, Y0)
  [no state change yet]

Phase 3 - Remap:
  state_{n+1} = state_L mapped from mesh_L onto mesh_n conservatively
```

### 3.2 守恒 Remap 核心

对每个**原始**(mesh_n)cell c,它在 Lagrangian phase 后被 mesh_L 的
若干 cell 覆盖。要把每个 Lagrangian cell 的物理量按**覆盖面积**
分配到相应的原始 cells。

具体:对 mesh_L 的每个 cell cL,计算它与 mesh_n 每个 cell 的交集面积
(polygon-polygon intersection),按此面积分摊物理量:

```
For each lagrangian cell cL:
  For each original cell cN overlapping cL:
    A_overlap = intersect_area(cL, cN)
    frac = A_overlap / Area(cL)
    # Conservative scalar quantities: mass, energy
    new_dm[cN]    += frac · old_dm[cL]
    new_ie[cN]    += frac · old_dm[cL] · old_e_int[cL]
    # Momentum via per-edge or per-corner scheme (trickier for staggered)
    ...
```

**关键实现决策**:
- **标量(dm, e_int_×dm):每个 cell 值**——最简单,按面积加权即可
- **动量/速度:节点值**——需要 **subcell** 水平的 remap(Kucharik-Shashkov 2012)
  - 每个节点的"subcell" = 节点周围四分之一相邻 cells 的四边形
  - Lagrangian phase 后 subcell 也变形
  - Remap 把 subcell 的动量守恒分配回原始 subcell

**简化版本**(第一次实现):
- mesh_L 和 mesh_n 偏移不大(一步位移 < h/10),因此**每个 cL 只和
  最多 4 个 cN 有交集**(最近的 2×2)。
- 用 "swept-edge" 近似:不真的算 polygon intersection,只算**每条边**
  在 lagrangian step 中扫过的面积,用它近似 remap flux。
- 这就是 hal3d 的 `advection_phase` 做的事。

### 3.3 算法细节(swept-edge remap)

对原始 mesh 的每条边 e(从 node A 到 node B):
1. Lagrangian 位移把 A,B 移到 A', B'。
2. 四边形 A-B-B'-A' 是这条边"扫过"的面积。
3. 这个扫过区域里的物理量,之前属于哪边的 cell(靠近 A 还是 B 所在的 cell),
   remap 后应该**移到另一边的 cell**(边两侧 cell 的 cell-center 换了)。
4. 按流入方向 donor-cell upwind 选择物理量。

数学上和有限体积 Godunov flux **完全等价**,但 flux 是从几何位移算的,
不用 Riemann solver。因此 ALE remap 的耗散只来自 donor-cell upwind 的
一阶误差,二阶远远更准。

### 3.4 需要实现的 kernel

| Kernel | 大概 LOC | 功能 |
|---|---|---|
| k_clag_swept_area | ~80 | 对每条边,算 A-B-B'-A' 有符号面积 |
| k_clag_remap_mass | ~60 | 按 swept-area 从 donor cell 转移 dm |
| k_clag_remap_energy | ~60 | 转移 dm·e_int |
| k_clag_remap_momentum_subcell | ~150 | 子格动量 remap(Kucharik-Shashkov) |
| k_clag_reset_mesh | ~20 | 把 X, Y 复位到 X0, Y0 |
| k_clag_repair | ~100 | 限制器,防 remap 引入非物理极值 |

**预计总共 ~500-700 LOC 加到 `cart_lag_kernels.cu` / `cart_lag_solver.cu`**。

### 3.5 Step() 新流程

```cpp
double CartLagSolver::step(double t, double t_end) {
    // Phase 1: Lagrangian (unchanged, but 存下 X0, Y0 快照方便 rezone)
    save_mesh_to_X0_Y0();          // 一次性,init 时存
    existing_lagrangian_step(dt);

    // Phase 2+3: Rezone + swept-edge Remap
    remap_conservative(X, Y, X0, Y0, dm, vX, vY, e_int);  // 守恒转移物理量
    reset_mesh_to_X0_Y0();          // 强制节点回到初始位置
    k_clag_node_mass(d_dm, d_mnode, nx, ny);  // 节点质量随 dm 变化

    step_count++;
    return dt;
}
```

### 3.6 预期结果

- **HSE 无扰动**:节点永远不动,dm 不变,|v| = 0 + roundoff。应该精确到
  机器精度一直稳。
- **HSE 扰动**:线性声波/g-mode 传播,每步 Lagrangian 位移 ε,remap 误差
  O(ε²)·amplitude。小扰动在 remap 中几乎无损失,可以跑非常长时间。
- **Sod**:激波在 Eulerian 网格上被分辨,reflection 不会 tangle。
- **非线性塌缩**:KE 从 1.6e-7 → 0.12 的大振幅非线性演化不再触发 mesh
  死亡。

### 3.7 会损失什么

- **Lagrangian 精度**。纯 Lagrangian 对某些接触不连续/材料界面是最准的。
  我们的目标是恒星脉动,没有材料界面,所以这不是问题。
- **守恒律精度**。conservative remap 能保证机器精度 dm 和 dm·e_int,
  但**动量守恒需要 subcell remap**(更复杂),用简单 cell-center 方案
  动量只到 O(h²) 守恒。对我们的应用可以接受。

---

## 4. 为什么先做 Eulerian Rezone 而不是 Lazy/Partial

| 选项 | 优点 | 缺点 | 复杂度 |
|---|---|---|---|
| **Eulerian Rezone** | 绝对稳,实现简单(节点=const) | 物理上退化成 Eulerian 方法 | 中 |
| Partial Rezone | 保留部分 Lagrangian 精度 | 需要 mesh-quality sensor,阈值调参 | 高 |
| Lazy Rezone | 平时纯 Lagrangian,坏了才救 | 需要检测 + 回滚逻辑 | 高 |

**决定先做 Eulerian Rezone**,因为:
1. 最简单实现,能最快验证猜想(hourglass 消失否?tangle 消失否?)
2. 失败的话再迭代 Partial/Lazy 至少有基线对比
3. 恒星脉动 case 动量守恒要求不那么严,Eulerian 够用

---

## 5. 实施计划

| 步骤 | 预计 LOC | 难点 |
|---|---|---|
| 1. `d_X0, d_Y0` 存初始位置 + reset kernel | 30 | 无 |
| 2. swept-area kernel(有符号面积) | 80 | 4-point 几何 |
| 3. scalar remap(mass + ie)kernel | 120 | 需要 atomic(多 donor) |
| 4. cell-centered velocity → 用 momentum remap | 80 | 速度从节点下插值到 cell-center,再 remap |
| 5. 节点速度重建(从 cell-center 平均) | 40 | |
| 6. repair limiter(防超过原值范围) | 80 | |
| 7. step() 集成 + 测试 | 50 | |
| 8. 诊断:monitor remap error(每步多少 dm 漂移) | 30 | |

**总估计 510 LOC**,1-2 天能出 working prototype。

---

## 6. 验证目标

### Test A:256² HSE 无扰动 tend=1000
目标:dt 保持在初始值(~2e-3)不退化,|v| < 1e-10 一直稳,mass/E 机器
精度守恒,完成全部 500000 步。这是**最关键的 test**。

### Test B:hse_perturbed amp=2% tend=100
目标:|v|_rms 震荡但不发散,dt 保持 O(1e-3)。

### Test C:Sod tend=1.0(激波反射后)
目标:激波反射、膨胀波反射都能正确处理,不 tangle。

---

## 7. 风险

1. **swept-edge remap 的 upwinding**:一阶 donor-cell 耗散太厉害,
   扰动会被 smoothed out。需要 MUSCL-like 二阶 reconstruction。
   **缓解**:先实现一阶,看物理是否可接受,再升级。
2. **动量守恒**:cell-centered velocity remap 不严格守恒动量。
   **缓解**:残差监控;如果太大再做 subcell remap。
3. **边界 cell 的几何**:固定墙边界在 Eulerian rezone 下不是问题,
   但如果 Lagrangian phase 后某边界节点穿过墙,**需要 clipping**。
   **缓解**:BC 已经 pin 墙节点位置,穿墙只有内部 roundoff,影响很小。

---

## 8. 结论

**"HSE 啥都没动的 mesh 会退化"这个现象的正解是 hourglass mode**,
不是 bug,是 Caramana compatible 离散的已知弱点。解决方案不是"修
Lagrangian 让它别 hourglass"(那是 Caramana 1998 附录的 sub-zonal
pressure trick,复杂且效果有限),而是"上真正的 ALE,让 mesh 不动"。

Eulerian Rezone + swept-edge Conservative Remap 是 hal3d 验证过的
主流路线,我们照搬即可。预计 500-700 LOC,1-2 天。

下一步:开 `cart_lag_remap.cu`(或类似),按 §5 顺序实施。
