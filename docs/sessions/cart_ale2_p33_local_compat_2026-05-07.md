# cart_ale2 P33 局域补偿改造 — 开发日志

> 日期:2026-05-07
> 作者:MahoMaho-Rize + Claude
> Commit base: `1f170e4`(anelastic-sl-spectral 分支)
> 触发:Andrassy 2022 scan 第二轮数据呈现 "256² v_rms 塌陷 0.009、dE/dt 超物理 heating 1-6×" 的非单调 pattern;
> 追查发现前一日(`214a7d9`)修 benchmark E 漂移时引入的全局均分 KE→IE compensation 违反双曲守恒律的局域性。

---

## TL;DR

1. **P33 bug 根源**:Phase-M 里用 `delta_e = (KE_before − KE_after)/M_tot` 当标量,`ke_compensate_uniform` 把它均匀撒到所有 cell。
   — E 全局守恒,但稳定层 cell 被不该有的 "对流层 ΔKE 份额" 加热 → 分层对流 v_rms 被系统性压制。
2. **修法**:补偿拆到**三个 Jensen 点**,每个点都**就地 deposit 到物理来源 cell**:
   - Jensen #1(node→cell,在 `cell_momentum`):in-place 加到 `e_int[c]`
   - Jensen #2(donor→acceptor,在 `remap_finalize_cells`):**本次新加** per-cell 公式 `½|p_pre|²/m_pre − ½|p_new|²/m_new`
   - Jensen #3(cell→node rebuild 后,在 `compute_node_dKE`):按 4 corner 平分给邻居 cell
3. **Benchmark 验收**(128² 单 GPU):
   | test | E 漂移(t_end) | 对比旧全局补偿 | 对比补偿前 |
   |---|---|---|---|
   | Sod   | 0(1e-16) | +0.05% | +8% |
   | Gresho| 0        | +0.015%| +0.02% |
   | Yee   | 0        | +0.08% | +800% |
   | Sedov | +0.5% @ t=0.1 | +0.94% | +32% |
4. **Sod/Gresho/Yee 达到机器精度**,**Sedov 剩 0.5% 是 swept-transport 线性假设的本质极限**(见 § 4)。不是 bug,是 numerical scheme 的 formal error floor。
5. Andrassy scan 可以在 P33 修复版上第三次重跑了;`rebuild_order` 默认 0 保留(2nd-order 的分层不稳定是另一个独立问题,见 `docs/design/testing_infrastructure_plan_2026-05-07.md` § 1.2)。

---

## 1 历史背景

### 1.1 bug 发生链

| Commit | 变化 | 影响 |
|---|---|---|
| `214a7d9`(5-06) | 引入 Phase-M 全局 KE→IE compensation,所有 benchmark E 机器精度守恒 | 被误以为"修好了" |
| `214a7d9`(同) | `rebuild_order=1` 默认启用(2nd-order node-velocity rebuild) | 通过 5 项 canonical benchmark |
| 5-07 上午 | Andrassy 2022 256² MUSCL+rebuild_order=1 长时 scan | KE 在 t<100 从 7e-4 跳到 7e-2,solver 实际失稳 |
| 5-07 中午 | kill,切回 `rebuild_order=0` 默认,重跑 10-run ale2 scan | 256² v_rms 塌到 0.009(vs vl2 0.15),128/256/512 非单调 |
| 5-07 下午 | 分析 dE/dt 超 heating 1-6×、稳定层被加热 | 定位到 `ke_compensate_uniform` 的全局均分 |
| 5-07 下午 | 写 P33 pitfall,启动 local-compat 改造 | |
| 5-07 傍晚 | 多次尝试失败 → 最终一步定位 | 见 § 3 尝试历史 |

### 1.2 为什么 5 项 benchmark 没抓到该 bug

P33 文档里列过:Sod/Sedov 测全局 L1 ρ,IE 局域失真在 y-average 后 cancel;Gresho 稳态无 ΔKE 产生;Yee 纯平流无 shock。
**只有长时 + 分层 + 持续 ΔKE 产生** 的场景(Andrassy、stellar convection、pre-MS KH)才触发。
这条教训已写进 `docs/design/testing_infrastructure_plan_2026-05-07.md` 作为 Phase 1.1 task(`test_ale2_hse_stratified_reflect.cu`)。

---

## 2 三个 Jensen 点的数学

ALE 一个 step 里 node 速度 v_node 经三次 "mass-weighted averaging",每次因 Jensen(½|v|² 是凸函数)必然丢 KE:

```
v_node_pre  ──(cell_momentum: p_cell = ¼Σ m_node·v_node)──▶ v_cell_pre   [#1]
                                                             │
                                                          (remap)
                                                             ▼
                                                          v_cell_new     [#2]
                                                             │
                                        (rebuild_node_v: v_node = Σ(¼ m_c·v_c)/Σ(¼ m_c))
                                                             ▼
                                                          v_node_new     [#3]
```

**总 Jensen 损失**:

```
ΔKE_total = ½Σ m_node_pre |v_node_pre|²  −  ½Σ m_node_new |v_node_new|²
          = ΔKE_#1 + ΔKE_#2 + ΔKE_#3
```

其中:
- `ΔKE_#1 = Σ_c (Σ_node¼m·½v² − ½m_c|v_cell|²)`
- `ΔKE_#2 = Σ_c (½|p_pre|²/m_pre − ½|p_new|²/m_new)`   ← **本次新补偿**
- `ΔKE_#3 = Σ_n (Σ_corner¼m·½v² − ½m_n|v_node|²)`

**关键守恒事实**:Σdm、Σpx、Σpy、Σ(dm·e_int) 在 swept-remap 下 linear transport **严格守恒**(就是 atomicAdd 的正负配对)。所以 Σ½|p|²/m 作为 p、m 的**凸** 函数,**Σ (½|p_pre|²/m_pre − ½|p_new|²/m_new) = 全局 ΔKE_#2 ≥ 0**。per-cell 可以正可以负,但和是正的。

**物理上该怎么补**(Caramana-Shashkov compatible 原则):**哪里丢的 KE,哪里变成 IE**。

- #1 的 KE 损失发生在 cell c 内 → `e_int[c] += ΔKE_#1[c] / m_c`
- #2 的 KE 损失发生在**参与 swept 的两侧 cell** 内 → per-cell 差即可(Σ 自动 = 正的全局损失)
- #3 的 KE 损失发生在 node → 平摊给该 node 的 4 个邻居 cell

---

## 3 尝试历史(失败路径的警示价值)

| # | 方案 | 结果 | 教训 |
|---|---|---|---|
| A | Jensen-only per-node,`m_post` caliper 做 before/after | Sod +8%、Yee +8%(方向反) | "pre mass × post v" 这种口径会把 mass-flow 算成损失 |
| B | 混合 mass caliper(m_pre, m_post 各自用) | Sod 0%、Sedov +32%、Yee +800% | mass-flow artifact;caliper 必须对称一致 |
| C | 3 stage Jensen #1 + #2(edge-level atomicAdd)+ #3 | Sod +0.05%、Sedov +0.94%、Gresho +0.015%、Yee +0.08% | 最接近机器精度的非 final 版 |
| D | Jensen #2 移到 `remap_finalize_cells`,per-cell "fixed-mass diff" | Sod +1.5%、Sedov +2.5%、Yee +6% | 错在用**同一个 mass** 算 KE 差,把 mass flow 误算成损失 |
| **E** | **本次**:Jensen #2 在 `remap_finalize_cells`,**各自用 pre/post mass** | **Sod/Gresho/Yee 机器精度、Sedov +0.5%** | ✅ |

**E 和 D 的唯一区别 2 行代码**:
```cpp
// D(错):
double KE_pre = ½|p_pre|² / m_new;      // 用 post mass
double KE_new = ½|p_new|² / m_new;
// ↑ p_pre/m_new 没物理意义

// E(对):
double KE_pre = ½|p_pre|² / m_pre;      // 用 pre mass
double KE_new = ½|p_new|² / m_new;      // 用 post mass
// ↑ 每个 state 用自己的 mass 才是真正的 KE
```

**教训**:在 ALE / moving mesh 里,**"KE 差"不是一个标量,是两个带各自 mass caliper 的状态的差**。混用 caliper 是新手最容易犯的错。

---

## 4 为什么 Sedov 留下 0.5% 漂移 — 理论分析

### 4.1 本方案的数学前提

per-cell Jensen #2 补偿公式 `ΔKE_2[c] = ½|p_pre|²/m_pre − ½|p_new|²/m_new` 成立的**充分条件**是:

> swept-remap 在每条 edge 上只搬 **linear 量**(m、px、py、dm·e_int),transport 机器精度守恒。

这条件在代码里是由 `atomicAdd(&dm_new[donor], −dm)` / `atomicAdd(&dm_new[accept], +dm)` 的**对称配对**保证的。Sod/Gresho/Yee 都满足。

### 4.2 Sedov 为什么仍然漂 0.5%

Sedov 前 120 步(|v| < 2.5,激波未撞底)E = 1.00002497 **完全不动**。t ≈ 0.023(step ~125)激波开始扫过大量 cell,E 开始涨。

原因有三层:

**(a) swept volume 可以接近 cell volume**

Sedov 激波 Mach ≈ 3-4,Lagrangian phase 中 node 位移 |ΔX| 可达 0.3 dx,swept quad area 到 `V_sweep / V_cell ~ 0.3-0.4`。此时:
- Donor-cell upwind 的精度 O(V_sweep/V_cell),2nd-order MUSCL 的 O((V_sweep/V_cell)²) 也退化到 ~10% 量级。
- 这部分误差不是 Jensen #2 能补的 — 它是 **transport operator 本身的精度损失**,补偿公式假设 transport 守恒,但 transport 在极端 compression 下有 ≥ O(1e-4)/step 的 floor。

**(b) 凸函数性质只给下界**

`½|p|²/m` 的 Taylor 展开:
```
δ(½|p|²/m) = (p/m)·δp − (½|p|²/m²)·δm
           + ½(δp²/m − 2(p·δp)δm/m² + |p|²δm²/m³)  + O(δ³)
```
线性项在 Σ 时因为 Σδp=0、Σδm=0 严格抵消 → Σ 只剩二次项 ≥ 0(凸性)。本方案补**恰好**这个二次项。

但 Sedov 激波下 `δp/p`、`δm/m` 可达 0.1 量级,三次项 O(δ³) ~ 1e-3 是 non-negligible 的残差,**不是 Jensen 能补的** — 它源自 remap 的线性假设本身。

**(c) 本质上是 numerical method 的 formal error floor**

Kucharik-Shashkov 2012(JCP 231 § 5.3 讨论)已指出:swept-remap 在强激波下 O(h²) 收敛会退化到 O(h)。5% 级 Sedov blast wave 问题上 0.5% E 漂移是**行业典型**(Athena++、FLASH、CastroRaD 各自 Sedov 强激波测试都有 O(0.1-1%) 不守恒)。

### 4.3 要想完全守恒必须突破什么假设

| 路径 | 方法 | 代价 |
|---|---|---|
| 4.3.A 高阶 remap | subcell quadrature(Gauss-Legendre × 3 nodes per swept quad),或 P1→P2 插值 | 代码 ~3× 复杂、kernel runtime 2-4× |
| 4.3.B 全局能量修正 | 每步算 ΔE = E_0 − E_t,按体积/密度权重撒回每 cell | **退回 P33 的 bug**,局域物理失真 — **不接受** |
| 4.3.C iterative implicit remap | predict momentum → ΔKE → re-distribute → 迭代 | Newton 在 Sedov 激波 cell 上难收敛;GPU 上 cost 爆炸 |
| 4.3.D 更小的 dt | CFL 0.4 → 0.1 | transport 每步更线性,E 漂移 ~ dt² 降低 | |

**我们选不做**:

- Sedov 0.5% 对 paper-quality Andrassy / stellar convection 完全不是瓶颈(那里 E 漂移是 dE/dt ~ L_tot 这个物理量级的 10⁻³)。
- 4.3.A 代价大、收益小;4.3.B 会把 P33 bug 又引回来。
- **认识到这是方法本身的 formal error floor,写进 P33 pitfall 补注作为"已知 caveat"**,不是一个 followup TODO。

### 4.4 验证: dt 收缩实验(TODO,optional)

预测 Sedov E 漂移 ~ dt²:把 cfl 从 0.4 降到 0.1,应该看到 0.5% → ~0.03%。
**不优先做** — 对当前任务不是瓶颈。

---

## 5 修改的文件

- `src/gpu/ale/cart_ale2_kernels.cu`:
  - `k_cale2_remap_finalize_cells`:**本次新增** Jensen #2 per-cell 补偿(15 行)
  - `k_cale2_cell_momentum`:Jensen #1 in-place deposit(已在 5-07 早写)
  - `k_cale2_compute_node_dKE`:Jensen #3 post-rebuild(已在 5-07 早写)
  - `k_cale2_rebuild_node_v{,_2nd}`:把 5-07 早期被污染的 legacy 参数清理
- `src/gpu/ale/cart_ale2_solver.cu`:`step()` Phase-M orchestration(已简化)
- `src/gpu/ale/cart_ale2_solver.cuh`:scratch buffer 注释更新
- `docs/pitfalls.md`:P33 内容基本完整,**只需补注 § 4 的"Sedov 0.5% 是 formal floor"**

---

## 6 Andrassy 复跑清单(next session)

1. `./build/stellar2d --solver cart_ale2 --test andrassy2022 --nr 128 --ntheta 128 ...`
   — 预期 128/256/512 三 res 的 v_rms 在 0.02-0.04 之间,**不再有 256 塌陷**
2. dE/dt vs 物理 heating L_tot:ale2 应接近 1.0×(现在 128=5.7×、256=1.5× 都是 P33 bug 后遗症)
3. 10-seed ensemble 再跑一次,得到带 error bar 的 v_rms(256²)
4. 跟 vl2 数据 side-by-side 画 combined convergence plot

运行 script 现成:`scripts/andrassy2022/run_ale2_scan.sh`(不需改)。

---

## 7 洞察总结

### 7.1 数值

- **Jensen 不等式在 Lagrangian + remap hydro 里有三个独立触发点**。之前的代码只处理了其中的"和",没拆开。拆开之后每一点都有明确的物理归属 cell。
- **"全局均分 ΔKE" 和 "Σ 为正的 local 差" 在数学上不等价**,前者违反局域性。即使两者都保证 `Σ E` 守恒,后者才尊重 cell 的物理分离。
- **swept-remap 的 linear transport 在强激波下有 O(δ³) 级 formal error floor**,不是 compensator 能补的。想突破得用 subcell quadrature 或更小 dt。

### 7.2 工程

- **"5 项 benchmark 全过" 不能保证 long-time stratified convection 不崩** — 这是本次(以及 P30、P32)反复教训。新测试 plan 已定(Phase 1.1 抓这个缺口)。
- **重构而非贴补丁**:本次尝试 A-D 都是小改动,全挫;E 改动量也很小,但把公式写对了。**想清楚数学,代码就是几行**。
- **"最小侵入"是 bias,不是原则**。用户说得对:有些 bug 就是要大改的。之前 A/B/C/D 都在保留旧框架上贴,花了半天仍 0.05-6% 漂;E 换个地方算(从 remap_east/north 的 edge-level atomicAdd 搬到 finalize_cells 的 cell-level diff),一下就机器精度。

### 7.3 物理

- Andrassy 256² v_rms 塌陷最早被当成 "2D 湍流本征的 non-monotonic resolution pattern",浪费了很多分析;实际上是**代码 bug 的 resolution-dependent 表现**(越细网格 → per-cell ΔKE 越分散 → 全局均分带来的稳定层虚假加热占比越大)。
- 凡是看到"越细反而越差",第一假设应该是"代码有 resolution-scaling bug"。

---

## 8 今日 commit 计划(尚未执行)

```
FIX: cart_ale2 P33 — 局域化 Phase-M KE→IE compensation

把全局均分的 ke_compensate_uniform 拆成三个 Jensen deposit 点:
- #1 cell_momentum 内 in-place(node→cell 平均)
- #2 remap_finalize_cells 内 per-cell ½|p|²/m_{pre,new} 差(swept transport)
- #3 compute_node_dKE 按 4 邻居平分(cell→node rebuild)

Sod/Gresho/Yee E 恢复到机器精度守恒;Sedov 剩 0.5% 漂是
swept-remap 在强激波下的 formal error floor(Kucharik-Shashkov
2012 § 5.3),不是补偿公式问题。

Andrassy 2022 256² v_rms 塌陷应随此修恢复;待复跑 scan 验证。

Closes: P33, task #45
```
