# Phase A — 2D MHD solver 物理可信度 benchmark 计划

**日期**:2026-05-08
**分支**:`athena-mhd-solver`
**求解器**:`src/gpu/explicit/athena_mhd_*`(VL2 + HLLD + CT)
**目标**:在不修改求解器代码的前提下,量化当前 2D MHD solver 的
多维耦合正确性、长时稳定性、数值耗散 —— 为后续物理题(Suzuki
-complement 方向)提供"可信分辨率 / 可信 Reynolds 数"的硬数据。

**不包含**:任何物理模型扩展(重力、导热、冷却、驱动)。这些留给
Phase B / C。

---

## 动机

现有 10 个 benchmark(Brio-Wu / RJ2a / RJ4d / CPAW 1D 收敛 /
CPAW 2D / OT / MHD blast / MHD rotor / field loop / linwave 收敛)
只验证了**离散层面**的三件事:

1. 收敛阶 ≈ 2(空间 + 时间)
2. ∇·B 在机器精度
3. 守恒量 / 对称性在合理容差内

**这些不等于"物理可用"**。未验证的关键问题:

- **A1**:2D 维度耦合是否正确?斜向传播的线性波会不会暴露 x / y
  方向分离式处理中的 bug?
- **A2**:发展到湍流后,数值耗散率有多大?对应的有效 Reynolds / 磁
  Reynolds 数是多少?
- **A3**:∇·B 在 100× 长于现有测试的时间窗内是否漂移?CT telescoping
  identity(推导书 §A5)在**有限 round-off** 下真的机器精度吗?
- **A4**:波模在长时间(数十周期)后的衰减率,是我们能量化物理题中
  Alfvén 波损耗的唯一方式。

**产出**:数值耗散表 + 三张诊断图 + Markdown 报告。
后续每个物理题的"结论置信区间"都引用此表。

---

## 四个 benchmark

### A1 — Oblique linear wave (2D 维度耦合)

**文献**:Stone+08 §6.2, Table 3。

**设置**:
- 周期域 Lx = 2, Ly = 1;波矢 **k** = (2π/Lx, 2π/Ly) → 斜向 k 与网格
  对角线不平行。
- 4 modes:fast / Alfvén / slow / entropy。
- 分辨率:N ∈ {32, 64, 128},Ny = N/2(保持 Lx/Ly = 2)。
- 一个完整周期,和 IC 做 L¹ 差。

**需要新增**:
- `AthenaMHDSolver::init_linear_wave_oblique(mode, kx, ky)` — 沿任意
  k 方向的右特征向量(来自推导书 §A3,乘以旋转矩阵)。
- `tests/test_athena_mhd_linwave_oblique.cu` — 4 × 3 网格,提取 slope。

**通过判据**:每个 mode slope ∈ [1.8, 2.2]。

**为什么重要**:`athena_mhd` 的 x-sweep 和 y-sweep 是分离 flux,
然后 CT corner EMF 做 4 点平均。若平均权重错了 / CT 的 GS05 分量
混用错了,1D 测试过,斜向 2D 会直接失败。

---

### A2 — Orszag-Tang 谱 + 有效 Reynolds

**文献**:Orszag-Tang 1979;Stone+08 Fig 22-23;Dai-Woodward 1998。

**设置**:
- 标准 OT IC(已有 `init_orszag_tang`)。
- 分辨率扫描:N² ∈ {128², 256², 512²}。
- 运行到 t = 0.5(湍流饱和时刻,Stone+08 用此)。
- 后处理:2D FFT 求 KE(k) 和 ME(k),取轴对称平均。

**需要新增**:
- `scripts/analyze_mhd_spectrum.py` — VTK / HDF5 → 2D FFT → E(k) 曲线。
- VTK 输出已有;确认 `init_orszag_tang` 输出 v_x, v_y, B_x, B_y 到
  VTK。

**输出**:
- 每个 N 的 log-log 谱图,标注惯性段 slope 和耗散尺度 k_diss。
- 耗散尺度定义:E(k) 下降到峰值 10% 的位置。
- ν_eff 估计:Kolmogorov phenomenology k_diss^{-4/3} ε^{1/3} = ν_eff,
  其中 ε 从 dKE/dt 测。

**通过判据**:
- 128² 到 512² 谱形自相似(惯性段范围增宽 2-3 倍)。
- k_diss 的分辨率 scaling ≈ N^{0.75}(Goldreich-Sridhar 预期 N^{3/4})。
- ν_eff 给出定量数字,写进最终表格。

**为什么重要**:所有 2D MHD 湍流题(包括方向 1 的 Alfvén 湍流)都要
预先知道这个数。"物理 Re 必须低于数值 Re" 是可信度底线。

---

### A3 — Field loop 长时 ∇·B + 磁能衰减

**文献**:Gardiner-Stone 2005 §4.3;Stone+08 Fig 24。

**设置**:
- 现有 `init_field_loop`(已有测试 `test_athena_mhd_field_loop.cu`),
  但现有测试只跑 2 次穿越。
- 扩展到 **10 次对角穿越**(t = 10),每 0.5 s 记录
  max|∇·B|、loop 内磁能 ME_loop、loop 中心场强 |B_loop|。
- 固定 N = 128。

**需要新增**:
- `tests/test_athena_mhd_field_loop_long.cu` — 复用 field-loop IC,
  加 100 步 snapshot + CSV 输出。
- `scripts/plot_field_loop_long.py` — divB(t), ME_loop(t) 曲线 +
  指数拟合 η_eff。

**通过判据**:
- max|∇·B|(t) 在 [0, 10] 全程 < 10⁻¹⁰(CT telescoping 精度维持)。
- ME_loop 指数衰减 ME ∝ exp(−η_eff k_loop² t),η_eff < 10⁻⁴ at N=128。
- 如果 divB 缓慢漂移 → CT round-off accumulates,需要记录 slope。

**为什么重要**:
- ∇·B 的 round-off 累积是**没人认真测过的**。CT 在代数上 exactly 保,
  但浮点舍入会不会在 10⁴ cell-updates 后累加到 1e-12?这个数字必须
  知道。
- 磁衰减率是"未运动"场下的 η_eff 底数,A4 的动态 Alfvén 波结果可以
  跟这个对照。

---

### A4 — CP Alfvén 波 2D 多周期衰减

**文献**:Tóth 2000;Stone+08 §6.2。

**设置**:
- 现有 `init_cpaw_2d`(traveling)已有测试,但只跑 1 周期。
- 扩展到 **10 周期 和 50 周期**,两个运行。
- 分辨率扫描:N ∈ {32, 64, 128} × Ny=N/2(原 CPAW 域 Lx:Ly = 2:1)。
- 每个 N + 两个时长 = **6 组运行**。
- 诊断:δB⊥ 包络振幅(沿波矢方向 FFT 取主模幅度)vs t。

**需要新增**:
- `tests/test_athena_mhd_cpaw_longtime.cu` — 循环 N × t_end,输出
  `t, A_y, A_z, divB` CSV。
- `scripts/plot_cpaw_decay.py` — 指数拟合 A(t) = A₀ exp(−γ_num t),
  导出 γ_num(N),拟合 N^{-2} scaling → η_eff 定量值。

**通过判据**:
- 振幅衰减率 γ_num 随 N 呈 N^{-2} scaling(2 阶 solver 应给出
  ν_eff ∝ Δx²)。
- η_eff(N=128) 数字明确写入表格。
- 50 周期后仍然稳定(不发散、无 NaN、max|∇·B| < 10⁻¹⁰)。

**为什么重要**:
- 物理意义上的 Alfvén 波损耗率。后续方向 1("2D 分层 Alfvén 湍流")
  的物理耗散率必须 >> 这个数才有意义。
- 50 周期是 Suzuki 类问题中"driver 对应的 Alfvén 周期数"的下限
  (典型 1.5 hr 波在 30-day 跑里是 ~500 周期)。

---

## 数值耗散表格式(最终产出)

运行完 A2 和 A4 后,汇总成:

| 测试 | N | Δx | 测得参数 | 物理对应量 |
|---|---|---|---|---|
| A2 OT | 128 | 1/128 | k_diss, ν_eff | 湍流惯性段 |
| A2 OT | 256 | 1/256 | ... | ... |
| A2 OT | 512 | 1/512 | ... | ... |
| A4 CPAW | 32 | 1/32 | γ_num, η_eff | 线性波耗散 |
| A4 CPAW | 64 | 1/64 | ... | ... |
| A4 CPAW | 128 | 1/128 | ... | ... |
| A3 loop | 128 | 1/128 | η_eff_static | 磁通量衰减 |

"物理要求的 Re / Rm" 与这个表比较后,就能直接说
"N ≥ 256 下此物理题可信"。

---

## 时间估计

| Benchmark | 新代码 | 运行(GPU) | 分析 | 合计 |
|---|---|---|---|---|
| A3 field loop long | 1 个测试 | ~30 s | 0.5 h | 2 h |
| A4 CPAW long-time | 1 个测试 × 6 runs | ~3 min | 1 h | 3 h |
| A2 OT spectrum | 1 个分析脚本 | ~30 min (512²) | 2 h | 4 h |
| A1 oblique linwave | 1 IC + 1 测试 | ~1 min | 1 h | 3 h |
| **合计** | | | | **~12 h**(1-2 天) |

---

## 执行顺序

**按工作量递增**,边做边积累信心:

1. **A3 field loop long**(最轻,纯扩展现有 ctest)→ 验证 divB round-off 行为
2. **A4 CPAW long-time**(IC 不变,scan 配置)→ 出 η_eff 数字
3. **A2 OT spectrum**(分析脚本为主)→ 出 ν_eff 数字
4. **A1 oblique linwave**(需要写新 IC)→ 最终正确性证书

每步完成都提交一个 commit,跑坏了方便 bisect。

---

## 成功标准

**Phase A 整体通过的充要条件**:

1. ✅ 所有 4 个 benchmark 的测试 / 脚本都在 `ctest` 里,且全部 PASS。
2. ✅ `docs/projects/mhd_verification/phase_A_results.md` 写成,含
   数值耗散表 + 三张图。
3. ✅ 表格数字全部合理(η_eff N=128 ≈ 10⁻⁴ 量级,ν_eff ≈ 10⁻³
   量级,OT 谱斜率在 [-5/3, -3/2] 之间)。
4. ✅ 若某项不合理,定位原因(bug?算子阶太低?recipe 不对?)
   并记录在 `docs/pitfalls.md`。

达到这四条后,**才有资格**进入 Phase B(KH)和 Phase C(重力 +
分层 Alfvén 波)。不达到的话 Suzuki-complement 方向的任何"物理
结果"都是海市蜃楼。

---

## Phase B / C 预告(后续)

- **Phase B**:加 1 个 IC(双 KH),测 B‖ 和 B⊥ shear。~2 天。
- **Phase C**:加重力源项(求解器需修改)+ 3 个 benchmark
  (HSE 长时、MHSE 长时、Hollweg 1982 Alfvén 分层传播)。~3 天。

过了 C 就能做 PhD 申请 proof-of-concept("2D 分层 Alfvén 湍流 + AD"
方向 1,见 `docs/research_survey/research_plan.md`)。
