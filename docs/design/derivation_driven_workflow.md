# Derivation-driven development workflow (athena_mhd & 类似物理扩展)

> **适用范围**:凡是在派生书(`docs/mhd_derivations/`)有节号的物理,落地到 solver 时都走这个流程。
> 不适用于纯工程改动(性能优化、IO、CLI、log 格式等)。
> 编写于 2026-05-08,源自 B-M1~M5.5 迭代经验。

---

## 0. 核心命题

派生书是 **spec**,solver 是 **实现**。两者角色不可混淆:

- 派生书(`docs/mhd_derivations/sections/*.md` + `scripts/*.py`)是**硬件无关、实现无关**的真相来源,内容是 sympy 验证过的方程和 identity。
- solver(`src/gpu/...`)是把 spec 在特定离散化 / GPU kernel / BC 下落地,并在测试中引用派生 identity 作为**阈值**。

**派生书里有但 solver 没做 = 待开发,不是 "bug";solver 做了但派生书里没写 = 工程层面的 implementation note,要补进派生书以免下次重踩。**

---

## 1. "新物理特性"迭代的标准 7 步

加入一个新物理(e.g. Spitzer 导热、Townsend 冷却、E1 driver、§B1 flux-tube 几何),**按顺序**走完下面 7 步。每一步没过不进下一步。

### Step 1 — 派生书先行

打开 `docs/mhd_derivations/sections/<x>.md` + `scripts/<x>.py`。

- 如果这节已经存在且 identity 够用 → 跳到 Step 2。
- 如果需要新 identity(比如新的闭式解、新的 ghost BC 行为、某个退化极限):**先在 sympy 里写出来、跑通 `assert_zero`**,再写 solver。写 kernel 之前不做任何 sympy 工作 = **最容易踩的坑**。
- 如果是**数值实现 gotcha**(不在 formal derivation 里但实测踩到的,比如 B-M4 的 ghost-T scalar mirror):落到 **"## 数值实现备忘 (not in formal derivation)"** 小节,不混入 formal 部分。

**输出产物**:更新后的 `*.md` + `*.py`,`python3 docs/mhd_derivations/scripts/<x>.py` 退出 0。

### Step 2 — Solver API 草案

在 `*_solver.cuh` 里加 state(默认关闭:bool `xxx_on = false`)+ 公共方法签名。
**不写实现**,仅让 header 能被 include 而不破坏现有测试的编译。

**输出产物**:header 改动 + 所有旧测试仍编译通过 + 行为不变。

### Step 3 — Kernel + method 实现

在 `*_kernels.cu` 写 `__global__ k_xxx_...` + `*_solver.cu` 写 host-side 包装。

**关键约束**:
- 任何 **on-by-default** 的行为改动禁止 —— 新功能必须 guard `xxx_on`。
- **退化测试先写**(见 Step 5 的 T1):当 `xxx_on=false` 时 bit-identical 旧行为;当参数退化(e.g. Λ₀=0)时也 bit-identical。这两条是 regression sentinel。

### Step 4 — 加 test 源文件 + CMake 条目

`tests/test_<solver>_<feature>.cu`,**一个 feature 一个文件**。套用现有 CHECK_LT 宏 + g_tests/g_failures 计数。

**阈值来源硬约束**:**每个 `CHECK_LT` 的 bound 必须能 trace 回一条 sympy 验证过的 identity**,或者标注 "numerical tolerance for X-stage dissipation"。**不接受 "我试了下大概是这个量级"**。
如果找不到 identity → 回 Step 1 补派生。

### Step 5 — 必跑的 4 类测试

每个物理特性至少覆盖这 4 类(对应 B-M3 Townsend 的 T1-T4 模板):

| 类别 | 检验 | 阈值来源 |
|---|---|---|
| **T1 闭式对照 / 数学极限** | solver 数值 vs sympy 派生的 closed form | identity |
| **T2 passive 性** | 和本物理无关的量不变(e.g. cooling 不改 ρ, mom, B, divB) | identity |
| **T3 单调性 / 2nd-law proxy** | 熵、能量符号、CFL 阶耗散 | identity |
| **T4 ghost BC / 物理守恒** | 域积分量,reflect/periodic 对对 | identity |

**物理性测试(可选但强烈推荐)**:如果 feature 描述了**波的产生或传播**(driver、wave,…),补一个"真物理发生了"的检验(例 B-M5 T5 Alfvén 发射 + 极化)。

### Step 6 — 全量回归

在 `build/` 跑**所有**同族 solver 的既有测试,**必须 0 回归**。

```bash
./test_athena_mhd_hse_preserve && ./test_athena_mhd_conduction \
  && ./test_athena_mhd_cooling && ./test_athena_mhd_combined \
  && ./test_athena_mhd_driver && ./test_athena_mhd_chromo \
  && ./test_athena_mhd_<new>
```

如果**任何** assertion 值变化(包括 "本来 0.000e+00 现在 1e-17"),**停**,查出是哪一行代码改的,判断是 bug 还是容忍变化(大多数是 bug,但比如 stride 改动导致 FP 加法顺序变动是可接受的)。

### Step 7 — 记录 + commit

- `docs/projects/mhd_verification/phase_B_benchmarks.md` 加一节 milestone(setup + 实测表 + 实现备忘)。
- commit message 用 `ADD:` / `FIX:` 前缀 + 一句话 + 测试结果行。
- 如果实现过程踩到的 numerical gotcha 派生书没说,**补回派生节的 "数值实现备忘"**(不是只留在 benchmark log 里就完事)。

---

## 2. 什么是"阈值必须 trace 到 identity"

反例(禁止):

```cpp
CHECK_LT(rel_err, 1e-3, "T matches analytic");   // 1e-3 是哪来的?
CHECK_LT(blob_decay_ratio, 0.95, "blob decays"); // 0.95 靠感觉定的?
```

正例:

```cpp
// §C7 Townsend: 闭式解 bit-identical 对应 kernel 公式,rel err 应 0 至 ULP
CHECK_LT(rel_err, 1e-10, "C7-T1: α=0.5 closed-form Townsend");

// §C6-T1: χ k² dt 线性衰减,VL2 PLM 在 k=2π/L, N=64 的已知空间 2 阶误差
// 约 (1/N)² ≈ 2.4e-4,留 100x 余量
CHECK_LT(rel_err, 2e-2, "C6-T1: parallel decay matches exp(-χk²t)");

// E1-T5a: Alfvén 到达时间;VL2 group velocity 在 PLM 下有 ~few% 相位误差,
// 加 reflect 回波干扰,保守 30% 容忍
CHECK_LT(time_err, 0.3 * tau_theory, "E1-T5a: Alfvén arrival at y*");
```

**每个 bound 前面必须有一行注释解释它从哪里来**。review 时看这行注释。

---

## 3. 派生书补 "数值实现备忘" 的触发条件

落地过程中发现的事情里,**哪些**必须回写派生书?标准:

✅ 需要回写:
- 离散格式下某个 ghost / BC 约定与连续派生不自洽(例 §C6 T-scalar-mirror)
- 一个 sympy identity 在浮点下要注意顺序才不失精度(例 §B4 per-stage defect)
- 多算子拼接时的 commute 顺序敏感(例 §C8 blend operator split)
- "这个派生给了连续 identity,但在 staggered grid 上要额外存 X" 的 data-structure 约束

❌ 不写进派生书,只留 benchmark log:
- 单次 kernel 的性能优化(block size、shared mem tile)
- 测试阈值的感性选择
- 配置参数推荐(IC 具体 A_rms、f_min 等)

---

## 4. 什么情况下可以**跳过**这套流程

只有一种:**bug fix**,且不涉及新物理。例:修一个 kernel 的 index off-by-one,但算法本身对。此时只跑全量回归就够。但 **regression 测试要证明修的对**(最小化 reproducer 固化到测试里)。

---

## 5. Phase B 证据(这套流程已经生效的地方)

| Milestone | Step 1 派生 | 补进书 | 回归 |
|---|---|---|---|
| B-M1 WB MHSE | §B4(已完) | 3 条数值备忘(per-stage / 6-var / prim) | 全过 |
| B-M2 Spitzer κ | §C6(已完) | 无(T ghost 问题 M4 才暴露) | 全过 |
| B-M3 Townsend cool | §C7(已完) | 无 | 全过 |
| B-M4 combined | — 组合测试 | §C6 补 T-scalar-mirror 备忘 | 全过 |
| B-M5 driver | §E1(已完) | §E1 补 SET-not-ADD + Parseval 离散归一化 | 全过 |
| B-M5.5 chromo | §C8(已完) | §C8 待补 operator split 非 exact 说明 | 全过 |

共 45/45 assertion,零跳过、零 hand-wavy 阈值。

---

## 6. Quick reference

```
新物理 → sympy identity → header state → kernel + method →
测试(T1 闭式 + T2 passive + T3 monotone + T4 BC + 可选 physics)→
全量回归 → benchmark log + 派生备忘(如果有 gotcha)→ commit
```

**一条违反即停,不硬推。**
