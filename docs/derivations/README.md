# stellar2d derivation books — project-level SOP

**状态**:2026-05-08 提出;首个示范实例是
`docs/derivations/mhd/`(MHD solver 开发前置手册,15 节 sympy 全验证,
23 页 PDF)。本文件把 **MHD derivation book 的方法论泛化**为项目级范式,
适用于所有 stellar2d 求解器的开发 / 扩建 / 回填。

---

## 为什么做这个范式(历史病因)

回顾 stellar2d 过去若干次"代数/物理 bug"经历,共同病因都是:

> **模型手推代数 → 编码 → 数值实验里 debug → 发现问题时已经写了 500 行 CUDA**。

具体实例(从 `docs/pitfalls.md` + session journals):

| 症状 | 根因 | 拦截它本应能看到的识别 |
|---|---|---|
| **P34** strang LM-HLLC 乘错 S* pressure jump | Rieper 2011 代数跳步 | sympy 验证 HLLC intermediate state 的 p* 表达式 |
| stellar_liouville $\mathsf{M} = \mathsf{L}^{-1}\mathsf{R}$ 装配失败 | 逐点除误用作逆算子 | sympy 对 Galerkin V_K + SL basis 的 $\mathsf{L}^{-1}$ 推导 |
| cart_ale2 swept-remap `ex` 符号错 | 外法向 vs 内法向混用 | sympy 推重建点的几何归属 |
| P33 local-compat Phase-M 补偿 | KE→IE 代数两次才做对 | sympy 验证各单元上 ΔKE = ΔIE 的 compatibility |
| anelastic_sl Chorin ω²=1 vs 0.8 | Chorin splitting 的本质 O(Δt) 偏差 | sympy 推 splitting 误差阶,提前说"做不到" |
| radial1d pre-MS KH 30Myr 跨不过 | Lagrangian + well-balanced 下 F_v≡0 | 物理-数值阻抗失配分析(非纯代数,但可文档化) |

这些 bug **在代码里 debug 一个要 2-5 天**,**在 sympy 里 ≤ 1 小时就能发现**。

---

## 范式的核心 (3 条强规则)

每个求解器在 `docs/derivations/<solver_name>/` 下建一本
"derivation book",遵守:

### 规则 1. 所有代数恒等式必须由 sympy 脚本生成

- 脚本末尾必须有 `assert sp.simplify(LHS − RHS) == 0`(通过 `assert_zero`
  helper,见 `docs/derivations/mhd/scripts/_common.py`)。
- **不允许**在 markdown 里写"通过代数可以得到..."这类跳步。
- 如果 sympy 不能符号化验证(嵌套根号、超越方程等),用**数值随机采样**
  (N ≥ 50 random states,atol ≤ 1e-10),并在 markdown 里**明确标注**
  "symbolic verification intractable — falls back to numerical".
  这是 Stone+08 Appendix B / Roe-Balsara 1996 等经典文献的做法。

### 规则 2. 每个脚本必须 self-check 且独立可跑

- 一个 `.py` 推一节的 proposition。
- **不允许**脚本间互相 import 中间结果。每个脚本从 `_common.py` 的
  符号定义开始重新推。
- 这样任一节都可以独立重跑 —— 前置条件是"sympy + numpy 可用",没有
  隐式依赖链。

### 规则 3. 每节必须连接到代码实现

markdown section 里强制有:

```markdown
> **code checkpoints:**
> `src/gpu/.../<file>.cu::<function_name>` (要实现的入口点)
> `tests/test_<solver>_<feature>.cu` (要加的回归测试)

## ✅ Verification checkpoint (to be wired)
{具体怎么把这节的 identity 锁进 ctest}
```

这样**手稿是 solver 实现的 single source of truth**,不是装饰性文档。
未来 kernel 注释里 `// Identity A4-SM: see docs/derivations/mhd/a4.md §eq:A4_SM`
可以往回追。

---

## 项目结构

```
docs/derivations/
├── README.md                      (本文件,meta-规范)
├── template/                       (新求解器克隆起点)
│   ├── scripts/_common.py          (符号 + LatexDump + assert_zero)
│   ├── run_all.sh                  (批跑 + pass/fail 报告)
│   └── build_manuscript.sh         (pandoc → PDF)
├── mhd/                            (首个完整示范,15+ sections,PDF ~23 页)
├── strang/                         (待补)
├── cart_ale2/                      (待补)
├── anelastic_sl/                   (待补)
├── pseudo_spectral/                (待补)
└── radial1d/                       (待补)
```

每个子目录结构与 `derivations/mhd/` 一致:

```
<solver>/
├── README.md            项目索引 + 四大部分 + 验证执行顺序
├── run_all.sh           跑所有 scripts/*.py,报告 pass/fail
├── build_manuscript.sh  拼接 sections/*.md → manuscript.{md,pdf}
├── scripts/             sympy 推导脚本(每节 1 个)+ _common.py
├── sections/            markdown 段落,每节对应一个 script
└── output/              sympy 脚本产生的 LaTeX snippets(可 \input)
```

---

## 新求解器启动清单

做一个新 solver 之前,必须先做这些事:

### 阶段 0 — 准入(在 pixi.toml 里放行前)
- [ ] 在 `docs/derivations/<new_solver>/` 建骨架(从 `template/` 克隆)
- [ ] 写 README.md 列出 **Part A/B/C/D**(见下方"Scope 划分")
- [ ] 第 1 节 `scripts/a1_*.py` + `sections/a1_*.md` 跑通 sympy

### 阶段 1 — 数学完备
- [ ] 所有核心方程组(守恒律、特征分析、离散格式、边界条件)都
      有对应 sympy 脚本 + markdown 章节
- [ ] 所有数值方法的**阶数、稳定性、CFL**都有符号推导(不是直接抄
      文献,而是自己 sympy 推)
- [ ] `bash run_all.sh` 全绿,`manuscript.pdf` 可审阅

### 阶段 2 — 代码绑定
- [ ] `src/gpu/.../<solver>.cu` 开建,每个函数的注释引用对应的
      `docs/derivations/<solver>/sections/<X>.md::<eq:Y>`
- [ ] `tests/test_<solver>_*.cu` 按 manuscript 里的 "Verification checkpoint"
      清单一条一条锁定
- [ ] CMake `add_gpu_test` 注册,`ctest -L fast` 里可跑

### 阶段 3 — 合主干
- [ ] PR description 必须列出:
    - 新 solver 的 derivation book 全路径
    - 所有 sympy identities 的清单(script 输出的 stdout)
    - 每个 identity 对应的 code checkpoint 是否已锁
- [ ] Review 时:reviewer **看 derivation book 和 kernel 的对应关系**,
      不是 line-by-line 读 CUDA。任一 kernel 改动**必须触发对应
      derivation 章节的重推**(或明确说"仅数值实现优化,代数不变")。

---

## Scope 划分(建议通用模板)

虽然每个求解器不同,但 Part A/B/C/D 的**结构**可以复用:

- **Part A — 物理方程 + 数值格式基础**:从物理原理推出守恒律,
  做守恒↔primitive 变换,推 flux Jacobian 及其 eigensystem / Riemann
  solver,证明离散守恒(例:MHD 的 A1-A5)
- **Part B — 几何降维 / 特殊 frame**:球坐标 → 1D,柱坐标,shearing
  sheet,moving mesh。几何源项 / 约束条件(flux conservation)
  (例:MHD 的 B1-B3)
- **Part C — 非理想 / 耗散项**:viscosity, conductivity, diffusion,
  cooling, heating。稳定性 + 正定性 + CFL。Closure relations
  (例:MHD 的 C1-C4)
- **Part D — 特殊应用场景**:shearing box, shearing-periodic BC,
  turbulence stress decomposition,特定物理的 limit cases
  (例:MHD 的 D1-D3)

---

## 已有求解器:回填策略

**不可能一次全重推**(工程量太大),但可以:

1. **触发式补推**:每次 carp 过一个 solver 的代码改动,顺手在
   `docs/derivations/<solver>/` 补一节。
2. **bug-修复捆绑重推**:以后发现任何求解器的代数 bug,**修复 PR 里
   必须附上对应 sympy 推导**。P34 应该做但没做。
3. **按重要性排序**:
   - **高优先级**(研究里常改):`cart_ale2`(Lagrangian + remap)、
     `anelastic_sl`(SL spectral)、`radial1d`(JFNK)
   - **中优先级**:`strang`(稳定但复杂 Riemann)、`pseudo_spectral`
   - **低优先级**:`wb2d / ale2d / cart_lag / fas / lowmach`(冻结,
     不再扩展)

建议每月**至少补 1 个 solver 的 Part A**,半年内高优先级求解器基本齐备。

---

## 常见异议 & 回应

### "这太慢了,我两天能写出来的 kernel 要花一周推代数"

- 那两天的 kernel **你之前已经 debug 了 3 天**(pitfalls.md 里全是证据)。
- sympy 推导不是"先推再写",而是"推的时候同步写"。A1-A5 的 sympy 全部
  是在思考 HLLD / CT 的过程中同步验证的,总工时 ≤ 写 kernel 本身。
- 真正费时的是**代数跳错 → 跑数值实验 → 看不到问题 → 回头怀疑 IC → 再改
  → 再跑**这个循环,sympy 一次就切断。

### "sympy 推不出来的怎么办?"

- 先退到**数值随机采样**(50+ 点,atol ≤ 1e-10)。这是 Stone+08 Appendix
  / Roe-Balsara 1996 等**经典文献自己用的办法**。
- 把"sympy 推不出" 明确写在 markdown 里,不要假装推出来了。
- 极少数情况(Chorin splitting 误差阶之类 asymptotic 分析),用**手写**
  但配 LaTeX 详细推导,并注明"sympy 不适用,手工推导已交叉查过 Wolfram"。

### "已有求解器没有这份文档,现在补太痛"

- 不要求一次到位。见上面"回填策略"。
- 关键是**新 solver 从 day 1 就走这套**,这样三年后整个 repo 都有了。

---

## 参考实现

`docs/derivations/mhd/` 是第一个完整示范:

- **15 sympy 脚本**,全部 `bash run_all.sh` 绿
- **14 markdown 章节**,每节有 "Verification checkpoint"
- **manuscript.pdf** 23 页,是 athena_mhd 求解器实现前的唯一真理源
- `scripts/_common.py` 是可复用的 helper(符号表 + LatexDump + assert_zero)

---

## 工具链

- `sympy >= 1.14`(已加到 `pixi.toml`)
- `pandoc >= 2.9`(已有)
- `xelatex`(已有,系统装 texlive-xetex texlive-fonts-recommended)

一次安装,永久受益。

---

## 从现在开始

**立即适用的约束**:
- 动 `athena_mhd`(MHD solver 实现阶段)必须在 `docs/derivations/mhd/`
  里引用对应 sympy identity。
- 任何新 solver 先建 derivation book 再写 kernel(**不是反过来**)。
- PR reviewer 权威要求作者引用 derivation 章节,没有则打回。

**鼓励性做法**:
- 碰到 already-shipped 求解器的代数疑问,顺手补一节 derivation。
- 开发讨论里用 "§A4 eq:A4_SM" 这样的锚点引用,比 commit hash 更持久。

这是给**未来的自己**(以及任何接手这个 repo 的人)留的最好礼物。
三年后回看,哪些求解器有 derivation book,哪些没有 —— 区别一目了然。
