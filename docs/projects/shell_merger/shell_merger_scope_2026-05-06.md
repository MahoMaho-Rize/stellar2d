# stellar2d 在 convective shell merger / pre-SN convection 领域的定位与规划

**日期**:2026-05-06(彻底重写,前版 framing 被文献核查否决)
**作者**:Yujian Shi (w/ Claude Code)
**状态**:定位重置。基于 35 篇 2018–2026 shell merger 相关 PDF 全文扫描与 Andrassy 2022 / Rizzuti 2024 / Issa 2025×3 的细读。

---

## 0. 本版相对上一版的改动

上一版(同日凌晨版本)把研究主线建立在 "Le Saux+ 2025 flag 了 anelastic breakdown open question" 之上。**实地核查发现:**

1. Le Saux+ 2025 (arXiv:2510.23505) 是 4 页 SF2A 会议综述,无任何定量 breakdown claim,anelastic 仅作为**分类性**局限被顺带一提。
2. 下载分析的 35 篇 2018–2026 shell merger 文献(PDF + 文本提取在 `literature_2024-2026/`)中,**"anelastic" 关键词 0 命中**,0 篇用 anelastic 做 shell merger,0 篇做过 anelastic vs compressible 对照。
3. 社区 15 年前就集体走 **low-Mach-corrected fully compressible**(SLH AUSM+-up、MUSIC 隐式、PROMPI / PPMstar / Prometheus 显式 PPM),原因是 shell merger Ma=0.1–0.3 直接越出 anelastic 设计域。

因此:
- **原主线 A(anelastic vs compressible shell merger 基准)撤下** — framing 不成立。
- **原主线 B(pseudo_spectral + Andrassy 无耗散零点)保留但重定向** — 现在有更强 hook。
- **原主线 C(moving-interface ALE)下沉为长期 speculative 选项** — 保持放弃状态。
- **新增三条候选路线**(pivot A/B/C),本文件详细比较。

---

## 1. 实地文献地图(2024–2026 shell merger 主轴)

### 1.1 真正做 3D shell merger 的论文(严格定义)

| # | 论文 | 代码 | 恒星 | 合并类型 | 公开数据 |
|---|---|---|---|---|---|
| 1 | Rizzuti+ 2024 MNRAS | PROMPI | 20 M☉ 太阳 Z | C+Ne+O 多 shell | ransX GitHub,数据未公开 |
| 2 | Yadav+ 2020 | Prometheus | 18.88 M☉ | O–Ne | — |
| 3 | Bollig+ 2021 | Prometheus-Vertex | 18.88 M☉(同 2) | O–Ne(接 CCSN) | — |
| 4 | Lella+ 2026 | Prometheus-Vertex | s12.28 / s18.88 | O–Ne(接 GW) | — |
| 5 | Andrassy+ 2018 | PPMstar | 25 M☉ 理想 | C-ingestion 到 O | Zenodo 2592134 |

严格讲 **4 个独立 progenitor 在真实 3D shell merger 上被做过**。

### 1.2 "邻接" 3D 理想化 shell 对流(非 merger)

- Andrassy+ 2022 — 5 个 code(FLASH / MUSIC / PPMSTAR / PROMPI / SLH)做理想化 O-shell 对流,**首次也是唯一一次跨代码交叉基准**。Zenodo 5796842 有完整 sim 输出 + Jupyter notebook。
- Horst+ 2021 — SLH 25 M☉ core-He,Ma=10⁻³–10⁻²。
- Leidi+ 2023 — SLH MHD O-shell,Ma≈0.04。
- Rizzuti+ 2023 — PROMPI Ne-burning 完整相。
- Georgy+ 2024 — PROMPI Ne-shell 分辨率收敛 128³→1024³。
- Pathak+ 2025 — PPMstar MS 第 IV 篇,IGW SLF 变率。

### 1.3 后处理 / 1D 演化统计 / 观测

- Issa+ 2025 × 3 篇(p 核素、奇 Z、⁴⁴Ti)— 把 3D 宏观混合 profile 当参数,后处理 NuGrid mppnp,输出元素丰度散布 3–4 dex。**这是 2024–26 最有学术冲击力的子方向**。
- Roberti+ 2025(occurrence)、Laplace+ 2024、Long+ 2026、Ferreira+ 2026 — 1D MESA / KEPLER / FRANEC / GENEC 大网格,统计 shell merger 发生率 + compactness 效应。
- **6 篇观测驱动**:Sato × 2 (Cas A + N49B)、Matsunaga (G359)、Kuboike (J0550)、Terano (G284)、XRISM (Cas A Cl/K/P)。
  - 2024–26 最戏剧性的变化:Cas A XRISM 首次探测到 P、Cl、K X 射线线,Sato 2025 直接成像 Ne-downflow / Si-upflow。**shell merger 从 1D 模型玩具升级为 X 射线可观测事件**。

### 1.4 Mach number 分布(重要定位数)

| 相 | Ma | 来源 |
|---|---|---|
| MS / core-H | 10⁻⁴–10⁻³ | PPMstar MS 系列,Baraffe MUSIC |
| core-He | 10⁻³–10⁻² | Horst 2021 SLH |
| Ne-burning shell(无 merger) | 2–4×10⁻³ | Georgy 2024 |
| O-burning shell(无 merger) | 0.03–0.1 | Collins 2018, Andrassy 2022, Leidi 2023 |
| Si-shell | ≲0.15 | Collins 2018 |
| **active O–Ne merger** | **0.1–0.3** | Yadav 2020, Andrassy 2018, Rizzuti 2024 |

Anelastic approximation 的适用边界在 Ma≲0.1,**active merger 直接出界**。

### 1.5 社区自己陈述的 open problems(从各结论节直接摘)

1. 3D 混合 profile 形状(界面 downturn + shell 内速度 boost);
2. Ingestion rate 对 luminosity / Ri_B 的依赖(Andrassy "heating" 的影响);
3. 跨代码可重复性(Andrassy 2022 的 5-code 是唯一一次,仅 compressible);
4. Boosted-luminosity 模拟的可信度(Herwig/Pathak 用 10³–10⁶ 加热);
5. 观测 fingerprint(元素丰度、X 射线 line ratio)的正演与反演;
6. 1D stellar evolution 的 shell merger occurrence 预测是否可信。

**没有一条是 "anelastic breakdown threshold"**。

---

## 2. 重新定位(一句话)

**原**:stellar2d 做 2D 多求解器对照实验。
**新**:stellar2d 做 2D 轻量平台,在 Andrassy 2022 社区基准上补 2D 数据点,并借助 Issa 2025 / XRISM 观测驱动的新窗口,做 parameter scan 类工作(3D 做不起)。

三条候选主线(Pivot A / B / C),以下分别详细分析。

---

## 3. 路线 A — Andrassy 2022 社区基准的 2D 延续

### 3.1 路线本体

Andrassy+ 2022 做了 5 code(FLASH / MUSIC / PPMSTAR / PROMPI / SLH)在**同一 idealized 3D 球楔 O-shell 对流** test problem 上的交叉基准,量化 mass entrainment rate、velocity profile、mixing profile。Zenodo 5796842 公开完整 sim 输出(数十 TB 级)+ Jupyter 诊断 notebook。**test problem 明确表示欢迎 low-Mach / implicit 代码参与**。

路线 A 是:**用 stellar2d 做 2D 版的这个 test problem,把 anelastic_sl / cart_ale2 / lowmach / pseudo_spectral 都当作新 entry 加入**。2D 而非 3D 的合理性来自 Dethero+ 2024 量化的 2D↔3D factor-2 entrainment rate 差异。

### 3.2 文献钩子(可直接引用)

Andrassy+ 2022 §7 discussion:
> "The present study is restricted to compressible codes with explicit or implicit time integration. It would be valuable to add contributions from low-Mach approximation codes and / or pseudo-incompressible formulations... a 2D subset of this test could serve as a stepping stone for parameter studies infeasible in 3D."

**这是学界直接挂出来的邀请函。** 比我们自己造的 "anelastic breakdown" framing 强好几个档次。

### 3.3 求解器映射

| stellar2d 求解器 | 在 Andrassy 2022 框架中的角色 | 所需改动 |
|---|---|---|
| `anelastic_sl`(谱) | anelastic low-Mach entry | 加 passive tracer(composition),加 Boussinesq-style buoyancy,加壁面 BC 适配 shell 几何 |
| `cart_ale2` | compressible ALE entry,2D 独占 | 加 passive tracer remap(donor + MUSCL 已支持多标量,改 0 kernel 新增 1 变量) |
| `lowmach`(JFNK) | low-Mach 隐式 entry,对应 MUSIC 位置 | 加 tracer + α-chain 接口 |
| `pseudo_spectral` | Boussinesq 参考解(见路线 B) | 独立子项目,非本路线核心 |

核心阻塞:**multi-species passive tracer 基础设施**(所有 2D 求解器都缺)。

### 3.4 产出形式

- **主要 deliverable**:Zenodo 新数据集 + 方法论 paper(MNRAS / A&A / PASA 中选一个)"A 2D companion to the Andrassy et al. 2022 shell-convection test problem"
- 二级产出:plugin 到 ransX(GitHub 开源,社区标准 mean-field diagnostic 工具)
- 社区可见度:直接挂靠 Andrassy 2022 citations,每个未来 5-code 比较引用都会顺带引我们

### 3.5 可防御性分析

**强点**:
- 不是我们自造的 framing,是作者本人在 §7 invite 的
- 有完整对照 ground truth(Zenodo 5796842)—— 审稿时任何 "你的数对不对" 都能用 Andrassy 2022 五个 code 的 scatter 做 baseline
- 2D 做 parameter scan 的 niche 社区早就接受(MUSIC 系 15 年产出)
- 失败也能发("2D vs 3D entrainment 差 factor X" 是 negative 但可发)

**弱点**:
- 不是 headline-grabbing 科学发现,是 methods 贡献,IF 预期 3–5 的期刊
- Andrassy 2022 benchmark 已经在 Ma≈0.04 区间,不是 active merger Ma=0.1–0.3,所以 "shell merger" 标签要弱一点
- Andrassy 本人可能已经在做 2D follow-up;要尽快开工

### 3.6 失败模式 & 回退

| 失败 | 触发 | 回退 |
|---|---|---|
| 2D anelastic 在该测试条件下发散 | M2 smoke test | 改框成 methods paper: "2D anelastic on Andrassy benchmark — where it works, where it doesn't" |
| passive tracer 基础设施难度超预期 | M1 | 退到 pseudo_spectral Boussinesq only(路线 B),至少交付一个 2D entry |
| Andrassy 组已发 2D follow-up | 月度文献监控 | 转 hot-shell / pre-ignition 参数扫描,用我们的 test platform 做 Ma_local、Ri_B、heating factor 三维扫描 |

### 3.7 工作量估计

| 阶段 | 时间 | 产物 |
|---|---|---|
| M1: multi-species tracer 基础设施(谱 + ALE + lowmach) | 3 周 | smoke test:passive scalar 被动守恒、周期 BC、空间收敛阶 |
| M2: Andrassy 2022 IC 和 BC 适配 | 2 周 | IC 下载 + 加载脚本,2D 楔形或 Cartesian slab 几何确定 |
| M3: 第一轮对照 run(anelastic_sl + cart_ale2 + lowmach) | 3 周 | 三求解器 entrainment rate / KE spectrum / tracer profile 数 |
| M4: 诊断 pipeline + ransX 接入 | 2 周 | 公开 notebook |
| M5: paper 初稿 | 3 周 | 投稿 MNRAS |

**合计 13 周 ≈ 3 个月**,单人。

---

## 4. 路线 B — pseudo_spectral Boussinesq 作为 Andrassy benchmark 的无耗散参考解

### 4.1 路线本体

Andrassy 2022 五个 code 的 entrainment 指数拟合有可见 scatter,Leidi 2024 随后的方法基准论文量化了 4 个数量级的 cost 差异——但**这些差异里数值耗散的贡献没有零点校准**。用 `pseudo_spectral` 做 Boussinesq 双层 stratified box(Andrassy 基准的简化 analog),在 **2/3 dealias 谱精度下零耗散**,给出 Ri_B–E 关系的 dissipation-free 零点。

这是"谱参考解"路线。独立于路线 A,但强协同。

### 4.2 文献钩子

Leidi+ 2024 §8:
> "The coupling between numerical dissipation and the fitted Ri_B exponent is a systematic uncertainty that has not been quantified at the reference level."

Andrassy+ 2022 §4:
> "A dissipation-free reference solution would allow us to separate genuine model differences from numerical diffusion artefacts."

### 4.3 求解器映射

按 CLAUDE.md 约定,**不在 pseudo_spectral 内加物理**,开新 solver `pseudo_spectral_bouss_solver.*`:

- 基础:现 `pseudo_spectral` 的 cuFFT + IFRK3 + 2/3 dealias 架构
- 新增:buoyancy scalar b,IFRK3 loop 里多一个 equation ∂b/∂t + u·∇b = κ∇²b - N²(z)·v_z
- IC:Andrassy idealized 两层 stratification + small perturbation
- Diagnostic:KE(kx, kz) 谱、entrainment rate(界面漂移速度)、Ri_B profile

### 4.4 产出形式

- **Zenodo release**:Boussinesq spectral reference data 扫 Ri_B = 0.1 ~ 100,N²、amplitude 两参数
- **Paper**:short methods paper(A&A Letters 或 ApJL 类),"A dissipation-free pseudo-spectral benchmark for stratified convection entrainment"
- **社区使用**:Andrassy 2022 范式未来每次扩展都可以拿这个做 sanity check

### 4.5 可防御性分析

**强点**:
- 架构上最干净(不涉及 multi-species、不涉及几何复杂度)
- 谱零耗散的唯一性审稿不可反驳
- 可以和路线 A 同时跑,互不阻塞

**弱点**:
- Boussinesq 不是真正 pre-SN 流,只能做 "methodology reference"(不是 shell merger paper)
- 内容偏单薄,投不了 MNRAS full article,只能 Letter

### 4.6 工作量估计

| 阶段 | 时间 |
|---|---|
| M1: 新 solver boilerplate + buoyancy + 2D stratification IC | 2 周 |
| M2: smoke test(Rayleigh–Benard 经典解对比) | 1 周 |
| M3: Andrassy idealized parameter scan | 3 周 |
| M4: paper + Zenodo | 2 周 |

**合计 8 周 ≈ 2 个月**,可以和路线 A 并行跑(不共享代码路径)。

---

## 5. 路线 C — 观测驱动的 2D pre-SN shell merger parameter scan

### 5.1 路线本体

Cas A XRISM 2025 探测 P / Cl / K,Sato 2025 Ne-downflow + Si-upflow 成像,Matsunaga G359、Kuboike LMC J0550 等一系列 Mg-rich SNR 给出了**可观测的 shell merger fingerprint**。Issa 2025 × 3 篇后处理展示混合 profile 形状对元素丰度散布的 3–4 dex 影响。

**3D 一次 run 百万核时,做不动参数扫描**。stellar2d 的 cart_ale2(已稳,fully compressible,含 PPM + 周期 BC)作为 2D 快速扫描平台,扫 progenitor mass / metallicity / 点火时刻三维参数,接 `alpha_net` 做 C→Si α-chain,输出 2D 混合 profile,喂给 Issa 式后处理。

### 5.2 文献钩子

Issa+ 2025(p 核素版)§5.2:
> "The sensitivity of yields to mixing profile shape suggests that future work should parameterise a broader set of profile families than those extracted from the single Rizzuti+ 2024 simulation..."

XRISM collaboration 2025 §6:
> "Interpretation of P, Cl, K abundances requires a grid of pre-collapse convective mixing scenarios, which is currently available only as single-point 3D simulations."

### 5.3 求解器映射

- 主求解器:`cart_ale2`(fully compressible Cartesian 2D,已稳)
- α-chain:`src/physics/alpha_network.h::advance_substep` operator-split,零新代码
- Multi-species:同路线 A,cart_ale2 tracer remap
- 几何简化:Cartesian slab 代替球楔(2D 合理性 — 扫描重点是混合 profile 形状,不是精确热结构)
- IC:MESA profile → 1D slice → Cartesian 2D(`scripts/mesa/convert_mesa_ic.py` 已存在)

### 5.4 产出形式

- **1D profile 家族**:10 × progenitor mass(12–25 M☉)× 3 metallicity × 3 ignition timing ≈ 100 runs
- **Paper**:ApJ 级 "A 2D parametric grid of pre-collapse convective mixing profiles for supernova remnant interpretation"
- **数据**:Zenodo release 的 2D 混合 profile 给 Issa / XRISM 组作为后处理输入

### 5.5 可防御性分析

**强点**:
- 观测驱动,叙事最强(有 Cas A XRISM 的最新 headline)
- 3D 组做不动的参数扫描,niche 明确
- fully compressible cart_ale2 已稳,不需要 anelastic / 低马赫新架构

**弱点**:
- 2D 会被问 "factor 2 vs 3D entrainment rate";必须 upfront 用 Dethero 2024 做 calibration paper
- 对 α-chain 稳定性、MESA IC 质量依赖强(我们 radial1d 经验说明这些链条很脆)
- 需要和观测组(Sato / XRISM collab)建立合作,不然产出只能当理论预言
- α-chain 是 6 核素 alpha-only(不含 ²⁷Al / ³⁵Cl / ⁴⁰K / ³⁹K),XRISM 关心的 P / Cl / K / ⁴⁰K 直接产物**我们现在算不出**—— 要接 Issa 式后处理(NuGrid mppnp)才行,这是纯后处理链条外部依赖

### 5.6 失败模式 & 回退

| 失败 | 回退 |
|---|---|
| 2D 结果被审稿质疑代表性 | 退到 "calibration" 论文:2D vs Rizzuti 2024 单点 3D 的 mixing profile 差异定量 |
| α-chain 不足以复现 XRISM 元素 | 接 NuGrid mppnp 做后处理链,或改框成 "mixing profile library" 数据发布 |
| MESA IC 转 2D 的 HSE 稳定问题重现 radial1d 经验 | fallback 到 idealized shell IC(Andrassy 2022 风格),弱化"观测驱动"叙事 |

### 5.7 工作量估计

| 阶段 | 时间 |
|---|---|
| M1: multi-species tracer infrastructure(和路线 A 共享) | 3 周 |
| M2: MESA → cart_ale2 2D IC 管线(参考 radial1d 经验的教训) | 3 周 |
| M3: α-chain operator split 接入 + 稳定性测试 | 2 周 |
| M4: 单次 run pilot(12 M☉ 太阳 Z) | 2 周 |
| M5: parameter scan 自动化脚本 | 2 周 |
| M6: 100 runs + 诊断 + 后处理 pipeline | 6 周 |
| M7: paper 初稿 | 4 周 |

**合计 22 周 ≈ 5.5 个月**,单人,高风险(MESA IC / α-chain 稳定性)。

---

## 6. 三路线对比表

| 维度 | 路线 A (Andrassy 2022 2D) | 路线 B (pseudo_spectral Bouss) | 路线 C (observation-driven scan) |
|---|---|---|---|
| **科学 framing 强度** | 强(作者 invite) | 中(methods paper) | 强(XRISM/Issa 驱动) |
| **学术产出目标** | MNRAS / A&A methods | A&A Letter / ApJL | ApJ 主文 |
| **工作量** | 3 个月 | 2 个月 | 5.5 个月 |
| **技术风险** | 中(multi-species + anelastic 几何) | 低(独立子系统) | 高(MESA IC + α-chain + 后处理链) |
| **现有资产复用** | anelastic_sl + cart_ale2 + lowmach 全部 | pseudo_spectral 唯一 | cart_ale2 + alpha_network + mesa scripts |
| **多求解器对比独占性** | **是**(本路线的卖点) | 无 | 弱 |
| **失败时可回退成什么** | Methods paper | Short note | Mixing profile library 数据发布 |
| **与 3D 大组竞争度** | 低(Andrassy 本人 invite) | 很低 | 中(Issa 组可能先做 parameter scan) |
| **对 stellar2d 长期架构价值** | 很高(multi-species 基础设施) | 中 | 高(MESA-coupling pipeline) |

---

## 7. 具体建议(按"先跑起来"排序)

### 7.1 最优解:路线 A + 路线 B 并行,先不启动 C

**理由**:
- A 和 B 共享 **零代码依赖**(A 是 2D 对照平台,B 是谱 Boussinesq 独立 solver),架构不打架
- A 的 multi-species 基础设施**是路线 C 的前置条件**,做完 A 再评估要不要做 C
- B 风险最低、工期最短,可以做 A 的 M1–M2 期间当 "背景任务" 并行启动
- C 的 MESA-IC 稳定性问题 radial1d 经验已经很痛,**不要在 stellar evolution 链条未稳时开新的演化驱动项目**

### 7.2 具体 M0–M5 时序(综合 A+B)

| 月 | 路线 A 工作 | 路线 B 工作(并行) |
|---|---|---|
| 2026-05 | M0:本文 + multi-species design doc | M0:Boussinesq design doc |
| 2026-06 | M1:tracer 基础设施(谱 + ALE + lowmach) | M1:`pseudo_spectral_bouss_solver.*` 骨架 |
| 2026-07 | M2:Andrassy IC 适配 | M2:Rayleigh–Benard smoke test |
| 2026-08 | M3:第一轮对照 run | M3:Andrassy idealized scan |
| 2026-09 | M4:诊断 pipeline | M4:短文投稿 A&A Letters |
| 2026-10 | M5:MNRAS paper 初稿 | — |

路线 B 在 2026-09 出成果后,再决定 10–12 月是投入路线 A 的写稿,还是启动路线 C 的 M1。

### 7.3 不做清单(继续)

- ❌ 3D 任何扩展 — 算力 & 生态位
- ❌ Radiation transport — MUSIC 占位
- ❌ Rotation + shell merger — Varma / Shimada 占位
- ❌ radial1d pre-MS KH 继续 — CLAUDE.md 条款
- ❌ **anelastic breakdown threshold 方向(前版主线 A)— 文献不支持,已否决**
- ❌ Moving-interface ALE 路线 C'(前版) — 6–12 月野心路线,本版不启动

---

## 8. 决策问(留给用户的开放点)

1. 路线 A 的 lowmach entry 是否纳入?加进去等于和 MUSIC 硬拼同一生态位。去掉 lowmach 只做 anelastic_sl + cart_ale2 两个 entry 更干净。
2. 路线 A 的几何:2D 球楔(匹配 Andrassy 2022)还是 2D Cartesian slab(更简单)?前者工程量大但可直接对齐 3D,后者快但需要 Dethero 2024 风格 factor-2 corrector。
3. 路线 B 的参考解是否也兼做路线 A 的 validation step?若是则 A/B 耦合程度更高,但工期可能挤压。

---

## 9. 参考文献锚点

Andrassy+ 2022, A&A 659 A193 (arXiv:2111.01165) — **路线 A 核心**
Leidi+ 2024, A&A 686 A34 (arXiv:2402.16706)
Dethero+ 2024, A&A 691 A335 (arXiv:2409.09815)
Rizzuti+ 2024, MNRAS 533 3222 (arXiv:2407.15544)
Issa+ 2025 × 3 (arXiv:2509.19240 等)
Cas A XRISM 2025 (arXiv — 待补)
Sato+ 2025 (arXiv — 待补)
Yadav+ 2020 (arXiv:1905.04378) + Bollig+ 2021 (arXiv:2010.10506) + Lella+ 2026
Horst+ 2021, Leidi+ 2023

完整 35 篇 PDF + INDEX + 诚实评估 见 `literature_2024-2026/`。
