# MHD Reproduction Dossier

**Purpose**: 1:1 复刻 Suzuki Takeru 组近年 MHD 工作的 reproduction-grade 资料库。**不是综述,是可执行的复刻手册**。

**使用者**:任何将在 stellar2d 里新增 MHD solver 的开发者 —— 必须先读这份 dossier、通过 benchmarks.md 里的 all pass 门槛、然后才能动 suzuki_*.md 里的 science 复刻。

---

## 文件结构

```
docs/research_survey/mhd_repro/
├── README.md                           # 本文件(总索引 + gating logic)
├── standard_mhd_benchmarks.md          # 10 个 community 标准 MHD 验证基准(706 行)
├── suzuki_papers_setups_part1.md       # 2501.00294 RGB + 2203.15280 solar longitudinal(298 行)
└── suzuki_papers_setups_part2.md       # 2403.18409 solar non-ideal + 2305.12112 cyl MRI(311 行)
```

**总行数**:1315 行,约 13K 字;内含 53 个 equation references、40+ 个 numerical-values-to-match、4 个完整参数 sweep 表。

---

## 两级 go/no-go 门槛(**必须严格执行**)

新 solver 的开发必须走这个顺序,**跳跃开发等于 1:1 失败**:

### Gate 1:通过 standard_mhd_benchmarks.md 的 code verification 套件

**目的**:确认 MHD solver 的数值正确性,排除 bug。**不过不能进入任何 science 复刻。**

10 个 benchmarks 按难度递增:

| # | Benchmark | Dim | 最先做 | 最难过 | 锁什么 |
|---|---|---|---|---|---|
| 1 | Brio-Wu shock tube | 1D | ✓ | | 七波 Riemann 是否齐,compound wave |
| 2 | Ryu-Jones 1a/2a/3a/4a/5a | 1D | | | 多个 MHD shock 形态 |
| 3 | CP Alfvén wave 1D | 1D | | | 阶数收敛(应该 p=2) |
| 4 | **CP Alfvén wave 2D** | **2D** | | **★** | CT + 多维特征追踪 + coord 旋转一致性 |
| 5 | Orszag-Tang vortex | 2D | | | 多维 MHD 湍流图像 match Stone+08 Fig 25 |
| 6 | MHD blast wave | 2D | | ★ | 低 β positivity,B₀=1 时不崩 |
| 7 | MHD rotor | 2D | | | 高马赫旋转流 + 磁张力 |
| 8 | Field-loop advection | 2D | | | ∇·B = 0 保持、CT 正确性 |
| 9 | Linear MHD waves conv | 1D | ✓ | | fast/slow/Alfvén/entropy 分别 p=2 |
| 10 (opt) | Torrilhon | 1D | | | compound wave 处理 |
| 11 (opt) | MHD KH | 2D | | | 磁场 regularize KH 混合 |

**Gate 1 pass 条件**:
- 所有 1D shock tube L1 error 符合 Stone+08 Fig 28-29 给定剖面
- Linear wave convergence slope > 1.9 for fast/Alfvén(符合 2nd-order 目标)
- CP Alfvén 2D convergence slope > 1.9(**最可能先掉的一项**)
- Orszag-Tang 在 N=128² 时的 ρ 剖面与 Stone+08 Fig 25 在 L∞ 范数下差距 < 2%
- MHD blast B₀=1 时 positivity 不崩(**HLLC 无法过,必须上 HLLD 才行**)
- Field-loop 在 100 个 crossing 后 |B_z| / |B_0| 漂移 < 1%

### Gate 2:通过 Suzuki 论文 1:1 复刻(每篇独立验证)

**每一篇论文的复刻都必须先拿到一组 paper 里给出的 "single data point" 匹配**,而不是一次跑完整 sweep。按下面这 4 个"最小可复刻单元"顺序验证:

#### 2.1 最轻 — Shimizu+2022 (`suzuki_papers_setups_part1.md` §Paper 2)

- **最小可复刻单元**:Case `BsV00`(baseline)
- **必须 match 的 3 个值**:
  - Ṁ = 1.32 × 10⁻¹⁴ M☉/yr
  - v_out = 688 km/s
  - T_corona ∈ [1.0, 1.2] × 10⁶ K at r ∈ [1.05, 1.1] R☉
- **所需物理模块**:1D 球形 ideal MHD + Alfvén 波注入 BC + 部分电离 H EOS + Saha + 辐射冷却(Sutherland-Dopita)+ 超径向 flux-tube A(r)
- **工作量估计**:solver 已就绪之后 2-3 周
- **原因这是最轻**:ideal MHD、setup 最 canonical、有完整 14-case sweep 表可以二次验证

#### 2.2 中 — Matsuoka+2024 (`part2.md` §Paper 3)

- **最小可复刻单元**:Case `M0`(ideal 参考运行)
- **必须 match**:Ṁ = 2.04 × 10⁻¹⁴ M☉/yr 至 ±5%
- **扩展**:Case `M3`(完全非理想)Ṁ = 3.52 × 10⁻¹⁵,验证 ideal → non-ideal 6 倍压制
- **所需物理模块**:前置 + Ohmic + ambipolar diffusion 源项
- **工作量估计**:Shimizu+2022 复刻 OK 之后 1-2 周加 non-ideal 项

#### 2.3 重 — Suzuki+2025 RGB wind (`part1.md` §Paper 1)

- **最小可复刻单元**:α Boo fiducial (B=0.65 G, Z=0.3 Z☉, non-ideal)
- **必须 match**:⟨Ṁ⟩ = 3.3 × 10⁻¹¹ M☉/yr;⟨v_out⟩ = 77 km/s
- **物理模块**:前置 + LTE Saha 多金属电离 + 对流湍动 boundary driver
- **工作量估计**:Matsuoka+2024 OK 之后 2-4 周

#### 2.4 最重 — Suzuki+2023 cylindrical MRI (`part2.md` §Paper 4)

- **最小可复刻单元**:Cartesian 控制运行(非圆柱)—— 有 S19/S21 published α_ss baseline 可对
- **必须 match**:α_M ≈ 0.072、α_R ≈ 0.016(Cartesian baseline)
- **物理模块**:前置 + 3D cylindrical shearing box + shearing-periodic BC + net-flux 初值
- **工作量估计**:所有前三篇 OK 之后 2-3 个月
- **风险最高**:要 3D,GPU memory 成本(256×320×256 ≈ 5M cells × 8 vars),turbulent statistics 需要 300 rotation 长运行

---

## 3 个 "gotcha" 重点提醒

### G1. HLLC → HLLD 不是 trivial swap
文献只说"换掉 Riemann solver",但:
- **HLLD 要 8 个守恒变量**(hydro 5 + B_x, B_y, B_z),而不是 5 个
- **需要另外的 signal-speed 公式**(fast magnetosonic,不是 hydro sound speed)
- Athena++ 的 `hlld.cpp` 里有 4 个 intermediate-state 公式,**每一个都必须抄对**,一个符号错了结果还可能 "看起来对"(smooth waves 会过,但 shock 内部就崩)

### G2. Constrained Transport 不是可选
- **cell-centered 方案发散清洁会失败**,Dedner GLM 也勉强;Gardiner & Stone 2005 的 CT corner EMF 是 athena_vl2 + MHD 唯一 robust 的路线
- B_x, B_y 必须 face-staggered,不是 cell-centered
- corner EMF 需要 4 点 average(GS05 Eq. 30 或 Stone+08 Eq. 42)—— **错了 field-loop 就漂移**,是 Gate 1 #8 必挂的源头
- **绝不要**把 B_x/B_y 放进 athena_vl2 的 cell-centered hydro 存储;要新开面心数组

### G3. 1D / 2D / 3D 拓扑不同
- Suzuki 风(Papers 1-3)是 **1D spherical + super-radial flux tube A(r)** —— 不是 2D / 3D。**不能照搬 athena_vl2 2D Cartesian setup**
- MRI(Paper 4)是 **3D cylindrical shearing box** —— athena_vl2 目前只支持 2D Cartesian,**需要两次架构扩展**
- **先做 1D 版本**(加 A(r) geometrical source term 很简单),拿 Gate 2.1 / 2.2 过了再考虑 3D

---

## 下一步执行路径

### 路径 A —— 稳妥(推荐)
1. 新开 `src/gpu/explicit/athena_mhd_solver.{cuh,cu}` + `athena_mhd_kernels.cu`(按 CLAUDE.md "不可覆盖"规则,不动 athena_vl2)
2. 实现 Gate 1 #1 Brio-Wu shock tube,走通整个 HLLD Riemann + CT 框架
3. Gate 1 #9 Linear wave convergence(最能早期发现 order 掉阶)
4. Gate 1 #8 Field-loop(早期发现 ∇·B = 0 违反)
5. Gate 1 #4 CP Alfvén 2D(阶数 + CT + 多维一致性,**最可能先失败,提早暴露**)
6. Gate 1 其它 + Gate 2.1 Shimizu+2022 baseline

**预期时间**:Gate 1 全套约 **4-6 周**(一人全职),Gate 2.1 baseline 再 **2-3 周**。

### 路径 B —— 激进
跳过完整 Gate 1,**只做 #1、#9、#4** 三个 key benchmark,直接上 Gate 2.1 —— 风险是 Gate 2 遇 bug 时不知道是物理问题还是 solver 问题,debug 成本反而高。**不推荐**。

### 路径 C —— 走 radial1d 1.5D
`radial1d` Lagrangian 1D 加 B_φ 做 1.5D flux-tube MHD(Suzuki-2006 numerics),绕过 HLLD + CT 复杂度。但这样**只能做 Gate 2.1 + 2.2 的 1D 部分**,Paper 4 的 MRI 完全做不了。也就是说**最终还是要做路径 A**。

**推荐**:**路径 A**,不走捷径。

---

## 每次 PR 必须引用本 dossier

动 `src/gpu/explicit/athena_mhd_*` 任一文件的 PR,description 必须明确:
- 正在 close 哪条 Gate(例如 "Close Gate 1 #4 CP Alfvén 2D convergence")
- 测量值 vs 论文参考值(数字对数字)
- 如果跑的是 science 复刻,必须引用具体 suzuki_papers_setups_*.md 里的 §P8 numeric 值

**不允许 "差不多" 式的复刻**。如果数字不 match,要么找 bug 要么 document 差距并给出物理解释。

---

## 已知 verification 缺口

以下几项在 deep-read 里被标记为 "paper 不明确" —— **新 solver 开发早期必须先通过邮件问原作者或查 GitHub 原码**:

1. **Ryu-Jones RJ1a / 3a / 5a** 的精确 IC 在原 ApJ 1995 PDF 里 JBIG2-compressed,未解压;Stone+08 只给 2a + 4d。→ 查 Athena++ `athinput.ryu*` 或 PLUTO test suite
2. **Matsuoka+2024** 精确径向网格数 N_r 未给(inherited from Shimizu+2022),需读 Suzuki+Inutsuka 2005
3. **Matsuoka+2024** Sutherland-Dopita Λ(T) 表未再现,需查原表
4. **Suzuki+2023** MRI 的 Riemann solver family + slope limiter + CFL 未明说,需读 S19 代码或 email 作者
5. **Suzuki+2025 RGB** 所有径向剖面只有图无表,需要 email Takeru 要 data products
