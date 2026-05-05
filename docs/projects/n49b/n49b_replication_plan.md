# N49B 论文复刻方案

**目标论文**: Sato, Matsunaga, Sawada, Takahashi, **Suwa**, Hughes, Uchida, Umeda
"Destratification in the Progenitor Interior of the Mg-rich Supernova Remnant N49B"
ApJ 2024, arXiv:2403.04156

**目的**: 完全复刻论文所有计算结果,作为申请 Suwa 老师的 introduction 材料,并建立继续做 2D 验证的基础。

**分支**: `n49b-replication`
**创建日期**: 2026-05-04

---

## 1. 论文计算内容分解

论文实际上做了两件事:

### A. 数据库 post-processing(论文 §3,主体)
- **输入**: Sukhbold, Woosley & Heger (2018) 公开数据集 (Harvard Dataverse DOI 10.7910/DVN/VOEXDE)
  - 1499 个 KEPLER 1D pre-SN profiles
  - ZAMS mass 范围 12–27 M☉
  - 每个 profile 包含:(r, m, ρ, T, P, X_i for ~20 elements) + convection history
- **输出**: Fig 2、Fig 4、Fig 5、Fig 6、Table 2
- **计算**: 纯 Python post-processing,不解方程

### B. 1D 爆炸性核合成模拟(论文 §A Appendix)
- **代码**: Sawada & Suwa (2023) 自研 1D Lagrangian hydro(arXiv:2301.03610)
- **方程**: 1D 球对称 Euler + light-bulb ν 源 + α-network
- **输入**: Sukhbold profile 作为 IC(12.02 / 12.75 / 15.28 / 15.90 M☉)
- **输出**: Fig 7 — pre-SN 与 post-SN mass fraction 对比
- **目的**: 验证 explosive nucleosynthesis 不会洗掉 pre-SN Mg/Ne signature

---

## 2. 我们的起点与缺口

### 已有(无需新增)
| 能力 | 文件 |
|---|---|
| 1D Lagrangian 隐式 hydro | `src/gpu/radial1d_*.cu` |
| Helm EOS | `src/physics/helmholtz_eos.cuh` |
| MLT | `src/gpu/radial1d_solver.cu` (`--mlt`) |
| pp-chain 核反应 | `src/physics/nuclear_pp.h` |
| opacity 表 | `src/physics/opacity_table.cu` |
| MESA profile reader | `scripts/mesa_profile.py` |
| h5py 依赖 | `pixi.toml` (PR #3 已加) |

### 需要新增
| 缺口 | 位置 | 估工时 |
|---|---|---|
| KEPLER pre-SN profile reader | `scripts/sukhbold_reader.py` | 1-2 天 |
| Compactness / M₄ / μ₄ 计算 | `scripts/stellar_structure_params.py` | 0.5 天 |
| O-rich layer 组成积分 | `scripts/composition_analysis.py` | 0.5 天 |
| Kippenhahn 图绘制 | `scripts/plot_kippenhahn.py` | 1 天 |
| α-network (~13 核) | `src/physics/alpha_network.{h,cu}` | 1 周 |
| Light-bulb ν 源 | `src/physics/light_bulb.h` | 1-2 天 |
| Explosive nucleosynthesis driver | `scripts/run_explosive_nucleo.py` | 2-3 天 |

---

## 3. 执行方案(三阶段)

### Stage 1 — 纯数据复刻(约 1 周)

**目标**: 在我们仓库里重现 Fig 2、Fig 4、Fig 5、Fig 6、Table 2。
**风险**: 低。纯 Python + 公开数据。
**Deliverable**: `docs/n49b_replication_stage1.md` 并排对比图。

执行顺序:
1. 下载 Sukhbold+2018 Dataverse(Task #16)
2. 写 KEPLER profile reader,单元测试选 5 个模型(Task #19)
3. 复现 Fig 2(compactness vs M_He_core)—— 最简单,先做(Task #11)
4. 复现 Table 2(Mg-rich rate by mass bin)(Task #14)
5. 复现 Fig 6(Mg/Ne vs Si/Ne scatter)—— N49B 观测点对比的核心(Task #20)
6. 复现 Fig 4/5(Kippenhahn + mass fraction)—— 最复杂,最后做(Task #10)
7. 写 stage1 report(Task #13)

**验收**: 和论文图**肉眼一致**(浮点 roundoff 以内),Table 2 数字完全一致。

### Stage 2 — 1D 爆炸核合成复刻(约 2-3 周)

**目标**: 重现 Fig 7(pre-SN vs post-SN mass fraction)。
**风险**: 中。α-network 是新东西,但 radial1d 的 hydro / 时间推进 / Newton 框架已有。
**Deliverable**: 4 个 progenitor 的 post-SN profile + 和论文 Fig 7 并排对比。

执行顺序:
1. α-network 实现(Task #18)
   - 13 个核:⁴He, ¹²C, ¹⁶O, ²⁰Ne, ²⁴Mg, ²⁸Si, ³²S, ³⁶Ar, ⁴⁰Ca, ⁴⁴Ti, ⁴⁸Cr, ⁵²Fe, ⁵⁶Ni
   - 反应率:REACLIB 或 Timmes aprox13 的解析 fit
   - 测试:α-freeze-out 在 T₉=5, ρ=10⁷ 的平衡 abundance
2. Light-bulb ν 源(Task #17)
   - ε_ν_heat = L_ν σ_heat / (4πr²) · ⟨...⟩
   - ε_ν_cool ~ T⁶ per nucleon(简化 Bruenn 表达)
   - CLI:`--nu-lum L_ν --nu-temp T_ν --nu-gain-radius r_g`
3. Explosive nucleosynthesis driver(Task #15)
   - 驱动脚本:读 Sukhbold profile → radial1d + α-net + light-bulb → dump 后激波过后的 X_i(m)

**验收**: 15.90 M☉ post-SN Mg/Ne ≈ 0.7–0.8(论文数字 0.75)。

### Stage 3 — 2D 扩展(本 PR 只做规划,不执行)

**目标**: 把 Stage 1 挑出的 Mg-rich 候选(12.75, 15.90 M☉)喂给 cart_ale2 做 2D 对流验证,检查 1D MLT 预测的 shell merger 在 2D 是否成立。

**Deliverable**: `docs/n49b_stage3_2d_plan.md` — 留给下一轮。

---

## 4. 分支/PR 策略

- 本地分支: `n49b-replication`(从 main 分出)
- 目标: 每完成一个 Stage 就单独 PR 进 main
- 命名:
  - PR A: `n49b-replication → main` (Stage 1)
  - PR B: `n49b-replication → main` (Stage 2,在 Stage 1 merge 后 rebase)
- 不开 draft stacked PR,顺序做顺序合。

---

## 5. 关键外部依赖

| 依赖 | URL / DOI | 预计大小 |
|---|---|---|
| Sukhbold+2018 Dataverse | doi:10.7910/DVN/VOEXDE | 未知,估计数 GB |
| REACLIB 反应率(如用) | https://reaclib.jinaweb.org/ | <10 MB |
| Sawada & Suwa 2023 light-bulb 细节 | arXiv:2301.03610 | PDF |

数据集下到 `third_party/sukhbold_2018/`(已加到 `.gitignore`)或 `~/data/sukhbold_2018/`。
大文件**不入 git**,在 README 里给下载命令。

---

## 6. 时间估计

| 阶段 | 最乐观 | 现实 | 悲观 |
|---|---|---|---|
| Stage 1 | 3 天 | 1 周 | 2 周(如果 KEPLER 格式意外难) |
| Stage 2 | 1 周 | 2-3 周 | 1 月(α-network 收敛调试) |
| 总计(到可发给 Suwa) | 10 天 | 3-4 周 | 6 周 |

---

## 7. 成功标准

**最小**:
- 重现 Fig 2、Fig 6、Table 2 到肉眼一致
- 附带一份 markdown report,写清楚我们的 pipeline、数据源、每个数字和他的对比
- Email 给 Suwa 时的自我介绍可以说"我复现了您 2024 N49B 论文的核心数据分析,并把工具链开源在 `github.com/MahoMaho-Rize/stellar2d` 的 `n49b-replication` 分支"

**理想**(+ Stage 2):
- 重现 Fig 7,post-SN Mg/Ne 数字吻合
- 能说"我的工具链能在您 4 个 test progenitor 上跑完整 explosive nucleosynthesis pipeline"
- 暗示下一步:"我提议扩展到 2D 验证 shell merger,这是您论文 §3.3 的 open question"

**超额**(Stage 3,后续):
- 2D shell merger demo,可能是共同论文的 seed
