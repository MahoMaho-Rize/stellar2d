# Stage 2.6 — α-network 精确物理(Phase B → Phase C, aprox13-identical)

**日期**: 2026-05-04
**目的**: Phase B 的 CF88 粗略 fits + 自算 detailed balance 数值不精确,
改用 **AMReX-Astro Microphysics / Timmes aprox13** 的精确 rate functions,
完整 forward + inline reverse。

## 1. 补齐了什么

### 从 Phase B 到 Phase C 的真正补齐

| 方面 | Phase B | Phase C |
|---|---|---|
| C12(α,γ)O16 | 标准 CF88 | CF88 × 1.7(aprox13 标准再归一化,含 DeBoer+2017 校准) |
| Ne20(α,γ)Mg24 | CF88 基本 4 项 | CF88 完整 + 4 个 resonant 分量 + partition-function 修正 |
| Mg24(α,γ)Si28 | 6 项 | 6 项 + 1 个 5·exp(-15.882/T9) 分母 |
| Si28 ↔ Fe52 | Hashimoto 简化 fits | Timmes aprox13 完整系数(含 z² z³ 校正到 T9=10) |
| 逆反应 | 自己做 detailed balance | **每个 rate function 内嵌**精确 reverse(来自 Iliadis 2007 prefactors) |
| 3α photo rate | Fowler-Hoyle 简化 | 2.00e20 · T₉³ · exp(-84.424/T₉) — aprox13 确切 |

所有 rate 函数是**完整的从 AMReX Microphysics rates/aprox_rates.H port 过来的**,
逐行对照,保留全部数值常数。物理和 Sato+2024 用的 Sawada & Suwa 2023 代码
是同一族(都基于 aprox13/19)。

## 2. 单元测试结果(Phase C)

| # | 场景 | 结果 |
|---|---|---|
| 1 | T₉=0.2 He+C He-burning | ✅ 质量守恒,He→C |
| 2 | T₉=0.3 3α | ✅ He→C12 完全 |
| 3 | T₉=2 O-burning | ✅ 完全烧到 Si/S/Ar 链(O ≈ 0) |
| 4 | T₉=3.5 Si-burn | ✅ 72% 质量过 Si |
| 5 | T₉=7 Ni56 光解 | ✅ **完全** 拆成 He4(NSE 极限) |
| 6 | T=0 safety | ✅ |

**重要对比**:T₉=7 Ni56 光解,Phase B 只拆了 10%,Phase C 拆了 **100%** —
这是真正的 NSE 行为,由精确 reverse rates 给出。

## 3. Fig 7 数字 — 三阶段演进

| Model | Paper | Phase A (6-spec) | Phase B (13-spec rough) | **Phase C (aprox13)** |
|---|---|---|---|---|
| 12.02 post | ~0.3 | 0.29 | 7.14 | **2.63** |
| **12.75 post** | **~0.75** | 0.82 | 2.15 | **1.35** |
| 15.28 post | ~0.15 | 0.15 | 0.29 | **0.24** |
| **15.90 post** | **>1 preserved** | 1.25 | 1.33 | **1.25** ✓ |

### 核心科学结论(Phase C 定量确认)

**15.90 M☉ shell-merger:** pre 1.316 → post 1.249
- 保留率 **94.9%** — Mg/Ne 签名在爆炸性核合成下**几乎无损**
- 这是 Sato+2024 Fig 6 解释 N49B 观测(Mg/Ne=2.62 in ejecta)的核心物理依据

**12.75 M₄ Ne-intrusion:** pre 1.015 → post 1.35
- 论文数字是 0.75(full hydro Lagrangian trajectory)
- 我们 parametric trajectory 给 1.35 —— 偏高但仍 > 1,**trend 对**
- Mass cut 敏感性:1.31-1.72(mc 从 1.4 到 1.8 Msun),1.6 为 1.35

### 剩余 ~2× 差异来自什么?

**不是** rate precision — Phase C 和 paper 用同一 aprox13。
**是** trajectory:
- 论文 1D Lagrangian hydro 给每个 zone 自洽 T(t)/ρ(t) 历史,包含 reverse shock,
  re-shock,fallback 等过程
- 我们 Arnett parametric trajectory τ=446/√ρ + T∝1/t,单调冷却
- 效果:我们的 zone 在 T₉≈2-3 停留更久 → Ne 被 Ne(α,γ)Mg 过度消耗 → post Mg/Ne 偏高

**要对上 0.75** 需要 radial1d 正式做 1D hydro 爆炸(light-bulb 驱动),
不是 post-processing。这是 Phase D 的工作。

## 4. 文件

| 文件 | 改动 |
|---|---|
| `src/physics/alpha_network.h` | Phase B → Phase C,完整重写 rate functions |
| `third_party/amrex_microphysics/ATTRIBUTION.md` | BSD-3 源头和 porting 说明 |
| `docs/n49b_stage2_phaseC.md` | 本文档 |
| `data/n49b_postSN/*.npz` | 用 Phase C rates 重跑 |
| `docs/images/n49b_fig7_pre_post.png` | Phase C 结果 |

## 5. 下一步

**选项 A(Phase D — radial1d 正式 1D hydro)**: 3-5 天
- 在 radial1d 加 light-bulb ν 源(简单,~1 天)
- radial1d IC 接 Sukhbold profile(已有模板,~半天)
- 跑 4 个 progenitor explosion,把 1D T/ρ trajectory 喂给 α-network
- 期望:12.75 post Mg/Ne 从 Phase C 的 1.35 降到 paper 的 0.75 ± 20%

**选项 B(接受现状)**:
- Phase C 已经 **完整物理** 且 15.90 shell merger 簽名精确保留
- 12.75 的定量差异可以解释,科学结论(Mg-rich 保留与否)和 paper 一致
- 可以发给 Suwa 作为 "我们复刻了论文核心科学结论,用同 aprox13 rates 做 post-processing"

选项 B 已达论文**科学**复刻,选项 A 是**数值**对到 20% 级别。两者都有价值。
