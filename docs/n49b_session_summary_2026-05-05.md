# N49B 复刻项目 — 2026-05-05 session 总结

## Session 范围

一天时间,完成 Phase D(radial1d 1D implicit hydro 爆炸核合成)全部 6 天计划工作,外加 aprox13e 反应网络升级。

## 完成清单

### Phase D Day 1-6(commits `5ef4054` → `06f6290` → `5d630c1`)

| Day | 内容 | commit |
|---|---|---|
| 1 | radial1d 13-species 缓冲区 infrastructure | `5ef4054` |
| 2 | Sukhbold IC loader + thermal bomb + proto-NS 内边界 | `3d1ad04` |
| 3 | α-network operator-split hook + profile 13-spec dump | `3f349b2` |
| 4 | implicit Newton + block-tridiag PC 解开 SN 炸彈 | `cd535c8` |
| 5 | 48-run Mg/Ne 敏感性扫描 | `4638c1f` |
| 6 | Fig 7 + 多维 audit + 部分复刻报告 | `5d630c1` |

### Phase E — aprox13e(commit `a607521`)

- 从 AMReX aprox_rates.H 逐行 port 8 对 (α,p)(p,γ) bypass 反应率
- RHS 增加 bypass 贡献(QSE fast-intermediate 极限)
- 编译 flag `ALPHA_NET_USE_AP_BYPASS`(默认开)
- 重跑 48-run sweep,对比 aprox13 vs aprox13e

## 关键数据

### Mg/Ne 对 Sato+2024(最佳匹配)

| Model | aprox13 | **aprox13e** | Paper | 最终误差 |
|---|---|---|---|---|
| 12.02 M⊙ | 0.300 | **0.298** | 0.30 | **0.7%** |
| 12.75 M⊙ | 0.715 | **0.760** | 0.75 | **1.3%** |
| 15.28 M⊙ | 0.145 | **0.144** | 0.15 | **4.0%** |
| 15.90 M⊙ | 1.240 | **1.256** | 1.25 | **0.5%** |

### 性能

- 单 SN run:6-8 Newton 步,t=0→10s,~10-70s wall(aprox13 vs aprox13e)
- 48-run sweep:4 min(aprox13)、30 min(aprox13e,更 stiff)

## 反模式实证 — Mg/Ne 对上 ≠ 物理对上

(follow `feedback_baseline_matching_antipattern.md`)

即使做完 aprox13e 升级,多维 audit 依然有两个主要问题没解决:

### ❌ M_Ni56 ≈ 0

- 真实 CCSN:0.03-0.15 M⊙
- 我们:< 0.015 M⊙(最佳 run 为 0.013,其余 ~0)
- 原因:**不是反应率问题**,是 bomb 模型问题。热炸弹 1e51 erg / 0.1 M⊙ 不够让 deep inner 层达到 NSE。真实 paper 用 piston + 中微子驱动。

### ❌ Si/Ne 偏离典型带(0.3-5)

- 12.75 mc=1.8 Si/Ne=0.12,15.28 Si/Ne=0.008
- 原因:**最佳 Mg/Ne 匹配的 mc=1.8 把 Si-burning 区 excise 掉了**

### ❌ M_Fe_peak < 0.04 M⊙(期望 0.1-0.3)

同上,mc 把 Fe-peak 合成区切掉。

## 诚实评估 — Suwa 邮件 readiness

### 现在有的

- 4 progenitor Mg/Ne < 5% 对 paper
- 13-species α-chain + (α,p) bypass network(aprox13e)
- implicit Newton 1D hydro + block-tridiag PC
- GPU 加速(Dual<1> AD 精确 J·v)
- 48-run 敏感性扫描(4 min wall)
- 多维 audit 脚本诚实暴露缺口

### 还缺的(给 Suwa 真正打动人需要的)

1. **Ni56 非零** — 需要真实 piston+ν bomb 模型,或 aprox21 full
2. **2D 对流实验** — **这才是真正 novel 的,超出 paper scope**
3. **物理洞察** — 不只数值对上,要讲 Ne(α,p) 在 12.75 case 为什么主导

### 关键反模式记录

- `feedback_baseline_matching_antipattern.md` — 有 paper baseline 时调参
  对上单一标量很容易但掩盖 bug,需要多维输出 + 逐公式 audit 才算物理对

## 下一步候选

| 选项 | 工作量 | Suwa 影响力 | 物理完整性 |
|---|---|---|---|
| **aprox21 full network** | 5-7 天 | 低 | 修 Ni56 |
| **Piston+ν bomb 模型** | 3-5 天 | 低 | 修 Ni56 |
| **Stage 3: 2D cart_ale2 shell merger** | 10-15 天 | **高** | paper 外的新物理 |
| **SkyNet REACLIB 集成** | 5-7 天 | 中 | 完整 reaction 覆盖 |

推荐:**Stage 3 2D 对流** 是 Suwa 没有的能力,radial1d implicit Newton
+ block-tridiag PC 已证明可做 Newton-Krylov hydro,这个 testbed 直接
port 到 cart_ale2 的 2D Lagrangian rezone grid 上,可以跑 15.90 M⊙
shell merger 的 2D bubble 结构 —— 这**是 paper 的 1D 看不到的新物理**。

## 今日总耗时

约 6-8 小时交互工作 + ~40 分钟计算(48 run × 2 sweeps)。

Phase D 原计划 6 天(现实)或 10 天(悲观),实际 < 半天完成。
多出来的时间做了 Phase E aprox13e 升级。

## 文件清单

### 新增
- `src/physics/alpha_network.h` — aprox13e 扩展(8 对 bypass)
- `scripts/n49b/run_phaseD_sweep.py` — 48-run 扫描
- `scripts/n49b/fig7_phaseD_pre_post.py` — Fig 7 从 radial1d npz
- `scripts/n49b/fig_aprox13_vs_aprox13e.py` — aprox13 vs aprox13e 对比
- `scripts/n49b/audit_phaseD.py` — 多维诚实 audit
- `scripts/n49b/convert_sukhbold_ic.py` — Sukhbold → radial1d IC
- `src/gpu/radial1d_kernels.cuh` — k_rad1d_alpha_burn kernel
- `src/gpu/radial1d_solver.{cu,cuh}` — init_from_sukhbold, M_inner BC
- `tests/test_radial1d_alpha13.cu` — Day 1 round-trip 测试
- `tests/test_radial1d_alpha_burn.cu` — Day 3 GPU vs CPU 测试
- `data/n49b_postSN/phaseD_sweep_aprox13.{csv,json}` — 48 rows
- `data/n49b_postSN/phaseD_sweep_aprox13e.{csv,json}` — 48 rows
- `data/n49b_postSN/postSN_*.npz` — 48 个 per-run 档
- `docs/n49b_phaseD_plan.md` — 6 天计划
- `docs/n49b_phaseD_day4_shock_stability.md` — Day 4 hydro 稳定性记录
- `docs/n49b_replication_stage2_phaseD.md` — Phase D 部分复刻报告
- `docs/n49b_session_summary_2026-05-05.md` — 本文档

### 记忆
- `feedback_baseline_matching_antipattern.md` — baseline 对齐反模式警告
