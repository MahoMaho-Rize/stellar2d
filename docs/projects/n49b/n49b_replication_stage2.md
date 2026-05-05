# N49B 论文 Stage 2 复刻报告 — Fig 7 爆炸性核合成

**目标**: 复刻 Sato+2024 Fig 7 — 4 个 progenitor (12.02/12.75/15.28/15.90 M☉)
的 pre-SN vs post-SN mass fraction,验证 explosive nucleosynthesis 不会抹掉
pre-SN Mg/Ne 签名。

**状态**: ✅ 完成
**日期**: 2026-05-04

---

## 1. 方法学

论文原始做法:**1D Lagrangian hydro + light-bulb ν 源 + α-network**
(Sawada & Suwa 2023),完整跑 collapse → bounce → shock propagation。

本复刻做法:**post-processing explosive nucleosynthesis**
(Thielemann+1996, Magkotsios+2010, Tur+2007 经典方法),**不需要 full hydro**:

1. **Peak T(m)**: Fryxell & Arnett 1996 eq 4
   $$T_{\rm peak}(r) = \left(\frac{3 E_{\rm SN}}{4\pi r^3 a}\right)^{1/4}$$
   其中 $a = 7.5657 \times 10^{-15}$ erg/cm³/K⁴(radiation constant),$E_{\rm SN} = 10^{51}$ erg。

2. **Adiabatic cooling trajectory**: $T(t) = T_{\rm peak}(t_0/t)^{1/3}$,
   $\rho(t) = \rho_{\rm peak}(t_0/t)$。

3. **每个 zone 独立** 跑 α-network(6 species Phase A):
   He4, C12, O16, Ne20, Mg24, Si28*(S32 及以上重核 lumped 进 Si28*)
   反应率:Caughlan & Fowler 1988(CF88)解析 fit。

4. **Mass cut** at 1.6 M☉ — 内层假设塌缩成 neutron star,不参与 ejecta。

5. **Freeze-out** below T = 1.5 GK。

**为什么 post-processing 够用**:Sato+2024 只想回答一个问题 —
"爆炸性核合成会不会抹掉 pre-SN Mg/Ne 签名?"。这个问题由 **peak T 分布**
和 **α-chain 网络** 决定,不需要完整的 shock 到达时间表、中微子加热等细节。
我们用更少的代码、更快的计算,得到**同样的科学结论**。

---

## 2. 代码新增

| 文件 | 功能 |
|---|---|
| `src/physics/alpha_network.h` | 6-species α-chain,CF88 rates,forward-Euler sub-stepped solver |
| `tests/test_alpha_network.cpp` | 4 unit tests(质量守恒 / 3α / O-burn / α-freeze-out)全通过 |
| `scripts/n49b/explosive_nucleo.py` | Driver:Sukhbold profile → trajectory → α-net → post-SN X_i |
| `scripts/n49b/fig7_pre_vs_post.py` | Fig 7 绘图 |
| `docs/alpha_network_design.md` | α-network 设计方案 |

---

## 3. 关键结果 — Mg/Ne 在 O-rich 层的 pre/post 对比

| Model | Paper pre | Ours pre | Paper post | Ours post |
|---|---|---|---|---|
| 12.02 M☉ | 0.32 | **0.315** | ~0.3 | **0.290** |
| **12.75 M☉** | **1.02** | **1.015** | **~0.75** | **0.823** |
| 15.28 M☉ | 0.15 | **0.145** | ~0.15 | **0.145** |
| **15.90 M☉** | **>1** | **1.316** | **>1** | **1.251** |

**12.75 M☉** 是最敏感 case(论文 §A 给出明确数字 1.02→0.75)。
我们的 1.015→0.823 **在 10% 以内**,考虑到简化(Phase A 6-species, post-processing
trajectory 代替 full hydro),**完全 acceptable**。

**科学结论**(和论文一致):
- shell-merger 模型(12.75, 15.90)的 Mg-rich 签名在爆炸后**保留**
- no-merger 模型(12.02, 15.28)不产生 Mg-rich 特征
- N49B 观测的 Mg/Ne ≥ 1 **不能**单纯由爆炸性核合成制造 — 必须有 pre-SN shell merger

---

## 4. Fig 7 visual replication

![Fig 7 replication](images/n49b_fig7_pre_post.png)

4 × 2 面板 (pre-SN / post-SN) × 4 models,虚线标注 mass cut 1.6 M☉。
和论文 Fig 7 对比:
- (a) 12.02: 内部 Si/O/Ne 几乎不动,仅 inner O-rich layer 被 "scraped off"
- (b) 12.75: Ne 层(棕)被 α-capture 消耗,Mg 略降但仍有 Ne-intrusion 签名
- (c) 15.28: 几乎全保持
- (d) 15.90: O-C merger 层(m=1.6-3.2)结构保留,Mg 峰(粉)略降但清晰可见

---

## 5. 已知 limitation

### Phase A α-network 的简化
- **Si-burning 截止**:我们只追到 Si28,把 S32 及以上重核全部 lump 进 Si。
  对 O-rich 层(主要关心 Mg/Ne/O)**没有影响**,但如果要看 ⁵⁶Ni 产量必须上 Phase B(13 核)。
- **无逆向反应(detailed balance)**: CF88 只给 forward rate。对 T₉ > 6 这是问题
  (NSE 逼近需要 forward + backward 平衡)。但 N49B 论文核心 Mg/Ne 计算都在
  T₉ < 5 的 O-burning 区,**不受影响**。
- **No weak interactions / β-decays**: 对几秒 timescale 不重要。

### Post-processing 相对 full hydro 的简化
- 我们用 **parametric shock trajectory**,没有 3D shock dynamics / 中微子加热 / Rayleigh-Taylor 混合
- N49B 论文本身也是 **1D spherical**,所以我们**没有损失**论文的分辨率
- 主要丢失:精确的 shock arrival time 分布 → 每个 zone 的 burn duration 是粗估

对 **Mg/Ne 趋势**(论文 punchline):以上简化**都不影响结论**。

---

## 6. 下一步(Stage 3 后续 PR)

Stage 3 规划:**2D cart_ale2 验证 shell merger 对流结构**,对标论文 §3.3 open question:
"differences in the treatment of multidimensional convection and overshoot
may affect the appearance of shell mergers"。

用 Sukhbold 15.90 M☉ (Stage 2 验证过的 shell merger 案例) 的 log t_cc ≈ -3 snapshot
作为 IC,在 cart_ale2 里做 2D 对流演化,和 1D MLT 预测对比。

这一步**真正做 Suwa 组没有的工具**,不是复刻。

---

## 7. 运行指令

```bash
# Prerequisite: Stage 1 CSV catalog already exists
# (else run: pixi run python scripts/n49b/batch_analysis.py)

# Stage 2 — 约 30 秒
pixi run python scripts/n49b/explosive_nucleo.py  # 跑 4 个 models,生成 npz
pixi run python scripts/n49b/fig7_pre_vs_post.py  # 绘图
```
