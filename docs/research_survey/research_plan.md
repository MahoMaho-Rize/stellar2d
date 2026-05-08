# 研究规划:Suwa Yudai × Suzuki Takeru × stellar2d 能力地图

**日期**:2026-05-08
**目的**:基于两位 PI 近 6 年(2020-2026)论文的深读,规划 stellar2d 现有求解器框架下可行的研究题目。

**上游材料**
- 论文目录:`suwa_yudai_papers.md`(23 篇)、`suzuki_takeru_papers.md`(26 篇)
- PDF:`pdfs/suwa/`、`pdfs/suzuki/`
- 深读笔记:`suwa_deep_notes.md`、`suzuki_deep_notes.md`

---

## 1. 研究兴趣地图

### 1.1 Suwa 组(核心:核塌缩超新星晚期 / 中微子 / compact remnant)

**旗舰线 A — SN 中微子 light curve 反演**(I–VI 系列 + template paper,2020-2025,约 9 篇)
> 方法论:1D 长时(数十秒)PNS 冷却 → 合成 Super-K / Hyper-K 中微子事件率 → 倒推 NS 质量/半径/EOS/距离。
> 关键工具:Sawada 的 "blcode"(1D Lagrangian 隐式 hydro + Helmholtz EOS) 或 GR1D(GR + M1 transport)。
> 近一年重点:加入真实探测器背景(Super-K VI:2505.19721);将 template 扩展到 BH-formation case(M31:2504.19510)。

**旗舰线 B — PNS 星震学 GW**
> 2302.00292 Mori+Suwa+Takiwaki:GR1D + M1,到 20 s postbounce,提取 g/f/p-mode 频率随时间漂移 → 3G GW 探测器(CE/ET)可跟踪。
> 方法是 Cowling / linearized 频率分析 on 1D background profile。

**旗舰线 C — ⁵⁶Ni problem / 爆炸能量**
> 2301.03610、2010.05615、2112.10782 (Sawada+Suwa):light-bulb neutrino heating + Lagrangian hydro 扫多个 progenitor,发现 M(⁵⁶Ni) 与 E_expl 的 tension。
> 2306.01682:PISN 里 ¹²C(α,γ) 反应率对 Ni 产量的影响。
> 2512.11404 Shinoda+Suwa:1D 球形 Athena++ + thermal bomb 扫 H-envelope mass → fallback 决定 remnant NS/BH boundary。

**旗舰线 D — compact remnant 群体统计**
> 2202.00230、2204.08851、2306.17381、2312.01948:binary population synthesis、AIC、²⁶Al 溶液。
> 非 hydro 工作,是 binary-stellar-evolution 代码 + 现象学拟合。

**共同特征**:
- 基本是 **1D**(沿径向),Lagrangian 或 log-r Eulerian
- 计算瓶颈在 ν-transport,**非 ν 的部分完全 stellar2d 兼容**
- 近 3 年越来越依赖 **合成观测量** 而非原始模拟(探测器响应、light curve template)

---

### 1.2 Suzuki 组(核心:Alfvén 波驱动风 / MHD 吸积盘 / 尘埃演化)

**旗舰线 A — Alfvén 波驱动恒星风**(9 篇)
> 方法论:1D 超径向 flux-tube、显式 nonideal MHD、光球注入 broadband Alfvén 波 → 波耗散加热 → supersonic wind。
> 自研 Godunov-MoCCT 代码,十几年来稳定地迭代。
> 近两年新方向:**Ohmic + ambipolar 扩散**(2501.00294 RGB winds、2403.18409 solar chromosphere)、**低金属度 Pop III 日冕**(2106.12740、2303.17469)。
> 2203.15280 Shimizu:光球纵波(sound)通过 mode conversion 给 wind 加热贡献 30%。

**旗舰线 B — 原行星盘风驱动尘埃演化**(7 篇)
> 2004.08839 Taki、2201.08840 Ogihara、2212.10796 Hasegawa。
> 1D α-disk 演化(有 MDW mass loss)+ dust coagulation eq。非 MHD 硬计算,是简化 1D。

**旗舰线 C — MRI / 磁化吸积盘**(6 篇)
> 2305.12112 Suzuki sole author:cylindrical shearing box MHD,intermittent accretion bursts + ring/gap 自发产生。
> 2211.01072、2412.14981 Takasao:3D 全局 MHD magnetospheric accretion onto T Tauri stars。
> 2502.12549、2411.00298、2312.15415 Tamilan:TDE disk + MDW self-similar 解析解(非模拟)。

**旗舰线 D — stellar rotation / 磁制动 / gyrochronology**(4 篇)
> 基于 flux-tube MHD wind simulations 产生 torque scaling,再与观测 ω(t) 对比。

**旗舰线 E — galactic-center MHD**(2 篇)
> 2306.15761 Kakiuchi:3D global MHD w/ radiative cooling,磁 loop buoyancy 驱动云块下沉。

**共同特征**:
- 基本是 **1D flux-tube** 或 **2.5D / 3D shearing-box / global MHD**
- **MHD 是核心物理**;纯 hydro 无法复刻旗舰结果
- **波驱动的加热** 可以用 prescribed heating profile Q(r, t) 在纯 hydro 下做 qualitative 映射

---

## 2. stellar2d 能力侧可用组件对照

| 能力 | 求解器 | 与 Suwa 组匹配 | 与 Suzuki 组匹配 |
|---|---|---|---|
| 1D Lagrangian implicit hydro + Helm EOS + MESA loader + pp-chain | `radial1d` | **极高**(直接替代 blcode) | 中(可做 hydro-only 风模型) |
| 2D Cartesian ALE(Andrassy benchmark / 恒星对流) | `cart_ale2` | 中(progenitor convection 背景) | 低(无 MHD) |
| 2D 不可压缩 pseudo-spectral NS | `pseudo_spectral` | 低 | 低(无 MHD / 无分层) |
| 2D anelastic + Sturm-Liouville spectral(g-mode EVP + TD integrator) | `anelastic_sl` | 中(PNS asteroseismology 替代方案) | 低 |
| 2D spherical anelastic | `sph2d_spectral` | 未知(没测试) | 未知 |
| Passive species tracer(2nd-order MUSCL)+ 可变 g(y)+ 自定义 q̇(y) 加热 | cart_ale2 | 中(nuclear burning 背景) | **高**(波加热 mock) |

**核心 gap**:stellar2d **没有 MHD 求解器**,也**没有 ν-transport**。这两个都是本领域标杆工作的核心物理。因此我们**不应该**去做"与他们 head-to-head"的工作,而应该做 **(a) 他们 setup 里不需要 MHD/ν 的那一子集** 或 **(b) 他们主方法下 inaccessible 的 angle(例如 nonlinear mode coupling)**。

---

## 3. 项目提案清单(按可行性排序)

### P1. **复刻 Sawada+Suwa 2023 "⁵⁶Ni problem" scan**(高匹配,小工作量)

**动机**:Sawada+Suwa 2023 用 **1D Lagrangian implicit hydro + Helm EOS + MESA progenitors + light-bulb ν heating/cooling** 扫多个质量段的 progenitor,在 E_expl–M(⁵⁶Ni) 平面展示观测张力(Eq. 4-5 的 light-bulb 模型)。

**与我们 radial1d 的差距**:
- ✅ Lagrangian + Helm EOS:我们有
- ✅ MESA IC 加载:我们有
- ❌ **光灯式 ν heating**:Q_ν⁺ = L_νₑ · σ · n_N / (4πr²)(Eq. 4)和冷却项(Eq. 5):**需要加 2 个源项 kernel**
- ❌ **⁵⁶Ni post-process 合成**(要么沿 mass shell 跟温度历史做 NSE equilibrium,要么简化为 "T > 5e9 K 的 shell mass 算 Ni yield"):**工作量小**

**预估工作量**:2-3 天(两个 kernel + 后处理脚本 + 扫 5 个 progenitor × 3 个 L_ν 值 = 15 个 runs)

**预期结果**:重现原论文 Fig. 3 的 E_expl–M_Ni 散点 + tension curve。**可以直接和 Sawada+Suwa 对比,定量 benchmark**。

**扩展**(可选,±2 天):加入 Kawashimo+2023 的 ¹²C(α,γ) 反应率扰动,刻画 PISN regime 的 Ni 敏感度。

---

### P2. **Shinoda+Suwa 2025 fallback scan 替代复刻**(中匹配,中工作量)

**动机**:2512.11404 用 1D Athena++ HLLC + thermal bomb + core-softening 扫 18-28 M☉ metal-poor progenitor,给出 M_remnant vs E_expl / M_env 的通用关系。

**与我们的差距**:
- ❌ Athena++ 的 1D 球形 Eulerian HLLC:我们最接近的是 `radial1d`(Lagrangian)。**不一样,但能看同一物理**。
- ❌ **Thermal bomb**:在 r < r_bomb 的 shell 一次注入能量 E_bomb:**简单,1 天工作量**
- ❌ **Core-softening**(Price-Monaghan potential,避免 r → 0 的 singular gravity):**已有 G·M/r² 就行,radial1d 用 Lagrangian zones,最内层 face r > 0,本质上不需要 softening**
- ❌ MESA Z=10⁻⁴ Z☉ progenitor:**可以用 MESA 生成,我们已有 loader**

**预估工作量**:4-5 天(thermal-bomb IC 分支 + 5 种 progenitor × 3 种 E_bomb = 15 runs + 分析脚本)

**预期结果**:**Lagrangian 替代 Eulerian 的 fallback 结果对比**。Lagrangian 不会有 shell-crossing 问题但要小心 r_surface 塌缩的表面边界。如果拿到一致的 M_remnant universal relation,就是 methodology cross-check paper 素材。

**风险**:radial1d 的隐式路径在 fallback accretion 的 supersonic regime 可能不稳定(CLAUDE.md 说 pre-MS KH 跨不过 τ_KH,但 fallback 是 hydro timescale 非 KH timescale,应该 OK)。需要先做 smoke test。

---

### P3. **PNS asteroseismology via anelastic_sl**(低-中匹配,核心不匹配)

**动机**:Mori+Suwa+Takiwaki 2023 在 20 s PNS 冷却背景上做 g/f/p-mode 频率追踪。

**为什么低匹配**:PNS **不是 anelastic** 的 —— 它高度可压缩、相对论、有中微子压。我们的 `anelastic_sl` 严格适用 low-Mach、低熵扰动 Boussinesq/anelastic 区间。**直接硬套是错的物理**。

**可做的 angle**:不直接做 PNS,而是做**"anelastic 极限下的可长时间稳定 g-mode 频谱演化"作为数值方法 benchmark**。即:拿一个 polytropic atmosphere,做长时间(数千周期)anelastic g-mode 跟踪,测量 **nonlinear frequency shift**(幅度 → 相位)。这类似 Goldreich 线性-非线性转换研究,可以补上我们现有 anelastic_sl 框架的一个非线性基准。

**与 Mori+2023 的连接**:方法论层面 —— 我们**不做 PNS**,但在 anelastic 极限产出"数值方法 能否稳定跟 g-mode 频率到 0.1 % over 1000 periods"的基准,这对他们用 LMM/leakage 在 GR1D 里做 asteroseismic fitting 是有方法学参考价值的。

**工作量**:2-3 天(主要是改写 `test_anelastic_sl_td_nonlinear` 扩展到 500+ 周期 + 加 FFT-based frequency tracker)

**建议优先级**:**低**。除非你有兴趣做数值方法论 paper,否则不如做 P1/P2。

---

### P4. **Alfvén 波加热风 的 hydro-only 模拟**(低匹配,纯定性)

**动机**:Suzuki 组 flagship(2501.00294、2203.15280、2403.18409)。关键物理是 Alfvén 波驱动 + 耗散(Ohmic / ambipolar / 非线性级联)给 wind 加热。

**为什么低匹配**:**加热机制是 MHD 的**。没有 MHD 的话,只能用 **prescribed heating profile Q(r, t)** mock 掉 "wave dissipation";这样我们就是在测 **"给定 Q,hydro 风结构是什么"**,这是 1970s-80s 的 Parker-wind + heating 研究 —— **已经被做烂了**。

**可做的子问题**:
- (a) Shimizu+2022 §Appendix A 的 **no-Alfvén baseline**:纯 hydro + 声波注入 → wind 结构。这是他们论文的 control run,我们能完全复刻,并作为 **radial1d 1D Lagrangian 和 Eulerian 在低-Mach 边界的 methodology 对比**。
- (b) Washinoue+Suzuki 2022 (2209.10156) 的 **chromospheric T 对 coronal 加热的影响**:1D,热波传播,radial1d 加一个 top-boundary acoustic driver + optically thin cooling 即可做。

**工作量**:(a) 2 天。(b) 3-4 天(需要加 Sutherland-Dopita cooling 表)。

**建议优先级**:**低**。和他们的 flagship 方法论差距大,结果是"validate 1970s 知识"。

---

### P5. **progenitor convection 作为 CCSN 初始条件**(中匹配,长远)

**动机**:Suwa 组(Shinoda+2025、Sato+2024 N49B paper)的 fallback / nucleosynthesis 模拟都依赖 MESA 1D 球形 progenitor。但 **Si/O/C shell convection** 在 CCSN 前几百秒强烈扰动,影响 progenitor 的对称性破坏,进而影响 3D 爆炸动力学。Couch+2015、Müller+2016 等 3D 工作已经证实。

**与我们的匹配**:**cart_ale2 做 Andrassy 2022 convection benchmark** 本质就是 stellar convection 的 analog。**我们可以扩展 cart_ale2 做 2D Si/O shell convection**:
- 2D Cartesian box 代表 progenitor 中一个径向 + 角向扇区
- Variable g(y)(我们有)、自定义 q̇(y)(我们有) → 模拟 nuclear burning heating
- Passive species tracer(我们有) → 核素 跟踪

**预期产出**:提供一个 **2D hydro Si/O shell convection snapshot** 数据集(角向速度谱、entropy 扰动幅度、eddy turnover 时间),可以喂给 Suwa 组的 1D progenitor 做 3D 扰动初始条件估计。

**工作量**:**大**(2-3 周)。需要:
- 从 MESA 拿到 CCSN progenitor 最后几百秒的 Si 或 O shell
- 转成 cart_ale2 的 HSE IC(gradients、composition、heating)
- 长时积分(turnover scale)
- 和 Couch+2015 / Müller+2016 / Yoshida+2021 的公开 3D 对比

**建议优先级**:**中长远题目**。如果有意做 PhD-level 科学题目,这是最有 paper-level 产出的方向。

---

### P6. **Sawada+Kurokawa+Suwa 2025 cosmic-ray → 行星形成**(跨学科,低工作量 demo)

**动机**:2512.09660 Science Advances。SN CR flux 提高 protoplanetary disk 的电离率 → 修改盘风结构 → 影响 planetesimal 形成。

**可做的部分**:我们可以**用 anelastic_sl 做 CR-ionization-driven vertical mixing 的线性稳定性分析**。这个 paper 没做任何 hydro/MHD 模拟,全是 analytical。

**工作量**:不明确,要再深读。**优先级低**,是跨学科冒险。

---

## 4. 推荐的优先级路径

**第 1 优先级(1-2 个月出第一篇 paper)**:**P1 复刻 Sawada+Suwa 2023**。
- 理由:match 我们所有硬件(radial1d stack 完备)、科学问题活跃(⁵⁶Ni problem 仍 unsolved)、和 Suwa 组有 direct dialogue、和原 PI 的"light-bulb + Lagrangian hydro"方法论 head-to-head 对比天然。
- Deliverable:replication paper + 可能加上一个新 angle(例如 γ=5/3 → Helmholtz EOS 的差距,或加入 opacity-enhanced ν-heating profile)。

**第 2 优先级(2-3 个月)**:**P2 Shinoda+Suwa 2025 fallback**。
- 理由:Athena++ Eulerian vs radial1d Lagrangian 的 methodology cross-check paper,不需要太多新物理。
- 结合:P1+P2 可以一起打包成 "Lagrangian 1D 在 CCSN 后期动力学的系统性测试"。

**第 3 优先级(长远)**:**P5 2D Si/O shell convection**。
- 理由:唯一能真正触及 Suwa 组 3D 前沿 frontier 的角度,需要 cart_ale2 扩展但基础设施已备。

**不推荐**:P3(物理不匹配)、P4(和 Suzuki 组核心差距太大)。

---

## 5. 下一步具体动作(如果选 P1)

1. Read Sawada+Suwa 2023(`2301.03610.pdf`)全文,提取 Eq. (4)、(5)、Table 1 的 L_ν 和 <E_ν>、progenitor list。
2. 检查 radial1d 的 source-term hook:`src/gpu/radial1d/radial1d_solver.cu` 有 `k_rad1d_nuclear_pp`(pp burning),照样子写 `k_rad1d_light_bulb_nu`(加热 + 冷却)。
3. 写 `src/init/ccsn_bomb.cpp` 或复用 `init_from_mesa` 并做 thermal bomb 的 inject。
4. 从 Kepler (Woosley+02) 或 MESA 拿到 12.3 / 15 / 17 / 19.5 M☉ solar-metallicity progenitors。
5. 扫 L_νₑ ∈ {5, 10, 20, 40} × 10⁵² erg/s,测量 shock 半径、爆炸能量、⁵⁶Ni 质量(后处理:Si burning 区域的 total shell mass at T > 5×10⁹ K)。
6. 出图对比 Fig. 3 of Sawada+Suwa 2023。

预估:**14-21 天**(2 周全职),产出可投稿到 PASJ / ApJ Letters 的 1 篇 short paper。

---

## 6. 最后的诚实提醒

- **stellar2d 的竞争力不在核心爆炸 ν-transport,而在 radial1d 的 Helm EOS + MESA loader + 隐式 Lagrangian 稳定 + 现代软件工程(tests, CI, reproducibility)** —— 这意味着我们能在 Sawada+Suwa "blcode" 做过的每个题目上做 **更干净的数值 benchmark**,但不能做他们做不了的新物理。
- **Suzuki 组我们几乎无法接触其 flagship MHD 工作**。真要追他们的方向,应该先在 stellar2d 里加一个 **2D / 1D ideal-MHD solver**(大工程,3-6 个月)。不建议短期做。
- **时间预算**:如果目标是 1 年内出 2 篇以上 paper,应该聚焦 **Suwa 侧**;Suzuki 侧最多做 P4(a) 作为 methodology warm-up。
