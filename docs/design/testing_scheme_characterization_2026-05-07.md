# stellar2d Scheme Characterization 测试方案

> 创建日期:2026-05-07
> 作者:MahoMaho-Rize + Claude(P33 排查后 / ale2 过度耗散定性之后)
> 触发事件:5-07 晚发现 ale2 在 Andrassy benchmark 上 v_rms 低 10× 于 vl2,
> 256² 非单调凹陷。诊断结论是 ale2 ∈ "过度 damp 的 ILES",ν_eff ~ V·dx
> 一阶大。此前 5 benchmark 全过,**耗散性质没有任何测试覆盖**,直到跑
> Andrassy 才暴露。**这是一类系统性覆盖缺口**,跟 5-07 第一份测试
> plan 互补,专门补耗散/分辨率尺度/收敛阶三轴。

## 0 与 `testing_infrastructure_plan_2026-05-07.md` 的关系

原 plan 关注 "correctness regression + CI 基础设施"(5 个 Phase、pytest + CTest、
compute_error in C++),覆盖 **"code 是否正确求解已知 IC"**。

本文件关注**互补但独立的一套**:**"code 求解 IC 的数值性质是什么"**。

| 问题 | 原 plan 能答 | 本方案能答 |
|---|---|---|
| L1(Sod, 256²) 比上次差了吗? | ✅ | — |
| solver 对均匀流守恒吗? | ✅ | — |
| 2 阶格式真的是 2 阶吗? | ⚠️ 部分(收敛率) | ✅ 全面(多 IC × 多 mode) |
| solver 的 ν_eff 是多少? | ❌ | ✅ |
| 细网格效果真的单调改善吗? | ❌ | ✅ |
| ILES 谱是 k⁻⁵/³ 还是 k⁻¹⁰? | ❌ | ✅ |
| 两个 solver 有效 Reynolds 数差几倍? | ❌ | ✅ |

**核心论点**:正确性(测对不对)和 scheme 性质(耗散多少、色散多大、刻度)是两条独立维度。
前者保护 regression,后者决定 **solver 能干什么 / 不能干什么**。Andrassy 256² 凹陷
事件是典型"正确性 OK,scheme 性质不适用"场景,必须有独立测试识别。

## 1 三轴分类

### 1.1 轴 A:收敛阶(Convergence Order)

对每个 smooth 解析解 IC,在 `res = (64, 128, 256, 512, ...)` 上测 `L1, L2, L∞` 误差,
log-log 斜率拟合得 p,断言 `p_expected − 0.2 ≤ p_measured ≤ p_expected + 0.5`。

- **原 plan Phase 2.4** 对 ale2 Linear wave 已经做了 `{plm, ppm_cs, ppm_cw} × {128, 256}`
  两点斜率,这是雏形。
- **本方案要求**:所有 solver 都有至少一个 smooth-mode 收敛测试,最少 3 个 res,给 slope。

### 1.2 轴 B:分辨率扫描(Resolution Scan)

**不是** 单 IC 的收敛率,而是 **固定物理问题、扫分辨率看物理量**。

- **典型场景**:turbulent / convective saturation,哪个 res 才够 resolve?
- **度量**:v_rms(res)、enstrophy(res)、vortex life time(res)、mass entrainment rate(res)
- **期望**:good scheme **单调** 收敛到一个物理极限;**非单调** 是警告信号(Andrassy 256 凹陷)
- **需要多 res,不单是比 slope**,要看曲线形状

### 1.3 轴 C:数值耗散性质(Dissipation Characterization)

**量化 ν_eff**:给一个已知解析衰减解,测 scheme 诱导的等效 kinematic viscosity。

|  | Good ILES (Godunov) | Jensen-ILES (ale2) | DNS (pseudo_spectral) |
|---|---|---|---|
| ν_eff scaling | ~ dx² (2 阶) | ~ V·dx (1 阶) | ~ 0(显式 ν 物理输入) |
| ν_eff(k) | k-independent | k-independent | k-independent |
| ν_eff(V) | ~ V²(weak) | ~ V(strong) | V-independent |
| 能量谱 | k⁻⁵/³ or k⁻³ | k⁻¹⁰(过 damp) | k⁻⁵/³ |

需要**多个独立的定量实验**,不是单看一个数字。

## 2 实验菜单(全 solver 通用的 6 个测试)

每个 solver 都应完成以下 6 个 test,适配到自己的 IC 系统。

### T1. Smooth convergence test(轴 A 主)

**IC**:一个有解析解且光滑的 IC。
- Hydro(Strang/ale2/cart_ale/vl2/cart_lag):**entropy wave**(δρ≠0,δP=0,δv=0.0,平流),t→t_end 与 IC 比 L1
- Low-Mach JFNK(lowmach):**acoustic p-mode**(sine P / v perturbation on HSE),频率测出
- Spectral(pseudo_spectral/anelastic_sl):**Taylor-Green 纯扩散** 已有
- Radial1d:**Lane-Emden p-mode** 频率

**扫描**:`res = (64, 128, 256, 512)`,至少 3 个点
**断言**:`2 阶格式 slope ≥ 1.8`,`1 阶 slope ≥ 0.8`,`spectral slope > 3`

### T2. Scheme-specific convergence(轴 A 补)

每个 solver 独特的数值路径(AV, limiter, rebuild, Riemann solver)触发一下:

- **ale2 / cart_ale / cart_lag**:shock-tube (Sod) L1 收敛 + Gresho 稳态保持 + symmetry break
- **strang**:linwave 3 个 flux × 3 limiter
- **vl2**:同 strang
- **lowmach**:unperturbed HSE residual + perturbed response
- **pseudo_spectral**:2/3 dealias enstrophy(ν=0)、forced k⁻⁵/³

### T3. 数值耗散 quantification(轴 C 主 — Linear shear mode decay)

对**全部 hydro / spectral solver** 通用:

**IC**:`vx(y, 0) = V₀ · sin(k·2π·y/Ly), vy=0, ρ=P=uniform, g=0`
**BC**: periodic (或 shear-periodic)
**解析**:NSE 线性解 `vx(t) = V₀ exp(−ν k² t) sin(...)`
**测量**:`max|vx|(t)` 随 t 指数衰减,拟合 slope → `ν_eff = −slope/k²`

**扫描**:
- `res = (64, 128, 256, 512, 1024)` — 验证 ν_eff ∝ dx^n 的 n 阶数
- `k = (1, 2, 4, 8)` — 验证 ν_eff(k) 是否 k-independent(真粘性的标记)
- `V₀ = (0.01, 0.1)` — 验证 Jensen 型 ν ∝ V 还是物理型 V-independent

**产出 figure**:`ν_eff · dx` vs `k² / res²` 的 Pareto 曲面。每 solver 一张图。
一目了然看出 "who is low-diss (spectral / Godunov) vs high-diss (Jensen)"。

### T4. Smooth vs Grid-scale mode 对照(轴 C — Jensen 机制的 direct probe)

验证 T3 的 slope 是否 V-dependent。两种 IC 同 V_rms:

- (a) `vx = V·sin(2πy/Ly)`(k=1)
- (b) `vx = V·sin(k_Nyq·y)`(k 接近 Nyquist)

同一解析衰减率(应当)预测 `dE/dt(b)/dE/dt(a) = (k_Nyq/k₁)² = (nx/2)²`。

实际比值随 scheme:
- 物理 Laplacian 粘性:准确 `(nx/2)²`
- Jensen-ILES:比值偏小(grid-scale 被 Jensen 额外打击,两者都 decay 快)
- Hyperdiffusion(pseudo_spectral 2/3-rule):比值偏大(grid-scale 被 hyperdiff 超额打击)

**识别指标**,单测即可分辨 3 种 scheme。

### T5. 分辨率扫描(轴 B — Turbulent saturation)

**IC**:适合该 solver 的湍流 saturation 场景(不一定有解析解)。
- hydro: **2D decaying turbulence**(broadband random IC → cascade)
- hydro w/ gravity: **Rayleigh-Taylor** 或 **Andrassy-type** (只 ale 族做)
- low-Mach: **stratified Rayleigh-Bénard**(lowmach 不支持,跳过)
- spectral: **forced turbulence** Kolmogorov k⁻⁵/³(pseudo_spectral 已有 `--test forced_turb`)

**扫描**:`res = (64, 128, 256, 512)`
**度量**:饱和 v_rms,关于 res 的曲线
**断言**:good scheme 应 **单调** 趋近物理极限(细 res 更好)。
  - **非单调** = 警告信号(记录但不 fail — 可能是 sub-critical Re transition)
  - **slope 方向反转** = fail(细网格反而更差 → code bug)

### T6. 谱斜率(轴 C — inertial range characterization)

**IC / 方法**:T5 的饱和态 → FFT 得 E(k) → 拟合 inertial range slope

**断言**:
- 2D: Kraichnan−5/3(inverse cascade)or −3(forward enstrophy),允许 ±0.5
- 3D-like(k shell averaged in 2D): −5/3,允许 ±0.5
- **k⁻¹⁰** 或更陡:记录,不 fail(是 scheme 特性描述,不是 regression)

**用途**:
- 描述每 solver 的 "effective LES filter" 性质
- 论文写法:"solver X resolves inertial range k⁻⁵/³ down to k_d ≈ N/8"

## 3 每 solver 的具体覆盖矩阵

| solver | T1 | T2 | T3 | T4 | T5 | T6 |
|---|---|---|---|---|---|---|
| **strang** | entropy wave ✅ | linwave 3×3 ✅ | 需加 | 需加 | 需加 | 需加 |
| **vl2** | 需加 | linwave 3 flux | 需加 | 需加 | ✅ (Andrassy as proxy) | 需加 |
| **cart_ale2** | 需加 | 5 benchmarks ✅ | **需加 (本次实验)** | 需加 | ✅ (Andrassy scan) | 需加 |
| **cart_ale** | 需加 | 5 benchmarks(旧) | 需加 | 需加 | — | — |
| **cart_lag** | 需加 | Sod ✅ | 需加 | 需加 | — | — |
| **ale2d** | 需加 | — | 需加 | — | — | — |
| **wb2d** | — | — | 需加 | — | — | — |
| **pseudo_spectral** | Taylor-Green ✅ | KH 已有 | 需加(Exp 1 本方案) | 需加 | forced turb ✅ | ✅(k⁻⁵/³) |
| **anelastic_sl** | 需加 | DNS triad ✅ | 需加 | — | 需加(RB) | — |
| **sph2d_spectral** | 需加 | — | 需加 | — | — | — |
| **lowmach** | lane_emden | HSE 0 residual ✅ | n/a(implicit damping) | — | — | — |
| **radial1d** | p-mode freq | 5 benchmarks ✅ | n/a(1D) | — | — | — |
| **fas/simple/proj** | 需加 | HSE origin ✅ | — | — | — | — |

总缺口:**37 个测试**(含 duplicates;实际 code 可以 T3-T6 共享 IC 层)。

## 4 实施顺序

### Phase A:最高 ROI — "三骨干"(1 周内)

| # | 任务 | 为什么 ROI 高 |
|---|---|---|
| A.1 | T3(ν_eff)for ale2 / vl2 / pseudo_spectral | **3 个 solver 对比即可写 paper**,一次实验收获 3 个定量结果 |
| A.2 | T5(res scan)for ale2 / vl2 | Andrassy 事件的直接对照,**paper 支撑数据**(v_rms vs res 曲线) |
| A.3 | T1(entropy wave)for ale2 / vl2 / cart_ale / cart_lag | 目前 scripts/tests_ale2 只测 L1,**未测 slope** — Andrassy regression 前 `de242a8` 错 swept-remap 1 阶退化 2 周没发现,是这缺口 |

**工作量**:A.1 新写 `init_shear_mode` IC + 2 个分析 Python(~200 行 C++ / 300 行 Python)。
A.2 直接复用现有 Andrassy scan 数据 + 新 plot 脚本。
A.3 改 sod_compare.py 成 entropy_wave_slope_check,复用 `_common.py`。

### Phase B:覆盖扩展(2-4 周)

Phase A 之后,**剩下 solver × 测试类型** 交叉矩阵里补未完成的格:

- 每 solver T1(smooth convergence)
- 每 hydro solver T4(smooth vs grid-mode Jensen probe)
- spectral solver T6(谱 k⁻⁵/³)

**工作量**:~15 个新测试,每个 1-2 h。

### Phase C:长期诊断(按需)

- T5 for 3+ 湍流/对流 benchmarks(RT instability, KH)
- T6 谱分析工具通用化(build scripts/spectrum_analysis.py,pseudo_spectral / ale2 公用)
- 跨 solver 自动对比 dashboard(matplotlib grid,每个 figure 一轴)

## 5 数据 / 脚本命名规范

沿用原 plan 的 `tst/` 框架:

```
tst/
├── scheme_char/                   # 新 — scheme characterization suite
│   ├── conftest.py
│   ├── test_shear_decay_gpu.py    # T3 × all hydro solvers (parametrized)
│   ├── test_convergence_smooth_gpu.py  # T1
│   ├── test_jensen_probe_gpu.py   # T4
│   ├── test_res_scan_turb_gpu.py  # T5
│   └── test_inertial_spectrum_gpu.py  # T6
└── ...
```

`@pytest.mark.scheme_char` 作为整套标签,`pytest -m scheme_char --solver=X` 挑选
特定 solver。

**数据落地**:
```
docs/scheme_char/
├── 2026-05-07_ale2_nu_eff.csv         # 手工归档
├── 2026-05-07_ale2_nu_eff.png
├── 2026-05-07_vl2_nu_eff.csv
├── 2026-05-07_comparison_all_solvers.png
└── README.md                          # "这些数据支持 XX paper / 决策"
```

## 6 paper relevance

本方案的 3 轴数据是 **每篇 paper 都该引用的底层 scheme characterization**,而不只是
regression 护栏。例如:

- Andrassy paper §5 写 "cart_ale2 ν_eff = 0.3·V·dx (T3 extracted), placing it in
  the high-dissipation ILES bracket comparable to PROMPI" → 读者可**解释** v_rms 差异
- KH paper 写 "pseudo_spectral 2/3-dealias 给 k⁻⁵/³ to k_d = N/3 (T6)",
  有量化 cutoff 比口头 "convergence is good" 强 10×
- 若日后新加 subzonal-ale2 solver,T3/T5 前后对比定量说明 "ν_eff 从 O(V·dx) 降到 O(V·dx²)"

## 7 风险 / 不做

- **不统一 IC 接口**:不同 solver IC convention 差异大(BC、dim、units),不值得
  为 testing 建立统一 IC 框架。每 solver 维护自己的 shear-mode IC kernel。
- **不追 analytic-ν DNS**:pseudo_spectral 可以加 `--nu` 参数跑 DNS 对照,但
  hydro 不强求加显式 ν(非 scheme 本来 scope)。
- **不强求 T5 单调**:非单调性 (Andrassy 256²) 本身是数据,记录不 fail。
- **不跑 3D**:全 2D(或 1D radial1d),符合代码库现状。

## 8 对 task 体系的更新

建议在 pending task 列表插入:

- **task A**:Phase A.1 — ale2/vl2/pseudo_spectral T3 对比(3 solver × ν_eff 定量)
- **task B**:Phase A.2 — Andrassy res scan formally 进 docs/scheme_char/
- **task C**:Phase A.3 — 全 hydro solver T1(entropy wave slope check)
- **task D**:Phase B — 补完 T4 / T6 覆盖
- **task E**:Phase C — 长期(按 paper 需求 pull)

---

**与 `testing_infrastructure_plan_2026-05-07.md` 不冲突**。那份管 "correctness 
regression + CI 基础设施",本份管 "scheme 性质的 characterization"。两套互补,
共用 `tst/` 框架 + `@pytest.mark` 标签分组。

---

## 9 Reviewer-endorsed 按族测试矩阵

> 2026-05-07 晚:外部 reviewer 对 § 2-3 做了按族重组并补充了几个族内共享
> 测试目标。以下矩阵是与 § 2-3 的**投影**(T1-T6 转成按族看),作为**执行
> 优先级参考**。矩阵中 **"高"** = Phase A 必须完成,**"中"** = Phase B,
> **"低"** = 按需 pull。

### 9.1 Explicit compressible 族(strang, athena_vl2, wb2d)

| 测试类型 | 方法 / IC | 优先级 | 现状 |
|---|---|---|---|
| Unit Test | MUSCL/HLLC/VL2 reconstruction、Riemann 解、BC、dt 更新 | 高 | strang ✅ 6 个;vl2 / wb2d **需补** |
| Convergence / Res Scan | entropy wave、smooth advection、Sod L1 多 res | 高 | strang ✅(1.92);vl2 / wb2d **需补** |
| E2E | Sod / Sedov / Noh / Gresho / Yee | 高 | strang / vl2 部分;wb2d 需 smoke |
| ν_eff / Dissipation | shear wave decay、KE→IE 账 | 中 | wb2d 稳定性未调好,先短时诊断即可 |

### 9.2 Implicit low-Mach 族(lowmach, fas, simple, projection)

| 测试类型 | 方法 / IC | 优先级 | 现状 |
|---|---|---|---|
| Unit Test | JFNK / GMRES / 预条件 / WB 背景减除 | 高 | lowmach ✅ 7 个;fas/simple/proj 薄 |
| Convergence / Res Scan | 小振幅轴对称波,量化实际阶数 | 中 | 全部 **需补** |
| E2E | lowmach 主力;fas/simple/proj regression | 高 | lowmach 有 lane_emden_perturbed |
| ν_eff / Dissipation | 小 shear 或 Gresho decay(定期 audit 即可) | 中 | 可选 |

### 9.3 ALE hydro 族(ale2d, cart_lag, cart_ale, cart_ale2)

| 测试类型 | 方法 / IC | 优先级 | 现状 |
|---|---|---|---|
| Unit Test | Lagrangian step / rezone / remap / 守恒 / BC | 高 | cart_ale2 ✅ 1 unit + trace hook;其他 smoke |
| Convergence / Res Scan | smooth advection / vortex / Sedov 多 res | 中 | cart_ale2 可做 ν_eff vs dx |
| E2E | Sod/Sedov/Noh/Gresho/Yee(周期 + 反射 BC 都测) | 高 | cart_ale2 ✅ 5;ale2d / cart_lag 需 smoke |
| ν_eff / Dissipation | shear mode decay + KE→IE ledger | 高 | cart_ale2 `cart_ale2_trace.{h,cu}` **已落地**,Phase A 跑 T3 |

### 9.4 1D Lagrangian implicit hydro(radial1d)

| 测试类型 | 方法 / IC | 优先级 | 现状 |
|---|---|---|---|
| Unit Test | Newton-Krylov、Dual AD matvec、EOS、核反应 / 传导一致性 | 高 | Helm / Dual 可借用 |
| Convergence / Res Scan | polytrope 脉动 / 小振幅径向 pulse **用 Cowling 近似频率作参考** | 高 | **需补** — 无专属 |
| E2E | ZAMS pp-chain 点火、径向脉动 | 高 | 有 session log 未 formalize 为测试 |
| ν_eff / Dissipation | 1D implicit 的 ν_eff 可忽略;dissipation 主要由 timestep / solver error 产生 | 低 | 跳过 |

### 9.5 Spectral / Pseudo-spectral 族(pseudo_spectral, anelastic_sl, sph2d_spectral)

| 测试类型 | 方法 / IC | 优先级 | 现状 |
|---|---|---|---|
| Unit Test | FFT / skew-sym convection / dealias / projection 正确性 | 高 | pseudo_spectral ✅ Taylor-Green;其他需 smoke |
| Convergence / Res Scan | Taylor-Green / shear wave decay(抽 ν_eff(dx, k)) | 高 | **需补** |
| E2E | forced turb / DNS triad / g-mode | 高 | ps & ansl 已有;sph2d 需 smoke |
| ν_eff / Dissipation | shear mode decay、能量衰减 vs analytic | 高 | **标准方案,需补** |
| Spectra | E(k) compensated,inertial range 定性 | 高 | 2D: inverse k⁻⁵/³, forward k⁻³,**不要求严格** |

---

## 10 补充策略(按族矩阵的横向准则)

### 10.1 Smoke test 分层(T1 最小覆盖)

所有 frozen / 低活跃 solver(ale2d / wb2d / cart_impl / sph2d_spectral)都要
**最简短时 HSE 或 advection** smoke(< 5 s)。目标不是收敛率,只是**能跑不崩**
+ 基本守恒。

- **判据**:100 step 后 `|M_end − M_0|/M_0 < 1e-10`,`max|v| < 10×|v|_IC`,
  无 NaN
- **标签**:`@pytest.mark.smoke`,PR 必跑

### 10.2 Regression / E2E(T2)分层

已知 benchmark(Sod/Sedov/Gresho/Taylor-Green)作为 **"长期一致性护栏"**:

- **L1 绝对值**(每个 res 一条):check 不比最后一次记录差 >10%
- **收敛率**:log-log slope ≥ 2阶格式-0.2
- **标签**:`@pytest.mark.fast`(128² 短时)和 `@pytest.mark.slow`(512² 长时)分层

### 10.3 Scheme characterization(T3)数据归档规范

每个 solver T3-T6 运行完毕写入 `docs/scheme_char/YYYY-MM-DD_<solver>_*.csv`,
**这是 paper/audit 数据**,不是测试通过与否:

- 不 fail 任何测试(除非 ν_eff 明显 > 上次 × 2 说明 scheme 有意外变化)
- 维护一张 `docs/scheme_char/README.md` 的 **solver-wide 总表**:
  ```
  | solver | ν_eff prefactor | order (n in dx^n) | measured date |
  ```
- paper 引用用此表

### 10.4 分辨率扫描准则

- 每族挑 1-2 个 IC 做**网格倍增** `dx → dx/2 → dx/4 → dx/8`
- **报告 3 个量**:
  - L1(res) log-log slope(收敛率)
  - v_rms(res) 单调性(是否 Andrassy-like 凹陷)
  - wall time × memory 随 res scaling(performance 回归)
- **标签**:`@pytest.mark.scan`(手动跑,永不 CI)

### 10.5 谱测量准则

- **只要求 pseudo_spectral / anelastic_sl / sph2d_spectral 强约束**
- 2D inertial range 断言:
  - inverse cascade k⁻⁵/³ 允许 slope ∈ [−2.3, −1.3]
  - forward enstrophy k⁻³ 允许 slope ∈ [−3.5, −2.5]
  - 不在这两条内 **记录不 fail**(scheme 特性而非 bug)
- hydro solver 的谱作为 T6 选做,不强求

### 10.6 ν_eff 测量方法论

**主测量**:Linear shear mode decay(T3),拟合 `ν_eff = −slope(ln V_max)/k²`

**辅助**:Modified Equation Analysis(MEA)— 手工推导数值格式的 leading-order
truncation error 项,对比 T3 测量值验证。

**不用**:energy spectrum slope fitting(太依赖 IC,误差大)

### 10.7 文档与 audit

每个 solver 维护一张 `docs/scheme_char/<solver>_matrix.md`:
```
测试类型   | Unit | Conv | E2E | ν_eff | Spectra
实现状态   | ✅   | ⚠️   | ✅  | ❌    | N/A
最后测量   | 5-07 | --   | 5-07| --    | --
```

作为 CI dashboard 和 code review 的 audit 依据。**新 commit 不允许把 "✅" 降级
成 "❌"**,这是 regression。

---

## 11 更新后的执行优先级

按 9-10 节综合,Phase A(本周完成)最终版:

| Task | 工作 | 产出 | 验收 |
|---|---|---|---|
| #50 | T3 ν_eff × {ale2, vl2, pseudo_spectral} | `docs/scheme_char/2026-05-07_nu_eff_comparison.png` + CSV | 3 solver 各给 ν_eff(dx,k,V) 表 |
| #49 | Andrassy res scan 入库 | `docs/scheme_char/2026-05-07_andrassy_vrms_vs_res.png` | ale2 vs vl2 三 res 曲线 + 256 凹陷文字说明 |
| #47 | entropy wave slope × {ale2, vl2, cart_ale, cart_lag} | `docs/scheme_char/2026-05-07_entropy_wave_convergence.csv` | 4 solver slope 表 |
| #48 | Phase B 全量覆盖 | 见矩阵 | smoke + spectra + 剩余 ν_eff |

Phase A 完成后,**解锁 paper writeup** —— ale2 在 Andrassy 的 "低 v_rms" 不再是
待解释现象,而是 **定量测量的 ν_eff = f(dx) 结果**。
