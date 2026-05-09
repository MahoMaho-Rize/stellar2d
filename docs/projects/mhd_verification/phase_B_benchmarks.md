# Phase B — athena_mhd 源项 / 大气 / Suzuki 物理栈实测记录

**Branch**: `athena-mhd-solver`
**Solver**: `src/gpu/explicit/athena_mhd_*`(VL2 + HLLD + CT + §B4 WB)
**派生依据**: §B4 (HSE), §C6 (Spitzer), §C7 (Townsend cooling), §C8 (chromo), §E1 (stochastic driver)

Phase A 证明了 ideal MHD 不炸(见 `phase_A_benchmarks.md`)。
Phase B 在此基础上加源项,逐个 milestone 向 Suzuki flux-tube 风物理靠近。

---

## 总览

| M | 内容 | 派生 | 测试 | 状态 |
|---|---|---|---|---|
| B-M1 | Well-balanced MHSE 源(§B4) | §B4 已完成 | `test_athena_mhd_hse_preserve.cu` 6/6 | ✅ |
| B-M2 | Spitzer κ₀T^{5/2} 各向异性导热 | §C6 已完成 | `test_athena_mhd_conduction.cu` 5/5 | ✅ |
| B-M3 | Townsend 光学薄冷却闭式积分 | §C7 已完成 | `test_athena_mhd_cooling.cu` 10/10 | ✅ |
| B-M4 | 分层 + 导热 + 冷却 combined | §B1/4/C6/7 | `test_athena_mhd_combined.cu` 10/10 | ✅ |
| B-M5 | Suzuki+25 stochastic broadband driver | §E1+§E2 已完成 | `test_athena_mhd_driver.cu` 9/9(T5 Alfvén 发射 + T6 §E2 吸收 + T7 Leroy80/Cranmer07 WKB v⊥∝ρ^{-1/4}) | ✅ |
| B-M5.5 | Shimizu+22 色球 blend thick+thin 冷却 | §C8 已完成 | `test_athena_mhd_chromo.cu` 7/7 | ✅ |
| B-M5.75 | 全算子集成 smoke(WB+κ+cool+chromo+driver) | §B4/C6/C7/C8/E1 | `test_athena_mhd_all_ops.cu` 14/14 | ✅ |

---

## B-M1 — Well-balanced MHSE 源(§B4)

**Date**: 2026-05-08
**Commit**: `fdbe383`

### Setup
等温指数分层大气:
- $\rho(y) = \rho_0 \exp(-y/H)$, $P(y) = \rho(y)\,c_s^2$, $\mathbf{v} = 0$
- 可选均匀 $B_y = B_0$(测试用 $B_0 = 0.1$,$\beta \gg 1$)
- Reflective y-BC, periodic x-BC, $g = 1$, $H = 1$,$N = 64 \times 64$
- 1000 步,$t \approx 3.6$(~3 个声波穿越)

### 实测对照

| 量 | naive(仅 ρg 源) | WB on(§B4 减 defect) |
|---|---|---|
| max\|δρ\|/ρ | 6.2×10⁻³ | **0**(精确) |
| max\|δP\|/P | 1.4×10⁻² | **0**(精确) |
| max\|v\|/c_s | 8.3×10⁻³ | **4.0×10⁻¹⁷**(1 ULP) |
| max\|∇·B\| | 0 | 0(CT 不受 WB 影响) |

6/6 断言通过。与派生书 §B4 "machine precision fixed point" 预期完全一致。

### B6 自检断言解释

B6 断言 "WB drift 严格 < naive drift",用来**抓"WB 没生效伪通过"**:早期调试里曾出现 snapshot 忘调 / wb_active 标志漏开,两个数值相同但各自都在阈值内混过。B6 输出的 "improvement factor = naive / max(wb, 1e-30) = 1.4×10²⁸" 不是真实物理比值——分母是除 0 兜底。实际含义:**WB drift 小到 ULP 以下,naive drift 在 1% 级**。

### 实现备忘(数值,非物理)

§B4 派生本身不需要改;下面 3 条是实现层面的 gotcha,未来复现 WB 或做类似操作时要记住:

1. **VL2 两阶段必须分开存 defect**
   - Predictor(order=1 donor-cell)与 corrector(order=xorder PLM)的离散残差 R(U_hse) 不同
   - 单 defect 抵消后残留 ~0.8% drift;双 defect 到 ULP
   - 代码:`d_rhs_hse_s1_*` + `d_rhs_hse_s2_*`,在 `apply_flux_divergence_and_ct` 按 stage 路由

2. **必须对全 6 个 conservative 都 subtract**
   - 原始直觉:只对 (m_x, m_y, m_z, E) 减(因为只有这些有重力源),ρ 和 Bz 不减
   - 实测:reflective 壁上 ρ 的**浮点舍入残差**(ULP 级)在 1000 步累积到 ~1% drift,B3 (δρ) 断言失败
   - 修正:六变量 (ρ, m_x, m_y, m_z, E, B_z) 全部 subtract

3. **snapshot 里两阶段用同一份 prim(U_hse)**
   - 实 step()中 stage-2 计算 flux 时 d_w_* = prim(U*)(因为 step 有 swap+refill)
   - snapshot 里只需 cons_to_prim 一次(prim = U_hse),不用模拟 swap——因为 step() 里 stage-1 WB 完美时 U* = U_hse,stage-2 实际见到的就是 prim(U_hse)
   - 如果 snapshot 里加 swap 模拟 stage-2 的 U* 状态,反而 capture 了"未 WB 的 U*"的残差,破坏协调

### 公共 API

```cpp
// 新增在 athena_mhd_solver.cuh
void init_hse_atmosphere(double g, double H, double rho0, double B0_y = 0.0);
void snapshot_hse();  // 也可对任意自定义 U_hse 手工调

// 新增 state
bool wb_active = false;  // 默认关,A1-A5 所有测试零影响
```

---

## B-M2 — Spitzer 各向异性导热(§C6)

**Date**: 2026-05-08
**Commit**: TBD

### Setup
- $\kappa_0 = 1$,$\rho_0 = 1$,$T_0 = 1$(code units,$k_B = \mu = 1$)
- $N = 64^2$(T4 用 $32^2$),$L_x = L_y = 1$,全周期 BC
- 温度扰动 $T(x, y) = T_0(1 + A\cos(\mathbf{k}\cdot\mathbf{r}))$,$A = 10^{-2}$
- 均匀 $|B| = 1$,方向按测试选(C6-T1/T4 沿 $\hat{x}$,T2 沿 $\hat{y}$)

实现(operator-split,独立于 VL2 step):
- `kappa0 > 0` 时 `apply_conduction(dt_target)` 被调用
- subcycle Euler:每步用 CFL $\Delta t_\mathrm{cond} = 0.45 \cdot \min_\mathrm{cell} \tfrac{1}{2}\rho c_v h^2 / \kappa_\parallel$
- flux 用 cell-centered $\hat{\mathbf{b}}$ 面平均后再归一化(§C6 "先 average B 后 normalize")
- 默认关(`kappa0 = 0`),A5 30/30 + B1 6/6 零影响

### 实测

| 测试 | 测量 | 阈值 | 状态 |
|---|---|---|---|
| C6-T1 parallel decay 匹配 $\exp(-\chi k^2 t)$ | rel err **1.4×10⁻⁵** | < 2×10⁻² | ✅ |
| C6-T2 perp quench (B⊥∇T) | rel change **0** | < 10⁻⁸ | ✅ |
| C6-T3 total E 守恒 | ΔE/E = **6.5×10⁻¹⁵** | < 10⁻¹⁰ | ✅ |
| C6-T3 ∇·B 不动 | **0** | < 10⁻¹⁰ | ✅ |
| C6-T4 2nd 法单调衰减 | 50/50 bins 无增长 | 0 violations | ✅ |

$\chi_\mathrm{eff} = \kappa_0 T_0^{5/2} / (\rho_0 c_v) = 2/3$, $\gamma_\mathrm{an} = \chi k^2 \approx 26.3$,$t_\mathrm{end} = 0.05/\gamma_\mathrm{an} \approx 1.9×10^{-3}$(5% decay linear regime)。
实测 ratio = 0.9512 vs analytic 0.9512 — 5 位有效数字对齐。

### 实现备忘

1. **IC 顺序陷阱**: 首版测试在 `sv.init()` 之前设 `sv.kappa0 = 1`,但 `init()` 重置为 0 → conduction 不触发(χ=0, t=inf)。修正:设 kappa0 **必须在 seed/init 之后**。记在这里提醒未来测试。
2. **T 于 ghost**: `k_athmhd_compute_T` 一次写全 $\mathsf{sx} \times \mathsf{sy}$ 包括 ghost 层,flux kernel 用到 $j\pm 1$ 的横向均值没问题。如果只写 interior 就需要 fill_ghost_T。
3. **No STS in v1**: Euler subcycling 够用,1000 sub-steps/hydro-step 在 $T \sim 1$ 够快。chromosphere ($T \sim 10^6$) 才需要 Meyer+12 RKL2 $\sqrt{s}$ 加速 — 留给 B-M4 integration.

### 公共 API

```cpp
// athena_mhd_solver.cuh
double kappa0 = 0.0;              // set > 0 to enable
void   apply_conduction(double dt_target);  // no-op if kappa0 ≤ 0
double compute_conduction_dt();             // returns ½ ρ c_v h² / κ_∥ min
```

---

## B-M3 — Townsend 2009 光学薄冷却(§C7)

**Date**: 2026-05-08
**Commit**: TBD

### Setup
- 单段幂律 $\Lambda(T) = \Lambda_0 (T/T_\mathrm{ref})^\alpha$
- Code units,$k_B = \mu = 1$,$\rho \Lambda_0 = 1$,$\gamma = 5/3$
- ODE $dT/dt = -C T^\alpha$,$C = (\gamma-1)\rho\Lambda_0 / T_\mathrm{ref}^\alpha$(cell 上常数)
- 闭式积分(Townsend 2009 lemma):
  - $\alpha \ne 1$: $T = [T_0^{1-\alpha} - C(1-\alpha)t]^{1/(1-\alpha)}$
  - $\alpha = 1$ 退化: $T = T_0 e^{-Ct}$
- **每 cell 一次 kernel launch,无 subcycle** —— exact integration,unconditionally stable
- 只更新 E 里的 $\rho e_\mathrm{th}$,其它 (ρ, m, B_f, KE, ME) 一概不动

### 实测

| 测试 | 测量 | 阈值 | 状态 |
|---|---|---|---|
| C7-T1 $\alpha=0.5$ 闭式匹配 | rel err **0** | < 10⁻¹⁰ | ✅ |
| C7-T1 $\alpha=1.0$ 指数支 | rel err **0** | < 10⁻¹⁰ | ✅ |
| C7-T1 $\alpha=2.0$ 闭式匹配 | rel err **0** | < 10⁻¹⁰ | ✅ |
| C7-T1 $\alpha=3.0$ 闭式匹配 | rel err **0** | < 10⁻¹⁰ | ✅ |
| C7-T2 ρ / ME / KE / divB 不动 | 四项 **0** | < 10⁻¹⁴–10⁻¹⁰ | ✅ |
| C7-T3 monotone 50 bins | violations = **0** | 0 | ✅ |
| C7-T4 ΔE = ρc_v·ΔT budget | rel err **5.4×10⁻¹⁵** | < 10⁻¹⁰ | ✅ |

10/10 断言通过。T1 各 $\alpha$ 分支 rel err = 0 —— 测试里的"analytic"和 kernel 的闭式解是 bit-wise 同一个表达式,正确则必然 exact。T4 才是独立验证能量预算(分别算 E 总和、$\rho c_v \Delta T$,比值对齐到 ULP)。

A5 30/30 + B-M1 6/6 + B-M2 5/5 回归零影响(`cool_on = false` 默认)。

### 为什么 B-M3 比 B-M2 快很多

Spitzer 导热 kernel 是 parabolic PDE,$\Delta t_\mathrm{cond} \sim h^2/\chi$ —— 必须 subcycle。
Townsend 冷却是 cell-local ODE,闭式解 **exact**,$\Delta t$ 可以任意大;代码路径就是
`fill_ghost + cons_to_prim + 1 kernel`,没有循环、没有 CFL 计算。

### 实现备忘

1. **Tfloor 必须有**:$\alpha > 1$ 的情况 $T \to 0$ 需要无穷时间,但 $\alpha < 1$ 时
   $T$ 会在 finite time 里打穿 0(`base = T0^{1-α} - C(1-α)dt ≤ 0`)。kernel
   检测后 clamp 到 `Tfloor`,避免 NaN 传播。
2. **`Tnew > T0` 兜底**:浮点误差可能让 $\alpha = 1$ 分支在 $Ct \ll 1$ 时
   `exp(-Ct) ≈ 1` 产生舍入尘埃,用 `if (Tnew > T0) Tnew = T0` 强制单调。
3. **ΔE 计算用 $\rho c_v (T_\mathrm{new} - T_0)$**:不是 `(E_new - E_old)`。
   E 里还有 KE + ME,冷却只改 $\rho e_\mathrm{th}$,用 $T$ 差值直接算避免把
   KE/ME 也卷进算术误差。
4. **v1 单段 $\Lambda$**:暂不支持 Sutherland-Dopita 多段表。未来 B-M5
   (coronal cooling) 会加分段 Townsend 查表(每段闭式,cross-segment 用
   temporal evolution function $Y(T)$)。

### 公共 API

```cpp
// athena_mhd_solver.cuh
bool   cool_on      = false;   // false = off
double cool_Lambda0 = 0.0;
double cool_Tref    = 1.0;
double cool_alpha   = 0.0;
double cool_Tfloor  = 1e-6;
void   apply_cooling(double dt);   // no-op if cool_on=false
```

---

## B-M4 — Combined operator integration(§B4 + §C6 + §C7)

**Date**: 2026-05-08
**Commit**: TBD

### 动机

B-M1~M3 各自证明了 WB MHSE / Spitzer 导热 / Townsend 冷却 operator **在独立场景下正确**。B-M4 验证 operator-split 组合
$$ U^{n+1} = L_\mathrm{cool}(\Delta t) \cdot L_\mathrm{cond}(\Delta t) \cdot L_\mathrm{vl2}(U^n; \Delta t, \mathrm{WB}) $$
不会破坏各自已经证明的性质,并抓 operator 之间的"**BC 幽灵**"。

### 发现的数值 gotcha:κ + reflect y-BC 的 ghost-T 错误

**症状**:等温大气 $T = c_s^2$ 严格均匀,应有 $\nabla T = 0$ → $F_c \equiv 0$。但实测单次 `apply_conduction(dt)` 后 $\delta E/E = 4\%$(不是 ULP!),而 $\delta\rho$ / $v$ / $\nabla\cdot B$ 都是 0 —— **只有 E 被 κ 算子错误地修改了**。

**根因**:reflect y-BC 下 B_y face 是反对称镜像(`Byf[ng-1] = -Byf[ng+1]`),因此 ghost cell 的
$$B_{y,cc}^\mathrm{ghost} = \tfrac12(\mathrm{Byf}[ng-1] + \mathrm{Byf}[ng]) = \tfrac12(-B_{0y} + B_{0y}) = 0$$
而 interior $B_{y,cc} = B_{0y}$。Ghost cell 的 $P = (\gamma-1)(E - \mathrm{KE} - \mathrm{ME})$ 由于 $\mathrm{ME}^\mathrm{ghost} \ne \mathrm{ME}^\mathrm{interior}$ 多算/少算一个 $\tfrac12(\gamma-1) B_{0y}^2$,导致 $T^\mathrm{ghost} = P^\mathrm{ghost}/\rho^\mathrm{ghost} \ne c_s^2$。$\nabla T$ 在 wall 上**假性**非零 → κ 把能量"泄"到 ghost。

这不是 §C6 派生的错,也不是 `cons_to_prim` 的错:`cons_to_prim` 正确地**按 face-B 的镜像规则**推算 ghost $B_{cc}$,只是 *$B_{cc}^\mathrm{ghost}$ 和 interior $B_{cc}$ 的"镜像"不是 scalar mirror*。而 κ 只看温度 $T$,$T$ 是标量,**它的 ghost 必须是 scalar mirror**,不能从 $(P, \rho)$ 重新算。

**修复**:加 `k_athmhd_ghost_T_{y_reflect,y_periodic,y_outflow,x_periodic,x_outflow}` 一组 kernel,在 `compute_T` 之后,标量镜像填 T_cc 的 ghost 层,覆盖 `cons_to_prim` 推算的"错"ghost T。κ flux kernel 仍然读 T_cc,但现在 ghost T 与 interior T 严格对齐。

### Setup

- D-T1:HSE 大气 + κ₀=1 + cool_on(Λ₀=0 trivial),跑 500 步 VL2 + κ + cool
- D-T2:periodic 无重力,T 均匀 + $\delta T = 0.2\,e^{-r^2/\sigma^2}$ Gaussian blob,B = 0.5 ŷ,κ=0.05,30 VL2 步
- D-T3:periodic 均匀 T + 均匀 cool(α=2),验证每 cell 独立更新

### 实测

| 测试 | 测量 | 阈值 | 状态 |
|---|---|---|---|
| D-T1 WB 在 κ+cool 下守恒 | max\|δρ\|/ρ = **1.3×10⁻¹⁶**,max\|δE\|/E = **0** | < 10⁻¹⁰ | ✅ |
| D-T1 v 保持静止 | max\|v\|/c_s = **2.8×10⁻¹⁶** | < 10⁻⁸ | ✅ |
| D-T1 ∇·B 不动 | **0** | < 10⁻¹⁰ | ✅ |
| D-T2 blob 衰减 | amp 0.185 → 0.048,ratio = **0.26** | < 0.95 | ✅ |
| D-T2 far-field δρ/ρ | **1.0×10⁻²** | < 3×10⁻² | ✅ |
| D-T2 far-field δP/P | **1.6×10⁻²** | < 3×10⁻² | ✅ |
| D-T2 ∇·B 不动 | **2.7×10⁻¹⁴** | < 10⁻¹⁰ | ✅ |
| D-T3 uniform cool spread | **0** (ULP) | < 10⁻¹⁴ | ✅ |
| D-T3 T 匹配 Townsend α=2 | rel err **1.4×10⁻¹⁶** | < 10⁻¹⁰ | ✅ |

10/10 断言通过。B-M1 6/6 + B-M2 5/5 + B-M3 10/10 回归零影响(T ghost kernel 仅在 `apply_conduction` / `compute_conduction_dt` 路径上生效,`cool_on` 路径和 VL2 主步不经过 T 填充)。

### 实现备忘

1. **T ghost 必须独立填,不要靠 cons_to_prim**。cons_to_prim 按 face-B 推算 ghost B_cc,reflect face BC 下 ghost B_cc ≠ interior B_cc,造成 ghost P 偏差,进而 T 偏差。`κ` 只看 T,T 是标量,按标量镜像填。
2. **D-T2 far-field 容忍 3% 是物理上限,不是算法松**。blob 在 N=32 的 30 步里激发的声波前沿已传到 |x−xc|=0.3 ≈ 4σ 区域;WB 只保证不**额外**漂移,不抑制 blob 自身的物理 acoustic spillover。
3. **operator 顺序**:`step` → `apply_conduction(dt)` → `apply_cooling(dt)`,这是第一阶 Strang 简化(Godunov split)。二阶 Strang `A·B·A` 对 HSE + 弱源没有明显优势,留给 B-M5 corona 场景(强 cooling ≫ dynamic time)再评估。
4. **apply_cooling 不需要 T ghost fix**:cooling 是 per-cell ODE,只读 cell 内的 (ρ, P),从不跨 cell 或读 ghost。T ghost 问题只影响空间 flux(κ 导热、未来 viscosity 都要注意)。

### 公共 API

无新增。B-M4 纯测试 + 内部 kernel 扩充(`k_athmhd_ghost_T_*`)。

---

## B-M5 — Suzuki+25 stochastic broadband driver(§E1)

**Date**: 2026-05-08
**Commit**: TBD

### Setup

- Photospheric 驱动波形:$v_x(t, j = n_g) = \sum_{n=1}^{N} A_n \sin(2\pi f_n t + \phi_n)$
- 频率 log-spaced on $[f_\min, f_\max]$(典型 100× bandwidth)
- 相位 $\phi_n \sim \mathrm{Uniform}[0, 2\pi)$,mt19937_64 seeded for 再现性
- 幅度 $A_n = A_\mathrm{rms} \sqrt{2/N}$ 使 $\langle v_x^2 \rangle_t = A_\mathrm{rms}^2$(§E1 Parseval identity)
- 作用方式:SET interior j=ng 一行的 $v_x$,E 按 KE 变化同步更新
- default `driver_on = false`,不影响主流程

### 实测

| 测试 | 测量 | 阈值 | 状态 |
|---|---|---|---|
| E1-T1 driver off HSE 保持 | max\|δρ\|/ρ = **0**,max\|δE\|/E = **0** | < 10⁻¹⁰ | ✅ |
| E1-T2 ⟨v_x²⟩ = A_rms² | mean_v² = **9.80e-3** vs target 1.00e-2,rel err 2.0% | < 10% | ✅ |
| E1-T3 device = host 波形 | max\|Δ\| = **3.2×10⁻¹⁶** (ULP) | < 10⁻¹² | ✅ |
| E1-T4 seed 可复现 | max\|run0−run1\| = **0** | < 10⁻¹⁴ | ✅ |
| E1-T5a Alfvén 到达时间 | τ_theory = **1.593**,measured = **1.517**,rel err 4.8% | < 30% | ✅ |
| E1-T5b 极化关系 δB_x = −√ρ δv_x | ratio = **1.019**,rel err 1.9% | < 30% | ✅ |

7/7 断言通过。B-M1/2/3/4 全部回归零影响。

### T5 Alfvén 发射物理测试

单模驱动 (f_drive=1.0,A_rms=0.05) 在 HSE 等温大气 (B0_y=0.5,ρ₀=1,H=1) 上,测:
- **到达时间**:WKB $\tau(y^*) = \int_0^{y^*} dy/v_A(y) = (2H/B_{0y})(1 - e^{-y^*/2H}) = 1.593$ for $y^*=1$;实测第一次 $|v_x(y^*)| > 0.3 A_\mathrm{rms}\sqrt{2}$ 在 $t=1.517$,差 4.8%(VL2+PLM group-velocity 误差 + reflection,<30% 容差)
- **极化** ($z^+$ 上行 Alfvén 本征矢):在 $t = \tau + 0.5/f$(arrival 后半周期),实测 $\delta v_x = +0.050$,$\delta B_x = -0.031$,$\sqrt\rho(y^*) = 0.602$;期望 $-\sqrt\rho \delta v_x = -0.030$,ratio = 1.019(1.9%)

两个独立物理事实:driver **真的发射** Alfvén 波,MHD 介质**真的用正确波速传**,线性极化**对**。比 T1–T4 的工程验证更严肃。

### 实现备忘

1. **幅度归一化 N√(2/N) 不是 √(2/ln(f_max/f_min))**:§E1 派生给两种归一化 —— (a) 连续 $P(\omega) = A^2/\omega$ 下 $A^2 = A_\mathrm{rms}^2/\ln(f_\max/f_\min)$;(b) $N$ iid 相位正弦和下 $\Sigma A_n^2 / 2 = A_\mathrm{rms}^2$。我们用离散 N-mode,选(b)。(a) 适合 Elsässer 带内能通量的积分计算,不是 Parseval。
2. **driver 是 SET,不是 ADD**:Suzuki+25 inner-BC 语义是**规定速度**(prescribed velocity),每步覆写一次,不是冲量叠加。反映到 kernel:KE 要按新 $v_x$ 重算并写回 E,否则会积累能量漂移。
3. **t=0 时 $v_x(0) = \sum A_n \sin(\phi_n) \ne 0$**:driver 打开时会有一个**阶跃式**速度跳变。如果要平滑启动,可加 envelope $w(t) = \tanh(t/\tau_{\mathrm{ramp}})$。v1 暂时不加,配合 HSE IC 用(v_x IC 可以先设为 driver 初值,但目前测试里 IC v_x=0,第一步就接管;实测 T1 driver off + 200 步 HSE 仍 ULP,证明阶跃不坏 WB)。
4. **所有 cell 同相位**:j=ng 一行上每个 x cell 注入**相同**波形 —— 这是 Suzuki+25 "1D photospheric driver in 2D atm" 的简化。真 3D 会有水平相干长度 $\ell_h$,给每个 cell 独立 seed 或空间 filter。v1 保持简化。
5. **无 Elsässer 吸收**:§E1 派生里顶边界要用 $\partial z^-/\partial r = 0$。目前 athena_mhd 顶 BC 是 outflow/reflect,不是 Elsässer。实际 Suzuki 跑 open-top,但开放 BC 的 MHD Riemann 也不是严格 $z^-$ 吸收。**这块留给 B-M6**(flux-tube wind setup)再正式补。

### 公共 API

```cpp
// athena_mhd_solver.cuh
bool driver_on       = false;
double driver_Arms   = 0.0;
double driver_fmin   = 0.0, driver_fmax = 0.0;
int    driver_Nmodes = 0;
void init_stochastic_driver(double A_rms, double f_min, double f_max,
                            int N_modes, unsigned seed);
void apply_driver(double t);   // no-op if driver_on=false
```

---

## B-M5.5 — Shimizu+22 / Suzuki+25 色球 blend thick+thin 冷却(§C8)

**Date**: 2026-05-08
**Commit**: TBD

### Setup

在 §C7 单段 Townsend 基础上,加色球 blend:
- 混合权重 $\xi = \max(0, 1 - p_\mathrm{chr}/p)$:$p < p_\mathrm{chr}$ 时 $\xi=0$(纯 optically-thin 薄),$p \gg p_\mathrm{chr}$ 时 $\xi \to 1$(纯 Newton relaxation 厚)
- **Thick**:$\partial T/\partial t = -(T - T_\mathrm{ref,thck})/\tau_\mathrm{thck}$(指数松弛)
- **Thin**:§C7 Townsend 单段幂律
- **Operator split**:先按权重 $\xi$ 走 thick 指数支 $T_a = T_\mathrm{ref} + (T_0 - T_\mathrm{ref}) e^{-\xi \Delta t/\tau}$,再在 $T_a$ 基础上按权重 $(1-\xi)$ 走 Townsend 闭式;保持 O($\Delta t^2$) 误差
- cell-local,无 subcycle,无条件稳定

### 实测

| 测试 | 测量 | 阈值 | 状态 |
|---|---|---|---|
| E1-T1 纯 thin 极限 (ξ=0) | T_num = 7.0e-1,rel err **0** | < 10⁻¹⁴ | ✅ |
| E1-T2 纯 thick 极限 (ξ≈1) | T_num = 0.8033,rel err **1.4×10⁻¹⁶** | < 10⁻¹⁴ | ✅ |
| E1-T3 blend 中间 (ξ=0.5) | T_num = 0.8611,rel err **0** | < 10⁻¹⁴ | ✅ |
| E1-T4 ρ/mom/B/divB 不变 | 四项 **0** | < 10⁻¹⁰–10⁻¹⁴ | ✅ |

7/7 断言通过。B-M1~5 全部回归零影响(`chromo_on = false` 默认,独立于 `cool_on`)。

### 实现备忘

1. **operator split 不是严格 exact**:thick + thin 两步拼起来并不等于真正的组合 ODE $\dot T = -\xi/\tau (T-T_\mathrm{ref}) - (1-\xi) C T^\alpha$,O($\Delta t^2$) 误差。但每个极限($\xi=0$ 或 $\xi=1$)都 exact 对上单支派生。实际科学用途(wind 演化)$\Delta t \ll \tau_\mathrm{thck}$,误差可忽略。
2. **$\xi$ 在 cell 上当 $\Delta t$ 内为常数**:实际 $\xi$ 是 $p$ 的函数,$p$ 会随 $T$ 变化。v1 每 cell 用 step-start 时的 $\xi$,不迭代。如果未来需要 stiff coronal regime,改 RK2 或 Rosenbrock。
3. **`cool_on` 和 `chromo_on` 独立**:两个都开会顺序生效(先 conduction,再 §C7,再 §C8);用户自己选一个,不要同时。
4. **`p_chr` 是 code units 的压强阈值**:真物理对应 chromosphere-corona transition 处约 0.1 dyn/cm²,需要根据 IC 无量纲化转换。v1 仅测算子,不测物理参数选择。

### 公共 API

```cpp
// athena_mhd_solver.cuh
bool   chromo_on         = false;
double chromo_p_chr      = 0.0;
double chromo_T_ref_thck = 1.0;
double chromo_tau_thck   = 0.1;
double chromo_Lambda0    = 0.0;
double chromo_T_ref_thin = 1.0;
double chromo_alpha      = 0.0;
double chromo_Tfloor     = 1e-6;
void   apply_chromo_cooling(double dt);
```

---

## B-M5.75 — 全算子集成 smoke(WB+κ+cool+chromo+driver)

**Date**: 2026-05-09
**Commit**: TBD

### Setup

B-M4 只测了 WB + κ + cool 三件套,B-M5 / B-M5.5 进来后没一次把 driver + chromo 也加入同一条 step 链。B-M5.75 闭合这个缺口,作为进 B-M6 前的回归哨兵。

每步操作链(operator split 顺序):
```
U^{n+1} = L_driver(t+dt) ∘ L_chromo(dt) ∘ L_cool(dt) ∘ L_cond(dt) ∘ L_vl2(U^n; dt, WB)
```

### 实测

| 测试 | 测量 | 阈值 | 状态 |
|---|---|---|---|
| F-T1 所有 toggle 关 | δρ/ρ=**0**,δE/E=**0**,divB=**0** | < 10⁻¹⁰ | ✅ |
| F-T2 所有 toggle 开、参数置零(κ=1,Λ=0) | δρ/ρ=**1.3e-16**,δE/E=**0** | < 10⁻⁸ | ✅ |
| F-T3 live params 300 步 smoke | 无 NaN,ρ_min=0.374,divB=**0**,v_rms=**6.7e-4**(target 1e-3),mass drift **3.1e-6** | 工程哨兵 | ✅ |

**14/14 通过**,B-M1~5.5 全部回归零影响。

### 关键发现(2026-05-09 修订)

**先前版本误判**:最初把 F-T3 的 O(10⁻⁶) 质量漂移归因于 "prescribed-velocity Dirichlet BC 非守恒",这是错的。B-M5.75 cleanup 把 driver 从 interior-SET 改成 **§E2 characteristic 内 BC**(Elsässer 不变量 ghost fill),和 Shoda+18 (ApJ 853 190)、Sakaue+Shibata+21 (arXiv 2106.12752) 的 1D Alfvén-wave-driven 风求解独立收敛到同一公式:$\tilde z^+|_\text{ghost} = -2 v_\text{drv}(t)$ + 吸收 $\tilde z^-$。

**F-T4 振幅扫描**(A∈{1e-3, 5e-4, 2.5e-4, 1.25e-4}): 严格 $A^2$ 缩放(s2/s1 = s3/s2 = 0.25,A=1e-6 ULP 噪声),证明 §E2 线性阶质量通量**恰好为零**,和派生一致。

**F-T3 的 3.1e-6 漂移实际来自两个独立源**:

1. **2D PLM+HLLD 在切向 Alfvén 间断处的 $O(A^2)$ 截断 floor** —— 1D 文献里不讨论这个(Shoda18/Sakaue21 都是 1D),是 2D 重构的数值伪影,PPM 或更薄 ghost 梯度应可压下去。
2. **cool + chromo $\Lambda > 0$ 对 HSE 的累积退化** —— F-T4 variant (g) 关驱动 (A=0) 跑完整 cool+chromo 链得 **3.15e-6**,几乎就是 F-T3 的全部漂移。这是算子本身的 HSE 残差,和驱动无关。

F-T1/F-T2 已锁定 driver 关时 5 算子 ULP 守恒;F-T3 阈值定在 **1e-5**,既捕得住任何真实 BC 回归,又能骑过已知的 $\Lambda t$ floor。

操作链顺序的**物理**根据:
1. VL2 先走(hyperbolic core,其他都是源项)
2. conduction 紧随(它也是要在最 fresh 的 T 上算热流)
3. cool → chromo(cool 是单段 Townsend,chromo 是 blend;顺序颠倒仅 O(Δt²) 差异)
4. driver 放最后(inner BC 覆写,避免上面任一算子再扰动 v_x)

### 公共 API

无新 API。沿用 B-M5 的 `apply_driver` + B-M5.5 的 `apply_chromo_cooling`,操作链顺序记录在 `test_athena_mhd_all_ops.cu` 顶部注释。

---

## Phase B 完成状态(2026-05-09)

| Milestone | 物理 | 断言 | 关键发现 |
|---|---|---|---|
| B-M1 | §B4 WB MHSE | 6/6 | per-stage defect + 6-var subtract + 单 prim |
| B-M2 | §C6 Spitzer κ | 5/5 | face-avg B → normalize 顺序 |
| B-M3 | §C7 Townsend cool | 10/10 | 闭式 exact,per-cell,无 subcycle |
| B-M4 | WB + κ + cool combined | 10/10 | **T ghost 必须独立 scalar mirror** |
| B-M5 | §E1+§E2 Suzuki driver + characteristic BC | 9/9 | Parseval √(2/N) + Alfvén 发射(T5a 10%/T5b 5%)+ §E2 吸收 vs reflect-wall(T6 R<0.35)+ **Leroy80/Cranmer07 WKB v⊥∝ρ^{-1/4} 外部对照(T7 5.6%,阈值 10%)** |
| B-M5.5 | §C8 chromo blend cool | 7/7 | operator-split blend 顺序 |
| B-M5.75 | 全 6 算子集成 smoke | 16/16 | driver 改 §E2 characteristic BC(Shoda18/Sakaue21 独立吻合);F-T3 残余 O(1e-6) 漂移来自 2D PLM+HLLD 截断 floor + cool/chromo Λt 残差,不是 BC 非守恒 |

**总计**:63/63 通过(B-M5 +1 for T6 §E2 absorber, +1 for T7 Leroy80/Cranmer07 WKB external benchmark; B-M5.75 +2 for F-T4 decomposition)。所有派生书 §B4/§C6/§C7/§E1/§E2 的 numerical gotcha 已在 `docs/derivations/mhd/sections/` 对应节补入"数值实现备忘"子章节。

---

## B-M6 — 占位

详细 setup 在进入时补。
