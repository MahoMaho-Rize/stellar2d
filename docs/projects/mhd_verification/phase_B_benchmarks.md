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
| B-M4 | 分层 + 导热 + 冷却 combined | §B1/4/C6/7 | TBD | ⏳ |

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

## B-M4 — 占位

详细 setup 在进入时补。
