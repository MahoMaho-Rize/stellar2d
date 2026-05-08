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
| B-M2 | Spitzer κ₀T^{5/2} 各向异性导热 | §C6 已完成 | TBD | 🔜 |
| B-M3 | Townsend 光学薄冷却闭式积分 | §C7 已完成 | TBD | ⏳ |
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

**Setup**: TBD(开始时补)

派生基础:
- $\mathbf{q}_\mathrm{Spitzer} = -\kappa_\parallel \hat{\mathbf{b}}\hat{\mathbf{b}}\cdot\nabla T - \kappa_\perp(\nabla T - \hat{\mathbf{b}}\hat{\mathbf{b}}\cdot\nabla T)$
- $\kappa_\parallel = \kappa_0 T^{5/2}$,$\kappa_\perp \ll \kappa_\parallel$(各向异性比 ~10⁷ 在 corona)
- 显式抛物 CFL:$\Delta t < h^2 \rho / (2\,\kappa_\parallel (\gamma-1)/k_B)$—— 热扩散 severe,需 STS

### 验证计划(未开工)

1. **C6-T1**: hot spot沿 B 扩散 — 各向异性比 ≥ 10³(测试取有限 κ⊥ 避免 0/0)
2. **C6-T2**: $\kappa_\perp \to 0$ 极限退化 1D — 横向温度 frozen
3. **C6-T3**: Kirchhoff potential $U = \int \kappa_\parallel dT \propto T^{7/2}/\kappa_0$ 守恒
4. **C6-T4**: 熵产 ≥ 0 单调

---

## B-M3 / B-M4 — 占位

详细 setup 在进入时补。
