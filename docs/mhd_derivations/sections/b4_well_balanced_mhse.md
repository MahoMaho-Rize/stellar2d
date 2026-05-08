# B4. Well-balanced MHSE at the operator level

> **sympy script:** `scripts/b4_well_balanced_mhse.py`
> **verified:** $F_\mathrm{wb}(\mathbf{U}_\mathrm{hse}) \equiv 0$ by
> construction; linear-wave Jacobian preserved,
> $\partial_\mathbf{U}F_\mathrm{wb} = \partial_\mathbf{U}R$ at
> $\mathbf{U}_\mathrm{hse}$ — perturbation dynamics unchanged.
> **code checkpoints:**
> `athena_mhd_solver.cu::compute_residual_wb`,
> `athena_mhd_solver.cu::snapshot_hse_if_needed`,
> `tests/test_athena_mhd_wind_hse_stationary.cu`.

## The problem

Direct application of the discrete residual
$R(\mathbf{U}) = -\partial_i \mathbf{F}_i + \mathbf{S}$ to a piecewise-
constant MHSE atmosphere leaves a truncation residual of order
$\mathcal{O}(\Delta r^{2})\,\rho g$. For a Suzuki-style wind run this
residual drives a $|\delta v_r|/c_s \sim 10^{-2}$ transient that
corrupts the wind-mass-loss diagnostic for hundreds of crossing
times.

## Well-balanced residual (Bermúdez-Vázquez 1994 / Botta+04)

$$\boxed{F_{\mathrm{wb}}(\mathbf{U}) \equiv R(\mathbf{U}) - R(\mathbf{U}_{\mathrm{hse}}).}$$

By construction
$F_{\mathrm{wb}}(\mathbf{U}_{\mathrm{hse}}) = 0$ (sympy-verified
trivially). So an atmosphere at MHSE stays at MHSE to machine
precision for arbitrarily long time, regardless of reconstruction
order or Riemann solver.

## Preservation of linear-wave dynamics

A small perturbation $\delta\mathbf{U}$ around the MHSE state:

$$F_\mathrm{wb}(\mathbf{U}_\mathrm{hse} + \delta\mathbf{U}) = \left.\partial_\mathbf{U}R\right|_{\mathbf{U}_\mathrm{hse}}\cdot\delta\mathbf{U} + \mathcal{O}(\delta\mathbf{U}^2).$$

Sympy-verified that the Jacobian of $F_\mathrm{wb}$ at
$\mathbf{U}_\mathrm{hse}$ **equals** the Jacobian of $R$ at the same
state. Thus linear MHD waves (Alfvén, magnetosonic) propagate
through $F_\mathrm{wb}$ identically to $R$ — no spurious damping or
dispersion is introduced by the well-balancing subtraction.

## What MHSE means in a super-radial flux tube

Reprise from §B1:

$$\boxed{\partial_r p + \rho g + B_r^2\,\partial_r(\ln A) = 0.}$$

Note the **last term** is critical. A naïve WB that only cancels
gravity (the hydrodynamic well-balancing) will leave an
$\mathcal{O}(B_r^2)$ residual. The correction for the super-radial
tube must include the area-divergence term.

## Practical recipe

1. **Snapshot MHSE on the grid.** Integrate the ODE
   $\partial_r p_\mathrm{hse} = -\rho_\mathrm{hse} g - B_r^2\,\partial_r(\ln A)$
   using the given $(\rho_\mathrm{hse}, B_r, A)$ profiles.
2. **Compute $R(\mathbf{U}_\mathrm{hse})$ once** at startup, store per
   cell. (Assumes $\mathbf{U}_\mathrm{hse}$ is time-independent — true
   for stationary background.)
3. **Evolve** $\partial_t\mathbf{U} = R(\mathbf{U}) - R(\mathbf{U}_\mathrm{hse})$.
4. **Initial condition**: seed $\mathbf{U}^0 = \mathbf{U}_\mathrm{hse}$.
   Initial RHS identically zero → no transient.

## Failure modes observed elsewhere

- `radial1d` without this correction: Newton iteration looks
  converged ($|F|<10^{-9}$) but the HSE slowly drifts at machine
  precision × resolution. Seen in pre-MS KH attempts: HSE-Newton is
  stable but cannot initiate KH because $F_v \equiv 0$ at MHSE.
- `cart_ale2` with WB: integration-grade HSE stability for
  $10^4$ crossings.

## Snapshot invalidation

If $(\rho_\mathrm{hse}, B_r, A)$ profiles change (e.g., user-driven
parameter sweep), re-snapshot. The kernel can detect staleness via a
hash of the profile arrays.

## ✅ Verification checkpoints

- `tests/test_athena_mhd_wind_hse_stationary.cu` — MHSE atmosphere,
  $10^4$ acoustic crossings, assert $\max|v_r|/c_s < 10^{-8}$ (with
  WB on) vs $\sim 10^{-2}$ (with WB off). The "off" case is a
  **positive control**: if turning off WB doesn't produce a
  transient, the WB machinery is a no-op and should be audited.

## 数值实现备忘 (not in formal derivation)

以下 3 条是 Phase B-M1 实测 (commit `fdbe383`, `test_athena_mhd_hse_preserve.cu`
6/6 通过) 发现的离散化陷阱。派生层面 $F_\mathrm{wb}(\mathbf{U}_\mathrm{hse}) \equiv 0$
是解析恒等式,但在 VL2 + PLM + reflective wall 的实现栈里,以下三点任何一条
写错,都会把"machine precision"退化成 $\sim 10^{-2}$–$10^{-3}$ 的漂移。

1. **VL2 两阶段必须分开存 defect $R(\mathbf{U}_\mathrm{hse})$。**
   Predictor 阶段走 donor-cell (order=1),corrector 阶段走 PLM (order=xorder);
   两者的**离散残差** $R(\mathbf{U}_\mathrm{hse})$ 在有限精度下并不相等
   (重建顺序不同 → face 值不同 → flux 不同)。只存一份 defect 做两次
   subtract,实测残留 $\sim 0.8\%$ drift;两份独立 defect (`d_rhs_hse_s1_*`,
   `d_rhs_hse_s2_*`,在 `apply_flux_divergence_and_ct` 按 stage 路由) 才到 ULP。

2. **必须对全 6 个守恒量 $(\rho, m_x, m_y, m_z, E, B_z)$ 都 subtract。**
   朴素直觉是只对有重力源的 $(m_x, m_y, m_z, E)$ 减。实测:reflective 壁
   上 $\rho$ 和 $B_z$ 的 flux 残差虽然是 ULP 级 ($\sim 10^{-16}$),但 1000
   步累积可以放大到 $\sim 1\%$ 的 $\delta\rho$ 漂移,使 B3 ($\delta\rho$)
   断言失败。WB 是**代数对消** (identical cancellation),不是"只减主要项",
   6 个守恒量必须全部参与。

3. **Snapshot 时两阶段共用同一份 $\mathrm{prim}(\mathbf{U}_\mathrm{hse})$,
   不要模拟 stage-2 swap。**
   实 `step()` 里 stage-2 的 flux 由 $\mathrm{prim}(\mathbf{U}^*)$ 计算,因为
   stage-1 末尾做了 swap + refill。但在 WB 完美时,$\mathbf{U}^* \equiv
   \mathbf{U}_\mathrm{hse}$ — stage-2 实际"看到"的就是 $\mathrm{prim}(\mathbf{U}_\mathrm{hse})$
   本身。snapshot 里只需 `cons_to_prim(U_hse)` 一次,两阶段共用;
   若人为加 swap 去模拟 stage-2 的 $\mathbf{U}^*$,反而 capture 了"未 WB
   过的 $\mathbf{U}^*$"的假残差,破坏自洽。

共通 takeaway: WB 是**两端离散表达式 bit-wise 相同**才 cancel,不是
"物理上等价"就行。任何改动重建顺序、变量顺序、或两阶段间 state 语义的
PR,都必须重跑 B-M1 以验证 ULP 对齐。
