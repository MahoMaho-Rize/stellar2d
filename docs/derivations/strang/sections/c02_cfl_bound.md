# C2. CFL bound

> **sympy script:** `scripts/c02_cfl_bound.py`
> **generated LaTeX:** `output/c02_cfl_bound.latex.tex`
> **verifies:** 1 strong-form + 1 numerical identity — 1 Lax-
> Friedrichs amplification identity
> ($|g|^2 = 1 + (\nu^2 - 1)\sin^2(k\Delta x)$, strong-form sympy);
> 1 max-wave-speed identity
> ($\max(|u-c|, |u+c|) = |u| + c$; sympy cannot fold absolute-value
> expressions for symbolic sign, so this is verified at 100
> random samples with max residual $0$)
> **code checkpoints:**
> `src/gpu/explicit/strang_solver.cu :: k_strang_cfl`
> (line 529-556: computes $(|u|+c)/\Delta x + (|v|+c)/\Delta y$ per cell;
> host reduction gives global max; $\Delta t = \sigma / \max\{\cdot\}$)

The kernel's time-step selection implements a **conservative**
CFL condition based on a 1D von-Neumann analysis of the linear
scalar advection equation, combined into a single 2D-style
estimator. This section proves the underlying 1D stability bound
and documents how the Strang kernel's choice relates to the
split-vs-unsplit stability regions.

## 1D linear advection: Lax-Friedrichs von-Neumann analysis

For $u_t + a u_x = 0$ the Lax-Friedrichs (LF) update in
amplification form gives

$$|g(k)|^{2} \;=\; 1 \;+\; (\nu^{2} - 1)\,\sin^{2}(k\,\Delta x), \qquad \nu \;=\; \frac{a\,\Delta t}{\Delta x}. \quad (\text{C2-1D-LF})$$

**Stability.** Von-Neumann requires $|g(k)| \le 1$ for all $k$.
Since $\sin^2(k\Delta x) \in [0, 1]$, the worst case is
$\sin^2 = 1$, giving $|g|^2_{\max} = 1 + (\nu^2 - 1) = \nu^2$, and
$|g|^2 \le 1$ iff $|\nu| \le 1$.

sympy verifies the amplification identity via direct expansion
$\mathrm{Re}(g)^2 + \mathrm{Im}(g)^2$ with $g = \cos\theta - i\nu\sin\theta$;
the analytic simplification reduces to
$\cos^2\theta + \nu^2\sin^2\theta = 1 + (\nu^2-1)\sin^2\theta$.
The stability condition $|\nu| \le 1$ then follows from elementary
calculus (supremum over $\theta$).

## Fastest Euler wave speed

From §A3, the flux Jacobian $\mathcal{A}_x$ has eigenvalues
$\{u-c, u, u, u+c\}$. The fastest signal speed — the absolute
value of the largest eigenvalue — is

$$\lambda_{\max}(\mathcal{A}_x) \;=\; \max\bigl(|u-c|,\, |u|,\, |u+c|\bigr) \;=\; |u| + c. \quad (\text{C2-euler-wave})$$

**Proof.** The identity $\max(|u-c|, |u+c|) = |u| + c$ is the
standard triangle inequality: for any reals $u, c$ (with $c > 0$),
$|u+c| + |u-c| \ge |u+c| - |u-c| = 2u$ if $u \ge 0$ (giving $|u+c|
= u+c$), or $2c$ if $u \le 0$ (giving $|u-c| = c-u = |u|+c$). In
either case $\max = |u| + c$.

sympy cannot fold nested absolute values for symbolic $u$ (sign
unknown), so this is verified at 100 random $(u, c)$ samples; the
residual is exactly $0$ in all samples.

## Strang-split 2D CFL

Each 1D sweep in the Strang kernel has its own stability condition:

$$\Delta t \cdot \max\biggl\{\frac{|u|+c}{\Delta x}\biggr\} \le \sigma_{1D}, \qquad \Delta t \cdot \max\biggl\{\frac{|v|+c}{\Delta y}\biggr\} \le \sigma_{1D}. \quad (\text{C2-Strang-cfl})$$

For the linear LF scheme $\sigma_{1D} = 1$. For MUSCL-Hancock the
1D stability limit is slightly tighter ($\sigma_{1D} \approx 2/3$
or $0.5$ depending on the limiter family) because the higher-order
reconstruction adds dispersive error that can go unstable at
$\nu = 1$.

Strang splitting means each sweep updates an intermediate state,
and the stability region is the **rectangular** product
$\{(\nu_x, \nu_y) : \nu_x \le \sigma_{1D}, \nu_y \le \sigma_{1D}\}$.
This allows $\nu_x = \nu_y = 1$ simultaneously (linear LF limit) —
a full factor of 2 more than the unsplit case below.

## Combined 2D estimator (used in kernel)

The kernel's CFL buffer computes

$$\text{buf}[i, j] \;=\; \frac{|u_{ij}| + c_{ij}}{\Delta x} \;+\; \frac{|v_{ij}| + c_{ij}}{\Delta y}. \quad (\text{C2-combined})$$

The global $\Delta t$ is $\sigma / \max\{\text{buf}\}$. This enforces

$$\nu_x + \nu_y \;\le\; \sigma, \qquad \nu_x = (|u|+c) \Delta t / \Delta x, \;\; \nu_y = (|v|+c) \Delta t / \Delta y.$$

This is the **unsplit 2D MUSCL** form of the CFL condition, which
is more conservative than the Strang-split rectangular region. It
over-restricts the time step for a Strang-split scheme — at the
cost of a factor of 2 in $\Delta t$ it buys robustness across
transients where the Strang splitting's "independent sweeps"
assumption breaks down (e.g., during large shocks where the
reconstruction error grows faster than the linear LF analysis
predicts).

## Stability margin comparison

| Scheme / bound | $\sigma_{\max}$ |
|---|---|
| split 1D linear LF | 1.0 |
| split 1D MUSCL | $\sim 0.67$ |
| unsplit 2D LF | 0.5 (diamond: $\nu_x + \nu_y \le 1$, worst corner at $\nu_x = \nu_y = 0.5$) |
| unsplit 2D MUSCL | $\sim 0.4$ |
| **kernel default** | **0.4** |

The kernel's default $\sigma = 0.4$ is at the unsplit 2D MUSCL
limit; it provides $\sim 60\%$ margin below the Strang-split 1D
LF bound. The choice trades per-step efficiency for robustness
across realistic hydrodynamic tests (shocks, contact discontinuities,
gravity-driven convection) where the linear analysis is optimistic.
Users can relax to $\sigma = 0.8$ or $1.0$ for smooth flows (e.g.,
§D1 entropy wave) if per-step cost matters.

## Acoustic vs. advective CFL

In the stratified-atmosphere setup (HSE background $\bar\rho, \bar p$
with $c \approx \sqrt{\gamma \bar p / \bar\rho}$), the sound speed
$c$ dominates $|u|, |v|$ throughout — the flow is low-Mach. The
CFL condition is therefore **acoustic**:

$$\Delta t \;\sim\; \frac{\sigma \min(\Delta x, \Delta y)}{c_{\max}}.$$

This is the restriction that motivates the LM-HLLC blending of
§C3: the pressure-dissipation of standard HLLC at this $\Delta t$
smears acoustic waves too aggressively, so a Mach-dependent blend
reduces the dissipation without violating CFL.

## ✅ Verification checkpoint (to be wired)

1. **Kernel CFL reduction.** After `k_strang_cfl` populates the
   buffer, the host reduction computes $\max\{\text{buf}\}$. The
   $\Delta t$ selected is $\sigma / \max\{\text{buf}\}$; verify
   that $\nu_x + \nu_y$ at the globally-tightest cell equals
   $\sigma$ to ULP precision. Test: `test_strang_step.cu` §C2-cfl.

2. **Linear stability.** On an IC with known advection velocity
   (e.g., uniform $u = 0.5$, $v = 0$, $c = 1$), run with
   $\sigma = 0.99$; the scheme remains stable for $> 100$ steps.
   Run with $\sigma = 1.01$ (just above the split 1D bound) and
   check that the scheme becomes unstable (entropy wave amplitude
   grows). Test: `test_strang_step.cu` §C2-near-limit.

3. **Low-Mach-dominant CFL.** On the HSE IC, measure $\Delta t$
   and confirm it is acoustic-limited ($\Delta t \approx \sigma
   \Delta y / c_{\max}$), not advective. Test:
   `test_strang_step.cu` §C2-acoustic.

Failure of (1) is a kernel arithmetic bug. Failure of (2) —
specifically stability at $\sigma > 1$ — indicates the linear
analysis is wrong, but in practice the kernel becomes unstable
slightly below $\sigma = 1$ due to non-linear terms. Failure of
(3) would mean the CFL formula is not computing the expected
acoustic speed — likely a missing factor of $c$ or wrong density
reference.
