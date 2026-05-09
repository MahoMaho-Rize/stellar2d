# E4. PML-style absorbing sponge for outgoing Alfvén waves

> **sympy script:** `scripts/e4_pml_sponge.py` (9 identities verified;
> implicit-Euler eigenvalues shown unconditionally L-stable; T7
> numerical attenuation $e^{-\tau_\text{PML}} = 0.207$ one-way,
> 4.3% worst-case round-trip reflection).
> **verified:**
> characteristic-split damping ODE $\partial_t \tilde z^+ = -\sigma z^+$,
> $\partial_t \tilde z^- = 0$; local energy decay
> $\tfrac{\mathrm d}{\mathrm dt}|\tilde z^+|^2 = -2\sigma|\tilde z^+|^2
> \le 0$; C⁰-matching at $y_\text{pml}$ ($\sigma(y_\text{pml}) = 0$)
> eliminates impedance jump; primitive-variable drag matrix
> $\mathbf M$ has eigenvalues $\{0, 2\}$ (rank-1 by construction, $z^-$
> channel preserved); implicit-Euler update $(\mathbf I + \tfrac{\Delta
> t \sigma}{2}\mathbf M)^{-1}$ eigenvalues $\{1, 1/(1 + \Delta t \sigma)\}$
> (unconditional L-stability); analytic outgoing attenuation
> $\tau_\text{PML} = \int_{y_\text{pml}}^{L_y} \sigma/v_A\,\mathrm dy$.
> **code checkpoints:**
> `athena_mhd_kernels.cu::k_athmhd_apply_pml`,
> `athena_mhd_solver.cu::apply_pml(dt)`,
> `athena_mhd_solver.cuh::AthenaMHDSolver::pml_on` (flag) +
> `pml_y_start`, `pml_sigma0` (profile),
> `tests/test_athena_mhd_driver.cu::E1-T7` (Hankel-exact benchmark in
> non-PML diagnostic region, target $|v_\perp(y_2)/v_\perp(y_1) -
> R_\text{Hankel}| < 3\%$).

## Motivation

§E3 gives the continuum-ideal non-reflecting top BC for outgoing
Alfvén waves; §E3.5 (Stone-1999 recursion) would give the
discrete-consistent version on a **uniform** background but is
unstable in a stratified atmosphere because the outgoing wave is a
Hankel function, not a plane wave. The derivation-clean fix that
works in both uniform and stratified regimes is a PML (Perfectly
Matched Layer) absorbing sponge in the upper portion of the column.

Classical PML (Bérenger 1994) works on Maxwell's equations by splitting
fields into artificial components with distinct damping profiles; the
characteristic-variable reformulation (Nataf 2013; Colonius 2004 review)
reduces to a simple damping term on the outgoing invariant when the
system is already characteristic-diagonal. The 1D-in-y Alfvén channel
in linearised 2D MHD has exactly this structure — two variables
$(v_x, B_x)$ that diagonalise into $\tilde z^+$ (upgoing) and
$\tilde z^-$ (downgoing) — so the PML reduces to a single drag on
$\tilde z^+$ inside a chosen top-layer region $y \ge y_\text{pml}$.

## PML equations

Inside the absorbing layer:

$$\boxed{\;
\partial_t \tilde z^+ + v_A(y)\,\partial_y \tilde z^+
  = -\sigma(y)\,\tilde z^+,
\qquad
\partial_t \tilde z^- - v_A(y)\,\partial_y \tilde z^- = 0.
\;}$$
<!-- label=E4-characteristic -->

* **$\tilde z^+$ is damped.** The term $-\sigma(y) \tilde z^+$ drains
  outgoing-wave energy as the wave crosses the sponge. Any $\tilde z^+$
  amplitude that reaches the numerical top wall has been attenuated by
  $e^{-\tau_\text{PML}}$ (see attenuation formula below), so the
  reflected wave at the wall — even if the wall BC is imperfect — is
  at most $e^{-2\tau_\text{PML}}$ of the original outgoing amplitude.
* **$\tilde z^-$ is untouched.** The incoming invariant continues to
  advect downward losslessly; no spurious incoming wave is generated
  by the PML.
* **Impedance matching at $y = y_\text{pml}$.** Choose $\sigma(y)$ C⁰
  with $\sigma(y_\text{pml}) = 0$ and polynomial growth thereafter;
  the PML PDE reduces to the lossless PDE at the interface, so there
  is NO reflection at the PML entry.

## Profile choice

We use the standard quadratic PML profile (Bérenger 1994 original):

$$\sigma(y) = \begin{cases}
0, & y < y_\text{pml},\\[4pt]
\sigma_0\bigl(\dfrac{y - y_\text{pml}}{L_y - y_\text{pml}}\bigr)^2,
& y \ge y_\text{pml}.
\end{cases}$$
<!-- label=E4-profile -->

This C¹-continuous profile gives a smooth transition (no ghost-cell
slope jump into the sponge). Cubic or higher-order profiles are
marginally better but add complexity; quadratic is standard and
sufficient for the T7 benchmark.

## Primitive-variable drag

Substituting $\tilde z^\pm = \mp v_x + B_x/\sqrt{\rho_0}$ and splitting
off the damping term:

$$\boxed{\;
\begin{aligned}
\partial_t v_x\bigr|_\text{PML}  &=
  \tfrac{\sigma(y)}{2}\bigl[-v_x + B_x/\sqrt{\rho_0}\bigr],\\[4pt]
\partial_t B_x\bigr|_\text{PML}  &=
  \tfrac{\sigma(y)}{2}\bigl[\sqrt{\rho_0}\,v_x - B_x\bigr].
\end{aligned}
\;}$$
<!-- label=E4-primitive -->

The z-polarised channel $(v_z, B_z)$ has the identical form by the
symmetry of the linearised 2D MHD system.

## Drag matrix and implicit-Euler solver

The primitive drag is a linear system $\partial_t \mathbf u = -\tfrac{\sigma}{2} \mathbf M \mathbf u$
with $\mathbf u = (v_x, B_x)^T$ and

$$\mathbf M = \begin{pmatrix} 1 & -1/\sqrt{\rho_0} \\
-\sqrt{\rho_0} & 1 \end{pmatrix}.$$

$\mathbf M$ is rank-1 (determinant 0) with eigenvalues $\{0, 2\}$
corresponding to $\tilde z^-$ (undamped) and $\tilde z^+$ (damped at
rate $\sigma$). Implicit Euler:

$$\begin{pmatrix} v_x^{n+1}\\ B_x^{n+1}\end{pmatrix}
 = \bigl(\mathbf I + \Delta t \cdot \tfrac{\sigma}{2}\mathbf M\bigr)^{-1}
 \begin{pmatrix} v_x^{n}\\ B_x^{n}\end{pmatrix}.$$
<!-- label=E4-implicit -->

sympy closed form for the inverse:

$$\bigl(\mathbf I + \Delta t \tfrac{\sigma}{2}\mathbf M\bigr)^{-1}
 = \frac{1}{1 + \Delta t \sigma}\begin{pmatrix}
  \tfrac{\Delta t \sigma + 2}{2}
  & \tfrac{\Delta t \sigma}{2\sqrt{\rho_0}}\\[4pt]
  \tfrac{\Delta t \sigma \sqrt{\rho_0}}{2}
  & \tfrac{\Delta t \sigma + 2}{2}
 \end{pmatrix}.$$
<!-- label=E4-implicit-inverse -->

Eigenvalues $\{1, 1/(1 + \Delta t \sigma)\}$ are both in $[0, 1]$ for
any $\Delta t > 0$, so the update is **unconditionally L-stable**.

## Outgoing attenuation

Steady-state solution of the damped characteristic ODE:

$$\tau_\text{PML} = \int_{y_\text{pml}}^{L_y}
  \frac{\sigma(y)}{v_A(y)}\,\mathrm dy,\qquad
  \frac{\lvert\tilde z^+(L_y)\rvert}{\lvert\tilde z^+(y_\text{pml})\rvert}
    = e^{-\tau_\text{PML}}.$$
<!-- label=E4-attenuation -->

**T7 numbers** (H=1, f=2, $B_{y0}=0.5$, $L_y = 2$, $y_\text{pml} = 1.5$,
$\sigma_0 = 10$, quadratic profile):

- $v_A(y_\text{pml}) = 0.5 / \sqrt{e^{-1.5}} \approx 1.059$
- $\Delta = L_y - y_\text{pml} = 0.5$
- $\tau_\text{PML} = \sigma_0 \Delta / (3 v_A) \approx 1.574$
  (closed form for quadratic profile over constant $v_A$)
- One-way attenuation: $e^{-1.574} \approx 0.207$ (79% absorbed)
- Worst-case round-trip reflection: $(0.207)^2 \approx 0.043$ (4.3%)

For stronger absorption increase $\sigma_0$; with $\sigma_0 = 20$ the
round-trip reflection drops to 0.18%.

## Stability constraint

Explicit Euler would require $0 < \sigma \Delta t < 2$ for monotone
decay; the implicit-Euler implementation removes this constraint
entirely. The CFL-limited hydrodynamic $\Delta t$ is always much
smaller than $2/\sigma_0$ for reasonable $\sigma_0$ (e.g. at T7,
$\Delta t \sim 3\times 10^{-3}$, $\sigma_0 = 10$, giving $\sigma \Delta t
\le 0.03 \ll 2$), so even an explicit implementation would be safe.
The implicit version is used as a safety net.

## Operator-split placement

The PML is a pure source term and integrates via operator splitting
with the hyperbolic VL2 step:

```
U^{n+1} = L_PML(Δt) ∘ L_chromo(Δt) ∘ L_cool(Δt) ∘ L_cond(Δt) ∘ L_vl2(U^n; Δt, WB)
```

Same 1st-order Godunov splitting as the other source operators
(§B4 + §C6 + §C7 + §C8). PML runs LAST so it can absorb whatever
outgoing amplitude the hyperbolic + other-source chain produced in
this step. The `apply_driver` call is unchanged (it just updates
`driver_t_now`; actual driver ghost fill happens in `fill_ghost` at
the next step).

## Where to place $y_\text{pml}$

Rule: place $y_\text{pml}$ at least $2\lambda_\text{Alfvén}(y_\text{pml})$
below the wall to allow a full wavelength of attenuation before the
wall. For T7 at $y_\text{pml} = 1.5$, $v_A \approx 1.06$, $f = 2$,
$\lambda = v_A/f = 0.53$, so $\Delta = 0.5 \approx \lambda$ — marginal
but sufficient for the quadratic profile (effective attenuation
depth is $\sim \Delta/3$).

For steady-state Alfvén-wind runs targeting a specific benchmark
measurement at height $y_\text{meas}$, choose $y_\text{pml} > y_\text{meas}$
so the PML does not affect the measurement.

## Default-off

The `pml_on` flag is false by default. All existing Phase-B tests
(B-M1–B-M5.75) run with `pml_on = false` and are bit-identical to
pre-§E4. Only B-M5 T7 (and future B-M6 wind-column runs) enable it.

## Broadband driver compatibility

Unlike §E3.5 Stone-1999 which needed a representative frequency, the
PML sponge damps ALL frequencies simultaneously with rate $\sigma(y)$
(no $k$-dependence in the damping term). Broadband §E1 drivers work
out-of-the-box; no parameter tuning per frequency band.

## Verification checkpoints

- `scripts/e4_pml_sponge.py` — sympy: 9 identities; closed-form
  implicit-Euler inverse matrix; outgoing-attenuation formula verified
  for T7 parameters (attenuation = 0.207 one-way, round-trip
  reflection ≤ 4.3%).
- `tests/test_athena_mhd_driver.cu::E1-T7` — Hankel benchmark in
  non-PML region ($y \in [0.25, 1.25]$ with PML at $y \ge 1.5$).
  Measured $v_\perp(y_2)/v_\perp(y_1)$ agrees with Hankel envelope
  within 3%.
- Regression: all Phase-B tests with `pml_on = false` bit-identical
  to pre-§E4 (checked via full suite rerun after §E4 lands).
- y-profile sanity: inside $y \in [0.25, 1.25]$ the RMS envelope is
  monotonic with |err vs Hankel| < 3% at every sampled height. No
  standing-wave pattern.
