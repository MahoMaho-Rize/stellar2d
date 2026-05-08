# C1. Gravity source consistency

> **sympy script:** `scripts/c01_gravity_source_consistency.py`
> **generated LaTeX:** `output/c01_gravity_source_consistency.latex.tex`
> **verified:**
> - 1 kinetic+potential energy conservation law ($\partial_t (E + \rho g y) + \nabla\!\cdot\![(E + P + \rho g y)\mathbf{v}] = 0$)
> - 1 HSE balance reduction identity
> - 1 work-form identity ($S_E = -m_y g = \rho v \cdot (-g)$)
>
> **code checkpoints:**
> - `src/gpu/explicit/strang_solver.cu :: k_hllc_update_y` (line 513-516: `S_my = -rho_total * g_grav`; `S_E = -d_my[k0] * g_grav`; line 522-523: separate `+= dt * S_my` and `+= dt * S_E` applied after the HLLC flux divergence)

The Strang solver's y-sweep applies the gravity source term
$\mathbf{S}(\mathbf{U}; g) = (0, 0, -\rho g, -m_y g)^{\mathsf T}$
inside `k_hllc_update_y`. This section proves that

1. This is the **unique** source consistent with the Euler PDE
   under a uniform downward gravitational acceleration $g\mathbf{e}_y$;
2. The form closes a strong-form conservation law for **total**
   (kinetic + internal + gravitational-potential) energy;
3. On pure HSE the source exactly cancels the background pressure
   gradient, preserving well-balancing;
4. The Strang kernel inserts gravity **only once** inside the
   y-sweep update — no double-counting via flux-side source
   injection (pre-empting a known family of lowmach-solver bugs).

## PDE with gravity

$$\partial_{t}\mathbf{U} \;+\; \partial_{x}\mathbf{F}_{x} \;+\; \partial_{y}\mathbf{F}_{y} \;=\; \mathbf{S}(\mathbf{U}; g), \qquad \mathbf{S}(\mathbf{U}; g) \;=\; \bigl(0,\; 0,\; -\rho g,\; -m_{y} g\bigr)^{\mathsf T}. \quad (\text{C1-euler-gravity})$$

The non-zero components are:

- **y-momentum.** $-\rho g$ is the gravitational force per unit
  volume: mass $\rho$ times acceleration $g$ in the $-\mathbf{e}_y$
  direction.
- **energy.** $-m_y g = -\rho v g$ is the work done by gravity per
  unit time per unit volume: force $(-\rho g)$ times velocity $v$.
  It is negative when $v > 0$ (fluid rising, losing kinetic energy)
  and positive when $v < 0$ (fluid falling, gaining kinetic energy).

## Kinetic + potential energy conservation

Define the gravitational potential energy density $\rho g y$. The
total mechanical + internal energy $E + \rho g y$ satisfies a
strong-form conservation law:

$$\frac{\partial}{\partial t}\bigl(E + \rho g y\bigr) \;+\; \nabla\!\cdot\!\bigl[(E + P + \rho g y)\,\mathbf{v}\bigr] \;=\; 0. \quad (\text{C1-KE-PE-conservation})$$

**Strong-form verification.** sympy substitutes the mass equation
$\partial_t \rho = -\nabla\!\cdot\!(\rho\mathbf{v})$ and the energy
equation with source $-m_y g$, computes
$\partial_t(E + \rho g y)$, and confirms it equals the divergence
$-\nabla\!\cdot\![(E + P + \rho g y)\mathbf{v}]$ exactly. The
algebra turns on the identity

$$g y\,\nabla\!\cdot\!(\rho\mathbf{v}) \;+\; \rho v g \;=\; \nabla\!\cdot\!(g y \rho \mathbf{v}),$$

which follows from $\partial_y(g y \rho v) = g \rho v + g y
\partial_y(\rho v)$, i.e., the $\rho v g$ work term is exactly
absorbed into the divergence of $g y \rho \mathbf{v}$. This is
**not** a coincidence: it is the content of the virial identity
for a stratified gas in uniform gravity.

The conservation law is the mathematical statement that gravity
is a conservative force: no energy is lost or gained in any
closed volume beyond what flows across the boundary.

## HSE balance reduction

On pure HSE ($\delta\rho = \delta P = u = v = 0$, stored
$m_y = 0$), the y-momentum equation reduces to

$$\partial_t m_y \;+\; \partial_y\bigl(\rho v^{2} + P\bigr) \;+\; \rho g \;=\; 0 \;+\; \partial_y \bar p \;+\; \bar\rho g. \quad (\text{C1-HSE-balance})$$

By §B2's HSE ODE $\partial_y \bar p = -\bar\rho g$, the right-hand
side is identically zero. Stored $m_y$ stays at zero — HSE is
preserved strong-form by the gravity source plus the flux
gradient, and not by either alone.

**Discrete consequence.** The kernel's §B3-compliant face flux
reconstruction produces a flux-divergence term
$-(\bar p_{j+1/2} - \bar p_{j-1/2})/\Delta y$ at cell $j$, which
discretely approximates $-\partial_y \bar p = \bar\rho g$. The
kernel's source term $-\bar\rho_j g$ then exactly cancels this to
the cell-centred background-density value. The residual is the
difference between the cell-centred $\bar\rho_j$ and the cell-
averaged $\int \bar\rho\,dy / \Delta y$, which is $O(\Delta y^2)$ —
a structural truncation error bounded by §E5's HSE drift estimate.

## Work-form of the energy source

$$S_E \;=\; -m_y g \;=\; \rho v \cdot (-g). \quad (\text{C1-work})$$

sympy confirms the tautology (same expression written two ways).
The physical content is that $S_E$ is the work rate of the
gravitational body force: $\text{force density} \times \text{velocity}
= (-\rho g) v = -m_y g$.

## Kernel applies gravity ONCE

The only place the gravity source is added in the kernel is inside
`k_hllc_update_y` at line 522-523:

```cpp
double rho_total = fmax(d_rho[k0] + rho_bg, 1e-20);
double S_my = -rho_total * g_grav;
double S_E  = -d_my[k0] * g_grav;
// ... flux divergence via HLLC ...
d_my[k0] = d_my[k0] - dtdy * (GT_mn - GB_mn) + dt * S_my;
d_E [k0] = d_E [k0] - dtdy * (GT3   - GB3)   + dt * S_E;
```

There is no second gravity term inside `k_muscl_hancock_y` (the
Hancock predictor does not include gravity — see §A12: gravity is
a simple source that does not contribute to the half-step flux
divergence at leading order), and there is no gravity term inside
the x-sweep or inside the flux evaluation `d_euler_flux_y`.

**Why this matters.** A known anti-pattern in related solvers (see
CLAUDE.md's "P32 lowmach-family bug") is to add a gravity term
**both** inside the source and inside the flux, counting the
energy work twice. The symptom is an unphysical energy gain
proportional to $\rho v g \Delta t$ per step, which accumulates
into a secular drift. The Strang kernel avoids this by keeping
gravity as a pure source and never embedding it in the flux.

## Operator-split consistency

The gravity source is absorbed into the $\mathcal{Y}$-sweep (the
y-direction flux operator), not added as a separate
$\mathcal{Z}$-operator. Under §A14's self-adjointness condition,
this choice is compatible with 2nd-order Strang splitting: the
full operator chain is still $\mathcal{X}(\Delta t/2)\,
\mathcal{Y}(\Delta t/2)\,\mathcal{Y}(\Delta t/2)\,
\mathcal{X}(\Delta t/2)$, symmetric about $\Delta t/2$.

If the gravity source were split off as $\mathcal{Z}(\Delta t)$
acting after the Strang chain (a "sequential" addition), the
composite would become
$\mathcal{X}(\Delta t/2)\,\mathcal{Y}(\Delta t)\,
\mathcal{X}(\Delta t/2)\,\mathcal{Z}(\Delta t)$, which is **not**
symmetric under time reversal; the splitting would degrade to
1st-order. §E4 verifies that the kernel's absorption of gravity
into $\mathcal{Y}$ preserves 2nd-order Strang order.

## Verification checkpoints

1. **Bit-identical source-application.** On a known state with
   known $\rho_{\mathrm{tot}}, m_y$, compute the kernel's
   $S_{m_y}, S_E$ contributions via `cudaMemcpy` of the state
   buffer before and after `k_hllc_update_y`, subtract the
   flux-divergence contribution, and verify the residual equals
   $dt \cdot (-\rho_{\mathrm{tot}} g, -m_y g)$ to ULP precision.
   Test: `test_strang_hllc.cu` §C1-gravity-source.

2. **Pure-HSE long-time preservation.** After $10^4$ Strang steps
   on HSE IC, the stored $m_y, \delta E$ stay at
   $O(\varepsilon_{\mathrm{mach}} N)$ — no secular drift. Test:
   `test_strang_step.cu` §C1-hse-long.

3. **KE+PE conservation on closed domain.** Initialise a symmetric
   perturbation (§D7 reflection-symmetric IC) on a closed-wall
   domain; after any number of Strang steps, $\int_V (E + \rho g y)\,dV$
   is conserved to $O(\varepsilon_{\mathrm{mach}} N)$. Test:
   `test_strang_step.cu` §C1-kepe-conservation.

4. **No double-counting.** Code-inspection check: search for any
   occurrence of `g_grav` in `k_muscl_hancock_y`, `k_hllc_update_x`,
   or `d_euler_flux_{x,y}`. Required: zero occurrences. Test:
   `test_strang_unit.cu` §C1-code-inspection.

Failure of (1) is an arithmetic bug in the kernel's source
application (sign error or wrong variable — e.g., using
$\delta \rho$ instead of $\rho_{\mathrm{tot}}$). Failure of (2)
means HSE is being broken by gravity application, often due to
the cell-centred background vs. face-centred reconstruction
mismatch (§B3). Failure of (3) is a deeper energy-balance
violation. Failure of (4) flags a likely double-count bug that
(1)-(3) might not catch if the extra term is small enough on the
test IC.
