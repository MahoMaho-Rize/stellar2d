# E3. Top outgoing characteristic BC for 2D MHD Alfvén wind

> **sympy script:** `scripts/e3_top_outgoing_bc.py`
> **verified:** Alfvén invariants $\tilde z^\pm$ advect at $\pm v_A$ (so
> $\tilde z^+$ is OUTGOING and $\tilde z^-$ is INCOMING at the top
> boundary $y = L_y$ — roles flipped from §E2); non-reflecting BC
> $\tilde z^-|_\text{top,ghost} = 0$, $\tilde z^+|_\text{top,ghost} =
> \tilde z^+|_\text{top,int}$; ghost-fill closure for primitives;
> reflection coefficient $R_\text{top} = 0$ at linear order; outgoing
> amplitude transmitted unchanged; quiescent interior $\Rightarrow$
> zero ghost; z-polarised Alfvén channel absorbs identically; face-B
> consistency; composite §E2 + §E3 linear steady state is unique and
> well-posed ($\tilde z^+(y) = -2 v_x^\text{drv}$, $\tilde z^-(y) = 0$);
> WKB growth law $A \propto \rho^{-1/4}$ (Leroy 1980, Cranmer+2007
> eq. 16) derived from §E3 steady state.
> **code checkpoints:**
> `athena_mhd_kernels.cu::k_athmhd_ghost_y_top_outgoing_cc` +
> `k_athmhd_ghost_y_top_outgoing_face`,
> `athena_mhd_solver.cuh::AthenaMHDSolver::top_outgoing` (flag),
> `athena_mhd_solver.cu::fill_ghost` (dispatch gate),
> `tests/test_athena_mhd_driver.cu::E1-T7` (Leroy80 / Cranmer07 WKB
> benchmark $v_\perp \propto \rho^{-1/4}$ within 10%).

## Motivation

§E2 gave a characteristic bottom BC that drives a $\tilde z^+$ Alfvén
wave into the domain and absorbs the returning $\tilde z^-$ exactly at
linear order. A complete Alfvén-wave wind column also needs a clean
**top** boundary that lets the upgoing $\tilde z^+$ wave EXIT without
reflection.

In the prototype T7 we observed that with the v1 top-reflect wall the
round-trip standing wave between the §E2 driver and a hard top
accumulates PLM+HLLD noise each transit; after about two $\tau_\text{top}$
the timestep collapsed from $\sim 5\times 10^{-3}$ to $\sim 10^{-20}$
and the column effectively froze. The pathology is structural:

* §E2 bottom injects $\tilde z^+ = -2 v_x^\text{drv}$ every step.
* Reflective top flips $\tilde z^+ \to \tilde z^-$ with sign change.
* That $\tilde z^-$ returns to the bottom, is picked up by the §E2
  absorbing extrapolation, and sets the bottom ghost $v_x$ and $B_x$
  away from pure-driven values.
* Repeat: each round trip bootstraps a higher-harmonic standing wave
  on top of the driver, PLM grows the harmonic, and the fast-wave
  speed in the growing standing wave blows up.

This is exactly the scenario §E2 memo point 5 already flagged
("Top BC is not an Elsässer absorber in v1 — must be patched before
running a full flux-tube wind to steady state"). §E3 is that patch,
derivation-first.

## Setup

Identical linearisation as §E2: around background
$(\rho, \mathbf v, p, \mathbf B) = (\rho_0, 0, p_0, B_{y0} \hat y)$.
The Alfvén-$x$ channel obeys

$$\partial_t v_x = \frac{B_{y0}}{\rho_0}\,\partial_y B_x,\qquad
  \partial_t B_x = B_{y0}\,\partial_y v_x,$$

with Riemann invariants

$$\tilde z^\pm = \mp v_x + B_x/\sqrt{\rho_0},\qquad
  \partial_t \tilde z^\pm \pm v_A\,\partial_y \tilde z^\pm = 0,\qquad
  v_A = B_{y0}/\sqrt{\rho_0}.$$

**Key observation.** The sign-of-propagation algebra is unchanged, but
at the top boundary $y = L_y$ the physical roles of $\tilde z^+$ and
$\tilde z^-$ are opposite to §E2:

| Invariant | Speed | Role at $y = 0$ (§E2) | Role at $y = L_y$ (§E3) |
|---|---|---|---|
| $\tilde z^+$ | $+v_A$ | INCOMING (driver) | **OUTGOING** (exits top) |
| $\tilde z^-$ | $-v_A$ | OUTGOING (extrapolated) | **INCOMING** (from above) |

The non-reflecting top BC is therefore dual to §E2: specify the
*incoming* invariant (zero, since there is no source outside the
domain) and extrapolate the outgoing one.

## Characteristic top BC

$$\boxed{\;
\tilde z^-\bigr|_\mathrm{top\;ghost} = 0,\qquad
\tilde z^+\bigr|_\mathrm{top\;ghost}
   = \tilde z^+\bigr|_\mathrm{top\;int}\;}$$
<!-- label=E3-BC -->

Inverting via $v_x = (\tilde z^- - \tilde z^+)/2$,
$B_x/\sqrt{\rho_0} = (\tilde z^+ + \tilde z^-)/2$ gives the primitive
closure:

$$\boxed{\;
v_x\bigr|_\mathrm{top\;ghost}
   = \tfrac{1}{2}\bigl[v_x^\mathrm{int} - B_x^\mathrm{int}/\sqrt{\rho_0}\bigr],
\qquad
\frac{B_x\bigr|_\mathrm{top\;ghost}}{\sqrt{\rho_0}}
   = \tfrac{1}{2}\bigl[-v_x^\mathrm{int} + B_x^\mathrm{int}/\sqrt{\rho_0}\bigr]\;}$$
<!-- label=E3-ghost-fill -->

The non-Alfvén channels (density, pressure, normal momentum, $v_y$) use
a standard outflow/zero-gradient mirror on the top, since the
acoustic/entropy modes do not couple to the Alfvén sector at linear
order.

## Reflection coefficient

For a pure upgoing incident pulse
$\tilde z^+|_\mathrm{top,int} = Z_0$, $\tilde z^-|_\mathrm{top,int} = 0$,
the corresponding primitives are $v_x^\mathrm{int} = -Z_0/2$,
$B_x^\mathrm{int}/\sqrt{\rho_0} = Z_0/2$. Plugging into the ghost
formula and recomputing:

$$R_\mathrm{top} \equiv
  \frac{\tilde z^-\bigr|_\mathrm{top\;ghost}}
       {\tilde z^+\bigr|_\mathrm{top\;int}}
  = 0.$$
<!-- label=E3-reflection -->

Zero by construction at linear order — the sympy script verifies this
as Identity 3.

## Composite §E2 + §E3 linear steady state

In the linear steady state ($\partial_t = 0$) the Alfvén equations
collapse to $\partial_y \tilde z^\pm = 0$, i.e. $\tilde z^+$ and
$\tilde z^-$ are constants along the column. The bottom (§E2) and top
(§E3) BCs give

$$\tilde z^+(y) = -2\,v_x^\mathrm{drv},\qquad
  \tilde z^-(y) = 0,\qquad \forall y \in [0, L_y].$$
<!-- label=E3-composite-steady -->

In terms of primitives:
$v_x(y) = -\tilde z^+(y)/2 = v_x^\mathrm{drv}$,
$B_x(y) = \sqrt{\rho_0}\,\tilde z^+(y)/2 = -\sqrt{\rho_0}\,v_x^\mathrm{drv}$.

At the nonlinear level this picture is modified by stratification
(the background $\rho_0$ varies with $y$). The WKB analysis of
Leroy 1980 / Velli 1993 / Cranmer+2007 upgrades this to:

$$\frac{\mathrm d}{\mathrm d y}\bigl[A^2\,\rho\,v_A\bigr] = 0
  \quad\Longrightarrow\quad A \propto \rho^{-1/4},$$
<!-- label=E3-wkb-growth -->

where $A(y) = |v_\perp(y)|$ is the local Alfvén amplitude. This is
the external-literature benchmark tested by B-M5 T7.

## z-polarised Alfvén channel

The $(v_z, B_z)$ channel has no driver in v1, so the top BC is a pure
absorber with $\tilde z^\pm_z|_\mathrm{top\;ghost} = 0$ and
$\tilde z^+_z|_\mathrm{top\;ghost} = \tilde z^+_z|_\mathrm{int}$. The
closed form matches the $x$-polarisation with
$(v_x, B_x) \to (v_z, B_z)$:

$$v_z\bigr|_\mathrm{top\;ghost}
   = \tfrac{1}{2}(v_z^\mathrm{int} - B_z^\mathrm{int}/\sqrt{\rho_0}),\qquad
  B_z\bigr|_\mathrm{top\;ghost}
   = \tfrac{1}{2}(-\sqrt{\rho_0}\,v_z^\mathrm{int} + B_z^\mathrm{int}).$$

## Face-B consistency

On the Yee grid the cell-centred $B_x^\mathrm{cc,ghost\;top}$ equals
the average of the two $x$-faces bounding the top ghost cell, so the
simplest consistent face fill is

$$B_x^{\mathrm{face},\,i\pm\tfrac12,\,j_\mathrm{top\;g}}
  = B_x^\mathrm{cc,ghost\;top},$$

giving $\tfrac12(\text{face}_\text{L} + \text{face}_\text{R}) = B_x^\mathrm{cc}$
by construction. The normal face $B_y^{\mathrm{face},\,j_\mathrm{top\;g}+\tfrac12}$
mirrors symmetrically from the interior $B_y^\mathrm{face}$ immediately
below the wall (Yee-consistent; same argument as the outflow BC), which
preserves $\nabla\!\cdot\!\mathbf B = 0$ to machine precision in the
ghost row.

## Implementation

* **Flag.** `AthenaMHDSolver::top_outgoing` (default `false` for
  backwards compatibility with all existing B-M1 – B-M5.75 tests).
  When `true`, `fill_ghost()` dispatches
  `k_athmhd_ghost_y_top_outgoing_cc` + `k_athmhd_ghost_y_top_outgoing_face`
  for the top row instead of the reflective mirror. The bottom is
  unchanged — it continues to use §E2 if `driver_on && driver_Nmodes > 0`,
  otherwise reflective.
* **When to enable.** Any long-time column that needs a steady state
  under a continuous Alfvén driver (B-M5 T7 Leroy-Cranmer benchmark,
  future B-M6 main-trunk wind). Do NOT enable when running a closed
  box Alfvén eigenmode convergence test — the reflect-wall is the
  physical setup there.

## Numerical implementation notes

1. **The top fill is a PURE ABSORBER** (no incoming driver). Unlike §E2
   there is no $v_\mathrm{drv}^{(\mathrm{top})}$ term; the BC is
   homogeneous.
2. **No coupling to pressure / entropy at the BC.** The isothermal HSE
   background fixes $\rho(y)$ and $p(y)$; the top ghost simply inherits
   the HSE mirror for those fields. Only the Alfvén fields
   $(v_x, B_x, v_z, B_z)$ use the characteristic formula.
3. **Composite with §E2 is well-posed.** Every identity verifies
   symbolically with no residual degrees of freedom. The steady state
   exists, is unique, and matches Leroy80 / Cranmer07 eq. 16.
4. **Does not replace §E2.** §E3 is the top-BC companion. The inner
   driver is still §E2; nothing in §E2 changes.
5. **Default-off flag** prevents any of B-M1 – B-M5.75 from regressing:
   the new kernel only runs when `top_outgoing = true` is set
   explicitly (currently only by B-M5 T7).

## Verification checkpoints

- `scripts/e3_top_outgoing_bc.py` — sympy: 9 identities verified
  (advection, ghost closure, $R_\text{top} = 0$, transmission
  invariance, quiescence, z-channel, face-B, composite well-posedness,
  WKB growth law).
- `tests/test_athena_mhd_driver.cu::E1-T7` — Leroy 1980 / Cranmer 2007
  WKB amplitude growth $v_\perp \propto \rho^{-1/4}$ on a stratified
  atm with §E2 bottom + §E3 top; expect agreement within 10% of the
  analytic prediction (PLM+HLLD per-wavelength damping $\sim$ 2% floor).
- Regression: every other Phase-B test keeps `top_outgoing = false`
  and must remain bit-identical to pre-§E3 (checked by full suite
  rerun after §E3 lands).
