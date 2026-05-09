# E2. Characteristic inner BC for 2D MHD

> **sympy script:** `scripts/e2_characteristic_bc.py`
> **verified:** Alfvén transport matrix eigenvalues $\pm v_A$;
> Riemann invariants $\tilde z^\pm = \mp v_x + B_x/\sqrt{\rho_0}$
> satisfy $\partial_t \tilde z^\pm \pm v_A \partial_y \tilde z^\pm = 0$;
> $+v_A$ mode polarisation $\delta B_x = -\sqrt{\rho_0}\,\delta v_x$;
> ghost-fill closure for prescribed $v_x^\mathrm{drv}(t)$ + absorbing
> $\tilde z^-$; reflection coefficient $R = 0$ for pure incident
> Alfvén pulse (linear order); face-B ghost fill consistent with
> cell-centred $B_x^\mathrm{cc,ghost}$.
> **code checkpoints:**
> `athena_mhd_kernels.cu::k_athmhd_ghost_y_characteristic`,
> `athena_mhd_solver.cu::apply_driver` (deprecated interior-SET path removed),
> `tests/test_athena_mhd_all_ops.cu::F-T3f` (ULP mass conservation),
> `tests/test_athena_mhd_driver.cu::E1-T6` (reflection coefficient $R < 10^{-3}$).

## Motivation

B-M5.75 (`test_athena_mhd_all_ops::F-T3`) showed that the interior-SET
driver — overwriting $v_x$ on the $j = n_g$ interior row each step — is
non-conservative in total mass at the $O(10^{-6})$ level over 300 steps.
The source of the drift is the step discontinuity of $v_x$ across the
$j = n_g - \tfrac12$ face: the reflective ghost still carries
$v_x = 0$ while the interior row is forcibly driven, producing an
unbalanced flux at the boundary Riemann problem.

This is not a physical limitation of prescribed-velocity inner boundaries.
Suzuki+25 and Shoda+2018a do not lose mass at the photospheric driver;
what they run is a **characteristic inner BC** in which only the
**incoming** Alfvén Riemann invariant $\tilde z^+$ is prescribed, while
the **outgoing** $\tilde z^-$ is extrapolated from the interior. This
section derives that BC symbolically and records the exact ghost-fill
formulas the kernel must use.

## Setup

Linearise 2D MHD around the background
$(\rho, \mathbf v, p, \mathbf B) = (\rho_0, 0, p_0, B_{y0}\hat y)$,
with the bottom boundary at $y = 0$. Perturbation fields
$(v_x, B_x, v_z, B_z, \rho', v_y, p')$ decouple into four one-dimensional
$y$-transport subsystems:

$$\begin{aligned}
\text{Alfvén-}x:\quad & \partial_t v_x = \frac{B_{y0}}{\rho_0}\,\partial_y B_x,\qquad
                       \partial_t B_x = B_{y0}\,\partial_y v_x, \\
\text{Alfvén-}z:\quad & (\text{identical, } v_z \leftrightarrow v_x, B_z \leftrightarrow B_x), \\
\text{Acoustic+entropy:}\ & (\rho', v_y, p')\ \text{system with speeds } \pm c_s \text{ and } 0.
\end{aligned}$$

Only the Alfvén-$x$ channel is actively driven in v1; the $z$ channel
and the acoustic / entropy channel are treated below.

## Alfvén Riemann invariants and polarisation

Sympy-verified: the transport matrix in $\partial_t U + A'\,\partial_y U = 0$
for $U = (v_x, B_x)^\mathrm{T}$ has eigenvalues $\pm v_A$ where
$v_A = B_{y0}/\sqrt{\rho_0}$.

$$\boxed{\tilde z^\pm = \mp v_x + \frac{B_x}{\sqrt{\rho_0}}, \qquad
\partial_t \tilde z^\pm \pm v_A\,\partial_y \tilde z^\pm = 0.} \quad (\text{E2-invariants})$$

Sign convention: $\tilde z^+$ propagates at $+v_A$, i.e. **from the bottom
boundary into the domain** (incoming). $\tilde z^-$ propagates at $-v_A$,
i.e. **from the domain toward the bottom** (outgoing).

The right eigenvector of the $+v_A$ mode is $(1, -\sqrt{\rho_0})^\mathrm{T}$,
giving the polarisation

$$\boxed{\delta B_x = -\sqrt{\rho_0}\,\delta v_x \quad\text{on the incoming Alfvén mode.}} \quad (\text{E2-polarisation})$$

The B-M5 T5 Alfvén emission test measured this ratio at 1.019 against
the theoretical 1.0 — an independent confirmation of the same sign
convention used here.

## Characteristic ghost-fill closure

The driver prescribes the horizontal velocity $v_x^\mathrm{drv}(t)$.
For an incoming Alfvén wave of that amplitude, the polarisation relation
forces $\delta B_x = -\sqrt{\rho_0}\,v_x^\mathrm{drv}$, so the incoming
Riemann invariant is

$$\tilde z^+ \bigr|_\mathrm{ghost}
  = -v_x^\mathrm{drv} + (-\sqrt{\rho_0}\,v_x^\mathrm{drv})/\sqrt{\rho_0}
  = -2\,v_x^\mathrm{drv}.$$

For the outgoing invariant, "absorbing" means $\partial_y \tilde z^- = 0$,
which on a discrete ghost row becomes

$$\tilde z^- \bigr|_\mathrm{ghost} = \tilde z^- \bigr|_\mathrm{int}
  \equiv v_x^\mathrm{int} + B_x^\mathrm{int}/\sqrt{\rho_0}.$$

Inverting the Elsässer system to recover primitives,

$$\boxed{\begin{aligned}
v_x \bigr|_\mathrm{ghost} &= v_x^\mathrm{drv}
  + \tfrac{1}{2}\bigl[v_x^\mathrm{int} + B_x^\mathrm{int}/\sqrt{\rho_0}\bigr], \\
\frac{B_x \bigr|_\mathrm{ghost}}{\sqrt{\rho_0}} &= -v_x^\mathrm{drv}
  + \tfrac{1}{2}\bigl[v_x^\mathrm{int} + B_x^\mathrm{int}/\sqrt{\rho_0}\bigr].
\end{aligned}} \quad (\text{E2-ghost-fill})$$

Limits of this formula:

- **Quiescent interior** ($v_x^\mathrm{int} = B_x^\mathrm{int} = 0$):
  ghost values collapse to $v_x^\mathrm{ghost} = v_x^\mathrm{drv}$,
  $B_x^\mathrm{ghost} = -\sqrt{\rho_0}\,v_x^\mathrm{drv}$ — a pure
  $+v_A$ Alfvén injection. Sympy-verified.
- **Driver off** ($v_x^\mathrm{drv} = 0$): the ghost values encode only
  the outgoing $\tilde z^-$; substituting back yields
  $\tilde z^+ \bigr|_\mathrm{ghost} = 0$ exactly. Sympy-verified.

## Reflection coefficient

Consider a pure incident $\tilde z^-$ pulse from the interior with
amplitude $Z_0(t)$ and $\tilde z^+|_\mathrm{int} = 0$. Corresponding
interior primitives are $v_x^\mathrm{int} = Z_0/2$,
$B_x^\mathrm{int} = \sqrt{\rho_0}\,Z_0/2$. With $v_x^\mathrm{drv} = 0$,
the characteristic ghost formulas give $v_x^\mathrm{ghost} = Z_0/2$,
$B_x^\mathrm{ghost}/\sqrt{\rho_0} = Z_0/2$, so

$$\tilde z^+ \bigr|_\mathrm{ghost}
  = -(Z_0/2) + (Z_0/2) = 0 \quad\Rightarrow\quad R \equiv 0.$$

Sympy-verified. This is the defining advantage over the interior-SET
or the naive reflective BC: outgoing Alfvén waves leave the domain
without generating a spurious incoming partner.

## Face-B consistency (Yee grid)

The CT-preserving face-B update requires that the ghost cell-centred
$B_x^\mathrm{cc,ghost}$ be the arithmetic mean of the two x-faces
bounding the ghost cell:

$$B_x^\mathrm{cc,ghost} = \tfrac{1}{2}\bigl(B_x^{\mathrm{face},i-\tfrac12,j_g}
                                          + B_x^{\mathrm{face},i+\tfrac12,j_g}\bigr).$$

The simplest face-fill that reproduces the characteristic $B_x^\mathrm{cc,ghost}$
is to set **both x-faces in the ghost row** to the same value
$B_x^\mathrm{cc,ghost}$. This is consistent because
(a) the mean of two equal numbers is the number itself (sympy-verified trivially);
(b) $B_y^\mathrm{face}$ in the ghost row is handled by the mirror rule
of the reflective-y BC, so $\nabla\!\cdot\!\mathbf B = 0$ is preserved
locally to ULP.

## z-polarised Alfvén channel

The $(v_z, B_z)$ system has the same structure as $(v_x, B_x)$ but no
driver. Characteristic ghost fill with $v_z^\mathrm{drv} = 0$:

$$v_z \bigr|_\mathrm{ghost} = \tfrac{1}{2}\bigl[v_z^\mathrm{int} + B_z^\mathrm{int}/\sqrt{\rho_0}\bigr],\qquad
\frac{B_z \bigr|_\mathrm{ghost}}{\sqrt{\rho_0}} = \tfrac{1}{2}\bigl[v_z^\mathrm{int} + B_z^\mathrm{int}/\sqrt{\rho_0}\bigr].$$

This is a **pure absorber** for the $z$-polarised Alfvén mode. The
kernel uses the same formula as (E2-ghost-fill) with the driver term
set to zero.

## Acoustic + entropy channel

The $(\rho', v_y, p')$ system is not involved in the Alfvén driver. In
v1 the BC is the usual **reflective-y for HSE**: $\rho$ and $p$ are
pinned to the HSE column values at the ghost $y$, and $v_y$ is mirrored
antisymmetrically. Formally:

$$\rho \bigr|_\mathrm{ghost} = \rho_\mathrm{HSE}(y_\mathrm{ghost}),\quad
  v_y \bigr|_\mathrm{ghost} = -v_y \bigr|_\mathrm{int},\quad
  p \bigr|_\mathrm{ghost} = p_\mathrm{HSE}(y_\mathrm{ghost}).$$

This is the same treatment the B-M1 HSE test uses and does not need
new symbolic verification.

## Interaction with B-M5 driver implementation

The previous interior-SET implementation is removed. `apply_driver(t)`
now computes $v_x^\mathrm{drv}(t)$ via the stochastic broadband waveform
(§E1 definition unchanged) and **writes only the ghost row**, leaving
the interior row to be updated by VL2 flux divergence like any other
cell. The kernel change is local; the public API is unchanged.

Regression implications:

- E1-T1 through E1-T4 (engineering tests) continue to pass — the
  waveform is the same, only the target row moves from $j=n_g$ to
  $j=n_g - 1$.
- E1-T5 (Alfvén emission) still passes because the interior sees an
  injected $+v_A$ mode with the polarisation derived above; the arrival
  time and polarisation ratio are unchanged.
- B-M5.75 F-T3f (mass conservation) tightens to ULP.
- New **E1-T6** (linear-order reflection coefficient): inject a pure
  downgoing Alfvén pulse from mid-domain, measure
  $\tilde z^+|_\mathrm{ghost}$ at arrival, assert amplitude ratio
  $R < 10^{-3}$ (WKB floor plus grid-dispersion error).

## ✅ Verification checkpoints

- `tests/test_athena_mhd_driver.cu::E1-T6` — downgoing Alfvén pulse,
  reflection coefficient $R < 10^{-3}$.
- `tests/test_athena_mhd_all_ops.cu::F-T3f` — tightened mass-drift
  threshold from $10^{-4}$ back to $10^{-10}$ (ULP).
- `tests/test_athena_mhd_hse_preserve.cu` — unchanged, regression sentinel.
- `tests/test_athena_mhd_driver.cu::E1-T1..T5` — unchanged, regression sentinel.
