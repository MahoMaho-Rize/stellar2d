# E4. Strang-split gravity source commutator

> **sympy script:** `scripts/e04_strang_split_source_commutator.py`
> **generated LaTeX:** `output/e04_strang_split_source_commutator.latex.tex`
> **verifies:** 4 strong-form identities — 4 non-commutative
> polynomial residual coefficients at $\Delta t^2$ showing the
> **wrong** (separate-$\mathcal{Z}$) operator-chain fails at
> $\Delta t^2$ with the commutator structure $\pm\tfrac{\Delta t^2}{2}[\mathcal{Z}, \mathcal{X}]$ and $\pm\tfrac{\Delta t^2}{2}[\mathcal{Z}, \mathcal{Y}_{\mathrm{hyd}}]$
> **code checkpoints:**
> `src/gpu/explicit/strang_solver.cu :: k_hllc_update_y`
> (gravity source `S_my, S_E` applied INSIDE the y-sweep kernel,
> line 513-523). The operator chain is X(Dt/2) * Y_total(Dt) *
> X(Dt/2), not a separate Z-operator.

The Strang kernel absorbs the gravity source **into** the
y-direction operator: the update `k_hllc_update_y` computes the
full HLLC flux divergence **and** adds the gravity source in the
same kernel call, treating them as a single $\mathcal{Y}_{\mathrm{total}} =
\mathcal{Y}_{\mathrm{hyd}} + \mathcal{Y}_{\mathrm{grav}}$ operator.
This is structurally critical for the 2nd-order accuracy claim
(§A13).

If one instead split gravity as a **separate** $\mathcal{Z}$
operator applied **after** the Strang chain, the composite would
be:

$$L_{\mathrm{wrong}}(\Delta t) \;=\; e^{\Delta t\,\mathcal{Z}}\,\underbrace{e^{(\Delta t/2)\mathcal{X}}\,e^{\Delta t\,\mathcal{Y}_{\mathrm{hyd}}}\,e^{(\Delta t/2)\mathcal{X}}}_{\text{Strang chain}}, \quad (\text{E4-wrong})$$

which introduces a Lie-type step for $\mathcal{Z}$ after the
Strang chain. Per §A14, Lie splitting is 1st-order — the
composite becomes 1st-order for the $\mathcal{Z}$ piece even
though the inner Strang chain is 2nd-order.

## Correct (kernel's) choice

$$L_{\mathrm{correct}}(\Delta t) \;=\; e^{(\Delta t/2)\mathcal{X}}\,e^{\Delta t\,(\mathcal{Y}_{\mathrm{hyd}} + \mathcal{Y}_{\mathrm{grav}})}\,e^{(\Delta t/2)\mathcal{X}}. \quad (\text{E4-correct})$$

Treating $\mathcal{Y}_{\mathrm{total}} = \mathcal{Y}_{\mathrm{hyd}} +
\mathcal{Y}_{\mathrm{grav}}$ as a single operator, §A13's Strang
proof applies directly: the residual against $e^{\Delta t(\mathcal{X}
+ \mathcal{Y}_{\mathrm{tot}})}$ is $O(\Delta t^3)$, 2nd-order
globally.

## Wrong-chain leading error (sympy verified)

For the wrong composite $L_{\mathrm{wrong}}$ vs. the exact
$e^{\Delta t(\mathcal{X} + \mathcal{Y}_{\mathrm{hyd}} + \mathcal{Z})}$, sympy computes the
$\Delta t^2$ residual monomial coefficients:

| monomial | coefficient |
|---|---|
| $\mathcal{Z}\mathcal{X}$ | $+\Delta t^2 / 2$ |
| $\mathcal{X}\mathcal{Z}$ | $-\Delta t^2 / 2$ |
| $\mathcal{Z}\mathcal{Y}_{\mathrm{hyd}}$ | $+\Delta t^2 / 2$ |
| $\mathcal{Y}_{\mathrm{hyd}}\mathcal{Z}$ | $-\Delta t^2 / 2$ |
| $\mathcal{X}\mathcal{Y}_{\mathrm{hyd}}, \mathcal{Y}_{\mathrm{hyd}}\mathcal{X}$ | 0 (Strang chain already cancels) |

Summed, the residual is

$$L_{\mathrm{wrong}}(\Delta t) \;-\; e^{\Delta t\,(\mathcal{X} + \mathcal{Y}_{\mathrm{hyd}} + \mathcal{Z})} \;=\; \frac{\Delta t^{2}}{2}\,\bigl[\mathcal{Z},\;\mathcal{X} + \mathcal{Y}_{\mathrm{hyd}}\bigr] \;+\; O(\Delta t^3). \quad (\text{E4-residual})$$

The non-zero $\Delta t^2$ term indicates 1st-order global error
(via time-reversal analysis of §A14): the $\Delta t^1$ error in
the forward-only update comes from the $\Delta t^2$ residual in
the composition $L_{\mathrm{wrong}} L_{\mathrm{wrong}}^{-1}$.

## Why gravity can go inside $\mathcal{Y}$

The gravity source contributes only to the y-momentum and energy
components ($\mathcal{S} = (0, 0, -\rho g, -m_y g)$, §C1). These
are both components whose physical y-sweep update is non-trivial
(the y-momentum flux contains $\rho v^2 + P$, and the energy flux
contains the pressure term $\bar p$). Gravity balances these flux
components pointwise by §B3 + §C1, so absorbing it into the
y-operator is natural — the operator is "what happens to the
cell during the y-direction update".

If the physics had gravity that coupled to x-momentum (e.g., a
rotating frame or a non-aligned gravity vector), it would need a
separate treatment or a more complex splitting.

## Kernel implementation verification

The gravity application in `k_hllc_update_y` (strang_solver.cu
line 513-523) is called **inside** the y-sweep kernel, after the
HLLC flux computation and before the state write-back:

```cpp
// Compute HLLC fluxes GT, GB at top and bottom faces.
// ...
// Apply gravity source INSIDE y-sweep (not outside):
double rho_total = fmax(d_rho[k0] + rho_bg, 1e-20);
double S_my = -rho_total * g_grav;
double S_E  = -d_my[k0] * g_grav;
// Combined update: flux divergence + gravity source.
d_my[k0] = d_my[k0] - dtdy * (GT_mn - GB_mn) + dt * S_my;
d_E [k0] = d_E [k0] - dtdy * (GT3   - GB3)   + dt * S_E;
```

This is the correct `L_correct` structure. No separate gravity
step is called before or after the Strang chain.

## ✅ Verification checkpoint (to be wired)

1. **Absence of separate Z operator.** Code inspection:
   `grep -n "g_grav" strang_solver.cu` — all occurrences must be
   inside `k_hllc_update_y` or in init code (for HSE background);
   none in a standalone `apply_gravity` function. Test:
   `test_strang_unit.cu` §E4-no-Z-operator.

2. **Convergence rate on non-trivial gravity problem.** At §D5
   bubble IC (which has gravity, and hence probes the kernel's
   gravity-inclusive Strang order), slope $p \in [1.8, 2.2]$.
   Test: `test_strang_convergence.cu` §E4-bubble-order.

3. **Symmetry under source-flip.** Running a canonical test with
   gravity sign reversed (`g_grav = -g` instead of `+g`) should
   give results related by a symmetry (the atmosphere "upside
   down"). This is primarily a consistency check for gravity's
   pure-source nature. Test: optional scheme-char diagnostic.

Failure of (1) means someone has added a separate gravity
operator call somewhere, degrading the scheme to 1st-order.
Failure of (2) with slope $\ll 1.8$ on a gravity-inclusive test
might indicate (1), or a deeper §C1 gravity-source bug.
