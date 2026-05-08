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

## Numerical implementation notes (not in formal derivation)

The three items below are discretisation pitfalls surfaced during
Phase B-M1 (commit `fdbe383`, `test_athena_mhd_hse_preserve.cu` 6/6
pass).  At the derivation level $F_\mathrm{wb}(\mathbf{U}_\mathrm{hse})
\equiv 0$ is an analytic identity, but in the VL2 + PLM + reflective
wall implementation stack any one of these three, if done wrong,
degrades "machine precision" into drift at the $10^{-2}$–$10^{-3}$
level.

1. **The two VL2 stages need separately stored defects $R(\mathbf{U}_\mathrm{hse})$.**
   The predictor uses donor-cell reconstruction (order=1) while the
   corrector uses PLM (order=xorder).  The discrete residual
   $R(\mathbf{U}_\mathrm{hse})$ is not the same under the two
   reconstructions in finite precision (different face values yield
   different fluxes).  Storing one defect and subtracting it in both
   stages leaves a residual $\sim 0.8\%$ drift; storing two
   independent defects (`d_rhs_hse_s1_*`, `d_rhs_hse_s2_*`, routed by
   stage in `apply_flux_divergence_and_ct`) reaches ULP.

2. **Subtract the defect from all six conservatives
   $(\rho, m_x, m_y, m_z, E, B_z)$, not just the ones with gravity
   source.**
   Naive intuition says subtract only
   $(m_x, m_y, m_z, E)$.  Empirically, the reflective-wall flux
   residuals on $\rho$ and $B_z$ are ULP-level ($\sim 10^{-16}$) per
   step, but accumulate over 1000 steps into $\delta\rho/\rho \sim
   1\%$, failing the B3 assertion.  Well-balancing is an **algebraic
   identity cancellation**, not "zero out the dominant terms"; all
   six fields must participate.

3. **The snapshot uses one $\mathrm{prim}(\mathbf{U}_\mathrm{hse})$
   for both stages — do not simulate the stage-2 swap.**
   In the real `step()` the stage-2 flux is computed from
   $\mathrm{prim}(\mathbf{U}^*)$ because stage 1 performs a swap +
   refill at the end.  When WB is exact, however,
   $\mathbf{U}^* \equiv \mathbf{U}_\mathrm{hse}$, so stage 2 genuinely
   sees $\mathrm{prim}(\mathbf{U}_\mathrm{hse})$ itself.  The snapshot
   only needs `cons_to_prim(U_hse)` once, shared by both stages; if
   one manually adds a swap to mimic stage 2's $\mathbf{U}^*$, the
   captured "defect" is actually the residual of a non-WB
   $\mathbf{U}^*$, breaking self-consistency.

Common takeaway: well-balancing only cancels if the two discrete
expressions are **bit-wise identical**, not merely "physically
equivalent".  Any PR that alters reconstruction order, variable
ordering, or inter-stage state semantics must re-run B-M1 to verify
ULP cancellation.
