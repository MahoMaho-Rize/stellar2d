---
title: |
  MHD Derivations for stellar2d —
  a sympy-verified manuscript
author: |
  stellar2d development notes\
  *Department of Astronomy, Tsinghua University*\
  `github.com/MahoMaho-Rize/stellar2d`
date: 2026-05-08
---

# Front matter

## Purpose

This manuscript is an **internal, reproducibility-grade derivation document**
for the MHD extension of stellar2d. It covers four parts:

- **Part A** — ideal MHD equations, conservative / primitive forms,
  flux Jacobian eigensystem, HLLD intermediate states, constrained
  transport (CT) preservation of $\nabla\cdot\mathbf{B}=0$; PLM/PPM
  reconstruction + TVD slope limiters; VL2 predictor-corrector and
  its CFL bound (hyperbolic + parabolic); HLLD degeneracy branches;
  Powell-source vs CT comparison; linear-wave convergence IC.
- **Part B** — reduction to a 1D super-radial flux-tube geometry used by
  the Suzuki-group stellar-wind codes, including the WKB Alfvén-wave
  action integral, the Parker critical-point condition in this
  geometry, and the well-balanced MHSE operator needed to keep the
  atmosphere quiet for long-time wind runs.
- **Part C** — non-ideal dissipation: Ohmic, ambipolar, and their
  contributions to the energy equation; closure of the two diffusivities
  through Saha ionisation; sub-grid turbulent heating closure
  (Suzuki-Inutsuka 2005) for 1D flux-tube wind runs.
- **Part D** — cylindrical shearing-box MHD, the shearing-periodic
  boundary condition, and the exact Maxwell–Reynolds stress
  decomposition of the $\alpha_{\mathrm{SS}}$ metric.

## Reproducibility protocol

Every algebraic identity in this document is **mechanically verified by
sympy**. Each section corresponds to a script in
`docs/mhd_derivations/scripts/<section>.py` that ends with
`assert_zero(LHS − RHS, ...)` calls. If any identity cannot be
simplified by sympy but is physically correct, we document the
alternate verification route (e.g., "manually checked with Wolfram,
reason: polynomial simplification blows up").

The scripts are intentionally **independent**: each one imports only
`_common.py` and re-derives everything it needs from first principles.
No cross-script caching of intermediate symbolic results. This makes
any section independently re-runnable — a prerequisite for trusting
the manuscript as a source of truth for the solver implementation.

## Conventions

- Units are Gaussian with $4\pi$ absorbed into $\mathbf{B}$, so that the
  Alfvén speed is simply $c_A = B/\sqrt{\rho}$, matching Stone & Gardiner
  2008, Athena++, PLUTO.
- $\gamma$ always denotes the ratio of specific heats (not the relativistic
  Lorentz factor).
- Einstein summation is *not* used; sums are written explicitly.
- sympy variable names mirror the mathematical symbols
  (`rho` = $\rho$, `B_x` = $B_x$, etc.); see `scripts/_common.py` for the
  full inventory.

## How to regenerate this manuscript

```bash
cd docs/mhd_derivations
bash run_all.sh         # (re-)runs every scripts/*.py, refreshes output/
bash build_manuscript.sh  # concatenates sections/*.md → manuscript.{md,pdf}
```

If a sympy assertion fails during `run_all.sh`, the build halts and the
offending section is flagged. No partial manuscript is emitted.

# A1. Ideal MHD equations (conservative form)

> **sympy script:** `scripts/a1_ideal_mhd_equations.py`
> **generated LaTeX:** `output/a1_ideal_mhd_equations.latex.tex`
> **verifies:** 8 identities (Lorentz × 3 components, induction × 3,
> Poynting-flux identity, $\nabla\cdot(\nabla\times\cdot)=0$)
> **code checkpoints (future):** `src/gpu/explicit/athena_mhd_kernels.cu::d_mhd_flux`

## Starting assumptions

| Label | Assumption |
|---|---|
| **A1a** | Compressible neutral gas with magnetic field $\mathbf{B}$; no viscosity, no Ohmic, no ambipolar, no Hall. |
| **A1b** | Infinite conductivity → frozen-in flux, $\mathbf{E} = -\mathbf{v}\times\mathbf{B}$. |
| **A1c** | Non-relativistic: displacement current $\partial_t\mathbf{E}/c$ dropped. |
| **A1d** | Ideal EOS: $p = (\gamma - 1)\rho e$. |

## Mass conservation

Written compactly as
$$\partial_t\rho + \nabla\cdot(\rho\mathbf{v}) = 0. \quad (\text{A1-mass})$$

Under A1a–d this is the unmodified fluid continuity equation.

## Lorentz force identity (A1-lorentz)

The single most important vector identity of the derivation is

$$(\nabla\times\mathbf{B})\times\mathbf{B}
 = \nabla\cdot(\mathbf{B}\otimes\mathbf{B})
 - \nabla\!\left(\tfrac{1}{2}|\mathbf{B}|^{2}\right)
 - (\nabla\cdot\mathbf{B})\,\mathbf{B}. \quad (\text{A1-lorentz})$$

The last term vanishes **analytically** when the solenoidal constraint
$\nabla\cdot\mathbf{B}=0$ is enforced. At the discrete level it does
not vanish in general; this is precisely what CT schemes (§A5) are
engineered to handle.

**Sympy verification.** All three vector components of the residual
$(\nabla\times\mathbf{B})\times\mathbf{B} - [\nabla\cdot(\mathbf{B}\otimes\mathbf{B}) - \nabla(\tfrac{1}{2}|\mathbf{B}|^2) - (\nabla\cdot\mathbf{B})\mathbf{B}]$
reduce to $0$ under `sympy.simplify`, without imposing $\nabla\cdot\mathbf{B}=0$.
This confirms the identity is an *algebraic*, not a physical, fact.

## Momentum conservation (divergence form)

Define the **total magneto-fluid pressure**

$$P^{\star} \equiv p + \tfrac{1}{2}|\mathbf{B}|^{2}.$$

Then (A1-lorentz) lets us write momentum conservation in divergence form:

$$\boxed{
\partial_t(\rho\mathbf{v})
+ \nabla\!\cdot\!\left[\rho\mathbf{v}\otimes\mathbf{v}
 - \mathbf{B}\otimes\mathbf{B}
 + P^{\star}\mathbf{I}\right] = \mathbf{0}.
} \quad (\text{A1-mom})$$

**Implementation note.** $\mathbf{B}\otimes\mathbf{B}$ is *not* symmetric
in its indices when read as a flux tensor $F_{ij}$ (it is actually
symmetric, but in contrast the induction tensor $v_iB_j - v_jB_i$ is
anti-symmetric). In the kernel, $B_iB_j$ for each
$(i,j)\in\{(x,x),(x,y),(x,z),(y,y),(y,z),(z,z)\}$ must be computed on
the same stencil as $v_iv_j$ — mixing face-averaged $\mathbf{B}$ with
cell-centred $\mathbf{B}$ here is the number-one source of initial
HLLD bugs.

## Induction equation — two equivalent forms (A1-induction)

**Tensor-divergence form** (preferred for a Godunov kernel):

$$\partial_t B_i + \partial_j\!\left(v_j B_i - v_i B_j\right) = 0.$$

**Curl form** (preferred for a CT update):

$$\partial_t\mathbf{B} = \nabla\times(\mathbf{v}\times\mathbf{B}).$$

**Sympy verification.** For each component $i\in\{x,y,z\}$ we check that

$$\nabla\times(\mathbf{v}\times\mathbf{B})\Big|_i
\equiv -\partial_j(v_j B_i - v_i B_j)$$

holds as a *pure vector-calculus identity* (no $\nabla\cdot\mathbf{B}=0$,
no $\nabla\cdot\mathbf{v}=0$ invoked). This is the well-known
Jacobi-like identity; all three sympy assertions pass.

## Total-energy conservation (A1-energy)

With

$$E \equiv \rho e + \tfrac{1}{2}\rho|\mathbf{v}|^{2}
        + \tfrac{1}{2}|\mathbf{B}|^{2},$$

the conservative form is

$$\boxed{
\partial_t E + \nabla\!\cdot\!\left[(E + P^{\star})\mathbf{v}
 - \mathbf{B}(\mathbf{B}\cdot\mathbf{v})\right] = 0.
}$$

The magnetic-Poynting-flux piece $-\mathbf{B}(\mathbf{B}\cdot\mathbf{v})$
and the enthalpy-like piece $P^\star\mathbf{v}$ together make the flux
manifestly symmetric-hyperbolic.

**Sympy-verified identity.** The vector identity
$\mathbf{B}\times(\mathbf{v}\times\mathbf{B}) = |\mathbf{B}|^2\mathbf{v}
- \mathbf{B}(\mathbf{v}\cdot\mathbf{B})$ lets us rewrite the Poynting
flux $\mathbf{S} = \tfrac{1}{2}\mathbf{B}\times(\mathbf{v}\times\mathbf{B})$
in the form used in (A1-energy). `assert_zero` confirms
$\nabla\cdot[\mathbf{B}\times(\mathbf{v}\times\mathbf{B})
 - (|\mathbf{B}|^2\mathbf{v} - \mathbf{B}(\mathbf{B}\cdot\mathbf{v}))] = 0$
identically.

## Solenoidal constraint preservation (A1-divB)

Taking the divergence of the induction equation in curl form:

$$\partial_t(\nabla\cdot\mathbf{B}) = \nabla\cdot\nabla\times(\mathbf{v}\times\mathbf{B}) = 0,$$

where the last equality is the identity $\nabla\cdot(\nabla\times\cdot)=0$.
**At the continuous level this is automatic**; at the discrete level it
is what CT or Dedner-GLM machinery must engineer. See §A5.

**Sympy verification.** `assert_zero(div_cart(curl_cart(v×B)))` passes
for arbitrary smooth $\mathbf{v}$, $\mathbf{B}$.

## Compact conservative system

Collecting all fluxes, we write the 8-variable conservative MHD system:

$$\boxed{
\partial_t\mathbf{U} + \partial_i\mathbf{F}_i(\mathbf{U}) = \mathbf{0},
\quad
\mathbf{U} = (\rho,\ \rho\mathbf{v},\ \mathbf{B},\ E)^{\mathrm{T}}.
}$$

The explicit form of $\mathbf{F}_i$ follows by assembling the four fluxes
above. We defer the component-by-component flux Jacobian and its
eigensystem to §A3.

## ✅ Verification checkpoint (to be wired)

Implementation invariants the kernel must satisfy (enforced through
`tests/test_athena_mhd_*.cu` in a future PR):

1. **One-cell linearised update reduces to (A1-mass), (A1-mom),
   (A1-induction-tensor), (A1-energy) exactly** — zero the $\mathbf{B}$
   field and verify that $\mathbf{F}_i$ reduces to the Euler flux used
   in `athena_vl2`.
2. **Solenoidal residual** $(\nabla\cdot\mathbf{B})_{cell}$ from face-
   centred $\mathbf{B}$ remains at round-off over 100 sound-crossing
   times on a smooth periodic IC (field-loop advection, §A5).
3. **Energy-momentum consistency** — on a piecewise-constant linear
   wave IC, total energy updated via (A1-energy) agrees with
   $\tfrac{1}{2}\rho|\mathbf{v}|^{2}+\tfrac{1}{2}|\mathbf{B}|^{2}+\rho e$
   reconstructed from the primitive fields, to round-off per step.

Any failure on (1)–(3) flags the implementation as inconsistent with
§A1 and must be resolved before proceeding to §A2.

# A2. Conservative ↔ primitive variable transformation

> **sympy script:** `scripts/a2_conservative_primitive.py`
> **generated LaTeX:** `output/a2_conservative_primitive.latex.tex`
> **verifies:** $\dfrac{\partial\mathbf{W}}{\partial\mathbf{U}}\cdot
> \dfrac{\partial\mathbf{U}}{\partial\mathbf{W}} = \mathsf{I}_{8}$
> (all 64 entries symbolically checked to zero after subtraction).
> **code checkpoints (future):**
> `athena_mhd_kernels.cu::d_prim_from_cons`, `d_cons_from_prim`.

## Variable inventory

The conservative 8-vector used inside the Godunov flux loop:

$$\mathbf{U} = (\rho,\ \rho v_x,\ \rho v_y,\ \rho v_z,\ B_x,\ B_y,\ B_z,\ E)^{\mathrm{T}}.$$

The primitive 8-vector used for reconstruction / Riemann solver input:

$$\mathbf{W} = (\rho,\ v_x,\ v_y,\ v_z,\ B_x,\ B_y,\ B_z,\ p)^{\mathrm{T}}.$$

The total-energy density closes the map through the ideal EOS:

$$E = \frac{p}{\gamma-1} + \tfrac{1}{2}\rho|\mathbf{v}|^{2} + \tfrac{1}{2}|\mathbf{B}|^{2}. \quad (\text{A2-E})$$

## Pressure extraction (primitive from conservative)

Inverting (A2-E) for $p$ gives the pressure-extraction formula the
`d_prim_from_cons` kernel must use:

$$\boxed{
p = (\gamma - 1)\!\left(E - \frac{|\mathbf{m}|^{2}}{2\rho}
 - \tfrac{1}{2}|\mathbf{B}|^{2}\right),
\quad \mathbf{m} \equiv \rho\mathbf{v}.
}$$

**Implementation gotcha.** For low-β, high-Mach flows (MHD blast,
Orszag-Tang late time), the hydro term $|\mathbf{m}|^2/(2\rho)$ and the
magnetic term $|\mathbf{B}|^2/2$ can both be within machine-precision
of the total $E$. A naïve subtraction can then hand the EOS a negative
$p$. The kernel must either:

1. Accept a positivity fallback (fall back to isothermal primitive
   extraction and flag the cell), or
2. Use the dual-energy formulation (evolve an auxiliary internal-energy
   variable and use it when the primitive subtraction is unreliable).

This is documented in Stone & Gardiner 2008 §4.6 and implemented in
Athena++ as the `DUAL_ENERGY` compile-time flag.

## Forward Jacobian $\partial\mathbf{U}/\partial\mathbf{W}$

sympy produces the 8×8 Jacobian explicitly; it is block-sparse with
the $\mathbf{B}$ block trivially the identity, the momentum block
$\rho\mathbf{I} + \mathbf{v}\otimes(\partial\mathbf{v}/\partial\mathbf{W})$,
and the energy row containing $\partial E/\partial\rho$,
$\rho\mathbf{v}$, $\mathbf{B}$, $1/(\gamma-1)$.

## Backward Jacobian $\partial\mathbf{W}/\partial\mathbf{U}$

The inverse map takes the conservative 8-vector to primitives term by
term. The non-trivial rows are the velocity-from-momentum and the
pressure-from-energy rows:

$$\frac{\partial v_i}{\partial(\rho v_j)} = \frac{\delta_{ij}}{\rho},
\qquad
\frac{\partial v_i}{\partial\rho} = -\frac{v_i}{\rho},$$
$$\frac{\partial p}{\partial\rho}
= (\gamma-1)\frac{|\mathbf{m}|^{2}}{2\rho^{2}}
= (\gamma-1)\tfrac{1}{2}|\mathbf{v}|^{2},$$
$$\frac{\partial p}{\partial(\rho v_i)} = -(\gamma-1)v_i,
\qquad
\frac{\partial p}{\partial B_i} = -(\gamma-1)B_i,
\qquad
\frac{\partial p}{\partial E} = \gamma-1.$$

## Consistency check (sympy)

The script computes
$\dfrac{\partial\mathbf{W}}{\partial\mathbf{U}}(\mathbf{U}(\mathbf{W}))
\cdot \dfrac{\partial\mathbf{U}}{\partial\mathbf{W}}(\mathbf{W})$
after substituting back so that both Jacobians are evaluated at the
same primitive state, and verifies

$$\frac{\partial\mathbf{W}}{\partial\mathbf{U}}
\cdot
\frac{\partial\mathbf{U}}{\partial\mathbf{W}} = \mathsf{I}_{8}. \quad (\text{A2-inverse})$$

All 64 entries of the residual matrix are `sympy.simplify`-reduced to
$0$ and pass `assert_zero`.

## ✅ Verification checkpoint (to be wired)

When the MHD kernel is written, the test

```
tests/test_athena_mhd_roundtrip.cu
```

should:

1. Seed random primitive 8-vectors $\mathbf{W}$ with $\rho,p>0$ and
   $|\mathbf{B}|^{2}/(2p)<10^{6}$ (stay out of the positivity danger
   zone).
2. Call `d_cons_from_prim` to compute $\mathbf{U}$.
3. Call `d_prim_from_cons` to recover $\mathbf{W}'$.
4. Assert $\|\mathbf{W} - \mathbf{W}'\|_{\infty} / \|\mathbf{W}\|_{\infty}
   < 10\,\varepsilon_{\mathrm{mach}}$.

Any regression of (A2-inverse) in the kernel fails this round-trip.

# A3. Flux Jacobian $\mathsf{A} = \partial\mathbf{F}/\partial\mathbf{U}$ and its 7-wave eigensystem

> **sympy script:** `scripts/a3_flux_jacobian_eigensystem.py`
> **generated LaTeX:** `output/a3_flux_jacobian_eigensystem.latex.tex`
> **symbolically verified:** hydro projection (25 entries), discriminant
> identities I1 & I2.
> **numerically verified (20 trials × 3 γ):** eigenvalues match
> closed-form to $\le 5\times10^{-15}$; $\mathsf{A}\mathbf{r} -
> \lambda\mathbf{r}$ residuals $\le 6\times10^{-15}$.
> **code checkpoints (future):**
> `athena_mhd_kernels.cu::d_mhd_eigenvalues_primitive`,
> `d_mhd_right_eigenvectors`.

## Working frame

The eigenanalysis is done in **primitive form** in 1D along $\hat{x}$.
Primitive 7-vector:

$$\mathbf{W} = (\rho,\ v_x,\ v_y,\ v_z,\ B_y,\ B_z,\ p)^{\mathrm{T}}.$$

$B_x$ is **not** an evolution variable in 1D: $\partial_x B_x = \nabla\cdot\mathbf{B} = 0$
forces $B_x = \text{const}$ across any $x$-interface. The 1D MHD system
therefore has **seven** propagating waves, not eight; the eighth mode
of the 3D system is the divergence-cleaning wave handled separately
by CT (§A5).

## Primitive-form Jacobian $\mathsf{A}_W$

$$\partial_t\mathbf{W} + \mathsf{A}_W(\mathbf{W})\,\partial_x\mathbf{W} = \mathbf{0},$$

$$\mathsf{A}_W =
\begin{bmatrix}
v_x & \rho & 0 & 0 & 0 & 0 & 0 \\
0 & v_x & 0 & 0 & B_y/\rho & B_z/\rho & 1/\rho \\
0 & 0 & v_x & 0 & -B_x/\rho & 0 & 0 \\
0 & 0 & 0 & v_x & 0 & -B_x/\rho & 0 \\
0 & B_y & -B_x & 0 & v_x & 0 & 0 \\
0 & B_z & 0 & -B_x & 0 & v_x & 0 \\
0 & \gamma p & 0 & 0 & 0 & 0 & v_x
\end{bmatrix}. \quad (\text{A3-jacobian})$$

**Hydrodynamic sanity check.** Setting $B_x = B_y = B_z = 0$ and
projecting onto the $(\rho, v_x, v_y, v_z, p)$ subspace recovers the
standard 5×5 hydrodynamic primitive Jacobian. sympy verifies all 25
entries of the projection match the hydro reference.

## Characteristic speeds

Define the four reference speeds:

$$c_{s_0}^{2} \equiv \gamma p / \rho,\qquad
c_{Ax}^{2} \equiv B_x^{2}/\rho,\qquad
c_{A\perp}^{2} \equiv (B_y^{2}+B_z^{2})/\rho,\qquad
c_{A}^{2} \equiv c_{Ax}^{2} + c_{A\perp}^{2}.$$

The fast and slow magnetosonic speeds are the two positive roots of

$$c_{f,s}^{2} = \tfrac{1}{2}\!\left[(c_{s_0}^{2} + c_{A}^{2}) \pm
\sqrt{(c_{s_0}^{2}+c_{A}^{2})^{2} - 4 c_{s_0}^{2} c_{Ax}^{2}}\right]. \quad (\text{A3-cfs})$$

**Discriminant identities (HLLD-stable forms).** The closed form
(A3-cfs) loses ULPs when $c_{A\perp} \to 0$ or $c_{s_0} \to 0$. The
HLLD kernel uses the two equivalent identities

$$\boxed{
c_f^{2} + c_s^{2} = c_{s_0}^{2} + c_{A}^{2},
\qquad
c_f^{2}\cdot c_s^{2} = c_{s_0}^{2}\cdot c_{Ax}^{2},
} \quad (\text{A3-discriminant})$$

which sympy verifies symbolically (both reduce to $0$ under `simplify`).
With these two identities, $c_f$ and $c_s$ can be recovered from
$(c_{s_0}^{2}, c_{A}^{2}, c_{Ax}^{2})$ without ever forming the
root-of-difference $c_A^2 - c_{s_0}^2$.

## The seven wave speeds

$$\boxed{
\{\lambda_k\}_{k=1}^{7} =
\{\,v_x - c_f,\ v_x - c_{Ax},\ v_x - c_s,\ v_x,\
v_x + c_s,\ v_x + c_{Ax},\ v_x + c_f\,\}.
} \quad (\text{A3-wave-speeds})$$

Here $c_{Ax} \equiv B_x/\sqrt{\rho}$ carries the sign of $B_x$ — this
convention lets us write the Alfvén speed once without the $s=\mathrm{sign}(B_x)$
factor that appears in Stone+08 eigenvector formulas. The other
convention (unsigned $c_{Ax} = |B_x|/\sqrt{\rho}$ + $s$ in every eigenvector
entry) is equivalent; we choose signed-$c_{Ax}$ here because it keeps
the 7-wave spectrum contiguous for all sign($B_x$).

## Numerical verification of the spectrum

sympy's `simplify()` does not handle nested radicals
$\sqrt{a \pm \sqrt{b}}$ reliably; the fast / slow eigenvector residuals
cannot be reduced to zero purely symbolically. This is a known limit —
Stone+08 Appendix B, Roe & Balsara 1996, and the Athena++ source all
fall back to **numerical random-sample verification** (the "Roe check")
for the eigensystem. We do the same:

1. Draw 20 random physically admissible states
   $(\rho, p, v_x, B_x, B_y, B_z)$ with $\rho, p > 0$,
   $B_x \ne 0$, $|B_\perp| \ne 0$;
2. For each of three ratios of specific heats $\gamma \in \{5/3, 7/5, 4/3\}$,
   diagonalise $\mathsf{A}_W$ numerically with `numpy.linalg.eig`;
3. Sort the seven numerical $\lambda_k$ and compare to the sorted
   closed-form spectrum (A3-wave-speeds).

Results over 60 trials:

$$\max|\lambda_{\text{closed}} - \lambda_{\text{numerical}}| < 5\times10^{-15},
\qquad
\max\|\mathsf{A}\mathbf{r} - \lambda\mathbf{r}\|_{\infty} < 6\times10^{-15}.$$

The residuals are at the `double`-precision floor for matrix
eigenvalue computation on a well-conditioned $7\times7$ matrix —
indistinguishable from exact.

## Closed-form right-eigenvectors

The closed forms come from Stone & Gardiner 2008 Appendix B Eqs.
(B11)–(B13); they are reproduced verbatim in the Athena++ source
(`src/eos/adiabatic_mhd.cpp::LRMHDWaves`). We document them here for
the kernel implementation; their correctness is pinned to the
community verification in Stone+08 Fig. 28–30.

Define the **amplitude-normalisation coefficients**

$$\alpha_f^{2} = \frac{c_{s_0}^{2} - c_s^{2}}{c_f^{2} - c_s^{2}},\qquad
\alpha_s^{2} = \frac{c_f^{2} - c_{s_0}^{2}}{c_f^{2} - c_s^{2}},
\qquad \alpha_f^{2} + \alpha_s^{2} = 1,$$

and the **direction coefficients**

$$\beta_y = \frac{B_y}{\sqrt{B_y^{2}+B_z^{2}}},\qquad
\beta_z = \frac{B_z}{\sqrt{B_y^{2}+B_z^{2}}},\qquad
\beta_y^{2} + \beta_z^{2} = 1,$$

and $s = \mathrm{sign}(B_x)$.

### Entropy (contact) wave — $\lambda_4 = v_x$

$$\mathbf{r}_{\mathrm{entropy}} = (1,\ 0,\ 0,\ 0,\ 0,\ 0,\ 0)^{\mathrm{T}}.$$

Pure density perturbation, no velocity, pressure, or magnetic-field
change. This is literally what "contact discontinuity" means in MHD —
it propagates unchanged at the flow speed.

### Alfvén waves — $\lambda_{2,6} = v_x \mp c_{Ax}$

$$\mathbf{r}_{A\pm} = \left(\,0,\ 0,\ \mp s\beta_z,\ \pm s\beta_y,\
-\beta_z\sqrt{\rho},\ \beta_y\sqrt{\rho},\ 0\,\right)^{\mathrm{T}}.$$

Transverse velocity and $B$-field perturbation, no density or pressure
change. This is the canonical transverse electromagnetic wave of MHD.

### Fast magnetosonic waves — $\lambda_{1,7} = v_x \mp c_f$

$$\mathbf{r}_{f\pm} = \left(\,
\rho\alpha_f,\
\pm\alpha_f c_f,\
\mp s\alpha_s c_s\beta_y,\
\mp s\alpha_s c_s\beta_z,\
\alpha_s c_{s_0}\sqrt{\rho}\beta_y,\
\alpha_s c_{s_0}\sqrt{\rho}\beta_z,\
\alpha_f\gamma p
\,\right)^{\mathrm{T}}.$$

Longitudinal compression plus in-phase perpendicular-$B$ compression.

### Slow magnetosonic waves — $\lambda_{3,5} = v_x \mp c_s$

$$\mathbf{r}_{s\pm} = \left(\,
\rho\alpha_s,\
\pm\alpha_s c_s,\
\pm s\alpha_f c_f\beta_y,\
\pm s\alpha_f c_f\beta_z,\
-\alpha_f c_{s_0}\sqrt{\rho}\beta_y,\
-\alpha_f c_{s_0}\sqrt{\rho}\beta_z,\
\alpha_s\gamma p
\,\right)^{\mathrm{T}}.$$

Longitudinal compression plus out-of-phase perpendicular-$B$ compression.

## Degenerate-limit eigenvectors

When $|\mathbf{B}_\perp| = 0$ (pure longitudinal $\mathbf{B}$), the
$\beta_y, \beta_z$ definitions above divide by zero. Stone+08
Eq. (B17)-(B20) gives replacement eigenvectors for this case. We do
not verify these here — they are documented in the Athena++ source
and must be implemented as branch-conditionals in the kernel.

## ✅ Verification checkpoint (to be wired)

The future

```
tests/test_athena_mhd_eigensystem.cu
```

must:

1. For 100 random admissible states, evaluate
   `d_mhd_eigenvalues_primitive` and verify each returned $\lambda_k$
   agrees with (A3-wave-speeds) to $10^{-12}$ absolute.
2. For each state, evaluate `d_mhd_right_eigenvectors` and verify
   $\|\mathsf{A}_W(\mathbf{W})\,\mathbf{r}_k - \lambda_k\mathbf{r}_k\|_{\infty}
   < 10^{-12}$ for all $k \in \{1,\dots,7\}$.
3. Pressure-positivity stress test: states with $\beta_{\mathrm{plasma}} < 0.01$,
   verify no NaN / no overflow in $\alpha_f$, $\alpha_s$.
4. Degenerate-limit coverage: states with $|\mathbf{B}_\perp|/|\mathbf{B}| <
   10^{-10}$, verify the degenerate-limit branch is taken and produces
   non-zero eigenvectors.

This eigenanalysis is the algebraic backbone of HLLD (§A4); any
regression here is a direct bug in every Riemann flux computation.

# A4. HLLD intermediate states (Miyoshi-Kusano 2005)

> **sympy script:** `scripts/a4_hlld_intermediate_states.py`
> **verified:** $p^{\star}_{\text{tot,L}} = p^{\star}_{\text{tot,R}}$
> symbolically; Rankine–Hugoniot outer-wave jumps numerically
> (20 trials, max err $3\times10^{-15}$).
> **code checkpoints:** `athena_mhd_kernels.cu::d_hlld_flux`.

## Wave fan

HLLD partitions the Riemann fan between $S_L$ and $S_R$ into four
intermediate plateaus separated by $S^\star_L, S_M, S^\star_R$ —
outer fast, Alfvén, contact, Alfvén, outer fast.

## Contact speed (MK Eq.\ 38)

$$\boxed{S_M = \frac{(S_R - v_{xR})\rho_R v_{xR} - (S_L - v_{xL})\rho_L v_{xL}
 - p^{\star}_{\text{tot,R}} + p^{\star}_{\text{tot,L}}}
 {(S_R - v_{xR})\rho_R - (S_L - v_{xL})\rho_L}.}$$

## Star-region state

$\rho^{\star}_K = \rho_K(S_K - v_{xK})/(S_K - S_M)$ (MK Eq.\ 43);
$p^{\star}_{\text{tot}} = p^{\star}_{\text{tot,L}} + \rho_L(S_L - v_{xL})(S_M - v_{xL})
= p^{\star}_{\text{tot,R}} + \rho_R(S_R - v_{xR})(S_M - v_{xR})$
(MK Eq.\ 41). **Sympy symbolically verifies** the L/R equality.

$B^{\star}_{yK}, v^{\star}_{yK}$ from MK Eqs.\ 44-47 (see script).
Alfvén speeds $S^{\star}_{L,R} = S_M \mp |B_x|/\sqrt{\rho^\star_{L,R}}$.

## Critical implementation note

**HLLD star state is NOT an EOS state.** Do NOT compute $\mathbf{F}^\star_K$
via $p = (\gamma-1)(E^\star - \tfrac{1}{2}\rho|\mathbf{v}|^2 - \tfrac{1}{2}|\mathbf{B}|^2)$
— it gives the wrong pressure. Use MK Eq.\ 64 directly with
$p^\star_{\text{tot}}$ from (A4-Ptot-star). Numerical verification
confirms: RH residuals are $\mathcal{O}(1)$ with EOS inversion,
$\mathcal{O}(10^{-15})$ with direct MK formula.

## ✅ Verification

`tests/test_athena_mhd_hlld.cu` — Brio-Wu shock tube match Stone+08
Fig 28 to $L^1 < 2\%$ at $N=512$.

# A5. Constrained Transport and discrete $\nabla\cdot\mathbf{B}=0$

> **sympy script:** `scripts/a5_ct_divergence_preservation.py`
> **verified:** CT telescoping identity
> $(\nabla\cdot\mathbf{B})^{n+1} - (\nabla\cdot\mathbf{B})^n = 0$;
> Gardiner-Stone 2005 corner-EMF averaging is 2nd-order accurate.
> **code checkpoints:** `athena_mhd_solver.cu::update_face_B_with_emf`,
> `tests/test_athena_mhd_field_loop.cu`.

## Grid layout

Yee-like staggered: $B_x$ on vertical faces $(i\pm\tfrac{1}{2}, j)$,
$B_y$ on horizontal faces $(i, j\pm\tfrac{1}{2})$, $E_z$ on corners.

## CT update (Evans-Hawley 1988)

$$B_x^{i\pm 1/2, j, n+1} = B_x^{i\pm 1/2, j, n} - \frac{\Delta t}{\Delta y}(E_z^{i\pm 1/2, j+1/2} - E_z^{i\pm 1/2, j-1/2}),$$
$$B_y^{i, j\pm 1/2, n+1} = B_y^{i, j\pm 1/2, n} + \frac{\Delta t}{\Delta x}(E_z^{i+1/2, j\pm 1/2} - E_z^{i-1/2, j\pm 1/2}).$$

## The central identity

$$\boxed{(\nabla\cdot\mathbf{B})^{n+1}_{i,j} = (\nabla\cdot\mathbf{B})^n_{i,j}}$$

**Sympy symbolically verifies** this as a pure telescoping identity —
the four corner-EMF contributions cancel exactly, regardless of what
values the EMFs take. This is why CT is qualitatively different from
Dedner GLM: ∇·B = 0 is an **exact** property, not a truncation
error.

## Gardiner-Stone 2005 corner-EMF averaging

$$E_z^{i+1/2, j+1/2} = \tfrac{1}{4}(E_z^{x,i+1/2,j} + E_z^{x,i+1/2,j+1} + E_z^{y,i,j+1/2} + E_z^{y,i+1,j+1/2}).$$

**Sympy-verified** via Taylor expansion: leading error is
$\tfrac{h^2}{8}(\partial_x^2 E_z + \partial_y^2 E_z) + \mathcal{O}(h^4)$,
no $\mathcal{O}(h)$ term — matches 2nd-order Godunov accuracy.

## ✅ Verification

`tests/test_athena_mhd_field_loop.cu` — GS05 field-loop, 10
crossings, lock $\max|\nabla\cdot\mathbf{B}| < 10\varepsilon_{\mathrm{mach}}$
at every step.

# A6. Reconstruction (PLM/PPM) and TVD slope limiters

> **sympy script:** `scripts/a6_reconstruction_tvd.py`
> **verified:** minmod / vL / MC opposite-sign cancellation; Sweby
> TVD region $0\le\varphi(r)\le\min(2,2r)$ for $r\in[0,4]$; PLM
> reconstruction second-order; PPM 4-point interpolant 4th-order
> (exact to cubics); PPM parabola $\int_0^1 W\,d\xi = a_0$.
> **code checkpoints:**
> `athena_mhd_kernels.cu::d_reconstruct_primitive_plm`,
> `athena_mhd_kernels.cu::d_reconstruct_primitive_ppm`,
> `tests/test_athena_mhd_linear_wave_convergence.cu`.

## Why this section matters

VL2 + HLLD without a **monotonic** reconstruction is 2nd-order but
gives spurious oscillations at every shock. The three limiters below
are the only ones we need in practice: **minmod** (safest,
dispersive), **van Leer** (smooth, slightly sharper), **MC** (most
aggressive, Stone+08 default).

A skipped step here directly causes the class of bugs already seen on
stellar2d:

- `cart_ale2` swept-remap used donor-cell then MUSCL without
  characteristic projection — caused the periodic-BC drift at
  `P30/P31`.
- Stone+08 App. A is explicit: **reconstruct in primitive variables,
  then project onto characteristic variables** via the eigenvectors
  of §A3. Skipping the projection converts crisp fast-mode jumps into
  diffusive blobs.

## Definitions

$$\sigma_{\text{minmod}}(a,b) =
\begin{cases} 0, & \mathrm{sign}(a)\neq\mathrm{sign}(b),\\
\mathrm{sign}(a)\,\min(|a|,|b|), & \text{else}.\end{cases}$$

$$\sigma_{\text{vL}}(a,b) =
\begin{cases} 0, & ab\le 0,\\
\dfrac{2ab}{a+b}, & ab>0.\end{cases}$$

$$\sigma_{\text{MC}}(a,b) = \mathrm{sign}(a)\,\min\!\left(2|a|,\tfrac{|a+b|}{2},2|b|\right)
\text{ when } ab>0, \text{ else } 0.$$

## Sweby TVD region (A6-Sweby)

All three limiters satisfy, with $r = b/a$:

$$\boxed{0 \leq \varphi(r) \leq \min(2,\,2r),\quad r\ge 0.}$$

**Sympy-verified** by numerical sweep ($r\in[0,4]$, 401 samples).
Any limiter in this region is TVD for 1D scalar advection with
CFL $\le 1$ (Harten 1983; LeVeque 2002 §16).

## PLM reconstruction and order

$$W^{L}_{i+1/2} = W_i + \tfrac{1}{2}\sigma_i,\quad
\sigma_i = \tfrac{1}{2}(W_{i+1} - W_{i-1}).$$

**Leading error** $W^{L}_{i+1/2} - W(x_{i+1/2}) = -\tfrac{h^{2}}{8}W''(x_{i+1/2}) + \mathcal{O}(h^{3})$,
so the reconstruction is **second order** (no $\mathcal{O}(h)$ term).
Sympy-verified: no $h^0$ or $h^1$ terms in the residual expansion.

## PPM (Colella-Woodward 1984) 4-point interpolant

$$\boxed{W_{i+1/2} = \tfrac{7}{12}(W_i + W_{i+1})
  - \tfrac{1}{12}(W_{i-1} + W_{i+2}),}$$

applied to the **cell-averaged** values. Sympy-verified: no
$\mathcal{O}(h^0..h^3)$ term in the expansion against the smooth
reference; leading error is $-\tfrac{1}{30} h^{4} W^{(4)}$.

The parabolic form inside a cell (Colella-Woodward 1984 Eq. 1.6),

$$W(\xi) = a_L + \xi[\Delta a + a_6(1 - \xi)], \quad
\Delta a = a_R - a_L,\quad a_6 = 6(a_0 - (a_L + a_R)/2),$$

satisfies $\int_0^1 W\,d\xi = a_0$ **exactly** (sympy-verified):
conservation of cell averages is built in.

## Practical recommendation (Stone+08 Appendix A)

1. Reconstruct $(\rho, \mathbf{v}, \mathbf{B}, p)$ in **primitive**
   variables (not conservative). This keeps positivity of $\rho, p$
   easier to enforce.
2. Project onto characteristic variables
   $\delta W^{(k)} = \ell_k \cdot \delta W$ using the left-
   eigenvectors of §A3, limit each wave family separately, then
   project back $\delta W = \sum_k r_k \, \delta W^{(k)}$.
3. Use MC as the default; switch to van Leer near strong shocks if
   MC produces staircase artefacts (rare but documented in Stone+08
   Fig 28).

## ✅ Verification checkpoints

- `tests/test_athena_mhd_linear_wave_convergence.cu` — 3-resolution
  linear fast-wave advection; L¹ convergence slope in $[1.9, 2.1]$.
- `tests/test_athena_mhd_shock_tube.cu` — Brio-Wu, MC limiter, no
  staircase within the fast rarefaction fan.

# A7. Van-Leer 2 predictor-corrector (Stone-Gardiner 2009)

> **sympy script:** `scripts/a7_vl2_predictor_corrector.py`
> **verified:** amplification factor $g(\xi)=1-i\nu\xi-\tfrac{\nu^2}{2}\xi^2+\mathcal{O}(\xi^3)$
> (Lax-Wendroff 2nd-order); $|g|^2\le 1$ on $|\nu|\le 1$;
> truncation error $\mathcal{O}(\Delta t^3 + \Delta t\,\Delta x^2)$.
> **code checkpoints:**
> `athena_mhd_solver.cu::vl2_predictor_step`,
> `athena_mhd_solver.cu::vl2_corrector_step`.

## The scheme

$$\boxed{\mathbf{U}^{\star} = \mathbf{U}^{n} + \tfrac{\Delta t}{2}\mathcal{L}[\mathbf{U}^{n}],\qquad
\mathbf{U}^{n+1} = \mathbf{U}^{n} + \Delta t\,\mathcal{L}[\mathbf{U}^{\star}].}$$

This is **midpoint-RK2** in time, paired with any semi-discrete
operator $\mathcal{L}$ (here: PLM reconstruction + HLLD flux from §A4,
plus CT EMF update from §A5).

## Why VL2 over Strang or CTU

| Feature | VL2 | Strang | CTU |
|---|---|---|---|
| Dimensional coupling | Unsplit, exact | Split, $\mathcal{O}(\Delta t^2)$ error | Unsplit |
| Auxiliary state | $\mathbf{U}^\star$ | per-direction sweep | corner-transport cells |
| CFL limit (unsplit) | $\le 1$ | $\le 1$ in each direction | $\le 1$ (multi-D) |
| Compatibility with CT | Native (Stone-Gardiner 2009) | Requires extra corner EMF gymnastics | Native |
| Kernel complexity | Low (2 stages) | Low (per-direction) | Higher (CTU corner sweeps) |

Stone+08 CTU is more accurate on discontinuities but significantly
more complex; VL2 is the default in Athena++ and what we adopt.

## Fourier (von Neumann) analysis on linear advection

With PLM + upwind flux for $a>0$ on a smooth Fourier mode,

$$g(\xi) = 1 - i\nu\xi - \tfrac{\nu^2}{2}\xi^2 + \mathcal{O}(\xi^3),
\quad \nu \equiv a\Delta t / \Delta x,$$

**matching Lax-Wendroff** at leading orders. Sympy-verified the three
coefficients. Numerical sweep confirms $|g|^2 \le 1$ for
$\nu\in\{0.1,\ldots,1.0\}$; at $\nu = 1.05$ the amplification is
$|g|^2 \approx 1.22 > 1$, confirming the CFL limit is **sharp**.

## Truncation error (A7-consistency)

With the central-flux semi-discrete operator
$\mathcal{L}[U]_j = -a(U_{j+1}-U_{j-1})/(2h)$ acting on smooth $U$,
after two VL2 stages,

$$\boxed{\mathbf{U}^{n+1}_j - U(x_j, t+\Delta t) = \mathcal{O}(\Delta t^3 + \Delta t\,\Delta x^2).}$$

Leading error is a purely dispersive $U_{xxx}\,a(a^2-1)/6$ term,
vanishing at $\nu=1$ (exact for advection on-CFL). Sympy expands the
predictor/corrector composition and confirms the residual has no
$\mathcal{O}(\varepsilon^0 \ldots \varepsilon^2)$ terms (with
$\varepsilon = \Delta t = h$).

## CFL constraint (1D)

$$\Delta t \leq C_{\mathrm{CFL}}\,\frac{\Delta x}
{\max_{\text{cells}}(|v_x| + c_f)},\quad C_{\mathrm{CFL}} \leq 1.$$

See §A8 for multidimensional generalisation and parabolic terms.

## Implementation tips (for `athena_mhd_solver.cu`)

- Save $\mathbf{U}^n$ **before** the predictor so the corrector can
  rewind if a positivity check fails.
- The predictor need **not** compute the flux to full order — a 1st-
  order Godunov flux is enough; the corrector is where the 2nd-order
  accuracy is paid for. Stone+08 explicitly notes this economy.
- For CT, compute **face-centred** $E_z$ at both predictor and
  corrector; average at the corners using Gardiner-Stone 2005 (§A5).

## ✅ Verification checkpoints

- `tests/test_athena_mhd_linear_wave_convergence.cu` — three
  resolutions, fast wave, expect L¹ convergence slope in $[1.9,2.1]$.
- `tests/test_athena_mhd_vl2_stability.cu` — CFL scan
  $\nu\in\{0.5, 0.9, 0.99\}$ stable; $\nu = 1.05$ must blow up.

# A8. MHD CFL time-step constraints

> **sympy script:** `scripts/a8_mhd_cfl.py`
> **verified:** FTCS diffusion amplification worst-case at $\xi=\pi$
> gives $\sigma \le 1/2$; $c_f$ limits in three degenerate cases
> ($B_\perp=0$, $B_x=0$, $\mathbf{B}=0$).
> **code checkpoints:**
> `athena_mhd_kernels.cu::d_mhd_dt_reduction`,
> `athena_mhd_solver.cu::compute_dt`.

## Hyperbolic CFL (fast wave dominant)

Combining §A3 (7-wave eigensystem) with §A7 ($|\nu|\le 1$), the
unsplit multidimensional bound is

$$\boxed{\Delta t_{\mathrm{hyp}} \leq
C_{\mathrm{CFL}}\ \Bigg/ \sum_{d=1}^{D}\max_{\text{cells}}\!\left(\frac{|v_d| + c_{f,d}}{\Delta x_d}\right),\quad C_{\mathrm{CFL}} \leq 1,}$$

where the fast-magnetosonic speed $c_{f,d}$ in direction $d$ is the
positive root of the §A3 discriminant.

## Parabolic CFL (Ohmic + ambipolar diffusion)

For the scalar diffusion $\partial_t U = \eta \partial_x^2 U$ with FTCS
central space + forward Euler time, the amplification factor is
$g(\xi) = 1 - 4\sigma\sin^2(\xi/2)$ with $\sigma = \eta\Delta t/\Delta x^2$.
The worst case $\xi=\pi$ gives $g = 1 - 4\sigma$; stability $|g|\le 1$
requires $\sigma \le 1/2$ (sympy-verified).

Combining Ohmic + ambipolar as $\eta_\text{eff} = \eta_\Omega + \eta_{\mathrm{AD}}$:

$$\boxed{\Delta t_{\mathrm{para}} \leq \tfrac{1}{2}\min_{\text{cells}}\frac{\Delta x^{2}}{\eta_\Omega + \eta_{\mathrm{AD}}}.}$$

## Fast-speed behaviour at degenerate limits (A8-cf-limits)

Sympy-verified:

| Limit | $c_f^2$ |
|---|---|
| $B_\perp = 0$ (tangential-free) | $\max(c_{s_0}^2, c_{Ax}^2)$ |
| $B_x = 0$ (perpendicular only) | $c_{s_0}^2 + c_{A\perp}^2$ |
| $\mathbf{B} = 0$ (hydro) | $c_{s_0}^2$ |

The first two use `numerical fall-back` in sympy because `sp.Max` does
not simplify against the nested radical; 30 random states confirm
agreement to machine precision.

## Combined rule

$$\Delta t = \min\!\left(\Delta t_{\mathrm{hyp}},\,\Delta t_{\mathrm{para}}\right).$$

## Kernel-form reduction

Implementation-wise, per cell:

$$\left(\frac{1}{\Delta t}\right)_{\!i,j,k} =
\frac{|v_x| + c_f}{\Delta x} + \frac{|v_y| + c_f}{\Delta y} + \frac{|v_z| + c_f}{\Delta z} + 2\frac{\eta_\Omega + \eta_{\mathrm{AD}}}{\min(\Delta x,\Delta y,\Delta z)^{2}}.$$

Then $\Delta t^{-1}$ is reduced (`max`) over the grid and the global
$\Delta t$ is $C_\text{CFL} / \Delta t^{-1}_\text{max}$.

## Super-time-stepping (optional, future)

When $\eta$ is large enough that $\Delta t_{\mathrm{para}} \ll \Delta t_{\mathrm{hyp}}$,
the RKL2 super-time-stepping scheme (Meyer+12) advances the parabolic
part with $N$ sub-steps per hyperbolic step at $\mathcal{O}(N^2)$ CFL
relaxation. Not included in the initial `athena_mhd` implementation;
reserved for §C5 future extension when Suzuki turbulent heating is
active.

## ✅ Verification checkpoints

- `tests/test_athena_mhd_cfl_advection.cu` — advection of smooth
  wave at $C_\text{CFL} = 0.95$; no growth. At $C_\text{CFL} = 1.05$
  the test **must** go unstable within 10 steps (positive control).
- `tests/test_athena_mhd_ohm_diffusion.cu` — pure Ohmic field-loop
  decay; $\Delta t_\text{para}$ honored; analytic decay rate matched
  to $< 10^{-5}$.

# A9. HLLD degenerate branches

> **sympy script:** `scripts/a9_hlld_degeneracy.py`
> **verified:** on the Alfvén locus $(S_K - v_{xK})^2 = B_x^2/\rho_K$,
> numerator of $B^\star_{yK}$ vanishes and denominator reduces to
> $\rho_K(S_K-v_{xK})(v_{xK} - S_M)$; $D_{S_M} = 0$ when both sides
> of the Riemann interface are identical.
> **code checkpoints:**
> `athena_mhd_kernels.cu::d_hlld_flux` (branch dispatch).

## Three degeneracies of the generic HLLD

The §A4 formulas $B^\star_{yK}, v^\star_{yK}$ blow up in three
cases. The kernel must dispatch on these **before** evaluating the
generic formulas.

| Branch | Condition | Fallback |
|---|---|---|
| **D1** | $B_x = 0$ | HLLC-MHD (Li 2005), 3-wave fan |
| **D2** | $(S_K - v_{xK})^2 = B_x^2/\rho_K$ | Upstream $B_y, v_y, B_z, v_z$ unchanged |
| **D3** | $S_L \to S_R$ and sides identical | Pure HLL (2-wave average) |

## D1 — Zero longitudinal field ($B_x = 0$)

$$B_x = 0 \Longrightarrow S^\star_L = S^\star_R = S_M,\ \mathbf{U}^\star_K = \mathbf{U}^{\star\star}_K,$$

so the two-plateau star structure collapses to one plateau. The
algorithm reduces to **HLLC-MHD** (Li 2005 Appendix A). Recommended
implementation: detect $|B_x| < \varepsilon \cdot (c_{s_0} + c_{A\perp})\sqrt{\rho}$
and dispatch.

## D2 — Alfvén locus removable singularity

The MK Eq. 44 denominator
$\rho_K(S_K-v_{xK})(S_K-S_M) - B_x^2$ vanishes on the locus
$(S_K-v_{xK})^2 = B_x^2/\rho_K$, where the upstream state already
sits on the Alfvén characteristic.

**Sympy-verified** along the locus:
- Numerator $\rho_K(S_K-v_{xK})^2 - B_x^2 \to 0$.
- Denominator reduces to $\rho_K(S_K-v_{xK})(v_{xK} - S_M)$.

So both vanish, and $B^\star_{yK}$ is $0/0$ — *removable*. The physical
content is that the transverse field does not jump across an Alfvén
wave that coincides with the upstream state.

**Kernel regularisation:**

$$\boxed{|\text{den}| < \epsilon\sqrt{\rho_K}\,|B_x|\ \Longrightarrow\ B^\star_{yK} \leftarrow B_{yK},\ v^\star_{yK} \leftarrow v_{yK}\ (\text{and same for } z).}$$

with $\epsilon \sim 10^{-12}$ in double precision. Falling through to
the upstream state is continuous with the generic formula.

## D3 — $S_L \approx S_R$ (vacuum / strongly-aligned)

When $\|(S_R - v_{xR})\rho_R - (S_L - v_{xL})\rho_L\| < \epsilon$, the
$S_M$ denominator vanishes. Physically: the Riemann fan has collapsed
(L = R), so there is no jump to resolve. Fall back to pure HLL:

$$\mathbf{F}_{\mathrm{HLL}} = \frac{S_R \mathbf{F}_L - S_L \mathbf{F}_R + S_L S_R(\mathbf{U}_R - \mathbf{U}_L)}{S_R - S_L}.$$

Pure HLL is diffusive but unconditionally stable, a safe last-resort.

## Implementation order (most to least specific)

```
if |D_SM| < ε:
    return HLL_flux()                  # D3
if |B_x| < ε * (cs0 + cA⊥)*sqrt(ρ):
    return HLLC_MHD_flux()             # D1
compute generic star states (§A4)
if |den_K| < ε * sqrt(ρ_K) * |B_x|:    # D2
    set B_y*_K, B_z*_K, v_y*_K, v_z*_K
      to upstream values
return HLLD_flux(generic)
```

## Why this matters

Without these dispatches, the Brio-Wu shock tube **will** produce NaN
at $t \sim 0.05$ when a rarefaction fan grazes the Alfvén locus.
Documented in Miyoshi-Kusano 2005 Sec. 3.4 and in Athena++
`src/hydro/rsolvers/mhd/hlld.cpp::HLLDTransport` branch logic.

## ✅ Verification checkpoints

- `tests/test_athena_mhd_brio_wu.cu` — Brio-Wu, 512 cells, match
  Stone+08 Fig 28 to $L^1 < 2\%$.
- `tests/test_athena_mhd_degenerate_Bx0.cu` — synthetic state with
  $B_x = 10^{-14}$, must not produce NaN and must match pure-HLLC
  hydrodynamic flux to $10^{-10}$.

# A10. Powell 8-wave source vs Constrained Transport

> **sympy script:** `scripts/a10_powell_vs_ct.py`
> **verified:** Powell source term $\mathbf{S}_\mathrm{P}$ vanishes
> identically when $\nabla\cdot\mathbf{B} = 0$; since §A5 gives CT
> preservation at the discrete level, $\mathbf{S}_\mathrm{P} \equiv 0$
> for all $n$ and no correction is needed.
> **code checkpoints:** no new code required — documented to prevent
> future "add a Powell source term for safety" PR.

## The 8-wave system (Powell+99)

The 8-wave formulation augments ideal MHD with a divergence-cleaning
wave and a **non-conservative** source term:

$$\partial_t \mathbf{U} + \partial_i \mathbf{F}_i(\mathbf{U}) = -(\nabla\!\cdot\!\mathbf{B})\,\mathbf{S}_\mathrm{P}(\mathbf{U}),$$

with

$$\mathbf{S}_\mathrm{P} = \begin{pmatrix}0\\ \mathbf{B}\\ \mathbf{v}\\ \mathbf{B}\cdot\mathbf{v}\end{pmatrix}.$$

At the **continuous** level, $\nabla\cdot\mathbf{B}=0$ exactly, so
$\mathbf{S}_\mathrm{P}\cdot 0 = 0$ — trivial.

## The non-trivial claim

At the **discrete** level, a cell-centred or vertex-centred $\mathbf{B}$
storage generally has $(\nabla\cdot\mathbf{B})_{i,j} \ne 0$ at
$\mathcal{O}(\Delta x)$, and the Powell source is a genuine
correction. **But** on the Yee-staggered grid with CT (§A5), the
discrete $(\nabla\cdot\mathbf{B})^n_{i,j}$ satisfies

$$(\nabla\!\cdot\!\mathbf{B})^{n}_{i,j} = (\nabla\!\cdot\!\mathbf{B})^{0}_{i,j}\ \forall\,n$$

(the §A5 telescoping identity). Provided initialisation seeds
$\mathbf{B}^0$ from a vector potential or via one projection solve
($\nabla^2 \phi = \nabla\!\cdot\mathbf{B}_\text{raw}$,
$\mathbf{B} \leftarrow \mathbf{B}_\text{raw} - \nabla\phi$),
$(\nabla\!\cdot\!\mathbf{B})^0 = 0$ and thus

$$\boxed{\mathbf{S}_\mathrm{P}^n \equiv 0\ \forall n.}$$

## GLM-MHD contrast (Dedner+02)

GLM adds an 8th field $\psi$ and hyperbolic-parabolic cleaning:

$$\partial_t \psi + c_h^{2}(\nabla\!\cdot\!\mathbf{B}) = -\alpha\psi,\quad
\partial_t \mathbf{B} + \nabla\psi = \dots$$

GLM **does not** preserve $\nabla\cdot\mathbf{B} = 0$ exactly; it
advects and damps the constraint violation at wave speed $c_h$. CT
is *exact* by construction; GLM trades exactness for a grid that
does not need face-centred field storage (ADER-DG codes etc.).

For `athena_mhd` we use CT. This section exists to lock in "no Powell
source needed" as a *derived* result, not an assumption.

## Implementation note

The kernel must:

1. Seed $\mathbf{B}^0$ from a vector potential $\mathbf{A}$ via
   $\mathbf{B} = \nabla\times\mathbf{A}$ at face centres. No
   projection needed; $\nabla\cdot\mathbf{B} \equiv 0$ by vector
   calculus.
2. Run the diagnostic `d_mhd_divB_check` after every time step to
   confirm $\max |\nabla\cdot\mathbf{B}| < 10\,\epsilon_{\text{mach}}$.
   If this check ever fails, the bug is in Step 1 or in the CT
   kernel; not a reason to add a Powell source.

## ✅ Verification checkpoints

- `tests/test_athena_mhd_field_loop.cu` — Gardiner-Stone 2005 field
  loop, 10 diagonal advections. Track $\max|\nabla\cdot\mathbf{B}|$
  — must remain at machine precision.
- No test for the Powell source — it is **not** in the kernel and
  should never be added.

# A11. Linear MHD wave initial conditions (Stone+08 Tab 1)

> **sympy script:** `scripts/a11_linear_wave_initial_conditions.py`
> **verified:** with $\rho_0=1, p_0=1/\gamma, \mathbf{B}_0=(1,\sqrt{2},1/2), \gamma=5/3$
> the wave speeds are $c_f=2$, $c_s=1/2$, $c_{Ax}=1$, $c_{s_0}=1$, all
> matching Stone+08 Table 1; the discriminant identities
> $c_f^2 + c_s^2 = c_{s_0}^2 + c_A^2$ and $c_f^2 c_s^2 = c_{s_0}^2 c_{Ax}^2$
> are sympy-verified.
> **code checkpoints:**
> `tests/test_athena_mhd_linear_wave_convergence.cu`;
> `tst/test_athena_mhd/test_linwave.py` (Python harness analogous to
> the existing `tst/test_ale2/test_linwave.py`).

## Why this is the "gold-standard" convergence test

Linear wave advection has a known exact solution: after one period
the perturbation returns to its initial shape, and any deviation is
**purely numerical**. The L¹ norm of the difference, sampled at
multiple resolutions, directly measures convergence slope and
diagnoses accuracy loss of the scheme.

A scheme that claims 2nd order but gives slope 1.3 in the fast-wave
test has a broken reconstruction, broken HLLD, or broken CT — and
this test finds that before `athena_mhd` touches a shock tube.

## Background (Stone+08 Table 1)

$$\rho_0 = 1,\quad p_0 = \frac{1}{\gamma},\quad \mathbf{v}_0 = \mathbf{0},\quad
\mathbf{B}_0 = \left(1,\ \sqrt{2},\ \tfrac{1}{2}\right),\quad \gamma = \tfrac{5}{3}.$$

**Derived wave speeds** (all sympy-verified):

$$c_f = 2,\quad c_{Ax} = 1,\quad c_s = \tfrac{1}{2},\quad c_{s_0} = 1.$$

## Perturbation

For mode $k$ with eigenvector $\mathbf{r}_k$ (from §A3) and amplitude
$A = 10^{-6}$:

$$\mathbf{W}(x, 0) = \mathbf{W}_0 + A\,\mathbf{r}_k\,\cos\!\left(\frac{2\pi x}{L}\right),\quad x\in[0, L].$$

## Periods on an $L$-periodic box

$$T_f = L/2,\quad T_A = L,\quad T_s = 2L.$$

## Entropy-mode exception

The entropy eigenvalue is $\lambda_\text{ent} = v_{0,x}$ which is $0$
at our base state. To make the entropy-wave convergence test
meaningful, **choose $v_{0,x} = 1$ for the entropy-mode case only**;
then $T_\text{ent} = L$. Stone+08 does the same (Sec 6.1).

## $\nabla\cdot\mathbf{B} = 0$ at IC (automatic)

In the 1D MHD system (§A3) $B_x$ is a parameter, not an evolution
variable; all seven right-eigenvectors live in
$(\rho, v_x, v_y, v_z, B_y, B_z, p)$ and do not perturb $B_x$. For a
plane wave along $\hat{x}$, $\nabla\cdot\delta\mathbf{B} = ik_x\delta B_x = 0$
automatically. **No extra projection step needed at IC.**

## Expected error

For a VL2 + HLLD + CT code (§A7, §A4, §A5),

$$\varepsilon_{L^1}(k, \Delta x) = C_k\,A\,(\Delta x / L)^{2},$$

with $C_k$ an $\mathcal{O}(1)$ mode-dependent constant. Convergence
slope measured across $\Delta x = L/64, L/128, L/256$ should fall in
$[1.9, 2.1]$ for all 7 modes.

## Practical test protocol

1. For each of the 7 modes:
   - Run the solver at resolutions $N \in \{64, 128, 256\}$.
   - Evolve for one period $T_k$.
   - Compute $\varepsilon_{L^1}(N) = \frac{1}{N}\sum_j |W_j^{N} - W_j^{\text{exact}}|$.
2. Fit slope: $\log\varepsilon \propto p \log\Delta x$.
3. Accept if $|p - 2| < 0.1$.

## ✅ Verification checkpoint

`tests/test_athena_mhd_linear_wave_convergence.cu` — single test that
cycles through 7 modes × 3 resolutions × 1 period; outputs a
21-row CSV; asserts slope ∈ [1.9, 2.1] per mode.

## Extension: 3D linear wave

A 3D linear wave IC is constructed identically, substituting
$k_x \to \mathbf{k}$ and rotating the eigenvector by the same angle.
Stone+08 §6.2 does the oblique test at $\mathbf{k} = (1,2,2)$. This
stresses dimensional coupling. Reserved for post-MVP; not included
in the initial test suite.

# B1. Super-radial flux-tube reduction

> **sympy script:** `scripts/b1_flux_tube_geometry.py`
> **verified:** flux conservation $AB_r = \text{const}$; MHSE
> $\partial_r p + \rho g = -B_r^2\partial_r(\ln A)$;
> Kopp-Holzer $f(R_*) = 1$, $f(\infty) = f_{\max}$.
> **code checkpoints:** `tests/test_mhd_wind_hse_stationary.cu`.

## Geometry

$A(r) = r^2 f(r)$, $f(R_*) = 1$, $f(\infty) = f_{\max}$. Kopp-Holzer
form:

$$f(r) = \frac{f_{\max}\,e^{(r-R_1)/\sigma} + f_1}{e^{(r-R_1)/\sigma}+1},\quad f_1 = 1 - (f_{\max}-1)e^{(R_*-R_1)/\sigma}.$$

## 1D flux-tube equations

$$\partial_t(\rho A) + \partial_r(\rho v_r A) = 0,$$
$$\partial_t(\rho v_r A) + \partial_r[A(\rho v_r^2 + p_{\text{tot}} - B_r^2)] = (p_{\text{tot}} - B_r^2)\partial_r A - \rho g A,$$
$$\partial_t(EA) + \partial_r[A((E+p_{\text{tot}})v_r - B_r(\mathbf{B}\cdot\mathbf{v}))] = -\rho g v_r A + AQ.$$

## Flux conservation

$$\boxed{A(r)B_r(r) = R_*^2 B_0 \Rightarrow B_r(r) = \frac{B_0}{f(r)}(R_*/r)^2.}$$

## MHSE

$$\boxed{\partial_r p + \rho g = -B_r^2\,\partial_r(\ln A).}$$

**IC gotcha**: for a Suzuki-style wind run, atmospheres must be
integrated from this MHSE, NOT plain HSE. Using HSE triggers a
$\sim 10^{-2}$ transient and corrupts $\dot M$ measurements.

## WKB amplitude (see B2)

$\delta v_\perp \propto (\rho v_A A)^{-1/2}$ (Poynting) or
$(\rho v_A A)^{-1/4}$ (per-mode amplitude convention).

## ✅ Verification

`tests/test_mhd_wind_hse_stationary.cu` — HSE atmosphere, 100
acoustic crossings, lock $\max|v_r|/c_s < 10^{-4}$.

# B2. WKB wave action conservation for Alfvén waves

> **sympy script:** `scripts/b2_wave_action_wkb.py`
> **verified:** dispersion $\omega^2 = v_A^2 k^2$; no-reflection limit
> under uniform $\rho_0, B_{r,0}$.
> **code checkpoints:** `athena_mhd_kernels.cu::d_alfven_wave_driver`,
> `tests/test_mhd_wind_amplitude_scaling.cu`.

## Linearised transverse MHD

Around a static background $(\rho_0(r), B_{r,0}(r))$ with $v_{r,0} = 0$:

$$\partial_t v_{\perp} = \frac{B_{r,0}}{\rho_0}\partial_r B_{\perp},\qquad
\partial_t B_{\perp} = B_{r,0}\partial_r v_{\perp}. \quad (\text{B2-linear})$$

WKB ansatz gives the Alfvén dispersion

$$\boxed{\omega^2 = v_A^2 k^2,\qquad v_A \equiv B_{r,0}/\sqrt{\rho_0}.} \quad (\text{B2-disp})$$

**Sympy verified** that the determinant of the 2×2 linearised matrix
equals $-\omega^2 + v_A^2 k^2$, giving the dispersion as the unique
non-trivial mode.

## Elsässer variables

$z_\pm \equiv v_\perp \mp B_\perp/\sqrt{\rho_0}$ decouple the outgoing
and incoming Alfvén waves:

$$\partial_t z_\pm \pm v_A\,\partial_r z_\pm = S_{\mathrm{refl}}\,z_\mp, \quad (\text{B2-Elsasser})$$

with a reflection source $S_{\mathrm{refl}}$ that depends on the
background gradients. In a uniform $(\rho_0, B_{r,0})$, sympy
verifies $S_{\mathrm{refl}} = 0$ — no wave reflection in a uniform
atmosphere (Alfvén waves are pure one-way in the uniform limit).

## Wave action conservation

Outgoing $z_+$ in the absence of reflection conserves the wave action

$$\boxed{\rho_0 A\,|z_+|^2/v_A = \text{const along characteristics.}} \quad (\text{B2-action})$$

For the amplitude-scaling law of a driver BC (relevant for Suzuki
winds), **Poynting flux** $\rho_0 v_A A |\delta v_\perp|^2$ conservation
gives

$$|\delta v_\perp| \propto (\rho_0 v_A A)^{-1/2}, \quad (\text{B2-Poynting})$$

while a **per-mode** amplitude normalisation gives the Jacques-1977
form

$$|\delta v_\perp| \propto (\rho_0 v_A A)^{-1/4}. \quad (\text{B2-Jacques})$$

The two differ because the former fixes the transmitted energy flux
while the latter fixes the wave amplitude in Elsässer space. Suzuki
wind codes use **the Poynting-flux convention** at the photospheric
driver (amplitude $\langle\delta v_\perp\rangle\approx 1.25$ km/s in
Shimizu+22) — make sure the kernel matches.

## ✅ Verification

`tests/test_mhd_wind_amplitude_scaling.cu` — linearised Alfvén wave
from a fixed BC driver. Measure $\delta v_\perp^{\mathrm{rms}}(r)$ at
$r = 5, 10, 20\,R_*$; lock deviation from $(\rho v_A A)^{-1/2}$
at $< 5\%$.

# B3. Parker critical point on a super-radial flux tube

> **sympy script:** `scripts/b3_parker_critical_point.py`
> **verified:** Parker wind equation derivation; classical spherical
> limit $r_c = GM_*/(2 c_s^2)$.
> **code checkpoints:** `tests/test_mhd_parker_wind.cu`.

## Parker wind equation

Steady isothermal flow on a super-radial tube $(A(r) = r^2 f(r))$ with
$p = c_s^2 \rho$ yields, after eliminating $\rho$ via mass conservation:

$$\boxed{\left(v - \frac{c_s^2}{v}\right)\frac{dv}{dr}
= c_s^2\,\frac{d\ln A}{dr} - \frac{GM_*}{r^2}.} \quad (\text{B3-Parker})$$

**Sympy derived** from mass + momentum + EOS.

## Critical (sonic) point

At $v = c_s$ the LHS vanishes, forcing

$$c_s^2\!\left(\frac{2}{r_c} + \frac{d\ln f}{dr}\bigg|_{r_c}\right) = \frac{GM_*}{r_c^2}. \quad (\text{B3-critical})$$

**Spherical limit $f \equiv 1$:** $r_c = GM_*/(2c_s^2)$ (classical
Parker radius). Sympy verifies this by solving (B3-critical) analytically.

## Asymptotic velocity

Far from $r_c$ (spherical limit), $v(r) \sim c_s\sqrt{4\ln(r/r_c) + \text{const}}$
— logarithmic growth, characteristic of the Parker isothermal wind.

## ✅ Verification

`tests/test_mhd_parker_wind.cu` — isothermal, no magnetic driver, set
$T = 2\times10^6$ K, $M_* = 1 M_\odot$, measure Mach number crossing
at $r = r_c$. Lock $|M(r_c) - 1| < 10^{-3}$.

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

# C1. Ohmic (resistive) dissipation

> **sympy script:** `scripts/c1_ohmic_dissipation.py`
> **verified:** $\nabla\times(\nabla\times B) = \nabla(\nabla\cdot B) - \nabla^2 B$
> (all 3 components); $\nabla\times(\eta_O J) = \eta_O \nabla\times J + (\nabla\eta_O)\times J$.
> **code checkpoints:** `athena_mhd_kernels.cu::d_ohmic_dissipation`,
> `tests/test_mhd_ohmic_diffusion.cu`.

## Ohm's law with finite conductivity

$$\mathbf{E} = -\mathbf{v}\times\mathbf{B} + \eta_O\,\mathbf{J},
\qquad \mathbf{J} = \nabla\times\mathbf{B}. \quad (\text{C1-Ohm})$$

## Induction equation

**Constant $\eta_O$:**
$$\partial_t\mathbf{B} = \nabla\times(\mathbf{v}\times\mathbf{B}) + \eta_O\,\nabla^2\mathbf{B}. \quad (\text{C1-induction-const})$$

**Variable $\eta_O(\mathbf{r})$:**
$$\partial_t\mathbf{B} = \nabla\times(\mathbf{v}\times\mathbf{B})
+ \eta_O\,\nabla^2\mathbf{B} - (\nabla\eta_O)\times\mathbf{J}. \quad (\text{C1-induction-var})$$

Sympy verified via the vector identity
$\nabla\times(\eta_O\mathbf{J}) = \eta_O\nabla\times\mathbf{J} + (\nabla\eta_O)\times\mathbf{J}$.

## Joule heating

$$\boxed{Q_{\mathrm{Ohm}} = \eta_O |\mathbf{J}|^2 \ge 0.} \quad (\text{C1-Q})$$

Manifestly non-negative; enters internal-energy source.

## CFL for explicit diffusion

$$\Delta t \le \frac{(\Delta x)^2}{2\eta_O}\ (\mathrm{1D}),\
\frac{(\Delta x)^2}{4\eta_O}\ (\mathrm{2D}),\
\frac{(\Delta x)^2}{6\eta_O}\ (\mathrm{3D}). \quad (\text{C1-CFL})$$

At $r \approx 1000$ km in Matsuoka+24, $\eta_O \sim 10^{12}$ cm²/s
and $\Delta x \sim 1$ km, so $\Delta t_{\mathrm{Ohm}} \sim 10^{-4}$ s
— much smaller than hydro CFL. In practice, Matsuoka+24 use **super-
time-stepping** (Alexiades-Amiez-Gremaud 1996) to accelerate the
explicit Ohmic update.

## ✅ Verification

`tests/test_mhd_ohmic_diffusion.cu` — sinusoidal $B_y$ perturbation,
$\eta_O = $ const. Lock $L^2(B_y)$ exponential decay rate matches
analytical $e^{-\eta_O k^2 t}$ to $< 1\%$ at $N = 128$.

# C2. Ambipolar diffusion (ion-neutral drift)

> **sympy script:** `scripts/c2_ambipolar_dissipation.py`
> **verified:** $\mathbf{J}_\perp\cdot\hat{\mathbf{B}}=0$;
> $\mathbf{J}_\|\times\mathbf{B}=0$; $(\mathbf{J}\times\hat{\mathbf{B}})\times\hat{\mathbf{B}} = -\mathbf{J}_\perp$;
> $(\mathbf{J}\times\mathbf{B})\times\mathbf{B}/|\mathbf{B}|^2 = (\mathbf{J}\times\hat{\mathbf{B}})\times\hat{\mathbf{B}}$;
> $Q_{\mathrm{amb}} = \eta_A |\mathbf{J}_\perp|^2 \ge 0$.
> **code checkpoints:** `athena_mhd_kernels.cu::d_ambipolar_flux`,
> `tests/test_mhd_ambipolar_bmin.cu`.

## Motivation

In a partially-ionised plasma, **neutrals** do not couple directly to
$\mathbf{B}$; they feel the Lorentz force only via ion-neutral
collisions. When the collision rate $\nu_{in}$ is slower than MHD
timescales, ions drift relative to neutrals at

$$\mathbf{v}_{\mathrm{drift}} = \mathbf{J}\times\mathbf{B}/(\rho_i\rho_n\nu_{in}),$$

yielding the non-ideal electric field

$$\mathbf{E}_{\mathrm{amb}} = \eta_A\,(\mathbf{J}\times\hat{\mathbf{B}})\times\hat{\mathbf{B}} = -\eta_A\,\mathbf{J}_\perp, \quad (\text{C2-Eamb})$$

with $\eta_A \equiv |\mathbf{B}|^2/(\rho_i\rho_n\gamma_{in})$.

## Parallel / perpendicular current selectivity

**Critical property**: only $\mathbf{J}_\perp$ is dissipated by
ambipolar; the parallel current $\mathbf{J}_\|$ flows freely along
$\hat{\mathbf{B}}$:

$$\mathbf{J}_\|\times\mathbf{B} = \mathbf{0}. \quad (\text{C2-selective})$$

## Kernel-friendly form

For implementation, the non-unit-vector form is preferred:

$$\boxed{\mathbf{E}_{\mathrm{amb}} = \eta_A\,\frac{(\mathbf{J}\times\mathbf{B})\times\mathbf{B}}{|\mathbf{B}|^2},} \quad (\text{C2-altform})$$

avoids dividing by $|\mathbf{B}|$ in the normalisation step — matters
for low-$|\mathbf{B}|$ robustness (chromosphere bottoms).

## Induction equation

$$\partial_t\mathbf{B} = \nabla\times(\mathbf{v}\times\mathbf{B}) + \nabla\times\!\left[\eta_A\frac{(\mathbf{J}\times\mathbf{B})\times\mathbf{B}}{|\mathbf{B}|^2}\right]. \quad (\text{C2-induction})$$

**Non-linear in $\mathbf{B}$** — key difference from Ohmic.

## Ambipolar heating

$$\boxed{Q_{\mathrm{amb}} = \eta_A |\mathbf{J}_\perp|^2 = \eta_A\frac{|\mathbf{J}|^2|\mathbf{B}|^2 - (\mathbf{J}\cdot\mathbf{B})^2}{|\mathbf{B}|^2} \ge 0.} \quad (\text{C2-Q})$$

## CFL

$$\Delta t \le \frac{(\Delta x)^2}{2\eta_A |\mathbf{B}|^2/\rho}\ \mathrm{(per direction)}. \quad (\text{C2-CFL})$$

Scales with $|\mathbf{B}|^2$ — tight CFL in strong-field regions.

## ✅ Verification

`tests/test_mhd_ambipolar_bmin.cu` — $B_x$ uniform, sinusoidal $B_y$
with $\eta_A = $ const. Lock decay rate against analytical; also
verify that when $B_y \parallel B_x$ (i.e., $\mathbf{J}\parallel\mathbf{B}$),
$Q_{\mathrm{amb}} = 0$ and $B_y$ does not decay. Catches any
accidental scalar-diffusion substitution for the tensor form.

# C3. Total-energy equation with non-ideal MHD

> **sympy script:** `scripts/c3_resistive_energy.py`
> **verified:** ideal Poynting flux identity; $Q_{\mathrm{Ohm}} = \eta_O|J|^2$;
> $Q_{\mathrm{amb}} = \eta_A|J_\perp|^2$ (via Lagrange's identity).
> **code checkpoints:** `athena_mhd_kernels.cu::d_total_energy_flux`.

## Total-energy equation

With both Ohmic + ambipolar, total energy is still conserved; only
the Poynting flux gets a non-ideal addition:

$$\boxed{\partial_t E + \nabla\cdot\!\left[(E+p^\star)\mathbf{v} - \mathbf{B}(\mathbf{v}\cdot\mathbf{B}) + \mathbf{E}_{\mathrm{ni}}\times\mathbf{B}\right] = 0,} \quad (\text{C3-energy})$$

with $\mathbf{E}_{\mathrm{ni}} = \eta_O\mathbf{J} + \eta_A(\mathbf{J}\times\mathbf{B})\times\mathbf{B}/|\mathbf{B}|^2$.

## Internal-energy source

The non-ideal dissipation becomes internal-energy heating:

$$\boxed{\partial_t(\rho e) + \nabla\cdot(\rho e\,\mathbf{v}) = -p\nabla\cdot\mathbf{v} + Q_{\mathrm{Ohm}} + Q_{\mathrm{amb}},} \quad (\text{C3-internal})$$

$$Q_{\mathrm{Ohm}} = \eta_O|\mathbf{J}|^2,\qquad
Q_{\mathrm{amb}} = \eta_A|\mathbf{J}_\perp|^2
= \eta_A\frac{|\mathbf{J}|^2|\mathbf{B}|^2 - (\mathbf{J}\cdot\mathbf{B})^2}{|\mathbf{B}|^2}.$$

**Sympy verified** the Lagrange identity $|\mathbf{J}\times\mathbf{B}|^2
= |\mathbf{J}|^2|\mathbf{B}|^2 - (\mathbf{J}\cdot\mathbf{B})^2$ which
maps between the two Q_amb forms.

## Sign convention

$Q = -\mathbf{E}_{\mathrm{ni}}\cdot\mathbf{J}$ is the rate at which EM
energy is dissipated **into the fluid's internal energy** — positive
for dissipative $\mathbf{E}_{\mathrm{ni}}$. Both Ohmic and ambipolar
give $Q > 0$.

## Kernel structure

- **Total energy**: evolved via conservative flux (C3-energy).
  Non-ideal Poynting flux $\mathbf{E}_{\mathrm{ni}}\times\mathbf{B}$
  added to the hydro energy flux.
- **No direct $Q$ kernel needed**: internal-energy source is
  implicitly accounted for by the flux divergence of the non-ideal
  Poynting term. Explicit computation of $Q$ only needed for
  diagnostics.

## ✅ Verification

`tests/test_mhd_ohmic_energy_conservation.cu` — uniform $\mathbf{B}_0$
with sinusoidal $B_y$ perturbation + $\eta_O = $ const. Measure total
energy drift over 100 decay timescales. Lock $|\Delta E/E_0| < 10^{-8}$ —
**any leak means the non-ideal Poynting flux term is missing from
the energy flux**.

# C4. Saha closure for $\eta_O$ and $\eta_A$

> **sympy script:** `scripts/c4_saha_ionization_closure.py`
> **verified:** Saha low-T limit $x_e \to 0$; high-T limit $x_e \to 1$.
> **code checkpoints:** `athena_mhd_kernels.cu::d_saha_eta`,
> `tests/test_mhd_saha_table.cu`.

## Saha equation (pure hydrogen, LTE)

$$\boxed{\frac{x_e^2}{1-x_e} = \frac{1}{n_H}\!\left(\frac{m_e k_B T}{2\pi\hbar^2}\right)^{3/2}\!e^{-\chi_H/k_B T},} \quad (\text{C4-Saha})$$

with $\chi_H = 13.6$ eV and $x_e = n_e/(n_e + n_H)$. **Sympy verified**
$x_e \to 0$ as $T \to 0$ and $x_e \to 1$ as $T \to \infty$.

## Diffusivity closures (Draine 1983 / Choi+09)

**Ohmic** (electron-neutral collisions):

$$\eta_O \approx 234\,\sqrt{T/10^4\,\mathrm{K}}\,x_e^{-1}\ \mathrm{cm^2/s}. \quad (\text{C4-etaO})$$

**Ambipolar** (ion-neutral drift):

$$\eta_A = \frac{|\mathbf{B}|^2}{\rho_i\rho_n\gamma_{in}},\qquad
\gamma_{in} \approx 3.5\times 10^{13}\ \mathrm{cm^3/g/s}. \quad (\text{C4-etaA})$$

These are the forms used in Suzuki+25 (RGB winds) and Matsuoka+24
(solar chromosphere).

## Weakly-ionised limit ($x_e \ll 1$)

$$\eta_O \sim \frac{234\sqrt{T/10^4}}{x_e},\qquad
\eta_A \sim \frac{|\mathbf{B}|^2}{x_e\rho^2\gamma_{in}}.$$

**Both diverge** as $x_e \to 0$ — non-ideal effects are maximal at
the chromosphere base where most hydrogen is neutral. This is why
the RGB winds in Suzuki+25 get their 15× $\dot M$ suppression from
ambipolar: it shuts down high-frequency Alfvén wave transmission
at the chromospheric floor.

## Magnetic Reynolds numbers

$$R_m^{\mathrm{Ohm}} = \frac{L v_A}{\eta_O},\qquad
R_m^{\mathrm{amb}} = \frac{L v_A\rho}{\eta_A|\mathbf{B}|^2}.$$

**Ideal MHD requires $R_m \gg 1$.** Matsuoka+24 reports $R_m \sim 1-10$
at $r \approx 1000$ km in the solar chromosphere — non-ideal MHD
kicks in below this altitude.

## Kernel implementation

Pre-tabulate $(\eta_O, \eta_A)$ as functions of $(\rho, T, |\mathbf{B}|)$
on a 3D table to avoid evaluating the Saha equation every timestep.
Suzuki+25 uses a 128×128×64 log-spaced table; cost is negligible
once built.

## ✅ Verification

`tests/test_mhd_saha_table.cu` — verify tabulated $(\eta_O, \eta_A)$
match Draine 1983 Table 2 to $<1\%$ at five reference $(T, \rho)$
pairs: $(10^4, 10^{-9})$, $(6000, 10^{-7})$, $(4500, 10^{-8})$,
$(3000, 10^{-11})$, $(10^5, 10^{-12})$.

# C5. Sub-grid turbulent heating closure (Suzuki-Inutsuka 2005)

> **sympy script:** `scripts/c5_suzuki_turbulent_heating.py`
> **verified:** positivity $\varepsilon_\mathrm{turb} > 0$; pure-outward
> limit $|z^-|\to 0$ gives no cascade
> ($\varepsilon_\mathrm{turb}^\mathrm{SY}\to 0$); timescale bound
> $\Delta t \le \lambda_\mathrm{cor}/(2 c_d|\delta v_\perp|)$.
> **code checkpoints:**
> `athena_mhd_kernels.cu::d_turbulent_heating_source`,
> `tests/test_athena_mhd_turbulent_heating_positivity.cu`.

## Why this closure exists

The 1D super-radial flux-tube code (§B1) cannot resolve the 3D Alfvén
wave turbulence that transfers outward-going wave energy to heat in
the corona. Suzuki-Inutsuka 2005 (SI05) added a phenomenological
sub-grid heating term that dissipates transverse wave energy at a
prescribed cascade rate. This is the crucial ingredient without
which the 1D wind models severely **under-estimate** coronal
temperature.

## Suzuki-Inutsuka 2005 form

$$\boxed{\varepsilon_{\mathrm{turb}}^\mathrm{SI} = c_d\,\rho\,\frac{|\delta v_\perp|^3}{\lambda_{\mathrm{cor}}},\quad c_d \approx 0.1,\quad \lambda_\mathrm{cor}\sim 10^7\,\mathrm{cm}.}$$

Dimensions: $[\rho][\delta v]^3/[\lambda] = \mathrm{erg/cm^3/s}$ ✓.

## Shoda-Yokoyama 2016 (Elsässer) form

$$\varepsilon_\mathrm{turb}^\mathrm{SY} = \frac{\rho}{2\lambda_\mathrm{cor}}\left(|\mathbf{z}^+|^2|\mathbf{z}^-| + |\mathbf{z}^-|^2|\mathbf{z}^+|\right).$$

This form makes the **non-linear coupling** between counter-
propagating modes explicit. Limit $|z^-|\to 0$ gives
$\varepsilon_\mathrm{turb}^\mathrm{SY} \to 0$ (sympy-verified): no
cascade without cross-polarisation — an essential physical
consistency check.

## Energy-equation coupling

$$\partial_t E + \nabla\!\cdot\!\left[(E + P^\star)\mathbf{v} - \mathbf{B}(\mathbf{B}\!\cdot\!\mathbf{v})\right] = \varepsilon_\mathrm{turb}\ \ge 0.$$

**Positivity.** Sympy-verified $\varepsilon_\mathrm{turb} > 0$ for
positive $\rho, \delta v_\perp, \lambda_\mathrm{cor}, c_d$. The
kernel must not inadvertently create a negative-definite heating
source — any code review must check this.

## Timescale bound (explicit-source CFL)

Per-step heating must not exceed the available wave KE:

$$\varepsilon_\mathrm{turb}\cdot\Delta t \le \tfrac{1}{2}\rho|\delta v_\perp|^2\ \Longleftrightarrow\ \Delta t \le \frac{\lambda_\mathrm{cor}}{2\,c_d\,|\delta v_\perp|}.$$

In practice, for solar wind parameters
($|\delta v_\perp|\sim 30\,\mathrm{km/s}$, $\lambda_\mathrm{cor} \sim 10^8\,\mathrm{cm}$),
this bound is well above the hyperbolic CFL; no concern.

## Shimizu+22 scaling

$$\lambda_\mathrm{cor}(r) = \lambda_0 \sqrt{A(r)/A(R_*)},\quad \lambda_0 \approx 10^7\,\mathrm{cm}$$

(i.e., correlation length scales with the tube area because energy-
containing eddies fill the tube cross-section). Replication of
Shimizu+22 figures requires this scaling **and** $c_d = 0.1$ exactly
— the paper's calibration.

## Extraction of $\delta v_\perp$

$\delta v_\perp$ is defined in the **Lagrangian / comoving frame**:

$$|\delta v_\perp|^2 \equiv \langle (v_y - \bar{v}_y)^2 + (v_z - \bar{v}_z)^2\rangle,$$

with $\bar{v}$ a running time-average. In a 1D tube code with no
transverse degree of freedom, this can instead be derived from the
**Alfvén wave amplitude**

$$|\delta v_\perp|^2 = |z^+|^2 + |z^-|^2 - 2\mathbf{z}^+\!\cdot\!\mathbf{z}^-$$

using the Elsässer variables tracked directly from §B2.

## ✅ Verification checkpoints

- `tests/test_athena_mhd_turbulent_heating_positivity.cu` — random
  initial $(\rho, \delta v, \lambda)$ states, assert
  $\varepsilon_\mathrm{turb} \ge 0$ across 100 samples; no NaN.
- `tests/test_athena_mhd_suzuki_wind.cu` — full replication of
  SI05 Fig. 2 mass-loss rate $\dot M \sim 2\times 10^{-14}\,M_\odot/\mathrm{yr}$
  within ±10% (calibration tolerance per paper's own sensitivity to
  $c_d$). Requires well-balanced MHSE (§B4).

# D1. Ideal MHD in cylindrical $(R, \phi, z)$ coordinates

> **sympy script:** `scripts/d1_cylindrical_mhd.py`
> **verified:** displays cylindrical divergence, curl, and axisymmetric
> reductions. No new nontrivial identities beyond Part A; this section
> is the reference for D2 and D3.
> **code checkpoints:** `athena_mhd_cylindrical_solver.cu` (future).

## Cylindrical operators

$$\nabla\cdot\mathbf{v} = \frac{1}{R}\partial_R(R v_R) + \frac{1}{R}\partial_\phi v_\phi + \partial_z v_z.$$

$$(\nabla\times\mathbf{B})_R = \frac{1}{R}\partial_\phi B_z - \partial_z B_\phi,\
(\nabla\times\mathbf{B})_\phi = \partial_z B_R - \partial_R B_z,\
(\nabla\times\mathbf{B})_z = \frac{1}{R}\!\left[\partial_R(R B_\phi) - \partial_\phi B_R\right].$$

## Radial momentum (axisymmetric)

$$\partial_t(\rho v_R) + \frac{1}{R}\partial_R(R\rho v_R^2) + \partial_z(\rho v_R v_z)
= \boxed{\frac{\rho v_\phi^2}{R}} - \partial_R p - \rho\partial_R\Phi_{\mathrm{grav}}. \quad (\text{D1-mom-R})$$

The $\rho v_\phi^2/R$ term is the **Christoffel-symbol / centrifugal
source**. It is what keeps a rotating disk in equilibrium against gravity.
In a kernel, it is the most common source of subtle bugs — easy to
drop when porting Cartesian kernels.

## Axisymmetric radial-flux conservation

$\partial_\phi = \partial_z = 0$ with $\mathbf{B} = B_R \hat{R}$
forces $\nabla\cdot\mathbf{B} = (1/R)\partial_R(R B_R) = 0$, so
$R\cdot B_R(R) = \text{const}$ — the cylindrical analog of the
spherical $r^2 B_r$ conservation.

## ✅ Verification

`tests/test_mhd_cyl_centrifugal.cu` — rotating equilibrium disk
with $v_\phi = \sqrt{GM/R}$. Lock $\max|v_R|/c_s < 10^{-4}$ over
10 rotations. Failure means the centrifugal source term is missing
or mis-signed.

# D2. Shearing-sheet approximation and shearing-periodic BC

> **sympy script:** `scripts/d2_shearing_sheet_bc.py`
> **verified:** shearing-sheet background is steady state (Coriolis +
> tidal cancel on $v_\phi^{\mathrm{bg}}$); shear wrap-around time
> $t_{\mathrm{shear}} = L_y/(q\Omega_0 L_x)$; effective potential
> is anti-confining.
> **code checkpoints:** `athena_mhd_shearingbox_bc.cu`,
> `tests/test_mhd_shearingbox_static.cu`.

## Hill expansion

Local Cartesian patch centred at $(R_0, \Omega_0)$, co-rotating with
the disk. Linearised rotational velocity:

$$\boxed{v_\phi^{\mathrm{bg}}(x) = -q\,\Omega_0\,x,\qquad q \equiv -\frac{d\ln\Omega}{d\ln R}}. \quad (\text{D2-shear})$$

$q = 3/2$ for Keplerian disks, $q = 2$ for rigid rotation.

## Frame-rotation sources

Coriolis:
$$\mathbf{F}_{\mathrm{Cor}} = -2\,\Omega_0\hat{z}\times\mathbf{u} = 2\Omega_0(u_y\hat{x} - u_x\hat{y}).$$

Tidal (Hill):
$$\mathbf{F}_{\mathrm{tidal}} = 2q\Omega_0^2\,x\,\hat{x} = -\nabla\Phi_{\mathrm{eff}},\qquad \Phi_{\mathrm{eff}} = -q\Omega_0^2 x^2.$$

**Sympy verified** that on the background $\mathbf{u}_{\mathrm{bg}} = (0, -q\Omega_0 x, 0)$,
$F_{\mathrm{Cor},x} + F_{\mathrm{tidal},x} = 0$. The shearing-sheet
background is an exact steady state of the force-balance.

## Shearing-periodic BC

$$\boxed{\mathrm{field}(x = L_x, y, z, t) = \mathrm{field}(x = 0, y - q\Omega_0 L_x t, z, t).} \quad (\text{D2-BC})$$

The $y$-shift $\Delta y(t) = q\Omega_0 L_x t$ grows linearly in time.
After $t_{\mathrm{shear}} = L_y/(q\Omega_0 L_x)$ the shift wraps to
$L_y$ and the BC becomes pure periodic again.

Under periodic-$y$ convention, the BC is always well-defined: take
$\Delta y\ \mathrm{mod}\ L_y$. **Sympy verified** $t_{\mathrm{shear}}$.

## Effective potential is anti-confining

$$\frac{d^2\Phi_{\mathrm{eff}}}{dx^2} = -2q\Omega_0^2 < 0,$$

so the tidal potential alone is unstable — **the Coriolis force is
what stabilises the shearing sheet**. Sympy-verified; documented to
flag that dropping the Coriolis kernel would immediately disperse
the disk.

## ✅ Verification

`tests/test_mhd_shearingbox_static.cu` — initial condition
$\mathbf{u} = (0, -q\Omega_0 x, 0)$, purely hydro, no perturbation.
Lock max perturbation energy $<10^{-6}\times E_0$ over $10/\Omega_0$.
Any drift means Coriolis or tidal coupling is mis-implemented.

# D3. MRI stress decomposition $\alpha_{\mathrm{SS}} = \alpha_R + \alpha_M$

> **sympy script:** `scripts/d3_mri_stress_decomposition.py`
> **verified:** Maxwell stress antisymmetric under $B_R\to-B_R$,
> invariant under $(B_R,B_\phi)\to(-B_R,-B_\phi)$; Reynolds stress
> antisymmetric under $v_R\to-v_R$; MRI dispersion vanishes at
> Balbus-Hawley extremum ($k_*^2 v_A^2 = 15/16\,\Omega^2$,
> $\gamma_{\max} = 3/4\,\Omega$).
> **code checkpoints:** `tests/test_mhd_mri_growth_rate.cu`,
> `tests/test_mhd_mri_stress_decomp.cu`.

## Shakura-Sunyaev stress

$$T_{R\phi} = \underbrace{\rho v_R\,\delta v_\phi}_{\text{Reynolds}} + \underbrace{(-B_R B_\phi)}_{\text{Maxwell}}. \quad (\text{D3-stress})$$

$$\alpha_{\mathrm{SS}} = \frac{\langle T_{R\phi}\rangle}{\langle p\rangle} = \alpha_R + \alpha_M.$$

## Suzuki 2023 sign-quadrant decomposition

Maxwell stress is:
- **antisymmetric** under single flip $B_R \to -B_R$: $M_{R\phi}\to -M_{R\phi}$.
- **invariant** under simultaneous flip $(B_R, B_\phi) \to (-B_R, -B_\phi)$.

This pair of symmetries is what makes the Suzuki 2305.12112 "triangle
diagnostic" meaningful: the four sign quadrants $(\pm B_R, \pm B_\phi)$
carry physically distinct stress contributions. The cylindrical vs
Cartesian sign-flip result in that paper (headline $[\phi \Rightarrow_R \phi]$
arrow $+6.42$ vs $-1.05$) is a direct consequence of this symmetry
structure.

## Balbus-Hawley MRI dispersion

For vertical $\mathbf{B}_0 = B_0\hat{z}$ and Keplerian $q = 3/2$
($\kappa^2 = \Omega^2$):

$$\omega^4 - \omega^2(\kappa^2 + 2k^2v_A^2) + k^2v_A^2(k^2v_A^2 + \kappa^2 - 4\Omega^2) = 0. \quad (\text{D3-BH-disp})$$

Instability requires $k^2 v_A^2 < 3\Omega^2$. The most-unstable mode:

$$\boxed{k_*^2 v_A^2 = \tfrac{15}{16}\Omega^2,\qquad \gamma_{\max} = \tfrac{3}{4}\Omega.} \quad (\text{D3-MRI-max})$$

**Sympy verified** by direct substitution into the dispersion
polynomial. This gives the MRI timescale $\sim 4/(3\Omega) \approx 1/3$
of an orbital period.

## Parseval for Maxwell

$$\frac{1}{V}\int B_R(\mathbf{x}) B_\phi(\mathbf{x})\,d^3x = \sum_{\mathbf{k}} \mathrm{Re}[\hat{B}_R(\mathbf{k})\hat{B}_\phi^*(\mathbf{k})]$$

Allows FFT-based computation of $\alpha_M$ at arbitrary resolution.

## ✅ Verification

**`tests/test_mhd_mri_growth_rate.cu`** — seed a linear $B_y$
perturbation at $k_*$ with amplitude $10^{-6}B_0$; lock measured
growth rate matches $(3/4)\Omega$ to $<1\%$ over $5/\Omega$.

**`tests/test_mhd_mri_stress_decomp.cu`** — in nonlinear saturated
state, lock $\alpha_M/\alpha_R \sim 3-5$ (Stone+96, Suzuki+23
Cartesian baseline). Suzuki+23 reports $\alpha_M \approx 0.072$,
$\alpha_R \approx 0.016$ for Cartesian Keplerian baseline.

