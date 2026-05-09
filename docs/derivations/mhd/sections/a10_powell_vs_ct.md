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
