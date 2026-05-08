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
