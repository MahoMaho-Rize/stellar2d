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
