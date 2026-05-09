# A1. Compressible Euler equations (strong-form conservation)

> **sympy script:** `scripts/a01_euler_equations.py`
> **generated LaTeX:** `output/a01_euler_equations.latex.tex`
> **verified:**
> - energy-flux factorisation $x$: $(E + p)u = \rho(h + \tfrac{1}{2}|\mathbf{v}|^{2})u$
> - energy-flux factorisation $y$: $(E + p)v = \rho(h + \tfrac{1}{2}|\mathbf{v}|^{2})v$
> - x-momentum material-derivative form: $\rho D_t u = -\partial_x p$
> - y-momentum material-derivative form: $\rho D_t v = -\partial_y p$
> - internal-energy material derivative: $\rho D_t e_{\mathrm{int}} = -p\,\nabla\!\cdot\!\mathbf{v}$
>
> **code checkpoints:**
> - `src/gpu/explicit/strang_device.cuh::d_euler_flux_x`
> - `src/gpu/explicit/strang_device.cuh::d_euler_flux_y`
> - `src/gpu/explicit/strang_device.cuh::d_cons2prim`

## Starting assumptions

| Label | Assumption |
|---|---|
| A1a | Compressible neutral gas; no viscosity, no thermal conduction, no radiation, no species diffusion. |
| A1b | Ideal EOS, $p = (\gamma - 1)\,\rho\,e_{\mathrm{int}}$, with $\gamma$ constant. |
| A1c | Solutions are smooth in $(x, y, t)$; identities in this section are pointwise strong-form equalities. Discontinuous solutions are admitted only via the entropy condition derived in §A5 and the Rankine-Hugoniot relations exploited by §A7. |
| A1d | Gravity, if present, enters as a source term in §C1 — it is **not** part of the flux $\mathbf{F}$ here. |

The two-dimensional state vector used by the kernel is
$\mathbf{U} = (\rho,\ \rho u,\ \rho v,\ E)^{\!\top}$, with total energy
per unit volume $E = p/(\gamma - 1) + \tfrac{1}{2}\rho(u^{2} + v^{2})$.

## Mass conservation

$$\partial_t \rho + \nabla\!\cdot\!(\rho\mathbf{v}) = 0. \quad (\text{A1-mass})$$

This is taken as a postulate, not a derived identity. Every identity
below is verified by sympy **while treating the mass residual as a
free symbolic object** — the identity can then be closed under the
strong-form assumption (A1-mass).

## Momentum conservation (divergence form)

$$
\partial_t(\rho u) + \partial_x(\rho u^{2} + p) + \partial_y(\rho u v) = 0, \quad (\text{A1-mom-x})
$$
$$
\partial_t(\rho v) + \partial_x(\rho u v)\;\; + \partial_y(\rho v^{2} + p) = 0. \quad (\text{A1-mom-y})
$$

These two equations define rows 1 and 2 of the flux vector
$\mathbf{F}_x, \mathbf{F}_y$ used by every Godunov kernel in the
codebase. The Riemann-solver derivation in §A7 onwards operates on
precisely these flux components.

## Total-energy conservation

$$E \;=\; \frac{p}{\gamma - 1} \;+\; \tfrac{1}{2}\rho\,(u^{2} + v^{2}). \quad (\text{A1-E})$$

$$\partial_t E + \partial_x\!\left[(E + p)\,u\right] + \partial_y\!\left[(E + p)\,v\right] = 0. \quad (\text{A1-energy})$$

The combination $(E + p)$ rather than $E$ alone in the flux is the
source of the *enthalpy* form that every Riemann solver exploits
(§A7) and that the HLLC contact-wave algebra of §A8 relies on.

## Enthalpy form of the energy flux

Define the specific enthalpy

$$h \;\equiv\; e_{\mathrm{int}} + \frac{p}{\rho} = \frac{\gamma\,p}{(\gamma - 1)\,\rho}.$$

Then $(E + p)\,v_i = \rho\,\bigl(h + \tfrac{1}{2}|\mathbf{v}|^{2}\bigr)\,v_i$
identically.

**sympy verification (strong form).** For both directions $i\in\{x, y\}$,

$$(E + p)\,v_i \;-\; \rho\bigl(h + \tfrac{1}{2}|\mathbf{v}|^{2}\bigr)v_i \;\xrightarrow{\text{sp.simplify}}\; 0$$

at the symbolic level, without any ODE integration.

## Momentum in material-derivative form

Expanding the conservative momentum equation with $\rho u = \rho\cdot u$
and using (A1-mass), one obtains

$$\rho\,D_t u = -\partial_x p, \qquad
\rho\,D_t v = -\partial_y p,
\qquad D_t \equiv \partial_t + u\,\partial_x + v\,\partial_y. \quad (\text{A1-material-mom})$$

**sympy verification (strong form).** Letting
$R_{\mathrm{mom}, x}$ and $R_{\mathrm{mass}}$ denote the left-hand
side residuals of (A1-mom-x) and (A1-mass),

$$R_{\mathrm{mom}, x} \,-\, u\,R_{\mathrm{mass}} \;-\; \bigl(\rho\,D_t u + \partial_x p\bigr) \;=\; 0,$$

verified in sympy without invoking either residual equal to zero.
This is important: (A1-material-mom) is not merely a consequence of
(A1-mass) + (A1-mom-x); it is an *algebraic identity* between the
two LHS residuals, and it remains valid even off-solution.

An analogous identity holds for the $y$-component.

## Internal-energy equation (first law on a trajectory)

$$\rho\,D_t e_{\mathrm{int}} = -\,p\,\nabla\!\cdot\!\mathbf{v}, \qquad
e_{\mathrm{int}} = \frac{p}{(\gamma - 1)\,\rho}. \quad (\text{A1-material-e})$$

**sympy verification (strong form).** Denote the energy residual
$R_E$. The algebraic identity

$$R_E \;-\; u\,R_{\mathrm{mom}, x} \;-\; v\,R_{\mathrm{mom}, y} \;-\; \bigl(e_{\mathrm{int}} - \tfrac{1}{2}|\mathbf{v}|^{2}\bigr) R_{\mathrm{mass}} \;-\; \bigl(\rho\,D_t e_{\mathrm{int}} + p\,\nabla\!\cdot\!\mathbf{v}\bigr) \;=\; 0$$

holds by `sp.simplify` alone. This is the reduction that underlies
the Riemann-solver energy balance: the interior of every Godunov
cell obeys (A1-material-e) as long as (A1-mass), (A1-mom-x),
(A1-mom-y), and (A1-energy) are simultaneously satisfied.

The product $p\,\nabla\cdot\mathbf{v}$ is the reversible
compression/expansion work of thermodynamics. In §A5 (entropy
condition) and in §C4 (entropy invariant) this identity is invoked
to show that $s = \log(p/\rho^{\gamma})$ is a Lagrangian invariant
on smooth flow.

## Compact conservative system

$$\boxed{\partial_t \mathbf{U} + \partial_x \mathbf{F}_x(\mathbf{U}) + \partial_y \mathbf{F}_y(\mathbf{U}) = \mathbf{0}, \qquad \mathbf{U} = (\rho,\ \rho u,\ \rho v,\ E)^{\!\top}.}$$

$$\mathbf{F}_x = \begin{pmatrix}\rho u\\ \rho u^{2} + p\\ \rho u v\\ (E + p)\,u\end{pmatrix}, \qquad
\mathbf{F}_y = \begin{pmatrix}\rho v\\ \rho u v\\ \rho v^{2} + p\\ (E + p)\,v\end{pmatrix}.$$

This is the strong-form system that every discretisation in the rest
of this book approximates. The kernel-level counterparts live in
`strang_device.cuh::d_euler_flux_x` and `d_euler_flux_y`; they must
return these four components verbatim, with the $(E + p)$ factoring
derived from §A1 above.

## Verification checkpoints

The kernel is required to satisfy three strong-form invariants
derivable from §A1, checked by one-cell test cases in
`tests/test_strang_muscl.cu` and `tests/test_strang_hllc.cu`:

1. **Zero-velocity flux.** For any state with $u = v = 0$, the only
   non-zero flux component is $F_x[1] = F_y[2] = p$. All other
   entries vanish identically. Regression signal: bitwise zero.

2. **Energy flux factorisation.** For any state with $\rho > 0$ and
   $p > 0$, the kernel's computed energy flux equals
   $\rho(h + \tfrac{1}{2}|\mathbf{v}|^{2})\,v_i$ to ULP-precision.

3. **Material-derivative equivalence.** Apply the kernel's flux
   Jacobian to a smooth test state and compare the resulting
   semi-discrete $D_t u$ against a directly computed
   $-\partial_x p / \rho$: agreement to $\mathcal{O}(\Delta x^2)$,
   i.e., MUSCL-consistent truncation error.

Failure of any of (1)–(3) flags inconsistency between the kernel
implementation and the continuum equations of this section; such a
failure must be resolved before anything in §A2 or later is relied
upon.
