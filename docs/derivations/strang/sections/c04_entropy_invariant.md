# C4. Smooth-flow entropy invariant

> **sympy script:** `scripts/c04_entropy_invariant.py`
> **generated LaTeX:** `output/c04_entropy_invariant.latex.tex`
> **verified:**
> - 1 chain-rule identity ($D_t s = (1/P) D_t P - (\gamma / \rho) D_t \rho$)
> - 1 entropy invariant ($D_t s = 0$ on smooth Euler)
> - 1 entropy-function chain identity
> - 1 entropy-function invariant ($D_t(P\rho^{-\gamma}) = 0$)
> - 1 entropy-function identity ($\log K = s$, via `expand_log(force=True)`)
>
> **code checkpoints:**
> - `src/gpu/explicit/strang_solver.cu :: write_vtk` (entropy computed as a post-processing diagnostic; the kernel does not enforce $D_t s = 0$, it emerges from §E1's smooth- convergence measurement)

On smooth solutions of the Euler equations (no shocks, no
discontinuities), the specific entropy

$$s \;=\; \log(P \rho^{-\gamma}) \;=\; \log P \;-\; \gamma \log \rho \quad (\text{C4-entropy})$$

is a **Lagrangian invariant**:

$$\bigl(\partial_t + \mathbf{v}\!\cdot\!\nabla\bigr)\,s \;=\; D_t s \;=\; 0. \quad (\text{C4-invariant})$$

Each fluid parcel maintains its initial entropy forever, provided
the flow stays smooth. This is the material-derivative form of
the adiabatic first law — no heat exchange, no dissipation, so
entropy is constant per parcel.

At **shocks** the entropy strictly **increases** (§A5), so this
invariant fails by design — shock-capturing schemes must
introduce numerical dissipation equivalent to entropy production
per the Lax condition. The identity in this section is the
smooth-flow limit.

## Derivation

From mass conservation $D_t \rho = -\rho\,\nabla\!\cdot\!\mathbf{v}$
and the adiabatic pressure equation $D_t P = -\gamma P\,
\nabla\!\cdot\!\mathbf{v}$ (which follows from energy + mass via
substitution), the chain rule gives

$$D_t s \;=\; \frac{D_t P}{P} \;-\; \gamma\,\frac{D_t \rho}{\rho} \;=\; -\gamma\,\nabla\!\cdot\!\mathbf{v} \;+\; \gamma\,\nabla\!\cdot\!\mathbf{v} \;=\; 0. \quad (\text{C4-deriv})$$

**Strong-form verification.** sympy substitutes the PDE-derived
forms of $\partial_t P, \partial_t \rho$ into the chain-rule
expansion of $D_t s$ and simplifies to zero. The equivalent form

$$D_t(P \rho^{-\gamma}) \;=\; 0 \quad (\text{C4-K})$$

is also verified independently via chain-rule + PDE substitution.
The two forms are equivalent via $s = \log(P\rho^{-\gamma})$; the
second is the **entropy function** used in the HSE background
definition of §B2 (where $K = P/\rho^\gamma$ is an explicit
constant).

## Gravity is irrelevant

The material derivative $D_t s$ depends only on the local thermodynamic
state, not on the gravity source. Gravity enters the momentum and
energy equations as non-zero RHS, but in the entropy derivation:

- $D_t P$ is derived from mass + (energy equation minus momentum $\cdot
  \mathbf{v}$ equation), and the gravity terms (which appear in both
  energy and momentum with consistent signs, §C1) **cancel** in the
  difference.
- $D_t \rho$ is derived from mass, which has no gravity source.

So $D_t s = 0$ holds **even with gravity** on smooth flows. This is
the statement that gravity is a conservative body force that does
not dissipate energy into entropy — Potential energy exchanges
with kinetic energy, with internal energy (and entropy) unchanged.

## HSE consistency

On pure HSE ($\mathbf{v} = \mathbf{0}$), the material derivative
$D_t s = (\partial_t + \mathbf{v}\cdot\nabla)s = \partial_t s = 0$
because $s = \log K$ is constant in $y$ (isentropic stratification,
§B2). The entropy invariant is satisfied trivially — there is no
advection to advect anything, and the Eulerian $\partial_t s$ is
zero because $s$ is constant.

## Shock contrast with §A5

The §A5 entropy inequality states

$$\partial_t \eta + \partial_i(\eta u_i) \le 0, \qquad \eta = -\rho s,$$

with strict equality only on smooth states. At shocks
$\partial_t \eta + \partial_i(\eta u_i) < 0$, which by standard
manipulation implies $D_t s > 0$ (entropy of a parcel crossing a
shock increases). The smooth-flow identity $D_t s = 0$ is the
equality case of the §A5 inequality — they are consistent, not
competing, statements.

## Numerical consequences

On the Strang kernel:

- **Smooth-flow tests** (§D1 entropy wave, §D5 bubble) should
  preserve $s$ to truncation-error precision. The kernel's
  numerical dissipation (from HLLC and MUSCL) introduces an
  $O(\Delta x^2 \nabla^2 s)$ entropy production on smooth flow,
  which converges to zero as $\Delta x \to 0$ (measured in §E1).
- **Shock tests** (§D3 Sod, §D4 Woodward-Colella blast) should
  show $s$ strictly increasing across shocks. The rate is set by
  HLLC's numerical entropy production (which matches the physical
  rate to leading order under sufficient resolution, per §A5 Lax
  condition).
- **HSE tests** (§D6 hse-zero-lock) preserve $s = \log K$ to
  machine precision, since the perturbation storage keeps
  $\delta\rho = \delta P = 0$ and the full state matches §B2's
  $K$-constant background.

## Diagnostic in the kernel

The kernel does **not** evolve $s$ explicitly — it is a derived
quantity. The VTK diagnostic output includes $s$ as a computed
field (line 946-947 of strang_solver.cu after cons2prim). Tests
in §D and §E series read this field and check the invariance
property at the numerical level.

## Verification checkpoints

1. **Entropy preservation on entropy wave.** At the end of one
   entropy-wave period, the entropy distribution $s(x, y)$ must
   return to its initial value to $O(\Delta x^2)$. Test:
   `test_strang_convergence.cu` §C4-entropy-wave.

2. **HSE entropy uniformity.** On pure HSE, $s(x, y)$ is
   identically $\log K$ to machine precision at every cell. Test:
   `test_strang_init.cu` §C4-hse-entropy.

3. **Shock entropy jump sign.** Across a shock (§D3 Sod), the
   measured post-shock entropy is strictly greater than the
   pre-shock entropy (agreeing with Rankine-Hugoniot prediction
   from §A5). Test: `test_strang_sod.cu` §C4-shock-entropy.

4. **Smooth bubble entropy tracking.** The bubble IC (§D5) has
   an initial entropy anomaly; the Lagrangian entropy (following
   fluid parcels) stays within $O(\Delta x^2)$ of the initial
   value throughout the bubble's rise. Test:
   `test_strang_step.cu` §C4-bubble-entropy.

Failure of (1) is an HLLC-dissipation issue (possibly with
`use_lm_fix` interference). Failure of (2) is an HSE-preservation
bug (usually §B3 or §C1). Failure of (3) would indicate a broken
Lax condition — the HLLC flux is producing negative entropy at
shocks, a sign of scheme instability or wrong wave-speed
estimates. Failure of (4) is an entropy dissipation issue — the
kernel is leaking entropy out of the bubble faster than
truncation error predicts.
