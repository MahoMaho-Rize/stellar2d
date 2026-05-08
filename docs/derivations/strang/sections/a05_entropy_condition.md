# A5. Entropy condition (thermodynamic and mathematical)

> **sympy script:** `scripts/a05_entropy_condition.py`
> **generated LaTeX:** `output/a05_entropy_condition.latex.tex`
> **verified:**
> - $D_t s_{\mathrm{alg}} = 0$, $D_t s_{\mathrm{therm}} = 0$, $\partial_t \eta + \partial_i(u_i\eta) = 0$
> - 1 numerical-consistency check (Hessian PSD on 80 random admissible states)
> - 1 documented **[WEAK]** inequality (Lax entropy condition across a shock)
>
> **code checkpoints:**
> - `src/gpu/explicit/strang_solver.cu :: write_vtk` (emits $s_{\mathrm{alg}}$ as diagnostic)
> - §C4 (kernel-level entropy invariance on smooth tests)

The Euler system is not uniquely solvable in the presence of
discontinuities — a single initial value problem can admit multiple
weak solutions. The **entropy condition** singles out the unique
physically admissible solution. This section derives the smooth-
flow version in strong form and documents the discontinuous version
as a labelled weak form.

## Algebraic entropy invariant

Define

$$s_{\mathrm{alg}} \;\equiv\; \frac{p}{\rho^{\gamma}}. \quad (\text{A5-s-alg})$$

This is the entropy **up to a monotone transformation**. The
thermodynamic specific entropy is

$$s_{\mathrm{therm}} \;=\; \frac{1}{\gamma - 1}\,\ln\!\left(\frac{p}{\rho^{\gamma}}\right) + \text{const}, \quad (\text{A5-s-therm})$$

a monotone function of $s_{\mathrm{alg}}$. The kernel emits
$s_{\mathrm{alg}}$ as the VTK diagnostic rather than
$s_{\mathrm{therm}}$ to avoid the $\ln$ near $\rho\to 0$ boundary.

## Smooth-flow invariance — strong form

On smooth flow (no shocks), the entropy is a Lagrangian invariant:

$$D_t s_{\mathrm{alg}} \;=\; 0, \qquad D_t \;\equiv\; \partial_t + u\,\partial_x + v\,\partial_y. \quad (\text{A5-Dt-s})$$

**Strong-form derivation.** Starting from mass conservation
(§A1-mass) and the internal-energy material-derivative equation
(§A1-material-e), one can derive the auxiliary pressure rule

$$D_t p \;=\; -\gamma\,p\,\nabla\!\cdot\!\mathbf{v} \quad (\text{A5-Dt-p})$$

which combined with $D_t \rho = -\rho\,\nabla\!\cdot\!\mathbf{v}$
(from mass) gives

$$D_t\!\left(\frac{p}{\rho^{\gamma}}\right) \;=\; \frac{D_t p}{\rho^{\gamma}} - \gamma\,\frac{p}{\rho^{\gamma+1}}\,D_t \rho \;=\; \frac{-\gamma p\,\nabla\!\cdot\!\mathbf{v} + \gamma\,p\,\nabla\!\cdot\!\mathbf{v}}{\rho^{\gamma}} \;=\; 0.$$

**sympy verification.** The script computes $D_t s_{\mathrm{alg}}$
symbolically and substitutes the two smooth-flow rules
(A5-Dt-p) and the mass rule $D_t \rho = -\rho\,\nabla\cdot\mathbf{v}$;
the result reduces to $0$ under `sp.simplify`. The thermodynamic
form $s_{\mathrm{therm}}$ inherits invariance by chain rule.

## Mathematical entropy (Harten 1983)

Define the convex scalar entropy

$$\eta(\mathbf{U}) \;=\; -\rho\,\ln\!\left(\frac{p}{\rho^{\gamma}}\right), \qquad q_i \;=\; u_i\,\eta. \quad (\text{A5-eta})$$

On smooth flow,

$$\partial_t \eta + \partial_x q_x + \partial_y q_y \;=\; 0. \quad (\text{A5-eta-conservation})$$

**Strong-form verification.** sympy substitutes the mass and
pressure-smooth-flow rules into $\partial_t\eta + \nabla\cdot(\mathbf{v}\eta)$;
the result is $0$ under `sp.simplify`.

**Convexity of $\eta(\mathbf{U})$ — [WEAK] numerical check.**
The Hessian $H_{ij} = \partial^2 \eta / \partial U_i \partial U_j$
on the admissible region $\{\rho > 0,\ p > 0\}$ is positive
definite. sympy does not admit a symbolic proof of positive-
definiteness (the analytical eigenvalues of a $4\times 4$ symbolic
matrix with logarithmic entries are intractable), so this is the
first Rule-4 numerical fallback of the book:

> _Numerical consistency: 80 random admissible states sampled with
> $\rho \in [0.1, 10]$, $|u|, |v| \in [0, 2]$, $p \in [0.1, 10]$,
> $\gamma \in \{1.4, 5/3, 2\}$. At each state, $H$ is evaluated and
> its 4 eigenvalues are computed via `numpy.linalg.eigvalsh`. All
> eigenvalues are verified positive (within $10^{-10}$ of zero).
> The ensemble passes uniformly._

Convexity is a cited result (Harten 1983 §3); the numerical check
here confirms the claim holds on the stellar2d admissible region
for the $\gamma$ values used in the code.

## Lax entropy condition across a shock — **[WEAK]**

For a $k$-shock with speed $\sigma$ separating left and right
states, the **Lax entropy condition** is

$$\sigma\,[\eta]_{\mathrm{L}}^{\mathrm{R}} \;-\; [q_n]_{\mathrm{L}}^{\mathrm{R}} \;\geq\; 0, \qquad (\text{A5-lax-inequality, [WEAK]})$$

where $[f]_L^R = f(\mathbf{U}_R) - f(\mathbf{U}_L)$ and $q_n$ is the
entropy flux along the shock normal.

**Why this is weak form.** At a shock, $\eta$ and $q_n$ are
distributional — $\partial_t \eta$ contains a delta function
supported on the shock trajectory. The inequality above is obtained
by integrating the strong-form equation against a smooth non-
negative test function $\varphi(x,t)$ over a neighbourhood of the
shock, invoking the Rankine-Hugoniot jump conditions on the flux,
and exploiting $\varphi \ge 0$ to flip the equality to an
inequality. This cannot be written as a pointwise algebraic
identity and is outside the sympy strong-form protocol.

**Rule-4 justification.** Reason: shocks are distributional; $D_t
\eta$ at the shock does not admit a strong-form pointwise
expression. Numerical consistency check for a specific shock-tube
IC is performed in §D3 (Sod) and §D4 (Woodward-Colella) where the
jump $[\eta]$ and $[q_n]$ across the solver's reconstructed shock
are computed and the inequality verified numerically on the
converged solution.

## Verification checkpoints

The kernel does not enforce §A5 explicitly; the Riemann solver (§A7)
embeds the entropy condition through its choice of wave speeds
(§A9, Davis bounds) and its contact-speed formula (§A8). §A5's
invariants are verified at the discrete level by:

1. **Smooth-flow entropy drift.** On a periodic, shock-free IC
   (e.g., D1 entropy wave or D2 acoustic linwave at $A = 10^{-6}$),
   the solver's $\max_t |s_{\mathrm{alg}}(x,t) - s_{\mathrm{alg}}(x,0)|$
   measured at every cell must remain below the modified-equation
   bound $O(\Delta x^2 + \Delta t^2)$. Test: to be added as
   `tests/test_strang_entropy_invariance.cu` during the Part-E
   wire-up.

2. **Shock entropy production.** For Sod's tube (§D3), the total
   entropy
   $\int \eta\,dV$ must be a non-increasing function of time. (The
   flux $q_n$ vanishes on periodic boundaries or cancels on
   reflective boundaries.) Test: `tests/test_strang_sod.cu` §E5
   entropy block.

3. **Harten convexity round-trip.** Starting from a smooth state,
   perturbed by $\varepsilon \xi$ with $\xi$ random and small, the
   change in $\eta$ must be non-negative to $O(\varepsilon^2)$ for
   every random $\xi$ — the discrete analogue of the Hessian
   positivity. Test: added to `test_strang_unit.cu` as a property-
   based check.

Failures on (1) indicate either a bug in the Riemann solver
(entropy fix missing in a transonic rarefaction) or a problem in
the slope limiter (over-damping triggering excess dissipation —
still entropy-conserving but characteristically wrong). Failures
on (2) indicate a positivity bug or a wave-speed estimate that
doesn't bound the shock, either of which is a deep Riemann-solver
problem.
