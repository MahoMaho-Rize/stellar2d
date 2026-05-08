# A13. Time integrator family (Strang / Lie / VL2 / RK2)

> **sympy script:** `scripts/a13_time_integrator_family.py`
> **generated LaTeX:** `output/a13_time_integrator_family.latex.tex`
> **verifies:** 10 strong-form identities on the linearised operator
> expansion — 2 Lie-splitting leading-commutator identities; 4
> Strang-splitting $\Delta t^2$ cancellation identities; 1 kernel-
> chain equivalence (all monomials match identically); 1 VL2
> leading-$\Delta t^3$ identity; plus the printed report of the
> 6 non-zero Strang $\Delta t^3$ residual coefficients
> **code checkpoints:**
> `src/gpu/explicit/strang_solver.cu :: StrangSolver::step`

This is the **fourth alternative-scheme comparison** of the book.
Four time-integrator templates are compared by Baker-Campbell-
Hausdorff (BCH) expansion on a linearised pair of non-commuting
operators $\mathcal{X}, \mathcal{Y}$. The Strang kernel uses
splitting; the comparison with the alternatives (Lie, VL2-unsplit,
RK2) quantifies the order gap and identifies the Strang scheme's
advantage.

## Operator-split abstraction

Write the 2D Euler update as

$$\partial_t \mathbf{U} \;=\; \mathcal{X}(\mathbf{U}) + \mathcal{Y}(\mathbf{U}), \qquad \mathcal{X} = -\partial_x \mathbf{F}_x, \quad \mathcal{Y} = -\partial_y \mathbf{F}_y. \quad (\text{A13-split-form})$$

Each of $\mathcal{X}, \mathcal{Y}$ is non-linear in $\mathbf{U}$,
but for order analysis we **linearise** them (treat as formal
non-commuting operators) and expand the exponential $e^{\Delta t
(\mathcal{X} + \mathcal{Y})}$ that represents the exact continuous-
time solution. Four integrator choices are then products of
one-operator exponentials to compare.

The non-linear order verification is deferred: for the non-linear
Euler system no closed-form BCH series exists. This section
delivers the strong-form proof on the linearised operator ring
(which is the tightest sympy can make the identity); the non-
linear analogue is verified numerically by the entropy-wave
convergence test of §E1.

## Lie splitting (1st order)

$$\mathbf{U}^{n+1} \;=\; e^{\Delta t\,\mathcal{Y}}\,e^{\Delta t\,\mathcal{X}}\,\mathbf{U}^{n}. \quad (\text{A13-Lie})$$

BCH expansion to $\Delta t^2$:

$$e^{\Delta t\,\mathcal{Y}}\,e^{\Delta t\,\mathcal{X}} \;-\; e^{\Delta t\,(\mathcal{X} + \mathcal{Y})} \;=\; -\,\frac{\Delta t^{2}}{2}\,[\mathcal{X},\,\mathcal{Y}] + O(\Delta t^{3}).$$

**Strong-form verification.** sympy expands both sides as non-
commutative polynomials in monomials of $\mathcal{X}, \mathcal{Y}$
up to total degree 3. The coefficient of $\mathcal{X}\mathcal{Y}$
in the difference is $-\Delta t^2/2$; the coefficient of $\mathcal
{Y}\mathcal{X}$ is $+\Delta t^2/2$. Their sum is the commutator
$-(\Delta t^2/2)[\mathcal{X}, \mathcal{Y}]$, verifying the
classical result.

## Strang splitting (2nd order)

$$\mathbf{U}^{n+1} \;=\; e^{(\Delta t/2)\,\mathcal{X}}\,e^{\Delta t\,\mathcal{Y}}\,e^{(\Delta t/2)\,\mathcal{X}}\,\mathbf{U}^{n}. \quad (\text{A13-Strang})$$

**Strong-form verification.** sympy expands the triple product up
to degree 3. All four $\Delta t^2$ monomials ($\mathcal{XX},
\mathcal{XY}, \mathcal{YX}, \mathcal{YY}$) have cancelling
coefficients in the residual — verified as four separate identities.

The first non-zero residuals appear at $\Delta t^3$; they are
iterated-commutator terms. sympy reports the six non-zero
coefficients:

| monomial | coefficient |
|---|---|
| $\mathcal{X}\mathcal{X}\mathcal{Y}$ | $-\Delta t^3 / 24$ |
| $\mathcal{X}\mathcal{Y}\mathcal{X}$ | $+\Delta t^3 / 12$ |
| $\mathcal{X}\mathcal{Y}\mathcal{Y}$ | $+\Delta t^3 / 12$ |
| $\mathcal{Y}\mathcal{X}\mathcal{X}$ | $-\Delta t^3 / 24$ |
| $\mathcal{Y}\mathcal{X}\mathcal{Y}$ | $-\Delta t^3 / 6$ |
| $\mathcal{Y}\mathcal{Y}\mathcal{X}$ | $+\Delta t^3 / 12$ |

These rearrange into linear combinations of the nested commutators
$[\mathcal{X}, [\mathcal{X}, \mathcal{Y}]]$ and $[[\mathcal{X},
\mathcal{Y}], \mathcal{Y}]$, which is the classical result for
Strang symmetric splitting (see, e.g., Sanz-Serna 1992).
Collectively they represent the leading $O(\Delta t^3)$ error
which vanishes as $\Delta t \to 0$, giving 2nd-order global
accuracy.

## Kernel operator chain

The Strang kernel's `step` function applies

$$\mathcal{X}(\Delta t/2)\;\mathcal{Y}(\Delta t/2)\;\mathcal{Y}(\Delta t/2)\;\mathcal{X}(\Delta t/2). \quad (\text{A13-kernel-chain})$$

By the semigroup property of the $\mathcal{Y}$-flow,
$\mathcal{Y}(\Delta t/2)\,\mathcal{Y}(\Delta t/2) = \mathcal{Y}(\Delta t)$,
so the kernel chain is **identically** the Strang split:

$$\mathcal{X}(\Delta t/2)\,\mathcal{Y}(\Delta t/2)\,\mathcal{Y}(\Delta t/2)\,\mathcal{X}(\Delta t/2) \;\equiv\; \mathcal{X}(\Delta t/2)\,\mathcal{Y}(\Delta t)\,\mathcal{X}(\Delta t/2).$$

**Strong-form verification.** sympy expands the four-factor
product and the three-factor Strang chain, subtracts, and
confirms every monomial coefficient up to degree 3 is zero. The
two forms are bit-identical at the operator level.

The kernel's choice of the four-factor form saves one ghost-cell
fill per full step: after the first $\mathcal{X}(\Delta t/2)$, the
ghost cells are refilled only for the next $\mathcal{Y}$ sweep;
the two half-$\mathcal{Y}$ sweeps share a single ghost refill.
This is an implementation optimisation, not an algorithmic change.

## Unsplit VL2 (Stone-Gardiner 2009, 2nd order)

The unsplit VL2 scheme — the basis of Athena++ — uses a single
unsplit predictor-corrector structure:

$$\mathbf{U}^{\star} \;=\; \mathbf{U}^{n} - \frac{\Delta t}{2}\,\mathcal{L}\,\mathbf{U}^{n}, \qquad \mathbf{U}^{n+1} \;=\; \mathbf{U}^{n} - \Delta t\,\mathcal{L}\,\mathbf{U}^{\star}, \quad (\text{A13-VL2})$$

with $\mathcal{L} = \mathcal{X} + \mathcal{Y}$ the full spatial
operator. Algebraically on linear $\mathcal{L}$:

$$\mathbf{U}^{n+1} \;=\; \bigl(I - \Delta t\,\mathcal{L} + \tfrac{\Delta t^{2}}{2}\,\mathcal{L}^{2}\bigr)\,\mathbf{U}^{n}.$$

Compared to $e^{-\Delta t\,\mathcal{L}} = I - \Delta t\,\mathcal{L}
+ \tfrac{\Delta t^2}{2}\,\mathcal{L}^2 - \tfrac{\Delta t^3}{6}\,
\mathcal{L}^3 + \ldots$,

$$\mathbf{U}^{n+1} \;-\; e^{-\Delta t\,\mathcal{L}}\,\mathbf{U}^{n} \;=\; -\,\frac{\Delta t^{3}}{6}\,\mathcal{L}^{3}\,\mathbf{U}^{n} \;+\; O(\Delta t^{4}).$$

**Strong-form verification.** sympy confirms the leading error is
$-(\Delta t^3/6)\,\mathcal{L}^3$ on the linearised operator ring,
so VL2 is also 2nd-order globally.

**VL2 vs. Strang trade-off.** Both integrators are $O(\Delta t^3)$
on the leading error; VL2 avoids the ghost-refill-between-sweeps
cost and therefore is faster in practice at the same accuracy.
Strang splitting is simpler and does not require building the
full 2D flux gradient in a single pass. The stellar2d kernel uses
Strang because it was implemented first; `athena_vl2_solver.cu`
later added VL2 as an Athena-parity alternative.

## RK2-MUSCL (Shu-Osher, 2nd order)

$$\mathbf{U}^{(1)} \;=\; \mathbf{U}^{n} + \Delta t\,\mathcal{L}(\mathbf{U}^{n}), \qquad \mathbf{U}^{n+1} \;=\; \tfrac{1}{2}\mathbf{U}^{n} + \tfrac{1}{2}\bigl(\mathbf{U}^{(1)} + \Delta t\,\mathcal{L}(\mathbf{U}^{(1)})\bigr).$$

On the linearised operator ring, this reduces algebraically to
the same $(I - \Delta t\,\mathcal{L} + \tfrac{\Delta t^2}{2}\,
\mathcal{L}^2)$ form as VL2, up to the specific choice of
intermediate stages. It is therefore also 2nd-order with the same
leading $\Delta t^3$ coefficient on the linearised operator. Not
used in the stellar2d kernel; listed here for completeness.

## Order comparison

| integrator | leading error | symmetric? |
|---|---|---|
| Lie | $-(\Delta t^2/2)\,[\mathcal{X}, \mathcal{Y}]$ | no |
| Strang | $\sim \Delta t^3\,[\mathcal{X}, [\mathcal{X}, \mathcal{Y}]] + \ldots$ | yes |
| VL2 | $-(\Delta t^3/6)\,\mathcal{L}^3$ | no (but operates on $\mathcal{X}+\mathcal{Y}$) |
| RK2-MUSCL | $-(\Delta t^3/6)\,\mathcal{L}^3$ (linearised) | no |

All three 2nd-order schemes are $\Delta t^3$-leading-error but with
different error structures. The sizes of the leading errors on a
given problem depend on the commutator structure of $\mathcal{X}$
and $\mathcal{Y}$, which for 2D Euler is non-trivial
(pressure gradient in $x$ couples to the $y$-momentum update,
and vice versa).

## Non-linear order: **[WEAK]** via §E1

The BCH analysis above is exact on linearised operators. For the
fully non-linear Euler system, the BCH series does not converge
in closed form (the commutators of non-linear PDE operators are
themselves non-linear PDE operators). We cannot verify 2nd-order
accuracy of the kernel on non-linear Euler symbolically. Per
Rule 4, this is a legitimate [WEAK] step for sympy — the physics
is real, the kernel behaves as expected, but the verification is
a numerical measurement of convergence rate rather than a
closed-form identity.

The non-linear verification is delivered in §E1 (entropy-wave
convergence order): we run the kernel on smooth entropy-wave IC
at $N = \{64, 128, 256, 512\}$, fit a log-log slope on $L^1$
error, and confirm the slope falls in $[1.8, 2.2]$ as predicted
by the modified-equation analysis of §E1.

## ✅ Verification checkpoint (to be wired)

The kernel does not expose $\mathcal{X}, \mathcal{Y}$ operators
directly; the A13 identities are verified at the scheme level
through §E1. However, one **mechanical** regression is possible:

1. **Symmetry of the Strang chain.** Starting from a symmetric IC
   $\mathbf{U}(x, y) = \mathbf{U}(-x, y)$, the Strang-kernel
   evolution preserves that symmetry to ULP precision — a
   consequence of the $X$-operator symmetry in the split plus the
   B4 periodic-x BC (§B4 to be derived). Violation indicates a
   bug in the ghost-cell fill OR a non-symmetric $\mathcal{X}$
   operator in the kernel. Test: `test_strang_reflection_symmetry.cu`
   §A13-time-symmetry.

Failure in (1) is a deep structural bug; more typically the A13
invariants are verified via §E1 (convergence rate).
