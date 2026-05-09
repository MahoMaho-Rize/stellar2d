# A14. Strang operator-chain self-adjointness

> **sympy script:** `scripts/a14_strang_operator_chain.py`
> **generated LaTeX:** `output/a14_strang_operator_chain.latex.tex`
> **verified:**
> - 1 half-flow reversal
> - 31 Strang-chain self-adjoint monomial identities (every monomial of degree $\le 4$ in $\mathcal{X}, \mathcal{Y}$ cancels identically in $L_\mathrm{fwd} L_\mathrm{bwd}$)
> - 2 Lie-splitting non-self- adjoint leading residuals ($\pm\Delta t^2$ on $[\mathcal{X}, \mathcal{Y}]$)
>
> **code checkpoints:**
> - `src/gpu/explicit/strang_solver.cu :: StrangSolver::step` (the four-factor symmetric chain is structurally self-adjoint)

This section proves the **structural reason** the Strang splitting
of §A13 is 2nd-order: it is **self-adjoint** (time-reversible).
Self-adjoint integrators have leading error at **even** powers of
$\Delta t$ only; Strang is 2nd-order because the $\Delta t^1$
leading error, which would exist for an asymmetric 1st-order
scheme like Lie, is structurally forbidden by the time-reversal
symmetry of the triple product.

## Self-adjointness definition

A numerical integrator $L(\Delta t)$ is **self-adjoint** (or
time-reversible) if

$$L(\Delta t)\,L(-\Delta t) \;=\; I \qquad \forall\,\Delta t. \quad (\text{A14-self-adjoint-def})$$

Equivalently, evolving forward by $\Delta t$ and then backward by
$\Delta t$ returns to the identity exactly, for any $\Delta t$.
This property is **structural**: either it holds for all $\Delta t$
(and then the leading error is at an even power of $\Delta t$), or
it fails at some $\Delta t^k$ for odd $k$, making the integrator
$k$-th-order rather than $(k+1)$-th.

## Strang splitting is self-adjoint

$$\bigl[e^{(\Delta t/2)\,\mathcal{X}}\,e^{\Delta t\,\mathcal{Y}}\,e^{(\Delta t/2)\,\mathcal{X}}\bigr] \;\cdot\; \bigl[e^{(-\Delta t/2)\,\mathcal{X}}\,e^{-\Delta t\,\mathcal{Y}}\,e^{(-\Delta t/2)\,\mathcal{X}}\bigr] \;=\; I. \quad (\text{A14-Strang-self-adjoint})$$

**Strong-form verification.** sympy expands both factors as non-
commutative polynomials in $\mathcal{X}, \mathcal{Y}$ to total
degree 4, multiplies them, and checks **every monomial** up to
degree 4. Every non-identity monomial coefficient is zero (31
scalar identities); only the empty monomial retains coefficient 1.
The proof extends trivially to higher orders by induction on the
time-reversal symmetry of the product.

## Consequence: even-order accuracy

Any self-adjoint integrator has a Taylor series in $\Delta t$ with
**only even-power** error terms. Proof sketch:

- Denote the local error $\mathbf{E}(\Delta t) = L(\Delta t) -
  e^{\Delta t\,\mathcal{L}}$.
- Self-adjointness of $L$ and $e^{\Delta t\,\mathcal{L}}$
  (the exact flow) implies $\mathbf{E}(\Delta t) =
  -\mathbf{E}(-\Delta t) \cdot e^{2\Delta t\,\mathcal{L}}$ or,
  after Taylor expansion, $\mathbf{E}(\Delta t)$ contains only
  even powers of $\Delta t$.

Strang is therefore $O(\Delta t^2)$-error = 2nd-order globally.
This is a structural result, independent of the commutator
identities of §A13.

## Lie splitting fails self-adjointness

$$\bigl[e^{\Delta t\,\mathcal{Y}}\,e^{\Delta t\,\mathcal{X}}\bigr] \;\cdot\; \bigl[e^{-\Delta t\,\mathcal{Y}}\,e^{-\Delta t\,\mathcal{X}}\bigr] \;=\; I \;-\; \Delta t^{2}\,[\mathcal{X}, \mathcal{Y}] + O(\Delta t^{3}). \quad (\text{A14-Lie-not-self-adjoint})$$

**Strong-form verification.** sympy reports the non-zero residuals
at degree 2: coefficient of $\mathcal{X}\mathcal{Y}$ is $-\Delta
t^2$, coefficient of $\mathcal{Y}\mathcal{X}$ is $+\Delta t^2$.
Their combination is exactly $-\Delta t^2[\mathcal{X},
\mathcal{Y}]$.

This explains why Lie is 1st-order: the leading error is at
$\Delta t^2$ (in the time-reversal composition), which under the
standard $\mathbf{E}(\Delta t)$ decomposition gives a $\Delta t$-
order term in the forward-only error (the split derivative of a
$\Delta t^2$-order residual).

## Kernel realisation

The Strang kernel applies

$$\mathcal{X}(\Delta t/2)\;\mathcal{Y}(\Delta t/2)\;\mathcal{Y}(\Delta t/2)\;\mathcal{X}(\Delta t/2),$$

which is identical to the three-factor Strang chain by §A13's
semigroup composition. Time-reversing the kernel chain factor-by-
factor,

$$\mathcal{X}(-\Delta t/2)\;\mathcal{Y}(-\Delta t/2)\;\mathcal{Y}(-\Delta t/2)\;\mathcal{X}(-\Delta t/2),$$

and composing the two gives identity to all orders. The kernel is
**structurally self-adjoint**; no quality-of-implementation
issue can change this.

The only thing the kernel can do wrong here is an **asymmetric**
modification: for example, if one applied a gravity source term as
a third operator `Z(Dt)` without a symmetric wrapper, the split
would become Lie-type and immediately degrade to 1st-order. §C1
and §E4 derive why the gravity source in the Strang kernel is
**absorbed inside** $\mathcal{Y}$ (as a momentum + energy source)
rather than inserted as a separate $\mathcal{Z}$ operator, so the
full step chain remains symmetric.

## Verification checkpoints

The self-adjointness of the Strang kernel is a structural
consequence of the code's topology (the order of sweep calls).
Its violation would appear only as:

1. **Asymmetric split.** If someone adds a new physics operator
   to `StrangSolver::step` without a symmetric wrapper, the split
   becomes Lie-like. Reference test: `test_strang_step.cu` §A14
   asymmetry-smoke — run $X(\Delta t/2) \cdot Y(\Delta t/2) \cdot
   Y(\Delta t/2) \cdot X(\Delta t/2)$ and $X(-\Delta t/2) \cdot
   Y(-\Delta t/2) \cdot Y(-\Delta t/2) \cdot X(-\Delta t/2)$ on
   the same IC and verify the composition returns the original
   state to ULP precision. Any drift indicates a bug in the
   kernel's operator order or a non-symmetric source insertion.

2. **Time-reversal smoke test.** Run the kernel forward for $N$
   steps at CFL $\sigma$, then run backward for $N$ steps at
   $-\sigma$. Returned state must equal the IC to round-off
   accumulation $O(\varepsilon_{\mathrm{mach}} N)$. Test:
   `test_strang_step.cu` §A14 time-reversal.

Failure of (1) is a structural bug (someone broke the split by
adding an asymmetric operator). Failure of (2) is either an
asymmetric split or an arithmetic bug in one of the half-sweeps
(most likely the CFL computation applying floor at different
sign conventions).
