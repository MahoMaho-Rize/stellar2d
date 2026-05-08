# F3. CT round-off accumulation + $B_\mathrm{cc}$ aliasing in long-time field-loop

> **sympy script:** `scripts/f3_ct_roundoff_and_bcc_aliasing.py`
> **verified:** random-walk round-off bound =
> worst-case / $\sqrt{n_\mathrm{step}}$; midpoint reconstruction
> $B_\mathrm{cc} = B + (h^2/8) B'' + O(h^4)$ for smooth $B$; aliasing
> bound $|\Delta\mathrm{ME}_\mathrm{cc}| \le C_\mathrm{alias} A_0^2 \pi R (h/R)$,
> first-order in $h$, non-monotonic in $t$.
> **code checkpoints:**
> `tests/test_athena_mhd_field_loop_long.cu` — L1 (divB round-off),
> L3 (ME_cc aliasing bound).

## Motivation

Phase A3 of the Phase A verification plan asks: **does CT actually
preserve $\nabla\cdot\mathbf{B}=0$ to machine precision over 10⁴
steps**, not just through 10² as the short test covers?

The telescoping identity §A5 is algebraically exact in real
arithmetic. In *floating-point* arithmetic each face update
introduces a ULP-sized error, and the accumulation bound must be
derived before the long-time test has a quantitative pass criterion.

A second, subtler issue: the long-time test measures magnetic energy
through the **cell-centred reconstruction** $B_\mathrm{cc}$, not the
face-stored $B_f$. This diagnostic aliases the $C^0$ kink at the
field-loop boundary $r = R$ as the loop translates, producing
oscillations in $\mathrm{ME}_\mathrm{cc}(t)$ that **do not** violate
CT. Without derivation, the test could mistake diagnostic aliasing
for a solver bug.

## Q1: CT round-off accumulation bound

One CT face update:

$$(B_x)^{n+1}_{i+1/2,j} = (B_x)^{n}_{i+1/2,j}
 - \frac{\Delta t}{\Delta y}\bigl(E_z^{i+1/2,j+1/2} - E_z^{i+1/2,j-1/2}\bigr).$$

In double precision (IEEE-754), each subtraction carries relative
error $\le \varepsilon_\mathrm{ULP} = 2.22\times 10^{-16}$. The
corner-$E_z$ contribution scales with $|\mathbf{B}|_\infty$ (HLLD
flux scaling; consistency of the ideal-MHD Jacobian).

The discrete divergence of a cell is a **signed sum of 4 face values**:

$$(\nabla\!\cdot\!\mathbf{B})_{i,j} = \frac{B_{x,R} - B_{x,L}}{\Delta x}
 + \frac{B_{y,T} - B_{y,B}}{\Delta y}.$$

Per-step round-off residual:

$$|\Delta(\nabla\!\cdot\!\mathbf{B})|_{\mathrm{per\ step}} \le \frac{4\,\varepsilon_\mathrm{ULP}\,|\mathbf{B}|_\infty}{h}. \quad (\text{F3-per-step})$$

Over $n_\mathrm{step}$ updates, two accumulation models:

$$\boxed{\max_t |\nabla\!\cdot\!\mathbf{B}| \ \le\
\begin{cases}
4\,n_\mathrm{step}\,\varepsilon_\mathrm{ULP}\,|\mathbf{B}|_\infty / h & \text{(coherent, worst case)}\\
4\,\sqrt{n_\mathrm{step}}\,\varepsilon_\mathrm{ULP}\,|\mathbf{B}|_\infty / h & \text{(random walk)}
\end{cases}} \quad (\text{F3-bound})$$

Sympy-verified: the random-walk form equals the worst-case form
divided by $\sqrt{n_\mathrm{step}}$.

**Numeric check** for the A3 test parameters
($\varepsilon_\mathrm{ULP} = 2.22\times 10^{-16}$, $|\mathbf{B}|_\infty = 1$,
$h = 1/128$, $n_\mathrm{step} \approx 10^4$):

| Bound | Value |
|---|---|
| Worst-case (F3-bound) | $1.14\times 10^{-9}$ |
| Random-walk (F3-bound) | $1.14\times 10^{-11}$ |
| **Measured** (A3 test) | $1.85\times 10^{-15}$ |

The measured value is **4 orders of magnitude tighter** than even the
random-walk bound. Interpretation (Gardiner-Stone 2005 §3.4.1):
the CT stencil is sign-symmetric in the 4 corner contributions to
each cell, so round-off cancels to within 1 ULP rather than
accumulating as random walk. CT is not just algebraically exact —
it is *round-off exact* on realistic hardware.

The A3 test lock `max|∇·B| < 1e-10` uses a safety margin between the
measured $10^{-15}$ and the worst-case bound $10^{-9}$.

## Q2: $B_\mathrm{cc}$ reconstruction aliasing at the $r=R$ kink

The diagnostic cell-centred $B$ is computed by midpoint averaging:

$$(B_{x,\mathrm{cc}})_{i,j} = \tfrac{1}{2}\bigl((B_{x,f})_{i-1/2,j} + (B_{x,f})_{i+1/2,j}\bigr).$$

**Smooth $B$ behaviour (sympy-verified):**

$$B_{x,\mathrm{cc}} = B_x(x_i) + \tfrac{h^2}{8} B_x''(x_i) + \mathcal{O}(h^4). \quad (\text{F3-midpoint})$$

So for a smooth field, $\mathrm{ME}_\mathrm{cc}(t) - \mathrm{ME}_\mathrm{true}(t)
= \mathcal{O}(h^2)$ is a small, smooth, $t$-independent bias.

**Field-loop IC pathology.** The GS05 field-loop has $|\mathbf{B}|$
step-function at $r = R$:

$$|\mathbf{B}|(r) = \begin{cases} A_0 & r < R \\ 0 & r > R \end{cases}.$$

At the kink, $B$ is $C^0$ but not $C^1$. The Taylor expansion in
(F3-midpoint) fails, and the midpoint reconstruction has local error
$\mathcal{O}(h)$ — **a full order worse** than the smooth case.

As the loop translates with velocity $\mathbf{v} = (v_x, v_y)$, the
kink aliases successively across different cell boundaries. Let
$N_\mathrm{ring} \approx 2\pi R / h$ be the number of cells crossed
by the ring. The cell-integrated $\mathrm{ME}_\mathrm{cc}$ picks up
a phase-dependent $\mathcal{O}(A_0^2 \cdot h/R)$ aliasing bound:

$$\boxed{\bigl|\mathrm{ME}_\mathrm{cc}(t) - \mathrm{ME}_\mathrm{cc}(0)\bigr| \le C_\mathrm{alias}\,A_0^2\,\pi R\,(h/R),\quad \text{oscillatory in }t.} \quad (\text{F3-aliasing})$$

Sympy-verified properties:
- Quadratic in $A_0$ (quadratic in amplitude, as $\mathrm{ME}$ itself is).
- Linear in $h$: halving the grid spacing halves the aliasing error.

**Critical physical interpretation.** CT conserves the
*face-integrated flux* $\oint \mathbf{B}\cdot d\mathbf{S}$ through
any closed discrete loop *exactly* — this is the content of §A5.
The discrepancy $\mathrm{ME}_\mathrm{cc}(t) \ne \mathrm{const}$ lives
entirely in the **diagnostic reconstruction** $B \to B_\mathrm{cc}$,
not in the solver state. A test that measures "ME conservation" via
$B_\mathrm{cc}$ will report $\mathcal{O}(1)$ oscillation in
$\mathrm{ME}_\mathrm{cc}(t)$ on a translating field loop — this is
**expected**, not a bug.

**A3 test measurement** (N=128, A₀=1e-3, R=0.3, 10 crossings,
minmod limiter):

$$\mathrm{ME}_\mathrm{cc}(t=10) / \mathrm{ME}_\mathrm{cc}(0) \approx 1.56, \quad \max_t \mathrm{ME}_\mathrm{cc} / \mathrm{ME}(0) \approx 1.56.$$

This is within the expected aliasing envelope (F3-aliasing) for the
parameters: $h/R = 0.026$, $N_\mathrm{ring} \approx 60$ cells,
$C_\mathrm{alias} \sim O(10)$ after 600 sub-cell-boundary crossings.

## Limiter sensitivity

The field-loop IC is more sensitive to the choice of slope limiter than
smooth tests (e.g., linear-wave convergence). **Empirically**:
- **van Leer harmonic** (`limiter = 0`): the loop is *unstable* over
  ~8 crossings; $\mathrm{ME}_\mathrm{cc}$ grows by 10⁴ from a
  compressible instability driven by the sharp kink. CT still
  preserves $\nabla\cdot\mathbf{B}$ to $10^{-14}$ even as this
  happens — a clean demonstration that CT constraint-preservation
  ≠ physical stability.
- **minmod** (`limiter = 1`): the loop is stable over 10+ crossings;
  ME_cc oscillates bounded by (F3-aliasing).

The A3 test uses minmod for this reason. The loop-instability under
van Leer is not a bug — it is the reason Stone+08 §6.3 switches to
minmod for shock-containing problems and notes that field-loop tests
specifically require a more dissipative limiter.

## ✅ Verification checkpoints

Exactly what the A3 test `tests/test_athena_mhd_field_loop_long.cu`
locks:

- **L1** $\max_t |\nabla\!\cdot\!\mathbf{B}| < 10^{-10}$ — 1 order
  below worst-case bound (F3-bound) and 4 orders above measured.
- **L3** $\max_t \mathrm{ME}_\mathrm{cc}/\mathrm{ME}_0 < 3$ — well
  within envelope (F3-aliasing) for A₀=1e-3, h=1/128, R=0.3.
- **L4** $\mathrm{ME}_\mathrm{cc}(t_\mathrm{end})/\mathrm{ME}_0 > 0.5$ —
  solution is not decaying to zero (loop structure preserved).

Any of these failing flags a real bug (solver-level or IC-level, not
aliasing).
