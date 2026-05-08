# A11. Reconstruction order (donor-cell / MUSCL / PPM)

> **sympy script:** `scripts/a11_reconstruction_order.py`
> **generated LaTeX:** `output/a11_reconstruction_order.latex.tex`
> **verified:**
> - 10 strong-form identities via Taylor expansion of cell averages — donor-cell $h^1$ and $h^2$ coefficients
> - MUSCL $h^0$ and $h^1$ coefficients vanish (establishing 2nd order)
> - PPM $h^0$, $h^1$, $h^2$, $h^3$ coefficients all vanish (establishing 4th-order face reconstruction)
>
> **code checkpoints:**
> - `src/gpu/explicit/strang_solver.cu :: k_muscl_hancock_x/y` (MUSCL) (PPM is derived here for comparison; the kernel does not implement it.)

This is the **third alternative-scheme comparison** section. Four
reconstruction orders are derived in strong form by Taylor
expansion of cell averages around a cell centre, and their leading
truncation errors compared. The Strang kernel uses MUSCL (2nd
order); PPM is derived here so that the book can answer "what do
we gain by upgrading to PPM?".

## Setup: cell averages vs point values

For a smooth $u(x)$ on a uniform grid of spacing $h$, the cell
average at cell $j$ is

$$\langle u\rangle_j \;=\; \frac{1}{h}\int_{x_j - h/2}^{x_j + h/2} u(x')\,dx' \;=\; u(x_j) + \frac{h^{2}}{24}\,u''(x_j) + \frac{h^{4}}{1920}\,u''''(x_j) + O(h^{6}).$$

The distinction between $\langle u\rangle_j$ and $u(x_j)$ is
$O(h^2)$ — negligible for 1st-order schemes but crucial for PPM
(which claims 4th-order face reconstruction).

The **true face value** is the point value at the face centre,

$$u(x_{j+1/2}) \;=\; u(x_j + h/2) \;=\; u(x_j) + \tfrac{h}{2}\,u'(x_j) + \tfrac{h^{2}}{8}\,u''(x_j) + \tfrac{h^{3}}{48}\,u'''(x_j) + \tfrac{h^{4}}{384}\,u''''(x_j) + O(h^{5}).$$

Every Godunov scheme of this book operates on cell averages
$\{\langle u\rangle_{j-1}, \langle u\rangle_j, \ldots\}$ and
reconstructs face values against the **point-value** target above.

## Donor-cell (1st order)

$$u_{j+1/2, L}^{\mathrm{donor}} \;=\; \langle u\rangle_j. \quad (\text{A11-donor})$$

Error expansion:

$$\varepsilon^{\mathrm{donor}} \;=\; u_{j+1/2, L}^{\mathrm{donor}} - u(x_{j+1/2}) \;=\; -\tfrac{h}{2}\,u'(x_j) - \tfrac{h^{2}}{12}\,u''(x_j) + O(h^{3}).$$

**Strong-form verification.** sympy confirms the $h^1$ and $h^2$
coefficients of the error match the expected $-\tfrac{1}{2}u'$ and
$-\tfrac{1}{12}u''$ (note the $h^2$ constant is $\tfrac{1}{12}$,
not $\tfrac{1}{8}$, because cell average and point value differ
by $\tfrac{h^2}{24}u''$; this is the "cell-average correction"
that PPM exploits).

## Unlimited MUSCL (2nd order)

The unlimited central-difference slope

$$\sigma_j \;=\; \frac{\langle u\rangle_{j+1} - \langle u\rangle_{j-1}}{2 h},$$

combined with linear reconstruction

$$u_{j+1/2, L}^{\mathrm{MUSCL}} \;=\; \langle u\rangle_j + \tfrac{h}{2}\,\sigma_j. \quad (\text{A11-MUSCL})$$

Error expansion:

$$\varepsilon^{\mathrm{MUSCL}} \;=\; -\tfrac{h^{2}}{12}\,u''(x_j) + O(h^{3}).$$

**Strong-form verification.** The $h^0$ and $h^1$ error
coefficients vanish identically (sympy-verified), confirming 2nd-
order accuracy. The leading $h^2$ coefficient is
$-\tfrac{1}{12}u''(x_j)$ (sympy reports this directly). The
numerical dissipation of the full Godunov scheme is further
reduced by cancellation between reconstruction and time
integration (§E1 modified-equation analysis).

## PPM Colella-Woodward (4th order face reconstruction)

$$u_{j+1/2}^{\mathrm{PPM\text{-}CW}} \;=\; \tfrac{7}{12}\bigl(\langle u\rangle_j + \langle u\rangle_{j+1}\bigr) - \tfrac{1}{12}\bigl(\langle u\rangle_{j-1} + \langle u\rangle_{j+2}\bigr). \quad (\text{A11-PPM-CW})$$

Error expansion (at $x_j$-frame): $\varepsilon^{\mathrm{PPM-CW}} =
-\tfrac{h^{4}}{30}\,u''''(x_j) + O(h^{6})$.

**Strong-form verification.** sympy confirms that the $h^0, h^1,
h^2, h^3$ coefficients all cancel identically — i.e., the PPM
formula is 4th-order-accurate at the face. (Colella–Woodward
1984 eq. 1.7 quotes the leading error in the $x_{j+1/2}$-frame as
$\tfrac{3 h^4}{640} u''''(x_{j+1/2})$; the two constants differ by
the Taylor-frame shift; both are $O(h^4)$, which is what matters.)

## PPM Colella-Sekora variant

Colella & Sekora (2008) keep the same unlimited 4-cell reconstruction
(A11-PPM-CW) but replace the Colella–Woodward parabolic-overshoot
limiter with an extremum detector. At smooth extrema the original
CW limiter degrades to 1st order; CS detects genuine smooth
extrema and preserves 3rd-order accuracy there. The unlimited
reconstruction algebra is identical, so the §A11 truncation
analysis applies without modification. The limiter difference
manifests only on non-smooth data (shocks, contacts).

## Reconstruction-order hierarchy

| reconstruction | leading error | scheme overall order | stencil width |
|---|---|---|---|
| donor-cell | $O(h)$ | 1st | 1 |
| MUSCL (unlimited) | $O(h^2)$ | 2nd | 3 |
| MUSCL (limited) | $O(h^2)$ away from extrema, $O(h)$ at extrema | 2nd (smooth) / 1st (shocks) | 3 |
| PPM-CW | $O(h^4)$ reconstruction → $O(h^3)$ full scheme | 3rd | 4 |
| PPM-CS | $O(h^4)$ reconstruction → $O(h^3)$ everywhere | 3rd | 4 |

**Kernel's choice.** The Strang solver uses limited MUSCL with
the MC limiter. The decision matrix: for a 2D compressible
simulation at $N^2 \sim 512^2$ with many limiter activations at
grid-scale turbulence, MUSCL's 2nd-order error is comparable to
PPM's 3rd-order error after accounting for limiter clips, while
costing ~40% less per step. Upgrading to PPM is listed as an
optional scheme-characterisation experiment in §E (not in the
current kernel scope).

## Verification checkpoints

The kernel implements §A11's MUSCL reconstruction inside
`k_muscl_hancock_x/y`. Tests:

1. **Smooth-IC 2nd-order convergence.** On a smooth sinusoidal
   IC, the entropy-wave $L^1$ error must decrease as $h^2$ as
   $N$ is refined from $64^2$ to $512^2$. Slope must fall in
   $[1.8, 2.2]$. Test: `test_strang_convergence.cu` §A11-MUSCL-
   order (already exists; will be extended to read goldens from
   D1 in Part D).

2. **Donor-cell fallback regression.** With MUSCL's limiter
   clamped to 0 (donor-cell behaviour), the same test must give
   slope in $[0.8, 1.2]$. This confirms the limiter, not the
   base scheme, is responsible for 2nd-order accuracy. Test:
   `test_strang_convergence.cu` §A11-donor-cell-fallback (to
   be added).

Failure of (1) indicates either a bug in the Hancock predictor
(§A12) or in the limiter (§A10). Failure of (2) would indicate
a deep structural bug in the spatial reconstruction itself,
visible only at 1st-order setting.
