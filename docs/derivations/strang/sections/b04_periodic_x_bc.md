# B4. Periodic-x boundary condition

> **sympy script:** `scripts/b04_periodic_x_bc.py`
> **generated LaTeX:** `output/b04_periodic_x_bc.latex.tex`
> **verifies:** 8 strong-form identities — 2 index-offset
> identities (left and right ghost offsets = $n_x$); 1 physical-
> distance identity ($x_{\mathrm{src}} - x_{\mathrm{ghost}} = L_x$);
> 1 periodic-manufactured-solution identity (sin wave with
> $k L_x = 2\pi m$); 4 flux-commutativity identities
> ($F_x(U_{\mathrm{ghost}})[i] = F_x(U_{\mathrm{phys}})[i]$,
> $i = 0..3$)
> **code checkpoints:**
> `src/gpu/explicit/strang_solver.cu :: k_ghost_x`
> (line 33, cell-data copy: `d_*[k_dst] = d_*[k_src]` where
> `ig_ghost = g` or `nx + g`)
> `src/gpu/explicit/strang_solver.cu :: k_ghost_face_x`
> (line 564, face-state copy: `d_wL[ig=ng-1]` from `d_wL[ig=ng+nx-1]`
> etc.)

Periodic-x is the simpler of the two BCs in the Strang kernel: the
solution is required to satisfy $\mathbf{U}(x + L_x, y, t) =
\mathbf{U}(x, y, t)$, and the ghost cells are filled by a direct
copy from the physical interior across the domain. No physics
transformation is applied (contrast with §B5's reflective BC).

## BC identity

$$\mathbf{U}(x + L_x, y, t) \;=\; \mathbf{U}(x, y, t) \quad \forall\,(x, y, t). \quad (\text{B4-periodic})$$

## Ghost-cell copy formulas

Using the kernel's physical index convention where $i_{\mathrm{phys}}
\in \{0, 1, \ldots, n_x - 1\}$ is the interior index (index-0 is the
leftmost physical cell) and ghost layers extend beyond:

$$\begin{aligned}\mathbf{U}_{\mathrm{ghost}}(i_{\mathrm{phys}} = -1 - g) \;&=\; \mathbf{U}(i_{\mathrm{phys}} = n_x - 1 - g), \\ \mathbf{U}_{\mathrm{ghost}}(i_{\mathrm{phys}} = n_x + g) \;&=\; \mathbf{U}(i_{\mathrm{phys}} = g),\end{aligned} \quad g \in \{0, \ldots, n_g - 1\}. \quad (\text{B4-ghost})$$

Equivalently, $i_{\mathrm{src}} - i_{\mathrm{ghost}} = n_x$ on the
left side and $i_{\mathrm{ghost}} - i_{\mathrm{src}} = n_x$ on the
right side. Both directions implement the wrap-around
$i \mapsto i + n_x$ or $i - n_x$.

## Consistency with the PDE

Taking a periodic-in-$x$ smooth solution
$\mathbf{U}(x, y, t) = \mathbf{U}(x + L_x, y, t)$ and substituting
$x_{\mathrm{ghost}} = i_{\mathrm{ghost}}\,\Delta x$,
$x_{\mathrm{src}} = (i_{\mathrm{ghost}} + n_x)\,\Delta x$, the
physical distance is $x_{\mathrm{src}} - x_{\mathrm{ghost}} = L_x$,
so $\mathbf{U}(x_{\mathrm{ghost}}) = \mathbf{U}(x_{\mathrm{src}})$
automatically. sympy verifies this for a sine-wave manufactured
solution with $k L_x = 2\pi m$ ($m$ integer); the more general case
(any periodic $\mathbf{U}$) follows from the fundamental periodic
hypothesis.

## Ghost-cell width

MUSCL-Hancock (§A12) reads a 3-point stencil $(i-1, i, i+1)$ in
each sweep to compute the MC slope (§A10). The predictor therefore
requires one layer of ghost. **Additionally**, the face-state
refill `k_ghost_face_x` (line 564) operates on the ghost face
**after** the predictor has written face states into
`d_wL, d_wR`, and the ghost faces are indexed at $i_g = n_g - 1$
and $i_g = n_g + n_x$ — i.e., the first and last **ghost** cell
positions. These positions hold cell data that feeds the next
sweep's boundary Riemann problem. The full round-trip requires
$n_g \ge 2$:

- Layer 1: feeds the MUSCL stencil at the boundary cell.
- Layer 2: holds face-state values at the boundary face for the
  HLLC Riemann problem.

The kernel uses $n_g = 2$ in `StrangSolver::init()` line 639.

## Flux commutativity

Since the Euler flux $\mathbf{F}_x$ is a pointwise function of
$\mathbf{U}$, the copy identity transfers automatically:

$$\mathbf{F}_x(\mathbf{U}_{\mathrm{ghost}}) \;=\; \mathbf{F}_x(\mathbf{U}_{\mathrm{phys}}), \quad (\text{B4-flux-commute})$$

component-wise. sympy verifies this as a trivial identity (the
ghost copy preserves the argument of $\mathbf{F}_x$, so the output
is identical). This closes the BC: no additional flux-side
correction is needed.

## Compatibility with perturbation storage

The perturbation storage $(\delta\rho, m_x, m_y, \delta E)$ is
periodic in $x$ whenever the full state $(\rho, m_x, m_y, E_{\mathrm{tot}})$
is periodic, because the HSE background $\bar\rho(y), \bar p(y)$
depends only on $y$: adding a $y$-only function to a $x$-periodic
function preserves $x$-periodicity. Therefore the copy on stored
variables is automatically the correct BC on the full state — no
re-encoding at the boundary is needed.

## Face-state ghost fill

After MUSCL-Hancock writes $\mathbf{w}_L, \mathbf{w}_R$ at each
internal cell, the face-state arrays have undefined values at the
ghost cells $i_g = n_g - 1$ and $i_g = n_g + n_x$. The kernel
`k_ghost_face_x` at line 564 fills them using the same periodic
copy pattern:

- $\mathbf{w}_L$ at ghost $i_g = n_g - 1$ (left boundary face) is
  copied from $\mathbf{w}_L$ at $i_g = n_g + n_x - 1$ (the last
  physical cell's left face, which by periodicity corresponds to
  the left-boundary face on the opposite side).
- $\mathbf{w}_R$ at ghost $i_g = n_g + n_x$ (right boundary face)
  is copied from $\mathbf{w}_R$ at $i_g = n_g$.

This preserves the MUSCL reconstructions computed on the first
ghost layer by `k_muscl_hancock_x` (which reads cell data from the
outer ghost layer, which is already periodic).

## ✅ Verification checkpoint (to be wired)

1. **Periodic-copy exactness.** After `k_ghost_x` has run, every
   ghost-cell value is bit-identical to its source. Test:
   `test_strang_unit.cu` §B4-ghost-copy.

2. **Smooth IC round-trip.** A periodic sine-wave IC (§D1 entropy
   wave) evolved for one period $T = L_x / u_0$ should return to
   the initial state to truncation-error precision (limited by
   §E1's $O(\Delta x^2)$ bound, not by the BC itself). Test:
   `test_strang_convergence.cu`.

3. **Ghost-face consistency.** After the y-sweep has touched the
   face-state ghost fill on the **x**-periodic axis, the HLLC flux
   at $i = n_g$ (left boundary) should equal the HLLC flux at
   $i = n_g + n_x$ (right boundary) to machine precision. Test:
   `test_strang_unit.cu` §B4-face-flux-equal.

Failure of (1) is a straight copy bug in `k_ghost_x`. Failure of
(2) is either (1) failing silently or a deeper bug in the HLLC /
update sequence that breaks periodic closure. Failure of (3) is
a `k_ghost_face_x` bug (index miscomputation or wrong
source-destination pairing).
