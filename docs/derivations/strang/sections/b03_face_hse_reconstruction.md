# B3. Face-centred HSE reconstruction (well-balancing necessary condition)

> **sympy script:** `scripts/b03_face_hse_reconstruction.py`
> **generated LaTeX:** `output/b03_face_hse_reconstruction.latex.tex`
> **verifies:** 13 strong-form identities — 4 face-state equality
> identities ($\rho_L = \rho_R$, $P_L = P_R$, $u_L = u_R$,
> $v_L = v_R$ on pure HSE); 4 face-flux equality identities
> ($F_{y,L}[k] = F_{y,R}[k]$, $k=0..3$); 4 face-flux form
> identities ($F_{y}$ at HSE = $(0, 0, \bar p, 0)$); 1
> cell-centred-reconstruction counter-example identity
> **code checkpoints:**
> `src/gpu/explicit/strang_solver.cu :: k_muscl_hancock_y`
> (line 343-372: `y_bot`, `y_top` at face; `d_hse_rho`, `d_hse_p`
> evaluated at face y-coord; `rL = rho_bar_bot + rhoP_bot`,
> `PL = p_bar_bot + PP_bot`)

The y-sweep reconstructs two face-state vectors $\mathbf{U}_L$ and
$\mathbf{U}_R$ at each face between cells $j$ and $j+1$. The
**well-balancing** (WB) requirement is: on pure HSE
($\delta \rho \equiv 0, \delta P \equiv 0, u \equiv v \equiv 0$),
the reconstructed face states must be **algebraically identical**
so that the HLLC flux jump vanishes at round-off precision. The
section proves that this is achieved only when the HSE background
$\bar\rho, \bar p$ is evaluated **at the face y-coordinate**, with
the perturbation variables added on both sides. Cell-centred
background reconstruction breaks WB at $O(\Delta y)$ and drives a
drift of order $|d\bar\rho/dy|$ per step.

## Face reconstruction formula

Let $y_{\mathrm{face}} = y_{\mathrm{lo}} + j_{\mathrm{face}}\,\Delta y$
(the face index is $j_{\mathrm{face}} = j + 1$ between cells $j$
and $j + 1$). Each side computes

$$\begin{aligned}\rho_{L/R} \;&=\; \bar\rho(y_{\mathrm{face}}) \;+\; \bigl(\delta \rho_{j / j+1} \;\mp\; \tfrac{1}{2}\,s^\rho_{j / j+1}\bigr), \\ P_{L/R} \;&=\; \bar p(y_{\mathrm{face}}) \;+\; \bigl(\delta P_{j / j+1} \;\mp\; \tfrac{1}{2}\,s^P_{j / j+1}\bigr), \\ u_{L/R} \;&=\; u_{j / j+1} \;\mp\; \tfrac{1}{2}\,s^u_{j / j+1}, \\ v_{L/R} \;&=\; v_{j / j+1} \;\mp\; \tfrac{1}{2}\,s^v_{j / j+1},\end{aligned} \quad (\text{B3-face-recon})$$

where $s^X_j$ is the MC-limited slope of variable $X$ in cell $j$
(§A10). The L-side (top face of cell $j$) uses a $+\tfrac{1}{2}s$
extrapolation, and the R-side (bottom face of cell $j+1$) uses
$-\tfrac{1}{2}s$.

## WB necessary condition (strong form)

On pure HSE, $\delta \rho \equiv 0$, $\delta P \equiv 0$, $u \equiv v
\equiv 0$, and all MC slopes are zero (the slope operator is
multi-linear in its inputs, so $s(\mathbf{0}, \mathbf{0}) =
\mathbf{0}$). Then **by direct substitution**:

$$\rho_L \;=\; \bar\rho(y_{\mathrm{face}}), \qquad \rho_R \;=\; \bar\rho(y_{\mathrm{face}}), \qquad \rho_L - \rho_R \;=\; 0, \quad (\text{B3-WB})$$

and analogously for $P, u, v$. sympy verifies all four component
equalities as strong-form identities (no simplification required —
direct substitution).

## HSE face-flux form

On pure HSE, $u = v = 0$, so the Euler flux simplifies to

$$\mathbf{F}_y(\mathbf{U}_{\mathrm{HSE}}) \;=\; (\bar\rho v,\; \bar\rho u v,\; \bar\rho v^2 + \bar p,\; (E + \bar p) v)^{\mathsf T} \;\big|_{u=v=0} \;=\; (0,\; 0,\; \bar p(y_{\mathrm{face}}),\; 0)^{\mathsf T}. \quad (\text{B3-face-flux})$$

sympy verifies both component-wise equalities $F_L[k] = F_R[k]$ on
pure HSE and the explicit form $(0, 0, \bar p, 0)^{\mathsf T}$ for
each component $k \in \{0, 1, 2, 3\}$.

## HLLC on HSE degenerates to the exact pressure flux

When $\mathbf{U}_L = \mathbf{U}_R = \mathbf{U}_{\mathrm{HSE}}$, the
HLLC Riemann solver (§A8) is exactly the left/right flux:

$$\mathbf{F}_{\mathrm{HLLC}}(\mathbf{U}_L, \mathbf{U}_R)\bigg|_{\mathbf{U}_L = \mathbf{U}_R} \;=\; \mathbf{F}_y(\mathbf{U}_L) \;=\; (0, 0, \bar p(y_{\mathrm{face}}), 0)^{\mathsf T}.$$

This follows algebraically from §A8's HLLC strong-form identities:
on an identical left-right pair, any wave-branch gives the same
flux, and that flux is $\mathbf{F}_y(\mathbf{U}_{\mathrm{HSE}})$. No
sympy re-verification is needed — the §A8 identities carry over.

## Balance with the gravity source

The kernel's y-sweep update combines the HLLC flux divergence with
the gravity source term $S_{m_y} = -\rho g$, $S_E = -m_y g$ (§C1).
The flux divergence at cell $j$ is

$$-\frac{1}{\Delta y}\bigl[F_y(U_{j+1/2}) - F_y(U_{j-1/2})\bigr]\bigg|_{\mathrm{HSE}} \;=\; -\frac{1}{\Delta y}\bigl[(0, 0, \bar p_{j+1/2}, 0) - (0, 0, \bar p_{j-1/2}, 0)\bigr] \;=\; -\bigl(0, 0, \frac{d\bar p}{dy}\big|_j, 0\bigr) \;=\; \bigl(0, 0, \bar\rho_j g, 0\bigr).$$

The gravity source contributes $(0, 0, -\bar\rho_j g, -0 \cdot g) =
(0, 0, -\bar\rho_j g, 0)$. Their sum is exactly zero, closing the
HSE balance to the pointwise discretisation accuracy of §B2's
finite-difference ODE residual (which is 2nd-order as $\Delta y
\to 0$, per §E5). This is the **only** way the kernel can preserve
HSE to round-off; the alternative (cell-centred background) fails
below.

## Cell-centred reconstruction: the counter-example

If the kernel had instead reconstructed the face from cell-centre
backgrounds,

$$\rho_L^{\mathrm{wrong}} \;=\; \bar\rho(y_j) \;+\; \delta\rho_j, \qquad \rho_R^{\mathrm{wrong}} \;=\; \bar\rho(y_{j+1}) \;+\; \delta\rho_{j+1},$$

then on pure HSE the difference is

$$\rho_L^{\mathrm{wrong}} - \rho_R^{\mathrm{wrong}} \;=\; -\Delta y\,\frac{d\bar\rho}{dy} \;+\; O(\Delta y^3), \quad (\text{B3-wrong})$$

which is non-zero (the HSE density changes between cells by
$\Delta y \, d\bar\rho/dy$). The HLLC solver would see this as a
spurious density jump at every face and produce a non-zero flux of
order $\Delta y$ per face per step, driving a drift of order
$|d\bar\rho/dy|$ that accumulates linearly in time. WB would be
broken at the leading 1st order.

## Implementation check: face-centred evaluation

The kernel correctness relies on the lines

```cpp
// strang_solver.cu, k_muscl_hancock_y, line 343-345
double y_bot = y_lo + j_phys * dy;          // face y-coordinate
double rho_bar_bot = d_hse_rho(y_bot, rho0_gm1, hse_coeff, inv_gm1);
double p_bar_bot   = d_hse_p(rho_bar_bot, K_poly, gamma);
```

(for the top face, line 360-362 uses `y_top = y_lo + (j_phys + 1) * dy`).
Both sides of the face evaluate $\bar\rho, \bar p$ at the **same**
$y_{\mathrm{face}}$ because the face index is a single integer —
there is no "L-side face y" and "R-side face y". This is the
structural guarantee of WB.

## ✅ Verification checkpoint (to be wired)

1. **Pure HSE face-state equality.** Start the solver with the HSE
   background, zero perturbations. At every internal face, assert
   $\mathbf{U}_L = \mathbf{U}_R$ to ULP precision (all four
   components). Test: `test_strang_muscl.cu` §B3-hse-face-equal.

2. **Pure HSE flux jump.** Compute the HLLC flux jump
   $\mathbf{F}_R - \mathbf{F}_L$ at every internal face after the
   MUSCL-Hancock predictor; required $\le 10\varepsilon_{\mathrm{mach}}
   \bar p(y_{\mathrm{face}})$ (only the $P$ component has a
   non-zero scale). Test: `test_strang_muscl.cu` §B3-hse-flux.

3. **Long-time HSE preservation.** After $10^4$ Strang steps on
   pure HSE IC, the max-norm of the perturbation state is bounded
   by $\varepsilon_{\mathrm{mach}} \cdot N \cdot \kappa(\bar p,
   \bar\rho)$, where $\kappa$ is the condition number of the
   face-centred HSE evaluation (§E5). Test:
   `test_strang_step.cu` §B3-hse-longtime.

Failure of (1) or (2) is a structural bug in `k_muscl_hancock_y`
— most likely a cell-centred background reuse (copy-paste of the
HSE background from cell $j$ instead of evaluating at the face).
Failure of (3) is either (1)/(2) failing silently, or a more
subtle bug in the HLLC flux assembly (see §C1 for the gravity-
source balance that must be correct for (3) to hold).
