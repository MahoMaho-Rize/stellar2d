# B6. Outflow-y top boundary condition

> **sympy script:** `scripts/b06_outflow_y_bc.py`
> **generated LaTeX:** `output/b06_outflow_y_bc.latex.tex`
> **verified:**
> - 4 ghost-uniform identities ($\mathbf{U}_{\mathrm{ghost}}(g_1)[k] = \mathbf{U}_{\mathrm{ghost}}(g_2)[k]$, $k = 0..3$)
> - plus 2 documentation identities (Neumann continuum interpretation; Riemann-invariant extrapolation error leading order)
>
> **code checkpoints:**
> - `src/gpu/explicit/strang_solver.cu :: k_ghost_y` (line 71, top branch: `jg_ghost = ng+ny+g`, `jg_src = ng+ny-1` — all ghosts copy from the same last physical cell)
> - `src/gpu/explicit/strang_solver.cu :: k_ghost_face_y` (line 618, top branch: outflow face-state copy)

The top of the Strang domain is an **outflow** boundary implemented
by zero-gradient copy: every ghost cell is a copy of the last
physical cell. The BC is non-reflecting for supersonic outflow and
only approximately non-reflecting for subsonic outflow (one
characteristic is incoming and its amplitude is set by zeroth-
order extrapolation, which admits an $O(\Delta y)$ reflection
error). This is a standard choice for 2D compressible hydro with
gravity — the alternatives (exact characteristic BC,
sponge layers) are more complex and the kernel opts for the
simple, robust zero-gradient form.

## Ghost-cell copy formula

$$\mathbf{U}_{\mathrm{ghost}}(j_g = n_g + n_y + g) \;=\; \mathbf{U}(j_g = n_g + n_y - 1), \quad g \in \{0, \ldots, n_g - 1\}. \quad (\text{B6-zero-gradient})$$

All ghost cells receive the **same** copy (the last physical cell's
value). This is a zeroth-order extrapolation — it is flat, not
linearly extrapolated.

## Continuum interpretation

In the limit $\Delta y \to 0$, the zero-gradient copy is equivalent
to the Neumann BC

$$\frac{\partial \mathbf{U}}{\partial y}\bigg|_{y = y_{\mathrm{top}}} \;=\; \mathbf{0}, \quad (\text{B6-neumann})$$

imposed componentwise on all four conservative variables. sympy
verifies that all ghosts are uniform copies (pairwise identical),
which is the discrete statement of zero normal derivative.

## Characteristic structure of the y-flux Jacobian

From §A3 (rotationally covariant: swap $x \leftrightarrow y$), the
eigenvalues of $\mathcal{A}_y(\mathbf{U})$ are

$$\mathrm{spec}(\mathcal{A}_y) \;=\; \{v - c,\;\, v,\;\, v,\;\, v + c\}, \quad (\text{B6-eigenvalues})$$

with corresponding eigenvectors (see §A3 rotated to y).

## Subsonic outflow

If $0 < v < c$ at the top, three characteristics $(v, v, v+c)$
propagate **out of** the domain and one ($v - c < 0$) propagates
**into** the domain. The incoming characteristic carries
information from outside the domain that the BC must supply.

The 1D Riemann-invariant carried by the incoming acoustic wave is

$$R_{-} \;=\; u - \frac{2 c}{\gamma - 1}, \quad (\text{B6-R-minus})$$

where $u$ is the velocity and $c$ the sound speed (in 1D along the
y-axis $u \leftrightarrow v$; we keep the 1D notation here). The
**correct** non-reflecting BC would set $R_{-}^{\mathrm{ghost}} =
R_{-}^{\mathrm{ext}}$ where $R_{-}^{\mathrm{ext}}$ is the value
implied by whatever condition holds outside (for stellar-atmosphere
outflow this is typically zero perturbation in $R_{-}$).

Zero-gradient copy instead sets $R_{-}^{\mathrm{ghost}} =
R_{-}^{\mathrm{phys, last}}$, the interior value at the last cell.
The leading-order error is

$$R_{-}^{\mathrm{extrap}} \;-\; R_{-}^{\mathrm{true}} \;=\; -\,\tfrac{\Delta y}{2}\,\frac{d R_{-}}{dy} \;+\; O(\Delta y^{2}), \quad (\text{B6-error})$$

i.e., an $O(\Delta y)$ reflection at the boundary. For smooth
steady-state outflow where $dR_{-}/dy \approx 0$ this is
acceptable. For strong transients (an acoustic pulse travelling up)
the zero-gradient BC will partially reflect the pulse, generating
an $O(\Delta y)$ returning wave. If the test requires clean
outflow, a sponge layer or a proper characteristic BC is needed;
the Strang kernel does not provide one.

## Supersonic outflow

For $v > c$ at the top, all four eigenvalues are positive (all
outgoing). No information flows in from outside the domain. Zero-
gradient copy is then **exact** in the sense that the interior
solution's evolution cannot depend on the ghost values — the BC is
irrelevant provided it does not cause instability. Zero-gradient
is stable (no additional reflection).

## HSE interaction

On **pure HSE**, the perturbation state is zero at every physical
cell. Zero-gradient copies zero to the ghost. The reconstructed
face state at the top boundary uses the face y-coordinate for the
HSE background (§B3), so the face-$(\rho, u, v, P)$ pair is
$(\bar\rho(y_{\mathrm{top}}), 0, 0, \bar p(y_{\mathrm{top}}))$ on
both sides. §B3's WB identity holds, and HSE is preserved.

For **non-HSE interior states**, zero-gradient does **not** preserve
HSE in the ghost — the interior perturbation values (some non-zero
combination of $\delta\rho, m_x, m_y, \delta E$) are copied, and
the reconstructed face state may differ from the pure-HSE face
state. This is intrinsic to the outflow BC: material is allowed to
leave the domain, and the ghost state represents what has "just
left". The gravity source still operates correctly because it
reads the physical-cell values, not the ghost values.

## Face-state ghost fill

The face-state ghost fill `k_ghost_face_y` (line 618) mirrors the
cell-data ghost fill: the top-boundary ghost faces receive copies
of the last physical cell's face states. For outflow, the top
ghost's $\mathbf{w}_L$ (bottom face) is set to the physical cell's
$\mathbf{w}_R$ (top face) and the ghost's $\mathbf{w}_R$ is set to
the same:

```cpp
// strang_solver.cu, k_ghost_face_y, line 620-625 (outflow top branch)
for (int c = 0; c < 4; ++c) {
    d_wL[kd*4+c] = d_wR[ks*4+c];   // ghost bottom = last cell top
    d_wR[kd*4+c] = d_wR[ks*4+c];   // ghost top = same (outflow)
}
```

This closes the top-boundary face-state buffers for the next
sweep's HLLC Riemann problem.

## Trade-off and alternatives

| BC | accuracy | incoming char | implementation | stability |
|---|---|---|---|---|
| zero-gradient (current) | $O(\Delta y)$ | linearly extrapolated | 1 copy | stable |
| linear extrapolation | $O(\Delta y^2)$ | linearly extrapolated | 2-point extrap | stable |
| characteristic BC | exact (smooth) | solved separately | complex; requires 1D Riemann at boundary | stable |
| sponge-layer damping | varies | attenuated | many layers, tunable | stable |

The Strang kernel uses the simplest (zero-gradient) because (a) it
is the standard Godunov outflow choice, (b) stellar-atmosphere
applications typically have steady-state outflow where
$dR_{-}/dy \approx 0$, and (c) tests in §D-series and §E-series do
not probe the outflow reflection sensitivity. Future work needing
quiet outflow should add a characteristic BC or sponge layer.

## Verification checkpoints

1. **Zero-gradient copy exactness.** All ghost cells are bit-
   identical to the last physical cell. Test:
   `test_strang_unit.cu` §B6-ghost-copy.

2. **Pure-HSE preservation.** On HSE IC, the top ghost cells retain
   zero perturbation state; the face reconstruction at the top
   boundary agrees with §B3's WB identity. Test:
   `test_strang_unit.cu` §B6-hse-pres.

3. **Supersonic outflow exit.** Initialise a supersonic y-flow at
   the top (e.g., Gaussian pulse with $v > c$) and measure the
   return flux at the top boundary; required: returned flux
   amplitude $< 10^{-6}$ of the outgoing pulse. Test:
   `test_strang_step.cu` §B6-supersonic-outflow.

4. **Subsonic outflow reflection.** Initialise a subsonic acoustic
   pulse travelling upwards and measure the reflection amplitude;
   expected: $O(\Delta y)$ reflection (which at $n_y = 256$ is
   $\sim 10^{-2}$ of the outgoing pulse). Test:
   `test_strang_step.cu` §B6-subsonic-reflection.

Failure of (1) is a copy bug in `k_ghost_y` (wrong source index).
Failure of (2) is usually a §B1 perturbation storage bug (ghost is
being filled with full state instead of perturbation). Failures
(3) and (4) indicate a deeper flux or reconstruction bug.
