# B5. Reflective-y bottom boundary condition

> **sympy script:** `scripts/b05_reflective_y_bc.py`
> **generated LaTeX:** `output/b05_reflective_y_bc.latex.tex`
> **verifies:** 13 strong-form identities — 1 involution identity
> ($\mathcal{R}_{\mathrm{ref}}^2 = \mathbf{I}$); 4 flux-reversal
> identities ($\mathbf{F}_y(\mathcal{R}\mathbf{U}) = \mathcal{R}'
> \mathbf{F}_y(\mathbf{U})$, 4 components); 4 wall-face flux
> identities ($\mathbf{F}_y(\rho, u, 0, P) = (0, 0, P, 0)$);
> 4 HSE-perturbation-zero identities
> **code checkpoints:**
> `src/gpu/explicit/strang_solver.cu :: k_ghost_y`
> (line 71, bottom branch: `jg_ghost = ng-1-g`, `jg_src = ng+g`,
> `d_my[k_dst] = -d_my[k_src]`)
> `src/gpu/explicit/strang_solver.cu :: k_ghost_face_y`
> (line 593, bottom branch: `d_wR[kd*4+2] = -d_wL[ks*4+2]`
> (v component negated))

The bottom of the Strang domain is a solid wall. The ghost cells
are the mirror image of the physical cells across the wall plane,
with the normal component of velocity (here $v$) negated. In
conservative-variable form this is the reflection

$$\mathcal{R}_{\mathrm{ref}} \;=\; \mathrm{diag}(+1,\, +1,\, -1,\, +1). \quad (\text{B5-R-ref})$$

The $+1$s keep $\rho, m_x, E_{\mathrm{tot}}$ unchanged; the $-1$
negates $m_y$. This is the correct choice: at the wall, a fluid
parcel moving towards the wall is bounced back with its normal
velocity reversed, while its tangential velocity and thermodynamic
state are preserved.

## Ghost-cell copy formula

With kernel index convention (cell $j_{\mathrm{phys}} \in \{0, 1,
\ldots, n_y - 1\}$ with ghost layers at $j_g \in \{0, \ldots,
n_g - 1\}$ below and $\{n_g + n_y, \ldots, n_g + n_y + n_g - 1\}$
above):

$$\mathbf{U}_{\mathrm{ghost}}(j_g = n_g - 1 - g) \;=\; \mathcal{R}_{\mathrm{ref}}\,\mathbf{U}(j_g = n_g + g), \quad g \in \{0, \ldots, n_g - 1\}. \quad (\text{B5-ghost})$$

For $n_g = 2$: $g = 0$ mirrors the first physical cell
($j_g = n_g = 2 \to j_g = n_g - 1 = 1$), and $g = 1$ mirrors the
second ($j_g = 3 \to j_g = 0$). The physical wall plane is at
$j_g = n_g - 1/2 = 1.5$.

## Flux-reversal identity (strong form)

Reflection and flux assembly commute up to a component-wise
sign-flip pattern:

$$\mathbf{F}_y(\mathcal{R}_{\mathrm{ref}}\,\mathbf{U}) \;=\; \mathrm{diag}(-1, -1, +1, -1)\,\mathbf{F}_y(\mathbf{U}), \quad (\text{B5-flux-reversal})$$

where the "flux-reflection matrix" differs from $\mathcal{R}_{\mathrm{ref}}$:
mass, x-momentum, and energy fluxes **do** flip sign (they are
linear in $v$), but the y-momentum flux does **not** (it depends on
$\rho v^2$ and $P$, both even in $v$). sympy verifies all four
components independently.

The immediate physical consequence: at the wall face (where
$\mathbf{U}_L = \mathcal{R}_{\mathrm{ref}} \mathbf{U}_R$), the
flux-difference contributions from mass, x-momentum, and energy
vanish after HLLC averaging by the sign-flip symmetry of the L/R
pair, while the y-momentum contribution remains non-zero but
isotropic (purely pressure).

## Wall-face flux (v = 0 at wall)

At the wall the normal velocity vanishes (this is the physical
content of the no-penetration boundary). Substituting $v = 0$:

$$\mathbf{F}_y(\rho, u, 0, P) \;=\; (0,\; 0,\; P,\; 0)^{\mathsf T}. \quad (\text{B5-wall-flux})$$

No mass or kinetic energy flux crosses the wall. The y-momentum
flux is the pressure at the wall — this is Newton's third law
(force on the wall = force on the fluid, transmitted by pressure).

**Consequence for HLLC at the wall.** The Riemann problem with
$\mathbf{U}_L = \mathcal{R}_{\mathrm{ref}} \mathbf{U}_R$ has
$S_\star = 0$ (by the L/R symmetry in §A8): the intermediate
contact wave sits exactly at the wall. The HLLC flux is the
contact-wave flux, which by §A8 evaluates to
$(0, 0, P^\star, 0)$ with $P^\star$ equal to the physical-side
pressure $P_R$ (by §A8's $p^\star_L = p^\star_R$ strong-form
identity). No mass or kinetic energy is transported across the
wall to round-off.

## Involution

$$\mathcal{R}_{\mathrm{ref}}^{2} \;=\; \mathbf{I}, \quad (\text{B5-involution})$$

which makes reflection a $\mathbb{Z}_2$ symmetry operation. Chaining
two reflective BCs returns to the original state; this is useful
for two-wall-bounded domains (not used in the Strang kernel's
single-wall bottom, but structurally important for §D7).

## HSE preservation

On pure HSE, the perturbation state is identically zero at every
cell. Reflection maps zero to zero component-wise:

$$\mathcal{R}_{\mathrm{ref}}\,(0, 0, 0, 0)^{\mathsf T} \;=\; (0, 0, 0, 0)^{\mathsf T}.$$

So the ghost perturbation is also zero, and the reconstructed face
state at the wall is exactly
$(\bar\rho, 0, 0, \bar p/(\gamma-1))^{\mathsf T}$. By §B3, $L = R$
on pure HSE, and the wall-face flux is $(0, 0, \bar p, 0)$ —
balanced by the cell-interior gravity source $(0, 0, -\bar\rho g,
0)$. The wall does not break well-balancing.

**Subtle point.** The ghost cell is at a y-coordinate $y_{\mathrm{ghost}}
= -y_{\mathrm{phys}}$ (mirrored across the wall $y = 0$). The HSE
background $\bar\rho, \bar p$ is **not** symmetric in $y$ (it
decreases monotonically with height), so naively one might worry
that reflection breaks HSE. The kernel's design avoids this by
storing only the **perturbation**, which is zero on pure HSE and
stays zero under reflection. The HSE background is evaluated only
at **physical** or **face** y-coordinates in the MUSCL predictor
(§B3), never at ghost y-coordinates. This decoupling is essential
for the reflective BC to be HSE-preserving.

## Face-state reflection

The face-state ghost fill `k_ghost_face_y` at line 593 applies the
same reflection on $\mathbf{w}_L, \mathbf{w}_R$, with the special
handling that the ghost cell's top face $\mathbf{w}_R$ is the
**reflected** version of the physical cell's bottom face
$\mathbf{w}_L$:

```cpp
// strang_solver.cu, k_ghost_face_y, line 601-609
d_wR[kd*4+0] =  d_wL[ks*4+0];   // rho unchanged
d_wR[kd*4+1] =  d_wL[ks*4+1];   // u unchanged
d_wR[kd*4+2] = -d_wL[ks*4+2];   // -v (reflect normal)
d_wR[kd*4+3] =  d_wL[ks*4+3];   // P unchanged
```

This implements the reflection matrix on the face-state primitive
vector $(\rho, u, v, P)$ (the $-1$ applies only to the normal
component $v$). The mapping is from the physical cell's $\mathbf{w}_L$
(bottom face) to the ghost's $\mathbf{w}_R$ (top face, which lies
on the wall).

## ✅ Verification checkpoint (to be wired)

1. **Pure-HSE preservation.** After arbitrary ghost-fill passes on
   HSE IC, the perturbation state remains bitwise zero at all ghost
   and physical cells. Test: `test_strang_unit.cu` §B5-hse-pres.

2. **Wall-flux zero-mass.** Initialise a non-HSE y-flux at the
   wall cell (e.g., a symmetric-bouncing IC), run one Strang step,
   and confirm that the total mass integrated over the physical
   domain is conserved to ULP precision (no leakage through the
   wall). Test: `test_strang_step.cu` §B5-wall-mass.

3. **Flux-reversal identity check.** Given random admissible
   $\mathbf{U}$, compute $\mathbf{F}_y(\mathcal{R}\mathbf{U})$ and
   $\mathcal{R}' \mathbf{F}_y(\mathbf{U})$ on the host; agreement
   to ULP precision for all four components. Test:
   `test_strang_unit.cu` §B5-flux-reversal.

4. **Face-state reflection.** After `k_ghost_face_y`, assert that
   $\mathbf{w}_R[\mathrm{ghost}] = \mathcal{R}\,\mathbf{w}_L[\mathrm{phys}]$
   for all four components. Test: `test_strang_unit.cu`
   §B5-face-reflection.

Failure of (1) is a sign error in `k_ghost_y` or a BG-related bug
(the kernel is subtracting or adding the wrong background at the
wall). Failure of (2) is a deeper issue — the wall is leaking
mass, which means the Riemann-problem symmetry is broken (either
$\mathcal{R}$ is wrong, or HLLC's L/R treatment is not symmetric).
Failure of (3) is a direct math bug and should be fixed by reading
§A1's flux formula and restoring the correct signs.
