# A2. Conservative ↔ primitive bijection

> **sympy script:** `scripts/a02_conservative_primitive.py`
> **generated LaTeX:** `output/a02_conservative_primitive.latex.tex`
> **verified:**
> - round-trip $\mathbf{W}\to\mathbf{U}\to\mathbf{W}$ and $\mathbf{U}\to\mathbf{W}\to\mathbf{U}$ on all 4 components
> - both Jacobian determinants (forward and reverse)
> - 16 chain-rule entries of $\partial \mathbf{U}/\partial\mathbf{W}$ times $\partial \mathbf{W}/\partial\mathbf{U}$
> - positivity envelope: $\rho > 0, P > 0$ in $\mathbf{W}$-space maps to admissible $\mathbf{U}$-space
>
> **code checkpoints:**
> - `src/gpu/explicit/strang_device.cuh :: d_cons2prim`

The strang solver stores $\mathbf{U} = (\rho, m_x, m_y, E)^\top$ on
disk but needs the primitive form
$\mathbf{W} = (\rho, u, v, p)^\top$ every time a Riemann solver, a
reconstruction, or a physically-motivated floor is applied. This
section proves strong-form that the two forms are related by a
smooth bijection on the admissible state space and derives the
consequence for the kernel's numerical floor.

## Forward map (primitive → conservative)

$$\mathbf{U}(\mathbf{W}) \;=\; \begin{pmatrix}\rho\\ \rho u\\ \rho v\\ \dfrac{p}{\gamma - 1} + \tfrac{1}{2}\rho(u^{2}+v^{2})\end{pmatrix}. \quad (\text{A2-prim2cons})$$

## Inverse map (conservative → primitive)

$$\mathbf{W}(\mathbf{U}) \;=\; \begin{pmatrix}\rho\\ m_x / \rho\\ m_y / \rho\\ (\gamma - 1)\left(E - \dfrac{m_x^{2} + m_y^{2}}{2\rho}\right)\end{pmatrix}. \quad (\text{A2-cons2prim})$$

## Round-trip identities

**Primitive → conservative → primitive.** For every admissible
$\mathbf{W}$ (i.e., $\rho > 0$, $p > 0$),

$$\mathbf{W}\bigl(\mathbf{U}(\mathbf{W})\bigr) \;\equiv\; \mathbf{W}.$$

sympy verifies this at the *symbolic* level on each of the four
components.

**Conservative → primitive → conservative.** For every admissible
$\mathbf{U}$ (i.e., $\rho > 0$, $E > (m_x^2 + m_y^2)/(2\rho)$),

$$\mathbf{U}\bigl(\mathbf{W}(\mathbf{U})\bigr) \;\equiv\; \mathbf{U}.$$

Again each of the four components is verified to reduce identically
to its input.

## Jacobian determinants

Expanding the $4\times 4$ Jacobian $\partial\mathbf{U}/\partial\mathbf{W}$
along the $E$-row (the only row with a $1/(\gamma-1)$ factor) and
using block triangularity,

$$\det\!\left(\frac{\partial \mathbf{U}}{\partial \mathbf{W}}\right) \;=\; \frac{\rho^{2}}{\gamma - 1}. \quad (\text{A2-det-forward})$$

Correspondingly,

$$\det\!\left(\frac{\partial \mathbf{W}}{\partial \mathbf{U}}\right) \;=\; \frac{\gamma - 1}{\rho^{2}}. \quad (\text{A2-det-inverse})$$

Neither determinant vanishes in the admissible region $\rho > 0$,
so the inverse function theorem applies locally everywhere in that
region; the $\mathbf{U}\leftrightarrow\mathbf{W}$ map is a
diffeomorphism on the admissible open set.

**Chain-rule verification.** Evaluated at the forward image
$\mathbf{U}(\mathbf{W})$ of any primitive state,

$$\frac{\partial \mathbf{U}}{\partial \mathbf{W}} \,\cdot\, \frac{\partial \mathbf{W}}{\partial \mathbf{U}}\bigg|_{\mathbf{U}(\mathbf{W})} \;=\; I_{4\times 4}. \quad (\text{A2-chain-identity})$$

sympy verifies this entry-by-entry — 16 scalar identities, all
reducing to the expected Kronecker delta.

## Positivity envelope

The pressure is recovered from the conservative state by

$$p \;=\; (\gamma - 1)\left(E - \frac{m_x^{2} + m_y^{2}}{2\rho}\right).$$

Admissibility ($p > 0$) is therefore equivalent to the strict
kinetic-energy inequality

$$E \;>\; \frac{m_x^{2} + m_y^{2}}{2\rho}. \quad (\text{A2-positivity})$$

The kernel's `d_cons2prim` applies the clamp
`P = fmax(P, 1e-30)` whenever round-off of a nearly-vacuum state
has driven the right-hand side below zero. This clamp is strictly
outside the admissible region of the continuum theory; its purpose
is to prevent catastrophic cancellation from propagating into
$\sqrt{\gamma p / \rho}$ (the sound-speed used by the Riemann
solver). No solution in the admissible region triggers the clamp.

## Verification checkpoints

Three invariants the kernel must satisfy, checked by
`tests/test_strang_unit.cu`:

1. **Round-trip identity at ULP precision.** For any admissible
   primitive state generated randomly (100 samples with $\rho \in
   [0.1, 10]$, $p \in [0.1, 10]$, $|u|, |v| \in [0, 3c_0]$),
   `cons_to_prim(prim_to_cons(W))` must agree with $W$ to within
   $10\,\varepsilon_{\mathrm{mach}}\,\|W\|_\infty$.

2. **Pressure-floor activation.** Construct a near-vacuum state
   with $E$ chosen so that the analytic pressure is $-10^{-20}$
   (below floor but above round-off). Verify that `d_cons2prim`
   returns $p = 10^{-30}$ rather than propagating a negative
   value. This is the **only** legitimate trigger path for the
   floor.

3. **Sound-speed positivity.** For any admissible state, the sound
   speed $c = \sqrt{\gamma p / \rho}$ returned by the kernel must
   be strictly positive. Failure indicates that the pressure floor
   was not applied or that a negative pressure was passed through
   without clamping — both are bugs the Riemann solver would
   silently turn into NaN.

Failures of (1) and (3) are correctness bugs; failure of (2) would
be catastrophic on a stratified-HSE simulation's low-pressure
atmosphere region.
