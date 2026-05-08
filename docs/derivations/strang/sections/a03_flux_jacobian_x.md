# A3. x-direction flux Jacobian eigensystem

> **sympy script:** `scripts/a03_flux_jacobian_x.py`
> **generated LaTeX:** `output/a03_flux_jacobian_x.latex.tex`
> **verified:**
> - characteristic polynomial
> - $A_x R_k = \lambda_k R_k$ for all 4 eigenpairs × 4 components
> - left-right orthogonality $L R = I$ (16 entries)
> - characteristic decomposition $R\,\mathrm{diag}(\lambda)\,L = A_x$ (16 entries)
>
> **code checkpoints:**
> - `src/gpu/explicit/strang_device.cuh :: d_lmhllc` (uses sound speed + wave speeds)
> - indirectly: every Riemann solver that invokes $S_L, S_R, S_\star$

The flux Jacobian $A_x \equiv \partial \mathbf{F}_x / \partial \mathbf{U}$
governs the linearised propagation of infinitesimal perturbations.
Its eigensystem is the algebraic foundation for four downstream
facts:

- The **sound speed** $c = \sqrt{\gamma p / \rho}$ appears as the
  magnitude-of-largest-eigenvalue; §C2 (CFL) bounds $\Delta t$ in
  terms of it.
- The **Davis wave-speed estimates** (§A9) use $u \pm c$ directly
  from $\{\lambda_k\}$.
- The **HLLC star region** (§A8) is parameterised by
  $p^\star, u^\star$ whose derivation invokes Rankine-Hugoniot
  jumps across $\lambda_0 = u - c$, $\lambda_{1,2} = u$, and
  $\lambda_3 = u + c$.
- The **acoustic linwave golden value** (§D2) is the right
  eigenvector $R_3$ scaled by amplitude $A$.

## Matrix form

Writing the primitive components in terms of the conservative ones,
$u = m_x/\rho$, $v = m_y/\rho$, and
$p = (\gamma-1)(E - (m_x^2+m_y^2)/(2\rho))$, the $4\times 4$ matrix
$A_x(\mathbf{U})$ is the strong-form Jacobian

$$A_x \;=\; \frac{\partial \mathbf{F}_x}{\partial \mathbf{U}}. \quad (\text{A3-Ax-def})$$

Its explicit entries are lengthy; sympy computes them automatically
and uses them throughout the section. The key results are the
eigensystem, not the matrix entries themselves.

## Characteristic polynomial and eigenvalues

$$\det\!\left(A_x - \lambda I\right) \;=\; (u - \lambda)^{2}\bigl[(u - \lambda)^{2} - c^{2}\bigr], \qquad c \;\equiv\; \sqrt{\gamma p / \rho}. \quad (\text{A3-charpoly})$$

Factoring gives the four eigenvalues

$$\{\lambda_k\}_{k=0}^{3} \;=\; \{\,u - c,\ u,\ u,\ u + c\,\}. \quad (\text{A3-eigvals})$$

**Physical interpretation.** The acoustic modes $u \mp c$ are the
two genuinely-nonlinear characteristic families; simple-wave solutions
in these families admit rarefactions or shocks (§A6). The
doubly-degenerate eigenvalue $u$ is linearly degenerate; its
eigenspace is two-dimensional and hosts the **entropy wave** (density
contrast at constant velocity and pressure) and the **shear wave**
(tangential-velocity contrast at constant density and pressure).

## Right eigenvectors

In conservative form, with specific total enthalpy

$$h \;=\; \frac{\gamma\,p}{(\gamma - 1)\,\rho} + \tfrac{1}{2}(u^{2} + v^{2}), \quad (\text{A3-enthalpy})$$

the right eigenvectors (columns of $R$) are

$$R_0 = \begin{pmatrix}1\\ u - c\\ v\\ h - u\,c\end{pmatrix},\qquad
R_1 = \begin{pmatrix}1\\ u\\ v\\ \tfrac{1}{2}(u^{2}+v^{2})\end{pmatrix},\qquad
R_2 = \begin{pmatrix}0\\ 0\\ 1\\ v\end{pmatrix},\qquad
R_3 = \begin{pmatrix}1\\ u + c\\ v\\ h + u\,c\end{pmatrix}. \quad (\text{A3-right-eigvecs})$$

**Degeneracy basis choice.** $R_1$ and $R_2$ together span the
two-dimensional eigenspace of $\lambda = u$. The choice above is
the conventional one: $R_1$ (entropy) has unit density component
and zero tangential-velocity component; $R_2$ (shear) has zero
density component and unit tangential-velocity component. Any
invertible linear combination of these two vectors is also a valid
basis, but the chosen one matches the decomposition used by Toro
§3.1.2 and by every Godunov-family reference in the literature.

**Strong-form verification.** For each $k \in \{0, 1, 2, 3\}$,

$$A_x R_k \;-\; \lambda_k R_k \;\overset{\text{sp.simplify}}{\longrightarrow}\; \mathbf{0}, \qquad (\text{after the substitution}\;\; c^2 \to \gamma p / \rho).$$

This is 16 scalar identities (4 eigenpairs × 4 components), all
verified.

## Left eigenvectors and orthogonality

$L = R^{-1}$ is computed symbolically by sympy. The normalisation
is such that

$$L R \;=\; I_{4\times 4}, \qquad (\text{A3-orthogonality})$$

verified entry-by-entry (16 scalar identities).

The $k$-th row $L_k$ of $L$ is the left eigenvector associated with
$\lambda_k$. In the projection interpretation,

$$\Delta \mathbf{U} \;=\; \sum_{k=0}^{3} (L_k \cdot \Delta \mathbf{U})\, R_k,$$

so $L_k \cdot \Delta \mathbf{U}$ is the **characteristic amplitude** in
the $k$-th wave family. This is the identity that §A7 (Riemann solver
comparison) and §D2 (acoustic linwave) invoke when constructing ICs
that excite exactly one wave family.

## Characteristic decomposition

$$A_x \;=\; R\,\mathrm{diag}(u - c,\ u,\ u,\ u + c)\,L, \qquad L = R^{-1}. \quad (\text{A3-diagonalisation})$$

sympy verifies this entry-by-entry (16 scalar identities), after
substituting $c = \sqrt{\gamma p / \rho}$ and simplifying the
resulting radical expressions. The diagonalisation confirms $A_x$
is strictly hyperbolic on the admissible region (real eigenvalues,
complete eigenvector basis).

## Verification checkpoints

The kernel does not explicitly assemble $A_x$ — it uses the Riemann
solver directly. Consequently the §A3 invariants are verified
indirectly through downstream sections:

1. **Acoustic linwave convergence.** An IC built from $R_3$
   (right-going acoustic) with amplitude $A = 10^{-6}$ must evolve
   for one wavelength $\lambda$ without any unphysical mode
   coupling. Test: `test_strang_linwave_convergence.cu` with
   `use_lm_fix = false` (§E2); ratios of $\{\delta\rho, \delta u,
   \delta p\}$ at each time must match the eigenvector components
   of $R_3$ to $O(\Delta x^2)$.

2. **Entropy wave non-mixing.** An IC built from $R_1$ (entropy
   only: non-zero $\delta\rho$, zero $\delta u, \delta v, \delta p$)
   must propagate without exciting any acoustic content. Test:
   `test_strang_convergence.cu` on an entropy IC in the co-moving
   frame — HLLC degenerates to pure upwind (§D1), and acoustic
   modes remain exactly zero to ULP.

3. **Sound speed positivity.** The kernel's $c = \sqrt{\gamma p / \rho}$
   must be strictly positive; negative $c$ indicates that §A2's
   positivity-envelope clamp failed upstream. Tested inside
   `test_strang_hllc.cu` on a Sod-tube IC.

Failures of (1) or (2) indicate an eigenvector bug (either in the
Riemann solver's star-region algebra of §A8, or in the slope limiter
of §A10 producing spurious characteristic mixing). Failure of (3) is
a positivity-preservation bug, caught earlier in §A2.
