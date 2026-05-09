# A4. Rotational covariance of the y-direction flux

> **sympy script:** `scripts/a04_rotational_covariance.py`
> **generated LaTeX:** `output/a04_rotational_covariance.latex.tex`
> **verified:**
> - involution $R^2 = I$ (16 entries), covariance $\mathbf{F}_y = R\,\mathbf{F}_x(R\,\mathbf{U})$ (4 components), Jacobian covariance $A_y = R\,A_x(R\,\mathbf{U})\,R$ (16 entries), y-direction characteristic polynomial
>
> **code checkpoints:**
> - `src/gpu/explicit/strang_solver.cu :: k_hllc_update_y` (argument permutation)
> - `src/gpu/explicit/strang_device.cuh :: d_euler_flux_y`

The 2D Euler flux is invariant under relabelling of the two
coordinate axes. This means the y-sweep Riemann solver does not
need to be written as a separate device function — the x-sweep
solver, invoked with its momentum arguments permuted, produces the
same numerical flux. The strang kernel exploits this in
`k_hllc_update_y`, which passes the $(m_x, m_y)$ components of its
left/right states in swapped order to the shared `d_lmhllc` routine.

## The permutation matrix

$$R \;=\; \begin{pmatrix}1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1\end{pmatrix}, \qquad R^{2} = I. \quad (\text{A4-R-def})$$

$R$ swaps the second and third components of the conservative
state vector $\mathbf{U} = (\rho, m_x, m_y, E)^\top$, mapping
$(\rho, m_x, m_y, E) \mapsto (\rho, m_y, m_x, E)$. Density and total
energy are unaffected. Because this is a permutation, $R$ is its
own inverse; sympy verifies all 16 entries of $R^2 = I$
individually.

## Flux covariance

$$\mathbf{F}_y(\mathbf{U}) \;=\; R\,\mathbf{F}_x\!\bigl(R\,\mathbf{U}\bigr). \quad (\text{A4-flux-covariance})$$

**Strong-form verification.** Expressing $\mathbf{U}$ in primitive
form via $\mathbf{U}(\rho, u, v, p)$ (§A1), computing both sides
and simplifying with sympy yields zero on all four conservative-
flux components.

**Interpretation at the kernel.** The component order in the
argument list of `d_lmhllc` is $(\rho, u_n, u_t, P)$, where $u_n$
is the velocity along the face normal and $u_t$ the velocity
parallel to the face. For an x-face this is $(u_n, u_t) = (u, v)$;
for a y-face it is $(u_n, u_t) = (v, u)$. The A4 identity says
that the same `d_lmhllc` device function produces the correct
numerical flux in both cases — what changes is only the argument
binding at the call site. This is why `k_hllc_update_y` writes

```cpp
d_lmhllc(d_wR[k0*4+0], d_wR[k0*4+2], d_wR[k0*4+1], d_wR[k0*4+3],
         d_wL[kT*4+0], d_wL[kT*4+2], d_wL[kT*4+1], d_wL[kT*4+3],
         ... );
```

with indices 1 and 2 swapped at both the left (`wR[k0]`) and the
right (`wL[kT]`) state. No new Riemann solver code is generated;
A4 guarantees correctness of this permutation.

## Jacobian covariance

By the chain rule (and $\partial (R\mathbf{U})/\partial \mathbf{U} = R$),

$$A_y(\mathbf{U}) \;=\; \frac{\partial \mathbf{F}_y(\mathbf{U})}{\partial \mathbf{U}} \;=\; \frac{\partial}{\partial \mathbf{U}}\Bigl(R\,\mathbf{F}_x(R\,\mathbf{U})\Bigr) \;=\; R\,\frac{\partial \mathbf{F}_x}{\partial \mathbf{U}}\bigg|_{R\mathbf{U}}\!\cdot R.$$

That is,

$$A_y(\mathbf{U}) \;=\; R\,A_x\!\bigl(R\,\mathbf{U}\bigr)\,R. \quad (\text{A4-Jacobian-covariance})$$

**Strong-form verification.** sympy computes the left-hand side by
direct Jacobian construction from the primitive-form substitution
of $\mathbf{F}_y$ into $A_y$; the right-hand side by the sequence
swap $(m_x, m_y) \to (m_y, m_x)$ inside $A_x$, followed by two $R$
multiplications. The 16 entries of the difference all vanish under
`sp.simplify`.

## Eigenvalues of $A_y$

Substituting the primitive form into the expression for $A_y$ and
taking its characteristic polynomial,

$$\det\!\left(A_y - \lambda I\right) \;=\; (v - \lambda)^{2}\bigl[(v - \lambda)^{2} - c^{2}\bigr], \qquad c = \sqrt{\gamma p / \rho},$$

giving eigenvalues

$$\{\lambda_k^{(y)}\}_{k=0}^{3} \;=\; \{\,v - c,\ v,\ v,\ v + c\,\}. \quad (\text{A4-eigvals-y})$$

This mirrors §A3 with $u \leftrightarrow v$. Rotational covariance
means every y-direction eigensystem result in the rest of the book
is obtained from the x-direction result of §A3 by the single
substitution $u \leftrightarrow v$.

## Verification checkpoints

The kernel expresses A4 as a permutation of arguments at the call
site, not as an explicit matrix multiplication. Consequently the
correctness of the A4 identity is verified through a kernel-level
equivalence test: for any admissible state $\mathbf{U}$ and a pair
of neighbouring states $(\mathbf{U}_L, \mathbf{U}_R)$ that differ
only in their tangential component,

1. **y-sweep swap-invariance.** Call `d_lmhllc` with
   $(u, v) = (u_0, v_0)$ and then again with the permuted argument
   binding representing $(u, v) = (v_0, u_0)$. The returned flux
   components, after the reciprocal permutation on the output side,
   must be bitwise identical. This is the operational form of
   $\mathbf{F}_y(\mathbf{U}) = R \mathbf{F}_x(R \mathbf{U})$ at the
   kernel level.

2. **Isotropic smoke IC.** Initialise a 2D Euler solution rotated
   by 90°; evolve for $N$ steps under both x-sweep and y-sweep
   alone and verify that the evolved states are related by $R$ to
   machine precision.

Test location: `tests/test_strang_hllc.cu`, §A4 smoke block.

Failure of (1) indicates an index-mismatch bug in the y-sweep
kernel call (the kind of bug that an off-by-one in `d_wL[k0*4+1]`
versus `d_wL[k0*4+2]` would introduce). Failure of (2) indicates a
deeper asymmetry in the solver — most likely in the ghost-cell BC
(§B5), not in the Riemann solver itself, but A4 is the
identification the test relies on.
