# A3. Flux Jacobian $\mathsf{A} = \partial\mathbf{F}/\partial\mathbf{U}$ and its 7-wave eigensystem

> **sympy script:** `scripts/a3_flux_jacobian_eigensystem.py`
> **generated LaTeX:** `output/a3_flux_jacobian_eigensystem.latex.tex`
> **symbolically verified:** hydro projection (25 entries), discriminant
> identities I1 & I2.
> **numerically verified (20 trials × 3 γ):** eigenvalues match
> closed-form to $\le 5\times10^{-15}$; $\mathsf{A}\mathbf{r} -
> \lambda\mathbf{r}$ residuals $\le 6\times10^{-15}$.
> **code checkpoints (future):**
> `athena_mhd_kernels.cu::d_mhd_eigenvalues_primitive`,
> `d_mhd_right_eigenvectors`.

## Working frame

The eigenanalysis is done in **primitive form** in 1D along $\hat{x}$.
Primitive 7-vector:

$$\mathbf{W} = (\rho,\ v_x,\ v_y,\ v_z,\ B_y,\ B_z,\ p)^{\mathrm{T}}.$$

$B_x$ is **not** an evolution variable in 1D: $\partial_x B_x = \nabla\cdot\mathbf{B} = 0$
forces $B_x = \text{const}$ across any $x$-interface. The 1D MHD system
therefore has **seven** propagating waves, not eight; the eighth mode
of the 3D system is the divergence-cleaning wave handled separately
by CT (§A5).

## Primitive-form Jacobian $\mathsf{A}_W$

$$\partial_t\mathbf{W} + \mathsf{A}_W(\mathbf{W})\,\partial_x\mathbf{W} = \mathbf{0},$$

$$\mathsf{A}_W =
\begin{bmatrix}
v_x & \rho & 0 & 0 & 0 & 0 & 0 \\
0 & v_x & 0 & 0 & B_y/\rho & B_z/\rho & 1/\rho \\
0 & 0 & v_x & 0 & -B_x/\rho & 0 & 0 \\
0 & 0 & 0 & v_x & 0 & -B_x/\rho & 0 \\
0 & B_y & -B_x & 0 & v_x & 0 & 0 \\
0 & B_z & 0 & -B_x & 0 & v_x & 0 \\
0 & \gamma p & 0 & 0 & 0 & 0 & v_x
\end{bmatrix}. \quad (\text{A3-jacobian})$$

**Hydrodynamic sanity check.** Setting $B_x = B_y = B_z = 0$ and
projecting onto the $(\rho, v_x, v_y, v_z, p)$ subspace recovers the
standard 5×5 hydrodynamic primitive Jacobian. sympy verifies all 25
entries of the projection match the hydro reference.

## Characteristic speeds

Define the four reference speeds:

$$c_{s_0}^{2} \equiv \gamma p / \rho,\qquad
c_{Ax}^{2} \equiv B_x^{2}/\rho,\qquad
c_{A\perp}^{2} \equiv (B_y^{2}+B_z^{2})/\rho,\qquad
c_{A}^{2} \equiv c_{Ax}^{2} + c_{A\perp}^{2}.$$

The fast and slow magnetosonic speeds are the two positive roots of

$$c_{f,s}^{2} = \tfrac{1}{2}\!\left[(c_{s_0}^{2} + c_{A}^{2}) \pm
\sqrt{(c_{s_0}^{2}+c_{A}^{2})^{2} - 4 c_{s_0}^{2} c_{Ax}^{2}}\right]. \quad (\text{A3-cfs})$$

**Discriminant identities (HLLD-stable forms).** The closed form
(A3-cfs) loses ULPs when $c_{A\perp} \to 0$ or $c_{s_0} \to 0$. The
HLLD kernel uses the two equivalent identities

$$\boxed{
c_f^{2} + c_s^{2} = c_{s_0}^{2} + c_{A}^{2},
\qquad
c_f^{2}\cdot c_s^{2} = c_{s_0}^{2}\cdot c_{Ax}^{2},
} \quad (\text{A3-discriminant})$$

which sympy verifies symbolically (both reduce to $0$ under `simplify`).
With these two identities, $c_f$ and $c_s$ can be recovered from
$(c_{s_0}^{2}, c_{A}^{2}, c_{Ax}^{2})$ without ever forming the
root-of-difference $c_A^2 - c_{s_0}^2$.

## The seven wave speeds

$$\boxed{
\{\lambda_k\}_{k=1}^{7} =
\{\,v_x - c_f,\ v_x - c_{Ax},\ v_x - c_s,\ v_x,\
v_x + c_s,\ v_x + c_{Ax},\ v_x + c_f\,\}.
} \quad (\text{A3-wave-speeds})$$

Here $c_{Ax} \equiv B_x/\sqrt{\rho}$ carries the sign of $B_x$ — this
convention lets us write the Alfvén speed once without the $s=\mathrm{sign}(B_x)$
factor that appears in Stone+08 eigenvector formulas. The other
convention (unsigned $c_{Ax} = |B_x|/\sqrt{\rho}$ + $s$ in every eigenvector
entry) is equivalent; we choose signed-$c_{Ax}$ here because it keeps
the 7-wave spectrum contiguous for all sign($B_x$).

## Numerical verification of the spectrum

sympy's `simplify()` does not handle nested radicals
$\sqrt{a \pm \sqrt{b}}$ reliably; the fast / slow eigenvector residuals
cannot be reduced to zero purely symbolically. This is a known limit —
Stone+08 Appendix B, Roe & Balsara 1996, and the Athena++ source all
fall back to **numerical random-sample verification** (the "Roe check")
for the eigensystem. We do the same:

1. Draw 20 random physically admissible states
   $(\rho, p, v_x, B_x, B_y, B_z)$ with $\rho, p > 0$,
   $B_x \ne 0$, $|B_\perp| \ne 0$;
2. For each of three ratios of specific heats $\gamma \in \{5/3, 7/5, 4/3\}$,
   diagonalise $\mathsf{A}_W$ numerically with `numpy.linalg.eig`;
3. Sort the seven numerical $\lambda_k$ and compare to the sorted
   closed-form spectrum (A3-wave-speeds).

Results over 60 trials:

$$\max|\lambda_{\text{closed}} - \lambda_{\text{numerical}}| < 5\times10^{-15},
\qquad
\max\|\mathsf{A}\mathbf{r} - \lambda\mathbf{r}\|_{\infty} < 6\times10^{-15}.$$

The residuals are at the `double`-precision floor for matrix
eigenvalue computation on a well-conditioned $7\times7$ matrix —
indistinguishable from exact.

## Closed-form right-eigenvectors

The closed forms come from Stone & Gardiner 2008 Appendix B Eqs.
(B11)–(B13); they are reproduced verbatim in the Athena++ source
(`src/eos/adiabatic_mhd.cpp::LRMHDWaves`). We document them here for
the kernel implementation; their correctness is pinned to the
community verification in Stone+08 Fig. 28–30.

Define the **amplitude-normalisation coefficients**

$$\alpha_f^{2} = \frac{c_{s_0}^{2} - c_s^{2}}{c_f^{2} - c_s^{2}},\qquad
\alpha_s^{2} = \frac{c_f^{2} - c_{s_0}^{2}}{c_f^{2} - c_s^{2}},
\qquad \alpha_f^{2} + \alpha_s^{2} = 1,$$

and the **direction coefficients**

$$\beta_y = \frac{B_y}{\sqrt{B_y^{2}+B_z^{2}}},\qquad
\beta_z = \frac{B_z}{\sqrt{B_y^{2}+B_z^{2}}},\qquad
\beta_y^{2} + \beta_z^{2} = 1,$$

and $s = \mathrm{sign}(B_x)$.

### Entropy (contact) wave — $\lambda_4 = v_x$

$$\mathbf{r}_{\mathrm{entropy}} = (1,\ 0,\ 0,\ 0,\ 0,\ 0,\ 0)^{\mathrm{T}}.$$

Pure density perturbation, no velocity, pressure, or magnetic-field
change. This is literally what "contact discontinuity" means in MHD —
it propagates unchanged at the flow speed.

### Alfvén waves — $\lambda_{2,6} = v_x \mp c_{Ax}$

$$\mathbf{r}_{A\pm} = \left(\,0,\ 0,\ \mp s\beta_z,\ \pm s\beta_y,\
-\beta_z\sqrt{\rho},\ \beta_y\sqrt{\rho},\ 0\,\right)^{\mathrm{T}}.$$

Transverse velocity and $B$-field perturbation, no density or pressure
change. This is the canonical transverse electromagnetic wave of MHD.

### Fast magnetosonic waves — $\lambda_{1,7} = v_x \mp c_f$

$$\mathbf{r}_{f\pm} = \left(\,
\rho\alpha_f,\
\pm\alpha_f c_f,\
\mp s\alpha_s c_s\beta_y,\
\mp s\alpha_s c_s\beta_z,\
\alpha_s c_{s_0}\sqrt{\rho}\beta_y,\
\alpha_s c_{s_0}\sqrt{\rho}\beta_z,\
\alpha_f\gamma p
\,\right)^{\mathrm{T}}.$$

Longitudinal compression plus in-phase perpendicular-$B$ compression.

### Slow magnetosonic waves — $\lambda_{3,5} = v_x \mp c_s$

$$\mathbf{r}_{s\pm} = \left(\,
\rho\alpha_s,\
\pm\alpha_s c_s,\
\pm s\alpha_f c_f\beta_y,\
\pm s\alpha_f c_f\beta_z,\
-\alpha_f c_{s_0}\sqrt{\rho}\beta_y,\
-\alpha_f c_{s_0}\sqrt{\rho}\beta_z,\
\alpha_s\gamma p
\,\right)^{\mathrm{T}}.$$

Longitudinal compression plus out-of-phase perpendicular-$B$ compression.

## Degenerate-limit eigenvectors

When $|\mathbf{B}_\perp| = 0$ (pure longitudinal $\mathbf{B}$), the
$\beta_y, \beta_z$ definitions above divide by zero. Stone+08
Eq. (B17)-(B20) gives replacement eigenvectors for this case. We do
not verify these here — they are documented in the Athena++ source
and must be implemented as branch-conditionals in the kernel.

## ✅ Verification checkpoint (to be wired)

The future

```
tests/test_athena_mhd_eigensystem.cu
```

must:

1. For 100 random admissible states, evaluate
   `d_mhd_eigenvalues_primitive` and verify each returned $\lambda_k$
   agrees with (A3-wave-speeds) to $10^{-12}$ absolute.
2. For each state, evaluate `d_mhd_right_eigenvectors` and verify
   $\|\mathsf{A}_W(\mathbf{W})\,\mathbf{r}_k - \lambda_k\mathbf{r}_k\|_{\infty}
   < 10^{-12}$ for all $k \in \{1,\dots,7\}$.
3. Pressure-positivity stress test: states with $\beta_{\mathrm{plasma}} < 0.01$,
   verify no NaN / no overflow in $\alpha_f$, $\alpha_s$.
4. Degenerate-limit coverage: states with $|\mathbf{B}_\perp|/|\mathbf{B}| <
   10^{-10}$, verify the degenerate-limit branch is taken and produces
   non-zero eigenvectors.

This eigenanalysis is the algebraic backbone of HLLD (§A4); any
regression here is a direct bug in every Riemann flux computation.
