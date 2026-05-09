# F1. Oblique linear MHD wave: rotated eigenvectors

> **sympy script:** `scripts/f1_oblique_linwave.py`
> **verified:** spectrum invariance under $B_y \leftrightarrow B_z$ split
> when $|B|, B\cdot\hat{\mathbf{k}}, \rho, p$ are held fixed
> (20 random trials, max err $4.9\times 10^{-15}$); solenoidal
> constraint $\mathbf{k}\cdot\delta\mathbf{B} = 0$ for rotated
> eigenvector; $c_f^2$ reduces to §A3 form at $\theta = 0$.
> **code checkpoints:**
> `AthenaMHDSolver::init_linear_wave_oblique` (to be added);
> `tests/test_athena_mhd_linwave_oblique.cu` (A1 test).

## Motivation

The §A3 MHD eigensystem is derived for 1D propagation along $\hat{\mathbf{x}}$.
For a 2D convergence test with wave-vector $\mathbf{k} = (k_x, k_y)$ —
Stone+08 §6.2 uses $\mathbf{k} \cdot \mathbf{L} = (2, 1)$ giving
$\theta = \arctan 2 \approx 63.4°$ on a $2\times 1$ domain — we need
the eigenvector *rotated* to the oblique direction.

This is the natural Phase A1 test: if the 1D linwave convergence (§A11)
passes at $p \approx 2$ but the 2D oblique test fails, the bug is
**purely in the x / y flux coupling** (i.e., the VL2 corrector wraps
or the CT corner-EMF averaging). No such bug is caught by 1D-only
tests.

## Rotation rule for primitive-form eigenvector

The primitive 7-vector eigenvector in the $\hat{\mathbf{x}}$-frame
from §A3:

$$\mathbf{r} = (\delta\rho,\ \delta v_x,\ \delta v_y,\ \delta v_z,\
\delta B_y,\ \delta B_z,\ \delta p)^{\mathrm{T}}$$

(with $\delta B_x = 0$: the $B_x$ component is not a wave variable in
1D, see §A3 discussion).

In the rotated frame with $\hat{\mathbf{k}} = (\cos\theta, \sin\theta, 0)$,
the vector components transform under $R(\theta)$:

$$\boxed{\begin{pmatrix}\delta v_x' \\ \delta v_y'\end{pmatrix}
= R(\theta)\begin{pmatrix}\delta v_x \\ \delta v_y\end{pmatrix},\qquad
\begin{pmatrix}\delta B_x' \\ \delta B_y'\end{pmatrix}
= R(\theta)\begin{pmatrix}0 \\ \delta B_y\end{pmatrix}.} \quad (\text{F1-rotation})$$

$\delta\rho, \delta v_z, \delta B_z, \delta p$ are scalars — invariant
under the z-axis rotation. Explicitly:

$$\delta B_x' = -\sin\theta\,\delta B_y,\qquad
\delta B_y' = +\cos\theta\,\delta B_y. \quad (\text{F1-B-rotation})$$

## Solenoidal constraint check

The rotated $\delta\mathbf{B}$ must still satisfy $\nabla\cdot\delta\mathbf{B} = 0$,
i.e., for a plane wave $\mathbf{k}\cdot\delta\mathbf{B} = 0$.

$$\mathbf{k}\cdot\delta\mathbf{B} = k_0\bigl(\cos\theta\cdot(-\sin\theta\,\delta B_y)
+ \sin\theta\cdot\cos\theta\,\delta B_y\bigr) = 0.$$

Sympy-verified symbolically. This is the fundamental reason the
rotation works: $\delta\mathbf{B}$ in the unrotated frame is
perpendicular to $\hat{\mathbf{x}}$ (via $\delta B_x = 0$), and the
rotation preserves orthogonality to the rotated axis.

## Wave-speed formula in the oblique frame

Under rotation, $c_{Ax}$ in the §A3 formula must be replaced by the
Alfvén speed component along the wave direction:

$$\boxed{c_{A,k} \equiv (\mathbf{B}\cdot\hat{\mathbf{k}})/\sqrt{\rho}.} \quad (\text{F1-cAk})$$

Then the fast-magnetosonic speed remains:

$$c_f^2 = \tfrac{1}{2}\bigl[(c_{s_0}^2 + c_A^2) + \sqrt{(c_{s_0}^2 + c_A^2)^2 - 4\,c_{s_0}^2\,c_{A,k}^2}\bigr],$$

with $c_A^2 = |B|^2/\rho$ unchanged (it is the *total* Alfvén speed,
not projected). Sympy verification: setting $\theta = 0$ (unrotated
frame) reduces (F1-cAk) to $c_{Ax} = B_x/\sqrt{\rho}$ as in §A3.

## Numerical verification

Random 20-state numerical check: given $(\rho, p, |\mathbf{B}|, B\cdot\hat{\mathbf{k}})$
fixed, the 7-wave spectrum is identical for *any* decomposition of
$\mathbf{B}$ into $B_y, B_z$ components. This confirms the spectrum
depends only on the invariants $(|B|, B\cdot\hat{\mathbf{k}}, \rho, p)$
— equivalently, it is rotationally invariant around $\hat{\mathbf{k}}$.

Max eigenvalue error over 20 trials: $4.9\times 10^{-15}$.

## Stone+08 §6.2 oblique-convergence setup

1. Domain $L_x = 2$, $L_y = 1$, fully periodic.
2. Wave vector $\mathbf{k} = 2\pi(1, 2)/L$ pointed diagonally;
   the wave crosses the domain in one period.
3. Background state from §A11:
   $\rho_0 = 1, p_0 = 1/\gamma, \mathbf{v}_0 = 0, \mathbf{B}_0 = (1, \sqrt{2}, 1/2), \gamma = 5/3$.
4. For each of the 4 modes (fast, Alfvén, slow, entropy):
   - Compute unrotated eigenvector $\mathbf{r}$ from §A3.
   - Compute $\theta$ from $(k_x, k_y)$; rotate vector components of
     $\mathbf{r}$ per (F1-rotation).
   - Plant IC on the 2D grid: $\mathbf{W}(x, y, 0) = \mathbf{W}_0 +
     A\,\mathbf{r}_\mathrm{rotated}\,\cos(\mathbf{k}\cdot\mathbf{x})$ with $A = 10^{-6}$.
5. Evolve for one wave period $T = 2\pi / (\lambda\, |\mathbf{k}|)$ where
   $\lambda$ is the oblique wave speed.
6. Measure $L^1$ error: $\varepsilon_{L^1}(N) = \frac{1}{N_x N_y}\sum |\mathbf{W}^{n+1} - \mathbf{W}^0|$.

## Pass criteria (A1 test)

1. For each of 4 modes × 3 resolutions ($N = 32, 64, 128$),
   $\varepsilon_{L^1}(N)$ ∝ $N^{-p}$ with $p \ge 1.8$.
2. $\max_t |\nabla\cdot\mathbf{B}| < 10^{-10}$ throughout (CT lock
   under 2D oblique propagation, the stronger case than 1D).
3. Solver remains stable for all modes (entropy / Alfvén / fast / slow
   must all propagate without NaN for $N \in \{32, 64, 128\}$).

## Why this specifically tests 2D coupling

A bug localised to x-sweep or y-sweep alone will NOT manifest in 1D
linwave tests (§A11) because only one direction is exercised. It WILL
show up here if:

- **CT corner-EMF averaging** (GS05 §A5) has a wrong 4-point weight —
  the rotated $\delta B_y$ requires exact y-flux of $E_z^x$ and
  x-flux of $E_z^y$ contributions to cancel.
- **VL2 predictor-corrector** has mismatched $dt/2$ vs $dt$ between
  directions — the oblique wave accumulates directional phase error.
- **PLM slope limiter** has different logic in x vs y — the oblique
  wave tests both slopes simultaneously, while §A11 tests only one.

## ✅ Verification checkpoints

- `tests/test_athena_mhd_linwave_oblique.cu` — Phase A1 test,
  4 modes × 3 resolutions, locks the three pass criteria above.

Failure on A1 after §A11 passes indicates a specifically 2D-coupling
bug — isolate by rerunning 1D (§A11), 2D-aligned (rerun §A11 on
$L_x = 1, L_y = 1$ rotated 0°), 2D-oblique (this test).
