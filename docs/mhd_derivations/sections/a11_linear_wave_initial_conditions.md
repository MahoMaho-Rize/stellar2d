# A11. Linear MHD wave initial conditions (Stone+08 Tab 1)

> **sympy script:** `scripts/a11_linear_wave_initial_conditions.py`
> **verified:** with $\rho_0=1, p_0=1/\gamma, \mathbf{B}_0=(1,\sqrt{2},1/2), \gamma=5/3$
> the wave speeds are $c_f=2$, $c_s=1/2$, $c_{Ax}=1$, $c_{s_0}=1$, all
> matching Stone+08 Table 1; the discriminant identities
> $c_f^2 + c_s^2 = c_{s_0}^2 + c_A^2$ and $c_f^2 c_s^2 = c_{s_0}^2 c_{Ax}^2$
> are sympy-verified.
> **code checkpoints:**
> `tests/test_athena_mhd_linear_wave_convergence.cu`;
> `tst/test_athena_mhd/test_linwave.py` (Python harness analogous to
> the existing `tst/test_ale2/test_linwave.py`).

## Why this is the "gold-standard" convergence test

Linear wave advection has a known exact solution: after one period
the perturbation returns to its initial shape, and any deviation is
**purely numerical**. The L¹ norm of the difference, sampled at
multiple resolutions, directly measures convergence slope and
diagnoses accuracy loss of the scheme.

A scheme that claims 2nd order but gives slope 1.3 in the fast-wave
test has a broken reconstruction, broken HLLD, or broken CT — and
this test finds that before `athena_mhd` touches a shock tube.

## Background (Stone+08 Table 1)

$$\rho_0 = 1,\quad p_0 = \frac{1}{\gamma},\quad \mathbf{v}_0 = \mathbf{0},\quad
\mathbf{B}_0 = \left(1,\ \sqrt{2},\ \tfrac{1}{2}\right),\quad \gamma = \tfrac{5}{3}.$$

**Derived wave speeds** (all sympy-verified):

$$c_f = 2,\quad c_{Ax} = 1,\quad c_s = \tfrac{1}{2},\quad c_{s_0} = 1.$$

## Perturbation

For mode $k$ with eigenvector $\mathbf{r}_k$ (from §A3) and amplitude
$A = 10^{-6}$:

$$\mathbf{W}(x, 0) = \mathbf{W}_0 + A\,\mathbf{r}_k\,\cos\!\left(\frac{2\pi x}{L}\right),\quad x\in[0, L].$$

## Periods on an $L$-periodic box

$$T_f = L/2,\quad T_A = L,\quad T_s = 2L.$$

## Entropy-mode exception

The entropy eigenvalue is $\lambda_\text{ent} = v_{0,x}$ which is $0$
at our base state. To make the entropy-wave convergence test
meaningful, **choose $v_{0,x} = 1$ for the entropy-mode case only**;
then $T_\text{ent} = L$. Stone+08 does the same (Sec 6.1).

## $\nabla\cdot\mathbf{B} = 0$ at IC (automatic)

In the 1D MHD system (§A3) $B_x$ is a parameter, not an evolution
variable; all seven right-eigenvectors live in
$(\rho, v_x, v_y, v_z, B_y, B_z, p)$ and do not perturb $B_x$. For a
plane wave along $\hat{x}$, $\nabla\cdot\delta\mathbf{B} = ik_x\delta B_x = 0$
automatically. **No extra projection step needed at IC.**

## Expected error

For a VL2 + HLLD + CT code (§A7, §A4, §A5),

$$\varepsilon_{L^1}(k, \Delta x) = C_k\,A\,(\Delta x / L)^{2},$$

with $C_k$ an $\mathcal{O}(1)$ mode-dependent constant. Convergence
slope measured across $\Delta x = L/64, L/128, L/256$ should fall in
$[1.9, 2.1]$ for all 7 modes.

## Practical test protocol

1. For each of the 7 modes:
   - Run the solver at resolutions $N \in \{64, 128, 256\}$.
   - Evolve for one period $T_k$.
   - Compute $\varepsilon_{L^1}(N) = \frac{1}{N}\sum_j |W_j^{N} - W_j^{\text{exact}}|$.
2. Fit slope: $\log\varepsilon \propto p \log\Delta x$.
3. Accept if $|p - 2| < 0.1$.

## ✅ Verification checkpoint

`tests/test_athena_mhd_linear_wave_convergence.cu` — single test that
cycles through 7 modes × 3 resolutions × 1 period; outputs a
21-row CSV; asserts slope ∈ [1.9, 2.1] per mode.

## Extension: 3D linear wave

A 3D linear wave IC is constructed identically, substituting
$k_x \to \mathbf{k}$ and rotating the eigenvector by the same angle.
Stone+08 §6.2 does the oblique test at $\mathbf{k} = (1,2,2)$. This
stresses dimensional coupling. Reserved for post-MVP; not included
in the initial test suite.
