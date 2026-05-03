# 6. Assembled-operator time stepping

## 6.1 Principle

*We enforce discrete closure by constructing the time-stepping
operator $\mathsf M = \mathsf L^{-1}\mathsf R$ as a single matrix,
computed once per horizontal wavenumber at setup time and applied
as a per-wavenumber dense matrix-vector product at every RK4
substage.*

This construction replaces the sequential application of discrete
operators in the matrix-free scheme of Section 5 --- differentiation
by $\mathsf D$, pointwise multiplication by $\rho_0$ and $N^2$,
division by $\rho_0$, and pressure projection --- with the direct
application of the assembled operator $\mathsf M$, whose action on
any interior vector $V$ is exactly $\mathsf L^{-1}\mathsf R V$.
Since $\mathsf M$ is the same operator that the g-mode EVP
diagonalises, its action on an EVP eigenvector reduces to scalar
multiplication; no mode mixing can occur.

This modification preserves the spectral transform, the pressure
projection, and the time-stepping scheme; it inserts a single
matrix-vector product per horizontal wavenumber in place of the
factored operator sequence applied by the matrix-free scheme,
closing the operator-consistency gap of Proposition 1 at minimal cost.

## 6.2 Eigenmode preservation under assembled RK4

The assembled construction admits a direct statement as a discrete
eigenmode-preservation result for RK4 time stepping, which we state
as Proposition 2 below.  The result is an application of standard
RK4 accuracy analysis [Higham, *Accuracy and Stability of Numerical
Algorithms*, Ch. 3] to a diagonalisable operator; what the present
work contributes is the *assembled construction* that makes the
diagonalisation step available, not the RK4 analysis itself.  Let
$\mathsf L, \mathsf R \in \mathbb R^{N_{\mathrm{int}} \times N_{\mathrm{int}}}$
denote the interior-restricted SL-compatible matrices of (3.7) at a
fixed horizontal wavenumber $k_x$. Let $(\omega_n^2, V_n)$ be an
eigenpair of $\mathsf R V = \omega^2\,\mathsf L V$, and let
$\mathsf M = \mathsf L^{-1}\mathsf R$ with spectral decomposition
$\mathsf M = Q\,\Lambda\,Q^{-1}$,
$\Lambda = \mathrm{diag}(\omega_1^2, \dots, \omega_{N_{\mathrm{int}}}^2)$;
denote the basis condition number $\kappa(Q) = \|Q\|_2\|Q^{-1}\|_2$.

Write the linear second-order ODE
$\mathsf L\,\ddot V = -\mathsf R\,V$ as a first-order system
$\dot U = \mathsf A\,U$ with $U = (V, W)^\top$ and
$$
\mathsf A = \begin{pmatrix} 0 & I \\ -\mathsf M & 0 \end{pmatrix}
\in \mathbb R^{2N_{\mathrm{int}} \times 2N_{\mathrm{int}}}.
\tag{6.1}
$$

**Lemma A (spectral decomposition of $\mathsf A$).**
*For each $(\omega_n^2, V_n)$, $\mathsf A$ has conjugate
eigenpairs $\mathsf A\,U_n^\pm = \pm \mathrm i\omega_n\,U_n^\pm$
with $U_n^\pm = (V_n, \pm \mathrm i\omega_n V_n)^\top$, so
$\mathrm{span}\{U_n^+, U_n^-\}$ is a two-dimensional invariant
subspace of $\mathsf A$.*

Direct substitution gives
$\mathsf A (V_n, c V_n)^\top = (c V_n, -\omega_n^2 V_n)^\top$; the
eigenvalue equation $-\omega_n^2 = \lambda c$ with $c = \lambda$
solves to $\lambda = \pm\mathrm i\omega_n$.

**Lemma B (RK4 on the pure imaginary axis).**
*For $z = \mathrm i\theta$, $|\theta| \le 2\sqrt 2$, the RK4 stability
function $R_4(z) = 1 + z + z^2/2 + z^3/6 + z^4/24$ satisfies
$|R_4(\mathrm i\theta)|^2 = 1 - \theta^8/576 + \mathcal O(\theta^{10})$
and $\arg R_4(\mathrm i\theta) = \theta + \theta^5/120 + \mathcal O(\theta^7)$,
so $R_4(\mathrm i\theta) = \mathrm e^{\mathrm i\theta}(1 + \rho)$
with $|\rho| \le (\omega_n\Delta t)^5 / 120$.*

This is a direct expansion of the RK4 polynomial on the imaginary
axis and establishes (i) non-dissipation to order $\theta^8$,
(ii) fifth-order phase accuracy, (iii) $|R_4| \le 1$ within the
stability bound.

**Proposition 2 (eigenmode preservation under assembled RK4).**
*Let $V(0) = V_n$, $W(0) = 0$, and advance the first-order system
$\dot U = \mathsf A\,U$ by RK4 with step size $\Delta t$ satisfying
$\omega_{\max}\Delta t \le 2\sqrt 2$. Write
$V^k := (I, 0)\,U^k$ for the upper component. Then for every
integer $k \ge 0$,*
$$
\bigl\|V^k - V_n \cos(\omega_n k\Delta t)\bigr\|_2
\;\le\;
\frac{1}{120}\,(\omega_n\Delta t)^5\,k\,\|V_n\|_2
\;+\;
2\,k\,\epsilon_{\mathrm{mach}}\,\kappa(Q)\,\|V_n\|_2.
\tag{6.2}
$$
*The bound is independent of $N_y$, $\rho_0(y)$, and $N^2(y)$.*

**Proof.** By Lemma A, $\mathsf A$ restricted to
$\mathrm{span}\{U_n^+, U_n^-\}$ is $\mathrm{diag}(\mathrm i\omega_n, -\mathrm i\omega_n)$
on a suitable basis; the RK4 polynomial $R_4(\Delta t\,\mathsf A)$
acts componentwise by $R_4(\pm\mathrm i\omega_n\Delta t)$. The initial
state $U^0 = (V_n, 0)^\top$ decomposes as $\tfrac{1}{2}(U_n^+ + U_n^-)$,
so
$V^k = \tfrac{1}{2}\bigl[R_4(\mathrm i\omega_n\Delta t)^k + R_4(-\mathrm i\omega_n\Delta t)^k\bigr] V_n.$
By Lemma B, each factor is $\mathrm e^{\pm\mathrm i\omega_n\Delta t}(1+\rho_n)$
with $|\rho_n| \le (\omega_n\Delta t)^5/120$, giving the first term of
(6.2). The second term is the standard mixed forward-backward error
of RK4 in the $Q$-basis (see Higham, *Accuracy and Stability of
Numerical Algorithms*, Ch.3). The sum of both gives the bound. $\square$

**Corollary (matrix-free counterexample).**
*The bound (6.2) does not hold when $\mathsf M$ is replaced by any
factored matrix-free operator $\widetilde{\mathsf M}$ whose
eigenvectors differ from $\{V_n\}$ by an $\mathcal O(\|\Delta\mathsf M\|)$
perturbation, where
$\Delta\mathsf M = \mathsf M_{\mathrm{mf}} - \mathsf M_{\mathrm{asm}}$
is the operator difference of Proposition 1. The per-step leak is
then bounded below by $c(\rho_0)\|V_n\|_2\Delta t + \mathcal O(\Delta t^2)$
with $c(\rho_0) > 0$ the structural constant of Proposition 1,
independently of $N_y$ refinement.*

## 6.3 Results

Per-step deviation measurements on an eigenmode initial condition
under the assembled-operator scheme:

| Background | Matrix-free | Full-Galerkin $V$-space | Full-Galerkin $\varphi$-space |
|---|---|---|---|
| Boussinesq ($\rho = 1$, $N^2 = 1$) | $4.8 \times 10^{-18}$ | $3.2 \times 10^{-18}$ | $3.2 \times 10^{-18}$ |
| Lane--Emden $n = 3/2$                | $1.70 \times 10^{-5}$ | $5.1 \times 10^{-18}$ | $4.9 \times 10^{-18}$ |

*Table 6.1: Per-step deviation of a g-mode eigenvector under three
time-stepping constructions on the same CGL grid.  The two
Full-Galerkin columns use the assembled $\mathsf L^{-1}\mathsf R$ and
agree with each other regardless of which state variable
($V$ versus $\varphi = \rho_0 V$) is advanced, confirming that
Proposition 2's conclusion is intrinsic to the assembled operator, not
to the chosen state.  Setup: $N_y = 64$, $\Delta t = 10^{-4}$,
amplitude $10^{-8}$, 100 RK4 steps.*

Both Full-Galerkin columns reach $5 \times 10^{-18}$ --- machine
precision times the spectral condition number of $\mathsf L$.  On the
Boussinesq background ($\rho_0 \equiv 1$, where the Proposition~1 gap
$c(\rho_0)$ vanishes by construction) the matrix-free column
reaches the same round-off floor.  On Lane--Emden the matrix-free
column sits at $1.70 \times 10^{-5}$, thirteen orders of magnitude
above the assembled scheme, and is independent of $N_y$ across
$N_y = 32, 48, 64, \dots, 256$ (Section 5.2) --- the time-domain
signature of Proposition~1's resolution-independent operator gap.

The GPU implementation on a single consumer-grade accelerator reaches
$3 \times 10^{-15}$ per-step deviation at the production resolution
$(N_x, N_y) = (64, 64)$; the gap to the Python $5 \times 10^{-18}$
floor is accounted for by $\sim 10^{-14}$ round-trip loss in the
real-to-complex / complex-to-real FFT pair and
$N_y\,\epsilon_{\mathrm{mach}} \approx 10^{-14}$ accumulation in the
dense inversion used to form $\mathsf L^{-1}$.  Both are normal
double-precision behaviour and neither is a discretisation-level
concern at the physical amplitude scales of interest.  Implementation
details are given in Appendix A.3.

Long-time regression on the same $n_g = 1$ g-mode at $N_y = 64$ on
Lane--Emden $n = 3/2$, integrated with $\Delta t = 2\times 10^{-3}$
(corresponding to $\omega_1\Delta t \approx 3\times 10^{-3}$):

| Metric | Matrix-free | Assembled-operator scheme |
|---|---|---|
| Stable integration horizon | $\lesssim 1$ period | $\ge 300$ periods |
| Fourier peak $\omega/\omega_{\mathrm{EVP}} - 1$ (300 periods) | not defined --- solution blows up | $-2.0 \times 10^{-5}$ |
| Final eigenmode deviation (300 periods) | not defined | $2.8 \times 10^{-13}$ |

*Table 6.2: Long-time behaviour of the g-mode under the two
stepping strategies.  The matrix-free scheme diverges within a
single oscillation period on Lane--Emden $n = 3/2$: an exponential
instability triggered by the Proposition~1 operator gap, whose
small per-step energy injection compounds over the fast oscillatory
motion.  The assembled scheme remains stable for 300 periods at the
$10^{-13}$ deviation level, with Fourier-peak frequency drift
consistent with RK4 truncation.*

The Fourier-peak frequency error of $2.0 \times 10^{-5}$ for the
assembled scheme over 300 periods is dominated by the finite
resolution of the Fourier peak extraction rather than by the time
integrator; the mean phase-drift rate at this $\Delta t$ is at the
$10^{-7}$ level, consistent with Proposition 2's first-term prediction
$(\omega_n\Delta t)^5/120 \times \text{step count}$.

## 6.4 Summary

The assembled-operator scheme resolves the operator-consistency
floor of Section 5 by replacing the local surrogate
$\mathrm{diag}(1/\rho_0)$ with the correct global inverse $\mathsf L^{-1}$,
stored as a setup-time dense matrix and applied as a DGEMV per
wavenumber per substep.  The mathematical content is captured in
Proposition 2: under RK4 time stepping on the assembled first-order
system $\dot U = \mathsf A U$, any EVP eigenvector is preserved to
$\mathcal O(\epsilon_{\mathrm{mach}}\kappa(Q) + (\omega\Delta t)^5)$
per step, independent of $N_y$, $\rho_0$, or $N^2$.
The reference implementation reaches $5 \times 10^{-18}$; the GPU
port, $3 \times 10^{-15}$.  The bandwidth footprint is comparable to
the FFT pair and the device-memory overhead is
$\mathcal O(n_h\,N_{\mathrm{int}}^2)$, of order 1 MB at production
resolution, so the assembled scheme does not alter the performance
envelope of the pseudo-spectral code.

The construction is specific to the *linear* part of the evolution,
for which $\mathsf L^{-1}\mathsf R$ is time-independent.  Extending
it to nonlinear time stepping via operator splitting is the subject
of Section 7.
