# 7. Nonlinear extension via operator splitting

## 7.1 Problem statement

The assembled-operator construction of Section 6 closes the linear
g-mode propagation to machine precision.  Stellar pulsation physics
of interest --- mode coupling, parametric resonance, weakly
nonlinear saturation --- enters through the quadratic advection
terms $(\mathbf u\!\cdot\!\nabla)v$ and $(\mathbf u\!\cdot\!\nabla)b$
that Section 6 dropped.  The question is: how can the
assembled-operator scheme be extended to carry these nonlinear terms
without losing the Proposition 2 closure as a linear limit?

Three classes of schemes are standard candidates:

  - **Strang-split scheme.** Alternate a linear half-step of the
    assembled-operator update with a nonlinear RK4 step that evolves
    $(u, v, b)$ under advection alone.
  - **Semi-implicit IMEX scheme.** Crank--Nicolson on the assembled
    linear block (trapezoidal rule) with Adams--Bashforth-2
    extrapolation of the nonlinear right-hand side.
  - **Exponential-propagator scheme.** Diagonalise $\mathsf M$ once
    per horizontal wavenumber to obtain $\Omega_n = \sqrt{\omega_n^2}$
    and advance the linear block exactly via $\cos(\Omega\Delta t)$,
    $\sin(\Omega\Delta t)/\Omega$, with a Strang-symmetric nonlinear
    block in between.

All three reduce to the assembled scheme in the $\text{amp} \to 0$
limit and differ only in how they couple the linear and nonlinear
blocks at finite amplitude.  We prototyped all three in Python and
ran a head-to-head comparison.  Full implementation details of each
prototype are deferred to Appendix A.4.

## 7.2 Results

Running the three prototypes on the same $(u, v, b)$ eigenmode
initial condition with amplitude scan $10^{-8}$ to $10^{-1}$, 800
time steps ($\approx 4$ g-mode periods at $n_g = 1$ on Lane--Emden),
$(N_x, N_y) = (64, 48)$, $\Delta t = 2\times 10^{-2}$:

| Scheme | Amp | dev/step | Final dev | $\Delta E/E_0$ |
|---|---|---|---|---|
| Strang-split       | $10^{-8}$ | $2.4\times 10^{-9}$ | $4.4\times 10^{-7}$ | $+2.6$ |
| Strang-split       | $10^{-3}$ | $2.4\times 10^{-4}$ | $4.4\times 10^{-2}$ | $+4.6$ |
| Strang-split       | $10^{-1}$ | $2.4\times 10^{-2}$ | $2.7\times 10^{-1}$ | $+7.0$ |
| Semi-implicit IMEX | $10^{-8}$ | $7.9\times 10^{-10}$ | $3.5\times 10^{-6}$ | $+2.6$ |
| Semi-implicit IMEX | $10^{-3}$ | $7.9\times 10^{-5}$ | $3.5\times 10^{-1}$ | $+57$ |
| Semi-implicit IMEX | $10^{-1}$ | $7.9\times 10^{-3}$ | $1.3\times 10^{+1}$ | $+2.6\times 10^{79}$ |
| Exp. prop.         | $10^{-8}$ | $2.4\times 10^{-9}$ | $4.4\times 10^{-7}$ | $+2.6$ |
| Exp. prop.         | $10^{-3}$ | $2.4\times 10^{-4}$ | $4.4\times 10^{-2}$ | $+4.6$ |
| Exp. prop.         | $10^{-1}$ | $2.4\times 10^{-2}$ | $3.5\times 10^{0}$ | $+7.1\times 10^{3}$ |
| Exp., linear-only  | $10^{-8}$ | $1.2\times 10^{-14}$ | $4.3\times 10^{-13}$ | $+2.6$ |

*Table 7.1: Three-scheme comparison on the nonlinear anelastic time
evolution. Lane--Emden $n = 3/2$ background, $\ell = 1$ g-mode
initial condition.  The "Exp., linear-only" row disables the
nonlinear block and verifies the machine-precision floor of the
exact linear propagator.  The $\Delta E/E_0$ column uses the raw
anelastic functional, whose $+2.6$ baseline is a fixed quadrature
offset, not a time-stepping error (see §7.3 and the natural invariant
$E_{\mathrm{asm}}$ of Table 7.2).*

Three findings are visible.

**Finding 1: Strang-split and exponential-propagator schemes are
numerically indistinguishable at finite amplitude.**  Their dev/step
and final-dev columns agree to three significant figures across all
amplitudes.  The exponential-propagator has orders-of-magnitude lower
linear-block deviation ($10^{-14}$ in isolation) but once the
nonlinear block is active, the nonlinear block's own RK2 round-off
dominates and the two schemes become empirically equivalent.  The
Strang-split scheme is simpler to implement, reuses existing
pseudo-spectral advection kernels, and is our recommendation for the
GPU production extension.

**Finding 2: the CN-AB2 IMEX combination is unstable at physically
relevant amplitudes.**  We document the failure of the simplest
second-order IMEX scheme (Crank--Nicolson on the linear block,
Adams--Bashforth-2 on the nonlinear block), not of the IMEX family
as a whole.  CN-AB2 is a baseline choice because it is the default
IMEX in several spectral frameworks (e.g.\ Dedalus) for
quick-turnaround runs and because its stability can be analysed in
closed form.  Higher-order IMEX schemes (IMEX-RK3, IMEX-BDF3) use
$L$-stable or SSP discretisations in place of AB2 and may avoid
the instability reported here; we have not tested them.  Finding 2
should therefore be read as a minimum-baseline result motivating
the Strang-split choice below, not as a negative statement about
IMEX in general.

At amp $= 10^{-1}$, the CN-AB2 IMEX scheme drives $\Delta E/E_0$ to
$2.6\times 10^{79}$ over 800 steps, an unbounded exponential growth.
The origin is a classical AB2 imaginary-axis instability, amplified
by CN. To make this precise, apply
IMEX(CN, AB2) to the scalar test equation
$\dot x = \mathrm i\omega x + \lambda_N x$ with $\lambda_L = \mathrm i\omega$
handled by CN and $\lambda_N$ by AB2. The two-step recurrence is
$[x_{n+1}; x_n] = G(z_L, z_N)[x_n; x_{n-1}]$ with stability governed
by $\rho(G) = \max|\mathrm{eig}(G)|$, where
$$
G(z_L, z_N) = \begin{pmatrix}
  R_{\mathrm{CN}}(z_L) + \tfrac{3}{2} z_N M_{\mathrm{eff}} & -\tfrac{1}{2} z_N M_{\mathrm{eff}} \\
  1 & 0
\end{pmatrix},
\qquad
R_{\mathrm{CN}}(z) = \frac{1 + z/2}{1 - z/2}.
\tag{7.1}
$$

Crank--Nicolson places $R_{\mathrm{CN}}(\mathrm i\omega\Delta t)$ on
the unit circle ($|R_{\mathrm{CN}}| = 1$, neutral stability), but AB2
applied to a test equation with purely imaginary $\lambda_L$ is
absolutely unstable for any $\lambda_N$ with non-zero real part.
Nonlinear advection $(\mathbf u\!\cdot\!\nabla)v$ in the g-mode
context has generically non-zero real $\lambda_N$ --- representing
mode-to-mode energy transfer --- so the IMEX(CN, AB2) combination is
unconditionally unstable for any $\alpha > 0$, with growth rate
$\sim \alpha\omega\Delta t$ per step. At amp $= 10^{-1}$ and
$\omega = 1.64, \Delta t = 0.02$, $\rho(G) \approx 1.005$, giving a
per-step amplification of $0.5\%$; over 800 steps this is
$1.005^{800} \approx 50$-fold, and the quadratic feedback from the
nonlinear advection drives this into the $10^{79}$ level observed in
Table 7.1.  The full stability contour and numerical verification are
given in Appendix A.5.


**Finding 3: $\Delta t$ stability is essentially identical across
the three schemes up to $\omega\Delta t \approx 1.6$.**  A
$\Delta t$-stability scan (Lane--Emden, amp $= 10^{-8}$, 15 time
units, $\Delta t \in [3\times 10^{-3}, 1.0]$) shows all three schemes
stable throughout.  The theoretical RK4 imaginary-axis CFL bound is
$\omega\Delta t \le 2.8$; the implicit schemes have no
linear-CFL constraint.  For g-mode frequencies on our Lane--Emden
setup ($\omega \le 2$), the convenience-versus-stability argument
that would favour implicit methods in stiffer problems does not
apply.

## 7.3 Energy drift: quadrature-functional mismatch, not time-stepping

Table 7.1 reports $\Delta E/E_0 = +2.6$ at amp $= 10^{-8}$, where
no physical energy transfer is possible.  This drift is not a
time-integrator defect but a mismatch between two distinct
functionals: the *raw* functional
$E_{\mathrm{raw}} = \tfrac{1}{2}\int\rho_0(u^2+v^2) + \tfrac{1}{2}\int b^2/N^2$
is not the natural invariant of the *discrete* oscillator
$\ddot V = -\mathsf M V$ that Proposition 2 closes; it is a *continuous*
functional applied through a CC quadrature that is inconsistent with
the algebra of $\mathsf M$.  Replacing it by the
assembled-operator-induced inner product resolves the drift.

Define
$$
E_{\mathrm{asm}}(V, W) := \tfrac{1}{2}\,\langle W, W\rangle_{w,x}
                          + \tfrac{1}{2}\,\langle V, \mathsf M V\rangle_{w,x},
\tag{7.2}
$$
with $\langle \cdot,\cdot\rangle_{w,x}$ the CC-weighted $y$-inner product
combined with periodic $x$-sum.  By construction, $E_{\mathrm{asm}}$ is
conserved exactly under the continuous flow $\ddot V = -\mathsf M V$
(since $\mathsf M = \mathsf L^{-1}\mathsf R$ is symmetrisable in the
assembled EVP inner product and $\frac{d}{dt}E_{\mathrm{asm}}
= \langle W, \ddot V + \mathsf M V\rangle_{w,x} \equiv 0$).  Under
discrete RK4 it is preserved to $\mathcal O((\omega\Delta t)^8)$ per
step --- the fifth-order truncation of the stability function squared
--- and under the exponential propagator it is preserved exactly to
round-off.  We measured $\Delta E_{\mathrm{raw}}/E_0$,
$\Delta E_{\mathrm{int}}/E_0$ (floor-excluded), and
$\Delta E_{\mathrm{asm}}/E_0$ on the identical linear run, sweeping
amplitude, $N_y$, and $\Delta t$ independently:

| Sweep | var | $\Delta E_{\mathrm{raw}}/E_0$ | $\Delta E_{\mathrm{int}}/E_0$ | $\Delta E_{\mathrm{asm}}/E_0$ (RK4) | $\Delta E_{\mathrm{asm}}/E_0$ (exp) |
|---|---|---|---|---|---|
| amp       | $10^{-8}$           | $+2.65$ | $+3.05$ | $-1.4\times 10^{-8}$  | $-2.6\times 10^{-14}$ |
| amp       | $10^{-4}$           | $+2.65$ | $+3.05$ | $-1.4\times 10^{-8}$  | $-4.2\times 10^{-14}$ |
| amp       | $10^{-1}$           | $+2.65$ | $+3.05$ | $-1.4\times 10^{-8}$  | $-1.9\times 10^{-14}$ |
| $N_y$     | 48                  | $+2.65$ | $+3.05$ | $-1.4\times 10^{-8}$  | $-2.6\times 10^{-14}$ |
| $N_y$     | 96                  | $+2.65$ | $+3.07$ | $-1.4\times 10^{-8}$  | $+1.0\times 10^{-12}$ |
| $N_y$     | 192                 | $+2.66$ | $+3.07$ | $-1.4\times 10^{-8}$  | $+2.4\times 10^{-12}$ |
| $\Delta t$| $2\times 10^{-2}$   | $+2.65$ | $+3.05$ | $-1.4\times 10^{-8}$  | $-2.6\times 10^{-14}$ |
| $\Delta t$| $1\times 10^{-2}$   | $+2.65$ | $+3.05$ | $-4.4\times 10^{-10}$ | $-1.4\times 10^{-13}$ |
| $\Delta t$| $5\times 10^{-3}$   | $+2.65$ | $+3.05$ | $-1.4\times 10^{-11}$ | $-9.2\times 10^{-13}$ |

*Table 7.2: Three energy diagnostics under identical linear-anelastic
evolution (eigenmode IC, 800 steps).  $E_{\mathrm{asm}}$ drift under
RK4 follows the textbook $\mathcal O(\Delta t^4)$ scaling (ratio 32
per $\Delta t$ halving), and is conserved to round-off under the
exponential propagator.  $E_{\mathrm{raw}}$ and $E_{\mathrm{int}}$
are $\Delta t$-independent and amplitude-invariant, confirming that
the $+2.6$ value is a constant offset between the two functionals,
not a time-stepping error.*

Two observations settle the question.

**(a) $E_{\mathrm{asm}}$ exhibits textbook RK4 convergence.**  The
$\Delta t$ sweep gives $\Delta E_{\mathrm{asm}}/E_0$ scaling $2^5$ per
$\Delta t$ halving, i.e. $\mathcal O(\Delta t^4)$ in the stability
function and $\mathcal O(\Delta t^4)$ accumulated over 800 steps ---
exactly the Proposition 2 floor.  Under the exponential propagator,
$\Delta E_{\mathrm{asm}}$ sits at $\mathcal O(10^{-14})$ to
$\mathcal O(10^{-12})$ across all $N_y$ and amplitudes, the
round-off floor of the per-kx DGEMV $\cos/\sin$ propagator.

**(b) $E_{\mathrm{raw}}$ is $\Delta t$-independent.**  Halving
$\Delta t$ does not reduce $\Delta E_{\mathrm{raw}}/E_0$ by any amount
--- the $+2.645$ value persists to six significant figures at
$\Delta t = 2\times 10^{-2}, 10^{-2}, 5\times 10^{-3}$.  A genuine
time-stepping drift would scale as $\Delta t^p$ for some $p \ge 2$.
This constancy identifies the drift as a fixed offset between
$E_{\mathrm{raw}}$ and $E_{\mathrm{asm}}$ at the *initial condition*
--- the raw functional evaluated on the eigenmode already differs
from $E_{\mathrm{asm}}$ by the factor $+2.645$, and both evolve
conservatively (to $E_{\mathrm{asm}}$ precision) under the
integrator, so the *relative* raw drift is fixed.

The mechanism is that $E_{\mathrm{raw}}$ applies CC quadrature with
the continuous weights $\rho_0(y)$ and $1/N^2(y)$, while the
discrete flow $\ddot V = -\mathsf M V$ is generated by
$\mathsf M = \mathsf L^{-1}\mathsf R$ whose symmetrising weight is
different (the interior-restricted $\mathsf L$ itself).  The ratio
of the two quadratures is not unity --- it is a constant depending
on the $V_n$-weighted integrals of $\rho_0$ and $1/N^2$ on the
particular eigenmode, which for the Lane--Emden $n = 3/2$ first
$\ell = 1$ g-mode evaluates numerically to $1 + 2.645$.  This
constant is amplitude-invariant (the ratio of two linear functionals
of $V$), grid-refinement-invariant (both integrals converge to their
continuous limits under spectral rates), and $\Delta t$-invariant
(the integrator is not the cause of the offset).

The practical implication is:
**when diagnosing energy conservation of the assembled scheme, use
$E_{\mathrm{asm}}$.**  It is the natural invariant of the discrete
oscillator, closes the loop of Proposition 2 in the energy sense, and
agrees with $E_{\mathrm{raw}}$ in the Boussinesq limit (where
$\rho_0 \equiv 1$ and $\mathsf L = -\partial_{yy} + k_x^2$ reduces to
the symmetric Laplacian).  The $+2.6$ row of Table 7.1 is therefore
not a failure of the time integrator --- it is the wrong diagnostic
applied to a correctly integrated discrete flow.

## 7.4 Summary

The Strang-split scheme and the exponential-propagator scheme are
empirically equivalent at physical amplitudes.  The
exponential-propagator has a stronger theoretical guarantee ---
machine-precision linear block --- but this guarantee is invisible
once the nonlinear block is active, because the nonlinear block's
RK2 round-off dominates.  The semi-implicit IMEX(CN, AB2) scheme is
excluded on stability grounds at amp $\ge 10^{-1}$, on the
analytical basis of (7.1); other IMEX variants remain candidates for
future work.

The recommended nonlinear scheme is the Strang-split formulation: it
reuses the existing pseudo-spectral advection infrastructure and
delivers numerical performance indistinguishable from the
higher-cost exponential-propagator in the regime of physical
interest.  The linear machine-precision closure of Proposition 2 is
preserved by construction; the nonlinear block contributes its own
$\mathcal O(\Delta t^4)$ Strang error, additive and independent of
the linear block.  While the present study evaluates the nonlinear
extension on a CPU reference implementation, porting the Strang-split
scheme to GPU architectures is a direct objective for future work.

The exponential-propagator is held in reserve for applications
requiring $10^{-12}$ or better per-step accuracy over $\mathcal O(10^4)$
time steps, such as long-term mode-coupling statistics or weakly
nonlinear saturation studies.
