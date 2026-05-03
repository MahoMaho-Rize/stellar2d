# Appendix A: Implementation details and auxiliary data

## A.1 Lane--Emden ODE integration and $\rho_{\mathrm{cut}}$ selection

The Lane--Emden background of Section 2.3 is obtained by integrating

$$
\frac{1}{\xi^2}\frac{d}{d\xi}\!\left(\xi^2\, \frac{d\theta}{d\xi}\right)
+ \theta^n = 0,
\qquad \theta(0) = 1,\quad \theta'(0) = 0,
\tag{A.1}
$$

to the first zero $\xi_1$ using an adaptive Runge--Kutta integrator
with event detection on $\theta = 0$.  For $n = 3/2$, $\xi_1 \approx 3.6538$;
for $n = 3$, $\xi_1 \approx 6.8969$.  The density profile is
$\rho_0(\xi) = \theta(\xi)^n$, mapped linearly to the CGL grid
$y \in [0, L_y]$ with $\xi \in [\xi_{\mathrm{lo}}, \xi_{\mathrm{hi}}]$ defined
by the $\rho_0 \ge \rho_{\mathrm{cut}}$ constraint.

**$\rho_{\mathrm{cut}}$ sensitivity.**  Varying the cutoff
$\rho_{\mathrm{cut}} \in \{0.01, 0.02, 0.05, 0.10\}$ shifts the first
ten g-mode frequencies of the assembled EVP on Lane--Emden $n = 3/2$:

| $\rho_{\mathrm{cut}}$ | $\omega(n_g=1)$ | $\omega(n_g=5)$ | $\Delta\omega/\omega_{\mathrm{ref}}$ |
|---|---|---|---|
| 0.01         | 1.964 | 0.715 | $+19.4\%$ |
| 0.02         | 1.857 | 0.671 | $+12.1\%$ |
| **0.05 (ref)** | **1.645** | **0.598** | **0** |
| 0.10         | 1.458 | 0.530 | $-11.4\%$ |

*Table A.1: $\rho_{\mathrm{cut}}$ sensitivity on the first g-mode
frequency for Lane--Emden $n = 3/2$.*

The cutoff sets the floor-region extent where $N^2 = 0$, which
narrows or widens the g-mode trap zone and shifts the spectrum
by $\pm 15\%$.  This is a parametric choice analogous to the
$T_{\mathrm{eff}}$ boundary-condition selection in stellar-atmosphere
models; it does *not* affect the operator-consistency arguments of
Sections 5 and 6, each of which holds pointwise in $\rho_{\mathrm{cut}}$.

## A.2 Spectral convergence: integer vs half-integer polytropic index

The convergence dichotomy stated in §3.5 follows from the boundary
regularity of the SL potential $W(y)$.  For integer polytropic
index $n \in \{1, 3\}$, $\theta \sim (\xi_1 - \xi)^1$ near $\xi_1$,
so $\rho_0 = \theta^n \sim (\xi_1 - \xi)^n$ is polynomial; $W(y)$ in
the Liouville-transformed variable is analytic on the closure of
the open interior with only a removable coordinate singularity.
Chebyshev collocation then converges exponentially in $N_y$: the
relative manufactured-Poisson error reaches $10^{-13}$ at
$N_y \approx 32$ and saturates at the Clenshaw--Curtis quadrature
floor.

For half-integer $n = 3/2$, $\rho_0 \sim (\xi_1 - \xi)^{3/2}$ has a
boundary branch point; $W(y)$ acquires an endpoint singularity
$\sim (L_y - y)^{-1/2}$.  The Chebyshev collocation error on such
profiles is bounded by the $C^k$ regularity of the target function
at the endpoints, giving $\mathcal O(N_y^{-2\sigma-1})$ algebraic
convergence with $\sigma = 1/2$, i.e. $\mathcal O(N_y^{-2})$.
Numerically, relative Poisson error reaches $10^{-6}$ at $N_y = 64$
and saturates near $10^{-7}$ at $N_y = 128$ due to combined
ill-conditioning in the SL eigendecomposition near the wall and
finite-precision arithmetic.

An alternative basis of the form $\rho_0^\alpha T_k$ with $\alpha$
matched to the fractional surface exponent $\sigma$ restores
exponential convergence on half-integer profiles [5, Ch. 17].  We
do not use this extension because the time-domain closure problem
(Sections 5, 6) is independent of the spatial floor once $N_y \ge 48$.

## A.3 CUDA implementation

The CUDA implementation of the assembled scheme uses cuFFT for
horizontal transforms, a custom per-wavenumber DGEMV kernel for the
$\mathsf M$-application, and host-side Gauss--Jordan for the
one-time construction of $\mathsf L^{-1}$.  The assembled matrices
$\{\mathsf M_{k_x}\}_{k = 1}^{n_h - 1}$ are packed into a contiguous
column-major slab of shape $(n_h, N_{\mathrm{int}}, N_{\mathrm{int}})$,
consuming $8\,n_h\,N_{\mathrm{int}}^2$ bytes; at production resolution
$(n_h, N_y) = (33, 64)$ this is $\approx 1$ MB VRAM.  The
$k_x = 0$ mode is zeroed as it carries no g-mode dynamics.

Per time substep, the state $v \in \mathbb R^{N_y \times N_x}$ is
transformed by a forward real-input FFT in $x$ to the spectral field
$\hat v \in \mathbb C^{N_y \times n_h}$.  For each $k_x$, the
$N_{\mathrm{int}}$-vector $\hat v_{\mathrm{int}}(k)$ is multiplied by
$\mathsf M_{k_x^{(k)}}$ via $n_h$ parallel DGEMVs ($N_{\mathrm{int}}^2$
FLOP each); wall entries are forced to zero.  An inverse FFT returns
$\mathsf M v$ to physical space.  Total arithmetic cost per RK4 step
is $4 n_h N_{\mathrm{int}}^2 \approx 5\times 10^5$ FLOP, dwarfed by the
four FFTs at $\mathcal O(N_x N_y \log N_x)$.

Observed performance on RTX 4080 SUPER (SM 8.9, CUDA 12.9):
$3\times 10^{-15}$ per-step deviation at $(N_x, N_y) = (64, 64)$,
approximately three decades above the Python prototype floor of
$5\times 10^{-18}$.  The gap is attributed to (i) $\sim 10^{-14}$
round-trip loss in the cuFFT real-to-complex / complex-to-real pair
and (ii) $N_y \epsilon_{\mathrm{mach}} \approx 10^{-14}$ accumulation
in the host-side Gauss--Jordan inversion, both consistent with
normal double-precision behaviour.

## A.4 Nonlinear scheme prototypes

The three prototypes of Section 7 share the same CGL grid,
$\mathsf L, \mathsf R$ assembly, and Lane--Emden background, differing
only in the time-integration wrapper.

- **Strang-split.** A Strang $(A, B, A)$ pattern per $\Delta t$,
  where $A$ is a half-step of the assembled linear RK4 on $(V, W, B)$
  with $\dot B = -N^2 V$ linear, and $B$ is a full-step RK4 advection
  on $(v, b)$ with $u$ reconstructed from $v$ by continuity at each
  RK4 substage.

- **Semi-implicit IMEX.** Per time step: (i) extrapolate the
  nonlinear right-hand side via Adams--Bashforth-2
  $f_{\mathrm{nl}}^{n+1} \approx 1.5 f^n - 0.5 f^{n-1}$; (ii) solve the
  $2 N_{\mathrm{int}} \times 2 N_{\mathrm{int}}$ linear system
  $(I - \tfrac{\Delta t}{2}\mathsf A)\,U^{n+1} = (I + \tfrac{\Delta t}{2}\mathsf A)\,U^n
   + \Delta t\,[0; f_{\mathrm{nl}}^{\mathrm{AB2}}]$ per wavenumber,
  the left-hand matrix pre-factored at setup; (iii) update $b$ by
  trapezoidal rule linearly and AB2 in advection.

- **Exponential-propagator.** Diagonalise $\mathsf M = Q\Lambda Q^{-1}$
  per wavenumber at setup.  Per step: (i) exact linear half-step
  to $(V, W, B)$ via analytic integrals
  $\int_0^{\Delta t/2} V(\tau)\,d\tau$, extending to $B$ by
  $B \leftarrow B - N^2\int V$; (ii) RK2 midpoint for the nonlinear
  advection on $(v, b)$; (iii) second exact linear half-step.

## A.5 IMEX stability derivation

For the scalar test equation $\dot x = \mathrm i\omega x + \lambda_N x$
with $\lambda_L = \mathrm i\omega$ treated by CN and $\lambda_N$ by
AB2, the IMEX(CN, AB2) recurrence is
$[x_{n+1}; x_n] = G(z_L, z_N)[x_n; x_{n-1}]$ with $G$ given by (7.1).
The spectral radius $\rho(G)$ is the amplification factor; stability
requires $\rho(G) \le 1$.

A grid scan over $\omega\Delta t \in [0.01, 3]$ and effective
nonlinear coupling $\alpha \in [0, 1.5]$ with three coupling phases
(pure imaginary, pure real, and mixed $45^\circ$) finds:

- *Imaginary $\lambda_N$*: $\rho(G) \approx 1$ for all $\alpha$;
  neutrally stable.
- *Real $\lambda_N$*: $\rho(G) > 1$ for any $\alpha > 0$,
  per-step growth $\sim \alpha\omega\Delta t$; unconditionally
  unstable.
- *Mixed*: intermediate.

At the paper's settings ($\omega = 1.64$, $\Delta t = 0.02$,
amp = $10^{-1}$), the effective $\alpha$ of the $(u\cdot\nabla)v$
coupling yields $\rho(G) \approx 1.005$, giving 800-step
amplification $\sim 50$; the quadratic feedback inherent to the
nonlinear block amplifies this to the $10^{79}$ energy growth
observed in Table 7.1.

## A.6 $c(\rho_0, n_g, k_x)$ table

The Proposition 1 constant $c(\rho_0)$ of Section 5.4 is a function of
both radial mode index $n_g$ and horizontal wavenumber $k_x$, through
the $V_n$-weighted measure $d\mu$ in (5.1b).  Table A.2 gives the
direct measurements on Lane--Emden $n = 3/2$, $\rho_{\mathrm{cut}} =
0.05$, $N_y = 96$.  In every entry the measured shape gap and the
variance prediction (5.1b) agree to all displayed digits.

| $k_x / (2\pi/L_y)$ | $n_g = 1$ | $n_g = 2$ | $n_g = 3$ | $n_g = 5$ | $n_g = 10$ |
|---|---|---|---|---|---|
| 1  | 19.65 | 58.27 | 111.91 | 277.23 | 1054.2  |
| 2  | 47.37 | 117.80 | 192.63 | 385.43 | 1192.4  |
| 4  | 121.24 | 269.98 | 402.27 | 683.45 | 1632.2  |
| 8  | 317.08 | 654.64 | 933.12 | 1454.3 | 2843.8  |

*Table A.2: $c(\rho_0, n_g, k_x)$ across 24 $(n_g, k_x)$ modes.  The
pattern is growth in both $n_g$ (higher radial order, more oscillatory
$V_n$, larger $\mathrm{Var}_\mu(k_x^2 N^2)$) and $k_x$ (scaling closer
to $k_x^{1}$ than $k_x^2$, because $\omega_n^2$ also grows with $k_x$).
Every entry is independent of $N_y$ to four significant figures.*

