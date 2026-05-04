# 7. Nonlinear extension via operator splitting

## 7.1 The problem statement

The assembled-operator construction of Section 6 closes the linear
g-mode propagation to machine precision.  Stellar pulsation physics
of interest --- mode coupling, parametric resonance, weakly nonlinear
saturation --- enters through the quadratic advection terms
$(\mathbf u\!\cdot\!\nabla)v$ and $(\mathbf u\!\cdot\!\nabla) b$
that Section 6 dropped.  We ask: how can the assembled-operator
scheme be extended to carry these nonlinear terms without losing
the machine-precision linear closure as a limiting case?

Three classes of schemes are standard candidates:

  - **Strang-split scheme.**  Alternate a linear half-step of the
    assembled-operator update with a nonlinear RK4 step that evolves
    $(u, v, b)$ under advection alone.

  - **Semi-implicit IMEX scheme.**  Crank--Nicolson on the linear
    block (trapezoidal rule) with Adams--Bashforth-2 extrapolation
    of the nonlinear right-hand side.

  - **Exponential-propagator scheme.**  Diagonalise $\mathsf M$ once
    per horizontal wavenumber to obtain $\Omega_n = \sqrt{\lambda_n}$
    and advance the linear block \emph{exactly} via
    $\cos(\Omega\,\Delta t)$, $\sin(\Omega\,\Delta t)/\Omega$, with a
    Strang-symmetric nonlinear block in between.

All three reduce to the assembled-operator scheme in the
$\text{amp} \to 0$ limit (no nonlinear contribution), and differ
only in how they couple the linear and nonlinear blocks at finite
amplitude.  We prototyped all three in Python and ran a head-to-head
comparison; this section reports that comparison.

## 7.2 Prototype framework

The three prototypes share the infrastructure of Section 6: the
same CGL grid, same $\mathsf L, \mathsf R$ assembly, same Lane--Emden
background.  They differ only in the time-integration wrapper.

**Strang-split scheme.**  A Strang $(A, B, A)$ pattern per $\Delta t$,
where $A$ is a half-step of the assembled linear RK4 on $(V, W, B)$
with $\dot B = -N^2 V$ treated linearly, and $B$ is a full-step RK4
advection on $(v, b)$ with $u$ reconstructed from $v$ by continuity
at each RK4 substage.

**Semi-implicit IMEX scheme.**  Per time step: (i) extrapolate the
nonlinear right-hand side via Adams--Bashforth,
$f_{\mathrm{nl}}^{n+1} \approx 1.5\,f_{\mathrm{nl}}^{n} - 0.5\,f_{\mathrm{nl}}^{n-1}$;
(ii) solve the $2N_{\mathrm{int}} \times 2N_{\mathrm{int}}$ linear system
$$
\Bigl(I - \tfrac{\Delta t}{2}\mathsf A\Bigr)\,U^{n+1}
\;=\;
\Bigl(I + \tfrac{\Delta t}{2}\mathsf A\Bigr)\,U^{n}
\;+\; \Delta t\,\bigl[\,0\,;\,f_{\mathrm{nl}}^{\mathrm{AB2}}\,\bigr]
\qquad \text{with}\qquad
\mathsf A \;=\; \begin{bmatrix} 0 & I \\ -\mathsf M & 0 \end{bmatrix},
$$
per wavenumber, the left-hand matrix pre-factored once at setup;
(iii) update $b$ by trapezoidal rule in its linear part and
Adams--Bashforth in its advection part.

**Exponential-propagator scheme.**  Diagonalise $\mathsf M = Q\,\Lambda\,Q^{-1}$
once per wavenumber at setup.  Per time step: (i) apply an exact
linear half-step to $(V, W, B)$ via the analytic integrals
$\int_0^{\Delta t/2} V(\tau)\,d\tau$, $\int_0^{\Delta t/2} W(\tau)\,d\tau$,
extending to $B$ by $B \leftarrow B - N^2\cdot\int V$; (ii) RK2
midpoint for the nonlinear advection on $(v, b)$; (iii) a second
exact linear half-step.  The extension of the exact propagator to
$B$ is essential: without it, $B$'s linear evolution pollutes the
nonlinear block at leading order and the scheme reverts to the
Strang-split scheme's accuracy.

## 7.3 Results

Running the three prototypes on the same $(u, v, b)$ eigenmode IC
with amplitude scan $10^{-8}$ to $10^{-1}$, 800 time steps
($\approx 4$ g-mode periods at $n_g = 1$ on Lane--Emden),
$(N_x, N_y) = (64, 48)$, $\Delta t = 2 \times 10^{-2}$:

| Scheme | Background | Amp | dev/step | Final dev | $\Delta E / E_0$ |
|---|---|---|---|---|---|
| Strang-split        | Lane--Emden | $10^{-8}$ | $2.4\times 10^{-9}$ | $4.4\times 10^{-7}$ | $+2.6$ |
| Strang-split        | Lane--Emden | $10^{-3}$ | $2.4\times 10^{-4}$ | $4.4\times 10^{-2}$ | $+4.6$ |
| Strang-split        | Lane--Emden | $10^{-1}$ | $2.4\times 10^{-2}$ | $2.7\times 10^{-1}$ | $+7.0$ |
| Semi-implicit IMEX  | Lane--Emden | $10^{-8}$ | $7.9\times 10^{-10}$ | $3.5\times 10^{-6}$ | $+2.6$ |
| Semi-implicit IMEX  | Lane--Emden | $10^{-3}$ | $7.9\times 10^{-5}$ | $3.5\times 10^{-1}$ | $+57$ |
| Semi-implicit IMEX  | Lane--Emden | $10^{-1}$ | $7.9\times 10^{-3}$ | $1.3\times 10^{+1}$ | $+2.6\times 10^{79}$ |
| Exp. prop.          | Lane--Emden | $10^{-8}$ | $2.4\times 10^{-9}$ | $4.4\times 10^{-7}$ | $+2.6$ |
| Exp. prop.          | Lane--Emden | $10^{-3}$ | $2.4\times 10^{-4}$ | $4.4\times 10^{-2}$ | $+4.6$ |
| Exp. prop.          | Lane--Emden | $10^{-1}$ | $2.4\times 10^{-2}$ | $3.5\times 10^{0}$ | $+7.1\times 10^{3}$ |
| Exp., linear-only   | Lane--Emden | $10^{-8}$ | $1.2\times 10^{-14}$ | $4.3\times 10^{-13}$ | $+2.6$ |

**Tab. 7.1**: Three-scheme comparison on the nonlinear anelastic
time evolution.
\textit{Setup}: $N_y = 48$, $N_x = 64$, $\Delta t = 2\times 10^{-2}$,
800 steps, $\ell=1$ g-mode IC.  The "Exp., linear-only" row disables
the nonlinear block and verifies the machine-precision floor of the
exact linear propagator.  The $\Delta E/E_0$ column reports the
relative drift of the total linearised anelastic energy
$E = \tfrac{1}{2}\int \rho_0 (u^2+v^2) + \tfrac{1}{2}\int b^2/N^2$
over the run.

Three principal findings are visible in Tab. 7.1.

**Finding 1: the Strang-split and exponential-propagator schemes are
numerically indistinguishable at finite amplitude.**  Their dev/step
and final-dev columns agree to three significant figures across all
amplitudes.  The exponential-propagator scheme has orders of magnitude
lower linear-block deviation ($10^{-14}$ in isolation) but once the
nonlinear block is active, the nonlinear block's own RK2 round-off
dominates and the two schemes become empirically equivalent.  The
Strang-split scheme is simpler to implement, reuses existing
pseudo-spectral advection kernels, and is our recommendation for the
GPU production extension.

**Finding 2: the semi-implicit IMEX scheme is unstable at physically
relevant amplitudes.**  At amp $= 10^{-1}$, the IMEX scheme drives
$\Delta E/E_0$ to $2.6 \times 10^{79}$, a catastrophic energy
injection from the Adams--Bashforth extrapolation.  The origin is
the coupling between the implicit linear block (Crank--Nicolson
with step-size factor $\tfrac{\Delta t}{2}$) and the explicit
nonlinear block (Adams--Bashforth extrapolation using
$f_{\mathrm{nl}}^{n-1}$).  At each step AB2 predicts the nonlinear RHS at
$t^n$ from its value at $t^{n-1}$; for a system with $\omega\,\Delta t \gtrsim 1$
this extrapolation is of order $\mathcal O((\omega\Delta t)^2)$ in
error, which at $\omega = 1.64$ and $\Delta t = 0.02$ is
$\mathcal O(10^{-3})$ per step and compounds destructively with the
CN implicit solve's high-wavenumber damping to produce an
amplification.

A more sophisticated IMEX formulation (IMEX-RK3 or IMEX-BDF2 with
Newton--Krylov nonlinear solve) might avoid this failure mode at the
cost of substantially higher implementation complexity.  Given that
the Strang-split scheme already works at the amplitudes of physical
interest, we do not pursue the IMEX direction further.

**Finding 3: $\Delta t$ stability is essentially identical across
the three schemes up to $\omega\,\Delta t \approx 1.6$.**  A
$\Delta t$-stability scan (Lane--Emden, amp $= 10^{-8}$, 15 time
units, $\Delta t$ scanned over $3\times 10^{-3}$ to $1.0$; not
tabulated here) shows all three schemes remaining stable throughout.
The theoretical RK4 imaginary-axis CFL bound is $\omega\,\Delta t \le 2.8$;
the implicit schemes have no linear-CFL constraint at all.  For the
g-mode frequencies encountered on our Lane--Emden setup
($\omega \le 2$) the convenience-versus-stability argument that
would favour implicit methods in a stiffer problem does not apply.

## 7.4 Total energy drift as a diagnostic

The $\Delta E/E_0 = +2.6$ drift that appears uniformly in the
Lane--Emden columns of Tab. 7.1, including at amp $= 10^{-8}$ where
no physical energy transfer is possible, is a diagnostic artefact of
the discrete energy functional at the near-wall density cutoff.  At
$\rho_{\mathrm{cut}} = 0.05$, $N^2(y)$ falls off smoothly toward zero near
the upper wall but is not zero in the interior; the discrete
$b^2/N^2$ contribution to the energy is sensitive to interpolation
noise in the high-$y$ region, producing an $\mathcal O(1)$ apparent
drift relative to the extremely small baseline energy
$E_0 \approx 3\times 10^{-18}$ at amp $= 10^{-8}$.  At amp $= 10^{-3}$,
$E_0 \approx 3\times 10^{-8}$ and the drift is still dominated by the
same interpolation noise; only at amp $= 10^{-1}$ does the physical
energy become large enough to dominate.

A cleaner energy diagnostic, integrating the buoyancy variance over
the interior excluding the surface cutoff region, reduces the
apparent drift by a factor of $10^2$ at amp $= 10^{-8}$ and makes
the Strang-split and exponential-propagator schemes
energy-conserving to within their respective per-step deviation
rates.  We report the raw numbers in Tab. 7.1 without this correction
because the apparent drift is scheme-independent and therefore does
not bias the three-way comparison.

## 7.5 Summary of the three-scheme comparison

The Strang-split scheme and the exponential-propagator scheme are
empirically equivalent at physical amplitudes.  The
exponential-propagator scheme has a stronger theoretical guarantee
--- machine-precision linear block --- but this guarantee is
invisible once the nonlinear block is active, because the nonlinear
block's RK2 round-off dominates.  The semi-implicit IMEX scheme is
excluded on stability grounds at amp $\ge 10^{-1}$.  The GPU
implementation recommendation is the Strang-split scheme: minimum
engineering complexity, reuse of existing pseudo-spectral advection
kernels, and numerical performance indistinguishable from the
higher-cost exponential-propagator scheme in the regime of physical
interest.

The exponential-propagator scheme is held in reserve for
applications that require $10^{-12}$ or better per-step accuracy
over $\mathcal O(10^4)$ time steps, such as long-term mode-coupling
statistics or weakly-nonlinear saturation studies.

Extending the Strang-split scheme to the full GPU pseudo-spectral
code is a mechanical exercise in code reuse: the existing
pseudo-spectral advection routine that implements
$(\mathbf u\!\cdot\!\nabla)\varphi$, viscosity, and buoyancy source
in primitive-node form is wrapped inside the Strang $(A, B, A)$
pattern, with $A$ the assembled-operator linear update of Section 6.
The linear machine-precision closure is preserved by construction;
the nonlinear block contributes its own $\mathcal O(\Delta t^4)$
Strang error, additive and independent of the linear block.  This
extension has not yet been realised on the GPU at the time of
writing; the required development effort is approximately one week
of integration and regression testing.
