# 2. Mathematical setting

## 2.1 Anelastic equations

We work throughout with the reduced-pressure anelastic equations in a
two-dimensional Cartesian periodic-in-$x$, wall-bounded-in-$y$
channel of horizontal extent $L_x$ and radial extent $L_y$.  With
$\mathbf u = (u, v)$ the horizontal and radial velocity components,
$b$ the buoyancy scalar (dimensionless perturbation to the background
specific entropy, written so that $b$ appears as an acceleration),
and $\pi$ the reduced pressure,

$$
\begin{aligned}
\partial_t u &= -\, (\mathbf u\!\cdot\!\nabla)\, u \;-\; \partial_x \pi, \\
\partial_t v &= -\, (\mathbf u\!\cdot\!\nabla)\, v \;-\; \partial_y \pi
               \;-\; \frac{\rho_0'}{\rho_0}\,\pi \;+\; b, \\
\partial_t b &= -\, (\mathbf u\!\cdot\!\nabla)\, b \;-\; N^2(y)\, v, \\
\partial_x(\rho_0\, u) \;+\;
\partial_y(\rho_0\, v) &= 0.
\end{aligned}
\tag{2.1}
$$

The background density $\rho_0(y)$ and squared Brunt--Väisälä
frequency $N^2(y)$ are time-independent prescribed functions of the
radial coordinate; primes denote $d/dy$.  The stratification is
captured entirely through $\rho_0$ and $N^2$; in particular, there
is no advected background state aside from the constraint that the
background is a hydrostatic equilibrium of the underlying fully
compressible system, $p_0' = -\rho_0\,g$, from which $N^2$ is
determined.

The boundary conditions are periodicity in $x$,
$f(x+L_x,y,t) = f(x,y,t)$ for $f \in \{u,v,b,\pi\}$, together with
impermeability $v(x,0,t) = v(x,L_y,t) = 0$ on the radial walls,
free-slip for $u$ (no wall viscous boundary layer in the stellar
interior interpretation), and $\pi(x,0,t) = \pi(x,L_y,t) = 0$
--- the latter a gauge choice that makes the pressure projection
well-posed on Dirichlet boundary conditions and does not affect the
divergence-free velocity.  The buoyancy $b$ is likewise taken to
have Dirichlet walls.  All these choices match the SL machinery of
Section 3.

## 2.2 Linearised evolution and the anelastic g-mode equation

Linearising (2.1) about the rest state and taking
$\varphi(x,y,t) = \tilde\varphi(y)\,e^{i k_x x}\,e^{-i\omega t}$
for $\varphi \in \{u, v, b, \pi\}$, algebraic elimination of
$\tilde u$ through continuity, $\tilde b$ through the buoyancy
equation, and $\tilde\pi$ through the momentum equations reduces
the linearised system to the single second-order ODE in the
radial velocity amplitude $V \equiv \tilde v$,

$$
-\Bigl(\rho_0\, V'\Bigr)' \;+\; k_x^2\, \rho_0\, V
\;=\;
\frac{k_x^2\, N^2\, \rho_0}{\omega^2}\, V,
\qquad
V(0) = V(L_y) = 0.
\tag{2.2}
$$

We write (2.2) in the form $\omega^2\, L\, V = R\, V$ with

$$
L = -\frac{d}{dy}\Bigl(\rho_0\,\frac{d}{dy}\Bigr) + k_x^2\,\rho_0, \qquad
R = k_x^2\, N^2\,\rho_0,
\tag{2.3}
$$

both acting on the interior of $[0, L_y]$ with Dirichlet boundary
conditions.  The generalised eigenvalue problem $L\,V = \omega^{-2}\,R\,V$
organises the radial internal-gravity-wave spectrum; its eigenvectors
are the *radial structure functions* of the anelastic g-modes that
this paper is concerned with recovering numerically.

Under the change of variable $\varphi \equiv \rho_0 V$, equation
(2.2) becomes the Boussinesq-like form

$$
-\varphi'' + k_x^2\,\varphi = \frac{k_x^2\, N^2}{\omega^2}\,\varphi,
\qquad
\varphi(0) = \varphi(L_y) = 0,
\tag{2.4}
$$

in which $\rho_0$ has disappeared from the leading operator.  This is
the Liouville reduction; we return to it in Section 3.1.

## 2.3 Background: Lane--Emden $n = 3/2$ polytrope

The paper tracks two backgrounds: a uniform ($\rho_0 \equiv 1$,
$N^2 \equiv 1$) Boussinesq baseline, and a Lane--Emden $n = 3/2$
polytrope that represents a monatomic, isentropic, radiatively
stable stellar envelope.  The Lane--Emden background is obtained by
integrating the standard dimensionless polytropic ODE to its first
zero at $\xi_1 \approx 3.6538$, mapping $[0, \xi_1] \to [0, L_y]$
linearly, and applying a surface truncation $\rho_0 \ge \rho_{\mathrm{cut}}$
($\rho_{\mathrm{cut}} = 5 \times 10^{-2}$ unless otherwise noted) to keep
$\rho_0'/\rho_0$ bounded.  The density contrast across the domain is
$\sim 20$ at this cut, well inside the regime where the Boussinesq
approximation is quantitatively wrong.  The Brunt--Väisälä frequency
is $N^2(y) = \max(-\rho_0'/\rho_0, 0)$, with the clipping excluding
numerical negatives from finite-difference reconstruction of
$\rho_0'$.  Details of the polytropic ODE integration, the
$\rho_{\mathrm{cut}}$ sensitivity (Appendix A.2), and the alternative
Lane--Emden $n = 3$ profile used for the GYRE benchmark are
deferred to the appendix and Section 4 respectively.

We deliberately adopt this standard pseudo-spectral formulation to
isolate operator consistency from approximation errors: any failure
observed in Section 5 will be attributable to the discrete algebra
of the time-stepping loop, not to an unconventional implementation.

## 2.4 Normalisation and Fourier convention

All results in this paper are in the dimensionless unit system
$G = M_\star = R_\star = 1$, which makes frequencies dimensionless
$\omega^2$ in units of $G M_\star / R_\star^3$.  Horizontal Fourier
transforms are taken with the convention
$\hat f(k_x) = \sum_n f(x_n)\, e^{-ik_x x_n}\,\Delta x$,
$f(x_n) = (1/L_x)\sum_{k_x} \hat f(k_x)\,e^{ik_x x_n}$,
so that the $k_x$ grid is $k_x^{(k)} = 2\pi k / L_x$ for
$k = 0, 1, \dots, n_h-1$ with $n_h = n_x/2 + 1$ for real input.
A two-thirds dealiasing rule is applied to all real-space products
by zeroing the top third of horizontal Fourier modes before any
backward transform.  Radial discretisation uses the Chebyshev--
Gauss--Lobatto (CGL) grid on $[0, L_y]$; grid and differentiation
matrix conventions are specified in Section 3.
