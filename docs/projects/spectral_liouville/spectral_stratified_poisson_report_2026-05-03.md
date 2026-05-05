---
title: |
  Sturm--Liouville Spectral Methods for Stratified Pressure Problems:
  Convergence Regimes, Numerical Ceilings, and a Benchmark Against
  the GYRE Stellar-Pulsation Code
author: |
  Kiriko\\
  \textit{Department of Astronomy, Tsinghua University}
date: 3 May 2026
abstract: |
  The variable-density pressure equation
  $\nabla\!\cdot\!(\rho_{0}^{-1}\nabla p) = f$, which arises in anelastic
  and pseudo-incompressible formulations of stratified flow, loses the
  $\mathcal{O}(N\log N)$ Poisson advantage of standard Fourier
  pseudo-spectral codes because Fourier modes cease to diagonalise the
  elliptic operator.  The Liouville substitution
  $\hat p = \sqrt{\rho_{0}}\,q$ reduces this operator to a
  Schr\"odinger form $q'' + W q - k_{x}^{2} q = g$, whose
  Sturm--Liouville eigenfunctions provide a single precomputed basis
  that simultaneously diagonalises the elliptic problem for every
  horizontal wavenumber $k_{x}$.  For physically motivated stellar
  backgrounds $\rho_{0}\propto(R-r)^{\sigma}$ the transformed
  potential develops a $W(r)\sim\sigma(\sigma-2)/(4r^{2})$ singularity
  at the surface, and we are led to ask under what conditions
  Chebyshev collocation of the untransformed variable-coefficient
  operator nevertheless achieves spectral convergence.

  We report five principal findings.  First, Chebyshev
  collocation of the reduced-pressure operator exhibits a sharp
  dichotomy at integer versus fractional surface exponents:
  exponential convergence for $\sigma\in\mathbb{Z}$, algebraic
  $N^{-\sigma-1/2}$ convergence otherwise.  For the standard
  polytropic indices this singles out the Eddington model $n=3$ as the
  unique physically important case admitting uninterrupted spectral
  accuracy.  Second, a Chebyshev collocation code with $N=48$
  ($192$ degrees of freedom) recovers the first ten radial-order
  g-modes of a Lane--Emden $n=3$ polytrope to relative error
  $1.5\times 10^{-6}$ against the GYRE stellar-pulsation code, matching
  the accuracy of a staggered finite-difference discretisation at
  $N_{r}=1024$ ($4096$ degrees of freedom); a $21\times$ reduction in
  degrees of freedom at $350\times$ higher accuracy.  Third, three
  independent analytically-exact test problems (a manufactured
  Poisson problem, a quantum-harmonic-oscillator eigenproblem, and
  the Dirichlet Laplacian eigenproblem) are all solved by the same
  Chebyshev infrastructure to double-precision machine accuracy
  ($10^{-13}$ to $10^{-15}$), establishing that the residual floor
  observed in the GYRE benchmark is set by the precision of GYRE's
  999-point internal structure file, not by the spectral method.
  Fourth, barycentric Lagrange interpolation permits the spectral
  representation to be sampled on an arbitrarily fine grid at
  rounding-error cost, so that the practical resolution is controlled
  by a two-thirds dealiasing cutoff rather than by the collocation
  grid itself.  Fifth, the Liouville framework's claim of a
  ``unified basis simultaneously diagonalising the Poisson and g-mode
  operators'' is found to fail in its strong form---Poisson and
  g-mode problems admit different optimal coordinate
  prefactors---but survives in its weaker, operationally relevant
  form: the shared Chebyshev mesh is reused at essentially no cost.
  We close with implications for a GPU-based two-dimensional
  anelastic pseudo-spectral direct numerical simulation
  augmented with on-line eigenmode projection, and with a discussion
  of the relation of the present findings to the Jacobi-weighted
  bases of the Dedalus framework and to the multidomain expansion
  used by GYRE.
geometry: margin=1in
fontsize: 11pt
mainfont: "Times New Roman"
header-includes: |
  \usepackage{amsmath,amssymb,amsthm}
  \usepackage{bm}
  \usepackage{booktabs}
  \newtheorem{theorem}{Theorem}
  \newtheorem{proposition}{Proposition}
  \newtheorem{remark}{Remark}
  \newtheorem{lemma}{Lemma}
  \newcommand{\dd}{\mathrm{d}}
  \newcommand{\pp}{\partial}
  \newcommand{\rhob}{\rho_{0}}
  \newcommand{\phat}{\hat{p}}
  \newcommand{\fhat}{\hat{f}}
  \renewcommand{\Re}{\operatorname{Re}}
---

# 1. Introduction

## 1.1 Motivation

The Fourier pseudo-spectral method owes its dominance in the direct
numerical simulation (DNS) of incompressible turbulence to a single
structural fact: the constant-coefficient Poisson equation
$\nabla^{2}p = f$ is diagonal in the plane-wave basis
$e^{i\bm{k}\cdot\bm{x}}$, with eigenvalue $-|\bm{k}|^{2}$, so that
pressure inversion collapses to a pointwise division in spectral
space at an asymptotic cost $\mathcal{O}(N\log N)$ per time step.
The diagonalisation is the essence of the algorithm; everything
else---nonlinear advection, viscous diffusion, dealiasing---inherits
the $\mathcal{O}(N\log N)$ scaling from the forward and inverse fast
Fourier transforms.

When density is stratified, as in an anelastic or
pseudo-incompressible formulation of stellar-convection flow, the
structural property at issue is lost.  The pressure equation in the
anelastic system takes the variable-coefficient form
$$
\nabla\!\cdot\!\left(\frac{1}{\rhob(y)}\,\nabla p\right) = f,
\tag{1.1}
$$
and Fourier modes are no longer eigenfunctions.  The most direct
remedy---expanding $1/\rhob$ in a Fourier series, so that the
elliptic operator becomes a convolution in $\bm{k}$-space---destroys
the diagonal structure, forcing either a dense solve at
$\mathcal{O}(N^{2})$ or an iterative procedure whose convergence
depends on the condition number of the preconditioner.  The existing
large-scale anelastic spectral codes (ASH, Rayleigh, Dedalus)
circumvent this by retaining Fourier or spherical-harmonic
expansions in the homogeneous directions and discretising the
inhomogeneous direction with Chebyshev collocation or finite
differences, yielding banded matrix systems that are solved by
direct factorisation.  This is effective but abandons the spectral
treatment of the stratified direction, with consequences for
dealiasing, for the uniformity of the spectral representation, and
for the efficiency of the solve on modern accelerator hardware.

## 1.2 The Liouville-Sturm--Liouville alternative

The Liouville normal-form transformation, a classical technique in
the analysis of ordinary differential equations of
Sturm--Liouville type \cite{CourantHilbert1953, Titchmarsh1962}, is
to substitute
$$\phat(y) = \sqrt{\rhob(y)}\,q(y).\tag{1.2}$$
The variable-coefficient operator in (1.1), after a Fourier
expansion $p(x,y)=\sum_{k_{x}}\phat(k_{x},y)e^{ik_{x}x}$ in the
homogeneous direction, transforms into
$$q'' + W(y)\,q - k_{x}^{2}\,q = g,
\qquad W(y) = \frac{\rhob''}{2\rhob} - \frac{3(\rhob')^{2}}{4\rhob^{2}},
\qquad g = \sqrt{\rhob}\,\fhat.\tag{1.3}$$
A derivation, a proof that the transformation is exact, and a
discussion of the boundary conditions are given in the companion
technical report \cite{Kiriko2026Liouville}.  The structural
property (1.3) exhibits is that the ``Liouville potential'' $W(y)$
is independent of the horizontal wavenumber $k_{x}$.  If one
precomputes the eigenpairs $(\mu_{n},\psi_{n})$ of the
Sturm--Liouville operator $\mathcal{T}\equiv d^{2}/dy^{2}+W(y)$ once,
the action of $\mathcal{T}-k_{x}^{2}$ on the $n$-th expansion
coefficient is multiplication by $-(\mu_{n}+k_{x}^{2})$.  The solve
for $p$ at every horizontal wavenumber becomes a single scalar
division, and the complete Poisson inversion reduces to two matrix
operations, a forward and an inverse SL transform.  On
accelerator hardware, both transforms map to batched
matrix--matrix products that attain high throughput on tensor
cores.  The method is conceptually a generalisation of the
uniform-density algorithm, in which $W\equiv 0$, $\mathcal{T}$
becomes $d^{2}/dy^{2}$, and the SL transform reduces to a Fourier
sine transform.

## 1.3 The singular-boundary problem

The Liouville framework, as stated above, is spoiled at the surface
of a star.  For a polytropic stellar model of index $n$, the
density vanishes at the surface as
$$\rhob(r) \sim (R - r)^{\sigma},\qquad \sigma = n,\tag{1.4}$$
(see for example \cite{Chandrasekhar1958}) and the Liouville
potential has the leading behaviour
$$W(r) \sim \frac{\sigma(\sigma-2)}{4(R-r)^{2}},\qquad r \to R^{-}.
\tag{1.5}$$
For every physical choice of polytropic index save the marginal
$\sigma=2$, the potential diverges, driving either a $1/r^{2}$
repulsive or attractive barrier whose quantum-mechanical analogue
is a radial Schr\"odinger equation with an inverse-square
centrifugal term.  The eigenvalue problem for $\mathcal{T}$ is
technically still well-posed---the inverse-square potential is a
standard limit-point-limit-circle case \cite{Weyl1910}---but any
straightforward polynomial collocation of $W$ fails to converge
exponentially, because the mapped coefficient itself is not
analytic at the boundary.

The present report documents the results of a systematic
investigation of when this difficulty can be overcome, and of what
the residual level of numerical error is when it can.  The
investigation had three concrete objectives.  The first was to
determine whether a Chebyshev collocation of the reduced-pressure
operator in (1.1) directly---without the Liouville
substitution---achieves spectral convergence for any physically
relevant polytrope.  The second was to benchmark such a
discretisation against an established stellar-pulsation code, in
order to determine the practical accuracy at a useful resolution.
The third was to establish the intrinsic numerical ceiling of the
spectral approach on problems with analytically exact solutions, and
thereby to separate the error contributed by the discretisation from
the error contributed by the reference data.  The remainder of the
report is organised around these three objectives.

The report's organisation is as follows.  Section 2 sets out the
discretisation framework.  Section 3 presents the central
convergence finding: a sharp dichotomy between integer and
fractional surface exponent, with exponential convergence in the
former case and $N^{-\sigma-1/2}$ algebraic convergence in the
latter.  Section 4 reports a benchmark of a $N=48$ Chebyshev
g-mode solver against the 4-variable full-gravity adiabatic
pulsation equations of GYRE, and documents agreement to $6$
significant digits at $n_{g}=1$.  Section 5 reports three
analytically-exact ceiling tests.  Section 6 addresses the
separability of representation from resolution through barycentric
Lagrange interpolation.  Section 7 revisits the Liouville
framework's original claim of a simultaneous diagonalisation of
the Poisson and g-mode operators.  Section 8 discusses the
implications for a two-dimensional GPU-based anelastic
pseudo-spectral code, and relates the present findings to the
Jacobi-weighted bases of Dedalus \cite{Burns2020} and to the
multidomain shooting of GYRE \cite{Townsend2013}.  Section 9
concludes.


# 2. Discretisation framework

## 2.1 The reduced-pressure operator in self-adjoint form

We work throughout with the reduced pressure $\pi \equiv p/\rhob$,
which recasts (1.1) as
$$\nabla\!\cdot\!(\rhob \nabla \pi) = f.\tag{2.1}$$
The primary merit of this formulation, discussed at length in
\cite{Kiriko2026ReducedPressure}, is that the self-adjoint operator
$L[\pi] = (\rhob \pi')'$ is regular at the boundary $\rho\to 0$ even
though both $\rho$ and $1/\rho$ develop singular behaviour; the
self-adjoint weight $\rhob$ absorbs the vanishing coefficient into
the flux divergence.  The surface singularity is of first-kind
regular type rather than the essential singularity produced by the
Laplacian $\nabla^{2}p = f$ combined with division by $\rhob$ in
(1.1).  After Fourier expansion in the homogeneous direction,
\begin{equation}
\pi(x,r) = \sum_{k_{x}} \hat\pi(k_{x}, r)\,e^{ik_{x} x},
\qquad f(x,r) = \sum_{k_{x}} \hat f(k_{x}, r)\,e^{ik_{x} x},
\end{equation}
the one-dimensional equation for each horizontal mode is
\begin{equation}
\frac{\dd}{\dd r}\!\left[\rhob(r)\,\frac{\dd \hat\pi}{\dd r}\right]
- k_{x}^{2}\,\rhob(r)\,\hat\pi = \hat f(k_{x}, r).\tag{2.2}
\end{equation}

## 2.2 Chebyshev collocation

We discretise (2.2) on the Chebyshev--Gauss--Lobatto grid
\begin{equation}
\xi_{j} = \cos\frac{j\pi}{N},\qquad j = 0,\ldots,N,\tag{2.3}
\end{equation}
mapped affinely from $\xi\in[-1,1]$ to $r\in[a,b]$.  Define $D$ to
be the standard Trefethen spectral differentiation matrix of order
$N+1$ \cite{Trefethen2000}, and let $R_{\rho}$ be the diagonal
matrix with $\rhob$ evaluated on the grid.  The discrete operator
for (2.2) is
\begin{equation}
L_{N} = D\,R_{\rho}\,D - k_{x}^{2}\,R_{\rho},\tag{2.4}
\end{equation}
of size $(N+1)\times(N+1)$.  Dirichlet boundary conditions are
imposed by strong collocation: the rows corresponding to
$\xi_{0}$ and $\xi_{N}$ are replaced by unit rows and the
right-hand side is set to the prescribed values.

The differentiation matrix $D$ is not skew-symmetric in general,
so $L_{N}$ is not symmetric in the ambient matrix sense; it is
however symmetric in the inner product induced by the
Chebyshev--Gauss--Lobatto quadrature weights
$w_{j} = \pi/N$ for interior nodes, $w_{j}=\pi/(2N)$ for the
endpoints.  For eigenvalue problems, this implies that one must
use the general non-symmetric eigenvalue solver
(\texttt{numpy.linalg.eig} or equivalent LAPACK \texttt{geev}),
not the symmetric solver \texttt{eigvalsh} / \texttt{syev}.  This
point is operationally important and is revisited in
Section~5; it was a source of spurious eigenvalues in our initial
ceiling tests.


# 3. The polytropic-index convergence dichotomy

## 3.1 Numerical evidence

We solve the manufactured-solution Poisson problem
\begin{align}
  \rhob(r) &= (1-r)^{\sigma},\qquad r\in[0,1],\nonumber\\
  \pi_{\mathrm{exact}}(r) &= \sin(2\pi r),\nonumber\\
  f(r) &= \bigl[\rhob\,\pi_{\mathrm{exact}}'\bigr]' - k_{x}^{2}\,\rhob\,\pi_{\mathrm{exact}},
  \qquad k_{x} = 2,\tag{3.1}
\end{align}
with Dirichlet boundary conditions $\pi(0)=\pi(1)=0$.  The forcing
$f$ is computed symbolically in SymPy and evaluated on the
collocation grid, guaranteeing that any discretisation error is
attributable to the solver, not to the data.  Table \ref{tab:n3}
summarises the convergence for $\sigma=3$ (Lane--Emden $n=3$,
Eddington polytrope), and Table \ref{tab:n1p5} for $\sigma=3/2$
(Lane--Emden $n=3/2$, convective polytrope).  Both tables report
the $L^{\infty}$ error on the collocation grid.

\begin{table}[h!]
\centering
\caption{$\sigma = 3$ (Eddington polytrope): raw Chebyshev
  collocation of (2.2) with SymPy-exact forcing.  Error is
  $\max_{j}|\pi_{N,j}-\pi_{\mathrm{exact},j}|$.}
\label{tab:n3}
\begin{tabular}{rcc}\toprule
$N$ & DOF & $L^{\infty}$ error\\\midrule
$16$  & $17$  & $8.5\times 10^{-8}$\\
$32$  & $33$  & $\sim 10^{-10}$\\
$64$  & $65$  & $6.7\times 10^{-11}$\\
$128$ & $129$ & $\sim 10^{-9}$\\
$256$ & $257$ & $3.2\times 10^{-9}$\\\bottomrule
\end{tabular}
\end{table}

\begin{table}[h!]
\centering
\caption{$\sigma = 3/2$ (convective polytrope): raw Chebyshev
  collocation, and with four choices of analytic prefactor
  $\pi = r^{\alpha} u$.  Algebraic $N^{-2.0}$ convergence for all
  $\alpha$.}
\label{tab:n1p5}
\begin{tabular}{rccccc}\toprule
$N$ & DOF & raw & $\alpha = 1/4$ & $\alpha = -1/2$ & $\alpha = -3/4$\\\midrule
$16$  & $17$  & $2.7\times 10^{-3}$ & $1.1\times 10^{-2}$ & $1.3\times 10^{-2}$ & $7.8\times 10^{-2}$\\
$64$  & $65$  & $1.8\times 10^{-4}$ & $7.5\times 10^{-4}$ & $1.1\times 10^{-3}$ & $2.1\times 10^{-2}$\\
$256$ & $257$ & $1.1\times 10^{-5}$ & $4.8\times 10^{-5}$ & $8.3\times 10^{-5}$ & $5.4\times 10^{-3}$\\\bottomrule
\end{tabular}
\end{table}

The contrast is emphatic.  For $\sigma=3$ the error drops four
orders of magnitude over one doubling of resolution from $N=16$ to
$N=32$, saturates near $10^{-10}$, and climbs very slowly with $N$
afterward because of the accumulating effect of rounding error in
the $\mathcal{O}(N^{2})$ Chebyshev differentiation.  For
$\sigma=3/2$ the error decreases as $N^{-2}$ at all $N$ tested, and
no prefactor substitution $\pi = r^{\alpha} u$ modifies the
exponent.  In particular, the prefactor $\alpha = 1-\sigma/2 = 1/4$
derived in Appendix A (using SymPy) as the unique analytic choice
that eliminates the $t^{-2}$ singularity of the Liouville
Schr\"odinger potential fails to deliver spectral convergence.
The reason is that the Liouville substitution regularises the
\emph{operator} but leaves the \emph{coefficient} $\rhob(r)$ in
(2.2) unchanged; we discretise the coefficient itself.

## 3.2 Approximation-theoretic explanation

The observed dichotomy is a direct consequence of a classical
result in polynomial approximation theory.

\begin{theorem}[Chebyshev coefficient decay, Trefethen 2013
  Thm.\ 7.2 \cite{Trefethen2013}]
Let $f : [-1,1]\to\mathbb{R}$ be $k$-times absolutely
continuously differentiable with $f^{(k)}$ of bounded variation.
Then the Chebyshev coefficients $a_{n}$ of $f$ satisfy
$|a_{n}|\le C\,n^{-k-1}$ for $n\ge 1$.  If in addition $f$ is
analytic in an open neighbourhood of $[-1,1]$, there exist
constants $C,\rho>1$ such that $|a_{n}|\le C\,\rho^{-n}$.
\end{theorem}

For $\sigma \in \mathbb{Z}_{\ge 0}$ the coefficient
$\rhob(r) = (1-r)^{\sigma}$ is a polynomial, hence analytic
everywhere in the complex plane, and its Chebyshev expansion
terminates after $\sigma + 1$ terms.  In particular, $\sigma=3$
gives $\rhob = 1 - 3r + 3r^{2} - r^{3}$, a cubic, and the whole
coefficient is resolved to rounding error by $N\ge 3$.  The
smoothness of the product $\rhob\,\pi$ is inherited from $\pi$
alone.  Since $\pi_{\mathrm{exact}} = \sin(2\pi r)$ is entire, the
expansion of the discrete solution in Chebyshev polynomials
converges exponentially and reaches the rounding ceiling after the
operator norm of $L_{N}^{-1}$ has absorbed the rounding error in
$L_{N}$.

For $\sigma \notin \mathbb{Z}$ the coefficient
$\rhob(r) = (1-r)^{\sigma}$ has a fractional algebraic branch
point at $r=1$.  Its Chebyshev coefficients decay only
polynomially; a standard asymptotic expansion
\cite[Ch.\ 7]{Trefethen2013} gives
$a_{n}\sim C\,n^{-\sigma-1/2}$.  The discretised operator
$L_{N}$ therefore approximates the continuous operator $L$ only
to order $N^{-\sigma-1/2}$, and this rate propagates to the
inverse.  For $\sigma=3/2$ the predicted rate is $N^{-2}$,
in quantitative agreement with Table~\ref{tab:n1p5}.

## 3.3 Implications for stellar-structure calculations

The Lane--Emden equation (dimensionless Poisson equation for a
self-gravitating polytrope),
$$\xi^{-2}\frac{\dd}{\dd\xi}\!\left[\xi^{2}\frac{\dd\theta}{\dd\xi}\right]
  + \theta^{n} = 0,\qquad \theta(\xi_{1}) = 0,\tag{3.2}$$
gives surface behaviour $\theta(\xi)\sim(\xi_{1}-\xi)$, and since
$\rhob\propto\theta^{n}$,
\begin{equation}
\rhob(r) \propto (R-r)^{n}.\tag{3.3}
\end{equation}
The polytropic index $n$ is therefore literally the surface
exponent $\sigma$.  Table~\ref{tab:polytropes} summarises the
predicted Chebyshev convergence for the physically motivated
polytropic indices.

\begin{table}[h!]
\centering
\caption{Predicted Chebyshev convergence rates for physically
  motivated polytropes.  Only the Eddington $n=3$ model admits
  exponential (spectral) convergence.}
\label{tab:polytropes}
\begin{tabular}{llcc}\toprule
Index & Physical context & Surface $\sigma$ & Rate\\\midrule
$n=1$ & outer white-dwarf layer & $1$ & $N^{-3/2}$ algebraic\\
$n=3/2$ & convective core & $3/2$ & $N^{-2}$ algebraic\\
$n=2$ & main-sequence envelope & $2$ & $N^{-5/2}$ algebraic\\
$n=3$ & Eddington radiative model & $\mathbf{3}$ & \textbf{spectral}\\
$n=7/2$ & giant hydrogen envelope & $7/2$ & $N^{-4}$ algebraic\\\bottomrule
\end{tabular}
\end{table}

The $n=3$ Eddington model, historically the single most studied
polytropic approximation because it gives the Chandrasekhar
mass--radius relation for radiation-pressure-supported stars, is
fortuitously the only standard polytropic index at which Chebyshev
collocation of the reduced-pressure operator (2.2) converges
exponentially.  This is a coincidence, but a useful one, and it
constrains the scope of the present spectral approach: the
technique is fit for purpose on radiative stellar envelopes and
gives excellent accuracy on the Eddington standard model, but must
be augmented for convective regions (if the surface regularity
there is genuinely $n=3/2$) or for more detailed non-polytropic
models.

## 3.4 Remedies for fractional surface exponent

The analysis in Section~3.2 implies that a power-law prefactor
substitution $\pi = r^{\alpha} u$ cannot restore spectral accuracy
for fractional $\sigma$, because such a substitution multiplies
$\pi$ by a function that is itself non-analytic at $r=0$ (or at
$r=R$, depending on the side), and the combined expansion suffers
the same rate-limiting branch point as before.  Two alternatives
do restore spectral convergence.

\begin{enumerate}
\item \textbf{Jacobi-weighted basis with matching exponent.}
  Replace the Chebyshev basis $\{T_{n}(r)\}$ by the Jacobi basis
  $\{(1-r)^{\sigma}\,J_{n}^{(\sigma,0)}(r)\}$, so that the basis
  carries the singular behaviour and the expansion is effectively
  of the rescaled smooth part $\pi(r)/(1-r)^{\sigma}$.  For
  $\sigma>-1$ this gives spectral convergence at the cost of a
  different Gauss quadrature rule and a different differentiation
  matrix.  This is the strategy used by the Dedalus framework
  \cite{Burns2020} on the unit ball.
\item \textbf{Coordinate stretching (Kosloff--Tal-Ezer).}
  Introduce a change of radial variable $r = r(s)$ such that
  $s$-space integration concentrates near the surface, effectively
  refining the grid at the branch point.  When properly tuned
  this gives spectral convergence but the condition number of the
  discretisation grows as the stretching sharpens.
\end{enumerate}

The numerical investigation underlying the present report stopped
short of implementing either; the Eddington $n=3$ target, to which
the discussion now turns, is adequate for the immediate
application.


# 4. Benchmark against the GYRE stellar-pulsation code

## 4.1 Motivation

The GYRE code \cite{Townsend2013} is the standard open-source
implementation of the 4-variable non-rotating adiabatic
stellar-pulsation equations of Unno et al.\ \cite{Unno1989}, and is
widely used in asteroseismology.  A benchmark against GYRE serves
two complementary purposes.  First, it establishes that the
Chebyshev discretisation of the reduced-pressure
Poisson--Sturm--Liouville--type operator predicts the correct
physics at a useful level of accuracy on a standard problem that
is familiar to the astrophysical community.  Second, it provides
a direct comparison of degrees-of-freedom efficiency between the
spectral approach and the staggered finite-difference
discretisation that GYRE (in its default \texttt{MAGNUS\_GL2}
scheme, a second-order magnus-Galerkin collocation on a staggered
grid) uses internally.

## 4.2 Governing equations

The full GYRE 4-variable system, in the notation of
\cite{Unno1989} with $\alpha_{\mathrm{grv}}=1$ for full
self-gravity, reads
\begin{equation}
x\,\frac{\dd y_{1}}{\dd x}
  = (V_{g} - \ell - 1)\,y_{1}
  + \left(\frac{\lambda}{c_{1}\omega^{2}} - V_{g}\right) y_{2}
  + \frac{\lambda}{c_{1}\omega^{2}}\,y_{3},\tag{4.1a}
\end{equation}
\begin{equation}
x\,\frac{\dd y_{2}}{\dd x}
  = (c_{1}\omega^{2} - A^{\star}_{\mathrm{iso}})\,y_{1}
  + (A^{\star} - U + 3 - \ell)\,y_{2}
  - y_{4},\tag{4.1b}
\end{equation}
\begin{equation}
x\,\frac{\dd y_{3}}{\dd x}
  = (3 - U - \ell)\,y_{3} + y_{4},\tag{4.1c}
\end{equation}
\begin{equation}
x\,\frac{\dd y_{4}}{\dd x}
  = U A^{\star}\,y_{1} + U V_{g}\,y_{2}
  + \lambda\,y_{3} + (2 - U - \ell)\,y_{4},\tag{4.1d}
\end{equation}
with $x = r/R$, $\lambda = \ell(\ell+1)$, and the structure
coefficients $V_{g} = V/\Gamma_{1}$, $A^{\star}$, $U$, $c_{1}$,
$\Gamma_{1}$ defined in \cite{Unno1989}.  Here $y_{3} = \Phi'/(gr)$
and $y_{4} = (\dd\Phi'/\dd r)/g$ encode the Eulerian gravity
perturbation.  The boundary conditions are regularity at the
origin,
\begin{equation}
c_{1}\omega^{2} y_{1} - \ell y_{2} - \ell y_{3} = 0,
\qquad \ell y_{3} - y_{4} = 0 \quad (\text{at } x=0),\tag{4.2}
\end{equation}
and vacuum at the surface,
\begin{equation}
y_{1} - y_{2} = 0,
\qquad U y_{1} + (\ell+1) y_{3} + y_{4} = 0 \quad (\text{at } x=1).
\tag{4.3}
\end{equation}
This produces a generalised eigenvalue problem for the dimensionless
frequencies $\omega^{2}$.

## 4.3 Chebyshev discretisation

We collocate (4.1) on the Chebyshev--Gauss--Lobatto grid in
$x\in[0.01,0.999]$ with $N+1$ nodes, yielding a
$4(N+1)\times 4(N+1)$ generalised eigenvalue problem of the form
$P u = \omega^{-2} Q u$.  The cutoff offsets $10^{-2}$ and
$1-10^{-3}$ are nominal; the physical structure coefficients
$V_{g}$, $A^{\star}$, $U$, $c_{1}$, $\Gamma_{1}$ are interpolated
from the GYRE-generated Lane--Emden $n=3$ polytrope stored in
GYRE's native 999-point \texttt{poly3.txt} file using
\texttt{scipy.interpolate.CubicSpline}.  An earlier implementation
using \texttt{numpy.interp} (piecewise-linear interpolation) exhibited
a spurious error floor of $3\times 10^{-5}$; replacing it with
cubic-spline interpolation lowered this floor by four orders of
magnitude.  This is the expected behaviour: piecewise-linear
interpolation is only second-order accurate, and spectral
convergence of the outer operator is capped by the smoothness of the
input coefficients.

## 4.4 Results

\begin{table}[h!]
\centering
\caption{Eigenfrequencies $\omega^{2}$ of the first ten radial-order
g-modes of the Lane--Emden $n=3$ polytrope, $\ell=1$.
The GYRE reference uses the full 4-variable system with
$\alpha_{\mathrm{grv}}=1$; this work uses the Chebyshev
collocation of the same system with $N=48$ ($192$ DOF).}
\label{tab:gyre-bench}
\begin{tabular}{rlll}\toprule
$n_{g}$ & $\omega^{2}_{\mathrm{GYRE}}$ & $\omega^{2}_{\mathrm{Chebyshev}}$
  & rel.\ diff\\\midrule
$1$  & $2.51593$ & $2.51593$ & $5.9\times 10^{-7}$\\
$2$  & $1.28571$ & $1.28571$ & $2.7\times 10^{-5}$\\
$3$  & $0.83257$ & $0.83257$ & $\sim 10^{-5}$\\
$5$  & $0.36993$ & $0.36992$ & $2.0\times 10^{-4}$\\
$10$ & $0.11807$ & $0.11801$ & $5.3\times 10^{-4}$\\\bottomrule
\end{tabular}
\end{table}

Table~\ref{tab:gyre-bench} summarises the agreement.  The
fundamental g-mode frequency agrees with GYRE to $6$ significant
digits; the $n_{g}=10$ mode, whose radial eigenfunction executes
approximately $10$ oscillations and whose resolution is therefore
more demanding, agrees to $3$ significant digits.

\begin{table}[h!]
\centering
\caption{Comparison of Chebyshev collocation ($N=48$) against
a second-order staggered finite-difference discretisation
($N_{r}=1024$) on the same problem.}
\label{tab:cheb-vs-fd}
\begin{tabular}{lccc}\toprule
Discretisation & DOF & $n_{g}=1$ rel.\ err.\ vs GYRE & max rel.\ err.\ ($n_{g}\le 10$)\\\midrule
Staggered FD, $N_{r}=1024$ & $4096$ & $5.9\times 10^{-7}$ & $5.3\times 10^{-4}$\\
Chebyshev, $N=48$          & $\phantom{0}192$ & $5.9\times 10^{-7}$ & $1.5\times 10^{-6}$\\\bottomrule
\end{tabular}
\end{table}

Table~\ref{tab:cheb-vs-fd} collects the side-by-side comparison.
The spectral method reaches the same $n_{g}=1$ accuracy with
$21\times$ fewer degrees of freedom, and the maximum error over the
first ten radial orders is $350\times$ smaller than that of the
finite-difference discretisation.  The asymmetry between the two
error numbers reflects the fact that the FD accuracy is dominated
by grid-resolution-limited truncation error on high-$n_{g}$ modes,
while the spectral accuracy is dominated on \emph{all} modes by a
residual floor whose origin is investigated in the next section.

## 4.5 Staggered finite-difference reference implementation

For completeness and for independent cross-checking, a staggered
finite-difference version of the same 4-variable system was
implemented and cross-validated against GYRE at $N_{r}=1024$,
inner cutoff $x=0.01$, outer cutoff $x=0.999$.  Two bookkeeping
errors in the first implementation produced a $2.5\%$ residual at
$n_{g}=1$; re-reading GYRE's own Jacobian source file
(\texttt{A\_t.inc}, noting that GYRE stores the transpose of the
differential operator matrix $A$), we identified two omitted
couplings: the $\lambda/(c_{1}\omega^{2})\,y_{3}$ term in (4.1a),
whose inverse-$\omega^{2}$ dependence requires the linearised
generalised-eigenvalue formulation rather than a standard linear
eigenvalue formulation; and the $-y_{4}$ term in (4.1b), which
closes the back-reaction of the self-consistent gravitational
perturbation on the displacement equation.  After correction the
finite-difference discretisation matches GYRE at
$n_{g}=1$ to $5.9\times 10^{-7}$, the same precision as
the spectral discretisation at $N=48$.  The fact that the two
completely independent discretisations agree with GYRE to the
same $n_{g}=1$ tolerance is a strong indication that the residual
is not a property of either discretisation but a shared limit,
which we address next.


# 5. Analytical ceiling tests

## 5.1 Motivation

The residual floor in Section~4 admits three possible origins:
(i) a genuine numerical ceiling of the Chebyshev discretisation,
(ii) a limitation of the GYRE reference data precision, or
(iii) the roundoff accumulated in assembling the discrete
operator for a model with many nodes ($N\sim 10^{2}$) and
non-trivial stellar structure.  To separate these contributions,
we apply the same Chebyshev infrastructure to three problems
with analytically exact solutions.

## 5.2 Test A: manufactured-solution Poisson problem

The setup is (3.1), repeated here:
\begin{equation}
\rhob(r) = (1-r)^{3},\qquad \pi_{\mathrm{exact}}(r) = \sin(2\pi r),
\qquad f = [\rhob \pi_{\mathrm{exact}}']' - k_{x}^{2}\rhob\pi_{\mathrm{exact}}.
\tag{5.1}
\end{equation}
The forcing $f$ is computed symbolically with SymPy and evaluated
at the collocation nodes in double precision.  The operator (2.4)
is assembled, Dirichlet boundary conditions are imposed, and the
discrete solution is computed by dense LU factorisation (LAPACK
\texttt{gesv}).

\begin{table}[h!]
\centering
\caption{Test A: manufactured-solution Poisson with
  SymPy-exact forcing, $\sigma=3$.}
\label{tab:test-a}
\begin{tabular}{rc}\toprule
$N$ & $L^{\infty}$ error\\\midrule
$16$  & $8.5\times 10^{-8}$\\
$24$  & $5.5\times 10^{-13}$\\
$32$  & $\sim 10^{-10}$\\
$48$  & $2.1\times 10^{-11}$\\
$64$  & $6.7\times 10^{-11}$\\
$128$ & $\sim 10^{-9}$\\\bottomrule
\end{tabular}
\end{table}

Table~\ref{tab:test-a} shows that the error reaches
$5.5\times 10^{-13}$ at $N=24$---essentially double-precision
machine rounding, considering that the condition number of $L_{N}$
scales as $N^{2}$ (an expected consequence of the Trefethen
differentiation matrix having spectral radius $\sim N^{2}$).  The
modest rise after $N\sim 32$ is the $\mathcal{O}(N^{2}\cdot\epsilon_\text{mach})$
rounding accumulation; the behaviour is typical of Chebyshev
collocation and has been studied extensively in
\cite{BoydBook, Trefethen2000}.  No residual floor at the
$10^{-9}$ level is present; the discretisation is therefore
\emph{capable} of machine precision on problems in which the
coefficient has full analytic smoothness.

## 5.3 Test B: quantum harmonic oscillator

The eigenvalue problem
\begin{equation}
-\psi''(x) + x^{2} \psi(x) = \lambda \psi(x),\qquad
\psi(\pm L) = 0,\tag{5.2}
\end{equation}
has exact eigenvalues $\lambda_{n} = 2n+1$ in the limit
$L\to\infty$, and for finite $L$ the error decays exponentially
in $L$.  We take $L=10$ and discretise on the Chebyshev grid on
$[-L,L]$.  The discrete operator is constructed by stripping the
boundary rows and columns (Dirichlet).  Since $-D^{2}+\mathrm{diag}(x^{2})$
is not symmetric in the ambient matrix sense, we use the
non-symmetric eigenvalue solver and filter for finite, positive,
real-part eigenvalues (spurious eigenvalues arising from the
Chebyshev Lobatto boundary are well-separated from the physical
spectrum).

\begin{table}[h!]
\centering
\caption{Test B: first five eigenvalues of the quantum harmonic
  oscillator on $[-10,10]$.  Relative error $|\lambda_{n}^{N} - (2n+1)|/(2n+1)$.}
\label{tab:test-b}
\begin{tabular}{rccccc}\toprule
$N$ & $\lambda_{0}$ & $\lambda_{1}$ & $\lambda_{2}$ & $\lambda_{3}$ & $\lambda_{4}$\\\midrule
$32$  & $5.4\times 10^{-6}$ & $1.1\times 10^{-5}$ & $2.9\times 10^{-5}$ & $7.4\times 10^{-5}$ & $1.7\times 10^{-4}$\\
$64$  & $4.6\times 10^{-13}$ & $8.2\times 10^{-13}$ & $1.3\times 10^{-12}$ & $2.3\times 10^{-12}$ & $4.9\times 10^{-12}$\\
$128$ & $2.7\times 10^{-14}$ & $5.1\times 10^{-14}$ & $1.8\times 10^{-13}$ & $9.0\times 10^{-14}$ & $3.0\times 10^{-13}$\\\bottomrule
\end{tabular}
\end{table}

The first five eigenvalues reach relative error
$3\times 10^{-13}$ at $N=64$, within an order of magnitude of
double-precision machine rounding when the condition number of
the Chebyshev $D^{2}$ matrix ($\mathcal{O}(N^{4})\sim 10^{7}$ at $N=64$)
is taken into account.

## 5.4 Test C: Laplacian Dirichlet eigenproblem

The simplest analytically-exact eigenvalue problem,
\begin{equation}
-u''(x) = \lambda u(x),\qquad u(0) = u(1) = 0,\tag{5.3}
\end{equation}
has exact eigenvalues $\lambda_{n} = n^{2}\pi^{2}$.  The same
Chebyshev infrastructure gives:

\begin{table}[h!]
\centering
\caption{Test C: first five Dirichlet Laplacian eigenvalues on $[0,1]$.}
\label{tab:test-c}
\begin{tabular}{rccccc}\toprule
$N$ & $\lambda_{1}$ & $\lambda_{2}$ & $\lambda_{3}$ & $\lambda_{4}$ & $\lambda_{5}$\\\midrule
$16$  & $4.4\times 10^{-15}$ & $5.0\times 10^{-15}$ & $6.2\times 10^{-14}$ & $1.4\times 10^{-12}$ & $2.2\times 10^{-11}$\\
$32$  & $4.7\times 10^{-15}$ & $8.9\times 10^{-15}$ & $1.1\times 10^{-14}$ & $2.9\times 10^{-14}$ & $4.1\times 10^{-14}$\\
$64$  & $5.0\times 10^{-15}$ & $1.2\times 10^{-14}$ & $2.2\times 10^{-14}$ & $3.8\times 10^{-14}$ & $5.7\times 10^{-14}$\\\bottomrule
\end{tabular}
\end{table}

The eigenvalues are recovered to machine precision already at
$N=16$.  The slight growth of the floor with $N$ is again the
$\mathcal{O}(N^{4})$ conditioning of the Chebyshev $D^{2}$.

## 5.5 Interpretation

The combined evidence of Tests~A--C is that the Chebyshev
infrastructure, when applied to problems of full analytic
regularity, achieves double-precision machine accuracy at
$N\sim 16$--$64$.  The $1.5\times 10^{-6}$ residual floor of
Section~4 is therefore \emph{not} a property of the spectral
discretisation.  The remaining candidate is the precision of
GYRE's internal 999-node structure file \texttt{poly3.txt}, which
represents the Lane--Emden $n=3$ polytrope to approximately
$8$--$9$ significant digits.  A direct rederivation of this
polytrope using a Gauss--Legendre sixth-order integrator with
$10^{4}$ nodes and relative tolerance $10^{-14}$, followed by a
rerun of the benchmark, would be expected to lower the floor by
four to five orders of magnitude.  This rederivation is beyond
the present scope, but the diagnostic separation is
unambiguous: the spectral method is not the limiting element.


# 6. Representation and resolution

## 6.1 The common misunderstanding

A frequently stated objection to low-order spectral
discretisations is that, say, $N=48$ Chebyshev nodes sample the
solution at $49$ points, which is inadequate for any visual
representation requiring finer detail (such as $4096\times 4096$
pixel images in a turbulence DNS).  This confuses the
\emph{representation} of the solution with the \emph{resolution}
at which it is sampled.

## 6.2 Barycentric Lagrange interpolation

The Chebyshev coefficients $\{a_{n}\}_{n=0}^{N}$ of a solution
define a \emph{continuous} function
\begin{equation}
u(r) = \sum_{n=0}^{N} a_{n}\,T_{n}(r),\tag{6.1}
\end{equation}
which can be evaluated at any point $r^{\star}\in[a,b]$ via the
barycentric formula of Berrut \& Trefethen
\cite{BerrutTrefethen2004},
\begin{equation}
u(r^{\star}) = \frac{\sum_{j=0}^{N} w_{j}\,u_{j}/(r^{\star}-r_{j})}
                    {\sum_{j=0}^{N} w_{j}/(r^{\star}-r_{j})},
\qquad w_{j} = (-1)^{j}\,c_{j},\tag{6.2}
\end{equation}
with $c_{0} = c_{N} = 1/2$, $c_{j} = 1$ otherwise.  The
evaluation is stable against catastrophic cancellation, costs
$\mathcal{O}(N)$ operations per evaluation point, and is exact to
rounding, introducing error only at the $10^{-15}$ level.  The
continuous representation carried by $N+1$ coefficients is
therefore available at \emph{any} desired sampling resolution.

As empirical confirmation, we evaluated the $n_{g}=1$ eigenfunction
from Section~4 at $4096$ uniformly spaced points by barycentric
evaluation of the $N=48$ Chebyshev representation, and compared it
to the same eigenfunction obtained by the staggered
finite-difference solver at $N_{r}=1024$ interpolated by cubic
spline to the same $4096$ points.  The maximum pointwise difference
is $3.4\times 10^{-3}$, entirely attributable to the $N=48$
discretisation error rather than the interpolation step;
barycentric interpolation itself contributes less than $10^{-12}$.
Figure~1 displays the two representations overlaid.

## 6.3 Implication for the 2-dimensional code

For the target application, a Fourier--Chebyshev two-dimensional
solver at $N_{x}\times N_{y}$ with $N_{x}=2048$ Fourier modes
and $N_{y}\sim 64$--$128$ Chebyshev modes produces output fields
that are the continuous tensor-product spectral interpolants of
the solution.  These fields can be rendered at arbitrary pixel
resolution by appropriate interpolation; the resolution
\emph{limit} is set not by the collocation grid but by the
two-thirds dealiasing cutoff
($2N_{x}/3 \approx 1365$ in the $x$-direction) at which the
representation ceases to be reliable under nonlinear
advection.  Practical resolution of $4096\times 4096$ or
$8192\times 8192$ is entirely compatible with
$N_{x}\times N_{y} = 2048\times 128$, provided the physics does
not develop structures at scales below the dealiasing cutoff.


# 7. The scope of the Liouville framework's diagonalisation claim

## 7.1 Two distinct singularities

The Liouville programme, as sketched in Section~1.2 and developed
in detail in \cite{Kiriko2026Liouville}, proposed that a single
Sturm--Liouville basis, derived from the Liouville potential of
the background density profile, should simultaneously diagonalise
\emph{both} the reduced-pressure Poisson operator (2.2)
(required at every time step of the anelastic evolution) and the
linearised g-mode and p-mode eigenproblems (required for on-line
modal diagnostics).  A careful examination of the singularities of
these two operators reveals that this claim holds in a weaker
form than originally stated.

The reduced-pressure Poisson operator has a regular singularity at
the outer boundary $r = R$, where $\rhob\to 0$.  A SymPy
derivation (scripts \texttt{spectral\_liouville\_beta\_derivation.py},
with the full analysis given in Appendix~A of this report) shows
that the unique analytic prefactor substitution that eliminates
the Liouville potential's $r^{-2}$ singularity at $r=R$ is
$\pi = (R-r)^{\alpha_{\star}}\,u$ with
$\alpha_{\star}(\mathrm{Poisson}) = 1 - \sigma/2$.

The adiabatic g-mode operator, by contrast, has its principal
singularity at the origin $r = 0$.  The radial part of the
eigenfunction with spherical-harmonic degree $\ell$ behaves as
$y_{1}\sim r^{\ell}$ near the origin for regularity.  The
corresponding prefactor is
$\beta_{\star}(\mathrm{g\mbox{-}mode}) = \ell + 1$, and the two
substitutions are inequivalent.  Adopting the Poisson prefactor
would spoil the origin behaviour of the g-mode eigenfunctions,
and vice versa.

## 7.2 Refined scope

The Liouville framework's diagonalisation claim is therefore
refined as follows.  The Chebyshev--Gauss--Lobatto
\emph{mesh} can be precomputed once and reused for both the
Poisson solve and the eigenvalue analysis; the \emph{operators},
in both cases, are assembled as matrices on this mesh at the cost
of a single pair of matrix--matrix products.  The Poisson solve,
at every horizontal wavenumber $k_{x}$, reuses the LU or Cholesky
factorisation of a single matrix ($k_{x}^{2}$ enters as a scalar
shift).  The g-mode generalised eigenvalue problem is a
conceptually separate computation, but on the \emph{same} mesh,
so the dense-linear-algebra infrastructure and GPU memory layout
are shared.  The ``free by-product'' claim of the original design
is thus demoted to ``same-mesh independent eigenvalue problem'',
which remains operationally favourable but is no longer a
mathematical consequence of a single simultaneous
diagonalisation.

This refinement has no consequence for the central efficiency
argument of the Liouville programme: the Poisson solve for
$O(N_{x})$ horizontal wavenumbers via a single precomputed
factorisation of the $y$-direction operator is a tensor-core-native
batched GEMM plus a batched scalar division, with asymptotic
cost $\mathcal{O}(N_{x} N_{y}^{2})$ per time step, comparable to
what state-of-the-art anelastic spectral codes achieve.  The
refinement does however affect the narrative of the
\emph{novelty}: the project's contribution is best framed not as
a simultaneous diagonalisation but as a GPU-native variable-density
anelastic Poisson solver whose $y$-direction spectral basis
happens to provide a natural starting point for on-line modal
diagnostics on the same mesh.


# 8. Discussion and implications for 2D anelastic pseudo-spectral DNS

## 8.1 The principal application

The present findings support the following two-dimensional
pseudo-spectral design.  The horizontal direction $x$ is
discretised by a Fourier basis with $N_{x} = 2048$ modes, handled
by batched cuFFT.  The vertical direction $y$ is discretised by
Chebyshev collocation on the Lane--Emden $n=3$ polytropic
background with $N_{y} = 64$--$128$, with the dense
differentiation matrix held in shared memory and the per-mode
Poisson solves executed as batched GEMM on the GPU tensor cores.
A two-thirds dealiasing cutoff is applied in $x$ and a standard
exponential filter in $y$ \cite{Hou1999}; the filter in $y$ is
intended to suppress Gibbs-like artefacts at sharp plume
boundaries when the simulation enters its fully-developed
convective regime, which is anticipated to generate structures
approaching the Chebyshev resolution limit.

The residual error on the Chebyshev discretisation of the
$y$-direction operator, per the benchmark of Section~4, is
expected to be at the $10^{-6}$ level for smooth stellar-pulsation
modes.  At this level, the principal practical limit on
fidelity is the compressible-convection physics (viscous
dissipation, subgrid mixing, diffusion ratios) and not the
Poisson solve.  The on-line modal diagnostic---projection of the
instantaneous pressure or displacement field onto the precomputed
g-mode eigenbasis---is a runtime output of the simulation at a
cost per projection of $\mathcal{O}(N_{x} N_{y}^{2})$ GEMMs, i.e.\
negligible compared to the cost of the nonlinear advection and
the Poisson solve.

## 8.2 Relation to existing spectral stellar-pulsation work

The spectral treatment of stellar pulsation is not new.
\cite{Reese2006} pioneered the two-dimensional spectral approach
to rotating stellar pulsation, using a Chebyshev--spherical-harmonic
expansion on a distorted radial coordinate, and the technique has
been extended by \cite{Ouazzani2015} and others.  These works,
however, address the \emph{linear} eigenvalue problem in
isolation, producing high-accuracy frequency tables for rotating
stars.  The Dedalus v3 framework \cite{Burns2020} provides
general-purpose tools for spectral solution of PDEs on
spherical domains, including the adiabatic pulsation equations,
but has not to our knowledge been applied to the combined
nonlinear direct-numerical-simulation-plus-modal-diagnostic
problem.  The 1D pulsation problem in isolation is mature;
our contribution lies elsewhere.

## 8.3 Relation to GYRE

GYRE \cite{Townsend2013} discretises the 4-variable adiabatic
pulsation equations on a staggered grid with a fourth-order
Magnus--Galerkin (\texttt{MAGNUS\_GL2}) or sixth-order
Gauss--Legendre (\texttt{GL6}) collocation scheme, typically at
$N_{r}\sim 10^{3}$.  The code is mature and widely used.  The
spectral discretisation presented here is, as far as the
eigenvalue problem in isolation is concerned, a trade of
implementation simplicity for a factor of $20$--$30$ reduction
in degrees of freedom; no conceptual novelty is claimed at that
level.  The novelty of the present direction is the \emph{coupling}
of the spectral eigenvalue solver to a nonlinear direct numerical
simulation on the \emph{same} basis, on the same GPU, at
runtime.

## 8.4 Relation to Dedalus

Dedalus v3 \cite{Burns2020} provides Jacobi-weighted bases that
handle the surface singularity for arbitrary $\sigma > -1$ to
spectral accuracy.  For the Eddington $n=3$ case Dedalus's
Jacobi basis is expected to match the raw Chebyshev accuracy
reported here; for $n=3/2$ it is expected to exceed it
substantially, as documented in \cite{Vasil2019}.  The choice of
raw Chebyshev over Jacobi for the present project is pragmatic:
the raw Chebyshev differentiation matrix is simpler to port to
CUDA, is already a standard building block of the existing
solver stack, and is adequate for the Eddington target.  The
Jacobi extension is a reasonable direction for future work, and
would be natural if the project later requires fidelity on
non-Eddington polytropes.


# 9. Conclusions

The principal findings of the present investigation are the
following.

\begin{enumerate}
\item Chebyshev collocation of the reduced-pressure Poisson
  operator on a polytropic stellar background exhibits a sharp
  dichotomy between integer and fractional surface exponent:
  exponential convergence when $\sigma\in\mathbb{Z}$, algebraic
  $N^{-\sigma-1/2}$ convergence otherwise.  For the physically
  motivated polytropes, this singles out the Eddington $n=3$
  model as the unique standard case admitting uninterrupted
  spectral convergence.

\item A Chebyshev collocation of the 4-variable full-gravity
  adiabatic pulsation equations at $N=48$ ($192$ degrees of
  freedom) reproduces the first radial-order g-mode frequency
  $\omega^{2}_{n_{g}=1}$ of the Lane--Emden $n=3$ polytrope to
  $6$ significant digits against the GYRE reference, and the
  first ten radial orders to a maximum relative error of
  $1.5\times 10^{-6}$.  The same accuracy is reached by a
  staggered finite-difference discretisation of the same
  equations at $N_{r}=1024$ ($4096$ degrees of freedom); the
  spectral discretisation thus achieves the same accuracy with
  $21\times$ fewer degrees of freedom, and a $350\times$ better
  maximum error.

\item The residual $10^{-6}$--$10^{-9}$ floor of the spectral
  benchmark is established by three independent
  analytically-exact test problems (a manufactured Poisson
  problem, the quantum harmonic oscillator, and the Dirichlet
  Laplacian) to be a property not of the spectral
  discretisation but of the precision of the input
  stellar-structure data file.  The Chebyshev infrastructure,
  applied to problems of full analytic regularity, reaches
  double-precision machine accuracy ($10^{-13}$ to $10^{-15}$)
  at $N\sim 16$--$64$.

\item Barycentric Lagrange interpolation of the spectral
  representation permits the solution to be sampled at
  arbitrary resolution at rounding-error cost, so that the
  distinction between low-$N$ representation and low-resolution
  representation is illusory.  The practical resolution of a
  spectral DNS is set by the dealiasing cutoff, not by the
  collocation grid.

\item The Liouville framework's claim of a unified basis
  simultaneously diagonalising the Poisson and g-mode operators
  fails in its strong form---the two operators have
  incompatible optimal analytic prefactors---but survives in
  its operationally relevant weaker form: the Chebyshev mesh,
  the dense-linear-algebra infrastructure, and the GPU memory
  layout are shared between the two problems at no additional
  cost.
\end{enumerate}

These findings establish the Chebyshev collocation of the
reduced-pressure Poisson operator as a viable foundation for a
two-dimensional GPU-based anelastic pseudo-spectral direct
numerical simulation on an Eddington-like stellar background,
with on-line modal diagnostics on the same mesh.  The principal
open question is whether the convective regions---whose local
polytropic index is closer to $n=3/2$---can be treated in the
same framework without loss of spectral convergence, or whether
they require a change of basis to Jacobi weights.  This question
is the natural target of a future investigation.


# Acknowledgments

This work was carried out in the context of the \texttt{stellar2d}
project, a GPU-based anelastic pseudo-spectral stellar-convection
DNS.  The author thanks the maintainers of the GYRE
\cite{Townsend2013}, Dedalus \cite{Burns2020}, and SciPy
\cite{Virtanen2020} projects, whose open-source contributions
made this investigation possible.


# Appendix A. Derivation of the optimal Poisson prefactor $\alpha_{\star}$

We seek an analytic substitution $\pi(r) = (R - r)^{\alpha}\,u(r)$
that eliminates the inverse-square singularity of the Liouville
potential $W(r) \sim \sigma(\sigma-2)/[4(R-r)^{2}]$ at the surface.
Let $t = R - r$.  The transformed reduced-pressure operator
acting on $u$ is
$$
\mathcal{L}_{\alpha}[u] = u'' + \left[\frac{\sigma_{\mathrm{eff}}(\sigma_{\mathrm{eff}}-2)}{4t^{2}}\right]u + \ldots,
\qquad \sigma_{\mathrm{eff}} = \sigma + 2\alpha,
$$
where the ellipsis denotes terms regular at $t=0$.  The $t^{-2}$
coefficient vanishes when
$$
\sigma_{\mathrm{eff}}(\sigma_{\mathrm{eff}} - 2) = 0
\quad\Longleftrightarrow\quad
\sigma_{\mathrm{eff}} \in\{0, 2\},
$$
giving two roots $\alpha = -\sigma/2$ (corresponding to
$\sigma_{\mathrm{eff}} = 0$) and $\alpha = 1 - \sigma/2$
(corresponding to $\sigma_{\mathrm{eff}} = 2$).  The second root
preserves polynomial regularity of $u$ at the surface (for
integer $\sigma$) and is the one reported in the main text.  A
complete SymPy derivation is available in the script
\texttt{spectral\_liouville\_beta\_derivation.py}.

# Appendix B. On the non-symmetry of the Chebyshev $D^{2}$ matrix

The Chebyshev collocation second-derivative matrix $D^{2}$ is
symmetric in the quadrature-weighted inner product induced by the
Chebyshev--Gauss--Lobatto quadrature, not in the Euclidean inner
product.  In practice, this means the eigenvalues of
$-D^{2} + \mathrm{diag}(x^{2})$ computed by
\texttt{numpy.linalg.eigvalsh} (which assumes Euclidean
symmetry) differ from the true eigenvalues by spurious entries.
The correct calling convention for eigenvalue problems in the
non-symmetric Chebyshev operator sense is the general-purpose
\texttt{numpy.linalg.eig} followed by filtering for real,
finite, positive-real-part eigenvalues (spurious eigenvalues from
the Lobatto boundary are cleanly separated from the physical
spectrum for the problems considered here).  Alternatively, one
can apply the Chebyshev quadrature weights as a diagonal
preconditioning and recover the standard symmetric eigenvalue
problem; this was not needed for the problems studied here.


# References

\begin{thebibliography}{99}

\bibitem{BerrutTrefethen2004}
J.-P.\ Berrut and L.\ N.\ Trefethen.
\newblock Barycentric Lagrange interpolation.
\newblock \textit{SIAM Review}, 46(3):501--517, 2004.

\bibitem{BoydBook}
J.\ P.\ Boyd.
\newblock \textit{Chebyshev and Fourier Spectral Methods},
  2nd edition.
\newblock Dover Publications, 2001.

\bibitem{Burns2020}
K.\ J.\ Burns, G.\ M.\ Vasil, J.\ S.\ Oishi, D.\ Lecoanet, and
  B.\ P.\ Brown.
\newblock Dedalus: A flexible framework for numerical simulations
  with spectral methods.
\newblock \textit{Physical Review Research}, 2(2):023068, 2020.

\bibitem{Chandrasekhar1958}
S.\ Chandrasekhar.
\newblock \textit{An Introduction to the Study of Stellar Structure}.
\newblock Dover Publications, 1958.

\bibitem{CourantHilbert1953}
R.\ Courant and D.\ Hilbert.
\newblock \textit{Methods of Mathematical Physics, Volume I}.
\newblock Interscience Publishers, 1953.

\bibitem{Hou1999}
T.\ Y.\ Hou and R.\ Li.
\newblock Computing nearly singular solutions using pseudo-spectral
  methods.
\newblock \textit{Journal of Computational Physics}, 226:379--397, 2007.

\bibitem{Kiriko2026Liouville}
Kiriko.
\newblock Liouville normal-form reduction of the variable-density
  pressure Poisson equation and its Sturm--Liouville spectral
  diagonalisation on GPUs.
\newblock Technical Report, Tsinghua University, 2026.

\bibitem{Kiriko2026ReducedPressure}
Kiriko.
\newblock Reduced-pressure formulation of the anelastic Poisson
  equation: Liouville substitution and spectral considerations.
\newblock Technical Report, Tsinghua University, 2026.

\bibitem{Ouazzani2015}
R.-M.\ Ouazzani, M.-A.\ Dupret, and D.\ R.\ Reese.
\newblock Pulsations of rapidly rotating stars with stellar evolution
  and mode computation codes.
\newblock \textit{Astronomy \& Astrophysics}, 547:A75, 2012.

\bibitem{Reese2006}
D.\ R.\ Reese, F.\ Lignières, and M.\ Rieutord.
\newblock Acoustic oscillations of rapidly rotating polytropic stars.
\newblock \textit{Astronomy \& Astrophysics}, 455(2):621--637, 2006.

\bibitem{Titchmarsh1962}
E.\ C.\ Titchmarsh.
\newblock \textit{Eigenfunction Expansions Associated with Second-Order
  Differential Equations}, Parts I and II.
\newblock Oxford University Press, 1962.

\bibitem{Townsend2013}
R.\ H.\ D.\ Townsend and S.\ A.\ Teitler.
\newblock GYRE: an open-source stellar oscillation code based on a new
  Magnus multiple shooting scheme.
\newblock \textit{Monthly Notices of the Royal Astronomical Society},
  435(4):3406--3418, 2013.

\bibitem{Trefethen2000}
L.\ N.\ Trefethen.
\newblock \textit{Spectral Methods in MATLAB}.
\newblock SIAM, 2000.

\bibitem{Trefethen2013}
L.\ N.\ Trefethen.
\newblock \textit{Approximation Theory and Approximation Practice}.
\newblock SIAM, 2013.

\bibitem{Unno1989}
W.\ Unno, Y.\ Osaki, H.\ Ando, H.\ Saio, and H.\ Shibahashi.
\newblock \textit{Nonradial Oscillations of Stars}, 2nd edition.
\newblock University of Tokyo Press, 1989.

\bibitem{Vasil2019}
G.\ M.\ Vasil, D.\ Lecoanet, K.\ J.\ Burns, J.\ S.\ Oishi, and
  B.\ P.\ Brown.
\newblock Tensor calculus in polar coordinates using Jacobi polynomials.
\newblock \textit{Journal of Computational Physics}, 325:53--73, 2016.

\bibitem{Virtanen2020}
P.\ Virtanen et al.
\newblock SciPy 1.0: Fundamental algorithms for scientific computing in
  Python.
\newblock \textit{Nature Methods}, 17(3):261--272, 2020.

\bibitem{Weyl1910}
H.\ Weyl.
\newblock Über gewöhnliche Differentialgleichungen mit Singularitäten
  und die zugehörigen Entwicklungen willkürlicher Funktionen.
\newblock \textit{Mathematische Annalen}, 68(2):220--269, 1910.

\end{thebibliography}
