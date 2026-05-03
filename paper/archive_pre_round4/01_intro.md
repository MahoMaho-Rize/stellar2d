# 1. Introduction

## 1.1 Motivation: the pseudo-spectral bottleneck for stratified stars

Fully compressible two-dimensional direct numerical simulation of stellar
interiors is limited, at the resolutions accessible to a single GPU, by
the acoustic Courant condition rather than by the gravity-wave or
convective transit timescale of physical interest.  The standard
remedies replace the fully compressible equations by either the
Boussinesq, anelastic, or pseudo-incompressible systems, in all of
which the only elliptic equation that survives in the time-stepping
loop is a Poisson-like pressure problem.  For a background density
profile $\rho_0(r)$ the pressure projection that enforces mass
conservation takes the form

$$
\nabla \!\cdot\! \bigl(\rho_0(r)\,\nabla \pi\bigr) \;=\; \mathrm{RHS}.
\tag{1.1}
$$

The standard GPU-native approach is a Fourier pseudo-spectral
discretisation, because the horizontal Fourier modes diagonalise the
horizontal part of the operator and reduce each time step to a
sequence of $\mathcal O(N\log N)$ FFTs plus a local tridiagonal solve
in the non-periodic direction.  This machinery, however, *requires*
that $\rho_0$ be a constant --- otherwise the horizontal Fourier
modes do not diagonalise (1.1), and the FFT-per-step advantage
vanishes.  The Boussinesq limit is therefore the only stellar
regime for which standard pseudo-spectral codes reach
machine-precision Poisson projections at $\mathcal O(N\log N)$
cost; for any density contrast large enough to matter physically
--- g-mode propagation cavities, convective overshoot, shell
burning, stellar pulsation --- the practitioner is forced either to
give up spectral accuracy and fall back to finite differences or
finite volumes, or to give up the fast Poisson solve and accept
$\mathcal O(N^3)$ per step from dense linear algebra.

This paper reports that the trade-off is unnecessary.  A
Sturm--Liouville basis derived from the Liouville reduction of
(1.1), combined with a careful treatment of how that same SL basis
propagates into the *discrete* time-stepping operator, produces a
two-dimensional spectral code that is variable-density,
machine-precision, externally benchmarked against the GYRE
stellar-pulsation code, and implementable on a single consumer GPU.

## 1.2 Two ideas, not one

A reader familiar with stratified spectral methods might expect the
story to end at the spatial discretisation.  It does not.  We find that
the natural intuition --- "replace the Fourier basis with a SL basis
adapted to the variable density, and the time-domain problem inherits
spectral accuracy automatically" --- is false in the discrete setting,
and that diagnosing and fixing this inheritance failure constitutes a
second, independent contribution.

**Spatial idea (the Liouville substitution).**  Under the standard
Liouville transformation $\hat p = \sqrt{\rho_0}\,q$, the variable-
coefficient operator $\nabla \!\cdot\! (\rho_0\nabla\,\cdot\,)$
reduces to a Schrödinger form $q'' + W\,q - k_x^2 q = g$, for which
a single precomputed Sturm--Liouville basis $\{\psi_n\}$ simultaneously
diagonalises the elliptic problem at every horizontal wavenumber
$k_x$.  The computational pattern --- one SL eigenproblem at setup,
then a per-mode $1/(\mu_n + k_x^2)$ multiplication per time step ---
preserves the $\mathcal O(N\log N)$ Fourier topology.  The same
basis, used as the inner-product weight on the Chebyshev collocation
grid, organises the radial g-mode eigenvalue problem.  Section 3
describes this construction and documents convergence dichotomies
associated with the polytropic surface exponent; Section 4 reports an
external benchmark against the GYRE code that recovers the first ten
l=1 g-modes of a Lane--Emden $n=3$ polytrope to $3.6\times 10^{-5}$
relative error at $N_r=96$ degrees of freedom per radial field.

**Temporal idea (assembled-matrix time-stepping).**  Suppose one has
the correct spatial discretisation: the eigenpair $(\omega^2, v_{\mathrm{EVP}})$
satisfies $R\,v_{\mathrm{EVP}} = \omega^2\,L\,v_{\mathrm{EVP}}$ to machine
precision.  The natural expectation is that pushing $v_{\mathrm{EVP}}$
through the time-stepping loop of the same code will produce a
harmonic oscillation $v(t) = v_{\mathrm{EVP}}\cos(\omega t)$ with deviation
per step at or below machine precision.

This expectation is false.  The time-stepping loop of a standard
pseudo-spectral code does not apply the matrix $L$ directly; it
applies a *sequence* of primitive operations --- spectral $\partial_y$,
pointwise multiplication by $\rho_0$, pointwise multiplication by
$N^2$ --- whose composition is continuously but *not discretely*
equal to $L$.  The discrete-level discrepancy is a variable-coefficient
Leibniz defect, at leading order $\partial_y(\rho_0 \partial_y v) \neq
\rho_0\,\partial_{yy} v + \rho_0'\,\partial_y v$ on a non-uniform
collocation grid, and it is irreducible: no refinement of $N$ removes
it, because increasing $N$ changes both sides by the same amount.
We measure its effect empirically as a $6.9\times 10^{-4}$ per-step
deviation of the eigenmode under Lane--Emden $n=3/2$ stratification
(Section 5), verify in a reduced 1D Python prototype that it cannot
be fixed by changing the spatial basis alone (the Fourier-basis and
SL-basis EVP substitutions of Section 5.3), and prove by direct
numerical construction
that it vanishes to $5\times 10^{-18}$ per step when the linearised
time-stepping matrix $L^{-1}R$ is assembled and applied as a single
per-wavenumber DGEMM (Section 6).  A CUDA implementation of the
assembled-matrix propagator delivers $3\times 10^{-15}$/step on the
same problem, only three orders of magnitude above the Python
floor, the gap consistent with accumulated round-off in the
host-side Gauss--Jordan factorisation used to form $L^{-1}$.

## 1.3 Methodology: Python prototypes before GPU implementations

A secondary theme of this work is cost-controlled algorithm selection.
Each decision point in the development --- which spatial basis to use,
whether to change the time-stepping variables, which time integrator
for the linear block, how to couple linear and nonlinear blocks ---
was validated first on a standalone Python prototype of a few hundred
lines, with the corresponding CUDA implementation begun only after
the prototype had closed the relevant convergence or stability gap.
Two decisions were reversed by this procedure after two or more days
of unwritten CUDA work had been saved.  Section 7 documents the
three-path comparison for nonlinear time-stepping that rejected a
Crank--Nicolson / Adams--Bashforth IMEX scheme (unstable at
$\mathcal O(10^{-1})$ amplitude) and an exponential integrator
(equivalent to a Strang split in the physically relevant amplitude
regime, at higher per-step cost) in favour of a simple Strang split
between the assembled linear propagator of Section 6 and an RK4
advection block.

## 1.4 Contributions

We contribute:

1. A *two-ingredient* closed-form spectral framework for variable-
   density anelastic DNS, in which the Liouville-SL spatial
   discretisation and the assembled-matrix time-stepping are each
   necessary and jointly sufficient for machine-precision closure on
   realistic stellar backgrounds.

2. An external benchmark against the GYRE adiabatic non-rotating
   stellar-pulsation code, giving $3.6\times 10^{-5}$ relative error
   on the first ten l=1 g-modes of a Lane--Emden $n=3$ polytrope at
   $N_r=96$ collocation points.

3. Identification, quantification, and resolution of a discrete
   variable-coefficient Leibniz defect that blocks eigenmode
   preservation in primitive-node time-stepping of stratified
   anelastic flow.  We name the resolution the *assembled-matrix*
   or *full-Galerkin* time-stepping.

4. A systematic Python-prototype comparison of candidate remedies
   --- two EVP basis substitutions, one reduced-variable
   time-stepping, and the assembled-operator scheme for the linear
   problem; Strang-split, semi-implicit IMEX, and
   exponential-propagator schemes for the nonlinear extension ---
   that illustrates how a variable-coefficient time-stepping
   defect can masquerade as a basis problem, and how a few hours
   of prototype code redirects several weeks of GPU development.

5. A working CUDA implementation that realises the full framework
   to $3\times 10^{-15}$ per-step accuracy on a single RTX 4080 SUPER,
   released together with this paper.

## 1.5 Roadmap

Section 2 fixes notation, writes down the anelastic equations and the
Lane--Emden background that is used throughout.  Section 3 develops
the Liouville-SL spatial discretisation and discusses its convergence
properties.  Section 4 assembles the GYRE-compatible four-variable
g-mode operator and reports the external benchmark.  Section 5
exposes the time-domain operator mismatch.  Section 6 presents the
assembled-matrix resolution and its CUDA realisation.  Section 7
extends the framework to nonlinear time-stepping via operator
splitting and reports the three-path prototype comparison.  Section
8 discusses scope, limits, and relation to the existing GYRE and
Dedalus code bases.  Section 9 summarises.  All reproducers and
full source code are available at
\texttt{https://github.com/MahoMaho-Rize/stellar2d}.
