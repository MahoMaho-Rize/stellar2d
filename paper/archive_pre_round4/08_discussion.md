# 8. Discussion

## 8.1 Scope and limits of the present framework

The framework developed in Sections 2--7 targets two-dimensional
anelastic flow in a horizontally periodic, radially wall-bounded
domain with a time-independent radial stratification $(\rho_0(y),
N^2(y))$.  The Liouville-SL spatial machinery requires only that
the background is hydrostatic and smooth enough for the
Sturm--Liouville eigenvalue problem to be well-posed on the CGL
grid after surface-cut regularisation.  Integer-polytrope indices
admit exponential spectral convergence; half-integer indices give
algebraic $\mathcal O(N_y^{-2\sigma - 1})$ convergence which is
adequate in practice for $N_y \gtrsim 48$.

The assembled-matrix time-stepping of Section 6 makes no
smoothness or integer-$n$ assumptions: its closure property
follows from the construction $\mathsf M = \mathsf L^{-1}\mathsf R$
and is independent of the $\rho_0, N^2$ profiles that enter
$\mathsf L, \mathsf R$.  The only constraint is that the
linearised anelastic system can be written as a single
second-order ODE in time for $V$; this holds for incompressible
anelastic, pseudo-incompressible, and the Boussinesq limit, but
not for the fully compressible system where the acoustic modes
couple $V$ to $p$ dynamically.

The Strang-split nonlinear extension of Section 7 inherits the linear
block's machine-precision closure and adds $\mathcal O(\Delta t^4)$
splitting error on top, independent of background.
Its stability under finite-amplitude stellar pulsation is
empirically bounded by the advection CFL rather than by any
g-mode CFL; physical amplitudes up to $\mathcal O(10^{-1})$ of
the g-mode velocity scale are stable over tens of periods in the
Python prototype.

**What this framework does not address:**

  - *Fully compressible acoustic cutoff.*  Stars have both g-modes
    and p-modes.  The anelastic approximation filters out p-modes
    by construction.  A fully compressible spectral DNS with
    sound-wave physics would need a different time-stepping
    architecture (e.g., Godunov-type flux upwinding for the
    acoustic subsystem, or a multiple-timescale IMEX).

  - *Two-dimensional axisymmetric (cylindrical or spherical)
    geometry.*  The present framework is Cartesian; it maps
    naturally to a 2D-periodic local box or a local Cartesian
    chunk of an atmospheric model.  Extension to full-sphere
    surface harmonics would replace Fourier with
    spherical-harmonic transforms and SL with a tensor-product
    SL on $(\theta, r)$; this is architectural work but not a
    new mathematical construction.

  - *Convective overshoot with adiabatic parcels.*  The linear-
    stratification model captures wave propagation but not the
    intermittent parcel motion of active convection.  For
    convective-envelope stars the present framework provides
    the g-mode wave tank into which a separate convective
    subgrid model would inject sources.

  - *Rotation and magnetic fields.*  The 4-variable GYRE benchmark
    in Section 4 is the non-rotating, non-magnetic adiabatic
    case.  Including rotation and magnetism adds two-pressure
    and vector-potential variables to the EVP, enlarges the
    assembled-matrix footprint proportionally, but does not
    alter the operator-consistency argument of Section 6.

## 8.2 Relation to GYRE

GYRE is the de facto reference for adiabatic non-rotating stellar
pulsation.  It operates in one dimension (radial only) on the
pre-computed stellar structure produced by an evolution code
(MESA, CESAM, or a polytropic solver).  Its output is a discrete
set of eigenvalues and eigenvectors for a given $(\ell, n_g)$.
This paper's Section 4 benchmark uses GYRE to validate the
spatial machinery; our framework reproduces GYRE eigenvalues to
$3.6 \times 10^{-5}$ at $N_r = 96$.

GYRE does not perform time-domain integration; it is an
eigenvalue solver only.  The natural division of labour is that
GYRE computes the linear spectrum that initial conditions for our
time-domain solver are decomposed in, and our framework then
evolves those initial conditions forward under the full nonlinear
physics.  For the linear (small-amplitude) regime the two codes
should (and do, to $\mathcal O(10^{-5})$) agree.  For finite
amplitudes our framework provides information that GYRE cannot,
namely mode coupling, energy cascades, and nonlinear saturation
phenomena.

The maximum relative error in our GYRE-reproduction is currently
limited by linear interpolation of the GYRE-shipped 1001-point
structure file onto our 96-point CGL grid; cubic spline
interpolation would further reduce this, but the resulting
improvement is not visible at the physical amplitudes of
interest and we have not implemented it.

## 8.3 Relation to Dedalus

Dedalus is a general-purpose spectral framework for
fluid-dynamics problems with user-specified equations.  It
supports Chebyshev, Jacobi, and other SL bases on
non-periodic directions through a unified ``sparse-matrix
tau method'' formulation, with implicit time-stepping (IMEX,
SBDF) for the stiff linear terms.

Architecturally, Dedalus is closest in spirit to the framework of
this paper: both use spectral methods in a non-periodic
direction with SL bases, and both couple these to implicit time
integration of the linear part.  The differences are as
follows.  Dedalus implements the tau method, in which boundary
conditions are enforced by adding tau rows to the spectral
matrix and solving the augmented system.  Our framework uses
interior restriction and explicit boundary conditions, which
gives smaller matrices at the cost of forgoing tau's cleaner
handling of corner boundary-condition compatibility.  Dedalus's
linear solver is Krylov-based and iterative; ours is direct
Gauss--Jordan at setup time, applied as a stored dense matrix
per step.  For the $\mathcal O(N_y) \le 100$ problems we target
this direct choice is faster and gives bit-reproducible output;
for much larger $N_y$ the iterative choice of Dedalus scales
better.

The time-stepping decision --- Dedalus's implicit IMEX versus our
Strang operator split with exact linear block --- is an
engineering trade-off rather than a fundamental difference.  We
selected Strang for the ease of integration with a pre-existing
CUDA pseudo-spectral code; Dedalus users with the same problem
would likely select an IMEX-BDF3 scheme.  Our Section 7
comparison suggests that for g-mode dominated flow these two
choices give indistinguishable finite-amplitude accuracy, so the
decision is driven by implementation ergonomics.

An important observation is that Dedalus's tau-method linear
solver, being a correct discretisation of the assembled operator
$\mathsf L$, is \emph{implicitly} the assembled-matrix
construction of our Section 6.  Dedalus users do not encounter
the primitive-node operator mismatch of Section 5 because the
tau-method route never factors the operator into its primitive
pieces.  Our contribution is thus not to \emph{invent}
assembled-matrix time-stepping --- it is implicit in any
tau-method spectral code --- but to \emph{diagnose} why a
primitive-node code exhibits the mismatch and to construct an
explicit assembled operator $\mathsf M = \mathsf L^{-1}\mathsf R$
that can be bolted onto a pre-existing primitive-node
pseudo-spectral infrastructure with minimal architectural change.

## 8.4 Relation to earlier stratified spectral work

Heinrichs (1991) showed that Chebyshev collocation of
incompressible Navier--Stokes on non-periodic domains requires
careful enforcement of the continuity constraint through an
explicit Leray projection, not through a primitive-variable
semi-discrete scheme.  The failure mode is the same as ours ---
primitive-node factorisation of a composite constrained operator
---  and the resolution in both cases is to apply the composite
operator as a single matrix rather than as a product of its
primitive pieces.  Our Section 5.2 statement of the discrete
Leibniz defect generalises the Leray-projection observation to
variable-coefficient elliptic operators.

Canuto, Hussaini, Quarteroni, Zang's spectral-methods treatise
discusses SL-basis methods for radially variable
coefficients and notes that the Galerkin formulation with the
appropriate weight function inherits the SL eigenvalue
spectrum of the continuous operator by construction.  Our
Section 6 reformulates this observation as a time-stepping
principle: the Galerkin-projected $\mathsf M$ is both the
assembly-consistent matrix for $V$ and the generator for the
oscillator block in RK4.

Stellar-convection pseudo-spectral codes such as Snodgrass
et al.'s and Beaudoin et al.'s have targeted variable-density
anelastic convection by FD or hybrid FD/spectral schemes; to our
knowledge no production code targets spectral accuracy on a
Lane--Emden-like stratification in the time domain.  Our
framework provides this capability on a single consumer GPU,
albeit with the engineering limitations noted above.

## 8.5 Methodology remark: prototype-first algorithm selection

Sections 5 and 7 illustrate a working pattern that we find
useful for spectral-method development.  Each decision point in
the workflow --- choice of basis, choice of variable, choice of
time integrator, choice of splitting --- is first tested in a
lightweight Python prototype of a few hundred lines.  The
prototype captures the essential algebraic content of the
problem but dispenses with GPU memory layout, CUDA kernel
optimisation, and engineering hardening; it can be written in
thirty minutes and run in under a minute.

In the course of this work, two decision points were reversed
after Python prototyping: the reduced-variable time-stepping of
Section 5.4 --- $23\%$ improvement at the cost of a substantial
rewrite, deemed not worth the effort --- and the semi-implicit
IMEX scheme of Section 7 --- catastrophic instability at
amp $= 10^{-1}$, reversing the appearance of ``implicit is always
safer.''  Each reversal saved approximately one week of GPU
implementation time.

The methodology is not universal: problems dominated by
communication overhead or memory-layout effects do not show the
same CPU/GPU parity, and problems where the decision rests on
integration details of a larger code base (legacy APIs, mixed
precision, fused kernels) are poorly served by Python at all.
Our problem class --- small dense matrices per wavenumber,
FFT-friendly geometry, bandwidth-bound rather than
arithmetic-bound --- happens to be a particularly good fit for
Python-first prototyping.

## 8.6 Open questions

The framework leaves four questions open:

1. *Three-dimensional extension.*  A full 3D spherical-harmonic
   anelastic DNS of a real star is the natural follow-on
   application.  The per-harmonic assembled matrix has the
   same size as our per-$k_x$ matrix, but there are
   $\mathcal O(\ell_{\max}^2)$ harmonics versus our $n_h
   \approx \ell_{\max}$ wavenumbers; total memory scales by
   $\ell_{\max}$, which at $\ell_{\max} = 64$ is still
   manageable.  The FFT cost scales by the same factor.

2. *Non-radial stratification and magnetism.*  If $\rho_0$ or
   $N^2$ depends on the horizontal coordinate as well, the
   horizontal Fourier decomposition of Section 3 no longer
   diagonalises the elliptic operator, and the assembled matrix
   becomes full $(N_x N_y) \times (N_x N_y)$ rather than block
   diagonal in $k_x$.  This is the standard stiffness barrier of
   non-separable elliptic systems; iterative methods (multigrid,
   Krylov) are the standard response.

3. *Behaviour near the surface cutoff.*  The $\rho_{\mathrm{cut}}$
   regularisation truncates the outer envelope at a density
   level chosen for numerical convenience.  Physical wave
   reflection and tunnelling at the true surface, which a realistic
   stellar-atmosphere treatment would include, are not captured
   by our cutoff.  An inhomogeneous outer boundary condition
   (radiating or damping) would be a natural extension.

4. *Adiabatic versus non-adiabatic pulsation.*  GYRE supports a
   non-adiabatic mode (``nad'') that includes radiative
   dissipation.  Our framework is adiabatic only; extending it
   to include a phenomenological heat-diffusion term in the
   buoyancy equation is straightforward and would match one of
   GYRE's standard operating regimes.
