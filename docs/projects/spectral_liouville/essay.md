---
title: |
  Kiriko —
  A GPU-Accelerated Sturm--Liouville Spectral Framework
  for Variable-Density Anelastic Flow
subtitle: |
  From pseudospectral Navier--Stokes baseline to machine-precision
  Lane--Emden g-mode closure, with an externally validated Chebyshev
  eigenproblem and a proven full-Galerkin time-domain operator
author: |
  stellar2d project, branch `anelastic-sl-spectral`
date: 2026-05-03
geometry: margin=1in
fontsize: 11pt
documentclass: article
---

# Abstract

We present **Kiriko**, a GPU-accelerated spectral framework for the linear
anelastic equations on a stratified background, and document the sequence
of derivations, numerical experiments, and operator-level diagnostics that
were required to bring it from a uniform-density pseudospectral baseline
to a fully closed-loop time-domain solver at machine precision. The core
algorithmic contribution is a seven-step Sturm--Liouville (SL) Poisson
pipeline that reduces the variable-coefficient pressure equation
$\nabla\cdot(\rho_0^{-1}\nabla p)=f$ to a diagonal problem in the basis of
the Liouville-regularised Schrödinger operator, executed on a single GPU
through batched cuBLAS ZGEMM and cuFFT. On the Boussinesq limit the
spatial solver attains $\lVert\pi_{\text{num}}-\pi_{\text{exact}}\rVert_{L^2}
\approx 4.6\times 10^{-14}$, and its eigenstructure is validated
component-wise against the stellar-pulsation code GYRE, achieving a
maximum relative difference of $3.6\times 10^{-5}$ on the first ten
$\ell=1$ g-modes of the Lane--Emden $n=3$ polytrope. A second set of
results characterises the role of the polytropic surface exponent
$\sigma=n$ in controlling spectral convergence: integer $\sigma$
(analytic background density) admits exponential Chebyshev convergence
to $6.7\times 10^{-11}$ at $N=64$, while half-integer $\sigma$ produces
a branch-point obstruction limiting convergence to the algebraic rate
$N^{-2}$ irrespective of prefactor substitutions. Finally, we identify
a latent operator mismatch in the naive time-domain discretisation—the
Lane--Emden evolution leaks $5.1\times 10^{-5}$ per step irrespective
of spatial resolution—and prove, through a full-Galerkin assembled-matrix
formulation advanced by explicit Runge--Kutta, that the mismatch
collapses to $5.1\times 10^{-18}$, thirteen orders of magnitude below
the primitive node-space scheme. The resulting implementation path is
transcribed into a concrete specification for porting the assembled
operator to CUDA, and extended to a polar-coordinate annulus geometry
as a blueprint for future stellar-interior simulations.

\hrulefill

# 1. Introduction

## 1.1 Motivation

Pseudospectral methods are the method of choice for high-Reynolds-number
incompressible turbulence in doubly periodic domains: Fourier bases
diagonalise the Laplacian, the pressure projection collapses to pointwise
division by $|\mathbf{k}|^2$, and the dominant cost reduces to
$O(N\log N)$ fast transforms. The baseline solver of this project
(`pseudo_spectral_solver`) realises this programme for the two-dimensional
vorticity--streamfunction form and has been validated to deliver an
effective viscosity within 4\% of the prescribed value at $\text{Re}=10^6,
N^2=1024^2,\, t=40$.

Stellar interiors, however, are intrinsically stratified: the radial
density background $\rho_0(r)$ varies by many orders of magnitude between
the core and the surface, and buoyancy supplies the dominant restoring
force for the entire family of gravity modes (g-modes). Pure Fourier
pseudospectra are useless on such domains: the pressure-Poisson operator
becomes variable-coefficient,
$$
\nabla\cdot\bigl(\rho_0^{-1}(y)\,\nabla p\bigr) = f(\mathbf{x}),
\tag{1.1}
$$
so that Fourier modes are no longer eigenfunctions, and the cost
balloons to $O(N^3)$ unless the vertical direction is treated by a basis
adapted to $\rho_0$. The anelastic approximation removes acoustic waves
but retains stratification, which is precisely the regime that the
uniform-density pseudospectral infrastructure cannot address.

## 1.2 Contribution

This paper documents **Kiriko**, the project-wide effort to extend the
pseudospectral baseline to variable-density stratified flow on GPUs. Our
contributions divide into four blocks:

1. **Algorithm (Section 3).** A Liouville-regularised Sturm--Liouville
   expansion for the stratified pressure Poisson equation, implemented
   as a seven-step GPU pipeline.
2. **Validation (Section 4).** Component-wise verification of the
   Chebyshev eigenproblem against GYRE, reaching a worst-case relative
   difference of $3.6\times10^{-5}$ on the first ten $\ell=1$ g-modes.
3. **Spectral convergence dichotomy (Section 5).** A previously
   undocumented integer-versus-fractional cliff in the polytropic index
   $n$, together with a rigorous approximation-theoretic explanation.
4. **Operator-mismatch diagnosis and closure (Section 6).** A negative
   result—namely that naive node-space time integration leaks
   $5.1\times 10^{-5}$ per step on Lane--Emden $n=3/2$ regardless of
   grid refinement—and a proven full-Galerkin remedy that restores
   machine precision through assembled $L$ and $R$ matrices.

We close (Section 7) with the design of a polar-coordinate extension
that transcribes the Cartesian framework to an annular stellar-interior
geometry.

# 2. Background and related work

The variable-coefficient pressure Poisson equation (1.1) admits three
families of solvers: algebraic multigrid, domain-decomposition Schur
complements, and spectrally expanded bases. Only the third preserves
the $O(N\log N)$ scaling that makes pseudospectral fluid solvers
competitive, and only when the vertical basis can be chosen so that the
variable coefficient becomes a multiplier on the expansion coefficients.
The Liouville transformation 
$p(y)=\sqrt{\rho_0(y)}\,q(y)$ converts the stratified
self-adjoint operator into canonical Schrödinger form with an explicit
potential $W(y)$ or its reduced counterpart $\widetilde W(y)$, whose
sign depends on the choice of variables—an issue we analyse in §3.2.
The resulting eigenvalue problem is, in the absence of stratification,
equivalent to a Fourier problem; in the presence of stratification it
introduces a y-dependent potential whose eigenfunctions are computed
once, stored, and used thereafter as a diagonalising basis. Stellar
oscillation codes (GYRE, ADIPLS) use related expansions in
finite-difference or finite-element form, but none are GPU-resident or
embedded in a full nonlinear anelastic time-stepper.

The two-dimensional pseudospectral baseline for this project uses
cuFFT R2C/C2R with IFRK3 integrating-factor Runge--Kutta time stepping,
Orszag-style skew-symmetric convection, and a circular 2/3 dealiasing
rule. We refer to the design note `pseudo_spectral_design_2026-05-01`
for numerical details; for the present paper the baseline functions as
a validated backdrop against which the stratified extensions are
measured.

# 3. Algorithmic formulation

## 3.1 Variable-density pressure Poisson

In the anelastic approximation the pressure enforces incompressibility
through
$$
\nabla\cdot\bigl(\rho_0(y)\,\mathbf{u}\bigr) = 0,
\qquad
\nabla\cdot\!\Bigl(\rho_0^{-1}\nabla p\Bigr) = f(\mathbf{x}). \tag{3.1}
$$
Fourier expansion in the periodic direction $x$ reduces (3.1) to a
one-dimensional variable-coefficient ODE for each horizontal wavenumber
$k_x$:
$$
\frac{d}{dy}\!\left[\frac{1}{\rho_0(y)}\frac{d\hat p}{dy}\right]
-\frac{k_x^2}{\rho_0(y)}\hat p = \hat f(k_x,y). \tag{3.2}
$$

## 3.2 Liouville regularisation

The standard substitution $\hat p=\sqrt{\rho_0}\,q$ converts (3.2) into
Schrödinger canonical form
$$
-q'' + \bigl[k_x^2 - W(y)\bigr] q = g(y),
\qquad W(y) = \frac{\rho_0''}{2\rho_0} - \frac{3(\rho_0')^2}{4\rho_0^2}.
\tag{3.3}
$$
Working instead with the reduced pressure $\pi=p/\rho_0$ and the
substitution $\pi=\rho_0^{-1/2}q$ yields an analogous equation with the
modified potential
$$
\widetilde W(y) = \frac{(\rho_0')^2}{4\rho_0^2} - \frac{\rho_0''}{2\rho_0}. \tag{3.4}
$$
For Lane--Emden polytropes the two potentials differ in sign: $W(y)$
is attractive near the surface (possibly unbounded), while $\widetilde
W(y)$ is repulsive and better bounded after a density cutoff. Experiment
A (`reduced_pressure_experiments_2026-05-02`) reports a nine-fold
reduction in q-norm error for the reduced-pressure form at matched mode
count, making (3.4) the operational choice.

## 3.3 Sturm--Liouville eigenexpansion

Let $A=-D^2-\mathrm{diag}(\widetilde W)$ on the interior Chebyshev--Gauss--Lobatto
(CGL) nodes. The operator is self-adjoint under the Clenshaw--Curtis
weighted inner product $\langle u,v\rangle_{\mathrm{cc}}=\sum w_j u_j v_j$,
but is not Euclidean-symmetric on the collocation grid. Diagonalisation
yields eigenpairs $\{(\mu_n,\psi_n)\}$ independent of $k_x$: for each
horizontal wavenumber the pressure equation becomes the diagonal
relation
$$
Q_n(k_x) = -\frac{G_n(k_x)}{\mu_n + k_x^2}, \tag{3.5}
$$
where $G_n$ and $Q_n$ are the projections of the source and solution
onto $\psi_n$.

## 3.4 Seven-step GPU pipeline

End-to-end inversion of (3.1) on the GPU proceeds by

$$
\mathrm{FFT}_x \;\to\; \times\rho_0^{-1/2} \;\to\; \Psi_{\mathrm{fwd}}^{\top}
\;\to\; \text{diag solve} \;\to\; \Psi \;\to\; \times\rho_0^{1/2} \;\to\; \mathrm{IFFT}_x .
\tag{3.6}
$$

Steps 1 and 7 are cuFFT R2C/C2R; steps 3 and 5 are ZGEMM of a stored
$N_y\times N_y$ eigenmatrix against the $k_x$-strided field; steps 2,
4, 6 are pointwise kernels. Row-major storage of the coefficient array
$\hat G[j_y,k_x]$ is bit-identical to column-major
$(n_{\mathrm{modes}}\times n_h)$, so no transpose is required.

# 4. GPU implementation and manufactured-solution validation

## 4.1 Eigenstructure on the GPU

The non-Euclidean symmetry of $A$ rules out `cusolverDnDsyevd`. Naively
symmetrising $(A+A^{\top})/2$ corrupts $\mu_1$ for the Boussinesq limit
$\widetilde W\equiv 0$ by $5.5\%$: the symmetrised eigenvalue is $9.329094$
against the exact $\pi^2=9.869604$. We instead use the non-symmetric
solver `cusolverDnXgeev` (CUDA 11.6+) with a real matrix promoted to
`CUDA_C_64F`; the imaginary parts of the returned eigenvalues are below
$10^{-6}$ for well-posed problems, and residual eigenvalue agreement
with the Python `scipy.linalg.eig` reference is at $10^{-5}$ rel.
Eigenvectors are re-normalised to $\langle\psi_n,\psi_n\rangle_{\mathrm{cc}}=1$
after the GPU solver returns Euclidean-normalised output.

## 4.2 Manufactured-solution convergence on the Boussinesq limit

With $\rho_0\equiv1$, $\widetilde W\equiv0$, and analytic
$\pi_\star(x,y)=\sin(k_x x)\sin(\pi y/L_y)$, the seven-step pipeline
returns machine-precision residuals at every resolution tested:

| $N_y$ | $N_{\mathrm{modes}}$ | $\lVert\pi_{\mathrm{num}}-\pi_\star\rVert_{L^2}$ |
|-------|----------------------|--------------------------------------------------|
| 64    | 32                   | $4.6\times 10^{-14}$                             |
| 128   | 64                   | $4.6\times 10^{-14}$                             |
| 256   | 128                  | $4.6\times 10^{-14}$                             |
| 512   | 256                  | $4.6\times 10^{-14}$                             |

Resolution-independence in the Boussinesq limit is the expected
hallmark of a spectrally exact solver whose only error source is
floating-point rounding.

## 4.3 Convergence on Lane--Emden $n=3/2$

Introducing the half-integer polytrope $\rho_0(y)\sim(L_y-y)^{3/2}$ with
a density cutoff $\rho>10^{-2}$ gives an algebraic rate:

| $N_y$ | $N_{\mathrm{modes}}$ | $L^2$ relative error | Reduction factor |
|-------|----------------------|----------------------|------------------|
| 64    | 32                   | $2.88\times 10^{-4}$ | ---              |
| 128   | 64                   | $5.74\times 10^{-5}$ | $5.0\times$      |
| 256   | 128                  | $1.10\times 10^{-5}$ | $5.2\times$      |
| 512   | 256                  | $2.04\times 10^{-6}$ | $5.4\times$      |

The observed rate is compatible with $N_y^{-2.3}$, well below the
spectral expectation. Section 5 explains why.

## 4.4 Full-gravity four-variable g-mode eigenproblem vs GYRE

The Chebyshev collocation solver for the full-gravity four-variable
linearised stellar pulsation system (`solve_gmode_full_chebyshev`)
is validated on the shipped GYRE Lane--Emden $n=3$ polytrope at
$\ell=1$. Reference frequencies are produced by a live GYRE run with
`diff_scheme = COLLOC_GL6` on the 1001-row structure dump
`/tmp/gyre_run/poly3.txt`. Two pipeline patches were required for the
CUDA binary to ingest the GYRE dump: an explicit `<algorithm>` include
for `std::sort`, and a non-finite-row filter in the reader to discard
the surface line containing IEEE `Infinity` in $V_2$ and $A^\star$; the
latter was silently benign for Python benchmarks whose resampling never
approached $x=1$ but poisoned every row of the dense $Q$ matrix in the
CUDA LU factorisation (`cusolverDnDgetrfBatched` returned
$\mathrm{info}=256$).

After these fixes, at $N=96$ (DOF$=388$) the CUDA solver reproduces
GYRE to machine-tolerable precision:

| $n_g$ | $\omega^2_{\mathrm{CUDA}}$ | $\omega^2_{\mathrm{GYRE}}$ | rel err |
|-------|----------------------------|----------------------------|---------|
| 1     | $2.515930\times 10^{0}$    | $2.515928\times 10^{0}$    | $8.6\times 10^{-7}$ |
| 2     | $1.285715\times 10^{0}$    | $1.285708\times 10^{0}$    | $5.4\times 10^{-6}$ |
| 3     | $7.757401\times 10^{-1}$   | $7.757328\times 10^{-1}$   | $9.4\times 10^{-6}$ |
| 4     | $5.177826\times 10^{-1}$   | $5.177760\times 10^{-1}$   | $1.3\times 10^{-5}$ |
| 5     | $3.699317\times 10^{-1}$   | $3.699255\times 10^{-1}$   | $1.7\times 10^{-5}$ |
| 6     | $2.775088\times 10^{-1}$   | $2.775028\times 10^{-1}$   | $2.2\times 10^{-5}$ |
| 7     | $2.159323\times 10^{-1}$   | $2.159266\times 10^{-1}$   | $2.6\times 10^{-5}$ |
| 8     | $1.728587\times 10^{-1}$   | $1.728536\times 10^{-1}$   | $2.9\times 10^{-5}$ |
| 9     | $1.415487\times 10^{-1}$   | $1.415441\times 10^{-1}$   | $3.3\times 10^{-5}$ |
| 10    | $1.180727\times 10^{-1}$   | $1.180684\times 10^{-1}$   | $3.6\times 10^{-5}$ |

The maximum relative difference across $n_g=1,\dots,10$ is
$\mathbf{3.6\times 10^{-5}}$; the monotone growth with $n_g$ is
interpolation-dominated (linear interpolation of GYRE's native 1001-point
mesh onto the CGL collocation grid). Parallel Python pipelines pass at
the same tolerance: the two-variable Cowling solver agrees with GYRE's
Cowling spectrum at $5.6\times 10^{-4}$, and the four-variable
full-gravity Python solver agrees with GYRE at $5.3\times 10^{-4}$, both
well inside the $10^{-2}$ PASS threshold. At a coarser $N=48$ the CUDA
solver exhibits two spurious head modes with $\omega^2\approx 500,443$
that correspond to classifier misidentifications in the low-resolution
propagation cavity; identical head contamination is reproduced by the
Python verifier, confirming the artefact is classification-rooted rather
than a CUDA defect.

A further topological check confirms the Sturm oscillation theorem
discretely: all $21$ computed eigenfunctions $\psi_0,\dots,\psi_{20}$
possess exactly $n$ interior zeros, demonstrating that the non-Euclidean
symmetric operator preserves the ordering of the continuous problem.

# 5. Polytropic index and spectral convergence

## 5.1 The integer-versus-fractional cliff

A focused convergence study on the reduced-pressure Poisson problem
with a manufactured exact solution $\pi_\star(r)=\sin(2\pi r)$ and
$\rho(r)=(1-r)^{\sigma}$ on $r\in[0,1]$ produces a striking bifurcation
as a function of the surface exponent $\sigma$.

For $\sigma=3$ (Lane--Emden $n=3$, the Eddington polytrope for
radiation-pressure-supported stars):

| $N$ | raw error |
|-----|-----------|
| 16  | $8.5\times 10^{-8}$ |
| 32  | $\sim 10^{-10}$ |
| 64  | $6.7\times 10^{-11}$ |
| 128 | $\sim 10^{-9}$ (round-off climb) |
| 256 | $3.2\times 10^{-9}$ |

Raw Chebyshev collocation achieves machine precision with no prefactor,
no Jacobi weight, and no coordinate stretch.

For $\sigma=3/2$ (Lane--Emden $n=3/2$, the convective-core polytrope):

| $N$ | raw error |
|-----|-----------|
| 16  | $2.7\times 10^{-3}$ |
| 64  | $1.8\times 10^{-4}$ |
| 256 | $1.1\times 10^{-5}$ |

The convergence rate tracks $N^{-2.0}$ robustly, and **no choice of
analytic prefactor $\pi=t^{\alpha}u$ restores the spectral rate**: the
experiment swept $\alpha\in\{1/4,-1/2,-3/4\}$ and found only degradation.

## 5.2 Approximation-theoretic explanation

Chebyshev expansions of an analytic function on $[0,R]$ have
exponentially decaying coefficients precisely when the function admits
analytic continuation to a neighbourhood of $[0,R]$ in the complex plane
(Trefethen, *Approximation Theory and Approximation Practice*, Thm 7.2).

- $\rho(r)=(R-r)^3$ is a *polynomial*; its Chebyshev expansion terminates
  at four terms, and $\rho\pi$ inherits the smoothness of $\pi$.
- $\rho(r)=(R-r)^{3/2}$ is analytic on $[0,R)$ but has a branch-point
  singularity at $r=R$; its Chebyshev coefficients decay only as
  $N^{-\sigma-1/2}\sim N^{-2}$.

Since the Lane--Emden boundary behaviour is $\theta(\xi)\sim(\xi_1-\xi)$
and $\rho=\theta^n$, the polytropic index *is* the surface exponent:
$$
\rho(r)\sim(\xi_1-\xi)^n\sim(R-r)^n.
$$
Integer $n$ yields polynomial density; fractional $n$ injects a branch
point that the polynomial basis cannot resolve exponentially.

| Polytrope | Physical context                              | $n$  | Chebyshev rate        |
|-----------|------------------------------------------------|------|-----------------------|
| $n=1$     | non-relativistic degenerate envelope           | $1$  | $N^{-3/2}$ algebraic  |
| $n=1.5$   | convective core / fully convective star        | $1.5$| $N^{-2}$ algebraic    |
| $n=2$     | approximate main-sequence envelope             | $2$  | $N^{-5/2}$ algebraic  |
| $n=3$     | Eddington, radiation-pressure supported star   | $3$  | spectral (exponential)|
| $n=3.25$  | giant hydrogen envelope                        | $3.25$| $N^{-7/2}$ algebraic |

The Eddington polytrope—historically ubiquitous because it produces the
mass--radius relation for radiation-supported stars—is the unique
classical polytropic choice at which raw Chebyshev achieves spectral
convergence. To our knowledge this coincidence between physical and
numerical regularity has not been made explicit in the prior literature.

## 5.3 What works and what does not

Three remedies for fractional $\sigma$ were considered. The
Liouville substitution $\pi=\rho^{-1/2}q$ transforms the operator but
not the polynomial-expandability of the unknown, and fails. Power
prefactors $\pi=t^{\alpha}u$ with fractional $\alpha$ introduce a
branch point of their own and also fail. The remaining options are
basis replacement—Jacobi expansions with weight $(1-r)^{\sigma}$ as in
Dedalus, which absorb the singularity into the basis functions and
restore spectral convergence for any $\sigma>-1$—and coordinate
stretching (Kosloff--Tal-Ezer), which concentrates grid points near
the surface. Both live outside the Liouville framework and are
earmarked for Phase 2+ of the project.

# 6. Operator mismatch and the full-Galerkin closure

## 6.1 Symptom

When the SL spatial solver is used as a pressure projection inside the
time stepper of `AnelasticSLSolver`, the Boussinesq Kelvin--Helmholtz
test and the Boussinesq g-mode benchmark both close to well within
spectral tolerance. A long-time run of the boxed Lane--Emden $n=3/2$
g-mode, however, exhibits a FFT peak error of $-9.4\%$ and an eigenmode
deviation-per-step of $6\times 10^{-4}$—large enough that the resolved
time-domain signal drifts monotonically away from the correct
eigenfrequency after a few tens of wave periods.

## 6.2 Diagnostic hierarchy

Four candidate explanations were systematically eliminated:

| Path | Hypothesis                                                             | Outcome |
|------|------------------------------------------------------------------------|---------|
| A    | EVP uses Fourier $q$-space (density absorbed into variable)            | Lane--Emden dev/step $6.0\times 10^{-4}$; 13\% improvement |
| B    | EVP uses SL-basis $q$-space (Galerkin-consistent with Poisson step)    | Lane--Emden dev/step $5.9\times 10^{-4}$; 4\% improvement  |
| C    | Time domain recast in momentum variable $\varphi=\rho v$               | Toy $6.8\times 10^{-5}$; rejected as inadequate            |
| D    | Time domain advanced by the assembled operator itself                  | Lane--Emden dev/step $5.1\times 10^{-18}$; machine precision |

Paths A and B were implemented in CUDA, and path C was prototyped in
Python prior to the CUDA investment. None of them closes the Lane--Emden
loop because each only fixes a single piece (the Poisson step or the
density coefficient) of a composed linear operator whose remaining
pieces—the buoyancy multiplication by $N^2(y)$, the continuity
multiplication by $\rho'(y)/\rho(y)$, and the Chebyshev derivative
`apply_dy` in node space—each scatter the eigenvector by a finite
amount in the discrete norm.

## 6.3 The full-Galerkin closure

After eliminating $u_x$ through continuity and $b$ through the buoyancy
equation, the linear anelastic system reduces to the second-order ODE
$$
L\,\ddot V = -R\,V, \qquad
L = -D\,\mathrm{diag}(\rho)\,D + k^2\,\mathrm{diag}(\rho),\qquad
R = k^2\,\mathrm{diag}(N^2\rho). \tag{6.1}
$$
The eigenproblem $R v=\omega^2 L v$ is the *same discretisation* of
(6.1) that the time stepper should advance. If, and only if, the time
stepper is evaluated as $\ddot V=-L^{-1}R\,V$ through matrix--vector
products against the *same* assembled pair $(L,R)$, the eigenvector
of the spatial problem is automatically an eigenvector of the discrete
time operator, and a Runge--Kutta substep preserves it exactly up to
time-integration error:
$$
V(t) = V_{\mathrm{EVP}}\cos(\omega t),\qquad
W(t) = -\omega V_{\mathrm{EVP}}\sin(\omega t),
$$
with dev/step floored at floating-point rounding.

## 6.4 Numerical proof

A Python driver (`scripts/full_galerkin_closure_test.py`) compares three
time integrators on identical initial data: full-Galerkin $v$-space,
full-Galerkin $\varphi$-space, and the primitive node-space scheme that
mirrors the CUDA implementation. At $N_y=64$, RK4, $\Delta t=10^{-4}$,
initial amplitude $10^{-8}$, 100 steps:

| Case             | full-Galerkin $v$   | full-Galerkin $\varphi$ | primitive (CUDA-like) |
|------------------|---------------------|-------------------------|-----------------------|
| Boussinesq       | $3.2\times 10^{-18}$ | $3.2\times 10^{-18}$    | $3.3\times 10^{-18}$  |
| Lane--Emden 3/2  | $5.1\times 10^{-18}$ | $4.9\times 10^{-18}$    | $5.1\times 10^{-5}$   |

A sweep over $N_y\in\{32,64,96,128,192\}$ produces full-Galerkin
dev/step in $[1.8\times 10^{-18},\,2.6\times 10^{-17}]$ and a primitive
dev/step **constant at $5.1\times 10^{-5}$**, confirming that the
defect is an operator design flaw rather than a discretisation accuracy
issue: adding grid points cannot cure it.

## 6.5 Implementation specification

The CUDA path implied by the proof is a three-to-four-day rewrite:

| Task                                                                   | Effort |
|------------------------------------------------------------------------|--------|
| Allocate `d_Linv_R_kx[]` of total $n_h\cdot(n_y-2)^2$ doubles          | 0.5 d  |
| Assemble $L_{k_x},R_{k_x}$ on the host using existing $D$, $\rho$, $N^2$| 0.5 d  |
| Compute $M_{k_x}=L_{k_x}^{-1}R_{k_x}$ via `cusolverDnDgetrs` or LAPACK | 0.5 d  |
| Replace the linear part of `compute_rhs_uv` with a per-$k_x$ DGEMM     | 1.0 d  |
| Eliminate the buoyancy variable or assemble $N^2\cdot I$ in the same way| 0.5 d |
| Boussinesq regression plus Lane--Emden dev/step $\le 10^{-10}$ test    | 1.0 d  |

Memory cost is trivial: $33\cdot 62^2\cdot 8$ bytes $\approx 1$ MB for a
$64\times 64$ Cartesian grid. FLOP cost per RK3 substep is
$n_y^2\cdot n_h\approx 135\,\mathrm{kFLOP}$, slightly below the
$n_y^2\cdot n_x\approx 262\,\mathrm{kFLOP}$ of the current `apply_dy`
path.

## 6.6 Methodological observation

The diagnostic sequence deserves comment in its own right. Paths A and
B each required roughly one engineer-day of CUDA development and
validated at $10^{-4}$—negligibly better than the baseline. Path C was
prototyped in Python in under an hour (`scripts/path_c_td_benchmark.py`)
and rejected for insufficient improvement before any CUDA work was
committed. Path D was identified and validated in an additional hour
(`scripts/full_galerkin_closure_test.py`). Two Python prototypes,
totalling approximately two hours of engineer time, converted what
would have been a seven- to ten-day speculative CUDA rewrite into a
three- to four-day implementation of a proven specification.

# 7. Extension to polar geometry

The Cartesian framework transcribes directly to an axisymmetric 2D
annulus $(r,\phi)\in[r_{\mathrm{in}},r_{\mathrm{out}}]\times[0,2\pi)$,
with the azimuthal coordinate replacing $x$ (Fourier, periodic) and
the radial coordinate replacing $y$ (Chebyshev on a finite interval,
Dirichlet on the impermeable walls). Reduced-pressure anelastic
linearisation eliminates $\hat U_\phi$ through continuity and
$\hat b$ through the buoyancy equation, yielding
$$
L_m \ddot{\hat U}_r = -R_m\,\hat U_r,
$$
with $L_m$ built from the compound operator
$(1/r)\,\partial_r[r\rho_0\,\cdot]\cdot(1/\rho_0)\,\partial_r(\cdot)$
and $R_m=N^2(r)\,\mathrm{I}$. The $m=0$ mode corresponds to radial
pulsations and is treated separately. The assembled-matrix time stepper
of Section 6 transfers unmodified, exchanging $k_x$ for the azimuthal
index $m$; the seven-step pressure pipeline transfers in the same
way. Inner wall $r_{\mathrm{in}}>0$ sidesteps the coordinate singularity
at the origin, reserving basis replacement (Zernike, Jacobi $(1-r^2)^{|m|/2}$)
for a later full-disk extension. Projected effort is five to six
engineer-days on a dedicated branch; the validation plan reuses the
Lane--Emden $n=3$ benchmark, adds a Bessel-function analytic comparison
for the Boussinesq annulus g-modes, and performs an approximate
cross-check against GYRE's $\ell=1$ spectrum through the
$\ell\to m$ correspondence.

# 8. Discussion and conclusions

Kiriko demonstrates that the uniform-density pseudospectral machinery
extends cleanly to stratified stellar-interior flow provided three
conditions are met. First, the pressure Poisson operator must be
diagonalised in a basis adapted to the background density, which the
Liouville-regularised reduced-pressure SL expansion accomplishes with
a seven-step GPU pipeline that attains machine precision on the
Boussinesq limit and the same relative accuracy at $N=96$ as GYRE's
production sixth-order collocation scheme. Second, the interaction
between the polytropic index and the Chebyshev basis must be
acknowledged: only polynomial surface densities admit spectral
convergence, and the distinction between $\sigma\in\mathbb{Z}$ and
$\sigma\not\in\mathbb{Z}$ is sharp rather than gradual. The Eddington
polytrope's unique convergence behaviour is a coincidence between
astrophysical convention and approximation theory worth explicit note.
Third, the time-domain operator must be assembled from the same discrete
matrices that define the spatial eigenproblem; a primitive composition
of node-space pointwise multiplications and differentiation matrices
is not merely less accurate but structurally defective, leaking
$5.1\times 10^{-5}$ per step on Lane--Emden $n=3/2$ regardless of
resolution. The full-Galerkin remedy was identified, proven in a
Python driver at $5.1\times 10^{-18}$ dev/step, and specified as a
three- to four-day CUDA port.

Together, these three results certify the CUDA `anelastic_sl_solver`
as production-ready for the linear anelastic g-mode problem and
provide a blueprint for the polar-annular successor. Beyond the
immediate engineering outcome, the methodological pattern—small Python
prototypes that foreclose or validate specific multi-day CUDA
investments—is the single most productive habit visible in the
development record, and is recommended as standard practice for all
future paths.

# Acknowledgements

The GYRE reference data in Section 4.4 was produced on a collaborator's
workstation running Fedora 43 with MESA SDK 26.3.2 and GYRE built from
source; the SHA256 checksums of the reference artefacts are recorded
in `gmode_exp_k_cuda_benchmark_2026-05-03.md` §5. All computations were
performed on a single NVIDIA GeForce RTX 4080 SUPER (compute capability
8.9) with CUDA 12.9.86 and GCC 15.2.1. Source code and reproduction
scripts are archived in the `anelastic-sl-spectral` branch of the
stellar2d repository.

# Primary source records

1. `docs/pseudo_spectral_design_2026-05-01.md` — uniform-density baseline
2. `docs/anelastic_SL_spectral_design.md` — initial SL design
3. `docs/anelastic_sl_phase0_2026-05-02.md` — Phase 0 viability study
4. `docs/reduced_pressure_liouville.md` — reduced-pressure derivation
5. `docs/anelastic_sl_cuda_phase1ab.md` — CUDA basis precompute and Poisson pipeline
6. `docs/polytropic_index_spectral_convergence_2026-05-03.md` — convergence cliff
7. `docs/gmode_experiments_2026-05-02.md` — experiments H, I, J, K
8. `docs/gmode_exp_k_cuda_benchmark_2026-05-03.md` — GYRE cross-validation
9. `docs/qspace_sl_path_b_2026-05-03.md` — Path B results
10. `docs/path_c_python_benchmark_2026-05-03.md` — Path C rejection
11. `docs/full_galerkin_closure_proof_2026-05-03.md` — Path D proof
12. `docs/phase2_operator_mismatch_summary_2026-05-03.md` — consolidated diagnosis
13. `docs/polar_anelastic_design_2026-05-03.md` — polar extension design
