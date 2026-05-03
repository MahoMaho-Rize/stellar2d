---
title: |
  Spectral Solver Design — Liouville / Sturm--Liouville / Chebyshev Methods
  for the Stratified Pressure Equation
author: |
  Kiriko, Tsinghua University
date: 2026-05-03 (consolidated edition)
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
  \newcommand{\dd}{\mathrm{d}}
  \newcommand{\pp}{\partial}
  \newcommand{\rhob}{\rho_{0}}
  \newcommand{\phat}{\hat{p}}
  \newcommand{\pihat}{\hat{\pi}}
  \newcommand{\fhat}{\hat{f}}
---

# 0. About this document

This document consolidates four prior technical notes produced during
2026-05-01..03 into a single integrated reference, updated with the
post-Phase-0-ext+ conclusions.  The four predecessors are:

- `anelastic_SL_spectral_design.md`
- `liouville_SL_spectral_derivation.md`
- `reduced_pressure_liouville.md`
- `liouville_singularity_causality.md`

All four original files are retained in the repository as a record of
the reasoning trajectory; each one now carries a prominent header
pointing the reader here and to the formal report.

**Relationship to the formal report**.  The formal English report
`docs/spectral_stratified_poisson_report_2026-05-03.md` is the
citation-ready document (18 pages, authored by Kiriko / Tsinghua
University) that presents the Phase 0 ext+ conclusions and quantitative
evidence.  The present document is the **internal engineering design
record**: it retains the full mathematical derivations, the discussion
of implementation trade-offs, and the historical context, and is the
intended starting point for a developer joining the project.

**Three sibling documents**:

- `docs/spectral_stratified_poisson_report_2026-05-03.md` — formal
  English report
- `docs/spectral_experiments.md` — experimental record (Phase 0,
  reduced-pressure, g-mode, polytropic-index convergence)
- `docs/singular_basis_survey_2026-05-02.md` — GYRE / Dedalus survey

---


# Part I  Motivation and Problem Setup

## 1. Motivation

The existing `pseudo_spectral` solver handles uniform-density,
doubly-periodic, two-dimensional incompressible Navier--Stokes
($\rho = \mathrm{const}$, $\nabla\cdot u = 0$) in
vorticity--streamfunction form.  Stellar convection requires stratified
density $\rhob(y)$, which promotes the pressure equation to a
**variable-coefficient elliptic problem**.  Fourier modes are no longer
eigenfunctions, and the standard $\mathcal{O}(N\log N)$ pseudo-spectral
Poisson solve fails.

This document integrates two technical strategies:

1. **Liouville--SL basis (Parts II-III)**.  The substitution
   $\hat p = \sqrt{\rhob}\,q$ reduces the variable-coefficient Poisson
   operator to a Schrödinger form whose Sturm--Liouville
   eigenfunctions are common to all $k_x$.  The Poisson solve becomes a
   pipeline "cuFFT + GEMM + pointwise division + GEMM + cuFFT".
2. **Direct Chebyshev collocation (Part IV)**.  The Phase 0 ext+
   investigation demonstrates that, for the Eddington standard model
   ($n = 3$, $\sigma = 3$), raw Chebyshev collocation attains spectral
   convergence to machine precision without any Liouville-type
   substitution.  **This is the Phase 1 main route**.

Both routes share the same Chebyshev-grid infrastructure; they differ
only in how the operator is assembled.  The SL route is retained as a
Phase 2+ optional backend.


## 2. Problem formulation

### 2.1 Variable-density incompressible pressure equation

Dividing the variable-density incompressible momentum equation by
$\rhob$ and taking the divergence gives

$$\nabla \cdot \left(\frac{1}{\rhob(y)} \nabla p\right) = f(x)
\tag{2.1}$$

where $\rhob(y)$ is the background stratified density, depending only
on $y$.  For the anelastic system $\nabla\cdot(\rhob u) = 0$, the
same operator appears after the analogous projection; the mathematical
structure is identical.

### 2.2 Fourier reduction in the homogeneous direction

The $x$ direction remains homogeneous and periodic.  Expanding
$p(x,y) = \sum_{k_x} \phat(k_x,y)\,e^{ik_x x}$ reduces (2.1) for each
horizontal mode to a one-dimensional variable-coefficient ODE:

$$\frac{\dd}{\dd y}\!\left[\frac{1}{\rhob(y)}\,\frac{\dd \phat}{\dd y}\right]
  - \frac{k_x^{2}}{\rhob(y)}\,\phat = \fhat(k_x,y).
\tag{2.2}$$

**Issue**. The $1/\rhob(y)$ factor in front of the $k_x^{2}$ term
couples SL modes.  A direct eigenfunction expansion of
$\tfrac{\dd}{\dd y}[(1/\rhob)\tfrac{\dd}{\dd y}]$ alone is not enough:
the $k_x^{2}$ term introduces dense mode coupling.

The two resolutions of this issue — the Liouville substitution
(Parts II-III) and direct Chebyshev collocation (Part IV) — are what
this document develops.


\clearpage

# Part II  Liouville Normal-Form Reduction

## 3. The Liouville substitution

### 3.1 Change of dependent variable

Set
$$\boxed{\phat(y) = \sqrt{\rhob(y)}\;q(y),}\tag{3.1}$$
and compute the transformed operator.

\begin{proposition}[Liouville reduction]
Under the substitution (3.1), equation (2.2) takes the form (3.2) below, with the Liouville potential given by (3.3).
\end{proposition}

$$\frac{\dd}{\dd y}\!\left[\frac{1}{\rhob}\,\frac{\dd \phat}{\dd y}\right] = \frac{1}{\sqrt{\rhob}}\,\bigl[q'' + W(y)\,q\bigr], \tag{3.2}$$

$$W(y) \;=\; \frac{\rhob''}{2\rhob} - \frac{3(\rhob')^{2}}{4\rhob^{2}}. \tag{3.3}$$

\begin{proof}
Differentiating $\phat = \sqrt{\rhob}\,q$:
\begin{align*}
\phat' &= \frac{\rhob'}{2\sqrt{\rhob}}\,q + \sqrt{\rhob}\,q',\\
\frac{1}{\rhob}\,\phat' &= \frac{\rhob'}{2\,\rhob^{3/2}}\,q + \frac{q'}{\sqrt{\rhob}}.
\end{align*}
Differentiating the second line:
\begin{align*}
\frac{\dd}{\dd y}\!\left[\frac{1}{\rhob}\,\phat'\right] &= I_1 + I_2,\\
I_1 &= \left[\frac{\rhob''}{2\,\rhob^{3/2}} - \frac{3(\rhob')^{2}}{4\,\rhob^{5/2}}\right]q + \frac{\rhob'}{2\,\rhob^{3/2}}\,q',\\
I_2 &= -\frac{\rhob'}{2\,\rhob^{3/2}}\,q' + \frac{q''}{\sqrt{\rhob}}.
\end{align*}
The $q'$ contributions cancel, leaving
$$I_1 + I_2 = \frac{1}{\sqrt{\rhob}}\left[q'' + W(y)\,q\right].$$
\end{proof}

### 3.2 The reduced equation

Substituting (3.1) and (3.2) into (2.2) and multiplying by
$\sqrt{\rhob}$:

$$\boxed{q'' + W(y)\,q - k_x^{2}\,q = g(y), \qquad g \equiv \sqrt{\rhob}\,\fhat.}\tag{3.4}$$

Defining the **Liouville--Schrödinger operator**
$$\mathcal{T} \equiv \frac{\dd^{2}}{\dd y^{2}} + W(y),\tag{3.5}$$
(3.4) reads
$$\bigl[\mathcal{T} - k_x^{2}\bigr]\,q = g. \tag{3.6}$$

**The key structural property**: $\mathcal{T}$ does not depend on
$k_x$.  The horizontal wavenumber enters only as an additive shift of
eigenvalues.  A single precomputed spectral basis therefore serves all
$k_x$ simultaneously.


## 4. Spectral diagonalisation

### 4.1 The eigenvalue problem

Consider the Sturm--Liouville (equivalently, time-independent
Schrödinger) eigenvalue problem

$$\mathcal{T}\,\psi_n \;=\; \psi_n'' + W(y)\,\psi_n \;=\; -\mu_n\,\psi_n,
\qquad n = 0, 1, 2, \ldots \tag{4.1}$$

with appropriate boundary conditions (periodic, Dirichlet, or Neumann).
Standard SL theory guarantees:

1. Eigenvalues $\{\mu_n\}$ are real and non-decreasing:
   $\mu_0 \le \mu_1 \le \mu_2 \le \cdots$.
2. Eigenfunctions $\{\psi_n\}$ form a complete orthonormal basis of
   $L^{2}([0, L_y])$.
3. When $W \equiv 0$ (uniform density), $\psi_n$ reduces to the
   standard Fourier modes and $\mu_n = n^{2}\pi^{2}/L_y^{2}$.

\begin{theorem}[Universal diagonalisation]
The eigenfunctions $\{\psi_n\}$ simultaneously diagonalise the operator
$\mathcal{T} - k_x^{2}$ for every $k_x$, as expressed in (4.2).
\end{theorem}

$$\bigl[\mathcal{T} - k_x^{2}\bigr]\,\psi_n = -(\mu_n + k_x^{2})\,\psi_n. \tag{4.2}$$

### 4.2 Solution by eigenfunction expansion

Expanding $q = \sum_n a_n\,\psi_n$ and $g = \sum_n g_n\,\psi_n$ and
substituting into (3.6), orthonormality gives

$$\boxed{a_n(k_x) = -\frac{g_n(k_x)}{\mu_n + k_x^{2}}.} \tag{4.3}$$

**No mode coupling.**  The variable coefficient $1/\rhob$ is entirely
encoded in the eigendata $\{(\mu_n, \psi_n)\}$.  Each $(k_x, n)$ solve
is a scalar division, structurally identical to the uniform-density
Fourier--Poisson solve
$\phat(k) = -\fhat(k)/|k|^{2}$.


## 5. Complete SL-GEMM algorithm

### 5.1 Precomputation (once, or whenever $\rhob$ changes)

1. Compute $W(y)$ from $\rhob(y)$ via (3.3).
2. Solve the 1D Schrödinger eigenvalue problem (4.1) for
   $\{\mu_n, \psi_n\}$.
3. Form the transform matrix $\Psi_{in} = \psi_n(y_i)$,
   $\Psi \in \mathbb{R}^{N_y \times N_y}$.
4. Precompute the vector $\sqrt{\rhob(y_i)}$.

### 5.2 Per-timestep Poisson solve

\begin{center}
\begin{tabular}{clll}\toprule
Step & Operation & Formula & Cost\\\midrule
1 & FFT in $x$          & $f(x,y) \to \fhat(k_x,y)$                   & $\mathcal{O}(N_x N_y \log N_x)$\\
2 & Weight              & $g = \sqrt{\rhob}\cdot\fhat$                 & $\mathcal{O}(N_x N_y)$\\
3 & Forward SL transform & $G = \Psi^{\!\top} g$                       & $\mathcal{O}(N_y^{2} N_x)$\\
4 & Pointwise divide    & $Q_n(k_x) = -G_n/({\mu_n+k_x^{2}})$          & $\mathcal{O}(N_x N_y)$\\
5 & Inverse SL transform & $q = \Psi\,Q$                               & $\mathcal{O}(N_y^{2} N_x)$\\
6 & Weight              & $\phat = \sqrt{\rhob}\cdot q$                & $\mathcal{O}(N_x N_y)$\\
7 & IFFT in $x$         & $\phat(k_x,y) \to p(x,y)$                    & $\mathcal{O}(N_x N_y \log N_x)$\\\bottomrule
\end{tabular}
\end{center}

Dominant cost: the pair of dense matrix--matrix products in steps 3
and 5, giving overall complexity $\mathcal{O}(N_y^{2}N_x)$ per Poisson
solve.

### 5.3 Complexity comparison

\begin{center}
\begin{tabular}{lccl}\toprule
Method & $y$ direction & Total ($N\times N$) & GPU characteristics\\\midrule
Fourier ($\rhob=\mathrm{const}$) & division & $N^{2}\log N$ & cuFFT (optimal)\\
Chebyshev + tridiagonal & Thomas alg.\ & $N^{2}$ & serial per column\\
\textbf{SL-GEMM}   & \textbf{GEMM} & $N^{3}$ & \textbf{cuBLAS batched (tensor cores)}\\
Iterative (CG / MG) & SpMV $\times k$ & $N^{2}k$ & convergence-dependent\\\bottomrule
\end{tabular}
\end{center}

While SL-GEMM has higher asymptotic complexity, dense matrix
multiplication achieves near-peak GPU arithmetic throughput (high
arithmetic intensity, fully parallel, tensor-core compatible).  For
$N_y \lesssim 4096$, batched DGEMM wall-clock time is competitive with
latency-bound Thomas, and often faster.


\clearpage

# Part III  Reduced-Pressure Formulation

## 6. Motivation — a change of dependent variable weakens the singularity

The $1/\rhob$ coefficient in (2.1) is introduced by the specific
algebraic manoeuvre of dividing by $\rhob$ before taking the
divergence.  **Changing the dependent variable changes the
singularity**.

Define the **reduced pressure** $\pi \equiv p/\rhob$ (specific enthalpy
perturbation).  Substituted into the anelastic momentum equation with
the continuity constraint, this yields

$$\nabla\cdot(\rhob\,\nabla\pi) = \tilde f,\tag{6.1}$$

where $\tilde f$ absorbs the buoyancy, advection, and $\nabla\rhob$
terms.  **The structural change**: the elliptic operator in (6.1) has
coefficient $\rhob$ (not $1/\rhob$).  The density now appears in the
**numerator**.


## 7. Reduced-pressure Liouville analysis

### 7.1 Fourier reduction

$$\frac{\dd}{\dd y}\!\left[\rhob\,\frac{\dd\pihat}{\dd y}\right] - k_x^{2}\,\rhob\,\pihat = \tilde f(k_x,y).\tag{7.1}$$

### 7.2 Substitution $\pihat = \rhob^{-1/2}\,q$

Direct symbolic computation (verified with SymPy) yields
$$q'' + \widetilde W(y)\,q - k_x^{2}\,q = \tilde g,\tag{7.2}$$
with
$$\widetilde W(y) = \frac{\rhob''}{2\rhob} - \frac{(\rhob')^{2}}{4\rhob^{2}}.\tag{7.3}$$

**Difference from the original potential $W$**: the $(\rhob')^2/\rhob^2$
coefficient is $-1/4$ rather than $-3/4$.

### 7.3 Surface-singularity comparison

For Lane--Emden $n = 3/2$, $\rhob \propto (R-y)^{3/2}$:

\begin{center}
\begin{tabular}{lccc}\toprule
Formulation & Operator & Surface potential & $|C|$\\\midrule
Original & $\nabla\cdot(\rhob^{-1}\nabla p)$ & $W \approx -21/(16t^{2})$ & $1.3125$\\
\textbf{Reduced pressure} & $\nabla\cdot(\rhob\nabla\pi)$ & $\widetilde W \approx +3/(16t^{2})$ & $0.1875$\\\bottomrule
\end{tabular}
\end{center}

**Singularity strength reduced by a factor of 7**, and the sign of $C$
flips from negative (attractive) to positive (repulsive).  Frobenius
indicial analysis:

\begin{center}
\begin{tabular}{lccc}\toprule
Formulation & $C$ & Indicial exponents & Square-integrable?\\\midrule
Original & $-21/16$ & $7/4,\;-3/4$ & only $\alpha = 7/4$ branch\\
\textbf{Reduced pressure} & $+3/16$ & $3/4,\;1/4$ & \textbf{both branches}\\\bottomrule
\end{tabular}
\end{center}

This is a qualitative improvement: both linearly independent solutions
are square-integrable, and SL theory applies directly.

**Scope notice (2026-05-03 update)**.  The factor-of-7 reduction
documented above is meaningful only for $\sigma = 3/2$.  For the
project's principal scenario $\sigma = 3$ (Eddington, where $\rhob$ is
a polynomial), raw Chebyshev collocation reaches machine precision
directly, and this optimisation offers no operational value.  See
Part IV.


\clearpage

# Part IV  Direct Chebyshev Collocation (Phase 1 main route)

## 8. The integer-vs-fractional $\sigma$ convergence cliff

The central finding of Phase 0 ext+ (2026-05-03): direct Chebyshev
collocation of the reduced-pressure operator, with no substitution
whatsoever, exhibits a **sharp dichotomy** in convergence behaviour
depending on whether the surface exponent $\sigma$ is integer or
fractional.

### 8.1 Numerical evidence

Solve the manufactured-solution Poisson problem
$\rhob(r) = (1-r)^\sigma$, $\pi_\text{exact} = \sin(2\pi r)$,
$f = [\rhob\pi']' - k_x^{2}\rhob\pi$ (with $f$ constructed symbolically
in SymPy).

\begin{center}
\begin{tabular}{rll}\toprule
$N$ & $\sigma = 3$ error & $\sigma = 3/2$ error (raw)\\\midrule
16  & $8.5\times 10^{-8}$  & $2.7\times 10^{-3}$\\
32  & $\sim 10^{-10}$      & $5.1\times 10^{-4}$\\
64  & $6.7\times 10^{-11}$ & $1.8\times 10^{-4}$\\
128 & $\sim 10^{-9}$ (roundoff) & $5.2\times 10^{-5}$\\
256 & $3.2\times 10^{-9}$  & $1.1\times 10^{-5}$\\\bottomrule
\end{tabular}
\end{center}

- $\sigma = 3$: exponential convergence to machine precision
  ($\sim 10^{-10}$ at $N = 64$), followed by a slow roundoff rise — the
  ideal behaviour of Chebyshev with a polynomial coefficient.
- $\sigma = 3/2$: steady $N^{-2}$ algebraic convergence.  No $r^\alpha$
  prefactor recovers spectral accuracy.

### 8.2 Approximation-theoretic explanation (Trefethen Thm 7.2)

The decay of Chebyshev coefficients is governed by the analytic
regularity of the function being expanded.  For
$\rhob(r) = (1-r)^{\sigma}$:

- $\sigma \in \mathbb{Z}_{\ge 0}$: $\rhob$ is a **polynomial**, its
  Chebyshev expansion terminates in $\sigma + 1$ terms, and the whole
  coefficient is captured to machine precision.  The spectral
  convergence of the outer problem is untouched by the surface.
- $\sigma \notin \mathbb{Z}$: $(1-r)^{\sigma}$ has a **fractional
  algebraic branch point**.  Its Chebyshev coefficients decay only as
  $N^{-\sigma - 1/2}$.  $L_N$ approximates $L$ only to this rate, and
  the rate propagates to the solution — hence $N^{-2}$ for $\sigma = 3/2$.

### 8.3 Physical significance for stellar structure

The Lane--Emden equation
$\theta'' + (2/\xi)\theta' + \theta^n = 0$ with $\theta(\xi_1) = 0$ has
surface behaviour $\theta \sim (\xi_1 - \xi)$, so
$\rhob \propto (\xi_1 - \xi)^n \propto (R - r)^n$.  **The polytropic
index $n$ is literally the surface exponent $\sigma$**.

\begin{center}
\begin{tabular}{llcl}\toprule
Index & Physical context & Surface $\sigma$ & Rate\\\midrule
$n=1$ & outer white-dwarf layer & 1 & $N^{-3/2}$ algebraic\\
$n=3/2$ & convective core & $3/2$ & $N^{-2}$ algebraic\\
$n=2$ & main-sequence envelope & 2 & $N^{-5/2}$ algebraic\\
$n=3$ & \textbf{Eddington radiative model} & \textbf{3} & \textbf{spectral}\\
$n=7/2$ & giant hydrogen envelope & $7/2$ & $N^{-4}$ algebraic\\\bottomrule
\end{tabular}
\end{center}

**The Eddington $n = 3$ model is the unique standard physical polytrope
in the spectrally-convergent branch**.  This is a coincidence between
"the most-studied historical polytrope" and "the best-behaved Chebyshev
regime", and it justifies choosing $n = 3$ as the background for the
Phase 1 main route.


## 9. Chebyshev discretisation

### 9.1 Chebyshev--Gauss--Lobatto grid

$$\xi_j = \cos\frac{j\pi}{N}, \qquad j = 0,\ldots,N,\tag{9.1}$$
mapped affinely to $r \in [a,b]$.  Let $D$ denote the Trefethen spectral
differentiation matrix of size $(N+1)\times(N+1)$.

### 9.2 Reduced-pressure operator

With $R_\rho \equiv \operatorname{diag}(\rhob(r_j))$,

$$L_N = D\,R_\rho\,D - k_x^{2}\,R_\rho.\tag{9.2}$$

Dirichlet boundary conditions are imposed by strong collocation (unit
rows at $j = 0$ and $j = N$, right-hand side set to prescribed values).

### 9.3 A symmetry caveat

$D$ is **not symmetric** in the standard Euclidean inner product; it is
symmetric in the Chebyshev--Gauss--Lobatto quadrature-weighted inner
product.  Eigenvalue problems must therefore be solved with
`numpy.linalg.eig` (LAPACK `geev`), not `eigvalsh`.  This was
operationally important in Phase 0 ext+ Tests B and C: an initial
implementation with `eigvalsh` produced spurious eigenvalues.  See
`docs/spectral_experiments.md` §19.


\clearpage

# Part V  Unified Basis Claim — Critical Examination

## 10. "g-mode as a free by-product" revisited

### 10.1 The two operators have different singular points

- **Poisson operator**: singular at the **surface** $r = R$ (where
  $\rhob \to 0$).  Liouville potential
  $\widetilde W \sim \sigma(\sigma - 2)/[4(R - r)^{2}]$.  The
  SymPy-derived optimal prefactor is
  $$\pi = (R - r)^{\alpha_\star}\,u,\qquad
    \alpha_\star(\text{Poisson}) = 1 - \sigma/2.\tag{10.1}$$
- **g-mode operator**: singular at the **origin** $r = 0$ (the
  centrifugal term $\ell(\ell+1)/r^{2}$).  The corresponding prefactor
  is
  $$y_1 \sim r^{\beta_\star},\qquad
    \beta_\star(\text{g-mode}) = \ell + 1.\tag{10.2}$$

$\alpha_\star \ne \beta_\star$: **no single set of eigenfunctions
simultaneously diagonalises both operators**.

### 10.2 The correct refined statement

The strong form of the original Liouville programme — "a unified SL
basis simultaneously diagonalises the Poisson and g-mode operators" —
fails.  The weaker, operationally relevant form survives:

- The Chebyshev **mesh** is shared (same CGL nodes).
- The dense-linear-algebra infrastructure (cuBLAS GEMM, VRAM layout) is
  shared.
- The Poisson solve at each $k_x$ reuses a single LU factorisation
  (only the scalar shift $k_x^{2}$ changes).
- The g-mode calculation is a **separate EVP on the same mesh**, not a
  free by-product.

The engineering benefit is preserved; the "free mathematical
by-product" claim is demoted.  This is the central conclusion of
Phase 0 ext+ experiments E5 and E6.


\clearpage

# Part VI  Liouville Singularity — A Physical Interpretation

## 11. Reversed causality: quantum mechanics vs.\ astrophysics

The $1/t^{2}$ behaviour of the Liouville potential as $\rhob \to 0$ is
mathematically identical to structures in quantum mechanics (the
Coulomb potential, the centrifugal barrier).  The **causal
relationship**, however, is reversed.

### 11.1 In quantum mechanics: the potential is primary

$$V(r) \;\text{(physical input)} \;\longrightarrow\; \psi(r) \to 0 \;\text{(mathematical consequence)}$$

A Coulomb potential or centrifugal barrier is the **fundamental
physical quantity**.  The wavefunction $\psi$ (and hence the
probability density $|\psi|^{2}$) vanishes at certain boundaries
*because* the potential forces it to.  Techniques such as
Coulomb--Sturmian bases ($r^\ell e^{-\alpha r} L_n$), Laguerre DVR, and
R-matrix matching are designed around a singularity that **cannot be
removed**, because it is a property of the physical system.

### 11.2 In astrophysics: the density is primary

$$\rhob(y) \to 0 \;\text{(physical input)} \;\longrightarrow\; W(y) \to -\infty \;\text{(mathematical artifact)}$$

The density profile $\rhob(y)$ is determined by hydrostatic equilibrium
(Lane--Emden, MESA).  The stellar surface is simply where the gas runs
out; there is no "infinite potential barrier" in the physics.  The
original operator $\nabla\cdot(\rhob^{-1}\nabla p)$ is a **degenerate
elliptic** operator at the surface, losing ellipticity smoothly as
$\rhob \to 0$, without any true singularity.  The divergence of $W$ is
**manufactured by the $\sqrt{\rhob}$ substitution** — it is a
coordinate singularity, not a physical one.

### 11.3 Consequences for basis design

1. **Quantum-mechanics methods are applicable but over-engineered.**
   Coulomb--Sturmian bases were designed for genuine physical
   singularities; applying them to an artificial coordinate singularity
   is using a sledgehammer for a thumbtack.
2. **Path A's limitations trace to the same observation.**  A prefactor
   substitution $r^\alpha$ to absorb $W$'s singularity is the
   astrophysical analogue of the Coulomb--Sturmian $r^\ell
   e^{-\alpha r}$ factor.  There, it is the correct method; here, it
   attempts to "unwind" a singularity that ought not to exist in the
   first place.
3. **The natural approach is to return to the original operator.**
   Reduced pressure (Part III) weakens the singularity from
   non-integrable attractive to two-branch-integrable repulsive.  Raw
   Chebyshev (Part IV) discretises the coefficient $\rhob$ directly,
   and for polynomial $\sigma$ encounters no singularity at all.

**Summary.**  Compared with the technical machinery the
quantum-mechanics community has developed for genuine Coulomb
singularities, the stratified-Poisson problem in astrophysics should be
handled with **minimal machinery**.  For $\sigma \in \mathbb{Z}$, no
special treatment is required; for $\sigma \notin \mathbb{Z}$, a
Jacobi-weighted basis (Dedalus) is the only additional machinery
needed.


\clearpage

# Part VII  Phase Roadmap

## 12. Current status and forward plan

### 12.1 Completed (Phase 0 ext+, 2026-05-02..03)

- Verified the convergence dichotomy between Lane--Emden $n = 3/2$ and
  $n = 3$.
- Benchmarked Chebyshev at $N = 48$ (192 DOF) against the GYRE
  full-gravity 4-variable system: max relative error
  $1.5\times 10^{-6}$, a $21\times$ reduction in DOF and $350\times$
  smaller max error compared with staggered FD at $N_r = 1024$ (4096
  DOF).
- Three analytical-ceiling tests (manufactured Poisson, quantum
  harmonic oscillator, Dirichlet Laplacian) all reach double-precision
  machine accuracy ($10^{-13}$ to $10^{-15}$).
- Verified via barycentric Lagrange that "N coefficients $\ne$ N pixels".
- Chose Path A (raw Chebyshev) as sufficient for $n = 3$.

Full evidence: `docs/spectral_experiments.md` and
`docs/spectral_stratified_poisson_report_2026-05-03.md`.

### 12.2 Phase 1: 2D Fourier--Chebyshev Boussinesq

- **$x$ direction**: Fourier (periodic), reusing the `pseudo_spectral`
  cuFFT infrastructure.
- **$y$ direction**: Chebyshev collocation, GPU-side $D^{(2)}$ by
  cuBLAS dense GEMM.
- **Physics**: 2D Boussinesq with buoyancy; Poisson via Chebyshev +
  dense solve.
- **Benchmark**: Rayleigh--Bénard Nu--Ra scaling (Ahlers et al.).
- **Background**: Gaussian transition, moving to Eddington $n = 3$
  polytrope.

A design document `docs/phase1_2d_spectral_design.md` will be written
when Phase 1 starts in earnest.

### 12.3 Phase 2: Anelastic upgrade

- Upgrade from Boussinesq to anelastic: $\nabla\cdot(\rhob u) = 0$.
- Chebyshev handles the variable-density Poisson directly (raw or
  reduced-pressure form).
- **SL-GEMM backend as an optional optimisation**: activated only if
  dense Poisson solves become a GPU-wall-clock bottleneck.

### 12.4 Phase 3: Live eigenmode projection (differentiator)

- Independent EVP on the same mesh yields g-mode / p-mode eigenpairs.
- Instantaneous flow fields are projected onto the mode basis as a
  runtime diagnostic.
- **This is the genuine novelty of the project**: convection--pulsation
  coupling in a 2D nonlinear DNS with live modal projection.

### 12.5 Phase 4+: Spherical-shell extension

See `docs/sph_spectral_roadmap.md` (long-term).

### 12.6 Publication strategy

- **Methods paper** (JCP): "Sturm--Liouville spectral methods for
  stratified astrophysical flows — convergence regimes and GPU
  implementation".
- **Applications paper** (A&C / ApJS): "GPU anelastic pseudo-spectral
  DNS with live eigenmode projection for convection--pulsation
  coupling diagnostics".

The applications paper carries the real novelty; the methods paper
functions as supporting citation.


\clearpage

# Appendix A  GPU implementation considerations

## A.1 Batched GEMM formulation

Steps 3 and 5 of §5.2 are a single matrix--matrix multiplication:
$$G = \Psi^{\!\top}g, \qquad q = \PsiQ,$$
with $g, G, Q, q \in \mathbb{R}^{N_y \times N_x}$.
On modern NVIDIA Ampere/Hopper GPUs, FP64 GEMM exceeds 1 TFLOP/s, and
the $\mathcal{O}(N)$ arithmetic intensity (flops per byte read) ensures
compute-bound execution for $N_y \ge 256$.

## A.2 Integration with the time integrator

The SL-GEMM Poisson solver slots into an existing pseudo-spectral time
integrator (e.g.\ IFRK3 + cuFFT) by replacing the spectral-space
division $\phat = -\fhat/|k|^{2}$ with the sequence:

$$\text{cuFFT (R2C in }x\text{)} \to \text{cuBLAS DGEMM} \to \text{pointwise divide} \to \text{cuBLAS DGEMM} \to \text{cuFFT (C2R in }x\text{)}.$$

All existing infrastructure — IFRK3 time integration, skew-symmetric
convection, 2/3 dealiasing, VRAM frame buffering — is reused without
modification.

## A.3 Memory footprint

The transform matrix $\Psi$ requires $N_y^{2}$ doubles $= 32$ MiB at
$N_y = 2048$, which is modest compared with the flow-field arrays
($\sim 100$ MiB each at $2048^{2}$) and the VRAM frame buffer
($\sim 10$ GiB).


\clearpage

# Appendix B  Related methods and literature positioning

## B.1 The three-community gap

| Community | Representative codes | Radial discretisation | Why SL-GEMM is absent |
|-----------|---------------------|----------------------|----------------------|
| Stellar pseudo-spectral | ASH, Rayleigh | spherical harmonics (angular) + Chebyshev/FD (radial) + banded | GEMM not competitive on CPU |
| GPU pseudo-spectral | hit3d, spectralDNS | Fourier in all directions | limited to uniform density |
| Variable-density GPU CFD | engineering LES codes | finite volume + multigrid | spectral methods rarely used |

No existing code combines Liouville reduction, SL spectral basis, GPU
GEMM, and anelastic stellar convection.

## B.2 SL theory background

- **Liouville normal form**: Sturm 1836, Liouville 1837; see Zettl
  (2005) for a modern treatment.
- **SL eigenfunction expansions for elliptic PDEs**: discussed by Boyd
  (2001) as theoretically elegant but rarely implemented due to the
  absence of a fast SL transform.
- **Chebyshev convergence theory**: Trefethen (2013) Ch.\ 7
  (asymptotic coefficient decay for integer vs fractional branch
  points).
- **Berrut--Trefethen barycentric Lagrange interpolation** (2004):
  enables a stable, machine-precision evaluation of the spectral
  representation on any finer grid.

## B.3 Relation to GYRE / Dedalus

See `docs/singular_basis_survey_2026-05-02.md` for a detailed survey.
The project's novelty positioning, in its final form, is "GPU 2D
anelastic DNS with live eigenmode projection" — it does not compete
with GYRE (the mature 1D stellar-pulsation code) or Dedalus (the
general-purpose spectral-PDE framework) in the 1D pulsation-benchmark
arena.


\clearpage

# Appendix C  Historical record — mapping to the four predecessor documents

| Section in this document | Predecessor | Original section |
|-------------------------|-------------|------------------|
| Part I §1-2     | anelastic_SL_spectral_design.md       | §1-2 |
| Part II §3-4    | liouville_SL_spectral_derivation.md   | §3-4 |
| Part II §5      | liouville_SL_spectral_derivation.md §5 + anelastic_SL_spectral_design.md §5 | |
| Part III §6-7   | reduced_pressure_liouville.md         | entire |
| Part IV §8-9    | polytropic_index_spectral_convergence_2026-05-03.md + formal report §3 | |
| Part V §10      | new (Phase 0 ext+ E5/E6 conclusions)  | |
| Part VI §11     | liouville_singularity_causality.md    | entire |
| Part VII §12    | new (post Phase 0 ext+ roadmap)       | |
| Appendix A      | anelastic_SL_spectral_design.md §8 + liouville_SL_spectral_derivation.md §8 | |
| Appendix B      | anelastic_SL_spectral_design.md §7 + liouville_SL_spectral_derivation.md §9 | |

All four predecessor files are retained in the repository as a record
of the reasoning trajectory.  Each carries a header update block
pointing the reader to this document and to the formal report; new
work should proceed from this document and the formal report only.
