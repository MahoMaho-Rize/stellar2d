# 1. Introduction

## 1.1 Motivation and the central claim

Nodal pseudo-spectral discretisations of variable-coefficient
anelastic flow in which the elliptic operator is applied in
*matrix-free* factored form --- that is, as a composition of
differentiation, pointwise multiplication, and pointwise division
kernels, without constructing the assembled matrix --- exhibit a
structural operator-consistency gap at the discrete level.  When the
resulting scheme is reduced to a scalar second-order oscillator by
eliminating the momentum/pressure coupling, the gap manifests as
an $\mathcal O(10^{-4})$ per-step deviation of an eigenmode from its
own EVP solution --- a deviation that is *independent of grid
resolution*, *independent of time-step refinement*, and *independent
of the boundary-condition formulation* --- and, under further factorisation
(§5.7.2), as an exponential instability that destroys the solution
within a single oscillation period.  The failure is not an artefact
of insufficient refinement: the defective step is already present at
$N_y = 32$ with the eigenvalue problem solved to machine precision.

The mechanism is not a truncation error, not an aliasing effect,
and not a singularity. It is a structural mismatch between a global
elliptic operator and a pointwise surrogate: the matrix-free scheme
replaces the inverse $\mathsf L^{-1}$ of the discrete elliptic
operator $\mathsf L = -\partial_y(\rho_0\partial_y\cdot) +
k_x^2\rho_0$ by the pointwise scaling $\mathrm{diag}(1/\rho_0)$.  Both
matrices are well-defined and invertible for $\rho_0 > 0$; they are
simply different operators. The consequent per-step consistency
error persists under any refinement of either the grid or the
physics amplitude.

The gap is structural rather than numerical: in the continuous limit,
the matrix-free discretisation converges to a local multiplication
operator, $\mathsf M_{\mathrm{mf}} \to b(y)/a(y)$, while the
assembled discretisation converges to the intended global elliptic
inverse, $\mathsf M_{\mathrm{asm}} \to \mathcal L^{-1}\mathcal R$.
The two agree only when $b(y)/a(y)$ is constant, which for the
anelastic problem means $N^2 \equiv \mathrm{const}$ (Boussinesq), and
which for the general variable-coefficient generalised eigenproblem
means constant coefficient ratio.  The stellar-anelastic failure is
the specific case of this general phenomenon that we document and
resolve quantitatively in this paper (Sections 5 and 6), with a
non-stellar confirmation on three unrelated coefficient pairs in
Section 5.6.  The issue is orthogonal to the choice of spectral basis
(Galerkin frameworks such as Dedalus's $\tau$-method assemble
$\mathsf L^{-1}$ implicitly through weak-form residual projection and
are unaffected); the axis that matters is whether the composite
elliptic operator is applied matrix-free or through its assembled
inverse.  Within the matrix-free family, different factorisation
choices produce qualitatively different time-domain behaviour: a
local pointwise reduction is bounded but inconsistent, a fully
factorised reduction is spectrally unstable, and a pressure-retaining
formulation is stable precisely because its pressure solve
reintroduces a global elliptic inversion analogous to $\mathsf
L^{-1}$ (Section 5.7).  The assembled construction of Section 6 makes
this global inversion explicit.

The unique minimal-cost remedy is to assemble $\mathsf L^{-1}\mathsf R$
explicitly as a setup-time dense matrix and apply it as a
per-wavenumber DGEMV per RK4 substage. This restores the discrete
closure to machine precision and requires no change to the spectral
transform pipeline, the pressure projection, or the time-stepping
scheme --- a single inserted matrix-vector multiplication per
wavenumber.  The CPU reference implementation achieves a per-step
deviation of $5\times 10^{-18}$, while a GPU-accelerated port
maintains accuracy at the $3\times 10^{-15}$ level; the spatial
machinery is externally validated against the GYRE stellar-pulsation
code [1] to $9.1\times 10^{-9}$ relative error on a Lane--Emden
$n = 3$ polytrope.

## 1.2 The two observations

This paper contributes two formal statements.

**Proposition 1** (Section 5): for any $\rho_0 \in C^2$ with
$\rho_0' \not\equiv 0$, the matrix-free time-stepping operator
$\mathsf M_{\mathrm{mf}} = \mathrm{diag}(1/\rho_0)\cdot \mathsf R_{\mathrm{applied}}$
exhibits a resolution-independent consistency gap relative to the
assembled operator $\mathsf M_{\mathrm{asm}} = \mathsf L^{-1}\mathsf R$,
with magnitude scaling linearly in $\|\rho_0'\|_\infty$ in the
small-perturbation limit and scale-invariant in the relative
norm. The gap does not vanish under grid refinement.

**Proposition 2** (Section 6): under RK4 time stepping of the
first-order system $\dot U = \mathsf A U$ with
$\mathsf A = \begin{pmatrix} 0 & I \\ -\mathsf M & 0 \end{pmatrix}$
built from the assembled $\mathsf M = \mathsf L^{-1}\mathsf R$, any
EVP eigenvector $V_n$ is preserved to within
$\mathcal O((\omega_n\Delta t)^5 k) + \mathcal O(\epsilon_{\mathrm{mach}}\kappa(Q) k)$
per step count $k$, independent of $N_y$, $\rho_0$, and $N^2$.

These statements are formally independent --- Proposition 2 describes
what a correct discrete algebra gives; Proposition 1 describes what
the incorrect pointwise substitution costs --- and are jointly
necessary for machine-precision closure on realistic stellar
backgrounds.

## 1.3 Contributions

1. **Proposition 1** (Section 5): a formal statement of the
   operator-consistency floor, including the resolution-independent
   lower bound, the $\varepsilon^{1.00}$ scaling law in the
   small-perturbation limit, and the structural (not truncation)
   character of the inconsistency.

2. **Proposition 2** (Section 6): a formal statement of discrete
   eigenmode preservation under RK4 time stepping on the assembled
   operator $\mathsf M = \mathsf L^{-1}\mathsf R$, with explicit
   error bound (6.2).

3. **Quantitative three-method comparison** (Section 8.3): the
   assembled construction achieves $10^{-18}$ per-step eigenmode
   deviation at $N_y = 64$, versus $10^{-14}$ for a reference
   $\tau$-method Galerkin implementation and $10^{-4}$ for the
   matrix-free scheme; working memory is approximately one-third of
   the $\tau$-method Chebyshev-coefficient formulation (a single
   $N_{\mathrm{int}}^2$ matrix vs.\ three); per-step runtime is
   within a factor of two of both alternatives.  The scheme
   requires only a single setup-time matrix insertion into an
   existing matrix-free pipeline.

4. **GYRE external validation** (Section 4): $9.1\times 10^{-9}$
   relative error on the first ten $\ell = 1$ g-modes of a
   Lane--Emden $n = 3$ polytrope at $N_r = 96$, a floor set by
   dense-eigensolver floating-point round-off rather than spectral
   truncation.

## 1.4 Roadmap

Section 2 fixes notation and writes down the anelastic equations
and the Lane--Emden background. Section 3 develops the Liouville-SL
spatial discretisation. Section 4 assembles the four-variable
GYRE-compatible g-mode operator and reports the external benchmark.
Section 5 presents the empirical time-domain failure (5.1), the
analysis of alternative error sources (5.2), the formal statement of
Proposition 1 (5.3), the scaling-law verification (5.4), and the
relation to Galerkin-tau methods (5.5). Section 6 presents the
assembled-operator resolution with Proposition 2 and its GPU
realisation. Section 7 extends the framework to nonlinear time
stepping via operator splitting. Section 8 discusses scope, limits,
and the underlying principle that spectral accuracy does not imply
operator consistency. Section 9 concludes. Appendix A collects
implementation details, convergence proofs, and the density-cutoff
sensitivity analysis; Appendix B is a code and data availability
statement.
