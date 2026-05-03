# 9. Conclusions

We identified, quantified, and resolved a structural
operator-consistency failure in the matrix-free pseudo-spectral
time-stepping of variable-coefficient anelastic flow.  Initialising
the solver with an exact g-mode eigenvector of its own discrete
eigenproblem produces an $\mathcal O(10^{-4})$ per-step deviation
that persists under eight-fold grid refinement and sixteen-fold
$\Delta t$ refinement; integrated forward in time this per-step
deviation compounds into an exponential instability that destroys
the solution within a single oscillation period.  The mechanism
(Proposition 1) is neither a truncation error nor an aliasing effect,
but a structural mismatch between a global elliptic operator
$\mathsf L^{-1}$ --- dense, nonlocal, with row decay algebraic in
$|i-j|$ --- and the pointwise surrogate
$\mathrm{diag}(1/\rho_0)$.  For the one-parameter family
$\rho_0 = 1 + \varepsilon f(y)$ the absolute consistency gap scales
as $\varepsilon^{1.00 \pm 0.001}$, while the relative gap is
scale-invariant at $\mathcal O(1)$: no refinement of either grid or
physics amplitude reduces the defect below a background-dependent
floor $c(\rho_0) \sim \|\rho_0'\|_\infty$.

The resolution (Proposition 2) is to assemble $\mathsf M = \mathsf
L^{-1}\mathsf R$ explicitly once at setup and to apply it as a
per-wavenumber dense matrix-vector product at every RK4 substage.
The resulting first-order system admits exact per-eigenmode invariant
subspaces under RK4, with $\mathcal O((\omega\Delta t)^5)$ phase
error and $\mathcal O(\epsilon_{\mathrm{mach}}\kappa(Q))$ round-off
accumulation per step, independent of $N_y$, $\rho_0$, and $N^2$.
The CPU reference implementation achieves a per-step deviation of
$5\times 10^{-18}$; the GPU-accelerated port, $3\times 10^{-15}$.
External validation against GYRE gives $9.1\times 10^{-9}$ relative
error on Lane--Emden $n = 3$ g-modes at $N_r = 96$.

The broader takeaway is that spectral accuracy of the basis does not
imply operator consistency of the time-stepping loop under variable
coefficients.  Pseudo-spectral codes written with matrix-free
operator application --- the standard GPU architecture for reasons
of memory layout and kernel simplicity --- inherit the
$\mathrm{diag}(1/\rho_0)$ surrogate by construction.  Galerkin
frameworks ($\tau$-method, spectral element, finite element) avoid
the failure by building the inverse implicitly through weak-form
assembly, at the architectural cost of a fundamentally different
pipeline.  The present work shows that the matrix-free architecture
need not be abandoned: the closure is recovered by a single
setup-time matrix assembly, requiring no change to the spectral
transform, the pressure projection, or the time-stepping scheme.
