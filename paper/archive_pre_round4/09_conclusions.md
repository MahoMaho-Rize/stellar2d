# 9. Conclusions

Spectral methods for variable-density anelastic flow on realistic
stellar backgrounds close to machine precision if, and only if, two
independent ideas are combined.  Each alone is insufficient; their
combination is the subject of this paper.

**First**, the Liouville substitution $\pi = \rho_0^{-1/2} q$
reduces the variable-coefficient Poisson operator of the anelastic
pressure projection to a Schrödinger form whose Sturm--Liouville
eigenfunctions provide a single precomputed basis that
simultaneously diagonalises the elliptic problem at every
horizontal wavenumber.  The same machinery organises the
generalised eigenvalue problem for g-mode frequencies.  Section 4
reports a benchmark of this construction against the GYRE stellar-
pulsation code: on the shipped Lane--Emden $n = 3$ polytrope our
$N_r = 96$ CGL Chebyshev collocation recovers the first ten
$\ell = 1$ g-mode frequencies to a maximum relative error of
$3.6 \times 10^{-5}$.

**Second**, the spatial SL consistency is \emph{not} sufficient
for time-domain closure.  The primitive-node time-stepping loop of
a standard pseudo-spectral code --- apply the spectral derivative,
multiply pointwise by $\rho_0$, multiply pointwise by $N^2$,
divergence-project --- is a correct discretisation of the
\emph{continuous} evolution operator but is \emph{not} equal, at
the matrix level, to the assembled $\mathsf L^{-1}\mathsf R$ that
defines the same operator's eigenvalue problem.  The discrepancy is
a variable-coefficient Leibniz defect at the discrete level.  On a
Lane--Emden $n = 3/2$ polytrope it produces a $6.9 \times 10^{-4}$
per-step deviation of the discrete eigenvector --- four orders of
magnitude worse than the Boussinesq baseline on the same code, and
entirely insensitive to refining the spatial discretisation.

**Third**, the resolution is constructive.  Build $\mathsf L^{-1}\mathsf
R$ explicitly as a dense per-wavenumber matrix at setup time, store
it, and use it for the time update via per-wavenumber matrix--vector
product.  The Python prototype reaches $5 \times 10^{-18}$/step; the
CUDA implementation reaches $3 \times 10^{-15}$/step.  On a 300-period
g-mode run, the long-time Fourier-peak frequency error drops from
$-9.4\%$ under primitive-node stepping to $-1.5 \times 10^{-4}$ under
the assembled-matrix stepping, and the eigenvector amplitude remains
stable at $4 \times 10^{-9}$ rather than collapsing to
$\mathcal O(1)$.

**Fourth**, the nonlinear extension is a routine Strang operator
split between the assembled-operator linear block and an RK4
advection block.  A Python three-scheme comparison that includes a
semi-implicit IMEX scheme (which fails catastrophically at
physically relevant amplitudes) and an exponential-propagator scheme
(numerically indistinguishable from the Strang-split scheme at
finite amplitude) establishes the Strang split as the GPU
implementation recommendation.

The combination of the four results above provides, on a single
consumer-grade GPU, a two-dimensional anelastic spectral code that
is variable-density, externally benchmarked, and machine-precision
exact on its own linear spectrum.  Source code, reproducers, and all
benchmark data are at \texttt{https://github.com/MahoMaho-Rize/stellar2d}.

Two pieces of work remain.  The GPU realisation of the Strang-split
nonlinear scheme is described here as a prototype in Python only;
its GPU implementation is a mechanical reuse of the pre-existing
pseudo-spectral advection kernels inside the Strang wrapper of
Section 6.  The nonlinear benchmark campaign --- long-time runs on
real MESA stellar profiles, testing mode-coupling and saturation
--- will be appended to the public repository as these runs
complete.

Beyond the present framework, three extensions suggest themselves.
Full three-dimensional spherical-harmonic DNS of a real star, using
the same per-harmonic assembled matrix construction, appears
tractable at $\ell_{\max} \approx 64$ on a single GPU.
Non-radiative boundary conditions representing realistic
atmospheric damping would replace our surface cutoff with a
physics-motivated outer boundary.  And a non-adiabatic extension,
matching GYRE's ``nad'' mode, is a one-line addition to the buoyancy
equation that would expand the application from pure pulsation to
driven, damped, stochastic oscillations.

## Acknowledgments

We thank the developers of the GYRE stellar-pulsation code for
maintaining a reliable external reference over more than a decade,
and the NVIDIA cuFFT and cuSOLVER development teams for API
stability across major CUDA versions.  Computational resources were
provided by the Tsinghua Department of Astronomy.

This work was undertaken in dialogue with Claude (Anthropic), whose
role in the workflow was limited to code review, diagnostic logging,
and prose editing; all algorithmic decisions, numerical experiments,
and final text are the author's.
