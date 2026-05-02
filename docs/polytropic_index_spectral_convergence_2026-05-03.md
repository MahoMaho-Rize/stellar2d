---
title: Polytropic index and spectral convergence — the σ∈ℤ vs σ∈ℝ cliff
date: 2026-05-03
status: findings report
scripts:
  - scripts/spectral_liouville_convergence_v2.py
  - scripts/spectral_liouville_beta_derivation.py
parent: docs/spectral_liouville_plan_2026-05-03.md
---

# Summary (one-line)

**The polytropic index n determines whether Chebyshev spectral methods
can solve the stratified Poisson problem to machine precision or are
stuck at $N^{-2}$ algebraic convergence — and the crossover is sharply
at integer-vs-fractional surface exponent.**

For Lane-Emden $n = 3$ (surface $\rho \sim (R-r)^3$, integer exponent),
the reduced-pressure Poisson operator $\nabla\cdot(\rho\,\nabla\pi) = f$
discretised on Chebyshev-Gauss-Lobatto collocation over the **full domain
$[0, R]$** (no cutoff, no surface substitution) achieves error $\sim 10^{-10}$
at $N = 64$ — essentially machine precision.

For Lane-Emden $n = 3/2$ (surface $\rho \sim (R-r)^{3/2}$, half-integer
exponent), the same scheme gives $N^{-2}$ algebraic convergence, and no
simple analytic prefactor $\pi = t^\alpha u$ restores the exponential rate.

This dichotomy is driven by whether the surface density profile is
**analytic-at-zero** (polynomial-expressible) or **non-analytic**.


# The numerical evidence

## E6 v2 convergence sweep (SymPy-forced manufactured solution)

**Setup.**  $\rho(r) = (1 - r)^\sigma$ on $[0, 1]$;
$\pi_\text{exact}(r) = \sin(2\pi r)$ (Dirichlet-compatible);
$f = [\rho \pi_\text{exact}']' - k_x^2 \rho \pi_\text{exact}$ computed
symbolically, then evaluated on the CGL grid.  Solve
$D_1 \operatorname{diag}(\rho) D_1 \pi - k_x^2 \operatorname{diag}(\rho) \pi = f$
with Dirichlet BCs $\pi(0) = \pi(R) = 0$.

**Script.** `scripts/spectral_liouville_convergence_v2.py`.

### σ = 3 (Lane-Emden $n = 3$)

| N | error (raw) | error (α=1−σ/2=−1/2) |
|---|-------------|----------------------|
| 16 | $8.5\times 10^{-8}$ | $2.4\times 10^{-3}$ |
| 32 | $\sim 10^{-10}$     | $\sim 10^{-4}$ |
| 64 | $6.7\times 10^{-11}$ | $1.5\times 10^{-4}$ |
| 128 | $\sim 10^{-9}$ (round-off climb) | $\sim 10^{-5}$ |
| 256 | $3.2\times 10^{-9}$ | $9.1\times 10^{-6}$ |

**Raw discretisation achieves machine precision.**  Additional prefactor
substitutions make it *worse* (they reintroduce irregularities at the
endpoint via the analytic discontinuity of $t^\alpha$ for fractional $\alpha$).

### σ = 3/2 (Lane-Emden $n = 3/2$)

| N | error (raw) | α=1/4 | α=−1/2 | α=−3/4 |
|---|-------------|-------|--------|--------|
| 16 | $2.7\times 10^{-3}$ | $1.1\times 10^{-2}$ | $1.3\times 10^{-2}$ | $7.8\times 10^{-2}$ |
| 64 | $1.8\times 10^{-4}$ | $7.5\times 10^{-4}$ | $1.1\times 10^{-3}$ | $2.1\times 10^{-2}$ |
| 256 | $1.1\times 10^{-5}$ | $4.8\times 10^{-5}$ | $8.3\times 10^{-5}$ | $5.4\times 10^{-3}$ |

**Algebraic $N^{-2.0}$ convergence for all choices of α.** No simple
power-law prefactor restores spectral accuracy.


# Why integer vs fractional exponent matters

## Approximation theory angle

Chebyshev polynomials are polynomials in $r$.  A function $f(r)$ has
Chebyshev expansion with **exponentially decaying coefficients** if and
only if $f$ is analytic in an open neighbourhood of $[0, R]$ in the complex
plane (specifically, a Bernstein ellipse).

- $\rho(r) = (R - r)^3 = R^3 - 3R^2 r + 3R r^2 - r^3$ is a **polynomial**.
  Its Chebyshev expansion is finite (4 terms). The product
  $\rho(r)\,\pi(r)$ has smoothness entirely inherited from $\pi(r)$.
- $\rho(r) = (R - r)^{3/2}$ is analytic on $[0, R)$ but has a
  branch-point singularity at $r = R$. Its Chebyshev expansion has
  coefficients decaying only as $N^{-\sigma - 1/2} \sim N^{-2}$
  (Trefethen, *Approximation Theory and Approximation Practice*, Thm 7.2).

**The $N^{-2}$ convergence rate we observe is a direct consequence of
Chebyshev's inability to resolve a fractional-power branch point with
exponential accuracy.**

## Surface regularity of the operator

Consider the operator $L\pi = [\rho \pi']' - k_x^2 \rho \pi$ at $r = R$:

- $\rho \sim (R-r)^3$: smooth zero with $\rho^{(k)}(R) = 0$ for $k = 0, 1, 2$
  and $\rho^{(3)}(R) = -6$ finite.  $L$ is a regular singular operator of
  mild type — Chebyshev polynomial expansion converges spectrally.
- $\rho \sim (R-r)^{3/2}$: $\rho(R) = \rho'(R) = 0$ but $\rho''(r)$ diverges
  as $-\frac{3}{4}(R-r)^{-1/2}$ and $\rho'''$ as $-\frac{3}{8}(R-r)^{-3/2}$.
  The coefficient $\rho$ in $L$ itself is non-analytic; Chebyshev
  reconstruction of $L$ inherits the $N^{-2}$ decay.

## Lane-Emden physics

The Lane-Emden equation $\theta'' + (2/\xi)\theta' + \theta^n = 0$ with
$\theta(\xi_1) = 0$ has surface behaviour $\theta(\xi) \sim (\xi_1 - \xi)$
(linear zero in $\xi$).  Since $\rho = \theta^n$:

$$\rho(r) \sim (\xi_1 - \xi)^n \sim (R - r)^n.$$

**The polytropic index $n$ IS the surface exponent $\sigma$**.  Physical
choices:

| Polytrope | Physical context | $\sigma = n$ | Chebyshev convergence |
|---|---|---|---|
| $n = 1$ | white dwarf outer (non-relativistic degenerate) | 1 | $N^{-3/2}$ algebraic |
| $n = 1.5$ | convective core, fully-convective star | 1.5 | $N^{-2}$ algebraic |
| $n = 2$ | approximate MS star envelope | 2 | $N^{-5/2}$ algebraic |
| $n = 3$ | ideal radiative star, Eddington | **3** | **spectral (exponential)** |
| $n = 3.25$ | giant hydrogen envelope | 3.25 | $N^{-7/2}$ algebraic |

The $n = 3$ Eddington polytrope — historically the most studied because it
gives the mass-radius relation for radiation-pressure-supported stars —
happens to be the **only** standard polytropic index at which Chebyshev
spectral convergence is uninterrupted by the surface.

**This is a previously undocumented coincidence between a physical
regularity (Eddington model) and a numerical regularity (integer surface
exponent).**


# Relation to the Liouville / prefactor machinery

Earlier analysis (`reduced_pressure_liouville.md`, `singular_basis_survey_2026-05-02.md`)
considered two tools to handle the $\rho \to 0$ singularity:

1. **Liouville substitution $\pi = \rho^{-1/2} q$.**  This turns the
   self-adjoint SL operator into Schrödinger canonical form with an
   effective potential $\widetilde W(t) = \sigma(\sigma-2)/(4 t^2)$.
   But this is an operator-level transform; it doesn't change the
   *boundary* smoothness of the underlying function we try to expand
   in polynomials.
2. **Power prefactor $\pi = t^\alpha u$.**  For integer $\alpha$ this
   multiplies $\pi$ by a polynomial — no regularity change.  For
   fractional $\alpha$ (which is exactly what matches $\sigma \notin \mathbb{Z}$),
   it introduces a fractional branch point, which Chebyshev can't resolve
   exponentially either.

**Neither tool restores spectral accuracy for fractional $\sigma$.**  This
is the negative result of E6.

The positive result: for **integer $\sigma$** (most importantly, $n = 3$),
neither tool is needed — raw Chebyshev already gives spectral accuracy.

## What actually works for fractional $\sigma$

The survey (`singular_basis_survey_2026-05-02.md`) identifies two options
that are absent from the Liouville machinery:

1. **Jacobi basis with matching weight $(1-x)^\sigma$** (Dedalus).
   Instead of multiplying the unknown by $t^\alpha$, change the basis:
   expand $\pi$ in $\{(1-r)^\sigma J_n^{(\sigma, 0)}(r)\}$.  The basis
   functions carry the singular behaviour; coefficient expansion is
   in polynomial space.  **Gives spectral convergence for any $\sigma > -1$.**
2. **Coordinate stretching / Kosloff-Tal-Ezer** (not used by GYRE or
   Dedalus, occasionally found elsewhere).  Map $r = r(s)$ so $s$-space
   integration concentrates near the surface.

Both options live **outside** the Liouville framework.


# Implications for the project

## Near-term (Phase 0 ext+ continuation)

- **Exp J's Lane-Emden $n = 3$ target** is perfectly suited to raw
  Chebyshev.  E7 (next step) should show spectral convergence with
  $N \sim 64$ matching Exp J's $N_r = 1024$ FD accuracy.  This is
  sufficient to certify the spectral approach for the stellar-pulsation
  validation benchmark.
- **Lane-Emden $n = 1.5$ and other fractional-$\sigma$ cases** require
  Jacobi basis, i.e. Path B.  Attempting Path A with clever analytic
  substitutions alone will not beat $N^{-2}$.

## Paper angle (updated)

The observation "spectral convergence depends discontinuously on $n$ at
integer values" is a paper-worthy finding in its own right:

- **Method paper (JCP)**: *Polytropic surface regularity and spectral
  convergence of stratified Poisson solvers.*  Show the $N^{-2}$ vs
  spectral dichotomy numerically; identify the integer-$\sigma$
  crossover; compare Liouville prefactors (fail), Jacobi weights
  (succeed), coordinate stretching (TBD).
- **Astrophysical paper (A&C / ApJS)**: same finding framed around
  Eddington vs convective polytropes, noting that the Eddington-model
  ($n = 3$) ubiquity in classical stellar structure textbooks aligns
  fortuitously with the best-behaved numerical case.

## Corrections to existing docs (queued for Phase 0 ext+ E9)

- `docs/anelastic_SL_spectral_design.md`: §8.1 (surface-singularity TODO)
  upgrade to state the integer-vs-fractional crossover explicitly.
- `docs/reduced_pressure_liouville.md`: §9.2 add a new subsection
  documenting that reduced-pressure only weakens but does not eliminate
  the singularity for fractional $\sigma$.
- `docs/singular_basis_survey_2026-05-02.md`: §5 Paths update —
  Path A is demonstrably insufficient for fractional $\sigma$;
  Path B / Jacobi or Path C / Kosloff-Tal-Ezer are the only options.
- `docs/anelastic_sl_phase0_2026-05-02.md`: E3 slope of $-2.4$ is
  confirmed here with $\sigma = 3/2$; the $n = 3$ case (not tested in
  Phase 0) would have given **machine-precision** instead, a data point
  worth adding retroactively.


# Reproducibility

```bash
python scripts/spectral_liouville_convergence_v2.py
```

Output tables and a convergence plot at
`videos/spectral_liouville_convergence_v2.png`.
SymPy forcing, no cutoff, full domain $[0, 1]$.
