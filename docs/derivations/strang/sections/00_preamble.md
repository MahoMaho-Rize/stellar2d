---
title: |
  Strang Derivations for stellar2d —
  a sympy-verified manuscript
author: |
  stellar2d development notes\
  *Department of Astronomy, Tsinghua University*\
  `github.com/MahoMaho-Rize/stellar2d`
date: 2026-05-08
---

# Front matter

## Purpose

This manuscript is an **internal, reproducibility-grade derivation
document** for the Strang-split Euler solver of stellar2d
(`src/gpu/explicit/strang_solver.{cu,cuh}`).  It is a companion to the
MHD book (`docs/derivations/mhd/`) and follows the same sympy-driven,
strong-form-only methodology.

The book covers five parts, 36 sections in total:

- **Part A-phys** (A1–A6) — Compressible Euler equations in strong
  form: mass / momentum / total energy; flux Jacobians and their
  eigensystems; rotational covariance of the y-flux; entropy condition
  and the Lax criterion; simple-wave families.
- **Part A-num** (A7–A14) — Riemann-solver family (Rusanov, HLLE,
  HLLC, Roe) with contact-wave resolution cross-comparison; HLLC
  intermediate states; Davis wave-speed estimates with Einfeldt
  entropy fix; slope-limiter family (MC, minmod, van Leer, superbee,
  Ospre) in Sweby form; reconstruction-order hierarchy (donor-cell,
  MUSCL, PPM Colella-Woodward, PPM Colella-Sekora); MUSCL-Hancock
  half-step identity; time-integrator family (Strang, Lie, unsplit
  VL2, RK2-MUSCL) via BCH expansion; the `step` operator chain.
- **Part B** (B1–B6) — Perturbation storage $(\delta\rho, m_x, m_y,
  \delta E)$; isentropic HSE with the closed-form density profile;
  face-centred HSE reconstruction as a necessary condition for well-
  balancing; ghost-cell boundary conditions (periodic-x, reflective-y,
  outflow-y).
- **Part C** (C1–C4) — Gravity source-term work consistency; strong-
  form CFL bound; LM-HLLC pressure-jump blending and its $M\to 0$
  dispersion analysis; smooth-flow entropy invariant.
- **Part D** (D1–D7) — Canonical initial conditions with analytic
  golden values: entropy wave; acoustic linwave; Sod shock tube;
  Woodward-Colella two-shock blast; bubble IC; HSE zero-perturbation
  lock; reflection-symmetric IC.
- **Part E** (E1–E5) — Post-hoc benchmark derivations: entropy-wave
  convergence order; acoustic-wave LM-HLLC order pathology;
  LM-HLLC effective viscosity; Strang split source-term commutator;
  long-time HSE drift bound.

## Reproducibility protocol

Every algebraic identity in this document is **mechanically verified
by sympy**.  Each section corresponds to a script
`docs/derivations/strang/scripts/<section>.py` that ends with
`assert_zero(LHS − RHS, label)` calls.  If sympy cannot symbolically
simplify an expression that is physically correct, the script falls
back to numerical random sampling at N ≥ 50 admissible points with
absolute tolerance ≤ 10⁻¹⁰, and the markdown section flags the
fallback explicitly.

## Strong-form rule

All derivations default to **strong-form**, pointwise identities of
the shape $A(x,t) \equiv B(x,t)$ rather than weak-form identities
against a test-function family.  Weak-form fallback is allowed only
for three situations: (1) non-linear operator identities with no
closed-form expansion (e.g., full-non-linear BCH), (2) benchmarks
with no closed-form solution (e.g., the Woodward-Colella blast after
wave interactions), and (3) the finite-volume cell average itself,
which is inherently integrated.  Every weak-form step is labelled
`[WEAK]` in the markdown, carries a sympy numerical-consistency
check, and includes a plain-English justification.

## Script independence

Scripts are intentionally **independent**.  Each one imports only
`_common.py` and re-derives everything it needs from first principles.
No cross-script caching of intermediate symbolic results.  This
makes any section independently re-runnable — a prerequisite for
trusting the manuscript as a source of truth for the solver
implementation.

## Conventions

- Units are arbitrary but consistent; the Strang solver in the
  codebase runs with $\gamma = 5/3$ and $G = g$ read from the
  configuration.  The derivation is $G$-agnostic and $\gamma$-generic.
- $\gamma$ denotes the ratio of specific heats; $\gamma - 1$ is
  commonly factored as `gm1` in the kernel.
- The perturbation-form variables $(\delta\rho, m_x, m_y, \delta E)$
  are the on-disk storage; primitive variables $(\rho, u, v, P)$ are
  reconstructed from the stored fields plus the HSE background
  $(\bar\rho(y), \bar p(y))$ every time the solver needs them.
- sympy variable names mirror the mathematical symbols wherever
  readable (`rho` = $\rho$, `u` = $u$, `p` = $p$, `c_sound` = $c$);
  see `scripts/_common.py` for the full inventory.
- Golden values for Part-D ICs live in `output/d*_goldens.json`,
  which is **not** committed to the repository.  `bash run_all.sh`
  is a build-time prerequisite of `ctest` and regenerates every
  JSON.

## How to regenerate this manuscript

```bash
cd docs/derivations/strang
bash run_all.sh             # refreshes output/*.latex.tex and output/d*_goldens.json
bash build_manuscript.sh    # assembles sections/*.md -> manuscript.{md,pdf}
```

If a sympy assertion fails during `run_all.sh`, the build halts and
the offending section is flagged.  No partial manuscript is emitted.
