---
title: |
  MHD Derivations for stellar2d —
  a sympy-verified manuscript
author: |
  stellar2d development notes\
  *Department of Astronomy, Tsinghua University*\
  `github.com/MahoMaho-Rize/stellar2d`
date: 2026-05-08
---

# Front matter

## Purpose

This manuscript is an **internal, reproducibility-grade derivation document**
for the MHD extension of stellar2d. It covers four parts:

- **Part A** — ideal MHD equations, conservative / primitive forms,
  flux Jacobian eigensystem, HLLD intermediate states, constrained
  transport (CT) preservation of $\nabla\cdot\mathbf{B}=0$; PLM/PPM
  reconstruction + TVD slope limiters; VL2 predictor-corrector and
  its CFL bound (hyperbolic + parabolic); HLLD degeneracy branches;
  Powell-source vs CT comparison; linear-wave convergence IC.
- **Part B** — reduction to a 1D super-radial flux-tube geometry used by
  the Suzuki-group stellar-wind codes, including the WKB Alfvén-wave
  action integral, the Parker critical-point condition in this
  geometry, and the well-balanced MHSE operator needed to keep the
  atmosphere quiet for long-time wind runs.
- **Part C** — non-ideal dissipation: Ohmic, ambipolar, and their
  contributions to the energy equation; closure of the two diffusivities
  through Saha ionisation; sub-grid turbulent heating closure
  (Suzuki-Inutsuka 2005) for 1D flux-tube wind runs.
- **Part D** — cylindrical shearing-box MHD, the shearing-periodic
  boundary condition, and the exact Maxwell–Reynolds stress
  decomposition of the $\alpha_{\mathrm{SS}}$ metric.

## Reproducibility protocol

Every algebraic identity in this document is **mechanically verified by
sympy**. Each section corresponds to a script in
`docs/mhd_derivations/scripts/<section>.py` that ends with
`assert_zero(LHS − RHS, ...)` calls. If any identity cannot be
simplified by sympy but is physically correct, we document the
alternate verification route (e.g., "manually checked with Wolfram,
reason: polynomial simplification blows up").

The scripts are intentionally **independent**: each one imports only
`_common.py` and re-derives everything it needs from first principles.
No cross-script caching of intermediate symbolic results. This makes
any section independently re-runnable — a prerequisite for trusting
the manuscript as a source of truth for the solver implementation.

## Conventions

- Units are Gaussian with $4\pi$ absorbed into $\mathbf{B}$, so that the
  Alfvén speed is simply $c_A = B/\sqrt{\rho}$, matching Stone & Gardiner
  2008, Athena++, PLUTO.
- $\gamma$ always denotes the ratio of specific heats (not the relativistic
  Lorentz factor).
- Einstein summation is *not* used; sums are written explicitly.
- sympy variable names mirror the mathematical symbols
  (`rho` = $\rho$, `B_x` = $B_x$, etc.); see `scripts/_common.py` for the
  full inventory.

## How to regenerate this manuscript

```bash
cd docs/mhd_derivations
bash run_all.sh         # (re-)runs every scripts/*.py, refreshes output/
bash build_manuscript.sh  # concatenates sections/*.md → manuscript.{md,pdf}
```

If a sympy assertion fails during `run_all.sh`, the build halts and the
offending section is flagged. No partial manuscript is emitted.
