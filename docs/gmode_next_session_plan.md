---
title: g-mode Validation — Next Session Plan (GYRE-Compatible Operator)
date: 2026-05-02 (late evening, pre-compact)
status: COMPLETED — see docs/gmode_experiments_2026-05-02.md §§13-15
---

# STATUS (2026-05-02, next day)

All items in this plan are DONE:

- Exp I (2-var Cowling GYRE-compat): PASS, max_rel 5.6e-4 (commit `953d49f`).
- Exp J (4-var full-gravity GYRE-compat): PASS, max_rel 5.3e-4, n_g=1 to
  6 significant digits (commit `be94af9`).  This is the production
  reference for CUDA port.
- `solve_gmode_full_gyre_compat` in `gmode_infra.py` is the finalised
  operator; both `--verify` regressions are frozen and must exit zero.

This document is retained for historical context on how the diagnosis
and fix path proceeded.  For current state read
`docs/gmode_experiments_2026-05-02.md` §§12-15.


# Why this doc exists (original)

Context is about to be compacted.  This document records:

1. What we found today that invalidates earlier claims.
2. Why Exp H (GYRE benchmark) shows ~2.2× disagreement with our Python.
3. The concrete plan agreed for the next session.

The reader (future-me after context compaction, or the user) should start
here, then re-read `docs/gmode_experiments_2026-05-02.md` §§0a, 11 only
after absorbing this summary.


# Current state (HEAD before compact)

Branch: `pseudo-astro-explore`, last pushed commit `7a80d09`.

Files added today that are still valid:
- `scripts/gmode_infra.py` — shared infrastructure (Lane-Emden, W, SL eig,
  Cowling scalar slab + spherical, Chebyshev).  Includes a new
  `solve_gmode_cowling_spherical_regular()` added late this session with a
  Robin / regular-at-origin inner BC — this did NOT close the gap vs GYRE.
- `scripts/gmode_exp_a_lane_emden.py` — pipeline smoke check, PASS.
- `scripts/gmode_exp_b_stratified.py` — FD Gaussian-bump cavity, PASS (converges
  to Tassoul).
- `scripts/gmode_exp_c_chebyshev.py` — Chebyshev version, PASS.
- `scripts/gmode_exp_d_polytrope_profile.py` — MESA-style parser regression, PASS.
- `scripts/gmode_exp_e_anelastic_linop.py` — 2-var vs slab scalar, PASS (but
  the PASS condition is weak; see §0a).
- `scripts/gmode_exp_f_variable_rho.py` — variable ρ₀, PASS.
- `scripts/gmode_exp_g_spherical_scalar.py` — full spherical scalar vs 2-var,
  PASS with O(Nr⁻²) convergence.
- `scripts/gmode_exp_h_run_gyre.sh` — runs GYRE on its shipped n=3 poly.
- `scripts/gmode_exp_h_gyre_benchmark.py` — external benchmark, **FAIL: 2.2×**
  disagreement at n_g = 1, decreasing to ~1× at high n_g.  Written but the
  FAIL mode is diagnosed (see below), not committed with a pass/fail label.
- `scripts/mesa_profile.py` — simple MESA-style column-table reader.

MESA SDK at `~/mesasdk-26.3.2`, GYRE at `~/gyre` (built).  GYRE run
artefacts at `/tmp/gyre_run/summary.h5` + `poly3.txt`.


# What we found (the honest version)

## Exp H disagreement is NOT a bug in our code.  It is a physics mismatch.

GYRE's dimensionless adiabatic Cowling equations (with `alpha_grv=1`) are a
coupled 4-variable system (y₁..y₄).  With `alpha_grv=0` (true Cowling
approximation) they reduce to 2 variables (y₁, y₂), still involving FOUR
structure coefficients:

- `V` = -dlnP/dlnr / x              (scaled pressure gradient)
- `U` = dlnM/dlnr                    (enclosed mass gradient)
- `A*` = rN²/g  = r³N²/(GM_r)        (dimensionless Brunt squared)
- `c_1` = x³ M / M_r                 (scaled mass ratio)
- `Γ₁`  (constant for polytrope)

Our `solve_anelastic_2var` only uses `rho_0` and `N²`.  It silently drops
the V, U, Γ₁ coupling and amounts to a **Boussinesq-like** reduction of
the full Cowling system.  This is NOT the "anelastic" approximation in
the stellar-oscillation sense; it is a simpler incompressible-momentum +
buoyancy-only reduction.

For the artificial Gaussian-bump cavities of Exps B-G (short cavity,
small fractional ρ₀ variation) the missing terms are small — so our
solver internally self-consistent and converges against its own scalar
reduction (Exp G) beautifully.

For a real polytrope (Exp H: full [0, R] domain, ρ₀ varies 10⁹×), the
missing V/U/Γ₁ coupling dominates at low n_g:
- n_g = 1: ratio 2.2× (large discrepancy)
- n_g = 5: ratio 1.3×
- n_g = 10: ratio 1.1×
- n_g ≫ 10: ratio → 1 (high-n Boussinesq limit)

Compare against GYRE with `alpha_grv=0` (pure Cowling):
- n_g = 1: ω² = 2.852 (full GYRE: 2.516, 13% difference)

So GYRE-Cowling and GYRE-full agree to ~13% at low n_g.  Our
Boussinesq-like reduction is a different beast, ~2× off from both.


# The plan for the next session

## Goal

Write a **GYRE-compatible Python 2-variable Cowling operator** that
takes (V, U, A*, c_1, Γ₁) as inputs and implements exactly the equations
GYRE solves with `alpha_grv=0`.  Verify it matches GYRE Cowling output
to <1%.  This becomes the reference implementation for the future
C++/CUDA anelastic operator port.

## Concrete equations to implement

From `docs/source/ref-guide/osc-equations/dimless-form.rst` with α_grv = 0:

```
x dy_1/dx = (V/Γ₁ - 1 - ℓ) y_1 + (λ/(c_1 ω²) - α_γ V/Γ₁) y_2
x dy_2/dx = (c_1 ω² - α_π A*) y_1 + (3 - U + A* - ℓ) y_2
```

where λ = ℓ(ℓ+1), and for standard adiabatic:  α_γ = α_π = 1.

Discretise on uniform x-grid as a generalised eigenproblem
`A u = ω² B u` where u = [y_1; y_2] and ω² is the eigenvalue.  BCs:

- Inner (x → 0, REGULAR):  `c_1 ω² y_1 - ℓ y_2 = 0`   (the IB_regular.inc
  condition we already extracted from GYRE source)
- Outer (x = 1, VACUUM):   `y_1 - y_2 = 0`   (the OB vacuum condition; find
  the exact form in `gyre_ad_obound_m.fypp`)

## Step-by-step

1. **Read GYRE outer BC** from `~/gyre/src/eqns/ad/gyre/gyre_ad_obound_m.fypp`
   and the corresponding `OB_*.inc` files.  Verify it matches the form
   `y_1 - y_2 = 0` for VACUUM adiabatic.
2. **Write `solve_gmode_cowling_gyre_compat(x, V, U, A_star, c_1, Gamma_1, ell, n_modes)`**
   in `gmode_infra.py`.  Handles BCs as algebraic constraints on boundary
   rows of A/B matrices (row 0 and row N-1).
3. **Write Exp I**: `scripts/gmode_exp_i_gyre_compat.py`.  Loads the
   GYRE `/tmp/gyre_run/poly3.txt` (which gives exactly the five structure
   coefficients), runs `solve_gmode_cowling_gyre_compat` at Nr=1024 or
   2048, compares to GYRE `summary_cowling.h5` (with `alpha_grv=0`).
   Target agreement: <1% relative error on n_g = 1..10 at Nr = 2048.
4. **Re-run Exp H** using `solve_gmode_cowling_gyre_compat` — should now
   agree with full GYRE to ~13% (the known Cowling vs full-gravity
   difference) rather than 220%.
5. **Add `docs/gmode_experiments_2026-05-02.md` §13 (Exp I)** + §14 a
   corrections log that supersedes earlier PASS verdicts:
   > Exps E, F, G tested an incompressible-buoyancy 2-var operator that
   > does NOT match GYRE's Cowling or full-gravity adiabatic equations.
   > The consistency they demonstrated was *among themselves* only.
   > Exp I is the first true external benchmark; Exps E-G are
   > reclassified as "internal consistency of the simplified operator"
   > and the simplified operator itself is retained as an educational
   > baseline, not a production reference.

## After that — then (and only then) start C++

With Exp I passing to <1% against GYRE, the Python
`solve_gmode_cowling_gyre_compat` is a trustworthy reference.  Port it to
C++:

```
src/gpu/anelastic_operator.{cu,cuh}   (new file, per CLAUDE.md)
```

Initial port: dense LAPACK or sparse shift-invert on host/GPU, matching
Nr ~ 2048.  Regression test: feed the same GYRE poly3.txt, compare ω²
to Python reference.


# What to commit before compact

Everything in the "Files added today" list except the Exp H partial
implementation (its PASS criterion is wrong — fix in next session).
Status notes:

- docs/gmode_experiments_2026-05-02.md: committed (last push 7a80d09)
- Exp G: committed and PASSes its own (correct) criterion
- Exp H files: LOCAL only, not yet committed (outputs FAIL; documentation
  of why is in this plan)
- gmode_infra.py — has the new `solve_gmode_cowling_spherical_regular`
  addition: LOCAL only.

**Action on commit**: add Exp H files + the new infra function + this
plan document in one commit labelled "WIP: Exp H external benchmark
(documented disagreement; see gmode_next_session_plan.md §Plan)".


# Session takeaways for the user to remember

1. **Exps B–G all used the SAME simplified operator** and validated each
   other.  That is internal consistency, not external correctness.
2. **The first external check (GYRE) exposes a 2.2× gap at low n_g.**
   The gap is a physics mismatch (missing V, U, Γ₁ coupling), not a bug.
3. **The fix is to re-implement the operator following GYRE's exact
   equations** (V, U, A*, c_1, Γ₁ as inputs), not to patch the current
   simplified one.
4. **Don't port to C++ until Exp I passes <1% against GYRE.** Everything
   before that would be porting a toy.
