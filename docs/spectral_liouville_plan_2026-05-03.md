---
title: Spectral Liouville Solver — Detailed Plan (Path A + Path B parallel)
date: 2026-05-03
status: active
supersedes: docs/gmode_next_session_plan.md
parents:
  - docs/anelastic_SL_spectral_design.md
  - docs/reduced_pressure_liouville.md
  - docs/singular_basis_survey_2026-05-02.md
  - docs/anelastic_sl_phase0_2026-05-02.md
---

# Context — what happened up to 2026-05-02

Previous work established:

1. **`anelastic_SL_spectral_design.md`** — the project's core design: 2D
   anelastic pseudo-spectral solver where the y-direction uses a single set of
   Sturm-Liouville eigenfunctions $\{\psi_n\}$ that simultaneously diagonalises
   the variable-coefficient Poisson operator for all $k_x$. The SL basis also
   yields the g-mode spectrum as a by-product.
2. **`reduced_pressure_liouville.md`** — reformulation using $\pi = p/\rho_0$
   changes the operator from $\nabla\cdot(\rho_0^{-1}\nabla p)$ to
   $\nabla\cdot(\rho_0\nabla\pi)$. The Liouville substitution $\pi = \rho_0^{-1/2} q$
   then yields a potential $\widetilde W$ whose singularity strength at
   $\rho_0 \to 0$ is $|C|=3/16 = 0.1875$, down from the original $|C|=21/16$
   (factor-of-7 reduction). Both Frobenius branches become square-integrable.
3. **Phase 0 / Phase 0 ext** (`anelastic_sl_phase0_2026-05-02.md`) —
   numerical validation:
   - Smooth Gaussian $\rho_0$: exponential convergence confirmed (E3).
   - Lane-Emden $n=3/2$ with cutoff $\rho > 0.01$: err $3.7\times10^{-6}$,
     **algebraic $N^{-2.4}$ convergence**. Singularity at the truncated
     surface is the sole cause.
   - Sturm oscillation (E1), Tassoul asymptote (E2), N² ↔ W physical
     decomposition (E4) all pass.
4. **`singular_basis_survey_2026-05-02.md`** — survey of how GYRE and Dedalus
   handle $\rho\to 0$:
   - GYRE: pre-defined variables $y_1 = x^{2-\ell}\xi_r/r$, $y_2 = x^{2-\ell}P'/(\rho g r)$
     absorb singularities at variable-definition level.
   - Dedalus: Jacobi basis weights $(1-x)^a(1+x)^b$ absorb singularities
     at basis-definition level.
   - Conclusion: our plain Liouville substitution is insufficient. We need
     either (A) a GYRE-style $r^\beta$ prefactor on top of $\sqrt{\rho_0}$,
     or (B) a switch to weighted Jacobi basis.
5. **2026-05-02 detour (Exp E–J)** — valuable side-quests that fed back:
   - Exp E–G: two Python operators for adiabatic Cowling g-modes (Boussinesq-like
     incompressible-buoyancy reduction). Verified self-consistent (Exp G
     algebraic equivalence) but exposed by first external benchmark (Exp H)
     as off by 120% vs GYRE full gravity.
   - Exp I: first GYRE-compatible operator — 2-var Cowling system
     (GYRE's α_grv=0 equations) in staggered FD. max_rel 5.6e-4 vs GYRE Cowling.
   - Exp J: full 4-var adiabatic operator (GYRE's α_grv=1). max_rel 5.3e-4
     vs GYRE full, n_g=1 to 6 significant digits.
   - These Python FD operators are a **reference oracle** for the spectral
     work: if our spectral Liouville solver reproduces Exp J's $\omega^2$ to
     better tolerance with far fewer unknowns, we have proven the spectral
     advantage on a realistic problem.

**What we did NOT do** and is the actual goal: a reduced-pressure Liouville
Chebyshev-spectral solver that handles the $\rho_0 \to 0$ surface singularity
to spectral (exponential) accuracy on Lane-Emden polytropes.


# Goal of this plan

Over the next 2-3 working days, execute **Phase 0 ext+** from the survey:

1. **Path A** — extend the Liouville substitution with a GYRE-style $r^\beta$
   prefactor, find the $\beta$ that restores exponential convergence on
   Lane-Emden, verify against Exp J's frozen reference.
2. **Path B** — in parallel, use Dedalus v3 as a library to solve the same
   Lane-Emden Poisson/g-mode problem with its weighted Jacobi basis.
3. **Decision point**: with both data sets, choose which to carry forward
   into Phase 1 (2D solver), or commit to path C (method-comparison paper).

No CUDA work until the spectral method is demonstrably better than the FD
reference on a realistic stratified problem.


# Deliverables

- **D1** `scripts/spectral_liouville_prefactor.py` — reduced-pressure Liouville
  + Chebyshev spectral + $r^\beta$ prefactor sweep. PASS: $\beta^\star$ found;
  convergence exponential.
- **D2** `scripts/spectral_liouville_gmode_benchmark.py` — apply the
  $\beta^\star$ solver to Exp J's Lane-Emden $n=3$ polytrope, compare
  $\omega^2$ to frozen GYRE EXPECTED. PASS: same 6-digit n_g=1 agreement with
  1/8–1/16 the DOF of Exp J's Nr=1024.
- **D3** `scripts/spectral_dedalus_gmode_benchmark.py` — Dedalus Lane-Emden
  g-mode EVP (Ball basis or Jacobi). PASS: independent $\omega^2$ reproducing
  GYRE EXPECTED.
- **D4** `docs/spectral_liouville_phase0ext_2026-05-03.md` — results report,
  includes method A / B side-by-side table: convergence rate, wall time,
  DOF-for-target-accuracy.
- **D5** Decision memo (appended to D4): which path for Phase 1.


# Step-by-step work items

## Step 1 — E5: Liouville + $r^\beta$ prefactor derivation and scan  (Day 1 AM)

**Goal.** Find analytic $\beta^\star$ that makes the transformed potential
$\widetilde W_\beta$ finite as $t \to 0$ ($t = R - r$, the stellar surface
coordinate).

**Setup.**  Start from the reduced-pressure Liouville equation
$q'' + \widetilde W q - k_x^2 q = \tilde g$ (`reduced_pressure_liouville.md`
eq 11). Make a second substitution $q = t^\beta \tilde q$:

$$\widetilde W_\beta(t) = \widetilde W(t) + \frac{\beta(\beta-1)}{t^2} + \frac{2\beta\,\rho_0'(t) / \rho_0(t)}{t} - \frac{\beta}{t}\,\frac{d}{dt}[\ln f_\beta]$$

(exact form to be derived symbolically at execution time).

**Action.**
1. Use SymPy to derive $\widetilde W_\beta(t)$ for Lane-Emden surface decay
   $\rho_0 \sim t^{3/2}$ (n=3/2) and $\rho_0 \sim t^3$ (n=3 — our Exp J
   target).
2. Solve analytically for the $\beta$ that cancels the leading $t^{-2}$
   divergence in $\widetilde W_\beta$. Expected result: $\beta_\star = 1/4$
   or $3/4$ (the Frobenius exponents found in `reduced_pressure_liouville.md`
   §4).
3. Document the derivation in new `docs/spectral_liouville_phase0ext_2026-05-03.md` §1.

**PASS.** $\widetilde W_\beta(t=0)$ finite under SymPy expansion; algebra
confirmed manually.

## Step 2 — E6: Chebyshev solver with prefactor, 1D SL eigenproblem  (Day 1 PM)

**Goal.** Implement the solver and verify exponential convergence on
Lane-Emden including $t = 0$ without cutoff.

**Setup.** `scripts/spectral_liouville_prefactor.py`:
- Chebyshev-Gauss-Lobatto grid on $t \in [0, R]$ (or $x \in [0, 1]$).
- Assemble $-\tilde q'' - \widetilde W_\beta \tilde q = \mu \tilde q$ with
  endpoint regularity conditions (at $t = 0$, $\tilde q$ finite;
  at $t = R$, either Dirichlet or inherited from outer BC of Exp J).
- Reference solutions: Exp J's EXPECTED $\omega^2$ (for comparison via
  $\mu_n \leftrightarrow \omega_n^2$ — derive the exact mapping; the
  Liouville $\mu_n$ ARE the negative SL eigenvalues, not $\omega^2$
  directly, so adapt).

Wait — rigorous sanity check needed: the Liouville operator of
`reduced_pressure_liouville.md` is for solving the **Poisson equation**, not
directly the g-mode equation. Those two share the basis but not the
eigenvalues. Re-read §4 of the design doc and decide: do we target Poisson
inversion convergence, or g-mode frequency convergence, or both?

**Plan.**
- **E6a** Target first: Poisson inversion on a manufactured solution
  $\pi_\text{exact}(r) = \sin(k_r r)(1 - r/R)^{3/4}$ (captures the $\alpha=3/4$
  Frobenius branch), full domain $r \in [0, R]$.
- **E6b** Separately: the g-mode problem is a SEPARATE generalised EVP
  $L\phi = \omega^2 M\phi$ on the same grid; verify against Exp J EXPECTED.
  (If we pick Path A's "unified basis" narrative, E6b demonstrates that the
  SAME Chebyshev basis discretises both operators.)

**PASS.**
- E6a: $\|q_\text{num} - q_\text{exact}\|_\infty / \|q_\text{exact}\|_\infty$
  exponential in $N$; e.g. $\le 10^{-8}$ at $N = 64$.
- E6b: max rel_diff vs Exp J EXPECTED $\omega^2$ $\le 10^{-3}$ at $N = 64$
  (i.e. 16× fewer DOF than Exp J's Nr = 1024 while matching 3-digit accuracy
  for n_g=1..10).

## Step 3 — E7: Apply to Lane-Emden $n=3$, compare to Exp J  (Day 2 AM)

**Goal.** Directly benchmark against the frozen GYRE EXPECTED from Exp J.

**Action.**
1. Write `scripts/spectral_liouville_gmode_benchmark.py`:
   - Same GYRE Lane-Emden $n=3$ structure as Exp J.
   - Discretise the full 4-variable GYRE adiabatic system in Chebyshev with
     $r^\beta$ prefactor on the $y_1, y_2$ pair that carries the $\rho_0$
     dependence. (Or equivalently, for Path A's narrative, apply the
     Liouville+prefactor *basis* to the Poisson sector only and handle the
     $\Phi'$ sector separately.)
2. Compare $\omega_n^2$ for $n_g = 1..10$ to `EXPECTED_OMSQ_GYRE` in
   `scripts/gmode_exp_j_full_gyre_compat.py`.
3. Run at N = 16, 32, 64, 128 to establish convergence.
4. Record wall-time (host) for comparison with Exp J's $4N-2 = 4094$ dense
   GEP ($\approx 10$ s on this hardware).

**PASS.** N = 64 gives max_rel $\le 10^{-6}$ on n_g=1 (matching Exp J's
$5.9\times 10^{-7}$) with 64 DOF vs Exp J's 4094.

## Step 4 — E8 Path B: Dedalus independent benchmark  (Day 2 PM)

**Goal.** Establish Path B baseline with zero implementation risk.

**Action.**
1. `pip install dedalus` in a fresh venv (if not already).
2. Adapt `examples/nlbvp_ball_lane_emden/lane_emden.py` to solve the
   adiabatic oscillation EVP. Follow GYRE dimensionless form.
3. Use Dedalus's weighted Jacobi basis appropriate for Lane-Emden n=3.
4. `scripts/spectral_dedalus_gmode_benchmark.py`: same interface as E7,
   compare to Exp J EXPECTED.

**PASS.** Dedalus reproduces Exp J EXPECTED to similar accuracy as E7.
Record DOF and wall time.

## Step 5 — E9: Convergence table + decision  (Day 3 AM)

**Goal.** Data-driven decision between Paths A / B / C.

**Action.** Write `docs/spectral_liouville_phase0ext_2026-05-03.md`:

1. **§1** $\beta^\star$ derivation (from Step 1).
2. **§2** E5/E6/E7 (Path A) results: $\beta^\star$ sweep, convergence plots,
   Exp J agreement.
3. **§3** E8 (Path B) results: Dedalus convergence, Exp J agreement.
4. **§4** Side-by-side comparison:
    | Method | DOF for $10^{-6}$ | Wall time | Implementation LOC | g-mode auto? |
    |---|---|---|---|---|
    | Exp J FD staggered | 4094 | 10 s | 500 | separate EVP |
    | Path A (Liouville + $r^\beta$) | ? | ? | ? | **yes — SL spectrum** |
    | Path B (Dedalus Jacobi) | ? | ? | ? | separate EVP |
5. **§5** Decision: which path for Phase 1.

## Step 6 — E10: Roll forward or retire the FD operator  (Day 3 PM)

Based on Step 5 decision:
- **If Path A chosen**: freeze Chebyshev + $r^\beta$ Liouville solver as
  the reference for Phase 1's y-direction discretisation. The Exp I/J FD
  operators remain as regression oracles.
- **If Path B chosen**: pivot Phase 1 design to Jacobi-Galerkin on
  y-direction. Exp I/J remain regression oracles. The "unified basis"
  selling point weakens; rewrite paper angle to JCP method comparison or
  pure A&C HPC.
- **If Path C chosen**: both A and B live on as dual Phase 1 implementations.


# Success criteria (gate for starting Phase 1 / CUDA work)

**All three must be met before any 2D or CUDA work:**

1. [_] Exponential convergence demonstrated on Lane-Emden without cutoff
   (E3 reproduced, but with prefactor instead of cutoff, OR with Jacobi
   basis).
2. [_] Exp J's frozen EXPECTED $\omega^2$ reproduced to $\le 10^{-3}$ by at
   least one spectral method with DOF $\le 128$.
3. [_] Path A or Path B (or both) committed to for Phase 1, with written
   rationale.


# Non-goals for this sprint

- No CUDA coding.
- No 2D solver.
- No anelastic (Phase 3) physics.
- No redesign of Exp I / J — they are frozen oracles.
- No paper writing — that waits until Phase 1 is running.


# Risks and fallback

- **R1** $\beta^\star$ may not be analytically clean for Lane-Emden $n=3$
  (vs $n=1.5$ where it's $1/4$ or $3/4$). Fallback: numerical $\beta$ sweep
  with $10^{-3}$ resolution; accept any value that gives exponential
  convergence.
- **R2** Dedalus Ball basis may not directly apply to our 1D radial problem;
  may need Jacobi on $[0,1]$ with specific $(a, b)$. Fallback: use Jacobi
  with $(a, b) = (0, 3)$ (matching Lane-Emden $n=3$ surface decay $\rho \sim t^3$).
- **R3** Both paths may turn out equivalent at the $10^{-6}$ level, making
  the decision subjective. Fallback: commit to Path C (dual implementation).
- **R4** Execution may exceed 3 days. Acceptable cost for avoiding a wrong
  CUDA architectural commitment.
