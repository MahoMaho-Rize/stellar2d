# DNS benchmark plan — response to reviewer #3 (missing nonlinear benchmark)

**Context.** The paper currently argues its case entirely through
eigenmode-preservation tests (linear regime) plus a 800-step
linear-adjacent nonlinear prototype (§7). A JCP numerical-methods
reviewer will (rightly) demand a non-trivial nonlinear benchmark that
exercises the advertised selling point — the assembled-operator
scheme preserving its linear closure while carrying quadratic
advection physics. This document is the working plan for adding two
such benchmarks before the next submission round, written so that
work can resume on either host.

## Selected experiments

After surveying the existing infrastructure, the two minimal,
reviewer-compelling benchmarks are:

### Experiment A — 3-wave triad mode coupling (mandatory)

**Goal.** Demonstrate that the assembled-operator scheme carries
quadratic advection over 50–100 oscillation periods while conserving
total energy to $\lesssim 10^{-10}$ and exhibiting physically correct
triad energy transfer.

**Setup.**
- Lane-Emden $n = 3/2$, $\rho_{\mathrm{cut}} = 0.05$, $N_y = 64$, $N_x
  = 64$, $L_x = L_y = 1$.
- IC: single g-mode $(n_g = 1, k_x = 2\pi/L_x)$ at amp $= 10^{-2}$
  (large enough to excite $\mathcal O(10^{-4})$ second-harmonic
  amplitude within 10 periods; small enough to remain weakly
  nonlinear).
- Strang-split integrator (Path 1 of §7) with assembled $\mathsf M$
  linear half-step + RK4 nonlinear full-step, 2/3 dealias on $x$.
- Run $300$ periods with $\Delta t$ targeting CFL ≤ 0.5 on the
  fastest retained g-mode.

**Diagnostics.**
- Total energy $E(t) = \tfrac12 \int \rho_0(u^2 + v^2)\,dy\,dx + \int
  (b^2/(2N^2))\,dy\,dx$ (kinetic + available potential), sampled every
  period.
- Modal energy $E_k(t)$ for $k \in \{1, 2, 3, 4\} \cdot 2\pi/L_x$ via
  horizontal FFT.
- Eigenmode deviation (same metric as §5.1) on the primary $k_1$
  component.

**Pass criteria.**
- $|\Delta E/E(0)| < 10^{-10}$ over 300 periods.
- $E_{k_2}(t)$ grows $\propto t^2$ in the weakly nonlinear window and
  plateaus (no secular exponential growth).
- Primary mode deviation remains at Proposition 2 floor
  ($\lesssim 10^{-13}$) in the linear component.

**Infrastructure.**
- Python reference: `scripts/nonlinear_path1_opsplit.py` already has
  the Strang-split integrator, `make_eigenmode_ic`, and
  `compute_advection`. Add diagnostics output + plot.
- GPU: `src/gpu/anelastic_sl_solver.{cu,cuh}` — **need to verify
  nonlinear advection block is wired in, and that IFRK3 ordering
  matches Strang.** If GPU currently only does linear, use GPU for
  the 300-period linear-closure check and Python for the nonlinear
  diagnostics run (Python is fast enough at $64^2$).

**New code.** ~50 lines of runner + plot. Target output:
`paper/figures/fig7_1_triad.png` replacing/complementing current
fig7_1.

**Cost estimate.** Python ~30 min/300 periods @ $64^2$; GPU ~5 min.

---

### Experiment B — Parametric resonance (secondary, strongly recommended)

**Goal.** Exhibit a physically non-trivial nonlinear phenomenon —
parametric subharmonic instability — as a stronger reviewer
demonstration than energy conservation alone.

**Setup.**
- Same background / resolution as A.
- Pre-select a g-mode pair $(a, b)$ such that $\omega_a \approx
  2\omega_b$ (approximately; within 1–2% detuning). Candidate from
  the EVP spectrum on Lane-Emden $n=3/2$, $N_y = 96$:
  - scan $(n_g, k_x)$ space for $\omega_a/\omega_b$ near integer 2
  - likely candidate: $(n_g=1, k_x=2\cdot2\pi/L_x)$ vs $(n_g=2,
    k_x=2\pi/L_x)$ — verify with a `scan_resonance.py` utility (~40
    lines).
- IC: $a$-mode at amp $= 10^{-2}$, $b$-mode at seed amp $= 10^{-6}$.
- Run until $b$ saturates or 500 periods, whichever first.

**Diagnostics.**
- $\log E_b(t)$ — expect linear-in-time growth rate
  $\gamma \approx \epsilon\, |V_{abb}| / 2$ (three-wave coefficient).
- Saturation level vs amp (sweep $10^{-2}, 10^{-2.5}, 10^{-3}$).

**Pass criteria.**
- Measured $\gamma$ agrees with three-wave theory to $\sim 10\%$.
- Post-saturation: no exponential blow-up (differentiates from the
  §5.7.2 reduced-operator pathology).

**Infrastructure.** Same as A, plus resonant-pair scan.

**New code.** ~80 lines (pair scan + runner + plot + theory comparison).

**Cost estimate.** Python multi-hour; **GPU essential** for practical
turnaround.

---

## Not selected (and why)

- **Broadband g-mode turbulence.** Reviewer would contest the
  physical meaning of 2D pseudo-anelastic turbulence. Not worth the
  space.
- **Boussinesq ↔ Lane-Emden crossover.** No new information beyond
  §5.1 / §6.3 linear results.
- **§7 as-is (three-method comparison prototype).** The current
  three-method comparison is linear-adjacent and will be demoted to
  Appendix; the Strang-split path becomes the §7 production
  integrator carrying Experiments A and B.

---

## Paper integration

### Current §7 structure (to be replaced)

1. Problem statement
2. Three candidate paths (Strang / IMEX / exponential)
3. Short-horizon prototype comparison (800 steps)
4. Conclusion: Strang recommended

### Proposed §7 structure

1. §7.1 Problem statement — same opening, but ending with "we
   demonstrate the scheme in two weakly nonlinear benchmarks".
2. §7.2 Strang-split integrator construction (inherits from current
   §7; compressed).
3. §7.3 **Experiment A — 3-wave triad**. Figure 7.1: modal energy
   vs time + total energy conservation. Main result text.
4. §7.4 **Experiment B — parametric resonance**. Figure 7.2: $\log
   E_b(t)$ + growth rate comparison. Main result text.
5. §7.5 Brief summary: scheme carries nonlinear physics without
   losing Proposition 2's linear floor.
- Appendix A: IMEX / exponential prototype comparison (demoted from
  main text).

### Abstract addition

One new sentence after the GPU benchmark line:

> "On two nonlinear benchmarks — three-wave triad energy transfer and
> parametric subharmonic resonance on Lane–Emden $n = 3/2$ — the
> Strang-split extension conserves total energy to $10^{-10}$ over
> $300$ oscillation periods and recovers the three-wave theoretical
> growth rate to within $10\%$."

---

## Host-switch handoff

### What's ready on this host (laptop)

- Python infra: `scripts/nonlinear_path1_opsplit.py` +
  `nonlinear_paths_infra.py` + `nonlinear_paths_compare.py`.
- Paper: all §5-§9 consolidated, PDF at `paper/paper.pdf` (25 pages).
- Review artefacts: `scripts/review_e{1,2,3,4}_*.py` for Proposition 1
  verification experiments.

### What needs verification on GPU host

1. `src/gpu/anelastic_sl_solver.cu` — does the current code path
   include the nonlinear advection block? Grep for `compute_advection`,
   `dealias_23`, or `nonlinear` inside the solver.
2. IFRK3 vs Strang — the GPU solver uses IFRK3 for the linear block;
   verify this matches the Python Strang-split semantics in the
   amp → 0 limit (should, since IFRK3 on assembled $\mathsf M$
   recovers the same linear propagator).
3. Output cadence — ensure diagnostics every period, not every step,
   to keep VRAM frame buffer manageable over 300 periods.

### Execution order (suggested)

1. Run Experiment A on Python first (small, quick reference).
2. Port A to GPU; match against Python to machine precision.
3. Write scan_resonance utility, identify $(a, b)$ pair.
4. Run Experiment B on GPU only.
5. Generate figures fig7_1_triad.png, fig7_2_resonance.png.
6. Rewrite §7 per structure above.
7. Update abstract, §1.3 contributions.
8. Rebuild PDF.

### Files likely to be edited in next round

- `paper/07_nonlinear.md` — rewrite.
- `paper/01_intro.md` — §1.3 add contribution bullet 5.
- `paper/paper-cas.tex` — abstract one-sentence addition.
- `paper/11_appendix_a.md` — move current §7.3 prototype comparison
  here.
- `scripts/nonlinear_path1_opsplit.py` — add diagnostics hooks.
- `scripts/dns_a_triad.py` — new runner (thin wrapper).
- `scripts/dns_b_resonance.py` — new runner.
- `scripts/scan_resonance.py` — new EVP utility.

---

*Last updated: session ending 2026-05-04 on laptop; next session on
GPU host.*
