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

The two benchmarks are physically meaningful in the asteroseismology
literature *and* mechanically adversarial for the matrix-free default
— they force the scheme to get nonlinear energy transfer *right*,
not merely stable.

### Experiment A — Two-mode near-resonant coupling + assembled vs subclass A comparison

**Physical goal.** Exhibit stable, physically correct mode-coupling
energy exchange between two near-resonant g-modes over hundreds of
oscillation periods, and demonstrate that the matrix-free pointwise
surrogate of Proposition 1 (subclass A) — although not exponentially
unstable — produces a *silently wrong* energy-exchange spectrum.
This is the asteroseismology community's target regime (Kumar &
Goldreich 1989; Weinberg et al.\ 2012) and directly showcases
Proposition 1's norm-level gap at the level of an observable.

**Setup.**
- Lane-Emden $n = 3/2$, $\rho_{\mathrm{cut}} = 0.05$, $N_y = 64$, $N_x
  = 64$, $L_x = L_y = 1$.
- Pick two g-modes $(n_g^{(1)}, k_x^{(1)}) = (1, 2\pi/L_x)$ and
  $(n_g^{(2)}, k_x^{(2)}) = (3, 2\pi/L_x)$ that form a *near-resonant*
  pair — i.e. $3\omega_1 - \omega_3$ small compared to either
  individual frequency.  Verify the specific pair via the EVP
  spectrum at setup time; if the default $(1, 3)$ pair is too
  off-resonant, fall back to $(1, 2)$ or search the EVP output.
- IC: equal-amplitude superposition, each at amp $= 5\times 10^{-3}$.
- Strang-split integrator (Path 1 of §7) with assembled $\mathsf M$
  linear half-step + RK4 nonlinear full-step, 2/3 dealias on $x$.
- Run $300$ periods (of the slower mode), $\Delta t$ targeting CFL
  $\le 0.5$ on the fastest retained g-mode.

**Critical design: A/B swap on the same framework.**  Run the *same*
simulation twice, differing only in how `apply_M` is implemented:
- *Run 1 (assembled)*: $\mathsf M = \mathsf L^{-1}\mathsf R$ per
  wavenumber, precomputed at setup.
- *Run 2 (subclass A)*: $\mathsf M = \mathrm{diag}(k_x^2 N^2)$
  pointwise (the Proposition 1 surrogate).
Every other detail — IC, $\Delta t$, dealias, pressure projection,
nonlinear block — is bitwise identical.  Subclass A is spectrally
bounded (Table 5.4) so it will *not* blow up; the failure mode it
exhibits is a **silent wrong answer**, which is the more compelling
criticism of the matrix-free default.

**Diagnostics.**
- Modal energy $E_k(t)$ for $k \in \{1, 2, 3, 4, 5, 6\}\cdot 2\pi/L_x$,
  sampled every period via horizontal FFT + projection onto EVP
  eigenvectors.
- Total energy $E(t) = \tfrac12 \int \rho_0(u^2 + v^2)\,dy\,dx + \int
  (b^2/(2N^2))\,dy\,dx$.
- Eigenmode deviation on the two primary $k$-components.
- Spectrum plot $E_k$ vs $t$ side-by-side (assembled | subclass A).

**Pass criteria (assembled run).**
- $|\Delta E/E(0)| < 10^{-10}$ over 300 periods.
- Clean two-mode beating pattern between primary and daughter
  harmonics; exchange period matches three-wave theory to $\sim 10\%$.

**Expected failure mode (subclass A run).**
- Total energy *does* drift (no physical conservation structure in
  the surrogate).
- Modal energy exchange pattern differs measurably from Run 1:
  either wrong exchange period, wrong equilibrium distribution, or
  slow monotone drift in modes that should beat.  The numerical
  value of the discrepancy becomes the quantitative criticism.

**Infrastructure.**
- Python reference: `scripts/nonlinear_path1_opsplit.py` has
  Strang-split + `apply_M` via `M_list` (assembled).  For subclass A
  replace with `M_list_surrogate[k] = kx[k]**2 * diag(N2)`.  Add
  modal-projection diagnostics hook.
- GPU: `src/gpu/anelastic_sl_solver.{cu,cuh}` — **verify nonlinear
  advection block is wired, and add a `--operator-mode {assembled,
  surrogate}` runtime flag** so the A/B swap is controlled by a
  single CLI argument on the same binary.

**New code.** ~80 lines runner + modal-projection utility + 2-panel
plot.  Output: `paper/figures/fig7_1_mode_coupling.png`.

**Cost estimate.** Python ~60 min/300 periods @ $64^2$ per run × 2
runs.  GPU ~10 min × 2 runs.

---

### Experiment B — Parametric subharmonic instability (PSI) with Lamb-Bretherton growth rate

**Physical goal.** A parent g-mode of frequency $\omega_0$ drives a
pair of daughter modes $(\omega_+, \omega_-)$ with $\omega_+ + \omega_-
\approx \omega_0$, exciting the daughters out of noise at an
analytically known growth rate.  This is the classical PSI geometry
[Lamb & Bretherton 1988, *J. Fluid Mech.* 187]; comparing the
numerical $\gamma$ to the Lamb-Bretherton prediction provides a
*quantitative* benchmark, not merely a plausibility check.

**Setup.**
- Same background and resolution as A.
- Pre-compute the EVP spectrum at $N_y = 96$; scan $(n_g, k_x)$ for a
  triad $(\omega_0, \omega_+, \omega_-)$ with $|\omega_+ + \omega_- -
  \omega_0|/\omega_0 \lesssim 0.02$ and a non-negligible coupling
  integral $I = \int \rho_0 V_0\,(V_+\cdot\nabla)V_-\,dy$.
- IC: parent at amp $a_0 = 10^{-2}$; daughter pair at seed amp
  $10^{-6}$ each (well above round-off floor, well below nonlinear
  crosstalk).
- Integrate until daughter pair saturates (typically 50-200 parent
  periods) or 500 parent periods, whichever first.

**Analytical prediction.**  Lamb-Bretherton-type three-wave theory
gives the linear growth rate of the daughter pair,
$$
\gamma_{\mathrm{LB}} = \tfrac14\, a_0\, \omega_0\, |I| \cdot
(\text{detuning factor}).
$$
Exact prefactor depends on the normalisation convention used for
$V_n$; we adopt the $\int \rho_0 V_n^2 \,dy = 1$ normalisation and
compute the coupling integral directly from the EVP eigenvectors.

**Diagnostics.**
- $\log E_\pm(t)$ vs $t$ — expect linear growth at rate
  $\gamma_{\mathrm{meas}}$ followed by saturation.
- Saturation amplitude vs parent amp $a_0$, sweeping $a_0 \in
  \{10^{-2}, 10^{-2.5}, 10^{-3}\}$.

**Pass criteria.**
- $|\gamma_{\mathrm{meas}} - \gamma_{\mathrm{LB}}|/\gamma_{\mathrm{LB}}
  \lesssim 10\%$ at the largest $a_0$ (higher signal-to-noise).
- Detuning sensitivity: vary the resonance mismatch, reproduce the
  Lorentzian-in-detuning shape of $\gamma_{\mathrm{meas}}(\Delta
  \omega)$.
- Post-saturation: no exponential blow-up (differentiates from
  §5.7.2 reduced-operator pathology).

**Infrastructure.** Same as A, plus:
- `scripts/scan_resonance.py` — EVP spectrum scan, triad search,
  coupling-integral calculator.  One-off utility, ~100 lines.
- GPU: the `--operator-mode assembled` path; subclass A is not needed
  here (the experiment is about fidelity of the *correct* scheme,
  not a comparison).

**New code.** ~150 lines (triad scan + runner + detuning sweep +
figure).  Output: `paper/figures/fig7_2_psi.png` with two panels:
(a) $\log E_\pm$ vs $t$ with analytical slope overlay; (b)
$\gamma_{\mathrm{meas}}$ vs $\Delta\omega$ Lorentzian.

**Cost estimate.** Scan + triad ID on Python (~10 min).  Main run
multi-hour on Python, **GPU strongly preferred**.  Detuning sweep: 5
runs × 100 periods each.

---

## Not selected (and why)

- **Broadband g-mode turbulence.** Reviewer would contest the
  physical meaning of 2D pseudo-anelastic turbulence.  Not worth
  the space.
- **Boussinesq ↔ Lane-Emden crossover.** No new information beyond
  §5.1 / §6.3 linear results.
- **§7 as-is (three-method comparison prototype).** The current
  three-method comparison is linear-adjacent and will be demoted to
  Appendix; the Strang-split path becomes the §7 production
  integrator carrying Experiments A and B.

---

## Paper integration

### Proposed §7 structure

1. §7.1 Problem statement — same opening, but ending with "we
   demonstrate the scheme in two nonlinear benchmarks that
   asteroseismology has identified as central".
2. §7.2 Strang-split integrator construction (inherits from current
   §7; compressed).
3. §7.3 **Experiment A — two-mode coupling, assembled vs surrogate
   comparison**.  Figure 7.1: modal energy vs time, two panels.  Key
   claim: the matrix-free surrogate produces a silently wrong mode
   spectrum that a typical benchmark (total energy, eigenmode
   deviation) would not catch.
4. §7.4 **Experiment B — PSI with Lamb-Bretherton growth rate**.
   Figure 7.2: $\log E_\pm(t)$ + detuning Lorentzian.  Key claim:
   quantitative agreement with three-wave theory to $\sim 10\%$.
5. §7.5 Brief summary: scheme carries asteroseismology-relevant
   nonlinear physics to quantitative accuracy without losing
   Proposition 2's linear floor.
- Appendix A: IMEX / exponential prototype comparison (demoted).

### Abstract addition

One sentence after the GPU benchmark line:

> "On two nonlinear benchmarks drawn from asteroseismology — two-mode
> near-resonant coupling and parametric subharmonic instability on
> Lane-Emden $n = 3/2$ — the Strang-split extension recovers the
> Lamb-Bretherton analytical growth rate to within $10\%$ and exposes
> a silent energy-transfer error in the matrix-free pointwise
> surrogate that conventional conservation diagnostics would not
> catch."

### §1.3 contributions add

> 5. **Two nonlinear benchmarks** (Section 7): two-mode coupling and
>    parametric subharmonic instability, demonstrating (i)
>    quantitative agreement with Lamb-Bretherton three-wave theory to
>    $\sim 10\%$ and (ii) a measurable silent failure mode of the
>    matrix-free pointwise surrogate on energy-transfer spectra.

---

## Host-switch handoff

### What's ready on this host (laptop)

- Python infra: `scripts/nonlinear_path1_opsplit.py` +
  `nonlinear_paths_infra.py` + `nonlinear_paths_compare.py`.
- Paper: all §1-§9 consolidated, PDF at `paper/paper.pdf` (26 pages).
- Review artefacts: `scripts/review_e{1,2,3,4}_*.py` for Proposition
  1 verification experiments.

### What needs verification on GPU host

1. `src/gpu/anelastic_sl_solver.cu` — does the current code path
   include the nonlinear advection block?  Grep for
   `compute_advection`, `dealias_23`, or `nonlinear` inside the
   solver.
2. IFRK3 vs Strang — the GPU solver uses IFRK3 for the linear block;
   verify this matches the Python Strang-split semantics in the
   amp → 0 limit (should, since IFRK3 on assembled $\mathsf M$
   recovers the same linear propagator).
3. Output cadence — ensure diagnostics every period, not every step,
   to keep VRAM frame buffer manageable over 300 periods.
4. **A/B operator swap**: add a runtime flag
   `--operator-mode {assembled, surrogate}` so the same binary runs
   both Experiment A variants.  `surrogate` replaces the
   precomputed $\mathsf M_k$ with `diag(kx^2 * N^2)`.

### Execution order (suggested)

1. Python: verify Experiment A pipeline with 20-period run, both
   assembled and surrogate paths.
2. Port A to GPU with `--operator-mode` flag; match linear limit
   against Python to machine precision.
3. Run Experiment A at production resolution + full horizon, both
   variants, on GPU.
4. Write `scan_resonance.py`, identify $(\omega_0, \omega_+,
   \omega_-)$ triad + coupling integral.
5. Run Experiment B on GPU only.  Detuning sweep in parallel if VRAM
   allows.
6. Generate figures `fig7_1_mode_coupling.png`,
   `fig7_2_psi.png`.
7. Rewrite §7 per structure above.
8. Update abstract, §1.3 contributions, §9 conclusions.
9. Rebuild PDF.

### Files likely to be edited in next round

- `paper/07_nonlinear.md` — rewrite.
- `paper/01_intro.md` — §1.3 add contribution bullet 5.
- `paper/09_conclusions.md` — add nonlinear benchmark line.
- `paper/paper-cas.tex` — abstract one-sentence addition.
- `paper/11_appendix_a.md` — move current §7.3 prototype comparison
  here.
- `scripts/nonlinear_path1_opsplit.py` — add modal projection +
  `--operator-mode` toggle.
- `scripts/dns_a_mode_coupling.py` — new runner.
- `scripts/dns_b_psi.py` — new runner.
- `scripts/scan_resonance.py` — new EVP utility (triad search +
  coupling integral).

### Citations to add (to 10_refs.md)

- Kumar & Goldreich 1989, *ApJ* 342 — mode coupling in stars
- Weinberg et al.\ 2012, *ApJ* 751 — nonlinear tidal couplings
- Lamb & Bretherton 1988, *J. Fluid Mech.* 187 — PSI analytical
  framework

---

*Last updated: session ending 2026-05-04 on laptop; next session on
GPU host.  Experiments A and B upgraded from the first draft
following the reviewer feedback that asteroseismology-relevant
physics (mode coupling, PSI) and quantitative theory comparison
(Lamb-Bretherton) make far stronger showcases than generic energy
conservation.*
