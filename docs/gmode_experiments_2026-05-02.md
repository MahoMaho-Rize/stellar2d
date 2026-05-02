---
title: |
  g-mode Infrastructure and First Validation Experiments
author: |
  Technical Report, stellar2d project, `anelastic-sl-spectral` branch
date: 2 May 2026
---

# 0. Purpose

This note documents the initial g-mode (internal gravity wave) infrastructure
in this repo and two first-day validation experiments that demonstrate the
pipeline is numerically sound before any integration with the anelastic
solver.

- **Experiment A.** Lane-Emden $n = 3/2$ polytrope with a $\widetilde W$-derived
  $N^2_\text{proxy}$.  This is a pipeline sanity check only: Lane-Emden is
  isentropic and supports no real g-modes, but the Cowling solver should
  still reproduce the Tassoul (1980) asymptotic period spacing
  $\Delta P_\infty$ when fed any positive $N^2(r)$.
- **Experiment B.** Artificial Gaussian-bump $N^2(r) > 0$ on a cleanly
  bounded cavity.  This is the first real g-mode test in the repo: as the
  radial resolution $N_r$ grows, the last-5-modes-mean $\Delta P$ should
  converge to the Tassoul integral.

All numeric code lives in `scripts/gmode_infra.py` (shared), with two
experiment drivers `scripts/gmode_exp_a_lane_emden.py` and
`scripts/gmode_exp_b_stratified.py`.  The binding protocol (every numeric
value in this document has `EXPECTED` constants and a `--verify` path in the
corresponding script) matches the one established in
`docs/reduced_pressure_experiments_2026-05-02.md`.

## Reproducibility protocol

Every table in §§2–3 is traceable to one script.  Each script:

1. Prints a **provenance banner** on startup (script path + git HEAD + date).
2. Embeds its published reference values as an `EXPECTED` constant.
3. Accepts `--verify` to compare a fresh run against `EXPECTED` within a
   stated relative tolerance and exit nonzero on drift.

Standard usage:

```bash
python scripts/gmode_exp_a_lane_emden.py --verify
python scripts/gmode_exp_b_stratified.py --verify
```


# 1. Infrastructure — `scripts/gmode_infra.py`

Shared module used by both experiments.  Key entry points:

| Function | Purpose |
|---|---|
| `solve_lane_emden(n)` | Emden function $\theta(\xi)$ via `scipy.solve_ivp`; used for $\rho_0$. |
| `compute_W_original` / `compute_W_reduced` | Liouville potentials; sign-consistent with `reduced_pressure_*.py`. |
| `solve_sl_eigenpairs(y, W, n_modes)` | Standard Sturm-Liouville: $(-d^2/dy^2 - W)\psi = \mu\psi$ on a uniform grid. |
| `solve_gmode_cowling(r, N2, ell, n_modes)` | **Generalised** eigenproblem $-\psi'' = \omega^{-2} \cdot \ell(\ell+1)\,N^2(r)/r^2 \cdot \psi$; returns ascending-$n$ modes (so index 0 = highest frequency, $n = 1$). |
| `tassoul_dP(r, N2, ell)` | $\Delta P_\infty = 2\pi^2 / [\sqrt{\ell(\ell+1)} \int N/r \, dr]$. |

### Design notes

- **Ordering gotcha (fixed in infra).**  The generalised eigenproblem has
  $\lambda = 1/\omega^2$; the $n = 1$ (lowest radial order, highest frequency)
  g-mode has the **smallest** $\lambda$, and $\omega \to 0$ as $n \to \infty$.
  The solver sorts by ascending $\lambda$ to return $n = 1, 2, 3, \ldots$.
  Early versions of this infra sorted the wrong way and returned a tail slice.
- **Cowling approximation.** The solver neglects the perturbation of
  self-gravity, which is standard for high-$n$ g-modes in stellar oscillation
  theory (Unno et al. 1989 §14).  Lifting this approximation requires the
  full fourth-order ADIPLS / GYRE structure and is out of scope for a
  validation infrastructure.
- **Limitations.** Uniform-grid FD only; Dirichlet BCs; Cartesian / spherical
  distinction is via the $\ell(\ell+1)/r^2$ centrifugal factor.  High-$n$
  modes and sharp $N^2$ gradients will eventually need Chebyshev collocation
  (see `reduced_pressure_chebyshev.py` for the template).


# 2. Experiment A — Lane-Emden $\widetilde W$-proxy heuristic

> **Provenance.** All numbers in this section were produced by
> `scripts/gmode_exp_a_lane_emden.py` at commit `8aa3476`.
> `python scripts/gmode_exp_a_lane_emden.py --verify` must exit zero.

## 2.1 Setup

Lane-Emden $n = 3/2$, density cutoff $\rho_\text{cut} = 0.05$, cavity
$r \in [0.15, 0.844]$ (core region excluded to avoid the $1/r^2$ centrifugal
blow-up, surface boundary layer excluded where `np.gradient` on $\widetilde W$
loses accuracy).  Proxy Brunt frequency $N^2_\text{proxy}(r) \equiv -\widetilde W(r)$,
masked to the positive region.

$\ell = 1$, 30 radial orders computed, last-5 modes averaged for the Tassoul
comparison.

## 2.2 Result

| quantity | value |
|---|---|
| $\Delta P_\text{Tassoul}$ | $137.6316$ |
| $\Delta P_\text{tail (n=26..30)}$ | $117.2542$ |
| ratio | $\mathbf{0.852}$ |
| pipeline sanity | PASS (acceptance window $0.80$–$1.20$) |

At low $n$ the match is nearly perfect ($n = 1$: ratio $0.996$; $n = 2$: $0.999$;
$n = 5$: $0.996$), degrading smoothly toward $0.85$ at $n = 30$.  This is
expected: the 90-point g-mode cavity cannot resolve 30 radial orders
accurately — the tail deviation reflects finite-grid bias, not a pipeline
defect.

## 2.3 Interpretation

This experiment has **no physical content** for Lane-Emden itself (the true
$N^2 = 0$ for an isentropic polytrope).  Its value is as a pipeline smoke
test: the numerically-computed $\Delta P_n$ tracks the Tassoul asymptote
whenever any positive $N^2$ profile is supplied, verifying that
`solve_gmode_cowling` and `tassoul_dP` are mutually consistent.

The low-$n$ agreement and the monotone drift at high $n$ — driven by grid
resolution — are the expected signatures of a correct 2nd-order FD
eigensolver on an under-resolved cavity.


# 3. Experiment B — stratified layer g-modes

> **Provenance.** All numbers in this section were produced by
> `scripts/gmode_exp_b_stratified.py` at commit `8aa3476`.
> `python scripts/gmode_exp_b_stratified.py --verify` must exit zero.

## 3.1 Setup

Artificial Gaussian-bump $N^2(r)$ on $r \in [0.2, 1.0]$:

$$N^2(r) \;=\; \exp\!\left[-\left(\tfrac{r - r_c}{\sigma}\right)^{\!2}\right] \cdot \sin^2\!\left(\tfrac{\pi(r - r_\text{lo})}{r_\text{hi} - r_\text{lo}}\right),$$

with $r_c = 0.6$, $\sigma = 0.2$.  The $\sin^2$ taper ensures $N^2 \to 0$
smoothly at the Dirichlet endpoints.  $\ell = 1$, 40 radial orders, last 5
averaged.  Resolution sweep $N_r \in \{256, 512, 1024, 2048\}$.

## 3.2 Result

| $N_r$ | $\Delta P_\text{Tassoul}$ | $\Delta P_\text{tail}$ | ratio | $|\text{ratio} - 1|$ |
|---|---|---|---|---|
| 256 | $20.993338$ | $19.523643$ | $0.9300$ | $7.00 \times 10^{-2}$ |
| 512 | $20.993276$ | $20.638170$ | $0.9831$ | $1.69 \times 10^{-2}$ |
| 1024 | $20.993261$ | $20.910023$ | $0.9960$ | $3.96 \times 10^{-3}$ |
| **2048** | $20.993257$ | $20.977525$ | $\mathbf{0.9993}$ | $\mathbf{7.49 \times 10^{-4}}$ |

The convergence rate is consistent with the 2nd-order FD eigensolver: each
doubling of $N_r$ roughly quarters the error, matching the expected
$\mathcal{O}(N_r^{-2})$ scaling.

## 3.3 Interpretation

This is **the first demonstrated real g-mode calculation in this repo**.  The
stratified layer hosts a genuine g-mode cavity (positive $N^2$), and the
numerical eigenvalues approach the Tassoul asymptote to within $0.07\%$ at
$N_r = 2048$ — a passing physical result, not a heuristic.

The infrastructure is therefore ready for:

1. **MESA profile ingestion**: replace `n2_profile()` with a radial profile
   from a real stellar model; the same Cowling solver applies.
2. **Multi-cavity / radiative-convective boundary**: piecewise $N^2(r)$ with
   evanescent regions will introduce mixed-mode / avoided-crossing
   behaviour; the generalised eigenproblem handles this natively.
3. **Coupling to the reduced-pressure SL Poisson solver** (parent doc
   `reduced_pressure_liouville.md` §6.2): the SL basis $\{\psi_n\}$ computed
   for the anelastic pressure solve and the g-mode basis use the *same*
   underlying eigenproblem structure (different potentials); code can share
   Cholesky factorisation and Chebyshev matrices.

## 3.4 What this experiment does **not** prove

- No coupling to the full anelastic momentum equation.  The Cowling
  simplification in `solve_gmode_cowling` isolates the radial oscillation
  structure but ignores back-reaction on gravity.
- No horizontal resolution.  The $\ell(\ell+1)/r^2$ term captures the
  horizontal angular dependence but a full 2D / 3D simulation is a separate
  task.
- No time integration.  The eigenvalues give oscillation frequencies and
  period spacings; actually *propagating* a g-mode in the anelastic solver
  is the eventual follow-up.

These are items for the next development cycle.


# 5. Experiment C — Chebyshev collocation g-mode solver

> **Provenance.** All numbers in this section were produced by
> `scripts/gmode_exp_c_chebyshev.py` at commit `e703991`.
> `python scripts/gmode_exp_c_chebyshev.py --verify` must exit zero.

## 5.1 Setup

Identical Gaussian-bump $N^2(r)$, $\ell = 1$, and cavity as Experiment B; the
only change is the discretisation.  The Cowling generalised eigenproblem

$$-\psi'' = \omega^{-2}\,\ell(\ell+1)\,N^2(r)/r^2 \cdot \psi$$

is now solved via `gmode_infra.solve_gmode_cowling_cheb` on CGL nodes with
the Trefethen $D^2$ matrix.  The Chebyshev helpers (`cheb`,
`clenshaw_curtis_weights`, `cheb_on_interval`) are hoisted from
`scripts/reduced_pressure_chebyshev.py` into `scripts/gmode_infra.py`, so the
reduced-pressure and g-mode suites now share a single spectral core.

### Spurious-mode guard

Chebyshev generalised eigensolvers contaminate the last $\sim N_\text{Cheb}/4$
modes with high-frequency states whose eigenfunctions have sub-grid
wavelength — returning 40 modes at $N_\text{Cheb} = 64$ produces a chaotic
tail with $\Delta P$ values in the hundreds.  The experiment caps
`n_modes = max(10, N_cheb // 5)` so the last-5-modes tail average always
lies inside the converged band.

## 5.2 Result

| $N_\text{Cheb}$ | modes used | $\Delta P_\text{Tassoul}$ | $\Delta P_\text{tail}$ | ratio | $|\text{ratio} - 1|$ |
|---|---|---|---|---|---|
| 64 | 12 | $21.00169$ | $21.04450$ | $1.00204$ | $2.04 \times 10^{-3}$ |
| 128 | 25 | $20.99536$ | $21.00747$ | $1.00058$ | $5.77 \times 10^{-4}$ |
| 256 | 51 | $20.99378$ | $20.99785$ | $1.00019$ | $1.94 \times 10^{-4}$ |
| **512** | 102 | $20.99339$ | $20.99483$ | $\mathbf{1.00007}$ | $\mathbf{6.85 \times 10^{-5}}$ |

## 5.3 Interpretation

Compared to Experiment B (FD, $N_r$ sweep):

- $N_\text{Cheb} = 64$ already matches $N_r = 1024$ FD accuracy ($\sim 4\times 10^{-3}$
  in B vs $2\times 10^{-3}$ here) — a $\mathbf{16\times}$ reduction in grid
  size for comparable error.
- $N_\text{Cheb} = 512$ beats $N_r = 2048$ FD by an order of magnitude
  ($7 \times 10^{-5}$ vs $7.5 \times 10^{-4}$).
- The convergence is **not exponential** (the spurious-mode band limits the
  highest resolved $n$); instead it appears to behave roughly as
  $N_\text{Cheb}^{-1}$ at the scales we tested, dominated by the
  discretisation of the $1/r^2$ weight rather than the $D^2$ operator itself.
  Pushing past $10^{-5}$ accuracy likely needs either a refined mode cap or
  a Cholesky-based generalised eigensolver robust to ill-conditioned $B$.

For the purposes of linear-stability analysis and cross-validation of the
anelastic momentum solver, $10^{-4}$ accuracy on $\omega^2$ is more than
sufficient; the infrastructure is ready to move on.


# 6. Experiment D — polytropic profile through a MESA-style parser

> **Provenance.** All numbers in this section were produced by
> `scripts/gmode_exp_d_polytrope_profile.py` at commit `<to be updated>`.
> `python scripts/gmode_exp_d_polytrope_profile.py --verify` must exit
> zero.  The polytropic fixture file (`videos/polytrope_fixture.dat`, ~30 KB)
> is regenerated deterministically at the top of `main()` so does not need
> to be tracked in git.

## 6.1 Setup

A thin column-table reader `scripts/mesa_profile.py` consumes a MESA-style
text file with columns `(r, rho, N², cs²)`.  The companion helper
`build_polytrope_fixture()` writes such a file from scratch: a Lane-Emden
$n = 3$ polytrope density profile plus the *same* Gaussian-bump $N^2(r)$
used in Experiments B and C, imposed rather than derived thermodynamically.

This is the minimal integration test for the future MESA reader: the
Cowling solvers consume only $(r, N^2)$, so feeding the fixture through the
parser must reproduce the Exp B/C numbers up to small interpolation
artifacts introduced by resampling the 600-row table onto the uniform FD or
CGL grid.

## 6.2 Result

| Suite | $\Delta P_\text{Tassoul}$ | $\Delta P_\text{tail}$ | ratio | $|\text{ratio} - 1|$ |
|---|---|---|---|---|
| FD, $N_r = 1024$ | $20.992871$ | $20.910504$ | $0.996076$ | $3.92 \times 10^{-3}$ |
| FD, $N_r = 2048$ | $20.992860$ | $20.977810$ | $0.999283$ | $7.17 \times 10^{-4}$ |
| Chebyshev, $N = 256$ | $20.993408$ | $20.997854$ | $1.000212$ | $2.12 \times 10^{-4}$ |
| **Chebyshev, $N = 512$** | $20.993003$ | $20.994706$ | $\mathbf{1.000081}$ | $\mathbf{8.11 \times 10^{-5}}$ |

Against the in-memory Exp B/C numbers:

| Suite | Exp D (file I/O) | Exp B/C (in-memory) | drift |
|---|---|---|---|
| FD, $N_r = 1024$ | $3.92 \times 10^{-3}$ | $3.96 \times 10^{-3}$ | $1.0\%$ |
| FD, $N_r = 2048$ | $7.17 \times 10^{-4}$ | $7.49 \times 10^{-4}$ | $4.3\%$ |
| Chebyshev, $N = 256$ | $2.12 \times 10^{-4}$ | $1.94 \times 10^{-4}$ | $9.3\%$ |
| Chebyshev, $N = 512$ | $8.11 \times 10^{-5}$ | $6.85 \times 10^{-5}$ | $18.4\%$ |

The drift grows with resolution because the interpolation error floor is
fixed by the 600-row fixture, so as the solver becomes more accurate the
relative contribution of the resampling error grows.  Absolute error
remains well below $10^{-3}$ in all cases — good enough for a file-I/O
sanity test.  A future MESA reader fed with higher-resolution real stellar
profiles should see the drift shrink.

## 6.3 Interpretation

The parser path is clean: the Cowling solvers accept file-loaded $N^2(r)$
and produce the expected Tassoul asymptote, with error dominated by the
fixture's 600-row resolution rather than anything in `mesa_profile.py`
itself.  This unblocks swapping in a real MESA `profile*.data` file as soon
as an actual stellar model is available.


# 7. Next steps

1. ~~**Chebyshev upgrade**~~ — done in §5.
2. ~~**MESA-style profile reader**~~ — done in §6 with a polytropic fixture;
   swapping to real MESA `profile*.data` is now a one-line
   `read_profile(path)` call once a physical model is chosen.
3. **Avoided-crossing benchmark.** Standard astrophysics test case with
   piecewise-linear $N^2$ and a radiative-convective boundary; expected
   spectra available in Unno et al. (1989) figures 17-19.  This needs the
   Lamb frequency $L_\ell^2 = \ell(\ell+1)\,c_s^2/r^2$ as well as $N^2$ —
   i.e.\ it uses the `cs2` column that the parser already extracts but
   the current solvers ignore.  A mixed g/p-mode solver (fourth-order
   Dziembowski / GYRE structure) is out of scope for a validation
   infrastructure; a truncated version using only the g-mode branch in a
   single cavity is a reasonable first step.
4. **Integration with anelastic momentum solver.** The anelastic time
   integrator needs a linear-stability analysis tool; the $\omega^2$
   output of the Cowling solvers is the natural reference against which
   the linearised RHS eigenvalues should agree.
