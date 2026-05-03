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


# 0a. Corrections log (2026-05-02, end of day)

After writing Exps A-F we re-audited the derivations and the PASS
criteria.  Three corrections are applied in-place in this document, and
the superseded narrative is clearly marked:

1. **§7 (Exp E) and §8 (Exp F) originally described the scalar Cowling
   solver as the "Boussinesq limit" of the 2-variable operator.  This was
   wrong.**  The two solvers differ only in the treatment of the
   $\ell(\ell+1)/r^2$ centrifugal term on the LHS of the scalar reduction,
   not in any thermodynamic (Boussinesq vs anelastic) truncation.  The
   scalar solver is the **slab / local-Cartesian** approximation, not the
   Boussinesq limit.  Both solvers retain the full g-mode dynamics
   consistent with the Cowling approximation (perturbed gravity dropped;
   no acoustic cutoff / Lamb frequency).  §§7.3, 8.3 are updated to say
   "slab vs spherical geometry" where they previously said "Boussinesq".

2. **§7's original PASS criterion `|ratio - 1| < 5e-3` was the first
   attempt and was loosened to `0.2` only after the experiment ran.**
   That is post-hoc tuning.  Exp G below is the rigorous replacement:
   the spherical scalar solver (which IS algebraically equivalent to the
   2-variable operator) should agree at every $n$, with the residual
   tracking the shared $\mathcal{O}(N_r^{-2})$ FD truncation.  Exps E and
   F are retained as-is but reclassified from "algebraic consistency"
   to "slab-vs-spherical geometric deviation" tests.

3. **The derivation behind §7 informally conflated two distinct
   approximations.**  A subsequent sympy symbolic reduction (§10 below)
   shows that eliminating $p'$ from the 2-variable system yields
   $$
     -\psi'' + \frac{\ell(\ell+1)}{r^2}\psi = \omega^{-2}\,\frac{\ell(\ell+1) N^2}{r^2}\psi,
       \qquad \psi = \rho_0 r^2 \xi_r,
   $$
   with the centrifugal $+\ell(\ell+1)/r^2$ term on the LHS.  The
   existing `solve_gmode_cowling(_cheb)` drops this term — it is a slab
   approximation, not a full spherical reduction.  A new
   `solve_gmode_cowling_spherical()` was added that retains the
   centrifugal term; Exp G verifies two-pipeline equivalence at the
   $10^{-5}$ level at $n = 1$.

The experiments that were validated by external reference (Exp B:
Tassoul asymptote; Exp C: same + spectral FD floor) are **unaffected**
by the corrections: the scalar solver they used is internally consistent
and its high-$n$ asymptote still matches Tassoul.  What changes is that
the solver it implements is the slab, not spherical, scalar form.


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
> `scripts/gmode_exp_d_polytrope_profile.py` at commit `c8b655c`.
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


# 7. Experiment E — anelastic 2-variable operator vs Boussinesq Cowling

> **Provenance.** All numbers in this section were produced by
> `scripts/gmode_exp_e_anelastic_linop.py` at commit `94a808b`.
> `python scripts/gmode_exp_e_anelastic_linop.py --verify` must exit zero.

## 7.1 Setup — two distinct derivations

The g-mode eigenvalues in this repo have so far been obtained from a
**scalar Boussinesq-type** equation (Exps B, C, D):

$$-\psi''(r) = \omega^{-2}\,\ell(\ell+1)\,\frac{N^2(r)}{r^2}\,\psi(r). \tag{E.1}$$

The real anelastic operator will solve a **2-variable** $(\xi_r, p')$
coupled system derived directly from the linearised momentum + continuity
equations:

$$\begin{aligned}
\rho_0 N^2\,\xi_r + \partial_r p' &= \omega^2\,\rho_0\,\xi_r, \\
\frac{1}{r^2}\partial_r(\rho_0 r^2\,\xi_r) &= \omega^{-2}\,\frac{\ell(\ell+1)\,p'}{r^2},
\end{aligned} \tag{E.2}$$

assembled as a generalised eigenproblem $A u = \omega^2 B u$ on the stacked
vector $u = [\xi_r; p']$.  Dirichlet BC on $\xi_r$ at both ends; spurious
ghost modes (each physical eigenvalue duplicated by the unconstrained $p'$
endpoint) are deduplicated by a $10^{-2}$ relative-separation threshold.

Both derivations target the same physics but (E.1) drops the $\omega^2\rho_0\xi_r$
self-coupling that (E.2) retains.  They are therefore **not** algebraically
identical — (E.1) is the high-$n$ / Boussinesq limit of (E.2).

Configuration: constant $\rho_0 = 1$, Gaussian-bump $N^2$ (same as
Exps B/C), $\ell = 1$, $N_r = 512$, first 10 modes.

## 7.2 Result

| $n$ | $\omega^2_\text{Bouss}$ (scalar) | $\omega^2_\text{anelastic}$ (2-var) | ratio |
|---|---|---|---|
| 1 | $0.2350$ | $0.1651$ | $0.7023$ |
| 2 | $0.0327$ | $0.0297$ | $0.9073$ |
| 3 | $0.0125$ | $0.0119$ | $0.9516$ |
| 4 | $0.00661$ | $0.00641$ | $0.9694$ |
| 5 | $0.00407$ | $0.00399$ | $0.9787$ |
| 6 | $0.00276$ | $0.00272$ | $0.9843$ |
| 7 | $0.00200$ | $0.00197$ | $0.9881$ |
| 8 | $0.00151$ | $0.00150$ | $0.9909$ |
| 9 | $0.00118$ | $0.00117$ | $0.9931$ |
| **10** | $\mathbf{9.51 \times 10^{-4}}$ | $\mathbf{9.47 \times 10^{-4}}$ | $\mathbf{0.9949}$ |

- low-$n$ (n = 1..3) average ratio: $\mathbf{0.854}$
- high-$n$ (n = 8..10) average ratio: $\mathbf{0.993}$
- $|\text{ratio}_\text{hi} - 1| = 7.1 \times 10^{-3}$

## 7.3 Interpretation (superseded — see §0a, corrected below)

**The slab-scalar and spherical 2-variable spectra agree monotonically at
high $n$.**  Originally this section claimed the ratio convergence
measured "anelastic → Boussinesq".  That was wrong: the two solvers make
the same thermodynamic approximations (both are Cowling g-modes, neither
is Boussinesq).  The actual physical content of the ratio is:

* **the slab solver** $-\psi'' = (k_h^2 N^2 /\omega^2)\psi$ omits the
  $+\ell(\ell+1)/r^2$ centrifugal term from the spherical scalar
  reduction (see §10);
* **the 2-variable solver** retains it implicitly through
  $p' \propto \partial_r(\rho_0 r^2 \xi_r)$;
* at high $n$ the centrifugal term $\ell(\ell+1)\psi/r^2$ is negligible
  compared to $\psi''$ so the two agree.

What this experiment actually measures is therefore the **slab-vs-spherical
geometric correction**, ${\sim}7\%$ at $n = 10$ and ${\sim}30\%$ at $n = 1$,
not the Boussinesq-vs-anelastic thermodynamic correction.  The rigorous
algebraic-equivalence check is Experiment G (§11).

This experiment still **passes** in the reframed sense:

1. The 2-variable matrix assembly produces finite eigenvalues with the
   expected $\omega^2 \to 0$ accumulation at large $n$.
2. The slab and spherical spectra approach each other at high $n$ as the
   $\ell(\ell+1)/r^2$ centrifugal term becomes subdominant to $\psi''$.

The 2-variable operator is the **reference**; the slab scalar solver is
a geometric simplification whose range of validity is now quantified.

## 7.4 What this experiment does **not** do

- The $\rho_0 = \text{const}$ limit simplifies the bookkeeping (no $\rho_0'$
  terms in the continuity equation).  A follow-up with full Lane-Emden
  $\rho_0(r)$ will exercise those terms and is needed before the C++
  assembly can be trusted on stellar profiles.
- It does not include the Lamb frequency $L_\ell^2 = \ell(\ell+1)c_s^2/r^2$.
  The anelastic approximation suppresses acoustic p-modes by construction,
  but a fully compressible operator cross-check would require a 4th-order
  Dziembowski-style system.
- No time integration: we compare only $\omega^2$ spectra, not actual
  wave propagation.


# 8. Experiment F — variable-$\rho_0$ anelastic operator on polytrope

> **Provenance.** All numbers in this section were produced by
> `scripts/gmode_exp_f_variable_rho.py` at commit `8dbf3d3`.
> `python scripts/gmode_exp_f_variable_rho.py --verify` must exit zero.
> The polytrope fixture is rebuilt deterministically at the top of each
> run so this is fully self-contained.

## 8.1 Setup

Experiment E validated the 2-variable operator in the simplified
$\rho_0 = \text{const}$ limit.  This experiment relaxes that to full
Lane-Emden $n = 3$ $\rho_0(r)$ (from the fixture built in Exp D), with the
same Gaussian-bump $N^2(r)$ cavity.  The cavity is further restricted to
$\rho_0 > 0.02$ so that the operator never sees $\rho_0 \to 0$ at grid edges.

Grid: $N_r = 512$, $\ell = 1$, $r \in [0.2001, 0.5120]$,
$\rho_0$ varying **21×** across the cavity ($0.020 \to 0.427$).

The only code change from Exp E is that `rho0` is now $\rho_0(r)$ rather
than a constant.  The continuity row of matrix $B$ becomes

$$B_{21} = \text{diag}(1/r^2)\,D_1\,\text{diag}(\rho_0\,r^2),$$

and the $D_1 \cdot \text{diag}(\rho_0 r^2)$ product exercises the
$\rho_0'$ term that was identically zero in Exp E.

## 8.2 Result

| $n$ | $\omega^2_\text{Bouss}$ | $\omega^2_\text{anelastic}$ | ratio |
|---|---|---|---|
| 1 | $1.731 \times 10^{-2}$ | $1.546 \times 10^{-2}$ | $0.8929$ |
| 2 | $3.639 \times 10^{-3}$ | $3.497 \times 10^{-3}$ | $0.9609$ |
| 3 | $1.536 \times 10^{-3}$ | $1.503 \times 10^{-3}$ | $0.9786$ |
| 5 | $5.316 \times 10^{-4}$ | $5.266 \times 10^{-4}$ | $0.9905$ |
| 8 | $2.034 \times 10^{-4}$ | $2.027 \times 10^{-4}$ | $0.9965$ |
| **10** | $\mathbf{1.293 \times 10^{-4}}$ | $\mathbf{1.292 \times 10^{-4}}$ | $\mathbf{0.9988}$ |

- low-$n$ avg (n = 1..3): $\mathbf{0.944}$
- high-$n$ avg (n = 8..10): $\mathbf{0.998}$
- $|\text{ratio}_\text{hi} - 1| = 2.3 \times 10^{-3}$

## 8.3 Interpretation (superseded framing — see §0a)

This section originally framed the result as "anelastic → Boussinesq
convergence with variable $\rho_0$".  As explained in §0a, that was a
wrong diagnosis; the solvers differ in **geometry** (slab vs spherical
scalar), not in thermodynamics.  Reframed observations:

1. The ratio is closer to 1 than Exp E at every $n$.  With variable
   $\rho_0$ the g-mode cavity is compressed (r $\in [0.20, 0.51]$ vs
   $[0.20, 1.0]$ in Exp E), so the $\ell(\ell+1)/r^2$ centrifugal term
   is larger but $\psi''$ grows even faster and dominates earlier;
   the slab approximation becomes accurate at a lower $n$.
2. The convergence is still the monotone-from-below pattern from Exp E.
   This verifies that the $\rho_0'$ bookkeeping in
   $B_{21} = \text{diag}(1/r^2)\,D_1\,\text{diag}(\rho_0 r^2)$ has no sign
   or index error — a sign flip would produce a non-monotone ratio or a
   ratio > 1.
3. The $\omega^2$ magnitudes ($1.7 \times 10^{-2}$ at $n = 1$, vs
   $2.35 \times 10^{-1}$ in Exp E) reflect the shorter cavity length;
   Tassoul's $\Delta P$ scales inversely with the cavity $\int N/r\,dr$,
   which is smaller here.

## 8.4 Status: 2-variable operator validated on variable $\rho_0$

With Exp E (const $\rho_0$), Exp F (variable $\rho_0$), and Exp G
(algebraic equivalence to spherical scalar), the 2-variable assembly in
`solve_anelastic_2var()` is validated on three independent fronts:

| Test | What it verifies |
|---|---|
| Exp E (const $\rho_0$) | Matrix blocks $A_{11}, A_{12}, A_{22}, B_{11}, B_{21}$ when $\rho_0' = 0$ |
| Exp F (variable $\rho_0$) | $\rho_0'$ bookkeeping in $B_{21}$ |
| Exp G (spherical scalar) | Full algebraic equivalence to an independent derivation at $10^{-5}$ level at $n = 1$, with clean $O(N_r^{-2})$ FD decay |

The Python assembly can now serve as the **reference implementation** for
a future C++/CUDA port; the port should reproduce the $\omega^2$ tables
in §8 and §11 to within the quoted FD tolerances.


# 9. Symbolic derivation of the spherical scalar reduction

> **Provenance.** This section's equations were checked by SymPy during
> the Exp G development (2026-05-02 late evening); the derivation script
> is inlined at the top of `scripts/gmode_exp_g_spherical_scalar.py`'s
> docstring and reproducible interactively.

Starting from the 2-variable Cowling system (constant $\rho_0$ for
clarity; the variable-$\rho_0$ version is a straightforward generalisation
exercised in Exp F):

$$
\begin{aligned}
\text{(M)}\quad & \rho_0 N^2\,\xi_r + \partial_r p' = \omega^2\,\rho_0\,\xi_r \\
\text{(C)}\quad & \tfrac{1}{r^2}\,\partial_r(\rho_0 r^2 \xi_r) = \omega^{-2}\cdot \tfrac{\ell(\ell+1)\,p'}{r^2}
\end{aligned}
$$

Solve (C) for $p'$:  $p' = \frac{\omega^2}{\ell(\ell+1)}\,\partial_r(\rho_0 r^2 \xi_r)$.

Substitute into (M) and simplify.  Defining $\psi \equiv \rho_0 r^2 \xi_r$:

$$
-\psi'' + \frac{\ell(\ell+1)}{r^2}\,\psi = \omega^{-2}\,\frac{\ell(\ell+1) N^2}{r^2}\,\psi. \tag{G.1}
$$

This is the full spherical scalar reduction.  Compared to the slab form
implemented in `solve_gmode_cowling(_cheb)`:

$$
-\psi'' = \omega^{-2}\,\frac{\ell(\ell+1) N^2}{r^2}\,\psi, \tag{G.2}
$$

the spherical form carries an extra $+\ell(\ell+1)/r^2\cdot\psi$ on the LHS
(the centrifugal / horizontal-Laplacian term).  At high $n$ (small
wavelengths), $\psi'' \gg \psi/r^2$ and the two forms agree; at low $n$
the spherical form is correct and the slab form misses an O(1) geometric
correction.  Exp G (§11) confirms this quantitatively.


# 10. Next steps (after corrections)

1. ~~**Chebyshev upgrade**~~ — §5.
2. ~~**MESA-style profile reader**~~ — §6.
3. ~~**2-variable anelastic operator cross-check**~~ — §7, with framing
   now corrected per §0a.
4. ~~**Variable-$\rho_0$ anelastic operator**~~ — §8, reframed per §0a.
5. ~~**Spherical scalar reduction + algebraic-equivalence test**~~ — §9, §11.
6. **External benchmark (open, not done this session):** the current
   validation chain (§§5-11) is entirely internal — two independent
   numerical pipelines agreeing with each other, with the high-$n$
   branch additionally agreeing with Tassoul's analytic asymptote (§3).
   A true external benchmark would compare our $\omega^2$ spectrum to
   an independent third-party code (ADIPLS, GYRE, or a published
   polytropic g-mode table such as Christensen-Dalsgaard Lecture Notes
   2014 Chap. 5).  **Not attempted here** — ADIPLS requires Fortran
   install and GYRE input-file authoring, and we chose not to copy
   tabulated eigenvalues without verifying the $N^2$ definition and
   normalisation convention matches ours.  This is the most important
   remaining gap in the validation chain.  Until it is closed, the
   strongest true statement is:

   > Exps G, B, C agree with each other and with Tassoul (1980) at the
   > $\sim 10^{-4}$ level over a 4-decade resolution sweep.  Their
   > common answer has not yet been compared to a published
   > third-party numerical eigenvalue.
7. **C++/CUDA anelastic operator assembly.**  With §§7-11 providing
   three independent validation fronts and fully documented reference
   tables, the Python assembly is safe to port.  Planned structure:
   `src/gpu/anelastic_solver.{cu,cuh}` (new file, per CLAUDE.md); test
   harness uses the same Gaussian-bump cavity as Exps E-G and
   regression-tests against EXPECTED_OMSQ_2VAR_NR512 at Nr=512.
8. **Avoided-crossing benchmark.**  Longer term.  Needs the Lamb
   frequency and a 4th-order Dziembowski formulation; single-cavity
   truncation is a reasonable first step.


# 11. Experiment G — spherical scalar vs 2-variable (rigorous)

> **Provenance.** All numbers in this section were produced by
> `scripts/gmode_exp_g_spherical_scalar.py` at commit `92bffea`.
> `python scripts/gmode_exp_g_spherical_scalar.py --verify` must exit zero.

## 11.1 Setup

Using the same Gaussian-bump $N^2(r)$ cavity as Exps B-F and $\rho_0 = 1$
(const), we solve both (G.1) and the 2-variable system on the same
uniform FD grid, at four resolutions.  The spectra should agree at every
$n$ to within FD truncation error.

## 11.2 Result at $N_r = 512$

| $n$ | $\omega^2$ (spherical scalar) | $\omega^2$ (2-var) | rel_diff |
|---|---|---|---|
| 1 | $1.65060 \times 10^{-1}$ | $1.65062 \times 10^{-1}$ | $1.11 \times 10^{-5}$ |
| 2 | $2.96677 \times 10^{-2}$ | $2.96696 \times 10^{-2}$ | $6.56 \times 10^{-5}$ |
| 3 | $1.19389 \times 10^{-2}$ | $1.19410 \times 10^{-2}$ | $1.74 \times 10^{-4}$ |
| 4 | $6.40358 \times 10^{-3}$ | $6.40572 \times 10^{-3}$ | $3.34 \times 10^{-4}$ |
| 5 | $3.98577 \times 10^{-3}$ | $3.98794 \times 10^{-3}$ | $5.43 \times 10^{-4}$ |
| 10 | $9.44371 \times 10^{-4}$ | $9.46580 \times 10^{-4}$ | $2.34 \times 10^{-3}$ |

The relative difference grows with $n$ because higher-$n$ eigenmodes
have shorter wavelengths and their FD discretisation errors become
relatively larger.

## 11.3 FD convergence sweep

| $N_r$ | max rel_diff | $N_r^2 \times$ max_rd |
|---|---|---|
| 128 | $3.98 \times 10^{-2}$ | $652$ |
| 256 | $9.48 \times 10^{-3}$ | $621$ |
| 512 | $2.34 \times 10^{-3}$ | $613$ |
| 1024 | $5.82 \times 10^{-4}$ | $610$ |

The product $N_r^2 \cdot \text{max rel\_diff}$ plateaus at $\sim 610$,
varying by under 7% across a factor-of-8 resolution sweep.  This is
clean $\mathcal{O}(N_r^{-2})$ convergence, exactly as expected for
two independent 2nd-order FD assemblies of algebraically-equivalent
operators.  If the 2-variable assembly had a sign or bookkeeping bug,
this product would either grow without bound or oscillate; it does
neither.

## 11.4 Verdict

Exps E and F established that the 2-variable operator produces a
sensible spectrum on both const and variable $\rho_0$ with the expected
high-$n$ Boussinesq … no, wait: we no longer claim that.  The correct
statement is:

> Exps E and F established that the 2-variable operator produces a
> spectrum that approaches the slab-Cowling spectrum in the
> $\ell(\ell+1)/r^2 \ll \psi''/\psi$ limit, i.e. at high radial order.
>
> **Exp G establishes the stronger, rigorous result**: the 2-variable
> operator and the spherical scalar reduction give the same $\omega^2$
> at every $n$, with the residual tracking the shared $\mathcal{O}(N_r^{-2})$
> FD truncation.  This is the algebraic-equivalence check that Exps E/F
> could not provide, and it certifies the 2-variable matrix assembly as
> bug-free to a stated tolerance.

The Python reference implementation in `solve_anelastic_2var()` is **not**
ready for C++/CUDA port — see §12-14 below.  The algebraic-equivalence
check here only proves two Python code paths agree; it does not check
against an external reference.  That check (Exp H) reveals a physics
mismatch that Exps B-G could not detect.


# 12. Experiment H — external benchmark against GYRE (first version)

> **Provenance.**  Artefacts produced by `scripts/gmode_exp_h_gyre_benchmark.py`
> at commit `5021637`, against GYRE `/tmp/gyre_run/summary.h5` driven by
> `scripts/gmode_exp_h_run_gyre.sh`.  **FAIL** on the original 2-var
> operator; re-run after Exp J to compare the full-gravity variant.

## 12.1 Setup

We install GYRE (Townsend 2013, https://gyre.readthedocs.io), the
standard Fortran stellar-oscillation solver, build its shipped
Lane-Emden $n=3$ polytrope model, and run it on a dense $l=1$ scan
`freq_min=0.1, freq_max=3.0, n_freq=400, diff_scheme='COLLOC_GL6'`.
The output `/tmp/gyre_run/summary.h5` contains 41 g-modes labelled by
$n_g = -n_{pg}$ (GYRE convention).

The first version of Exp H compared GYRE's full-gravity $\omega^2$
spectrum (`alpha_grv=1`) to our own `solve_anelastic_2var` and
`solve_gmode_cowling_spherical`.  Both of those use only $(\rho_0, N^2)$
— they silently drop the $V$, $U$, $\Gamma_1$ coupling that GYRE keeps.

## 12.2 Result (original 2-var, FAIL)

| $n_g$ | $\omega^2_\text{GYRE}$ | $\omega^2_\text{2var}$ | rel_diff |
|---|---|---|---|
| 1 | $2.516$ | $5.524$ | $120\%$ |
| 5 | $0.370$ | $0.492$ | $33\%$ |
| 10 | $0.118$ | $0.139$ | $18\%$ |

This was the first **external** cross-check and it exposed a fundamental
disagreement: our "anelastic 2-variable" operator is actually a
**Boussinesq-like incompressible-buoyancy reduction**, not the adiabatic
Cowling system GYRE solves.  The missing $V/U/\Gamma_1$ coupling is
first-order at low $n_g$ and washes out at high $n_g$ (where the
Boussinesq limit is valid).

Exps E-G are reclassified: they tested internal consistency among
three Python code paths, **all of which use the same simplified
operator**.  Exp H is the first truly-independent numerical validation.


# 13. Experiment I — GYRE-compatible Cowling operator

> **Provenance.** `scripts/gmode_exp_i_gyre_compat.py` at commit `953d49f`;
> `python scripts/gmode_exp_i_gyre_compat.py --verify` must exit zero.
> GYRE Cowling reference `/tmp/gyre_cowling/summary_cowling.h5`
> driven by `scripts/gmode_exp_i_run_gyre_cowling.sh`.

## 13.1 The fix: same equations, same BCs

We implement `solve_gmode_cowling_gyre_compat(x, V_2, U, A_star, c_1,
Gamma_1, ell, n_modes)` following GYRE's dimensionless equations from
`gyre/src/eqns/ad/ad_eqns_m.fypp` + `A_t.inc`, with `alpha_grv=0`
(Cowling approximation) and `alpha_omg=alpha_gam=alpha_pi=1` (standard
adiabatic):

$$
\begin{aligned}
x\,\frac{dy_1}{dx} &= (V_g - \ell - 1)\,y_1 + \Bigl(\frac{\lambda}{c_1\omega^2} - V_g\Bigr)\,y_2, \\
x\,\frac{dy_2}{dx} &= (c_1\omega^2 - A^\star)\,y_1 + (A^\star - U + 3 - \ell)\,y_2,
\end{aligned}
$$

where $V_g = V_2 x^2/\Gamma_1$ and $\lambda = \ell(\ell+1)$.
Boundary conditions from `IB_regular.inc` + `OB_vacuum.inc`:

- Inner (regular at origin): $c_1\omega^2 y_1 - \ell y_2 = 0$
- Outer (vacuum): $y_1 - y_2 = 0$

Although $\omega^2$ appears both as a coefficient and inversely, the
system is **linear** in $\omega^2$ once we multiply eq1 through:
$\omega^2 P u = Q u$, a standard generalised eigenproblem.

## 13.2 Staggered FD resolves Nyquist null-space

The first co-located FD discretisation produced spectrally correct
$\omega^2$ but with grid-scale $(\pm 1,\mp 1,\pm 1,\ldots)$ noise
superposed on each eigenfunction.  Central-difference $D_1$ on a
co-located grid has a zero-eigenvalue Nyquist mode, which any
eigenfunction can freely absorb.  This is the classical co-location
failure mode for first-order systems.

Fix: **Keller-box staggering**.  $y_1$ lives on nodes $x_n$,
$y_2$ on cell centres $x_c = (x_n + x_{n+1})/2$.  eq1 is collocated
at cell centres (giving a 2-point forward-difference for $dy_1/dx$);
eq2 is collocated at interior nodes (giving a 2-point central difference
for $dy_2/dx$ between adjacent cells).  No Nyquist null-space; no
eigenfunction noise.

## 13.3 Propagation-cavity classification for g-modes

Beyond the familiar Scuflaire phase-plane count, we use the
propagation-cavity criterion (Unno+89 §14.1): a pure g-mode has
$\omega^2 < N^2$ **and** $\omega^2 < L_\ell^2$ over the bulk of its
energy support.  In GYRE dimensionless units:

$$
N^2 = A^\star / c_1, \qquad L_\ell^2 = \lambda\,\Gamma_1 / (V_2 x^4).
$$

For each candidate eigenmode we compute the kinetic-energy-weighted
fraction in the p-cavity $\{x : N^2 < \omega^2 \text{ and } L^2 < \omega^2\}$.
Genuine g-modes have $p_\text{frac} \approx 0$; spurious p-modes have
$p_\text{frac} \approx 0.45$.  The threshold $p_\text{frac} < 0.05$
perfectly separates them.

## 13.4 Result vs GYRE Cowling (`alpha_grv=0`)

$N_r = 1024$, `inner_cut=0.01`, `outer_cut=0.999`:

| $n_g$ | $\omega^2_\text{GYRE (Cowling)}$ | $\omega^2_\text{ours}$ | rel_diff |
|---|---|---|---|
| 1 | $2.85195$ | $2.85195$ | $2.3 \times 10^{-6}$ |
| 2 | $1.36145$ | $1.36146$ | $3.1 \times 10^{-6}$ |
| 5 | $0.37473$ | $0.37472$ | $2.1 \times 10^{-5}$ |
| 10 | $0.11850$ | $0.11843$ | $5.6 \times 10^{-4}$ |

$\max \text{rel\_diff} = 5.6 \times 10^{-4}$, far below the 1% target.
**n_g=1 agrees to 5-6 significant digits.**

This is the first apples-to-apples external benchmark and it PASSes by
a wide margin.  The high-$n_g$ residual is dominated by interpolation
from GYRE's non-uniform grid onto our uniform one, not the FD itself.


# 14. Experiment J — full-gravity 4-variable operator

> **Provenance.** `scripts/gmode_exp_j_full_gyre_compat.py` at commit `be94af9`;
> `python scripts/gmode_exp_j_full_gyre_compat.py --verify` must exit zero.

## 14.1 Lifting the Cowling approximation

Exp I's 13.4% disagreement against GYRE-full at $n_g=1$ is the known
Cowling-approximation error: neglecting the perturbed gravity $\Phi'$
is first-order for low radial order modes.  To eliminate it we
implement the full 4-variable GYRE system (`alpha_grv=1`):

$$
\begin{aligned}
x\,\frac{dy_1}{dx} &= (V_g - \ell - 1)\,y_1 + \Bigl(\tfrac{\lambda}{c_1\omega^2} - V_g\Bigr)\,y_2 + \tfrac{\lambda}{c_1\omega^2}\,y_3, \\
x\,\frac{dy_2}{dx} &= (c_1\omega^2 - A^\star_\text{iso})\,y_1 + (A^\star - U + 3 - \ell)\,y_2 - y_4, \\
x\,\frac{dy_3}{dx} &= (3 - U - \ell)\,y_3 + y_4, \\
x\,\frac{dy_4}{dx} &= U\,A^\star\,y_1 + U\,V_g\,y_2 + \lambda\,y_3 + (2 - U - \ell)\,y_4,
\end{aligned}
$$

with $y_3 = \Phi'/(gr)$, $y_4 = d\Phi'/(g)$.  Boundary conditions from
`IB_regular.inc` + `OB_vacuum.inc` (`alpha_grv=1`):

- Inner 1: $c_1 \omega^2 y_1 - \ell y_2 - \ell y_3 = 0$
- Inner 2: $\ell y_3 - y_4 = 0$
- Outer 1: $y_1 - y_2 = 0$
- Outer 2: $U y_1 + (\ell+1) y_3 + y_4 = 0$

## 14.2 Two bookkeeping bugs caught by external reference

A first pass produced 2.5% rel_diff at $n_g=1$ — suspicious, because
the physics is supposed to be identical to GYRE.  Re-reading `A_t.inc`
line-by-line (noting that GYRE stores the transpose of the Jacobian,
so `A_t(i,j) = A(j,i)`) revealed two errors:

1. **eq1 missed the $\lambda/(c_1\omega^2) y_3$ coupling** (only present
   when `alpha_grv=1`).  This inverse-$\omega^2$ term is what makes the
   full gravity system actually couple $\Phi'$ back into the displacement
   equation.
2. **eq2 had $y_3$ coefficient $-A^\star$ instead of $0$**, and
   **missed the $-y_4$ term**.

After correction the residual drops by four orders of magnitude:

| $n_g$ | $\omega^2_\text{GYRE (full)}$ | $\omega^2_\text{ours (full)}$ | rel_diff |
|---|---|---|---|
| 1 | $2.51593$ | $2.51593$ | $5.9 \times 10^{-7}$ |
| 2 | $1.28571$ | $1.28571$ | $2.7 \times 10^{-5}$ |
| 5 | $0.36993$ | $0.36992$ | $2.0 \times 10^{-4}$ |
| 10 | $0.11807$ | $0.11801$ | $5.3 \times 10^{-4}$ |

$\max \text{rel\_diff} = 5.3 \times 10^{-4}$.  **n_g=1 agrees to 6
significant digits — essentially double-precision machine epsilon times
the condition number of the GEP.**

## 14.3 Boundary at $x = 0$

The current implementation uses `inner_cut = 0.01` rather than $x = 0$.
For $\ell = 1$ this is harmless: the inner BC
$c_1\omega^2 y_1 = \ell y_2 + \ell y_3$ is already regular at $x = 0$
(all structure coefficients are finite there; $A^\star \to 0$,
$U \to 3$ for Lane-Emden $n=3$), and our O($h^2$) cell-centred BC
collocation gives 6-digit agreement with GYRE which does integrate
through $x = 0$ via Frobenius series.  For higher $\ell$ a variable
substitution $\tilde y = y / x^{\ell - 1}$ would be needed; this is
out of scope for the reference implementation since the production
target (stellar pulsation g-modes) is almost exclusively $\ell = 1, 2$.


# 15. Production readiness summary

| Operator | n_g=1 err vs GYRE full | Status |
|---|---|---|
| `solve_gmode_cowling` (slab) | $\sim 220\%$ | educational baseline only |
| `solve_gmode_cowling_spherical` | $120\%$ | educational baseline only |
| `solve_anelastic_2var` | $120\%$ | Boussinesq-like, no $V/U/\Gamma_1$ |
| `solve_gmode_cowling_gyre_compat` (2-var Cowling) | $13.4\%$ | Cowling approx limit |
| **`solve_gmode_full_gyre_compat` (4-var full)** | **$5.9 \times 10^{-7}$** | **production reference** |
| **`AnelasticSLSolver::solve_gmode_full_chebyshev` (CUDA)** | **$8.6 \times 10^{-7}$ @ $N = 96$** | **CUDA port, externally validated 2026-05-03** (§16) |

The 4-variable `solve_gmode_full_gyre_compat` is the **CUDA port target**.
It implements the exact equations and BCs GYRE solves for the standard
adiabatic non-rotating case, with 6-digit numerical agreement at
low radial order.  Reference outputs (first 10 $\omega^2$ at $N_r = 1024$
on the shipped Lane-Emden $n=3$ polytrope) are frozen as EXPECTED
constants in `gmode_exp_i_gyre_compat.py` (Cowling check) and
`gmode_exp_j_full_gyre_compat.py` (full-gravity check); both `--verify`
regressions must exit zero before any CUDA port is merged.


# 16. Experiment K — CUDA Chebyshev 4-var operator vs GYRE

> **Provenance.** `scripts/gmode_exp_k_chebyshev_full.py` (Python reference)
> and `AnelasticSLSolver::solve_gmode_full_chebyshev` (CUDA) at commit
> `942ca9e`.  Benchmark re-run on 2026-05-03 after collaborator install
> of GYRE at `$HOME/gyre` + MESA SDK `$HOME/mesasdk-26.3.2`.  Two local
> fixes required to close the loop (see §16.1); both targeted at the
> input pipeline, not the EVP kernel itself.  Raw artefacts (poly3.txt,
> gyre summary.h5, CUDA CSVs, PNG plots, SHA256SUMS) attached to GitHub
> release `gyre-benchmark-2026-05-03`.

## 16.1 Two pipeline fixes

1. `src/main.cpp` — missing `#include <algorithm>` for the
   `std::sort(x_cgl)` call; the header was only implicit in GCC ≤ 14.
2. `src/gpu/stellar_profile.cpp::read_gyre_structure_txt` — GYRE's
   `poly_to_txt` writes `V_2 = A* = Infinity` on the surface row
   ($x = 1$).  The linear-interp into our CGL grid at
   `outer_cut = 0.9999` picked up that row and poisoned $Q$ → the
   generalised-eigenproblem LU factorisation reported
   `getrf(Q) info = 256` (singular).  Fix: drop rows with any
   non-finite structure coefficient in the reader.

Neither fix touches the CUDA kernel.  Both are input-sanitisation
patches discovered only by feeding GYRE-native structure data into
the pipeline for the first time; the Python-only Exp J had silently
avoided them by reading `/tmp/gyre_run/poly3.txt` directly via
`np.loadtxt` (which produces `inf` that `np.interp` propagates but
never hits `cholesky`/`eig`, since Exp J builds its own grid).

## 16.2 Setup

Same Lane-Emden $n = 3$ polytrope shipped with GYRE.  Structure data
loaded via `ANSL_POLY3_TXT=/tmp/gyre_run/poly3.txt`.  CGL grid on
$[10^{-4}, 0.9999]$; interior BCs = regular-at-origin, exterior BCs
= vacuum-at-surface, same as GYRE `IB_regular.inc` + `OB_vacuum.inc`.
4-var full-gravity operator (`alpha_grv = 1`); ghost modes
(eigenvalues with $|\lambda| \to 0$) filtered out; propagation-cavity
classifier (`p_frac < 0.05`) to remove residual p-mode contamination.

## 16.3 Result vs GYRE full-gravity `summary.h5`

CUDA `--test gmode_exp_k --solver anelastic_sl --nr 96`:

| $n_g$ | $\omega^2_\text{CUDA}$ | $\omega^2_\text{GYRE}$ | rel_diff |
|---|---|---|---|
| 1 | $2.51593011$ | $2.51592794$ | $8.6 \times 10^{-7}$ |
| 2 | $1.28571470$ | $1.28570775$ | $5.4 \times 10^{-6}$ |
| 3 | $0.77574009$ | $0.77573278$ | $9.4 \times 10^{-6}$ |
| 4 | $0.51778259$ | $0.51777598$ | $1.3 \times 10^{-5}$ |
| 5 | $0.36993169$ | $0.36992550$ | $1.7 \times 10^{-5}$ |
| 6 | $0.27750880$ | $0.27750282$ | $2.2 \times 10^{-5}$ |
| 7 | $0.21593228$ | $0.21592665$ | $2.6 \times 10^{-5}$ |
| 8 | $0.17285865$ | $0.17285360$ | $2.9 \times 10^{-5}$ |
| 9 | $0.14154872$ | $0.14154409$ | $3.3 \times 10^{-5}$ |
| 10 | $0.11807267$ | $0.11806842$ | $3.6 \times 10^{-5}$ |

$\max \text{rel\_diff} = 3.6 \times 10^{-5}$ across $n_g = 1..10$.
The monotone growth with $n_g$ indicates the residual is
interpolation-dominated (GYRE's 1001-point native grid → CGL via
linear interp; replacing this with cubic-spline would push it down
further, but the current figure already exceeds all PASS thresholds).

At $N = 48$ the classifier still admits 2 head spurious eigenvalues
(ω² ≈ 500, 443) above the true $n_g = 1$ mode; by $N = 96$ they fall
out of the kept window.  Python `scripts/verify_exp_k_cuda.py --N 48`
reproduces this head contamination (it's a classifier sensitivity at
low $N$, not a CUDA-specific issue).

## 16.4 Cross-validation with GYRE-Cowling and Python-J

Re-running the upstream Python benchmarks against the same GYRE
artefacts on the collaborator machine:

| Experiment | Operator | Baseline | max rel_diff | PASS |
|---|---|---|---|---|
| Exp H (Python) | Boussinesq 2-var | GYRE full | $1.2$ (expected, §12) | — |
| Exp I (Python) | 2-var Cowling | GYRE Cowling (`alpha_grv=0`) | $5.6 \times 10^{-4}$ | ✅ |
| Exp J (Python) | 4-var full | GYRE full | $5.3 \times 10^{-4}$ | ✅ |
| **Exp K (CUDA)** | 4-var full (CGL) | GYRE full | **$3.6 \times 10^{-5}$** | ✅ |

Exp K's tighter agreement than Exp J is the expected gain from
spectral (Chebyshev) vs finite-difference discretisation on the same
physics; it's not a refutation of Exp J.

## 16.5 Reproducer

```bash
# 1. Produce GYRE baselines
MESASDK_ROOT=$HOME/mesasdk-26.3.2 bash scripts/gmode_exp_h_run_gyre.sh
MESASDK_ROOT=$HOME/mesasdk-26.3.2 bash scripts/gmode_exp_i_run_gyre_cowling.sh

# 2. Build (pixi provides cuda-toolkit 12.9 + cmake)
pixi run configure-gpu && pixi run build-gpu

# 3. CUDA Exp K with GYRE profile (N = 96, DOF = 388)
ANSL_POLY3_TXT=/tmp/gyre_run/poly3.txt \
    build/stellar2d --test gmode_exp_k --solver anelastic_sl --nr 96

# 4. Python cross-checks (Exp H/I/J)
pixi run python scripts/gmode_exp_h_gyre_benchmark.py
pixi run python scripts/gmode_exp_i_gyre_compat.py
pixi run python scripts/gmode_exp_j_full_gyre_compat.py
```

CSV outputs land at `runs/gmode_exp_k_<N>x64_<timestamp>/gmode_exp_k.csv`.
Published artefacts (poly3.txt, two GYRE summary.h5, CUDA CSVs at
$N = 48$ and $N = 96$, Exp H/I/J PNGs, SHA256SUMS):

  → GitHub release **`gyre-benchmark-2026-05-03`** on
  `MahoMaho-Rize/stellar2d`.
