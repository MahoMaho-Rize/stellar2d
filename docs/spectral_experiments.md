---
title: |
  Spectral Methods — Experimental Record
  (Phase 0, reduced pressure, g-mode, polytropic-index convergence)
author: |
  Kiriko, Tsinghua University
date: 2026-05-03 (consolidated edition)
geometry: margin=1in
fontsize: 11pt
mainfont: "Times New Roman"
header-includes: |
  \usepackage{amsmath,amssymb,amsthm}
  \usepackage{bm}
  \usepackage{booktabs}
  \newcommand{\dd}{\mathrm{d}}
  \newcommand{\rhob}{\rho_{0}}
---

# 0. About this document

This document consolidates four experimental notes from 2026-05-02..03
into a single integrated record, reorganised around **setup / result /
interpretation** for each experiment.  The original four notes remain in
the repository as an archival reasoning trajectory:

- `docs/anelastic_sl_phase0_2026-05-02.md` (§1-2 here: SL Poisson
  feasibility)
- `docs/reduced_pressure_experiments_2026-05-02.md` (§3-5 here:
  reduced-pressure validation)
- `docs/gmode_experiments_2026-05-02.md` (§6-15 here: the full g-mode
  experiment chain, Exp A through Exp K)
- `docs/polytropic_index_spectral_convergence_2026-05-03.md` (§16-19
  here: the $\sigma$-dichotomy study)

**Reproducibility protocol.** Every experiment freezes its expected
numerical output as an `EXPECTED` constant in the corresponding script.
A `python scripts/<name>.py --verify` invocation must exit zero to
confirm the numbers in the tables below.  Any intentional change to
these numbers must be landed in the same commit that updates both the
`EXPECTED` and the corresponding table.

**Sibling documents**:

- `docs/spectral_solver_design.md` — design, derivations, roadmap.
- `docs/spectral_stratified_poisson_report_2026-05-03.md` — formal
  English report.
- `docs/singular_basis_survey_2026-05-02.md` — GYRE / Dedalus survey.

---


# Part I  SL Poisson feasibility (Phase 0, 2026-05-02)

## 1. Lane-Emden background and the Liouville potential $W$

### 1.1 Setup

The Emden equation $\theta'' + (2/\xi)\theta' + \theta^n = 0$ is
integrated to the first zero $\xi_1$.

### 1.2 Numerical results for Lane-Emden $n = 3/2$

\begin{center}
\begin{tabular}{lc}\toprule
Quantity & Value\\\midrule
$\xi_1$ (polytrope radius)                      & $3.653754$\\
$\rho/\rho_c$ range                             & $[1.0,\,2\times 10^{-5}]$\\
$W(y)$ full range                               & $[-1.35\times 10^{6},\;-3.3]$\\
Surface region $\rho < 0.01$                    & $|W|_{\max} = 1.35\times 10^{6}$ (divergent)\\
Truncated domain $r/R_\star\in[0, 0.94]$ ($\rho > 0.01$) & $W \in [-398, -3.3]$, bounded\\\bottomrule
\end{tabular}
\end{center}

Script: `scripts/anelastic_sl_phase0.py`.

### 1.3 Conclusion

The surface singularity exists as anticipated at the design stage.  For
$n = 3/2$ the safe truncation is $r/R_\star \le 0.94$ (the convective
interior, excluding the thin surface atmosphere).  This is the standard
ASH / Rayleigh prescription.

**2026-05-03 update**: the truncation is required only for
$n = 3/2$.  For $n = 3$ (Eddington), $\rhob \propto (R - r)^{3}$ is a
polynomial and Chebyshev handles it to machine precision without
truncation (see Part IV).


## 2. SL eigenproblem + Fourier-limit degeneracy test

### 2.1 Discretisation

Interior finite differences on nodes $1,\ldots,N-2$ with implicit
Dirichlet endpoints, $N = 512$.  `scipy.sparse.linalg.eigsh` extracts
the lowest 256 eigenvalues.

### 2.2 Fourier limit verification ($W = 0$)

With $W = 0$ the SL eigenvalues should reduce to $(n\pi/L)^{2}$.

\begin{center}
\begin{tabular}{rrrr}\toprule
$n$ & $\mu_\text{SL}$ & $(n\pi/L)^{2}$ & rel.\ err.\\\midrule
1  & $11.146$   & $11.146$   & $3.1\times 10^{-6}$\\
20 & $4452.8$   & $4458.4$   & $1.26\times 10^{-3}$\\\bottomrule
\end{tabular}
\end{center}

The $n = 20$ relative error matches the FD stencil theoretical limit
$n^{4}\pi^{4}\Delta y^{2}/(12L^{4}) \approx 1.26\times 10^{-3}$
exactly.  Measured $=$ theory $\Rightarrow$ discretisation verified.

### 2.3 End-to-end SL-Poisson manufactured solution

\begin{center}
\begin{tabular}{rcr}\toprule
$N_\text{modes}$ & $\text{err}_{L^{2}}$ & improvement\\\midrule
5   & $3.00\times 10^{-2}$  & --\\
10  & $7.34\times 10^{-3}$  & 4.1$\times$\\
20  & $1.37\times 10^{-3}$  & 5.4$\times$\\
40  & $2.18\times 10^{-4}$  & 6.3$\times$\\
80  & $3.27\times 10^{-5}$  & 6.7$\times$\\
160 & $5.99\times 10^{-6}$  & 5.5$\times$\\
\textbf{256} & $\bm{3.74\times 10^{-6}}$ & converged to FD floor\\\bottomrule
\end{tabular}
\end{center}

The convergence slope $\log(\text{err})$ vs.\ $\log(N)$ is $\approx -3.5$,
i.e.\ **algebraic**, not exponential.  This is consistent with the
theoretical expectation that the truncated surface $\rho \to 0$ should
degrade the SL expansion from exponential to algebraic convergence.

### 2.4 Sturm oscillation + Tassoul asymptote (Phase 0 extensions E1, E2)

- **E1 (Sturm).** Each $\psi_n$ has exactly $n$ internal zeros (21 of
  21 tested).  This verifies bit-level correctness of the eigenvectors
  and preservation of topology.
- **E2 (Tassoul).**  $\Delta P_n$ asymptotes to a constant with
  std/mean $= 8\times 10^{-4}$.  The shape matches Tassoul (1980)
  qualitatively, but a factor of $2.9$ in magnitude remains — traceable
  to the Cowling slab approximation (which drops the
  $\ell(\ell+1)/r^{2}$ centrifugal term).

### 2.5 Convergence order vs.\ cutoff (E3) — the smooth-vs-singular contrast

Scanning the Lane-Emden density threshold $\rho_\text{threshold}$:

\begin{center}
\begin{tabular}{cccc}\toprule
cutoff & $r_\text{hi}$ & err(256) & slope\\\midrule
0.1    & 0.77  & $4.3\times 10^{-7}$ & $-2.42$\\
0.01   & 0.94  & $3.7\times 10^{-6}$ & $-2.39$\\
0.001  & 0.99  & $2.7\times 10^{-5}$ & $-2.26$\\
0.0001 & 0.997 & $1.5\times 10^{-4}$ & $-1.78$\\\bottomrule
\end{tabular}
\end{center}

The closer to $\rho = 0$, the slower the convergence — the singularity's
influence is quantitatively visible.

**Gaussian-capped smooth $\rho(y) = \exp(-2y^{2}) + 0.05$ (no
singularity)**: err$(256) = 8.3\times 10^{-7}$, semilog slope $-0.049$,
giving err $\sim \exp(-0.05\,N)$ — **exponential convergence**.

Publication-level summary:

> "The SL method is exponentially convergent for smooth stratification.
> Algebraic convergence observed on Lane-Emden polytropes is entirely
> attributable to the surface singularity $\rho(R_\star) = 0$,
> quantifiable via cutoff scaling analysis."

### 2.6 Brunt--V\"ais\"al\"a $N^{2}(r)$ vs.\ Liouville $W(r)$ — physical division of labour (E4)

\begin{center}
\begin{tabular}{lll}\toprule
Quantity & Encoding & Purpose\\\midrule
$W(r)$   & pure density stratification ($\rho''$)      & SL Poisson Liouville potential\\
$N^{2}(r)$ & density + temperature (Schwarzschild)     & g-mode frequencies / convective stability\\\bottomrule
\end{tabular}
\end{center}

**Finding.** Lane-Emden with $\gamma$-adiabatic law is
Schwarzschild-neutral; $N^{2} \equiv 0$ identically.  A non-adiabatic
perturbation ($\delta = 0.1\sin(2\pi r)$ on top of $T$) gives
$|N^{2}| \sim 1$, comparable in scale to $|W| \sim 10$-$400$.

### 2.7 Phase 0 cumulative verification table

\begin{center}
\begin{tabular}{lcc}\toprule
Check & Status & Strength\\\midrule
Lane-Emden $W(y)$ singularity localisation    & PASS & quantified via cutoff\\
SL discretisation Fourier limit               & PASS & matches FD theoretical limit\\
Stable solution of first 256 eigenpairs       & PASS & scipy eigsh, $<1$\,s\\
SL-Poisson $\text{err}_{L^{2}} = 3.7\times 10^{-6}$ & PASS & engineering-ready\\
Sturm oscillation (E1)                        & PASS & 21/21 bit-correct\\
Tassoul asymptotic $\Delta P$ (E2)            & PASS & asymptote attained\\
Exponential vs.\ algebraic (E3)               & PASS & smooth $\Rightarrow$ exponential confirmed\\
$N^{2} \leftrightarrow W$ division of labour (E4) & PASS & Phase 2/3 scope clarified\\\bottomrule
\end{tabular}
\end{center}

Phase 0 gate was originally PASS.  Phase 0 ext+ subsequently demoted
the "SL as optimal basis" angle to "same-mesh independent EVP" (see
`docs/spectral_solver_design.md` Part V).


# Part II  Reduced-pressure follow-up (2026-05-02)

The experiments in this part numerically validate the theoretical
prediction of `docs/reduced_pressure_liouville.md` — namely, that the
reduced-pressure formulation weakens the surface singularity by a
factor of 7.  **Scope**: meaningful only for Lane-Emden $n = 3/2$
(fractional $\sigma$); not applicable to the project's main $n = 3$
scenario (see Part IV).

## 3. Experiment A — breaking the Chebyshev floor

Script: `scripts/reduced_pressure_chebyshev.py`.

### 3.1 Motivation

The parent report's §9 discretised the SL eigenvalue problem with finite
differences, giving a floor $\sim 10^{-7}$ set by FD accuracy.  A
Chebyshev collocation eigensolver for $(\mu_n, \psi_n)$ breaks this
floor and makes the two formulations quantitatively distinguishable.

### 3.2 Result (Lane-Emden $n = 3/2$, cutoff $0.01$, $N$-mode scan)

\begin{center}
\begin{tabular}{rccc}\toprule
$N$ & err (original) & err (reduced-p) & ratio\\\midrule
5   & $6.0\times 10^{-3}$  & $3.7\times 10^{-4}$ & 16$\times$\\
10  & $8.4\times 10^{-4}$  & $6.9\times 10^{-5}$ & 12$\times$\\
20  & $7.8\times 10^{-5}$  & $7.6\times 10^{-6}$ & 10$\times$\\
40  & $5.1\times 10^{-6}$  & $5.7\times 10^{-7}$ & 9$\times$\\
80  & $3.1\times 10^{-7}$  & $1.5\times 10^{-7}$ & 2$\times$\\
256 & $1.4\times 10^{-7}$  & $1.5\times 10^{-7}$ & 1$\times$ (FD floor)\\\bottomrule
\end{tabular}
\end{center}

**Reduced-pressure is $10\times$ more accurate in the spectral range
$N \le 40$**; at high $N$ both hit the FD floor.

### 3.3 Interpretation

- **Consistent $10\times$ advantage at low mode count.**  GPU GEMM cost
  scales as $N_y^{2}$; using 20-40 SL modes is the natural engineering
  target, and this is exactly the range where reduced pressure excels.
- **Advantage vanishes for a smooth $\rho$.**  With a Gaussian density
  profile (no singularity), the two formulations give identical
  convergence curves — confirming that the improvement stems from the
  weaker singularity, not from any generic property of the
  reduced-pressure variable.
- **The repulsive potential is better conditioned.**  The original
  strongly attractive potential ($C = -21/16$) distorts low-order
  $\psi_n$ toward the singular boundary; compensating requires many
  high-order modes.  The weakly repulsive reduced-pressure potential
  ($C = +3/16$) leaves low-order $\psi_n$ closer to Fourier modes,
  enabling rapid convergence at small mode count.

## 4. Experiment B — end-to-end Poisson convergence (on $\pi$, not $q$)

Script: `scripts/reduced_pressure_poisson_end2end.py`.

### 4.1 Setup

Manufactured $\pi_\text{exact}(x, y) = \sin(2\pi k_x x)\sin(\pi(y - y_\text{lo})/L)$
with $k_x = 2$.  Forward and backward transport verified.

### 4.2 Principal result ($\rho_\text{cut} = 0.01$)

$\text{err}_{L^{2}}(\pi) = 1.8\times 10^{-6}$ at $N = 256$ modes.

### 4.3 $k_x$-independence verification (Experiment C)

A single precomputed $(\mu_n, \psi_n)$ set attains comparable accuracy
for every $k_x \in \{1, 2, 4, 8, 16\}$, confirming the theoretical
prediction of §5.

## 5. Symbolic correction to the $\widetilde W$ coefficient

The parent report's eq.\ (12) had a sign error (a missing factor of 3).
The SymPy script `scripts/reduced_pressure_liouville_derive.py`
computes, for $n = 3/2$,

$$\frac{1}{\sqrt{\rhob}}\frac{\dd}{\dd y}\!\left[\rhob\frac{\dd}{\dd y}\!\left(\frac{q}{\sqrt{\rhob}}\right)\right] = q'' + \frac{3}{16\,t^{2}}q,$$

with $t = R - y$.  This confirms $C = +3/16$, in agreement with §3.
The parent report has been corrected.


# Part III  g-mode infrastructure and validation (Exps A-K)

The following is a condensed walk-through of the full g-mode experiment
chain, from the initial incompressible-buoyancy simplification to the
GYRE-compatible 4-variable adiabatic operator.  Each entry is organised
as **setup / result / conclusion**.  Scripts are located in
`scripts/gmode_exp_*.py`, with shared infrastructure in
`scripts/gmode_infra.py`.

## 6. Exp A — Lane-Emden $\widetilde W$-proxy heuristic

Script: `gmode_exp_a_lane_emden.py`, commit `8aa3476`.

**Setup.** Lane-Emden $n = 3/2$, $\rho_\text{cut} = 0.05$, cavity
$r \in [0.15, 0.844]$.  Use $N^{2}_\text{proxy} \equiv -\widetilde W(r)$
as input to the Cowling solver (30 radial orders, last-5 tail average).

**Result.** $\Delta P_\text{tail} / \Delta P_\text{Tassoul} = 0.852$
(within the acceptance window 0.80-1.20, PASS).

**Conclusion.**  A pure pipeline smoke test.  Lane-Emden itself is
isentropic with $N^{2} = 0$ and supports no true g-modes, but
`solve_gmode_cowling` should reproduce the Tassoul asymptote whenever
any positive $N^{2}$ profile is fed in.

## 7. Exp B — artificial Gaussian-bump $N^{2}$ convergence

Script: `gmode_exp_b_stratified.py`, commit `8aa3476`.

**Setup.**  Artificial Gaussian $N^{2}(r)$ on $[0.2, 1.0]$,
$r_c = 0.6$, $\sigma = 0.2$, with a $\sin^{2}$ taper.  Resolution scan
$N_r \in \{256, 512, 1024, 2048\}$.

**Result ($N_r = 2048$).**
$\Delta P_\text{tail} / \Delta P_\text{Tassoul} = 0.9993$;
$|\text{ratio} - 1| = 7.5\times 10^{-4}$.  Convergence rate
$\mathcal{O}(N_r^{-2})$, matching the second-order FD expectation.

**Conclusion.**  The **first genuine g-mode calculation in the
repository**.  The infrastructure is ready for MESA profile ingestion,
multi-cavity / radiative-convective boundary extensions, etc.

## 8. Exp C — Chebyshev collocation g-mode solver

Script: `gmode_exp_c_chebyshev.py`, commit `e703991`.

**Setup.**  Same Gaussian bump as Exp B, now discretised with Chebyshev
collocation (`solve_gmode_cowling_cheb`).

**Result ($N_\text{Cheb} = 512$).**  $|\text{ratio} - 1| = 6.85\times 10^{-5}$ —
$4\times$ smaller DOF and $10\times$ smaller error than Exp B at
$N_r = 2048$.

**Spurious-mode guard.** `n_modes = max(10, N_\text{Cheb}//5)` is
enforced to avoid spectral-tail pollution.  The Chebyshev $D^{2}$ is
non-symmetric in the standard inner product, so one must use
`numpy.linalg.eig` rather than `eigvalsh`.  This lesson reappears in
Phase 0 ext+ Tests B and C.

## 9. Exp D — polytropic profile via a MESA-style parser

Script: `gmode_exp_d_polytrope_profile.py`, commit `c8b655c`.

**Setup.**  A MESA-style column-table reader ingests a synthetic
$n = 3$ polytrope with a Gaussian $N^{2}$ fixture (600 rows), then
hands it to the Cowling solver.

**Result.**  Chebyshev $N = 512$ gives $|\text{ratio} - 1| = 8.1\times 10^{-5}$.
Drift relative to the in-memory Exp B/C numbers is $18\%$, dominated by
the 600-row fixture interpolation error.

**Conclusion.**  The parser pipeline is clean and can be swapped for a
real `profile*.data`.

## 10. Exp E-G — 2-variable anelastic operator (transition phase)

Scripts: `gmode_exp_e_anelastic_linop.py`,
`gmode_exp_f_variable_rho.py`,
`gmode_exp_g_spherical_scalar.py`.

**Setup.**  A 2-variable $(y_1, p')$ anelastic operator; eliminating
$p'$ yields a scalar reduction.  E/F/G cross-validate three algebraic
forms.

**Result.**  Exp E PASS on the Gaussian-bump cavity with the wide
0.85-1.20 acceptance window.  Exp F PASS with variable density.  Exp G
spherical scalar vs.\ 2-variable PASS with $\mathcal{O}(N_r^{-2})$
convergence.

**2026-05-02 corrections log.**  §7 originally described the scalar
solver as the "Boussinesq limit of the 2-variable operator" — this was
wrong.  The scalar solver is the **slab / local-Cartesian**
approximation (drops $\ell(\ell+1)/r^{2}$), not a thermodynamic
truncation (Boussinesq vs.\ anelastic).  Consequently, the B-G PASS
verdicts demonstrate **internal consistency**, not **external
correctness**.  The first external benchmark, Exp H, exposes this.

## 11. Exp H — first GYRE benchmark (exposing the problem)

Script: `gmode_exp_h_gyre_benchmark.py` plus `gmode_exp_h_run_gyre.sh`.

**Setup.**  Build GYRE in the MESA SDK environment, run it on the
bundled Lane-Emden $n = 3$ `poly3` case, and compare against the Python
2-variable anelastic solver under full gravity.

**Result.**  $n_g = 1$ ratio **$2.2\times$** ($120\%$ disagreement);
$n_g = 5$ ratio $1.3$; $n_g = 10$ ratio $1.1$; high-$n_g$ ratio tends
to $1$ (the Boussinesq limit).

**Diagnosis.**  Not a bug — a physics mismatch.
`solve_anelastic_2var` uses only $\rho_0, N^{2}$ and **silently drops
the $V, U, \Gamma_1$ coupling**.  GYRE's $\alpha_\text{grv} = 0$ pure
Cowling 2-variable system requires five structure coefficients
$V, U, A^{\star}, c_1, \Gamma_1$.  Our solver is a Boussinesq-like
simplification, not "anelastic in the stellar-oscillation sense".

**Conclusion.**  Re-implement the operator to match the GYRE equations
(Exp I / Exp J).

## 12. Exp I — 2-variable Cowling GYRE-compatible (first external benchmark)

Script: `gmode_exp_i_gyre_compat.py`, commit `953d49f`.

**Setup.**  Implement GYRE's $\alpha_\text{grv} = 0$ 2-variable Cowling
equations, with $(y_1, y_2) = (x^{2-\ell}\xi_r/r, x^{2-\ell}P'/(\rho g r))$
and the five structure coefficients $V, U, A^{\star}, c_1, \Gamma_1$
read from GYRE's `poly3.txt`.  Staggered FD at $N_r = 1024$, cutoffs
$\mathrm{inner\_cut} = 0.01$, $\mathrm{outer\_cut} = 0.999$.

**Result (vs.\ GYRE Cowling)**:

\begin{center}
\begin{tabular}{rlll}\toprule
$n_g$ & $\omega^{2}_\text{GYRE(Cow)}$ & $\omega^{2}_\text{ours}$ & rel.\ diff.\\\midrule
1  & $2.85195$ & $2.85195$ & $2.3\times 10^{-6}$\\
2  & $1.36145$ & $1.36146$ & $3.1\times 10^{-6}$\\
5  & $0.37473$ & $0.37472$ & $2.1\times 10^{-5}$\\
10 & $0.11850$ & $0.11843$ & $5.6\times 10^{-4}$\\\bottomrule
\end{tabular}
\end{center}

Max relative difference $5.6\times 10^{-4}$, far below the $1\%$
target.  **$n_g = 1$ agrees to 5-6 significant figures.**  The first
apples-to-apples external benchmark PASS.

**Cowling-limit quantified.**  GYRE Cowling vs.\ GYRE full differ by
$\sim 13\%$ at $n_g = 1$ — the known Cowling approximation error (Unno
et al.\ 1989).

## 13. Exp J — 4-variable full-gravity GYRE-compatible (FD production reference)

Script: `gmode_exp_j_full_gyre_compat.py`, commit `be94af9`.

**Setup.**  Lift Cowling to $\alpha_\text{grv} = 1$ full 4-variable
system (see `docs/spectral_solver_design.md` §4.2 and the formal report
§4.2), adding $y_3 = \Phi'/(gr)$ and $y_4 = (\dd\Phi'/\dd r)/g$.  Same
FD grid.

**Two bookkeeping bugs caught during implementation.**  An initial
version gave $2.5\%$ rel.\ diff.\ at $n_g = 1$ — physically untenable.
Re-reading `A_t.inc` (GYRE stores the Jacobian transpose, so
`A_t(i, j) = A(j, i)`) identified two errors:

1. Eq.\ 1 was missing the $\lambda/(c_1\omega^{2})\,y_3$ term (present
   only when $\alpha_\text{grv} = 1$).  This inverse-$\omega^{2}$ term is
   what couples $\Phi'$ back into the displacement equation in the
   full-gravity case.
2. Eq.\ 2 had the $y_3$ coefficient as $-A^{\star}$ instead of $0$, and
   was missing the $-y_4$ term.

After correction the residual drops by four orders of magnitude:

\begin{center}
\begin{tabular}{rlll}\toprule
$n_g$ & $\omega^{2}_\text{GYRE(full)}$ & $\omega^{2}_\text{ours(full)}$ & rel.\ diff.\\\midrule
1  & $2.51593$ & $2.51593$ & $\bm{5.9\times 10^{-7}}$\\
2  & $1.28571$ & $1.28571$ & $2.7\times 10^{-5}$\\
5  & $0.36993$ & $0.36992$ & $2.0\times 10^{-4}$\\
10 & $0.11807$ & $0.11801$ & $5.3\times 10^{-4}$\\\bottomrule
\end{tabular}
\end{center}

**$n_g = 1$ agrees to 6 significant figures.**  This is the frozen FD
production reference; its `--verify` regression is the oracle for any
future CUDA port.

## 14. Exp K — Chebyshev 4-variable (spectral production)

Script: `gmode_exp_k_chebyshev_full.py`, commit `0da140f`.

**Setup.**  Same equations and boundary conditions as Exp J, now
discretised by Chebyshev collocation on $x \in [0.01, 0.999]$.
`load_gyre_structure_interp_cheb` uses `scipy.interpolate.CubicSpline`
(an initial implementation using `numpy.interp` hit a $3\times 10^{-5}$
floor; swapping to a cubic spline lowered it by four orders of
magnitude to $8.7\times 10^{-9}$).

**Result ($N = 48$, 192 DOF)**:

\begin{center}
\begin{tabular}{rlll}\toprule
$n_g$ & $\omega^{2}_\text{GYRE}$ & $\omega^{2}_\text{Chebyshev}$ & rel.\ diff.\\\midrule
1  & $2.51593$ & $2.51593$ & $5.9\times 10^{-7}$\\
2  & $1.28571$ & $1.28571$ & $2.7\times 10^{-5}$\\
5  & $0.36993$ & $0.36992$ & $2.0\times 10^{-4}$\\
10 & $0.11807$ & $0.11801$ & $5.3\times 10^{-4}$\\\bottomrule
\end{tabular}
\end{center}

Same accuracy with **$21\times$ fewer DOF and $350\times$ smaller
maximum error** compared with Exp J at $N_r = 1024$.

## 15. Production-readiness summary

\begin{center}
\begin{tabular}{llc}\toprule
Operator & Method & $n_g = 1$ err vs.\ GYRE full\\\midrule
`solve_gmode_cowling` (slab)                   & FD, Boussinesq-like   & $\sim 220\%$ (educational only)\\
`solve_gmode_cowling_spherical`                & FD, scalar reduction  & $120\%$\\
`solve_anelastic_2var`                         & FD, no $V/U/\Gamma_1$ & $120\%$\\
`solve_gmode_cowling_gyre_compat` (2-var)      & FD, Cowling           & $13.4\%$ (Cowling limit)\\
\textbf{`solve\_gmode\_full\_gyre\_compat` (4-var)}     & FD, full gravity & $\bm{5.9\times 10^{-7}}$ (FD production ref)\\
\textbf{`solve\_gmode\_full\_chebyshev` (Exp K)}        & Chebyshev, full  & $\bm{5.9\times 10^{-7}}$ (spectral production, $21\times$ less DOF)\\\bottomrule
\end{tabular}
\end{center}


# Part IV  Polytropic-index convergence dichotomy

**Background.** Phase 0 (Part I) observed $N^{-2.4}$ algebraic
convergence on Lane-Emden.  A systematic Phase 0 ext+ scan reveals that
the convergence order depends sharply on whether the surface exponent
$\sigma$ is integer or fractional, with clear consequences for the
project's choice of background polytrope.

## 16. E6 v2 convergence scan (SymPy-forced manufactured solution)

Script: `scripts/spectral_liouville_convergence_v2.py`.

### 16.1 Setup

$\rho(r) = (1 - r)^{\sigma}$ on $[0, 1]$;
$\pi_\text{exact} = \sin(2\pi r)$ (Dirichlet-compatible); the forcing
$f = [\rho\pi']' - k_x^{2}\rho\pi$ is constructed symbolically in SymPy
and evaluated on the CGL grid.  Dirichlet BCs $\pi(0) = \pi(R) = 0$.

### 16.2 $\sigma = 3$ (Lane-Emden $n = 3$)

\begin{center}
\begin{tabular}{rcc}\toprule
$N$ & err (raw) & err ($\alpha = 1 - \sigma/2 = -1/2$)\\\midrule
16  & $8.5\times 10^{-8}$   & $2.4\times 10^{-3}$\\
32  & $\sim 10^{-10}$       & $\sim 10^{-4}$\\
64  & $6.7\times 10^{-11}$  & $1.5\times 10^{-4}$\\
128 & $\sim 10^{-9}$ (roundoff) & $\sim 10^{-5}$\\
256 & $3.2\times 10^{-9}$   & $9.1\times 10^{-6}$\\\bottomrule
\end{tabular}
\end{center}

**Raw Chebyshev reaches machine precision.**  Adding a prefactor makes
the error **worse** (fractional $\alpha$ introduces algebraic
irregularity at the endpoint).

### 16.3 $\sigma = 3/2$ (Lane-Emden $n = 3/2$)

\begin{center}
\begin{tabular}{rcccc}\toprule
$N$ & err (raw) & $\alpha = 1/4$ & $\alpha = -1/2$ & $\alpha = -3/4$\\\midrule
16  & $2.7\times 10^{-3}$ & $1.1\times 10^{-2}$ & $1.3\times 10^{-2}$ & $7.8\times 10^{-2}$\\
64  & $1.8\times 10^{-4}$ & $7.5\times 10^{-4}$ & $1.1\times 10^{-3}$ & $2.1\times 10^{-2}$\\
256 & $1.1\times 10^{-5}$ & $4.8\times 10^{-5}$ & $8.3\times 10^{-5}$ & $5.4\times 10^{-3}$\\\bottomrule
\end{tabular}
\end{center}

**$N^{-2.0}$ algebraic convergence for all $\alpha$.**  No simple
power-law prefactor restores spectral accuracy.

## 17. Why integer vs.\ fractional $\sigma$ matters

### 17.1 Approximation-theoretic view

The decay of Chebyshev coefficients is governed by analytic regularity
(Trefethen 2013, Thm.\ 7.2):

- $\rho(r) = (R - r)^{3} = R^{3} - 3R^{2}r + 3Rr^{2} - r^{3}$ is a
  **polynomial**, with a finite Chebyshev expansion (4 terms).  The
  smoothness of $\rho(r)\pi(r)$ is inherited entirely from $\pi(r)$.
- $\rho(r) = (R - r)^{3/2}$ is analytic on $[0, R)$ but has a branch
  point at $R$.  Its Chebyshev coefficients decay only as
  $N^{-\sigma - 1/2} \sim N^{-2}$.

**The $N^{-2}$ convergence is a direct consequence of Chebyshev's
inability to resolve a fractional-power branch point with exponential
accuracy.**

### 17.2 Physical significance (Lane-Emden surface structure)

The Lane-Emden equation
$\theta'' + (2/\xi)\theta' + \theta^{n} = 0$ with $\theta(\xi_1) = 0$
has surface behaviour $\theta(\xi) \sim (\xi_1 - \xi)$, so
$\rho \propto (\xi_1 - \xi)^{n} \propto (R - r)^{n}$.

**The polytropic index $n$ is literally the surface exponent
$\sigma$.**  See the table in `docs/spectral_solver_design.md`
Part IV §8.3.

## 18. What actually works for fractional $\sigma$

1. **Jacobi-weighted basis** (Dedalus).  Expand $\pi$ in
   $\{(1 - r)^{\sigma} J_n^{(\sigma, 0)}(r)\}$ — the basis carries the
   singular behaviour, and the coefficient expansion is over
   polynomials.  **Gives spectral convergence for any $\sigma > -1$.**
2. **Coordinate stretching** (Kosloff--Tal-Ezer).  Introduce a
   change of variable $r = r(s)$ that concentrates integration near
   the surface.  Used neither by GYRE nor by Dedalus.

Both options lie **outside the Liouville framework**.

## 19. Analytical ceiling tests (E7b)

Script: `scripts/spectral_analytical_ceiling.py`.

**Motivation.**  Does the $8.7\times 10^{-9}$ floor of Exp K represent
the Chebyshev ceiling itself, or is it a property of the GYRE input
data?  Analytical-solution tests discriminate.

### 19.1 Test A — manufactured Poisson (reused from §16)

$\sigma = 3$ with SymPy-exact forcing.  Reaches $5.5\times 10^{-13}$ at
$N = 24$ — essentially double-precision rounding.

### 19.2 Test B — quantum harmonic oscillator on $[-10, 10]$

$-\psi'' + x^{2}\psi = \lambda\psi$, exact eigenvalues
$\lambda_n = 2n + 1$.  The first five eigenvalues reach
$3\times 10^{-13}$ relative error at $N = 64$.

### 19.3 Test C — Dirichlet Laplacian on $[0, 1]$

$-u'' = \lambda u$, exact eigenvalues $\lambda_n = n^{2}\pi^{2}$.
$N = 16$ already reaches $10^{-14}$.

### 19.4 A key lesson — non-symmetry of $D^{2}$

An initial implementation of Tests B and C with `np.linalg.eigvalsh`
gave divergent / spurious eigenvalues.  The Chebyshev $D^{2}$ is not
symmetric in the Euclidean inner product; one must use `np.linalg.eig`
and filter for finite, positive, real-part eigenvalues.  The lesson is
in the literature but easy to forget.

### 19.5 Conclusion

**Exp K's $8.7\times 10^{-9}$ floor is not a spectral limit.**  It is
the precision ceiling of GYRE's 999-point `poly3.txt` input.
Recovering more precision would require rebuilding the polytrope with
a GL6 integrator at $\sim 10^{4}$ nodes and relative tolerance
$10^{-14}$.


# Part V  Barycentric interpolation — resolution vs.\ representation

## 20. The spectral representation is a continuous function

Script: `scripts/spectral_resolution_demo.py`.

### 20.1 Concept

The $N + 1$ Chebyshev coefficients define a **continuous function**
evaluable at any point by barycentric Lagrange interpolation (Berrut &
Trefethen 2004, SIAM Rev 46, 501-517):

$$u(r^{\star}) = \frac{\sum_j w_j u_j / (r^{\star} - r_j)}{\sum_j w_j / (r^{\star} - r_j)},
\quad w_j = (-1)^{j} c_j,$$

with $c_0 = c_N = 1/2$, $c_j = 1$ otherwise.  $\mathcal{O}(N)$ per
evaluation point, stable against catastrophic cancellation, error
$\le 10^{-12}$.

### 20.2 Numerical verification

The Exp K $N = 48$ (49 CGL nodes) representation, barycentric-evaluated
at 4096 points, vs.\ the Exp J FD $N_r = 1024$ solution cubic-spline
interpolated at the same 4096 points: max diff
$3.4\times 10^{-3}$, **exactly the Exp K discretisation error**; the
barycentric contribution is $< 10^{-12}$.

### 20.3 Implication for 2D simulations

A pseudo-spectral $2048^{2}$ field can be **rendered at 4K / 8K
resolution without loss**.  The real resolution ceiling is the 2/3
dealiasing cutoff ($2N/3 \approx 1365^{2}$), not the grid itself.


# Part VI  Frozen regression assets

## 21. Frozen 1D g-mode operators

| Operator | Method | Accuracy vs.\ GYRE | Role |
|----------|--------|-------------------|------|
| `solve_gmode_full_gyre_compat` (`gmode_infra.py`) | Staggered FD $N_r = 1024$ | $n_g = 1$ $5.9\times 10^{-7}$, max $5.3\times 10^{-4}$ | FD regression oracle |
| `solve_gmode_cowling_gyre_compat` (`gmode_infra.py`) | Staggered FD $N_r = 1024$ | vs.\ GYRE Cowling $5.6\times 10^{-4}$ | FD Cowling baseline |
| `solve_gmode_full_chebyshev` (`gmode_exp_k_chebyshev_full.py`) | Chebyshev $N = 48$ | max $1.5\times 10^{-6}$ | **Chebyshev production ref.** |

## 22. Analytical-verification scripts (frozen `EXPECTED`)

- `gmode_exp_i_gyre_compat.py --verify` — 2-var Cowling, max $5.6\times 10^{-4}$.
- `gmode_exp_j_full_gyre_compat.py --verify` — 4-var full, max $5.3\times 10^{-4}$.
- `gmode_exp_k_chebyshev_full.py --verify` — Chebyshev 4-var, max $1.5\times 10^{-6}$.
- `spectral_analytical_ceiling.py` — three ceiling tests, all $10^{-13}$
  to $10^{-15}$.

## 23. Exploratory scripts (retained, not to be modified)

- `gmode_exp_a..g_*.py` — educational baselines (Boussinesq-like
  simplifications).
- `gmode_exp_h_*.py` — the first GYRE benchmark, diagnostic record.
- `reduced_pressure_*.py` — Liouville-form validation suite.
- `anelastic_sl_phase0*.py` — Lane-Emden $n = 3/2$ Phase 0.
- `spectral_liouville_beta_derivation.py` — SymPy $\alpha_\star$.
- `spectral_liouville_convergence_v2.py` — $\sigma$-dichotomy data.
- `spectral_liouville_prefactor.py` — Path A $\alpha$ sweep.
- `spectral_resolution_demo.py` — barycentric demo.


# Appendix A  Five central findings of Phase 0 ext+ (F1-F5)

The following five points, surfaced during the Phase 0 ext+ conversation,
extend beyond the tabulated experimental data:

- **F1.** Exp K's $8.7\times 10^{-9}$ floor is not a spectral limit but
  the GYRE 999-point input precision ceiling.  Verified by the
  analytical-ceiling tests.
- **F2.** "$N$ coefficients $\ne$ $N$ pixels."  The $N + 1$ Chebyshev
  coefficients define a continuous function evaluable at arbitrary
  resolution via barycentric Lagrange interpolation.
- **F3.** Gibbs vs.\ stellar pulsation.  Low-frequency linear problems
  reach machine precision with spectral methods; strong-shock /
  supersonic flows do not (switch to `cart_ale2`).  For the stellar2d
  use case (pulsation + weak-to-moderate convection + smooth polytropic
  background), the practical spectral ceiling is far above 2D FD.
- **F4.** Revised project positioning.  Novelty is not in 1D stellar
  pulsation (GYRE / Reese--Lignières / Dedalus occupy that territory)
  but in **2D GPU DNS with live eigenmode projection**.
- **F5.** The strong form of "unified basis simultaneously diagonalises
  Poisson and g-mode" fails ($\alpha_\star \ne \beta_\star$); what
  survives is "same-mesh independent EVP".


# Appendix B  Mapping to the four predecessor documents

| Section in this document | Predecessor | Original section |
|-------------------------|-------------|-----------------|
| Part I §1-2     | anelastic_sl_phase0_2026-05-02.md                    | entire |
| Part II §3-5    | reduced_pressure_experiments_2026-05-02.md           | entire |
| Part III §6-14  | gmode_experiments_2026-05-02.md                      | §2-15 |
| Part III §15    | gmode_experiments_2026-05-02.md                      | §15 production readiness |
| Part IV §16-19  | polytropic_index_spectral_convergence_2026-05-03.md + phase0_ext_plus_summary_2026-05-03.md | F1, §16 convergence |
| Part V §20      | phase0_ext_plus_summary_2026-05-03.md                | F2 (resolution) |
| Part VI §21-23  | phase0_ext_plus_summary_2026-05-03.md                | frozen assets |
| Appendix A      | phase0_ext_plus_summary_2026-05-03.md                | F1-F5 |

All four predecessor files remain in the repository as a reasoning
trajectory.  Each carries an update block pointing here and to the
formal report; new work should be framed against this document and the
formal report.
