---
title: Exp K CUDA benchmark — GYRE baseline reproducer (2026-05-03)
author: closing the loop on commit 942ca9e (branch `anelastic-sl-spectral`)
date: 2026-05-03
---

# Purpose

Commit `942ca9e ADD: Exp K CUDA + Lane-Emden polytrope profile builder`
landed the CUDA port of the 4-variable full-gravity Chebyshev EVP
(`AnelasticSLSolver::solve_gmode_full_chebyshev`) but noted:

> *"vs GYRE EXPECTED (n_g=1 ω²=2.52): 需要 GYRE poly3.txt baseline,
>  本地没装.  我们自建 polytrope 和 GYRE shipped poly.h5 细节有差 ...
>  移交合作者验证: 装 GYRE 后 bash scripts/gmode_exp_h_run_gyre.sh
>  生成 /tmp/gyre_run/poly3.txt + summary.h5"*

This document is the collaborator-machine re-run on 2026-05-03 that
closes the validation loop for Exp H, I, J (Python) and Exp K (CUDA)
against a live GYRE install.

# Host environment

| Component | Value |
|---|---|
| OS | Linux 6.19.8-200.fc43.x86_64 (Fedora 43) |
| GPU | NVIDIA GeForce RTX 4080 SUPER (compute 8.9) |
| CUDA | 12.9.86 (via `pixi` workspace env) |
| GCC | 15.2.1 |
| GYRE | `$HOME/gyre`, built from source |
| MESA SDK | `$HOME/mesasdk-26.3.2` |
| stellar2d HEAD | `942ca9e` (branch `anelastic-sl-spectral`) |

Two local patches were required on top of `942ca9e` before the CUDA
binary could consume GYRE's structure dump; these are documented in
§3 and also summarised in `gmode_experiments_2026-05-02.md` §16.1.

# 1. GYRE baselines

Two GYRE runs on the shipped Lane-Emden $n = 3$ polytrope
(`$HOME/gyre/models/poly/3.0/poly.h5`), both $\ell = 1$, dense inverse
frequency scan `freq_min = 0.1, freq_max = 3.0, n_freq = 400`,
`diff_scheme = COLLOC_GL6`.

```bash
MESASDK_ROOT=$HOME/mesasdk-26.3.2 bash scripts/gmode_exp_h_run_gyre.sh
#   → /tmp/gyre_run/poly3.txt                 1001 rows structure (x, V_2, A*, U, c_1, Γ_1)
#   → /tmp/gyre_run/summary.h5                38 l=1 modes, alpha_grv = 1 (full gravity)

MESASDK_ROOT=$HOME/mesasdk-26.3.2 bash scripts/gmode_exp_i_run_gyre_cowling.sh
#   → /tmp/gyre_cowling/summary_cowling.h5    39 l=1 modes, alpha_grv = 0 (Cowling)
```

SHA256 (also bundled in release):

| File | SHA256 |
|---|---|
| `poly3.txt` | `528fb14bc045faf13988f7024c3ccf8db8b2bba6413029601c48089e548766cf` |
| `summary.h5` (full) | `d438906a6a30743d97493240a90b54b64b6d000ec6b1a5a3e54ada822435c08d` |
| `summary_cowling.h5` | `37a0ae7e5151b39a14b95f1824b20861abbbed64cbfde579ec3bb0aa27e87082` |

GYRE's $\ell = 1$ full-gravity $\omega^2$ spectrum (first 10 g-modes,
$n_g = -n_{pg}$):

| $n_g$ | $\omega^2$ |
|---|---|
| 1 | 2.515927936088 |
| 2 | 1.285707754486 |
| 3 | 0.775732776477 |
| 4 | 0.517775976232 |
| 5 | 0.369925495676 |
| 6 | 0.277502815460 |
| 7 | 0.215926647338 |
| 8 | 0.172853603270 |
| 9 | 0.141544090462 |
| 10 | 0.118068423527 |

These are the reference numbers all three downstream benchmarks compare
against (Exp H/J directly; Exp K through the EXPECTED array frozen in
`src/main.cpp`).

# 2. Pipeline fixes applied

Both patches are input-pipeline only; the CUDA eigenproblem kernel
itself was untouched.

## 2.1 `src/main.cpp`: missing `<algorithm>`

`std::sort(x_cgl.begin(), x_cgl.end())` at line 1321 compiled on the
author's machine only through a transitively-included standard header;
on GCC 15.2.1 it triggers:

```
src/main.cpp:1320:18: error: 'sort' is not a member of 'std'; did you mean 'sqrt'?
```

Fix: explicit include.

```diff
 #include <functional>
+#include <algorithm>
 #include <sys/stat.h>
```

## 2.2 `src/gpu/stellar_profile.cpp`: GYRE surface-row `Infinity`

`gyre/bin/poly_to_txt` writes $V_2$ and $A^\star$ as `Infinity` on the
surface row ($x = 1$).  Example tail of `poly3.txt`:

```
  0.999000...  0.40088e+04  0.60012e+03  0.4071e-08  0.99700  1.6667
  1.000000...     Infinity     Infinity         0.0  1.00000  1.6667
```

`read_gyre_structure_txt` used `sscanf` which happily parses
`"Infinity"` into IEEE `inf`.  Downstream linear interpolation from
the 1001 GYRE points onto the CGL grid at `outer_cut = 0.9999` picked
up this infinite row (since $0.9999 < 1.0$) and poisoned every row of
$Q$ that touched the upper boundary.  The CUDA LU factorisation
(cuSOLVER `DgetrfBatched`) reported:

```
solve_gmode_full_chebyshev: getrf(Q) info=256
```

— Q is singular; no eigenvalues computed.

Fix: skip non-finite rows at read time.

```diff
 while (std::fgets(line, sizeof(line), fp)) {
     double x, V_2, A_star, U, c_1, G1;
     if (std::sscanf(line, "%lf %lf %lf %lf %lf %lf",
                     &x, &V_2, &A_star, &U, &c_1, &G1) == 6) {
+        // GYRE poly_to_txt writes V_2=A*=Infinity at the surface row (x=1);
+        // skip non-finite rows so downstream interpolation stays clean.
+        if (!std::isfinite(V_2) || !std::isfinite(A_star)
+            || !std::isfinite(U) || !std::isfinite(c_1)
+            || !std::isfinite(G1)) continue;
         out.x.push_back(x);
         ...
```

After this patch, `read_gyre_structure_txt` returns 1000 clean rows
(vs 1001 before), the interp on `[inner_cut, outer_cut] = [1e-4,
0.9999]` never touches the surface row, and $Q$ is full-rank.

Note: the Python Exp J (`gmode_exp_j_full_gyre_compat.py`) had avoided
this silently because it resamples onto its own uniform grid inside
`(0.01, 0.998)` and never queries near $x = 1$.  Exp K's CGL grid
extends to $x = 0.9999$ (to squeeze the last $n_g$ modes out of the
propagation cavity) and thus hit it first.

# 3. Benchmark results

All four runs re-done on 2026-05-03 using the GYRE artefacts from §1.

## 3.1 Exp H — Python, original simplified operators

`scripts/gmode_exp_h_gyre_benchmark.py`, compares three Python solvers
(`sph`, `2var`, `gyre_compat`) against GYRE-full.

```
  n_g       ω²_GYRE        ω²_sph       ω²_2var     ω²_compat     rd_sph    rd_2var  rd_compat
    1    2.5159e+00    5.5236e+00    5.5237e+00    2.8519e+00   1.20e+00   1.20e+00   1.34e-01
    5    3.6993e-01    4.9195e-01    4.9202e-01    3.7472e-01   3.30e-01   3.30e-01   1.30e-02
   10    1.1807e-01    1.3919e-01    1.3928e-01    1.1843e-01   1.79e-01   1.80e-01   3.05e-03

  max rel_diff (sph vs GYRE)         = 1.195e+00
  max rel_diff (2var vs GYRE)        = 1.195e+00
  max rel_diff (gyre_compat vs GYRE) = 1.336e-01
```

The ~120% disagreement on `sph`/`2var` is the Boussinesq simplification
error that motivated Exp I/J/K; the ~13% on `gyre_compat` is the
Cowling approximation error.  Both are expected and documented in
§12/§13 of `gmode_experiments_2026-05-02.md`.  **No PASS threshold
applies to Exp H** — it is a teaching plot showing why the simpler
operators are not the CUDA port target.

PNG: `gmode_exp_h_gyre_benchmark.png`.

## 3.2 Exp I — Python, 2-var Cowling vs GYRE-Cowling

`scripts/gmode_exp_i_gyre_compat.py`, compares
`solve_gmode_cowling_gyre_compat` (Python, `alpha_grv = 0`) against
`/tmp/gyre_cowling/summary_cowling.h5`.

```
  n_g    omega^2_GYRE    omega^2_ours     rel_diff
     1    2.851953e+00    2.851946e+00    2.300e-06
     4    5.277955e-01    5.277956e-01    8.207e-08
    10    1.184953e-01    1.184290e-01    5.588e-04

  max rel_diff = 5.588e-04
  PASS criterion: max rel_diff < 1e-2
  Result: PASS
```

PNG: `gmode_exp_i_gyre_compat.png`.

## 3.3 Exp J — Python, 4-var full vs GYRE-full

`scripts/gmode_exp_j_full_gyre_compat.py`, compares
`solve_gmode_full_gyre_compat` (Python, `alpha_grv = 1`) against
`/tmp/gyre_run/summary.h5`.

```
  n_g    ω²_GYRE_full    ω²_ours_full     rel_diff
     1    2.515928e+00    2.515927e+00    4.046e-07
     4    5.177760e-01    5.177796e-01    6.914e-06
    10    1.180684e-01    1.180062e-01    5.271e-04

  max rel_diff = 5.271e-04
  PASS criterion: max rel_diff < 1e-2
  Result: PASS
```

PNG: `gmode_exp_j_full_gyre_compat.png`.

## 3.4 Exp K — CUDA, 4-var full on Chebyshev collocation grid

```bash
ANSL_POLY3_TXT=/tmp/gyre_run/poly3.txt \
    build/stellar2d --test gmode_exp_k --solver anelastic_sl --nr 96
```

### $N = 48$ (DOF = 196): ghost modes above $n_g = 1$

```
  [Exp K] eigenvalues: total=196, valid_real_pos=69, kept=28
  n_g         ω²_CUDA         ω²_GYRE     rel err
    1  4.9903171683e+02  2.5159279361e+00   1.973e+02
    2  4.4333003501e+02  1.2857077545e+00   3.438e+02
    3  2.5159299872e+00  7.7573277648e-01   2.243e+00
    ...
```

The two head entries (ω² ≈ 500, 443) are spurious — the
propagation-cavity classifier at $N = 48$ misclassifies them.  Starting
at $n_g = 3$ the true g-mode ladder appears, shifted by 2 slots.
Python `scripts/verify_exp_k_cuda.py --N 48` reproduces this head
contamination identically (it's a classifier-at-low-$N$ issue, not a
CUDA bug).  CSV: `gmode_exp_k_N48.csv`.

### $N = 96$ (DOF = 388): clean agreement

```
  Exp K CUDA:  N=96 (DOF=388), ell=1, polytrope n=3
  n_g         ω²_CUDA         ω²_GYRE     rel err
    1  2.5159301091e+00  2.5159279361e+00   8.637e-07
    2  1.2857146951e+00  1.2857077545e+00   5.398e-06
    3  7.7574008734e-01  7.7573277648e-01   9.424e-06
    4  5.1778258843e-01  5.1777597623e-01   1.277e-05
    5  3.6993169173e-01  3.6992549568e-01   1.675e-05
    6  2.7750880199e-01  2.7750281546e-01   2.157e-05
    7  2.1593227798e-01  2.1592664734e-01   2.608e-05
    8  1.7285865166e-01  1.7285360327e-01   2.921e-05
    9  1.4154872063e-01  1.4154409046e-01   3.271e-05
   10  1.1807266549e-01  1.1806842353e-01   3.593e-05
  Exp K max_rel = 3.593e-05
```

**$\max \text{rel\_diff} = 3.6 \times 10^{-5}$** across $n_g = 1..10$.
CSV: `gmode_exp_k_N96.csv`.

The monotone growth with $n_g$ is interpolation-dominated (GYRE's
1001-point native grid → CGL via linear interp).  Replacing the
reader's linear interp with cubic-spline would tighten this further,
but the current agreement is 4 decades tighter than the PASS
threshold used for Exp I/J.

# 4. Summary

| Check | Operator | Baseline | max rel_diff | PASS |
|---|---|---|---|---|
| Exp H | Boussinesq 2-var / sph | GYRE full | $1.2$ | expected per §12 |
| Exp I | 2-var Cowling (Python) | GYRE Cowling | $5.6 \times 10^{-4}$ | ✅ |
| Exp J | 4-var full (Python) | GYRE full | $5.3 \times 10^{-4}$ | ✅ |
| **Exp K** | **4-var full (CUDA, CGL)** | **GYRE full** | **$3.6 \times 10^{-5}$** | **✅** |

The CUDA port of the 4-variable full-gravity g-mode EVP is externally
validated against GYRE.  Commit `942ca9e` can be declared production-
ready for the anelastic SL-spectral Phase 1 closure.

# 5. Attached artefacts (GitHub release `gyre-benchmark-2026-05-03`)

| File | Purpose |
|---|---|
| `poly3.txt` | GYRE Lane-Emden $n=3$ structure dump (1001 rows) |
| `gyre_summary_full.h5` | GYRE $\ell = 1$ full-gravity spectrum |
| `gyre_summary_cowling.h5` | GYRE $\ell = 1$ Cowling (`alpha_grv = 0`) spectrum |
| `gmode_exp_k_N48.csv` | CUDA Exp K raw $\omega^2$ at $N = 48$ (shows head spurious modes) |
| `gmode_exp_k_N96.csv` | CUDA Exp K raw $\omega^2$ at $N = 96$ (converged, $3.6\text{e-}5$) |
| `gmode_exp_h_gyre_benchmark.png` | Exp H plot: Boussinesq vs GYRE-full |
| `gmode_exp_i_gyre_compat.png` | Exp I plot: 2-var Cowling vs GYRE-Cowling |
| `gmode_exp_j_full_gyre_compat.png` | Exp J plot: 4-var full vs GYRE-full |
| `SHA256SUMS.txt` | Checksums for all of the above |
