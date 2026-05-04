---
title: |
  Reduced-Pressure Liouville Formulation — Experimental Follow-up:
  End-to-End Convergence, $k_x$-Independence, and Chebyshev Floor
author: |
  Technical Report (Supplement to `reduced_pressure_liouville.md`),
  stellar2d project, `anelastic-sl-spectral` branch
date: 2 May 2026
---

# 0. Purpose

This note supplements `docs/reduced_pressure_liouville.md` with three numerical
experiments that extend, verify, and in places **correct** the original report:

- **Experiment B.** End-to-end Poisson convergence measured on the physical
  quantity $\pi$ (rather than the intermediate $q$).
- **Experiment C.** Numerical confirmation of the $k_x$-independence claim of §5.
- **Experiment A.** Chebyshev collocation eigensolver to break the
  finite-difference error floor identified in §9.2.

In addition, a **sign error** in the original derivation of $\widetilde{W}$
(eq (12) of the parent document) is identified and corrected.

All three scripts are self-contained in `scripts/`, produce plots in `videos/`,
and can be rerun without rebuilding any C++ / CUDA code.

**Reproducibility protocol.** Every numeric table in §§2–4 is traceable to one
script.  Each script:

1. Prints a **provenance banner** on startup (script path + git HEAD + date).
2. Embeds its published reference values as an `EXPECTED` constant.
3. Accepts `--verify` to compare a fresh run against `EXPECTED` within a
   stated relative tolerance and exit nonzero on drift.

So the workflow for checking any claim here is:

```bash
python scripts/reduced_pressure_<experiment>.py --verify
```

A green run reproduces the report's numbers; a red run points at exactly which
row drifted.  Updating the reference values after an intentional change
requires editing `EXPECTED` in the script and the matching table in this
document in the same commit.


# 1. Correction to eq (12) of the parent document

The parent document states the reduced-pressure Liouville potential as

$$\widetilde{W} \;\stackrel{?}{=}\; \frac{\rho_0''}{2\rho_0} \;-\; \frac{(\rho_0')^2}{4\rho_0^2}. \tag{12}$$

For Lane-Emden $n = 3/2$ with $\rho_0 = c\,t^{3/2}$ near the surface:

$$\frac{\rho_0''}{2\rho_0} = \frac{3}{8\,t^2}, \qquad
\frac{(\rho_0')^2}{4\rho_0^2} = \frac{9}{16\,t^2},$$

so (12) as written gives $\widetilde{W} \to 3/(8t^2) - 9/(16t^2) = -3/(16\,t^2)$,
i.e.\ an **attractive** potential of strength $-3/16$.  This contradicts eq (13)
of the parent document, which was directly verified by computer algebra to give
the **repulsive** result $+3/(16\,t^2)$.

The correct formula is obtained by carrying out the substitution
$\pi = \rho_0^{-1/2}\,q$ in full:

$$\frac{\dd}{\dd y}\!\left[\rho_0\,\frac{\dd\pi}{\dd y}\right] = \sqrt{\rho_0}\,[\,q'' + \widetilde{W}\,q\,], \qquad
\boxed{\widetilde{W} = \frac{(\rho_0')^2}{4\rho_0^2} \;-\; \frac{\rho_0''}{2\rho_0}.} \tag{12'}$$

For Lane-Emden this gives $\widetilde{W} \to 9/(16t^2) - 6/(16t^2) = +3/(16\,t^2)$,
matching (13).  The **magnitude** $|C| = 3/16$ and the **sign** (repulsive) — the
two features used throughout §§3–5 of the parent document — are unchanged; only
the closed-form expression in (12) needed the sign flip.

All scripts in this supplement use (12').


# 2. Experiment B — end-to-end Poisson convergence on $\pi$

> **Provenance.**  All numbers in this section were produced by
> `scripts/reduced_pressure_endtoend.py` at commit `f8fcf96`.  To reproduce
> and verify:
> ```bash
> python scripts/reduced_pressure_endtoend.py            # fresh run
> python scripts/reduced_pressure_endtoend.py --verify   # compare against EXPECTED
> ```
> The script embeds the numbers below in its `EXPECTED` dict; `--verify`
> exits nonzero on drift beyond tolerance.
>
> **Plot:** `videos/reduced_pressure_endtoend.png`.

## 2.1 Setup

The parent document's §9 measures the $L^2$ error of the reconstructed
Liouville-space field $q$, after the substitution $\pi = \rho_0^{-1/2}\,q$.
This supplement runs the **full pipeline** end-to-end and measures the error on
the physically meaningful field $\pi$ itself:

$$
\pi_\text{exact}(y) = \sin\!\left(\frac{\pi (y-y_0)}{L}\right),
\qquad
p_\text{exact}(y) = \rho_0(y)\,\pi_\text{exact}(y)
$$

with the appropriate analytic RHS for each formulation, passed through
weighting → SL expansion → diagonal solve → unweighting → recovery of $\pi$
(original form: solve for $p$, divide by $\rho_0$; reduced form: solve for $\pi$
directly).  Both paths therefore end on the **same physical grid** with a
comparable target.

## 2.2 Result

| $\rho_\text{cut}$ | err_orig $(N{=}20)$ | err_redu $(N{=}20)$ | ratio | slope_orig | slope_redu |
|---|---|---|---|---|---|
| 0.1   | $1.74\times 10^{-4}$ | $1.75\times 10^{-4}$ | $1.0\times$ | $-2.33$ | $-2.33$ |
| 0.01  | $5.67\times 10^{-4}$ | $5.89\times 10^{-4}$ | $1.0\times$ | $-2.24$ | $-2.29$ |
| 0.001 | $1.11\times 10^{-3}$ | $1.57\times 10^{-3}$ | $0.7\times$ | $-1.84$ | $-2.11$ |

**The two formulations converge at essentially the same rate when the error is
measured on $\pi$.** The order-of-magnitude advantage reported in §9 of the
parent document is an artifact of measuring the error on $q$ instead of $\pi$.

## 2.3 Interpretation

The substitution $q = \rho_0^{+1/2}\,\pi$ multiplies by a function that vanishes
at the boundary.  A fixed $L^2$-on-$\pi$ error budget therefore translates into
a non-uniform budget on $q$: errors near the surface are "amplified" when viewed
in the original potential's $q$ space because $\sqrt{\rho_0}$ is small there.
The reduced-pressure form's $\sqrt{\rho_0}^{-1}$ weighting is the mirror of this,
and the two effects combine so that the final $\pi$-space error depends only on
the basis's representation capacity for $\pi$, which is similar in the two forms
in the interior.

This does **not** invalidate the reduced-pressure approach.  It does downgrade
the operational claim "4× GEMM cost reduction" (parent document §9.3) to "no
GEMM cost reduction in $\pi$-norm at the current FD eigensolver accuracy".  The
genuine benefits of the reformulation — repulsive vs attractive potential, both
indicial exponents integrable, better numerical conditioning — remain, but the
convergence-order advantage should not be used as a sales pitch without a
corresponding $q$-norm physical justification.


# 3. Experiment C — $k_x$-independence verification

> **Provenance.**  All numbers in this section were produced by
> `scripts/reduced_pressure_kx_independence.py` at commit `f8fcf96`.  To
> reproduce and verify:
> ```bash
> python scripts/reduced_pressure_kx_independence.py
> python scripts/reduced_pressure_kx_independence.py --verify
> ```
>
> **Plot:** `videos/reduced_pressure_kx_independence.png`.

## 3.1 Setup

Build the SL eigenpairs $\{(\mu_n, \psi_n)\}$ **once** from the chosen
$\widetilde{W}(y)$, then run the end-to-end Poisson solve for many $k_x$ values
using the same basis.  If §5 of the parent document is correct, the error on
$\pi$ should be bounded independently of $k_x$.

Parameters: $N_y = 512$, $N_m = 80$, $\rho_\text{cut} = 0.01$,
$k_x / (2\pi) \in \{1, 2, 4, 8, 16, 32, 64, 128\}$ (4 decades of $k_x^2$).

## 3.2 Result

| $k_x$ | $k_x^2$ | err_orig | err_redu |
|---|---|---|---|
| $2\pi$ | $3.95\times 10^1$ | $2.3741\times 10^{-5}$ | $2.3828\times 10^{-5}$ |
| $4\pi$ | $1.58\times 10^2$ | $2.3724\times 10^{-5}$ | $2.3802\times 10^{-5}$ |
| $8\pi$ | $6.32\times 10^2$ | $2.3722\times 10^{-5}$ | $2.3800\times 10^{-5}$ |
| $16\pi$ | $2.53\times 10^3$ | $2.3721\times 10^{-5}$ | $2.3800\times 10^{-5}$ |
| $32\pi$ | $1.01\times 10^4$ | $2.3721\times 10^{-5}$ | $2.3799\times 10^{-5}$ |
| $64\pi$ | $4.04\times 10^4$ | $2.3721\times 10^{-5}$ | $2.3799\times 10^{-5}$ |
| $128\pi$ | $1.62\times 10^5$ | $2.3721\times 10^{-5}$ | $2.3799\times 10^{-5}$ |
| $256\pi$ | $6.47\times 10^5$ | $2.3721\times 10^{-5}$ | $2.3799\times 10^{-5}$ |

Error spread (max/min): **1.00× for both formulations**.  The error is
identical to four significant figures across four decades of $k_x^2$.

## 3.3 Interpretation

The operator structure $\mathcal{T} - k_x^2$ (eq (15) of the parent document) is
confirmed: $k_x$ enters only as a shift on the diagonal eigenvalues
$\mu_n \to \mu_n + k_x^2$, and the basis $\{\psi_n\}$ requires no rebuild.

This is the single most important **engineering** claim of the parent document,
and it holds numerically.  It validates the GPU pipeline in §6: on an
$N_x \times N_y$ grid, a single $\Psi$ matrix of size $N_y \times N_\text{modes}$
serves all $N_x$ horizontal Fourier modes through two GEMMs.


# 4. Experiment A — Chebyshev collocation breaking the FD floor

> **Provenance.**  All numbers in this section were produced by
> `scripts/reduced_pressure_chebyshev.py` at commit `5803b6a`
> (previous commit `f8fcf96` had a sign bug in `W_from_rho`; see §4.3 note).
> To reproduce
> and verify:
> ```bash
> python scripts/reduced_pressure_chebyshev.py
> python scripts/reduced_pressure_chebyshev.py --verify
> ```
>
> **Plot:** `videos/reduced_pressure_chebyshev.png`.

## 4.1 Setup

The FD-based SL eigensolver used in Phase 0 and in §9 of the parent document
has 2nd-order accuracy, capping the achievable $q$-space $L^2$ error at
$\sim 10^{-7}$ regardless of how many SL modes are retained.

This experiment rebuilds the SL eigenvalue problem on Chebyshev-Gauss-Lobatto
(CGL) nodes using the Trefethen $D^2$ matrix, with Clenshaw-Curtis quadrature
weights used in both the normalisation of eigenfunctions and the forward SL
transform.  The manufactured $q$-space test from §9 is then rerun at
$N_\text{Cheb} \in \{64, 128, 256, 512\}$ and $N_\text{modes}$ up to $320$.

## 4.2 Result

| $N_\text{Cheb}$ | err_orig (max $N_m$) | err_redu (max $N_m$) | ratio |
|---|---|---|---|
| 64 | $5.10\times 10^{-6}$ | $5.38\times 10^{-7}$ | $9.47\times$ |
| 128 | $2.65\times 10^{-7}$ | $2.95\times 10^{-8}$ | $8.99\times$ |
| 256 | $1.24\times 10^{-8}$ | $1.40\times 10^{-9}$ | $8.85\times$ |
| 512 | $5.58\times 10^{-10}$ | $6.34\times 10^{-11}$ | $8.81\times$ |

- **The FD floor is broken.**  At $N_\text{Cheb} = 512$, $N_\text{modes} = 320$,
  the reduced-pressure $q$-space error drops to $6.3\times 10^{-11}$ — almost
  four orders of magnitude below the FD ceiling of $1.4\times 10^{-7}$ found
  in §9.
- **The low-$N$ advantage persists under spectral discretisation.**  The
  reduced-pressure form is consistently $\sim 9\times$ more accurate than the
  original across all Chebyshev grid sizes.  The ratio is nearly constant with
  resolution, confirming that the factor is intrinsic to the potential shape
  rather than an eigensolver artifact.
- **Convergence slopes are nearly identical** ($-3.9$ orig, $-3.8$ redu), with
  the reduced-p curve shifted down by roughly one decade at every $N_\text{modes}$.
  The advantage is in the **pre-factor**, not the rate.

## 4.3 Interpretation

The parent document's §9 low-$N$ advantage is **real** and robust under
spectral eigensolvers, not a finite-difference artifact as initially
conjectured.  The attractive $-21/(16\,t^2)$ well in the original form forces
low-order SL eigenfunctions to concentrate near the singular boundary to
satisfy the indicial balance $\alpha(\alpha-1) + C = 0$; the mildly repulsive
$+3/(16\,t^2)$ barrier in the reduced-pressure form lets the eigenfunctions
spread out more uniformly, giving better overlap with smooth source functions.

That overlap advantage persists regardless of how accurately we resolve the
eigenvalue problem — it is a property of the eigenfunctions themselves.  An
order-of-magnitude lower error at matched $N_\text{modes}$ translates, via
$N_m \propto \text{err}^{-1/\alpha}$ with $\alpha \approx 3.85$, to **roughly
$10^{1/3.85} \approx 1.8\times$ fewer modes** for a target accuracy, i.e.\ a
**~1.8× GEMM cost reduction** when measured in $q$-norm.

### A correction to an earlier draft

An earlier run of this script contained a sign bug in the helper function
`W_from_rho`: the $\rho''/(2\rho)$ term's sign was hard-coded as $+$ and
only the $(\rho')^2/\rho^2$ coefficient was parameterised.  This gave the
correct $W_\text{orig}$ (coincidentally — the original form's $\rho''/(2\rho)$
term does enter with sign $+$) but the wrong $\widetilde{W}_\text{wrong} =
+\rho''/(2\rho) + (\rho')^2/(4\rho^2) \to +15/(16\,t^2)$ instead of the
correct $\widetilde{W} = -\rho''/(2\rho) + (\rho')^2/(4\rho^2) \to +3/(16\,t^2)$.
Both potentials are repulsive but differ by a factor of 5 in strength, and
the eigenfunctions differ enough that the wrong $\widetilde{W}$ gave only a
$1.3\times$ advantage — leading to a mistaken draft of this section that
claimed the 10× advantage was an FD artifact.  The scripts in this commit
use the corrected `W_from_rho` with independent sign controls for both terms;
the numbers above are from that corrected run.  `compute_W_reduced` (used by
Experiments B and C) was always correct and symbolically verified.


# 5. Revised summary of the reformulation's benefits

With the three experiments in hand, the parent document's claims update as
follows:

| Claim | Status |
|---|---|
| Repulsive $\widetilde{W}$ vs attractive $W$ ($C$-sign flip) | **Confirmed** (eq (12') verified symbolically) |
| Both indicial exponents integrable | **Confirmed** (theoretical, robust) |
| $k_x$-independence of the SL operator | **Confirmed numerically to $k_x \sim 10^3$** (Experiment C) |
| 7× weaker singularity strength $|C|$ | **Confirmed** (derivation, symbolic) |
| ~10× lower error at low $N$ in $q$-norm | **Confirmed under Chebyshev too** (Experiment A: 9× at every $N_\text{Cheb}$) |
| ~4× GEMM cost reduction | **Partially supported**: ~1.8× cost reduction in $q$-norm (slopes match, prefactor differs by 9×, so $N_m$ scales as $\text{err}^{-1/3.85}$) |
| Same $\pi$-norm error end-to-end | **Confirmed** (Experiment B: 1:1 on physical $\pi$) |
| FD error floor at $\sim 10^{-7}$ | **FD-specific**, not physical (Experiment A reaches $6\times 10^{-11}$ under Chebyshev) |

The apparent tension between A (reduced-p wins 9× in $q$-norm) and B (two
forms tie in $\pi$-norm) is physical, not contradictory.  The substitution
$q = \rho_0^{+1/2}\,\pi$ down-weights the boundary region where the original
form's error concentrates; in $q$-norm that region contributes less but is
measured in a different metric.  The practical takeaway:

- If the downstream quantity of interest is $\pi$ (specific enthalpy
  perturbation), both forms deliver the same accuracy — **no $\pi$-side cost
  reduction**.
- If the downstream quantity is $q$ (the Liouville-normal-form field used
  internally for SL-basis storage, orthogonality, or numerical stability), the
  reduced-p form is **9× more accurate at matched $N_\text{modes}$**, giving a
  **~1.8× cost reduction** in GEMM operations when a target $q$-accuracy is
  fixed.

The **engineering recommendation stands**: the reduced-pressure form should be
preferred in GPU implementations.  The reasons are:

1. **Cleaner boundary** — both indicial branches integrable, no need to
   manually excise a divergent branch in the numerical eigensolver.
2. **Physical interpretability** — $\pi = p/\rho_0$ is the specific enthalpy
   perturbation, the natural variable for pressure-driven acceleration per
   unit mass.
3. **Degenerate operator** — $\nabla\cdot(\rho_0\nabla\pi)$ smoothly loses
   ellipticity at the surface rather than exhibiting a $1/\rho_0$ singularity
   in the elliptic coefficient.
4. **Better internal numerics** — the 9× $q$-norm advantage improves
   conditioning of the SL expansion coefficients $Q_n = -G_n/(\mu_n + k_x^2)$,
   reducing round-off when a narrow mode window is retained.
5. **Same cost** — the algorithm (parent §6) has identical GEMM, FFT, VRAM
   footprint, and GPU-side structure.  Only the weighting and the output
   variable change.


# 6. Reproduction

```bash
# End-to-end ($\pi$-norm): ~30 s
python scripts/reduced_pressure_endtoend.py

# k_x independence sweep: ~5 s
python scripts/reduced_pressure_kx_independence.py

# Chebyshev convergence: ~1 min (largest grid N=512)
python scripts/reduced_pressure_chebyshev.py
```

Each script writes one PNG to `videos/` and a numerical summary to stdout.  The
results reported above are from the commit that introduced these scripts; any
divergence in future runs would indicate changes to the Lane-Emden solver, the
grid construction, or the eigensolver.
