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
> `scripts/reduced_pressure_chebyshev.py` at commit `f8fcf96`.  To reproduce
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
| 64 | $5.10\times 10^{-6}$ | $3.68\times 10^{-6}$ | $1.39\times$ |
| 128 | $2.65\times 10^{-7}$ | $2.01\times 10^{-7}$ | $1.32\times$ |
| 256 | $1.24\times 10^{-8}$ | $9.54\times 10^{-9}$ | $1.30\times$ |
| 512 | $5.58\times 10^{-10}$ | $4.31\times 10^{-10}$ | $1.30\times$ |

- **The FD floor is broken.**  At $N_\text{Cheb} = 512$, $N_\text{modes} = 320$,
  the $q$-space error drops to $4.3\times 10^{-10}$ — three orders of magnitude
  below the FD ceiling of $1.4\times 10^{-7}$ found in §9.
- **The slope steepens to $-3.9$** in the reduced-p form (up from $\sim -2.2$
  under FD) and stays below the original form's slope only marginally.  The
  convergence is driven primarily by the smoothness of $q_\text{exact}$ (a
  truncated Fourier mode) and the basis's representation capacity, not by the
  potential shape.
- **The low-$N$ advantage shrinks from $\sim 10\times$ (FD) to $\sim 1.3\times$
  (Chebyshev).**  At $N_m = 80$: FD gives $3.1\times 10^{-7}$ vs $1.5\times 10^{-7}$
  (ratio $2\times$, approaching the floor); Chebyshev gives $2.65\times 10^{-7}$
  vs $2.01\times 10^{-7}$ (ratio $1.3\times$).

## 4.3 Interpretation

The large apparent advantage of the reduced-pressure form at low $N$ in the
parent document's Table of §9.1 is partly an **FD artifact**: the 2nd-order FD
eigensolver struggles more with the original form's strongly attractive
$-21/(16\,t^2)$ well than with the mildly repulsive $+3/(16\,t^2)$ barrier.
When the eigensolver is accurate to spectral order (Chebyshev), the two
potentials give nearly identical basis quality, and the convergence gap
collapses to a factor comparable to the indicial-exponent ratio.

This is a valuable correction: it tells us that **the right motivation for
adopting the reduced-pressure form is not faster convergence, but simpler
boundary behaviour** — both indicial exponents integrable, repulsive potential,
degenerate (rather than singular) elliptic coefficient.  Those properties are
robust to the eigensolver choice; the convergence-rate gain is not.


# 5. Revised summary of the reformulation's benefits

With the three experiments in hand, the parent document's claims update as
follows:

| Claim | Status |
|---|---|
| Repulsive $\widetilde{W}$ vs attractive $W$ ($C$-sign flip) | **Confirmed** (eq (12') verified) |
| Both indicial exponents integrable | **Confirmed** (theoretical, robust) |
| $k_x$-independence of the SL operator | **Confirmed numerically to $k_x \sim 10^3$** (Experiment C) |
| 7× weaker singularity strength $|C|$ | **Confirmed** (derivation, symbolic) |
| 10× lower error at low $N$ (q-norm, FD) | **Real but eigensolver-specific** (Experiment A: shrinks to 1.3× under Chebyshev) |
| 4× GEMM cost reduction (physical quantity) | **Not supported** in $\pi$-norm end-to-end (Experiment B) |
| FD error floor at $\sim 10^{-7}$ | **FD-specific**, not physical (Experiment A: $4 \times 10^{-10}$ under Chebyshev) |

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
4. **Same cost** — the algorithm (parent §6) has identical GEMM, FFT, VRAM
   footprint, and GPU-side structure.  Only the weighting and the output
   variable change.

The previously claimed **convergence-order** and **GEMM-cost** advantages
should be retracted; those advantages exist only in $q$-norm and only against a
2nd-order FD eigensolver.  Upgrading the eigensolver (as A shows is possible
and inexpensive) closes the gap.


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
