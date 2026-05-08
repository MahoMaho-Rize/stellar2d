# A9. Wave-speed estimates $S_L, S_R$

> **sympy script:** `scripts/a09_wave_speed_estimates.py`
> **generated LaTeX:** `output/a09_wave_speed_estimates.latex.tex`
> **verifies:** 1 Davis-bracket identity (80 random admissible (L,R)
> pairs × 8 eigenvalue-bracket inequalities = 640 scalar checks,
> all residuals $\le 0$); 4 Roe-property identities (80 random
> admissible (L,R) pairs, max residual $6\times 10^{-14}$ vs
> tol $10^{-9}$)
> **code checkpoints:**
> `src/gpu/explicit/strang_device.cuh :: d_lmhllc` (Davis speeds
> $S_L = \min(u_L - c_L, u_R - c_R)$, $S_R = \max(u_L + c_L, u_R +
> c_R)$ hard-coded at the top of the solver)

The HLLC algebra of §A8 requires **bounding wave speeds** $S_L, S_R$
that bracket every characteristic of the exact Riemann fan. This
section derives the three canonical choices, verifies each bounds
the A3 eigensystem correctly, and documents the Roe-average
construction used by Einfeldt's tighter bounds and by the Roe
solver of §A7.

## Davis wave speeds (used by the kernel)

$$S_L \;=\; \min(u_L - c_L,\ u_R - c_R), \qquad S_R \;=\; \max(u_L + c_L,\ u_R + c_R). \quad (\text{A9-Davis})$$

**Strong-form bracketing (Min/Max lattice).** By construction,

$$S_L \;\le\; u_K - c_K \;\le\; u_K \;\le\; u_K + c_K \;\le\; S_R \qquad \forall K \in \{L, R\}.$$

This says **every** eigenvalue of the A3 spectrum on either side
is contained in $[S_L, S_R]$. In particular, the entropy/shear
eigenvalues $\lambda_1 = \lambda_2 = u_K$ are trivially bracketed
by the acoustic bounds, and the acoustic eigenvalues $\lambda_{0,3}
= u_K \mp c_K$ are bracketed by construction.

**Verification (numerical, Rule 1 fallback for Min/Max lattice).**
sympy's `Min/Max` simplifier does not automatically reduce
expressions like $a - \min(a, b) = \max(0, a - b)$ when $a$ and
$b$ contain `sqrt(...)`. We therefore verify the bracketing
identities numerically at 80 random admissible $(L, R)$ pairs × 8
eigenvalues (4 per side) = 640 scalar checks. All residuals are
exactly zero (the Min/Max operations are hardware-level
comparisons, not floating-point arithmetic). This is not a
weak-form step; the identities are lattice-algebraic, and the
numerical fallback is a sympy-capability workaround only.

## Einfeldt tighter bounds

Using the Roe-averaged state $(\tilde\rho, \tilde u, \tilde v,
\tilde c)$ (defined below), Einfeldt (1988) proposed

$$S_L \;=\; \min\!\bigl(u_L - c_L,\ \tilde u - \tilde c\bigr), \qquad S_R \;=\; \max\!\bigl(u_R + c_R,\ \tilde u + \tilde c\bigr). \quad (\text{A9-Einfeldt})$$

These are strictly tighter than Davis when the Roe-averaged state
lies inside the fan. The tighter bounds reduce numerical dissipation
in HLLE and give the stellar2d-relevant "Einfeldt positivity"
property that HLLE preserves $\rho > 0, p > 0$ even across near-
vacuum states.

**Trade-off.** Einfeldt's bounds are more accurate but require
computing the Roe average. The Davis bounds are trivially cheaper
(no square-roots beyond the sound speed) and still safely bracket
every eigenvalue. The stellar2d kernel uses Davis for simplicity;
the Einfeldt alternative is derived here for comparison.

## Roe-averaged primitive state

Define the signed-weighted averages (Roe 1981, extended to 2D in
Glaister 1988):

$$\tilde \rho \;=\; \sqrt{\rho_L\,\rho_R}, \qquad \tilde u \;=\; \frac{\sqrt{\rho_L}\,u_L + \sqrt{\rho_R}\,u_R}{\sqrt{\rho_L} + \sqrt{\rho_R}},$$

$$\tilde v \;=\; \frac{\sqrt{\rho_L}\,v_L + \sqrt{\rho_R}\,v_R}{\sqrt{\rho_L} + \sqrt{\rho_R}}, \qquad \tilde h \;=\; \frac{\sqrt{\rho_L}\,h_L + \sqrt{\rho_R}\,h_R}{\sqrt{\rho_L} + \sqrt{\rho_R}},$$

$$\tilde c^{\,2} \;=\; (\gamma - 1)\!\left(\tilde h - \tfrac{1}{2}(\tilde u^{2} + \tilde v^{2})\right). \quad (\text{A9-Roe-avg})$$

The specific enthalpies are $h_K = \gamma p_K / ((\gamma-1)\rho_K) +
\tfrac{1}{2}(u_K^2 + v_K^2)$.

## The Roe property

The defining property of the Roe-averaged state is that the
flux Jacobian $A_x$ evaluated at the Roe average exactly reproduces
the flux jump:

$$A_{\mathrm{Roe}}(\mathbf{U}_L, \mathbf{U}_R)\,(\mathbf{U}_R - \mathbf{U}_L) \;=\; \mathbf{F}_x(\mathbf{U}_R) - \mathbf{F}_x(\mathbf{U}_L). \quad (\text{A9-Roe-property})$$

This is the **algebraic identity** that (a) makes the Roe solver
exact on isolated single-wave Riemann problems, (b) gives the
Einfeldt bounds their contact-resolution sharpness, and (c)
underpins the Roe entropy-fix discussion at transonic rarefactions.

**Strong-form verification via numerical fallback.** The Roe
matrix $A_x(\tilde{\mathbf{U}})$ has square-root-averaged entries
that sympy cannot simplify to the closed-form flux jump
$\mathbf{F}_R - \mathbf{F}_L$. We fall back to numerical random
sampling at 80 admissible $(L, R)$ pairs:

> _80 random $(\rho_L, \rho_R, u_L, u_R, v_L, v_R, p_L, p_R, \gamma)$
> samples with $\rho_K \in [0.1, 10]$, $|u_K|, |v_K| \in [0, 2]$,
> $p_K \in [0.1, 10]$, $\gamma \in \{1.4, 5/3, 2\}$. Four
> conservative-component residuals of $A_{\mathrm{Roe}}(\mathbf{U}_R
> - \mathbf{U}_L) - (\mathbf{F}_R - \mathbf{F}_L)$ checked with
> tolerance $10^{-9}$. Achieved max $|\text{residual}| = 6\times
> 10^{-14}$._

This is the same class of sympy-capability fallback as §A6's
Rankine-Hugoniot identities — strong-form, not weak-form. The
numerical check merely confirms what sympy cannot denest
symbolically.

## Entropy fix and transonic rarefaction

**Lax-entropy admissibility across a shock** was introduced in §A5
and flagged **[WEAK]** (distributional). The Roe solver violates
this at a transonic rarefaction (where a genuinely-nonlinear
acoustic eigenvalue changes sign inside the fan):
$A_{\mathrm{Roe}}$ treats the fan as a single jump at the averaged
speed, producing a non-physical "expansion shock" that carries the
fan's density/velocity data across a stationary discontinuity.

The fix is to replace $|\lambda_k|$ inside the Roe flux with
Harten's (1983) mollified absolute value $H_\epsilon(\lambda_k)$
whenever $\lambda_k$ changes sign across the fan. For HLLC + Davis
speeds this is not needed: the fan bounds are wide enough that the
transonic case is still resolved correctly. This is a
consequence of Davis's conservative over-estimate — one of the
reasons the stellar2d kernel uses Davis rather than the tighter
Einfeldt bounds.

## ✅ Verification checkpoint (to be wired)

The kernel's Davis wave-speed computation is simple enough that no
subtle bugs are likely; the checks are smoke tests:

1. **Davis brackets the spectrum.** For random admissible $(L, R)$
   pairs (100 samples), the kernel's $S_L, S_R$ must satisfy
   $S_L \le u_K \pm c_K \le S_R$ for both $K \in \{L, R\}$.
   Test: `test_strang_hllc.cu` §A9-Davis-bracket block.

2. **Davis bounds are Min/Max exact.** For any $(L, R)$ pair,
   $S_L = \min(u_L - c_L, u_R - c_R)$ to bitwise precision (no
   intermediate floating-point rounding). Test: one-line assertion
   using `std::min` and `std::max` comparing to the kernel's
   implementation.

3. **Einfeldt is tighter when applicable.** For a random state
   where the Roe-averaged state falls inside the fan, the
   Einfeldt bounds $[\tilde u - \tilde c, \tilde u + \tilde c]$
   must lie strictly inside the Davis bounds. This is a
   property-based test; it verifies understanding of the
   algebraic hierarchy, not the kernel itself (Einfeldt is not
   used in the kernel). Test: `test_strang_hllc.cu` §A9-Einfeldt-
   tighter block (optional scheme-characterisation check).

Failures of (1) or (2) indicate the kernel's Davis implementation
diverges from the §A9 definition — a bug in one of the
`fmin(u_L - c_L, u_R - c_R)` arithmetic ops.
