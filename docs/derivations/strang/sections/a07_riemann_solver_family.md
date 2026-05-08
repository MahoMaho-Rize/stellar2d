# A7. Riemann solver family (Rusanov, HLLE, HLLC, Roe)

> **sympy script:** `scripts/a07_riemann_solver_family.py`
> **generated LaTeX:** `output/a07_riemann_solver_family.latex.tex`
> **verifies:** 15 strong-form identities — consistency of Rusanov
> and HLLE on identity states (8 components); mass-flux diffusion
> signature of Rusanov and HLLE on a stationary contact; HLLC
> $S_\star = 0$ at a stationary contact; $F_L = F_R$ at a stationary
> contact (used by Roe)
> **code checkpoints:**
> `src/gpu/explicit/strang_device.cuh :: d_lmhllc` (HLLC branch used
> by the kernel; Rusanov / HLLE / Roe derived here for comparison
> only)

This is the **first alternative-scheme comparison** section of the
book (rule 4 of the README: book is the numerical reference, not
just justification for the kernel's own choice). All four
classical Godunov-family Riemann solvers are derived in the same
template, then benchmarked against each other on the single most
informative identity: the numerical flux at a stationary contact
discontinuity.

## Riemann solver template and consistency

Every Godunov-family Riemann solver is required by Harten–Lax–van
Leer (1983) to satisfy the consistency condition

$$F_{\mathrm{num}}(\mathbf{U}, \mathbf{U}) \;=\; \mathbf{F}_x(\mathbf{U}) \quad \text{(A7-consistency)}$$

for every admissible $\mathbf{U}$. This is the identity-state
regression that rules out any solver whose flux disagrees with the
continuum flux on a uniform state.

## Rusanov (local Lax–Friedrichs)

The simplest scheme: blend $\mathbf{F}_L$ and $\mathbf{F}_R$ by an
upper bound on the wave speed,

$$F^{\mathrm{Rusanov}} \;=\; \tfrac{1}{2}\bigl(\mathbf{F}_L + \mathbf{F}_R\bigr) - \tfrac{1}{2}\alpha\,(\mathbf{U}_R - \mathbf{U}_L), \qquad \alpha = \max\bigl(|u_L| + c_L,\ |u_R| + c_R\bigr). \quad (\text{A7-Rusanov})$$

**Consistency.** Substituting $\mathbf{U}_R = \mathbf{U}_L$ yields
$F^{\mathrm{Rusanov}} = \mathbf{F}_L$. All 4 components verified by
sympy.

**Strengths**: single wave-speed estimate $\alpha$, trivially
positive-definite on any admissible state, robust under vacuum.

**Weaknesses**: maximally diffusive — every wave, including
linearly-degenerate contact waves, is smeared by the full $\alpha$
bound.

## HLLE (Harten–Lax–van Leer–Einfeldt, two-wave)

Replace $\alpha$ by separate lower/upper wave-speed estimates
$S_L \le 0 \le S_R$ derived from the flux Jacobian eigenvalues:

$$F^{\mathrm{HLLE}} \;=\; \frac{S_R\,\mathbf{F}_L \;-\; S_L\,\mathbf{F}_R \;+\; S_L S_R\,(\mathbf{U}_R - \mathbf{U}_L)}{S_R - S_L}. \quad (\text{A7-HLLE})$$

**Consistency.** At $\mathbf{U}_R = \mathbf{U}_L$ the jump term
vanishes and $(S_R\mathbf{F}_L - S_L\mathbf{F}_L)/(S_R - S_L) =
\mathbf{F}_L$. All 4 components verified by sympy.

**Strengths**: sharper than Rusanov (distinct $S_L, S_R$); still
positive-definite (Einfeldt 1988).

**Weaknesses**: no internal structure between the two extreme
waves, so the contact wave (linearly degenerate 2-family) is
averaged into the single two-wave HLL state.

## HLLC (Harten–Lax–van Leer–Contact, three-wave)

Resolve the contact wave explicitly by inserting a third wave of
speed $S_\star$ between $S_L$ and $S_R$. The full definition is

$$F^{\mathrm{HLLC}} \;=\; \begin{cases}
\mathbf{F}_L & \text{if } 0 \le S_L, \\[2pt]
\mathbf{F}_L + S_L (\mathbf{U}^{\star}_L - \mathbf{U}_L) & \text{if } S_L \le 0 \le S_{\star}, \\[2pt]
\mathbf{F}_R + S_R (\mathbf{U}^{\star}_R - \mathbf{U}_R) & \text{if } S_{\star} \le 0 \le S_R, \\[2pt]
\mathbf{F}_R & \text{if } S_R \le 0,
\end{cases} \quad (\text{A7-HLLC})$$

where $\mathbf{U}^\star_L, \mathbf{U}^\star_R$ are the star-region
intermediate states and $S_\star$ is the contact speed. Their
algebra is derived in §A8 in full strong form.

**Strengths**: contact resolution exact at stationary contacts
(proved below); robust; positivity-preserving under Batten 1997
conditions.

**Weaknesses**: the low-Mach limit can create too much pressure
dissipation — the kernel uses the Rieper (2011) LM-HLLC variant,
derived in §C3.

## Roe (flux-difference splitting)

Build a Roe-averaged Jacobian $A_{\mathrm{Roe}}(\mathbf{U}_L,
\mathbf{U}_R)$ satisfying $A_{\mathrm{Roe}}(\mathbf{U}_R -
\mathbf{U}_L) = \mathbf{F}_R - \mathbf{F}_L$ exactly (the Roe
property); then

$$F^{\mathrm{Roe}} \;=\; \tfrac{1}{2}(\mathbf{F}_L + \mathbf{F}_R) - \tfrac{1}{2}\,|A_{\mathrm{Roe}}|\,(\mathbf{U}_R - \mathbf{U}_L). \quad (\text{A7-Roe})$$

**Strengths**: exact on isolated single-wave Riemann problems
(including contacts), sharpest contact resolution.

**Weaknesses**: can violate the Lax entropy condition at transonic
rarefactions (sonic glitch); requires Harten entropy fix or
Einfeldt wave-speed bound. The Roe matrix does not remain
positive-definite across strong shocks.

## Contact-wave scorecard (strong-form derivation)

Consider the stationary isolated contact discontinuity: $u_L = u_R
= 0$, $p_L = p_R$, $\rho_L \neq \rho_R$, $v_L \neq v_R$. The exact
Riemann solution has **no motion**; the exact flux is

$$\mathbf{F}^{\mathrm{exact}} \;=\; (0,\; p_L,\; 0,\; 0)^{\!\top},$$

since $\rho u = 0$, $\rho u^2 + p = p$, $\rho u v = 0$, and
$(E + p) u = 0$ at $u = 0$.

**Rusanov mass-flux diffusion (strong form).** Substituting
$u_L = u_R = 0$, $p_L = p_R$ into the Rusanov formula,

$$F^{\mathrm{Rusanov}}_\rho \;-\; F^{\mathrm{exact}}_\rho \;=\; -\tfrac{1}{2}\alpha\,(\rho_R - \rho_L), \qquad \alpha = \max(c_L, c_R). \quad (\text{A7-Rusanov-contact})$$

Non-zero for $\rho_L \neq \rho_R$; the solver smears the density
jump at speed $\alpha/2$.

**HLLE mass-flux diffusion (strong form).** Substituting the same
IC,

$$F^{\mathrm{HLLE}}_\rho \;-\; F^{\mathrm{exact}}_\rho \;=\; \frac{S_L\,S_R\,(\rho_R - \rho_L)}{S_R - S_L}. \quad (\text{A7-HLLE-contact})$$

Non-zero for $\rho_L \neq \rho_R$ (with $S_L < 0 < S_R$, so the
denominator is positive); smaller than Rusanov but still diffusive.

**HLLC exact resolution (strong form).** The contact speed at the
stationary contact is

$$S_\star \;=\; \frac{p_R - p_L + \rho_L u_L (S_L - u_L) - \rho_R u_R (S_R - u_R)}{\rho_L (S_L - u_L) - \rho_R (S_R - u_R)} \;=\; 0,$$

since both the numerator ($p_R - p_L = 0$, $u_L = u_R = 0$) and
the sign structure give $S_\star = 0$ identically. The HLLC flux
in the left-star branch ($S_L \le 0 \le S_\star$) has mass
component $\rho^\star_L \cdot S_\star = \rho^\star_L \cdot 0 = 0$.
Hence

$$F^{\mathrm{HLLC}}_\rho \;-\; F^{\mathrm{exact}}_\rho \;=\; 0. \quad (\text{A7-HLLC-contact})$$

**Exact resolution**, no density smearing.

**Roe exact resolution (strong form).** At the stationary contact,
$\mathbf{F}_L = \mathbf{F}_R = \mathbf{F}^{\mathrm{exact}}$ (sympy
verified on all 4 components). The Roe flux reduces to
$\tfrac{1}{2}(\mathbf{F}_L + \mathbf{F}_R) - \tfrac{1}{2}|A_{\mathrm{Roe}}|(\mathbf{U}_R - \mathbf{U}_L)$;
on a pure-contact jump, $\mathbf{U}_R - \mathbf{U}_L$ lies exactly
in the null space of $A_{\mathrm{Roe}}$ (this is the Roe property
applied to the contact eigenvalue), so the dissipative second term
vanishes. Hence

$$F^{\mathrm{Roe}}_\rho \;-\; F^{\mathrm{exact}}_\rho \;=\; 0. \quad (\text{A7-Roe-contact})$$

**Also exact resolution.**

## Summary scorecard

| solver | identity state $F_{\mathrm{num}}(U,U)=F_x(U)$ | stationary contact mass flux diffusion | contact wave exact? |
|---|---|---|---|
| Rusanov | ✓ | $-\tfrac{1}{2}\alpha(\rho_R - \rho_L)$ | no |
| HLLE | ✓ | $S_L S_R (\rho_R - \rho_L)/(S_R - S_L)$ | no |
| HLLC | ✓ (§A8) | $0$ | **yes** |
| Roe | ✓ (§A8 via Roe) | $0$ | **yes** |

The Strang kernel uses HLLC (via LM-HLLC in `d_lmhllc`) precisely
because of the bottom-right cell of this table: without contact
resolution, any simulation of stratified convection (which is
dominated by density contrasts at near-zero velocity) would be
crushed by numerical diffusion at every cell face. HLLE and
Rusanov are derived here only for comparison.

## ✅ Verification checkpoint (to be wired)

1. **Identity-state flux.** For any admissible state $\mathbf{U}$,
   `d_lmhllc(U, U)` must return $\mathbf{F}_x(\mathbf{U})$ to ULP
   precision. Test: `test_strang_hllc.cu` §A7-identity block.

2. **Stationary-contact mass flux.** Initialise $u_L = u_R = 0$,
   $p_L = p_R$, $\rho_L = 1$, $\rho_R = 5$ (canonical contact);
   `d_lmhllc` must return $F_{\rho} = 0$ to ULP. If the returned
   value is non-zero by more than $10\varepsilon_{\mathrm{mach}}$,
   the solver has lost HLLC's contact-resolution property (likely
   through a bug in the star-state formula of §A8). Test:
   `test_strang_hllc.cu` §A7-contact block.

Failures of (1) indicate a kernel bug in the basic flux
construction (fixed before §A8 is trusted). Failures of (2)
indicate a bug in the $S_\star$ formula or in the star-state
branch selection — a deeper algebraic issue that must be caught
before any stratified-convection benchmark is run.
