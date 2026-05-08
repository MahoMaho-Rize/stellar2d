# C8. Chromospheric (optically-thick) cooling and blended closure

> **sympy script:** `scripts/c8_chromospheric_cooling.py`
> **verified:** blend weight $\xi_\mathrm{rad} \in [0, 1)$ with anchor
> at $p = p_\mathrm{chr}$; Newton cooling has closed-form exponential
> solution $T(t) = T_\mathrm{ref} + (T_0-T_\mathrm{ref})\exp(-t/\tau_\mathrm{thck})$;
> GN05 scaling $\mathrm{d}\ln\tau_\mathrm{thck}/\mathrm{d}\ln\rho = -1/2$;
> Anderson-Athay cooling is monotone in $Z$, saturated above $\rho_\mathrm{cr}$;
> convex-combination partials $\partial_{Q_\mathrm{thck}} Q_R = \xi$,
> $\partial_{Q_\mathrm{thin}} Q_R = 1-\xi$.
> **code checkpoints:** `athena_mhd_kernels.cu::d_chromo_newton_cooling`,
> `athena_mhd_kernels.cu::d_anderson_athay_cooling`,
> `athena_mhd_solver.cu::apply_blended_cooling`,
> `tests/test_athena_mhd_chromo_relax.cu`.

## Why the chromosphere needs its own closure

At $T \lesssim 1.5\times 10^4\,\mathrm{K}$ the dominant radiative loss
channels (Lyman-$\alpha$, Ca II H+K, Mg II h+k, Balmer continuum) are
**saturated** — the emergent intensity equals the Planck function at
the optical-surface temperature, so cooling becomes independent of
the *local* $n_e n_i$ and instead relaxes toward the photospheric
$T_\mathrm{eff}$. The Sutherland-Dopita 1993 optically-thin rate
(§C7) overestimates cooling by $10^4\times$ in this regime
(Vernazza-Avrett-Loeser 1981 VAL3 atmosphere). Suzuki-group codes
patch this with a two-component closure.

## Shimizu+22 blending formula

$$\boxed{Q_R(T, \rho, p) = \xi_\mathrm{rad}(p)\,Q_R^\mathrm{thck}(T, \rho)
 + (1 - \xi_\mathrm{rad}(p))\,Q_R^\mathrm{thin}(T, \rho, Z).} \quad (\text{C8-blend})$$

with pressure-switch blending

$$\xi_\mathrm{rad}(p) = \max\!\bigl(0,\ 1 - p_\mathrm{chr}/p\bigr),\qquad
p_\mathrm{chr} \approx 0.1\,p_\odot. \quad (\text{C8-xi})$$

**Physical interpretation.** In the dense chromosphere $p \gg p_\mathrm{chr}$
gives $\xi \to 1$ (pure Newton cooling toward the photospheric
temperature). In the tenuous corona $p \lesssim p_\mathrm{chr}$ gives
$\xi \to 0$ (pure optically-thin SD93). The transition region at
$T \sim 15000\,\mathrm{K}$ is where $p \sim p_\mathrm{chr}$ and the
closure smoothly blends both channels.

Sympy-verified $\xi_\mathrm{rad} \in [0, 1)$ and anchors at zero at
the cutoff $p = p_\mathrm{chr}$.

## Gudiksen-Nordlund 2005 Newton cooling

$$\boxed{Q_R^\mathrm{thck} = \frac{e_\mathrm{int} - e_\mathrm{int}^\mathrm{ref}(r)}{\tau_\mathrm{thck}(\rho)},
\qquad \tau_\mathrm{thck}(\rho) = 0.1\,(\rho/\bar{\rho})^{-1/2}\,\mathrm{s},} \quad (\text{C8-GN05})$$

with $\bar{\rho} = 1.87\times 10^{-7}\,\mathrm{g\,cm^{-3}}$ (Shimizu+22
calibration) and $T^\mathrm{ref}(r) = T_\odot$ (or $T_\mathrm{eff}$ in
generic stars).

**Exponential relaxation.** For fixed $\rho$, the ODE
$\mathrm{d}T/\mathrm{d}t = -(T - T_\mathrm{ref})/\tau_\mathrm{thck}$
has the closed-form solution

$$T(t) = T_\mathrm{ref} + (T_0 - T_\mathrm{ref})\,\exp(-t/\tau_\mathrm{thck}). \quad (\text{C8-relax})$$

Sympy-verified by direct substitution. The GN05 scaling
$\tau_\mathrm{thck} \propto \rho^{-1/2}$ (sympy: log-slope $= -1/2$)
lengthens relaxation in tenuous layers — crucial to keep chromospheric
shock-train structures from being over-damped.

## Anderson-Athay 1989 empirical chromospheric rate

An independent branch used in Suzuki-Ohnaka-Yasuda 2025 (eq. 17):

$$\boxed{Q_R^\mathrm{AA}(\rho, Z) = 4.5\!\times\!10^9\,(0.2 + 0.8\,Z/Z_\odot)
\,\min(1,\,\rho/\rho_\mathrm{cr})\ \mathrm{erg\,cm^{-3}\,s^{-1}},\qquad
\rho_\mathrm{cr} = 10^{-16}\,\mathrm{g\,cm^{-3}}.} \quad (\text{C8-AA})$$

**Properties (sympy-verified):**
- Strictly non-negative for $\rho, Z \ge 0$.
- Monotone increasing in $Z/Z_\odot$: $\partial_Z Q_\mathrm{AA} = 3.6\times 10^9 \cdot \min(1, \rho/\rho_\mathrm{cr}) \ge 0$.
- Saturates at $\rho \ge \rho_\mathrm{cr}$: above the critical
  density, the chromosphere becomes fully optically thick and the
  cooling rate is density-independent (a photospheric-photon property).

The Anderson-Athay form is **not** exchangeable with the GN05 Newton
form — the former is a *rate* (per volume), the latter is a
*relaxation toward a reference state*. The Suzuki+25 code selects
AA for RGB-wind atmospheres, Shimizu+22 selects GN05 for solar wind.

## Two-piece thin cooling piece

$$Q_R^\mathrm{thin}(T, \rho, Z) = \begin{cases}
Q_R^\mathrm{SD93}(T, Z)\,n_e n_i, & T > 1.2\times 10^4\,\mathrm{K} \\
Q_R^\mathrm{AA}(\rho, Z), & T \le 1.2\times 10^4\,\mathrm{K} \\
0, & T \le T_\mathrm{cut} = 0.7\,T_\mathrm{eff}
\end{cases}. \quad (\text{C8-twopiece})$$

$T_\mathrm{cut}$ is a **numerical-stability floor**, not a physical
cutoff. Without it the chromospheric rate diverges at the
photospheric boundary and ejects the density to $\rho \to 0$. The
$0.7\,T_\mathrm{eff}$ threshold is Suzuki+25 calibration (Sec 2.5);
at $T_\mathrm{cut} = 3010\,\mathrm{K}$ for $\alpha$ Boo.

## Convex combination preserves positivity

For $\xi_\mathrm{rad} \in [0, 1]$ and $Q_R^\mathrm{thck}, Q_R^\mathrm{thin} \ge 0$:

$$Q_R = \xi\,Q_R^\mathrm{thck} + (1 - \xi)\,Q_R^\mathrm{thin} \ge 0. \quad (\text{C8-positivity})$$

Sympy partial derivatives confirm $\partial_{Q_\mathrm{thck}} Q_R = \xi$
and $\partial_{Q_\mathrm{thin}} Q_R = 1 - \xi$ — the blend is
*linear in each component*, so the combined cooling inherits positivity
from its parts.

## Implicit update (backward-Euler Newton cooling)

Because $\tau_\mathrm{thck}$ can drop below the hyperbolic CFL in
dense regions, the GN05 branch is usually integrated **implicitly**.
For fixed $\rho$ the single-step Backward-Euler reduces to

$$T^{n+1} = \frac{T^n + \nu\,T_\mathrm{ref}}{1 + \nu},\qquad
\nu \equiv \Delta t/\tau_\mathrm{thck}, \quad (\text{C8-BE})$$

which is **unconditionally stable** and exactly monotone toward
$T_\mathrm{ref}$. Kernel cost: one division per cell. No GMRES /
Newton-Krylov required because $Q_R^\mathrm{thck}$ is linear in $T$.

The thin-cooling branch uses Townsend 2009 (§C7 C7-Townsend); the
combined step applies BE for the thick piece and Townsend for the
thin piece in operator-split fashion.

## Smooth blending (optional)

The piecewise $\xi_\mathrm{rad}$ of (C8-xi) is $C^0$ at
$p = p_\mathrm{chr}$. For applications sensitive to the smoothness
of $Q_R$ (e.g., linear g-mode propagation through the transition
region), a $C^\infty$ alternative is

$$\xi_\mathrm{rad}^\mathrm{smooth}(p) = \tfrac{1}{2}\!\left[1 + \tanh\!\bigl((p - p_\mathrm{chr})/\Delta p\bigr)\right]. \quad (\text{C8-smooth})$$

$\Delta p \sim p_\mathrm{chr}/10$ keeps the transition width small.
Shimizu+22 uses the piecewise form; Suzuki+25 uses neither (pure AA
below the $T = 1.2\times 10^4$ threshold, no blending).

## Explicit vs implicit CFL comparison

| Branch | Explicit CFL | Implicit CFL |
|---|---|---|
| GN05 Newton (thick) | $\Delta t \le 0.3\,\tau_\mathrm{thck}$ | unconditional |
| SD93 (thin) | $\Delta t \le 0.1\,\tau_\mathrm{cool}$ | Townsend exact |
| AA chromosphere | depends on $\rho$; trivially sub-CFL | direct multiply |

**Recommendation**: in the kernel, use BE for GN05 and Townsend for
SD93 in an operator-split cooling pass applied after the hyperbolic
step. Neither branch touches $\rho, \mathbf{v}, \mathbf{B}$ — only
internal energy.

## ✅ Verification checkpoints

- `tests/test_athena_mhd_chromo_relax.cu` — 1D isobaric atmosphere
  seeded at $T_0 = 6000\,\mathrm{K}$ above $T_\mathrm{ref} = 5770\,\mathrm{K}$
  with $\tau_\mathrm{thck} = 0.1\,\mathrm{s}$; lock that $T(t)$
  matches (C8-relax) at $t = \{0.1, 1, 10\}\,\tau_\mathrm{thck}$
  to $<10^{-10}$ relative error.
- `tests/test_athena_mhd_chromo_blend.cu` — 1D atmosphere spanning
  $T \in [3000, 2\times 10^6]\,\mathrm{K}$; assert $\xi_\mathrm{rad}(p)$
  profile matches (C8-xi) cell-by-cell; assert smoothness /
  monotonicity.
- `tests/test_athena_mhd_chromo_positivity.cu` — 100 random
  $(T, \rho, p, Z)$ samples, assert $Q_R \ge 0$ and no NaN.
- `tests/test_athena_mhd_chromo_floor.cu` — atmosphere cooled past
  $T_\mathrm{cut} = 0.7\,T_\mathrm{eff}$; assert $T$ stays at floor
  and $Q_R^\mathrm{thin} = 0$ there.

The positivity test is the most important — it catches any sign-flip
regression in the AA or SD93 branches, which would let the code
"heat via radiation" and crash.
