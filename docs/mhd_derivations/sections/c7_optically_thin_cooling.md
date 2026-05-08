# C7. Optically-thin radiative cooling

> **sympy script:** `scripts/c7_optically_thin_cooling.py`
> **verified:** $\Lambda(T) > 0$ on piecewise power-law segments;
> $\tau_\mathrm{cool}$ positive definite; Townsend 2009 closed-form
> $T(t) = [T_0^{1-\alpha} - C(1-\alpha) t]^{1/(1-\alpha)}$
> numerically verified to satisfy $\mathrm{d}T/\mathrm{d}t = -C T^\alpha$
> across 21 $(\alpha, T_0, C, t)$ samples (max err $0.0$);
> $\alpha = 1$ degenerate exponential decay; logarithmic slope
> identity $\mathrm{d}\ln\Lambda/\mathrm{d}\ln T = \alpha$.
> **code checkpoints:** `athena_mhd_kernels.cu::d_cooling_source`,
> `athena_mhd_solver.cu::apply_radiative_cooling`,
> `tests/test_athena_mhd_cooling_townsend.cu`.

## Regime of validity

Optically-thin cooling applies when the mean-free-path of emitted
photons exceeds the local scale height,
$\ell_{\mathrm{photon}} \gtrsim H_\rho$. For a coronal plasma at
$T \gtrsim 10^{5.2}\,\mathrm{K}$, $\rho \lesssim 10^{-13}\,\mathrm{g\,cm^{-3}}$
this is satisfied and the spectrum is dominated by bremsstrahlung +
line emission (Sutherland-Dopita 1993, henceforth SD93).

Below $T \sim 1.5\times 10^4\,\mathrm{K}$ the chromosphere is
**optically thick** to the dominant Lyman-α and Ca II lines; see
§C8 for the blended thick-thin closure used in Suzuki+25 / Shimizu+22.

## SD93 cooling rate

$$\boxed{Q_R(T, \rho, Z) = n_e\,n_i\,\Lambda(T, Z)\ \ge\ 0,\qquad
n_e \approx n_i \approx \rho/(\mu_e m_u).} \quad (\text{C7-QR})$$

$\Lambda(T, Z)$ is tabulated from the SD93 ionisation-equilibrium
synthesis for $Z \in \{0, 10^{-3}, 10^{-2}, 10^{-1}, 1, 3\}\,Z_\odot$
over $\log T \in [4, 8.5]$. Piecewise power-law fit:

$$\Lambda(T) \approx \Lambda_k\,(T/T_k)^{\alpha_k},\qquad T \in [T_k, T_{k+1}]. \quad (\text{C7-piecewise})$$

**Typical slopes** (SD93 Table 6, solar Z):

| Regime | $\log T$ | $\alpha$ | Physics |
|---|---|---|---|
| Chromospheric tail | $4.0$–$4.3$ | $+3$ to $+5$ | HI, Ca II lines |
| Line-dominated | $4.3$–$7.0$ | $-0.5$ to $-1$ | Fe, O, Si |
| Bremsstrahlung | $7.0$–$8.5$ | $+1/2$ | free-free |

## Cooling timescale

$$\tau_\mathrm{cool} = \frac{\varepsilon_\mathrm{th}}{Q_R}
= \frac{p}{(\gamma - 1)\,n_e n_i\,\Lambda(T)}
= \frac{\mu_e^2 m_u^2 k_B T}{(\gamma-1)\,\mu m_u\,\rho\,\Lambda(T)}. \quad (\text{C7-tau})$$

Sympy-verified $\tau_\mathrm{cool} > 0$ for positive inputs.

**Order-of-magnitude**: at solar corona base ($T \sim 10^6\,\mathrm{K}$,
$\rho \sim 10^{-15}\,\mathrm{g\,cm^{-3}}$, $\Lambda \sim 10^{-22.7}\,\mathrm{erg\,cm^3\,s^{-1}}$)
$\tau_\mathrm{cool} \sim 10^5\,\mathrm{s}$, comparable to a sound-
crossing time over $\sim 1\,R_\odot$. At the chromospheric transition
region $\tau_\mathrm{cool}$ drops to $\sim 10^0\,\mathrm{s}$ —
dramatically sub-CFL — motivating operator splitting with implicit
/ exactly-integrable sub-cycles (Townsend 2009).

## Townsend 2009 closed-form integration

If $\Lambda(T) = \Lambda_0 (T/T_0)^\alpha$ on a power-law segment and
$n_e, \rho$ vary slowly (isochoric + slow-$\rho$ splitting), the
cooling ODE

$$\frac{\mathrm{d}T}{\mathrm{d}t} = -C\,T^\alpha,\quad
C \equiv (\gamma-1)(\mu m_u/k_B)\,n_e\,\Lambda_0/T_0^\alpha > 0$$

admits the closed-form solution (Townsend 2009 Eq. 26):

$$\boxed{T(t) = \bigl[\,T_0^{1-\alpha}
 - C(1-\alpha)\,t\,\bigr]^{1/(1-\alpha)}\qquad (\alpha \ne 1).} \quad (\text{C7-Townsend})$$

Numerically verified (21 samples across $\alpha \in \{-1, -\tfrac12, 0,
\tfrac12, \tfrac32, 2, 3\}$ and representative $(T_0, C, t)$) to
satisfy $\mathrm{d}T/\mathrm{d}t + C T^\alpha = 0$ to machine precision.

**Degenerate $\alpha = 1$ limit:**

$$T(t) = T_0\,\exp(-C t). \quad (\text{C7-exp})$$

**Kernel significance:** on any time-step $\Delta t_\mathrm{hyp}$
that spans many $\tau_\mathrm{cool}$, (C7-Townsend) gives the *exact*
sub-segment update — no sub-cycling required. This is the trick that
makes Athena++ / PLUTO cooling kernels run at hyperbolic CFL instead
of $0.1\,\tau_\mathrm{cool}$.

The full Townsend scheme handles crossing segment boundaries via a
**temporal evolution function** $Y(T)$ that monotonically maps
$T \mapsto$ integrated time; look-up + inversion stays closed-form.

## Energy-equation coupling

$$\partial_t E + \nabla\!\cdot\!\bigl[\text{(hydro fluxes)}\bigr]
 = -Q_R(T, \rho, Z). \quad (\text{C7-energy})$$

Cooling is a **pure sink**: $Q_R \ge 0$ subtracts from internal
energy. Sympy confirms $\mathrm{d}\ln\Lambda/\mathrm{d}\ln T = \alpha$
on each segment, so the Field 1965 isobaric stability criterion
reads

$$\partial_T\Lambda\bigr|_p < 0 \ \Longleftrightarrow\ \alpha < 2, \quad (\text{C7-field})$$

satisfied almost everywhere in $\log T \in [4.3, 7.0]$. This is the
root of **thermal instability** in the ISM: the warm neutral / cold
neutral bistability, and the Parker-like corona / chromosphere
thermal segregation.

## Explicit-source CFL (fallback, no Townsend)

When Townsend integration is disabled or the $\Lambda(T)$ table is not
piecewise-power-law, sub-cycle:

$$\Delta t_\mathrm{rad} \le \beta_\mathrm{rad}\,\tau_\mathrm{cool},\qquad
\beta_\mathrm{rad} \approx 0.1. \quad (\text{C7-CFL})$$

$\beta_\mathrm{rad} = 0.1$ is the Townsend 2009 calibrated safety
margin; $> 0.2$ gives $\gtrsim 1\%$ errors in thermal-front
propagation (SD93 shock tube benchmarks).

## Metallicity scaling

SD93 $\Lambda(T, Z)$ tables decompose as

$$\Lambda(T, Z) = \Lambda_\mathrm{H,He}(T) + (Z/Z_\odot)\,\Lambda_\mathrm{metals}(T).$$

Two-parameter bilinear table lookup $\Lambda(\log T, \log Z)$
implemented as a 64×16 device texture; cost negligible once built
(Suzuki+25 approach). For $Z < 10^{-4}\,Z_\odot$ (Pop III) the
metal term is $\lesssim 10^{-3}$ of the total — atomic-H-only
cooling tables (Anninos+1997) suffice instead.

## Implementation recipe

1. **Tabulate $\Lambda(\log T, \log Z)$** at build time or load from
   disk; store on GPU as 2D texture.
2. **Each step**: compute $T$ from primitive (requires μ from §C4).
3. **Cooling sub-stage**: either (a) Townsend closed-form on the
   segment, or (b) explicit sub-cycle at $\beta_\mathrm{rad}\tau_\mathrm{cool}$.
4. **Apply** $\Delta E = -Q_R \cdot \Delta t$ after the hyperbolic
   step. No change to $\rho, \mathbf{v}, \mathbf{B}$ — the cooling
   is pure internal-energy relaxation.
5. **Safety floor**: clip $T \ge T_\mathrm{floor}$ (e.g., $T_\mathrm{floor} = 0.7 T_\mathrm{eff}$
   in Suzuki+25) to prevent numerical runaway on under-resolved
   thermal fronts. The floor is a **known bias**, not a bug —
   document it in the driver log.

## ✅ Verification checkpoints

- `tests/test_athena_mhd_cooling_townsend.cu` — uniform box,
  $\Lambda(T) = \Lambda_0 (T/T_0)^{-1/2}$ (bremsstrahlung), fixed
  $\rho$. Assert $T(t)$ matches (C7-Townsend) at $t =
  \{0.1, 1, 10\}\,\tau_\mathrm{cool,0}$ to $<10^{-10}$ relative
  error.
- `tests/test_athena_mhd_cooling_segment_cross.cu` — initial $T_0$
  straddles two power-law segment boundaries; lock that the
  integrated $T(t)$ matches piecewise Townsend to $<10^{-8}$.
- `tests/test_athena_mhd_cooling_energy_floor.cu` — pathological
  $T_0 < T_\mathrm{floor}$, assert $T$ sticks at floor and no NaN.
- `tests/test_athena_mhd_cooling_field_instability.cu` — seed an
  isobaric $T$ perturbation in a Field-unstable regime
  ($\alpha = -1$); lock linear growth rate to analytic $\sigma =
  |\alpha - 2|/(2\tau_\mathrm{cool,0})$ within $5\%$.

The Field-instability test is the second-law certificate: any
regression that drops the $Q_R$ sign will mistake cooling for heating
and wreck this test first.
