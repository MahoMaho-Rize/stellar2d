# C5. Sub-grid turbulent heating closure (Suzuki-Inutsuka 2005)

> **sympy script:** `scripts/c5_suzuki_turbulent_heating.py`
> **verified:** positivity $\varepsilon_\mathrm{turb} > 0$; pure-outward
> limit $|z^-|\to 0$ gives no cascade
> ($\varepsilon_\mathrm{turb}^\mathrm{SY}\to 0$); timescale bound
> $\Delta t \le \lambda_\mathrm{cor}/(2 c_d|\delta v_\perp|)$.
> **code checkpoints:**
> `athena_mhd_kernels.cu::d_turbulent_heating_source`,
> `tests/test_athena_mhd_turbulent_heating_positivity.cu`.

## Why this closure exists

The 1D super-radial flux-tube code (§B1) cannot resolve the 3D Alfvén
wave turbulence that transfers outward-going wave energy to heat in
the corona. Suzuki-Inutsuka 2005 (SI05) added a phenomenological
sub-grid heating term that dissipates transverse wave energy at a
prescribed cascade rate. This is the crucial ingredient without
which the 1D wind models severely **under-estimate** coronal
temperature.

## Suzuki-Inutsuka 2005 form

$$\boxed{\varepsilon_{\mathrm{turb}}^\mathrm{SI} = c_d\,\rho\,\frac{|\delta v_\perp|^3}{\lambda_{\mathrm{cor}}},\quad c_d \approx 0.1,\quad \lambda_\mathrm{cor}\sim 10^7\,\mathrm{cm}.}$$

Dimensions: $[\rho][\delta v]^3/[\lambda] = \mathrm{erg/cm^3/s}$ ✓.

## Shoda-Yokoyama 2016 (Elsässer) form

$$\varepsilon_\mathrm{turb}^\mathrm{SY} = \frac{\rho}{2\lambda_\mathrm{cor}}\left(|\mathbf{z}^+|^2|\mathbf{z}^-| + |\mathbf{z}^-|^2|\mathbf{z}^+|\right).$$

This form makes the **non-linear coupling** between counter-
propagating modes explicit. Limit $|z^-|\to 0$ gives
$\varepsilon_\mathrm{turb}^\mathrm{SY} \to 0$ (sympy-verified): no
cascade without cross-polarisation — an essential physical
consistency check.

## Energy-equation coupling

$$\partial_t E + \nabla\!\cdot\!\left[(E + P^\star)\mathbf{v} - \mathbf{B}(\mathbf{B}\!\cdot\!\mathbf{v})\right] = \varepsilon_\mathrm{turb}\ \ge 0.$$

**Positivity.** Sympy-verified $\varepsilon_\mathrm{turb} > 0$ for
positive $\rho, \delta v_\perp, \lambda_\mathrm{cor}, c_d$. The
kernel must not inadvertently create a negative-definite heating
source — any code review must check this.

## Timescale bound (explicit-source CFL)

Per-step heating must not exceed the available wave KE:

$$\varepsilon_\mathrm{turb}\cdot\Delta t \le \tfrac{1}{2}\rho|\delta v_\perp|^2\ \Longleftrightarrow\ \Delta t \le \frac{\lambda_\mathrm{cor}}{2\,c_d\,|\delta v_\perp|}.$$

In practice, for solar wind parameters
($|\delta v_\perp|\sim 30\,\mathrm{km/s}$, $\lambda_\mathrm{cor} \sim 10^8\,\mathrm{cm}$),
this bound is well above the hyperbolic CFL; no concern.

## Shimizu+22 scaling

$$\lambda_\mathrm{cor}(r) = \lambda_0 \sqrt{A(r)/A(R_*)},\quad \lambda_0 \approx 10^7\,\mathrm{cm}$$

(i.e., correlation length scales with the tube area because energy-
containing eddies fill the tube cross-section). Replication of
Shimizu+22 figures requires this scaling **and** $c_d = 0.1$ exactly
— the paper's calibration.

## Extraction of $\delta v_\perp$

$\delta v_\perp$ is defined in the **Lagrangian / comoving frame**:

$$|\delta v_\perp|^2 \equiv \langle (v_y - \bar{v}_y)^2 + (v_z - \bar{v}_z)^2\rangle,$$

with $\bar{v}$ a running time-average. In a 1D tube code with no
transverse degree of freedom, this can instead be derived from the
**Alfvén wave amplitude**

$$|\delta v_\perp|^2 = |z^+|^2 + |z^-|^2 - 2\mathbf{z}^+\!\cdot\!\mathbf{z}^-$$

using the Elsässer variables tracked directly from §B2.

## ✅ Verification checkpoints

- `tests/test_athena_mhd_turbulent_heating_positivity.cu` — random
  initial $(\rho, \delta v, \lambda)$ states, assert
  $\varepsilon_\mathrm{turb} \ge 0$ across 100 samples; no NaN.
- `tests/test_athena_mhd_suzuki_wind.cu` — full replication of
  SI05 Fig. 2 mass-loss rate $\dot M \sim 2\times 10^{-14}\,M_\odot/\mathrm{yr}$
  within ±10% (calibration tolerance per paper's own sensitivity to
  $c_d$). Requires well-balanced MHSE (§B4).
