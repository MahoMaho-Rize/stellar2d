# C2. Ambipolar diffusion (ion-neutral drift)

> **sympy script:** `scripts/c2_ambipolar_dissipation.py`
> **verified:** $\mathbf{J}_\perp\cdot\hat{\mathbf{B}}=0$;
> $\mathbf{J}_\|\times\mathbf{B}=0$; $(\mathbf{J}\times\hat{\mathbf{B}})\times\hat{\mathbf{B}} = -\mathbf{J}_\perp$;
> $(\mathbf{J}\times\mathbf{B})\times\mathbf{B}/|\mathbf{B}|^2 = (\mathbf{J}\times\hat{\mathbf{B}})\times\hat{\mathbf{B}}$;
> $Q_{\mathrm{amb}} = \eta_A |\mathbf{J}_\perp|^2 \ge 0$.
> **code checkpoints:** `athena_mhd_kernels.cu::d_ambipolar_flux`,
> `tests/test_mhd_ambipolar_bmin.cu`.

## Motivation

In a partially-ionised plasma, **neutrals** do not couple directly to
$\mathbf{B}$; they feel the Lorentz force only via ion-neutral
collisions. When the collision rate $\nu_{in}$ is slower than MHD
timescales, ions drift relative to neutrals at

$$\mathbf{v}_{\mathrm{drift}} = \mathbf{J}\times\mathbf{B}/(\rho_i\rho_n\nu_{in}),$$

yielding the non-ideal electric field

$$\mathbf{E}_{\mathrm{amb}} = \eta_A\,(\mathbf{J}\times\hat{\mathbf{B}})\times\hat{\mathbf{B}} = -\eta_A\,\mathbf{J}_\perp, \quad (\text{C2-Eamb})$$

with $\eta_A \equiv |\mathbf{B}|^2/(\rho_i\rho_n\gamma_{in})$.

## Parallel / perpendicular current selectivity

**Critical property**: only $\mathbf{J}_\perp$ is dissipated by
ambipolar; the parallel current $\mathbf{J}_\|$ flows freely along
$\hat{\mathbf{B}}$:

$$\mathbf{J}_\|\times\mathbf{B} = \mathbf{0}. \quad (\text{C2-selective})$$

## Kernel-friendly form

For implementation, the non-unit-vector form is preferred:

$$\boxed{\mathbf{E}_{\mathrm{amb}} = \eta_A\,\frac{(\mathbf{J}\times\mathbf{B})\times\mathbf{B}}{|\mathbf{B}|^2},} \quad (\text{C2-altform})$$

avoids dividing by $|\mathbf{B}|$ in the normalisation step — matters
for low-$|\mathbf{B}|$ robustness (chromosphere bottoms).

## Induction equation

$$\partial_t\mathbf{B} = \nabla\times(\mathbf{v}\times\mathbf{B}) + \nabla\times\!\left[\eta_A\frac{(\mathbf{J}\times\mathbf{B})\times\mathbf{B}}{|\mathbf{B}|^2}\right]. \quad (\text{C2-induction})$$

**Non-linear in $\mathbf{B}$** — key difference from Ohmic.

## Ambipolar heating

$$\boxed{Q_{\mathrm{amb}} = \eta_A |\mathbf{J}_\perp|^2 = \eta_A\frac{|\mathbf{J}|^2|\mathbf{B}|^2 - (\mathbf{J}\cdot\mathbf{B})^2}{|\mathbf{B}|^2} \ge 0.} \quad (\text{C2-Q})$$

## CFL

$$\Delta t \le \frac{(\Delta x)^2}{2\eta_A |\mathbf{B}|^2/\rho}\ \mathrm{(per direction)}. \quad (\text{C2-CFL})$$

Scales with $|\mathbf{B}|^2$ — tight CFL in strong-field regions.

## ✅ Verification

`tests/test_mhd_ambipolar_bmin.cu` — $B_x$ uniform, sinusoidal $B_y$
with $\eta_A = $ const. Lock decay rate against analytical; also
verify that when $B_y \parallel B_x$ (i.e., $\mathbf{J}\parallel\mathbf{B}$),
$Q_{\mathrm{amb}} = 0$ and $B_y$ does not decay. Catches any
accidental scalar-diffusion substitution for the tensor form.
