# C3. Total-energy equation with non-ideal MHD

> **sympy script:** `scripts/c3_resistive_energy.py`
> **verified:** ideal Poynting flux identity; $Q_{\mathrm{Ohm}} = \eta_O|J|^2$;
> $Q_{\mathrm{amb}} = \eta_A|J_\perp|^2$ (via Lagrange's identity).
> **code checkpoints:** `athena_mhd_kernels.cu::d_total_energy_flux`.

## Total-energy equation

With both Ohmic + ambipolar, total energy is still conserved; only
the Poynting flux gets a non-ideal addition:

$$\boxed{\partial_t E + \nabla\cdot\!\left[(E+p^\star)\mathbf{v} - \mathbf{B}(\mathbf{v}\cdot\mathbf{B}) + \mathbf{E}_{\mathrm{ni}}\times\mathbf{B}\right] = 0,} \quad (\text{C3-energy})$$

with $\mathbf{E}_{\mathrm{ni}} = \eta_O\mathbf{J} + \eta_A(\mathbf{J}\times\mathbf{B})\times\mathbf{B}/|\mathbf{B}|^2$.

## Internal-energy source

The non-ideal dissipation becomes internal-energy heating:

$$\boxed{\partial_t(\rho e) + \nabla\cdot(\rho e\,\mathbf{v}) = -p\nabla\cdot\mathbf{v} + Q_{\mathrm{Ohm}} + Q_{\mathrm{amb}},} \quad (\text{C3-internal})$$

$$Q_{\mathrm{Ohm}} = \eta_O|\mathbf{J}|^2,\qquad
Q_{\mathrm{amb}} = \eta_A|\mathbf{J}_\perp|^2
= \eta_A\frac{|\mathbf{J}|^2|\mathbf{B}|^2 - (\mathbf{J}\cdot\mathbf{B})^2}{|\mathbf{B}|^2}.$$

**Sympy verified** the Lagrange identity $|\mathbf{J}\times\mathbf{B}|^2
= |\mathbf{J}|^2|\mathbf{B}|^2 - (\mathbf{J}\cdot\mathbf{B})^2$ which
maps between the two Q_amb forms.

## Sign convention

$Q = -\mathbf{E}_{\mathrm{ni}}\cdot\mathbf{J}$ is the rate at which EM
energy is dissipated **into the fluid's internal energy** — positive
for dissipative $\mathbf{E}_{\mathrm{ni}}$. Both Ohmic and ambipolar
give $Q > 0$.

## Kernel structure

- **Total energy**: evolved via conservative flux (C3-energy).
  Non-ideal Poynting flux $\mathbf{E}_{\mathrm{ni}}\times\mathbf{B}$
  added to the hydro energy flux.
- **No direct $Q$ kernel needed**: internal-energy source is
  implicitly accounted for by the flux divergence of the non-ideal
  Poynting term. Explicit computation of $Q$ only needed for
  diagnostics.

## ✅ Verification

`tests/test_mhd_ohmic_energy_conservation.cu` — uniform $\mathbf{B}_0$
with sinusoidal $B_y$ perturbation + $\eta_O = $ const. Measure total
energy drift over 100 decay timescales. Lock $|\Delta E/E_0| < 10^{-8}$ —
**any leak means the non-ideal Poynting flux term is missing from
the energy flux**.
