# B2. WKB wave action conservation for Alfvén waves

> **sympy script:** `scripts/b2_wave_action_wkb.py`
> **verified:** dispersion $\omega^2 = v_A^2 k^2$; no-reflection limit
> under uniform $\rho_0, B_{r,0}$.
> **code checkpoints:** `athena_mhd_kernels.cu::d_alfven_wave_driver`,
> `tests/test_mhd_wind_amplitude_scaling.cu`.

## Linearised transverse MHD

Around a static background $(\rho_0(r), B_{r,0}(r))$ with $v_{r,0} = 0$:

$$\partial_t v_{\perp} = \frac{B_{r,0}}{\rho_0}\partial_r B_{\perp},\qquad
\partial_t B_{\perp} = B_{r,0}\partial_r v_{\perp}. \quad (\text{B2-linear})$$

WKB ansatz gives the Alfvén dispersion

$$\boxed{\omega^2 = v_A^2 k^2,\qquad v_A \equiv B_{r,0}/\sqrt{\rho_0}.} \quad (\text{B2-disp})$$

**Sympy verified** that the determinant of the 2×2 linearised matrix
equals $-\omega^2 + v_A^2 k^2$, giving the dispersion as the unique
non-trivial mode.

## Elsässer variables

$z_\pm \equiv v_\perp \mp B_\perp/\sqrt{\rho_0}$ decouple the outgoing
and incoming Alfvén waves:

$$\partial_t z_\pm \pm v_A\,\partial_r z_\pm = S_{\mathrm{refl}}\,z_\mp, \quad (\text{B2-Elsasser})$$

with a reflection source $S_{\mathrm{refl}}$ that depends on the
background gradients. In a uniform $(\rho_0, B_{r,0})$, sympy
verifies $S_{\mathrm{refl}} = 0$ — no wave reflection in a uniform
atmosphere (Alfvén waves are pure one-way in the uniform limit).

## Wave action conservation

Outgoing $z_+$ in the absence of reflection conserves the wave action

$$\boxed{\rho_0 A\,|z_+|^2/v_A = \text{const along characteristics.}} \quad (\text{B2-action})$$

For the amplitude-scaling law of a driver BC (relevant for Suzuki
winds), **Poynting flux** $\rho_0 v_A A |\delta v_\perp|^2$ conservation
gives

$$|\delta v_\perp| \propto (\rho_0 v_A A)^{-1/2}, \quad (\text{B2-Poynting})$$

while a **per-mode** amplitude normalisation gives the Jacques-1977
form

$$|\delta v_\perp| \propto (\rho_0 v_A A)^{-1/4}. \quad (\text{B2-Jacques})$$

The two differ because the former fixes the transmitted energy flux
while the latter fixes the wave amplitude in Elsässer space. Suzuki
wind codes use **the Poynting-flux convention** at the photospheric
driver (amplitude $\langle\delta v_\perp\rangle\approx 1.25$ km/s in
Shimizu+22) — make sure the kernel matches.

## ✅ Verification

`tests/test_mhd_wind_amplitude_scaling.cu` — linearised Alfvén wave
from a fixed BC driver. Measure $\delta v_\perp^{\mathrm{rms}}(r)$ at
$r = 5, 10, 20\,R_*$; lock deviation from $(\rho v_A A)^{-1/2}$
at $< 5\%$.
