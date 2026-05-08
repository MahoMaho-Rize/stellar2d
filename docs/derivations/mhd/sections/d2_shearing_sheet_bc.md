# D2. Shearing-sheet approximation and shearing-periodic BC

> **sympy script:** `scripts/d2_shearing_sheet_bc.py`
> **verified:** shearing-sheet background is steady state (Coriolis +
> tidal cancel on $v_\phi^{\mathrm{bg}}$); shear wrap-around time
> $t_{\mathrm{shear}} = L_y/(q\Omega_0 L_x)$; effective potential
> is anti-confining.
> **code checkpoints:** `athena_mhd_shearingbox_bc.cu`,
> `tests/test_mhd_shearingbox_static.cu`.

## Hill expansion

Local Cartesian patch centred at $(R_0, \Omega_0)$, co-rotating with
the disk. Linearised rotational velocity:

$$\boxed{v_\phi^{\mathrm{bg}}(x) = -q\,\Omega_0\,x,\qquad q \equiv -\frac{d\ln\Omega}{d\ln R}}. \quad (\text{D2-shear})$$

$q = 3/2$ for Keplerian disks, $q = 2$ for rigid rotation.

## Frame-rotation sources

Coriolis:
$$\mathbf{F}_{\mathrm{Cor}} = -2\,\Omega_0\hat{z}\times\mathbf{u} = 2\Omega_0(u_y\hat{x} - u_x\hat{y}).$$

Tidal (Hill):
$$\mathbf{F}_{\mathrm{tidal}} = 2q\Omega_0^2\,x\,\hat{x} = -\nabla\Phi_{\mathrm{eff}},\qquad \Phi_{\mathrm{eff}} = -q\Omega_0^2 x^2.$$

**Sympy verified** that on the background $\mathbf{u}_{\mathrm{bg}} = (0, -q\Omega_0 x, 0)$,
$F_{\mathrm{Cor},x} + F_{\mathrm{tidal},x} = 0$. The shearing-sheet
background is an exact steady state of the force-balance.

## Shearing-periodic BC

$$\boxed{\mathrm{field}(x = L_x, y, z, t) = \mathrm{field}(x = 0, y - q\Omega_0 L_x t, z, t).} \quad (\text{D2-BC})$$

The $y$-shift $\Delta y(t) = q\Omega_0 L_x t$ grows linearly in time.
After $t_{\mathrm{shear}} = L_y/(q\Omega_0 L_x)$ the shift wraps to
$L_y$ and the BC becomes pure periodic again.

Under periodic-$y$ convention, the BC is always well-defined: take
$\Delta y\ \mathrm{mod}\ L_y$. **Sympy verified** $t_{\mathrm{shear}}$.

## Effective potential is anti-confining

$$\frac{d^2\Phi_{\mathrm{eff}}}{dx^2} = -2q\Omega_0^2 < 0,$$

so the tidal potential alone is unstable — **the Coriolis force is
what stabilises the shearing sheet**. Sympy-verified; documented to
flag that dropping the Coriolis kernel would immediately disperse
the disk.

## ✅ Verification

`tests/test_mhd_shearingbox_static.cu` — initial condition
$\mathbf{u} = (0, -q\Omega_0 x, 0)$, purely hydro, no perturbation.
Lock max perturbation energy $<10^{-6}\times E_0$ over $10/\Omega_0$.
Any drift means Coriolis or tidal coupling is mis-implemented.
