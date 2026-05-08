# B1. Super-radial flux-tube reduction

> **sympy script:** `scripts/b1_flux_tube_geometry.py`
> **verified:** flux conservation $AB_r = \text{const}$; MHSE
> $\partial_r p + \rho g = -B_r^2\partial_r(\ln A)$;
> Kopp-Holzer $f(R_*) = 1$, $f(\infty) = f_{\max}$.
> **code checkpoints:** `tests/test_mhd_wind_hse_stationary.cu`.

## Geometry

$A(r) = r^2 f(r)$, $f(R_*) = 1$, $f(\infty) = f_{\max}$. Kopp-Holzer
form:

$$f(r) = \frac{f_{\max}\,e^{(r-R_1)/\sigma} + f_1}{e^{(r-R_1)/\sigma}+1},\quad f_1 = 1 - (f_{\max}-1)e^{(R_*-R_1)/\sigma}.$$

## 1D flux-tube equations

$$\partial_t(\rho A) + \partial_r(\rho v_r A) = 0,$$
$$\partial_t(\rho v_r A) + \partial_r[A(\rho v_r^2 + p_{\text{tot}} - B_r^2)] = (p_{\text{tot}} - B_r^2)\partial_r A - \rho g A,$$
$$\partial_t(EA) + \partial_r[A((E+p_{\text{tot}})v_r - B_r(\mathbf{B}\cdot\mathbf{v}))] = -\rho g v_r A + AQ.$$

## Flux conservation

$$\boxed{A(r)B_r(r) = R_*^2 B_0 \Rightarrow B_r(r) = \frac{B_0}{f(r)}(R_*/r)^2.}$$

## MHSE

$$\boxed{\partial_r p + \rho g = -B_r^2\,\partial_r(\ln A).}$$

**IC gotcha**: for a Suzuki-style wind run, atmospheres must be
integrated from this MHSE, NOT plain HSE. Using HSE triggers a
$\sim 10^{-2}$ transient and corrupts $\dot M$ measurements.

## WKB amplitude (see B2)

$\delta v_\perp \propto (\rho v_A A)^{-1/2}$ (Poynting) or
$(\rho v_A A)^{-1/4}$ (per-mode amplitude convention).

## ✅ Verification

`tests/test_mhd_wind_hse_stationary.cu` — HSE atmosphere, 100
acoustic crossings, lock $\max|v_r|/c_s < 10^{-4}$.
