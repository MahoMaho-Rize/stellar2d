# B3. Parker critical point on a super-radial flux tube

> **sympy script:** `scripts/b3_parker_critical_point.py`
> **verified:** Parker wind equation derivation; classical spherical
> limit $r_c = GM_*/(2 c_s^2)$.
> **code checkpoints:** `tests/test_mhd_parker_wind.cu`.

## Parker wind equation

Steady isothermal flow on a super-radial tube $(A(r) = r^2 f(r))$ with
$p = c_s^2 \rho$ yields, after eliminating $\rho$ via mass conservation:

$$\boxed{\left(v - \frac{c_s^2}{v}\right)\frac{dv}{dr}
= c_s^2\,\frac{d\ln A}{dr} - \frac{GM_*}{r^2}.} \quad (\text{B3-Parker})$$

**Sympy derived** from mass + momentum + EOS.

## Critical (sonic) point

At $v = c_s$ the LHS vanishes, forcing

$$c_s^2\!\left(\frac{2}{r_c} + \frac{d\ln f}{dr}\bigg|_{r_c}\right) = \frac{GM_*}{r_c^2}. \quad (\text{B3-critical})$$

**Spherical limit $f \equiv 1$:** $r_c = GM_*/(2c_s^2)$ (classical
Parker radius). Sympy verifies this by solving (B3-critical) analytically.

## Asymptotic velocity

Far from $r_c$ (spherical limit), $v(r) \sim c_s\sqrt{4\ln(r/r_c) + \text{const}}$
— logarithmic growth, characteristic of the Parker isothermal wind.

## ✅ Verification

`tests/test_mhd_parker_wind.cu` — isothermal, no magnetic driver, set
$T = 2\times10^6$ K, $M_* = 1 M_\odot$, measure Mach number crossing
at $r = r_c$. Lock $|M(r_c) - 1| < 10^{-3}$.
