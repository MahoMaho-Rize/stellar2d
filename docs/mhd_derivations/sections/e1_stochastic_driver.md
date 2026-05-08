# E1. Stochastic broadband photospheric driver

> **sympy script:** `scripts/e1_stochastic_driver.py`
> **verified:** $\int_{\omega_{{\min}}}^{\omega_{{\max}}} (A^2/\omega)\mathrm{d}\omega = A^2 \ln(\omega_{{\max}}/\omega_{{\min}})$;
> target-variance normalisation $A^2 = \langle\delta v^2\rangle/\ln(\omega_{{\max}}/\omega_{{\min}})$;
> single-sinusoid time-variance $\langle\sin^2\rangle_t = 1/2$;
> cross-frequency terms vanish in time average (Parseval);
> linearised Elsässer $\partial_t z^\pm \pm v_A\partial_r z^\pm = 0$
> (pure one-way advection around uniform background);
> zero-gradient BC on $z^-$ forces incoming amplitude to zero
> (absorbing / no reflection in WKB limit).
> **code checkpoints:**
> `athena_mhd_kernels.cu::d_photospheric_driver`,
> `athena_mhd_kernels.cu::d_absorbing_bc_zminus`,
> `tests/test_athena_mhd_driver_spectrum.cu`.

## Why a broadband stochastic driver

Suzuki-group solar / red-giant wind simulations require continuous
wave injection at the inner boundary to replace the granulation-driven
photospheric convection. The driver must

1. **Carry the correct Poynting flux** so that the corona is heated
   to observed $T \sim 10^6\,\mathrm{K}$.
2. **Span a broadband spectrum** because phenomenological sub-grid
   turbulence (§C5) needs a realistic range of interacting
   frequencies to cascade.
3. **Be stochastic in phase** so that coherent resonances with the
   global domain modes do not build up.
4. **Pair with an absorbing BC for the incoming Alfvén wave** so
   that coronal material reflecting back down does not artificially
   amplify the driver.

## Driver form (Shimizu+22 eq. 38 / 42; Suzuki+25 Sec 2.8)

The transverse Elsässer variable outgoing from the photosphere is
driven as

$$\boxed{z^+_{\perp,\odot}(t) = A_\perp\,\sum_{N=0}^{N_{{\max}}}
\frac{\sin(2\pi f_N t + \varphi_N)}{\sqrt{f_N}},\qquad
\varphi_N \sim U[0, 2\pi).} \quad (\text{E1-driver-transverse})$$

The longitudinal radial velocity is driven independently:

$$\delta v_{\parallel,\odot}(t) = A_\parallel\,\sum_{N=0}^{N_{{\max}}}
\frac{\sin(2\pi f_N^\parallel t + \varphi_N^\parallel)}{\sqrt{f_N^\parallel}}. \quad (\text{E1-driver-long})$$

**Log-spaced sampling** mimics the continuous $\omega^{-1}$ spectrum:

$$f_N = f_{\min}\,(f_{\max}/f_{\min})^{N/N_{{\max}}},\qquad f_{\max} = 100\,f_{\min}. \quad (\text{E1-logspace})$$

**Suzuki-group calibration**:
- Transverse (Alfvén): $f_{\min} \approx 10^{-3}\,\mathrm{Hz}$, $f_{\max} \approx 10^{-2}\,\mathrm{Hz}$
  (16.7 min – 100 s);
- Longitudinal (p-mode): $f_{\min}^\parallel \approx 3.33\times 10^{-3}\,\mathrm{Hz}$,
  $f_{\max}^\parallel \approx 10^{-2}\,\mathrm{Hz}$ (5 min – 100 s).

## Power spectrum and normalisation

The $1/\sqrt{f_N}$ amplitude weighting reproduces a **flat-power-per-
log-octave** spectrum: $P(\omega) \propto 1/\omega$. Sympy-verified:

$$\int_{\omega_{{\min}}}^{\omega_{{\max}}} \frac{A^2}{\omega}\,\mathrm{d}\omega
= A^2 \ln(\omega_{{\max}}/\omega_{{\min}}). \quad (\text{E1-spectrum})$$

Normalising to the observed target rms fluctuation (Suzuki+25 solar
calibration: $\langle\delta v_\perp\rangle_\odot = 1.25\,\mathrm{km/s}$;
RGB $\alpha$ Boo: $2.50\,\mathrm{km/s}$ via $\delta v \propto (T_\mathrm{eff}^4/\rho)^{1/3}$):

$$\boxed{A^2 = \frac{\langle\delta v^2\rangle}{\ln(\omega_{{\max}}/\omega_{{\min}})},\qquad
\int P(\omega)\,\mathrm{d}\omega = \langle\delta v^2\rangle.} \quad (\text{E1-norm})$$

## Parseval variance identity

For a sum of sinusoids with distinct frequencies and independent
phases:

$$\bigl\langle\bigl[\sum_N A_N \sin(\omega_N t + \varphi_N)\bigr]^2\bigr\rangle_t
= \tfrac{1}{2}\,\sum_N A_N^2. \quad (\text{E1-parseval})$$

Sympy-verified symbolically on the time integral over a common
period: single-sinusoid variance is $A^2/2$, cross-terms integrate
to zero. Combining with $A_N = A/\sqrt{f_N}$ gives

$$\sum_N A_N^2 = A^2 \sum_N 1/f_N \approx A^2 \ln(f_{\max}/f_{\min})$$

(discrete approximation of the continuous integral).

The factor of $1/2$ means **the driver must be pre-multiplied by
$\sqrt{2}$** to hit $\langle\delta v^2\rangle$ — this is the source of
the explicit $\sqrt{2}$ factor in the Shimizu+22 driver (their footnote
to Eq. 41).

## Random phases — why essential

If $\varphi_N$ are deterministic, the driver is periodic with period
$T_{\min} = 2\pi/\gcd(\omega_N)$ (for rational $\omega_N$ ratios) or
quasi-periodic. Modes commensurate with the domain resonate — in the
solar wind problem this shows up as spurious peaks in the coronal
temperature spectrum.

**Phase regeneration frequency.** Draw new $\varphi_N$ each time one
simulation-correlation time elapses; practical choice
$\Delta t_\varphi \sim 10/f_{\min}$. Alternative: draw once per run
with a fixed random seed, document the seed in run metadata (Suzuki+25
approach — makes individual runs reproducible).

## Absorbing BC for incoming Alfvén

The coronal run generates downward-propagating Alfvén modes that must
not pile up at the photosphere. The physically-correct boundary
condition is the **free-outgoing** form in Elsässer variables:

$$\boxed{\partial_r z^-_\perp\bigr|_{r=R_*} = 0.} \quad (\text{E1-BC})$$

For a plane wave $z^-(r, t) = Z_0\,e^{i(kr - \omega t)}$, this forces
$ik\,Z_0 = 0$ — i.e., $Z_0 = 0$ (sympy-verified via `solve`). In WKB:
**zero reflection**, all incoming wave energy is absorbed.

**Kernel form.** Apply zero-gradient copy on the incoming Elsässer
variable at the inner ghost cells:

$$z^-_\perp(r=R_*-\mathrm{ghost}) \leftarrow z^-_\perp(r=R_*+1\text{ cell}).$$

Outgoing $z^+_\perp$ is clamped to the driver value (E1-driver-transverse).
Both together decouple the driver from the interior reflection.

## Elsässer propagation check (background dynamics)

Around a uniform static background $(\rho_0, B_{r,0})$, the linearised
transverse equations reduce to **pure one-way advection** of the
Elsässer variables:

$$\partial_t z^\pm \pm v_A\,\partial_r z^\pm = 0,\qquad v_A = B_{r,0}/\sqrt{\rho_0}. \quad (\text{E1-advect})$$

Sympy-verified directly by substituting $z^\pm = v_\perp \mp B_\perp/\sqrt{\rho_0}$
into the linearised induction + momentum equations.

Consequence: the driver at $r = R_*$ couples *only* to $z^+$;
corrections from finite background gradients (refraction, reflection,
mode conversion) appear at $\mathcal{O}(\Delta r/\lambda_A)$ and are
**part of the physical solution**, not BC artifacts. This is the
content of Parker (1965) and Hollweg (1981) WKB treatments.

## Driver implementation recipe

```c
// One-time initialisation at kernel start-up
srand(seed);
for (N = 0; N < N_max; ++N) {
    f[N] = f_min * pow(f_max/f_min, double(N)/N_max);   // log-spaced
    phi[N] = 2*M_PI * rand01();                          // uniform U[0,2π)
}
double A_perp = sqrt(2) * <dv_rms> / sqrt(log(f_max/f_min));

// Per time-step at inner ghost
double zplus = 0.0;
for (N = 0; N < N_max; ++N) {
    zplus += A_perp * sin(2*M_PI*f[N]*t + phi[N]) / sqrt(f[N]);
}
// Absorbing BC: copy interior value onto incoming Elsässer ghost
zminus_ghost = zminus_interior_1cell;
// Reconstruct v_perp, B_perp from (z+, z-)
v_perp = 0.5*(zplus + zminus_ghost);
B_perp = 0.5*sqrt(rho_0) * (zminus_ghost - zplus);
```

Key points:
- **Compute $A_\perp$ once**, not per-step; the log-factor is fixed.
- **$\sqrt{2}$ is important**; Parseval demands it.
- **Absorbing BC copies one cell inward** (not two, not zeroth-order
  extrapolation). Two-cell copy is fine but a zeroth-order
  extrapolation will produce a small-amplitude reflection.
- **Low-frequency end $f_{\min}$ sets the run length floor**: one
  $f_{\min}^{-1}$ period must fit within the simulation, else the driver
  is aliased.

## Special case: acoustic-only (no transverse driver)

Shimizu+22 Appendix A runs "B0V06" with $A_\perp = 0$ to isolate the
contribution of longitudinal acoustic waves to coronal heating via
mode-conversion. Set $A_\perp = 0$ and drive only $\delta v_\parallel$
at the 5-min p-mode band. This probes the chromospheric Alfvén
generation rate via the $\partial_r\ln p$ coupling term (§B2 Elsässer
source $S_\mathrm{refl}$).

## Per-star calibration scaling

For an arbitrary star, the solar calibration is scaled via Shimizu+22
Eq. 40:

$$\langle\delta v_0\rangle \propto (T_\mathrm{eff}^4 / \rho_0)^{1/3},\qquad
\omega_{{\max}}^{-1} \propto c_{s,0}\,R_*^2/M_*\ (\text{photospheric transit}),$$

with solar anchors $\langle\delta v_0\rangle_\odot = 1.25\,\mathrm{km/s}$,
$(\omega_{{\max}}^{-1})_\odot = 0.3\,\mathrm{min}$. Numbers in Suzuki+25
Table 3:

- $\alpha$ Boo: $\langle\delta v_0\rangle = 2.50\,\mathrm{km/s}$,
  $\omega_{{\max}}^{-1} = 150\,\mathrm{min}$, $\omega_{{\min}}^{-1} = 1.5\times 10^4\,\mathrm{min}$.
- $\alpha$ Tau: $\langle\delta v_0\rangle = 2.56\,\mathrm{km/s}$,
  $\omega_{{\max}}^{-1} = 340\,\mathrm{min}$, $\omega_{{\min}}^{-1} = 3.4\times 10^4\,\mathrm{min}$.

## ✅ Verification checkpoints

- `tests/test_athena_mhd_driver_spectrum.cu` — build the driver
  time-series over $10^6 f_{\min}^{-1}$, compute the FFT-measured
  spectrum, assert $\log P$ vs $\log f$ has slope $-1 \pm 0.05$
  within the driven band.
- `tests/test_athena_mhd_driver_variance.cu` — compute
  $\langle[z^+]^2\rangle$ over a multi-period sample; assert within
  $5\%$ of target $\langle\delta v^2\rangle$.
- `tests/test_athena_mhd_absorbing_bc.cu` — inject a Gaussian
  down-going Alfvén pulse, measure the reflected-amplitude
  coefficient; lock $R < 10^{-4}$ (WKB floor).
- `tests/test_athena_mhd_driver_reproducibility.cu` — same seed
  produces byte-identical driver output across runs; reseed gives
  uncorrelated phases.

The absorbing-BC test is the non-trivial one: a wrong
ghost-extrapolation order shows up here as a visible reflection pulse
but would pass every other spectrum / variance test silently.
