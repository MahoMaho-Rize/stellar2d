# F2. 2D MHD turbulence spectrum and $\nu_\mathrm{eff}$ extraction

> **sympy script:** `scripts/f2_mhd_turbulence_spectrum.py`
> **verified:** K41 log-log slope $-5/3$; IK65 slope $-3/2$;
> dissipation cutoff $k_\mathrm{diss} = (\epsilon/\nu_\mathrm{eff}^3)^{1/4}$;
> inversion $\nu_\mathrm{eff} = (\epsilon/k_\mathrm{diss}^4)^{1/3}$;
> scheme-order scaling $k_\mathrm{diss}(N) \propto N^{3/2}$ for
> $\nu_\mathrm{eff} \propto h^2$.
> **code checkpoints:**
> `scripts/analyze_orszag_tang_spectrum.py` (driver-external analysis);
> `docs/projects/mhd_verification/phase_A_results.md` — table entries.

## Motivation

The Orszag-Tang (OT) vortex, run to $t = 0.5$, develops a fully
nonlinear MHD turbulent cascade. Its energy spectrum $E(k)$ provides
the **direct, resolution-independent** measurement of our solver's
effective viscosity $\nu_\mathrm{eff}$ — the unique number that
determines whether our 2D MHD turbulence runs are resolving the
inertial range of interest.

Without this derivation, the A2 analysis script has no basis to
interpret the spectrum cutoff or claim a quantitative $\nu_\mathrm{eff}$.

## Kolmogorov (K41) and Iroshnikov-Kraichnan (IK65)

Two competing predictions for the 2D MHD inertial-range spectrum:

$$\boxed{E_K(k) = C_K\,\epsilon^{2/3}\,k^{-5/3},\qquad
E_{IK}(k) = C_{IK}\,(\epsilon\,v_A)^{1/2}\,k^{-3/2}.} \quad (\text{F2-K41},\text{F2-IK})$$

The slopes differ:
- **K41**: hydrodynamic Kolmogorov-Obukhov cascade, applies when
  kinetic and magnetic energies are approximately equipartitioned and
  the cascade is local in $k$-space.
- **IK65**: Iroshnikov-Kraichnan, applies when strong Alfvén-wave
  collisions dominate the cascade; the extra factor $v_A^{1/2}$
  encodes the Alfvén-wave crossing time.

Both slopes are sympy-verified via symbolic log differentiation.

For OT at $t = 0.5$ the literature consensus (Dahlburg-Picone 1989,
Politano-Pouquet 1989, Biskamp-Welter 1989) places the **observed**
slope between $-5/3$ and $-3/2$, closer to the K41 value.

## Dissipation cutoff and $\nu_\mathrm{eff}$ inversion

Below the viscous scale, the cascade is truncated by dissipation.
Classical Kolmogorov dissipation scale:

$$k_\mathrm{diss} = \bigl(\epsilon/\nu_\mathrm{eff}^3\bigr)^{1/4}. \quad (\text{F2-kdiss})$$

Inverting, given a *measured* $k_\mathrm{diss}$ from the spectrum:

$$\boxed{\nu_\mathrm{eff} = \bigl(\epsilon/k_\mathrm{diss}^4\bigr)^{1/3}.} \quad (\text{F2-nu-inv})$$

Sympy-verified: (F2-kdiss) and (F2-nu-inv) are each other's
functional inverse.

## Scheme-order consistency

For a 2nd-order finite-volume scheme with $\nu_\mathrm{eff} \propto \Delta x^2$:

$$\nu_\mathrm{eff}(N) = C_\mathrm{visc}/N^2,\qquad
k_\mathrm{diss}(N) = (\epsilon/C_\mathrm{visc}^3)^{1/4}\,N^{3/2}. \quad (\text{F2-scaling})$$

Sympy-verified: $\mathrm{d}\log k_\mathrm{diss} / \mathrm{d}\log N = 3/2$.

**Doubling the resolution** should shift $k_\mathrm{diss}$ by a factor
$2^{3/2} = 2.83$.  This is the A2 consistency check:
$k_\mathrm{diss}(256) / k_\mathrm{diss}(128) \in [2.0, 3.5]$.

## Measurement protocol (A2 test)

1. Run OT at $N \in \{128, 256, 512\}$ with `init_orszag_tang()` to
   $t = 0.5$.
2. Dump VTK of $(v_x, v_y, B_x, B_y)$ at $t = 0.5$.
3. FFT each to 2D $(k_x, k_y)$; compute 1D axisymmetric spectrum
   via shell averaging:

   $$E(k) = \tfrac{1}{2}\sum_{k-1/2 < |\mathbf{k'}| \le k+1/2} \bigl(|\hat v|^2 + |\hat B|^2\bigr).$$

4. Identify inertial range: sliding window of log-log slope; pick
   $k_\mathrm{iner}$ where slope is stable over 1 decade.
5. Identify dissipation cutoff: smallest $k$ where $E(k)$ drops to
   $< 10^{-3}$ of its inertial-range peak (or use the break in
   slope from $-5/3$ to exponential fall).
6. Compute $\nu_\mathrm{eff}$ via (F2-nu-inv) using measured
   $\epsilon$ (energy flux from $-\mathrm{d}E_\mathrm{tot}/\mathrm{d}t$
   at $t = 0.5$) and $k_\mathrm{diss}$.
7. Verify scheme-order scaling (F2-scaling) between N=128→256 and
   N=256→512.

## Pass criteria (A2 test)

1. **Inertial-range slope at N=256**: fitted log-log slope in
   $[-1.8, -1.4]$ over a continuous decade in $k$. Slope outside
   this range = wrong physics or cascaded too weakly.
2. **$k_\mathrm{diss}$ scaling**:
   $k_\mathrm{diss}(256)/k_\mathrm{diss}(128) \in [2.0, 3.5]$ and
   $k_\mathrm{diss}(512)/k_\mathrm{diss}(256) \in [2.0, 3.5]$.
3. **$\nu_\mathrm{eff}(N)$ table entry** written to the Phase A
   results document (no pass/fail judgement; the number itself is
   the deliverable).

## Why this matters for later Suzuki physics

Estimated at our expected parameters ($\epsilon \sim 0.1$, $\nu_\mathrm{eff}
\sim 10^{-4}$ at $N=128$, extrapolated to $\sim 10^{-5}$ at $N=512$):

| $N$ | $\nu_\mathrm{eff}$ | Re$_\mathrm{num}$ | inertial range (decades) |
|---|---|---|---|
| 128 | $\sim 10^{-4}$ | $\sim 10^4$ | ≈ 1.2 |
| 256 | $\sim 2\times 10^{-5}$ | $\sim 3\times 10^4$ | ≈ 1.6 |
| 512 | $\sim 6\times 10^{-6}$ | $\sim 10^5$ | ≈ 2.0 |

A Suzuki-type 2D Alfvén-turbulence extension needs
$\mathrm{Re}_m \gtrsim 10^3$ to set up a realistic inertial range.
**At N=256 or above**, this condition is met. Below N=256 the
cascade is marginally resolved; above N=512 we are firmly in the
inertial regime.

This is the single most important quantitative output of Phase A.

## ✅ Verification checkpoints

- `scripts/analyze_orszag_tang_spectrum.py` implements the protocol
  above and writes to `phase_A_results.md`.
- Solver correctness is already verified by the existing
  `test_athena_mhd_benchmarks.cu::test_orszag_tang` (smoke +
  $\nabla\cdot\mathbf{B}$ lock); A2 is an **analysis**, not a new
  solver test.

If the spectrum violates pass criterion (1), the likely cause is a
broken HLLD branch at low-$\beta$ (OT has $\beta \sim 0.01$ late in
time); if (2) fails, the effective viscosity doesn't scale like a
2nd-order scheme, indicating a time-stepping bug (check VL2
corrector wraps in A8).
