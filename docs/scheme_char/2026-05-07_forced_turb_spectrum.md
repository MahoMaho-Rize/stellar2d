# T6 forced-turb spectrum (smoke) — 2026-05-07

**Phase B** (task #48, third slice) for
`docs/design/testing_scheme_characterization_2026-05-07.md`.

## What this is — and is not

This is the **end-to-end pipeline test** that our T6 spectrum-measurement
machinery works: `pseudo_spectral --test forced_turb` → VRAM-buffered
binary VTK → `scripts/pseudo_spectral/spectrum_fit_pseudo_spectral.py`
→ compensated E(k) plot + inertial-range fit.

It is **not** a paper-grade Kraichnan inverse-cascade result.  For that
we would need longer `t_end` (≥ 20 eddy turnovers, not 5), lower drag
(≤ 1e-3, not 0.1, so the inverse cascade has room to develop), and
probably 512² resolution.  That's Phase C scope per the design doc.

## Configuration

```
--nr 256 --ntheta 256 --ps-Lx 1 --ps-Ly 1 --ps-nu 5e-5
--ps-forcing-eps 0.1 --ps-forcing-kf 32 --ps-forcing-dk 1
--ps-drag 0.1 --ps-hyper 1 --tend 5 --cfl 0.3
```

Forcing injects ε = 0.1 in a shell at `k = 32` (physical `k_phys ≈ 201`).
2D Kraichnan predicts inverse cascade `k⁻⁵/³` for `k < 201` and forward
enstrophy cascade `k⁻³` for `k > 201`.

## Result (last frame, t = 5)

- ν = 5.9e-5, ε = 9.4e-2, η = 1.2e-3, k_η ≈ 818
- Inverse cascade range `[6.28, 161]` (25 bins):
  - slope = **−0.38 ± 0.19**, R² = 0.15 (theoretical −5/3; 6.8σ off)
- Enstrophy cascade range: too few bins (k ∈ [201, 818] but grid-scale
  dissipation kills the top end quickly)

The slope is flat / weakly declining rather than −5/3 because 5 s is
too short for the inverse cascade to feed down from k=32 to the
domain-scale modes, and drag = 0.1 is too large — it damps the
large-scale side before the cascade establishes.

## What this does validate

1. **Frame capture + FFT + binning pipeline works** — the tool
   produces E(k) with the expected number of bins and the forcing peak
   is visible at k = 32.
2. **Forced-turb driver is numerically stable** for 5 s at 256² with
   forcing ε = 0.1 (KE settles ≈ 0.07, ε_KE ≈ 0.09 matches input).
3. **Hook is in place for paper-grade T6 measurements** — a longer
   drag-free run with `--ps-drag 1e-3 --tend 20` on 512² would produce
   a fittable inertial range using the same script path.

## Artifacts

- `2026-05-07_forced_turb_spectrum.png` — E(k) with inverse / enstrophy
  fit lines overlaid
- `scripts/scheme_char/run_spectra_smoke.sh` — reproduces both the run
  and the fit

## Next

For the next round (Phase C):
- Re-run 256² at `--ps-drag 1e-3 --tend 20` (≈ 2× wall time of this
  run); inverse cascade should establish.
- Re-run 512² same config; verify k⁻³ enstrophy cascade becomes visible
  (enstrophy range k ≈ [201, 1638]).
- Compare measured slopes against `scripts/pseudo_spectral` reference
  runs from `docs/dns_*` (they already include Kraichnan checks).
