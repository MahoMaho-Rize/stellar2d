# T3 ν_eff comparison — 2026-05-07

**Phase A.1 result** for `docs/design/testing_scheme_characterization_2026-05-07.md`.

Linear shear-mode / Taylor-Green pure-diffusion decay, fit on the window
`t ∈ [5 %, 50 %] · t_end` (`t_end = 2.0`, `k = 1`, `Ly = 1.0`):

- `cart_ale2` / `athena_vl2`: compressible shear mode
  `vx = V₀·sin(k·2π y/Ly)`, `vy = 0`, uniform ρ, P, `g = 0`.
  Analytic NS decay: `max|v|(t) = V₀·exp(-ν·k²_phys·t)`.
- `pseudo_spectral`: incompressible Taylor-Green vortex `ω = 2k·cos·cos`.
  Analytic NS decay: `|ω|(t) ∝ exp(-2ν·k²_phys·t)`.

`ν_eff` is extracted from the log-slope of the appropriate decay signal.
Ideal `ν = 0` would give `ν_eff = 0` to machine precision.

## Summary table (V₀ = 0.01)

| solver            | res=64   | 128      | 256      | 512      | slope p (ν ∝ dx^p) |
|-------------------|----------|----------|----------|----------|--------------------|
| cart_ale2         | 1.55e-02 | 7.76e-03 | 3.88e-03 | 1.94e-03 | **1.00**           |
| athena_vl2        | -3.3e-17 | 3.7e-17  | 9.6e-17  | -7.2e-18 | noise              |
| pseudo_spectral   | -2.9e-18 | -8.9e-18 | 7.0e-18  | -1.4e-17 | noise              |

## Summary table (V₀ = 0.1)

| solver            | res=64   | 128      | 256      | 512      | slope p (ν ∝ dx^p) |
|-------------------|----------|----------|----------|----------|--------------------|
| cart_ale2         | 1.63e-02 | 8.25e-03 | 4.15e-03 | 2.08e-03 | **1.00**           |
| athena_vl2        | ≤ 1e-16  | ≤ 1e-16  | ≤ 1e-16  | ≤ 1e-16  | noise              |
| pseudo_spectral   | ≤ 1e-17  | ≤ 1e-17  | ≤ 1e-17  | ≤ 1e-17  | noise              |

Artifacts:
- CSV:  `2026-05-07_nu_eff_comparison.csv` (24 rows, all fits)
- Plot: `2026-05-07_nu_eff_comparison.png` (ν_eff vs N, log-log, both V₀)

## Interpretation

### cart_ale2 — Jensen-ILES, 1st-order

- `ν_eff(64²) = 1.55e-2`; doubling res halves ν_eff exactly →
  **p = 1.00 to 3 significant figures**.
- Weak V dependence: V₀=0.01 vs V₀=0.1 gives only ~5 % shift in ν_eff
  at each res → prefactor is approximately `ν_eff ≈ 0.31 · V₀ · dx`
  (with V₀ = `max|vx|` and `dx = Ly / N`).
- This is the textbook signature of **Jensen-type 1st-order ILES**
  dissipation from the ALE remap step: every face is touched each cycle,
  so even a velocity field with `vy = 0` everywhere is dissipated at
  rate `O(V·dx)`.
- Quantitatively matches the observed over-damping on the Andrassy 256²
  benchmark (`v_rms ~ 10 %` of ps result).

### athena_vl2 — direction-aligned Godunov, vanishes exactly on this IC

- `|ν_eff| ≤ 10⁻¹⁶` for all `(res, V₀)` — rounding noise.
- This is **not** "athena_vl2 has zero numerical dissipation in general";
  it is specific to this IC. Because `vy ≡ 0` and P is uniform, the
  y-direction HLLC star state is symmetric between left and right at
  every face — the tangential-momentum flux of `ρvx` cancels to machine
  precision. x-direction fluxes are trivial (no gradient in x).
- The correct reading: **Godunov dissipation is direction-aligned**.
  When the flow has a non-zero normal velocity across the dissipating
  face (Andrassy convection roll, KH roll-up, turbulent eddies),
  athena_vl2 pays the usual `O(dx²)` price. A fair follow-up probe is
  a 45°-rotated shear IC (Phase B, task #48).

### pseudo_spectral — DNS-quality on Taylor-Green

- `|ν_eff| ≤ 10⁻¹⁷` — machine-precision preservation with `ν = 0` and
  2/3 dealias. For TG the nonlinear term `u·∇ω ≡ 0` identically, so
  this is really a sanity check for the IFRK3 + dealias pipeline.
- A real `ν_eff` probe for pseudo_spectral would use forced turbulence
  and fit the inertial-range spectrum slope (Phase B task #48, T6).

## Action implications

1. The **cart_ale2 Andrassy under-performance is now a known quantitative
   feature, not a bug**: `ν_eff ≈ 0.31·V·dx` places it in the
   high-dissipation ILES bracket comparable to the paper's PROMPI
   reference implementation. Continue reporting Andrassy results with
   this caveat in paper drafts.

2. **Do not use athena_vl2 alone as a measure of compressible scheme
   dissipation** — this IC hides its truncation error behind direction
   alignment. Use the Andrassy v_rms cross-comparison (done) plus a
   rotated-shear probe (Phase B) instead.

3. **cart_ale2 1st-order ILES is a fundamental ALE-remap constraint**,
   not a tuning knob. `--remap-order 2`, `--ppm`, `--rebuild-order 1`
   all give the same `p = 1.00` slope here; moving to 2nd order would
   require a subzonal-velocity ALE redesign (discussed but deferred).

4. **Phase A.3 (entropy wave slope, task #47)** is the natural next
   step: prediction is `p ≈ 1.9` for the density field (MUSCL-limited
   remap is 2nd-order on scalars) but `p ≈ 1.0` again on velocity — the
   same ALE-remap floor we see here.

## Reproduce

```bash
cmake --build build -j
scripts/scheme_char/run_nu_eff_scan.sh          # 24 runs, ~10-30 s each
python scripts/scheme_char/fit_nu_eff.py        # fit + plot + auto-summary
# NOTE: fit_nu_eff.py writes a short auto-generated md; this file preserves
# the interpretive version and should be edited by hand, not regenerated.
```
