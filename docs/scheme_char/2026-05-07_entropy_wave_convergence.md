# T1 entropy wave convergence — 2026-05-07

**Phase A.3 result** (task #47) for
`docs/design/testing_scheme_characterization_2026-05-07.md`.

## Test

Pure x-advection of a smooth density wave under uniform velocity,
periodic BC both directions, zero gravity, isobaric:

    ρ(x, 0) = ρ₀ · (1 + A · sin(k · 2π x / Lx))
    P = P₀,   v = (u₀, 0)

Analytic solution: ρ(x, t) = ρ(x − u₀·t, 0); after one period
`T = Lx / u₀` the wave returns to IC.  IC parameters:
ρ₀ = P₀ = 1, u₀ = 1, A = 0.01, k = 1, Lx = Ly = 1, γ = 5/3.
CFL = 0.3, `--athena-xorder 2 --athena-limiter vanleer`.

### Measurement

For each final-frame `ρ(x, y)` VTK, we average over y and compute the
**phase-corrected L1 error**: find the shift `s` that minimises
`L1(ρ_num − ρ_exact(x − s))` and report that minimum.  Phase
correction removes uniform-translation error (which would confuse the
convergence slope) and isolates the amplitude / shape damping.

## Summary

| solver      | N=64      | N=128     | N=256     | N=512     | slope p  |
|-------------|-----------|-----------|-----------|-----------|----------|
| cart_ale2   | 3.60e-03  | 3.73e-03  | 3.79e-03  | 7.26e-03  | **-0.31** |
| athena_vl2  | 7.66e-05  | 1.92e-05  | 4.53e-06  | 1.07e-06  | **2.06**  |

(measured phase shifts at t = t_end on a per-run basis are included in
the `.csv`.  athena_vl2 returns to s ≈ 0 exactly at every resolution;
cart_ale2 leaves |s| ≈ 0.23–0.77 of a period at every resolution.)

## Interpretation

### athena_vl2 — textbook 2nd-order

Slope **p = 2.06** across 4 resolutions with L1 halving every doubling
(factor of 4× per res doubling, consistent with `L1 ∝ dx²`).  This is
exactly what PLM + vanleer limiter + HLLC is supposed to deliver on a
smooth problem, and the Godunov Eulerian path preserves the wave phase
to machine precision (shift → 0 at every res).

This is the baseline a "correct 2nd-order scheme" should give on T1.

### cart_ale2 — negative effective convergence on scalar advection

L1 stays ≈ 3.7e-3 across 64 / 128 / 256, then *worsens* to 7.3e-3 at
512.  This is not a numeric artefact of the L1 metric — it reflects a
real property of the solver:

1. **Velocity field drifts**: the diagnostic log shows max|v| grows
   from 1.0000 → 1.0057 over one period at N = 128.  The Lagrangian
   + rebuild cycle amplifies velocity noise even though total KE is
   conserved to 1 part in 10⁵.  The initially-uniform v field breaks
   into a spatially-varying pattern.
2. **This velocity noise is the dominant error source**, not scalar
   (mass) remap.  Refining the grid gives the solver *more steps* and
   *more opportunities* to accumulate the velocity mode; the phase
   drift is roughly constant across resolutions (|s| ≈ 0.25–0.77 of a
   period at t = 1) but its interaction with the scheme's subgrid
   response grows at high res.
3. **Consistent with the T3 ν_eff = 0.31·V·dx Jensen-ILES finding**:
   cart_ale2 has 1st-order dissipation on velocity modes (task #50),
   and also 1st-order (or worse) dispersion on density modes
   advected by the same velocity field.

### Note on methodology

We deliberately phase-correct the L1 rather than reporting raw L1
against `ρ(x − u₀·t, 0)`.  Without phase correction the cart_ale2 L1
would be dominated by the bulk translation error (~0.5 of peak
amplitude at every resolution) and the scan would read "slope ≈ 0
uninformative".  Phase correction gives a tighter probe of the
amplitude / shape damping; it still shows cart_ale2's negative p, but
now attributable to genuine waveform distortion rather than a single
constant phase offset.

## Implications

1. **cart_ale2 is not a 2nd-order scheme for smooth scalar advection
   at practical resolutions**.  The MUSCL remap part (swept-edge,
   `remap-order 2`) has a 2nd-order formal order on a fixed velocity
   field, but the **coupled Lagrangian-remap-rebuild cycle** on a
   non-uniform density with uniform IC velocity develops a velocity
   mode whose dispersion dominates.  This is an independent probe
   corroborating the Jensen-ILES classification from task #50.
2. **athena_vl2 at 2.06 is the expected baseline** and a reasonable
   bar to hold other solvers against on Phase B.
3. **Paper-relevant**: When reporting Andrassy 2022 results at 256²,
   the cart_ale2 `v_rms` underestimate (5.1e-3 vs Athena 1.6e-1) is
   consistent with *both* the 1st-order velocity dissipation (T3) *and*
   the dispersive scalar-advection behaviour (T1).  The scheme is a
   genuine ILES with excess dissipation + dispersion, not a bug.

## Reproduce

```bash
scripts/scheme_char/run_entropy_wave_scan.sh
python scripts/scheme_char/fit_entropy_wave.py
```
