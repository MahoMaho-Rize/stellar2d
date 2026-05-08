# MHD base-solver assessment

Goal: pick a single existing `stellar2d` solver to extend for ideal MHD
(HLLD + 8 conserved vars), eventually non-ideal (Ohmic + ambipolar).
Priority: 1-2D time-domain solver that reproduces Suzuki-group work
(flux-tube Alfvén-wave winds, shearing-box MRI, cool-star chromospheres).

What MHD adds in general:
- `B = (Bx, By, Bz)` extra conserved variables (2.5D keeps all three even
  in 2D; wind flux-tube is 1.5D: radial flow + Bφ).
- `∇·B = 0` constraint: constrained transport (CT, face-staggered B,
  Evans-Hawley 1988 + corner EMF of Gardiner-Stone 2005) or hyperbolic
  Dedner cleaning on cell-centered B.
- Riemann solver: HLLD (Miyoshi-Kusano 2005) replaces HLLC. Same interface
  (L/R primitive states in → flux out) so the swap is mechanical.
- Extra CFL term: Alfvén wave speed `c_A = |B|/√(4πρ)` added to
  `max(|u|+c_fast)`. Trivial.

## Per-solver assessment

### `strang_solver` — 2D Strang-split HLLC+MUSCL (`src/gpu/explicit/strang_solver.{cuh,cu}`)
- Eulerian cartesian, cell-centered, ng=2, perturbation form `(δρ, mx, my, δE)`.
- Dimension-split: X(dt/2) → Y(dt) → X(dt/2); each sweep = MUSCL-Hancock
  predictor + LM-HLLC Riemann.
- MHD add: Strang-split MHD with CT is *awkward* — CT requires unsplit EMF
  averaging at corners (Gardiner-Stone 2005); dimensional splitting couples
  poorly to face-staggered B. Dedner is easier to bolt on but introduces
  an 8th var + a cleaning wave speed. Perturbation form conflicts with
  standard conservative MHD formulation.
- Difficulty: **4/5**. Verdict: *usable but suboptimal; Athena is the
  natural home for HLLD+CT.*

### `athena_vl2_solver` — 2D Athena++ VL2 port (`src/gpu/explicit/athena_vl2_solver.{cuh,cu}`)
- Eulerian cartesian, cell-centered, ng=3, SoA conserved `(ρ, mx, my, E, s)`.
- Van-Leer 2-stage *unsplit* predictor-corrector (Stone-Gardiner 2009):
  `calc_hydro_flux(order=1) → u* → calc_hydro_flux(order=2) → u^{n+1}`.
- Face-flux buffers already separate per direction (`d_Fx_*`, `d_Fy_*`),
  matching Athena's MHD layout. Passive-scalar machinery is a template
  for adding new advected fields.
- MHD add: **direct port path**. Athena++ src/mhd/rsolvers/hlld.cpp is a
  ~400-line drop-in replacement for hllc.cpp; `src/field/field.cpp` holds
  face-staggered B and CT EMF averaging (Gardiner-Stone 2005). The unsplit
  VL2 is *exactly* what Athena's MHD uses — no architectural changes.
- Difficulty: **2/5**. Verdict: *clear winner; port Athena MHD source
  files alongside existing Athena hydro scaffolding.*

### `cart_ale2_solver` — 2D Cartesian ALE (`src/gpu/ale/cart_ale2_solver.{cuh,cu}`)
- Lagrangian (Caramana subcell forces) + Eulerian rezone + swept-edge
  remap. Node-staggered velocity, cell-centered (ρ, e, P, X).
- No Riemann solver (artificial viscosity). MHD in ALE requires Lagrangian
  MHD (Caramana-Barlow-Shashkov) — a whole research programme, not a port.
- Difficulty: **5/5**. Verdict: *abandon; Lagrangian MHD is a multi-year
  side project with no Suzuki-style precedent.*

### `pseudo_spectral_solver` — 2D incompressible NS (`src/gpu/spectral/pseudo_spectral_solver.{cuh,cu}`)
- Vorticity-streamfunction, cuFFT R2C, IFRK3, 2/3 dealias, double-periodic.
- MHD add: 2D incompressible MHD in (ω, ψ, A, j) form IS a known formulation
  (2D reduced MHD = Strauss; ∇·B = 0 automatic via `B = ∇ × (A ẑ)`).
  Clean for turbulence/MRI-lite but useless for compressible Alfvén-wave
  winds (Suzuki's main physics is transonic + compressible).
- Difficulty: **3/5** for 2D reduced MHD only. Verdict: *good for RMHD
  turbulence studies, wrong physics for wind problem.*

### `radial1d_solver` — 1D Lagrangian radial hydro (`src/gpu/radial1d/radial1d_solver.{cuh,cu}`)
- Face-staggered (r, v) + zone-centered (ρ, e, P). Tscharnuter-Winkler AV,
  RK2 explicit + JFNK implicit BE. Already carries MLT, radiation, species.
- MHD add: 1D spherically-symmetric MHD is trivial for Bφ (flux freezing
  `d(r·Bφ)/dt = r·Bφ·(∂v/∂r − v/r)`); Suzuki-style 1.5D wind (v_r, v_φ,
  Bφ) fits naturally into Lagrangian faces. But NO Riemann solver exists
  here — would replicate Suzuki 2006 AV-style Lagrangian MHD, which is
  its own thing (works, but is a different numerical lineage).
- Difficulty: **3/5** for 1.5D Suzuki wind. Verdict: *viable for winds
  only; bad fit for 2D MRI.*

### `anelastic_sl_solver` — 2D anelastic spectral (`src/gpu/spectral/anelastic_sl_solver.{cuh,cu}`)
- Reduced-pressure Poisson solve via SL diagonalisation. Anelastic filters
  sound waves.
- MHD add: anelastic MHD filters fast magnetosonic waves — wrong regime
  for wind launching (Alfvén speed typically >> cs in chromosphere).
- Difficulty: **5/5**. Verdict: *wrong physics; skip.*

## Recommendation: extend `athena_vl2_solver`

Open **new** solver files (`athena_mhd_solver.{cuh,cu}` +
`athena_mhd_kernels.cu` per CLAUDE.md §"不可覆盖"), reusing Athena++'s
MHD source tree as the reference port target. 5 reasons:

1. **Cleanest Riemann interface**. `calc_hydro_flux(order)` already
   dispatches L/R primitive states into HLLC via a single device function.
   Replacing with HLLD is localised to one kernel file; the SoA flux
   buffers (`d_Fx_mx`, `d_Fy_E`, …) scale straightforwardly to 8-variable
   MHD by adding `d_Fx_Bx, d_Fx_By, d_Fx_Bz` etc. The primitive/conservative
   conversion pair `cons_to_prim()` has the same shape Athena MHD uses.

2. **Cleanest ∇·B story**. VL2 is the *exact* integrator Athena uses for
   its unsplit MHD with constrained transport (Stone+08 Appendix). The
   CT EMF averaging (Gardiner-Stone 2005) lives at cell corners, orthogonal
   to face-flux buffers — no architectural retrofit needed. Face-staggered
   B can be added as four new device arrays parallel to the existing
   `d_Fx_*` layout. Dedner cleaning is also available as a fallback
   (8th cell-centered field + wave speed).

3. **1D reduction for Suzuki winds is trivial**. Set `ny=1` with symmetric
   ghosts in y, keep only the x-sweep, add `Bφ` and `v_φ` as rotational
   (azimuthal) momentum components — standard 1.5D MHD wind setup
   (Velli 1994; Suzuki-Inutsuka 2006). Spherical-geometry source terms
   slot into `add_source_terms()` which already handles variable `g(y)`
   and volumetric heating `q̇(y)`.

4. **Existing test + CI infrastructure**. `compute_entropy_wave_error`,
   `compute_sod_error`, `compute_acoustic_wave_error` establish the
   convergence-test pattern; MHD analogs (Brio-Wu shock tube, circularly
   polarised Alfvén wave, Orszag-Tang vortex) drop into the same harness.
   Andrassy-2022 IC machinery + VTK output carry over unchanged.

5. **Community precedent is overwhelming**. Athena → Athena++ MHD is the
   canonical compressible-MHD stellar-wind / MRI code (used by Suzuki,
   Stone, Gardiner, Bai, many others). Porting their hlld.cpp and
   field.cpp CT module into our Athena-parity VL2 solver is the shortest
   route from ideal-MHD to nonideal: Athena++ already ships Ohmic +
   ambipolar diffusion modules (`src/hydro/diffusion/`) that plug into
   the same `add_source_terms` step the current solver uses.

## Relevant paths
- `/home/yujian_shi/stellar2d/src/gpu/explicit/athena_vl2_solver.cuh`
- `/home/yujian_shi/stellar2d/src/gpu/explicit/athena_vl2_solver.cu`
- `/home/yujian_shi/stellar2d/src/gpu/explicit/athena_vl2_kernels.cu`
- `/home/yujian_shi/stellar2d/src/gpu/radial1d/radial1d_solver.cuh` (future
  1.5D wind fallback if we want Lagrangian-AV MHD rather than Eulerian HLLD)
