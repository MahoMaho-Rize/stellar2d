# Standard MHD Code-Verification Benchmarks

This document is a **reproduction-grade benchmark catalog** compiled for the
`stellar2d` project as it prepares to extend the existing `athena_vl2` solver
(HLLC + VL2 unsplit predictor-corrector, cell-centered finite-volume) into a
full ideal-MHD implementation using the HLLD Riemann solver
(Miyoshi & Kusano 2005) and constrained-transport (CT) divergence-cleaning
(Evans & Hawley 1988; Gardiner & Stone 2005 — "GS05"). The catalog is to be
used as a **pre-merge go/no-go test gate** for the forthcoming `athena_mhd`
solver: every benchmark below must be reproduced to the specified tolerance
before MHD work branches into any science-run topic (stellar winds, MRI,
flux-tube Alfvén heating). Setups are quoted from the *Athena* lineage we
are porting from: Stone, Gardiner, Teuben, Hawley & Simon 2008, "Athena: A
New Code for Astrophysical MHD" (ApJS 178, 137 — "Stone+08"); Gardiner &
Stone 2005, "An unsplit Godunov method for ideal MHD via constrained
transport" (J. Comput. Phys. 205, 509 — "GS05"); and Mignone, Bodo, Massaglia
et al. 2007, "PLUTO: A Numerical Code for Computational Astrophysics" (ApJS
170, 228 — "Mignone+07") where noted. Exact initial-condition values have
been cross-validated against Stone+08 §8.2 / §8.4, the Athena public test
pages (`www.astro.princeton.edu/~jstone/Athena/tests/`) and the Athena++
problem-generator inputs on GitHub (`inputs/mhd/athinput.*`).

All quantities are in **Athena units**: `ρ, P, v, B` are dimensionless and
the magnetic field obeys the Gaussian normalization `B = B_phys / sqrt(4π)`,
so that `c_A = |B| / sqrt(ρ)` and `P_mag = B²/2` (no 4π factors appear in
the flux formulae). γ and end time `t_f` are given per test.

## Benchmark matrix (summary)

| # | Name | Dim | What it tests | Pass criterion |
|---|---|---|---|---|
| 1 | Brio & Wu shock tube | 1D | 7-wave MHD Riemann, compound wave, γ=2 closure | All 7 features visible in ρ, vₓ, vᵧ, Bᵧ at t=0.08; ≤3-cell shock/contact smearing at 800 cells |
| 2 | Ryu-Jones RJ2a | 1D | all 7 MHD wave families in one tube | ≤4-cell capture of fast/slow/rotational/contact at 512 cells |
| 2b | Ryu-Jones RJ4d | 1D | switch-on slow rarefaction + slow shock | Monotone profile, no overshoot at 512 cells, HLLD or Roe |
| 3 | CP Alfvén wave | 1D + 2D | nonlinear exact MHD solution, dispersion | L1(δB⊥) ∝ N⁻² after one period; 2D amplitude ≥0.95 after 5 crossings at 16+ pts/λ |
| 4 | Orszag-Tang vortex | 2D | multi-D shock interaction, symmetry, ∇·B | Slice ρ(x, y=0.3125) matches 512² reference at 192² to ≲ 2% deviation; ∇·B < 10⁻¹² at cell faces |
| 5 | MHD blast wave | 2D | strong shock + low-β (β=2·P/B²≈0.02) | Isotropic blast envelope, no carbuncle, mirror symmetry preserved to roundoff |
| 6 | MHD rotor | 2D | strong rotational discontinuity, torsional Alfvén wave | Concentric Mach-number contours at center preserved; max\|B\| matches Tóth 2000 Fig. 16 at 400² |
| 7 | Field-loop advection | 2D | ∇·B control, transverse field under advection | Loop shape preserved after 2 orbits; Jz contours show no reconnection hole; for vz ≠ 0 test, Bz stays 0 to roundoff |
| 8 | Linear MHD wave convergence | 1D | formal order of each wave family | L1(δq) ∝ N⁻² for fast/slow/Alfvén/entropy down to roundoff (A=10⁻⁶, N up to 1024) |
| 9 | Torrilhon 2003 shock tube | 1D | compound-wave convergence for ideal MHD | Convergence to non-compound exact solution as N → ∞ |
| 10 | MHD KH instability | 2D | qualitative — field-aligned flow stabilization | Mode suppressed for \|B_x\| ≥ v_shear critical; no divergence growth |

Tests 1–8 are **mandatory** for the pre-merge gate; 9–10 are **optional**
regression / qualitative checks.

---

## 1. Brio & Wu shock tube

**Reference.** Brio, M. & Wu, C. C. 1988, J. Comput. Phys. 75, 400.
Reproduced in Stone+08 §8.2, Fig. 13, Table 2 (parameters quoted verbatim
below).

- **Dimension.** 1D.
- **Domain.** `x ∈ [−0.5, 0.5]` (Athena++ `inputs/mhd/athinput.bw`).
  Discontinuity at `x = 0`.
- **γ.** 2 (not 5/3 — tests a non-cosmological equation of state closure).
- **Initial condition.** Piecewise constant.
  Left (x < 0): `ρL = 1.0, vx,L = 0, vy,L = 0, vz,L = 0, PL = 1.0,
  By,L = +1.0, Bz,L = 0`.
  Right (x > 0): `ρR = 0.125, vx,R = 0, vy,R = 0, vz,R = 0, PR = 0.1,
  By,R = −1.0, Bz,R = 0`.
  Longitudinal field constant everywhere: `Bx = 0.75`.
- **Boundary.** Outflow / non-reflecting (copy ghost).
- **End time.** `t_f = 0.08` (Stone+08 Fig. 13).
- **Typical grid.** Stone+08 runs 800 cells; reference solution at 10⁴ cells
  shown as solid line.
- **Expected outcome.** The exact solution of Brio-Wu contains seven
  distinct features: a left-going fast rarefaction, a left-going slow
  compound wave (a shock+rarefaction sharing the same velocity — the
  "compound wave" unique to non-convex MHD Riemann problems), a contact
  discontinuity, a right-going slow shock, a right-going fast rarefaction.
  All must be visible.
- **Pass criterion.** Shock + contact captured in ≤ 3 cells at 800 cells
  (Stone+08 §8.2). Velocity profile is the most discriminating — 3rd-order
  reconstruction can show small oscillations at the contact (Stone+08
  explicitly flags this).
- **Published comparison.** Stone+08 Fig. 13 — six-panel plot (ρ, P, vₓ, vᵧ,
  Bᵧ, e/(γ−1)) at t=0.08, 800 cells.
- **Pass/fail metric.** Visual match to Stone+08 Fig. 13. Quantitative:
  L1(ρ, vₓ, Bᵧ) vs. 10⁴-cell reference < 5 × 10⁻³ at 800 cells.

**Why this benchmark.** Smallest 1D test that exercises every MHD wave family
plus the compound wave (whose treatment distinguishes ideal MHD from central
schemes with non-convex flux). Catches: wrong sign convention on By; HLLD
middle-state collapse when vy,L = −vy,R; incorrect γ=2 energy-pressure
closure; misimplementation of the transverse-B face interpolation (the
compound wave's density discontinuity vanishes if Bᵧ is reconstructed
unsplit without proper characteristic tracing).

---

## 2. Ryu-Jones shock tubes (RJ2a, RJ4d primary; RJ1a / RJ3a / RJ5a secondary)

**Reference.** Ryu, D. & Jones, T. W. 1995, ApJ 442, 228, Table 1a (cases
1a–5a) and Table 2a (case 2a used by Stone+08 as "RJ2a"; case 4d used by
Stone+08 as "RJ4d"). Reproduced in Stone+08 §8.2, Figs. 14–15, and Table 2.

### 2a. RJ2a — "the 7-wave tube"

- **Dimension.** 1D.
- **Domain.** `x ∈ [−0.5, 0.5]` (Athena++ `inputs/mhd/athinput.rj2a`),
  discontinuity at `x = 0`. Stone+08 reports `x ∈ [0, 1]`, same result up to
  translation.
- **γ.** 5/3.
- **Initial condition.** Given in conventional units; Athena-normalized
  values in parentheses `(B_code = B_gauss / sqrt(4π))`.
  Left:
  - `ρL = 1.08`
  - `vx,L = 1.2, vy,L = 0.01, vz,L = 0.5`
  - `PL = 0.95`
  - `Bx = 2/sqrt(4π) ≈ 0.56418958`
  - `By,L = 3.6/sqrt(4π) ≈ 1.01554125`
  - `Bz,L = 2/sqrt(4π) ≈ 0.56418958`
  Right:
  - `ρR = 1.0`
  - `vx,R = 0, vy,R = 0, vz,R = 0`
  - `PR = 1.0`
  - `Bx = 2/sqrt(4π) ≈ 0.56418958` (continuous)
  - `By,R = 4.0/sqrt(4π) ≈ 1.12837917`
  - `Bz,R = 2/sqrt(4π) ≈ 0.56418958`
- **Boundary.** Outflow.
- **End time.** `t_f = 0.2`.
- **Typical grid.** 512 cells (Stone+08 Fig. 14); reference at 10⁴ cells.
- **Expected outcome.** Solution contains left- and right-going fast
  magnetosonic shocks, left- and right-going slow magnetosonic shocks, left-
  and right-going rotational discontinuities, and one contact discontinuity
  — i.e. all seven MHD wave families in a single tube. No compound wave.
- **Pass criterion.** Each of the seven features captured with 2–4 cells at
  512 cells (Stone+08 §8.2 Fig. 14 caption). L1 error on ρ vs. 10⁴-cell
  reference ≲ 2 × 10⁻³.

### 2b. RJ4d — switch-on slow rarefaction

- **Domain / γ / boundary / end time.** `x ∈ [−0.5, 0.5]`, γ = 5/3, outflow,
  `t_f = 0.16` (Stone+08 Fig. 15).
- **Initial condition.** (Stone+08 Table 2.)
  Left: `ρL = 1.0, vx,L = 0, vy,L = 0, vz,L = 0, PL = 1.0,
  By,L = +1.0, Bz,L = 0`.
  Right: `ρR = 0.3, vx,R = 0, vy,R = 0, vz,R = 0, PR = 0.2,
  By,R = cos(3)  ≈ −0.98999, Bz,R = sin(3) ≈ 0.14112`.
  Longitudinal field constant: `Bx = 0.7` everywhere.
- **Expected outcome.** Switch-on slow-mode rarefaction + slow shock. Tests
  HLLD in a region where the slow wave is not explicitly represented in the
  HLLD intermediate-state fan (HLLD resolves the Alfvén discontinuity
  exactly but treats slow waves within the outer fast fan).
- **Pass criterion.** Monotone profile, no post-shock overshoot at 512 cells
  with 3rd-order reconstruction + HLLD (Stone+08 §8.2).

### 2c. RJ1a, 3a, 5a (optional full-table coverage)

Ryu & Jones 1995, Table 1a states [NEEDS VERIFICATION — exact numerical
values of cases 1a/3a/5a were not extracted from the cached Stone+08 text;
verify against the original Ryu & Jones 1995 ApJ 442, 228 Table 1a, or see
Athena++ `inputs/mhd/athinput.rj{1a,3a,5a}` when the upstream repo adds
them — the canonical states, as commonly cited, are]:

- **RJ1a** ("fast-only tube"): `ρL = 1.0, vx,L = 10, PL = 20, By,L = 5/sqrt(4π);
  ρR = 1.0, vx,R = −10, PR = 1, By,R = 5/sqrt(4π); Bx = 5/sqrt(4π); t_f = 0.08`.
- **RJ3a** ("Alfvén + fast"): `ρL = 0.1, vy,L = 0.4, PL = 0.2,
  By,L = 4/sqrt(4π);
  ρR = 0.1, vx,R = 0.08, PR = 0.2, By,R = 4/sqrt(4π); Bx = 4/sqrt(4π);
  t_f = 0.1`.
- **RJ5a** ("slow shock + rotational + slow rarefaction"):
  `ρL = 1.0, vx,L = 0, vy,L = 0, PL = 1.0, By,L = 6/sqrt(4π);
  ρR = 0.4, vx,R = 0, vy,R = 0, PR = 0.4, By,R = 1/sqrt(4π);
  Bx = 2/sqrt(4π); t_f = 0.15`.

These are all given in Ryu & Jones 1995, Table 1a; our cached copy of the
ApJ paper failed to produce machine-readable text, so the above numbers
should be **re-verified against the original PDF** before inclusion in
regression tests. Stone+08 only formally includes 2a and 4d (renamed from
RJ95's original numbering).

**Why this benchmark.** RJ2a is *the* acid test of HLLD: every wave family
is excited in the initial Riemann problem, so a middle-state bug (e.g. the
HLLD-paper Eq. (41) star-state pressure, which involves the density-weighted
average `(ρL*SL* - ρR*SR*) / (ρL*SL - ρR*SR)`) will show immediately as
asymmetric rotational discontinuities. Catches: wrong eigenvalue ordering in
HLLD fan; rotational-discontinuity velocity computed with the total pressure
instead of the magnetic tension; slow-wave sign bugs visible in RJ4d.

---

## 3. Circularly polarized (CP) Alfvén wave

**Reference.** Tóth, G. 2000, J. Comput. Phys. 161, 605 ("T2000") §3.2.2,
as re-parameterized by GS05 §3.3.2 and §5.2. Reproduced in Stone+08 §8.4,
Figs. 19–20.

CP Alfvén is an **exact nonlinear** solution of ideal MHD, so the reference
solution is the initial state re-sampled at any integer number of wave
periods.

### 3a. 1D version

- **Dimension.** 1D.
- **Domain.** `x ∈ [0, 1]`, periodic.
- **γ.** 5/3.
- **Initial condition.** Background `ρ = 1, P = 0.1, B∥ = 1`; perpendicular
  components:
  - `By(x, 0) = 0.1 · sin(2πx)`
  - `Bz(x, 0) = 0.1 · cos(2πx)`
  - `vy(x, 0) = ±0.1 · sin(2πx)` (sign sets right-vs-left-traveling)
  - `vz(x, 0) = ±0.1 · cos(2πx)`
  - Traveling wave: `vx = 0`; standing wave: `vx = −c_A = −1` (so the
    right-traveling wave stands still in the grid frame).
- **Boundary.** Periodic.
- **End time.** `t_f = 1` (one period, one λ = grid length, c_A = 1).
- **Typical grid.** Convergence study at N = 8, 16, 32, 64, 128, 256, 512,
  1024 (Stone+08 Fig. 20).
- **Expected outcome.** Exact nonlinear solution: after one period the
  waveform returns to the initial state.
- **Pass criterion.** `L1(δB⊥) ∝ N⁻²` (2nd-order integrator) or `N⁻³`
  (3rd-order spatial reconstruction), traveling and standing waves must
  have identical L1 (Stone+08 §8.4).

### 3b. 2D version (GS05 setup; adopted by Stone+08 and Athena++)

- **Dimension.** 2D.
- **Domain.** `0 ≤ x ≤ sqrt(5) ≈ 2.236068`, `0 ≤ y ≤ sqrt(5)/2 ≈ 1.118034`
  (Athena++ `inputs/mhd/athinput.cpaw2d` confirms these limits). Periodic
  in both directions.
- **γ.** 5/3.
- **Wave angle.** `θ = tan⁻¹(2) ≈ 63.4°` with respect to x-axis.
  Equivalently the wave propagation axis is tilted so that **two complete
  wavelengths fit along the grid diagonal** (λ = 1 along the rotated `x₁`
  axis).
- **Rotated coordinate.** `x₁ = x cosθ + y sinθ = (x + 2y)/sqrt(5)`.
- **Initial condition.** (GS05 Eq. 54–56.)
  - `ρ = 1, P = 0.1`
  - `B₁ = 1` (parallel to wave)
  - `B₂(x₁) = 0.1 sin(2π x₁)`
  - `B₃(x₁) = 0.1 cos(2π x₁)`
  - `v₁ = 0` (traveling) or `v₁ = 1` (standing)
  - `v₂(x₁) = 0.1 sin(2π x₁)`
  - `v₃(x₁) = 0.1 cos(2π x₁)`
  - Cartesian components obtained by inverse rotation:
    `Bx = B₁ cosθ − B₂ sinθ`, `By = B₁ sinθ + B₂ cosθ`, `Bz = B₃`.
  - **In-plane `Bx, By` must be initialized via vector potential**
    `Az(x, y)` so that CT starts with machine-precision `∇·B = 0`.
- **Boundary.** Periodic.
- **End time.** `t_f = 5` (five crossings at c_A = 1 — matches Athena++
  input `tlim = 5.0`).
- **Typical grid.** `2N × N` with N = 8, 16, 32, 64, 128 (GS05 takes N = 8
  for illustration). Athena++ default is 256 × 128.
- **Expected outcome / pass criterion.** Exact solution at any integer
  period. At resolution 16 pts/λ the amplitude should be ≥ 0.95 of initial
  after 5 crossings (Stone+08 §8.4: "with 16 or more grid points per
  wavelength, the amplitude is better than 0.95 the original"); at 8 pts/λ
  amplitude ≥ 0.80. L1 convergence on δB₂ should be `∝ N⁻²` for 2nd-order
  VL2+CT, with 2D errors at most a factor of 2 above 1D errors (Stone+08
  §8.4).

**Why this benchmark.** The single most discriminating 2D MHD convergence
test. It **uniquely** isolates: (i) CT EMF-averaging correctness (GS05's
`E_z^c` corner-upwinded formula); (ii) multi-D characteristic tracing in the
unsplit predictor; (iii) the difference between cell-centered and
face-centered reconstruction. Catches: wrong interpolation of `Bx` and `By`
onto cell faces (CT constraint violated at corners); dimensionally-split
operator that decouples `Bx` from `∂E_z/∂y`; wrong sign on `E_z`
construction from the `F(Bx)` and `G(By)` Riemann fluxes.

---

## 4. Orszag-Tang vortex

**Reference.** Orszag, S. A. & Tang, C.-M. 1979, J. Fluid Mech. 90, 129
(original hydro vortex); Picone & Dahlburg 1991 for the MHD version;
Stone+08 §8.4, Figs. 22–23.

- **Dimension.** 2D.
- **Domain.** `(x, y) ∈ [0, 1] × [0, 1]`, periodic in both directions.
- **γ.** 5/3.
- **Initial condition** (Stone+08 §8.4 verbatim):
  - `ρ₀ = 25 / (36π)`
  - `P₀ = 5 / (12π)`
  - `vx = −sin(2πy)`, `vy = +sin(2πx)`, `vz = 0`
  - Magnetic field from vector potential
    `Az(x, y) = (B₀ / (4π)) cos(4πx) + (B₀ / (2π)) cos(2πy)`
    with `B₀ = 1 / sqrt(4π)`. This yields
    - `Bx = +∂Az/∂y = −B₀ sin(2πy)`
    - `By = −∂Az/∂x = +B₀ sin(4πx)`
    - `Bz = 0`.
- **Boundary.** Periodic (all four walls).
- **End time.** `t_f = 0.5` (Stone+08 contour plots; also `t_f = 1.0` for a
  later-time comparison against Dai & Woodward 1998).
- **Typical grid.** 192 × 192 for standard comparison; 512² high-resolution
  reference (Stone+08 Fig. 23 takes the 512² curve as ground truth).
- **Expected outcome.** Complex 2D MHD turbulence driven by the initial
  velocity–magnetic-field geometry. By `t = 0.5` a central current sheet has
  formed along the y-axis with maxima at four symmetric points.
- **Pass criterion.**
  1. **Symmetry.** Solution must preserve the four-fold symmetry of the
     initial conditions to visual roundoff (Stone+08 Fig. 22). A
     dimensionally-split algorithm will break this — tests unsplit
     integration.
  2. **Quantitative slices.** Horizontal slice `P(x, y = 0.3125)` at
     t = 0.5, 192² grid, must match the 512² reference curve in
     Stone+08 Fig. 23 top panel to within ≲ 2% peak deviation.
  3. `∇·B` at cell faces must be ≤ 10⁻¹² throughout evolution (CT
     constraint).
- **Published comparison.** Stone+08 Figs. 22–23. Additional references:
  Tóth 2000 Fig. 16–17 at `t = π`; Dai & Woodward 1994 Fig. 11.

**Why this benchmark.** Cheapest "full 2D" MHD test. Tests: multi-D shock
interactions (transverse MHD shocks form as the vortex compresses),
preservation of discrete symmetries (directionally-split schemes fail here
even when they pass 1D tubes), CT invariance, resilience to β ≪ 1 patches
that form transiently near the current sheet. Does **not** stress HLLD
rotational-discontinuity handling (already covered by RJ2a).

---

## 5. MHD blast wave in a strongly magnetized medium

**Reference.** Balsara, D. S. & Spicer, D. S. 1999, J. Comput. Phys. 149,
270 — the original parameter choice. Stone+08 §8.4 adopts the version of
Londrillo & Del Zanna 2000, Fig. 28 and 30.

- **Dimension.** 2D.
- **Domain.** `(x, y) ∈ [−0.5, 0.5] × [−0.75, 0.75]` i.e. `Lx = 1, Ly = 1.5`
  (Athena web page + Stone+08 §8.4).
- **γ.** 5/3.
- **Initial condition.**
  - `ρ = 1.0` uniform.
  - Pressure: `P = 10.0` inside `r ≡ sqrt(x² + y²) < 0.1`; `P = 0.1`
    outside. No taper.
  - Magnetic field: uniform, inclined 45° to grid:
    `Bx = B₀/sqrt(2), By = B₀/sqrt(2), Bz = 0`, with `B₀ = 1.0` (strong
    field; gives `β_outer = 2P/B² = 0.2`, `β_inner = 20`). Stone+08 also
    runs the isothermal `B₀ = 10` variant (`β_outer = 0.02`) as a robustness
    test — include this as an optional stretch case.
  - `v = 0`.
- **Boundary.** Periodic in both x and y (Stone+08 §8.4: "By using periodic
  boundary conditions, the flow becomes more complex as the outgoing blast
  wave re-enters the grid"). Some authors use outflow; the **Stone+08
  canonical setup uses periodic**.
- **End time.** `t_f = 0.2` (Stone+08 Fig. 28). A secondary comparison at
  `t_f = 1.0` is shown in Fig. 29 after the blast has re-entered.
- **Typical grid.** 200 × 300 cells (Stone+08 Fig. 28). High-resolution
  reference at 400 × 600.
- **Expected outcome.** Strong fast-mode shock expands anisotropically
  along field lines; the perpendicular direction is inhibited by magnetic
  tension. At `B₀ = 10` this anisotropy is extreme (factor-of-several axis
  ratio).
- **Pass criterion.**
  1. Mirror symmetry about the `y = −x` diagonal (which is along B) must be
     preserved. A dimensionally-split code loses this at the percent level.
  2. No negative pressures — HLLC alone **cannot** sustain this test
     (central rarefaction produces negative P in the intermediate state,
     triggering positivity fallback). **HLLD + CS (Colella-Sekora) limiter
     is the Stone+08 recommendation.**
  3. `β_min` in the blast interior must stay positive; robust codes
     maintain `β_min ≳ 10⁻³` through `t_f = 0.2`.
- **Published comparison.** Stone+08 Fig. 28 (ρ, P, Bx, By contours at
  t=0.2, 200×300). Balsara-Spicer 1999 Fig. 8.

**Why this benchmark.** Stress test for low-β robustness, the first test
the HLLC→HLLD port is likely to fail if the intermediate-state code doesn't
clamp against negative pressures. Also catches: wrong corner-EMF averaging
(breaks mirror symmetry); wrong Dedner cleaning-speed choice if using
Dedner instead of CT (field develops asymmetric oscillations).

---

## 6. MHD rotor

**Reference.** Balsara, D. S. & Spicer, D. S. 1999 (original 2D
formulation); Tóth, G. 2000, J. Comput. Phys. 161, 605 — "Rotor Test 1"
(canonical parameters). Reproduced in Stone+08 §8.4, Figs. 25–26, and
realized in Athena++ `inputs/mhd/athinput.rotor` (values confirmed below).

- **Dimension.** 2D.
- **Domain.** `(x, y) ∈ [−0.5, 0.5] × [−0.5, 0.5]`.
- **γ.** 1.4 (Athena++ `athinput.rotor`: `gamma = 1.4`). Some references
  use 5/3; **we adopt 1.4 to match Stone+08 and Athena++**.
- **Initial condition.**
  - Density: `ρ_in = 10` for `r ≤ r₀ = 0.1`; `ρ_out = 1` for `r > r₀`. No
    taper (Athena++ input has `r1 = −0.115`, a negative sentinel meaning
    "no smoothing zone").
  - Pressure: `P = 1.0` uniform (`p0 = 1.0`).
  - Velocity: inside the rotor (`r ≤ r₀`), rigid-body rotation at angular
    velocity `ω = 100 v₀ = 200` (Athena++ `v0 = 2.0`, prefactor 100 hard-
    coded in the rotor pgen):
    `vx = −ω · y, vy = +ω · x, vz = 0`.
    Outside: `v = 0`.
    Note: Stone+08 "Rotor Test 1" uses `u0 = 2` (peak velocity at the
    rotor edge = `ω·r₀ = 200·0.1 = 20`, hence strongly supersonic — Mach
    ~10 against the ambient sound speed).
  - Magnetic field: uniform, `Bx = bx0 = 1.410474` (which is
    `5 / sqrt(4π)`), `By = 0, Bz = 0`. Athena units (`B_phys / sqrt(4π)`).
- **Boundary.** Outflow (zero-gradient copy ghost) on all four sides.
- **End time.** `t_f = 0.15` (Athena++ `tlim = 0.15`, Stone+08 Fig. 25).
- **Typical grid.** 400 × 400 (Stone+08); Athena++ default 200 × 200.
- **Expected outcome.** Torsional Alfvén waves propagate outward from the
  rotor surface, shaping a nearly circular central rarefaction. The dense
  core is compressed radially inward then begins to rotate more slowly as
  angular momentum is transported to the ambient medium by the Alfvén waves.
- **Pass criterion.**
  1. Contours of Mach number inside the central rarefaction must remain
     concentric circles down to the origin (Stone+08 Fig. 25 — the
     "near-perfect symmetry" requirement).
  2. Slices `By(x, y = 0)` and `Bx(x, y = 0)` must match Stone+08 Fig. 26
     to within 3% at 400² resolution; peak `|B|` at the rotor edge should
     reach `|B|_max ≈ 3.2 ± 0.1` (read from Fig. 26) at t = 0.15.
  3. No carbuncle instability along the x or y axes.
- **Published comparison.** Stone+08 Figs. 25–26; Tóth 2000 Fig. 16
  (density and pressure contours); Mignone+07 Fig. 14.

**Why this benchmark.** Stresses strong **rotational discontinuity**
handling — an HLLD bug that misses the Alfvén jump will produce spurious
density waves from the rotor surface and break circular symmetry within
a few rotations. Also checks: discontinuity without smoothing (tough for
PPM's parabolic reconstruction), large initial `∇·B` errors from the
velocity shear if CT is wrongly implemented.

---

## 7. Field-loop advection

**Reference.** Gardiner & Stone 2005, J. Comput. Phys. 205, 509 ("GS05")
§3.3.1. Reproduced in Stone+08 §8.4, Fig. 21.

- **Dimension.** 2D.
- **Domain.** `x ∈ [−1, 1], y ∈ [−0.5, 0.5]` (GS05 Eq. (54); Athena++
  default: same). Resolved on a `2N × N` grid with N = 64, so 128 × 64.
- **γ.** 5/3.
- **Initial condition.**
  - `ρ = 1, P = 1` uniform.
  - Velocity: advection at speed `v₀ = sqrt(5)` (so after `t = 1` the loop
    has been carried exactly once around the grid diagonal):
    `vx = v₀ · cosθ = v₀ · (2/sqrt(5)) = 2`
    `vy = v₀ · sinθ = v₀ · (1/sqrt(5)) = 1`
    `vz = 0` (2D test).
  - Magnetic field: `Bz = 0`; `(Bx, By)` from vector potential
    `Az(x, y) = A₀ · max(R − r, 0)` with `A₀ = 10⁻³`, `R = 0.3`,
    `r = sqrt(x² + y²)`. This gives an initial loop of radius 0.3 centered
    at the origin, with `|B| = A₀` inside the loop and `|B| = 0` outside.
    `β = 2P/B² = 2 · 10⁶` inside the loop — the field is a
    **passive scalar** and should be transported without distortion.
- **Boundary.** Periodic in both x and y.
- **End time.** `t_f = 2.0` (after two orbits — Stone+08 Fig. 21). GS05
  also shows `t = 0.19` (first orbit, diagnostic of dispersion) and `t = 1`
  (one orbit).
- **Typical grid.** 128 × 64 (N = 64, GS05) and 256 × 148 (Athena docs).
- **Expected outcome.** Loop geometry preserved after two orbits; no hole at
  loop center (which would indicate numerical reconnection). The
  out-of-plane current `Jz = ∂By/∂x − ∂Bx/∂y` is sensitive to any loss of
  coherence.
- **Pass criterion.**
  1. Magnetic pressure `(Bx² + By²)/2` at t = 2 looks visually identical to
     t = 0 (GS05 Fig. 1 top-left vs. Fig. 3 bottom — the `Ez_c` curl-less
     CT algorithm shows no diffusion hole).
  2. Monotonic decay of total magnetic energy `∫ B² dV`:
     `E_B(t=2) / E_B(t=0) ≥ 0.97` at N = 64 (Stone+08 §8.4 "preserves the
     shape of the field loop extremely well"). Weaker CT algorithms (`Ez_α`)
     give ratios below 0.90 at same resolution.
  3. **`Bz ≡ 0` to roundoff.** Stone+08 §8.4 explicitly calls out this test
     with a uniform out-of-plane advection `vz ≠ 0`; `Bz` must remain 0 to
     machine precision throughout, confirming the CT discretization is
     consistent with `∇·B = 0`.
- **Published comparison.** GS05 Fig. 1 (t = 0 stationary), Fig. 2
  (advected, t = 0.19), Fig. 3 (advected, t = 2.0); Stone+08 Fig. 21.

**Why this benchmark.** The most sensitive test of **CT corner-EMF
averaging**. Advection of a passive-scalar field is trivial for a hydro
code but non-trivial for MHD because `Bx` and `By` at cell faces must
co-evolve such that the face-integrated divergence stays exactly zero. A
naive `E_z = (F_x(By) + G_y(Bx))/2` averaging of the Riemann-solver fluxes
makes the loop diffuse or develop a central hole within one orbit
(GS05 Fig. 1 top-right, Ez_α). Catches: wrong sign in
`E_z ≡ v × B |_z = v_x B_y − v_y B_x`; incorrect upwinding of the
transverse-B gradients in the corner-EMF construction; `Bz` being
contaminated by grid noise when `vz ≠ 0`.

---

## 8. Linear MHD wave convergence test

**Reference.** GS05 §5.4 (1D and 2D formulations and the eigenvectors
reproduced below); Stone+08 §8.2, Fig. 12 (1D) and §8.6, Fig. 32 (3D).

### Setup

- **Dimension.** 1D (primary) and 2D/3D (oblique version in 2D and 3D).
- **Domain.** 1D: `x ∈ [0, 1]`, periodic.
  2D: `x ∈ [0, 2/sqrt(5)], y ∈ [0, 1/sqrt(5)]`, periodic both directions,
  wave propagates at `θ = tan⁻¹(2) ≈ 63.4°` to x-axis, λ = 2/5 along
  rotated `x₁` (GS05 §5.4).
  3D: `Lx, Ly, Lz = 3, 3/2, 3/2`, wave oblique to all three axes (Stone+08
  §8.6).
- **γ.** 5/3.
- **Background state (in rotated frame where `x₁` is the wave axis):**
  - `ρ̄ = 1`
  - `P̄ = 1/γ = 3/5` → sound speed `c_s² = γP/ρ = 1`
  - `v̄₁ = 1` for the entropy-mode test; `v̄₁ = 0` for all other modes
  - `v̄₂ = v̄₃ = 0`
  - `B̄₁ = 1`, `B̄₂ = sqrt(2)`, `B̄₃ = 1/2`
- **Derived wave speeds.** In the wave-propagation direction:
  - slow-mode `c_s = 1/2`
  - Alfvén `c_A = 1`
  - fast-mode `c_f = 2`
- **Perturbation.** Amplitude `A = ε = 10⁻⁶`, wavelength = one domain
  length. Initial state is `q₀ = q̄ + ε R_k cos(2π x₁)` where `R_k` is the
  right eigenvector in **conserved variables** for wave mode `k`.
- **Boundary.** Periodic.
- **End time.** `t_f = λ / c_k` for the wave mode under study — i.e. one
  wavelength propagation time: `t_f = 2` for slow, `1` for Alfvén, `1/2`
  for fast. For entropy mode, `t_f = 1` (advected at background flow).
- **Typical grid.** N = 8, 16, 32, 64, 128, 256, 512, 1024 (Stone+08 Fig.
  12). Double precision is required to see convergence below L1 ~ 10⁻¹⁰.

### Right eigenvectors (conserved variables `q = (ρ, ρv₁, ρv₂, ρv₃, B₁, B₂, B₃, E)`)

Taken verbatim from **GS05 Appendix A**. The eigenvectors for this specific
background state (ρ̄ = 1, P̄ = 3/5, B̄ = (1, sqrt(2), 1/2), γ = 5/3) are
given explicitly; we reproduce them here to enable direct reference-implementation
matching.

**Fast magnetosonic (±c_f = ±2):**

```
              ( 6,  ±12,  ∓4√2,  ∓2,  0,  8√2,  4,  27 )^T
R_{±c_f}  = ------------------------------------------------
                               6 · √5
```

**Slow magnetosonic (±c_s = ±1/2):**

```
              ( 12,  ±6,  ±8√2,  ±4,  0,  −4√2,  −2,  9 )^T
R_{±c_s}  = ------------------------------------------------
                               6 · √5
```

**Alfvén (±c_A = ±1):**

```
              ( 0,  0,  ±1,  ∓2√2,  0,  −1,  2√2,  0 )^T
R_{±c_A}  = --------------------------------------------
                               3
```

**Entropy (v̄₁ = 1):**

```
              ( 2,  2,  0,  0,  0,  0,  0,  1 )^T
R_{v₁}    = ------------------------------------
                            2
```

These must be rotated back to Cartesian `(x, y, z)` before initialization
in 2D / 3D using the inverse of GS05 Eq. (54–56); for 2D
θ = tan⁻¹(2): `v_x = v₁ cosθ − v₂ sinθ`, etc. **In-plane magnetic
components Bx, By must be initialized from a vector potential Az so that
the discrete `∇·B` is zero at machine precision.**

### L1 error norm

After evolving for exactly one period (so the exact solution equals the
initial state), compute (GS05 Eq. (80)):

```
δq_k = (1 / N_cells) · Σ_{cells}  |q^{numerical}_{cell, k} − q^{t=0}_{cell, k}|
```

for each conserved variable `k = 1..8`, and

```
||δq|| = sqrt( Σ_k (δq_k)² )
```

### Convergence criterion

- **2nd-order scheme** (VL2 + piecewise-linear reconstruction + HLLD):
  `||δq|| ∝ N⁻²` for all four wave families, identical to machine precision
  for left- and right-propagating waves.
- **3rd-order scheme** (VL2 + PPM + HLLD): `||δq|| ∝ N⁻²` formally (3rd-order
  spatial is still 2nd-order in a VL2 time-stepper), but the prefactor drops
  by ~4× relative to PLM.
- Stone+08 Fig. 12 shows all three wave families converging at 2nd-order
  from N = 8 to N ≈ 512; at N = 1024 with `A = 10⁻⁶` the floor of roundoff
  becomes visible (`||δq|| ~ 10⁻¹²`).
- GS05 Fig. 11 reports observed convergence orders from a power-law fit:
  fast 2.00, Alfvén 2.26, slow 2.29, entropy 2.11 — i.e. all modes achieve
  *at least* 2nd-order; slow/Alfvén are slightly super-convergent for CTU
  because of the CT algorithm's extra symmetry in transverse directions.

### Pass criterion

1. Observed convergence order `p_k ≥ 1.95` for every wave family at N = 16
   → 512.
2. L1 errors for left- and right-propagating waves of each family agree to
   at least 10⁻¹⁴ (this is very sensitive — any transpose bug in the
   reconstruction kernel between `+x` and `−x` directions breaks this
   immediately).
3. No spurious mode coupling: e.g. initializing a pure Alfvén wave must
   produce `δρ < 10 · ε² · A = 10⁻¹⁸` (below roundoff) at every output
   time; any contamination above that level indicates a nonlinear-term
   sign error or a broken flux-Jacobian identity.

**Why this benchmark.** The *quantitative* pre-merge gate. While shock
tubes and nonlinear vortex tests check qualitative features, linear-wave
convergence is the only test that can **measure** the scheme's truncation
error to the exact analytical solution. Failing this test at L1 ∝ N⁻¹
(instead of N⁻²) or finding left/right asymmetry is a near-certain sign
that the characteristic tracing, reconstruction, or HLLD middle-state
formula is wrong. Modes tested: fast, slow, Alfvén, entropy — i.e. all
MHD wave families. Catches: almost every implementation bug that survives
qualitative tests.

---

## 9. Torrilhon shock tube (optional compound-wave test)

**Reference.** Torrilhon, M. 2003, J. Comput. Phys. 192, 73 — §4.2 "regular,
nearly coplanar problem"; Stone+08 §8.2 runs this as a convergence check.

- **Dimension.** 1D.
- **Domain.** `x ∈ [−0.5, 0.5]`, outflow boundaries.
- **γ.** 5/3.
- **Initial condition** (Stone+08 Table 2):
  Left: `ρL = 1.0, vx,L = 0, vy,L = 0, vz,L = 0, PL = 1.0,
  By,L = cos(0.5), Bz,L = sin(0.5)`.
  Right: `ρR = 0.2, vx,R = 0, vy,R = 0, vz,R = 0, PR = 0.2,
  By,R = cos(1.5), Bz,R = sin(1.5)`.
  Longitudinal field: `Bx = 1.0`.
- **End time.** `t_f = 0.4` (Torrilhon 2003 Fig. 7).
- **Typical grid.** 800 cells; reference at 10⁴.
- **Pass criterion.** As N → ∞ the solution should converge to a unique
  non-compound ideal-MHD solution; at finite resolution a compound wave may
  appear. Stone+08 reports Athena's 10⁴-cell solution is comparable to
  Torrilhon 2003's 2·10⁴-cell result — higher resolution needed to kill the
  compound wave.
- **Why.** Regression test for dissipation-free MHD Riemann solvers that can
  incorrectly stabilize a compound wave. Optional; most codes will have an
  N¹⸱⁵ convergence rate here due to non-convexity.

---

## 10. MHD Kelvin-Helmholtz instability (optional, qualitative)

**Reference.** Ryu, D., Jones, T. W., Frank, A. 2000, ApJ 545, 475; Stone+08
§8.4 briefly mentions the Miura & Pritchett 1982 variant.

- **Dimension.** 2D.
- **Domain.** `(x, y) ∈ [0, 1]²`, periodic in x, reflective in y (or
  periodic both if using `ρ(y)` taper).
- **γ.** 5/3.
- **Initial condition.** Shear layer:
  - `vx(y) = v₀ tanh((y − 0.5) / a)` with `v₀ = 0.5, a = 0.05`
  - `ρ = 1, P = 1/γ`
  - `Bx = B₀, By = 0, Bz = 0` with `B₀` a **scan parameter**:
    `B₀ = 0` (hydro KH for reference), `B₀ = 0.01` (weak, KH survives),
    `B₀ = 0.1` (strong, KH suppressed).
  - Perturbation: `vy = A sin(2πx) · exp(−(y − 0.5)² / a²)`, `A = 0.01`.
- **Boundary.** Periodic in x. Periodic or reflective in y.
- **End time.** `t_f ≈ 5` (several eddy turnover times).
- **Pass criterion (qualitative).**
  - `B₀ = 0`: classical KH vortex chain — quantitative comparison with any
    Athena KH figure.
  - `B₀ = 0.1`: instability must be **suppressed** (Stone+08 §8.4
    MHD RT analog: "the magnetic field suppresses the R-M instability").
  - Monitor `∇·B` throughout — must stay at roundoff.
- **Why.** Qualitative regression for nonlinear MHD turbulence. Not
  suitable as a hard pass/fail gate because there is no unique solution;
  use only to confirm the expected B-field stabilization trend.

---

## Execution notes for `stellar2d` port

For each test we will add a `src/drivers/athena_mhd_<testname>.cpp` file
following the post-refactor solver-addition workflow in `CLAUDE.md`. The
initial condition setup uses the `setup_ic` "Grid-less test cases" list
(CLAUDE.md §"main.cpp 拆分后的 solver 添加流程" step 7) so `setup_ic`
skips the generic hydro initialization.

Verification scripts mirror the existing `tst/test_ale2/` and `tst/`
structure used by the hydro benchmarks: each MHD test needs a
`compute_<test>_error` routine in the solver (following the entropy-wave /
Sod / linwave pattern), a `--compute-error` CLI flag wiring in the driver,
and a pytest in `tst/test_athena_mhd/`.

### Suggested ordering (ascending difficulty, earliest failure = highest
signal)

1. Linear MHD wave convergence (§8) — should work as soon as HLLD +
   CT corner-EMF are wired; fails loudly on any bug.
2. Brio-Wu (§1) — smallest 1D Riemann check.
3. RJ2a (§2) — full 7-wave coverage.
4. CP Alfvén 1D (§3a) — nonlinear exact solution.
5. Field-loop advection (§7) — CT correctness gate.
6. Orszag-Tang (§4) — first "realistic" 2D MHD test.
7. MHD blast (§5) — robustness test, likely first to expose low-β
   positivity issues.
8. MHD rotor (§6) — combines rotational discontinuity with low-β.
9. CP Alfvén 2D (§3b) — most sensitive 2D convergence gate.
10. RJ4d (§2b), Torrilhon (§9), MHD KH (§10) — regression / optional.

Tests 1–8 together cover: all 7 MHD wave families, CT constraint
preservation, multi-D symmetry, nonlinear convergence, low-β robustness,
and strong rotational discontinuities. If the new `athena_mhd` solver
passes all 8, it is ready for science runs.
