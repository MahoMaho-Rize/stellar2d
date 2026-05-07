# scheme_char — quantitative solver characterization data

Data + figures that support the scheme characterization matrix in
`docs/design/testing_scheme_characterization_2026-05-07.md`.

Each data drop is `YYYY-MM-DD_<test>_<solver(s)>.{csv,png,md}` plus a
self-contained markdown write-up. Treat this folder as **paper / audit
evidence**, not regression gates (those live under `tst/` / `tests/`).

## Layout

```
docs/scheme_char/
├── README.md                          ← this file
├── 2026-05-07_nu_eff_comparison.*     ← T3: ν_eff on cart_ale2, athena_vl2,
│                                        pseudo_spectral (task #50)
├── 2026-05-07_andrassy_vrms_vs_res.*  ← T5: Andrassy 2022 v_rms / Ṁ_e
│                                        resolution scan + non-monotonic
│                                        dip explanation (task #49)
├── 2026-05-07_entropy_wave_convergence.*  ← T1: smooth scalar advection
│                                        convergence slope, ale2 vs vl2
│                                        (task #47)
├── 2026-05-07_jensen_probe.*          ← T4: ν_eff(k) at N=256, verifies
│                                        cart_ale2 dissipation is a real
│                                        Laplacian (k-independent) on
│                                        resolved modes (task #48)
└── runs/                              ← raw diagnostics.csv from each scan
    └── <solver>/res_N_V_V0/
        ├── diagnostics.csv            (flattened copy of the actual run)
        └── shear_mode_NxN_<ts>/       (original run dir, keep for provenance)
```

## Solver ν_eff summary table

Kept in sync by hand as new T3 runs land. The slope p is the log-log fit
of `ν_eff` vs `N`, so `ν_eff ∝ dx^p` (larger `p` = less dissipation at
coarse resolution).

| solver           | ν_eff @ 128² (V₀=0.01) | slope p | measured  | notes                                           |
|------------------|------------------------|---------|-----------|-------------------------------------------------|
| cart_ale2        | 7.8e-3                 | 1.00    | 2026-05-07| Jensen-ILES, ν ≈ 0.31·V·dx                      |
| athena_vl2       | ≤ 10⁻¹⁶                | —       | 2026-05-07| Godunov direction-aligned; zero on this IC      |
| pseudo_spectral  | ≤ 10⁻¹⁷                | —       | 2026-05-07| TG nonlinear vanishes identically, machine ε   |

## Solver T1 entropy-wave convergence

| solver           | L1 @ 128²  | slope p | measured  | notes                                  |
|------------------|------------|---------|-----------|----------------------------------------|
| cart_ale2        | 3.7e-3     | -0.31   | 2026-05-07| dispersive, L1 *grows* at 512² (coupled Lagrangian-rebuild issue) |
| athena_vl2       | 1.9e-5     | 2.06    | 2026-05-07| textbook 2nd-order, phase returns to 0 |

## Solver T4 ν_eff(k) k-dependence (Jensen probe)

| solver           | ν_eff(k=1) | ν_eff(k=16) | k-flat? | measured  | notes                          |
|------------------|-----------|-------------|---------|-----------|--------------------------------|
| cart_ale2        | 3.88e-3   | 3.88e-3     | ✅ (0.3 % across k=1..16) | 2026-05-07| confirmed Laplacian-like viscosity |
| athena_vl2       | ≤ 1e-17   | ≤ 1e-17     | N/A     | 2026-05-07| still zero on this IC — Phase B rotated-shear followup pending |
| strang           | —                      | —       | —         | pending task #47 / #50 extension                |
| cart_ale         | —                      | —       | —         | pending                                         |
| cart_lag         | —                      | —       | —         | pending (HSE-incompatible BC, need cook IC)     |
| anelastic_sl     | —                      | —       | —         | pending (anelastic ν probe differs; see Phase B)|

## Reproduce

```bash
cmake --build build -j
scripts/scheme_char/run_nu_eff_scan.sh          # T3 scan (24 runs)
python scripts/scheme_char/fit_nu_eff.py        # fit + plot + summary md
```

## Conventions

- Run directories under `runs/<solver>/res_N_V_V0/` are overwritten by
  rerunning the scan; the flattened `diagnostics.csv` at that level is
  the canonical input to fit scripts.
- When a measurement is known to be direction-aligned (athena_vl2 T3)
  or geometry-degenerate (pseudo_spectral TG with ν=0), record the
  caveat in the markdown write-up — don't hide it behind a single
  "< ε" cell in the summary table.
- For `p` values, quote to 2 decimal places when `r² > 0.99` on the
  log-log fit; else quote as `~N` with uncertainty language.
