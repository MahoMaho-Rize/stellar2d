# ν_eff comparison — 2026-05-07

T3 Linear shear-mode / Taylor-Green pure-diffusion decay:

- cart_ale2 / athena_vl2:  `vx = V₀·sin(k·2π y/Ly)`, analytic decay `max|v| ∝ exp(-ν·k²_phys·t)`
- pseudo_spectral:  Taylor-Green `ω = 2k·cos·cos`, analytic `max|ω| ∝ exp(-2ν·k²_phys·t)`

ν_eff extracted by log-slope fit on t ∈ [5 %, 50 %] of t_end.

## Summary table (V₀=0.01)

| solver | res=64 | 128 | 256 | 512 | slope p (ν∝dx^p) |
|---|---|---|---|---|---|
| cart_ale2 | 1.55e-02 | 7.76e-03 | 3.88e-03 | 1.94e-03 | 1.00 |
| athena_vl2 | -3.31e-17 | 3.75e-17 | 9.58e-17 | -7.17e-18 | -0.14 |
| pseudo_spectral | -2.89e-18 | -8.86e-18 | 6.97e-18 | -1.37e-17 | -0.94 |

- CSV: `2026-05-07_nu_eff_comparison.csv`
- Plot: `2026-05-07_nu_eff_comparison.png`

## Reading

- slope p ≈ 2 → Godunov-like 2nd-order dissipation (ν_eff ≈ V·dx² prefactor varies)
- slope p ≈ 1 → Jensen-ILES 1st-order (ν_eff ≈ V·dx — 'over-damped' at coarse grid)
- slope p ≫ 2 / negligible ν_eff → spectral / DNS-quality
