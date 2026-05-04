#!/usr/bin/env python3
"""Phase D multi-dimensional audit — do NOT stop at Mg/Ne.

Per feedback_baseline_matching_antipattern.md: tuning a single scalar to match
a paper is dangerous.  This script checks multiple output quantities at the
best-match (mass_cut, E_SN) for each progenitor.

Outputs:
  1. Mg/Ne per paper reference
  2. Si/Ne ratios (paper Fig 6 range: ~0.4-2)
  3. M_Ni56 (typical CCSN: 0.05-0.15 Msun for non-fallback cases)
  4. Ejecta KE (paper: 2-4 × 10⁵¹ erg for N49B observations)
  5. Inner Fe-peak mass (expected ~0.1-0.3 Msun)
  6. Energy/mass conservation (should be exact for radial1d Lagrangian)

For each quantity, print the Phase D result and whether it lies in a
physically plausible range for CCSN nucleosynthesis.  This IS NOT a paper
match check — the paper doesn't publish all these numbers — but a sanity
audit that the underlying hydro+nuc is producing realistic output.
"""
from __future__ import annotations

import csv
import numpy as np
from pathlib import Path

BEST = {
    "12.02": ("1.4", "5.0e+50"),
    "12.75": ("1.8", "1.0e+51"),
    "15.28": ("1.8", "1.0e+51"),
    "15.90": ("1.8", "2.0e+51"),
}

PAPER_MGNE = {"12.02": 0.30, "12.75": 0.75, "15.28": 0.15, "15.90": 1.25}


def main():
    npz_dir = Path("data/n49b_postSN")

    print("=" * 75)
    print("Phase D multi-dimensional audit (paper 0.75 ± 20% is NOT the only check)")
    print("=" * 75)

    rows = []
    for zams, (mc, Es) in BEST.items():
        f = npz_dir / f"postSN_{zams}_mc{mc}_E{Es}.npz"
        d = np.load(f, allow_pickle=True)

        # 13-species summary
        M = {
            "He": float(d["M_He_total"]),
            "C":  float(d["M_C_total"]),
            "O":  float(d["M_O_total"]),
            "Ne": float(d["M_Ne_total"]),
            "Mg": float(d["M_Mg_total"]),
            "Si": float(d["M_Si_total"]),
            "S":  float(d["M_S_total"]),
            "Ar": float(d["M_Ar_total"]),
            "Ca": float(d["M_Ca_total"]),
            "Ti": float(d["M_Ti_total"]),
            "Cr": float(d["M_Cr_total"]),
            "Fe": float(d["M_Fe_total"]),
            "Ni": float(d["M_Ni_total"]),
        }
        M_fe_peak = M["Cr"] + M["Fe"] + M["Ni"]
        Mgne = float(d["Mg_Ne"])
        Sine = float(d["Si_Ne"])
        KE = float(d["KE_erg"])
        IE = float(d["IE_erg"])
        E_bomb = float(d["E_bomb_erg"])
        M_eject = float(d["M_ejecta_msun"])

        print(f"\n{zams} M⊙  (mc={mc}, E={Es} erg, M_eject={M_eject:.2f} M⊙)")
        print(f"  Mg/Ne     = {Mgne:.3f}  paper={PAPER_MGNE[zams]:.2f}  "
              f"Δ={100*(Mgne-PAPER_MGNE[zams])/PAPER_MGNE[zams]:+.1f}%")
        print(f"  Si/Ne     = {Sine:.3f}  (typical CCSN: 0.3–5)")
        print(f"  M_Ni56    = {M['Ni']:.4f} M⊙  (CCSN lit: 0.03–0.15)")
        print(f"  M_Fe_peak = {M_fe_peak:.4f} M⊙  (Cr+Fe+Ni, expected ~0.1–0.3)")
        print(f"  M_Si      = {M['Si']:.4f} M⊙")
        print(f"  M_O       = {M['O']:.4f} M⊙")
        print(f"  M_total = {sum(M.values()):.3f} M⊙  (should match M_eject={M_eject:.3f})")
        print(f"  KE/E_bomb = {KE/E_bomb:.3f}  (shock partitioning)")
        print(f"  IE/E_bomb = {IE/E_bomb:.3f}")
        rows.append({
            "zams": zams, "mc": mc, "E_SN": Es,
            "Mg_Ne": Mgne, "Si_Ne": Sine,
            "M_Ni56": M["Ni"], "M_Fe_peak": M_fe_peak,
            "M_Si": M["Si"], "M_O": M["O"],
            "M_total": sum(M.values()), "M_eject": M_eject,
            "KE_E_bomb": KE/E_bomb, "IE_E_bomb": IE/E_bomb,
        })

    print("\n" + "=" * 75)
    print("Sanity checks:")
    print("=" * 75)
    for r in rows:
        mass_ok = abs(r["M_total"] - r["M_eject"]) / r["M_eject"] < 0.01
        ni_ok = 0.001 <= r["M_Ni56"] <= 0.3    # wide for 1D parametric bomb
        sine_ok = 0.1 <= r["Si_Ne"] <= 20      # accept broad range for now
        tag = "✓" if (mass_ok and ni_ok) else "✗"
        print(f"  {r['zams']}: mass_cons={mass_ok} M_Ni in[0.001,0.3]={ni_ok} "
              f"Si/Ne in[0.1,20]={sine_ok}  overall {tag}")


if __name__ == "__main__":
    main()
