#!/usr/bin/env python3
"""Reproduce the left panels of Figs 4 & 5 of Sato+2024 (N49B paper).

Mass fraction X_i(m_enc) for key elements, on 4 progenitors:
  12.02 M☉ (no Ne-shell merger)
  12.75 M☉ (Ne-shell intrusion)
  15.28 M☉ (no merger)
  15.90 M☉ (violent O-C shell merger)

Right panels (Kippenhahn convection history) are provided as PNG figures in
the Sukhbold+2018 dataset (convection_plots/) — the raw convection-by-zone
time history is not published in ASCII form, so we reference the paper
directly for that panel.

Input  : ~/data/sukhbold_2018/mdotone/{M}.dat
Output : docs/images/n49b_fig4_5_mass_fraction.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sukhbold_reader import read_one, MSUN_CGS


ELEMENTS = [
    ("H1",   "H1",   "C0"),
    ("He4",  "He4",  "C1"),
    ("C12",  "C12",  "C2"),
    ("N14",  "N14",  "C3"),
    ("O16",  "O16",  "C4"),
    ("Ne20", "Ne20", "C5"),
    ("Mg24", "Mg24", "C6"),
    ("Si28", "Si28", "C7"),
    ("Fe56", "Fe56", "C8"),
]

MODELS = [
    ("12.02", "12.02 $M_\\odot$ — low-mass, no Ne-shell merger"),
    ("12.75", "12.75 $M_\\odot$ — Ne-shell intrusion"),
    ("15.28", "15.28 $M_\\odot$ — no shell merger"),
    ("15.90", "15.90 $M_\\odot$ — violent O-C shell merger"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="~/data/sukhbold_2018/mdotone")
    ap.add_argument("--out", default="docs/images/n49b_fig4_5_mass_fraction.png")
    args = ap.parse_args()
    data_dir = Path(args.data_dir).expanduser()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=130)
    axes = axes.ravel()

    for ax, (mstr, title) in zip(axes, MODELS):
        p = data_dir / f"{mstr}.dat"
        if not p.exists():
            ax.text(0.5, 0.5, f"{p}\nnot found", ha="center", va="center")
            ax.set_title(title)
            continue
        prof = read_one(p)
        m = prof.m_enc_msun
        for label, col, c in ELEMENTS:
            ax.plot(m, prof.X(col), label=label, color=c, lw=1.2)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.5)
        ax.set_xlim(0, min(m[-1], 4.5))
        ax.set_xlabel("Enclosed mass [solar masses]")
        ax.set_ylabel("Mass fraction")
        ax.set_title(title, fontsize=11)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, ncol=3, loc="lower right", framealpha=0.9)

    fig.suptitle(
        "N49B Fig 4/5 replication (left panels) — Sukhbold+2018 mass fractions",
        y=0.995, fontsize=12,
    )

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
