#!/usr/bin/env python3
"""aprox13 vs aprox13e comparison across 48-run sweep.

Shows: Mg/Ne across (mass_cut × E_SN) for both networks, annotated with
paper targets, and scatter of Δ(Mg/Ne) vs (ρ, T) proxies (E/dm).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path: Path) -> dict:
    t = {}
    for r in csv.DictReader(path.open()):
        if not r.get("Mg_Ne"):
            continue
        key = (r["zams"], float(r["mass_cut"]), float(r["E_bomb_erg"]))
        t[key] = float(r["Mg_Ne"])
    return t


def main():
    root = Path("data/n49b_postSN")
    t13  = load(root / "phaseD_sweep_aprox13.csv")
    t13e = load(root / "phaseD_sweep_aprox13e.csv")
    paper = {"12.02": 0.30, "12.75": 0.75, "15.28": 0.15, "15.90": 1.25}
    zams_list = ["12.02", "12.75", "15.28", "15.90"]
    mcs = [1.4, 1.6, 1.8]
    Es  = [5e50, 1e51, 2e51, 4e51]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=130)

    for idx, zams in enumerate(zams_list):
        ax = axes.flat[idx]
        for mc, color in zip(mcs, ["C0", "C1", "C2"]):
            vals13  = [t13.get((zams, mc, E), np.nan) for E in Es]
            vals13e = [t13e.get((zams, mc, E), np.nan) for E in Es]
            ax.plot(Es, vals13,  "o--", color=color, alpha=0.5,
                    label=f"mc={mc} (aprox13)")
            ax.plot(Es, vals13e, "s-",  color=color,
                    label=f"mc={mc} (aprox13e)")
        ax.axhline(paper[zams], color="k", linestyle=":", lw=1.5,
                   label=f"paper = {paper[zams]}")
        # tolerance band 0.8-1.2x paper
        ax.axhspan(0.8*paper[zams], 1.2*paper[zams], color="k", alpha=0.05)

        ax.set_xscale("log")
        ax.set_xlabel("E_SN [erg]")
        ax.set_ylabel("Mg/Ne (post-SN, O-rich mask)")
        ax.set_title(f"{zams} M⊙  (paper {paper[zams]})", fontsize=10)
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.9)

    fig.suptitle(
        "Phase D sensitivity — aprox13 (dashed) vs aprox13e with (α,p) bypass (solid)\n"
        "gray band = paper ± 20%",
        fontsize=11, y=0.995,
    )
    fig.tight_layout()
    out = Path("docs/images/n49b_aprox13_vs_aprox13e.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
