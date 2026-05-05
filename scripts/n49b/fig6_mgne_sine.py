#!/usr/bin/env python3
"""Reproduce Fig 6 of Sato+2024 (N49B paper) — Mg/Ne vs Si/Ne scatter.

Colored by M_He_core.  Overplots the N49B, G284, G292, Pup A observational
points from the paper.  Shows the Mg/Ne = 1 threshold as a dashed line.

Input : data/n49b_progenitor_catalog.csv (from batch_analysis.py)
Output: docs/images/n49b_fig6_mgne_sine.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Observational points from Sato+2024 Fig 6 and Table 1
#   N49B   : Mg/Ne = 2.62, Si/Ne very low
#   G284   : Mg/Ne ~ 1.3,  Si/Ne ~ 0.5-1 (magenta box in paper Fig 6)
#   Pup A  : Mg/Ne ~ 0.3,  Si/Ne ~ 0.3   (red circle)
#   G292   : Mg/Ne ~ 0.2,  Si/Ne ~ 0.3   (red square)
# Values digitised from Fig 6; paper cites Park+03/17, Katsuda+10,
# Kamitsukasa+14, Williams+15.
def _f(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


OBSERVATIONS = {
    "N49B":  dict(MgNe=2.62, SiNe=0.03, marker="*", color="magenta", size=300),
    "G284":  dict(MgNe=1.30, SiNe=0.70, marker="s", color="magenta", size=120),
    "Pup A": dict(MgNe=0.30, SiNe=0.30, marker="o", color="red",      size=100),
    "G292":  dict(MgNe=0.20, SiNe=0.30, marker="s", color="red",      size=100),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/n49b_progenitor_catalog.csv")
    ap.add_argument("--out", default="docs/images/n49b_fig6_mgne_sine.png")
    args = ap.parse_args()

    with open(args.catalog) as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    cols = {k: np.array([_f(r[k]) for r in rows]) for k in rows[0].keys()}

    # Mask out NaN and very-small M_O (models without a proper O-rich layer)
    valid = np.isfinite(cols["MgNe"]) & np.isfinite(cols["SiNe"]) & (cols["M_O"] > 0.01)
    d = {k: v[valid] for k, v in cols.items()}

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=130)

    # Main scatter
    sc = ax.scatter(
        d["MgNe"], d["SiNe"],
        c=d["M_He"], cmap="viridis",
        s=14, alpha=0.85, edgecolors="none",
        vmin=3, vmax=9,
    )
    cb = fig.colorbar(sc, ax=ax, label=r"He core mass [$M_\odot$]")

    # Mg/Ne = 1 threshold line
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax.text(1.05, 5e-3, "Mg/Ne = 1", color="gray", fontsize=9,
            rotation=90, va="bottom")

    # Annotation "Mg-rich progenitors" box (upper right)
    ax.fill_betweenx([1e-2, 1e2], 1.0, 50, alpha=0.05, color="red", zorder=0)
    ax.text(3, 30, "Mg-rich\nprogenitors", color="red", fontsize=10,
            alpha=0.7, ha="center")

    # Observations
    for name, p in OBSERVATIONS.items():
        ax.scatter(p["MgNe"], p["SiNe"], marker=p["marker"], s=p["size"],
                   color=p["color"], edgecolor="black", linewidth=1.2,
                   zorder=5, label=name)

    # "Shell merger (O-shell + C-shell)" direction arrow (upper right, per paper)
    ax.annotate("Shell merger\n(O-shell + C-shell)",
                xy=(6, 5), xytext=(10, 2),
                fontsize=9, color="black",
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    # "Ne-shell intrusion" arrow (lower right)
    ax.annotate("Ne-shell intrusion",
                xy=(1.5, 0.03), xytext=(2.5, 3e-3),
                fontsize=9, color="black",
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.05, 50)
    ax.set_ylim(2e-3, 1e2)
    ax.set_xlabel("Mg/Ne (mass ratio in O-rich layer)")
    ax.set_ylabel("Si/Ne (mass ratio in O-rich layer)")
    ax.set_title("N49B Fig 6 replication — Sukhbold+2018 progenitors\n"
                 f"({len(d['MgNe'])}/{len(cols['MgNe'])} models with M(O-rich) > 0.01 $M_\\odot$)")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")

    # Quick stats
    n_mgrich = int(np.sum((d["MgNe"] > 1) & (d["M_He"] > 5) & (d["SiNe"] > 1)))
    n_ne_intrusion = int(np.sum((d["MgNe"] > 1) & (d["SiNe"] < 0.1)))
    print(f"  Total O-rich progenitors : {len(d['MgNe'])}")
    print(f"  Mg-rich + high Si/Ne (shell merger regime) : {n_mgrich}")
    print(f"  Mg-rich + low Si/Ne (Ne-shell intrusion regime) : {n_ne_intrusion}")


if __name__ == "__main__":
    main()
