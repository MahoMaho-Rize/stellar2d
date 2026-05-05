#!/usr/bin/env python3
"""Reproduce Fig 2 of Sato+2024 (N49B paper).

Four panels:
  top-left   : ξ_{2.5}  vs M_He_core,  colored by Mg/Ne regime
  top-middle : M_4       vs M_He_core
  top-right  : μ_4       vs M_He_core
  bottom     : μ_4       vs μ_4 · M_4   (Ertl+2016 style with BH/SN line)

Mg/Ne coding:
  gray   : Mg/Ne < 1
  blue   : Mg/Ne > 1, M_ZAMS < 14 Msun   (low-mass shell intrusion)
  red    : Mg/Ne > 1, M_ZAMS >= 14 Msun  (high-mass shell merger)

Ertl+2016 BH/SN separation (w18 calibration):
    y = 0.283 * x + 0.0430

Input : data/n49b_progenitor_catalog.csv
Output: docs/images/n49b_fig2_compactness.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _f(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/n49b_progenitor_catalog.csv")
    ap.add_argument("--out", default="docs/images/n49b_fig2_compactness.png")
    args = ap.parse_args()

    with open(args.catalog) as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    c = {k: np.array([_f(r[k]) for r in rows]) for k in rows[0].keys()}

    zams = c["zams_mass"]
    Mhe  = c["M_He"]
    mgne = c["MgNe"]
    xi25 = c["xi25"]
    M4   = c["M4"]
    mu4  = c["mu4"]

    # Classification
    finite_mg = np.isfinite(mgne)
    mg_low  = ~(finite_mg & (mgne > 1))       # Mg/Ne < 1 or NaN
    mg_hi_low_mass  = finite_mg & (mgne > 1) & (zams <  14)
    mg_hi_high_mass = finite_mg & (mgne > 1) & (zams >= 14)

    fig = plt.figure(figsize=(13, 9), dpi=130)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3,
                          height_ratios=[1, 1.2])

    # ---- top-left: xi_{2.5} vs M_He ----
    ax = fig.add_subplot(gs[0, 0])
    _scatter3(ax, Mhe, xi25, mg_low, mg_hi_low_mass, mg_hi_high_mass)
    ax.set_xlabel(r"He core mass [$M_\odot$]")
    ax.set_ylabel(r"Compactness $\xi_{2.5}$")
    ax.set_xlim(3, 9); ax.set_ylim(0, 0.5)

    # ---- top-middle: M_4 vs M_He ----
    ax = fig.add_subplot(gs[0, 1])
    _scatter3(ax, Mhe, M4, mg_low, mg_hi_low_mass, mg_hi_high_mass)
    ax.set_xlabel(r"He core mass [$M_\odot$]")
    ax.set_ylabel(r"$M_4$ [$M_\odot$]")
    ax.set_xlim(3, 9); ax.set_ylim(1.4, 2.2)

    # ---- top-right: mu_4 vs M_He ----
    ax = fig.add_subplot(gs[0, 2])
    _scatter3(ax, Mhe, mu4, mg_low, mg_hi_low_mass, mg_hi_high_mass)
    ax.set_xlabel(r"He core mass [$M_\odot$]")
    ax.set_ylabel(r"$\mu_4$")
    ax.set_xlim(3, 9); ax.set_ylim(0, 0.18)

    # ---- bottom: mu_4 vs mu_4 * M_4  (Ertl+2016) ----
    ax = fig.add_subplot(gs[1, :])
    x_all = mu4 * M4
    _scatter3(
        ax, x_all, mu4, mg_low, mg_hi_low_mass, mg_hi_high_mass,
        labels=("Mg/Ne < 1",
                "Mg/Ne > 1 ($M_{ZAMS} < 14\\,M_\\odot$)",
                "Mg/Ne > 1 ($M_{ZAMS} \\geq 14\\,M_\\odot$)"),
    )
    # BH/SN separation line (w18)
    xx = np.linspace(0.05, 0.35, 100)
    ax.plot(xx, 0.283 * xx + 0.0430, "k--", lw=1.2,
            label=r"$y = 0.283\,x + 0.0430$ (w18)")
    ax.set_xlabel(r"$x = \mu_4 \cdot M_4$")
    ax.set_ylabel(r"$y = \mu_4$")
    ax.set_xlim(0, 0.38); ax.set_ylim(0, 0.18)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92)

    fig.suptitle(
        f"N49B Fig 2 replication — Sukhbold+2018 ({len(zams)} progenitors)",
        y=0.995, fontsize=12,
    )

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")

    # Summary stats
    print(f"  Mg/Ne < 1            : {mg_low.sum():4d}")
    print(f"  Mg/Ne > 1, ZAMS < 14 : {mg_hi_low_mass.sum():4d}")
    print(f"  Mg/Ne > 1, ZAMS ≥ 14 : {mg_hi_high_mass.sum():4d}")


def _scatter3(ax, x, y, mask_low, mask_mid, mask_hi, labels=None):
    """Three scatter groups: gray low, blue low-mass Mg-rich, red high-mass Mg-rich."""
    lo, mi, hi = labels if labels else (None, None, None)
    ax.scatter(x[mask_low], y[mask_low], s=10, c="#888888", alpha=0.5,
               edgecolors="none", label=lo)
    ax.scatter(x[mask_mid], y[mask_mid], s=14, c="blue", alpha=0.8,
               edgecolors="none", label=mi)
    ax.scatter(x[mask_hi],  y[mask_hi],  s=14, c="red",  alpha=0.8,
               edgecolors="none", label=hi)
    ax.grid(True, alpha=0.25)


if __name__ == "__main__":
    main()
