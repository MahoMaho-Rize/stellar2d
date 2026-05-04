#!/usr/bin/env python3
"""Plot dt ceiling lift from block-tridiag JFNK preconditioner.

Reads two radial1d diagnostics.csv files (PC on / PC off) and shows dt(t)
across the runs. Intended to accompany docs/radial1d_precond_tridiag_2026-05-03.md.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc-on",  default="/tmp/pc_on.csv")
    ap.add_argument("--pc-off", default="/tmp/pc_off.csv")
    ap.add_argument("--out",    default="docs/images/precond_tridiag_speedup.png")
    args = ap.parse_args()

    d_on  = read_csv(args.pc_on)
    d_off = read_csv(args.pc_off)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(d_off["t"] + 1, d_off["dt"] + 1,  "o-", color="#c33",
              label=f"identity PC — {len(d_off)-1} steps, reached t={d_off['t'][-1]:.2e} s", alpha=0.7, markersize=3)
    ax.loglog(d_on["t"] + 1,  d_on["dt"] + 1,   "s-", color="#693",
              label=f"block-tridiag PC — {len(d_on)-1} steps, reached t={d_on['t'][-1]:.2e} s", markersize=6)
    ax.axvspan(3.15e12, 1.4e15, color="gray", alpha=0.08)  # τ_KH range
    ax.text(7e13, 1e2, "τ_KH for 1 M⊙\npre-MS", ha="center", color="gray")
    ax.set_xlabel("physical time t [s]")
    ax.set_ylabel("timestep dt [s]")
    ax.set_title("radial1d JFNK preconditioner lift (pre-MS 1 M⊙ ZAMS IC)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
