#!/usr/bin/env python3
"""Reproduce Fig 7 of Sato+2024 — pre-SN vs post-SN mass fraction for 4 models.

Each model has a (pre-SN, SN) pair of panels showing X_i(m_enc).  The paper's
Fig 7 layout: 12.02 top-left, 12.75 top-right, 15.28 bottom-left, 15.90
bottom-right, each with two sub-panels.

Input : data/n49b_postSN/postSN_{M}.npz (from explosive_nucleo.py)
Output: docs/images/n49b_fig7_pre_post.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# X_pre/X_post columns match alpha_net::Species (Phase B, 13 species):
#   0: He4   1: C12   2: O16   3: Ne20   4: Mg24   5: Si28
#   6: S32   7: Ar36  8: Ca40  9: Ti44  10: Cr48  11: Fe52  12: Ni56
# For Fig 7 we show the 6 main species tracked by the paper.  We add
# "Fe*" for the sum Ti44+Cr48+Fe52+Ni56 to reflect iron-peak abundance.
SPECIES = [
    (0, "He",    "C1"),
    (1, "C",     "C2"),
    (2, "O",     "C4"),
    (3, "Ne",    "C5"),
    (4, "Mg",    "C6"),
    (5, "Si",    "C7"),
]

MODELS = [
    ("12.02", "(a) 12.02 $M_\\odot$"),
    ("12.75", "(b) 12.75 $M_\\odot$"),
    ("15.28", "(c) 15.28 $M_\\odot$"),
    ("15.90", "(d) 15.90 $M_\\odot$"),
]


def plot_panel(ax, m_enc, X, species, title, xlim):
    for idx, label, color in species:
        ax.plot(m_enc, X[:, idx], color=color, lw=1.4, label=label)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Mass radius [solar masses]")
    ax.set_ylabel("Mass fraction")
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir",  default="data/n49b_postSN")
    ap.add_argument("--out", default="docs/images/n49b_fig7_pre_post.png")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()

    # 4 rows × 2 cols (pre / SN)
    fig, axes = plt.subplots(4, 2, figsize=(11, 14), dpi=130)

    for i, (mstr, title) in enumerate(MODELS):
        fn = in_dir / f"postSN_{mstr}.npz"
        if not fn.exists():
            axes[i, 0].text(0.5, 0.5, f"{fn}\nnot found", ha="center", va="center")
            continue
        z = np.load(fn, allow_pickle=True)
        m_enc = z["m_enc_msun"]
        X_pre  = z["X_pre"]
        X_post = z["X_post"]
        mass_cut = float(z["mass_cut"])
        mgne_pre  = float(z["mgne_pre"])
        mgne_post = float(z["mgne_post"])

        # x-range: focus on the 0 to 4.5 Msun range that the paper uses
        xlim = (0, min(m_enc[-1], 4.5))

        plot_panel(axes[i, 0], m_enc, X_pre, SPECIES,
                   f"{title} — Pre-SN  (Mg/Ne={mgne_pre:.3f})", xlim)
        plot_panel(axes[i, 1], m_enc, X_post, SPECIES,
                   f"{title} — Post-SN  (Mg/Ne={mgne_post:.3f})", xlim)
        # mark mass cut
        for ax in axes[i, :]:
            ax.axvline(mass_cut, color="k", linestyle=":", alpha=0.5, lw=0.8)

        if i == 0:
            axes[i, 0].legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.95)

    fig.suptitle(
        "N49B Fig 7 replication — explosive nucleosynthesis on Sukhbold+2018 IC\n"
        "Phase-C 13-species α-network (aprox13 rates ported from AMReX Microphysics)",
        fontsize=11, y=0.995,
    )
    fig.tight_layout()
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
