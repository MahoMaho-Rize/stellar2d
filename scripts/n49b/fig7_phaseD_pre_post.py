#!/usr/bin/env python3
"""Fig 7 (Sato+2024) replication via radial1d implicit hydro.

4×2 panel grid: pre-SN + post-SN mass-fraction profiles for the four
progenitors (12.02 / 12.75 / 15.28 / 15.90 Msun).  Post-SN source data
comes from the Phase D sweep (data/n49b_postSN/postSN_*.npz) using the
best-matching (mass_cut, E_SN) per progenitor (see phaseD_sweep.csv).

Unlike the Phase C post-processing version (which read explosive_nucleo.py
output), this script reads the radial1d implicit-hydro output directly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sukhbold_reader import read_one, MSUN_CGS  # noqa: E402


# Paper's best-match configuration per progenitor (from phaseD_sweep.csv).
# Note: baseline-matching antipattern warning — these are Mg/Ne-best,
# subsequent audit should confirm Si/Ne/Ni56 agree at the same knobs.
BEST = {
    "12.02": ("1.4", "5.0e+50"),
    "12.75": ("1.8", "1.0e+51"),
    "15.28": ("1.8", "1.0e+51"),
    "15.90": ("1.8", "2.0e+51"),
}

SPECIES_POST = [
    (0, "He",  "C1"),
    (1, "C",   "C2"),
    (2, "O",   "C4"),
    (3, "Ne",  "C5"),
    (4, "Mg",  "C6"),
    (5, "Si",  "C7"),
]

# Sukhbold pre-SN: use raw X for the same 6 species (fold H into He).
def load_presn(dat_path: Path):
    prof = read_one(dat_path)
    m_enc = prof.m_enc / MSUN_CGS
    X = np.zeros((len(m_enc), 6))
    X[:, 0] = prof.X("He4") + prof.X("He3") + prof.X("H1")
    X[:, 1] = prof.X("C12") + prof.X("N14")
    X[:, 2] = prof.X("O16")
    X[:, 3] = prof.X("Ne20")
    X[:, 4] = prof.X("Mg24")
    X[:, 5] = prof.X("Si28")
    return m_enc, X


def load_post(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    nz = d["rho"].shape[0]
    mc_g = float(d["mass_cut"]) * MSUN_CGS
    # Recompute zone enclosed mass assuming equal-mass Lagrangian shells
    # above the mass cut.  Per-zone dm is stored implicitly via
    # M_ejecta_msun / nz.
    M_eject = float(d["M_ejecta_msun"]) * MSUN_CGS
    dm = M_eject / nz
    m_enc = (mc_g + (np.arange(nz) + 0.5) * dm) / MSUN_CGS
    # Map 13-species → 6 (He, C, O, Ne, Mg, Si)
    X13 = d["X"]
    X6 = X13[:, :6]
    return m_enc, X6


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
    ap.add_argument("--npz-dir", default="data/n49b_postSN")
    ap.add_argument("--presn-dir", default="~/data/sukhbold_2018/mdotone")
    ap.add_argument("--out", default="docs/images/n49b_fig7_phaseD.png")
    args = ap.parse_args()

    npz_dir = Path(args.npz_dir)
    presn_dir = Path(args.presn_dir).expanduser()

    fig, axes = plt.subplots(4, 2, figsize=(11, 14), dpi=130)

    models = [
        ("12.02", "(a) 12.02 $M_\\odot$"),
        ("12.75", "(b) 12.75 $M_\\odot$"),
        ("15.28", "(c) 15.28 $M_\\odot$"),
        ("15.90", "(d) 15.90 $M_\\odot$"),
    ]

    for i, (zams, title) in enumerate(models):
        mc, Es = BEST[zams]
        presn_path = presn_dir / f"{zams}.dat"
        npz_path = npz_dir / f"postSN_{zams}_mc{mc}_E{Es}.npz"
        if not (presn_path.exists() and npz_path.exists()):
            axes[i, 0].text(0.5, 0.5, f"missing:\n{presn_path}\n{npz_path}",
                            ha="center", va="center")
            continue

        m_pre, X_pre = load_presn(presn_path)
        m_post, X_post = load_post(npz_path)

        # Restrict to 0-4.5 Msun (paper convention)
        xlim = (0, 4.5)

        # Compute diagnostic Mg/Ne in pre-SN O-rich region (X_O>0.4)
        mask_pre = X_pre[:, 2] > 0.4
        dm_pre = np.gradient(m_pre)
        m_Ne_pre = float((X_pre[:, 3] * dm_pre)[mask_pre].sum())
        m_Mg_pre = float((X_pre[:, 4] * dm_pre)[mask_pre].sum())
        mgne_pre = m_Mg_pre / max(m_Ne_pre, 1e-30)

        d = np.load(npz_path, allow_pickle=True)
        mgne_post = float(d["Mg_Ne"])
        paper_mgne = float(d["Mg_Ne_paper"])

        plot_panel(axes[i, 0], m_pre, X_pre, SPECIES_POST,
                   f"{title} — Pre-SN (Mg/Ne={mgne_pre:.3f})", xlim)
        plot_panel(axes[i, 1], m_post, X_post, SPECIES_POST,
                   f"{title} — Post-SN (Mg/Ne={mgne_post:.3f}, paper {paper_mgne})",
                   xlim)
        mc_val = float(mc)
        for ax in axes[i, :]:
            ax.axvline(mc_val, color="k", linestyle=":", alpha=0.5, lw=0.8)

        if i == 0:
            axes[i, 0].legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.95)

    fig.suptitle(
        "N49B Fig 7 replication — radial1d implicit 1D hydro SN explosion\n"
        "13-species α-chain (aprox13 rates), mass-cut inner BC + thermal bomb",
        fontsize=11, y=0.995,
    )
    fig.tight_layout()
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
