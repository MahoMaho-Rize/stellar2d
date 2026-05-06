#!/usr/bin/env python3
"""Build the Andrassy 2022 resolution-scan summary figure.

Reads timeseries.csv from runs/{128,256,512} and overlays them against
the 5-code Andrassy rprof reference at 256³.

Usage:
    python3 scripts/andrassy2022/resolution_scan.py --out paper/resolution_scan.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timeseries import rprof_time_series

import matplotlib.pyplot as plt


def load_run(d: Path):
    a = np.loadtxt(d / "timeseries.csv", delimiter=",", skiprows=1)
    return dict(t=a[:, 0], y_ub=a[:, 2], v_rms_conv=a[:, 3],
                v_rms_stable=a[:, 4], v_rms_peak=a[:, 5])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--r128", type=Path,
                    default=Path("runs/andrassy2022_128x128_20260506_155112"))
    ap.add_argument("--r256", type=Path,
                    default=Path("runs/andrassy2022_256x256_20260506_152128"))
    ap.add_argument("--r512", type=Path,
                    default=Path("runs/andrassy2022_512x512_20260506_155217"))
    ap.add_argument("--andrassy-root", type=Path,
                    default=Path("data/andrassy2022/1D-profiles"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ours = {"128²": load_run(args.r128),
            "256²": load_run(args.r256),
            "512²": load_run(args.r512)}
    our_colors = {"128²": "#1f77b4", "256²": "black", "512²": "#d62728"}
    our_lw = {"128²": 1.4, "256²": 2.3, "512²": 1.4}

    codes = ["FLASH", "MUSIC", "PPMSTAR", "PROMPI", "SLH"]
    andrassy = {}
    for c in codes:
        t, vc, vs, yubs = rprof_time_series(args.andrassy_root, c, 256)
        if t.size:
            andrassy[c] = (t, vc, vs, yubs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax_vc, ax_yub = axes

    for c, (t, vc, vs, yubs) in andrassy.items():
        ax_vc.plot(t, vc, "-", color="gray", lw=1.0, alpha=0.55,
                   label=f"{c} 3D" if c == "FLASH" else None)
        ax_yub.plot(t, yubs, "-", color="gray", lw=1.0, alpha=0.55,
                    label=f"{c} 3D" if c == "FLASH" else None)

    for label, d in ours.items():
        m = np.isfinite(d["v_rms_conv"])
        ax_vc.plot(d["t"][m], d["v_rms_conv"][m], "-",
                   color=our_colors[label], lw=our_lw[label],
                   label=f"stellar2d 2D {label}")
        ax_yub.plot(d["t"][m], d["y_ub"][m], "-",
                    color=our_colors[label], lw=our_lw[label],
                    label=f"stellar2d 2D {label}")

    ax_vc.axhline(0.0345, color="magenta", ls="-.", lw=0.9, alpha=0.55,
                  label="3D 5-code mean ≈ 0.0345")
    ax_vc.axvspan(1000, 2000, color="gold", alpha=0.08,
                  label="saturation window")
    ax_yub.axhline(2.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax_yub.axvspan(1000, 2000, color="gold", alpha=0.08)

    ax_vc.set_xlabel("t")
    ax_vc.set_ylabel(r"$\tilde v_{\rm rms}$ (convective bulk, mass-weighted)")
    ax_vc.set_title("v_rms(t) — resolution scan")
    ax_vc.grid(alpha=0.3)
    ax_vc.legend(fontsize=8, loc="lower right", ncol=2)

    ax_yub.set_xlabel("t")
    ax_yub.set_ylabel(r"$y_{\rm ub}(t)$  (Andrassy coords)")
    ax_yub.set_title("upper boundary y_ub(t)")
    ax_yub.grid(alpha=0.3)
    ax_yub.set_ylim(1.9, 3.0)
    ax_yub.legend(fontsize=8, loc="lower right")

    # Saturation stats table in a footnote.
    tbl_lines = ["Saturation window t∈[1000, 2000]   ⟨v_rms_conv⟩ ± σ"]
    for label, d in ours.items():
        msk = (d["t"] >= 1000) & (d["t"] <= 2000) & np.isfinite(d["v_rms_conv"])
        vc = d["v_rms_conv"][msk]
        mark = "  ⚠ not saturated" if (label == "512²") else ""
        tbl_lines.append(f"  stellar2d 2D {label}:  {vc.mean():.4f} ± {vc.std():.4f}{mark}")
    for c, (t, vc, vs, yubs) in andrassy.items():
        msk = (t >= 1000) & (t <= 2000)
        tbl_lines.append(f"  {c} 3D 256³:      {vc[msk].mean():.4f} ± {vc[msk].std():.4f}")
    fig.text(0.02, -0.02, "\n".join(tbl_lines), fontsize=8, family="monospace",
             va="top")

    fig.suptitle("Andrassy 2022 resolution scan — cart_ale2 2D vs Andrassy 5-code 3D",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
