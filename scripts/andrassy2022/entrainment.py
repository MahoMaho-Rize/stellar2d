#!/usr/bin/env python3
"""Entrainment diagnostic for cart_ale2 Andrassy 2022 runs w/ species tracer.

Andrassy 2022 §3.2 defines the entrained mass:
    M_e(t) = ∫_{conv layer} ρ · X_1 dV
where X_1 is the μ_1 mass fraction (= 1 in the stable layer, 0 in the conv
layer at t=0) and "conv layer" = y < y_ub(t).  Entrainment rate Ṁ_e comes
from d M_e / d t in the saturated regime.

Reads every output_*.vtk in --run-dir, extracts density + species_X,
computes M_e(t) (mass of μ_1 fluid that's been dragged below y_ub), and
writes entrainment.csv + entrainment.png with Andrassy Fig. 11/12 overlay.

Usage:
    python3 scripts/andrassy2022/entrainment.py \\
        --run-dir runs/andrassy2022_256x256_20260506_XXXX
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose import parse_vtk, horizontal_profiles, find_y_ub, Y_SHIFT, GAMMA

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def compute_entrained_mass(vtk_path: Path, buffer: float = 0.1) -> tuple[float, float]:
    """Return (y_ub_local, M_e) for this VTK frame.

    M_e = ∫_{y < y_ub − buffer} ρ·X dV, integrated as Σ_cells ρ·X·dV using
    uniform dA.  Only counts mass that has been *dragged down* into the
    conv-layer bulk (excluding the transition zone near y_ub to match
    Andrassy §3.1 bulk-averaging convention).  Set buffer=0 to integrate
    up to y_ub directly.
    """
    data = parse_vtk(vtk_path)
    # Try to read species_X as a scalar.
    import re as _re
    txt = vtk_path.read_text()
    m = _re.search(r"SCALARS\s+species_X\s+double\s+1\s*\nLOOKUP_TABLE\s+default\s*\n",
                   txt)
    if m is None:
        return float("nan"), float("nan")
    start = m.end()
    end_m = _re.search(r"\n(SCALARS|VECTORS)", txt[start:])
    end = start + end_m.start() if end_m else len(txt)
    nx = data["nx"]; ny = data["ny"]
    X_arr = np.fromstring(txt[start:end], sep=" ")[: nx*ny].reshape(ny, nx)

    rho = data["rho"]  # (ny, nx)
    y_nodes = data["y_nodes"]
    x_nodes = data["x_nodes"]
    dx = float(x_nodes[-1] - x_nodes[0]) / nx
    dy = float(y_nodes[-1] - y_nodes[0]) / ny
    dV = dx * dy

    prof = horizontal_profiles(data)
    y_ub_local = find_y_ub(prof)
    y_c = prof["y"]
    mask_conv = y_c < (y_ub_local - buffer)   # row mask, with buffer
    M_e = 0.0
    for jy in range(ny):
        if not mask_conv[jy]:
            continue
        M_e += rho[jy, :].dot(X_arr[jy, :]) * dV
    return y_ub_local, M_e


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--vtk-dt", type=float, default=12.06)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or args.run_dir

    vtks = sorted(f for f in args.run_dir.glob("output_*.vtk")
                  if "final" not in f.name)
    if not vtks:
        print(f"No output_*.vtk in {args.run_dir}"); return 1

    print(f"Processing {len(vtks)} VTK frames")
    rows = []
    for i, f in enumerate(vtks):
        m = re.search(r"output_(\d+)\.vtk", f.name)
        t = (int(m.group(1)) + 1) * args.vtk_dt if m else float("nan")
        y_ub, M_e = compute_entrained_mass(f)
        rows.append((t, y_ub + Y_SHIFT, M_e))
        if i % 20 == 0 or i == len(vtks) - 1:
            print(f"  [{i+1}/{len(vtks)}] t={t:.2f}  y_ub={rows[-1][1]:.3f}  M_e={M_e:.6e}")

    out_csv = out_dir / "entrainment.csv"
    with out_csv.open("w") as fp:
        fp.write("t,y_ub,M_e\n")
        for r in rows:
            fp.write(f"{r[0]:.6e},{r[1]:.6e},{r[2]:.6e}\n")
    print(f"Wrote {out_csv}")

    if plt is None:
        print("matplotlib unavailable — skipping plot"); return 0

    rows_a = np.array(rows)
    t_a, y_ub_a, Me_a = rows_a.T

    # Linear fit Ṁ_e in saturation window t ∈ [500, 2000] (skip initial ramp).
    mask = (t_a >= 500) & (t_a <= 2000) & np.isfinite(Me_a)
    if mask.sum() > 5:
        p = np.polyfit(t_a[mask], Me_a[mask], 1)
        Mdot_e = p[0]
    else:
        Mdot_e = float("nan")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(t_a, Me_a, "k-", lw=1.8, label="stellar2d 2D 256²")
    if np.isfinite(Mdot_e):
        tl = np.array([500, 2000])
        yl = np.polyval(p, tl)
        ax1.plot(tl, yl, "r--", lw=1.2,
                 label=f"fit Ṁ_e = {Mdot_e:.3e}")
    # Paper reports Ṁ_e ~ 1e-4 total entrainment rate (Fig. 12) -
    # reference line.
    ax1.axhline(0.0, color="gray", lw=0.5)
    ax1.set_xlabel("t")
    ax1.set_ylabel(r"$M_e$ = $\int_{\rm conv} \rho X dV$")
    ax1.set_title("Entrained mass M_e(t)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.plot(t_a, y_ub_a, "k-", lw=1.8)
    ax2.axhline(2.0, color="gray", ls="--", lw=0.8)
    ax2.set_xlabel("t"); ax2.set_ylabel("y_ub (Andrassy coords)")
    ax2.set_title("Upper boundary y_ub(t)")
    ax2.grid(alpha=0.3)
    ax2.set_ylim(1.9, 2.6)

    fig.suptitle(f"Andrassy 2022 entrainment — {args.run_dir.name}")
    fig.tight_layout()
    out_png = out_dir / "entrainment.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
