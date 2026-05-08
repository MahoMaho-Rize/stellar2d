#!/usr/bin/env python3
"""Interface-sharpness & wrinkling diagnostics for Andrassy 2022 runs.

For each VTK frame with a species_X tracer, compute:
    δ_int      = 1 / max_y |⟨∂X/∂y⟩_x|    (horizontally-averaged, "mean sharpness")
    δ_max      = 1 / max_{y,x} |∂X/∂y|    (local peak sharpness)
    L_iso      = length of the X = 0.5 isocontour            (wrinkling proxy)
    L_iso / Lx = dimensionless wrinkling factor              (1.0 = flat)
    y_iso_bar  = x-mean of the X = 0.5 isocontour y-coord    (matches y_ub)

Why these?  Following Leidi+ 2024 and classical entrainment theory
    Ṁ_e ∝ D_eff · (ρ · Ā) / δ_int,   Ā ≡ L_iso · Lx
so measuring δ and L_iso separately lets us decompose the scheme-dependent
spread in Ṁ_e into (1) numerical diffusivity D_eff/δ  and (2) wrinkling Ā/Lx.

Output: interface.csv  with columns t, y_iso_bar, delta_int, delta_max,
L_iso_over_Lx.

Usage:
    python3 scripts/andrassy2022/interface_diagnostics.py \\
        --run-dir runs/scheme_scan_512/muscl_vanleer/andrassy2022_512x512_* \\
        --vtk-dt 20.0
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose import parse_vtk


def read_species_X(vtk_path: Path, rho: np.ndarray) -> np.ndarray | None:
    """Read the species_X scalar (cell-centered, shape (ny, nx))."""
    txt = vtk_path.read_text()
    ny, nx = rho.shape
    m = re.search(
        r"SCALARS\s+species_X\s+double\s+1\s*\nLOOKUP_TABLE\s+default\s*\n", txt)
    if m is None:
        return None
    start = m.end()
    end_m = re.search(r"\n(SCALARS|VECTORS)", txt[start:])
    end = start + end_m.start() if end_m else len(txt)
    arr = np.fromstring(txt[start:end], sep=" ")[: nx * ny]
    return arr.reshape(ny, nx)


_FIG_CACHE = {"fig": None, "ax": None}


def isocontour_length(X: np.ndarray, x_nodes: np.ndarray, y_nodes: np.ndarray,
                      level: float = 0.5) -> float:
    """Length of the level-set contour X = level via matplotlib contour generator.

    Uses `ax.contour(...).allsegs`, which returns a list-of-polylines per level.
    Reuses a hidden figure to avoid PyPlot overhead.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    # Min or max values bracketing level — contour fails if all equal or one side
    vmin, vmax = float(np.min(X)), float(np.max(X))
    if not (vmin < level < vmax):
        return 0.0

    ny, nx = X.shape
    xc = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    yc = 0.5 * (y_nodes[:-1] + y_nodes[1:])

    if _FIG_CACHE["fig"] is None:
        _FIG_CACHE["fig"], _FIG_CACHE["ax"] = plt.subplots()
    ax = _FIG_CACHE["ax"]
    ax.clear()
    cs = ax.contour(xc, yc, X, levels=[level])
    total = 0.0
    # allsegs is list-of-level; each level is list of polylines (Nx2 arrays).
    # Try newer API (allsegs attribute) then fall back to get_paths().
    try:
        segs = cs.allsegs[0]
    except AttributeError:
        segs = [p.vertices for p in cs.get_paths()]
    for poly in segs:
        if len(poly) < 2:
            continue
        d = np.diff(np.asarray(poly), axis=0)
        total += float(np.hypot(d[:, 0], d[:, 1]).sum())
    return total


def mean_isocontour_y(X: np.ndarray, yc: np.ndarray, level: float = 0.5) -> float:
    """Mean y of the X=level contour (per x-column, linear interp)."""
    ny, nx = X.shape
    ys = []
    for ic in range(nx):
        col = X[:, ic]
        # find first crossing
        above = col >= level
        if above.all() or (~above).all():
            continue
        # bracket
        idx = np.where(np.diff(above.astype(int)) != 0)[0]
        if idx.size == 0:
            continue
        j = idx[-1]   # last crossing (bottom-to-top)
        f0, f1 = col[j], col[j+1]
        if f1 == f0:
            y_cross = 0.5 * (yc[j] + yc[j+1])
        else:
            t = (level - f0) / (f1 - f0)
            y_cross = yc[j] + t * (yc[j+1] - yc[j])
        ys.append(y_cross)
    return float(np.mean(ys)) if ys else float('nan')


def interface_thickness(X: np.ndarray, yc: np.ndarray) -> tuple[float, float]:
    """Return (δ_int_mean, δ_max_local).

    δ_int_mean:  1 / max_y |⟨∂X/∂y⟩_x|   — horizontally averaged profile first
    δ_max_local: 1 / max_{y,x} |∂X/∂y|   — raw peak
    """
    ny, nx = X.shape
    dy = yc[1] - yc[0] if ny > 1 else 1.0
    # mean profile
    Xbar = X.mean(axis=1)                          # (ny,)
    dXbar_dy = np.gradient(Xbar, dy)               # (ny,)
    peak_mean = float(np.max(np.abs(dXbar_dy))) if np.any(np.isfinite(dXbar_dy)) else 0.0
    # local
    dX_dy = np.gradient(X, dy, axis=0)             # (ny, nx)
    peak_local = float(np.max(np.abs(dX_dy)))
    return (1.0 / peak_mean if peak_mean > 0 else float('nan'),
            1.0 / peak_local if peak_local > 0 else float('nan'))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--vtk-dt", type=float, default=20.0,
                    help="time interval between VTK frames (driver --vtk-dt)")
    ap.add_argument("--level", type=float, default=0.5,
                    help="isocontour level for wrinkling (default 0.5)")
    args = ap.parse_args()

    # Allow glob in run-dir
    candidates = sorted(glob.glob(args.run_dir))
    if not candidates:
        print(f"no match for {args.run_dir}"); return 1
    run_dir = Path(candidates[0])

    vtk_files = sorted(
        [f for f in run_dir.glob("output_*.vtk") if "final" not in f.name])
    if not vtk_files:
        print(f"no output_*.vtk in {run_dir}"); return 1

    print(f"Processing {len(vtk_files)} VTK frames in {run_dir}")
    rows = []
    for i, f in enumerate(vtk_files):
        m = re.search(r"output_(\d+)\.vtk", f.name)
        frame_idx = int(m.group(1)) if m else i
        t = (frame_idx + 1) * args.vtk_dt

        data = parse_vtk(f)
        X = read_species_X(f, data["rho"])
        if X is None:
            continue
        x_nodes = np.asarray(data["x_nodes"], dtype=float)
        y_nodes = np.asarray(data["y_nodes"], dtype=float)
        yc = 0.5 * (y_nodes[:-1] + y_nodes[1:])
        Lx = float(x_nodes[-1] - x_nodes[0])

        y_iso = mean_isocontour_y(X, yc, args.level)
        delta_mean, delta_local = interface_thickness(X, yc)
        L_iso = isocontour_length(X, x_nodes, y_nodes, args.level)

        rows.append(dict(
            t=t, y_iso=y_iso,
            delta_int=delta_mean,
            delta_max=delta_local,
            L_iso_over_Lx=L_iso / Lx if Lx > 0 else float('nan'),
        ))
        if i % 20 == 0 or i == len(vtk_files) - 1:
            print(f"  [{i+1}/{len(vtk_files)}] t={t:.2f}  y_iso={y_iso:.3f}  "
                  f"δ_int={delta_mean:.3e}  L/Lx={rows[-1]['L_iso_over_Lx']:.3f}")

    csv_path = run_dir / "interface.csv"
    with csv_path.open("w") as fp:
        fp.write("t,y_iso,delta_int,delta_max,L_iso_over_Lx\n")
        for r in rows:
            fp.write(f"{r['t']:.6e},{r['y_iso']:.6e},{r['delta_int']:.6e},"
                     f"{r['delta_max']:.6e},{r['L_iso_over_Lx']:.6e}\n")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
