#!/usr/bin/env python3
"""Time-series diagnostics for an Andrassy 2022 cart_ale2 pilot run.

Reads every output_*.vtk in --run-dir, computes horizontally-averaged
diagnostics at each frame (using the faithful Eq. 15 mass-weighted
fluctuation v_rms definition from diagnose.py), and writes:
  - timeseries.csv: t, y_ub, v_rms_conv, v_rms_stable, max_v_rms
  - timeseries.png: 4-panel time series overlaid with 5-code Andrassy data

Matches Andrassy 2022 Fig. 3 (v_rms conv/stable vs t) and the yub(t) plot.

Usage:
    python3 scripts/andrassy2022/timeseries.py \\
        --run-dir runs/andrassy_formal_256 \\
        --out-dir runs/andrassy_formal_256
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np

# Reuse the parsers from diagnose.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose import (parse_vtk, horizontal_profiles, find_y_ub,
                      bulk_averages, parse_rprof, Y_SHIFT)

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def vtk_time(path: Path, diag_csv_map: dict, output_interval: int,
             vtk_dt: float = 0.0) -> float:
    """Estimate simulation time for VTK frame `output_XXXX.vtk`.

    If --vtk-dt > 0 (time-based output), frame N was written at t = (N+1)·Δt
    (index base 1; cart_ale2 driver increments `frame` before writing).
    Otherwise the driver writes every `output_interval` steps, so frame N
    corresponds to step (N+1)·output_interval — look up the nearest recorded
    step in diagnostics.csv.
    """
    m = re.search(r"output_(\d+)\.vtk", path.name)
    if m is None:
        return float("nan")
    frame_idx = int(m.group(1))
    if vtk_dt > 0.0:
        return (frame_idx + 1) * vtk_dt
    step_target = (frame_idx + 1) * output_interval
    steps = np.array(sorted(diag_csv_map.keys()))
    if steps.size == 0:
        return float("nan")
    j = np.abs(steps - step_target).argmin()
    return diag_csv_map[int(steps[j])]


def load_csv(path: Path) -> tuple[dict, int]:
    """Parse the cart_ale2 diagnostics.csv into a step→t mapping."""
    m = {}
    with path.open() as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    m[int(parts[0])] = float(parts[1])
                except ValueError:
                    pass
    return m


def rprof_time_series(root: Path, code: str, res: int):
    """Return arrays (t, v_rms_conv, v_rms_stable, y_ub) across all rprofs
    for a given code and resolution, using Andrassy's STDEV_V{X,Y,Z} columns.

    v_rms_conv: spatially mass-weighted mean of sqrt(σ²_vx + σ²_vy + σ²_vz) in
                the region y < y_ub - 0.1.  Uses the rprof RHO column for
                weighting (Andrassy §3.1 mass-weighted bulk).
    """
    pattern = root / f"{code}/{code}-{res}/{code}-{res}-*.rprof"
    files = sorted(glob.glob(str(pattern)))
    ts = []
    vr_conv = []; vr_stab = []; yubs = []
    for f in files:
        rd = parse_rprof(Path(f))
        if not all(k in rd for k in ("RHO", "P", "STDEV_VX", "STDEV_VY")):
            continue
        y = rd["y"]; rho = rd["RHO"]
        sigma_vx = rd["STDEV_VX"]; sigma_vy = rd["STDEV_VY"]
        sigma_vz = rd.get("STDEV_VZ", np.zeros_like(sigma_vx))
        v_rms = np.sqrt(sigma_vx**2 + sigma_vy**2 + sigma_vz**2)
        # y_ub from pseudo-entropy rise (match diagnose.py: 0.1% threshold,
        # start search at Schwarzschild boundary y_paper=2.0).
        A = rd["P"] / rd["RHO"]**(5.0/3.0)
        mask_base = (y > 1.2) & (y < 1.8)
        if mask_base.sum() < 4:
            y_ub = y[-1]
        else:
            A_conv = A[mask_base].mean()
            y_ub = y[-1]
            for j, yj in enumerate(y):
                if yj >= 2.0 and A[j] > A_conv * 1.001:
                    y_ub = yj; break
        # Bulk mass-weighted v_rms² then sqrt.
        def bulk(mask):
            if mask.sum() == 0: return float("nan")
            w = rho[mask]; v2 = v_rms[mask]**2
            return float(np.sqrt((w*v2).sum() / w.sum()))
        ts.append(rd["t"])
        vr_conv.append(bulk(y < y_ub - 0.1))
        vr_stab.append(bulk(y > y_ub + 0.1))
        yubs.append(y_ub)
    return np.array(ts), np.array(vr_conv), np.array(vr_stab), np.array(yubs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--andrassy-root", type=Path,
                    default=Path("data/andrassy2022/1D-profiles"))
    ap.add_argument("--codes", nargs="+",
                    default=["FLASH", "MUSIC", "PPMSTAR", "PROMPI", "SLH"])
    ap.add_argument("--andrassy-res", type=int, default=256)
    ap.add_argument("--output-interval", type=int, default=4000,
                    help="VTK output interval in steps (must match driver)")
    ap.add_argument("--vtk-dt", type=float, default=12.06,
                    help="VTK output interval in simulation time (matches "
                         "driver --vtk-dt; 0 disables and uses step-based)")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    vtk_files = sorted(
        [f for f in args.run_dir.glob("output_*.vtk") if "final" not in f.name]
    )
    if not vtk_files:
        print(f"No output_*.vtk in {args.run_dir}"); return 1

    diag_csv = args.run_dir / "diagnostics.csv"
    diag_map = load_csv(diag_csv) if diag_csv.exists() else {}

    print(f"Processing {len(vtk_files)} VTK frames")
    rows = []
    for i, f in enumerate(vtk_files):
        t = vtk_time(f, diag_map, args.output_interval, args.vtk_dt)
        data = parse_vtk(f)
        prof = horizontal_profiles(data)
        y_ub_local = find_y_ub(prof)
        bulk = bulk_averages(prof, y_ub_local)
        rows.append(dict(
            t=t, frame=i,
            y_ub=y_ub_local + Y_SHIFT,
            v_rms_conv=bulk["v_rms_conv"],
            v_rms_stable=bulk["v_rms_stable"],
            v_rms_peak=prof["v_rms"].max(),
        ))
        if i % 20 == 0 or i == len(vtk_files) - 1:
            print(f"  [{i+1}/{len(vtk_files)}] t={t:.2f}  y_ub={rows[-1]['y_ub']:.3f}  "
                  f"v_rms_conv={rows[-1]['v_rms_conv']:.4e}")

    # CSV.
    csv_out = args.out_dir / "timeseries.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w") as f:
        f.write("t,frame,y_ub,v_rms_conv,v_rms_stable,v_rms_peak\n")
        for r in rows:
            f.write(f"{r['t']:.6e},{r['frame']},{r['y_ub']:.6e},"
                    f"{r['v_rms_conv']:.6e},{r['v_rms_stable']:.6e},"
                    f"{r['v_rms_peak']:.6e}\n")
    print(f"Wrote {csv_out}")

    if plt is None:
        print("matplotlib unavailable — skipping plot"); return 0

    t_ours = np.array([r["t"] for r in rows])
    vc_ours = np.array([r["v_rms_conv"] for r in rows])
    vs_ours = np.array([r["v_rms_stable"] for r in rows])
    yub_ours = np.array([r["y_ub"] for r in rows])

    # Load each code's time series.
    andrassy_series = {}
    for code in args.codes:
        ts, vc, vs, yubs = rprof_time_series(args.andrassy_root, code, args.andrassy_res)
        if ts.size:
            andrassy_series[code] = (ts, vc, vs, yubs)
            print(f"  {code}-{args.andrassy_res}: {ts.size} snapshots, "
                  f"t∈[{ts.min():.1f}, {ts.max():.1f}]")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    (ax_vc, ax_vs), (ax_yub, ax_peak) = axes
    colors = {"FLASH": "C0", "MUSIC": "C1", "PPMSTAR": "C2",
              "PROMPI": "C3", "SLH": "C4"}
    for code, (ts, vc, vs, yubs) in andrassy_series.items():
        c = colors.get(code, "gray")
        ax_vc.plot(ts, vc, "-", color=c, lw=1.2, alpha=0.75, label=f"{code}")
        ax_vs.plot(ts, vs, "-", color=c, lw=1.2, alpha=0.75, label=f"{code}")
        ax_yub.plot(ts, yubs, "-", color=c, lw=1.2, alpha=0.75, label=f"{code}")

    ax_vc.plot(t_ours, vc_ours, "k-", lw=2.2, label="stellar2d cart_ale2")
    ax_vs.plot(t_ours, vs_ours, "k-", lw=2.2, label="stellar2d cart_ale2")
    ax_yub.plot(t_ours, yub_ours, "k-", lw=2.2, label="stellar2d cart_ale2")
    ax_peak.plot(t_ours, np.array([r["v_rms_peak"] for r in rows]),
                 "k-", lw=2.2)

    ax_vc.axhline(0.034, color="magenta", lw=1, ls="-.", alpha=0.5,
                  label="paper saturation 0.034")
    ax_yub.axhline(2.0, color="gray", lw=0.8, ls="--", alpha=0.5)

    ax_vc.set_title("v_rms (convective layer)")
    ax_vs.set_title("v_rms (stable layer)")
    ax_yub.set_title("upper boundary y_ub(t)")
    ax_peak.set_title("peak v_rms(y)")
    for ax in axes.flat:
        ax.set_xlabel("t")
        ax.grid(alpha=0.3)
    ax_vc.legend(fontsize=8, loc="best")
    ax_yub.legend(fontsize=8, loc="best")
    ax_yub.set_ylim(1.9, 2.6)

    fig.suptitle(
        f"Andrassy 2022 pilot time series — {args.run_dir.name} vs 5-code "
        f"{args.andrassy_res}³", fontsize=12
    )
    fig.tight_layout()
    png_out = args.out_dir / "timeseries.png"
    fig.savefig(png_out, dpi=140, bbox_inches="tight")
    print(f"Wrote {png_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
