#!/usr/bin/env python3
"""Analyze Mach number evolution across VTK frames.

Usage:
    python scripts/check_mach.py <run_dir>
    python scripts/check_mach.py <run_dir> --plot   # also save mach_history.png
"""

import numpy as np
import os, sys, glob


def parse_vtk_fields(path):
    """Parse VTK structured grid, return (nr, nt, field_dict)."""
    with open(path) as f:
        lines = f.readlines()
    for l in lines:
        if l.startswith("DIMENSIONS"):
            nj, ni = int(l.split()[1]), int(l.split()[2])
            break
    nr, nt = ni - 1, nj - 1

    data = {}
    idx = 0
    while idx < len(lines):
        if lines[idx].startswith("SCALARS"):
            name = lines[idx].split()[1]
            idx += 2
            vals = []
            while len(vals) < nr * nt:
                vals.extend(float(x) for x in lines[idx].split())
                idx += 1
            data[name] = np.array(vals).reshape(nr, nt)
        elif lines[idx].startswith("VECTORS"):
            name = lines[idx].split()[1]
            idx += 1
            vals = []
            while len(vals) < nr * nt * 3:
                vals.extend(float(x) for x in lines[idx].split())
                idx += 1
            v = np.array(vals).reshape(nr, nt, 3)
            data[name + "_x"] = v[:, :, 0]
            data[name + "_z"] = v[:, :, 2]
        else:
            idx += 1
    return nr, nt, data


def analyze_frame(nr, nt, data, rho_thresh=1e-6):
    """Compute Mach statistics for one frame."""
    rho = data["density"]
    mach = data.get("mach", np.zeros_like(rho))
    vx = data.get("velocity_x", np.zeros_like(rho))
    vz = data.get("velocity_z", np.zeros_like(rho))
    speed = np.sqrt(vx**2 + vz**2)

    mask = rho > rho_thresh
    if not mask.any():
        return dict(max_mach=0, max_speed=0, min_rho=rho.min(), max_rho=rho.max(),
                    mach_99=0, mach_loc=(0, 0), rho_at_maxmach=0)

    max_mach = mach[mask].max()
    max_speed = speed[mask].max()
    mach_99 = np.percentile(mach[mask], 99)

    idx_max = np.unravel_index(np.argmax(mach * mask), mach.shape)

    return dict(
        max_mach=max_mach,
        max_speed=max_speed,
        min_rho=rho.min(),
        max_rho=rho.max(),
        mach_99=mach_99,
        mach_loc=idx_max,
        rho_at_maxmach=rho[idx_max],
    )


def main():
    do_plot = "--plot" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run_dir = args[0] if args else "."

    files = sorted(glob.glob(os.path.join(run_dir, "output_????.vtk")))
    if not files:
        print(f"No VTK files in {run_dir}")
        sys.exit(1)

    print(f"Analyzing {len(files)} frames in {run_dir}")
    print(f"{'frame':>6} {'max_M':>8} {'99%_M':>8} {'max|v|':>10} "
          f"{'min_rho':>10} {'max_rho':>10} {'rho@Mmax':>10} {'loc(i,j)':>10}")

    history = []
    for path in files:
        frame = os.path.basename(path).replace("output_", "").replace(".vtk", "")
        nr, nt, data = parse_vtk_fields(path)
        s = analyze_frame(nr, nt, data)
        history.append(s)

        loc = f"({s['mach_loc'][0]},{s['mach_loc'][1]})"
        print(f"{frame:>6} {s['max_mach']:8.3f} {s['mach_99']:8.4f} {s['max_speed']:10.3e} "
              f"{s['min_rho']:10.3e} {s['max_rho']:10.3e} {s['rho_at_maxmach']:10.3e} {loc:>10}")

    if do_plot and len(history) > 1:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        frames = range(len(history))
        max_m = [h["max_mach"] for h in history]
        m99 = [h["mach_99"] for h in history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        ax1.plot(frames, max_m, "r.-", label="max Mach")
        ax1.plot(frames, m99, "b.--", label="99th %ile Mach")
        ax1.axhline(1.0, color="k", ls=":", alpha=0.5, label="M=1")
        ax1.set_ylabel("Mach number")
        ax1.legend()
        ax1.set_title(f"Mach history — {os.path.basename(run_dir)}")
        ax1.grid(True, alpha=0.3)

        rho_min = [h["min_rho"] for h in history]
        rho_max = [h["max_rho"] for h in history]
        ax2.semilogy(frames, rho_max, "r.-", label="max ρ")
        ax2.semilogy(frames, rho_min, "b.-", label="min ρ")
        ax2.set_xlabel("output frame")
        ax2.set_ylabel("density")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(run_dir, "mach_history.png")
        plt.savefig(out_path, dpi=150)
        print(f"\nSaved plot: {out_path}")


if __name__ == "__main__":
    main()
