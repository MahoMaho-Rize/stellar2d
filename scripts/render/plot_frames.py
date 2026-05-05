#!/usr/bin/env python3
"""Render all VTK frames as PNG (density + velocity + Mach), then optionally GIF.

Usage:
    python scripts/plot_frames.py <run_dir>
    python scripts/plot_frames.py <run_dir> --gif   # also assemble GIF
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os, sys, glob


def parse_vtk(path):
    with open(path) as f:
        lines = f.readlines()
    for l in lines:
        if l.startswith("DIMENSIONS"):
            nj, ni = int(l.split()[1]), int(l.split()[2])
            break
    nr, nt = ni - 1, nj - 1
    pts_start = None
    for idx, l in enumerate(lines):
        if l.startswith("POINTS"):
            pts_start = idx + 1
            npts = int(l.split()[1])
            break
    pts = []
    for i in range(pts_start, pts_start + npts):
        pts.append([float(x) for x in lines[i].split()])
    pts = np.array(pts).reshape(ni, nj, 3)
    x_grid, z_grid = pts[:, :, 0], pts[:, :, 2]

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
    return nr, nt, x_grid, z_grid, data


def main():
    do_gif = "--gif" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run_dir = args[0] if args else "."

    out_dir = os.path.join(run_dir, "frames")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(run_dir, "output_????.vtk")))
    if not files:
        print(f"No VTK files in {run_dir}")
        sys.exit(1)
    print(f"{len(files)} frames → {out_dir}/")

    for fi, path in enumerate(files):
        frame = os.path.basename(path).replace("output_", "").replace(".vtk", "")
        nr, nt, xg, zg, data = parse_vtk(path)

        rho = data["density"]
        vx = data.get("velocity_x", np.zeros_like(rho))
        vz = data.get("velocity_z", np.zeros_like(rho))
        mach = data.get("mach", np.zeros_like(rho))
        speed = np.sqrt(vx**2 + vz**2)

        xc = 0.25 * (xg[:-1, :-1] + xg[1:, :-1] + xg[:-1, 1:] + xg[1:, 1:])
        zc = 0.25 * (zg[:-1, :-1] + zg[1:, :-1] + zg[:-1, 1:] + zg[1:, 1:])

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        ax = axes[0]
        rho_plot = np.maximum(rho, 1e-15)
        pcm = ax.pcolormesh(xg, zg, rho_plot, cmap="inferno",
                            norm=LogNorm(vmin=1e-6, vmax=2.0))
        ax.set_aspect("equal")
        ax.set_title("density (log)")
        plt.colorbar(pcm, ax=ax, shrink=0.8)

        ax2 = axes[1]
        smax = max(speed.max(), 1e-8)
        pcm2 = ax2.pcolormesh(xg, zg, speed, cmap="coolwarm", vmin=0, vmax=smax)
        skip = max(1, nr // 16)
        ax2.quiver(xc[::skip, ::skip], zc[::skip, ::skip],
                   vx[::skip, ::skip], vz[::skip, ::skip],
                   color="k", scale_units="xy", angles="xy",
                   scale=max(smax * 5, 1e-5), width=0.003)
        ax2.set_aspect("equal")
        ax2.set_title(f"|v| + flow  (max={smax:.2e})")
        plt.colorbar(pcm2, ax=ax2, shrink=0.8)

        ax3 = axes[2]
        mach_plot = np.maximum(mach, 1e-10)
        mach_max = max(mach_plot.max(), 1e-5)
        if mach_max > 1:
            pcm3 = ax3.pcolormesh(xg, zg, mach_plot, cmap="RdYlBu_r",
                                  norm=LogNorm(vmin=1e-4, vmax=mach_max))
        else:
            pcm3 = ax3.pcolormesh(xg, zg, mach_plot, cmap="RdYlBu_r",
                                  vmin=0, vmax=max(mach_max, 0.01))
        ax3.set_aspect("equal")
        ax3.set_title(f"Mach (max={mach_max:.2e})")
        plt.colorbar(pcm3, ax=ax3, shrink=0.8)

        for ax in axes:
            ax.set_xlabel("x")
            ax.set_ylabel("z")

        fig.suptitle(f"Frame {frame}  (#{fi})", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"frame_{frame}.png"),
                    dpi=120, bbox_inches="tight")
        plt.close(fig)

        if (fi + 1) % 20 == 0 or fi == len(files) - 1:
            print(f"  [{fi + 1}/{len(files)}] done")

    print(f"All frames saved to {out_dir}/")

    if do_gif:
        from PIL import Image
        pngs = sorted(glob.glob(os.path.join(out_dir, "frame_*.png")))
        if pngs:
            images = [Image.open(f) for f in pngs]
            gif_path = os.path.join(run_dir, "evolution.gif")
            images[0].save(gif_path, save_all=True, append_images=images[1:],
                           duration=300, loop=0)
            print(f"{len(images)} frames → {gif_path}")


if __name__ == "__main__":
    main()
