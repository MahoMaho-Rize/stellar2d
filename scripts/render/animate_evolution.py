#!/usr/bin/env python3
"""Generate a GIF animation of stellar2d density evolution from VTK output."""

import os, re, sys, glob
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image


def parse_vtk(path):
    """Parse stellar2d VTK structured grid file."""
    with open(path) as f:
        text = f.read()

    # Dimensions
    m = re.search(r"DIMENSIONS\s+(\d+)\s+(\d+)", text)
    nt_nodes, nr_nodes = int(m.group(1)), int(m.group(2))
    nt, nr = nt_nodes - 1, nr_nodes - 1

    # Points
    m = re.search(r"POINTS\s+\d+\s+\w+\n([\s\S]*?)(?=CELL_DATA)", text)
    pts = np.fromstring(m.group(1), sep=" ").reshape(-1, 3)

    # r_face and theta_face from node coords
    r_face = np.sqrt(pts[::nt_nodes, 0] ** 2 + pts[::nt_nodes, 2] ** 2)
    theta_face = np.arctan2(pts[-nt_nodes:, 0], pts[-nt_nodes:, 2])

    # Cell data fields
    fields = {}
    for m in re.finditer(
        r"SCALARS\s+(\w+)\s+\w+.*?\n(?:LOOKUP_TABLE.*?\n)?([\s\S]*?)(?=SCALARS|VECTORS|\Z)",
        text,
    ):
        name = m.group(1)
        vals = np.fromstring(m.group(2), sep=" ")
        if len(vals) == nr * nt:
            fields[name] = vals.reshape(nr, nt)

    return nr, nt, r_face, theta_face, fields


def make_animation(run_dir, output_path="animation.gif", fps=15, skip=1):
    files = sorted(glob.glob(os.path.join(run_dir, "output_????.vtk")))
    if not files:
        print(f"No VTK files in {run_dir}")
        return
    files = files[::skip]
    n_frames = len(files)
    print(f"Generating animation: {n_frames} frames from {run_dir}")

    # Parse first frame for grid geometry
    nr, nt, r_face, theta_face, _ = parse_vtk(files[0])

    # Build mesh for pcolormesh (x-z plane)
    R, T = np.meshgrid(r_face, theta_face, indexing="ij")
    X = R * np.sin(T)
    Z = R * np.cos(T)

    # Pre-scan for global color range
    rho_min, rho_max = 1e30, 0
    for f in files[:: max(1, n_frames // 20)]:
        _, _, _, _, fields = parse_vtk(f)
        rho = fields.get("density")
        if rho is not None:
            rho_min = min(rho_min, rho[rho > 0].min())
            rho_max = max(rho_max, rho.max())
    rho_min = max(rho_min, 1e-6)
    print(f"  ρ range: [{rho_min:.2e}, {rho_max:.2e}]")

    frames = []
    for idx, fpath in enumerate(files):
        _, _, _, _, fields = parse_vtk(fpath)
        rho = fields.get("density", np.ones((nr, nt)))
        rho = np.clip(rho, rho_min, None)

        fig, axes = plt.subplots(
            1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1.2, 1]}
        )

        # Left: 2D density map (half-star cross section)
        ax = axes[0]
        pcm = ax.pcolormesh(
            X,
            Z,
            rho,
            norm=LogNorm(vmin=rho_min, vmax=rho_max),
            cmap="inferno",
            shading="flat",
        )
        # Mirror to show full star
        ax.pcolormesh(
            -X,
            Z,
            rho,
            norm=LogNorm(vmin=rho_min, vmax=rho_max),
            cmap="inferno",
            shading="flat",
        )
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("z (polar axis)")
        ax.set_title(f"Density (log scale)")
        plt.colorbar(pcm, ax=ax, label="ρ", shrink=0.8)

        # Right: radial profile at equator
        ax2 = axes[1]
        r_center = 0.5 * (r_face[:-1] + r_face[1:])
        j_eq = nt // 2
        rho_eq = rho[:, j_eq]

        # Also get pressure if available
        P = fields.get("pressure")
        ax2.semilogy(r_center, rho_eq, "r-", lw=2, label="ρ (equator)")
        if P is not None:
            P_eq = np.clip(P[:, j_eq], 1e-10, None)
            ax2.semilogy(r_center, P_eq, "b--", lw=1.5, label="P (equator)")
        ax2.set_xlabel("r")
        ax2.set_ylabel("value")
        ax2.set_ylim(rho_min * 0.1, rho_max * 10)
        ax2.legend(loc="upper right", fontsize=9)
        ax2.set_title("Radial profile")
        ax2.grid(True, alpha=0.3)

        frame_num = int(re.search(r"(\d+)", os.path.basename(fpath)).group(1))
        fig.suptitle(
            f"stellar2d — Lane-Emden perturbed (frame {frame_num})",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()

        # Render to PIL Image
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), buf).convert(
            "RGB"
        )
        frames.append(img)
        plt.close(fig)

        if (idx + 1) % 20 == 0 or idx == n_frames - 1:
            print(f"  rendered {idx + 1}/{n_frames}")

    # Save GIF
    duration = int(1000 / fps)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
    print(
        f"Saved: {output_path} ({len(frames)} frames, {os.path.getsize(output_path) / 1024:.0f} KB)"
    )


if __name__ == "__main__":
    run_dir = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "runs/lane_emden_perturbed_64x32_20260428_223528"
    )
    out = sys.argv[2] if len(sys.argv) > 2 else "stellar_evolution.gif"
    skip = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    make_animation(run_dir, out, fps=12, skip=skip)
