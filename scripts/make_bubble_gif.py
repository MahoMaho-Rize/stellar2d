#!/usr/bin/env python3
"""Generate 3-panel GIF: density, velocity streamlines, speed field."""

import os, re, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from PIL import Image


def parse_vtk(path):
    with open(path) as f:
        text = f.read()

    m = re.search(r"DIMENSIONS\s+(\d+)\s+(\d+)", text)
    nt_nodes, nr_nodes = int(m.group(1)), int(m.group(2))
    nt, nr = nt_nodes - 1, nr_nodes - 1

    m = re.search(r"POINTS\s+\d+\s+\w+\n([\s\S]*?)(?=CELL_DATA)", text)
    pts = np.fromstring(m.group(1), sep=" ").reshape(-1, 3)

    r_face = np.sqrt(pts[::nt_nodes, 0]**2 + pts[::nt_nodes, 2]**2)
    theta_face = np.arctan2(pts[-nt_nodes:, 0], pts[-nt_nodes:, 2])

    fields = {}
    # Scalars
    for m in re.finditer(
        r"SCALARS\s+(\w+)\s+\w+.*?\n(?:LOOKUP_TABLE.*?\n)?([\s\S]*?)(?=SCALARS|VECTORS|\Z)", text):
        name = m.group(1)
        vals = np.fromstring(m.group(2), sep=" ")
        if len(vals) == nr * nt:
            fields[name] = vals.reshape(nr, nt)
    # Vectors
    for m in re.finditer(
        r"VECTORS\s+(\w+)\s+\w+\n([\s\S]*?)(?=SCALARS|VECTORS|\Z)", text):
        name = m.group(1)
        vals = np.fromstring(m.group(2), sep=" ")
        if len(vals) == nr * nt * 3:
            vdata = vals.reshape(nr, nt, 3)
            fields[name + "_x"] = vdata[:, :, 0]
            fields[name + "_y"] = vdata[:, :, 1]
            fields[name + "_z"] = vdata[:, :, 2]

    return nr, nt, r_face, theta_face, fields


def make_animation(run_dir, output_path="bubble_evolution.gif", fps=12, skip=1, max_frames=0):
    files = sorted(glob.glob(os.path.join(run_dir, "output_????.vtk")))
    if not files:
        print(f"No VTK files in {run_dir}"); return
    files = files[::skip]
    if max_frames > 0:
        files = files[:max_frames]
    n_frames = len(files)
    print(f"Generating 3-panel animation: {n_frames} frames from {run_dir}")

    nr, nt, r_face, theta_face, _ = parse_vtk(files[0])
    R, T = np.meshgrid(r_face, theta_face, indexing="ij")
    X = R * np.sin(T)
    Z = R * np.cos(T)

    # Cell centers for streamlines / quiver
    r_c = 0.5 * (r_face[:-1] + r_face[1:])
    t_c = 0.5 * (theta_face[:-1] + theta_face[1:])
    Rc, Tc = np.meshgrid(r_c, t_c, indexing="ij")
    Xc = Rc * np.sin(Tc)
    Zc = Rc * np.cos(Tc)

    # Pre-scan for color ranges (skip first 2 frames — initial transient)
    rho_min, rho_max = 1e30, 0
    spd_max = 0
    scan_files = files[2:] if len(files) > 2 else files
    for f in scan_files[::max(1, len(scan_files) // 10)]:
        _, _, _, _, flds = parse_vtk(f)
        rho = flds.get("density")
        if rho is not None:
            rho_min = min(rho_min, rho[rho > 0].min())
            rho_max = max(rho_max, rho.max())
        vx = flds.get("velocity_x")
        vz = flds.get("velocity_z")
        if vx is not None and vz is not None:
            spd = np.sqrt(vx**2 + vz**2)
            # Use 99th percentile to avoid outliers from atmosphere
            spd_max = max(spd_max, np.percentile(spd, 99.5))
    rho_min = max(rho_min, 1e-6)
    if spd_max < 1e-30:
        spd_max = 1.0
    print(f"  rho: [{rho_min:.2e}, {rho_max:.2e}], |v|_max: {spd_max:.2e}")

    frames = []
    for idx, fpath in enumerate(files):
        _, _, _, _, flds = parse_vtk(fpath)
        rho = np.clip(flds.get("density", np.ones((nr, nt))), rho_min, None)
        vx = flds.get("velocity_x", np.zeros((nr, nt)))
        vz = flds.get("velocity_z", np.zeros((nr, nt)))
        speed = np.sqrt(vx**2 + vz**2)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # --- Panel 1: Density (log scale, mirrored) ---
        ax = axes[0]
        pcm = ax.pcolormesh(X, Z, rho, norm=LogNorm(vmin=rho_min, vmax=rho_max),
                            cmap="inferno", shading="flat")
        ax.pcolormesh(-X, Z, rho, norm=LogNorm(vmin=rho_min, vmax=rho_max),
                      cmap="inferno", shading="flat")
        ax.set_aspect("equal")
        ax.set_title("Density (log)", fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("z")
        plt.colorbar(pcm, ax=ax, shrink=0.75, label="ρ")

        # --- Panel 2: Streamlines over density ---
        ax = axes[1]
        ax.pcolormesh(X, Z, rho, norm=LogNorm(vmin=rho_min, vmax=rho_max),
                      cmap="inferno", shading="flat", alpha=0.5)
        ax.pcolormesh(-X, Z, rho, norm=LogNorm(vmin=rho_min, vmax=rho_max),
                      cmap="inferno", shading="flat", alpha=0.5)
        # Quiver (subsample for clarity)
        s_r = max(1, nr // 20)
        s_t = max(1, nt // 15)
        qx = Xc[::s_r, ::s_t]
        qz = Zc[::s_r, ::s_t]
        qvx = vx[::s_r, ::s_t]
        qvz = vz[::s_r, ::s_t]
        q_spd = np.sqrt(qvx**2 + qvz**2)
        scale = spd_max * 15 if spd_max > 1e-20 else 1.0
        ax.quiver(qx, qz, qvx, qvz, q_spd, cmap="cool", scale=scale,
                  width=0.003, headwidth=4, alpha=0.9)
        # Mirror
        ax.quiver(-qx, qz, -qvx, qvz, q_spd, cmap="cool", scale=scale,
                  width=0.003, headwidth=4, alpha=0.9)
        ax.set_aspect("equal")
        ax.set_title("Flow field", fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("z")

        # --- Panel 3: Speed (|v|, linear scale, mirrored) ---
        ax = axes[2]
        spd_plot = speed
        pcm2 = ax.pcolormesh(X, Z, spd_plot, vmin=0, vmax=spd_max,
                             cmap="plasma", shading="flat")
        ax.pcolormesh(-X, Z, spd_plot, vmin=0, vmax=spd_max,
                      cmap="plasma", shading="flat")
        ax.set_aspect("equal")
        ax.set_title("Speed |v|", fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("z")
        plt.colorbar(pcm2, ax=ax, shrink=0.75, label="|v|")

        frame_num = int(re.search(r"(\d+)", os.path.basename(fpath)).group(1))
        fig.suptitle(f"stellar2d bubble — frame {frame_num}", fontsize=13, fontweight="bold")
        plt.tight_layout()

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), buf).convert("RGB")
        frames.append(img)
        plt.close(fig)

        if (idx + 1) % 20 == 0 or idx == n_frames - 1:
            print(f"  rendered {idx+1}/{n_frames}")

    duration = int(1000 / fps)
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=True)
    sz = os.path.getsize(output_path) / 1024
    print(f"Saved: {output_path} ({len(frames)} frames, {sz:.0f} KB)")


if __name__ == "__main__":
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/bubble_128x64_20260429_214056"
    out = sys.argv[2] if len(sys.argv) > 2 else "bubble_evolution.gif"
    skip = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    max_frames = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    make_animation(run_dir, out, fps=12, skip=skip, max_frames=max_frames)
