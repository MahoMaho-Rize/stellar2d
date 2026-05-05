#!/usr/bin/env python3
"""Render bubble GIF: entropy perturbation + density + speed, high contrast."""

import re, os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from PIL import Image


def parse_vtk(path):
    with open(path) as f:
        text = f.read()
    m = re.search(r"DIMENSIONS\s+(\d+)\s+(\d+)", text)
    if not m:
        return None
    nt_nodes, nr_nodes = int(m.group(1)), int(m.group(2))
    nt, nr = nt_nodes - 1, nr_nodes - 1
    m = re.search(r"POINTS\s+\d+\s+\w+\n([\s\S]*?)(?=CELL_DATA)", text)
    if not m:
        return None
    pts = np.fromstring(m.group(1), sep=" ").reshape(-1, 3)
    r_face = np.sqrt(pts[::nt_nodes, 0]**2 + pts[::nt_nodes, 2]**2)
    theta_face = np.arctan2(pts[-nt_nodes:, 0], pts[-nt_nodes:, 2])
    fields = {}
    for m in re.finditer(
        r"SCALARS\s+(\w+)\s+\w+.*?\n(?:LOOKUP_TABLE.*?\n)?([\s\S]*?)(?=SCALARS|VECTORS|\Z)", text):
        vals = np.fromstring(m.group(2), sep=" ")
        if len(vals) == nr * nt:
            fields[m.group(1)] = vals.reshape(nr, nt)
    for m in re.finditer(
        r"VECTORS\s+(\w+)\s+\w+\n([\s\S]*?)(?=SCALARS|VECTORS|\Z)", text):
        vals = np.fromstring(m.group(2), sep=" ")
        if len(vals) == nr * nt * 3:
            vdata = vals.reshape(nr, nt, 3)
            fields[m.group(1) + "_x"] = vdata[:, :, 0]
            fields[m.group(1) + "_z"] = vdata[:, :, 2]
    return nr, nt, r_face, theta_face, fields


def make_animation(run_dir, output_path, fps=12, skip=1):
    files = sorted(glob.glob(os.path.join(run_dir, "output_????.vtk")))
    if not files:
        print(f"No VTK files in {run_dir}")
        return

    valid = []
    for f in files[::skip]:
        result = parse_vtk(f)
        if result is not None and "density" in result[4]:
            valid.append(f)
    print(f"Rendering {len(valid)} frames from {run_dir}")

    first = parse_vtk(valid[0])
    nr, nt, r_face, theta_face = first[0], first[1], first[2], first[3]
    R, T = np.meshgrid(r_face, theta_face, indexing="ij")
    X = R * np.sin(T)
    Z = R * np.cos(T)

    r_c = 0.5 * (r_face[:-1] + r_face[1:])
    t_c = 0.5 * (theta_face[:-1] + theta_face[1:])
    Rc, Tc = np.meshgrid(r_c, t_c, indexing="ij")
    Xc = Rc * np.sin(Tc)
    Zc = Rc * np.cos(Tc)

    gamma = 5.0 / 3.0

    # Compute HSE entropy baseline: s0(r) = P(r,j=0) / rho(r,j=0)^gamma at t=0
    flds0 = first[4]
    rho0 = flds0["density"]
    P0 = flds0["pressure"]
    s0_profile = P0[:, 0] / np.maximum(rho0[:, 0], 1e-20)**gamma  # 1D radial

    # Pre-scan for color ranges
    ds_all, spd_all = [], []
    scan = valid[2::max(1, len(valid) // 15)]
    for f in scan:
        result = parse_vtk(f)
        if result is None:
            continue
        flds = result[4]
        rho = np.maximum(flds["density"], 1e-20)
        P = np.maximum(flds["pressure"], 1e-30)
        s = P / rho**gamma
        ds = (s - s0_profile[:, None]) / np.maximum(s0_profile[:, None], 1e-20)
        ds_all.append(np.clip(ds[2:-2], -2, 2).ravel())
        vx = flds.get("velocity_x", np.zeros_like(rho))
        vz = flds.get("velocity_z", np.zeros_like(rho))
        spd_all.append(np.sqrt(vx[2:-2]**2 + vz[2:-2]**2).ravel())

    ds_cat = np.concatenate(ds_all)
    spd_cat = np.concatenate(spd_all)
    ds_lim = np.percentile(np.abs(ds_cat), 98)
    ds_lim = max(ds_lim, 0.05)
    spd_hi = np.percentile(spd_cat, 98)
    if spd_hi < 0.01:
        spd_hi = 1.0
    print(f"  ds/s0 range: +/-{ds_lim:.3f}  |v|_98%: {spd_hi:.3e}")

    frames = []
    for idx, fpath in enumerate(valid):
        result = parse_vtk(fpath)
        if result is None:
            continue
        flds = result[4]
        rho = np.maximum(flds.get("density", np.ones((nr, nt))), 1e-20)
        P = np.maximum(flds.get("pressure", np.ones((nr, nt))), 1e-30)
        vx = flds.get("velocity_x", np.zeros((nr, nt)))
        vz = flds.get("velocity_z", np.zeros((nr, nt)))
        speed = np.sqrt(vx**2 + vz**2)

        s = P / rho**gamma
        ds = (s - s0_profile[:, None]) / np.maximum(s0_profile[:, None], 1e-20)
        ds = np.clip(ds, -ds_lim, ds_lim)
        speed = np.clip(speed, 0, spd_hi)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Panel 1: Entropy perturbation (diverging colormap, high contrast)
        ax = axes[0]
        norm_ds = TwoSlopeNorm(vmin=-ds_lim, vcenter=0, vmax=ds_lim)
        pcm = ax.pcolormesh(X, Z, ds, norm=norm_ds, cmap="RdBu_r", shading="flat")
        ax.pcolormesh(-X, Z, ds, norm=norm_ds, cmap="RdBu_r", shading="flat")
        ax.set_aspect("equal")
        ax.set_title("Entropy perturbation δs/s₀", fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("z")
        plt.colorbar(pcm, ax=ax, shrink=0.75, label="δs/s₀")

        # Panel 2: Density (linear, focused range)
        ax = axes[1]
        rho_lo, rho_hi = 0.02, 1.05
        pcm2 = ax.pcolormesh(X, Z, np.clip(rho, rho_lo, rho_hi),
                             vmin=rho_lo, vmax=rho_hi,
                             cmap="magma", shading="flat")
        ax.pcolormesh(-X, Z, np.clip(rho, rho_lo, rho_hi),
                      vmin=rho_lo, vmax=rho_hi,
                      cmap="magma", shading="flat")
        ax.set_aspect("equal")
        ax.set_title("Density (linear)", fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("z")
        plt.colorbar(pcm2, ax=ax, shrink=0.75, label="ρ")

        # Panel 3: Speed with quiver overlay
        ax = axes[2]
        pcm3 = ax.pcolormesh(X, Z, speed, vmin=0, vmax=spd_hi,
                             cmap="plasma", shading="flat")
        ax.pcolormesh(-X, Z, speed, vmin=0, vmax=spd_hi,
                      cmap="plasma", shading="flat")
        s_r = max(1, nr // 20)
        s_t = max(1, nt // 15)
        qx = Xc[::s_r, ::s_t]; qz = Zc[::s_r, ::s_t]
        qvx = vx[::s_r, ::s_t]; qvz = vz[::s_r, ::s_t]
        scale = spd_hi * 15 if spd_hi > 1e-20 else 1.0
        ax.quiver(qx, qz, qvx, qvz, color="white", scale=scale,
                  width=0.002, headwidth=4, alpha=0.6)
        ax.quiver(-qx, qz, -qvx, qvz, color="white", scale=scale,
                  width=0.002, headwidth=4, alpha=0.6)
        ax.set_aspect("equal")
        ax.set_title("Speed |v| + flow", fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("z")
        plt.colorbar(pcm3, ax=ax, shrink=0.75, label="|v|")

        frame_num = int(re.search(r"(\d+)", os.path.basename(fpath)).group(1))
        fig.suptitle(f"stellar2d bubble (hybrid mass mesh) — frame {frame_num}",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), buf).convert("RGB")
        frames.append(img)
        plt.close(fig)

        if (idx + 1) % 20 == 0 or idx == len(valid) - 1:
            print(f"  rendered {idx+1}/{len(valid)}")

    duration = int(1000 / fps)
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=True)
    sz = os.path.getsize(output_path) / 1024
    print(f"Saved: {output_path} ({len(frames)} frames, {sz:.0f} KB)")


if __name__ == "__main__":
    run_dir = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "bubble_hybrid.gif"
    skip = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    make_animation(run_dir, out, fps=12, skip=skip)
