#!/usr/bin/env python3
"""Render axisymmetric data interpolated onto Cartesian grid (x-z plane)."""

import re, os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import RegularGridInterpolator
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


def interp_to_cartesian(field, r_c, theta_c, x_grid, z_grid, R_max):
    """Interpolate (r, theta) data onto (x, z) Cartesian grid."""
    r_pts = np.sqrt(x_grid**2 + z_grid**2)
    theta_pts = np.arctan2(np.abs(x_grid), z_grid)  # 0..pi
    theta_pts = np.clip(theta_pts, theta_c[0], theta_c[-1])

    interp = RegularGridInterpolator(
        (r_c, theta_c), field,
        method="linear", bounds_error=False, fill_value=np.nan)
    pts = np.stack([r_pts.ravel(), theta_pts.ravel()], axis=-1)
    result = interp(pts).reshape(x_grid.shape)
    result[r_pts > R_max] = np.nan
    return result


def make_animation(run_dir, output_path, fps=8, skip=1):
    files = sorted(glob.glob(os.path.join(run_dir, "output_????.vtk")))
    valid = []
    for f in files[::skip]:
        result = parse_vtk(f)
        if result is not None and "density" in result[4]:
            valid.append(f)
    if not valid:
        print("No valid VTK files"); return
    print(f"Rendering {len(valid)} frames from {run_dir}")

    first = parse_vtk(valid[0])
    nr, nt, r_face, theta_face = first[0], first[1], first[2], first[3]
    r_c = 0.5 * (r_face[:-1] + r_face[1:])
    theta_c = 0.5 * (theta_face[:-1] + theta_face[1:])
    R_max = r_face[-1]
    gamma = 5.0 / 3.0

    # HSE baseline entropy
    flds0 = first[4]
    rho0 = flds0["density"]
    P0 = flds0["pressure"]
    s0_profile = P0[:, 0] / np.maximum(rho0[:, 0], 1e-20)**gamma

    # Cartesian grid
    N_cart = 512
    x1d = np.linspace(-R_max, R_max, N_cart)
    z1d = np.linspace(-R_max, R_max, N_cart)
    Xg, Zg = np.meshgrid(x1d, z1d, indexing="ij")

    # Pre-scan color ranges
    ds_all, spd_all = [], []
    for f in valid[1::max(1, len(valid)//10)]:
        result = parse_vtk(f)
        if result is None: continue
        flds = result[4]
        rho = np.maximum(flds["density"], 1e-20)
        P = np.maximum(flds["pressure"], 1e-30)
        s = P / rho**gamma
        ds = (s - s0_profile[:, None]) / np.maximum(s0_profile[:, None], 1e-20)
        ds_all.append(ds[4:].ravel())
        vx = flds.get("velocity_x", np.zeros_like(rho))
        vz = flds.get("velocity_z", np.zeros_like(rho))
        spd_all.append(np.sqrt(vx[4:]**2 + vz[4:]**2).ravel())

    ds_cat = np.concatenate(ds_all)
    spd_cat = np.concatenate(spd_all)
    ds_lim = np.percentile(np.abs(ds_cat), 98)
    ds_lim = max(ds_lim, 0.01)
    spd_hi = np.percentile(spd_cat, 98)
    if spd_hi < 0.001: spd_hi = 0.1
    print(f"  ds/s0: +/-{ds_lim:.4f}  |v|_98%: {spd_hi:.3e}")

    frames = []
    for idx, fpath in enumerate(valid):
        result = parse_vtk(fpath)
        if result is None: continue
        flds = result[4]
        rho = np.maximum(flds.get("density", np.ones((nr, nt))), 1e-20)
        P = np.maximum(flds.get("pressure", np.ones((nr, nt))), 1e-30)
        vx = flds.get("velocity_x", np.zeros((nr, nt)))
        vz = flds.get("velocity_z", np.zeros((nr, nt)))
        speed = np.sqrt(vx**2 + vz**2)

        s = P / rho**gamma
        ds = (s - s0_profile[:, None]) / np.maximum(s0_profile[:, None], 1e-20)

        # Interpolate to Cartesian
        ds_cart = interp_to_cartesian(ds, r_c, theta_c, Xg, Zg, R_max)
        rho_cart = interp_to_cartesian(rho, r_c, theta_c, Xg, Zg, R_max)
        spd_cart = interp_to_cartesian(speed, r_c, theta_c, Xg, Zg, R_max)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Panel 1: Entropy perturbation
        ax = axes[0]
        norm_ds = TwoSlopeNorm(vmin=-ds_lim, vcenter=0, vmax=ds_lim)
        pcm = ax.pcolormesh(x1d, z1d, ds_cart.T, norm=norm_ds,
                            cmap="RdBu_r", shading="auto")
        ax.set_aspect("equal"); ax.set_title("Entropy δs/s₀", fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("z")
        plt.colorbar(pcm, ax=ax, shrink=0.75)

        # Panel 2: Density
        ax = axes[1]
        rho_lo, rho_hi = 0.01, np.nanpercentile(rho_cart, 99)
        pcm2 = ax.pcolormesh(x1d, z1d, np.clip(rho_cart.T, rho_lo, rho_hi),
                             vmin=rho_lo, vmax=rho_hi,
                             cmap="magma", shading="auto")
        ax.set_aspect("equal"); ax.set_title("Density ρ", fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("z")
        plt.colorbar(pcm2, ax=ax, shrink=0.75)

        # Panel 3: Speed
        ax = axes[2]
        pcm3 = ax.pcolormesh(x1d, z1d, np.clip(spd_cart.T, 0, spd_hi),
                             vmin=0, vmax=spd_hi,
                             cmap="plasma", shading="auto")
        ax.set_aspect("equal"); ax.set_title("Speed |v|", fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("z")
        plt.colorbar(pcm3, ax=ax, shrink=0.75)

        frame_num = int(re.search(r"(\d+)", os.path.basename(fpath)).group(1))
        fig.suptitle(f"Lane-Emden perturbed (Cartesian view) — frame {frame_num}",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), buf).convert("RGB")
        frames.append(img)
        plt.close(fig)

        if (idx + 1) % 10 == 0 or idx == len(valid) - 1:
            print(f"  rendered {idx+1}/{len(valid)}")

    duration = int(1000 / fps)
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=True)
    sz = os.path.getsize(output_path) / 1024
    print(f"Saved: {output_path} ({len(frames)} frames, {sz:.0f} KB)")


if __name__ == "__main__":
    run_dir = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "evolution_cartesian.gif"
    skip = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    make_animation(run_dir, out, fps=8, skip=skip)
