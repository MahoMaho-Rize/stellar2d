#!/usr/bin/env python3
"""
sph2d 求解器的 Mollweide 投影動畫。

輸入 VTK 是球面 STRUCTURED_GRID,N_theta × N_phi cell-centered data:
  - vorticity (ζ)
  - u_theta, u_phi, speed

為了省事我們直接讀 VTK 的 cell data,假設 cell 順序與 (j, i) = (theta_idx, phi_idx) 一致
(寫出端就是這個 layout)。

用法:
  pixi run python scripts/render_sph2d.py <run_dir> <out.mp4> [fps] [duration]
"""
import os
import sys
import re
import struct
import glob
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_vtk_binary(path):
    """讀 sph2d 寫出的 BINARY VTK STRUCTURED_GRID。返回 (dims, scalars_dict)。"""
    with open(path, "rb") as f:
        data = f.read()
    # Header(ASCII 到第一個 DIMENSIONS 和 POINTS)
    header_end = data.find(b"CELL_DATA")
    header = data[:header_end].decode("ascii", errors="ignore")
    m = re.search(r"DIMENSIONS\s+(\d+)\s+(\d+)\s+(\d+)", header)
    nnx, nny, nnz = int(m.group(1)), int(m.group(2)), int(m.group(3))
    n_theta, n_phi = nny - 1, nnx - 1
    n_cells = n_theta * n_phi

    # POINTS double → skip
    m2 = re.search(r"POINTS\s+(\d+)\s+double", header)
    n_points = int(m2.group(1))
    # 跳過 points 數據 (3 * n_points * 8B) + 結尾 '\n'
    # 計算起始 offset:header 結束位置 = POINTS 行尾
    # 實際 points data 從第一個 '\n' 後 header 裡 "POINTS n double\n" 之後開始。
    points_hdr = re.search(r"POINTS\s+\d+\s+double\n", header)
    off = points_hdr.end()
    off += 3 * n_points * 8
    # 可能有一個 '\n'
    if data[off:off+1] == b"\n": off += 1

    # 從 CELL_DATA 開始的 ASCII 頭部找 SCALARS sections
    scalars = {}
    cd_start = data.find(b"CELL_DATA", off)
    cursor = cd_start
    # CELL_DATA n\n 後可能有多個 SCALARS
    cd_hdr_end = data.find(b"\n", cursor)
    cursor = cd_hdr_end + 1

    while cursor < len(data):
        # 尋找 "SCALARS <name> double 1\nLOOKUP_TABLE default\n"
        chunk = data[cursor:cursor+200].decode("ascii", errors="ignore")
        m3 = re.match(r"SCALARS\s+(\S+)\s+double\s+1\nLOOKUP_TABLE\s+\S+\n", chunk)
        if not m3:
            break
        name = m3.group(1)
        cursor += m3.end()
        vals = np.frombuffer(data[cursor:cursor + n_cells * 8],
                             dtype=">f8").astype(np.float64)
        scalars[name] = vals.reshape(n_theta, n_phi)
        cursor += n_cells * 8
        if data[cursor:cursor+1] == b"\n": cursor += 1
    return (n_theta, n_phi), scalars


def mollweide_project(theta, phi, n_theta, n_phi):
    """從 (θ, φ) grid 得 Mollweide 座標。
    θ ∈ [0, π]: 0 = 北極, π/2 = 赤道, π = 南極
    phi ∈ [0, 2π]: 0 = meridian
    Mollweide: lat = π/2 - θ (緯度), lon = φ - π (經度 ∈ [-π, π])
    """
    lat = np.pi / 2.0 - theta  # north pole → +π/2
    lon = phi - np.pi
    return lon, lat


def render_sph2d(run_dir: str, out_mp4: str, fps: int = 30, duration: float = 10.0):
    vtk_files = sorted(glob.glob(os.path.join(run_dir, "output_*.vtk")))
    if not vtk_files:
        print(f"ERROR: no VTK in {run_dir}")
        sys.exit(1)

    print(f"Found {len(vtk_files)} VTK frames")
    # Read first to get grid
    (n_theta, n_phi), _ = read_vtk_binary(vtk_files[0])
    print(f"Grid: N_theta={n_theta} N_phi={n_phi}")

    # Gauss-Legendre θ nodes (與 solver 一致,簡化用 equispaced — 對視覺差異小)
    # 精確起見:用 numpy 的 roots_legendre
    from numpy.polynomial.legendre import leggauss
    x, _ = leggauss(n_theta)
    theta_nodes = np.arccos(-x)  # (和 solver 排序一致,θ 從小到大)
    theta_nodes = np.sort(theta_nodes)
    phi_nodes = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    lon_grid, lat_grid = np.meshgrid(phi_nodes - np.pi,
                                     np.pi/2.0 - theta_nodes)

    # 決定 colormap 範圍:掃前 N 幀取 percentile
    sample_n = min(10, len(vtk_files))
    zmax_est = 0.0
    for p in vtk_files[:sample_n]:
        _, sc = read_vtk_binary(p)
        z = sc.get("vorticity", None)
        if z is not None:
            zmax_est = max(zmax_est, np.percentile(np.abs(z), 99))
    if zmax_est <= 0:
        zmax_est = 1.0

    tmp_dir = Path(run_dir) / "render_tmp"
    tmp_dir.mkdir(exist_ok=True)

    for k, p in enumerate(vtk_files):
        _, sc = read_vtk_binary(p)
        z = sc.get("vorticity", None)
        if z is None: continue
        fig = plt.figure(figsize=(9, 5), dpi=144)
        ax = fig.add_subplot(111, projection="mollweide")
        im = ax.pcolormesh(lon_grid, lat_grid, z,
                           vmin=-zmax_est, vmax=zmax_est,
                           cmap="RdBu_r", shading="auto")
        ax.set_title(f"ω  (frame {k+1}/{len(vtk_files)})")
        ax.grid(True, alpha=0.3)
        ax.set_xticklabels([])
        plt.colorbar(im, ax=ax, shrink=0.7, label=r"$\zeta$")
        fig.tight_layout()
        fig.savefig(tmp_dir / f"f_{k:05d}.png")
        plt.close(fig)
        if (k + 1) % 20 == 0:
            print(f"  rendered {k+1}/{len(vtk_files)}")

    # ffmpeg 串成 MP4
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(tmp_dir / "f_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-crf", "18",
        "-movflags", "+faststart",
        out_mp4,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # 清理 PNG
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"✓ {out_mp4}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: render_sph2d.py <run_dir> <out.mp4> [fps] [duration]")
        sys.exit(1)
    rd = sys.argv[1]
    out = sys.argv[2]
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    dur = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
    render_sph2d(rd, out, fps, dur)
