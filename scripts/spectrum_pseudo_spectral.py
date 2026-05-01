#!/usr/bin/env python3
"""2D 動能譜 E(k) 分析 — pseudo_spectral 輸出。

用法:
  scripts/spectrum_pseudo_spectral.py <run_dir>                    # 掃 8 幀疊圖
  scripts/spectrum_pseudo_spectral.py <run_dir>/output_NNNN.vtk   # 單幀
  scripts/spectrum_pseudo_spectral.py <run_dir> out.png [n_snap]  # 自訂輸出/取樣數

Parseval 正規化後總和 Σ E(k)·dk = ∫½(u²+v²) dA = KE_total。
疊上 Kraichnan 2D 湍流參考斜率:
  k^{-5/3}  逆級串 (能量向大尺度上傳, k < k_inj)
  k^{-3}    正向 enstrophy 級串 (渦量向小尺度下傳, k > k_inj)
"""

import numpy as np
import os, sys, glob, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_pseudo_spectral import parse_vtk_cells, parse_vtk_dims


def compute_spectrum(u, v, Lx=1.0, Ly=1.0):
    """回傳 (k_centers, E_k, KE_total_check)。

    E_k 為譜密度:Σ E_k · dk ≈ KE_total。
    k_centers 為整數 bin · dk (dk = 2π / max(Lx, Ly))。
    """
    ny, nx = u.shape
    dx, dy = Lx / nx, Ly / ny
    dA = dx * dy
    N = nx * ny

    U = np.fft.fft2(u)
    V = np.fft.fft2(v)
    E_mode = (dA / N) * 0.5 * (np.abs(U) ** 2 + np.abs(V) ** 2)

    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)

    dk = 2 * np.pi / max(Lx, Ly)
    kmax_bin = min(nx, ny) // 2
    bin_idx = np.clip(np.round(K / dk).astype(int), 0, kmax_bin)
    E_k_raw = np.bincount(bin_idx.ravel(),
                          weights=E_mode.ravel(),
                          minlength=kmax_bin + 1)[:kmax_bin + 1]
    k_centers = np.arange(kmax_bin + 1) * dk
    E_k = E_k_raw / dk
    KE_check = E_k_raw.sum()
    return k_centers, E_k, KE_check


def load_velocity(path, nx, ny):
    fields = parse_vtk_cells(path, nx, ny, {"velocity_x", "velocity_y"})
    if fields is None or "velocity_x" not in fields:
        raise RuntimeError(f"缺 velocity 欄位: {path}")
    return fields["velocity_x"], fields["velocity_y"]


def load_frame_times(run_dir):
    csv = os.path.join(run_dir, "frames.csv")
    if not os.path.exists(csv):
        return {}
    out = {}
    with open(csv) as fh:
        next(fh)
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                out[int(parts[0])] = float(parts[2])
    return out


def frame_number(path):
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def detect_kinj_from_diagnostics(run_dir):
    """若 run_dir 下有 argv-like 記錄 (stdout log) 就抓 ps-k;這裡先回 None。"""
    return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    arg = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    n_snap = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    if arg.endswith(".vtk"):
        files = [arg]
        run_dir = os.path.dirname(arg) or "."
    else:
        run_dir = arg.rstrip("/")
        all_files = sorted(glob.glob(os.path.join(run_dir, "output_????.vtk")))
        # 跳過 polar dummy (output_0000) 與小 ASCII 首幀,只取 binary
        def is_binary(p):
            with open(p, "rb") as fh:
                return b"BINARY\n" in fh.read(256)
        all_files = [f for f in all_files
                     if os.path.getsize(f) > 1024 and is_binary(f)]
        if not all_files:
            print(f"無 binary VTK 於 {run_dir}")
            sys.exit(1)
        if len(all_files) > n_snap:
            idx = np.linspace(0, len(all_files) - 1, n_snap).astype(int)
            files = [all_files[i] for i in idx]
        else:
            files = all_files

    if out_path is None:
        out_path = os.path.join(run_dir, "spectrum.png")

    nx, ny = parse_vtk_dims(files[0])
    print(f"Grid: {nx}×{ny}, processing {len(files)} frames")

    ftimes = load_frame_times(run_dir)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
    cmap = plt.get_cmap("viridis")

    curves = []
    for i, path in enumerate(files):
        u, v = load_velocity(path, nx, ny)
        k, Ek, KE = compute_spectrum(u, v)
        m = (k > 0) & (Ek > 1e-30)
        fnum = frame_number(path)
        t = ftimes.get(fnum)
        lbl = f"t={t:6.3f}" if t is not None else f"frame {fnum:04d}"
        color = cmap(i / max(1, len(files) - 1))
        ax.loglog(k[m], Ek[m], "-", color=color, label=lbl, lw=1.4)
        curves.append((t if t is not None else i, k, Ek, KE))
        print(f"  {os.path.basename(path):20s} {lbl}  KE={KE:.4e}")

    # 參考斜率 (Kraichnan 1967 2D 湍流)
    k_ref_lo = np.array([2.0, 30.0])
    k_ref_hi = np.array([10.0, 300.0])
    # 錨點:以中間值壓到畫面中央
    # 取所有 curves 的中間 k 處的 median E 當錨
    mid_k = 8.0
    Emids = []
    for (_, k, Ek, _) in curves:
        idx_near = np.argmin(np.abs(k - mid_k))
        if Ek[idx_near] > 0:
            Emids.append(Ek[idx_near])
    E_anchor = np.median(Emids) if Emids else 1e-3
    ax.loglog(k_ref_lo,
              E_anchor * (k_ref_lo / mid_k) ** (-5.0 / 3.0),
              "k--", alpha=0.55, lw=1.2, label=r"$k^{-5/3}$ (inverse cascade)")
    ax.loglog(k_ref_hi,
              E_anchor * 0.1 * (k_ref_hi / mid_k) ** (-3.0),
              "k:", alpha=0.55, lw=1.2, label=r"$k^{-3}$ (enstrophy cascade)")

    # 2/3 dealias cutoff 垂直線
    k_max_dealias = (nx / 3) * (2 * np.pi)  # 假 Lx=1
    ax.axvline(k_max_dealias, color="red", alpha=0.3, lw=1, ls="--")
    ax.text(k_max_dealias, ax.get_ylim()[1] * 0.3,
            " 2/3 dealias", color="red", alpha=0.6, fontsize=9,
            rotation=90, va="top", ha="left")

    ax.set_xlabel(r"wavenumber $k$", fontsize=12)
    ax.set_ylabel(r"$E(k)$", fontsize=12)
    ax.set_title(f"Energy spectrum — {os.path.basename(run_dir)}  ({nx}×{ny})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9, loc="lower left", ncol=2, framealpha=0.85)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
