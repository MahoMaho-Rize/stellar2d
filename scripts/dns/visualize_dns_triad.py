#!/usr/bin/env python3
"""Visualize DNS Experiment E1 (three-wave triad) snapshots.

Reads <run>/snapshots/snap_NNNN.bin files (written by dns_triad_coupled with
ANSL_DNS_SNAP_EVERY=1) and produces:

  1. paper/figures/fig7_3_triad_snapshots.png  — static 2x3 grid showing
     v(x,y), u(x,y), b(x,y) at two times (IC + a later mixing snapshot).
  2. paper/figures/fig7_4_triad_kspec.png      — static 2x3 grid showing
     |v̂(k_x, y)|² at four times: IC, early (≈ period 5), mid (period 15),
     late (period 45) to visualise the k_a + k_b → k_c energy transfer
     directly in Fourier space.
  3. runs/dns_expE1_triad.gif  — animation of v(x, y) colour map over all
     sampled periods (30 fps).

Usage:
    python3 scripts/visualize_dns_triad.py \
        runs/dns_expE1_snapshots runs/dns_expE1_triad.csv
"""
import argparse
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def load_snap(path):
    with open(path, "rb") as f:
        hdr = np.fromfile(f, dtype=np.int32, count=2)
        ny, nx = int(hdr[0]), int(hdr[1])
        t = np.fromfile(f, dtype=np.float64, count=1)[0]
        n = ny * nx
        u = np.fromfile(f, dtype=np.float32, count=n).reshape(ny, nx)
        v = np.fromfile(f, dtype=np.float32, count=n).reshape(ny, nx)
        b = np.fromfile(f, dtype=np.float32, count=n).reshape(ny, nx)
    return t, u, v, b


def load_all_snaps(snap_dir):
    files = sorted(glob.glob(os.path.join(snap_dir, "snap_*.bin")))
    print(f"  {len(files)} snapshots in {snap_dir}")
    ts, us, vs, bs = [], [], [], []
    for p in files:
        t, u, v, b = load_snap(p)
        ts.append(t); us.append(u); vs.append(v); bs.append(b)
    return (np.array(ts),
            np.stack(us),   # (nt, ny, nx)
            np.stack(vs),
            np.stack(bs))


def build_y_grid(ny, Ly, coord_map, beta):
    """Reconstruct CUDA's y-grid.  coord_map='cgl' uses standard CGL;
    'tanh' mirrors ANSL_COORD_MAP=tanh ANSL_COORD_BETA=beta (default 2).
    Must match the solver's grid or imshow's y-axis labels will be wrong."""
    N = ny - 1
    xcheb = np.cos(np.pi * np.arange(N + 1) / N)
    s_asc = (1.0 + xcheb[::-1]) * Ly / 2.0
    if coord_map == "cgl":
        return s_asc
    if coord_map == "tanh":
        u = beta * (2.0 * s_asc / Ly - 1.0)
        return 0.5 * Ly + 0.5 * Ly * np.tanh(u) / np.tanh(beta)
    raise ValueError(f"unknown coord_map: {coord_map}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snap_dir", help="path to snapshots/ dir")
    ap.add_argument("csv_path",  help="path to dns_triad_coupled.csv")
    ap.add_argument("--outprefix", default="paper/figures/fig7_")
    ap.add_argument("--gifpath", default="runs/dns_expE1_triad.gif")
    ap.add_argument("--fps",     type=int, default=30)
    ap.add_argument("--coord-map", choices=["cgl", "tanh"], default="tanh",
                    help="must match CUDA's ANSL_COORD_MAP (default tanh, "
                         "consistent with dns_triad_coupled runs)")
    ap.add_argument("--coord-beta", type=float, default=2.0,
                    help="must match CUDA's ANSL_COORD_BETA")
    ap.add_argument("--Ly", type=float, default=1.0)
    args = ap.parse_args()

    ts, us, vs, bs = load_all_snaps(args.snap_dir)
    nt, ny, nx = vs.shape

    # Parse csv header for period.
    with open(args.csv_path) as f:
        f.readline()    # triad config
        hdr2 = f.readline()
    T_period = float([p for p in hdr2.split() if p.startswith("period_a=")][0].split("=")[1])
    n_periods = ts / T_period
    print(f"  domain: Ny={ny} × Nx={nx}, periods: 0 → {n_periods[-1]:.1f}")

    # Grid: match CUDA's coordinate map for correct y-axis display.
    x = np.linspace(0, 1, nx, endpoint=False)
    y = build_y_grid(ny, args.Ly, args.coord_map, args.coord_beta)
    print(f"  grid: coord_map={args.coord_map}  "
          f"(y range {y.min():.3f}..{y.max():.3f}, "
          f"min dy={np.min(np.diff(y)):.3e})")

    # ── Fig 7.3: v, u, b snapshots at t=0 and a later mixing time ──────
    # Pick mixing time as the period where E_k6 peaks in the first burst.
    data_csv = np.loadtxt(args.csv_path, comments="#")
    mask = np.isfinite(data_csv[:, 3:9]).all(axis=1)
    data_csv = data_csv[mask]
    tcsv = data_csv[:, 0]
    Ek6  = data_csv[:, 8]
    # First local max of Ek6 in periods 2..30 (skip the initial IC transients).
    idx_range = np.where((tcsv > 2 * T_period) & (tcsv < 30 * T_period))[0]
    mixing_period = int(round(tcsv[idx_range[np.argmax(Ek6[idx_range])]] / T_period))
    mixing_idx = int(min(mixing_period, nt - 1))
    print(f"  Selected mixing snapshot at period {mixing_period} (E_k6 peak)")

    fig = plt.figure(figsize=(14, 7))
    for row, (title, t_idx) in enumerate([("IC  (t = 0)", 0),
                                          (f"Mixing  (period {mixing_period})",
                                           mixing_idx)]):
        for col, (fld, name) in enumerate([(us[t_idx], "u  (x-velocity)"),
                                           (vs[t_idx], "v  (y-velocity)"),
                                           (bs[t_idx], "b  (buoyancy)")]):
            ax = fig.add_subplot(2, 3, row * 3 + col + 1)
            # Symmetric colour range scaled to this snapshot.
            mx = max(1e-40, np.abs(fld).max())
            im = ax.imshow(fld, aspect="auto", origin="lower",
                           extent=[0, 1, y.min(), y.max()],
                           cmap="RdBu_r", vmin=-mx, vmax=mx)
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(f"{title}\n{name}  (|·|≤{mx:.2e})", fontsize=10)
            if col == 0:
                ax.set_ylabel("y")
            if row == 1:
                ax.set_xlabel("x")
    plt.suptitle("DNS E1 triad: field snapshots  "
                 "(ρ-truncated Lane-Emden, 128×128)", fontsize=11)
    plt.tight_layout()
    out1 = f"{args.outprefix}3_triad_snapshots.png"
    plt.savefig(out1, dpi=140, bbox_inches="tight")
    print(f"  saved → {out1}")
    plt.close(fig)

    # ── Fig 7.4: |v̂(k_x, y)|² heatmaps showing k-space energy transfer ──
    t_picks = [0,
               max(1, nt // 12),                # early  (≈ period 5)
               min(nt - 1, mixing_idx),        # mid    (first triad peak)
               min(nt - 1, 3 * mixing_idx)]    # late   (3 peaks later)
    labels = ["t = 0", f"period {int(n_periods[t_picks[1]]):d}",
              f"period {int(n_periods[t_picks[2]]):d}",
              f"period {int(n_periods[t_picks[3]]):d}"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), sharey=True)
    vmax_global = 0.0
    vhat_list = []
    for tidx in t_picks:
        vhat = np.fft.rfft(vs[tidx], axis=1) / nx
        spec = np.abs(vhat) ** 2
        vhat_list.append(spec)
        vmax_global = max(vmax_global, spec.max())
    for ax, label, spec in zip(axes, labels, vhat_list):
        # Log colour scale clamped to [vmax*1e-10, vmax].
        sp = np.clip(spec, vmax_global * 1e-10, vmax_global)
        im = ax.imshow(sp, aspect="auto", origin="lower",
                       extent=[0, spec.shape[1] - 1, y.min(), y.max()],
                       cmap="viridis",
                       norm=matplotlib.colors.LogNorm(
                           vmin=vmax_global * 1e-10, vmax=vmax_global))
        ax.set_title(label)
        ax.set_xlabel("k_x")
        ax.set_xlim(0, 12)      # focus on first 12 Fourier modes
        # Mark the seeded (a, b) and partner (c) k's.
        for k, lab in [(1, "k_a=1"), (5, "k_b=5"), (6, "k_c=6")]:
            ax.axvline(k, color="red" if k == 6 else "white",
                       ls="--" if k == 6 else ":",
                       lw=1.0, alpha=0.7)
    axes[0].set_ylabel("y")
    plt.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02,
                 label="|v̂(k_x, y)|²")
    plt.suptitle("DNS E1: Fourier-x power spectrum of v — "
                 "energy injection into k_c = k_a + k_b = 6",
                 fontsize=11)
    out2 = f"{args.outprefix}4_triad_kspec.png"
    plt.savefig(out2, dpi=140, bbox_inches="tight")
    print(f"  saved → {out2}")
    plt.close(fig)

    # ── Animation: v(x, y) over all sampled periods ─────────────────────
    vmax_anim = np.abs(vs).max()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(vs[0], aspect="auto", origin="lower",
                   extent=[0, 1, y.min(), y.max()],
                   cmap="RdBu_r", vmin=-vmax_anim, vmax=vmax_anim)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("v(x, y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title("")

    def update(i):
        im.set_data(vs[i])
        title.set_text(f"DNS E1 triad — v(x, y)   "
                       f"period {n_periods[i]:.1f}/{n_periods[-1]:.0f}")
        return im, title

    anim = FuncAnimation(fig, update, frames=nt, interval=1000.0 / args.fps,
                         blit=False)
    writer = PillowWriter(fps=args.fps)
    anim.save(args.gifpath, writer=writer)
    print(f"  saved → {args.gifpath}  ({nt} frames at {args.fps} fps)")


if __name__ == "__main__":
    main()
