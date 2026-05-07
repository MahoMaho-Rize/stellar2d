#!/usr/bin/env python3
"""T4 / Phase B (task #48) — Jensen probe analysis.

Fits ν_eff(k) at fixed resolution N=256 across k = 1..64.  Reports
whether ν_eff is k-independent (true Laplacian viscosity), rises with
k (Jensen-ILES over-damping at grid scale), or falls with k
(hyperviscous / spectral dealias).
"""
from __future__ import annotations
import csv
import math
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO / "docs" / "scheme_char" / "runs_jensen"
OUT_DIR = REPO / "docs" / "scheme_char"
DATE = "2026-05-07"

SOLVERS = ("cart_ale2", "athena_vl2")
K_LIST = (1, 2, 4, 8, 16, 32, 64)
LY = 1.0
N = 256


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    t = np.array([float(r["t"]) for r in rows])
    v = np.array([float(r["max_v"]) for r in rows])
    return t, v


def fit_decay(t: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Exponential slope on [5 %, 50 %] of t_end."""
    if len(t) < 4:
        return float("nan"), 0
    t_end = t[-1]
    mask = (t >= 0.05 * t_end) & (t <= 0.5 * t_end) & (y > 0)
    if mask.sum() < 4:
        mask = (y > 0)
    tt = t[mask]
    yy = np.log(y[mask])
    p = np.polyfit(tt, yy, 1)
    return -float(p[0]), int(mask.sum())


def scan() -> list[dict]:
    rows = []
    for solver in SOLVERS:
        for k in K_LIST:
            path = RUN_ROOT / solver / f"k_{k}" / "diagnostics.csv"
            if not path.exists():
                print(f"  [miss] {path}")
                continue
            try:
                t, v = load_csv(path)
                lam, n = fit_decay(t, v)
                kphys = k * 2.0 * math.pi / LY
                nu_eff = lam / (kphys * kphys) if math.isfinite(lam) else float("nan")
            except Exception as e:
                print(f"  [err] {path}: {e}")
                continue
            rows.append(dict(solver=solver, k=k, kphys=kphys, lam=lam,
                             nu_eff=nu_eff, n=n))
            print(f"  {solver:<12} k={k:<3} λ={lam:+.3e}  ν_eff={nu_eff:.3e}")
    return rows


def write_csv(rows: list[dict]) -> Path:
    path = OUT_DIR / f"{DATE}_jensen_probe.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["solver", "k_mode", "k_phys", "lambda", "nu_eff", "n_fit_points"])
        for r in rows:
            w.writerow([r["solver"], r["k"], f"{r['kphys']:.6e}",
                        f"{r['lam']:.6e}", f"{r['nu_eff']:.6e}", r["n"]])
    print(f"Wrote {path}")
    return path


def plot(rows: list[dict]) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"cart_ale2": "#d62728", "athena_vl2": "#1f77b4"}
    markers = {"cart_ale2": "o", "athena_vl2": "s"}

    # Panel 1: ν_eff vs k
    for solver in SOLVERS:
        xs, ys = [], []
        for r in rows:
            if r["solver"] == solver and math.isfinite(r["nu_eff"]) and r["nu_eff"] > 0:
                xs.append(r["k"]); ys.append(r["nu_eff"])
        if xs:
            ax1.loglog(xs, ys, marker=markers[solver], color=colors[solver],
                       label=solver, linewidth=2, markersize=8)
    ax1.axhline(7.76e-3, ls="--", color="gray", alpha=0.5,
                label="cart_ale2 T3 ν_eff @ k=1 (T3, N=128)")
    ax1.set_xlabel("k (mode number)")
    ax1.set_ylabel(r"$\nu_{\rm eff}$")
    ax1.set_title(f"ν_eff(k) at N={N}, V₀=0.01")
    ax1.grid(which="both", alpha=0.3)
    ax1.legend(fontsize=8, loc="best")

    # Panel 2: λ = ν·k² vs k
    #   Physical Laplacian: λ ∝ k²
    #   Jensen-ILES: λ ∝ k² at low k, but scheme saturates at grid scale
    for solver in SOLVERS:
        xs, ys = [], []
        for r in rows:
            if r["solver"] == solver and math.isfinite(r["lam"]) and r["lam"] > 0:
                xs.append(r["k"]); ys.append(r["lam"])
        if xs:
            ax2.loglog(xs, ys, marker=markers[solver], color=colors[solver],
                       label=solver, linewidth=2, markersize=8)
    xref = np.array([1, K_LIST[-1]])
    ax2.loglog(xref, 1e-2 * xref**2, ls=":", color="gray", alpha=0.5,
               label=r"$\lambda \propto k^2$ (Laplacian)")
    ax2.set_xlabel("k")
    ax2.set_ylabel(r"$\lambda$ (decay rate)")
    ax2.set_title("decay rate vs mode number")
    ax2.grid(which="both", alpha=0.3)
    ax2.legend(fontsize=8, loc="best")

    fig.suptitle("T4 Jensen probe — ν_eff as a function of mode k", fontsize=11)
    fig.tight_layout()
    path = OUT_DIR / f"{DATE}_jensen_probe.png"
    fig.savefig(path, dpi=140)
    print(f"Wrote {path}")
    return path


if __name__ == "__main__":
    rows = scan()
    if not rows:
        raise SystemExit("no data — run run_jensen_probe_scan.sh first")
    write_csv(rows)
    plot(rows)
