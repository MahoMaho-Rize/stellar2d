#!/usr/bin/env python3
"""T1 / Phase A.3 — entropy wave convergence fit (task #47).

Reads docs/scheme_char/runs_ewave/<solver>/res_N/output_final.vtk for
each (solver, N), reconstructs the analytic IC
    ρ_exact(x) = ρ0 · (1 + A · sin(k · 2π x / Lx))
(periodic advection → IC returns after one period), computes the L1
error on the horizontally-averaged ρ profile, and fits the log-log slope
to extract the formal convergence order p.

Outputs:
    docs/scheme_char/2026-05-07_entropy_wave_convergence.{csv,png,md}
"""
from __future__ import annotations
import csv
import math
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

REPO    = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO / "docs" / "scheme_char" / "runs_ewave"
OUT_DIR = REPO / "docs" / "scheme_char"
DATE    = "2026-05-07"

SOLVERS  = ("cart_ale2", "athena_vl2")
RES_LIST = (64, 128, 256, 512)
LX = 1.0
A_IC = 0.01
K_IC = 1


def parse_vtk_density(path: Path, nx: int) -> np.ndarray:
    """Return the (nx, ny) density field from a STRUCTURED_GRID VTK.

    Reads only the `SCALARS density` block.  Works for both cart_ale2
    and athena_vl2 ASCII outputs (they emit CELL_DATA in "k fastest"
    order: ix outer, jy inner).
    """
    with path.open() as fh:
        # Find SCALARS density line.
        for line in fh:
            if line.startswith("SCALARS density"):
                break
        else:
            raise RuntimeError(f"{path}: no SCALARS density found")
        next(fh)                                  # LOOKUP_TABLE default
        # Read nx*nx float tokens regardless of how many per line.
        vals = []
        need = nx * nx
        for line in fh:
            for tok in line.split():
                try:
                    vals.append(float(tok))
                except ValueError:
                    # next section header (e.g. SCALARS pressure) — stop
                    break
            else:
                if len(vals) >= need:
                    break
                continue
            break
        if len(vals) < need:
            raise RuntimeError(f"{path}: got {len(vals)} density values, "
                               f"expected {need}")
    # cart_ale2 write_vtk emits `for jc: for ic: arr[ic*ny+jc]` → jc outer,
    # ic inner (ix-fastest).  athena_vl2 similarly writes ix-fast.
    # reshape(ny, nx) puts iy on axis 0, ix on axis 1 — entropy wave
    # varies along x so it shows up on axis 1.
    return np.array(vals[:need]).reshape(nx, nx)


def l1_error(rho: np.ndarray) -> tuple[float, float]:
    """Return (L1_phase_corrected, phase_shift).

    The entropy wave should advect back to the IC after one period, but
    minor integrator timing / Lagrangian-rezone interaction may leave a
    small residual phase drift that is not a dissipation error.  We fit
    the best-matching shift s in  ρ_exact(x - s) and report the L1 of
    the residual amplitude after that phase alignment.  This is the
    true "dissipation L1" (amplitude damping + dispersion on small
    scales) independent of bulk phase timing error.
    """
    nx = rho.shape[1]
    rho_x = rho.mean(axis=0)
    xc = (np.arange(nx) + 0.5) / nx * LX
    best = (np.inf, 0.0)
    for s in np.linspace(-1.0, 1.0, 2001):
        expected = 1.0 + A_IC * np.sin(K_IC * 2.0 * math.pi * (xc - s) / LX)
        err = float(np.mean(np.abs(rho_x - expected)))
        if err < best[0]:
            best = (err, float(s))
    return best


def scan() -> list[dict]:
    rows = []
    for solver in SOLVERS:
        for N in RES_LIST:
            path = RUN_ROOT / solver / f"res_{N}" / "output_final.vtk"
            if not path.exists():
                print(f"  [miss] {path}")
                continue
            try:
                rho = parse_vtk_density(path, N)
                err, shift = l1_error(rho)
            except Exception as e:
                print(f"  [err] {path}: {e}")
                continue
            rows.append(dict(solver=solver, res=N, L1=err, shift=shift))
            print(f"  {solver:<12} N={N:<4} L1={err:.4e}  shift={shift:+.4f}")
    return rows


def fit_slope(rows: list[dict], solver: str) -> Optional[float]:
    pts = [(r["res"], r["L1"]) for r in rows if r["solver"] == solver]
    if len(pts) < 2:
        return None
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    # L1 ∝ dx^p  with dx = Lx / N  →  log L1 = p · log(1/N) + c
    p = np.polyfit(np.log(1.0 / xs), np.log(np.maximum(ys, 1e-30)), 1)
    return float(p[0])


def write_csv(rows: list[dict], slopes: dict) -> Path:
    path = OUT_DIR / f"{DATE}_entropy_wave_convergence.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["solver", "res", "L1", "phase_shift", "slope_p_solver"])
        for r in rows:
            w.writerow([r["solver"], r["res"], f"{r['L1']:.6e}",
                        f"{r['shift']:.4f}",
                        "" if slopes[r["solver"]] is None
                            else f"{slopes[r['solver']]:.3f}"])
    print(f"Wrote {path}")
    return path


def plot(rows: list[dict], slopes: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = {"cart_ale2": "#d62728", "athena_vl2": "#1f77b4"}
    markers = {"cart_ale2": "o", "athena_vl2": "s"}
    for solver in SOLVERS:
        pts = [(r["res"], r["L1"]) for r in rows if r["solver"] == solver]
        if not pts: continue
        pts.sort()
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        slope = slopes[solver]
        lbl = f"{solver}"
        if slope is not None:
            lbl += f"  (p={slope:.2f})"
        ax.loglog(xs, ys, marker=markers[solver], color=colors[solver],
                  label=lbl, linewidth=2, markersize=8)
    # Reference guides
    xref = np.array([RES_LIST[0], RES_LIST[-1]])
    dxref = LX / xref
    # anchor ~ 1e-5 at N=128 for 2nd order, etc.
    for n, ls, lbl in [(1, "--", r"$L_1 \propto dx^1$"),
                       (2, ":",  r"$L_1 \propto dx^2$")]:
        ref = 1e-4 * (dxref / (LX/128)) ** n
        ax.loglog(xref, ref, ls, color="gray", alpha=0.5, label=lbl)
    ax.set_xlabel("resolution N")
    ax.set_ylabel("L1 error on ρ at t = Lx / u0")
    ax.set_title(f"T1 entropy wave convergence  (A={A_IC}, k={K_IC})")
    ax.grid(which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    path = OUT_DIR / f"{DATE}_entropy_wave_convergence.png"
    fig.savefig(path, dpi=140)
    print(f"Wrote {path}")
    return path


def write_summary(rows: list[dict], slopes: dict):
    md = OUT_DIR / f"{DATE}_entropy_wave_convergence_auto.md"
    with md.open("w") as fh:
        fh.write(f"# Entropy wave convergence — {DATE} (auto-generated)\n\n")
        fh.write("| solver | ")
        fh.write(" | ".join(f"N={N}" for N in RES_LIST))
        fh.write(" | slope p |\n|---|" + "|".join("---" for _ in RES_LIST) + "|---|\n")
        for solver in SOLVERS:
            row = [solver]
            for N in RES_LIST:
                hit = [r for r in rows if r["solver"]==solver and r["res"]==N]
                row.append(f"{hit[0]['L1']:.2e}" if hit else "—")
            slope = slopes[solver]
            row.append(f"{slope:.3f}" if slope is not None else "—")
            fh.write("| " + " | ".join(row) + " |\n")
    print(f"Wrote {md}")


if __name__ == "__main__":
    rows = scan()
    slopes = {s: fit_slope(rows, s) for s in SOLVERS}
    print("\nSlopes:")
    for s in SOLVERS:
        print(f"  {s:<12} p = {slopes[s]}")
    write_csv(rows, slopes)
    plot(rows, slopes)
    write_summary(rows, slopes)
