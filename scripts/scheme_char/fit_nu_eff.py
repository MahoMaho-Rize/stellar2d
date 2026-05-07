#!/usr/bin/env python3
"""T3 (Phase A.1) ν_eff fit + comparison figure.

Reads diagnostics.csv files produced by scripts/scheme_char/run_nu_eff_scan.sh
for three solvers (cart_ale2, athena_vl2, pseudo_spectral) and fits the
exponential decay rate of the shear mode / Taylor-Green pattern to extract
an effective kinematic viscosity.

Output:
    docs/scheme_char/2026-05-07_nu_eff_comparison.csv
    docs/scheme_char/2026-05-07_nu_eff_comparison.png
    docs/scheme_char/2026-05-07_nu_eff_comparison.md    (summary)
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO / "docs" / "scheme_char" / "runs"
OUT_DIR = REPO / "docs" / "scheme_char"
DATE = "2026-05-07"

LY = 1.0
K_MODE = 1
KPHYS = K_MODE * 2.0 * math.pi / LY
V0_LIST = (0.01, 0.1)
RES_LIST = (64, 128, 256, 512)
SOLVERS = ("cart_ale2", "athena_vl2", "pseudo_spectral")


@dataclass
class FitResult:
    solver: str
    res: int
    V0: float
    lam: float            # decay rate −d ln(max|v or ω|)/dt
    nu_eff: float         # derived effective viscosity
    n_fit_points: int


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (t, signal) where signal is max|v| (hydro) or max|ω| (ps)."""
    with path.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"empty diagnostics.csv: {path}")
    t = np.array([float(r["t"]) for r in rows])
    # Hydro schemes: max_v.  Spectral: max_omega (proxy for mode amplitude).
    if "max_omega" in rows[0]:
        signal = np.array([float(r["max_omega"]) for r in rows])
    elif "max_v" in rows[0]:
        signal = np.array([float(r["max_v"]) for r in rows])
    else:
        raise RuntimeError(f"csv missing max_v / max_omega: {path}")
    return t, signal


def fit_decay(t: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Fit y ≈ y0 · exp(−λ t).  Uses the window t ∈ [5 % · t_end, 50 % · t_end]
    to avoid IC transients and late-time floors.  Returns (lam, n_pts).
    """
    if len(t) < 4:
        return float("nan"), 0
    t_end = t[-1]
    mask = (t >= 0.05 * t_end) & (t <= 0.5 * t_end) & (y > 0)
    if mask.sum() < 4:
        mask = (y > 0)
    tt = t[mask]
    yy = np.log(y[mask])
    # np.polyfit(deg=1): slope is d(ln y)/dt; λ = −slope.
    p = np.polyfit(tt, yy, 1)
    return -float(p[0]), int(mask.sum())


def scan() -> list[FitResult]:
    out: list[FitResult] = []
    # For ps Taylor-Green, analytic decay is y ∝ exp(-2·ν·k_phys²·t):
    #   ω(x,y,0)=2k·cos(kx)·cos(ky) → |ω|(t) decays at rate 2νk²
    # For shear_mode (hydro), v=V₀·exp(-ν k²t) → rate νk²
    for solver in SOLVERS:
        denom = 2.0 * KPHYS**2 if solver == "pseudo_spectral" else KPHYS**2
        for V0 in V0_LIST:
            for N in RES_LIST:
                path = RUN_ROOT / solver / f"res_{N}_V_{V0}" / "diagnostics.csv"
                if not path.exists():
                    print(f"  [miss] {path}")
                    continue
                try:
                    t, y = load_csv(path)
                except Exception as e:
                    print(f"  [err]  {path}: {e}")
                    continue
                lam, n_pts = fit_decay(t, y)
                nu_eff = lam / denom if math.isfinite(lam) else float("nan")
                out.append(FitResult(solver, N, V0, lam, nu_eff, n_pts))
                print(f"  {solver:<18} res={N:<4} V0={V0:<6}"
                      f" λ={lam:.3e}  ν_eff={nu_eff:.3e}  (n={n_pts})")
    return out


def write_csv(results: list[FitResult]) -> Path:
    path = OUT_DIR / f"{DATE}_nu_eff_comparison.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["solver", "res", "V0", "lambda", "nu_eff", "n_fit_points"])
        for r in results:
            w.writerow([r.solver, r.res, r.V0,
                        f"{r.lam:.6e}", f"{r.nu_eff:.6e}", r.n_fit_points])
    print(f"\nWrote {path}")
    return path


def plot(results: list[FitResult]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    markers = {"cart_ale2": "o", "athena_vl2": "s", "pseudo_spectral": "^"}
    colors  = {"cart_ale2": "#d62728", "athena_vl2": "#2ca02c",
               "pseudo_spectral": "#1f77b4"}

    for ax, V0 in zip(axes, V0_LIST):
        for solver in SOLVERS:
            xs, ys = [], []
            for r in results:
                if r.solver == solver and r.V0 == V0 and math.isfinite(r.nu_eff):
                    xs.append(r.res)
                    ys.append(r.nu_eff)
            if not xs:
                continue
            order = np.argsort(xs)
            xs = np.array(xs)[order]
            ys = np.array(ys)[order]
            ax.loglog(xs, ys, marker=markers[solver], color=colors[solver],
                      label=solver, linewidth=2, markersize=8)
            # Fit slope in log-log (ν_eff vs 1/res).
            if len(xs) >= 2:
                dx = np.array([1.0 / x for x in xs])
                p = np.polyfit(np.log(dx), np.log(np.maximum(ys, 1e-20)), 1)
                slope = p[0]
                ax.text(xs[-1] * 1.02, ys[-1],
                        f"  slope={slope:.2f}",
                        color=colors[solver], fontsize=8, va="center")
        # Reference ν ~ dx¹ and dx² guides.
        xref = np.array([RES_LIST[0], RES_LIST[-1]])
        dxref = 1.0 / xref
        for n, ls, lbl in [(1, "--", r"$\nu \propto dx^1$ (ILES)"),
                           (2, ":",  r"$\nu \propto dx^2$ (Godunov-ILES)")]:
            # Anchor at (res=128, midway between ps and ale2)
            ref_nu = 1e-3 * dxref**n / (1/128)**n
            ax.loglog(xref, ref_nu, ls, color="gray", alpha=0.5, label=lbl)

        ax.set_xlabel("resolution (N)")
        ax.set_title(f"V₀ = {V0}")
        ax.grid(which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel(r"$\nu_{\rm eff}$")
    fig.suptitle(
        f"T3 ν_eff comparison — shear-mode / Taylor-Green decay "
        f"(k={K_MODE}, Ly={LY})",
        fontsize=11,
    )
    fig.tight_layout()
    path = OUT_DIR / f"{DATE}_nu_eff_comparison.png"
    fig.savefig(path, dpi=140)
    print(f"Wrote {path}")
    return path


def write_summary(results: list[FitResult], csv_path: Path, png_path: Path):
    md = OUT_DIR / f"{DATE}_nu_eff_comparison.md"
    with md.open("w") as fh:
        fh.write(f"# ν_eff comparison — {DATE}\n\n")
        fh.write("T3 Linear shear-mode / Taylor-Green pure-diffusion decay:\n\n")
        fh.write(f"- cart_ale2 / athena_vl2:  `vx = V₀·sin(k·2π y/Ly)`, "
                 f"analytic decay `max|v| ∝ exp(-ν·k²_phys·t)`\n")
        fh.write(f"- pseudo_spectral:  Taylor-Green `ω = 2k·cos·cos`, "
                 f"analytic `max|ω| ∝ exp(-2ν·k²_phys·t)`\n\n")
        fh.write(f"ν_eff extracted by log-slope fit on "
                 f"t ∈ [5 %, 50 %] of t_end.\n\n")
        fh.write("## Summary table (V₀=0.01)\n\n")
        fh.write("| solver | res=64 | 128 | 256 | 512 | slope p (ν∝dx^p) |\n")
        fh.write("|---|---|---|---|---|---|\n")
        for solver in SOLVERS:
            row = [solver]
            nus = {}
            for N in RES_LIST:
                hit = [r for r in results
                       if r.solver == solver and r.res == N and r.V0 == 0.01]
                if hit and math.isfinite(hit[0].nu_eff):
                    nus[N] = hit[0].nu_eff
                    row.append(f"{hit[0].nu_eff:.2e}")
                else:
                    row.append("—")
            # Compute slope ν ∝ dx^p  ⇒ log ν = p·log(1/N)+c
            if len(nus) >= 2:
                xs = np.array(list(nus.keys()))
                ys = np.array(list(nus.values()))
                p = np.polyfit(np.log(1.0 / xs), np.log(np.maximum(ys, 1e-20)), 1)
                row.append(f"{p[0]:.2f}")
            else:
                row.append("—")
            fh.write("| " + " | ".join(row) + " |\n")
        fh.write("\n")
        fh.write(f"- CSV: `{csv_path.name}`\n")
        fh.write(f"- Plot: `{png_path.name}`\n")
        fh.write("\n")
        fh.write("## Reading\n\n")
        fh.write("- slope p ≈ 2 → Godunov-like 2nd-order dissipation "
                 "(ν_eff ≈ V·dx² prefactor varies)\n")
        fh.write("- slope p ≈ 1 → Jensen-ILES 1st-order "
                 "(ν_eff ≈ V·dx — 'over-damped' at coarse grid)\n")
        fh.write("- slope p ≫ 2 / negligible ν_eff → spectral / DNS-quality\n")
    print(f"Wrote {md}")


if __name__ == "__main__":
    print(f"Scanning {RUN_ROOT} ...")
    results = scan()
    if not results:
        print("No diagnostics.csv found — run scripts/scheme_char/run_nu_eff_scan.sh first.")
        raise SystemExit(1)
    csv_path = write_csv(results)
    png_path = plot(results)
    write_summary(results, csv_path, png_path)
