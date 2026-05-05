#!/usr/bin/env python3
"""Phase 3 Path 1/2/3 head-to-head comparison.

Runs all three paths over a matrix of scenarios:
    bg          ∈ {boussinesq, lane_emden}
    amp         ∈ {1e-8 (linear), 1e-3 (mild nonlin), 1e-1 (strong nonlin)}
    dt_scan     to find maximum stable dt per path (bisection up to crash)

Outputs:
  - Summary table to stdout
  - Combined PNG with:
    (a) dev(t) per path, linear amp (left panel, log scale)
    (b) energy drift per path, linear amp
    (c) dev/step vs dt for Path 2/3 (stability)
    (d) v_center trajectory sample for each path, nonlin amp

Saves per-run .npz under /tmp/nonlinear_bench_2026-05-03/.
"""
from __future__ import annotations
import os
import pathlib
import subprocess
import sys

import numpy as np


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
OUT_DIR = pathlib.Path("/tmp/nonlinear_bench_2026-05-03")


def run_path(script, tag, bg, amp, dt, n_steps, ny=48, nx=32, extra=None):
    """Invoke one of the Path N scripts; return parsed .npz or None on failure."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suf = f"_{extra}" if extra else ""
    out_npz = OUT_DIR / f"{tag}_{bg}_amp{amp:.0e}_dt{dt:.0e}{suf}.npz"
    cmd = [
        sys.executable, str(SCRIPT_DIR / script),
        "--bg", bg, "--ny", str(ny), "--nx", str(nx),
        "--amp", str(amp), "--dt", str(dt),
        "--n_steps", str(n_steps),
        "--save", str(out_npz),
    ]
    if extra == "no_nonlin":
        cmd.append("--no_nonlin")
    print(f"  → {tag}{suf}  bg={bg}  amp={amp}  dt={dt}  n_steps={n_steps}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("     CRASHED:")
        print(r.stdout[-400:])
        print(r.stderr[-400:])
        return None
    if not out_npz.exists():
        print("     no output file")
        return None
    return np.load(out_npz, allow_pickle=True)


def main():
    # Matrix: 3 paths × 2 backgrounds × 3 amplitudes.  Path 3 also includes
    # a linear-only run per scenario as reference.
    paths = [
        ("nonlinear_path1_opsplit.py",  "path1"),
        ("nonlinear_path2_imex.py",     "path2"),
        ("nonlinear_path3_expint.py",   "path3"),
    ]
    bgs = ["boussinesq", "lane_emden"]
    amp_scan = [1e-8, 1e-3, 1e-1]

    ny = 48; nx = 64
    # dt sized to give 4 periods (ω≈1.64, T≈3.82 → 4T≈15, /800 steps ≈ 0.019).
    dt_default = 2e-2
    n_steps_default = 800  # ≈ 4 periods

    summary = []
    for path_script, tag in paths:
        for bg in bgs:
            for amp in amp_scan:
                data = run_path(path_script, tag, bg, amp,
                                dt_default, n_steps_default, ny=ny, nx=nx)
                if data is None:
                    summary.append((tag, bg, amp, None))
                    continue
                summary.append((tag, bg, amp, dict(
                    dev_rate=float(data["dev_rate"]),
                    dev_final=float(data["dev"][-1]),
                    dE_rel=float(data["dE_rel"]),
                    rel_freq_err=float(data["rel_freq_err"]),
                    omega_evp=float(data["omega_evp"]),
                )))

    # Path 3 linear-only floor (reference)
    print()
    print("─" * 60)
    print(" Path 3 LINEAR-ONLY floor (reference for machine precision)")
    print("─" * 60)
    for bg in bgs:
        data = run_path("nonlinear_path3_expint.py", "path3",
                        bg, 1e-8, dt_default, n_steps_default,
                        ny=ny, nx=nx, extra="no_nonlin")
        if data is None:
            summary.append(("path3_lin", bg, 1e-8, None)); continue
        summary.append(("path3_lin", bg, 1e-8, dict(
            dev_rate=float(data["dev_rate"]),
            dev_final=float(data["dev"][-1]),
            dE_rel=float(data["dE_rel"]),
            rel_freq_err=float(data["rel_freq_err"]),
            omega_evp=float(data["omega_evp"]),
        )))

    # Print summary table
    print()
    print("═" * 108)
    print(f"{'path':<8} {'bg':<12} {'amp':>8}   "
          f"{'dev/step':>12} {'dev_final':>12} "
          f"{'ΔE/E':>12} {'rel_freq_err':>14}")
    print("─" * 108)
    for row in summary:
        if row[3] is None:
            print(f"{row[0]:<8} {row[1]:<12} {row[2]:>8.0e}   CRASH")
            continue
        d = row[3]
        print(f"{row[0]:<8} {row[1]:<12} {row[2]:>8.0e}   "
              f"{d['dev_rate']:>12.3e} {d['dev_final']:>12.3e} "
              f"{d['dE_rel']:>+12.3e} {d['rel_freq_err']:>+14.3e}")
    print("═" * 108)

    # dt stability scan: find ω·dt at which each path starts to degrade.
    # ω_lane ≈ 1.64, so scan dt up to large multiples of 1/ω.
    print()
    print("═" * 60)
    print(" dt stability scan  (Lane-Emden, amp=1e-8, 4-period run)")
    print("═" * 60)
    dts = [3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
    for tag, script in [("path1", "nonlinear_path1_opsplit.py"),
                        ("path2", "nonlinear_path2_imex.py"),
                        ("path3", "nonlinear_path3_expint.py")]:
        print(f"\n  {tag}:")
        for dt in dts:
            t_total = 15.0  # ~4 periods
            n_steps = max(20, int(t_total / dt))
            data = run_path(script, f"{tag}_stab", "lane_emden", 1e-8, dt, n_steps,
                            ny=32, nx=16)
            if data is None:
                print(f"    dt={dt:.0e}  (ω·dt={1.64*dt:.3f}):  CRASH")
                continue
            dev_f = float(data["dev"][-1])
            print(f"    dt={dt:.0e}  (ω·dt={1.64*dt:.3f}):  "
                  f"dev_final={dev_f:.3e}  ({n_steps} steps)")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=140)

        dt_tag = f"{dt_default:.0e}"

        def load(tag, bg, amp, extra=""):
            suf = f"_{extra}" if extra else ""
            fn = OUT_DIR / f"{tag}_{bg}_amp{amp:.0e}_dt{dt_tag}{suf}.npz"
            return np.load(fn) if fn.exists() else None

        # (a) Linear regime dev on Lane-Emden (+ linear floor reference)
        ax = axes[0, 0]
        for tag in ["path1", "path2", "path3"]:
            d = load(tag, "lane_emden", 1e-8)
            if d is not None:
                ax.semilogy(d["t"], np.maximum(d["dev"], 1e-20),
                            label=tag)
        d_ref = load("path3", "lane_emden", 1e-8, "no_nonlin")
        if d_ref is not None:
            ax.semilogy(d_ref["t"], np.maximum(d_ref["dev"], 1e-20),
                        "k--", label="path3 linear-only", lw=1)
        ax.set_xlabel("t"); ax.set_ylabel("eigmode deviation")
        ax.set_title("Linear regime dev(t)  (Lane-Emden, amp=1e-8)")
        ax.legend(); ax.grid(True, alpha=0.3)

        # (b) Energy drift on Lane-Emden, nonlin amp=1e-3
        ax = axes[0, 1]
        for tag in ["path1", "path2", "path3"]:
            d = load(tag, "lane_emden", 1e-3)
            if d is not None:
                E = d["energy"]
                ax.plot(d["t"], (E - E[0]) / max(abs(E[0]), 1e-300), label=tag)
        ax.set_xlabel("t"); ax.set_ylabel("ΔE / E(0)")
        ax.set_title("Energy drift  (Lane-Emden, amp=1e-3)")
        ax.legend(); ax.grid(True, alpha=0.3)

        # (c) v_center trajectory, amp=1e-3 (visible oscillations)
        ax = axes[1, 0]
        for tag in ["path1", "path2", "path3"]:
            d = load(tag, "lane_emden", 1e-3)
            if d is not None:
                ax.plot(d["t"], d["v_center"], label=tag, alpha=0.7)
        ax.set_xlabel("t"); ax.set_ylabel("v(Lx/4, Ly/2)")
        ax.set_title("v_center trajectory  (Lane-Emden, amp=1e-3)")
        ax.legend(); ax.grid(True, alpha=0.3)

        # (d) Strong nonlin regime (amp=0.1)
        ax = axes[1, 1]
        for tag in ["path1", "path2", "path3"]:
            d = load(tag, "lane_emden", 1e-1)
            if d is not None:
                E = d["energy"]
                ax.semilogy(
                    d["t"],
                    np.maximum(np.abs(E - E[0]) / max(abs(E[0]), 1e-300), 1e-20),
                    label=tag)
        ax.set_xlabel("t"); ax.set_ylabel("|ΔE/E₀|  (log)")
        ax.set_title("Strong nonlinear regime  (Lane-Emden, amp=1e-1)")
        ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_png = SCRIPT_DIR.parent / "videos" / "nonlinear_paths_compare.png"
        out_png.parent.mkdir(exist_ok=True)
        fig.savefig(out_png)
        print(f"\n  → {out_png}")
    except ImportError:
        print("  matplotlib not available, skipping PNG")


if __name__ == "__main__":
    main()
