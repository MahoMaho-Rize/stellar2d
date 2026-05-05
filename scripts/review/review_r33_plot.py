#!/usr/bin/env python3
"""Review Round-3 P3: render the complexity-accuracy figure from
runtime_scan.csv.  Output: review/r33_complexity/complexity_accuracy.png

The figure has 3 panels (shared N_y axis):
  (left)   per-step eigenmode dev vs N_y  (loglog)
  (middle) setup + per-step time vs N_y  (loglog)
  (right)  memory vs N_y (loglog)

Three methods: primitive, tau, assembled.
"""
from __future__ import annotations
import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
IN_CSV = SCRIPT_DIR.parent / "review" / "r33_complexity" / "runtime_scan.csv"
OUT_PNG = SCRIPT_DIR.parent / "review" / "r33_complexity" / "complexity_accuracy.png"


def load_rows():
    rows = []
    with open(IN_CSV) as f:
        for r in csv.DictReader(f):
            rows.append({
                "method": r["method"],
                "N_y": int(r["N_y"]),
                "setup_time": float(r["setup_time"]),
                "step_time": float(r["step_time"]),
                "memory_bytes": int(r["memory_bytes"]),
                "per_step_dev": float(r["per_step_dev"]),
            })
    return rows


def group_by_method(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["method"]].append(r)
    for k in out:
        out[k].sort(key=lambda r: r["N_y"])
    return out


def main():
    rows = load_rows()
    by_method = group_by_method(rows)

    style = {
        "primitive": dict(color="C3", marker="s", label="Primitive-node"),
        "tau":       dict(color="C1", marker="^", label=r"$\tau$-method"),
        "assembled": dict(color="C0", marker="o",
                          label="Assembled $L^{-1}R$ (this work)"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=130)

    # Panel 1: per-step dev vs N_y
    ax = axes[0]
    for method, data in by_method.items():
        Ns = [r["N_y"] for r in data]
        dev = [r["per_step_dev"] for r in data]
        # Filter infinite / nan
        Ns_f, dev_f = [], []
        for n, d in zip(Ns, dev):
            if np.isfinite(d) and d > 0:
                Ns_f.append(n); dev_f.append(d)
        ax.loglog(Ns_f, dev_f, lw=1.5, ms=8, **style[method])
    ax.axhline(1e-4, ls=":", color="gray", lw=1, label="§5.1 floor")
    ax.axhline(5e-16, ls="--", color="gray", lw=1,
               label="machine precision")
    ax.set_xlabel(r"$N_y$")
    ax.set_ylabel("per-step eigenmode deviation")
    ax.set_title("(a) Accuracy")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="best")

    # Panel 2: step_time + setup_time vs N_y
    ax = axes[1]
    for method, data in by_method.items():
        Ns = [r["N_y"] for r in data]
        setup = [r["setup_time"] * 1e3 for r in data]  # ms
        step = [r["step_time"] * 1e6 for r in data]    # μs
        s = style[method]
        ax.loglog(Ns, step, "-", lw=1.5, ms=7,
                  color=s["color"], marker=s["marker"],
                  label=s["label"] + " step")
        if method != "primitive":  # primitive has 0 setup
            ax.loglog(Ns, setup, "--", lw=1.0, ms=5,
                      color=s["color"], marker=s["marker"],
                      alpha=0.6, label=s["label"] + " setup (ms)")
    ax.set_xlabel(r"$N_y$")
    ax.set_ylabel(r"time (solid: μs/step, dashed: ms setup)")
    ax.set_title("(b) Runtime")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="best")

    # Panel 3: memory vs N_y
    ax = axes[2]
    for method, data in by_method.items():
        Ns = [r["N_y"] for r in data]
        mem_kb = [r["memory_bytes"] / 1024 for r in data]
        ax.loglog(Ns, mem_kb, lw=1.5, ms=8, **style[method])
    # Reference N² and N scaling lines
    ax.loglog([32, 256], [9 * 64, 9 * 4096], ls=":", color="black",
              alpha=0.3, lw=1, label=r"$\propto N_y^2$")
    ax.set_xlabel(r"$N_y$")
    ax.set_ylabel("working memory (kB)")
    ax.set_title("(c) Memory footprint")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Three-method comparison: accuracy / runtime / memory "
        "— Lane--Emden $n=3/2$, $\\ell=1$ g-mode",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
