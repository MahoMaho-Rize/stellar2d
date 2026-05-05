#!/usr/bin/env python3
"""Plot DNS Experiment E1 (three-wave triad) — CSV-based overview.

CSV columns (from --test dns_triad_coupled):
    t  max_v  E_kin_total  E_k1..E_k6  E_pot

This is the quick-look plot from raw CSV diagnostics (per-k Fourier-band
energy, NOT eigenmode projection).  For clean triad energies on the
eigenmode basis, use scripts/project_eigenmodes_e1.py (which consumes the
snapshot cubes with TANH-aware EVP projection).

Config (amp, mode indices, omega, period) is parsed from the CSV header
so the figure always matches the data — no hard-coded amp literals.

Panels:
  (1) Modal E_k time series for k=1..6 (log y)
  (2) b↔c partner sum with predicted beat period markers
  (3) Non-resonant modes — selectivity
  (4) E_c vs periods (linear y, no slope fit — signal sits near round-off
      in both raw CSV and TANH-projection; log-log slope on noise is not
      meaningful.  Use project_eigenmodes_e1.py for clean growth curves.)
"""
import argparse
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HDR_LINE1_RE = re.compile(
    r"#\s*triad:\s*a=\(n(?P<na>\d+),kx(?P<ka>\d+),amp(?P<ampa>[^\)]+)\)"
    r"\s+b=\(n(?P<nb>\d+),kx(?P<kb>\d+),amp(?P<ampb>[^\)]+)\)"
    r"\s+c=\(\?,kx(?P<kc>\d+)\)")


def parse_header(path):
    """Parse first two header lines for triad config + period + dt + spp.
    Returns a dict; raises if malformed."""
    with open(path) as f:
        line1 = f.readline()
        line2 = f.readline()
    m1 = HDR_LINE1_RE.search(line1)
    if not m1:
        raise ValueError(f"Unrecognised E1 CSV header line 1:\n  {line1.rstrip()}")
    cfg = {
        "n_a": int(m1.group("na")), "k_a": int(m1.group("ka")),
        "n_b": int(m1.group("nb")), "k_b": int(m1.group("kb")),
        "k_c": int(m1.group("kc")),
        "amp_a": float(m1.group("ampa")),
        "amp_b": float(m1.group("ampb")),
    }
    for tag in ("omega_a", "period_a", "dt", "spp"):
        mm = re.search(rf"{tag}=(\S+)", line2)
        if not mm:
            raise ValueError(f"Missing {tag} in header line 2:\n  {line2.rstrip()}")
        cfg[tag] = float(mm.group(1)) if tag != "spp" else int(float(mm.group(1)))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", nargs="?", default="runs/dns_expE1_triad.csv")
    ap.add_argument("--outfig", default="paper/figures/fig7_2_triad_coupled.png")
    args = ap.parse_args()

    cfg = parse_header(args.csv_path)
    T_period = cfg["period_a"]

    data = np.loadtxt(args.csv_path, comments="#")
    mask = np.isfinite(data[:, 2:9]).all(axis=1)
    if mask.sum() < len(data):
        print(f"WARNING: {len(data) - mask.sum()} non-finite rows dropped "
              f"(blowup at period ≈ {data[mask][-1, 0] / T_period:.1f})")
    data = data[mask]

    t     = data[:, 0]
    max_v = data[:, 1]
    E_kin = data[:, 2]
    Ek    = data[:, 3:9]
    E_pot = data[:, 9] if data.shape[1] > 9 else np.zeros_like(t)
    n_per = t / T_period

    # Report.
    print(f"CSV: {args.csv_path}")
    print(f"Config (from header): a=(n{cfg['n_a']}, k={cfg['k_a']}, amp={cfg['amp_a']:g})  "
          f"b=(n{cfg['n_b']}, k={cfg['k_b']}, amp={cfg['amp_b']:g})  "
          f"→ k_c={cfg['k_c']}")
    print(f"ω_a={cfg['omega_a']:.6f}, period={T_period:.4f}, dt={cfg['dt']:.4f}, "
          f"{cfg['spp']} spp")
    print(f"Samples: {len(t)}, up to period {n_per[-1]:.1f}")
    print(f"E_k1 drift (mode a): {(Ek[-1,0]-Ek[0,0])/Ek[0,0]*100:+.4f}%")
    print(f"E_k{cfg['k_b']} drift (mode b, kx={cfg['k_b']}): "
          f"{(Ek[-1, cfg['k_b']-1]-Ek[0, cfg['k_b']-1])/Ek[0, cfg['k_b']-1]*100:+.3f}%")
    print(f"E_kin drift: {(E_kin[-1]-E_kin[0])/E_kin[0]*100:+.4f}%")
    print(f"E_pot max: {E_pot.max():.3e}  (=0 expected when b frozen in linear block)")

    kb = cfg["k_b"]  # seeded-b column index (1-based = kb, 0-based = kb-1)
    kc = cfg["k_c"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── (1) All six modal energies ─────────────────────────────────────
    ax = axes[0, 0]
    colors = {"a": "#1f77b4", "b": "#d62728", "c": "#2ca02c",
              "q": "#bbbbbb"}
    for k in range(1, 7):
        if k == cfg["k_a"]:
            col, lbl, lw = colors["a"], f"$E_{{k={k}}}$ (pump $a$)", 1.8
        elif k == kb:
            col, lbl, lw = colors["b"], f"$E_{{k={k}}}$ (seeded $b$)", 1.8
        elif k == kc:
            col, lbl, lw = colors["c"], f"$E_{{k={k}}}$ (partner $c$)", 2.0
        else:
            col, lbl, lw = colors["q"], f"$E_{{k={k}}}$ (quiet)", 0.7
        ax.semilogy(n_per, np.clip(Ek[:, k - 1], 1e-50, None),
                    "-", color=col, lw=lw, label=lbl)
    ax.set_xlabel("periods of mode $a$")
    ax.set_ylabel("modal kinetic energy $E_k$")
    ax.set_title("Three-wave triad modal energies (raw CSV Fourier-band)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── (2) b ↔ c partner energy ──────────────────────────────────────
    ax = axes[0, 1]
    E_b = Ek[:, kb - 1]
    E_c = Ek[:, kc - 1]
    ax.plot(n_per, E_b, "-", color=colors["b"], lw=1.5,
            label=f"$E_b$ $(k_x={kb})$")
    ax.plot(n_per, E_c, "-", color=colors["c"], lw=1.5,
            label=f"$E_c$ $(k_x={kc})$")
    # Do NOT plot E_b + E_c on log axis — when E_c ≪ E_b the sum
    # visually coincides with E_b and is a trivial "conservation".
    ax.axhline(E_b[0], color=colors["b"], ls=":", lw=0.6, alpha=0.5,
               label=f"$E_b(0) = {E_b[0]:.2e}$")
    ax.set_yscale("log")
    ax.set_xlabel("periods of mode $a$")
    ax.set_ylabel("energy")
    ax.set_title("Partner-mode energies (b, c) — "
                 "see project_eigenmodes_e1.py for clean Manley–Rowe")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── (3) Non-resonant modes ─────────────────────────────────────────
    ax = axes[1, 0]
    q_modes = [k for k in range(1, 7)
               if k not in (cfg["k_a"], kb, kc)]
    cmap = plt.get_cmap("tab10")
    for i, k in enumerate(q_modes):
        ax.semilogy(n_per, np.clip(Ek[:, k - 1], 1e-50, None),
                    "-", color=cmap(i), lw=1.0, label=f"$k={k}$")
    ax.set_xlabel("periods of mode $a$")
    ax.set_ylabel("$E_k$")
    ax.set_title("Non-resonant modes (selectivity)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── (4) E_c vs period — no slope fit ────────────────────────────────
    # On raw Fourier bands E_c is dominated by (b·b) + (a·b) products,
    # which are NOT the triad partner.  Plot it linearly + mark round-off
    # floor; any "growth exponent" on log-log axis would be fitting
    # contamination.  For the real growth rate, rely on EVP-projected
    # data (project_eigenmodes_e1.py).
    ax = axes[1, 1]
    ax.plot(n_per, E_c, "o-", color=colors["c"], ms=3, lw=0.8,
            label=f"$E_{{k={kc}}}$ (raw band)")
    # IC value = round-off floor.
    ax.axhline(E_c[0], color="k", ls=":", lw=0.8,
               label=f"IC level = {E_c[0]:.2e}")
    # amp² scale.
    amp_scale = cfg["amp_a"] * cfg["amp_b"]
    ax.axhline(amp_scale ** 2, color="gray", ls="--", lw=0.8,
               label=fr"amp_a·amp_b$^2$ = {amp_scale**2:.1e}")
    ax.set_xlabel("periods of mode $a$")
    ax.set_ylabel(f"$E_{{k={kc}}}$")
    ax.set_yscale("log")
    ax.set_title("Partner-band energy  (use EVP projection for growth rate)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    plt.suptitle(
        f"DNS E1 raw overview:  "
        f"$a=(n_g{{=}}{cfg['n_a']}, k_x{{=}}{cfg['k_a']})$ + "
        f"$b=(n_g{{=}}{cfg['n_b']}, k_x{{=}}{cfg['k_b']})$ $\\to$ "
        f"$k_c{{=}}{kc}$.  "
        f"amp_a={cfg['amp_a']:g}, amp_b={cfg['amp_b']:g}, "
        f"Lane-Emden $n{{=}}3/2$, 128×128, GPU Strang-split",
        fontsize=10)
    plt.tight_layout()
    plt.savefig(args.outfig, dpi=140, bbox_inches="tight")
    print(f"\nsaved → {args.outfig}")


if __name__ == "__main__":
    main()
