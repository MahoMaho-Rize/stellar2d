#!/usr/bin/env python3
"""Plot DNS Experiment A (3-wave triad) GPU results.
Output: paper/figures/fig7_1_triad.png

Shows modal energy E_k1..4 and total energy drift across an amplitude
scan on Lane-Emden n=3/2 + GPU Strang-split RK4.  Amp list configurable
via the AMPS environment variable (space-separated), defaulting to the
three-amp "clean weakly-nonlinear" panel used in the paper.

  AMPS="1e-6 1e-5 1e-4 1e-3 3e-3 1e-2" python3 scripts/plot_dns_triad.py

Reads runs/dns_expA/triad_amp{amp}.csv produced by scripts/run_dns_expA_scan.sh
or a manual `--solver anelastic_sl --test dns_triad` invocation.

LONGTIME=1 switches source to runs/dns_expA_longtime/, adds a linear
extrapolation of the 100-T baseline dev(t) to the dev panel so visible
deviation from linearity = secular drift.  Output changes to
paper/figures/fig7_1_triad_longtime.png.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

amps = os.environ.get("AMPS", "1e-6 1e-5 1e-4").split()
longtime = os.environ.get("LONGTIME", "") not in ("", "0", "false")
src_dir = "runs/dns_expA_longtime" if longtime else "runs/dns_expA"
baseline_dir = "runs/dns_expA" if longtime else None
palette = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(amps)))
colors = {a: palette[i] for i, a in enumerate(amps)}

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
(ax_ek, ax_dev, ax_et, ax_sp) = axes.flat

summary_rows = []

for amp in amps:
    path = f"{src_dir}/triad_amp{amp}.csv"
    if not os.path.exists(path):
        print(f"  ! missing {path} — skipping")
        continue
    data = np.loadtxt(path, comments="#")
    # columns: t, v_center, eigmode_dev, E_total, E_k1, E_k2, E_k3, E_k4, max_abs_v
    m = np.isfinite(data[:, 2:8]).all(axis=1)
    n_blow = len(data) - int(m.sum())
    if n_blow > 0:
        print(f"  amp={amp}: dropped {n_blow} non-finite rows (blowup)")
    data = data[m]
    if data.shape[0] < 2:
        print(f"  ! amp={amp}: not enough finite rows after filter")
        continue
    t = data[:, 0]
    dev = data[:, 2]
    Et = data[:, 3]
    Ek = data[:, 4:8]   # k=1..4

    with open(path) as f:
        hdr = f.readline()
    T_period = float([p for p in hdr.split() if p.startswith("period=")][0].split("=")[1])
    n_periods = t / T_period

    blow_tag = " ⚠" if n_blow > 0 else ""
    lbl = f"amp={amp}{blow_tag}"
    E_k1_0 = Ek[0, 0]
    ax_ek.plot(n_periods, Ek[:, 0] / E_k1_0, label=f"k=1 ({lbl})",
               color=colors[amp], lw=1.5)
    ax_ek.plot(n_periods, Ek[:, 1] / E_k1_0,
               color=colors[amp], lw=1.0, ls="--")

    ax_dev.plot(n_periods, dev, label=lbl, color=colors[amp])
    # Linear-in-t extrapolation from the first 20% of the long-time run;
    # if the actual curve follows this line to end, drift is bounded
    # (non-secular).  Divergence above = secular accumulation.
    if longtime and n_periods[-1] > 50:
        fit_end = int(0.2 * len(n_periods))
        if fit_end >= 3:
            slope = (dev[fit_end - 1] - dev[0]) / max(
                n_periods[fit_end - 1] - n_periods[0], 1e-12)
            extrap = dev[0] + slope * (n_periods - n_periods[0])
            ax_dev.plot(n_periods, extrap, color=colors[amp],
                        lw=0.8, ls=":", alpha=0.7)

    if Et[0] > 0:
        ax_et.plot(n_periods, np.abs(Et - Et[0]) / Et[0] + 1e-16,
                   label=lbl, color=colors[amp])

    ax_sp.semilogy(np.arange(1, 5), Ek[-1, :] / E_k1_0 + 1e-40,
                   "o-", label=lbl, color=colors[amp])

    row = {
        "amp": amp,
        "last_period": n_periods[-1],
        "blowup": n_blow > 0,
        "dev_per_period": dev[-1] / max(n_periods[-1], 1e-12),
        "E_k1_drift": (Ek[-1, 0] - Ek[0, 0]) / Ek[0, 0],
        "E_k2_ratio_end": Ek[-1, 1] / E_k1_0,
        "E_k2_ratio_max": Ek[:, 1].max() / E_k1_0,
    }
    # When in LONGTIME mode, compare against 100-T_a baseline: if the
    # drift is truly ∝ t, the 500-T rate should equal the 100-T rate.
    if baseline_dir is not None:
        bpath = f"{baseline_dir}/triad_amp{amp}.csv"
        if os.path.exists(bpath):
            bdata = np.loadtxt(bpath, comments="#")
            bm = np.isfinite(bdata[:, 2:8]).all(axis=1)
            bdata = bdata[bm]
            if bdata.shape[0] >= 2:
                with open(bpath) as f:
                    bhdr = f.readline()
                bT = float([p for p in bhdr.split()
                            if p.startswith("period=")][0].split("=")[1])
                bnp = bdata[:, 0] / bT
                bdev_rate = bdata[-1, 2] / max(bnp[-1], 1e-12)
                bE_rate = ((bdata[-1, 4] - bdata[0, 4]) / bdata[0, 4]
                           / max(bnp[-1], 1e-12))
                row["baseline_dev_rate"] = bdev_rate
                row["baseline_E_rate"] = bE_rate
    summary_rows.append(row)

ax_ek.set_xlabel("periods of g-mode n_g=1")
ax_ek.set_ylabel("E_k / E_k1(0)")
ax_ek.set_yscale("log")
ax_ek.set_title("Modal energy: k=1 (solid) + k=2 (dashed)")
ax_ek.legend(loc="best", fontsize=8, ncol=2)
ax_ek.grid(True, alpha=0.3)

ax_dev.set_xlabel("periods")
ax_dev.set_ylabel("eigmode deviation")
ax_dev.set_title("Primary-mode deviation vs time")
ax_dev.set_yscale("log")
ax_dev.legend(fontsize=8)
ax_dev.grid(True, alpha=0.3)

ax_et.set_xlabel("periods")
ax_et.set_ylabel("|ΔE / E(0)|")
ax_et.set_yscale("log")
ax_et.set_title("Total-energy drift (anelastic functional)")
ax_et.legend(fontsize=8)
ax_et.grid(True, alpha=0.3)

ax_sp.set_xlabel("horizontal mode k")
ax_sp.set_ylabel("E_k / E_k1(0) at last valid sample")
ax_sp.set_yscale("log")
ax_sp.set_title("Modal spectrum at end of run")
ax_sp.set_xticks([1, 2, 3, 4])
ax_sp.legend(fontsize=8)
ax_sp.grid(True, alpha=0.3)

plt.suptitle("DNS Experiment A: 3-wave triad, Lane-Emden n=3/2, 64×64, GPU Strang-split RK4",
             fontsize=11)
plt.tight_layout()
out = ("paper/figures/fig7_1_triad_longtime.png" if longtime
       else "paper/figures/fig7_1_triad.png")
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved → {out}")

print("\nSummary:")
print(f"  {'amp':>6}  {'last T':>8}  {'blow':>5}  {'dev/T':>11}  "
      f"{'E_k1 drift':>11}  {'E_k2/E_k1(0) end':>17}  {'E_k2/E_k1(0) max':>17}")
for r in summary_rows:
    print(f"  {r['amp']:>6}  {r['last_period']:>8.1f}  "
          f"{'yes' if r['blowup'] else 'no':>5}  "
          f"{r['dev_per_period']:>11.3e}  "
          f"{r['E_k1_drift']:>+11.3e}  "
          f"{r['E_k2_ratio_end']:>17.3e}  "
          f"{r['E_k2_ratio_max']:>17.3e}")

if baseline_dir is not None and any("baseline_dev_rate" in r for r in summary_rows):
    print("\nSecular-drift check (long-time rate vs 100-T baseline rate):")
    print(f"  {'amp':>6}  {'dev/T (long)':>13}  {'dev/T (100T)':>13}  "
          f"{'ratio':>7}  {'ΔE_k1/T (long)':>15}  {'ΔE_k1/T (100T)':>15}  {'ratio':>7}")
    for r in summary_rows:
        if "baseline_dev_rate" not in r:
            continue
        long_E_rate = r["E_k1_drift"] / max(r["last_period"], 1e-12)
        dev_ratio = r["dev_per_period"] / max(r["baseline_dev_rate"], 1e-30)
        E_ratio = long_E_rate / r["baseline_E_rate"] if r["baseline_E_rate"] != 0 else float("nan")
        print(f"  {r['amp']:>6}  {r['dev_per_period']:>13.3e}  "
              f"{r['baseline_dev_rate']:>13.3e}  {dev_ratio:>7.2f}  "
              f"{long_E_rate:>+15.3e}  {r['baseline_E_rate']:>+15.3e}  "
              f"{E_ratio:>7.2f}")
    print("  (ratio ≈ 1 → bounded linear drift; ratio > 1 → secular)")
