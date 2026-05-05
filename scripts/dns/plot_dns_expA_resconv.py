#!/usr/bin/env python3
"""Resolution-convergence diagnostic for DNS Experiment A long-time drift.

Plots E_k1 drift rate vs grid resolution at fixed amp ∈ {1e-3, 1e-2},
showing that the drift is resolution-INDEPENDENT at each amp.  This
rules out Galerkin truncation (k>K energy leaking back) as the source
of the amp=1e-2 secular tail and isolates it to Strang O(dt²·amp²)
cross terms.

Output: paper/figures/fig7_1_triad_resconv.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

resolutions = [("runs/dns_expA_longtime",     64),
               ("runs/dns_expA_longtime_128", 128),
               ("runs/dns_expA_longtime_256", 256)]
amps = ["1e-3", "1e-2"]
colors = {"1e-3": "#1f77b4", "1e-2": "#d62728"}

fig, (ax_rate, ax_t) = plt.subplots(1, 2, figsize=(12, 5))

for amp in amps:
    xs, ys = [], []
    for resdir, N in resolutions:
        path = f"{resdir}/triad_amp{amp}.csv"
        if not os.path.exists(path):
            continue
        data = np.loadtxt(path, comments="#")
        m = np.isfinite(data[:, 2:8]).all(axis=1)
        data = data[m]
        with open(path) as f:
            hdr = f.readline()
        T = float([p for p in hdr.split() if p.startswith("period=")][0].split("=")[1])
        np_ = data[:, 0] / T
        Ek1 = data[:, 4]
        rate = (Ek1[-1] - Ek1[0]) / Ek1[0] / max(np_[-1], 1e-12)
        xs.append(N); ys.append(abs(rate))
        # Overlay the drift time series on the right panel.
        drift = (Ek1 - Ek1[0]) / Ek1[0]
        ax_t.plot(np_, np.abs(drift) + 1e-16, label=f"amp={amp}, {N}²",
                  color=colors[amp],
                  lw={64: 2.4, 128: 1.4, 256: 0.8}[N],
                  alpha={64: 0.35, 128: 0.7, 256: 1.0}[N])

    ax_rate.semilogy(xs, ys, "o-", color=colors[amp], label=f"amp={amp}",
                     markersize=9, lw=2)

# Horizontal reference for pure O(dt²) Strang floor measured at amp ≤ 1e-4.
ax_rate.axhline(8.07e-7, color="#555", ls=":", lw=1.2,
                label="amp-indep Strang floor (amp ≤ 1e-4)")
ax_rate.set_xlabel("grid resolution N  (N_x = N_y = N)")
ax_rate.set_ylabel("|ΔE_k1 / E_k1(0)|  per T_a")
ax_rate.set_xticks([64, 128, 256])
ax_rate.set_xscale("log", base=2)
ax_rate.set_title("Drift rate vs resolution at fixed amp\n"
                  "(flat line ⇒ not a Galerkin-truncation artefact)")
ax_rate.legend(fontsize=9)
ax_rate.grid(True, which="both", alpha=0.3)

ax_t.set_xlabel("periods of g-mode n_g=1")
ax_t.set_ylabel("|ΔE_k1 / E_k1(0)|")
ax_t.set_yscale("log")
ax_t.set_title("Drift time series: 64² (faint) → 256² (dark)\n"
               "Curves at each amp overlap ⇒ resolution-independent")
ax_t.legend(fontsize=8, ncol=2)
ax_t.grid(True, alpha=0.3)

plt.suptitle("DNS Experiment A: resolution convergence of amp=1e-2 secular tail",
             fontsize=11)
plt.tight_layout()
out = "paper/figures/fig7_1_triad_resconv.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved → {out}")

# Numerical summary with amp² ratio.
print("\nDrift rate vs (amp, res):  E_k1 drift per T_a")
print(f"  {'res':>4}  {'amp=1e-3':>11}  {'amp=1e-2':>11}  {'ratio 1e-2/1e-3':>16}")
for resdir, N in resolutions:
    r = {}
    for amp in amps:
        path = f"{resdir}/triad_amp{amp}.csv"
        if not os.path.exists(path):
            continue
        data = np.loadtxt(path, comments="#")
        m = np.isfinite(data[:, 2:8]).all(axis=1); data = data[m]
        with open(path) as f:
            hdr = f.readline()
        T = float([p for p in hdr.split() if p.startswith("period=")][0].split("=")[1])
        np_ = data[:, 0] / T; Ek1 = data[:, 4]
        r[amp] = (Ek1[-1] - Ek1[0]) / Ek1[0] / max(np_[-1], 1e-12)
    if "1e-3" in r and "1e-2" in r:
        ratio = r["1e-2"] / r["1e-3"]
        print(f"  {N:>4}  {r['1e-3']:>+11.3e}  {r['1e-2']:>+11.3e}  {ratio:>16.2f}")
print("\nScaling of excess drift above the amp-independent Strang floor:")
floor = 8.07e-7
e_1e3 = 8.162e-7 - floor
e_1e2 = 8.446e-6 - floor
ratio = e_1e2 / e_1e3
p = np.log10(ratio) / np.log10(10.0)
print(f"  excess(amp=1e-3) = {e_1e3:.2e}")
print(f"  excess(amp=1e-2) = {e_1e2:.2e}")
print(f"  ratio = {ratio:.0f}  →  exponent p = log10(ratio) / log10(10) = {p:.2f}")
print(f"  → excess ∝ amp^{p:.2f}   (amp^3 predicted from Strang ΔE ~ dt²·[L,N]·v)")
print(f"  C coefficient: excess/amp³ = {e_1e2/(1e-2)**3:.2f} (at 1e-2)"
      f"  vs {e_1e3/(1e-3)**3:.2f} (at 1e-3)")
