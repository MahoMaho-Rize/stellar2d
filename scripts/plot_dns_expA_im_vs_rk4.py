#!/usr/bin/env python3
"""IM vs RK4 side-by-side for DNS Experiment A (amp=1e-6 × 500 T_a).

Illustrates the textbook trade-off:
  - IM is exactly symplectic → H_IM drifts only at round-off (~1e-11)
    but has 2nd-order phase error that accumulates as stroboscopic
    amplitude decay in the anelastic KE diagnostic.
  - RK4 is non-symplectic → H_IM leaks at ~8e-7/T (consistent with
    the "amp-independent Strang floor" observed in commit d21409a)
    but has 4th-order phase error so stroboscopic amplitude stays clean.

Inputs (produced by an earlier session):
  runs/dns_expA_im/triad_amp1e-6_im.csv
  runs/dns_expA_im/triad_amp1e-6_rk4.csv
Output: paper/figures/fig7_1_triad_im_vs_rk4.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

paths = {"IM":  "runs/dns_expA_im/triad_amp1e-6_im.csv",
         "RK4": "runs/dns_expA_im/triad_amp1e-6_rk4.csv"}
colors = {"IM": "#d62728", "RK4": "#1f77b4"}

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
(ax_him, ax_ephys, ax_dev, ax_vc) = axes.flat

for tag, path in paths.items():
    if not os.path.exists(path):
        print(f"  ! missing {path}")
        continue
    data = np.loadtxt(path, comments="#")
    with open(path) as f:
        hdr = f.readline()
    T = float([p for p in hdr.split() if p.startswith("period=")][0].split("=")[1])
    np_ = data[:, 0] / T
    # cols: t, v_c, dev, E_tot, E_k1..4, max_v, H_im
    v_c = data[:, 1]
    dev = data[:, 2]
    Ephys = data[:, 3]
    Him = data[:, 9]

    eps = 1e-16
    ax_him.semilogy(np_, np.abs(Him - Him[0]) / Him[0] + eps,
                    label=tag, color=colors[tag], lw=1.3)
    ax_ephys.semilogy(np_, np.abs(Ephys - Ephys[0]) / Ephys[0] + eps,
                     label=tag, color=colors[tag], lw=1.3)
    ax_dev.semilogy(np_, dev + eps, label=tag, color=colors[tag], lw=1.3)
    ax_vc.plot(np_, v_c / v_c[0], label=tag, color=colors[tag], lw=1.2)

ax_him.set_title("|ΔH_IM / H_IM(0)|  —  symplectic invariant")
ax_him.set_xlabel("periods of g-mode n_g=1")
ax_him.set_ylabel("|ΔH_IM / H_IM(0)|")
ax_him.grid(True, which="both", alpha=0.3)
ax_him.legend()

ax_ephys.set_title("|ΔE_phys / E_phys(0)|  —  anelastic kinetic+potential")
ax_ephys.set_xlabel("periods of g-mode n_g=1")
ax_ephys.set_ylabel("|ΔE_phys / E_phys(0)|")
ax_ephys.grid(True, which="both", alpha=0.3)
ax_ephys.legend()

ax_dev.set_title("Eigenmode shape deviation (dev)")
ax_dev.set_xlabel("periods")
ax_dev.set_ylabel("dev")
ax_dev.grid(True, which="both", alpha=0.3)
ax_dev.legend()

ax_vc.set_title("v_center / v_center(0)  —  stroboscopic sample at t=NT")
ax_vc.set_xlabel("periods")
ax_vc.set_ylabel("v_c(NT) / v_c(0)")
ax_vc.axhline(1.0, color="grey", ls=":", lw=0.8)
ax_vc.grid(True, alpha=0.3)
ax_vc.legend()

plt.suptitle("DNS Experiment A: implicit midpoint vs RK4 — amp=1e-6, 500 T_a,\n"
             "IM preserves H_IM to round-off; RK4 preserves phase to round-off",
             fontsize=11)
plt.tight_layout()
out = "paper/figures/fig7_1_triad_im_vs_rk4.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved → {out}")

print("\nAt t = 500 T_a:")
print(f"  {'scheme':>6}  {'dev':>10}  {'|ΔE_phys|':>11}  {'|ΔH_IM|':>11}")
for tag, path in paths.items():
    data = np.loadtxt(path, comments="#")
    dev = data[-1, 2]
    E = abs(data[-1, 3] - data[0, 3]) / data[0, 3]
    H = abs(data[-1, 9] - data[0, 9]) / data[0, 9]
    print(f"  {tag:>6}  {dev:>10.3e}  {E:>11.3e}  {H:>11.3e}")
