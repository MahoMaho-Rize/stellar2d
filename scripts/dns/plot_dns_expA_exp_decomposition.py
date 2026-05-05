#!/usr/bin/env python3
"""Test 2: decompose Strang drift into linear-block + commutator parts.

Compares Strang(RK4 linear + NL_RK4) against Strang(Exp linear + NL_RK4) at
amp ∈ {1e-6, 1e-4, 1e-3, 1e-2} × 500 T_a.  Exp-propagator has zero phase
and zero amplitude error on the linear block; any residual drift in
Strang-Exp must come from the Strang commutator [L, N] at the block
interface — scaling as O(dt²·amp³).

Confirms the Phase 3 §4.1 conjecture: amp=1e-2 secular tail is dominated
by the commutator, not the linear-block floor.

Output: paper/figures/fig7_1_triad_exp_decomposition.png
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

amps = ["1e-6", "1e-4", "1e-3", "1e-2"]
colors = {"1e-6": "#440154", "1e-4": "#3b528b",
          "1e-3": "#21918c", "1e-2": "#fde725"}
rk4_dir = "runs/dns_expA_longtime"
exp_dir = "runs/dns_expA_exp"

def load(path):
    data = np.loadtxt(path, comments="#")
    with open(path) as f: hdr = f.readline()
    T = float([p for p in hdr.split() if p.startswith("period=")][0].split("=")[1])
    m = np.isfinite(data[:, 2:8]).all(axis=1)
    data = data[m]
    return data[:, 0] / T, data

fig, (ax_rate, ax_ts) = plt.subplots(1, 2, figsize=(13, 5))

# Left: log-log drift-rate vs amplitude comparison.
rk4_rates = []
exp_rates = []
for amp in amps:
    rk4_path = f"{rk4_dir}/triad_amp{amp}.csv"
    exp_path = f"{exp_dir}/triad_amp{amp}.csv"
    if os.path.exists(rk4_path):
        tt, d = load(rk4_path)
        rk4_rates.append(abs((d[-1, 4] - d[0, 4]) / d[0, 4] / max(tt[-1], 1e-12)))
    else:
        rk4_rates.append(np.nan)
    if os.path.exists(exp_path):
        tt, d = load(exp_path)
        exp_rates.append(abs((d[-1, 4] - d[0, 4]) / d[0, 4] / max(tt[-1], 1e-12)))
    else:
        exp_rates.append(np.nan)

amp_num = [float(a) for a in amps]
ax_rate.loglog(amp_num, rk4_rates, "o-", color="#1f77b4",
               lw=2, markersize=9, label="Strang(RK4, NL_RK4)")
ax_rate.loglog(amp_num, exp_rates, "s-", color="#d62728",
               lw=2, markersize=9, label="Strang(Exp, NL_RK4)")
# Reference lines
ref_amp = np.array([1e-6, 1e-2])
ax_rate.loglog(ref_amp, 8e-7 * np.ones_like(ref_amp), ":",
               color="#1f77b4", alpha=0.6, label="RK4 linear-block floor ~8e-7")
ax_rate.loglog(ref_amp, 8e-3 * ref_amp**3, ":",
               color="#d62728", alpha=0.6, label="commutator ~amp³")
ax_rate.set_xlabel("IC amplitude")
ax_rate.set_ylabel("|ΔE_k1 / E_k1(0)|  per T_a")
ax_rate.set_title("Linear-block / commutator decomposition of E_k1 drift\n"
                  "(500 T_a, Lane-Emden n=3/2, 64²)")
ax_rate.grid(True, which="both", alpha=0.3)
ax_rate.legend(fontsize=9)

# Right: time series |ΔE_k1/E_k1(0)| for amp=1e-2 to show RK4 and Exp nearly
# identical (commutator-dominated), while amp=1e-6 shows huge gap.
for amp, col in [("1e-6", "#440154"), ("1e-2", "#fde725")]:
    for style, label_prefix, dirpath in [
            ("-", "RK4", rk4_dir), ("--", "Exp", exp_dir)]:
        path = f"{dirpath}/triad_amp{amp}.csv"
        if not os.path.exists(path): continue
        tt, d = load(path)
        drift = np.abs(d[:, 4] - d[0, 4]) / d[0, 4] + 1e-18
        ax_ts.semilogy(tt, drift, style, color=col, lw=1.3,
                       label=f"amp={amp}, {label_prefix}")

ax_ts.set_xlabel("periods of g-mode n_g=1")
ax_ts.set_ylabel("|ΔE_k1 / E_k1(0)|")
ax_ts.set_title("Drift time series: RK4 (solid) vs Exp (dashed)\n"
                "amp=1e-6 gap huge (linear-block-dominated); "
                "amp=1e-2 overlap (commutator-dominated)")
ax_ts.grid(True, which="both", alpha=0.3)
ax_ts.legend(fontsize=9)

plt.suptitle("DNS Experiment A: linear block vs nonlinear commutator decomposition",
             fontsize=11)
plt.tight_layout()
out = "paper/figures/fig7_1_triad_exp_decomposition.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved → {out}")

print("\nTable: linear-block and commutator contributions to 500-T_a E_k1 drift rate")
print(f"  {'amp':>6}  {'RK4 rate':>11}  {'Exp rate':>11}  "
      f"{'ratio RK4/Exp':>14}")
for amp, r, e in zip(amps, rk4_rates, exp_rates):
    ratio = r / e if e > 0 else float("nan")
    print(f"  {amp:>6}  {r:>11.3e}  {e:>11.3e}  {ratio:>14.2e}")
