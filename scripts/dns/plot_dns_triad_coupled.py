#!/usr/bin/env python3
"""Plot DNS Experiment E1 (true three-wave triad) GPU results.
Output: paper/figures/fig7_2_triad_coupled.png

Shows modal-energy exchange between (a, b, c) in a resonant triad:
  a = (n_g=6, kx=1)   ω_a ≈ 0.4501
  b = (n_g=3, kx=5)   ω_b ≈ 1.5903
  c = (?  , kx=6)     ω_c ≈ ω_a + ω_b  (detune < 0.2%)

IC seeds a and b at equal amplitude; c is born from noise and should
exhibit Manley-Rowe / Lamb-Bretherton oscillatory energy exchange
between (b, c) while a loses energy slowly.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = "runs/dns_expE1_triad.csv"
data = np.loadtxt(path, comments="#")
# cols: t max_v E_total E_k1 E_k2 E_k3 E_k4 E_k5 E_k6
t   = data[:, 0]
Et  = data[:, 2]
Ek  = data[:, 3:9]   # k=1..6

# Strip NaN tail.
mask = np.isfinite(Ek).all(axis=1)
t = t[mask]; Et = Et[mask]; Ek = Ek[mask]

# Header parse: period
with open(path) as f:
    hdr1 = f.readline()        # triad config
    hdr2 = f.readline()        # omega / period / dt
    hdr3 = f.readline()        # columns
T_period = float([p for p in hdr2.split() if p.startswith("period_a=")][0].split("=")[1])
n_periods = t / T_period

print(f"Loaded {len(t)} samples, up to period {n_periods[-1]:.1f}")
print(f"E_k1 drift:  {(Ek[-1,0]-Ek[0,0])/Ek[0,0]*100:+.2f}%")
print(f"E_k5 range:  {Ek[:,4].min():.3e}  to  {Ek[:,4].max():.3e}  (IC: {Ek[0,4]:.3e})")
print(f"E_k6 range:  {Ek[:,5].min():.3e}  to  {Ek[:,5].max():.3e}  (IC: {Ek[0,5]:.3e})")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ── (1) Three-mode modal energy vs time ────────────────────────────
ax = axes[0, 0]
ax.plot(n_periods, Ek[:, 0], "-",  color="#1f77b4", lw=1.8,
        label="E_k1 (mode a, seeded)")
ax.plot(n_periods, Ek[:, 4], "-",  color="#d62728", lw=1.8,
        label="E_k5 (mode b, seeded)")
ax.plot(n_periods, Ek[:, 5], "-",  color="#2ca02c", lw=2.2,
        label="E_k6 (mode c, grows from noise)")
ax.set_yscale("log")
ax.set_xlabel("periods of mode a")
ax.set_ylabel("E_k")
ax.set_title("Three-wave triad: modal energy exchange")
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)

# ── (2) Manley-Rowe: E_b + E_c conservation check  ─────────────────
ax = axes[0, 1]
ax.plot(n_periods, Ek[:, 4], "-",  color="#d62728", lw=1.5, label="E_b (k=5)")
ax.plot(n_periods, Ek[:, 5], "-",  color="#2ca02c", lw=1.5, label="E_c (k=6)")
ax.plot(n_periods, Ek[:, 4] + Ek[:, 5], "--", color="k", lw=2.0,
        label="E_b + E_c (Manley-Rowe partner)")
# Reference E_b(0) for scale
ax.axhline(Ek[0, 4], color="#d62728", ls=":", lw=0.8, alpha=0.5,
           label=f"E_b(0) = {Ek[0,4]:.2e}")
ax.set_yscale("log")
ax.set_xlabel("periods of mode a")
ax.set_ylabel("energy")
ax.set_title("b ↔ c resonant energy exchange")
ax.legend(loc="lower left", fontsize=9)
ax.grid(True, alpha=0.3)

# ── (3) Quiet modes E_k2, E_k3, E_k4 ───────────────────────────────
ax = axes[1, 0]
for kidx, c in [(1, "#ff7f0e"), (2, "#9467bd"), (3, "#8c564b")]:
    ax.plot(n_periods, Ek[:, kidx] + 1e-40,
            "-", color=c, lw=1.0, label=f"E_k{kidx+1}")
ax.set_yscale("log")
ax.set_ylim(bottom=1e-20)
ax.set_xlabel("periods of mode a")
ax.set_ylabel("E_k")
ax.set_title("Non-resonant modes remain at numerical noise level")
ax.legend()
ax.grid(True, alpha=0.3)

# ── (4) E_k6 early-time growth: expect ∝ t² in weakly nonlinear ────
ax = axes[1, 1]
# Focus on first 15 periods where c grows from zero.
mask_early = n_periods <= 16
ax.loglog(n_periods[mask_early] + 1e-3, Ek[mask_early, 5],
          "o-", color="#2ca02c", lw=1.5, label="E_k6 (mode c)")
# Reference line E_c ∝ t² = (n_periods)²
t_ref = n_periods[mask_early][1:]
E_fit = Ek[mask_early, 5][2]
t_fit = n_periods[mask_early][2]
ax.plot(t_ref, E_fit * (t_ref / t_fit) ** 2, "k--", lw=1.2,
        label="∝ t² (triad coupling theory)")
ax.plot(t_ref, E_fit * (t_ref / t_fit) ** 4, "k:", lw=1.2,
        label="∝ t⁴")
ax.set_xlabel("periods of mode a")
ax.set_ylabel("E_k6")
ax.set_title("Early growth of resonance partner c (noise → triad)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, which="both", alpha=0.3)

plt.suptitle(
    "DNS Experiment E1: 3-wave triad (6, k=1) + (3, k=5) → (?, k=6)\n"
    "Lane-Emden n=3/2, Ny=128×Nx=128, amp=5e-5, GPU Strang-split",
    fontsize=11)
plt.tight_layout()
out = "paper/figures/fig7_2_triad_coupled.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nsaved → {out}")
