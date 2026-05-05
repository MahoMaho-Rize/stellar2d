#!/usr/bin/env python3
"""Plot DNS Experiment E1 (true three-wave triad) — REVISED 2026-05-04.

CSV columns (new layout): t max_v E_kin_total E_k1..E_k6 E_pot

Showcases:
  1. Three-wave frequency + wavenumber resonance:
        n_a=6 (kx=1, ω=0.4501) + n_b=3 (kx=5, ω=1.5903)
        → k_c=6, ω_c=2.0366   (detune 0.19%)
  2. Manley-Rowe b↔c partner exchange (sinc²-modulated by residual detune)
  3. Non-resonant modes stay at numerical noise
  4. |c(t)| rise from noise floor — amplitude plot with slope-fit
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = "runs/dns_expE1_triad.csv"
data = np.loadtxt(path, comments="#")

# Strip NaN rows (post-blowup).
mask = np.isfinite(data[:, 2:9]).all(axis=1)
data = data[mask]

t   = data[:, 0]
max_v = data[:, 1]
E_kin = data[:, 2]         # sum of E_k1..E_k6
Ek    = data[:, 3:9]       # k=1..6
E_pot = data[:, 9] if data.shape[1] > 9 else np.zeros_like(t)

with open(path) as f:
    f.readline(); hdr = f.readline()
T_period = float([p for p in hdr.split() if p.startswith("period_a=")][0].split("=")[1])
n_per = t / T_period

print(f"Loaded {len(t)} samples, up to period {n_per[-1]:.1f}")
print(f"E_k1 drift: {(Ek[-1,0]-Ek[0,0])/Ek[0,0]*100:+.3f}%")
print(f"E_kin drift: {(E_kin[-1]-E_kin[0])/E_kin[0]*100:+.3f}%")
print(f"E_pot max:  {E_pot.max():.3e}  (should be 0 — b frozen in linear block)")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ── (1) Modal energy time series ────────────────────────────────────
ax = axes[0, 0]
ax.semilogy(n_per, Ek[:, 0], "-",  color="#1f77b4", lw=1.8,
            label="$E_{k=1}$ (mode $a$, seeded)")
ax.semilogy(n_per, Ek[:, 4], "-",  color="#d62728", lw=1.8,
            label="$E_{k=5}$ (mode $b$, seeded)")
ax.semilogy(n_per, np.clip(Ek[:, 5], 1e-40, None), "-",
            color="#2ca02c", lw=2.0,
            label="$E_{k=6}$ (mode $c$: $k_a+k_b$ = 6)")
ax.axhline(Ek[0, 0], color="#1f77b4", ls=":", lw=0.6, alpha=0.5)
ax.axhline(Ek[0, 4], color="#d62728", ls=":", lw=0.6, alpha=0.5)
ax.set_xlabel("periods of mode $a$")
ax.set_ylabel("modal kinetic energy $E_k$")
ax.set_title("Three-wave triad modal energies")
ax.legend(loc="center right", fontsize=10)
ax.grid(True, alpha=0.3)

# ── (2) Manley-Rowe b↔c exchange ────────────────────────────────────
ax = axes[0, 1]
ax.plot(n_per, Ek[:, 4], "-",  color="#d62728", lw=1.5, label="$E_b$ $(k_x=5)$")
ax.plot(n_per, Ek[:, 5], "-",  color="#2ca02c", lw=1.5, label="$E_c$ $(k_x=6)$")
ax.plot(n_per, Ek[:, 4] + Ek[:, 5], "--", color="k", lw=1.5,
        label="$E_b + E_c$")
ax.axhline(Ek[0, 4], color="#d62728", ls=":", lw=0.6, alpha=0.5,
           label=f"$E_b(0) = {Ek[0,4]:.2e}$")
ax.set_yscale("log")
ax.set_xlabel("periods of mode $a$")
ax.set_ylabel("energy")
ax.set_title("Manley–Rowe: $b \\leftrightarrow c$ energy exchange  "
             "($\\mathrm{sinc}^2$-modulated by $\\Delta\\omega \\approx 0.2\\%$)")
ax.legend(loc="lower left", fontsize=9)
ax.grid(True, alpha=0.3)

# ── (3) Selectivity: non-resonant modes stay at noise ───────────────
ax = axes[1, 0]
for kidx, cl, lab in [(1, "#ff7f0e", "$k=2$"),
                       (2, "#9467bd", "$k=3$"),
                       (3, "#8c564b", "$k=4$")]:
    ax.semilogy(n_per, np.clip(Ek[:, kidx], 1e-30, None),
                "-", color=cl, lw=1.0, label=lab)
ax.set_ylim(bottom=1e-25, top=1e-10)
ax.set_xlabel("periods of mode $a$")
ax.set_ylabel("$E_k$")
ax.set_title("Non-resonant modes: stay at round-off noise (selectivity proof)")
ax.legend()
ax.grid(True, alpha=0.3)

# ── (4) |c(t)| amplitude vs  t, log-log, early regime ───────────────
ax = axes[1, 1]
n_early = 20
m_early = n_per <= n_early
# Use sqrt(E_c) as mode-c amplitude proxy.  Filter out pre-birth noise.
c_amp = np.sqrt(np.clip(Ek[m_early, 5], 1e-40, None))
# Find the first "real" (above noise) index.
noise_floor = np.sqrt(Ek[0, 5]) if Ek[0, 5] > 1e-30 else 1e-20
valid = c_amp > 3 * noise_floor
if valid.sum() > 4:
    t_valid = n_per[m_early][valid]
    c_valid = c_amp[valid]
    # Log-log fit slope on first few points.
    fit_n = min(10, len(t_valid))
    coef = np.polyfit(np.log(t_valid[:fit_n]),
                      np.log(c_valid[:fit_n]), 1)
    ax.loglog(n_per[m_early] + 1e-2, c_amp, "o",
              color="#2ca02c", ms=3, label="$|c(t)|$ from sim")
    tref = np.linspace(t_valid[1], t_valid[-1], 30)
    ax.loglog(tref, np.exp(coef[1]) * tref ** coef[0],
              "k--", lw=1.3,
              label=f"fit $\\propto t^{{{coef[0]:.2f}}}$")
    # Reference: weakly-nonlinear theory predicts |c| ∝ t.
    ax.loglog(tref, np.exp(coef[1]) * (tref / tref[0]) * t_valid[1],
              "r:", lw=1.0, label="$\\propto t$ (theory)")
else:
    ax.text(0.5, 0.5, "Insufficient signal", transform=ax.transAxes,
            ha="center", fontsize=14)
ax.set_xlabel("periods (log)")
ax.set_ylabel("$|c(t)| = \\sqrt{E_c}$")
ax.set_title("Mode-$c$ amplitude growth from noise (early regime)")
ax.legend(loc="lower right")
ax.grid(True, which="both", alpha=0.3)

plt.suptitle(
    "DNS E1: true 3-wave triad  $n_a = 6$ $(k_x{=}1)$ + "
    "$n_b = 3$ $(k_x{=}5)$ $\\to$ $k_c = 6$\n"
    "Lane-Emden $n{=}3/2$, $128{\\times}128$, amp=$5{\\times}10^{-5}$, "
    "GPU Strang-split.  Undepleted pump ($E_a{\\gg}E_{b,c}$).",
    fontsize=11)
plt.tight_layout()
out = "paper/figures/fig7_2_triad_coupled.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nsaved → {out}")
