#!/usr/bin/env python3
"""Canonical E1 triad figure:  paper/figures/fig7_2_triad_coupled.png

Reads a *densely-sampled* dns_triad_coupled CSV (ANSL_DIAG_EVERY_STEP=1)
and produces a 4-panel summary that exposes the real b↔c Manley–Rowe
oscillation — the result is invisible on per-period sampling because the
2·ω_b signal aliases to a spurious ~15 T_a beat.

Panels:
  (1) Raw E_a, E_b, E_c time series (shows mode-b self-oscillation at 2ω_b)
  (2) Low-pass-filtered envelopes E_b_lp, E_c_lp (the real slow triad beat)
  (3) FFT of E_b(t), with 2ω_b marker and triad |ω_a+ω_b-ω_c| beat marker
  (4) E_c_lp envelope with T_beat/2 markers predicted from detuning

Input CSV header (parsed via regex):
  # triad: a=(n6,kx1,amp1e-05)  b=(n3,kx5,amp1e-05)  c=(?,kx6)
  # omega_a=X period_a=Y dt=Z spp=W

Usage:
    python3 scripts/plot_dns_triad_coupled_dense.py <csv> [--outfig FN]
"""
import argparse
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HDR1_RE = re.compile(
    r"#\s*triad:\s*a=\(n(?P<na>\d+),kx(?P<ka>\d+),amp(?P<ampa>[^\)]+)\)"
    r"\s+b=\(n(?P<nb>\d+),kx(?P<kb>\d+),amp(?P<ampb>[^\)]+)\)"
    r"\s+c=\(\?,kx(?P<kc>\d+)\)")


def parse_header(path):
    with open(path) as f:
        line1, line2 = f.readline(), f.readline()
    m = HDR1_RE.search(line1)
    if not m:
        raise ValueError(f"bad header line 1: {line1!r}")
    cfg = {k: (int if k in ("na", "ka", "nb", "kb", "kc") else float)(v)
           for k, v in m.groupdict().items()}
    for tag in ("omega_a", "period_a", "dt"):
        mm = re.search(rf"{tag}=(\S+)", line2)
        cfg[tag] = float(mm.group(1))
    mm = re.search(r"spp=(\S+)", line2)
    cfg["spp"] = int(float(mm.group(1)))
    return cfg


def lowpass_ma(sig, window):
    """Simple moving-average low-pass.  window in samples."""
    kernel = np.ones(window) / window
    return np.convolve(sig, kernel, mode="same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--outfig", default="paper/figures/fig7_2_triad_coupled.png")
    # For the beat-period overlay we need ω_b and ω_c.  We can't pull them
    # from the CSV header alone — they depend on background + TANH grid.
    # Allow override, otherwise default to the canonical E1 run values
    # (n_b=3, k_b=5, Lane-Emden ρ_cut=0.1, TANH β=2).
    ap.add_argument("--omega_b", type=float, default=1.590169)
    ap.add_argument("--omega_c", type=float, default=2.036564)
    args = ap.parse_args()

    cfg = parse_header(args.csv_path)
    data = np.loadtxt(args.csv_path, comments="#")
    mask = np.isfinite(data[:, 2:9]).all(axis=1)
    if mask.sum() < len(data):
        print(f"dropped {len(data) - mask.sum()} non-finite rows")
    data = data[mask]

    t   = data[:, 0]
    Ek  = data[:, 3:9]
    T_a = cfg["period_a"]
    dt  = t[1] - t[0]
    n_per = t / T_a

    E_a = Ek[:, cfg["ka"] - 1]
    E_b = Ek[:, cfg["kb"] - 1]
    E_c = Ek[:, cfg["kc"] - 1]

    # Expected frequencies / beat.
    omega_a = cfg["omega_a"]
    omega_b = args.omega_b
    omega_c = args.omega_c
    dw = omega_a + omega_b - omega_c
    T_beat = 2 * np.pi / max(abs(dw), 1e-12)
    T_beat_a = T_beat / T_a

    print(f"Loaded {len(t)} samples, dt={dt:.4f}, {n_per[-1]:.1f} periods")
    print(f"ω_a={omega_a:.4f}, ω_b={omega_b:.4f}, ω_c={omega_c:.4f}")
    print(f"Δω = {dw:+.4e}  ({abs(dw)/omega_c*100:.3f}%)")
    print(f"T_beat = {T_beat:.1f} time units ({T_beat_a:.1f} T_a)")

    # Low-pass to remove the 2·ω_b self-oscillation.  Half-period of 2ω_b
    # is π/(2ω_b) = T_b/2.  Use a window of ~T_b/2 samples.
    T_b = 2 * np.pi / omega_b
    window = max(3, int(round(T_b / 2.0 / dt)))
    print(f"low-pass window: {window} samples ({window*dt:.2f} time units "
          f"= T_b/2)")
    E_b_lp = lowpass_ma(E_b, window)
    E_c_lp = lowpass_ma(E_c, window)

    # Strip window/2 edge artefacts from the LP envelope.
    edge = window
    valid = slice(edge, len(t) - edge)
    n_per_v = n_per[valid]
    E_b_lp_v = E_b_lp[valid]
    E_c_lp_v = E_c_lp[valid]

    # ── Figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) Raw E_a, E_b, E_c.  Show only a short window so the 2ω_b fast
    # oscillation is visible (plotting 200 T_a with 6400 points makes the
    # fast sinusoid look like noise).  Pick periods 0..20.
    ax = axes[0, 0]
    show = n_per <= 20
    ax.semilogy(n_per[show], E_a[show], "-", color="#1f77b4", lw=1.2,
                label=f"$E_a$ $(k={cfg['ka']})$ pump")
    ax.semilogy(n_per[show], E_b[show], "-", color="#d62728", lw=1.0,
                label=f"$E_b$ $(k={cfg['kb']})$ seeded")
    ax.semilogy(n_per[show], np.clip(E_c[show], 1e-40, None), "-",
                color="#2ca02c", lw=1.0,
                label=f"$E_c$ $(k={cfg['kc']})$ partner")
    ax.set_xlabel("periods of mode $a$")
    ax.set_ylabel("modal band energy")
    ax.set_title("Raw densely-sampled signal (first 20 $T_a$) — "
                 "fast $2\\omega_b$ oscillation visible")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # (2) Low-pass envelopes: the real slow triad beat.
    ax = axes[0, 1]
    ax.plot(n_per_v, E_b_lp_v, "-", color="#d62728", lw=1.5,
            label="$\\langle E_b \\rangle$ (slow envelope)")
    ax.plot(n_per_v, E_c_lp_v, "-", color="#2ca02c", lw=1.5,
            label="$\\langle E_c \\rangle$ (slow envelope)")
    # Mark T_beat/2 points.
    for m in range(1, int(n_per_v[-1] / (T_beat_a / 2)) + 2):
        xm = m * T_beat_a / 2
        if xm <= n_per_v[-1]:
            lbl = f"$T_{{\\rm beat}}/2$ = {T_beat_a/2:.0f} $T_a$" if m == 1 else None
            ax.axvline(xm, color="gray", ls=":", lw=0.8, alpha=0.6, label=lbl)
    ax.set_yscale("log")
    ax.set_xlabel("periods of mode $a$")
    ax.set_ylabel("low-pass $\\langle E \\rangle$")
    ax.set_title("Manley–Rowe $b \\leftrightarrow c$ (self-oscillation filtered out)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # (3) FFT of E_b(t) — shows 2ω_b self-oscillation + optional triad beat.
    ax = axes[1, 0]
    sig = E_b - E_b.mean()
    window_fn = np.hanning(len(sig))
    SP = np.fft.rfft(sig * window_fn)
    om = 2 * np.pi * np.fft.rfftfreq(len(sig), d=dt)
    ampl = np.abs(SP)
    ax.loglog(om[1:], ampl[1:], "-", color="#333333", lw=0.8,
              label="$|\\widetilde{E}_b(\\omega)|$ (FFT)")
    # Mark 2ω_b, 2ω_c, beat.
    ax.axvline(2 * omega_b, color="#d62728", ls="--", lw=1.2,
               label=f"$2\\omega_b = {2*omega_b:.3f}$")
    ax.axvline(2 * omega_c, color="#2ca02c", ls="--", lw=1.0,
               label=f"$2\\omega_c = {2*omega_c:.3f}$")
    ax.axvline(abs(dw), color="blue", ls=":", lw=1.2,
               label=f"$|\\Delta\\omega| = {abs(dw):.4f}$ (triad beat)")
    ax.set_xlabel("$\\omega$ (rad / time unit)")
    ax.set_ylabel("FFT amplitude")
    ax.set_title("Spectrum of $E_b(t)$ — self-oscillation at $2\\omega_b$ dominates")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(1e-3, om[-1])

    # (4) E_c_lp envelope zoom with theory markers.
    ax = axes[1, 1]
    ax.plot(n_per_v, E_c_lp_v, "-", color="#2ca02c", lw=1.6,
            label="$\\langle E_c \\rangle(t)$ (low-pass)")
    # First T_beat/2 = predicted first peak.
    xp = T_beat_a / 2
    if xp <= n_per_v[-1]:
        ax.axvline(xp, color="gray", ls="--", lw=1.0,
                   label=f"theory: peak @ $T_{{\\rm beat}}/2$ "
                         f"= {xp:.0f} $T_a$")
    for m in range(2, int(n_per_v[-1] / (T_beat_a / 2)) + 2):
        xm = m * T_beat_a / 2
        if xm <= n_per_v[-1]:
            ax.axvline(xm, color="gray", ls=":", lw=0.7, alpha=0.5)
    # Theoretical peak amplitude (undepleted-pump detuned triad).
    amp_a = cfg["ampa"]; amp_b = cfg["ampb"]
    Ec_theory = (amp_a * amp_b / abs(dw)) ** 2
    ax.axhline(Ec_theory, color="#ff7f0e", ls="-.", lw=1.0,
               label=f"$|c_c|^2_{{\\max}} \\sim |V_{{abc}}|^2 (c_a c_b/\\Delta\\omega)^2$"
                     f" = {Ec_theory:.1e}")
    ax.axhline(E_c_lp_v[0] if E_c_lp_v[0] > 0 else 1e-20,
               color="k", ls=":", lw=0.6,
               label=f"IC / floor = {E_c_lp_v[0]:.1e}")
    ax.set_yscale("log")
    ax.set_xlabel("periods of mode $a$")
    ax.set_ylabel("$\\langle E_c \\rangle$")
    ax.set_title(f"Partner growth  vs  detuned-triad prediction  "
                 f"($|\\Delta\\omega|/\\omega_c = {abs(dw)/omega_c*100:.2f}\\%$)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"DNS E1 three-wave triad:  "
        f"$a=(n_g{{=}}{cfg['na']}, k_x{{=}}{cfg['ka']})$ "
        f"+ $b=(n_g{{=}}{cfg['nb']}, k_x{{=}}{cfg['kb']})$ "
        f"$\\to$ $k_c{{=}}{cfg['kc']}$;  "
        f"amp=$10^{{-5}}$, dense sampling (dt={dt:.3f}), 200 $T_a$.\n"
        f"ω: {omega_a:.3f} + {omega_b:.3f} = {omega_a+omega_b:.3f}  vs  "
        f"$\\omega_c$ = {omega_c:.3f}  (detune {abs(dw)/omega_c*100:.2f}%)",
        fontsize=10)
    plt.tight_layout()
    plt.savefig(args.outfig, dpi=140, bbox_inches="tight")
    print(f"saved → {args.outfig}")


if __name__ == "__main__":
    main()
