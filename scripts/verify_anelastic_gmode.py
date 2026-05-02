#!/usr/bin/env python3
"""Phase 1d gmode probe analysis.

Reads gmode_probe.csv (columns: t, v_center) and runs FFT to extract the
dominant pulsation frequency.  For a Lane-Emden n=3/2 anelastic cavity
with a k_y=1 sinusoidal IC, we expect a single dominant g-mode oscillation
whose ω² sits in the Exp K spectrum.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    args = ap.parse_args()

    data = np.loadtxt(args.csv, comments="#")
    t = data[:, 0]
    v = data[:, 1]
    n = len(t)
    dt = float(np.median(np.diff(t)))
    print(f"loaded {n} samples, dt = {dt:.3e},  t ∈ [{t[0]:.3f}, {t[-1]:.3f}]")
    print(f"  |v_center|_max = {np.max(np.abs(v)):.4e}")

    # Detrend + window + FFT
    v_ac = v - v.mean()
    window = np.hanning(n)
    V = np.fft.rfft(v_ac * window)
    freqs = np.fft.rfftfreq(n, d=dt)        # cycles per unit t
    omega = 2 * np.pi * freqs                # angular frequency

    power = np.abs(V) ** 2
    mask = omega > 0
    if mask.sum() == 0:
        print("FAIL: no non-zero frequencies.")
        sys.exit(1)
    peak = np.argmax(power[mask])
    omega_peak = omega[mask][peak]
    omega_sq_peak = omega_peak ** 2

    print(f"  peak ω = {omega_peak:.4f}  rad/unit_t")
    print(f"  peak ω² = {omega_sq_peak:.4e}")
    print(f"  period T = {2 * np.pi / omega_peak:.4f}")

    # Top 5 peaks for diagnostic
    p = power[mask].copy()
    w = omega[mask].copy()
    for i in range(5):
        k = np.argmax(p)
        print(f"    [{i+1}] ω = {w[k]:.4f}   ω² = {w[k]**2:.4e}   power = {p[k]:.3e}")
        lo, hi = max(0, k - 2), min(len(p), k + 3)
        p[lo:hi] = 0.0

    if not np.isfinite(omega_peak) or omega_peak <= 0:
        print("FAIL: invalid peak frequency.")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
