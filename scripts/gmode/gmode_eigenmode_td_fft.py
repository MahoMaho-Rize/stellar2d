#!/usr/bin/env python3
"""FFT the v_center time series from --test gmode_eigenmode_td.

Fits the dominant frequency of v(t), compares against the EVP ω stored in
the CSV header.  Usage:

    python scripts/gmode_eigenmode_td_fft.py /tmp/ansl_eigen_td/<run>/gmode_eigenmode_td.csv

For a dt-scan (Chorin order diagnosis) pass multiple files:

    python scripts/gmode_eigenmode_td_fft.py run_dt1.csv run_dt2.csv ...
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import numpy as np


HDR_RE = re.compile(
    r"kx_int=(\S+)\s+n_g=(\S+)\s+N2=(\S+)\s+omega_sq_evp=(\S+)\s+omega_evp=(\S+)\s+amp=(\S+)"
)


def load_probe(path: Path):
    with path.open() as fp:
        header = fp.readline().lstrip("#").strip()
    m = HDR_RE.search(header)
    if not m:
        sys.exit(f"cannot parse header in {path}: {header!r}")
    kx_int = int(m.group(1))
    n_g    = int(m.group(2))
    N2     = float(m.group(3))
    om2    = float(m.group(4))
    om     = float(m.group(5))
    amp    = float(m.group(6))
    data = np.loadtxt(path, comments="#")
    t = data[:, 0]
    v = data[:, 1]
    return dict(kx_int=kx_int, n_g=n_g, N2=N2, om2_evp=om2, om_evp=om,
                amp=amp, t=t, v=v, path=path)


def estimate_frequency(t: np.ndarray, v: np.ndarray, om_expected: float):
    """Resample to uniform dt, FFT, and pick the strongest peak near the
    expected frequency.  Parabolic interpolation runs only on the bin of
    that local peak, avoiding pull from neighbouring spurious peaks."""
    t0, t1 = t[0], t[-1]
    n = len(t)
    dt_uniform = (t1 - t0) / (n - 1)
    t_uni = np.linspace(t0, t1, n)
    v_uni = np.interp(t_uni, t, v)
    w = np.hanning(n)
    V = np.fft.rfft((v_uni - v_uni.mean()) * w)
    freqs = np.fft.rfftfreq(n, d=dt_uniform)
    P = np.abs(V) ** 2
    f_expected = om_expected / (2.0 * np.pi)
    # Restrict search to a narrow window around the expected frequency
    # (± 10 %) so the peak-picker cannot latch onto the second g-mode.
    f_lo = 0.90 * f_expected
    f_hi = 1.10 * f_expected
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not mask.any():
        mask = freqs > 0
    local_idx = np.argmax(P[mask])
    kmax = np.flatnonzero(mask)[local_idx]
    # Parabolic interp ONLY if both neighbours are strictly smaller (i.e. this
    # is a resolved isolated peak).  With a g-mode ladder whose spacing is
    # comparable to the FFT bin width, neighbours can themselves be peaks —
    # parabolic fit would then pull the estimate across bins.
    if (1 < kmax < len(P) - 1
            and P[kmax - 1] < P[kmax] and P[kmax + 1] < P[kmax]
            and P[kmax - 1] > 0 and P[kmax + 1] > 0):
        y0, y1, y2 = np.log(P[kmax - 1]), np.log(P[kmax]), np.log(P[kmax + 1])
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    f_peak = freqs[kmax] + delta * (freqs[1] - freqs[0])
    om_measured = 2.0 * np.pi * f_peak
    # Report leakage: ratio of peak power to total.
    leakage = 1.0 - (P[kmax] / P[1:].sum())
    return om_measured, dt_uniform, leakage


def analyse(rec: dict):
    om_evp = rec["om_evp"]
    om_meas, dt_u, leakage = estimate_frequency(rec["t"], rec["v"], om_evp)
    rel = (om_meas - om_evp) / om_evp
    print(f"{rec['path'].parent.name}/{rec['path'].name}")
    print(f"  kx_int={rec['kx_int']}  n_g={rec['n_g']}  N²={rec['N2']:g}")
    print(f"  samples={len(rec['t'])}  dt~{dt_u:.4e}  t_end={rec['t'][-1]:.4f}")
    print(f"  ω_EVP           = {om_evp:.10e}")
    print(f"  ω_measured      = {om_meas:.10e}")
    print(f"  rel_err         = {rel:+.4e}")
    print(f"  spectral leakage = {leakage:.3f}  (0 = pure tone)")
    out = dict(rec)
    out.update(dt=dt_u, om_meas=om_meas, rel=rel, leakage=leakage)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", type=Path)
    args = ap.parse_args()
    results = []
    for p in args.csv:
        rec = load_probe(p)
        results.append(analyse(rec))
        print()

    if len(results) >= 2:
        print("dt-scan summary (diagnose Chorin splitting order):")
        print(f"  {'dt':>12}  {'ω_meas':>18}  {'rel_err':>14}")
        for r in results:
            print(f"  {r['dt']:12.4e}  {r['om_meas']:18.10e}  {r['rel']:+14.4e}")
        # Fit rel_err ∝ dt^p by log-log regression.
        dts  = np.array([r["dt"]  for r in results])
        errs = np.array([abs(r["rel"]) for r in results])
        if np.all(errs > 0) and np.all(dts > 0):
            p, logC = np.polyfit(np.log(dts), np.log(errs), 1)
            print(f"  fit: |rel_err| ∝ dt^{p:.3f}  (pure Chorin splitting ⇒ p≈1)")


if __name__ == "__main__":
    main()
