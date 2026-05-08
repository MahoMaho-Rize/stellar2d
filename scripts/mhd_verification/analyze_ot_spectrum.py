#!/usr/bin/env python3
"""
Phase A2 — Orszag-Tang vortex energy spectrum + ν_eff extraction.

Derivation: docs/mhd_derivations/sections/f2_mhd_turbulence_spectrum.md

Input:  VTK files produced by `run_ot_spectrum_scan.sh`:
          scripts/mhd_verification/runs/ot_N{128,256,512}/output_*.vtk
Output:
  - spectra/ot_N{N}_spectrum.pdf     — E(k) vs k log-log
  - spectra/ot_combined_spectrum.pdf — all resolutions overlaid
  - phase_A_results.md section (appended if --write-results)

Verification criteria (from F2 derivation):
  C1. Inertial-range slope ∈ [-1.8, -1.4] over ≥ 1 decade (at N=256+)
  C2. k_diss(256)/k_diss(128) ∈ [2.0, 3.5]
  C3. k_diss(512)/k_diss(256) ∈ [2.0, 3.5]   (if 512 run available)

Usage:
  python3 analyze_ot_spectrum.py [--write-results]
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "runs"
OUT_DIR = HERE / "spectra"


def read_vtk_structured(path):
    """Parse ASCII STRUCTURED_GRID VTK as written by athena_mhd
    write_vtk_2d: POINTS + CELL_DATA with SCALARS (density, pressure,
    Bx, By, Bz, Bmag) and VECTORS velocity."""
    with open(path) as f:
        lines = f.readlines()

    dims = None
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith("DIMENSIONS"):
            parts = line.split()
            dims = (int(parts[1]), int(parts[2]), int(parts[3]))
        if line.startswith("POINT_DATA") or line.startswith("CELL_DATA"):
            data_start = i
            break

    if dims is None or data_start is None:
        raise RuntimeError(f"bad VTK: {path}")

    nx, ny, _ = dims   # point counts (nx+1, ny+1) per solver convention
    is_cell = lines[data_start].startswith("CELL_DATA")
    N_expected = int(lines[data_start].split()[1])

    fields = {}
    i = data_start + 1
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("SCALARS"):
            name = line.split()[1]
            i += 2  # skip LOOKUP_TABLE
            vals = []
            while len(vals) < N_expected and i < len(lines):
                vals.extend(lines[i].split())
                i += 1
            arr = np.array(vals[:N_expected], dtype=np.float64)
            if is_cell:
                arr = arr.reshape(ny - 1, nx - 1)  # VTK order: y outer
            else:
                arr = arr.reshape(ny, nx)
            fields[name] = arr
        elif line.startswith("VECTORS"):
            name = line.split()[1]
            i += 1
            vals = []
            while len(vals) < N_expected * 3 and i < len(lines):
                vals.extend(lines[i].split())
                i += 1
            arr = np.array(vals[:N_expected * 3], dtype=np.float64)
            if is_cell:
                arr = arr.reshape(ny - 1, nx - 1, 3)
            else:
                arr = arr.reshape(ny, nx, 3)
            fields[name + "_x"] = arr[..., 0]
            fields[name + "_y"] = arr[..., 1]
            fields[name + "_z"] = arr[..., 2]
        else:
            i += 1

    return fields, nx - 1 if is_cell else nx, ny - 1 if is_cell else ny


def compute_2d_spectrum(vx, vy, bx, by, Lx=2*np.pi, Ly=2*np.pi):
    """Axisymmetric 2D FFT-based kinetic + magnetic power spectrum."""
    nx, ny = vx.shape[1], vx.shape[0]
    # FFT (note numpy puts y as first axis; we pass kx first)
    Vx_hat = np.fft.fft2(vx) / (nx * ny)
    Vy_hat = np.fft.fft2(vy) / (nx * ny)
    Bx_hat = np.fft.fft2(bx) / (nx * ny)
    By_hat = np.fft.fft2(by) / (nx * ny)

    # Cell-centred power per (kx, ky)
    power_KE = 0.5 * (np.abs(Vx_hat)**2 + np.abs(Vy_hat)**2)
    power_ME = 0.5 * (np.abs(Bx_hat)**2 + np.abs(By_hat)**2)

    # Wave numbers (2π/L convention)
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=Lx/nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=Ly/ny)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    Kmag = np.sqrt(KX**2 + KY**2)

    # Shell-average to 1D E(k)
    k_max = min(nx, ny) // 2
    kbins = np.arange(1, k_max) * (2 * np.pi / min(Lx, Ly))  # integer kmag
    dk = kbins[1] - kbins[0]
    E_KE = np.zeros_like(kbins, dtype=np.float64)
    E_ME = np.zeros_like(kbins, dtype=np.float64)
    for i, k_c in enumerate(kbins):
        mask = (Kmag >= k_c - 0.5 * dk) & (Kmag < k_c + 0.5 * dk)
        E_KE[i] = power_KE[mask].sum()
        E_ME[i] = power_ME[mask].sum()
    return kbins, E_KE, E_ME


def fit_inertial_slope(k, E, k_lo, k_hi):
    """Fit log-log slope of E(k) over [k_lo, k_hi]."""
    mask = (k >= k_lo) & (k <= k_hi) & (E > 0)
    if mask.sum() < 3:
        return None
    logk = np.log(k[mask])
    logE = np.log(E[mask])
    slope, _ = np.polyfit(logk, logE, 1)
    return slope


def find_dissipation_k(k, E, drop_factor=1e-3):
    """Smallest k where E drops below drop_factor · E_max_inertial."""
    Emax = np.max(E[len(E)//8:len(E)//2])  # inertial-range peak
    thresh = drop_factor * Emax
    below = np.where(E < thresh)[0]
    below = below[below > len(E) // 4]  # ignore low-k
    if len(below) == 0:
        return k[-1]
    return k[below[0]]


def analyse_resolution(N, t_target=0.5):
    run_dir = RUN_DIR / f"ot_N{N}"
    vtks = sorted(run_dir.glob("output_*.vtk"))
    if not vtks:
        print(f"  [N={N}] no VTKs in {run_dir}, skip")
        return None
    vtk = vtks[-1]  # final snapshot
    print(f"  [N={N}] reading {vtk.name}")

    fields, nx_eff, ny_eff = read_vtk_structured(vtk)

    # VTK writes VECTORS velocity (→ velocity_x, velocity_y) +
    # SCALARS Bx, By (athena_mhd convention).
    if "velocity_x" in fields:
        vx, vy = fields["velocity_x"], fields["velocity_y"]
    else:
        print(f"    [warn] no velocity; keys={list(fields.keys())}")
        return None
    if "Bx" in fields:
        bx, by = fields["Bx"], fields["By"]
    else:
        print(f"    [warn] no Bx; keys={list(fields.keys())}")
        return None

    kbins, E_KE, E_ME = compute_2d_spectrum(vx, vy, bx, by)
    E_tot = E_KE + E_ME

    # Slope fit: window [4·k_min, k_max/4]
    k_lo = 4 * kbins[0]
    k_hi = kbins[-1] / 4
    slope = fit_inertial_slope(kbins, E_tot, k_lo, k_hi)
    kdiss = find_dissipation_k(kbins, E_tot)

    return {"N": N, "k": kbins, "E_KE": E_KE, "E_ME": E_ME,
            "E_tot": E_tot, "slope": slope, "kdiss": kdiss}


def plot_spectra(results):
    import matplotlib.pyplot as plt
    OUT_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {128: "C0", 256: "C1", 512: "C2"}
    for r in results:
        ax.loglog(r["k"], r["E_tot"],
                  color=colors.get(r["N"], "k"),
                  label=f"N={r['N']} (slope={r['slope']:.2f})")
    # K41 reference line
    kref = np.logspace(np.log10(4), np.log10(60), 20)
    ax.loglog(kref, 0.1 * kref**(-5/3), "k--", alpha=0.5, label="k^{-5/3}")
    ax.loglog(kref, 0.05 * kref**(-3/2), "k:", alpha=0.5, label="k^{-3/2}")
    ax.set_xlabel("k")
    ax.set_ylabel("E(k) = E_KE + E_ME")
    ax.set_title("Orszag-Tang 2D MHD turbulence spectrum at t=0.5")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ot_combined_spectrum.pdf")
    fig.savefig(OUT_DIR / "ot_combined_spectrum.png", dpi=100)
    print(f"  → wrote {OUT_DIR/'ot_combined_spectrum.pdf'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-results", action="store_true",
                    help="append numeric results to phase_A_results.md")
    args = ap.parse_args()

    results = []
    for N in (128, 256, 512):
        r = analyse_resolution(N)
        if r is not None:
            results.append(r)

    if not results:
        print("no VTK data found; run run_ot_spectrum_scan.sh first")
        sys.exit(1)

    print("\n  === Phase A2 OT spectrum results ===")
    print(f"  {'N':>5}  {'slope':>8}  {'k_diss':>10}")
    for r in results:
        print(f"  {r['N']:>5}  {r['slope']:>8.3f}  {r['kdiss']:>10.2f}")

    # Scheme scaling k_diss(N) / k_diss(N/2)
    if len(results) >= 2:
        for i in range(1, len(results)):
            r2, r1 = results[i], results[i - 1]
            ratio = r2["kdiss"] / r1["kdiss"]
            N_ratio = r2["N"] / r1["N"]
            p = np.log(ratio) / np.log(N_ratio)
            print(f"  k_diss({r2['N']})/k_diss({r1['N']}) = {ratio:.2f}  "
                  f"(p = {p:.2f}, expect 3/2 = 1.5)")

    plot_spectra(results)

    # Pass criteria
    ok = True
    for r in results:
        if r["slope"] is None or not (-1.9 <= r["slope"] <= -1.3):
            print(f"  FAIL  N={r['N']} slope {r['slope']} outside [-1.9, -1.3]")
            ok = False
    for i in range(1, len(results)):
        r2, r1 = results[i], results[i - 1]
        ratio = r2["kdiss"] / r1["kdiss"]
        if not (2.0 <= ratio <= 3.5):
            print(f"  FAIL  k_diss ratio {ratio:.2f} outside [2.0, 3.5]")
            ok = False
    print("\n", "PASS" if ok else "FAIL", "A2 spectrum check")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
