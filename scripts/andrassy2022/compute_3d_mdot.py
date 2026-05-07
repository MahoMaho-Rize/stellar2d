#!/usr/bin/env python3
"""Compute Ṁ_e directly from Andrassy rprof X1 column, matching our 2D convention.

This bypasses Andrassy's published Ṁ_e normalization (unclear: total 3D or
per unit area?) by re-integrating from raw rprof data using exactly the same
integral we apply to our 2D data:

    M_e(t) = ∫_{conv layer} ⟨ρ(y)⟩ · ⟨X1(y)⟩ · dA_horizontal

For 3D, dA_horizontal = Lx · Lz = 2 · 2 = 4.
For 2D, dA_horizontal = Lx = 2 (our runs use Lx=2, no Lz).

Both we integrate ⟨ρ·X⟩(y) dy over y < y_ub − 0.1 buffer. Then rates.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose import parse_rprof
from timeseries import rprof_time_series

ROOT = Path("data/andrassy2022/1D-profiles")
CODES = ["FLASH", "MUSIC", "PPMSTAR", "PROMPI", "SLH"]
GAMMA = 5.0 / 3.0

# Andrassy domain: Lx=2, Lz=2, y∈[1,3]
LX_3D = 2.0
LZ_3D = 2.0


def compute_Me_from_rprof(rpath, buffer=0.1):
    rd = parse_rprof(rpath)
    if "X1" not in rd or "RHO" not in rd:
        return None
    y = rd["y"]; rho = rd["RHO"]; X1 = rd["X1"]; P = rd["P"]
    dy = float(y[1] - y[0])
    # y_ub: same A-rise threshold as our 2D diagnose
    A = P / rho**GAMMA
    mask_base = (y > 1.2) & (y < 1.8)
    A_conv = A[mask_base].mean()
    y_ub = y[-1]
    for j, yj in enumerate(y):
        if yj >= 2.0 and A[j] > A_conv * 1.001:
            y_ub = yj; break
    # Integrate ⟨ρ·X1⟩(y) dy over y < y_ub - buffer
    mask_conv = y < (y_ub - buffer)
    integrand = rho * X1 * dy
    M_e_column = integrand[mask_conv].sum()   # per unit horizontal area
    # For total 3D mass:
    M_e_total = M_e_column * LX_3D * LZ_3D
    return y_ub, M_e_column, M_e_total


def main():
    print("=== Andrassy 3D Ṁ_e recomputed from rprof X1 column ===")
    print()
    print(f"{'code':<10} {'Ṁ_e per area':<18} {'Ṁ_e total (×L_x·L_z)':<22} {'y_ub':<10}")

    for code in CODES:
        import glob
        files = sorted(glob.glob(f"{ROOT}/{code}/{code}-256/{code}-256-*.rprof"))
        ts, Me_col, Me_tot, yubs = [], [], [], []
        for f in files:
            try:
                rd = parse_rprof(Path(f))
                t = rd["t"]
                res = compute_Me_from_rprof(Path(f))
                if res is None: continue
                y_ub, M_col, M_tot = res
                ts.append(t); Me_col.append(M_col); Me_tot.append(M_tot)
                yubs.append(y_ub)
            except Exception as e:
                pass
        ts = np.array(ts); Me_col = np.array(Me_col); Me_tot = np.array(Me_tot)
        # Saturated Ṁ_e in [1000, 2000]
        m = (ts >= 1000) & (ts <= 2000)
        if m.sum() < 4:
            continue
        p_col = np.polyfit(ts[m], Me_col[m], 1)
        p_tot = np.polyfit(ts[m], Me_tot[m], 1)
        ybar = np.mean(yubs)
        print(f"{code:<10} {p_col[0]:<18.3e} {p_tot[0]:<22.3e} {ybar:<10.3f}")


if __name__ == "__main__":
    main()
