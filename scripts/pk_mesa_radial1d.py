#!/usr/bin/env python3
"""Side-by-side PK: MESA LOGS/profileN.data vs the corresponding radial1d
profile_*.txt (any step). Both are expressed on (M_enc / M_star, r, ρ, T, P)
so we can watch how much the two solvers drift after N τ_dyn of HSE hold.

Usage:
    python3 scripts/pk_mesa_radial1d.py \
        /tmp/mesa_work_1Msol/LOGS/profile5.data \
        runs/lane_emden_128x64_<timestamp>/profile_0019.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mesa_profile import read_profile


def read_radial1d_profile(path: Path):
    """Parser for runs/<tag>/profile_NNNN.txt (emitted by radial1d)."""
    # Format: header comments, then zones `k r_face v_face rho P e_int [X Y]`
    # with a trailing "nz r_face v_face - - -" row for the outer face.
    zones = []
    r_faces = []
    v_faces = []
    t = None
    step = None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # "# t = ... step = N"
                if "t = " in line and "step = " in line:
                    # parse tokens
                    parts = line.split()
                    for i, tok in enumerate(parts):
                        if tok == "t" and parts[i+1] == "=":
                            t = float(parts[i+2])
                        if tok == "step" and parts[i+1] == "=":
                            step = int(parts[i+2])
                continue
            tokens = line.split()
            if tokens[3] == "-":
                # trailing face row
                r_faces.append(float(tokens[1]))
                v_faces.append(float(tokens[2]))
                continue
            k = int(tokens[0])
            r_faces.append(float(tokens[1]))
            v_faces.append(float(tokens[2]))
            zones.append({
                "rho": float(tokens[3]),
                "P":   float(tokens[4]),
                "e":   float(tokens[5]),
            })
    r = np.array(r_faces)
    v = np.array(v_faces)
    rho = np.array([z["rho"] for z in zones])
    P   = np.array([z["P"]   for z in zones])
    e   = np.array([z["e"]   for z in zones])
    # Zone i sits between face i and face i+1. Compute zone enclosed mass.
    dv = (4.0/3.0) * np.pi * (r[1:]**3 - r[:-1]**3)
    dm = rho * dv
    m_enc_face = np.concatenate([[0.0], np.cumsum(dm)])
    r_center = 0.5 * (r[:-1] + r[1:])
    m_center = 0.5 * (m_enc_face[:-1] + m_enc_face[1:])
    return {
        "t": t,
        "step": step,
        "r_face": r,
        "v_face": v,
        "r": r_center,
        "m_enc_face": m_enc_face,
        "m": m_center,
        "rho": rho,
        "P": P,
        "e": e,
        "M_star": float(m_enc_face[-1]),
        "R_star": float(r[-1]),
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: pk_mesa_radial1d.py <MESA profile.data> <radial1d profile_*.txt>",
              file=sys.stderr)
        return 2
    mesa_path = Path(sys.argv[1])
    r1d_path = Path(sys.argv[2])

    mp = read_profile(mesa_path)
    mc = mp.to_cgs_profile()
    # Flip MESA to core→surface to match radial1d ordering.
    m_mesa = mc["m_enc"][::-1]
    r_mesa = mc["r"][::-1]
    rho_m  = mc["rho"][::-1]
    T_m    = mc["T"][::-1]
    P_m    = mc["P"][::-1]

    r1d = read_radial1d_profile(r1d_path)
    frac = r1d["m"] / r1d["M_star"]

    # Interpolate MESA onto radial1d's m/M grid for point-by-point diff.
    frac_mesa = m_mesa / mc["M_star"]
    # Drop the last (surface) point to keep the interp one-sided clean.
    rho_interp = np.interp(frac, frac_mesa, rho_m)
    T_interp   = np.interp(frac, frac_mesa, T_m)
    P_interp   = np.interp(frac, frac_mesa, P_m)
    r_interp   = np.interp(frac, frac_mesa, r_mesa)

    # We don't emit T from radial1d — reconstruct from ideal-gas floor (needs
    # actual EOS for Helm; here we print the P/ρ proxy as a sanity tag).
    T_proxy = r1d["P"] / r1d["rho"] / 8.314e7 * 1.28   # ideal w/ μ=1.28

    print(f"MESA : {mesa_path.name}  "
          f"Teff={mc['Teff']:.0f} K, L={mc['photosphere_L_Lsun']:.3f} Lsun, "
          f"n_zones={len(m_mesa)}")
    print(f"r1d  : {r1d_path.name}  "
          f"t={r1d['t']:.3e} s, step={r1d['step']}, "
          f"n_zones={len(r1d['rho'])}, |v|_max={np.max(np.abs(r1d['v_face'])):.3e}")
    print()
    print("Mass-coord PK at selected m/M fractions:")
    print(f"{'m/M':>8} {'r/R (r1d)':>12} {'r/R (MESA)':>12} "
          f"{'ρ (r1d)':>12} {'ρ (MESA)':>12} "
          f"{'P (r1d)':>12} {'P (MESA)':>12} "
          f"{'T_proxy':>12} {'T (MESA)':>12}")
    for f in [0.005, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]:
        i = int(np.argmin(np.abs(frac - f)))
        print(f"{frac[i]:8.4f} "
              f"{r1d['r'][i]/r1d['R_star']:12.4f} "
              f"{r_interp[i]/mc['R_star']:12.4f} "
              f"{r1d['rho'][i]:12.3e} {rho_interp[i]:12.3e} "
              f"{r1d['P'][i]:12.3e} {P_interp[i]:12.3e} "
              f"{T_proxy[i]:12.3e} {T_interp[i]:12.3e}")

    print()
    # Global diagnostics
    rho_err = np.abs(r1d["rho"] - rho_interp) / rho_interp
    P_err   = np.abs(r1d["P"] - P_interp) / P_interp
    print(f"ρ relative error: median {np.median(rho_err):.2e}, "
          f"max {np.max(rho_err):.2e}")
    print(f"P relative error: median {np.median(P_err):.2e}, "
          f"max {np.max(P_err):.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
