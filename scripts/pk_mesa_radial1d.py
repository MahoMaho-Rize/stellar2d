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
    """Parser for runs/<tag>/profile_NNNN.txt (emitted by radial1d).

    Auto-detects lean (`k r_face v_face rho P e_int`) vs rich
    (`... T kap gamma1 grada gradr L_face mixing_type conv_vel`) layout
    from the column-header comment line.
    """
    zones = []
    r_faces = []
    v_faces = []
    t = None
    step = None
    columns: list[str] = []
    with path.open() as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                if "t = " in s and "step = " in s:
                    parts = s.split()
                    for i, tok in enumerate(parts):
                        if tok == "t" and parts[i+1] == "=":
                            t = float(parts[i+2])
                        if tok == "step" and parts[i+1] == "=":
                            step = int(parts[i+2])
                elif s.startswith("# k") and "r_face" in s:
                    columns = s[1:].split()
                continue
            tokens = s.split()
            if tokens[3] == "-":
                r_faces.append(float(tokens[1]))
                v_faces.append(float(tokens[2]))
                continue
            k = int(tokens[0])
            r_faces.append(float(tokens[1]))
            v_faces.append(float(tokens[2]))
            zone: dict[str, float | int] = {}
            # columns list starts with "k r_face v_face rho P e_int ..."
            for ci, name in enumerate(columns[3:], start=3):
                if ci >= len(tokens):
                    break
                if name == "mixing_type":
                    zone[name] = int(tokens[ci])
                else:
                    zone[name] = float(tokens[ci])
            zones.append(zone)
    r = np.array(r_faces)
    v = np.array(v_faces)
    def arr(name: str) -> np.ndarray | None:
        if name not in zones[0]:
            return None
        return np.array([z[name] for z in zones])
    rho = arr("rho"); P = arr("P"); e = arr("e_int")
    assert rho is not None and P is not None and e is not None
    # Zone i sits between face i and face i+1. Compute zone enclosed mass.
    dv = (4.0/3.0) * np.pi * (r[1:]**3 - r[:-1]**3)
    dm = rho * dv
    m_enc_face = np.concatenate([[0.0], np.cumsum(dm)])
    r_center = 0.5 * (r[:-1] + r[1:])
    m_center = 0.5 * (m_enc_face[:-1] + m_enc_face[1:])
    out = {
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
    for extra in ("T", "kap", "gamma1", "grada", "gradr",
                  "L_face", "mixing_type", "conv_vel"):
        v_extra = arr(extra)
        if v_extra is not None:
            out[extra] = v_extra
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: pk_mesa_radial1d.py <MESA profile.data> <radial1d profile_*.txt>",
              file=sys.stderr)
        return 2
    mesa_path = Path(sys.argv[1])
    r1d_path = Path(sys.argv[2])

    mp = read_profile(mesa_path)
    mc = mp.to_cgs_profile()
    # MESA zone arrays — reverse to core→surface.
    m_mesa = mc["m_enc"][::-1]
    r_mesa = mc["r"][::-1]
    rho_m  = mc["rho"][::-1]
    T_m    = mc["T"][::-1]
    P_m    = mc["P"][::-1]
    frac_mesa = m_mesa / mc["M_star"]

    # Extended MESA columns (if requested in profile_columns.list)
    def mesa_col(name: str):
        if name in mp.zone_names:
            return mp.col(name)[::-1]   # reverse to core→surface
        return None
    opa_m    = mesa_col("opacity")
    grada_m  = mesa_col("grada")
    gradr_m  = mesa_col("gradr")
    gamma1_m = mesa_col("gamma1")
    L_m_sun  = mesa_col("luminosity")   # [Lsun]
    mix_m    = mesa_col("mixing_type")
    cvel_m   = mesa_col("conv_vel")

    r1d = read_radial1d_profile(r1d_path)
    frac = r1d["m"] / r1d["M_star"]

    rho_interp = np.interp(frac, frac_mesa, rho_m)
    T_interp   = np.interp(frac, frac_mesa, T_m)
    P_interp   = np.interp(frac, frac_mesa, P_m)
    r_interp   = np.interp(frac, frac_mesa, r_mesa)

    def interp_if(a):
        if a is None:
            return None
        return np.interp(frac, frac_mesa, a)
    opa_interp   = interp_if(opa_m)
    grada_interp = interp_if(grada_m)
    gradr_interp = interp_if(gradr_m)
    gamma1_interp = interp_if(gamma1_m)
    L_interp_sun  = interp_if(L_m_sun)
    cvel_interp   = interp_if(cvel_m)

    Lsun = mp.Lsun

    print(f"MESA : {mesa_path.name}  "
          f"Teff={mc['Teff']:.0f} K, L={mc['photosphere_L_Lsun']:.3f} Lsun, "
          f"n_zones={len(m_mesa)}")
    print(f"r1d  : {r1d_path.name}  "
          f"t={r1d['t']:.3e} s, step={r1d['step']}, "
          f"n_zones={len(r1d['rho'])}, |v|_max={np.max(np.abs(r1d['v_face'])):.3e}")
    print()

    # Build the wide per-fraction table, only showing columns both sides have
    have_rich = "T" in r1d
    header = ["m/M", "r/R (r1d)", "r/R (MESA)",
              "ρ r1d", "ρ MESA", "P r1d", "P MESA"]
    if have_rich:
        header += ["T r1d", "T MESA", "κ r1d", "κ MESA",
                   "∇_a r1d", "∇_a MESA", "∇_r r1d", "∇_r MESA",
                   "L r1d/Lsun", "L MESA/Lsun",
                   "mix r1d", "mix MESA"]
    print("Mass-coord PK at selected m/M fractions:")
    print(("{:>6} " + "{:>11} " * (len(header) - 1)).format(*header))
    for f in [0.005, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]:
        i = int(np.argmin(np.abs(frac - f)))
        row = [f"{frac[i]:6.4f}",
               f"{r1d['r'][i]/r1d['R_star']:11.4f}",
               f"{r_interp[i]/mc['R_star']:11.4f}",
               f"{r1d['rho'][i]:11.3e}", f"{rho_interp[i]:11.3e}",
               f"{r1d['P'][i]:11.3e}",   f"{P_interp[i]:11.3e}"]
        if have_rich:
            row += [
                f"{r1d['T'][i]:11.3e}",        f"{T_interp[i]:11.3e}",
                f"{r1d['kap'][i]:11.3e}",      f"{opa_interp[i]:11.3e}" if opa_interp is not None else "—",
                f"{r1d['grada'][i]:11.3e}",    f"{grada_interp[i]:11.3e}" if grada_interp is not None else "—",
                f"{r1d['gradr'][i]:11.3e}",    f"{gradr_interp[i]:11.3e}" if gradr_interp is not None else "—",
                f"{r1d['L_face'][i]/Lsun:11.3e}",
                f"{L_interp_sun[i]:11.3e}" if L_interp_sun is not None else "—",
                f"{int(r1d['mixing_type'][i]):>11d}",
                f"{int(mix_m[int(np.argmin(np.abs(frac_mesa - frac[i])))]):>11d}" if mix_m is not None else "—",
            ]
        print(" ".join(row))

    print()
    # Global relative-error medians
    def summarize(name: str, r1d_vals, mesa_vals):
        ok = mesa_vals > 0
        if not np.any(ok):
            return
        rel = np.abs(r1d_vals[ok] - mesa_vals[ok]) / mesa_vals[ok]
        print(f"  {name:>8} rel err: median {np.median(rel):.2e}, "
              f"p90 {np.percentile(rel, 90):.2e}, max {np.max(rel):.2e}")
    print("Relative error summary (cell-by-cell on radial1d grid):")
    summarize("ρ",   r1d["rho"], rho_interp)
    summarize("P",   r1d["P"],   P_interp)
    if have_rich:
        summarize("T",   r1d["T"],   T_interp)
        if opa_interp is not None:
            summarize("κ",   r1d["kap"], opa_interp)
        if gamma1_interp is not None:
            summarize("Γ₁",  r1d["gamma1"], gamma1_interp)
        if grada_interp is not None:
            summarize("∇_ad",r1d["grada"],  grada_interp)
        if cvel_interp is not None and "conv_vel" in r1d:
            # v_conv: MESA reports 0 in stable zones; comparing relative
            # error there is meaningless. Restrict to MESA conv-active cells.
            both_conv = (cvel_interp > 1.0) & (r1d["conv_vel"] > 1.0)
            if np.any(both_conv):
                rel = np.abs(r1d["conv_vel"][both_conv] - cvel_interp[both_conv]) \
                      / cvel_interp[both_conv]
                print(f"  {'v_conv':>8} rel err (both-conv zones only, "
                      f"n={both_conv.sum()}): "
                      f"median {np.median(rel):.2e}, "
                      f"p90 {np.percentile(rel, 90):.2e}")
                print(f"           MESA median: {np.median(cvel_interp[both_conv]):.2e} cm/s, "
                      f"r1d median: {np.median(r1d['conv_vel'][both_conv]):.2e} cm/s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
