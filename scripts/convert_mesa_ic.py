#!/usr/bin/env python3
"""Convert a MESA `LOGS/profile*.data` into a radial1d-friendly ASCII IC.

Output format (human-inspectable, consumed by Radial1DSolver::init_from_mesa):

    # MESA-derived radial1d IC
    # M_star_g   <double>
    # R_star_cm  <double>
    # n_zones    <int>      -- number of MESA zones (surface → core)
    # columns: m_enc_g r_cm rho_cgs T_K P_cgs X Y Z
    <m_enc> <r> <rho> <T> <P> <X> <Y> <Z>
    ...

Zones are kept in MESA's native ordering (surface first). The consumer
handles interpolation to its own Lagrangian zone count.

Usage:
    python3 scripts/convert_mesa_ic.py LOGS/profile5.data out.ic
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mesa_profile import read_profile


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: convert_mesa_ic.py <LOGS/profileN.data> <out.ic>",
              file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    prof = read_profile(src)
    c = prof.to_cgs_profile()
    n = len(c["r"])
    with dst.open("w") as f:
        f.write("# MESA-derived radial1d IC\n")
        f.write(f"# source      {src.resolve()}\n")
        f.write(f"# M_star_g    {c['M_star']:.15e}\n")
        f.write(f"# R_star_cm   {c['R_star']:.15e}\n")
        f.write(f"# n_zones     {n}\n")
        f.write(f"# Teff_K      {c['Teff']:.6e}\n")
        f.write(f"# L_Lsun      {c['photosphere_L_Lsun']:.6e}\n")
        f.write("# columns: m_enc_g r_cm rho_cgs T_K P_cgs X Y Z\n")
        for i in range(n):
            f.write(
                f"{c['m_enc'][i]:.15e} {c['r'][i]:.15e} "
                f"{c['rho'][i]:.15e} {c['T'][i]:.15e} {c['P'][i]:.15e} "
                f"{c['X'][i]:.15e} {c['Y'][i]:.15e} {c['Z'][i]:.15e}\n"
            )
    print(f"wrote {dst} ({n} zones, M={c['M_star']:.3e} g, R={c['R_star']:.3e} cm)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
