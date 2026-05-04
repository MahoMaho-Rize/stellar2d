#!/usr/bin/env python3
"""Sukhbold+2018 .dat → radial1d IC file converter.

Output format (13-species extension of convert_mesa_ic.py):

    # M_star_g <M_star_remaining after mass-cut, g>
    # R_star_cm <R_out, cm>
    # M_inner_g <M_cut, g>           ← gravitational mass inside r_in (proto-NS)
    # r_inner_cm <r_at_mass_cut, cm>
    # n_zones <nz_after_cut>
    # n_species 13
    # columns: m_enc_g r_cm rho_gcc T_K P_erg_cc X_He4 X_C12 X_O16 X_Ne20 X_Mg24 X_Si28 X_S32 X_Ar36 X_Ca40 X_Ti44 X_Cr48 X_Fe52 X_Ni56
    <surface-first rows, as in convert_mesa_ic.py>

13-species mapping from Sukhbold's 19 element columns:
    He4  ← He3 + He4                            (3He folded into 4He; trivial mass)
    C12  ← C12 + N14                            (N14 → C12 at α-freeze under detail balance)
    O16  ← O16
    Ne20 ← Ne20
    Mg24 ← Mg24
    Si28 ← Si28
    S32  ← S32
    Ar36 ← Ar36
    Ca40 ← Ca40
    Ti44 ← Ti44
    Cr48 ← Cr48
    Fe52 ← Fe52 + Fe54 + Fe56                   (iron-peak bulk lumped)
    Ni56 ← Ni56
    (dropped: H1 — Sukhbold presn has X_H~0 in the C/O core; neutrons — also ~0 outside NS)

Any residual neutron / hydrogen mass is folded into He4 so that Σ X_i = 1 in the
output.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sukhbold_reader import read_one, MSUN_CGS  # noqa: E402


ALPHA13 = [
    "He4", "C12", "O16", "Ne20", "Mg24", "Si28", "S32",
    "Ar36", "Ca40", "Ti44", "Cr48", "Fe52", "Ni56",
]


def fold_species(prof) -> np.ndarray:
    """Fold the 19-element Sukhbold composition into 13 α-chain mass fractions.

    Returns X[nz, 13] in alpha_net::Species order.  Rows sum to 1 by construction.
    """
    nz = prof.nz
    X = np.zeros((nz, 13), dtype=np.float64)
    X[:, 0]  = prof.X("He3")  + prof.X("He4")
    X[:, 1]  = prof.X("C12")  + prof.X("N14")
    X[:, 2]  = prof.X("O16")
    X[:, 3]  = prof.X("Ne20")
    X[:, 4]  = prof.X("Mg24")
    X[:, 5]  = prof.X("Si28")
    X[:, 6]  = prof.X("S32")
    X[:, 7]  = prof.X("Ar36")
    X[:, 8]  = prof.X("Ca40")
    X[:, 9]  = prof.X("Ti44")
    X[:, 10] = prof.X("Cr48")
    X[:, 11] = prof.X("Fe52") + prof.X("Fe54") + prof.X("Fe56")
    X[:, 12] = prof.X("Ni56")
    residual = prof.X("H1") + prof.X("neutrons")
    X[:, 0] += residual
    s = X.sum(axis=1, keepdims=True)
    X /= np.where(s > 0, s, 1.0)
    return X


def apply_mass_cut(prof, mass_cut_msun: float):
    """Return (m_enc_g, r_cm, rho, T, P, X13) for zones above M_cut.

    Zones are ordered surface-first in the output (matching convert_mesa_ic.py),
    but we select by enclosed mass which is computed from Sukhbold's core-first
    dm array.  The zone whose *outer* face crosses M_cut is kept (its inner
    face becomes the new inner boundary, shifted to exactly r(M_cut)).
    """
    m_enc = prof.m_enc
    M_cut = mass_cut_msun * MSUN_CGS
    if M_cut >= m_enc[-1]:
        raise ValueError(f"mass_cut {mass_cut_msun} Msun >= M_star")
    # Zones with outer enclosed mass > M_cut survive.  Sukhbold rows are
    # core-first, so we pick first index where m_enc > M_cut.
    k0 = int(np.searchsorted(m_enc, M_cut, side="right"))
    if k0 == 0:
        raise ValueError("mass cut below innermost zone — pick larger M_cut")
    # Recompute m_enc relative to the raw dm so that the first surviving
    # zone starts at M_cut (so r_inner = r at M_cut).  Interpolate r(M_cut)
    # between zone k0-1 and k0 by linear interpolation in m_enc.
    r_all = prof.r
    rho_all = prof.rho
    T_all = prof.T
    P_all = prof.P
    X_all = fold_species(prof)

    m_prev = m_enc[k0 - 1] if k0 > 0 else 0.0
    r_prev = r_all[k0 - 1] if k0 > 0 else 0.0
    m_next = m_enc[k0]
    r_next = r_all[k0]
    frac = (M_cut - m_prev) / (m_next - m_prev)
    r_cut = r_prev + frac * (r_next - r_prev)

    # Surviving zones: k0..nz-1.  In the output we need m_enc, r, rho, T, P, X
    # row-by-row with surface-first ordering.  First zone has inner face at
    # r_cut and enclosed mass at the outer face = m_enc[k0]; subsequent
    # zones just keep their Sukhbold values.
    idx = np.arange(k0, prof.nz)
    m_out = m_enc[idx]     # enclosed mass at zone's outer face
    r_out = r_all[idx]
    rho   = rho_all[idx]
    T     = T_all[idx]
    P     = P_all[idx]
    X13   = X_all[idx]

    # Reverse to surface-first for the IC file.
    return {
        "M_star": float(m_out[-1]),
        "R_star": float(r_out[-1]),
        "M_inner": float(M_cut),
        "r_inner": float(r_cut),
        "m_enc": m_out[::-1].copy(),
        "r":     r_out[::-1].copy(),
        "rho":   rho[::-1].copy(),
        "T":     T[::-1].copy(),
        "P":     P[::-1].copy(),
        "X":     X13[::-1].copy(),
    }


def write_ic(out_path: Path, ic: dict, header_note: str = "") -> None:
    nz = len(ic["m_enc"])
    with out_path.open("w") as f:
        f.write(f"# Sukhbold+2018 pre-SN IC for radial1d (13-species)\n")
        if header_note:
            f.write(f"# {header_note}\n")
        f.write(f"# M_star_g     {ic['M_star']:.10e}\n")
        f.write(f"# R_star_cm    {ic['R_star']:.10e}\n")
        f.write(f"# M_inner_g    {ic['M_inner']:.10e}\n")
        f.write(f"# r_inner_cm   {ic['r_inner']:.10e}\n")
        f.write(f"# n_zones      {nz}\n")
        f.write(f"# n_species    13\n")
        f.write("# columns: m_enc_g r_cm rho_gcc T_K P_erg_cc "
                + " ".join(f"X_{sp}" for sp in ALPHA13) + "\n")
        for k in range(nz):
            m = ic["m_enc"][k]
            r = ic["r"][k]
            rho = ic["rho"][k]
            T = ic["T"][k]
            P = ic["P"][k]
            X = ic["X"][k]
            row = [m, r, rho, T, P] + list(X)
            f.write(" ".join(f"{v:.10e}" for v in row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True,
                    help="Sukhbold .dat profile (e.g. 15.90.dat)")
    ap.add_argument("--out", required=True,
                    help="Output IC file for radial1d")
    ap.add_argument("--mass-cut", type=float, default=1.6,
                    help="Mass cut in Msun (default 1.6; typical 1.4-1.8)")
    args = ap.parse_args()

    prof = read_one(Path(args.in_path))
    ic = apply_mass_cut(prof, args.mass_cut)
    note = (f"ZAMS={prof.zams_mass:.2f} Msun, M_total={prof.M_star:.2f} Msun, "
            f"mass_cut={args.mass_cut:.3f} Msun")
    write_ic(Path(args.out), ic, header_note=note)
    print(f"wrote {args.out}")
    print(f"  {note}")
    print(f"  kept nz={len(ic['m_enc'])} zones")
    print(f"  r_inner = {ic['r_inner']:.3e} cm, "
          f"M_ejecta = {(ic['M_star']-ic['M_inner'])/MSUN_CGS:.2f} Msun")
    print(f"  X (outermost zone) = {ic['X'][0]}")


if __name__ == "__main__":
    main()
