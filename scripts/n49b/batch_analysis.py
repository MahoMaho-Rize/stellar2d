#!/usr/bin/env python3
"""Batch post-processing of Sukhbold+2018 pre-SN profiles.

Computes, for all ~1499 progenitors:
  - ZAMS mass, current mass, R_star
  - M_He_core (outermost zone with X(He4) > 0.1 going inward until X(H1) < 1e-3)
  - O-rich layer integrated composition: M_O, M_Ne, M_Mg, M_Si
  - Mg/Ne, Si/Ne, Si/O mass ratios
  - Compactness xi_{2.5}: (M/Msun)/(R(M)/1000km) at M=2.5 Msun
  - M_4: enclosed mass where specific entropy reaches 4 k_B/baryon
  - mu_4: dm/(M_sun) / (dr/1000km) at M_4

Output: CSV file with one row per progenitor, for downstream figure scripts.

Runs in ~60 s with Python multiprocessing.

Usage:
    pixi run python scripts/n49b/batch_analysis.py \\
        --data-dir ~/data/sukhbold_2018/mdotone \\
        --out data/n49b_progenitor_catalog.csv
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sukhbold_reader import read_one, PresnProfile, MSUN_CGS


RSUN_CGS = 6.96e10
KB = 1.380649e-16  # erg/K
NA = 6.02214076e23


def he_core_mass(prof: PresnProfile) -> float:
    """Helium core mass (M_sun): mass interior to the outermost zone where
    X(H1) > 0.1 (i.e. the hydrogen envelope base).  If no H envelope present,
    M_He_core = total M_star.
    """
    X_H = prof.X("H1")
    m_enc = prof.m_enc_msun
    # search outward for first zone with X_H > 0.1
    idx = np.where(X_H > 0.1)[0]
    if len(idx) == 0:
        return m_enc[-1]
    return m_enc[idx[0] - 1] if idx[0] > 0 else 0.0


def o_rich_composition(prof: PresnProfile, X_O_thresh: float = 0.4) -> dict:
    """Integrate element masses over the O-rich layer (X_O > threshold).

    Returns dict with M_O, M_Ne, M_Mg, M_Si (all in Msun) and ratios.
    """
    mask = prof.X("O16") > X_O_thresh
    out = {"M_O": 0.0, "M_Ne": 0.0, "M_Mg": 0.0, "M_Si": 0.0,
           "MgNe": np.nan, "SiNe": np.nan, "SiO": np.nan,
           "m_O_inner": np.nan, "m_O_outer": np.nan}
    if not mask.any():
        return out
    dm = prof.dm
    out["M_O"]  = (prof.X("O16")  * dm)[mask].sum() / MSUN_CGS
    out["M_Ne"] = (prof.X("Ne20") * dm)[mask].sum() / MSUN_CGS
    out["M_Mg"] = (prof.X("Mg24") * dm)[mask].sum() / MSUN_CGS
    out["M_Si"] = (prof.X("Si28") * dm)[mask].sum() / MSUN_CGS
    out["MgNe"] = out["M_Mg"] / max(out["M_Ne"], 1e-30)
    out["SiNe"] = out["M_Si"] / max(out["M_Ne"], 1e-30)
    out["SiO"]  = out["M_Si"] / max(out["M_O"],  1e-30)
    m_enc = prof.m_enc_msun
    out["m_O_inner"] = m_enc[mask][0]
    out["m_O_outer"] = m_enc[mask][-1]
    return out


def compactness(prof: PresnProfile, M_target: float = 2.5) -> float:
    """ξ_{M_target} = (M_target / Msun) / (R(M_target) / 1000 km).

    R(M) is the radius at the zone enclosing M_target Msun.  Return NaN if
    M_star < M_target.
    """
    m = prof.m_enc_msun
    if m[-1] < M_target:
        return np.nan
    idx = np.searchsorted(m, M_target)
    # linear interp in (m, r)
    r0, r1 = prof.r[idx - 1], prof.r[idx]
    m0, m1 = m[idx - 1], m[idx]
    r_at = r0 + (r1 - r0) * (M_target - m0) / (m1 - m0)
    return M_target / (r_at / 1e8)  # R in 1000 km = 1e8 cm


def m4_mu4(prof: PresnProfile, s_crit: float = 4.0) -> tuple[float, float]:
    """M_4 = enclosed mass (Msun) where specific entropy first exceeds s_crit
    (in k_B / baryon).  μ_4 = dM/(Msun) / dR/(1000 km) evaluated in a 0.3 Msun
    window outside M_4 (Ertl+2016 definition).  NaN if not reached.
    """
    s = prof.s  # k_B / baryon already
    m = prof.m_enc_msun
    idx = np.where(s >= s_crit)[0]
    if len(idx) == 0:
        return np.nan, np.nan
    i = idx[0]
    if i == 0:
        M4 = m[0]
    else:
        # linear interp in (s, m)
        s0, s1 = s[i - 1], s[i]
        m0, m1 = m[i - 1], m[i]
        M4 = m0 + (m1 - m0) * (s_crit - s0) / (s1 - s0)

    # mu_4: over an 0.3 Msun window outside M4
    M4_high = M4 + 0.3
    if m[-1] < M4_high:
        return M4, np.nan
    # find r at M4 and M4_high
    def r_at(M):
        j = np.searchsorted(m, M)
        if j == 0:
            return prof.r[0]
        if j >= len(m):
            return prof.r[-1]
        r0, r1 = prof.r[j - 1], prof.r[j]
        m0, m1 = m[j - 1], m[j]
        return r0 + (r1 - r0) * (M - m0) / (m1 - m0)

    dM = 0.3
    dR = (r_at(M4_high) - r_at(M4)) / 1e8  # in 1000 km
    if dR <= 0:
        return M4, np.nan
    return M4, dM / dR


def process_one(path: Path) -> dict:
    prof = read_one(path)
    comp = o_rich_composition(prof)
    xi25 = compactness(prof, 2.5)
    M4, mu4 = m4_mu4(prof, 4.0)
    return {
        "zams_mass": prof.zams_mass,
        "M_star":    prof.M_star,
        "nz":        prof.nz,
        "R_star":    prof.r[-1],
        "M_He":      he_core_mass(prof),
        "xi25":      xi25,
        "M4":        M4,
        "mu4":       mu4,
        **comp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="~/data/sukhbold_2018/mdotone")
    ap.add_argument("--out", default="data/n49b_progenitor_catalog.csv")
    ap.add_argument("--nproc", type=int, default=0,
                    help="0 = use cpu_count()")
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: process only first N files")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paths = sorted(data_dir.glob("*.dat"))
    if args.limit:
        paths = paths[: args.limit]
    print(f"Processing {len(paths)} profiles from {data_dir}")

    nproc = args.nproc or mp.cpu_count()
    with mp.Pool(nproc) as pool:
        rows = pool.map(process_one, paths)

    # Write CSV
    import csv
    cols = list(rows[0].keys())
    with out_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {out_path}")

    # Quick sanity summary
    import numpy as np
    zams = np.array([r["zams_mass"] for r in rows])
    mgne = np.array([r["MgNe"] for r in rows])
    print("\nZAMS mass bin   N    frac(Mg/Ne>1)")
    bins = [(12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18),
            (18, 19), (19, 22), (22, 23), (23, 24), (24, 27)]
    for lo, hi in bins:
        mask = (zams >= lo) & (zams < hi)
        n = mask.sum()
        if n == 0:
            frac = 0
        else:
            frac = np.sum((mgne[mask] > 1) & np.isfinite(mgne[mask])) / n
        print(f"  {lo:2d}-{hi:2d} Msun   {n:4d}   {frac:.2f}")


if __name__ == "__main__":
    main()
