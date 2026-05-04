#!/usr/bin/env python3
"""Phase D Day 5 — 4-progenitor × 3 mass_cuts × 4 E_SN sensitivity sweep.

For each (zams, mass_cut, E_SN):
  1. convert_sukhbold_ic.py writes IC
  2. stellar2d runs implicit Newton SN explosion to t=10s
  3. parse final profile, compute Mg/Ne, Si/Ne, Ni56 mass, ejecta KE
  4. append row to data/n49b_postSN/phaseD_sweep.csv

Output:
  data/n49b_postSN/phaseD_sweep.csv   — tabular results for all 48 runs
  data/n49b_postSN/postSN_<zams>_mc<mc>_E<E>.npz  — per-run profile dump

Record EVERY output quantity, not just Mg/Ne, so downstream analysis can
inspect multi-dimensional consistency with the paper (per feedback_baseline
_matching_antipattern.md — do not tune on a single scalar).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sukhbold_reader import read_one, MSUN_CGS  # noqa: E402

ALPHA_NAMES = ["He", "C", "O", "Ne", "Mg", "Si", "S",
               "Ar", "Ca", "Ti", "Cr", "Fe", "Ni"]

# Paper reference values (Sato+2024 Fig 7) — target Mg/Ne in post-SN O-rich region
PAPER_MGNE = {
    "12.02": 0.30,
    "12.75": 0.75,
    "15.28": 0.15,
    "15.90": 1.25,
}


def convert_ic(zams: str, mass_cut: float, out_path: Path, data_dir: Path) -> None:
    src = data_dir / f"{zams}.dat"
    cmd = [
        "python3", "scripts/n49b/convert_sukhbold_ic.py",
        "--in", str(src),
        "--out", str(out_path),
        "--mass-cut", f"{mass_cut}",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def run_sn(ic_path: Path, bomb_E: float, run_dir: Path, nr: int = 128,
           tend: float = 10.0, timeout: float = 600) -> Path:
    """Runs implicit radial1d SN explosion, returns path to final profile_*.txt."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "./build/stellar2d",
        "--solver", "radial1d", "--test", "sukhbold_bomb",
        "--nr", str(nr),
        "--gamma", "1.66667", "--G", "6.674e-8",
        "--eos", "helmholtz",
        "--ic-sukhbold", str(ic_path),
        "--bomb-E", f"{bomb_E:.3e}", "--bomb-dm", "0.1",
        "--implicit", "--no-rhse", "--jfnk-autodiff", "--precond-tridiag",
        "--newton-tol", "1e-4",
        "--dt-implicit-scale", "10.0",
        "--tend", f"{tend}",
        "--output-interval", "1",
    ]
    env = os.environ.copy()
    # Pin output dir via a sentinel file (stellar2d uses runs/<testname>_<timestamp>).
    # We don't control the name; instead we move the latest one after the run.
    before = set(Path("runs").glob("sukhbold_bomb_*"))
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
    after = set(Path("runs").glob("sukhbold_bomb_*"))
    new_dirs = after - before
    if not new_dirs:
        raise RuntimeError("no new run dir produced")
    latest = sorted(new_dirs, key=lambda p: p.stat().st_mtime)[-1]
    profiles = sorted(latest.glob("profile_*.txt"))
    if not profiles:
        raise RuntimeError(f"no profile in {latest}")
    # Rename the run dir to our deterministic name
    target = run_dir / "run"
    if target.exists():
        import shutil
        shutil.rmtree(target)
    latest.rename(target)
    return target / profiles[-1].name


def parse_profile(profile_path: Path) -> dict:
    """Return dict with arrays: r_face, v_face, rho, P, e, X (nz, 13)."""
    rows = []
    t = None
    with profile_path.open() as f:
        for ln in f:
            if ln.startswith("# t ="):
                t = float(ln.split("=")[1].split()[0])
            if ln.startswith("#") or not ln.strip():
                continue
            toks = ln.split()
            if toks[3] == "-":
                continue
            rows.append([float(x) for x in toks])
    arr = np.asarray(rows)
    return {
        "t": t,
        "k": arr[:, 0].astype(int),
        "r": arr[:, 1],
        "v": arr[:, 2],
        "rho": arr[:, 3],
        "P": arr[:, 4],
        "e": arr[:, 5],
        "X": arr[:, 6:19],
    }


def parse_ic_header(ic_path: Path) -> dict:
    meta = {}
    with ic_path.open() as f:
        for ln in f:
            if not ln.startswith("#"):
                break
            if "M_star_g" in ln: meta["M_star"] = float(ln.split()[-1])
            elif "M_inner_g" in ln: meta["M_inner"] = float(ln.split()[-1])
            elif "R_star_cm" in ln: meta["R_star"] = float(ln.split()[-1])
            elif "r_inner_cm" in ln: meta["r_inner"] = float(ln.split()[-1])
    return meta


def ic_X_O_on_m(ic_path: Path, m_target: np.ndarray) -> np.ndarray:
    """Interpolate pre-SN X_O16 at the target enclosed-mass values."""
    rows = []
    with ic_path.open() as f:
        for ln in f:
            if ln.startswith("#") or not ln.strip():
                continue
            rows.append([float(x) for x in ln.split()])
    ic = np.asarray(rows)[::-1]  # core-first
    ic_m = ic[:, 0]
    ic_XO = ic[:, 5 + 2]
    return np.interp(m_target, ic_m, ic_XO)


def analyze(profile: dict, ic_path: Path, bomb_E: float) -> dict:
    meta = parse_ic_header(ic_path)
    M_star = meta["M_star"]
    M_inner = meta["M_inner"]
    nz = len(profile["k"])
    dm = (M_star - M_inner) / nz
    m_center = M_inner + (np.arange(nz) + 0.5) * dm

    X_O_pre = ic_X_O_on_m(ic_path, m_center)
    mask = X_O_pre > 0.4

    X = profile["X"]
    names = ALPHA_NAMES
    # Total masses by species (all zones)
    totals = {f"M_{n}_total": float((X[:, i] * dm).sum() / MSUN_CGS)
              for i, n in enumerate(names)}
    # Masses in pre-SN O-rich region
    orich = {f"M_{n}_orich": float((X[:, i] * dm)[mask].sum() / MSUN_CGS)
             for i, n in enumerate(names)}
    m_Ne = orich["M_Ne_orich"]
    m_Mg = orich["M_Mg_orich"]
    m_O  = orich["M_O_orich"]
    m_Si = orich["M_Si_orich"]

    # Ejecta kinetic energy + internal energy (integrate over all zones)
    v_cell = 0.5 * (profile["v"][:-1] + profile["v"][1:]) if len(profile["v"]) > nz \
             else profile["v"]
    KE = float(0.5 * (dm * v_cell ** 2).sum()) if v_cell.shape[0] == nz else np.nan
    IE = float((dm * profile["e"]).sum())

    out = {
        "nz_orich": int(mask.sum()),
        "M_orich_total": float(mask.sum() * dm / MSUN_CGS),
        "Mg_Ne": m_Mg / max(m_Ne, 1e-30),
        "Si_Ne": m_Si / max(m_Ne, 1e-30),
        "M_ejecta_msun": (M_star - M_inner) / MSUN_CGS,
        "KE_erg": KE,
        "IE_erg": IE,
        "E_bomb_erg": bomb_E,
        "t_final": profile["t"],
    }
    out.update(totals)
    out.update(orich)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="~/data/sukhbold_2018/mdotone")
    ap.add_argument("--out-dir", default="data/n49b_postSN")
    ap.add_argument("--zams", nargs="+",
                    default=["12.02", "12.75", "15.28", "15.90"])
    ap.add_argument("--mass-cuts", nargs="+", type=float,
                    default=[1.4, 1.6, 1.8])
    ap.add_argument("--E-SN", nargs="+", type=float,
                    default=[0.5e51, 1.0e51, 2.0e51, 4.0e51])
    ap.add_argument("--nr", type=int, default=128)
    ap.add_argument("--tend", type=float, default=10.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "phaseD_sweep.csv"
    json_path = out_dir / "phaseD_sweep.json"
    rows = []

    total = len(args.zams) * len(args.mass_cuts) * len(args.E_SN)
    i = 0
    t_start = time.time()
    for zams in args.zams:
        for mc in args.mass_cuts:
            ic_path = Path("/tmp") / f"sukhbold_{zams}_mc{mc:.1f}.ic"
            convert_ic(zams, mc, ic_path, data_dir)
            for E in args.E_SN:
                i += 1
                label = f"{zams}_mc{mc:.1f}_E{E:.1e}"
                run_dir = out_dir / f"runs/{label}"
                print(f"[{i}/{total}] {label} ...", end=" ", flush=True)
                if args.dry_run:
                    print("(dry)")
                    continue
                t0 = time.time()
                try:
                    prof_path = run_sn(ic_path, E, run_dir, nr=args.nr, tend=args.tend)
                    prof = parse_profile(prof_path)
                    res = analyze(prof, ic_path, E)
                    res["zams"] = zams
                    res["mass_cut"] = mc
                    res["nr"] = args.nr
                    res["wall_s"] = time.time() - t0
                    paper = PAPER_MGNE.get(zams)
                    if paper is not None:
                        res["Mg_Ne_paper"] = paper
                        res["Mg_Ne_ratio_to_paper"] = res["Mg_Ne"] / paper
                    rows.append(res)
                    print(f"Mg/Ne={res['Mg_Ne']:.3f}  "
                          f"(paper {paper}) {res['wall_s']:.0f}s")
                    # Also save npz
                    np.savez(out_dir / f"postSN_{label}.npz",
                             r=prof["r"], v=prof["v"], rho=prof["rho"],
                             P=prof["P"], e=prof["e"], X=prof["X"],
                             **{k: v for k, v in res.items()
                                if isinstance(v, (int, float, str))})
                except subprocess.TimeoutExpired:
                    print("TIMEOUT")
                    rows.append({"zams": zams, "mass_cut": mc,
                                 "E_bomb_erg": E, "status": "timeout"})
                except Exception as e:
                    print(f"FAIL: {e}")
                    rows.append({"zams": zams, "mass_cut": mc,
                                 "E_bomb_erg": E, "status": f"fail: {e}"})

    wall = time.time() - t_start
    print(f"\nTotal wall time: {wall:.0f}s ({wall/60:.1f} min)")

    # Write CSV (union of all keys)
    keys = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2, default=float)
    print(f"wrote {csv_path} ({len(rows)} rows)")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
