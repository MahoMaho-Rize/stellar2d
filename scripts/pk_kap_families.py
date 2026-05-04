#!/usr/bin/env python3
"""Opacity-family PK: plot κ_R(ρ, T) for a solar-like profile across all
Rosseland-mean families MESA ships.

Draws a ρ(T) stellar-like track that passes through photosphere, envelope,
and core regimes, evaluates every family's κ at those points (trilinear,
bisect-on-logT), and prints a wide table so opacity differences between
OPAL/OP/OPLIB and across metal mixtures jump out immediately.

Usage:
    python3 scripts/pk_kap_families.py [--Z 0.02] [--X 0.7] [--csv out.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from convert_mesa_kap import read_kap_bin  # noqa: E402


def trilin(d: dict, X_val: float, rho: float, T: float) -> float:
    """Trilinear κ_R lookup matching src/physics/opacity_table.cuh."""
    X = d["X"]; lt = d["logT"]; lr = d["logR"]; lk = d["log_kap"]
    logT = float(np.log10(T))
    logR = float(np.log10(rho) - 3.0 * logT + 18.0)

    iX = int(np.searchsorted(X, X_val, side="right") - 1)
    iX = max(0, min(len(X) - 2, iX))
    xt = (X_val - X[iX]) / (X[iX + 1] - X[iX])

    jT = int(np.searchsorted(lt, logT, side="right") - 1)
    jT = max(0, min(len(lt) - 2, jT))
    tt = (logT - lt[jT]) / (lt[jT + 1] - lt[jT])

    dR = (lr[-1] - lr[0]) / (len(lr) - 1)
    iR = int((logR - lr[0]) / dR)
    iR = max(0, min(len(lr) - 2, iR))
    rt = (logR - lr[iR]) / (lr[iR + 1] - lr[iR])

    xt = float(np.clip(xt, 0.0, 1.0))
    tt = float(np.clip(tt, 0.0, 1.0))
    rt = float(np.clip(rt, 0.0, 1.0))

    acc = 0.0
    for dx, wx in ((0, 1 - xt), (1, xt)):
        for dT, wT in ((0, 1 - tt), (1, tt)):
            for dR_, wR in ((0, 1 - rt), (1, rt)):
                acc += wx * wT * wR * lk[iX + dx, jT + dT, iR + dR_]
    return 10.0 ** acc


# Solar-like stellar track (a coarse diagonal through the ρ-T plane that
# covers every regime we care about).
TRACK = [
    ("dust / molecular",        1.0e-14,    800.0),
    ("photosphere",             1.0e-7,    5800.0),
    ("H ionisation zone",       1.0e-4,    1.2e4),
    ("He-II ionisation",        1.0e-3,    5.0e4),
    ("outer envelope",          1.0e-2,    1.0e6),
    ("deep envelope",           1.0e0,     3.0e6),
    ("core",                    1.5e2,     1.5e7),
    ("degenerate WD-like",      1.0e6,     1.0e7),
]


def family_priority(name: str) -> tuple[int, str]:
    """Ordering to keep OPAL → OPLIB → OP → lowT grouped in the output."""
    if name.startswith("gs98_aFe"):
        return (4, name)
    if name.startswith("OP_"):
        return (5, name)
    if name.startswith("oplib_"):
        return (2, name)
    if name.startswith("lowT_"):
        return (6, name)
    if name in ("gs98", "a09", "gn93"):
        return (1, name)
    return (3, name)


def pick_binaries(root: Path, Z: float) -> list[Path]:
    """Return all .kapbin that match the given Z."""
    matches: list[tuple[tuple[int, str], Path]] = []
    for p in sorted(root.glob("*.kapbin")):
        # filename format: <family>_z<Z>.kapbin
        rhs = p.stem.split("_z")[-1]
        try:
            Z_val = float(rhs)
        except ValueError:
            continue
        if abs(Z_val - Z) > 1e-12:
            continue
        family = p.stem[: -(len(rhs) + len("_z"))]
        matches.append((family_priority(family), p))
    matches.sort()
    return [p for _, p in matches]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--Z", type=float, default=0.02)
    ap.add_argument("--X", type=float, default=0.7)
    ap.add_argument("--dir", default="third_party/mesa_kap")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    root = Path(args.dir)
    binaries = pick_binaries(root, args.Z)
    if not binaries:
        print(f"no .kapbin matched Z={args.Z} in {root}", file=sys.stderr)
        return 1

    print(f"PK opacity families at X={args.X}, Z={args.Z}")
    print(f"kap_data: {root} ({len(binaries)} families)\n")

    # Header row: stellar-track columns
    header = ["family"] + [f"{lab} κ [cm²/g]" for lab, _, _ in TRACK]
    rows: list[list[str]] = [header]

    csv_rows: list[dict] = []

    for p in binaries:
        d = read_kap_bin(p)
        family = p.stem.split("_z")[0]
        row = [family]
        csv_entry = {"family": family, "Z": d["Z"], "X_query": args.X}
        # Clamp X to the table's supported range for families whose minimum
        # X ≠ 0 (e.g. oplib_*_z0.1 starts at X = 0.01).
        X_use = float(np.clip(args.X, d["X"][0], d["X"][-1]))
        for label, rho, T in TRACK:
            kappa = trilin(d, X_use, rho, T)
            row.append(f"{kappa:.3g}")
            csv_entry[label] = kappa
        rows.append(row)
        csv_rows.append(csv_entry)

    widths = [max(len(r[j]) for r in rows) for j in range(len(header))]
    for r in rows:
        print("  ".join(f"{c:<{widths[j]}}" for j, c in enumerate(r)))

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as f:
            w = _csv.DictWriter(
                f, fieldnames=["family", "Z", "X_query"]
                           + [lab for lab, _, _ in TRACK]
            )
            w.writeheader()
            for row in csv_rows:
                w.writerow(row)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
