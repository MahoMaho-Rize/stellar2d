#!/usr/bin/env python3
"""Inspect MESA kap tables (Type-1 only).

Usage:
    python3 scripts/inspect_mesa_kap.py [root] [prefix]

Examples:
    python3 scripts/inspect_mesa_kap.py
    python3 scripts/inspect_mesa_kap.py /home/kiriko/mesa-ref/data/kap_data gs98_z0.02
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mesa_kap import (  # noqa: E402
    KapTable,
    UnsupportedKapFile,
    list_type1_families,
    parse_kap_file,
)


def summarize(tables: list[KapTable]) -> None:
    if not tables:
        print("(no tables matched)")
        return
    print(f"Parsed {len(tables)} table(s)")
    print()
    groups: dict[tuple[str, float], list[KapTable]] = {}
    for t in tables:
        import re
        stem = t.path.name
        family = re.split(r"_z\d|_x\d", stem)[0]
        groups.setdefault((family, t.Z), []).append(t)
    for (family, Z), ts in sorted(groups.items()):
        ts.sort(key=lambda x: x.X)
        print(f"  family={family}  Z={Z:.4g}  X={[t.X for t in ts]}")
        print(
            f"    logT ∈ [{ts[0].logT_min:.3f}, {ts[0].logT_max:.3f}] "
            f"({ts[0].n_logT} pts)   "
            f"logR ∈ [{ts[0].logR_min:.3f}, {ts[0].logR_max:.3f}] "
            f"({ts[0].n_logR} pts)"
        )
    print()
    solar = next(
        (t for t in tables if abs(t.X - 0.7) < 1e-3 and abs(t.Z - 0.02) < 1e-4),
        tables[0],
    )
    print(f"Sanity probe on {solar.path.name}  (X={solar.X}, Z={solar.Z})")
    print(f"  title: {solar.title}")
    probes = [
        ("photosphere ρ=1e-7 T=5800 K", 1e-7, 5800.0),
        ("core     ρ=150  T=1.5e7 K", 150.0, 1.5e7),
        ("envelope ρ=1e-2 T=1e6  K",   1e-2, 1e6),
        ("H-peak   ρ=1e-4 T=1e4  K",   1e-4, 1e4),
    ]
    for label, rho, T in probes:
        logT = float(np.log10(T))
        logR = float(np.log10(rho) - 3.0 * logT + 18.0)
        in_bounds = (
            solar.logT_min <= logT <= solar.logT_max
            and solar.logR_min <= logR <= solar.logR_max
        )
        mark = "in-table" if in_bounds else "clamped"
        kappa = 10.0 ** solar.kap_at(logT, logR)
        print(
            f"  {label:<34} logT={logT:6.3f} logR={logR:6.3f}  "
            f"κ≈{kappa:.3g} cm²/g  [{mark}]"
        )


def main() -> int:
    args = sys.argv[1:]
    root = Path(args[0]) if args else Path("/home/kiriko/mesa-ref/data/kap_data")
    prefix = args[1] if len(args) > 1 else "gs98_z0.02"

    if not root.exists():
        print(f"ERROR: {root} not found", file=sys.stderr)
        return 1

    print(f"Scanning {root}/{prefix}*.data")
    files = sorted(root.glob(f"{prefix}*.data"))
    if not files:
        fams = sorted({p.name.split("_z")[0] for p in root.glob("*.data")})
        print("  (no files matched; known family prefixes):", file=sys.stderr)
        for f in fams:
            print(f"    {f}", file=sys.stderr)
        return 2

    tables: list[KapTable] = []
    for p in files:
        try:
            tables.append(parse_kap_file(p))
        except UnsupportedKapFile as e:
            print(f"  skip {p.name}: {e}", file=sys.stderr)

    summarize(tables)
    return 0


if __name__ == "__main__":
    sys.exit(main())
