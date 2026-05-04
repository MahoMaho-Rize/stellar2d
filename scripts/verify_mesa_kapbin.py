#!/usr/bin/env python3
"""Verify KAPv1 binaries round-trip against the original ASCII.

For each written file, re-parse both sides and assert every logκ entry
agrees to 1e-12 absolute. Also dumps a physical κ at a few probe (ρ, T)
points so we can eyeball that the binary is usable end-to-end.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from convert_mesa_kap import read_kap_bin  # noqa: E402
from mesa_kap import (  # noqa: E402
    UnsupportedKapFile,
    list_type1_families,
    parse_kap_file,
)


def verify_one(family: str, Z: float, ascii_paths, bin_path: Path) -> tuple[int, int]:
    """Returns (checked_cells, max_diff_abs * 1e18 as int-like proxy)."""
    data = read_kap_bin(bin_path)
    if data["family"] != family:
        raise AssertionError(f"family mismatch: bin={data['family']} want={family}")
    if abs(data["Z"] - Z) > 1e-12:
        raise AssertionError(f"Z mismatch: bin={data['Z']} want={Z}")

    # Rebuild the reference 3-D array from the ASCII files
    tables = []
    for p in ascii_paths:
        try:
            tables.append(parse_kap_file(p))
        except UnsupportedKapFile as e:
            print(f"  skip {p.name}: {e}", file=sys.stderr)
    tables.sort(key=lambda t: t.X)
    if len(tables) != data["log_kap"].shape[0]:
        raise AssertionError(
            f"n_X mismatch: bin={data['log_kap'].shape[0]} ascii={len(tables)}"
        )

    ref = np.stack([t.log_kap for t in tables], axis=0)
    diff = np.abs(ref - data["log_kap"])
    max_abs = float(diff.max())
    if max_abs > 1e-12:
        raise AssertionError(
            f"{bin_path.name}: logκ max abs diff={max_abs:.3e} exceeds 1e-12"
        )
    return ref.size, max_abs


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1
               else "/home/kiriko/mesa-ref/data/kap_data")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "third_party/mesa_kap")

    groups = list_type1_families(src)
    if not groups:
        print("No Type-1 groups found", file=sys.stderr)
        return 1

    total_cells = 0
    worst = 0.0
    worst_file = ""
    tested = 0
    missing = 0
    for (family, Z), paths in sorted(groups.items()):
        z_str = f"{Z:g}"
        bin_path = dst / f"{family}_z{z_str}.kapbin"
        if not bin_path.exists():
            missing += 1
            continue
        try:
            cells, max_abs = verify_one(family, Z, paths, bin_path)
        except AssertionError as e:
            print(f"  FAIL {bin_path.name}: {e}", file=sys.stderr)
            return 2
        total_cells += cells
        if max_abs > worst:
            worst = max_abs
            worst_file = bin_path.name
        tested += 1

    print(
        f"verified {tested} binary(ies), {missing} missing, "
        f"{total_cells:,d} cells compared"
    )
    print(f"worst |Δlogκ| = {worst:.2e} in {worst_file or '(n/a)'}")

    # Physical κ sanity on the solar gs98 Z=0.02 binary
    solar_bin = dst / "gs98_z0.02.kapbin"
    if solar_bin.exists():
        sol = read_kap_bin(solar_bin)
        # Use the X slice closest to 0.7
        iX = int(np.argmin(np.abs(sol["X"] - 0.7)))
        print()
        print(f"solar sanity  gs98 Z=0.02 X={sol['X'][iX]}")
        probes = [
            ("photosphere ρ=1e-7 T=5800 K", 1e-7, 5800.0),
            ("core     ρ=150  T=1.5e7 K", 150.0, 1.5e7),
            ("envelope ρ=1e-2 T=1e6  K",   1e-2, 1e6),
        ]
        logT_arr = sol["logT"]
        logR_arr = sol["logR"]
        for label, rho, T in probes:
            logT = float(np.log10(T))
            logR = float(np.log10(rho) - 3.0 * logT + 18.0)
            jt = int(np.clip(
                int((logT - sol["logT_min"])
                    / ((sol["logT_max"] - sol["logT_min"]) / (logT_arr.size - 1))),
                0, logT_arr.size - 2,
            ))
            ir = int(np.clip(
                int((logR - sol["logR_min"])
                    / ((sol["logR_max"] - sol["logR_min"]) / (logR_arr.size - 1))),
                0, logR_arr.size - 2,
            ))
            # nearest-neighbour (just a sanity dump)
            kappa = 10.0 ** sol["log_kap"][iX, jt, ir]
            print(f"  {label:<34} κ ≈ {kappa:.3g} cm²/g")

    return 0


if __name__ == "__main__":
    sys.exit(main())
