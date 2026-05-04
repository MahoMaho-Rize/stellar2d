#!/usr/bin/env python3
"""Convert MESA Type-1 kap tables into self-describing binaries for radial1d.

For every (family, Z) group in the MESA kap_data directory, stacks the
per-X-slice logκ_R grids into a 3-D tensor (n_X, n_logT, n_logR) and writes
it to <out>/<family>_z<Z>.kapbin with a KAPv1 header.

Binary layout (little-endian, all doubles are float64, all ints are int32):

  0     magic       8 B   "KAPv1\0\0\0"
  8     family     48 B   zero-padded UTF-8 (e.g. "gs98", "OP_a09_nans_removed_by_hand")
  56    Z           8 B   float64, metal mass fraction
  64    n_X         4 B   int32
  68    n_logT      4 B   int32
  72    n_logR      4 B   int32
  76    pad         4 B
  80    logT_min    8 B   float64
  88    logT_max    8 B   float64
  96    logR_min    8 B   float64
  104   logR_max    8 B   float64
  112   reserved   16 B
  128   payload:
            X       float64 × n_X
            logT    float64 × n_logT
            logR    float64 × n_logR
            log_kap float64 × n_X × n_logT × n_logR
                     (X outer, logR inner-fastest)

Usage:
    python3 scripts/convert_mesa_kap.py [src] [dst] [family1 family2 ...]

Default src: /home/kiriko/mesa-ref/data/kap_data
Default dst: third_party/mesa_kap/
Default families: all Type-1 (ignore _co, kR_*, oplib_* if not requested).

Every group must have identical (logT, logR) grids across its X slices —
this is true for every MESA family I've looked at. The tool checks and
refuses to write otherwise.
"""
from __future__ import annotations

import struct
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

MAGIC = b"KAPv1\0\0\0"
HEADER_SIZE = 128


def write_kap_bin(
    dst: Path,
    family: str,
    Z: float,
    X: np.ndarray,          # (n_X,)
    logT: np.ndarray,       # (n_logT,)
    logR: np.ndarray,       # (n_logR,)
    log_kap: np.ndarray,    # (n_X, n_logT, n_logR)
) -> int:
    """Serialize one (family, Z) group. Returns byte count written."""
    assert log_kap.shape == (X.size, logT.size, logR.size), log_kap.shape
    assert log_kap.dtype == np.float64
    family_bytes = family.encode("utf-8")
    if len(family_bytes) > 48:
        raise ValueError(f"family name too long: {family}")
    family_padded = family_bytes.ljust(48, b"\0")

    header = bytearray(HEADER_SIZE)
    header[0:8] = MAGIC
    header[8:56] = family_padded
    struct.pack_into("<d", header, 56, Z)
    struct.pack_into("<iii", header, 64, X.size, logT.size, logR.size)
    # offset 76 is pad
    struct.pack_into("<dddd", header, 80,
                     float(logT[0]), float(logT[-1]),
                     float(logR[0]), float(logR[-1]))
    # 112..128 reserved zero

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        f.write(header)
        f.write(np.ascontiguousarray(X, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(logT, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(logR, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(log_kap, dtype=np.float64).tobytes())
    return dst.stat().st_size


def read_kap_bin(src: Path):
    """Parse a previously-written KAPv1 binary. Returns dict for round-trip tests."""
    with src.open("rb") as f:
        header = f.read(HEADER_SIZE)
        if header[:8] != MAGIC:
            raise ValueError(f"{src.name}: bad magic")
        family = header[8:56].rstrip(b"\0").decode("utf-8")
        (Z,) = struct.unpack_from("<d", header, 56)
        n_X, n_logT, n_logR = struct.unpack_from("<iii", header, 64)
        logT_min, logT_max, logR_min, logR_max = struct.unpack_from("<dddd", header, 80)

        X = np.frombuffer(f.read(n_X * 8), dtype=np.float64)
        logT = np.frombuffer(f.read(n_logT * 8), dtype=np.float64)
        logR = np.frombuffer(f.read(n_logR * 8), dtype=np.float64)
        log_kap = np.frombuffer(
            f.read(n_X * n_logT * n_logR * 8), dtype=np.float64
        ).reshape(n_X, n_logT, n_logR)

    return dict(
        family=family, Z=Z, X=X, logT=logT, logR=logR,
        logT_min=logT_min, logT_max=logT_max,
        logR_min=logR_min, logR_max=logR_max,
        log_kap=log_kap,
    )


def convert_group(
    family: str,
    Z: float,
    paths: list[Path],
    dst_root: Path,
) -> tuple[Path, int, int] | None:
    """Parse + stack + write. Returns (out_path, n_X, bytes) or None on skip."""
    tables: list[KapTable] = []
    for p in paths:
        try:
            tables.append(parse_kap_file(p))
        except UnsupportedKapFile as e:
            print(f"  skip {p.name}: {e}", file=sys.stderr)
    if not tables:
        return None

    # Consistency across X slices
    ref = tables[0]
    for t in tables[1:]:
        if (t.n_logT != ref.n_logT or t.n_logR != ref.n_logR
                or abs(t.logT_min - ref.logT_min) > 1e-9
                or abs(t.logT_max - ref.logT_max) > 1e-9
                or abs(t.logR_min - ref.logR_min) > 1e-9
                or abs(t.logR_max - ref.logR_max) > 1e-9):
            print(
                f"  FAIL {family} Z={Z}: grid mismatch between "
                f"{ref.path.name} and {t.path.name}",
                file=sys.stderr,
            )
            return None

    X = np.array([t.X for t in tables], dtype=np.float64)
    if not np.all(np.diff(X) > 0):
        # Sort just in case (should already be sorted by list_type1_families)
        order = np.argsort(X)
        X = X[order]
        tables = [tables[i] for i in order]

    log_kap = np.stack([t.log_kap for t in tables], axis=0)   # (n_X, n_logT, n_logR)
    logT = tables[0].logT
    logR = tables[0].logR

    # Filename: "<family>_z<Z>.kapbin". Z formatting matches MESA conventions:
    # float with up to 6 sig figs, trimmed trailing zeros.
    z_str = f"{Z:g}"
    out = dst_root / f"{family}_z{z_str}.kapbin"
    nbytes = write_kap_bin(out, family, Z, X, logT, logR, log_kap)

    # Round-trip sanity
    rt = read_kap_bin(out)
    if not (rt["log_kap"].shape == log_kap.shape
            and np.allclose(rt["log_kap"], log_kap)
            and np.allclose(rt["X"], X)):
        print(f"  FAIL {out.name}: round-trip mismatch", file=sys.stderr)
        return None

    return out, len(tables), nbytes


def main() -> int:
    args = sys.argv[1:]
    src = Path(args[0]) if len(args) >= 1 else Path("/home/kiriko/mesa-ref/data/kap_data")
    dst = Path(args[1]) if len(args) >= 2 else Path("third_party/mesa_kap")
    families_filter = set(args[2:]) if len(args) >= 3 else None

    if not src.is_dir():
        print(f"ERROR: {src} not found", file=sys.stderr)
        return 1

    def pred(family: str, Z: float) -> bool:
        if families_filter is None:
            return True
        return family in families_filter

    groups = list_type1_families(src, predicate=pred)
    if not groups:
        print("No Type-1 kap files matched.", file=sys.stderr)
        return 2

    print(f"src = {src}")
    print(f"dst = {dst.resolve()}")
    print(f"groups: {len(groups)}")
    print()

    total_bytes = 0
    total_groups = 0
    skipped = 0
    for (family, Z), paths in sorted(groups.items()):
        res = convert_group(family, Z, paths, dst)
        if res is None:
            skipped += 1
            continue
        out, n_X, nbytes = res
        total_bytes += nbytes
        total_groups += 1
        print(
            f"  {out.name}  n_X={n_X:2d}  size={nbytes / 1024:.1f} KB"
        )

    print()
    print(f"wrote {total_groups} group(s), {total_bytes / 1024 / 1024:.2f} MB, "
          f"skipped {skipped}")
    return 0 if skipped == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
