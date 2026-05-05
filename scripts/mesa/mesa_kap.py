"""MESA OPAL / Ferguson kap table reader (Type-1 layout only).

Type-1 files contain a single (X, Z) pair and a 2-D grid of logκ_R on
(logT, logR). Used by families: gs98, a09, gn93, OP_*, oplib_*, lowT_*,
and the gs98_aFe_* alpha-enhanced variants.

Skipped here (require separate parsers):
  - `*_co*`  (form=2, 4-D in C/O enhancement axes)
  - `kR_*`   (electron conduction tables, different layout)
  - `*.h5`   (AESOPUS HDF5)
  - `cache/`, `condtabl.data`

Public API:
  parse_kap_file(path)   → KapTable
  list_type1_families(root, predicate=None) → dict[(family, Z) → sorted list[Path]]
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class KapTable:
    path: Path
    title: str
    form: int
    version: int
    X: float
    Z: float
    n_logR: int
    logR_min: float
    logR_max: float
    n_logT: int
    logT_min: float
    logT_max: float
    logR: np.ndarray     # (n_logR,)
    logT: np.ndarray     # (n_logT,)
    log_kap: np.ndarray  # (n_logT, n_logR)  log10 κ_R [cm²/g]

    @property
    def dlogR(self) -> float:
        return (self.logR_max - self.logR_min) / (self.n_logR - 1)

    @property
    def dlogT(self) -> float:
        return (self.logT_max - self.logT_min) / (self.n_logT - 1)

    def kap_at(self, logT: float, logR: float) -> float:
        """Bilinear (log-space) probe. Clamped to grid edges; for sanity only."""
        jt = int(np.clip(int((logT - self.logT_min) / self.dlogT), 0, self.n_logT - 2))
        ir = int(np.clip(int((logR - self.logR_min) / self.dlogR), 0, self.n_logR - 2))
        t = (logT - self.logT[jt]) / self.dlogT
        r = (logR - self.logR[ir]) / self.dlogR
        k00 = self.log_kap[jt, ir]
        k01 = self.log_kap[jt, ir + 1]
        k10 = self.log_kap[jt + 1, ir]
        k11 = self.log_kap[jt + 1, ir + 1]
        return (
            (1 - t) * ((1 - r) * k00 + r * k01)
            + t * ((1 - r) * k10 + r * k11)
        )


_HEADER_RE_TYPE1 = re.compile(
    r"^\s*(\d+)\s+(\d+)"
    r"\s+([-+\d.Ee]+)\s+([-+\d.Ee]+)"
    r"\s+(\d+)\s+([-+\d.Ee]+)\s+([-+\d.Ee]+)"
    r"\s+(\d+)\s+([-+\d.Ee]+)\s+([-+\d.Ee]+)"
)


class UnsupportedKapFile(Exception):
    """Raised when a file is not a Type-1 table we can parse."""


def parse_kap_file(path: Path) -> KapTable:
    """Parse a single Type-1 kap file. Raises UnsupportedKapFile otherwise."""
    with path.open() as f:
        lines = f.readlines()
    if len(lines) < 7:
        raise UnsupportedKapFile(f"{path.name}: file too short")
    title = lines[0].rstrip()

    m = _HEADER_RE_TYPE1.match(lines[2])
    if not m:
        raise UnsupportedKapFile(f"{path.name}: not a Type-1 header")
    (form, version, X, Z,
     n_logR, logR_min, logR_max,
     n_logT, logT_min, logT_max) = m.groups()
    form = int(form)
    version = int(version)
    if form != 1:
        raise UnsupportedKapFile(f"{path.name}: form={form} (only form=1 supported)")
    X = float(X)
    Z = float(Z)
    n_logR = int(n_logR)
    logR_min = float(logR_min)
    logR_max = float(logR_max)
    n_logT = int(n_logT)
    logT_min = float(logT_min)
    logT_max = float(logT_max)

    # Scan forward for the first T row (numeric, right field count)
    data_start = None
    for i in range(3, len(lines)):
        parts = lines[i].split()
        if len(parts) != n_logR + 1:
            continue
        try:
            v = float(parts[0])
        except ValueError:
            continue
        # First T row has v == logT_min within float tolerance
        if abs(v - logT_min) < 1e-6:
            data_start = i
            break
    if data_start is None:
        raise UnsupportedKapFile(f"{path.name}: no T rows found")

    log_kap = np.empty((n_logT, n_logR), dtype=np.float64)
    logT = np.empty(n_logT, dtype=np.float64)
    for j in range(n_logT):
        row = lines[data_start + j].split()
        if len(row) != n_logR + 1:
            raise UnsupportedKapFile(
                f"{path.name}: row {j} expected {n_logR + 1} fields, got {len(row)}"
            )
        logT[j] = float(row[0])
        log_kap[j, :] = [float(v) for v in row[1:]]

    logR = np.linspace(logR_min, logR_max, n_logR)

    return KapTable(
        path=path,
        title=title,
        form=form,
        version=version,
        X=X,
        Z=Z,
        n_logR=n_logR,
        logR_min=logR_min,
        logR_max=logR_max,
        n_logT=n_logT,
        logT_min=logT_min,
        logT_max=logT_max,
        logR=logR,
        logT=logT,
        log_kap=log_kap,
    )


# Regex that splits a filename into (family, Z_literal, X_literal).
# family = everything before first `_z<num>_`; Z = the number after _z; X = after _x.
_NAME_RE = re.compile(
    r"^(?P<family>.+?)_z(?P<Z>[0-9]+(?:\.[0-9]*)?(?:[eE][-+]?[0-9]+)?)"
    r"_x(?P<X>[0-9]+(?:\.[0-9]*)?(?:[eE][-+]?[0-9]+)?)\.data$"
)

SKIP_FAMILIES = {
    # C/O-enhanced (form=2, 4D) — separate parser needed.
    "a09_co", "gn93_co", "gs98_co",
    "gs98_aFe_p4_co",
}


def list_type1_families(
    root: Path,
    predicate=None,
) -> dict[tuple[str, float], list[Path]]:
    """Group every Type-1 `<family>_z<Z>_x<X>.data` into { (family, Z): [paths] }.

    Order within each group is ascending X. Non-Type-1 (kR_*, condtabl.data,
    .h5, _co) are silently dropped. `predicate(family, Z)` can filter further.
    """
    groups: dict[tuple[str, float], list[tuple[float, Path]]] = {}
    for p in sorted(root.glob("*.data")):
        m = _NAME_RE.match(p.name)
        if m is None:
            continue   # e.g. kR_Z1E-2.data, condtabl.data
        family = m.group("family")
        if family in SKIP_FAMILIES:
            continue
        Z = float(m.group("Z"))
        X = float(m.group("X"))
        if predicate is not None and not predicate(family, Z):
            continue
        groups.setdefault((family, Z), []).append((X, p))

    return {
        key: [p for _, p in sorted(lst, key=lambda t: t[0])]
        for key, lst in groups.items()
    }
