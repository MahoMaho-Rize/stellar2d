#!/usr/bin/env python3
"""Sukhbold+2018 pre-SN profile reader.

Format of each .dat file in mdotone.tar.gz:

    line 1: blank
    line 2: "current star mass = <M_star> Msun; structural units are in [cgs] ..."
    line 3: blank
    line 4: column headers (space-separated names)
    line 5-6: blank
    line 7+: data rows, first column is zone index, others are cgs doubles

Columns provided (30 total):
    grid, mass, radius, velocity, density, temperature, pressure,
    specific_entropy, Abar, Ye, neutrons,
    H1, He3, He4, C12, N14, O16, Ne20, Mg24, Si28, S32, Ar36, Ca40,
    Ti44, Cr48, Fe52, Fe54, Ni56, Fe56

`mass` is the *zone mass dm in grams* (Lagrangian shell mass, sums to M_star);
`radius` is zone outer radius in cm.  Composition columns are mass fractions.

Use `prof.m_enc` / `prof.dm` to access the enclosed mass / zone mass.

Dataset: Sukhbold, Woosley & Heger 2018, doi:10.7910/DVN/VOEXDE
Paper:  ApJ 860, 93 (2018)
"""
from __future__ import annotations

import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np

MSUN_CGS = 1.989e33

ELEMENT_COLS = [
    "neutrons", "H1", "He3", "He4", "C12", "N14", "O16", "Ne20", "Mg24",
    "Si28", "S32", "Ar36", "Ca40", "Ti44", "Cr48", "Fe52", "Fe54", "Ni56",
    "Fe56",
]

STRUCTURE_COLS = [
    "grid", "mass", "radius", "velocity", "density", "temperature",
    "pressure", "specific_entropy", "Abar", "Ye",
]

ALL_COLS = STRUCTURE_COLS + ELEMENT_COLS


@dataclass
class PresnProfile:
    """A single Sukhbold+2018 pre-SN profile.

    All structure arrays are length `nz` (number of zones).
    ZAMS mass is inferred from the filename stem (e.g. "12.02.dat" -> 12.02).
    `M_star` is the current (post-mass-loss) total mass in Msun.
    """
    zams_mass: float                 # Msun — from filename
    M_star: float                    # Msun — current mass after mass loss
    nz: int
    data: Dict[str, np.ndarray]      # all columns by name, cgs units

    @property
    def dm(self) -> np.ndarray:
        """Zone mass (g) — the 'mass' column in the raw file."""
        return self.data["mass"]

    @property
    def m_enc(self) -> np.ndarray:
        """Enclosed mass (g) — cumulative sum of dm."""
        return np.cumsum(self.data["mass"])

    @property
    def m_enc_msun(self) -> np.ndarray:
        return self.m_enc / MSUN_CGS

    @property
    def r(self) -> np.ndarray:
        return self.data["radius"]

    @property
    def rho(self) -> np.ndarray:
        return self.data["density"]

    @property
    def T(self) -> np.ndarray:
        return self.data["temperature"]

    @property
    def P(self) -> np.ndarray:
        return self.data["pressure"]

    @property
    def s(self) -> np.ndarray:
        """Specific entropy (k_B per baryon)."""
        return self.data["specific_entropy"]

    def X(self, element: str) -> np.ndarray:
        """Mass fraction of an element (e.g. 'O16')."""
        return self.data[element]


def read_one(path: Path) -> PresnProfile:
    """Parse a single Sukhbold+2018 .dat profile file."""
    path = Path(path)
    zams_mass = float(path.stem)

    with path.open() as f:
        text = f.read()

    M_star = _parse_mstar(text)

    # Find the header line containing "grid  mass  radius ..." and then numeric rows
    lines = text.splitlines()
    header_idx = None
    for i, ln in enumerate(lines):
        toks = ln.split()
        if toks and toks[0] == "grid" and "mass" in toks and "radius" in toks:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"{path.name}: could not find column header row")

    header = lines[header_idx].split()
    if header != ALL_COLS:
        raise ValueError(
            f"{path.name}: unexpected header columns\n"
            f"  got : {header}\n"
            f"  want: {ALL_COLS}"
        )

    rows = []
    for ln in lines[header_idx + 1 :]:
        ln = ln.strip()
        if not ln:
            continue
        toks = ln.split()
        if len(toks) != len(ALL_COLS):
            raise ValueError(
                f"{path.name}: row has {len(toks)} cols, expected {len(ALL_COLS)}"
            )
        rows.append([float(t) for t in toks])

    arr = np.asarray(rows, dtype=np.float64)
    data = {col: arr[:, i] for i, col in enumerate(ALL_COLS)}
    return PresnProfile(
        zams_mass=zams_mass,
        M_star=M_star,
        nz=arr.shape[0],
        data=data,
    )


def read_from_tar(tar_path: Path, members: Optional[list[str]] = None) -> Iterator[PresnProfile]:
    """Stream profiles directly from the tar.gz (no full extraction required).

    If `members` is None, all .dat files are yielded.  Otherwise members is a
    list of ZAMS mass strings (e.g. ["12.02", "12.75"]) or full paths inside
    the tarball.
    """
    tar_path = Path(tar_path)
    with tarfile.open(tar_path, "r:gz") as tf:
        if members is None:
            names = [n for n in tf.getnames() if n.endswith(".dat")]
        else:
            names = []
            for m in members:
                if "/" in m or m.endswith(".dat"):
                    names.append(m)
                else:
                    names.append(f"mdotone/{m}.dat")

        for name in names:
            try:
                ti = tf.getmember(name)
            except KeyError:
                raise KeyError(f"{name} not in {tar_path}")
            with tf.extractfile(ti) as f:
                text = f.read().decode("ascii")
            yield _parse_text(text, zams_mass=_zams_from_name(name))


def iter_dir(dir_path: Path) -> Iterator[PresnProfile]:
    """Iterate all .dat profiles in a directory (after `tar xzf`)."""
    for p in sorted(Path(dir_path).glob("*.dat")):
        yield read_one(p)


def _parse_mstar(text: str) -> float:
    m = re.search(r"current star mass\s*=\s*([\d.]+)\s*Msun", text)
    if not m:
        raise ValueError("could not parse 'current star mass' line")
    return float(m.group(1))


def _zams_from_name(tar_name: str) -> float:
    """Extract ZAMS mass from tarball member name like 'mdotone/12.02.dat'."""
    stem = Path(tar_name).stem
    return float(stem)


def _parse_text(text: str, zams_mass: float) -> PresnProfile:
    M_star = _parse_mstar(text)
    lines = text.splitlines()
    header_idx = None
    for i, ln in enumerate(lines):
        toks = ln.split()
        if toks and toks[0] == "grid" and "mass" in toks and "radius" in toks:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"M={zams_mass}: could not find header")
    header = lines[header_idx].split()
    if header != ALL_COLS:
        raise ValueError(f"M={zams_mass}: unexpected header {header}")
    rows = []
    for ln in lines[header_idx + 1 :]:
        s = ln.strip()
        if not s:
            continue
        toks = s.split()
        rows.append([float(t) for t in toks])
    arr = np.asarray(rows, dtype=np.float64)
    data = {col: arr[:, i] for i, col in enumerate(ALL_COLS)}
    return PresnProfile(
        zams_mass=zams_mass,
        M_star=M_star,
        nz=arr.shape[0],
        data=data,
    )


if __name__ == "__main__":
    # Smoke test on 4 key profiles
    import sys
    dir_path = Path(sys.argv[1] if len(sys.argv) > 1 else
                    "~/data/sukhbold_2018/mdotone").expanduser()
    for mstr in ["12.02", "12.75", "15.28", "15.90"]:
        p = dir_path / f"{mstr}.dat"
        if not p.exists():
            print(f"SKIP {p} (not found)")
            continue
        prof = read_one(p)
        mz_msun = prof.m_enc_msun[-1]
        r_out = prof.r[-1]
        X_O = prof.X("O16")
        X_Ne = prof.X("Ne20")
        X_Mg = prof.X("Mg24")
        # integrate over O-rich region
        X_O_thresh = 0.4
        mask = X_O > X_O_thresh
        if mask.any():
            dm = prof.dm
            m_O = (X_O * dm)[mask].sum() / MSUN_CGS
            m_Ne = (X_Ne * dm)[mask].sum() / MSUN_CGS
            m_Mg = (X_Mg * dm)[mask].sum() / MSUN_CGS
            mgne = m_Mg / max(m_Ne, 1e-30)
        else:
            m_O = m_Ne = m_Mg = 0.0
            mgne = np.nan
        print(
            f"  ZAMS {prof.zams_mass:5.2f} Msun  "
            f"M_final {prof.M_star:5.2f}  nz={prof.nz:5d}  "
            f"M_total {mz_msun:5.2f} Msun  R_out {r_out:.2e} cm  "
            f"M_O_rich {m_O:.3f}  Mg/Ne {mgne:.3f}"
        )
