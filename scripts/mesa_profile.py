"""Reader for MESA LOGS/profile*.data files.

The format is three logical blocks separated by a blank line:

    block 1  line 1  column-number header           "1 2 3 ..."
             line 2  field names                    "model_number num_zones ..."
             line 3  field values                   (globals for the whole run)
    (blank line)
    block 2  line 5  column-number header again
             line 6  zone column names              "zone mass logR ..."
             line 7+ one row per zone, surface → core

Zone units:
    mass    M_enc / M_star (dimensionless, zone 1 = 1 at surface)
    logR    log10(r / R_sun)
    logT    log10(T / K)
    logRho  log10(ρ / g cm^-3)
    logP    log10(P / erg cm^-3)
    X, Y, Z  mass fractions

Reader also pulls the MSUN / RSUN / LSUN constants MESA records in the
header so downstream scripts can convert without redefining them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MesaProfile:
    header: dict[str, float | str]     # globals (model_number, Teff, …)
    zone_names: list[str]
    zone_data: np.ndarray              # (n_zones, n_cols) float64, surface→core

    @property
    def n_zones(self) -> int:
        return int(self.header["num_zones"])

    @property
    def Msun(self) -> float:
        return float(self.header["msun"])

    @property
    def Rsun(self) -> float:
        return float(self.header["rsun"])

    @property
    def Lsun(self) -> float:
        return float(self.header["lsun"])

    @property
    def M_star(self) -> float:
        """Total stellar mass [g]."""
        return float(self.header["star_mass"]) * self.Msun

    @property
    def R_star(self) -> float:
        """Photospheric radius [cm]."""
        return float(self.header["photosphere_r"]) * self.Rsun

    def col(self, name: str) -> np.ndarray:
        i = self.zone_names.index(name)
        return self.zone_data[:, i]

    def to_cgs_profile(self) -> dict:
        """Return a dict of (surface→core) cgs arrays plus global scalars.

        Keys:
            r       face radius [cm], including both surface and r≈0 entry
            m_enc   enclosed mass at each face [g]
            rho     zone density [g/cm^3]
            T       zone temperature [K]
            P       zone pressure [erg/cm^3]
            X, Y, Z zone mass fractions
            logR_mesa, logT_mesa, logRho_mesa, logP_mesa (diagnostic)

        Zone 1 corresponds to the surface; MESA does not output the r=0
        face explicitly — we synthesise it by appending r=0 at the
        innermost mass point.
        """
        mass_frac = self.col("mass")          # (N,) surface=1, core≈0
        logR = self.col("logR")
        logT = self.col("logT")
        logRho = self.col("logRho")
        logP = self.col("logP")
        X = self.col("x_mass_fraction_H")
        Y = self.col("y_mass_fraction_He")
        Z = self.col("z_mass_fraction_metals")

        Msun = self.Msun
        Rsun = self.Rsun

        r   = 10.0 ** logR * Rsun              # [cm]
        m   = mass_frac * self.M_star          # [g]
        rho = 10.0 ** logRho
        T   = 10.0 ** logT
        P   = 10.0 ** logP

        return {
            "r": r,
            "m_enc": m,
            "rho": rho,
            "T": T,
            "P": P,
            "X": X,
            "Y": Y,
            "Z": Z,
            "logR_mesa": logR,
            "logT_mesa": logT,
            "logRho_mesa": logRho,
            "logP_mesa": logP,
            "R_star": self.R_star,
            "M_star": self.M_star,
            "Teff": float(self.header["Teff"]),
            "photosphere_L_Lsun": float(self.header["photosphere_L"]),
        }


def _parse_value(tok: str) -> float | str:
    s = tok.strip()
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    try:
        return float(s)
    except ValueError:
        return s


def read_profile(path: Path) -> MesaProfile:
    """Parse one MESA `LOGS/profile*.data` file."""
    with path.open() as f:
        lines = f.readlines()

    # Find block boundaries (blank-line separator after globals).
    blanks = [i for i, ln in enumerate(lines) if ln.strip() == ""]
    if not blanks:
        raise ValueError(f"{path.name}: missing blank line separator")
    end_globals = blanks[0]
    start_zones = blanks[0] + 1
    # Globals block: 3 lines (colnum, names, values). Skip to values.
    if end_globals < 3:
        raise ValueError(f"{path.name}: globals block too short")
    hdr_names = lines[end_globals - 2].split()
    hdr_values = [_parse_value(t) for t in lines[end_globals - 1].split()]
    if len(hdr_names) != len(hdr_values):
        raise ValueError(
            f"{path.name}: globals header/value mismatch "
            f"({len(hdr_names)} vs {len(hdr_values)})"
        )
    header = dict(zip(hdr_names, hdr_values))

    # Zone block: skip colnum line, names, then data rows.
    if start_zones + 2 >= len(lines):
        raise ValueError(f"{path.name}: zone block missing")
    zone_names_line = lines[start_zones + 1]
    zone_names = zone_names_line.split()
    data_rows = []
    for ln in lines[start_zones + 2:]:
        if not ln.strip():
            break
        parts = ln.split()
        if len(parts) != len(zone_names):
            raise ValueError(
                f"{path.name}: zone row width mismatch "
                f"({len(parts)} vs {len(zone_names)})"
            )
        data_rows.append([float(p) for p in parts])
    zone_data = np.array(data_rows, dtype=np.float64)

    n_zones = int(header.get("num_zones", zone_data.shape[0]))
    if zone_data.shape[0] != n_zones:
        raise ValueError(
            f"{path.name}: header says {n_zones} zones, found {zone_data.shape[0]}"
        )

    return MesaProfile(header=header, zone_names=zone_names, zone_data=zone_data)
