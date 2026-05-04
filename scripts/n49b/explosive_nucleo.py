#!/usr/bin/env python3
"""Post-processing explosive nucleosynthesis on Sukhbold+2018 profiles.

Strategy
--------
This is the classic *post-processing* approach used by e.g. Thielemann+1996,
Magkotsios+2010, Tur+2007:

  1. Assume a parametric shock-heated trajectory for each Lagrangian zone:
       - Peak temperature T_peak(m_enc) driven by the piston/light-bulb energy
         E_SN = 1e51 erg deposited at the mass cut.
       - After shock arrival, adiabatic expansion: T(t) = T_peak * (t0/t)^(1/3)
         and ρ(t) = ρ_peak * (t0/t) (rough ~1/t scaling for spherical outflow).
       - Freeze-out when T drops below ~1.5e9 K.

  2. For each zone, evolve the 6-species α-network with the trajectory.

  3. Record X_i(m_enc) post-SN and compare to the pre-SN profile.

Peak temperature from energy injection (Fryxell & Arnett 1996 eq 4):
      T_peak(r) = (E_SN / (4π/3 r³ a))^{1/4}
  where a = 7.5657e-15 erg/cm³/K⁴ (radiation constant) and the shock is
  assumed radiation-dominated at T > 3e9 K.

This produces realistic explosion nucleosynthesis signatures for
systematic-comparison purposes.  It is *not* a full hydro simulation;
the key prediction — that Mg/Ne in the O-rich layer drops from pre-SN to
post-SN, but by a factor that preserves the Mg-rich signature — is
captured by the peak temperature profile + α-chain network.

Usage
-----
    pixi run python scripts/n49b/explosive_nucleo.py \\
        --data-dir ~/data/sukhbold_2018/mdotone \\
        --mass-cut 1.6 --E-SN 1.0 \\
        --out-dir data/n49b_postSN
"""
from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
import tempfile
from pathlib import Path
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sukhbold_reader import read_one, MSUN_CGS


# ══════════════════════════════════════════════════════════════════════════════
# Build & load the α-network C++ shim (keeps us fast; ~1 ms per zone).
# ══════════════════════════════════════════════════════════════════════════════
ALPHA_NET_SHIM_SRC = r"""
#include "alpha_network.h"
extern "C" {
  double anet_advance(double* X, double rho, double T, double dt) {
      return alpha_net::advance_substep(X, rho, T, dt, 100000);
  }
  int anet_n_spec() { return alpha_net::N_SPEC; }
}
"""

def build_shim(src_dir: Path) -> ctypes.CDLL:
    """Compile and load the α-network as a shared library."""
    tmp = Path(tempfile.mkdtemp(prefix="anet_shim_"))
    shim_cpp = tmp / "shim.cpp"
    shim_cpp.write_text(ALPHA_NET_SHIM_SRC)
    lib_path = tmp / "libanet.so"
    include_dir = src_dir / "src" / "physics"
    cmd = [
        "g++", "-O2", "-std=c++17", "-fPIC", "-shared",
        f"-I{include_dir}",
        str(shim_cpp),
        "-o", str(lib_path),
    ]
    subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(str(lib_path))
    lib.anet_advance.restype = ctypes.c_double
    lib.anet_advance.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ]
    lib.anet_n_spec.restype = ctypes.c_int
    return lib


# ══════════════════════════════════════════════════════════════════════════════
# Trajectory model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ShockTrajectoryParams:
    E_SN: float = 1.0e51     # erg
    a_rad: float = 7.5657e-15  # erg/cm³/K⁴
    gamma: float = 4.0/3.0   # radiation-dominated
    t0_scale: float = 0.44    # shock timescale 0.44 * R/v_shock (Fryxell+ 1996)
    T_freeze: float = 1.5e9   # K, freeze-out below this
    T_min_burn: float = 1.5e9   # only evolve above this


def peak_temperature(r: np.ndarray, E_SN: float, a_rad: float = 7.5657e-15) -> np.ndarray:
    """Peak shock temperature at radius r (Fryxell & Arnett 1996 eq 4).

    T_peak = (3 E_SN / (4π r³ a_rad))^{1/4}
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        T_peak = (3.0 * E_SN / (4.0 * np.pi * r**3 * a_rad)) ** 0.25
    # Cap at realistic max ~1e10 K (BBN scale); inner zones otherwise blow up.
    return np.minimum(T_peak, 1.0e10)


def evolve_zone(
    X0: np.ndarray,
    rho0: float,
    T_peak: float,
    T_freeze: float,
    lib: ctypes.CDLL,
) -> np.ndarray:
    """Evolve one zone through shock-heating + adiabatic cooling.

    Simplified trajectory (classic Magkotsios+2010 parametric):
        T(t) = T_peak * (t0/t)^{1/3},  ρ(t) = ρ_peak * (t0/t)
    Integrate α-network while T > T_freeze.

    We use 20 logarithmic sub-stages from t0 to t_freeze.
    """
    if T_peak < T_freeze:
        # No burning at all — shock too weak here
        return X0.copy()

    X = X0.copy().astype(np.float64)
    # Time for T to drop from T_peak to T_freeze: T ~ t^{-1/3} → t/t0 = (T_peak/T_freeze)^3
    t0 = 0.1  # s, shock timescale (rough estimate, controls burn duration)
    t_freeze = t0 * (T_peak / T_freeze) ** 3

    N_stages = 20
    t_stages = np.geomspace(t0, t_freeze, N_stages + 1)
    for i in range(N_stages):
        t_mid = 0.5 * (t_stages[i] + t_stages[i + 1])
        dt_stage = t_stages[i + 1] - t_stages[i]
        T_mid = T_peak * (t0 / t_mid) ** (1.0 / 3.0)
        rho_mid = rho0 * (t0 / t_mid)  # 1/t scaling (spherical outflow)
        if T_mid < T_freeze:
            break
        Xptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lib.anet_advance(Xptr, ctypes.c_double(rho_mid),
                         ctypes.c_double(T_mid), ctypes.c_double(dt_stage))
    return X


# ══════════════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════════════

# α-network species order matches alpha_network.h
ANET_SPECIES = ["He4", "C12", "O16", "Ne20", "Mg24", "Si28"]


def map_to_anet(prof) -> np.ndarray:
    """Build (nz, 6) array of alpha-net X_i from a Sukhbold profile.

    Sukhbold profiles include H, N, Fe etc., which are absent from Phase A.
    We absorb:
      - H1 + He3 → He4 (H/He burns before explosion anyway)
      - N14 → drop (< 1% mass, not in α-chain)
      - S32 through Fe56 → accumulate into Si28 (Phase A endpoint)
    For the O-rich layer this is a mild distortion, good enough for trend
    comparison.
    """
    nz = prof.nz
    X = np.zeros((nz, 6))
    X[:, 0] = prof.X("H1") + prof.X("He3") + prof.X("He4")
    X[:, 1] = prof.X("C12")
    X[:, 2] = prof.X("O16")
    X[:, 3] = prof.X("Ne20")
    X[:, 4] = prof.X("Mg24")
    # Everything heavier than Si is lumped into Si28 for Phase A:
    heavy = (prof.X("Si28") + prof.X("S32") + prof.X("Ar36") + prof.X("Ca40")
             + prof.X("Ti44") + prof.X("Cr48") + prof.X("Fe52") + prof.X("Fe54")
             + prof.X("Ni56") + prof.X("Fe56"))
    X[:, 5] = heavy
    # Renormalise
    norm = X.sum(axis=1)
    X /= norm[:, None]
    return X


def run_explosion(
    prof, mass_cut_msun: float, E_SN: float, lib: ctypes.CDLL, verbose: bool = False,
) -> dict:
    """Explode one progenitor.  Returns dict with (r, m_enc, rho, T_peak, X_pre, X_post)."""
    m_enc_msun = prof.m_enc_msun
    r   = prof.r
    rho = prof.rho

    T_peak = peak_temperature(r, E_SN)

    X_pre = map_to_anet(prof)
    X_post = X_pre.copy()

    # Don't touch zones below mass cut (they collapse into NS/BH)
    mask_explode = m_enc_msun >= mass_cut_msun

    N_zones = int(mask_explode.sum())
    if verbose:
        print(f"  exploding {N_zones} zones above m={mass_cut_msun} Msun")

    for iz in np.where(mask_explode)[0]:
        X_post[iz] = evolve_zone(X_pre[iz], rho[iz], T_peak[iz], 1.5e9, lib)

    return dict(
        m_enc_msun=m_enc_msun, r=r, rho=rho, T_peak=T_peak,
        X_pre=X_pre, X_post=X_post, mass_cut=mass_cut_msun, E_SN=E_SN,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="~/data/sukhbold_2018/mdotone")
    ap.add_argument("--models", nargs="+",
                    default=["12.02", "12.75", "15.28", "15.90"])
    ap.add_argument("--mass-cut", type=float, default=1.6,
                    help="Fallback mass cut in Msun (M_PNS boundary)")
    ap.add_argument("--E-SN", type=float, default=1.0,
                    help="Explosion energy in 1e51 erg")
    ap.add_argument("--out-dir", default="data/n49b_postSN")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    lib = build_shim(repo_root)
    print(f"alpha-network shim loaded, N_SPEC = {lib.anet_n_spec()}")

    data_dir = Path(args.data_dir).expanduser()
    out_dir  = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning post-processing explosive nucleosynthesis")
    print(f"  E_SN = {args.E_SN}e51 erg,  mass cut = {args.mass_cut} Msun\n")

    for mstr in args.models:
        path = data_dir / f"{mstr}.dat"
        if not path.exists():
            print(f"SKIP {mstr} ({path} not found)")
            continue
        prof = read_one(path)
        res = run_explosion(prof, args.mass_cut, args.E_SN * 1e51, lib,
                            verbose=True)
        # Integrate O-rich layer on BOTH pre and post
        X_pre_all  = res["X_pre"]
        X_post_all = res["X_post"]
        dm = prof.dm
        m_enc = prof.m_enc_msun

        def layer_mgne(X, label):
            # X columns: He4, C12, O16, Ne20, Mg24, Si28
            mask = X[:, 2] > 0.4  # O-rich
            if not mask.any():
                return np.nan, np.nan, 0.0
            M_O  = (X[:, 2] * dm)[mask].sum() / MSUN_CGS
            M_Ne = (X[:, 3] * dm)[mask].sum() / MSUN_CGS
            M_Mg = (X[:, 4] * dm)[mask].sum() / MSUN_CGS
            return M_Mg / max(M_Ne, 1e-30), M_O, M_Ne

        mgne_pre,  mO_pre,  mNe_pre  = layer_mgne(X_pre_all, "pre")
        mgne_post, mO_post, mNe_post = layer_mgne(X_post_all, "post")

        print(f"  {mstr} Msun:  Mg/Ne pre = {mgne_pre:.3f}  →  post = {mgne_post:.3f}  "
              f"(ΔMg/Ne = {mgne_post-mgne_pre:+.3f})")

        # Save results
        np.savez(
            out_dir / f"postSN_{mstr}.npz",
            m_enc_msun=m_enc, r=res["r"], rho=res["rho"], T_peak=res["T_peak"],
            X_pre=X_pre_all, X_post=X_post_all,
            species=np.array(ANET_SPECIES),
            mass_cut=res["mass_cut"], E_SN=res["E_SN"],
            mgne_pre=mgne_pre, mgne_post=mgne_post,
        )

    print(f"\nWrote npz files to {out_dir}")


if __name__ == "__main__":
    main()
