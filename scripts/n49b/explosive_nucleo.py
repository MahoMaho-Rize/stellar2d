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
    r_zone: float,
    T_peak: float,
    T_freeze: float,
    lib: ctypes.CDLL,
    v_shock: float = 1.0e9,  # ~1e4 km/s, post-shock velocity (Arnett 1996)
) -> np.ndarray:
    """Evolve one zone through shock-heating + free expansion cooling.

    Homologous-expansion trajectory (Magkotsios+2010, Tur+2007):
      - Hydrodynamic timescale τ_hydro = r / v_shock (the shock-crossing time)
      - After shock passage, free expansion: T ∝ 1/t, ρ ∝ 1/t³  (adiabatic γ=4/3)
      - We use T(t) = T_peak · (τ/t) and ρ(t) = ρ₀ · (τ/t)³
      - Freeze-out when T drops below T_freeze

    This is the standard parametric form used in post-processing SN
    nucleosynthesis (see Magkotsios+2010 eq 1 & Table 1).  It avoids the
    t^{-1/3} over-burning of the previous version.
    """
    if T_peak < T_freeze:
        return X0.copy()

    X = X0.copy().astype(np.float64)
    # Hydrodynamic timescale = r/v, but Magkotsios+2010 §2.2 recommend that
    # the effective burn duration is much shorter than r/v — adiabatic
    # expansion brings T below freeze-out within several τ_dyn.  Use
    # τ_hydro = 446 s / sqrt(ρ_peak/1 g/cc) as in Fowler-Hoyle; this is the
    # analytic cooling timescale for a shock-heated fluid element
    # (see Arnett 1996 "Supernovae and Nucleosynthesis" eq 8.40).
    tau_hydro = 446.0 / max(np.sqrt(max(rho0, 1.0)), 1.0)
    # Clamp: at least 10 ms (inner zones have small r), at most 30 s
    tau_hydro = max(min(tau_hydro, 30.0), 1e-2)

    # Freeze-out time: T ~ 1/t  →  t_freeze = tau * (T_peak / T_freeze)
    t0 = tau_hydro
    t_freeze = t0 * (T_peak / T_freeze)
    t_freeze = min(t_freeze, 100.0)

    N_stages = 40
    t_stages = np.geomspace(t0, t_freeze, N_stages + 1)
    for i in range(N_stages):
        t_mid = 0.5 * (t_stages[i] + t_stages[i + 1])
        dt_stage = t_stages[i + 1] - t_stages[i]
        T_mid = T_peak * (t0 / t_mid)
        rho_mid = rho0 * (t0 / t_mid) ** 3
        if T_mid < T_freeze:
            break
        Xptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lib.anet_advance(Xptr, ctypes.c_double(rho_mid),
                         ctypes.c_double(T_mid), ctypes.c_double(dt_stage))
    return X


# ══════════════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════════════

# α-network species order matches alpha_network.h (Phase B, 13 species)
ANET_SPECIES = ["He4", "C12", "O16", "Ne20", "Mg24", "Si28",
                "S32", "Ar36", "Ca40", "Ti44", "Cr48", "Fe52", "Ni56"]
N_SPEC = 13


def map_to_anet(prof) -> np.ndarray:
    """Build (nz, 13) array of alpha-net X_i from a Sukhbold profile.

    Phase B maps Sukhbold's ~19 species into the 13 α-chain slots:
      - H1 + He3 → He4 (H burns before explosion)
      - N14 → drop (< 1% mass, not in α-chain)
      - Fe54 + Fe56 → Ni56 (iron-peak lumped at Ni56 for α-network)
    Everything else maps one-to-one.
    """
    nz = prof.nz
    X = np.zeros((nz, N_SPEC))
    X[:,  0] = prof.X("H1") + prof.X("He3") + prof.X("He4")
    X[:,  1] = prof.X("C12")
    X[:,  2] = prof.X("O16")
    X[:,  3] = prof.X("Ne20")
    X[:,  4] = prof.X("Mg24")
    X[:,  5] = prof.X("Si28")
    X[:,  6] = prof.X("S32")
    X[:,  7] = prof.X("Ar36")
    X[:,  8] = prof.X("Ca40")
    X[:,  9] = prof.X("Ti44")
    X[:, 10] = prof.X("Cr48")
    X[:, 11] = prof.X("Fe52")
    X[:, 12] = prof.X("Ni56") + prof.X("Fe54") + prof.X("Fe56")
    # Renormalise (drops N14 etc.)
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
        X_post[iz] = evolve_zone(
            X_pre[iz], rho[iz], r[iz], T_peak[iz], 1.5e9, lib)

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

        # Define the O-rich layer ONCE from pre-SN structure (X_O > 0.4).
        # Post-SN composition is integrated over the SAME Lagrangian mask —
        # this is how Sato+2024 measures Mg/Ne survival (O-rich layer is a
        # structural property of the pre-SN progenitor).
        pre_mask = X_pre_all[:, 2] > 0.4

        def layer_mgne(X, mask):
            if not mask.any():
                return np.nan, 0.0, 0.0
            M_O  = (X[:, 2] * dm)[mask].sum() / MSUN_CGS
            M_Ne = (X[:, 3] * dm)[mask].sum() / MSUN_CGS
            M_Mg = (X[:, 4] * dm)[mask].sum() / MSUN_CGS
            return M_Mg / max(M_Ne, 1e-30), M_O, M_Ne

        mgne_pre,  mO_pre,  mNe_pre  = layer_mgne(X_pre_all,  pre_mask)
        mgne_post, mO_post, mNe_post = layer_mgne(X_post_all, pre_mask)

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
