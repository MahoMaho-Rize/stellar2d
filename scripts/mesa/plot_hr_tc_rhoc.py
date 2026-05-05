#!/usr/bin/env python3
"""Stellar-evolution standard plots from radial1d diagnostics.csv.

Produces two publication-quality figures:
  1. T_c - ρ_c plane with regime boundaries + H/He/C ignition lines.
     The evolution trajectory is colored by log(t) and drawn as a smooth
     LineCollection so the gradient reads at a glance.
  2. HR diagram (log L - log T_eff) with Hayashi track reference,
     ZAMS reference, and evolution colored by log(t).

Expected CSV columns (from the updated src/main.cpp schema):
  step, t, dt, mass, KE, IE, PE, total_E, max_mach, max_vr,
  T_c, rho_c, L_nuc, ..., L_surf, ..., T_phot, phot_zone, R_surf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LogNorm

L_SUN = 3.828e33   # erg/s
T_SUN = 5778.0     # K


def load_csv(path: Path) -> dict[str, np.ndarray]:
    """Robust CSV loader that handles variable column count via pandas-style
    dict-of-arrays, without requiring pandas."""
    with path.open() as fh:
        header = fh.readline().strip().split(",")
        rows = [ln.strip().split(",") for ln in fh if ln.strip()]
    arr = np.array(rows, dtype=float)
    out = {}
    for i, col in enumerate(header):
        out[col] = arr[:, i]
    return out


def draw_regime_background(ax):
    """Shade ideal-gas, radiation-dominated, and degenerate-electron regions
    on the log ρ - log T plane, plus the ion-degenerate boundary.
    Reference formulas (cgs):
      P_rad = P_gas  ⇒  aT⁴/3 = ρ kT/(μ m_p)  ⇒  T ≈ 3.2e7 (ρ/μ)^{1/3}
      P_deg_nr = P_gas  ⇒  K ρ^{5/3} = ρkT/(μ_e m_p)
                 ⇒  ρ/T^{3/2} ≈ 2.4e-8 μ_e (mu_e = A/Z ≈ 2)
      For simplicity use MESA textbook (Kippenhahn eq 2.69, 4.71).
    """
    log_rho = np.linspace(-4, 6, 400)
    rho = 10.0 ** log_rho
    mu = 0.62  # fully ionized H+He mix

    # Radiation pressure dominates above T_rad(ρ):
    # aT⁴/3 = ρkT/(μ m_p) → T = (3ρk / (a μ m_p))^{1/3}
    k_B = 1.381e-16
    m_p = 1.673e-24
    a_rad = 7.566e-15
    T_rad = (3.0 * rho * k_B / (a_rad * mu * m_p)) ** (1.0 / 3.0)

    # Electron degeneracy (non-relativistic): ρ/T^{3/2} > 2.4e-8 μ_e
    mu_e = 2.0
    T_deg = (rho / (2.4e-8 * mu_e)) ** (2.0 / 3.0)

    ax.fill_between(log_rho, np.log10(T_rad), 10, color="#b8d4e8",
                    alpha=0.35, label=r"radiation-dominated", zorder=0)
    ax.fill_between(log_rho, 0, np.log10(T_deg), color="#e8d0b8",
                    alpha=0.35, label=r"electron degenerate", zorder=0)

    # Ignition lines (approximate):
    # pp-chain:  T_c ≈ 4e6 K  (10% branch active)
    # CNO:       T_c ≈ 1.5e7 K
    # He:        T_c ≈ 1e8 K
    # C:         T_c ≈ 6e8 K
    for (T_ign, label) in [(4e6, "pp ignition"),
                           (1.5e7, "CNO ignition"),
                           (1e8, "He ignition"),
                           (6e8, "C ignition")]:
        ax.axhline(np.log10(T_ign), ls="--", lw=0.8, color="#555",
                   alpha=0.6, zorder=1)
        ax.text(-3.5, np.log10(T_ign) + 0.05, label,
                fontsize=8, color="#555", alpha=0.9)


def draw_hayashi_zams(ax):
    """Overlay schematic Hayashi track and ZAMS on the HR diagram.
    Hayashi track for solar metallicity pre-MS is near-vertical at
    log(T_eff) ≈ 3.6 for a wide L range.
    """
    log_Teff_hay = 3.63
    log_L_hay = np.linspace(-1, 3, 40)
    ax.plot([log_Teff_hay] * len(log_L_hay), log_L_hay,
            "--", color="#c33", lw=1, alpha=0.7, label="Hayashi (log T_eff≈3.63)")

    # ZAMS points (approximate, Tout et al. 1996 fits):
    # M=0.5:  log Teff=3.60, log L=-1.5
    # M=1.0:  log Teff=3.76, log L=0.0
    # M=2.0:  log Teff=3.95, log L=1.2
    # M=5.0:  log Teff=4.15, log L=2.7
    # M=15:   log Teff=4.45, log L=4.3
    zams_Te = [3.60, 3.76, 3.95, 4.15, 4.45]
    zams_L  = [-1.5, 0.0, 1.2, 2.7, 4.3]
    ax.plot(zams_Te, zams_L, "o-", color="#444", lw=1.2, ms=4,
            label="ZAMS (Tout+96)", alpha=0.8)

    # Solar reference point
    ax.plot(np.log10(T_SUN), 0.0, "*", color="#e8a000", ms=16,
            markeredgecolor="k", markeredgewidth=0.5, label="Sun", zorder=5)


def _thin_by_change(x: np.ndarray, y: np.ndarray, min_sep=2e-4) -> np.ndarray:
    """Return indices that keep points where (x, y) moves more than min_sep
    in log-Euclidean distance — removes the duplicate endpoints when the
    star sits at the Helm table floor for thousands of steps."""
    keep = [0]
    last = np.array([x[0], y[0]])
    for i in range(1, len(x)):
        cur = np.array([x[i], y[i]])
        if np.linalg.norm(cur - last) >= min_sep:
            keep.append(i)
            last = cur
    if keep[-1] != len(x) - 1:
        keep.append(len(x) - 1)
    return np.array(keep)


def plot_tc_rhoc(data: dict, out_path: Path) -> None:
    rho_c = data["rho_c"]
    T_c   = data["T_c"]
    t     = data["t"]

    good = (rho_c > 0) & (T_c > 0) & (t > 0)
    rho_c = rho_c[good]
    T_c   = T_c[good]
    t     = t[good]

    log_rho = np.log10(rho_c)
    log_T   = np.log10(T_c)
    log_t   = np.log10(np.maximum(t, 1.0))
    idx = _thin_by_change(log_rho, log_T, min_sep=5e-5)
    log_rho, log_T, log_t = log_rho[idx], log_T[idx], log_t[idx]

    fig, ax = plt.subplots(figsize=(9, 7))
    draw_regime_background(ax)

    # Evolution as colored segments
    pts = np.column_stack([log_rho, log_T])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    norm = Normalize(vmin=log_t.min(), vmax=log_t.max())
    lc = LineCollection(segs, cmap="viridis", norm=norm,
                        linewidth=2.4, zorder=3, capstyle="round")
    lc.set_array(0.5 * (log_t[:-1] + log_t[1:]))
    ax.add_collection(lc)

    # Mark start / end
    ax.plot(log_rho[0],  log_T[0],  "o", color="#20b020", ms=8,
            markeredgecolor="k", markeredgewidth=0.8, zorder=5,
            label=f"start t={t[0]:.2e}s")
    ax.plot(log_rho[-1], log_T[-1], "s", color="#d02020", ms=8,
            markeredgecolor="k", markeredgewidth=0.8, zorder=5,
            label=f"end t={t[-1]:.2e}s")

    cb = plt.colorbar(lc, ax=ax, pad=0.02, shrink=0.8)
    cb.set_label(r"$\log_{10}(t/{\rm s})$")

    ax.set_xlabel(r"$\log_{10}(\rho_c\,/\,{\rm g\,cm^{-3}})$")
    ax.set_ylabel(r"$\log_{10}(T_c\,/\,{\rm K})$")
    ax.set_title("Core evolution in the $T_c$–$\\rho_c$ plane\n" +
                 "(radial1d 1 $M_\\odot$ pre-MS, Helm + OPAL + pp-chain, rad-in-F)")

    ax.set_xlim(-4, 6)
    ax.set_ylim(4, 9)
    ax.grid(alpha=0.25, zorder=1)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")

    # Zoomed companion — focus on actual trajectory
    zoom_path = out_path.parent / f"{out_path.stem}_zoom.png"
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    pts2 = np.column_stack([log_rho, log_T])
    segs2 = np.stack([pts2[:-1], pts2[1:]], axis=1)
    lc2 = LineCollection(segs2, cmap="viridis", norm=norm,
                         linewidth=2.4, zorder=3)
    lc2.set_array(0.5 * (log_t[:-1] + log_t[1:]))
    ax2.add_collection(lc2)
    cb2 = plt.colorbar(lc2, ax=ax2, pad=0.02, shrink=0.85)
    cb2.set_label(r"$\log_{10}(t/{\rm s})$")

    # Draw ignition lines (just pp ignition for reference)
    ax2.axhline(np.log10(4e6), ls="--", lw=0.8, color="#555", alpha=0.7)
    ax2.text(log_rho.mean(), np.log10(4e6) + 0.004,
             "pp ignition (T≈4×10⁶ K)", fontsize=8, color="#555", ha="center")

    ax2.plot(log_rho[0], log_T[0], "o", color="#20b020", ms=10,
             markeredgecolor="k", markeredgewidth=0.8, zorder=5,
             label=f"start t={10**log_t[0]:.2e}s")
    ax2.plot(log_rho[-1], log_T[-1], "s", color="#d02020", ms=10,
             markeredgecolor="k", markeredgewidth=0.8, zorder=5,
             label=f"end t={10**log_t[-1]:.2e}s")

    # Autoscale with at least 0.05 dex padding on each axis
    dr = max((log_rho.max() - log_rho.min()) * 0.15, 0.03)
    dT = max((log_T.max()   - log_T.min())   * 0.15, 0.02)
    ax2.set_xlim(log_rho.min() - dr, log_rho.max() + dr)
    ax2.set_ylim(log_T.min()   - dT, log_T.max()   + dT)
    ax2.set_xlabel(r"$\log_{10}(\rho_c\,/\,{\rm g\,cm^{-3}})$")
    ax2.set_ylabel(r"$\log_{10}(T_c\,/\,{\rm K})$")
    ax2.set_title("Core evolution (zoomed)")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(zoom_path, dpi=150)
    print(f"wrote {zoom_path}")


def plot_hr(data: dict, out_path: Path) -> None:
    T_phot = data.get("T_phot", None)
    R_surf = data.get("R_surf", None)
    L_surf = data.get("L_surf", None)
    t      = data["t"]

    if T_phot is None or R_surf is None or L_surf is None:
        print("CSV lacks T_phot/R_surf/L_surf — upgrade CSV schema first")
        return

    good = (T_phot > 0) & (R_surf > 0) & (L_surf > 0) & (t > 0)
    T_phot = T_phot[good]
    L_surf = L_surf[good]
    t      = t[good]

    log_Teff = np.log10(T_phot)
    log_L    = np.log10(L_surf / L_SUN)
    log_t    = np.log10(np.maximum(t, 1.0))
    idx = _thin_by_change(log_Teff, log_L, min_sep=5e-5)
    log_Teff, log_L, log_t = log_Teff[idx], log_L[idx], log_t[idx]

    fig, ax = plt.subplots(figsize=(9, 7))
    draw_hayashi_zams(ax)

    pts = np.column_stack([log_Teff, log_L])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    norm = Normalize(vmin=log_t.min(), vmax=log_t.max())
    lc = LineCollection(segs, cmap="viridis", norm=norm,
                        linewidth=2.2, zorder=3)
    lc.set_array(0.5 * (log_t[:-1] + log_t[1:]))
    ax.add_collection(lc)

    ax.plot(log_Teff[0],  log_L[0],  "o", color="#20b020", ms=8,
            markeredgecolor="k", markeredgewidth=0.8, zorder=5,
            label=f"start t={t[0]:.2e}s")
    ax.plot(log_Teff[-1], log_L[-1], "s", color="#d02020", ms=8,
            markeredgecolor="k", markeredgewidth=0.8, zorder=5,
            label=f"end t={t[-1]:.2e}s")

    cb = plt.colorbar(lc, ax=ax, pad=0.02, shrink=0.8)
    cb.set_label(r"$\log_{10}(t/{\rm s})$")

    ax.set_xlabel(r"$\log_{10}(T_{\rm eff}/{\rm K})$")
    ax.set_ylabel(r"$\log_{10}(L/L_\odot)$")
    ax.set_title("HR diagram (radial1d 1 $M_\\odot$ pre-MS)")
    ax.invert_xaxis()   # convention: hot on the left

    # Reasonable viewing window
    ax.set_xlim(max(log_Teff.max(), 4.6) + 0.1, min(log_Teff.min(), 3.0) - 0.1)
    ax.set_ylim(min(log_L.min(), -2) - 0.2, max(log_L.max(), 4.5) + 0.2)

    ax.grid(alpha=0.25, zorder=1)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")

    # HR diagram — full range + early-time highlight panel
    hi_path = out_path.parent / f"{out_path.stem}_earlytime.png"
    # Focus on first log-decade of evolution: often the most interesting
    # dynamics happen when the IC relaxes + pre-MS descends the Hayashi
    # track. Pick points below log(t) = 14 (< 3 Myr) and render separately.
    early = log_t < 14.0
    if early.sum() > 5:
        figh, axh = plt.subplots(figsize=(8, 7))
        pts_e = np.column_stack([log_Teff[early], log_L[early]])
        segs_e = np.stack([pts_e[:-1], pts_e[1:]], axis=1)
        norm_e = Normalize(vmin=log_t[early].min(),
                           vmax=log_t[early].max())
        lc_e = LineCollection(segs_e, cmap="plasma", norm=norm_e,
                              linewidth=2.6, zorder=3)
        lc_e.set_array(0.5 * (log_t[early][:-1] + log_t[early][1:]))
        axh.add_collection(lc_e)
        cbh = plt.colorbar(lc_e, ax=axh, pad=0.02, shrink=0.85)
        cbh.set_label(r"$\log_{10}(t/{\rm s})$")

        draw_hayashi_zams(axh)
        axh.plot(log_Teff[early][0], log_L[early][0], "o",
                 color="#20b020", ms=9, markeredgecolor="k",
                 markeredgewidth=0.8, zorder=5, label="IC relaxation start")
        axh.plot(log_Teff[early][-1], log_L[early][-1], "s",
                 color="#d02020", ms=9, markeredgecolor="k",
                 markeredgewidth=0.8, zorder=5, label="transient end")

        axh.set_xlabel(r"$\log_{10}(T_{\rm eff}/{\rm K})$")
        axh.set_ylabel(r"$\log_{10}(L/L_\odot)$")
        axh.set_title("HR diagram — early relaxation + Hayashi descent")
        axh.invert_xaxis()
        axh.set_xlim(log_Teff[early].max() + 0.2,
                     log_Teff[early].min() - 0.1)
        axh.set_ylim(log_L[early].min()    - 0.3,
                     log_L[early].max()    + 0.3)
        axh.grid(alpha=0.25)
        axh.legend(loc="upper left", fontsize=9)
        figh.tight_layout()
        figh.savefig(hi_path, dpi=150)
        print(f"wrote {hi_path}")


def plot_dashboard(data: dict, out_path: Path) -> None:
    """Bonus: 4-panel summary of key evolutionary variables vs time."""
    t = data["t"]
    good = t > 0

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # T_c, rho_c
    ax = axes[0, 0]
    ax.semilogx(t[good], data["T_c"][good], color="#c33")
    ax.set_ylabel(r"$T_c$ [K]")
    ax.set_xlabel("t [s]")
    ax.grid(alpha=0.3)
    ax.set_title("central temperature")

    ax = axes[0, 1]
    ax.loglog(t[good], data["rho_c"][good], color="#38c")
    ax.set_ylabel(r"$\rho_c$ [g/cc]")
    ax.set_xlabel("t [s]")
    ax.grid(alpha=0.3, which="both")
    ax.set_title("central density")

    # L_surf vs L_nuc
    ax = axes[1, 0]
    L_surf = data.get("L_surf", None)
    L_nuc  = data.get("L_nuc",  None)
    if L_surf is not None:
        ax.loglog(t[good], np.maximum(L_surf[good], 1e-30) / L_SUN,
                  color="#c33", label=r"$L_{\rm surf}$")
    if L_nuc is not None:
        ax.loglog(t[good], np.maximum(L_nuc[good], 1e-30) / L_SUN,
                  color="#38c", label=r"$L_{\rm nuc}$")
    ax.set_ylabel(r"$L / L_\odot$")
    ax.set_xlabel("t [s]")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    ax.set_title("luminosities")

    # T_phot
    ax = axes[1, 1]
    T_phot = data.get("T_phot", None)
    if T_phot is not None:
        ax.semilogx(t[good], T_phot[good], color="#c70")
    ax.set_ylabel(r"$T_{\rm eff}$ [K]")
    ax.set_xlabel("t [s]")
    ax.grid(alpha=0.3)
    ax.set_title("photospheric temperature")

    fig.suptitle("radial1d 1 $M_\\odot$ pre-MS dashboard", fontsize=13)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="diagnostics.csv path")
    ap.add_argument("--outdir", type=Path, default=Path("docs/images"))
    ap.add_argument("--prefix", type=str, default="ignition_1Msol")
    args = ap.parse_args()

    data = load_csv(args.csv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    plot_tc_rhoc(data,   args.outdir / f"{args.prefix}_TcRhoc.png")
    plot_hr(data,        args.outdir / f"{args.prefix}_HR.png")
    plot_dashboard(data, args.outdir / f"{args.prefix}_dashboard.png")


if __name__ == "__main__":
    main()
