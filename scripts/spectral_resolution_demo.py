#!/usr/bin/env python3
"""
Demo: why N=48 Chebyshev doesn't mean "48-pixel resolution".

Key point: N+1 Chebyshev coefficients define a continuous function that
can be evaluated at any number of points via barycentric interpolation
(Berrut & Trefethen 2004, SIAM Rev 46, 501-517) WITHOUT additional
discretisation error. Only round-off (~1e-15) accrues.

We compare:
  - N=48 Chebyshev g-mode eigenfunction evaluated at 4096 points (spectral)
  - Exp J Nr=1024 FD eigenfunction interpolated to 4096 points (FD+cubic)
to show the spectral representation has HIGHER visual fidelity despite
having 20× fewer "nodes".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import BarycentricInterpolator, CubicSpline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gmode_infra as gi
from gmode_exp_k_chebyshev_full import (solve_gmode_full_chebyshev,
                                         cheb_D1_on_interval,
                                         load_gyre_structure_interp_cheb)
from gmode_exp_j_full_gyre_compat import main as exp_j_main  # not used directly


def run_cheb_ng1(N):
    x_cgl, V_2, A_star, U, c_1, G1 = load_gyre_structure_interp_cheb(
        '/tmp/gyre_run/poly3.txt', N)
    omsq, u_out = solve_gmode_full_chebyshev(x_cgl, V_2, U, A_star, c_1, G1, 1, 5)
    Nr = len(x_cgl)
    # Return y_1 for n_g = 1 (highest-frequency g-mode)
    return x_cgl, u_out[:Nr, 0].real, omsq[0]


def run_fd_ng1(Nr):
    from gmode_infra import solve_gmode_full_gyre_compat
    data = np.loadtxt('/tmp/gyre_run/poly3.txt', skiprows=1)
    x_full = data[:, 0]
    mask = (x_full > 0.01) & (x_full < 0.999)
    x_src = x_full[mask]
    x_u = np.linspace(x_src[0], x_src[-1], Nr)
    V_2u = np.interp(x_u, x_src, data[:, 1][mask])
    A_u  = np.interp(x_u, x_src, data[:, 2][mask])
    U_u  = np.interp(x_u, x_src, data[:, 3][mask])
    c_1u = np.interp(x_u, x_src, data[:, 4][mask])
    G1_u = np.interp(x_u, x_src, data[:, 5][mask])
    omsq, u_out = solve_gmode_full_gyre_compat(x_u, V_2u, U_u, A_u, c_1u, G1_u, 1, 5)
    # u_out staggered format is [y1_nodes; y3_nodes; y2_cells; y4_cells]
    n_node = Nr
    return x_u, u_out[:n_node, 0].real, omsq[0]


def main():
    gi.provenance_banner("scripts/spectral_resolution_demo.py",
                         "docs/spectral_liouville_plan_2026-05-03.md")
    print(" Spectral-vs-FD resolution demo: same g-mode, different representations")
    print("=" * 72)

    # Compute both
    N_cheb = 48
    Nr_fd = 1024
    x_cgl, y1_cheb, om_cheb = run_cheb_ng1(N_cheb)
    x_fd, y1_fd, om_fd = run_fd_ng1(Nr_fd)

    print(f"  Chebyshev: N={N_cheb} → {len(x_cgl)} CGL nodes, ω²={om_cheb:.10f}")
    print(f"  FD       : Nr={Nr_fd} → {len(x_fd)} uniform nodes, ω²={om_fd:.10f}")

    # Evaluate both on a common 4096-point fine grid
    N_fine = 4096
    x_fine = np.linspace(max(x_cgl[0], x_fd[0]),
                          min(x_cgl[-1], x_fd[-1]), N_fine)
    # Chebyshev barycentric — exact to machine epsilon
    bary = BarycentricInterpolator(x_cgl, y1_cheb)
    y1_cheb_fine = bary(x_fine)
    # FD cubic spline — O(h⁴) interpolation error
    spl = CubicSpline(x_fd, y1_fd)
    y1_fd_fine = spl(x_fine)

    # Normalise both (sign-align)
    y1_cheb_fine /= np.max(np.abs(y1_cheb_fine))
    y1_fd_fine /= np.max(np.abs(y1_fd_fine))
    if np.sign(y1_cheb_fine[N_fine // 2]) != np.sign(y1_fd_fine[N_fine // 2]):
        y1_fd_fine = -y1_fd_fine

    # Metrics
    diff = y1_cheb_fine - y1_fd_fine
    diff_max = np.max(np.abs(diff))
    print(f"\n  max |y1_cheb - y1_fd| on {N_fine}-point fine grid = {diff_max:.3e}")
    print("  (the two SOLUTIONS agree; they are different REPRESENTATIONS of the")
    print("   same eigenfunction.  Chebyshev stores 49 coefficients, FD stores 1024 values.)")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=140)

    # Top-left: both on fine grid (overlapping)
    ax = axes[0, 0]
    ax.plot(x_fine, y1_cheb_fine, "C2-", lw=1.2, label=f"Chebyshev N={N_cheb}, eval 4096 pts")
    ax.plot(x_fine, y1_fd_fine, "C3--", lw=0.9, alpha=0.7, label=f"FD Nr={Nr_fd}, cubic spline to 4096")
    ax.set_xlabel("r")
    ax.set_ylabel("y_1(r) (normalised)")
    ax.set_title("n_g=1 eigenfunction y_1 on 4096-point viewing grid")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Top-right: zoom in inner region where oscillation is rapid
    ax = axes[0, 1]
    idx_zoom = x_fine < 0.3
    ax.plot(x_fine[idx_zoom], y1_cheb_fine[idx_zoom], "C2-", lw=1.2,
            label=f"Chebyshev N={N_cheb}")
    ax.plot(x_fine[idx_zoom], y1_fd_fine[idx_zoom], "C3--", lw=0.9,
            alpha=0.7, label=f"FD Nr={Nr_fd}")
    # Overlay the actual CGL nodes as markers
    ax.plot(x_cgl[x_cgl < 0.3], y1_cheb[x_cgl < 0.3] / np.max(np.abs(y1_cheb)),
            "C2o", ms=4, label=f"{np.sum(x_cgl < 0.3)} CGL nodes used")
    ax.set_xlabel("r")
    ax.set_ylabel("y_1(r)")
    ax.set_title("Inner region zoom (r < 0.3)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Bottom-left: density of nodes
    ax = axes[1, 0]
    ax.plot(x_cgl, np.ones_like(x_cgl) * 1, "C2|", ms=12,
            label=f"{len(x_cgl)} CGL nodes (Chebyshev)")
    ax.plot(x_fd, np.ones_like(x_fd) * 0.5, "C3|", ms=6, alpha=0.5,
            label=f"{len(x_fd)} uniform nodes (FD)")
    ax.set_yticks([])
    ax.set_xlabel("r")
    ax.set_ylim(0, 1.5)
    ax.set_title("Node distributions (CGL clusters at endpoints)")
    ax.legend(fontsize=9, loc="upper center")
    ax.grid(alpha=0.3, axis="x")

    # Bottom-right: difference between representations
    ax = axes[1, 1]
    ax.semilogy(x_fine, np.abs(diff), "C4-", lw=0.7)
    ax.set_xlabel("r")
    ax.set_ylabel("|y1_cheb - y1_fd|")
    ax.set_title(f"Representation difference (max = {diff_max:.2e})")
    ax.grid(alpha=0.3, which="both")
    ax.set_ylim(1e-8, 1e-1)

    fig.suptitle(
        f"Chebyshev N=48 vs FD Nr=1024 — same eigenfunction, {Nr_fd/N_cheb:.0f}× fewer DOF",
        fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = gi.VID / "spectral_resolution_demo.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"\n  => {out}")


if __name__ == "__main__":
    main()
