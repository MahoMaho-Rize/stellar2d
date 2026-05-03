#!/usr/bin/env python3
"""E3: Spectrum-level consequence of primitive-node factorisation choices.

Three discrete operators, all claiming to implement the linear
g-mode dynamics   v̈ = -(L^{-1} R) v   on Lane-Emden n=3/2:

  (Asm)  M_asm  = L^{-1} R                            (assembled, Theorem 6.1)
  (PrA)  M_prA  = diag(k_x^2 N^2)                     (Proposition 1 surrogate
                                                       = diag(1/ρ)·R)
  (PrB)  M_prB  = -ρ^{-1} D·diag(ρ)·D + k_x^2(1-N^2) I
         (= ρ^{-1}(L - R), a factorisation often implemented when a code
         expands both L and R into apply_dy + pointwise-multiply primitives
         and cancels the common ρ outside.)

All three agree in the continuous limit on a smooth bulk, but in the
discrete setting on Lane-Emden they differ both in norm (Proposition 1
gap) AND, for variant B, in spectral *type*: M_prB acquires negative
eigenvalues in the surface layer where N^2(y) > 1.

This script:
  1. Builds all three operators at N_y = 64, k_x = 2π/L_y.
  2. Computes their spectra and flags negative real parts.
  3. Measures the per-step deviation of a g-mode eigenvector under each,
     verifying that variant B triggers exponential blow-up within one
     oscillation period while variant A and the assembled scheme remain
     bounded.
  4. Produces fig5_3_spectra.png showing the three spectra side by side.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import scipy.linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nonlinear_paths_infra import cgl_grid, cc_weights, bg_lane_emden

OUT_DIR = SCRIPT_DIR.parent / "review" / "e3_spectral"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = SCRIPT_DIR.parent / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def build_operators(Ny=64, Ly=1.0, kx=2 * np.pi, rho_cut=0.05):
    y, D = cgl_grid(Ny, Ly)
    rho, N2 = bg_lane_emden(y, Ly, rho_cut=rho_cut)
    intr = slice(1, Ny - 1)

    L = -D @ (np.diag(rho) @ D) + kx**2 * np.diag(rho)
    R = kx**2 * np.diag(N2 * rho)
    L_int = L[intr, intr]
    R_int = R[intr, intr]

    M_asm = scipy.linalg.solve(L_int, R_int)
    M_prA = kx**2 * np.diag(N2[intr])
    M_prB_full = (kx**2 * (np.eye(Ny) - np.diag(N2))
                  - np.diag(1.0 / rho) @ (D @ (np.diag(rho) @ D)))
    M_prB = M_prB_full[intr, intr]

    return dict(y=y, rho=rho, N2=N2, D=D, L=L, R=R, intr=intr,
                M_asm=M_asm, M_prA=M_prA, M_prB=M_prB, kx=kx, Ny=Ny)


def spectrum_summary(name, M):
    lam = scipy.linalg.eigvals(M)
    re = lam.real
    im = lam.imag
    n_neg = int(np.sum(re < -1e-10))
    n_complex = int(np.sum(np.abs(im) > 1e-10))
    print(f"  {name:8s}  n={len(lam):3d}  "
          f"min Re = {re.min():.3e}  max Re = {re.max():.3e}  "
          f"# Re<0 = {n_neg}  # complex = {n_complex}")
    return dict(name=name, n=len(lam), min_re=float(re.min()),
                max_re=float(re.max()), n_negative=n_neg,
                n_complex=n_complex, eigenvalues=lam)


def evolve_vspace_rk4(M_op, v0_int, n_steps, dt):
    """RK4 on the second-order system ẅ = -M·w, starting from w=v0, ẇ=0.
    Return (v_final, per_step_dev_first_step, max_abs_v_history)."""
    v = v0_int.copy()
    w = np.zeros_like(v)
    v0_norm = np.linalg.norm(v0_int)
    max_abs = v0_norm
    dev_hist = []
    for step in range(n_steps):
        def rhs(V_, W_):
            return W_, -(M_op @ V_)
        k1v, k1w = rhs(v, w)
        k2v, k2w = rhs(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w)
        k3v, k3w = rhs(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w)
        k4v, k4w = rhs(v + dt * k3v, w + dt * k3w)
        v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        w = w + dt / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
        if not np.all(np.isfinite(v)):
            return v, dev_hist, step, np.inf
        cur = np.linalg.norm(v)
        max_abs = max(max_abs, cur)
        coeff = np.dot(v, v0_int) / max(v0_norm**2, 1e-300)
        r = v - coeff * v0_int
        dev_hist.append(float(np.linalg.norm(r) / v0_norm))
    return v, dev_hist, None, max_abs


def main():
    print("E3: Spectral pathology of primitive-node factorisation variants")
    print("    Lane-Emden n=3/2, Ny=64, kx=2π/Ly, rho_cut=0.05")
    print("=" * 74)

    data = build_operators()
    Ny = data["Ny"]; kx = data["kx"]
    intr = data["intr"]

    print("\nOperator spectra:")
    info_asm = spectrum_summary("M_asm", data["M_asm"])
    info_prA = spectrum_summary("M_prA", data["M_prA"])
    info_prB = spectrum_summary("M_prB", data["M_prB"])

    # Take top eigenvector of M_asm as IC (the g-mode n_g=1)
    lam_asm, V_asm = scipy.linalg.eig(data["M_asm"])
    lam_asm = lam_asm.real; V_asm = V_asm.real
    mask = np.isfinite(lam_asm) & (lam_asm > 0)
    lam_asm = lam_asm[mask]; V_asm = V_asm[:, mask]
    order = np.argsort(lam_asm)[::-1]
    omega_n1 = np.sqrt(lam_asm[order[0]])
    V_n1 = V_asm[:, order[0]]
    V_n1 = V_n1 / np.max(np.abs(V_n1)) * 1e-8  # amplitude 1e-8

    period = 2 * np.pi / omega_n1
    print(f"\nIC: g-mode n=1 eigenvector, ω_1 = {omega_n1:.4f}, T = {period:.3f}")

    # Time evolution at dt = 2e-3, run 1.5 periods
    dt = 2e-3
    n_steps = int(1.5 * period / dt)
    print(f"\nRK4 evolution: dt = {dt}, {n_steps} steps ≈ 1.5 periods")
    print(f"{'scheme':10s}  {'status':18s}  {'max |v|/amp':>12s}  {'dev@first step':>14s}")
    print("-" * 68)
    results = {}
    for name, M in (("assembled", data["M_asm"]),
                    ("primitive A", data["M_prA"]),
                    ("primitive B", data["M_prB"])):
        v_fin, dev_hist, blew_step, max_abs = evolve_vspace_rk4(
            M, V_n1, n_steps, dt)
        amp_ratio = max_abs / np.linalg.norm(V_n1)
        if blew_step is not None:
            t_blow = blew_step * dt
            status = f"blew up at t={t_blow:.3f} (={t_blow/period:.2f} periods)"
        else:
            status = f"stable, final t={n_steps*dt:.3f}"
        first_dev = dev_hist[0] if dev_hist else float('nan')
        print(f"  {name:10s}  {status:18s}  {amp_ratio:12.3e}  {first_dev:14.3e}")
        results[name] = dict(status=status, max_abs_over_amp=amp_ratio,
                             first_dev=first_dev, blew_step=blew_step)

    # ------------ Figure 5.3: spectra ------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=140)
    all_ev = [info_asm["eigenvalues"], info_prA["eigenvalues"], info_prB["eigenvalues"]]
    labels = [r"Assembled:  $M_{\rm asm} = L^{-1} R$",
              r"Primitive var. A:  $M_{\rm prim} = {\rm diag}(k^2 N^2)$",
              r"Primitive var. B:  $M_{\rm eff} = -\rho^{-1}\partial_y(\rho\partial_y) + k^2(1-N^2)$"]
    for ax, ev, lab in zip(axes, all_ev, labels):
        re = ev.real
        im = ev.imag
        # symlog: linear near 0, log far away
        pos = re > 0
        neg = re < 0
        ax.scatter(re[pos], im[pos], c="C0", marker="o", s=25,
                   label=f"positive ({int(pos.sum())})")
        if neg.any():
            ax.scatter(re[neg], im[neg], c="red", marker="x", s=55, lw=2,
                       label=f"negative ({int(neg.sum())})")
        ax.axvline(0, color="k", lw=0.4)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_title(lab, fontsize=9)
        ax.set_xscale("symlog", linthresh=1e-2)
        ax.set_xlabel(r"Re $\lambda$")
        ax.set_ylabel(r"Im $\lambda$")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig_path = FIG_DIR / "fig5_3_spectra.png"
    plt.savefig(fig_path, dpi=140)
    plt.close(fig)
    print(f"\nWrote figure {fig_path}")

    # Save CSV
    csv_path = OUT_DIR / "spectra_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "n_eigs", "min_re", "max_re", "n_negative_re"])
        for info in (info_asm, info_prA, info_prB):
            w.writerow([info["name"], info["n"], info["min_re"],
                        info["max_re"], info["n_negative"]])
    print(f"Wrote {csv_path}")

    # Save evolution CSV
    ev_csv = OUT_DIR / "evolution.csv"
    with open(ev_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "status", "max_abs_v_over_amp",
                    "first_step_dev", "blew_step"])
        for name, r in results.items():
            w.writerow([name, r["status"], r["max_abs_over_amp"],
                        r["first_dev"], r["blew_step"] or ""])
    print(f"Wrote {ev_csv}")


if __name__ == "__main__":
    main()
