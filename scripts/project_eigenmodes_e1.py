#!/usr/bin/env python3
"""Post-process DNS E1 snapshots using CUDA's TANH-warped y-grid.

The CUDA solver uses ANSL_COORD_MAP=tanh ANSL_COORD_BETA=2, which warps
the CGL nodes and rescales the differentiation matrix by 1/y'(s).  Any
comparison with CUDA output MUST use the same warped grid; using standard
CGL via nonlinear_paths_infra.cgl_grid gives eigenvectors that look
"different", but that's just the two grids having different sampling.

This script:
  1. Reconstructs CUDA's TANH grid (y_asc, D_y) and (ρ, N²) at those nodes.
  2. Solves GEVP R V = ω² L V with ρ-weighted CC quadrature on THIS grid.
  3. For each snapshot, projects v̂(k_x, y) onto the full eigenbasis and
     tracks target triad (a, b, c) amplitudes.
  4. Uses R-norm (ρN²-weighted) inner product — that's the GEVP's
     natural orthogonality.
  5. Verifies Manley–Rowe beat period against ω_a + ω_b − ω_c.
"""
import argparse
import os
import glob
import numpy as np
import scipy.linalg
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Grid and background reconstruction (TANH-warped) ────────────────
def cgl_tanh_grid(ny, Ly, beta=2.0):
    """Standard CGL in s ∈ [0, Ly], then TANH-warp to y ∈ [0, Ly].
    Returns y_asc, D_y (ny × ny, operator d/dy on y-grid), dy_ds."""
    N = ny - 1
    x = np.cos(np.pi * np.arange(N + 1) / N)      # [-1, 1] descending
    idx = np.arange(N, -1, -1)
    s_asc = (1.0 + x[idx]) * Ly / 2.0             # [0, Ly] ascending

    # TANH warp.
    u = beta * (2.0 * s_asc / Ly - 1.0)
    y_asc = 0.5 * Ly + 0.5 * Ly * np.tanh(u) / np.tanh(beta)
    dy_ds = beta * (1.0 / np.cosh(u)) ** 2 / np.tanh(beta)

    # Standard D_raw on [-1, 1], then scale to [0, Ly] and reorder ascending.
    c = np.ones(N + 1); c[0] = 2.0; c[-1] = 2.0
    c = c * ((-1.0) ** np.arange(N + 1))
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D_raw = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D_raw = D_raw - np.diag(D_raw.sum(axis=1))

    scale = 2.0 / Ly
    D_asc = scale * D_raw[np.ix_(idx, idx)]

    # Chain-rule row-scale: d/dy = (1/y'(s)) d/ds
    D_y = D_asc.copy()
    for i in range(ny):
        D_y[i, :] /= max(dy_ds[i], 1e-30)

    return y_asc, D_y, dy_ds


def cc_weights_nonuniform(ny, Ly, dy_ds):
    """Clenshaw-Curtis weights mapped to the warped y-grid.
    Standard CC weights on s ∈ [0, Ly] times dy/ds at each node give
    the quadrature weights for ∫ f(y) dy = Σ w_cc(s_k)·y'(s_k)·f(y_k)."""
    N = ny - 1
    w = np.zeros(N + 1)
    for k in range(N + 1):
        s = 0.0
        J = N // 2
        for j in range(1, J + 1):
            b = 2.0 if (2 * j != N) else 1.0
            s += b / (4.0 * j * j - 1) * np.cos(2.0 * j * k * np.pi / N)
        w[k] = (1.0 - s) * 2.0 / float(N)
    w[0] /= 2.0; w[-1] /= 2.0
    # Reverse to ascending, scale to [0, Ly], then multiply by dy/ds.
    w_ascending = w[::-1] * (Ly / 2.0)
    return w_ascending * dy_ds


def bg_lane_emden_at_y(y_grid, Ly, rho_cut=0.1):
    """Lane-Emden n=3/2 ρ(y), N²(y) interpolated onto given (possibly
    non-uniform) y_grid."""
    sol = solve_ivp(
        lambda xi, st: [st[1],
                        -max(st[0], 0.0) ** 1.5 - 2.0 * st[1] / max(xi, 1e-8)],
        [1e-6, 4.0], [1.0, 0.0], dense_output=True,
        events=lambda t, st: st[0], max_step=1e-3,
    )
    xi_s = sol.t_events[0][0]
    xi = np.linspace(1e-6, xi_s, 4000)
    rho_full = np.clip(sol.sol(xi)[0], 0.0, None) ** 1.5
    mask = rho_full > rho_cut
    xi_lo, xi_hi = xi[mask].min(), xi[mask].max()
    xi_q = xi_lo + (y_grid / Ly) * (xi_hi - xi_lo)
    rho = np.clip(np.interp(xi_q, xi, rho_full), rho_cut, None)
    drho = np.gradient(rho, y_grid, edge_order=2)
    N2 = np.maximum(-drho / rho, 0.0)
    return rho, N2


def solve_evp_tanh(y, D, rho, N2, kx_phys, n_keep):
    """GEVP on TANH grid: L V = (1/ω²) R V, return omega and orthonormalised V."""
    ny = len(y)
    intr = slice(1, ny - 1)
    L = -D @ (np.diag(rho) @ D) + kx_phys ** 2 * np.diag(rho)
    R = kx_phys ** 2 * np.diag(N2 * rho)
    Li, Ri = L[intr, intr], R[intr, intr]
    lam, V_int = scipy.linalg.eig(Ri, Li)
    lam = np.real(lam); V_int = np.real(V_int)
    m = np.isfinite(lam) & (lam > 0)
    lam, V_int = lam[m], V_int[:, m]
    order = np.argsort(lam)[::-1]
    lam = lam[order][:n_keep]
    V_int = V_int[:, order][:, :n_keep]
    omegas = np.sqrt(lam)
    V_full = np.zeros((ny, n_keep))
    V_full[1:-1, :] = V_int
    # Sign-fix at midpoint.
    mid = ny // 2
    for n in range(n_keep):
        if V_full[mid, n] < 0:
            V_full[:, n] = -V_full[:, n]
    return omegas, V_full


# ── I/O ─────────────────────────────────────────────────────────────
def load_snap(path):
    with open(path, "rb") as f:
        hdr = np.fromfile(f, dtype=np.int32, count=2)
        ny, nx = int(hdr[0]), int(hdr[1])
        t = np.fromfile(f, dtype=np.float64, count=1)[0]
        n = ny * nx
        u = np.fromfile(f, dtype=np.float32, count=n).reshape(ny, nx)
        v = np.fromfile(f, dtype=np.float32, count=n).reshape(ny, nx)
        b = np.fromfile(f, dtype=np.float32, count=n).reshape(ny, nx)
    return t, u, v, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snap_dir")
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--Ly", type=float, default=1.0)
    ap.add_argument("--Lx", type=float, default=1.0)
    ap.add_argument("--rho_cut", type=float, default=0.1)
    ap.add_argument("--tanh_beta", type=float, default=2.0)
    ap.add_argument("--n_keep", type=int, default=10)
    ap.add_argument("--n_a", type=int, default=6)
    ap.add_argument("--k_a", type=int, default=1)
    ap.add_argument("--n_b", type=int, default=3)
    ap.add_argument("--k_b", type=int, default=5)
    ap.add_argument("--outfig", default="paper/figures/fig7_2_triad_coupled.png")
    ap.add_argument("--outcsv", default="runs/dns_expE1_projection_tanh.csv")
    args = ap.parse_args()

    # TANH grid and background.
    y, D, dy_ds = cgl_tanh_grid(args.ny, args.Ly, args.tanh_beta)
    w_cc = cc_weights_nonuniform(args.ny, args.Ly, dy_ds)
    rho, N2 = bg_lane_emden_at_y(y, args.Ly, args.rho_cut)
    print(f"TANH grid: min(dy)={np.min(np.diff(y)):.2e}, "
          f"max(dy)={np.max(np.diff(y)):.2e}")

    # EVP per k_x.
    k_list = list(range(1, 9))
    evp = {}
    for k in k_list:
        kx_phys = k * 2.0 * np.pi / args.Lx
        om, V = solve_evp_tanh(y, D, rho, N2, kx_phys, args.n_keep)
        # ρN²-normalize each eigenvector to <V,V>_{ρN²} = 1 (R-norm)
        rho_N2 = rho * N2
        for n in range(args.n_keep):
            nrm = np.sqrt(np.sum(w_cc * rho_N2 * V[:, n] ** 2))
            if nrm > 0:
                V[:, n] /= nrm
        evp[k] = (om, V)

    # Select c = (n_c, k_c = k_a + k_b) minimizing |ω_a + ω_b - ω_c|.
    k_c = args.k_a + args.k_b
    w_a = evp[args.k_a][0][args.n_a - 1]
    w_b = evp[args.k_b][0][args.n_b - 1]
    wc_cands = evp[k_c][0]
    detune_abs = np.abs(w_a + w_b - wc_cands)
    n_c_best = int(np.argmin(detune_abs)) + 1
    w_c = wc_cands[n_c_best - 1]
    dw = w_a + w_b - w_c
    T_a = 2 * np.pi / w_a
    T_beat = 2 * np.pi / max(abs(dw), 1e-12)

    print(f"\nTriad (TANH grid):")
    print(f"  a = (n={args.n_a}, k={args.k_a}, ω={w_a:.6f})")
    print(f"  b = (n={args.n_b}, k={args.k_b}, ω={w_b:.6f})")
    print(f"  c = (n={n_c_best}, k={k_c}, ω={w_c:.6f})  [auto-selected]")
    print(f"  ω_a + ω_b = {w_a + w_b:.6f}")
    print(f"  Δω = {dw:+.6e}   (|Δω|/ω_c = {abs(dw)/w_c*100:.4f}%)")
    print(f"  T_beat = 2π/|Δω| = {T_beat:.2f}  ({T_beat/T_a:.1f} periods of mode a)")

    # Load snapshots.
    files = sorted(glob.glob(os.path.join(args.snap_dir, "snap_*.bin")))
    if not files:
        raise RuntimeError(f"No snapshots in {args.snap_dir}")
    print(f"\nLoading {len(files)} snapshots...")

    t_arr = []
    coef = {k: [] for k in k_list}
    resid = {k: [] for k in k_list}

    rho_N2 = rho * N2

    for fn in files:
        t, u, v, b = load_snap(fn)
        vhat = np.fft.rfft(v, axis=1) / args.nx
        t_arr.append(t)
        for k in k_list:
            vk = vhat[:, k]
            om_k, V_k = evp[k]
            cs = np.zeros(args.n_keep, dtype=complex)
            for n in range(args.n_keep):
                Vn = V_k[:, n]
                # R-norm projection: c_n = <V_n, v>_{ρN²} (V_n already R-unit).
                cr = np.sum(w_cc * rho_N2 * Vn * vk.real)
                ci = np.sum(w_cc * rho_N2 * Vn * vk.imag)
                cs[n] = cr + 1j * ci
            coef[k].append(cs)

            # Total kx=k R-weighted energy and residual.
            E_tot_k = (np.sum(w_cc * rho_N2 * vk.real ** 2)
                       + np.sum(w_cc * rho_N2 * vk.imag ** 2))
            E_modes_k = np.sum(np.abs(cs) ** 2)
            resid[k].append(max(E_tot_k - E_modes_k, 0.0))

    t_arr = np.array(t_arr)
    for k in k_list:
        coef[k] = np.array(coef[k])
        resid[k] = np.array(resid[k])

    n_per = t_arr / T_a
    E_a = np.abs(coef[args.k_a][:, args.n_a - 1]) ** 2
    E_b = np.abs(coef[args.k_b][:, args.n_b - 1]) ** 2
    E_c = np.abs(coef[k_c][:, n_c_best - 1]) ** 2

    print(f"\nProjection results:")
    print(f"  E_a(0) = {E_a[0]:.3e}, E_a(end) = {E_a[-1]:.3e}   "
          f"drift = {(E_a[-1]-E_a[0])/E_a[0]*100:+.3f}%")
    print(f"  E_b(0) = {E_b[0]:.3e}, E_b(end) = {E_b[-1]:.3e}")
    print(f"  E_c(0) = {E_c[0]:.3e}, E_c(end) = {E_c[-1]:.3e}")
    print(f"  E_b + E_c range: [{(E_b+E_c).min():.3e}, "
          f"{(E_b+E_c).max():.3e}]  ratio = "
          f"{(E_b+E_c).max()/max((E_b+E_c).min(),1e-50):.2f}")

    # Detect E_b dip locations.
    from scipy.signal import find_peaks
    inv = -np.log10(E_b + 1e-50)
    peaks, _ = find_peaks(inv, prominence=0.3)
    if len(peaks) >= 2:
        dip_spacing = np.diff(n_per[peaks])
        print(f"\n  E_b dip positions (periods of mode a): "
              f"{n_per[peaks].round(1).tolist()}")
        print(f"  Dip spacing: {dip_spacing.round(2).tolist()}")
        print(f"  Mean spacing: {np.mean(dip_spacing):.2f}  "
              f"vs predicted T_beat/2 = {T_beat/T_a/2:.2f}")

    # ── Figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) Three modal energies.
    ax = axes[0, 0]
    ax.semilogy(n_per, E_a, "-", color="#1f77b4", lw=1.8,
                label=f"$E_a = |c_{{{args.n_a},{args.k_a}}}|^2$ (pump)")
    ax.semilogy(n_per, E_b, "-", color="#d62728", lw=1.8,
                label=f"$E_b = |c_{{{args.n_b},{args.k_b}}}|^2$ (seeded)")
    ax.semilogy(n_per, np.clip(E_c, 1e-50, None), "-",
                color="#2ca02c", lw=2.0,
                label=f"$E_c = |c_{{{n_c_best},{k_c}}}|^2$ (grows from noise)")
    ax.set_xlabel("periods of mode $a$")
    ax.set_ylabel("eigenmode energy (R-norm)")
    ax.set_title("Clean triad via TANH-grid EVP projection")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # (2) Manley-Rowe.
    ax = axes[0, 1]
    ax.plot(n_per, E_b, "-", color="#d62728", lw=1.5, label="$E_b$")
    ax.plot(n_per, E_c, "-", color="#2ca02c", lw=1.5, label="$E_c$")
    ax.plot(n_per, E_b + E_c, "--", color="k", lw=1.8,
            label="$E_b + E_c$ (Manley-Rowe)")
    # Predicted beat half-period markers.
    n_marks = int(2 * n_per[-1] / (T_beat / T_a)) + 2
    for m in range(1, n_marks):
        xp = m * T_beat / (2 * T_a)
        if xp < n_per[-1] * 1.05:
            ax.axvline(xp, color="gray", ls=":", lw=0.7, alpha=0.6,
                       label=("predicted dip (T_beat/2)" if m == 1 else None))
    ax.set_yscale("log")
    ax.set_xlabel("periods of mode $a$")
    ax.set_ylabel("energy")
    ax.set_title(f"Manley–Rowe: Δω/ω = {abs(dw)/w_c*100:.3f}%, "
                 f"predicted dip period = {T_beat/T_a/2:.1f} periods")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (3) Non-target residuals.
    ax = axes[1, 0]
    for k, cl, lab in [(2, "#ff7f0e", "k=2"), (3, "#9467bd", "k=3"),
                        (4, "#8c564b", "k=4"), (7, "#17becf", "k=7")]:
        ax.semilogy(n_per, np.clip(resid[k], 1e-35, None),
                    "-", color=cl, lw=1.0, label=lab + " residual")
    ax.set_ylim(bottom=1e-32)
    ax.set_xlabel("periods of mode $a$")
    ax.set_ylabel("non-eigenmode residual (R-norm)")
    ax.set_title("Non-target residuals — selectivity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (4) |c_c(t)| linear-axis view.  We avoid log-log slope fits on
    # near-round-off data; if the signal is above IC floor the envelope
    # is easy to read by eye against the amp-a·amp-b scale reference.
    # The real γ is best recovered from the E_b dip spacing (panel 2).
    ax = axes[1, 1]
    c_amp = np.sqrt(np.clip(E_c, 0.0, None))
    ax.plot(n_per, c_amp, "o-", color="#2ca02c", ms=3, lw=0.8,
            label="$|c_c(t)|$ (EVP projection)")
    noise = c_amp[0] if c_amp[0] > 0 else float(c_amp[1:5].mean())
    ax.axhline(noise, color="k", ls=":", lw=0.8,
               label=f"IC / round-off floor = {noise:.2e}")
    quarter_beat = T_beat / T_a / 4
    if quarter_beat < n_per[-1]:
        ax.axvline(quarter_beat, color="gray", ls="--", lw=0.8,
                   label=f"T_beat/4 = {quarter_beat:.1f} periods")
    ax.set_xlabel("periods of mode $a$")
    ax.set_ylabel("$|c_c(t)|$")
    ax.set_title("Mode-$c$ amplitude (linear axis, no slope fit)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"DNS E1 (TANH-aware projection): 3-wave triad  "
        f"$a=(n_g{{=}}{args.n_a}, k_x{{=}}{args.k_a})$ + "
        f"$b=(n_g{{=}}{args.n_b}, k_x{{=}}{args.k_b})$ $\\to$ "
        f"$c=(n_g{{=}}{n_c_best}, k_x{{=}}{k_c})$\n"
        f"ω_a+ω_b = {w_a+w_b:.4f}, ω_c = {w_c:.4f}, Δω = {dw:+.2e}.  "
        f"Lane-Emden n=3/2, ρ_cut={args.rho_cut}, TANH β={args.tanh_beta}, 128×128.",
        fontsize=10)
    plt.tight_layout()
    plt.savefig(args.outfig, dpi=140, bbox_inches="tight")
    print(f"\nsaved → {args.outfig}")

    # Save CSV for future work.
    header = ["t", "n_periods", "E_a", "E_b", "E_c", "E_b_plus_E_c",
              "resid_k2", "resid_k3", "resid_k4", "resid_k7"]
    import csv
    with open(args.outcsv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, t in enumerate(t_arr):
            w.writerow([t, n_per[i], E_a[i], E_b[i], E_c[i], E_b[i]+E_c[i],
                        resid[2][i], resid[3][i], resid[4][i], resid[7][i]])
    print(f"saved → {args.outcsv}")


if __name__ == "__main__":
    main()
