#!/usr/bin/env python3
"""Post-process E1 snapshots: project v̂(k_x, y) onto specific
(n_g, k_x) eigenvectors of the 1D g-mode EVP on Lane-Emden n=3/2.

For each snapshot:
  1) FFT v in x → v̂(k_x, y)
  2) For target modes (a, b, c) and any k we care about, solve the
     1D EVP -(ρ V')' + k² ρ V = (k²N²ρ/ω²) V and obtain V_{n,k}(y)
     for n_g = 1..n_max
  3) Compute c_{n,k}(t) = ⟨V_{n,k}, v̂_k⟩_ρ / ⟨V_{n,k}, V_{n,k}⟩_ρ
     using Clenshaw-Curtis quadrature
  4) Modal energy per target:  E_{n,k} = c_{n,k}² · ⟨V_{n,k}, V_{n,k}⟩_ρ
  5) Residual per k: E_{k,resid} = E_{total,k} - Σ_n E_{n,k}

Verifies:
  - Panel 1: E_a, E_b, E_c are CLEAN single-eigenmode energies
  - Panel 2: Manley-Rowe beat period matches predicted 2π/Δω
  - Panel 3: non-target residuals stay at noise
  - Panel 4: |c_{n_c,k_c}(t)| early growth = γ·t (linear in t)

Output: paper/figures/fig7_2_triad_coupled.png (overwrite) + CSV
"""
import argparse
import glob
import os
import numpy as np
import scipy.linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nonlinear_paths_infra import cgl_grid, cc_weights, bg_lane_emden


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


def solve_evp_all_modes(y, D, rho, N2, kx_phys, n_keep):
    """Return (omega[n], V[n, y]) for top n_keep g-modes at kx_phys.
    V normalised so ⟨V, V⟩_ρ_cc = 1."""
    ny = len(y)
    intr = slice(1, ny - 1)
    L = -D @ (np.diag(rho) @ D) + kx_phys ** 2 * np.diag(rho)
    R = kx_phys ** 2 * np.diag(N2 * rho)
    Li, Ri = L[intr, intr], R[intr, intr]
    lam, V_int = scipy.linalg.eig(Ri, Li)
    lam = np.real(lam); V_int = np.real(V_int)
    mask = np.isfinite(lam) & (lam > 0)
    lam, V_int = lam[mask], V_int[:, mask]
    order = np.argsort(lam)[::-1]
    lam = lam[order][:n_keep]
    V_int = V_int[:, order][:, :n_keep]
    omegas = np.sqrt(lam)
    V_all = np.zeros((n_keep, ny))
    V_all[:, 1:-1] = V_int.T
    return omegas, V_all


def norm_rho(V, rho, w_cc):
    """⟨V, V⟩_ρ with Clenshaw-Curtis weights."""
    return np.sum(w_cc * rho * V * V)


def inner_rho(V, W, rho, w_cc):
    """⟨V, W⟩_ρ."""
    return np.sum(w_cc * rho * V * W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snap_dir", default="runs/dns_expE1_snapshots", nargs="?")
    ap.add_argument("--ny",      type=int,   default=128)
    ap.add_argument("--nx",      type=int,   default=128)
    ap.add_argument("--Ly",      type=float, default=1.0)
    ap.add_argument("--Lx",      type=float, default=1.0)
    ap.add_argument("--rho_cut", type=float, default=0.1)
    ap.add_argument("--n_keep",  type=int,   default=10)
    # Target triad: (n_a, k_a), (n_b, k_b).  Mode c auto-selected as
    # the (n_c, k_c=k_a+k_b) that minimises detuning.
    ap.add_argument("--n_a",  type=int, default=6)
    ap.add_argument("--k_a",  type=int, default=1)
    ap.add_argument("--n_b",  type=int, default=3)
    ap.add_argument("--k_b",  type=int, default=5)
    ap.add_argument("--outfig", default="paper/figures/fig7_2_triad_coupled.png")
    ap.add_argument("--outcsv", default="runs/dns_expE1_projection.csv")
    args = ap.parse_args()

    # Grid / background.
    y, D = cgl_grid(args.ny, args.Ly)
    w_cc = cc_weights(args.ny, args.Ly)
    rho, N2 = bg_lane_emden(y, args.Ly, rho_cut=args.rho_cut)

    # Pre-solve EVP per k_x for the k values we care about: k=1..8.
    k_list = list(range(1, 9))
    evp = {}
    for k in k_list:
        kx_phys = k * 2.0 * np.pi / args.Lx
        omegas, V_all = solve_evp_all_modes(y, D, rho, N2, kx_phys, args.n_keep)
        # Normalise each eigenvector to ⟨V,V⟩_ρ = 1 for clean projection.
        for n in range(args.n_keep):
            nrm = norm_rho(V_all[n], rho, w_cc)
            if nrm > 0:
                V_all[n] /= np.sqrt(nrm)
        # Sign convention: positive at midpoint.
        mid = args.ny // 2
        for n in range(args.n_keep):
            if V_all[n, mid] < 0:
                V_all[n] = -V_all[n]
        evp[k] = (omegas, V_all)

    # Select c = (n_c, k_c = k_a + k_b) that minimises |ω_a + ω_b - ω_c|.
    k_c = args.k_a + args.k_b
    w_a = evp[args.k_a][0][args.n_a - 1]
    w_b = evp[args.k_b][0][args.n_b - 1]
    wc_candidates = evp[k_c][0]
    detune_abs = np.abs(w_a + w_b - wc_candidates)
    n_c_best = int(np.argmin(detune_abs)) + 1
    w_c = wc_candidates[n_c_best - 1]
    dw = w_a + w_b - w_c
    print(f"Triad: a=(n={args.n_a}, k={args.k_a}, ω={w_a:.6f})")
    print(f"       b=(n={args.n_b}, k={args.k_b}, ω={w_b:.6f})")
    print(f"       c=(n={n_c_best}, k={k_c}, ω={w_c:.6f}) — selected by min detuning")
    print(f"  ω_a + ω_b = {w_a+w_b:.6f}")
    print(f"  Δω = {dw:+.6e}  (|Δω|/ω_c = {abs(dw)/w_c*100:.3f}%)")
    T_beat = 2 * np.pi / max(abs(dw), 1e-12)
    T_a    = 2 * np.pi / w_a
    print(f"  Predicted sinc² beat period = 2π/|Δω| = {T_beat:.2f}  "
          f"= {T_beat/T_a:.1f} periods of mode a")

    # Load snapshots.
    files = sorted(glob.glob(os.path.join(args.snap_dir, "snap_*.bin")))
    if not files:
        raise RuntimeError(f"No snapshots in {args.snap_dir}")
    print(f"\nFound {len(files)} snapshots")

    # Output arrays.
    t_arr = []
    # For each k we track all n_keep modal coefficients c_{n,k}(t).
    coef = {k: [] for k in k_list}   # coef[k][t][n] = c_{n,k}
    # Non-eigenmode residual energy per k (after subtracting all n_keep modes).
    resid = {k: [] for k in k_list}

    for fn in files:
        t, u, v, b = load_snap(fn)
        # FFT v in x: v̂(k_x, y).  Use rfft with normalisation 1/nx.
        vhat = np.fft.rfft(v, axis=1) / args.nx  # (ny, nh)

        t_arr.append(t)
        for k in k_list:
            vk = vhat[:, k]   # (ny,) complex
            # Real and imaginary parts separately project; energy sum
            # contains both.
            omegas_k, V_all_k = evp[k]
            coefs_k = np.zeros(args.n_keep, dtype=complex)
            for n in range(args.n_keep):
                Vn = V_all_k[n]
                # ⟨V_n, v̂_k⟩_ρ.  Since ⟨V_n,V_n⟩_ρ = 1, coefs_k = projection
                coefs_k[n] = inner_rho(Vn, vk.real, rho, w_cc) \
                             + 1j * inner_rho(Vn, vk.imag, rho, w_cc)
            coef[k].append(coefs_k)

            # Total kx=k energy:  E_k = 2 · ⟨v̂_k, v̂_k⟩_ρ   (factor 2 = single-sided)
            E_tot_k = 2.0 * (norm_rho(vk.real, rho, w_cc)
                             + norm_rho(vk.imag, rho, w_cc))
            # Sum of eigenmode contributions.
            E_modes_k = 2.0 * np.sum(np.abs(coefs_k) ** 2)
            resid[k].append(max(E_tot_k - E_modes_k, 0.0))

    t_arr = np.array(t_arr)
    for k in k_list:
        coef[k] = np.array(coef[k])     # (nt, n_keep) complex
        resid[k] = np.array(resid[k])

    n_per = t_arr / T_a
    print(f"Samples: {len(t_arr)}, up to period {n_per[-1]:.1f}")

    # Energies of target modes (factor 2 for single-sided spectrum).
    E_a = 2.0 * np.abs(coef[args.k_a][:, args.n_a - 1]) ** 2
    E_b = 2.0 * np.abs(coef[args.k_b][:, args.n_b - 1]) ** 2
    E_c = 2.0 * np.abs(coef[k_c][:, n_c_best - 1]) ** 2

    print(f"\nAt t=0:   E_a = {E_a[0]:.3e}, E_b = {E_b[0]:.3e}, E_c = {E_c[0]:.3e}")
    print(f"At end:   E_a = {E_a[-1]:.3e}, E_b = {E_b[-1]:.3e}, E_c = {E_c[-1]:.3e}")
    print(f"E_a drift: {(E_a[-1]-E_a[0])/E_a[0]*100:+.3f}%")
    print(f"E_b drift: {(E_b[-1]-E_b[0])/E_b[0]*100:+.3f}%")
    print(f"E_b + E_c range: [{(E_b+E_c).min():.3e}, {(E_b+E_c).max():.3e}]"
          f"  ratio = {(E_b+E_c).max()/(E_b+E_c).min():.2f}")

    # ── Figure: 4 panels ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) Modal energies of (a, b, c) — clean single-eigenmode projection.
    ax = axes[0, 0]
    ax.semilogy(n_per, E_a, "-",  color="#1f77b4", lw=1.8,
                label=f"$E_a$ = $|c_{{{args.n_a},{args.k_a}}}|^2$ (pump)")
    ax.semilogy(n_per, E_b, "-",  color="#d62728", lw=1.8,
                label=f"$E_b$ = $|c_{{{args.n_b},{args.k_b}}}|^2$ (seeded partner)")
    ax.semilogy(n_per, np.clip(E_c, 1e-40, None), "-",
                color="#2ca02c", lw=2.0,
                label=f"$E_c$ = $|c_{{{n_c_best},{k_c}}}|^2$ (grows from noise)")
    ax.set_xlabel("periods of mode $a$")
    ax.set_ylabel("eigenmode energy  $|c_{n,k}|^2$")
    ax.set_title("Clean triad energies via EVP projection")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # (2) Manley-Rowe b ↔ c with predicted beat period overlay.
    ax = axes[0, 1]
    ax.plot(n_per, E_b, "-", color="#d62728", lw=1.5, label="$E_b$")
    ax.plot(n_per, E_c, "-", color="#2ca02c", lw=1.5, label="$E_c$")
    ax.plot(n_per, E_b + E_c, "--", color="k", lw=1.5, label="$E_b + E_c$")
    # Mark predicted beat period (first zero of sinc² at T_beat/2).
    if T_beat < 10 * n_per[-1]:
        for m in range(1, int(n_per[-1] / (T_beat / T_a)) + 2):
            ax.axvline(m * T_beat / T_a, color="gray", ls=":", lw=0.7,
                       alpha=0.6)
    ax.axhline(E_b[0], color="#d62728", ls=":", lw=0.6, alpha=0.5,
               label=f"$E_b(0) = {E_b[0]:.2e}$")
    ax.set_yscale("log")
    ax.set_xlabel("periods of mode $a$")
    ax.set_ylabel("energy")
    ax.set_title(f"Manley–Rowe $b \\leftrightarrow c$  "
                 f"(Δω/ω={abs(dw)/w_c*100:.3f}%, predicted beat = "
                 f"{T_beat/T_a:.1f} periods, grey dashes)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # (3) Non-target residual energies — true selectivity.
    ax = axes[1, 0]
    # Per-k residual (after subtracting n_keep eigenmode contribution).
    for k, cl in [(2, "#ff7f0e"), (3, "#9467bd"),
                   (4, "#8c564b"), (7, "#17becf")]:
        ax.semilogy(n_per, np.clip(resid[k], 1e-30, None),
                    "-", color=cl, lw=1.0, label=f"$k={k}$ residual")
    ax.set_ylim(bottom=1e-25)
    ax.set_xlabel("periods of mode $a$")
    ax.set_ylabel("non-eigenmode residual energy")
    ax.set_title("Non-target residuals — true selectivity metric")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (4) |c_c(t)| vs t, log-log, expect linear rise.
    ax = axes[1, 1]
    c_amp = np.sqrt(np.clip(E_c, 1e-40, None))
    # Noise floor = |c_c(0)|.
    noise = c_amp[0]
    # Early regime: find first 10 periods of growth above 3×noise.
    mask = (n_per > 0.5) & (n_per < 25) & (c_amp > 3 * noise)
    if mask.sum() > 3:
        t_fit = n_per[mask]
        c_fit = c_amp[mask]
        # Linear fit of log|c| vs log(t).  Should give slope ≈ 1.
        coeffs = np.polyfit(np.log(t_fit), np.log(c_fit), 1)
        ax.loglog(n_per[1:], c_amp[1:], "o",
                  color="#2ca02c", ms=3, label="$|c_c(t)|$ from projection")
        tref = np.linspace(t_fit.min(), t_fit.max(), 30)
        ax.loglog(tref, np.exp(coeffs[1]) * tref ** coeffs[0],
                  "k--", lw=1.3, label=f"fit $\\propto t^{{{coeffs[0]:.2f}}}$")
        ax.loglog(tref, c_fit[0] * tref / t_fit[0],
                  "r:", lw=1.0, label="$\\propto t$ (triad theory)")
    ax.set_xlabel("periods of mode $a$  (log)")
    ax.set_ylabel("$|c_c(t)|$")
    ax.set_title("Mode-$c$ amplitude growth  (clean projection)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # Suptitle.
    plt.suptitle(
        f"DNS E1 (revised projection): 3-wave triad  "
        f"$a=(n_g{{=}}{args.n_a}, k_x{{=}}{args.k_a})$ + "
        f"$b=(n_g{{=}}{args.n_b}, k_x{{=}}{args.k_b})$ $\\to$ "
        f"$c=(n_g{{=}}{n_c_best}, k_x{{=}}{k_c})$\n"
        f"$\\omega_a+\\omega_b = {w_a+w_b:.4f}$, $\\omega_c = {w_c:.4f}$, "
        f"$\\Delta\\omega = {dw:+.2e}$.  "
        f"Lane-Emden $n=3/2$, $\\rho_{{\\rm cut}}={args.rho_cut}$, 128×128.",
        fontsize=10)
    plt.tight_layout()
    plt.savefig(args.outfig, dpi=140, bbox_inches="tight")
    print(f"\nsaved → {args.outfig}")

    # CSV with all the projection coefficients, for future inspection.
    import csv
    with open(args.outcsv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["t", "n_periods", "E_a", "E_b", "E_c", "E_b_plus_E_c"]
        for k in k_list:
            header.append(f"resid_k{k}")
            for n in range(args.n_keep):
                header.append(f"|c_{n+1}_{k}|")
        w.writerow(header)
        for i, t in enumerate(t_arr):
            row = [t, n_per[i], E_a[i], E_b[i], E_c[i], E_b[i]+E_c[i]]
            for k in k_list:
                row.append(resid[k][i])
                for n in range(args.n_keep):
                    row.append(abs(coef[k][i, n]))
            w.writerow(row)
    print(f"saved → {args.outcsv}")


if __name__ == "__main__":
    main()
