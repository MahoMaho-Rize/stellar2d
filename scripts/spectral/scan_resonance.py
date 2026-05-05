#!/usr/bin/env python3
"""Scan Lane-Emden n=3/2 g-mode spectrum for three-wave triad candidates.

Selection rules (anelastic 2D, periodic x, Dirichlet y):
  - k_a + k_b = k_c    (wavenumber, integer index on periodic x)
  - ω_a + ω_b ≈ ω_c    (frequency, allow detuning)
  - y-overlap  <φ_a φ_b φ_c>_ρ  non-zero

This script enumerates g-modes at a given (ρ_cut, Ny, rho_cut, kx_int ∈ 1..K),
builds the table of (n_g, k_x, ω), then lists:

  [TRIAD]  best (a, b, c) triples by frequency detuning
  [PSI]    best (parent, daughter) pairs with ω_p ≈ 2 ω_d (same k)
  [2MODE]  best commensurate pairs (nearly equal ω with matched k)

For each candidate the host-space overlap integral with ρ-weighted CC
quadrature is printed as a proxy for the three-wave coupling strength.
"""
import argparse
import numpy as np
import scipy.linalg
from nonlinear_paths_infra import cgl_grid, cc_weights, bg_lane_emden


def solve_gmode_per_kx(y, D, rho, N2, kx_int, Lx, n_modes_keep):
    """Returns list of (omega, V_full) for the top n_modes_keep g-modes at kx_int."""
    ny = len(y)
    kx_phys = kx_int * 2.0 * np.pi / Lx
    intr = slice(1, ny - 1)
    L = -D @ (np.diag(rho) @ D) + kx_phys ** 2 * np.diag(rho)
    R = kx_phys ** 2 * np.diag(N2 * rho)
    Li, Ri = L[intr, intr], R[intr, intr]
    lam, V_int = scipy.linalg.eig(Ri, Li)
    lam = np.real(lam); V_int = np.real(V_int)
    mask = np.isfinite(lam) & (lam > 0)
    lam, V_int = lam[mask], V_int[:, mask]
    order = np.argsort(lam)[::-1]
    lam = lam[order][:n_modes_keep]
    V_int = V_int[:, order][:, :n_modes_keep]
    results = []
    for k in range(len(lam)):
        V_y = np.zeros(ny)
        V_y[1:-1] = V_int[:, k]
        # normalise: ρ-weighted L² norm = 1
        nrm = np.sqrt(np.sum(y_wt_ccρ * V_y ** 2))   # uses module-level y_wt_ccρ
        V_y /= max(nrm, 1e-30)
        # deterministic sign: positive at V(midpoint)
        if V_y[ny // 2] < 0:
            V_y = -V_y
        results.append((float(np.sqrt(lam[k])), V_y))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ny",      type=int,   default=128)
    ap.add_argument("--Ly",      type=float, default=1.0)
    ap.add_argument("--Lx",      type=float, default=1.0)
    ap.add_argument("--rho_cut", type=float, default=0.1)
    ap.add_argument("--kmax",    type=int,   default=6,
                    help="scan kx_int = 1..kmax")
    ap.add_argument("--nmax",    type=int,   default=5,
                    help="keep top n_g = 1..nmax per kx")
    ap.add_argument("--detune",  type=float, default=0.02,
                    help="max |ω mismatch| / ω for listing")
    ap.add_argument("--top",     type=int,   default=10,
                    help="list top N candidates per category")
    args = ap.parse_args()

    # Grid + background
    y, D = cgl_grid(args.ny, args.Ly)
    w_cc = cc_weights(args.ny, args.Ly)
    rho, N2 = bg_lane_emden(y, args.Ly, rho_cut=args.rho_cut)

    # ρ-weighted quadrature weights for L² inner product.
    global y_wt_ccρ
    y_wt_ccρ = w_cc * rho

    # Enumerate modes.
    modes = []   # list of (n_g, kx_int, omega, V_y)
    for kx_int in range(1, args.kmax + 1):
        lst = solve_gmode_per_kx(y, D, rho, N2, kx_int, args.Lx, args.nmax)
        for n_g, (omega, V_y) in enumerate(lst, start=1):
            modes.append((n_g, kx_int, omega, V_y))

    print(f"Lane-Emden n=3/2, ρ_cut={args.rho_cut}, Ny={args.ny}, Lx={args.Lx}")
    print(f"Scanned  kx_int ∈ 1..{args.kmax},  n_g ∈ 1..{args.nmax}  → "
          f"{len(modes)} g-modes\n")

    # Print the spectrum table.
    print("── g-mode spectrum (ω in units of √(gr/R) if ρ₀~1) ──")
    print("  n_g  kx   ω           period")
    for (n_g, kx_int, omega, _) in modes:
        T = 2 * np.pi / omega
        print(f"  {n_g:3d}  {kx_int:3d}  {omega:.6f}  {T:.4f}")
    print()

    # ── Triad candidates: k_a + k_b = k_c, ω_a + ω_b ≈ ω_c ───────────
    # Compute ρ-weighted 3-overlap ⟨φ_a φ_b φ_c⟩_ρ = Σ w_cc ρ V_a V_b V_c.
    def overlap3(Va, Vb, Vc):
        return float(np.sum(w_cc * rho * Va * Vb * Vc))

    triads = []
    for ia in range(len(modes)):
        na, ka, wa, Va = modes[ia]
        for ib in range(ia, len(modes)):    # a ≤ b (unordered)
            nb, kb, wb, Vb = modes[ib]
            kc = ka + kb
            if kc > args.kmax: continue
            # find c candidates
            for ic in range(len(modes)):
                nc, kc_, wc, Vc = modes[ic]
                if kc_ != kc: continue
                detune = abs(wa + wb - wc) / wc
                if detune > args.detune: continue
                V = overlap3(Va, Vb, Vc)
                triads.append({
                    "a": (na, ka, wa), "b": (nb, kb, wb), "c": (nc, kc_, wc),
                    "detune": detune, "V_abc": abs(V),
                })
    # Rank by (small detune) × (large overlap): maximize V/detune.
    triads.sort(key=lambda t: t["detune"] - 1.0 * np.log10(t["V_abc"] + 1e-20))
    print(f"── [TRIAD]  ω_a + ω_b ≈ ω_c,  k_a + k_b = k_c   (detune ≤ {args.detune}) ──")
    print("   a: (n,k,ω)              b: (n,k,ω)              c: (n,k,ω)              detune     |V_abc|")
    for t in triads[:args.top]:
        na, ka, wa = t["a"]; nb, kb, wb = t["b"]; nc, kc, wc = t["c"]
        print(f"  ({na},{ka},{wa:.4f})   ({nb},{kb},{wb:.4f})   "
              f"({nc},{kc},{wc:.4f})   {t['detune']:.2e}   {t['V_abc']:.3e}")
    print(f"  ({len(triads)} triad candidates total)\n")

    # ── PSI candidates: ω_parent ≈ 2 ω_daughter, same k_x, overlap ⟨V_p V_d²⟩ ─
    psi = []
    for ip in range(len(modes)):
        np_, kp, wp, Vp = modes[ip]
        for jd in range(len(modes)):
            nd, kd, wd, Vd = modes[jd]
            if kp != kd: continue      # same k (simplest 1D PSI)
            if wd >= wp: continue
            detune = abs(wp - 2.0 * wd) / wp
            if detune > args.detune: continue
            V = overlap3(Vp, Vd, Vd)
            psi.append({
                "p": (np_, kp, wp), "d": (nd, kd, wd),
                "detune": detune, "V_pdd": abs(V),
            })
    # Also allow  k_p = 2 k_d (spatial subharmonic)  to broaden the search.
    for ip in range(len(modes)):
        np_, kp, wp, Vp = modes[ip]
        for jd in range(len(modes)):
            nd, kd, wd, Vd = modes[jd]
            if kp != 2 * kd: continue
            if wd >= wp: continue
            detune = abs(wp - 2.0 * wd) / wp
            if detune > args.detune: continue
            V = overlap3(Vp, Vd, Vd)
            psi.append({
                "p": (np_, kp, wp), "d": (nd, kd, wd),
                "detune": detune, "V_pdd": abs(V), "note": "k_p=2k_d"
            })
    psi.sort(key=lambda t: t["detune"] - 1.0 * np.log10(t["V_pdd"] + 1e-20))
    print(f"── [PSI]    ω_p ≈ 2 ω_d   (detune ≤ {args.detune}) ──")
    print("   parent                   daughter                 detune     |V_pdd|  note")
    for t in psi[:args.top]:
        np_, kp, wp = t["p"]; nd, kd, wd = t["d"]
        note = t.get("note", "k_p=k_d")
        print(f"  ({np_},{kp},{wp:.4f})   ({nd},{kd},{wd:.4f})   "
              f"{t['detune']:.2e}   {t['V_pdd']:.3e}  {note}")
    print(f"  ({len(psi)} PSI candidates total)\n")

    # ── 2MODE commensurate: ω_a ≈ ω_b with same k_x (avoided crossings)
    #    or  ω_a / ω_b ≈ p/q  for small integers.
    two = []
    for ia in range(len(modes)):
        na, ka, wa, Va = modes[ia]
        for ib in range(ia + 1, len(modes)):
            nb, kb, wb, Vb = modes[ib]
            ratio = wa / wb if wb > 0 else 0
            # Find nearest low-order rational p/q with q ≤ 4.
            best_pq = None; best_det = 1.0
            for q in (1, 2, 3, 4):
                for p in range(1, 4 * q + 1):
                    r = p / q
                    d = abs(ratio - r) / r
                    if d < best_det:
                        best_det = d; best_pq = (p, q)
            if best_det > args.detune: continue
            two.append({
                "a": (na, ka, wa), "b": (nb, kb, wb),
                "ratio": ratio, "pq": best_pq, "detune": best_det,
            })
    two.sort(key=lambda t: t["detune"])
    print(f"── [2MODE]  ω_a : ω_b ≈ p/q   (detune ≤ {args.detune}, q ≤ 4) ──")
    print("   a                        b                        ratio     p/q     detune")
    for t in two[:args.top]:
        na, ka, wa = t["a"]; nb, kb, wb = t["b"]
        p, q = t["pq"]
        print(f"  ({na},{ka},{wa:.4f})   ({nb},{kb},{wb:.4f})   "
              f"{t['ratio']:.4f}   {p}/{q}   {t['detune']:.2e}")
    print(f"  ({len(two)} 2-mode candidates total)\n")


if __name__ == "__main__":
    main()
