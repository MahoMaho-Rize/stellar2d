#!/usr/bin/env python3
"""Manufactured solution test: convergence order of cart_ale2 species remap.

The C++ kernels do the following each step:
  1. Lagrangian: nodes drift by v·dt  (X_new = X_0 + v·dt)
  2. Rezone:     nodes snap back to X_0 (uniform mesh)
  3. Remap:      swept-edge flux of mX = X·dm through (X_0 → X_new) quads
     - donor-cell:   d_mX = (mX[donor] / V0) · V_sweep
     - 2nd-order:    d_mX = (mXd + sx·ex + sy·ey) · V_sweep
                     with minmod-limited slopes on mXd = mX/V0
  4. Finalize:   X = mX_new / dm

For a uniform x-velocity v_x = const (no y-velocity), and smooth
X(x, 0) = ½[1 + sin(2πkx/L)], the exact solution is
  X(x, t) = ½[1 + sin(2πk(x − v_x·t)/L)]

We run this pure advection for T periods and measure L1 error.
Expected convergence:
  donor-cell:  L1 error ∝ Δx
  2nd-order:   L1 error ∝ Δx²

Usage:
    python3 scripts/andrassy2022/tracer_order_test.py
"""
from __future__ import annotations

import numpy as np


# ─── Limiters ────────────────────────────────────────
def minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)

def vanleer(a, b):
    denom = a + b
    safe = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    r = np.where(a * b > 0, 2 * a * b / safe, 0.0)
    return r

def mc_lim(a, b):
    avg = 0.5 * (a + b)
    return minmod(minmod(2*a, 2*b), avg)


def apply_limiter(dfL, dfR, kind):
    if kind == "minmod":  return minmod(dfL, dfR)
    if kind == "vanleer": return vanleer(dfL, dfR)
    if kind == "mc":      return mc_lim(dfL, dfR)
    if kind == "centered": return 0.5 * (dfL + dfR)   # no limiter (unstable but clean test)
    raise ValueError(kind)


# ─── Swept-edge species remap (1D, x-periodic) ────────
def swept_remap_species(X, dm, dx, v_x, dt, order=1, limiter="vanleer"):
    """Perform one ALE step on a 1D x-periodic strip.

    Geometry: cells x_c[i] = (i + 0.5)·dx.
    Node drift: nodes at x_{i+1/2} move by v_x·dt, then snap back.
    Signed swept area on east edge i (between cell i and cell i+1):
      A_s = v_x · dt                          (positive = sweeps into i+1)
    V_sweep = |A_s|, donor = i if A_s > 0 else (i+1) mod N.

    Here dm is uniform (no hydro) so we can work directly with X.
    Let V0 = dx (1D analog of cell volume).  Species mass carried:
      1st-order:  d_mX = X[donor] · dm_donor · (V_sweep / V0)
      2nd-order:  d_mX = (X[donor] + sx[donor]·e_x) · dm_donor · (V_sweep / V0)
                    where e_x = (swept_centroid - donor_centroid).

    After remap:  X_new[cell] = mX_new[cell] / dm[cell]
    """
    # Matches cart_ale2 C++ kernel structure exactly:
    #   cL = i, cR = (i+1) mod N for each east edge i
    #   As = v_x · dt (signed swept "area" — positive means sweep into cL's right)
    #   donor = cL if As > 0 else cR
    #   if As > 0:  mX_new[cL] -= d_mX;  mX_new[cR] += d_mX
    #   if As < 0:  mX_new[cR] -= d_mX;  mX_new[cL] += d_mX
    N = len(X)
    mX = X * dm
    mX_new = mX.copy()
    As = v_x * dt
    frac = abs(As) / dx
    if frac > 0.5:
        raise ValueError(f"CFL too high: frac={frac}")
    V_sweep = abs(As)

    # Pre-build slopes for all cells (needed for 2nd-order) on mXd = mX/V0 = X·dm/dx.
    if order == 2:
        mXd = X * dm / dx
        mXd_roll_L = np.roll(mXd, 1)
        mXd_roll_R = np.roll(mXd, -1)
        dfL_all = (mXd - mXd_roll_L) / dx
        dfR_all = (mXd_roll_R - mXd) / dx
        sx_all = apply_limiter(dfL_all, dfR_all, limiter)

    L = N * dx
    for i in range(N):
        cL = i
        cR = (i + 1) % N
        donor = cL if As > 0 else cR
        # Overlap centroid (mirror of swept-quad centroid across old edge).
        # 1D reduction of the general formula:
        #   c_overlap = ¾·(A_old + B_old) − ¼·(A_new + B_new)
        # In 1D with node positions Ax = Bx = (i+1)·dx, Anx = Bnx = Ax + v·dt:
        #   cx = ¾·(2·(i+1)·dx) − ¼·(2·((i+1)·dx + v·dt))
        #      = (i+1)·dx − 0.5·v·dt
        # This is INSIDE the donor cell (mirrored across the east edge).
        cx = (i + 1) * dx - 0.5 * v_x * dt
        xd = (donor + 0.5) * dx
        ex = cx - xd
        if   ex >  0.5 * L: ex -= L
        elif ex < -0.5 * L: ex += L

        if order == 1:
            d_mX = (mX[donor] / dx) * V_sweep   # (mX/V0) · V_sweep
        else:
            d_mX = (mXd[donor] + sx_all[donor] * ex) * V_sweep

        if As > 0:
            mX_new[cL] -= d_mX
            mX_new[cR] += d_mX
        else:
            mX_new[cR] -= d_mX
            mX_new[cL] += d_mX

    X_new = mX_new / dm
    return X_new


def run_advection(N, v_x=1.0, L=1.0, T_total=1.0, cfl=0.4,
                  order=1, limiter="vanleer", k=1):
    """Advect X(x, 0) = ½[1 + sin(2πkx/L)] for T_total time units.
    Return L1 error against exact solution.
    """
    dx = L / N
    dt = cfl * dx / abs(v_x)
    n_steps = int(np.ceil(T_total / dt))
    dt = T_total / n_steps  # adjust for exact endpoint

    x_c = (np.arange(N) + 0.5) * dx
    X = 0.5 * (1 + np.sin(2 * np.pi * k * x_c / L))
    dm = np.ones(N)

    for _ in range(n_steps):
        X = swept_remap_species(X, dm, dx, v_x, dt, order=order, limiter=limiter)

    X_exact = 0.5 * (1 + np.sin(2 * np.pi * k * (x_c - v_x * T_total) / L))
    L1 = np.mean(np.abs(X - X_exact))
    return L1, n_steps


def main():
    print("Manufactured advection test: X(x,0) = ½[1 + sin(2πx/L)]")
    print("v_x = 1.0, L = 1.0, T = 1.0 (one full period)")
    print()
    print(f"{'N':>5}  {'Δx':>10}  {'1st-order L1':>16}  {'order':>6} |  "
          f"{'2nd-MUSCL L1':>16}  {'order':>6}")

    Ns = [32, 64, 128, 256, 512, 1024]
    tests = [("donor-cell", 1, None),
             ("MUSCL minmod",  2, "minmod"),
             ("MUSCL vanleer", 2, "vanleer"),
             ("MUSCL MC",      2, "mc"),
             ("centered (no limiter)", 2, "centered")]
    for label, order, lim in tests:
        print(f"\n--- {label} ---")
        print(f"{'N':>5}  {'L1':>12}  {'order':>6}")
        errs = []
        for N in Ns:
            L1, _ = run_advection(N, order=order, limiter=lim)
            errs.append(L1)
            if len(errs) > 1:
                o = np.log2(errs[-2] / errs[-1])
                print(f"{N:>5}  {L1:>12.4e}  {o:>6.3f}")
            else:
                print(f"{N:>5}  {L1:>12.4e}  {'—':>6}")
        p = np.polyfit(np.log(1.0/np.array(Ns[-3:])), np.log(errs[-3:]), 1)
        print(f"  → asymptotic slope = {p[0]:.3f}")


if __name__ == "__main__":
    main()
