#!/usr/bin/env python3
"""
Verify that the C++/CUDA AnelasticSLSolver's SL basis matches the Python
reference from scripts/reduced_pressure_chebyshev.py.

Compares eigenvalues mu_n and the |W̃|_max signature on a common ny grid.

Usage:
  ./build/stellar2d run --solver anelastic_sl --test sl_basis_check \
      --nr N --ntheta 256 --ps-Lx L --ps-Ly L --run-base /tmp/ansl_test

  python scripts/verify_sl_basis_cuda.py /tmp/ansl_test/sl_basis_check_*/sl_basis.csv
"""
import sys
import glob
from pathlib import Path
import numpy as np
import scipy.integrate


def solve_lane_emden(n=1.5, n_pts=5000):
    def rhs(xi, y):
        theta, dtheta = y
        if xi < 1e-10:
            return [dtheta, -theta**n / 3.0]
        theta_pow = np.sign(theta)*np.abs(theta)**n if theta >= 0 else 0.0
        return [dtheta, -2.0/xi*dtheta - theta_pow]
    def ev(xi, y): return y[0]
    ev.terminal = True; ev.direction = -1
    sol = scipy.integrate.solve_ivp(
        rhs, [1e-6, 10.0], [1-1e-12, 0],
        events=ev, max_step=0.01, rtol=1e-10, atol=1e-12, dense_output=True)
    xi_1 = sol.t_events[0][0]
    xi = np.linspace(1e-5, xi_1*0.999, n_pts)
    theta = sol.sol(xi)[0]
    return xi/xi_1, np.abs(theta)**1.5


def cheb(N):
    if N == 0:
        return np.zeros((1,1)), np.array([1.0])
    x = np.cos(np.pi * np.arange(N+1) / N)
    c = np.ones(N+1); c[0]=2; c[-1]=2
    c *= (-1.0)**np.arange(N+1)
    X = np.tile(x, (N+1,1)).T
    dX = X - X.T
    D = np.outer(c, 1.0/c) / (dX + np.eye(N+1))
    D -= np.diag(D.sum(axis=1))
    return D, x


def parse_cuda_csv(path):
    with open(path) as f:
        lines = f.readlines()
    section = None
    mu = []; y = []; W = []
    ny = None; n_modes = None
    for line in lines:
        line = line.strip()
        if line.startswith("# ny="):
            # "# ny=256, n_modes=128, Ly=0.941, background=lane_emden_1_5"
            parts = [p.strip() for p in line[2:].split(",")]
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    if k.strip() == "ny": ny = int(v)
                    if k.strip() == "n_modes": n_modes = int(v)
            continue
        if line.startswith("# mu_n"): section = "mu"; continue
        if line.startswith("# y_cgl"): section = "y"; continue
        if line.startswith("# W_tilde"): section = "W"; continue
        if line.startswith("#"): continue
        if not line: continue
        val = float(line)
        if section == "mu": mu.append(val)
        elif section == "y": y.append(val)
        elif section == "W": W.append(val)
    return ny, n_modes, np.array(mu), np.array(y), np.array(W)


def python_reference(ny, rho_cut=0.01):
    """Recompute mu_n, y_cgl, W_tilde using the same algorithm as C++."""
    r_ode, rho_ode = solve_lane_emden()
    mask = rho_ode > rho_cut
    r_in, rho_in = r_ode[mask], rho_ode[mask]
    r_lo, r_hi = float(r_in[0]), float(r_in[-1])
    Ly = r_hi - r_lo

    N = ny - 1
    D_raw, x_cheb = cheb(N)
    # Ascending CGL grid on [0, Ly]
    idx = np.argsort(x_cheb)
    x_asc = x_cheb[idx]
    y_asc = (1.0 - x_asc) * Ly / 2.0
    # Actually we need y ∈ [0, Ly] and ascending, which means
    # y_asc = (1 - x_asc) * Ly / 2 is descending as x_asc ascending...
    # Match C++: C++ uses idx[k] = N - k so y_asc[0] = 0, y_asc[-1] = Ly.
    # That corresponds to x going descending 1 -> -1 then reversed.
    # Let's do it explicitly:
    idx_desc_to_asc = np.arange(ny-1, -1, -1)
    x_permuted = x_cheb[idx_desc_to_asc]
    y_asc = (1.0 - x_permuted) * Ly / 2.0

    # Permute D and scale
    P = np.zeros((ny, ny))
    for k in range(ny):
        P[k, idx_desc_to_asc[k]] = 1.0
    scale = -2.0 / Ly
    D_scaled = scale * (P @ D_raw @ P.T)

    # Interpolate rho onto y_asc (same as C++: y_asc maps to [r_lo, r_hi])
    def rho_of_y(y):
        # Linear: y ∈ [0, Ly] → r ∈ [r_lo, r_hi]
        r = r_lo + (y / Ly) * (r_hi - r_lo)
        return np.interp(r, r_in, rho_in)

    rho = rho_of_y(y_asc)
    drho = D_scaled @ rho
    d2rho = D_scaled @ drho
    # Reduced W̃ = (ρ')²/(4ρ²) − ρ''/(2ρ)
    W = drho**2 / (4.0 * rho**2) - d2rho / (2.0 * rho)

    # Eigenproblem: A = -D² - diag(W) on interior
    D2 = D_scaled @ D_scaled
    M = ny - 2
    A = -D2[1:-1, 1:-1] - np.diag(W[1:-1])
    # Symmetrise
    A = 0.5 * (A + A.T)
    mu_all, V = np.linalg.eigh(A)
    return mu_all, y_asc, W, Ly


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cuda_csv = sys.argv[1]
    ny, n_modes, mu_cuda, y_cuda, W_cuda = parse_cuda_csv(cuda_csv)
    print(f"CUDA: ny={ny}, n_modes={n_modes}, Ly={y_cuda[-1]:.6f}")
    print(f"  |W̃|_max = {np.max(np.abs(W_cuda)):.3e}")
    print(f"  μ_0 = {mu_cuda[0]:.4e}, μ_5 = {mu_cuda[5]:.4e}, μ_{n_modes-1} = {mu_cuda[-1]:.4e}")

    mu_py, y_py, W_py, Ly_py = python_reference(ny, rho_cut=0.01)
    print(f"\nPython ref: ny={ny}, Ly={Ly_py:.6f}")
    print(f"  |W̃|_max = {np.max(np.abs(W_py)):.3e}")
    print(f"  μ_0 = {mu_py[0]:.4e}, μ_5 = {mu_py[5]:.4e}, μ_{ny-3} = {mu_py[-1]:.4e}")

    # Eigenvalue comparison (first n_modes)
    mu_py_trunc = mu_py[:n_modes]
    rel_err = np.abs(mu_cuda - mu_py_trunc) / np.abs(mu_py_trunc)
    print(f"\nEigenvalue agreement (first {n_modes} modes):")
    print(f"  max rel err = {rel_err.max():.3e}")
    print(f"  median rel err = {np.median(rel_err):.3e}")
    for k in [0, 1, 5, 10, 50, n_modes - 1]:
        if k < n_modes:
            print(f"  n={k:4d}:  CUDA={mu_cuda[k]:12.6e}  Python={mu_py_trunc[k]:12.6e}  "
                  f"rel_err={rel_err[k]:.3e}")

    # Grid agreement
    grid_err = np.max(np.abs(y_cuda - y_py))
    print(f"\nGrid agreement: max |y_cuda - y_py| = {grid_err:.3e}")

    # W agreement
    W_err = np.max(np.abs(W_cuda - W_py))
    print(f"W̃ agreement: max |W_cuda - W_py| = {W_err:.3e}")

    if rel_err.max() < 1e-10 and grid_err < 1e-12 and W_err < 1e-6:
        print("\n✓ PASS: CUDA and Python references agree to machine precision.")
    elif rel_err.max() < 1e-4:
        print("\n~ ACCEPTABLE: small numerical differences (likely different interpolation paths).")
    else:
        print("\n✗ FAIL: significant disagreement.")
        sys.exit(1)


if __name__ == "__main__":
    main()
