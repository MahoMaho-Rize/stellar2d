#!/usr/bin/env python3
"""Verify that solutions stored on N CGL nodes interpolate to full spectral
accuracy everywhere on the interval, not only at grid points.

Test:
  1) Evaluate a known smooth function f(y) = sin(3πy/Ly)·exp(-2y) on the
     coarse CGL grid (N=32, 48, 64).
  2) Reconstruct it at 1024 uniformly-spaced evaluation points via
     barycentric Chebyshev interpolation.
  3) Compare to the analytic value.  Expect: L∞ error ~ machine precision
     once N is large enough to resolve the function.

  4) Repeat for the Lane-Emden n=3/2 g-mode eigenvector (n_g=1, kx=2π):
     solve the EVP at N=64, N=128, N=192, reconstruct on 1024 points, and
     compare N=64 reconstruction to N=192 reconstruction on the same fine
     grid.  This mimics what we'd do in post-processing a DNS run.
"""
import numpy as np
import scipy.linalg
from nonlinear_paths_infra import cgl_grid, bg_lane_emden


def barycentric_cheb_interp(y_src, f_src, y_eval):
    """Barycentric interpolation of f sampled on CGL nodes y_src (ascending)
    evaluated at arbitrary y_eval.  Trefethen, Spectral Methods in MATLAB, (5.1).

    Weights for CGL on any interval: w_k = (-1)^k · (1/2 at endpoints, 1 otherwise).
    Because y_src was built by reversing cos(kπ/N) onto [0, Ly] (see cgl_grid),
    we mirror the standard weights the same way.
    """
    N = len(y_src) - 1
    w = np.ones(N + 1)
    w[0] = 0.5; w[N] = 0.5
    # y_src = (1 + x[::-1]) * Ly/2 with x=cos(kπ/N), so node k corresponds to
    # descending Chebyshev index N-k.  Sign pattern: (-1)^k after reversal.
    signs = (-1.0) ** np.arange(N + 1)
    w = w * signs[::-1]

    y_eval = np.atleast_1d(y_eval).astype(float)
    out = np.empty_like(y_eval)
    for i, y in enumerate(y_eval):
        # Exact grid-point match
        match = np.where(np.abs(y_src - y) < 1e-14)[0]
        if len(match):
            out[i] = f_src[match[0]]
            continue
        num = np.sum(w * f_src / (y - y_src))
        den = np.sum(w           / (y - y_src))
        out[i] = num / den
    return out


def smooth_testfn(y):
    return np.sin(3.0 * np.pi * y) * np.exp(-2.0 * y)


def test1_known_function():
    print("── Test 1: reconstruct f(y) = sin(3πy)·exp(-2y) ─────────────")
    y_fine = np.linspace(0.0, 1.0, 1024)
    f_exact = smooth_testfn(y_fine)

    for N_y in [16, 24, 32, 48, 64]:
        y_src, _ = cgl_grid(N_y, 1.0)
        f_src   = smooth_testfn(y_src)
        f_recon = barycentric_cheb_interp(y_src, f_src, y_fine)
        err = np.max(np.abs(f_recon - f_exact))
        print(f"  N={N_y:3d}:  L∞ error on 1024-pt fine grid = {err:.3e}")
    print()


def solve_gmode_evp(ny, rho_cut, kx_phys, n_g):
    """Returns (omega, y, V_full) for the n_g-th g-mode on Lane-Emden.
    V is normalised so max|V|=1."""
    y, D = cgl_grid(ny, 1.0)
    rho, N2 = bg_lane_emden(y, 1.0, rho_cut=rho_cut)
    intr = slice(1, ny - 1)
    L = -D @ (np.diag(rho) @ D) + kx_phys ** 2 * np.diag(rho)
    R = kx_phys ** 2 * np.diag(N2 * rho)
    Li, Ri = L[intr, intr], R[intr, intr]
    lam, V_int = scipy.linalg.eig(Ri, Li)
    lam = np.real(lam); V_int = np.real(V_int)
    mask = np.isfinite(lam) & (lam > 0)
    lam, V_int = lam[mask], V_int[:, mask]
    order = np.argsort(lam)[::-1]
    lam = lam[order]; V_int = V_int[:, order]
    omega = np.sqrt(lam[n_g - 1])
    V_y = np.zeros(ny)
    V_y[1:-1] = V_int[:, n_g - 1]
    V_y /= np.max(np.abs(V_y))
    # Fix a deterministic sign: positive at node jy = ny // 2.
    if V_y[ny // 2] < 0:
        V_y = -V_y
    return float(omega), y, V_y


def test2_gmode_eigenvector():
    print("── Test 2: Lane-Emden n=3/2 g-mode eigenvector ─────────────")
    print("  n_g=1, kx=2π/Ly=2π, rho_cut=0.05")
    y_fine = np.linspace(1e-4, 1.0 - 1e-4, 1024)
    kx_phys = 2.0 * np.pi
    omegas, y_srcs, V_srcs = {}, {}, {}
    for N_y in [48, 64, 96, 128, 192, 256]:
        omega, y_src, V_src = solve_gmode_evp(N_y, 0.05, kx_phys, 1)
        omegas[N_y] = omega
        y_srcs[N_y] = y_src
        V_srcs[N_y] = V_src
        print(f"  N={N_y:3d}:  ω = {omega:.12f}")
    print()

    # Compare N=48 reconstruction to N=256 reconstruction on the same 1024-pt
    # fine grid.  Because eigenvectors from different Ny are different linear
    # solutions (but should converge to the same function), normalise each by
    # its L2 norm on the fine grid before comparing.
    V256 = barycentric_cheb_interp(y_srcs[256], V_srcs[256], y_fine)
    V256 /= np.linalg.norm(V256)

    print("  Reconstruction convergence on 1024-pt fine grid (vs N=256):")
    for N_y in [48, 64, 96, 128, 192]:
        V_recon = barycentric_cheb_interp(y_srcs[N_y], V_srcs[N_y], y_fine)
        V_recon /= np.linalg.norm(V_recon)
        # Sign flip ambiguity: pick the sign that minimises error.
        err_pos = np.max(np.abs(V_recon - V256))
        err_neg = np.max(np.abs(-V_recon - V256))
        err = min(err_pos, err_neg)
        domega = abs(omegas[N_y] - omegas[256]) / omegas[256]
        print(f"    N={N_y:3d}:  L∞ err(V) = {err:.3e},  "
              f"|Δω/ω| = {domega:.3e}")
    print()


def test3_advection_aliasing():
    """Show why advection has a real resolution ceiling despite the
    spectral-accuracy interpolation result."""
    print("── Test 3: pointwise product frequency content ─────────────")
    print("  f(x) = sin(k_a x),  g(x) = sin(k_b x)")
    print("  f·g has content at k = k_a±k_b.  Interpolating the product")
    print("  from N Fourier modes loses k > N/2 (aliasing).  Dealiased")
    print("  effective resolution ≈ 2N/3.\n")
    N = 32
    nx_fine = 1024
    x_src  = np.linspace(0, 2*np.pi, N,       endpoint=False)
    x_fine = np.linspace(0, 2*np.pi, nx_fine, endpoint=False)
    # Resolved: k_a=3, k_b=5, product has k=|k_a±k_b|∈{2,8}, both < N/2=16.
    print("  Case A (resolved):  k_a=3, k_b=5  →  product has k ∈ {2, 8}")
    fa = np.sin(3 * x_src); ga = np.sin(5 * x_src)
    prod_grid = fa * ga
    prod_exact = np.sin(3 * x_fine) * np.sin(5 * x_fine)
    # Reconstruct via FFT interpolation.
    prod_hat = np.fft.rfft(prod_grid)
    prod_interp = np.fft.irfft(prod_hat * (nx_fine / N), n=nx_fine)
    err = np.max(np.abs(prod_interp - prod_exact))
    print(f"    L∞(reconstruct - exact) = {err:.3e}  (≈ machine precision)\n")

    # Aliased: k_a=5, k_b=14 → product has k=19 (aliased to k=13 on N=32 grid)
    print("  Case B (aliased):   k_a=5, k_b=14 →  product has k ∈ {9, 19};")
    print("                       k=19 aliases to k=32-19=13 on N=32 grid")
    fb = np.sin(5  * x_src); gb = np.sin(14 * x_src)
    prod_grid = fb * gb
    prod_exact = np.sin(5 * x_fine) * np.sin(14 * x_fine)
    prod_hat = np.fft.rfft(prod_grid)
    prod_interp = np.fft.irfft(prod_hat * (nx_fine / N), n=nx_fine)
    err = np.max(np.abs(prod_interp - prod_exact))
    print(f"    L∞(reconstruct - exact) = {err:.3e}  (aliased, NOT spectral)\n")


if __name__ == "__main__":
    test1_known_function()
    test2_gmode_eigenvector()
    test3_advection_aliasing()
