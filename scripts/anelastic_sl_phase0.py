#!/usr/bin/env python3
"""
Phase 0: Anelastic SL-Spectral 可行性驗證 (對應 docs/anelastic_SL_spectral_design.md §8, §9)

五個段落:
  0.1  Lane-Emden n=1.5 polytrope ρ(r) + W(r) + 奇異性分析
  0.2  截斷域 [y_lo, y_hi] 內解 Schrödinger 本徵問題 Tψ_n = -μ_n ψ_n
       + 測 ρ=const 極限 → Fourier 模的退化性
  0.3  SL-Poisson 端到端 manufactured-solution 驗證 ∇·((1/ρ)∇p) = f
  0.4  g-mode 頻率 cross-check(SL 本徵值 μ_n 對應 N²)+ scipy DGEMM 壁鐘
  0.5  彙整輸出到 videos/anelastic_sl_phase0_*.png 並 print table

用法:
  pixi run python scripts/anelastic_sl_phase0.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.integrate
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
VID = REPO / "videos"
VID.mkdir(exist_ok=True)


# ==============================================================================
# 0.1  Lane-Emden n=1.5 polytrope (Emden 函數 θ(ξ))
#      ρ = ρ_c · θ^n, P = K·ρ^(1+1/n), ξ 是 dim-less radius
# ==============================================================================
def solve_lane_emden(n: float = 1.5, xi_max: float = 10.0, n_pts: int = 5000):
    """
    θ''(ξ) + 2/ξ · θ'(ξ) + θ^n = 0,  θ(0)=1, θ'(0)=0
    Returns: ξ_arr, θ(ξ), θ'(ξ);  zero_ξ_1 (θ 首零點).
    """
    def rhs(xi, y):
        theta, dtheta = y
        if xi < 1e-10:
            return [dtheta, -theta ** n / 3.0]  # L'Hopital in limit
        theta_pow = np.sign(theta) * np.abs(theta) ** n if theta >= 0 else 0.0
        return [dtheta, -2.0 / xi * dtheta - theta_pow]

    def event_zero(xi, y):
        return y[0]
    event_zero.terminal = True
    event_zero.direction = -1

    sol = scipy.integrate.solve_ivp(
        rhs, [1e-6, xi_max], [1.0 - 1e-12, 0.0],
        events=event_zero, max_step=0.01, rtol=1e-10, atol=1e-12,
        dense_output=True,
    )
    xi_1 = sol.t_events[0][0] if sol.t_events[0].size else sol.t[-1]
    xi = np.linspace(1e-5, xi_1 * 0.999, n_pts)
    theta = sol.sol(xi)[0]
    return xi, theta, xi_1


def compute_W_from_rho(y, rho):
    """
    W(y) = ρ''/(2ρ) - 3(ρ')²/(4ρ²)
    用 4 階 central FD on non-uniform grid.
    """
    from numpy import gradient
    drho = gradient(rho, y, edge_order=2)
    d2rho = gradient(drho, y, edge_order=2)
    W = d2rho / (2.0 * rho) - 3.0 * drho ** 2 / (4.0 * rho ** 2)
    return W, drho, d2rho


def phase_0p1_lane_emden(save_path):
    print("\n=== 0.1 Lane-Emden n=1.5 polytrope + W(y) ===")
    xi, theta, xi_1 = solve_lane_emden(n=1.5)
    # 將 ξ → 歸一化半徑 r/R★ ∈ [0, 1)
    r_norm = xi / xi_1
    # ρ/ρ_c = θ^n
    rho = np.abs(theta) ** 1.5
    # 以 y = r/R 當譜域座標
    W, drho, d2rho = compute_W_from_rho(r_norm, rho)
    # 表面奇異性:截斷到 ρ > rho_thresh
    rho_thresh = 0.01  # 1% of center
    mask = rho > rho_thresh
    r_in = r_norm[mask]
    rho_in = rho[mask]
    W_in = W[mask]
    r_lo, r_hi = r_in[0], r_in[-1]

    print(f"  Lane-Emden n=1.5:  ξ_1 = {xi_1:.6f}")
    print(f"  ρ/ρ_c: [{rho[0]:.4f}, {rho[-1]:.2e}]")
    print(f"  W(y)  range over domain:  [{W.min():.3e}, {W.max():.3e}]")
    print(f"  W 在表面 (ρ<0.01): max|W| = {np.max(np.abs(W[~mask])):.3e}  <<顯示奇異性>>")
    print(f"  截斷域 [r_lo, r_hi] = [{r_lo:.4f}, {r_hi:.4f}]  (ρ > {rho_thresh})")
    print(f"  截斷內 W range: [{W_in.min():.3e}, {W_in.max():.3e}]")

    # 圖
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=140)
    axes[0].plot(r_norm, rho, "b-", lw=1.5)
    axes[0].axvline(r_hi, ls=":", c="r", label=f"cutoff r={r_hi:.3f}")
    axes[0].set_xlabel("r/R★")
    axes[0].set_ylabel(r"ρ/ρ_c")
    axes[0].set_title("Lane-Emden n=1.5 density")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(r_norm, drho, "g-", lw=1.2, label="ρ'")
    axes[1].plot(r_norm, d2rho, "m-", lw=1.0, label="ρ''")
    axes[1].axvline(r_hi, ls=":", c="r")
    axes[1].set_xlabel("r/R★")
    axes[1].set_title("ρ derivatives")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].semilogy(r_norm, np.abs(W), "r-", lw=1.2)
    axes[2].axvline(r_hi, ls=":", c="k", label=f"cutoff r={r_hi:.3f}")
    axes[2].set_xlabel("r/R★")
    axes[2].set_ylabel("|W(y)|  (log)")
    axes[2].set_title("Liouville potential W(y)")
    axes[2].legend(); axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    print(f"  → {save_path}")
    plt.close(fig)
    return r_in, rho_in, W_in


# ==============================================================================
# 0.2  SL 本徵問題 Tψ_n = -μ_n ψ_n on uniform 1D grid, Dirichlet BC
#      T = d²/dy² + W(y)  (W 外推到均勻格)
# ==============================================================================
def solve_sl_eigenpairs(y_uniform, W_uniform, n_modes):
    """
    離散 -(d²/dy² + W) ψ = μ ψ, Dirichlet(ψ(y_lo) = ψ(y_hi) = 0).

    y_uniform 包含 y_lo 和 y_hi(邊界上);我們**只對 interior nodes 做 FD**
    (index 1..N-2,共 N-2 個),邊界 ψ=0 隱式嵌入 stencil。
    返回的 psi 包含完整 N 格點(兩端填 0),方便後續 SL 變換用。
    """
    N = len(y_uniform)
    dy = y_uniform[1] - y_uniform[0]
    # Interior FD operator on nodes 1..N-2  (size M = N-2)
    M = N - 2
    W_int = W_uniform[1:-1]
    main = 2.0 / dy**2 - W_int
    off = -np.ones(M - 1) / dy**2
    A = scipy.sparse.diags([off, main, off], [-1, 0, 1], format="csr")
    # 最小 n_modes 個 eigenvalues
    mu, psi_int = scipy.sparse.linalg.eigsh(A, k=n_modes, which="SA")
    # embed 兩端 0 使 psi shape = (N, n_modes)
    psi = np.zeros((N, n_modes))
    psi[1:-1, :] = psi_int
    # Normalize ∫ ψ² dy = 1
    for i in range(psi.shape[1]):
        norm = np.sqrt(np.sum(psi[:, i] ** 2) * dy)
        psi[:, i] /= norm
    order = np.argsort(mu)
    return mu[order], psi[:, order]


def phase_0p2_sl_eigenpairs(r_in, rho_in, W_in, save_path, n_modes=20):
    print("\n=== 0.2 SL 本徵問題 + Fourier 極限退化檢驗 ===")
    # 將 W 插值到均勻格(SL 有限差分需要)
    N = 512
    y = np.linspace(r_in[0], r_in[-1], N)
    W = np.interp(y, r_in, W_in)
    rho = np.interp(y, r_in, rho_in)
    mu, psi = solve_sl_eigenpairs(y, W, n_modes)
    L = r_in[-1] - r_in[0]

    print(f"  Domain [y_lo, y_hi] = [{y[0]:.4f}, {y[-1]:.4f}],  L = {L:.4f}, N = {N}")
    print(f"  前 10 個本徵值 μ_n:")
    mu_fourier = np.array([(n * np.pi / L) ** 2 for n in range(1, n_modes + 1)])
    for n in range(min(10, n_modes)):
        mu_fr = mu_fourier[n]
        print(f"    n={n:2d}  μ_SL = {mu[n]:12.4e}   μ_Fourier = {mu_fr:12.4e}"
              f"   ratio = {mu[n] / mu_fr:.4f}")

    # 測 Fourier 極限:ρ = const 的 box,預期前低階 n 退化到 (nπ/L)²
    # 只比較前 20 個(大 n 被 FD 離散誤差 O(n⁴/N²) 污染,屬於格點本身局限)
    n_cmp = min(20, n_modes)
    y_box = y.copy()
    W_box = np.zeros_like(W)
    mu_box, psi_box = solve_sl_eigenpairs(y_box, W_box, n_cmp)
    max_rel_err = np.max(np.abs(mu_box - mu_fourier[:n_cmp]) / mu_fourier[:n_cmp])
    print(f"  ρ=const 極限 (n<={n_cmp}): μ_SL vs (nπ/L)²,max rel err = {max_rel_err:.3e}"
          f"  ← FD stencil 理論極限 O(n⁴·dy²) ≈ {(n_cmp*np.pi/L)**2 * (L/(N-1))**2 / 12:.3e}")

    # 畫前 6 個 ψ_n
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=140)
    for k, ax in enumerate(axes.flat):
        ax.plot(y, psi[:, k], "b-", lw=1.5, label=f"ψ_{k} SL (μ={mu[k]:.3f})")
        # Fourier 模作參照
        fmode = np.sin((k + 1) * np.pi * (y - y[0]) / L) * np.sqrt(2.0 / L)
        ax.plot(y, fmode, "r--", lw=1.0, alpha=0.5, label=f"sin((n+1)πy/L)")
        ax.set_xlabel("y")
        ax.set_ylabel("ψ")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("SL 本徵函數 ψ_n vs Fourier 模 (前 6 個)")
    fig.tight_layout()
    fig.savefig(save_path)
    print(f"  → {save_path}")
    plt.close(fig)
    return y, W, rho, mu, psi


# ==============================================================================
# 0.3  SL-Poisson 端到端 manufactured-solution 驗證
#      給定 p_exact(x,y), 算 f = ∇·((1/ρ)∇p_exact),
#      用 SL pipeline 反演得 p_num, 比對 err_L2.
# ==============================================================================
def manufactured_p_and_f(x, y, rho, Lx):
    """
    取 p_exact = sin(k_x x) · φ(y),  φ(y) 任意光滑且 Dirichlet.
    ∇·((1/ρ)∇p) = (1/ρ)[-k_x² p + d/dy((1/ρ)·ρ·∂_y p)]
                = (1/ρ)[-k_x² p + (1/ρ)∂_y((1/ρ)∂_y(ρ·...))]
    為簡化直接解析算:  f = ∂_x(ρ⁻¹ ∂_x p) + ∂_y(ρ⁻¹ ∂_y p)
    """
    k_x = 2.0 * np.pi / Lx * 2  # k=2 mode
    X, Y = np.meshgrid(x, y, indexing="xy")
    # φ(y) = sin(π (y-y_lo)/(y_hi-y_lo))^2 ?  取 zeroable:
    y_lo, y_hi = y[0], y[-1]
    L = y_hi - y_lo
    # φ = sin(πη) where η = (y - y_lo)/L  →  Dirichlet on both ends
    eta = (Y - y_lo) / L
    phi = np.sin(np.pi * eta)
    dphi = (np.pi / L) * np.cos(np.pi * eta)
    d2phi = -(np.pi / L) ** 2 * phi

    p = np.sin(k_x * X) * phi
    dpx = k_x * np.cos(k_x * X) * phi
    dpy = np.sin(k_x * X) * dphi
    d2px = -(k_x ** 2) * p
    d2py = np.sin(k_x * X) * d2phi

    # ρ(y) — 僅 y 依賴
    RHO, _ = np.meshgrid(rho, x, indexing="xy")
    RHO = RHO.T  # shape (Ny, Nx)
    drho = np.gradient(rho, y, edge_order=2)
    DRHO, _ = np.meshgrid(drho, x, indexing="xy")
    DRHO = DRHO.T

    # f = (1/ρ) ∂²p/∂x² + ∂_y[(1/ρ)∂p/∂y]
    #   = (1/ρ)·(-k_x² p) + (-drho/ρ²)·dpy + (1/ρ)·d2py
    inv_rho = 1.0 / RHO
    f = inv_rho * d2px + (-DRHO / RHO ** 2) * dpy + inv_rho * d2py
    return p, f


def sl_poisson_solve(f_xy, rho, y, psi, mu, Lx, k_x_modes):
    """
    §5 的 7 步 pipeline(此處簡化為用已有 Python array operations):
      1. FFT f in x → f̂(k_x, y)
      2. g(k_x, y) = √ρ · f̂
      3. G_n(k_x) = Σ_j ψ_n(y_j) · g(k_x, y_j) · dy  (正 SL transform)
      4. Q_n(k_x) = -G_n(k_x) / (μ_n + k_x²)
      5. q(k_x, y) = Σ_n ψ_n(y) · Q_n(k_x)   (逆 SL)
      6. p̂(k_x, y) = √ρ · q
      7. IFFT in x → p(x, y)
    """
    Ny, Nx = f_xy.shape
    dy = y[1] - y[0]
    sqrt_rho = np.sqrt(rho)[:, None]  # (Ny, 1)

    # 1. FFT in x (last axis)
    f_hat = np.fft.rfft(f_xy, axis=1) / Nx  # (Ny, Nmx), complex
    # 2. g = √ρ · f̂
    g = sqrt_rho * f_hat
    # 3. G = Ψᵀ · g · dy (正 SL)  — ψ columns → Ψᵀ shape (n_modes, Ny)
    PsiT = psi.T  # (n_modes, Ny)
    G = PsiT @ g * dy  # (n_modes, Nmx)
    # 4. Q = -G / (μ_n + k_x²)
    kx_sq = k_x_modes ** 2  # (Nmx,)
    denom = mu[:, None] + kx_sq[None, :]  # (n_modes, Nmx)
    # 避開 denom=0(k=0 且 μ=0,這裡 μ 都 > 0)
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    Q = -G / denom
    # 5. q = Ψ · Q
    q = psi @ Q  # (Ny, Nmx)
    # 6. p̂ = √ρ · q
    p_hat = sqrt_rho * q
    # 7. IFFT
    p_rec = np.fft.irfft(p_hat, n=Nx, axis=1) * Nx  # undo earlier /Nx
    return p_rec


def phase_0p3_sl_poisson(y, rho, W, mu, psi, save_path):
    print("\n=== 0.3 SL-Poisson 端到端 manufactured-solution ===")
    Ny = len(y)
    Nx = 128
    Lx = 1.0
    x = np.arange(Nx) * Lx / Nx
    # manufactured p and corresponding f
    p_exact, f = manufactured_p_and_f(x, y, rho, Lx)

    # k_x modes for R2C
    k_x_modes = 2 * np.pi * np.fft.rfftfreq(Nx, d=Lx / Nx)
    p_num = sl_poisson_solve(f, rho, y, psi, mu, Lx, k_x_modes)

    err = p_num - p_exact
    err_L2 = np.sqrt(np.mean(err ** 2))
    err_Linf = np.max(np.abs(err))
    print(f"  N_modes used = {psi.shape[1]},  Nx = {Nx}, Ny = {Ny}")
    print(f"  err_L2   = {err_L2:.3e}")
    print(f"  err_Linf = {err_Linf:.3e}")

    # 收斂掃描:變 N_modes
    n_modes_list = [5, 10, 20, 40, 80, min(160, psi.shape[1])]
    errs = []
    for Nm in n_modes_list:
        if Nm > psi.shape[1]:
            continue
        p_n = sl_poisson_solve(f, rho, y, psi[:, :Nm], mu[:Nm], Lx, k_x_modes)
        e = np.sqrt(np.mean((p_n - p_exact) ** 2))
        errs.append((Nm, e))
        print(f"    N_modes = {Nm:3d}  err_L2 = {e:.3e}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=140)
    im0 = axes[0].pcolormesh(x, y, p_exact, cmap="RdBu_r", shading="auto")
    axes[0].set_title("p_exact")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].pcolormesh(x, y, p_num, cmap="RdBu_r", shading="auto")
    axes[1].set_title("p_num (SL reconstruct)")
    plt.colorbar(im1, ax=axes[1])
    im2 = axes[2].pcolormesh(x, y, np.abs(err), cmap="viridis", shading="auto")
    axes[2].set_title(f"|err|  (L2={err_L2:.2e})")
    plt.colorbar(im2, ax=axes[2])
    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(save_path)
    print(f"  → {save_path}")
    plt.close(fig)

    # 收斂曲線圖
    fig2, ax = plt.subplots(figsize=(5.5, 4), dpi=140)
    xs, ys = zip(*errs)
    ax.loglog(xs, ys, "o-", lw=1.5)
    ax.set_xlabel("N_modes (SL eigenfunctions used)")
    ax.set_ylabel("err_L2")
    ax.set_title("SL expansion convergence")
    ax.grid(True, which="both", alpha=0.3)
    fig2.tight_layout()
    conv_path = str(save_path).replace(".png", "_convergence.png")
    fig2.savefig(conv_path)
    print(f"  → {conv_path}")
    plt.close(fig2)
    return err_L2, errs


# ==============================================================================
# 0.4  g-mode 頻率 cross-check + DGEMM 壁鐘
# ==============================================================================
def phase_0p4_gmode_and_dgemm(mu, y, rho, save_path):
    print("\n=== 0.4 g-mode 頻率 + DGEMM benchmark ===")
    # Brunt-Vaisala 頻率 N² = g·(1/ρ)(dρ/dy - dρ_ad/dy_ad)
    # 對 adiabatic polytrope,N² 由重力梯度和絕熱梯度差決定;
    # 這裡我們只做結構 check:SL 本徵值 μ_n 的 asymptotic scaling 對 n 應 ~ n²
    # 對應 high-order g-modes 漸近 ω_g ~ √N̄·n⁻¹(Tassoul 1980)
    print("  SL 本徵值前 20 個(期望大 n 近 n² scaling):")
    for n in range(len(mu)):
        ratio = mu[n] / ((n + 1) ** 2) if (n + 1) > 0 else np.nan
        print(f"    n={n:2d}  μ = {mu[n]:12.4e}   μ/(n+1)² = {ratio:.4e}")
    # g-mode 分析圖:  √μ vs n  — 線性 suggests Fourier-like; 曲線 suggests g-mode asymptote
    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=140)
    n_arr = np.arange(1, len(mu) + 1)
    ax.plot(n_arr, np.sqrt(mu), "bo-", label=r"$\sqrt{\mu_n}$ (SL)")
    L = y[-1] - y[0]
    ax.plot(n_arr, n_arr * np.pi / L, "k--", alpha=0.5,
            label=r"$n\pi/L$ (Fourier ref)")
    ax.set_xlabel("n (mode index)")
    ax.set_ylabel(r"$\sqrt{\mu_n}$")
    ax.set_title("SL eigenvalue asymptote (g-mode spectrum analogue)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    print(f"  → {save_path}")
    plt.close(fig)

    # DGEMM benchmark(CPU 端 scipy.linalg.blas.dgemm 作基準;GPU 下 cuBLAS 會更快)
    from scipy.linalg.blas import dgemm
    for N in [256, 512, 1024, 2048]:
        A = np.random.rand(N, N)
        B = np.random.rand(N, N // 2)
        # warmup
        _ = dgemm(1.0, A, B)
        t0 = time.time()
        for _ in range(3):
            C = dgemm(1.0, A, B)
        dt = (time.time() - t0) / 3
        gflop = 2.0 * N * N * (N // 2) / 1e9
        print(f"  DGEMM ({N}×{N})·({N}×{N//2})  wall = {dt*1000:.2f} ms"
              f"  → {gflop / dt:.1f} GFLOPS FP64 (CPU)")


def main():
    print("=" * 70)
    print(" Phase 0: Anelastic SL-Spectral 可行性驗證")
    print("=" * 70)

    # 0.1 Lane-Emden W(y)
    r_in, rho_in, W_in = phase_0p1_lane_emden(
        VID / "anelastic_sl_0p1_lane_emden.png"
    )

    # 0.2 SL eigenpairs
    y, W, rho, mu, psi = phase_0p2_sl_eigenpairs(
        r_in, rho_in, W_in,
        VID / "anelastic_sl_0p2_eigenfunctions.png",
        n_modes=256,
    )

    # 0.3 SL-Poisson end-to-end
    err_L2, errs = phase_0p3_sl_poisson(
        y, rho, W, mu, psi,
        VID / "anelastic_sl_0p3_poisson.png"
    )

    # 0.4 g-mode analogue + DGEMM bench
    phase_0p4_gmode_and_dgemm(
        mu[:20], y, rho,
        VID / "anelastic_sl_0p4_gmode.png"
    )

    print("\n" + "=" * 70)
    print(" Phase 0 done.")
    print(f" 最終 SL-Poisson err_L2 = {err_L2:.3e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
