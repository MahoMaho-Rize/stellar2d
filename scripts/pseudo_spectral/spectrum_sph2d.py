#!/usr/bin/env python3
"""
sph2d 球諧譜 E(l) 分析。

若 solver 已寫 spectrum.csv(將來擴展),直接讀;否則從 diagnostics.csv + VTK 估算。
暫時:從 VTK 讀 ζ(θ,φ),做 FFT + Legendre 得球諧係數,算 E(l) = Σ_m |ζ̂_l^m|²·R²/(l(l+1)·2)。

用法:
  pixi run python scripts/spectrum_sph2d.py <run_dir> <out.png> [n_snap=6]
"""
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from render_sph2d import read_vtk_binary


def compute_spectrum(zeta_phys, n_theta, n_phi, R=1.0, L_max=None):
    """從物理 ζ(θ,φ) 得 E(l),球諧展開後 E_l = Σ_m |ζ̂|²·R²/(l(l+1)·2)。
    用 Gauss-Legendre quadrature 求 ζ̂_l^m。"""
    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(n_theta)         # nodes and weights on [-1,1]
    # 節點排序 — solver 把 θ_j = acos(-x) 排成升序
    # 但 GL roots 給出 x sorted ascending;cos θ 也 ascending → θ desc 實際上
    # 為保守我們重排讓 θ ascending:
    order = np.argsort(-x)          # larger x → smaller θ
    x = x[order]
    w = w[order]
    sint = np.sqrt(1.0 - x * x)
    theta = np.arccos(x)

    if L_max is None:
        L_max = min(n_theta - 1, n_phi // 2 - 2)

    # φ FFT
    f_mphi = np.fft.rfft(zeta_phys, axis=1) / n_phi

    # Associated Legendre table (4π-normalized)
    P = np.zeros((n_theta, L_max + 1, L_max + 1))
    for j in range(n_theta):
        xj = x[j]
        pmm = 1.0 / np.sqrt(4.0 * np.pi)
        P[j, 0, 0] = pmm
        for m in range(1, L_max + 1):
            pmm = np.sqrt((2.0 * m + 1.0) / (2.0 * m)) * sint[j] * pmm
            P[j, m, m] = pmm
        for m in range(L_max + 1):
            if m + 1 <= L_max:
                P[j, m + 1, m] = np.sqrt(2.0 * m + 3.0) * xj * P[j, m, m]
            for l in range(m + 2, L_max + 1):
                a = np.sqrt((2.0 * l - 1.0) * (2.0 * l + 1.0)
                            / ((l - m) * (l + m)))
                b = np.sqrt((2.0 * l + 1.0) * (l - 1 - m) * (l - 1 + m)
                            / ((2.0 * l - 3.0) * (l - m) * (l + m)))
                P[j, l, m] = a * xj * P[j, l - 1, m] - b * P[j, l - 2, m]

    # 球諧係數 ẑ_l^m = Σ_j w_j · P_l^m(x_j) · f_m(θ_j)
    zhat = np.zeros((L_max + 1, L_max + 1), dtype=complex)
    for m in range(min(L_max + 1, f_mphi.shape[1])):
        for l in range(m, L_max + 1):
            zhat[l, m] = np.sum(w * P[:, l, m] * f_mphi[:, m]) * 2.0 * np.pi

    E = np.zeros(L_max + 1)
    for l in range(1, L_max + 1):
        s = np.abs(zhat[l, 0]) ** 2
        for m in range(1, l + 1):
            s += 2.0 * np.abs(zhat[l, m]) ** 2
        E[l] = 0.5 * s * R * R / (l * (l + 1))
    return E


def main():
    if len(sys.argv) < 3:
        print("Usage: spectrum_sph2d.py <run_dir> <out.png> [n_snap=6]")
        sys.exit(1)
    rd = sys.argv[1]
    out = sys.argv[2]
    n_snap = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    vtks = sorted(glob.glob(os.path.join(rd, "output_*.vtk")))
    if not vtks:
        print(f"ERROR: no VTK in {rd}")
        sys.exit(1)

    # 取 evenly spaced n_snap 幀
    idxs = np.linspace(0, len(vtks) - 1, n_snap).astype(int)
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=144)
    for k in idxs:
        (nt, nph), sc = read_vtk_binary(vtks[k])
        z = sc["vorticity"]
        E = compute_spectrum(z, nt, nph)
        ls = np.arange(len(E))
        mask = (E > 0) & (ls > 0)
        ax.loglog(ls[mask], E[mask], label=f"frame {k}")
    # 參考斜率
    ls_ref = np.arange(2, 50)
    for slope, lbl in [(-5/3, "-5/3"), (-3, "-3")]:
        ref = E[10] * (ls_ref / 10.0) ** slope if E[10] > 0 else ls_ref ** slope
        ax.loglog(ls_ref, ref, "--", alpha=0.3, label=lbl)
    ax.set_xlabel("l (spherical harmonic degree)")
    ax.set_ylabel("E(l)")
    ax.set_title(f"Sph2D spectrum — {os.path.basename(rd)}")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    print(f"✓ {out}")


if __name__ == "__main__":
    main()
