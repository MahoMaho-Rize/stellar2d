#!/usr/bin/env python3
"""2D 湍流譜斜率定量回歸 — Kraichnan 雙級串驗證。

在用戶指定的 k-range(或自動選)做 log-log 線性回歸:
  - Inverse cascade range (k < k_inj):    E(k) ~ k^α,  α 應接近 -5/3
  - Enstrophy cascade range (k > k_inj):  E(k) ~ k^β,  β 應接近 -3

關鍵:k_inj 必須是 **IC 注入波數**,不是當前幀的譜峰(後者在 2D 衰減
湍流中會隨時間逆級串到最低模 k=2π/L)。

輸出:
  1. 單張圖 — 譜 + 兩段擬合直線 + 帶陰影的 ±σ 95% CI + 斜率數值
  2. stdout — 擬合斜率 ± 標準誤、R², 理論偏差

複用 spectrum_pseudo_spectral.py 的 compute_spectrum / load_velocity / parse_vtk_*。

Usage:
  scripts/spectrum_fit_pseudo_spectral.py <run_dir> [out.png] [t_snap] [k_mode]
     t_snap  = 要做 fit 的物理時間 (預設 = 最後一幀)
     k_mode  = IC 擾動模數 (例如 --ps-k 的值); 預設 4
              物理 k_inj = k_mode · 2π/L
  scripts/spectrum_fit_pseudo_spectral.py <file.vtk> [out.png] [- k_mode]
     直接指定 VTK 檔 (t_snap 給 '-' 跳過)
"""

import numpy as np
import os, sys, glob, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_pseudo_spectral import parse_vtk_cells, parse_vtk_dims
from spectrum_pseudo_spectral import compute_spectrum, load_velocity, load_frame_times, frame_number


# ───────────────────────────────────────────────────────────
# Fit 工具
# ───────────────────────────────────────────────────────────
def loglog_linregress(k, E, k_min, k_max):
    """線性回歸 log E vs log k ∈ [k_min, k_max]。
    回傳 (slope, intercept, slope_se, r2, n_points)。
    intercept 是 log₁₀(E) at log₁₀(k)=0。
    """
    m = (k >= k_min) & (k <= k_max) & (E > 0)
    x = np.log10(k[m])
    y = np.log10(E[m])
    n = len(x)
    if n < 3:
        return None
    # 最小二乘
    xbar, ybar = x.mean(), y.mean()
    Sxx = ((x - xbar) ** 2).sum()
    Sxy = ((x - xbar) * (y - ybar)).sum()
    slope = Sxy / Sxx
    intercept = ybar - slope * xbar
    y_pred = slope * x + intercept
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - ybar) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    # 斜率標準誤 (Gauss-Markov)
    s2 = ss_res / max(n - 2, 1)
    slope_se = np.sqrt(s2 / Sxx)
    return dict(slope=slope, intercept=intercept, slope_se=slope_se,
                r2=r2, n=n, k_min=k_min, k_max=k_max)


def auto_split_ranges(k, E, k_inj, nu=None, eps=None):
    """挑 inverse / enstrophy cascade 的 k-range。
    k_inj 必須由呼叫者提供(IC 擾動物理 k = k_mode · 2π/L)。
    - Inverse:   [dk, k_inj·0.8]  (k < 注入)
    - Enstrophy: [k_inj·1.5, k_eta·0.1]  (k > 注入,遠離耗散)
    """
    kmax = k.max()
    dk = k[1] if len(k) > 1 and k[1] > 0 else 1.0

    # Inverse cascade k 範圍: [dk, k_inj·0.8]
    if k_inj > 2.5 * dk:
        inv = (dk, k_inj * 0.8)
    else:
        inv = None

    # Enstrophy cascade 上限: 避開 Kolmogorov 耗散範圍
    if nu is not None and eps is not None and eps > 0:
        eta = (nu ** 3 / eps) ** 0.25
        k_eta = 1.0 / eta
        k_high = min(k_eta * 0.1, kmax * 0.3)
    else:
        k_high = kmax * 0.05

    k_low_enst = k_inj * 1.5
    enst = (k_low_enst, max(k_high, k_low_enst * 2.0)) if k_high > k_low_enst * 1.5 else None
    return inv, enst


# ───────────────────────────────────────────────────────────
# 主流程
# ───────────────────────────────────────────────────────────
def pick_vtk(run_dir, t_request=None):
    all_files = sorted(glob.glob(os.path.join(run_dir, "output_????.vtk")))
    def is_binary(p):
        with open(p, "rb") as fh:
            return b"BINARY\n" in fh.read(256)
    all_files = [f for f in all_files if os.path.getsize(f) > 1024 and is_binary(f)]
    if not all_files:
        return None, None
    ftimes = load_frame_times(run_dir)
    if t_request is None or not ftimes:
        # 取最後一幀
        path = all_files[-1]
        fnum = frame_number(path)
        return path, ftimes.get(fnum)
    # 找最接近 t_request 的幀
    best_path, best_t, best_err = None, None, np.inf
    for p in all_files:
        fnum = frame_number(p)
        t = ftimes.get(fnum)
        if t is None: continue
        err = abs(t - t_request)
        if err < best_err:
            best_err, best_path, best_t = err, p, t
    return best_path, best_t


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    arg = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    # argv[3] = t_snap, argv[4] = k_mode
    t_req = None
    if len(sys.argv) > 3 and sys.argv[3] not in ("-", ""):
        t_req = float(sys.argv[3])
    k_mode = int(sys.argv[4]) if len(sys.argv) > 4 else 4

    if arg.endswith(".vtk"):
        path = arg
        run_dir = os.path.dirname(arg) or "."
        ftimes = load_frame_times(run_dir)
        t = ftimes.get(frame_number(path))
    else:
        run_dir = arg.rstrip("/")
        path, t = pick_vtk(run_dir, t_req)
        if path is None:
            print(f"無 binary VTK 於 {run_dir}")
            sys.exit(1)
    if out_path is None:
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(os.path.dirname(path), f"spectrum_fit_{base}.png")

    nx, ny = parse_vtk_dims(path)
    print(f"載入 {path}")
    print(f"  grid {nx}×{ny},  t = {t if t is not None else '?':.4f}" if t else f"  grid {nx}×{ny}")

    u, v = load_velocity(path, nx, ny)
    k, Ek, KE = compute_spectrum(u, v)

    # 從 diagnostics.csv 找 ν 和 ε (若存在),以便估 Kolmogorov 耗散波數
    nu_est, eps_est = None, None
    csv_path = os.path.join(run_dir, "diagnostics.csv")
    if os.path.exists(csv_path):
        try:
            d = np.genfromtxt(csv_path, delimiter=",", names=True)
            if t is not None and 'eps_KE' in d.dtype.names:
                idx = int(np.argmin(np.abs(d['t'] - t)))
                eps_est = float(d['eps_KE'][idx])
                # ν = eps / (2·Ω)  (eps_KE = 2ν·Ω)
                Omega = float(d['enstrophy'][idx])
                if Omega > 0:
                    nu_est = eps_est / (2.0 * Omega)
        except Exception:
            pass

    # 物理注入波數 (IC k_mode · 2π/L,假設 L=1)
    k_inj = k_mode * 2.0 * np.pi
    inv_rng, enst_rng = auto_split_ranges(k, Ek, k_inj, nu=nu_est, eps=eps_est)
    if nu_est and eps_est:
        eta = (nu_est ** 3 / eps_est) ** 0.25
        print(f"  ν≈{nu_est:.2e},  ε≈{eps_est:.2e},  η≈{eta:.3e},  k_η≈{1.0/eta:.1f}")
    print(f"  k_inj = k_mode·2π = {k_mode}·2π ≈ {k_inj:.2f}")
    print(f"  inverse range:   {inv_rng}")
    print(f"  enstrophy range: {enst_rng}")
    print()

    # 理論參考
    THEORY_INV = -5.0 / 3.0
    THEORY_ENS = -3.0

    fit_inv = loglog_linregress(k, Ek, *inv_rng) if inv_rng else None
    fit_ens = loglog_linregress(k, Ek, *enst_rng) if enst_rng else None

    def report(label, fit, theory):
        if fit is None:
            print(f"  {label}: 區間點數不足")
            return
        dev = fit['slope'] - theory
        sigma_away = abs(dev) / max(fit['slope_se'], 1e-12)
        tag = "✓" if sigma_away < 2 else ("≈" if sigma_away < 4 else "✗")
        print(f"  {label:20s} slope = {fit['slope']:+.3f} ± {fit['slope_se']:.3f}  "
              f"(理論 {theory:+.3f}, 偏差 {sigma_away:.1f}σ)  R²={fit['r2']:.3f}  n={fit['n']}  {tag}")

    print("擬合結果:")
    report("Inverse cascade",   fit_inv, THEORY_INV)
    report("Enstrophy cascade", fit_ens, THEORY_ENS)
    print()

    # ─── 繪圖 ───
    fig, ax = plt.subplots(figsize=(10, 7), dpi=130)
    mask = (k > 0) & (Ek > 1e-30)
    ax.loglog(k[mask], Ek[mask], "-", color="#2260aa", lw=1.3, label=f"E(k), t={t:.2f}" if t else "E(k)")

    # Injection k 垂直線
    ax.axvline(k_inj, color="gray", lw=0.9, ls=":", alpha=0.7)
    ax.text(k_inj * 1.05, ax.get_ylim()[1] * 0.3,
            f"  k_inj = {k_mode}·2π ≈ {k_inj:.1f}", color="gray", fontsize=9, va="top")

    # dealias cutoff
    kmax_dealias = k.max()   # compute_spectrum 回傳的 k 已含全部到 kmax_bin
    # 2/3 rule 圓形截斷
    Nmin = min(nx, ny)
    k_cut = 2 * np.pi * (Nmin / 3.0)   # Lx=Ly=1 假設
    ax.axvline(k_cut, color="red", lw=0.9, ls="--", alpha=0.4)
    ax.text(k_cut * 0.95, ax.get_ylim()[1] * 0.3,
            "2/3 dealias  ", color="red", fontsize=9, va="top", ha="right", alpha=0.8)

    # 擬合線 + 陰影 CI
    def plot_fit(fit, color, theory, label):
        if fit is None:
            return
        kk = np.linspace(fit['k_min'], fit['k_max'], 50)
        yy = 10 ** (fit['intercept'] + fit['slope'] * np.log10(kk))
        # ± 1.96·SE 陰影帶 (約 95% CI)
        se = fit['slope_se']
        yy_hi = 10 ** (fit['intercept'] + (fit['slope'] + 1.96 * se) * np.log10(kk))
        yy_lo = 10 ** (fit['intercept'] + (fit['slope'] - 1.96 * se) * np.log10(kk))
        ax.fill_between(kk, yy_lo, yy_hi, color=color, alpha=0.15)
        ax.loglog(kk, yy, "-", color=color, lw=2.2,
                  label=f"{label}: slope = {fit['slope']:+.2f} ± {fit['slope_se']:.2f}  "
                        f"(theory {theory:+.2f})")
        # 理論斜率的 dashed 參考
        yy_th = yy * (kk / np.sqrt(fit['k_min'] * fit['k_max']))**(theory - fit['slope'])
        # 讓理論線通過擬合區間中心
        kc = np.sqrt(fit['k_min'] * fit['k_max'])
        yc = 10 ** (fit['intercept'] + fit['slope'] * np.log10(kc))
        yy_ref = yc * (kk / kc) ** theory
        ax.loglog(kk, yy_ref, "--", color=color, lw=1.0, alpha=0.55)

    plot_fit(fit_inv, "#2ea02e", THEORY_INV, "Inverse (k^{-5/3})")
    plot_fit(fit_ens, "#d62728", THEORY_ENS, "Enstrophy (k^{-3})")

    ax.set_xlabel(r"wavenumber $k$", fontsize=12)
    ax.set_ylabel(r"$E(k)$", fontsize=12)
    title = f"Kraichnan 2D turbulence spectrum fit — {os.path.basename(run_dir)}"
    if t is not None:
        title += f"  (t={t:.2f}, {nx}×{ny})"
    ax.set_title(title, fontsize=12)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=10, loc="lower left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    print(f"寫入 {out_path}")


if __name__ == "__main__":
    main()
