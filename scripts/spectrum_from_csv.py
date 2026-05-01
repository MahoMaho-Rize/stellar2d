#!/usr/bin/env python3
"""從 `spectrum.csv` (solver 實時輸出 / backfill 生成) 產 3 種分析圖。

CSV schema:
  header : index,step,t,<k0>,<k1>,...,<k_{nbin-1}>
  row    : idx,step,t,E(k0),E(k1),...

規則與 solver 端 (compute_spectrum_bins) 一致,Σ E_k·dk = KE(t)。

模式:
  fit      單幀譜 + log-log 線性回歸 (Kraichnan k^{-5/3} / k^{-3})
  heatmap  E(k,t) 熱圖 (log10 E 的 pcolormesh)
  overlay  多 run 譜疊圖

Usage:
  spectrum_from_csv.py fit     <csv> <t_snap> [k_lo k_hi] [out.png]
                       —— k_lo,k_hi 可指定 inertial/enstrophy range(mode 單位,預設 auto)
                       —— t_snap = -1 取最後一幀
  spectrum_from_csv.py heatmap <csv> [out.png]
  spectrum_from_csv.py overlay <csv1> [<csv2> ...] --out <out.png>
                       —— 每個 csv 取最後一幀譜疊圖

特殊 sentinel k_lo=k_hi=0 → 全自動 Kraichnan 雙段(需要 --kf <k_mode>)。
"""
import os, sys, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ───────────────────────────────────────────────────────────
# CSV 載入
# ───────────────────────────────────────────────────────────
def load_spectrum_csv(path):
    """回傳 (k_axis, t_arr, step_arr, E_mat).  E_mat shape = (n_frames, n_bins)."""
    with open(path) as fh:
        header = fh.readline().strip().split(",")
    k_axis = np.array([float(x) for x in header[3:]])
    rows = np.loadtxt(path, delimiter=",", skiprows=1)
    if rows.ndim == 1:
        rows = rows[None, :]
    idx_arr  = rows[:, 0].astype(int)
    step_arr = rows[:, 1].astype(int)
    t_arr    = rows[:, 2]
    E_mat    = rows[:, 3:]
    return k_axis, t_arr, step_arr, E_mat


# ───────────────────────────────────────────────────────────
# Fit (log-log 線性回歸)
# ───────────────────────────────────────────────────────────
def loglog_linregress(k, E, k_lo, k_hi):
    m = (k >= k_lo) & (k <= k_hi) & (E > 0)
    x = np.log10(k[m]); y = np.log10(E[m])
    n = len(x)
    if n < 3:
        return None
    xbar, ybar = x.mean(), y.mean()
    Sxx = ((x - xbar) ** 2).sum()
    Sxy = ((x - xbar) * (y - ybar)).sum()
    slope = Sxy / Sxx
    intercept = ybar - slope * xbar
    y_pred = slope * x + intercept
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - ybar) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    slope_se = np.sqrt(ss_res / max(n - 2, 1) / Sxx)
    return dict(slope=slope, intercept=intercept, slope_se=slope_se,
                r2=r2, n=n, k_lo=k_lo, k_hi=k_hi)


def pick_frame(t_arr, t_req):
    if t_req is None or t_req < 0:
        return len(t_arr) - 1
    return int(np.argmin(np.abs(t_arr - t_req)))


def load_diagnostics(run_dir):
    """從 diagnostics.csv 讀 t, nu, eps_enstrophy。回傳 None 若缺欄位。"""
    p = os.path.join(run_dir, "diagnostics.csv")
    if not os.path.exists(p):
        return None
    try:
        d = np.genfromtxt(p, delimiter=",", names=True)
        if 't' not in d.dtype.names:
            return None
        # nu 反推:eps_KE = 2·ν·Ω  →  ν = eps_KE / (2·Ω)
        if 'eps_KE' in d.dtype.names and 'enstrophy' in d.dtype.names:
            Omega = np.maximum(d['enstrophy'], 1e-30)
            nu_arr = d['eps_KE'] / (2.0 * Omega)
        else:
            nu_arr = None
        eps_ens = d['eps_enstrophy'] if 'eps_enstrophy' in d.dtype.names else None
        return dict(t=d['t'], nu=nu_arr, eps_enstrophy=eps_ens)
    except Exception:
        return None


def estimate_k_scales(run_dir, t_snap):
    """回傳 (nu, eps_enstrophy, k_eta_ens) 或 (None, None, None)."""
    D = load_diagnostics(run_dir)
    if D is None or D['eps_enstrophy'] is None or D['nu'] is None:
        return None, None, None
    # 找最接近 t_snap 的 diag
    idx = int(np.argmin(np.abs(D['t'] - t_snap)))
    nu      = float(D['nu'][idx])
    eps_ens = float(D['eps_enstrophy'][idx])
    if nu <= 0 or eps_ens <= 0:
        return None, None, None
    # Kraichnan enstrophy dissipation wavenumber: k_η_ens = (η_diss / ν³)^{1/6}
    k_eta_ens = (eps_ens / (nu ** 3)) ** (1.0 / 6.0)
    return nu, eps_ens, k_eta_ens


def auto_split_with_guard(k, kf_mode, k_eta_ens, k_cut, verbose=True):
    """回傳 (inv_rng, enst_rng);若某段 decade 不足 0.5 則回 None 並印原因。"""
    k_inj = kf_mode * 2.0 * np.pi if kf_mode is not None else None

    # Inverse cascade: [dk, k_inj·0.7]
    dk = k[1] if len(k) > 1 else 1.0
    inv_rng = None
    if k_inj is not None and k_inj > 3.0 * dk:
        inv_rng = (dk, k_inj * 0.7)
        dec = np.log10(inv_rng[1] / inv_rng[0])
        if dec < 0.5:
            if verbose:
                print(f"  [auto-split] inverse range {inv_rng} 只有 {dec:.2f} decade < 0.5 → 跳過")
            inv_rng = None
        elif verbose:
            print(f"  [auto-split] inverse range = [{inv_rng[0]:.1f}, {inv_rng[1]:.1f}] ({dec:.2f} decade)")

    # Enstrophy cascade: [k_inj·1.3, min(k_η_ens·0.5, k_cut·0.9)]
    enst_rng = None
    if k_inj is not None:
        k_hi = k_cut * 0.9
        if k_eta_ens is not None:
            k_hi = min(k_hi, k_eta_ens * 0.5)
        k_lo = k_inj * 1.3
        if k_hi > k_lo:
            enst_rng = (k_lo, k_hi)
            dec = np.log10(k_hi / k_lo)
            if dec < 0.5:
                if verbose:
                    k_eta_str = f"k_η_ens={k_eta_ens:.1f}" if k_eta_ens else "k_η_ens=?"
                    print(f"  [auto-split] enstrophy range {enst_rng} 只有 {dec:.2f} decade < 0.5 → 跳過")
                    print(f"              ({k_eta_str}, k_cut={k_cut:.1f}, k_inj·1.3={k_lo:.1f})")
                    print(f"              建議降 ν 或降 k_f 讓 inertial range 開到 ≥ 1 decade")
                enst_rng = None
            elif verbose:
                print(f"  [auto-split] enstrophy range = [{enst_rng[0]:.1f}, {enst_rng[1]:.1f}] ({dec:.2f} decade)")
    return inv_rng, enst_rng


# ───────────────────────────────────────────────────────────
# Mode: fit
# ───────────────────────────────────────────────────────────
def cmd_fit(argv):
    if len(argv) < 2:
        print(__doc__); sys.exit(1)
    csv = argv[0]
    t_req = float(argv[1])
    # 解析 [k_lo k_hi] 與 [out.png]
    rest = argv[2:]
    out_png = None
    k_lo = k_hi = None
    kf_mode = None
    # 掃 --kf / 位置參數
    i = 0
    pos = []
    while i < len(rest):
        if rest[i] == "--kf" and i + 1 < len(rest):
            kf_mode = int(rest[i+1]); i += 2
        else:
            pos.append(rest[i]); i += 1
    if len(pos) >= 2:
        try:
            k_lo = float(pos[0]); k_hi = float(pos[1])
            if len(pos) >= 3: out_png = pos[2]
        except ValueError:
            out_png = pos[0]
    elif len(pos) == 1:
        out_png = pos[0]
    if out_png is None:
        out_png = os.path.join(os.path.dirname(csv) or ".", "spectrum_fit.png")

    k, t_arr, _, E_mat = load_spectrum_csv(csv)
    fi = pick_frame(t_arr, t_req)
    E = E_mat[fi]
    t = t_arr[fi]
    print(f"frame {fi}: t={t:.4f}, Σ E·dk = {(E*(k[1]-k[0])).sum():.4e}")

    fig, ax = plt.subplots(figsize=(10, 7), dpi=130)
    m = (k > 0) & (E > 0)
    ax.loglog(k[m], E[m], "-", color="#2260aa", lw=1.4, label=f"E(k), t={t:.2f}")

    # Kraichnan injection 線
    if kf_mode is not None:
        k_inj = kf_mode * 2.0 * np.pi
        ax.axvline(k_inj, color="gray", lw=0.9, ls=":", alpha=0.7)
        ax.text(k_inj * 1.05, ax.get_ylim()[1] * 0.3,
                f"k_inj = {kf_mode}·2π ≈ {k_inj:.1f}",
                color="gray", fontsize=9, va="top")

    # 2/3 dealias cutoff (假 L=1)
    nx_approx = 3 * len(k)  # rough: nbins = N/2+1 → N ≈ 2(nbins-1); k_cut = N/3·2π
    N = 2 * (len(k) - 1)
    k_cut = (N / 3.0) * 2.0 * np.pi
    ax.axvline(k_cut, color="red", lw=0.9, ls="--", alpha=0.4)
    ax.text(k_cut * 0.95, ax.get_ylim()[1] * 0.3, "2/3 dealias  ",
            color="red", fontsize=9, va="top", ha="right", alpha=0.8)

    # Fit
    if k_lo is not None and k_hi is not None:
        # 單段 fit (顯式範圍)
        fit = loglog_linregress(k, E, k_lo, k_hi)
        if fit:
            theory = -3.0 if kf_mode is None or k_lo > kf_mode * 2 * np.pi \
                     else -5.0 / 3.0
            plot_fit(ax, fit, color="#d62728", theory=theory,
                     label=f"fit [{k_lo:.1f}, {k_hi:.1f}]")
            print(f"slope = {fit['slope']:+.3f} ± {fit['slope_se']:.3f}"
                  f"  R²={fit['r2']:.3f}  n={fit['n']}")
    elif kf_mode is not None:
        # 自動雙段 Kraichnan — 讀 diagnostics.csv 算 k_η_ens,range 太窄時拒絕 fit
        run_dir = os.path.dirname(csv) or "."
        nu_est, eps_ens, k_eta_ens = estimate_k_scales(run_dir, t)
        if k_eta_ens is not None:
            print(f"  物理尺度 (t={t:.2f}):  ν={nu_est:.2e}, η_diss={eps_ens:.2e}, "
                  f"k_η_ens={k_eta_ens:.1f}")
        inv_rng, enst_rng = auto_split_with_guard(k, kf_mode, k_eta_ens, k_cut)
        f_inv = loglog_linregress(k, E, *inv_rng) if inv_rng else None
        f_ens = loglog_linregress(k, E, *enst_rng) if enst_rng else None
        # k_η_ens 垂直線
        if k_eta_ens is not None:
            ax.axvline(k_eta_ens, color="purple", lw=0.9, ls="-.", alpha=0.55)
            ax.text(k_eta_ens * 1.03, ax.get_ylim()[1] * 0.08,
                    f" k_η_ens≈{k_eta_ens:.0f}", color="purple", fontsize=9, va="top")
        if f_inv:
            plot_fit(ax, f_inv, "#2ea02e", -5.0/3.0, "Inverse (k^{-5/3})")
            print(f"  Inverse   slope = {f_inv['slope']:+.3f} ± {f_inv['slope_se']:.3f}"
                  f"  R²={f_inv['r2']:.3f}  n={f_inv['n']}")
        if f_ens:
            plot_fit(ax, f_ens, "#d62728", -3.0, "Enstrophy (k^{-3})")
            print(f"  Enstrophy slope = {f_ens['slope']:+.3f} ± {f_ens['slope_se']:.3f}"
                  f"  R²={f_ens['r2']:.3f}  n={f_ens['n']}")
        if not f_inv and not f_ens:
            print("  兩段 inertial range 都太窄無法 fit。請降 ν 或降 k_f。")

    ax.set_xlabel(r"wavenumber $k$")
    ax.set_ylabel(r"$E(k)$")
    ax.set_title(f"{os.path.basename(os.path.dirname(csv))}  (t={t:.2f})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    print(f"寫入 {out_png}")


def plot_fit(ax, fit, color, theory, label):
    kk = np.linspace(fit['k_lo'], fit['k_hi'], 80)
    yy = 10 ** (fit['intercept'] + fit['slope'] * np.log10(kk))
    se = fit['slope_se']
    yy_hi = 10 ** (fit['intercept'] + (fit['slope'] + 1.96*se) * np.log10(kk))
    yy_lo = 10 ** (fit['intercept'] + (fit['slope'] - 1.96*se) * np.log10(kk))
    ax.fill_between(kk, yy_lo, yy_hi, color=color, alpha=0.15)
    ax.loglog(kk, yy, "-", color=color, lw=2.2,
              label=f"{label}: {fit['slope']:+.2f} ± {fit['slope_se']:.2f} "
                    f"(th {theory:+.2f})")
    # 理論參考 dashed
    kc = np.sqrt(fit['k_lo'] * fit['k_hi'])
    yc = 10 ** (fit['intercept'] + fit['slope'] * np.log10(kc))
    yy_ref = yc * (kk / kc) ** theory
    ax.loglog(kk, yy_ref, "--", color=color, lw=1.0, alpha=0.55)


# ───────────────────────────────────────────────────────────
# Mode: heatmap
# ───────────────────────────────────────────────────────────
def cmd_heatmap(argv):
    if len(argv) < 1:
        print(__doc__); sys.exit(1)
    csv = argv[0]
    out_png = argv[1] if len(argv) > 1 else \
              os.path.join(os.path.dirname(csv) or ".", "spectrum_heatmap.png")

    k, t_arr, _, E_mat = load_spectrum_csv(csv)
    # log₁₀ E (遮罩 E<=0)
    E_safe = np.where(E_mat > 1e-30, E_mat, 1e-30)
    logE = np.log10(E_safe)
    # pcolormesh 用 edge (cell-centered → edges)
    k_edges = np.concatenate(([0], 0.5 * (k[:-1] + k[1:]), [k[-1] * 1.5]))
    if len(t_arr) > 1:
        dt = np.diff(t_arr)
        t_edges = np.concatenate(([t_arr[0] - 0.5 * dt[0]],
                                   0.5 * (t_arr[:-1] + t_arr[1:]),
                                   [t_arr[-1] + 0.5 * dt[-1]]))
    else:
        t_edges = np.array([t_arr[0] - 1e-3, t_arr[0] + 1e-3])

    fig, ax = plt.subplots(figsize=(12, 6), dpi=130)
    # 截斷 colormap 范圍到有用的動態
    vmax = np.percentile(logE[E_mat > 1e-20], 99) if np.any(E_mat > 1e-20) else 0
    vmin = vmax - 8.0
    mesh = ax.pcolormesh(t_edges, k_edges, logE.T,
                         shading="auto", cmap="inferno",
                         vmin=vmin, vmax=vmax)
    ax.set_yscale("log")
    ax.set_xlabel("time")
    ax.set_ylabel(r"wavenumber $k$")
    ax.set_ylim(k[1], k[-1])
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label(r"$\log_{10} E(k, t)$")
    ax.set_title(f"Energy spectrum heatmap — {os.path.basename(os.path.dirname(csv))}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    print(f"寫入 {out_png}  (nframes={len(t_arr)}, nbins={len(k)})")


# ───────────────────────────────────────────────────────────
# Mode: overlay
# ───────────────────────────────────────────────────────────
def cmd_overlay(argv):
    out_png = None
    csvs = []
    i = 0
    while i < len(argv):
        if argv[i] == "--out" and i + 1 < len(argv):
            out_png = argv[i+1]; i += 2
        else:
            csvs.append(argv[i]); i += 1
    if not csvs:
        print(__doc__); sys.exit(1)
    if out_png is None:
        out_png = "spectrum_overlay.png"

    fig, ax = plt.subplots(figsize=(10, 7), dpi=130)
    cmap = plt.get_cmap("tab10")
    for i, csv in enumerate(csvs):
        k, t_arr, _, E_mat = load_spectrum_csv(csv)
        E = E_mat[-1]
        m = (k > 0) & (E > 0)
        label = os.path.basename(os.path.dirname(csv)) or csv
        ax.loglog(k[m], E[m], "-", color=cmap(i % 10), lw=1.4,
                  label=f"{label}  (t={t_arr[-1]:.2f})")
    ax.set_xlabel(r"wavenumber $k$")
    ax.set_ylabel(r"$E(k)$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.9)
    ax.set_title("Spectrum overlay")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    print(f"寫入 {out_png}  ({len(csvs)} runs)")


# ───────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__); sys.exit(1)
    mode = sys.argv[1]
    argv = sys.argv[2:]
    if mode == "fit":
        cmd_fit(argv)
    elif mode == "heatmap":
        cmd_heatmap(argv)
    elif mode == "overlay":
        cmd_overlay(argv)
    else:
        print(f"Unknown mode: {mode}\n" + __doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
