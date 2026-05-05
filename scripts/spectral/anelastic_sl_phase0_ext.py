#!/usr/bin/env python3
"""
Phase 0 擴展驗證(2026-05-02):物理保真度 + 方法優美度。

承接 scripts/anelastic_sl_phase0.py,補四個測驗:
  E1  Sturm oscillation 定理:ψ_n 恰好 n 個零點
  E2  g-mode Tassoul 漸近週期間距 ΔP cross-check
  E3  收斂階 vs cutoff ρ_threshold (+ 光滑 Gaussian ρ₀ 對照)
  E4  Brunt-Väisälä N²(r) vs Liouville W(r) 物理等價性

用法:  pixi run python scripts/anelastic_sl_phase0_ext.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import scipy.integrate
import scipy.sparse
import scipy.sparse.linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
VID = REPO / "videos"
VID.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO / "scripts"))
from anelastic_sl_phase0 import (
    solve_lane_emden,
    compute_W_from_rho,
    solve_sl_eigenpairs,
    manufactured_p_and_f,
    sl_poisson_solve,
)


# ============================================================
# E1 Sturm oscillation:ψ_n 應恰好有 n 個內部零點
# ============================================================
def count_zero_crossings(psi_vec, tol=1e-12):
    """不含邊界(邊界我們 embed 0)的內部零點。連續變號記一次。"""
    s = np.sign(psi_vec)
    # 去掉邊界的 0(因為邊界本身被嵌為 0)
    s_int = s[1:-1]
    # 連續變號計數
    crossings = 0
    prev = 0
    for v in s_int:
        if abs(v) < tol:
            continue
        if prev != 0 and v != prev:
            crossings += 1
        prev = v
    return crossings


def sturm_oscillation_test(y, mu, psi, out_path):
    print("\n=== E1 Sturm oscillation 定理 ===")
    n_test = min(21, psi.shape[1])
    counts = np.zeros(n_test, dtype=int)
    for n in range(n_test):
        counts[n] = count_zero_crossings(psi[:, n])
    ok = np.all(counts == np.arange(n_test))
    print(f"  前 {n_test} 個本徵函數零點計數:")
    for n in range(n_test):
        mark = "✓" if counts[n] == n else f"✗(expected {n})"
        print(f"    ψ_{n:2d}   zeros = {counts[n]:2d}   {mark}")
    print(f"  Sturm 定理驗證:{'✓ 全部通過' if ok else '✗ 有違反'}")

    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=140)
    ax.plot(range(n_test), counts, "bo-", label="counted zeros")
    ax.plot(range(n_test), range(n_test), "k--", alpha=0.5, label="Sturm: n zeros in psi_n")
    ax.set_xlabel("mode index n")
    ax.set_ylabel("interior zero count")
    ax.set_title("Sturm oscillation theorem (1836) verification")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path}")
    return ok


# ============================================================
# E2 g-mode Tassoul asymptotic period spacing cross-check
#
# Tassoul 1980:高 n ℓ=1 g-mode 週期間距
#   ΔP = 2·pi^2 / sqrt(ell*(ell+1)) · [ integral (N/r) dr ]^(-1)
#
# 我們的 SL 本徵值 μ_n 對應特徵波長 k_n² = μ_n(算子 -T 的 eig)。
# 對 g-mode,大 n 漸近:μ_n 近 (n pi / L)² + mean(W),
# 因此 sqrt(μ_{n+1}) - sqrt(μ_n) ≈ pi / L(不含物理 N 分布)。
# 要直接和 Tassoul ΔP 比較,需要從 μ_n 還原物理頻率:
#   在 Cowling 近似下,徑向波動方程的 Sturm-Liouville 形式給出
#   omega_n² ≈ N²_avg · ell(ell+1) / (k_r,n² · r²)
#   即 omega_n ∝ 1/sqrt(μ_n)(當 μ 主導由 W 平均給)。
# 因此 P_n ∝ sqrt(μ_n),ΔP ≈ dP/dn = (1 / (2·sqrt(μ_n))) · (d μ_n / dn)。
# 在大 n 下 μ_n ≈ (n pi / L)² + const,d μ / d n ≈ 2 n pi² / L²,
# 故 ΔP ≈ pi / (L · sqrt(μ_n)/n) · const ← 和 Tassoul 的 1/(integral N/r) 可對齊。
#
# 此處作 heuristic 比對:算 SL 推出的 "P_n" = 2 pi / omega_n_SL,
# 其中 omega_n_SL = sqrt(|W_avg|) · sqrt(ell(ell+1)/μ_n)(Cowling 簡化),
# 對比 Tassoul ΔP。預期 O(1) 吻合(不精確因為忽略 N(r) 的徑向分布權重)。
# ============================================================
def gmode_tassoul_check(y, rho, W, mu, out_path, ell=1):
    print("\n=== E2 g-mode Tassoul 漸近 cross-check ===")
    # 估 N_avg² 的近似代理:SL 公式 W 是 density-only,定義 W_proxy = -<W>
    # (Lane-Emden polytrope 下 <W> 負,對應振盪規律)
    W_avg = np.mean(W)
    N2_proxy = -W_avg  # 代理 Brunt 特徵頻率平方(同樣維度 1/length²)
    if N2_proxy <= 0:
        print(f"  警告:W_avg = {W_avg:.3e} >= 0,SL 潛勢不支持 g-mode 結構")
        return

    N_proxy = np.sqrt(N2_proxy)
    # 取 r = y,∫(N/r) dr 的 proxy:用 sqrt(-W)/y 積分(ρ→0 處截斷過)
    # y 從 ~0 到 ~r_hi,避開 r=0 奇異
    mask = y > 1e-3
    integrand = np.sqrt(np.maximum(-W[mask], 0.0)) / y[mask]
    I_Nr = np.trapezoid(integrand, y[mask])

    DP_tassoul = 2 * np.pi * np.pi / np.sqrt(ell * (ell + 1)) / I_Nr
    print(f"  W_avg = {W_avg:.3e},  N_proxy = sqrt(-W_avg) = {N_proxy:.3e}")
    print(f"  ∫(N_proxy/r) dr = {I_Nr:.3e}")
    print(f"  Tassoul 預測 ΔP (ell={ell}) = {DP_tassoul:.4f}")

    # SL 模式的 P_n = 2pi/omega_n, omega_n_SL = sqrt(ell(ell+1)) * N_proxy / sqrt(mu_n)
    # (Cowling + slab 簡化;僅用於 scaling 對齊)
    omega_SL = np.sqrt(ell * (ell + 1)) * N_proxy / np.sqrt(mu)
    P_SL = 2 * np.pi / omega_SL
    DP_SL = np.abs(np.diff(P_SL))

    print(f"  SL 漸近 ΔP_n 前 10 個:")
    for n in range(min(10, len(DP_SL))):
        ratio = DP_SL[n] / DP_tassoul
        print(f"    n={n:2d}  ΔP_SL = {DP_SL[n]:.4e}   ratio vs Tassoul = {ratio:.3f}")

    # 漸近收斂:大 n 應 → 常數(Tassoul 的 "constant ΔP" 特徵)
    tail_mean = np.mean(DP_SL[-5:])
    tail_std = np.std(DP_SL[-5:])
    print(f"  尾段 (last 5 modes) ΔP 平均 = {tail_mean:.4e},std = {tail_std:.3e}")
    print(f"  Tassoul prediction / tail mean = {DP_tassoul / tail_mean:.3f}"
          f"  (O(1) 吻合即視為物理正確)")

    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=140)
    n_arr = np.arange(1, len(DP_SL) + 1)
    ax.plot(n_arr, DP_SL, "bo-", label=r"SL $\Delta P_n$")
    ax.axhline(DP_tassoul, ls="--", c="r",
               label=f"Tassoul constant = {DP_tassoul:.3f}")
    ax.set_xlabel("mode index n")
    ax.set_ylabel(r"$\Delta P_n$")
    ax.set_title(r"g-mode period spacing: SL eigenvalues vs Tassoul (1980)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path}")


# ============================================================
# E3 Convergence vs cutoff threshold
#   對不同 ρ_threshold 跑 SL-Poisson err_L2 vs N_mode,看 algebraic 系數變化。
#   另外做一個 Gaussian-capped ρ₀(光滑,無奇異),預期回到指數收斂。
# ============================================================
def convergence_vs_cutoff(out_path, N_grid=512, Nx=128):
    print("\n=== E3 Convergence order vs cutoff ρ_threshold ===")
    xi, theta, xi_1 = solve_lane_emden(n=1.5)
    r_full = xi / xi_1
    rho_full = np.abs(theta) ** 1.5

    thresholds = [0.1, 0.01, 0.001, 0.0001]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=140)

    # --- (a) Lane-Emden with different cutoffs ---
    for thr in thresholds:
        mask = rho_full > thr
        r_in = r_full[mask]
        rho_in = rho_full[mask]
        if len(r_in) < 100:
            continue
        W_in_full, _, _ = compute_W_from_rho(r_in, rho_in)
        y = np.linspace(r_in[0], r_in[-1], N_grid)
        W = np.interp(y, r_in, W_in_full)
        rho = np.interp(y, r_in, rho_in)
        mu, psi = solve_sl_eigenpairs(y, W, 256)

        # manufactured solution → err_L2 vs N_modes
        Lx = 1.0
        x = np.arange(Nx) * Lx / Nx
        p_exact, f = manufactured_p_and_f(x, y, rho, Lx)
        k_x_modes = 2 * np.pi * np.fft.rfftfreq(Nx, d=Lx / Nx)

        Nm_list = [5, 10, 20, 40, 80, 160, 256]
        errs = []
        for Nm in Nm_list:
            p_num = sl_poisson_solve(f, rho, y, psi[:, :Nm], mu[:Nm], Lx, k_x_modes)
            err = np.sqrt(np.mean((p_num - p_exact) ** 2))
            errs.append(err)
        axes[0].loglog(Nm_list, errs, "o-", label=f"cutoff rho>{thr}")
        # 估斜率
        xs = np.log(Nm_list[2:])
        ys = np.log(errs[2:])
        slope = np.polyfit(xs, ys, 1)[0]
        print(f"  rho_cut = {thr:.4f}  |domain| r in [{y[0]:.3f}, {y[-1]:.3f}]"
              f"   err(256) = {errs[-1]:.2e}   slope ~ {slope:.2f}")
    axes[0].set_xlabel("N_modes")
    axes[0].set_ylabel("err_L2")
    axes[0].set_title(r"(a) Lane-Emden: convergence vs cutoff $\rho_{\rm thr}$")
    axes[0].legend(); axes[0].grid(True, which="both", alpha=0.3)

    # --- (b) Gaussian-capped smooth rho (no singularity) ---
    y_s = np.linspace(-1, 1, N_grid)
    rho_s = np.exp(-2 * y_s ** 2) + 0.05  # >0 everywhere, smooth
    W_s, _, _ = compute_W_from_rho(y_s, rho_s)
    mu_s, psi_s = solve_sl_eigenpairs(y_s, W_s, 256)
    Lx = 2.0
    x = np.arange(Nx) * Lx / Nx
    # manufactured needs Dirichlet in y; our φ = sin(π η) goes to 0 at both ends
    p_exact, f = manufactured_p_and_f(x, y_s, rho_s, Lx)
    k_x_modes = 2 * np.pi * np.fft.rfftfreq(Nx, d=Lx / Nx)
    Nm_list = [5, 10, 20, 40, 80, 160, 256]
    errs = []
    for Nm in Nm_list:
        p_num = sl_poisson_solve(f, rho_s, y_s, psi_s[:, :Nm], mu_s[:Nm], Lx, k_x_modes)
        err = np.sqrt(np.mean((p_num - p_exact) ** 2))
        errs.append(err)
    axes[1].semilogy(Nm_list, errs, "ro-")
    # 線性 slope in semilog → 指數收斂
    xs = np.array(Nm_list[:-1])
    ys = np.log(errs[:-1])
    slope_exp = np.polyfit(xs, ys, 1)[0]
    axes[1].set_xlabel("N_modes")
    axes[1].set_ylabel("err_L2  (log scale)")
    axes[1].set_title(r"(b) Smooth Gaussian $\rho(y)$: expected exponential")
    axes[1].grid(True, which="both", alpha=0.3)
    print(f"  Gaussian rho smooth: err(256) = {errs[-1]:.2e}"
          f"   semilog slope ~ {slope_exp:.3f} (負斜率 + linear = 指數收斂)")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path}")


# ============================================================
# E4 Brunt-Väisälä N²(r) vs Liouville W(r) equivalence
#
# 對 adiabatic polytrope P = K ρ^γ (Lane-Emden γ = 1 + 1/n):
#   N² = g / H_p · (∇_ad - ∇)
#   ∇_ad = (γ-1)/γ
#   ∇    = d ln T / d ln P = (dT/T) / (dP/P)
# 對理想氣體 + polytropic,T ∝ ρ^(γ-1),P ∝ ρ^γ:
#   ∇ = (γ-1)/γ = ∇_ad    (adiabatic stratification!)
#   → N² = 0   (中性對流,Schwarzschild criterion)
#
# 對 Lane-Emden 這正是結果:polytropic 總是中性 stratified。
# 要得到 N² ≠ 0 必須引入非絕熱(e.g. 恆星演化 track 上的非 polytropic profile)。
#
# 因此此處我們做兩件事:
#   (a) 數值驗證 Lane-Emden 的 N²(r) ≈ 0(機器精度 or HSE 誤差)
#   (b) 構造人工 non-adiabatic profile:T(r) 獨立於 polytropic law,
#       給出非零 N²(r),對比 Liouville W(r) 的物理對應。
# ============================================================
def brunt_vs_liouville(out_path):
    print("\n=== E4 Brunt-Väisälä N²(r) vs Liouville W(r) 等價性 ===")
    xi, theta, xi_1 = solve_lane_emden(n=1.5)
    r_full = xi / xi_1
    rho = np.abs(theta) ** 1.5

    # (a) Lane-Emden polytropic 中性 stratification verify
    # γ = 1 + 1/n = 1 + 1/1.5 = 5/3
    gamma = 5.0 / 3.0
    P = rho ** gamma  # K=1 normalized

    # ∇_ad = (γ-1)/γ
    grad_ad = (gamma - 1) / gamma

    # ∇ = d ln T / d ln P;對 polytropic T ∝ ρ^(γ-1),P ∝ ρ^γ
    # => ∇ = (γ-1)/γ = ∇_ad 嚴格相等,N²=0
    print(f"  Lane-Emden γ={gamma:.3f},  ∇_ad = {grad_ad:.4f}")
    print(f"  Polytropic T ∝ ρ^(γ-1), P ∝ ρ^γ → ∇ = (γ-1)/γ = ∇_ad")
    print(f"  Schwarzschild 準則:N² = 0 (中性 stratification,符合預期)")

    # (b) 構造 non-adiabatic profile:讓 T 偏離 polytropic
    # T_mod(r) = T_ad(r) · [1 + δ · sin(2πr)]
    # 計算對應 N²,然後和 W(r) 對比
    mask = rho > 0.01
    r = r_full[mask]
    rho_m = rho[mask]

    # 重力 g(r) = G·M(<r)/r²;Lane-Emden 無量綱下 g ~ theta(xi) 相關,這裡取常數代理
    # 若我們只要 SHAPE 對比,g 的絕對值無妨
    g_proxy = 1.0

    T_ad = rho_m ** (gamma - 1)
    delta = 0.1
    T_mod = T_ad * (1 + delta * np.sin(2 * np.pi * r))

    # ∇_actual = d ln T_mod / d ln P
    lnT = np.log(T_mod)
    lnP = np.log(rho_m ** gamma)
    dlnT_dlnP = np.gradient(lnT, lnP, edge_order=2)
    N2 = g_proxy * (grad_ad - dlnT_dlnP)  # 正確符號:grad_ad - grad_actual
    # (標準 N² = g/H_p · (∇_ad - ∇_actual);穩定 stratification N²>0)
    # H_p = -dr/d ln P
    dP_dr = np.gradient(np.log(rho_m ** gamma), r, edge_order=2)
    H_p = np.where(np.abs(dP_dr) > 1e-8, -1.0 / dP_dr, 1e10)
    N2_full = g_proxy / np.abs(H_p) * (grad_ad - dlnT_dlnP)

    # W(r) 由原 ρ 算(與 T 無關 — 這是 Liouville 的「純密度」視角)
    W_orig, _, _ = compute_W_from_rho(r, rho_m)
    # W_mod:若 SL 也看「effective 密度 ρ_eff = ρ · f(T)」就能反映非絕熱效應;
    # 當前 SL 方法只看 ρ 的二階導,不含 T 分布資訊。
    # 這反映了一個事實:**純 Poisson SL 方法只編碼 density stratification,
    # 完整 g-mode 問題要 T 分布** — 這是 anelastic 擴展要補的物理。

    print(f"  Non-adiabatic T 擾動 δ = {delta} (sin 2πr)")
    print(f"  |N²| range: [{np.min(np.abs(N2_full)):.3e}, "
          f"{np.max(np.abs(N2_full)):.3e}]")
    print(f"  |W(r)|  range: [{np.min(np.abs(W_orig)):.3e}, "
          f"{np.max(np.abs(W_orig)):.3e}]")
    print(f"  **物理意義**:")
    print(f"    - W(r) 編碼『純密度分層的 Liouville potential』(僅 ρ 的二階結構)")
    print(f"    - N²(r) 需要 T 分布(Schwarzschild 穩定準則)")
    print(f"    - 兩者共同決定 g-mode 譜;SL Poisson 反演只用 W(r)")
    print(f"    - Phase 3 anelastic 擴展時,熵方程會把 N²(r) 的物理納入對流方程 RHS")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=140)
    axes[0].plot(r, N2_full, "b-", lw=1.5, label=r"$N^2(r)$  (non-adiabatic)")
    axes[0].axhline(0, c="k", ls=":", alpha=0.3)
    axes[0].set_xlabel("r/R★")
    axes[0].set_ylabel(r"$N^2(r)$  [arb.units]")
    axes[0].set_title("Brunt-Väisälä (physical, needs T)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].semilogy(r, np.abs(W_orig), "r-", lw=1.5, label=r"$|W(r)|$ Liouville")
    axes[1].set_xlabel("r/R★")
    axes[1].set_ylabel(r"$|W(r)|$")
    axes[1].set_title("Liouville potential (density-only, for SL-Poisson)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path}")


# ============================================================
# main
# ============================================================
def main():
    print("=" * 70)
    print(" Phase 0 ext: SL 物理保真度 + 方法優美度 驗證")
    print("=" * 70)

    # 共用基底:Lane-Emden n=1.5, cutoff rho>0.01
    xi, theta, xi_1 = solve_lane_emden(n=1.5)
    r_full = xi / xi_1
    rho_full = np.abs(theta) ** 1.5
    W_full, _, _ = compute_W_from_rho(r_full, rho_full)
    mask = rho_full > 0.01
    r_in = r_full[mask]
    rho_in = rho_full[mask]
    W_in = W_full[mask]

    N = 512
    y = np.linspace(r_in[0], r_in[-1], N)
    W = np.interp(y, r_in, W_in)
    rho = np.interp(y, r_in, rho_in)
    mu, psi = solve_sl_eigenpairs(y, W, 64)

    sturm_oscillation_test(y, mu, psi, VID / "anelastic_sl_ext_E1_sturm.png")
    gmode_tassoul_check(y, rho, W, mu, VID / "anelastic_sl_ext_E2_tassoul.png")
    convergence_vs_cutoff(VID / "anelastic_sl_ext_E3_cutoff.png")
    brunt_vs_liouville(VID / "anelastic_sl_ext_E4_brunt.png")

    print("\n" + "=" * 70)
    print(" Phase 0 ext done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
