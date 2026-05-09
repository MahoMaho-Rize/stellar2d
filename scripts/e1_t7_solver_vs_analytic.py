r"""
E1-T7 solver-vs-analytic comparison on the SAME (y, t) mesh.

Purpose.  The T7 test measures RMS(v_x) at several heights from a
finite sample of solver snapshots, and compares the ratio to the exact
Hankel envelope ratio.  This could mask TWO separate errors:

  (A) true discretisation error in the solver (PLM dissipation, BC
      imperfection, CT-MHD top-cell issues);
  (B) finite-sample RMS bias (only 6 periods averaged, non-integer CFL
      step phase → RMS has a 1-few % aliasing bias).

This script isolates (A) from (B) by computing the exact analytic
v_x(y, t) at the SAME (y_k, t_i) samples written by the solver to
t7_timeseries.csv, then running the identical RMS pipeline on both.
The ratio (solver RMS ratio) / (analytic-on-mesh RMS ratio) is a
pure measure of (A).

Analytic model (stratified isothermal atm, linear Alfvén wave injected
at the bottom with amplitude A_drv·√2 peak, single-mode sinusoidal):

  v_A(y) = B0 / √ρ(y) = B0 · exp(y/2H) / √ρ_0
  Wave equation in y:  ∂²_t v_x = ∂_y(v_A²(y) ∂_y v_x)
  Upgoing-wave solution in complex form:
    v_x(y, t) = Re[A_0 · H_0^{(2)}(ξ(y)) · exp(-i ω t + i φ_bc)]
  where ξ(y) = (2H·ω/B0) · exp(-y/2H) · √ρ_0  (Hankel argument).

Boundary condition at y = y_d (driver row, cell-centred): match
driver waveform  v_x(y_d, t) = A_drv·√2·sin(ω·t + φ_drv).
Solve for the constant A_0·φ_bc from this matching; then evaluate
v_x at each (y_k, t_i) in the CSV.

Usage:
  python scripts/e1_t7_solver_vs_analytic.py \
      build/t7_timeseries.csv
"""
from __future__ import annotations
import sys
import csv
from pathlib import Path
import numpy as np
import mpmath as mp

mp.mp.dps = 40


def hankel_vx_upgoing(y, t, omega, H, B0, rho0, y_d, A_peak, phi_drv):
    r"""
    Exact upgoing-wave Hankel solution for v_x(y, t) matching a
    sinusoidal driver at y = y_d.

    v_A(y)   = B0 · exp(y/(2H)) / sqrt(rho0)
    k(y)     = omega / v_A(y)
    ξ(y)     = (2H ω / B0) · exp(-y/(2H)) · sqrt(rho0)
              (so dξ/dy = -ξ/(2H), and Bessel's equation is satisfied)

    Upgoing-wave solution (outgoing at y → ∞):
      v_x(y, t) = Re[ C · H_0^{(2)}(ξ(y)) · exp(-i ω t) ]

    Matching at y = y_d to  v_x(y_d, t) = A_peak · sin(ω t + phi_drv):
      sin(ω t + phi_drv) = Im[ exp(i(ω t + phi_drv)) ]
                         = -Im[ exp(-i(ω t + phi_drv)) ]
                         = -Im[ exp(-i ω t) · exp(-i phi_drv) ]
      Re[C · H_0^{(2)}(ξ_d) · exp(-i ω t)] = A_peak · sin(ω t + phi_drv)
    So C · H_0^{(2)}(ξ_d) = −i · A_peak · exp(+i phi_drv)
    ⇒  C = −i · A_peak · exp(+i phi_drv) / H_0^{(2)}(ξ_d)
    """
    # Use mpmath for high-precision Hankel
    xi = lambda yy: (2 * H * omega / B0) * mp.exp(-yy / (2 * H)) * mp.sqrt(rho0)
    xi_d = xi(y_d)
    H0_xi_d = mp.hankel2(0, xi_d)
    # C = -i · A_peak · exp(+i phi_drv) / H_0^(2)(ξ_d)
    C = (-1j) * A_peak * mp.exp(1j * phi_drv) / H0_xi_d
    # Evaluate for a (possibly vectorised) (y, t)
    def one(y_val, t_val):
        H0_xi = mp.hankel2(0, xi(y_val))
        vx_complex = C * H0_xi * mp.exp(-1j * omega * t_val)
        return float(mp.re(vx_complex))
    if np.isscalar(y) and np.isscalar(t):
        return one(y, t)
    # Array form: broadcast
    y_arr = np.atleast_1d(y)
    t_arr = np.atleast_1d(t)
    out = np.empty((len(t_arr), len(y_arr)))
    # Pre-compute H_0(ξ(y)) per y (expensive): do once
    H0_per_y = np.array([complex(mp.hankel2(0, xi(float(yy)))) for yy in y_arr])
    C_np = complex(C)
    for i, tv in enumerate(t_arr):
        vx_complex = C_np * H0_per_y * np.exp(-1j * float(omega) * float(tv))
        out[i] = vx_complex.real
    return out


def solver_driver_waveform(t, A_rms, f, phi):
    r"""
    The §E1 single-mode driver normalisation:
      v_x^drv(t) = A_N · sin(2π f t + phi),  A_N = A_rms · √(2/N_modes)
    With N_modes = 1, A_N = A_rms · √2.
    Peak amplitude of v_x at the driver ghost row equals A_rms · √2.
    """
    A_N = A_rms * np.sqrt(2.0)
    return A_N * np.sin(2 * np.pi * f * t + phi)


def main(csv_path: str):
    print(f"== E1-T7 solver vs analytic-on-mesh comparison ==")
    print(f"   loading {csv_path}")
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        field_names = reader.fieldnames
        for row in reader:
            rows.append(row)
    if not rows:
        print("ERROR: empty CSV", file=sys.stderr)
        return 1

    # Extract heights from column headers (vx_y<value>)
    height_cols = [c for c in field_names if c.startswith("vx_y")]
    heights = np.array([float(c[len("vx_y"):]) for c in height_cols])
    n_y = len(heights)

    # Metadata columns are constant per row; pull from the first row.
    meta = {
        "A_rms":     float(rows[0]["A_drv"]),
        "f_drive":   float(rows[0]["f_drive"]),
        "H":         float(rows[0]["H"]),
        "B0y":       float(rows[0]["B0y"]),
        "rho0":      float(rows[0]["rho0"]),
        "yd_bottom": float(rows[0]["yd_bottom"]),
    }
    print(f"   heights: {heights}")
    print(f"   meta: {meta}")
    print(f"   n_samples = {len(rows)}")

    # Convert to arrays
    t_arr = np.array([float(r["t"]) for r in rows])
    vx_solver = np.array([[float(r[c]) for c in height_cols] for r in rows])

    # Solver driver uses seed-derived phase phi; we recover phi empirically
    # by fitting sin(2π f t + phi) to the driver row waveform.  Actually
    # we don't have the driver row v_x recorded; but we can recover phi
    # by fitting the solver's y_0 (lowest recorded height) using Hankel
    # theory OR by just scanning phi ∈ [0, 2π] and minimising RMS error
    # against the ground-truth solver waveform at y_0.
    #
    # Simpler: bracket phi empirically by matching the OUTGOING-wave
    # Hankel amplitude AND phase at y_0 = 0.254.
    omega = 2 * np.pi * meta["f_drive"]
    A_peak = meta["A_rms"] * np.sqrt(2.0)

    # Fit phi_drv so that analytic v_x at (y=heights[0], t=t_arr) best
    # matches solver v_x at that same column.  This isolates the phase
    # of the driver (which depends on seed + phi_n drawn).
    # Fit: v_solver[t, 0] = analytic(y_0, t, phi); minimise L2 over phi.
    best_phi = 0.0
    best_err = 1e30
    for phi_trial in np.linspace(0, 2 * np.pi, 361):
        v_an_trial = np.array([
            hankel_vx_upgoing(heights[0], t_arr[i], omega,
                              meta["H"], meta["B0y"], meta["rho0"],
                              meta["yd_bottom"], A_peak, phi_trial)
            for i in range(min(len(t_arr), 50))
        ])
        err = np.mean((v_an_trial - vx_solver[:50, 0])**2)
        if err < best_err:
            best_err = err
            best_phi = phi_trial
    # Coarse-fit refinement
    for phi_trial in np.linspace(best_phi - 0.02, best_phi + 0.02, 41):
        v_an_trial = np.array([
            hankel_vx_upgoing(heights[0], t_arr[i], omega,
                              meta["H"], meta["B0y"], meta["rho0"],
                              meta["yd_bottom"], A_peak, phi_trial)
            for i in range(min(len(t_arr), 50))
        ])
        err = np.mean((v_an_trial - vx_solver[:50, 0])**2)
        if err < best_err:
            best_err = err
            best_phi = phi_trial
    print(f"   fitted phi_drv = {best_phi:.4f} rad "
          f"(residual L2 = {np.sqrt(best_err):.3e})")

    # Compute analytic on the full (y, t) mesh
    print(f"   computing Hankel analytic on full {len(t_arr)} × {n_y} mesh...")
    vx_analytic = hankel_vx_upgoing(heights, t_arr, omega,
                                     meta["H"], meta["B0y"], meta["rho0"],
                                     meta["yd_bottom"], A_peak, best_phi)

    # RMS at each height
    rms_solver   = np.sqrt(np.mean(vx_solver**2, axis=0))
    rms_analytic = np.sqrt(np.mean(vx_analytic**2, axis=0))

    # Exact Hankel RMS prediction (infinite-time average, "Parseval"):
    #   |v_x(y)|_pk = |C · H_0^(2)(ξ(y))|
    #   RMS = |v_x|_pk / √2
    xi_y = lambda yy: ((2 * meta["H"] * omega) / meta["B0y"]) * \
                      mp.exp(-mp.mpf(yy) / (2 * meta["H"])) * mp.sqrt(meta["rho0"])
    H0_xi_d = mp.hankel2(0, xi_y(meta["yd_bottom"]))
    C_analytic = A_peak / abs(H0_xi_d)
    rms_exact = np.array([
        float(C_analytic * abs(mp.hankel2(0, xi_y(yy))) / mp.sqrt(2))
        for yy in heights
    ])

    print()
    print(f"  {'y':>7s} {'RMS_solver':>14s} {'RMS_analytic@mesh':>20s} "
          f"{'RMS_exact':>14s}")
    for k, yy in enumerate(heights):
        print(f"  {yy:7.4f} {rms_solver[k]:14.6e} {rms_analytic[k]:20.6e} "
              f"{rms_exact[k]:14.6e}")

    y1_idx, y2_idx = 0, len(heights) - 1
    R_solver   = rms_solver[y2_idx]   / rms_solver[y1_idx]
    R_analytic = rms_analytic[y2_idx] / rms_analytic[y1_idx]
    R_exact    = rms_exact[y2_idx]    / rms_exact[y1_idx]

    print()
    print(f"  R_solver         = {R_solver:.6f}  (solver RMS ratio)")
    print(f"  R_analytic@mesh  = {R_analytic:.6f}  (Hankel RMS on SAME sample mesh)")
    print(f"  R_exact          = {R_exact:.6f}  (Hankel infinite-time RMS)")
    print()
    print(f"  Decomposition of solver error:")
    err_solver_vs_exact  = R_solver / R_exact - 1
    err_analytic_vs_exact = R_analytic / R_exact - 1       # sampling-window bias
    err_solver_vs_analytic = R_solver / R_analytic - 1      # true discretisation
    print(f"    solver/exact - 1   = {err_solver_vs_exact * 100:+.3f}%   "
          f"(total error, what T7 tests)")
    print(f"    analytic@mesh/exact - 1 = {err_analytic_vs_exact * 100:+.3f}%   "
          f"(sampling-window bias; goes to zero at long T)")
    print(f"    solver/analytic@mesh - 1 = {err_solver_vs_analytic * 100:+.3f}%   "
          f"(TRUE solver discretisation error, no sampling bias)")

    return 0


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "build/t7_timeseries.csv"
    sys.exit(main(csv_path))
