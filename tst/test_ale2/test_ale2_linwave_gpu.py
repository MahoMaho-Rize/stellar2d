"""cart_ale2 — Linear acoustic wave (Athena++ linwave core test).

Right-running sound wave: δρ = A·sin(kx), δvx = c₀·δρ/ρ₀, δP = c₀²·δρ
on uniform (ρ₀, P₀, v=0). After one period T = Lx/c₀ the exact solution
returns to the IC (periodic BC).

Measured 2026-05-07 at CFL=0.3, A=1e-4, k=1, γ=5/3, one period:
  N=64   L1=1.59e-5  L1_phase=1.58e-5
  N=128  L1=8.53e-6  L1_phase=8.50e-6  (ratio 0.54 vs N=64)
  N=256  L1=4.42e-6  L1_phase=4.41e-6  (ratio 0.52 vs N=128)

cart_ale2's Lagrangian rebuild is ~1st-order on smooth advection (known
scheme_char Phase A.3 finding), so linwave ratios are ~0.5 rather than
the formal 2nd-order ~0.25. Tests lock regression, not order.
"""
from __future__ import annotations

import pytest

from testutils import (
    fresh_run_base, run_stellar2d, read_error_dat, find_error_dat,
)


A = 1e-4
K = 1


def _run(nx: int, tag: str):
    run_base = fresh_run_base(tag)
    run_stellar2d(
        [
            "--solver", "cart_ale2", "--test", "acoustic_wave",
            "--bc-x", "periodic", "--bc-y", "periodic",
            "--nr", str(nx), "--ntheta", str(nx),
            "--cfl", "0.3",
            "--awave-rho0", "1.0", "--awave-P0", "0.6",
            "--awave-A", str(A), "--awave-k", str(K),
            "--awave-periods", "1.0",
            "--diag-interval", "100000", "--vtk-dt", "999",
            "--compute-error",
        ],
        run_base,
    )
    rows = read_error_dat(find_error_dat(run_base, "acoustic_wave"))
    assert len(rows) == 1
    return rows[0]


@pytest.mark.fast
def test_linwave_n128_absolute():
    r = _run(128, "ale2_linwave_n128")
    assert r["Nx"] == 128
    # Measured L1_phase = 8.5e-6. Lock 3× above.
    assert r["L1_phase"] < 3e-5, (
        f"L1_phase={r['L1_phase']:.3e} > 3e-5 (linwave dissipation regressed)")


@pytest.mark.fast
def test_linwave_convergence_N64_to_N128():
    r64  = _run(64,  "ale2_linwave_n64_pair")
    r128 = _run(128, "ale2_linwave_n128_pair")
    ratio = r128["L1"] / r64["L1"]
    # Measured ratio 0.54 (~0.9-order). Lock 0.75 so a regression to
    # 1st-order-or-worse (ratio ≈ 0.9) fires.
    assert ratio < 0.75, (
        f"L1(128)/L1(64) = {ratio:.3f} ≥ 0.75 (cart_ale2 linwave stalled)")
