"""athena_vl2 — Linear acoustic wave (Athena++ linwave core test).

This is the single best convergence test for a Godunov scheme: smooth
sinusoidal perturbation, one period round-trip, no shocks to activate
limiters. Clean 2nd-order convergence is the expected signature of a
correctly-working PLM + HLLC + vl2 stack.

Measured 2026-05-07 at CFL=0.3, A=1e-4, k=1, γ=5/3, one period:
  N=64   L1=6.82e-7   Linf=2.71e-6
  N=128  L1=1.66e-7   Linf=1.01e-6  (ratio 0.24 ≈ log₂=2.03)
  N=256  L1=5.57e-8   Linf=3.73e-7  (ratio 0.34 ≈ log₂=1.55)

The N=256 ratio is slightly above 2nd-order floor (1/4 = 0.25) because
at A=1e-4 we're approaching the limiter-floor / round-off regime; at
larger amplitudes A ~ 1e-3 the ratio sharpens back toward 0.25.
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
            "--solver", "athena_vl2", "--test", "acoustic_wave",
            "--nr", str(nx), "--ntheta", str(nx),
            "--cfl", "0.3",
            "--athena-xorder", "2", "--athena-limiter", "vanleer",
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
    r = _run(128, "vl2_linwave_n128")
    assert r["Nx"] == 128
    # Measured L1 = 1.66e-7 at N=128. Lock 5× above.
    assert r["L1"] < 1e-6, (
        f"L1={r['L1']:.3e} > 1e-6 (vl2 2nd-order broken on linwave?)"
    )


@pytest.mark.fast
def test_linwave_convergence_2nd_order():
    """Key assertion: ratio L1(128)/L1(64) < 0.35 = log₂(0.35)≈−1.5."""
    r64  = _run(64,  "vl2_linwave_n64_pair")
    r128 = _run(128, "vl2_linwave_n128_pair")
    ratio = r128["L1"] / r64["L1"]
    # Measured 0.24 (textbook 2.03-order). Lock 0.35 — anything worse
    # indicates limiter activation, flux sign regression, or PLM bug.
    assert ratio < 0.35, (
        f"L1(128)/L1(64) = {ratio:.3f} ≥ 0.35 — vl2 lost 2nd-order on linwave")
