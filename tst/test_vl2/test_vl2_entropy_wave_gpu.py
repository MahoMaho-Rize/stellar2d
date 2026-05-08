"""athena_vl2 — T1 entropy wave convergence test (compute_error in C++).

athena_vl2 is a textbook 2nd-order scheme (PLM + HLLC + vl2 predictor-
corrector), so this IC gives a clean p≈2 slope. We lock:
  - L1 at N=128 below a 2e-5 absolute threshold (characterization value
    from Phase A.3 was ~6e-6; leave 3× slack for GPU atomic noise).
  - Ratio L1(128)/L1(64) < 0.35 (log₂ 0.35 ≈ −1.5; safely above the
    p=2 nominal 0.25 with slack for limiter variation).

See scheme_char Phase A.3 (task #47) for baseline values.
"""
from __future__ import annotations

import pytest

from testutils import (
    fresh_run_base, run_stellar2d, read_error_dat, find_error_dat,
)


A = 0.01
K = 1
U0 = 1.0


def _run(nx: int, run_base_name: str):
    run_base = fresh_run_base(run_base_name)
    run_stellar2d(
        [
            "--solver", "athena_vl2",
            "--test", "entropy_wave",
            "--athena-xorder", "2", "--athena-limiter", "vanleer",
            "--nr", str(nx), "--ntheta", str(nx),
            "--cfl", "0.3",
            "--ewave-rho0", "1.0", "--ewave-P0", "1.0",
            "--ewave-u0", str(U0),
            "--ewave-A", str(A), "--ewave-k", str(K),
            "--ewave-periods", "1.0",
            "--diag-interval", "100000", "--vtk-dt", "999",
            "--compute-error",
        ],
        run_base,
    )
    rows = read_error_dat(find_error_dat(run_base, "entropy_wave"))
    assert len(rows) == 1
    return rows[0]


@pytest.mark.fast
def test_entropy_wave_n128_absolute():
    r = _run(128, "vl2_ewave_n128")
    assert r["Nx"] == 128
    # athena_vl2 PLM+vanleer on this IC hits ~6e-6 at N=128 (Phase A.3).
    # Lock 3× above that floor so atomic-noise run-to-run is not flaky.
    assert r["L1"] < 2.0e-5, f"L1={r['L1']:.3e} > 2e-5 (vl2 2nd-order broken?)"


@pytest.mark.fast
def test_entropy_wave_convergence_order():
    """2nd-order scheme: L1(128)/L1(64) < 0.35 (formal p>=1.5 with slack)."""
    r64  = _run(64,  "vl2_ewave_n64")
    r128 = _run(128, "vl2_ewave_n128_pair")
    ratio = r128["L1"] / r64["L1"]
    # 2nd-order nominal ratio = 0.25; allow 0.35 for limiter activation +
    # GPU atomic noise. If the scheme silently drops to 1st order
    # (ratio ≈ 0.5) this fires.
    assert ratio < 0.35, (
        f"L1(128)/L1(64) = {ratio:.3f} ≥ 0.35 — vl2 convergence degraded "
        f"(L1_64={r64['L1']:.3e}, L1_128={r128['L1']:.3e})")
