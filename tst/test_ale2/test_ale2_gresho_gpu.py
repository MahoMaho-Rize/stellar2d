"""cart_ale2 — Gresho stationary vortex (compute_error in C++).

Gresho has an exact stationary solution with vφ(r) piecewise linear
peaking at 1.0 at r=0.2. We score L1/Linf of |v_sim − v_exact| inside
the vortex disk (r < 0.5); compute_gresho_error does the scoring on-device.

Tolerances (2026-05-07 measurements with --shear-aware-av --rebuild-order 1):
  - N=128  t=1.0  L1=1.5e-2  v_max=0.88
  - N=64   t=1.0  L1=3.5e-2  v_max=0.79
  - ratio  L1(128)/L1(64) ≈ 0.44 (~1.2-order)

Runbook expects L1 < 0.02 at N=128, t=3.0; we lock at t=1.0 to stay in
the fast bucket (<1s per run). Tolerances are 2× above measured.

Required flags:
  --shear-aware-av      — reduce AV in pure rotation (keeps v_max close
                          to 1.0 instead of dissipating to 0.2)
  --rebuild-order 1     — 2nd-order corner MUSCL node velocity rebuild
  --bc-x reflect --bc-y reflect
"""
from __future__ import annotations

import pytest

from testutils import (
    fresh_run_base, run_stellar2d, read_error_dat, find_error_dat,
)


def _run(nx: int, tag: str):
    run_base = fresh_run_base(tag)
    run_stellar2d(
        [
            "--solver", "cart_ale2", "--test", "gresho",
            "--bc-x", "reflect", "--bc-y", "reflect",
            "--shear-aware-av",
            "--rebuild-order", "1",
            "--nr", str(nx), "--ntheta", str(nx),
            "--cfl", "0.3", "--tend", "1.0",
            "--diag-interval", "100000", "--vtk-dt", "999",
            "--compute-error",
        ],
        run_base,
    )
    rows = read_error_dat(find_error_dat(run_base, "gresho"))
    assert len(rows) == 1
    return rows[0]


@pytest.mark.fast
def test_gresho_n128_absolute():
    r = _run(128, "ale2_gresho_n128")
    assert r["Nx"] == 128
    # Measured L1 = 1.5e-2 at N=128 t=1.0. Lock 3× above (reasonable guard
    # against AV toggle regressions which would dissipate fast).
    assert r["L1"]   < 0.045, f"L1={r['L1']:.3e} > 0.045 (Gresho dissipated)"
    assert r["Linf"] < 0.30,  f"Linf={r['Linf']:.3e} > 0.30"
    # v_max_sim captures the peak speed at r=0.2 — should be close to 1.0
    # (the IC peak). Measured 0.88; lock > 0.7 so heavy dissipation fires.
    assert r["v_max_sim"] > 0.7, (
        f"v_max_sim={r['v_max_sim']:.3f} ≤ 0.7 (vortex too dissipative)")


@pytest.mark.fast
def test_gresho_convergence_N64_to_N128():
    r64  = _run(64,  "ale2_gresho_n64_pair")
    r128 = _run(128, "ale2_gresho_n128_pair")
    ratio = r128["L1"] / r64["L1"]
    # Measured ratio ≈ 0.44 (partial 2nd order). Lock 0.7 so total stall
    # (ratio ≈ 1.0) or regression fires.
    assert ratio < 0.7, (
        f"L1(128)/L1(64) = {ratio:.3f} ≥ 0.7 (Gresho convergence broke)")
