"""athena_vl2 — Sod shock tube test (compute_error in C++).

athena_vl2 now supports `--bc-x {periodic, reflect, outflow}`; for Sod we
use `outflow` (zero-gradient) to match Athena++'s Sod setup. Measured at
t=0.1, CFL=0.3, vanleer PLM+HLLC:

  N=128  L1=4.35e-3  Linf=7.7e-2
  N=256  L1=2.19e-3  Linf=7.7e-2    ratio = 0.50 → 1st-order (expected
                                    near a shock — PLM degrades to 1st
                                    at the discontinuity).

Before --bc-x outflow landed, the solver was hard-coded x-periodic and
L1 sat at 0.050 (wrap contamination); see commit TBD. The tight ratio
here will catch a regression of that hardcode coming back.
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
            "--solver", "athena_vl2", "--test", "sod",
            "--bc-x", "outflow",
            "--nr", str(nx), "--ntheta", str(nx),
            "--cfl", "0.3", "--tend", "0.1",
            "--athena-xorder", "2", "--athena-limiter", "vanleer",
            "--diag-interval", "100000", "--vtk-dt", "999",
            "--compute-error",
        ],
        run_base,
    )
    rows = read_error_dat(find_error_dat(run_base, "sod"))
    assert len(rows) == 1
    return rows[0]


@pytest.mark.fast
def test_sod_n128_absolute():
    r = _run(128, "vl2_sod_n128")
    assert r["Nx"] == 128
    # With outflow BC: L1 ≈ 4.4e-3 at N=128. Lock 2× the measured value
    # (catches ~factor-2 regression like a BC bug returning wrap, or
    # limiter activation issue).
    assert r["L1"] < 1.0e-2, f"L1={r['L1']:.3e} > 1e-2 (vl2 Sod degraded)"


@pytest.mark.fast
def test_sod_convergence_N128_to_N256():
    """vl2 Sod with outflow BC shows 1st-order shock convergence (expected
    for PLM+HLLC near a discontinuity): measured ratio ≈ 0.50.  Lock 0.65
    so a regression above 2nd-fold or total stagnation (ratio > 0.9)
    fires immediately."""
    r128 = _run(128, "vl2_sod_n128_pair")
    r256 = _run(256, "vl2_sod_n256_pair")
    ratio = r256["L1"] / r128["L1"]
    assert ratio < 0.65, (
        f"L1(256)/L1(128) = {ratio:.3f} ≥ 0.65 (vl2 Sod convergence lost)")
