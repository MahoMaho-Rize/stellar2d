"""cart_ale2 — Sod shock tube test (compute_error in C++).

Replaces the 120-line python Toro port in scripts/tests_ale2/sod_compare.py.
Exact solution lives in src/gpu/common/sod_exact.h, shared with athena_vl2.

L1 is scored on x ∈ [0.05, 0.95]·Lx (central 90%) — the 5% edge zones are
excluded because periodic-BC solvers (athena_vl2 hard-codes x-periodic)
see wrap contamination there; cart_ale2 with --bc-x reflect doesn't wrap
but we use the same window for apples-to-apples comparison.

Tolerances (from 2026-05-07 characterization, N=128, t=0.1):
  - L1 < 0.08  (actual ~0.042 at N=128, ~0.029 at N=256)
  - Linf < 0.30  (actual ~0.20 at N=128, ALE swept-remap smears contact)
  - L1(256)/L1(128) < 0.9  (partial convergence; ALE+AV is ~1st order
    on shocks, not 2nd, which is by design — catches blow-up not order)
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
            "--solver", "cart_ale2", "--test", "sod",
            "--bc-x", "reflect", "--bc-y", "reflect",
            "--nr", str(nx), "--ntheta", str(nx),
            "--cfl", "0.3", "--tend", "0.1",
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
    r = _run(128, "ale2_sod_n128")
    assert r["Nx"] == 128
    # cart_ale2 + swept-remap on Sod @ t=0.1: L1 ≈ 0.042, Linf ≈ 0.20.
    # Lock 2× above the measured baseline.
    assert r["L1"]   < 0.08, f"L1={r['L1']:.3e} > 0.08 (Sod blew up?)"
    assert r["Linf"] < 0.30, f"Linf={r['Linf']:.3e} > 0.30"


@pytest.mark.fast
def test_sod_convergence_N128_to_N256():
    """Lock that the L1 at 256² is not worse than at 128²."""
    r128 = _run(128, "ale2_sod_n128_pair")
    r256 = _run(256, "ale2_sod_n256_pair")
    # Partial convergence for ALE+AV on shocks (formal 1st order with
    # CFL/AV limiter). Measured ratio = 0.69. Lock 0.9 so a 2x
    # regression (ratio > 1.0) or total stall (ratio ≈ 1.0) fires.
    ratio = r256["L1"] / r128["L1"]
    assert ratio < 0.9, (
        f"L1(256)/L1(128) = {ratio:.3f} ≥ 0.9 (cart_ale2 Sod stagnated)")
