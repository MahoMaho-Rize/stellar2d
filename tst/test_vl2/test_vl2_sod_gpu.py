"""athena_vl2 — Sod shock tube test (compute_error in C++).

athena_vl2 uses x-periodic BC (hard-coded), so at t=0.1 the left-running
rarefaction has wrapped and pollutes x≈0; the scored window [0.05, 0.95]
clips most of this. Current measured L1 at N=128 is ~0.050 — higher than
a "clean" 2nd-order Godunov Sod (~0.01) because of the wrap contamination.

This test therefore locks **regression**, not absolute quality:
  - L1 at N=128 < 0.10 (2× the measured baseline)
  - L1(256) not worse than L1(128) (ratio < 1.0)

Future work: athena_vl2 could support reflect-x BC for Sod, which would
drop L1 to ~0.01; when that lands, tighten these tolerances.
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
    # athena_vl2 + x-periodic BC + wrap pollution: L1 ≈ 0.050 at N=128.
    # Lock 2× above that.
    assert r["L1"] < 0.10, f"L1={r['L1']:.3e} > 0.10 (vl2 Sod blew up?)"


@pytest.mark.fast
def test_sod_convergence_N128_to_N256():
    """vl2 Sod with periodic BC is not converging on this IC (ratio ~0.9),
    so we only lock that finer grid is not WORSE than coarser."""
    r128 = _run(128, "vl2_sod_n128_pair")
    r256 = _run(256, "vl2_sod_n256_pair")
    ratio = r256["L1"] / r128["L1"]
    assert ratio < 1.0, (
        f"L1(256)/L1(128) = {ratio:.3f} ≥ 1.0 (vl2 Sod regressed)")
