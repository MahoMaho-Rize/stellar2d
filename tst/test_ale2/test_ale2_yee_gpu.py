"""cart_ale2 — Yee-Vinokur-Djomehri isentropic vortex (compute_error in C++).

Yee has a closed-form isentropic vortex solution on [-5,5]² periodic;
after one full period T = Lx/u_inf = 10 the state returns to the IC.
However, running to full t=10 at N=128 takes ~5 minutes (too slow for
fast-bucket) and requires non-default flags (--ppm or --rebuild-order 1
with careful AV tuning) to avoid dissipating the vortex core.

This test uses **short-t** (t=1.0 = 1/10 period) as a fast-bucket smoke:
the vortex drifts by 1 unit out of 10, so we're really testing that the
remap doesn't blow up a smooth vortex core over ~100 Lagrangian-rezone
cycles. Full-period convergence belongs in a `slow` variant run offline.

Measured (2026-05-07, default flags, cfl=0.3, t=1.0):
  - N=64   Ncycle=69   L1=2.6e-2  Linf=0.40  ρ∈[0.62,1.05]
  - N=128  Ncycle=142  L1=2.9e-2  Linf=0.43  ρ∈[0.55,1.16]

Lock: ρ stays bounded (no blow-up), L1 < 0.10.  True convergence on
this IC requires `--rebuild-order 1 --ppm` at N=256 running for t=10
and is tracked by the runbook, not this unit test.
"""
from __future__ import annotations

import pytest

from testutils import (
    fresh_run_base, run_stellar2d, read_error_dat, find_error_dat,
)


def _run(nx: int, tag: str, tend: float = 1.0):
    run_base = fresh_run_base(tag)
    run_stellar2d(
        [
            "--solver", "cart_ale2", "--test", "yee_vortex",
            "--bc-x", "periodic", "--bc-y", "periodic",
            "--nr", str(nx), "--ntheta", str(nx),
            "--cfl", "0.3", "--tend", str(tend),
            "--diag-interval", "100000", "--vtk-dt", "999",
            "--compute-error",
        ],
        run_base,
    )
    rows = read_error_dat(find_error_dat(run_base, "yee"))
    assert len(rows) == 1
    return rows[0]


@pytest.mark.fast
def test_yee_n128_short_t_smoke():
    r = _run(128, "ale2_yee_n128_short")
    assert r["Nx"] == 128
    # After t=1 (1/10 period): L1 ≈ 0.03, Linf ≈ 0.43. Blow-up would
    # push L1 > 0.5 easily (seen when rebuild-order 1 goes unstable on
    # this IC). Lock L1 < 0.10 as a generous guard.
    assert r["L1"] < 0.10, f"L1={r['L1']:.3e} > 0.10 (vortex dissipated)"
    # IC has ρ_max ≈ 1.05 at center and ρ_min ≈ 0.52 at edge. Lock
    # non-blow-up bounds with margin: an unstable run pushes ρ_max > 8.
    assert r["rho_max"] < 2.0, (
        f"ρ_max={r['rho_max']:.3f} ≥ 2.0 (Yee blew up)")
    assert r["rho_min"] > 0.1, (
        f"ρ_min={r['rho_min']:.3f} ≤ 0.1 (Yee developed vacuum)")


@pytest.mark.fast
def test_yee_convergence_smoke():
    """N=64 vs N=128 at t=1: L1 shouldn't worsen dramatically."""
    r64  = _run(64,  "ale2_yee_n64_pair")
    r128 = _run(128, "ale2_yee_n128_pair")
    # Measured ratio ≈ 1.1 at t=1 (remap diffusion dominates Linf, not
    # a clean convergence case at this short t). Lock ratio < 2.0 so
    # ratio doubling or blow-up fires.
    ratio = r128["L1"] / r64["L1"]
    assert ratio < 2.0, (
        f"L1(128)/L1(64) = {ratio:.3f} ≥ 2.0 (Yee got much worse at higher N)")
