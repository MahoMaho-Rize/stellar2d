#!/usr/bin/env bash
# T1 / Phase A.3 — entropy wave smooth convergence (task #47).
#
# cart_ale2 and athena_vl2 advect ρ(x, 0) = ρ0·(1 + A·sin(k·2π x/Lx))
# with uniform P, v = (u0, 0), periodic BC.  After one period the
# solution should return to the IC; L1 error on ρ at t=T measures the
# scheme's formal order for smooth advection.
#
# Usage:
#   scripts/scheme_char/run_entropy_wave_scan.sh [stellar2d binary]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${1:-$ROOT/build/stellar2d}"
OUT_ROOT="$ROOT/docs/scheme_char/runs_ewave"
mkdir -p "$OUT_ROOT"

if [[ ! -x "$BIN" ]]; then
    echo "error: binary not found: $BIN" >&2; exit 1
fi

RES_LIST=(64 128 256 512)
A=0.01
K=1
U0=1.0
PERIODS=1.0

run_one() {
    local tag=$1 solver=$2 extra=$3 nx=$4
    local dir="$OUT_ROOT/${tag}/res_${nx}"
    mkdir -p "$dir"
    echo "=== ${tag}  res=${nx} ==="
    (cd "$dir" && \
     "$BIN" --solver "$solver" --nr "$nx" --ntheta "$nx" --cfl 0.3 \
            --ewave-rho0 1.0 --ewave-P0 1.0 --ewave-u0 "$U0" \
            --ewave-A "$A" --ewave-k "$K" --ewave-periods "$PERIODS" \
            --diag-interval 10000 --vtk-dt 999 \
            --run-base "$dir" $extra >run.log 2>&1)
    # Pick the highest-numbered output_NNNN.vtk — that's the t_end dump
    # by the driver's `do_vtk = (t >= t_end)` rule.  (output_final.vtk is
    # written by main.cpp from the *polar-mesh* ctx.state and is all zero
    # for Cartesian solvers — do not use it.)
    local vtk
    vtk=$(find "$dir" -maxdepth 2 -name 'output_[0-9]*.vtk' 2>/dev/null |
          sort | tail -1)
    if [[ -n "$vtk" && -f "$vtk" ]]; then
        cp "$vtk" "$dir/output_final.vtk"
        echo "  -> $dir/output_final.vtk  (from $(basename "$vtk"))"
    else
        echo "  [warn] no output_NNNN.vtk under $dir; check run.log"
    fi
}

for N in "${RES_LIST[@]}"; do
    run_one cart_ale2  cart_ale2  "--test entropy_wave --bc-x periodic --bc-y periodic" "$N"
    run_one athena_vl2 athena_vl2 "--test entropy_wave --athena-xorder 2 --athena-limiter vanleer" "$N"
done

echo
echo "Scan complete.  Analyse with:"
echo "  python scripts/scheme_char/fit_entropy_wave.py"
