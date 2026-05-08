#!/usr/bin/env bash
# Phase A2 — Orszag-Tang spectrum resolution scan.
# Runs stellar2d at N ∈ {128, 256, 512} on the orszag_tang test,
# writes VTK to runs/ot_N<N>/output_*.vtk, to be analysed by
# analyze_ot_spectrum.py.
#
# Derivation: docs/mhd_derivations/sections/f2_mhd_turbulence_spectrum.md
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BIN="$ROOT/build/stellar2d"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: need built stellar2d binary at $BIN" >&2
    exit 1
fi

mkdir -p "$HERE/runs"
for N in 128 256 512; do
    RUN_DIR="$HERE/runs/ot_N${N}"
    if [[ -f "$RUN_DIR/output_0002.vtk" ]]; then
        echo "  [skip] $RUN_DIR already has t=0.5 VTK"
        continue
    fi
    mkdir -p "$RUN_DIR"
    echo "  [N=$N] running OT to t=0.5 ..."
    "$BIN" \
        --solver athena_mhd \
        --test orszag_tang \
        --nr $N --ntheta $N \
        --t-end 0.5 \
        --output-interval 100000 \
        --vtk-interval 100000 \
        --run-dir "$RUN_DIR" \
        2>&1 | tail -5
done
echo "done."
