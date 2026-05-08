#!/usr/bin/env bash
# Phase A2 — Orszag-Tang spectrum resolution scan.
# Runs stellar2d at N ∈ {128, 256, 512} on the orszag_tang test,
# writes VTK to runs/ot_N<N>/output_*.vtk, to be analysed by
# analyze_ot_spectrum.py.
#
# Derivation: docs/derivations/mhd/sections/f2_mhd_turbulence_spectrum.md
#
# NOTE: the binary writes to <run_base>/<test>_<N>x<N>_<timestamp>/
# (auto-timestamped, no --run-dir flag).  We point --run-base at a
# per-N dir then rename the timestamp subdir to "final" for a stable
# path the analyzer can find.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BIN="$ROOT/build/stellar2d"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: need built stellar2d binary at $BIN" >&2
    echo "  cmake --build build --target stellar2d" >&2
    exit 1
fi

mkdir -p "$HERE/runs"
for N in 128 256 512; do
    BASE="$HERE/runs/ot_N${N}"
    FINAL="$BASE/final"
    if [[ -f "$FINAL/output_final.vtk" ]]; then
        echo "  [skip] $FINAL already has output_final.vtk"
        continue
    fi
    rm -rf "$BASE"
    mkdir -p "$BASE"
    echo "  [N=$N] running OT to t=0.5 ..."
    "$BIN" \
        --solver athena_mhd \
        --test orszag_tang \
        --nr $N --ntheta $N \
        --tend 0.5 \
        --output-interval 100000 \
        --vtk-interval 100000 \
        --run-base "$BASE" \
        2>&1 | tail -8

    # Binary writes to BASE/orszag_tang_<N>x<N>_<ts>/ — rename to 'final'
    TS_DIR=$(ls -d "$BASE"/orszag_tang_* 2>/dev/null | head -1)
    if [[ -z "$TS_DIR" ]]; then
        echo "  [error] no timestamped output dir in $BASE" >&2
        exit 1
    fi
    mv "$TS_DIR" "$FINAL"
    echo "    → $FINAL"
done
echo "done."
