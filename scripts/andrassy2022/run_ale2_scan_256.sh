#!/usr/bin/env bash
# 256-only ale2 scheme scan, rerun after 2026-05-07 P33 local-compat fix.
#
# Purpose: verify v_rms no longer collapses to 0.009 on 256²
# (the pre-P33 global mean-field compensation heated stable layers
#  and systematically suppressed convection amplitude).
#
# 4 schemes: MUSCL-MC / MUSCL-VL / MUSCL-MM / PPM-CS-prim-char
# CFL 0.4, rebuild_order=0 (default, stratified instability guard).

set -euo pipefail

cd "$(dirname "$0")/../.."
BIN=build/stellar2d
TEND=2000
AMP=5e-5
RES=256

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: build/stellar2d missing. Run: cmake --build build -j" >&2
    exit 1
fi

timestamp() { date +"%F %T"; }

run_ale2 () {
    local tag=$1 limflag=$2 ppmflag=$3 outdir=$4
    echo "[$(timestamp)] ==== ale2 ${RES} $tag ===="
    mkdir -p "$outdir"
    $BIN run --solver cart_ale2 --test andrassy2022 \
        --nr "$RES" --ntheta "$RES" --cfl 0.4 \
        --ic-slab "data/andrassy2022/pilot_ic_n${RES}.txt" \
        --bc-x periodic --bc-y reflect \
        --tend "$TEND" --diag-interval 500 --vtk-dt 20 \
        --andrassy-amp "$AMP" \
        --remap-order 2 --remap-limiter "$limflag" $ppmflag \
        --run-base "$outdir" 2>&1 | tail -3
}

run_ale2 MUSCL-MC    mc       ""  runs/scheme_scan_256/muscl_mc
run_ale2 MUSCL-VL    vanleer  ""  runs/scheme_scan_256/muscl_vanleer
run_ale2 MUSCL-MM    minmod   ""  runs/scheme_scan_256/muscl_minmod
run_ale2 PPM-CS-prim vanleer "--ppm --ppm-limiter cs --ppm-space prim --ppm-char" \
                                  runs/scheme_scan_256/ppm_cs_prim

echo "[$(timestamp)] ==== ale2 256 scan complete ===="
