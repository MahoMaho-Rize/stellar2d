#!/usr/bin/env bash
# ale2-only rerun: 10 runs after the 2026-05-07 fix pack.
#
# cart_ale2 default now rebuild_order=0 (the 2nd-order rebuild has a
# stratified-atmosphere instability on reflect-wall long-t convection;
# see docs/andrassy_scan/README.md). remap_order=2 (MUSCL) stays default
# because it is the mature, physics-critical feature that defines ale2's
# sharp-interface advantage over Godunov schemes.
#
# Matrix (matches paper_scan_figure.py globs):
#   ale2 128 × {MUSCL-VL, MUSCL-MM, PPM-CS char}        = 3 runs
#   ale2 256 × {MUSCL-MC, MUSCL-VL, MUSCL-MM, PPM-CS}    = 4 runs
#   ale2 512 × {MUSCL-VL, MUSCL-MM, PPM-CS char}        = 3 runs
#
# vl2 already rerun 2026-05-07 and lives in runs/athena_vl2_scan_v2/.

set -euo pipefail

cd "$(dirname "$0")/../.."
BIN=build/stellar2d
TEND=2000
AMP=5e-5

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: build/stellar2d missing. Run: cmake --build build -j" >&2
    exit 1
fi

timestamp() { date +"%F %T"; }

run_ale2 () {
    local res=$1 tag=$2 limflag=$3 ppmflag=$4 outdir=$5
    echo "[$(timestamp)] ==== ale2 $res $tag ===="
    mkdir -p "$outdir"
    $BIN --solver cart_ale2 --test andrassy2022 \
        --nr "$res" --ntheta "$res" --cfl 0.4 \
        --ic-slab "data/andrassy2022/pilot_ic_n${res}.txt" \
        --bc-x periodic --bc-y reflect \
        --tend "$TEND" --diag-interval 500 --vtk-dt 20 \
        --andrassy-amp "$AMP" \
        --remap-order 2 --remap-limiter "$limflag" $ppmflag \
        --run-base "$outdir" 2>&1 | tail -3
}

# ---------- ale2 128 (3 runs) ----------
run_ale2 128 MUSCL-VL vanleer ""                                runs/scheme_scan_128/muscl_vanleer
run_ale2 128 MUSCL-MM minmod  ""                                runs/scheme_scan_128/muscl_minmod
run_ale2 128 PPM-CS-char vanleer "--ppm --ppm-limiter cs --ppm-space prim --ppm-char" \
                                                                runs/scheme_scan_128/ppm_cs_char

# ---------- ale2 256 (4 runs) ----------
run_ale2 256 MUSCL-MC mc      ""                                runs/scheme_scan_256/muscl_mc
run_ale2 256 MUSCL-VL vanleer ""                                runs/scheme_scan_256/muscl_vanleer
run_ale2 256 MUSCL-MM minmod  ""                                runs/scheme_scan_256/muscl_minmod
run_ale2 256 PPM-CS-prim vanleer "--ppm --ppm-limiter cs --ppm-space prim --ppm-char" \
                                                                runs/scheme_scan_256/ppm_cs_prim

# ---------- ale2 512 (3 runs) ----------
run_ale2 512 MUSCL-VL vanleer ""                                runs/scheme_scan_512/muscl_vanleer
run_ale2 512 MUSCL-MM minmod  ""                                runs/scheme_scan_512/muscl_minmod
run_ale2 512 PPM-CS-char vanleer "--ppm --ppm-limiter cs --ppm-space prim --ppm-char" \
                                                                runs/scheme_scan_512/ppm_cs_char

echo "[$(timestamp)] ==== ale2 scan complete ===="
