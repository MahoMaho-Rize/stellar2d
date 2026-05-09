#!/usr/bin/env bash
# Andrassy 2022 scheme × resolution scan — 16 runs total.
#
# Matches the matrix in docs/andrassy_scan/SUMMARY_2026-05-07.md so the new
# (post-compensation + 2nd-order rebuild) results are drop-in replacements.
#
#   vl2  {128,256,512} × {VL, MM}                      = 6 runs
#   ale2 128 × {MUSCL-VL, MUSCL-MM, PPM-CS char}       = 3 runs
#   ale2 256 × {MUSCL-MC, MUSCL-VL, MUSCL-MM, PPM-CS prim} = 4 runs
#   ale2 512 × {MUSCL-VL, MUSCL-MM, PPM-CS char}       = 3 runs
#
# Usage:
#   bash scripts/andrassy2022/run_scheme_scan.sh 2>&1 | tee docs/andrassy_scan/scan_log.txt
#
# Output layout (keeps paper_scan_figure.py globs valid):
#   runs/athena_vl2_scan_v2/{128,256,512}_{vanleer,minmod}/andrassy2022_*/
#   runs/scheme_scan_128/{muscl_vanleer,muscl_minmod,ppm_cs_char}/andrassy2022_*/
#   runs/scheme_scan_256/{muscl_mc,muscl_vanleer,muscl_minmod,ppm_cs_prim}/andrassy2022_*/
#   runs/scheme_scan_512/{muscl_vanleer,muscl_minmod,ppm_cs_char}/andrassy2022_*/

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
    $BIN run --solver cart_ale2 --test andrassy2022 \
        --nr "$res" --ntheta "$res" --cfl 0.4 \
        --ic-slab "data/andrassy2022/pilot_ic_n${res}.txt" \
        --bc-x periodic --bc-y reflect \
        --tend "$TEND" --diag-interval 500 --vtk-dt 20 \
        --andrassy-amp "$AMP" \
        --remap-order 2 --remap-limiter "$limflag" $ppmflag \
        --run-base "$outdir" 2>&1 | tail -3
    # Move latest matching run dir into $outdir (run-base already points there,
    # so the auto-created andrassy2022_${res}x${res}_<ts> will be inside).
}

run_vl2 () {
    local res=$1 limtag=$2 limflag=$3 outdir=$4
    echo "[$(timestamp)] ==== vl2 $res $limtag ===="
    mkdir -p "$outdir"
    $BIN run --solver athena_vl2 --test andrassy2022 \
        --nr "$res" --ntheta "$res" --cfl 0.4 \
        --ic-slab "data/andrassy2022/pilot_ic_n${res}.txt" \
        --bc-x periodic --bc-y reflect \
        --tend "$TEND" --diag-interval 500 --vtk-dt 20 \
        --andrassy-amp "$AMP" \
        --athena-xorder 2 --athena-limiter "$limflag" \
        --run-base "$outdir" 2>&1 | tail -3
}

# ---------- vl2 (6 runs) ----------
run_vl2 128 VL vanleer runs/athena_vl2_scan_v2/128_vanleer
run_vl2 128 MM minmod  runs/athena_vl2_scan_v2/128_minmod
run_vl2 256 VL vanleer runs/athena_vl2_scan_v2/256_vanleer
run_vl2 256 MM minmod  runs/athena_vl2_scan_v2/256_minmod
run_vl2 512 VL vanleer runs/athena_vl2_scan_v2/512_vanleer
run_vl2 512 MM minmod  runs/athena_vl2_scan_v2/512_minmod

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

echo "[$(timestamp)] ==== scan complete ===="
