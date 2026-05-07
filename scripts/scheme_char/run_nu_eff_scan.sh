#!/usr/bin/env bash
# T3 (Phase A.1) — ν_eff comparison: shear-mode decay on 3 solvers.
#
# For each solver, run a zero-viscosity shear-mode decay with 4 resolutions
# and 2 injection amplitudes (V₀).  The measured exponential decay rate
# slope(log max|v|)/t gives ν_eff = −slope/k²_phys (compressible) or
# ν_eff = −slope/(2 k²_phys) (spectral Taylor-Green, cos·cos geometry).
#
# Usage:
#   scripts/scheme_char/run_nu_eff_scan.sh  [/path/to/stellar2d binary]
#
# Outputs:
#   docs/scheme_char/runs/{cart_ale2,athena_vl2,pseudo_spectral}/\
#     res_N_V_V0/diagnostics.csv
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${1:-$ROOT/build/stellar2d}"
OUT_ROOT="$ROOT/docs/scheme_char/runs"
mkdir -p "$OUT_ROOT"

if [[ ! -x "$BIN" ]]; then
    echo "error: binary not found or not executable: $BIN" >&2
    exit 1
fi

RES_LIST=(64 128 256 512)
V0_LIST=(0.01 0.1)
K=1
TEND=2.0
DIAG=10
VTK=0         # no VTK (diagnostics-only scan)

run_one() {
    local tag=$1 solver=$2 extra=$3 nx=$4 V0=$5
    local dir="$OUT_ROOT/${tag}/res_${nx}_V_${V0}"
    mkdir -p "$dir"
    if [[ -f "$dir/diagnostics.csv" && "${SKIP_DONE:-0}" == "1" ]]; then
        echo "  [skip] $dir (already has diagnostics.csv)"; return
    fi
    echo "=== ${tag}  res=${nx}  V0=${V0} ==="
    (cd "$dir" && \
     "$BIN" --solver "$solver" --nr "$nx" --ntheta "$nx" \
            --tend "$TEND" --diag-interval "$DIAG" \
            --run-base "$dir" --output-interval 10000 \
            $extra >run.log 2>&1)
    # The driver writes into $dir/runs/.../diagnostics.csv (via SimContext).
    # Flatten: copy the newest diagnostics.csv up to $dir/
    local latest=$(ls -1dt "$dir"/runs/*/diagnostics.csv 2>/dev/null | head -1 || true)
    if [[ -n "$latest" ]]; then
        cp "$latest" "$dir/diagnostics.csv"
        echo "  -> $dir/diagnostics.csv"
    else
        # Some drivers write diagnostics.csv directly in run_dir; fallback.
        echo "  [warn] no diagnostics.csv found under $dir/runs; check run.log"
    fi
}

for V0 in "${V0_LIST[@]}"; do
    for N in "${RES_LIST[@]}"; do
        run_one cart_ale2   cart_ale2 \
            "--test shear_mode --bc-x periodic --bc-y periodic \
             --shear-V0 $V0 --shear-k $K --shear-rho 1.0 --shear-P 1.0 \
             --cfl 0.3" \
            "$N" "$V0"

        run_one athena_vl2  athena_vl2 \
            "--test shear_mode \
             --shear-V0 $V0 --shear-k $K --shear-rho 1.0 --shear-P 1.0 \
             --cfl 0.3 --athena-xorder 2 --athena-limiter vanleer" \
            "$N" "$V0"

        run_one pseudo_spectral pseudo_spectral \
            "--test taylor_green --ps-tg-k $K --ps-nu 0 \
             --cfl 0.3 --ps-Lx 1.0 --ps-Ly 1.0" \
            "$N" "$V0"
    done
done

echo
echo "Scan complete.  Analyse with:"
echo "  python scripts/scheme_char/fit_nu_eff.py"
