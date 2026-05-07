#!/usr/bin/env bash
# T4 / Phase B (task #48) — Jensen probe: ν_eff(k) at fixed resolution.
#
# Reuses the T3 shear_mode machinery with fixed N=256, sweeping k from
# 1 (smooth) to N/4 = 64 (approaching Nyquist).  A true Laplacian
# viscosity would fit the same ν_eff regardless of k; a Jensen-ILES
# scheme shows ν_eff that may rise with k (the scheme's effective
# dissipation over-damps grid-scale modes), while hyperdissipation
# (spectral 2/3-rule) would fall at low k.
#
# Usage:
#   scripts/scheme_char/run_jensen_probe_scan.sh [stellar2d binary]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${1:-$ROOT/build/stellar2d}"
OUT_ROOT="$ROOT/docs/scheme_char/runs_jensen"
mkdir -p "$OUT_ROOT"

if [[ ! -x "$BIN" ]]; then
    echo "error: binary not found: $BIN" >&2; exit 1
fi

N=256
K_LIST=(1 2 4 8 16 32 64)
V0=0.01
TEND=0.5      # shorter than T3's 2.0; high-k modes decay faster (λ ∝ k²)
DIAG=10

run_one() {
    local tag=$1 solver=$2 extra=$3 kk=$4
    local dir="$OUT_ROOT/${tag}/k_${kk}"
    mkdir -p "$dir"
    echo "=== ${tag}  k=${kk} ==="
    (cd "$dir" && \
     "$BIN" --solver "$solver" --nr $N --ntheta $N --cfl 0.3 \
            --tend $TEND --diag-interval $DIAG \
            --shear-V0 $V0 --shear-k "$kk" --shear-rho 1.0 --shear-P 1.0 \
            --run-base "$dir" --output-interval 10000 \
            $extra >run.log 2>&1)
    local latest
    latest=$(find "$dir" -maxdepth 2 -name diagnostics.csv -printf '%T@ %p\n' \
             2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)
    if [[ -n "$latest" && -f "$latest" ]]; then
        cp "$latest" "$dir/diagnostics.csv"
        echo "  -> $dir/diagnostics.csv"
    else
        echo "  [warn] no diagnostics.csv; check run.log"
    fi
}

for K in "${K_LIST[@]}"; do
    run_one cart_ale2  cart_ale2 \
        "--test shear_mode --bc-x periodic --bc-y periodic" "$K"
    run_one athena_vl2 athena_vl2 \
        "--test shear_mode --athena-xorder 2 --athena-limiter vanleer" "$K"
done

echo
echo "Scan complete.  Analyse with:"
echo "  python scripts/scheme_char/fit_jensen_probe.py"
