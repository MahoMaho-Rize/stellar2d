#!/usr/bin/env bash
# T6 / Phase B (task #48) — pseudo_spectral forced-turb spectrum smoke.
#
# Runs one canonical forced_turb at 256² and emits E(k) via the existing
# scripts/pseudo_spectral/spectrum_fit_pseudo_spectral.py tool.  This
# DOES NOT chase a pixel-perfect Kraichnan k⁻⁵/³ — per §10.5 of the
# scheme characterization plan, we record the spectrum as "scheme
# characteristic data", not a pass/fail regression.
#
# To do a proper Kraichnan inverse-cascade fit, longer t_end (~20-50
# eddy turnovers), lower drag, and a 512² grid are required; this
# smoke run is just enough to exercise the full pipeline and to archive
# a reproducible baseline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${1:-$ROOT/build/stellar2d}"
OUT_ROOT="$ROOT/docs/scheme_char/runs_spectra"
mkdir -p "$OUT_ROOT"

if [[ ! -x "$BIN" ]]; then
    echo "error: binary not found: $BIN" >&2; exit 1
fi

echo "=== pseudo_spectral forced_turb 256² ==="
"$BIN" run --solver pseudo_spectral --test forced_turb \
    --nr 256 --ntheta 256 --cfl 0.3 \
    --ps-Lx 1.0 --ps-Ly 1.0 --ps-nu 5e-5 \
    --ps-forcing-eps 0.1 --ps-forcing-kf 32 --ps-forcing-dk 1 \
    --ps-drag 0.1 --ps-hyper 1 \
    --tend 5.0 --diag-interval 500 --vtk-dt 1.0 \
    --frame-buffer \
    --run-base "$OUT_ROOT" 2>&1 | tail -3

run=$(ls -dt "$OUT_ROOT"/forced_turb_256x256_* | head -1)

echo
echo "=== spectrum fit ==="
PYTHONPATH="$ROOT/scripts/render" \
    python "$ROOT/scripts/pseudo_spectral/spectrum_fit_pseudo_spectral.py" \
        "$run/" \
        "$ROOT/docs/scheme_char/2026-05-07_forced_turb_spectrum.png" \
        5.0 32

echo "Artifacts:"
echo "  $ROOT/docs/scheme_char/2026-05-07_forced_turb_spectrum.png"
echo "  $run/frames.csv  (per-frame spectrum bins for downstream use)"
