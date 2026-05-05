#!/usr/bin/env bash
# DNS Experiment A — long-time (500 T_a) on the stable weakly-nonlinear
# amplitudes.  Purpose: verify that post-fix O(dt²) Strang drift is
# *bounded* (∝ t, rate independent of amp) rather than secularly
# accumulating.  The 100-T_a scan already showed dev/T ∝ amp and
# E_k2/E_k1 ∝ amp²; long-time checks the drift's asymptotic character.
#
# Pass criterion (see plot_dns_triad.py summary):
#   - 500-T_a dev/T and E_k1 drift rate within 10% of 100-T_a values
#     for the same amp → drift is linear, bounded; pass.
#   - Otherwise superlinear → residual nonlinear contamination; fail.
#
# Default amp set {1e-6, 1e-5, 1e-4} are known stable from the 100-T_a
# run.  Larger amps (1e-3, 3e-3, 1e-2) belong to the stability-boundary
# scan (run_dns_expA_scan.sh), not long-time.
set -euo pipefail

AMPS="${AMPS:-1e-6 1e-5 1e-4}"
PERIODS="${PERIODS:-500}"
SPP="${SPP:-32}"
RHO_CUT="${RHO_CUT:-0.05}"
NR="${NR:-64}"
NTHETA="${NTHETA:-64}"

OUTDIR=runs/dns_expA_longtime
mkdir -p "$OUTDIR"

for amp in $AMPS; do
    echo ""
    echo "── amp=$amp   periods=$PERIODS   rho_cut=$RHO_CUT ──────────"
    ANSL_TD_KIND=strang_nonlinear \
    ANSL_COORD_MAP=tanh ANSL_COORD_BETA=2 \
    ANSL_DNS_PERIODS="$PERIODS" ANSL_DNS_SPP="$SPP" ANSL_RHO_CUT="$RHO_CUT" \
    ./build/stellar2d --solver anelastic_sl --test dns_triad \
        --ntheta "$NTHETA" --nr "$NR" --ps-Lx 1 --ps-Ly 1 \
        --ps-k 1 --perturb "$amp" \
        --tend 1.0 --cfl 1.0 --ps-nu 0 2>&1 \
      | tee "$OUTDIR/run_amp${amp}.log"
    latest=$(ls -td runs/dns_triad_* | head -1)
    cp "$latest/dns_triad.csv" "$OUTDIR/triad_amp${amp}.csv"
    echo "  → $OUTDIR/triad_amp${amp}.csv"
done

echo ""
echo "Plot (comparing against 100-T_a baseline in runs/dns_expA):"
echo "  AMPS=\"$AMPS\" LONGTIME=1 python3 scripts/plot_dns_triad.py"
