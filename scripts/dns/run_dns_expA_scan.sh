#!/usr/bin/env bash
# DNS Experiment A — post-bug-fix amp scan.
#
# Re-tests the claim in docs/dns_expA_triad_gpu_2026-05-04.md §4 that
# amp=1e-3 blows up around 45 periods.  With the three 2026-05-04 fixes
# (drop W advection, zero kx=0 Reynolds mode, Galerkin V_K closure on
# state + RHS) the stability range is expected to widen.
#
# Produces runs/dns_expA/triad_amp{...}.csv for each amplitude.  Override
# AMPS via env:  AMPS="1e-6 1e-5 1e-4 1e-3 3e-3 1e-2" ./scripts/run_dns_expA_scan.sh
set -euo pipefail

AMPS="${AMPS:-1e-6 1e-5 1e-4 1e-3 3e-3 1e-2}"
PERIODS="${PERIODS:-100}"
SPP="${SPP:-32}"
RHO_CUT="${RHO_CUT:-0.05}"
NR="${NR:-64}"
NTHETA="${NTHETA:-64}"

OUTDIR=runs/dns_expA
mkdir -p "$OUTDIR"

for amp in $AMPS; do
    tag=$(printf '%s' "$amp" | tr '+-' 'pm')
    echo ""
    echo "── amp=$amp   periods=$PERIODS   rho_cut=$RHO_CUT ──────────"
    ANSL_TD_KIND=strang_nonlinear \
    ANSL_COORD_MAP=tanh ANSL_COORD_BETA=2 \
    ANSL_DNS_PERIODS="$PERIODS" ANSL_DNS_SPP="$SPP" ANSL_RHO_CUT="$RHO_CUT" \
    ./build/stellar2d run --solver anelastic_sl --test dns_triad \
        --ntheta "$NTHETA" --nr "$NR" --ps-Lx 1 --ps-Ly 1 \
        --ps-k 1 --perturb "$amp" \
        --tend 1.0 --cfl 1.0 --ps-nu 0 2>&1 \
      | tee "$OUTDIR/run_amp${tag}.log"
    # Solver writes dns_triad.csv under a timestamped run dir; locate and
    # copy into the canonical location used by plot_dns_triad.py.
    latest=$(ls -td runs/dns_triad_* | head -1)
    cp "$latest/dns_triad.csv" "$OUTDIR/triad_amp${amp}.csv"
    echo "  → $OUTDIR/triad_amp${amp}.csv"
done

echo ""
echo "All runs done.  Plot with:"
echo "  AMPS=\"$AMPS\" python3 scripts/plot_dns_triad.py"
