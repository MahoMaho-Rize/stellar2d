#!/bin/bash
# Preconditioner benchmark: run 300 steps with each PC, collect metrics.
set -e
cd "$(dirname "$0")"

BIN=./stellar2d
# output-interval=10 to get Step lines every 10 steps
COMMON="--test lane_emden_perturbed --nr 64 --ntheta 32 --tend 100 --output-interval 10 --solver lowmach"

PRECONDS="block_jacobi line_jacobi simple block_schur"
RESULTS_DIR=bench_results
mkdir -p $RESULTS_DIR

for PC in $PRECONDS; do
    echo "========== Running PC=$PC =========="
    OUTFILE="$RESULTS_DIR/${PC}.log"
    # Separate stdout (Step lines) and stderr (Newton debug).
    # Merge for now, but also capture progress bars.
    timeout 180 $BIN $COMMON --precond $PC > "$RESULTS_DIR/${PC}_stdout.log" 2> "$RESULTS_DIR/${PC}_stderr.log" || true
    echo "  Done."
done

echo ""
echo "===== STEP PROGRESS COMPARISON ====="
echo ""

for PC in $PRECONDS; do
    STDOUT="$RESULTS_DIR/${PC}_stdout.log"
    STDERR="$RESULTS_DIR/${PC}_stderr.log"
    echo "--- $PC ---"
    # Step lines go to stdout
    if [ -s "$STDOUT" ]; then
        head -3 "$STDOUT"
        echo "..."
        grep "^Step " "$STDOUT" | head -30
        NSTEPS=$(grep -c "^Step " "$STDOUT" 2>/dev/null || echo "0")
        echo "  (Total Step lines: $NSTEPS)"
    else
        echo "  No stdout output"
    fi

    # Extract from progress bars in stderr
    PROGRESS=$(grep "step " "$STDERR" 2>/dev/null | grep -o "step [0-9]*" | tail -1 || echo "")
    echo "  Last progress: $PROGRESS"

    CONVERGED=$(grep -c "converged at newton" "$STDERR" 2>/dev/null || echo "0")
    DT_CUTS=$(grep -c "Newton failed, cutting\|Newton failed (cut" "$STDERR" 2>/dev/null || echo "0")
    LS_FAILS=$(grep -c "line search stalled" "$STDERR" 2>/dev/null || echo "0")
    echo "  Converged: $CONVERGED  dt-cuts: $DT_CUTS  LS-stalls: $LS_FAILS"
    echo ""
done

echo "===== dt COMPARISON (from Step lines) ====="
echo ""
printf "%-14s %8s %12s %12s %12s %12s\n" "Precond" "Steps" "FinalTime" "AvgDt" "MaxDt" "MinDt"
echo "--------------------------------------------------------------------------------"

for PC in $PRECONDS; do
    STDOUT="$RESULTS_DIR/${PC}_stdout.log"
    DTS=$(grep "^Step " "$STDOUT" 2>/dev/null | grep -oP 'dt = \K[0-9.e+-]+' || true)
    NSTEPS=$(grep -c "^Step " "$STDOUT" 2>/dev/null || echo "0")
    FINAL_T=$(grep "^Step " "$STDOUT" 2>/dev/null | tail -1 | grep -oP 't = \K[0-9.e+-]+' || echo "-")

    if [ -n "$DTS" ] && [ "$NSTEPS" -gt 0 ]; then
        AVG_DT=$(echo "$DTS" | awk '{s+=$1; n++} END {printf "%.3e", s/n}')
        MAX_DT=$(echo "$DTS" | sort -g | tail -1)
        MIN_DT=$(echo "$DTS" | sort -g | head -1)
        printf "%-14s %8s %12s %12s %12s %12s\n" "$PC" "$NSTEPS" "$FINAL_T" "$AVG_DT" "$MAX_DT" "$MIN_DT"
    else
        printf "%-14s %8s %12s %12s %12s %12s\n" "$PC" "$NSTEPS" "$FINAL_T" "-" "-" "-"
    fi
done
