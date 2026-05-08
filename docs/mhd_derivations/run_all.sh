#!/usr/bin/env bash
# Run every sympy derivation script in docs/mhd_derivations/scripts/.
# Stops on first failure (any assert_zero violation) and reports the
# offending script.
#
# Usage:
#   cd docs/mhd_derivations
#   bash run_all.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/scripts"

# enumerate scripts in filename order; skip _common.py + hidden
SCRIPTS=$(ls -1 *.py 2>/dev/null | grep -v '^_' | sort)

n_total=0
n_ok=0
n_fail=0
failed=()

for s in $SCRIPTS; do
    n_total=$((n_total+1))
    printf "  [%2d] %-40s " "$n_total" "$s"
    if python3 "$s" > "../output/${s%.py}.log" 2>&1; then
        printf "OK\n"
        n_ok=$((n_ok+1))
    else
        printf "FAIL (see output/${s%.py}.log)\n"
        n_fail=$((n_fail+1))
        failed+=("$s")
    fi
done

echo
echo "────────────────────────────────────────────"
echo "  $n_ok / $n_total scripts passed sympy verification"
if [[ $n_fail -gt 0 ]]; then
    echo "  Failures:"
    for f in "${failed[@]}"; do
        echo "    - $f"
    done
    exit 1
fi
