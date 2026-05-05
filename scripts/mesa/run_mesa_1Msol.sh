#!/usr/bin/env bash
# Drive the MESA r26 1 M_sun pre-MS → ZAMS run in /tmp/mesa_work_1Msol.
#
# Background-safe:
#   cd ~/stellar2d
#   nohup bash scripts/run_mesa_1Msol.sh > /dev/null 2>&1 &
#   disown
#
# Progress:
#   tail -f /tmp/mesa_work_1Msol/mesa.log
#
# Wall-time estimate: ~60–90 seconds to ZAMS on a 9950X (single-thread hot
# path, MESA is mostly serial).
#
# Output:
#   /tmp/mesa_work_1Msol/LOGS/profileN.data
#   /tmp/mesa_work_1Msol/LOGS/history.data
#   /tmp/mesa_work_1Msol/1M_at_ZAMS.mod

set -euo pipefail

MESA_DIR=/home/kiriko/mesa-ref
MESASDK_ROOT=/home/kiriko/mesasdk-26.3.2
WORK=/tmp/mesa_work_1Msol

if [[ ! -d "$WORK" ]]; then
    echo "ERROR: $WORK missing — run `cp -r $MESA_DIR/star/work $WORK` first" >&2
    exit 1
fi
if [[ ! -f "$WORK/inlist_project" ]]; then
    echo "ERROR: $WORK/inlist_project missing" >&2
    exit 1
fi

export MESASDK_ROOT
# shellcheck disable=SC1091
set +u
source "$MESASDK_ROOT/bin/mesasdk_init.sh"
set -u
export MESA_DIR
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

cd "$WORK"
rm -f restart_photo mesa.log
rm -rf LOGS photos 1M_at_ZAMS.mod

{
    echo "========================================"
    echo "MESA 1M run at $(date -Is)"
    echo "  MESA_DIR=$MESA_DIR"
    echo "  WORK=$WORK"
    echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
    echo "========================================"
} | tee mesa.log

make 2>&1 | tee -a mesa.log
./rn 2>&1 | tee -a mesa.log

rc=${PIPESTATUS[0]}

{
    echo "========================================"
    echo "MESA 1M finished at $(date -Is), rc=$rc"
    if [[ $rc -eq 0 ]]; then
        echo "LOGS contents:"
        ls -lh LOGS/profile*.data 2>/dev/null | tail
        echo
        ls -lh LOGS/history.data 2>/dev/null
        ls -lh 1M_at_ZAMS.mod 2>/dev/null
    fi
    echo "========================================"
} | tee -a mesa.log

exit $rc
