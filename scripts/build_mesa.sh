#!/usr/bin/env bash
# Build MESA r26.x from /home/kiriko/mesa-ref using the local SDK.
#
# Usage:
#   bash scripts/build_mesa.sh              # foreground
#   nohup bash scripts/build_mesa.sh &      # background, survives logout
#
# Logs land in /home/kiriko/mesa-ref/build.log. Check progress with:
#   tail -f /home/kiriko/mesa-ref/build.log
#
# Wall-time estimate: ~45–60 min on this box (32-core Zen, 48 GB free).
# Peak RAM during link step: ~6 GB. Final tree occupies ~12 GB on disk.
#
# Exit code:
#   0  — build + self-test passed (final line "MESA installation was successful")
#   1  — install script bailed (see build.log tail)

set -euo pipefail

# ---- paths (edit only if you move things around) ---------------------
MESA_DIR=/home/kiriko/mesa-ref
MESASDK_ROOT=/home/kiriko/mesasdk-26.3.2

# ---- sanity checks ---------------------------------------------------
if [[ ! -d "$MESA_DIR" ]]; then
    echo "ERROR: MESA_DIR not found at $MESA_DIR" >&2
    exit 1
fi
if [[ ! -f "$MESASDK_ROOT/bin/mesasdk_init.sh" ]]; then
    echo "ERROR: mesasdk_init.sh not found under $MESASDK_ROOT" >&2
    exit 1
fi

# ---- activate SDK ----------------------------------------------------
export MESASDK_ROOT
# The SDK's init script references $MANPATH / $INFOPATH without nounset
# guards, which trips `set -u`. Disable nounset just for the sourcing.
set +u
# shellcheck disable=SC1091
source "$MESASDK_ROOT/bin/mesasdk_init.sh"
set -u

export MESA_DIR
# Use ~half the cores to leave headroom for the dev box; parallel fortran
# linking is the main RAM pressure point.
export NPROCS=${NPROCS:-16}
# MESA's own OMP: keep modest during build; test_suite runs can set higher.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

# Optional: skip the trailing test_suite run if you only want libraries.
# The 'install' script runs `star/work/mk` + `each_test_run_and_diff` at
# the end; setting SKIP_INSTALL_TEST=1 cuts that last ~5 min.
: "${SKIP_INSTALL_TEST:=0}"

# ---- summary banner --------------------------------------------------
cd "$MESA_DIR"
{
    echo "========================================"
    echo "MESA build starting at $(date -Is)"
    echo "  MESA_DIR     = $MESA_DIR"
    echo "  MESASDK_ROOT = $MESASDK_ROOT"
    echo "  SDK version  = $($MESASDK_ROOT/bin/mesasdk_version)"
    echo "  git HEAD     = $(git -C "$MESA_DIR" describe --tags --dirty --always 2>/dev/null || echo '(no git)')"
    echo "  NPROCS       = $NPROCS"
    echo "  OMP_NUM_THREADS = $OMP_NUM_THREADS"
    echo "  free RAM GB  = $(awk '/MemAvailable/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)"
    echo "========================================"
} | tee build.log

# ---- run the install -------------------------------------------------
# `./install` pipes its own progress to stdout and writes to build.log.
# We tee again so this wrapper's banner is retained above it.
if [[ "$SKIP_INSTALL_TEST" == "1" ]]; then
    # `./install` always runs the end-of-build test; to skip it we call
    # the underlying per-module build script directly and then `star/work/mk`.
    echo "[wrapper] SKIP_INSTALL_TEST=1 — running build only, no self-test" | tee -a build.log
    bash -c './each_package_do "make"' 2>&1 | tee -a build.log
else
    ./install 2>&1 | tee -a build.log
fi

rc=${PIPESTATUS[0]}

{
    echo "========================================"
    echo "MESA build finished at $(date -Is), exit=$rc"
    if [[ $rc -eq 0 ]]; then
        echo "  final libs:"
        ls -lh "$MESA_DIR/lib"/libstar* 2>/dev/null | head || echo "  (no libstar.* — check build.log)"
        echo
        echo "Next step — compile the star/work template:"
        echo "  source $MESASDK_ROOT/bin/mesasdk_init.sh"
        echo "  export MESA_DIR=$MESA_DIR"
        echo "  cp -r \$MESA_DIR/star/work /tmp/mesa_work_1Msol"
        echo "  cd /tmp/mesa_work_1Msol && ./mk && ./rn"
    fi
    echo "========================================"
} | tee -a build.log

exit $rc
