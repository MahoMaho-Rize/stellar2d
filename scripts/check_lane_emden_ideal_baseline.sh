#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${1:-ideal_baseline_64x32_t0.005}"
RUN_DIR="${REPO_ROOT}/runs/${RUN_NAME}"

NR="${NR:-64}"
NTHETA="${NTHETA:-32}"
TEND="${TEND:-0.005}"
OUTPUT_INTERVAL="${OUTPUT_INTERVAL:-2000}"
OMP_THREADS="${OMP_THREADS:-8}"
POLAR_THETA_GEOM="${POLAR_THETA_GEOM:-default}"
POLAR_CAP_BLEND="${POLAR_CAP_BLEND:-1.0}"

mkdir -p "${RUN_DIR}"

echo "Repo root: ${REPO_ROOT}"
echo "Run dir   : ${RUN_DIR}"
echo "Grid      : ${NR} x ${NTHETA}"
echo "t_end     : ${TEND}"
echo "OMP       : ${OMP_THREADS}"
echo "polar src : ${POLAR_THETA_GEOM}"

"${REPO_ROOT}/scripts/build_cpu_mac.sh"

pushd "${RUN_DIR}" >/dev/null
OMP_NUM_THREADS="${OMP_THREADS}" \
    "${REPO_ROOT}/build-mac/stellar2d_cpu" \
    --test lane_emden \
    --eos ideal \
    --nr "${NR}" \
    --ntheta "${NTHETA}" \
    --tend "${TEND}" \
    --output-interval "${OUTPUT_INTERVAL}" \
    --polar-cap-blend "${POLAR_CAP_BLEND}" \
    --polar-theta-geom "${POLAR_THETA_GEOM}" \
    2>&1 | tee run.log
popd >/dev/null

source "${REPO_ROOT}/.venv/bin/activate"
python "${REPO_ROOT}/tests/plot_lane_emden.py" --input-dir "${RUN_DIR}"

echo
echo "Done. Inspect:"
echo "  ${RUN_DIR}/run.log"
echo "  ${REPO_ROOT}/results/"
