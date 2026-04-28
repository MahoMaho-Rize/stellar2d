#!/bin/zsh
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_root"

binary="$repo_root/build-mac/stellar2d_cpu"
out_dir="$repo_root/build-mac"

if [[ ! -x "$binary" ]]; then
  echo "Binary not found: $binary"
  echo "Run: ./scripts/build_cpu_mac.sh"
  exit 1
fi

cd "$out_dir"
"$binary" \
  --test lane_emden \
  --nr 32 \
  --ntheta 16 \
  --tend 0.05 \
  --output-interval 5000 \
  2>&1 | tee run.log
