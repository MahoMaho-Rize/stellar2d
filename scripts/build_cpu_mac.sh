#!/bin/zsh
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_root"

if ! xcode-select -p >/dev/null 2>&1; then
  echo "Xcode Command Line Tools are not installed."
  echo "Run: xcode-select --install"
  exit 1
fi

if ! command -v clang++ >/dev/null 2>&1; then
  echo "clang++ not found."
  exit 1
fi

mkdir -p build-mac

omp_flags=()
omp_enabled=0

for prefix in /opt/homebrew /usr/local; do
  if [[ -f "$prefix/opt/libomp/include/omp.h" && -f "$prefix/opt/libomp/lib/libomp.dylib" ]]; then
    omp_flags=(
      -Xpreprocessor -fopenmp
      -I"$prefix/opt/libomp/include"
      -L"$prefix/opt/libomp/lib"
      -Wl,-rpath,"$prefix/opt/libomp/lib"
      -lomp
    )
    omp_enabled=1
    break
  fi
done

clang++ -std=c++17 -O2 -Isrc "${omp_flags[@]}" \
  src/grid.cpp \
  src/state.cpp \
  src/burn/pp_chain.cpp \
  src/hydro/reconstruct.cpp \
  src/hydro/riemann.cpp \
  src/hydro/flux.cpp \
  src/hydro/integrate.cpp \
  src/gravity/gmg.cpp \
  src/bc/boundary.cpp \
  src/io/output.cpp \
  src/init/lane_emden.cpp \
  src/init/sedov.cpp \
  src/init/jeans.cpp \
  src/init/evrard.cpp \
  src/main.cpp \
  -o build-mac/stellar2d_cpu

clang++ -std=c++17 -O2 -Isrc "${omp_flags[@]}" \
  src/burn/pp_chain.cpp \
  src/onezone_pp_main.cpp \
  -o build-mac/onezone_pp

echo "Built: $repo_root/build-mac/stellar2d_cpu"
echo "Built: $repo_root/build-mac/onezone_pp"
if [[ $omp_enabled -eq 1 ]]; then
  echo "OpenMP: enabled"
else
  echo "OpenMP: not found (serial build). Install libomp to use multiple CPU cores:"
  echo "  brew install libomp"
fi
