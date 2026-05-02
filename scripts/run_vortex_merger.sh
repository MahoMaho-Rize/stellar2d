#!/usr/bin/env bash
set -euo pipefail

./build/stellar2d \
    --solver pseudo_spectral \
    --test vortex_merger \
    --nr 2048 --ntheta 2048 \
    --ps-Lx 1.0 --ps-Ly 1.0 \
    --ps-vm-gamma 100 \
    --ps-vm-sigma 0.04 \
    --ps-vm-dist  0.10 \
    --ps-nu 1e-11 \
    --ps-hyper 2 \
    --cfl 0.35 \
    --tend 5.0 \
    --vtk-dt 0.0025 \
    --output-interval 10000 \
    --diag-interval 200 \
    --frame-buffer \
    --frame-headroom-mb 4096 \
    --ps-ckpt-every 20000
