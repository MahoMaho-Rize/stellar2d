#!/usr/bin/env bash
# One-shot driver: run GYRE on its shipped n=3 polytrope to produce
# /tmp/gyre_run/summary.h5 + /tmp/gyre_run/poly3.txt for Exp H.
#
# Prerequisites:
#   $HOME/mesasdk-*           (MESA SDK, 200 MB, http://user.astro.wisc.edu/~townsend/static.php?ref=mesasdk)
#   $HOME/gyre                (git clone https://github.com/rhdtownsend/gyre + git submodule update --init --recursive + make)
#
# The Python-side Exp H (scripts/gmode_exp_h_gyre_benchmark.py) reads
# summary.h5 (41 l=1 g-modes of Lane-Emden n=3 polytrope) and poly3.txt
# (dimensionless stellar structure) and compares to our own solvers.

set -euo pipefail

MESA_ROOT=${MESASDK_ROOT:-}
if [[ -z "$MESA_ROOT" ]]; then
    # auto-find
    MESA_ROOT=$(ls -d $HOME/mesasdk-* 2>/dev/null | head -n 1)
fi
if [[ -z "$MESA_ROOT" || ! -d "$MESA_ROOT" ]]; then
    echo "ERROR: MESA SDK not found.  Install from http://user.astro.wisc.edu/~townsend/static.php?ref=mesasdk"
    exit 2
fi

if [[ ! -x "$HOME/gyre/bin/gyre" ]]; then
    echo "ERROR: GYRE binary not found at \$HOME/gyre/bin/gyre.  Build it first."
    exit 2
fi

export MESASDK_ROOT="$MESA_ROOT"
# Skip the official init script (which invokes csh / interactive probes);
# we only need the SDK gfortran runtime libs in LD path, not the whole env.
export PATH="$MESA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$MESA_ROOT/lib:${LD_LIBRARY_PATH:-}"
export GYRE_DIR="$HOME/gyre"

RUN_DIR=/tmp/gyre_run
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

# 1. Copy the shipped n=3 polytrope H5 + dump dimensionless structure to text
cp "$HOME/gyre/models/poly/3.0/poly.h5" .
"$HOME/gyre/bin/poly_to_txt" poly.h5 poly3.txt >/dev/null

# 2. Write GYRE input for a dense l=1 g-mode scan (catches n_g = 1..~40)
cat > gyre.in << 'EOF'
&constants
/

&model
  model_type = 'POLY'
  file = 'poly.h5'
/

&mode
  l = 1
  tag = 'l=1'
/

&osc
  inner_bound = 'REGULAR'
  outer_bound = 'VACUUM'
/

&rot
/

&num
  diff_scheme = 'COLLOC_GL6'
/

&scan
  grid_type = 'INVERSE'
  freq_min = 0.1
  freq_max = 3.0
  n_freq = 400
  tag_list = 'l=1'
/

&grid
/

&ad_output
  summary_file = 'summary.h5'
  summary_item_list = 'id,l,n_pg,omega,freq'
/

&nad_output
/
EOF

# 3. Run GYRE; we only keep the log tail for regression baseline.
echo "==> running GYRE ..."
"$HOME/gyre/bin/gyre" gyre.in > gyre.log 2>&1
tail -5 gyre.log
echo "==> artefacts:"
ls -la summary.h5 poly3.txt
