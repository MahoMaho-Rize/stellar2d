#!/usr/bin/env bash
# One-shot driver: run GYRE on its shipped n=3 polytrope with alpha_grv=0
# (pure Cowling approximation) to produce /tmp/gyre_cowling/summary_cowling.h5
# for Exp I to compare against.
#
# Companion to scripts/gmode_exp_h_run_gyre.sh (which runs the SAME model but
# with full gravity, alpha_grv=1).  Running both gives us two baselines:
#   - /tmp/gyre_cowling/summary_cowling.h5 -- same physics as our gyre_compat
#                                             operator (Exp I target)
#   - /tmp/gyre_run/summary.h5              -- full gravity (Exp H target)

set -euo pipefail

MESA_ROOT=${MESASDK_ROOT:-}
if [[ -z "$MESA_ROOT" ]]; then
    MESA_ROOT=$(ls -d $HOME/mesasdk-* 2>/dev/null | head -n 1)
fi
if [[ -z "$MESA_ROOT" || ! -d "$MESA_ROOT" ]]; then
    echo "ERROR: MESA SDK not found."
    exit 2
fi
if [[ ! -x "$HOME/gyre/bin/gyre" ]]; then
    echo "ERROR: GYRE binary not found at \$HOME/gyre/bin/gyre."
    exit 2
fi

export MESASDK_ROOT="$MESA_ROOT"
export PATH="$MESA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$MESA_ROOT/lib:${LD_LIBRARY_PATH:-}"
export GYRE_DIR="$HOME/gyre"

RUN_DIR=/tmp/gyre_cowling
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

cp "$HOME/gyre/models/poly/3.0/poly.h5" .

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
  alpha_grv = 0.0
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
  summary_file = 'summary_cowling.h5'
  summary_item_list = 'id,l,n_pg,omega,freq'
/

&nad_output
/
EOF

echo "==> running GYRE (alpha_grv=0, pure Cowling) ..."
"$HOME/gyre/bin/gyre" gyre.in > gyre.log 2>&1
tail -5 gyre.log
echo "==> artefacts:"
ls -la summary_cowling.h5
