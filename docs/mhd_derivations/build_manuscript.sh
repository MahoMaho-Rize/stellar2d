#!/usr/bin/env bash
# Assemble sections/*.md into manuscript.md; optionally render to PDF
# via pandoc + xelatex (mirrors paper/build_paper_pdf.sh).
#
# Usage:
#   cd docs/mhd_derivations
#   bash build_manuscript.sh [--pdf]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PDF=0
for arg in "$@"; do
    [[ "$arg" == "--pdf" ]] && PDF=1
done

OUT_MD="manuscript.md"

# 1. Concatenate sections in numeric/alphabetic order.
#    Section order: 00_preamble, a0_*, a1_..a5_, b0_*, b1_..b3_,
#                   c0_*, c1_..c4_, d0_*, d1_..d3_, 99_*
order=$(ls -1 sections/*.md 2>/dev/null | sort)
if [[ -z "$order" ]]; then
    echo "ERROR: no sections/*.md files found" >&2
    exit 1
fi

: > "$OUT_MD"
for f in $order; do
    cat "$f" >> "$OUT_MD"
    echo "" >> "$OUT_MD"
done
echo "→ $OUT_MD ($(wc -l < "$OUT_MD") lines)"

# 2. Optional PDF via pandoc + xelatex.
#    Uses the system pandoc (pixi adds pandoc 3.9).
if [[ $PDF -eq 1 ]]; then
    command -v pandoc >/dev/null || {
        echo "ERROR: pandoc not found — install via pixi (pandoc dep) or apt" >&2
        exit 1
    }
    echo "  pandoc → PDF (xelatex)"
    pandoc "$OUT_MD" \
        --pdf-engine=xelatex \
        -V geometry:margin=1in \
        -V fontsize=10pt \
        -V mainfont="DejaVu Serif" \
        -V monofont="DejaVu Sans Mono" \
        -V linkcolor=blue \
        --toc --toc-depth=2 \
        --number-sections \
        -o manuscript.pdf 2>/dev/null || {
        echo "  xelatex failed — manuscript.md is still valid; PDF skipped"
        echo "  (install texlive-xetex texlive-fonts-recommended to enable PDF)"
    }
    [[ -f manuscript.pdf ]] && echo "→ manuscript.pdf"
fi
