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

# 0. Lint the sections for the two structural errors no fixer can handle:
#    unbalanced \boxed{ and $$...$$ with a blank line.
echo "→ lint sections/"
if ! python3 scripts/lint_sections.py; then
    echo "ERROR: lint failed — fix the sections/*.md errors above before building." >&2
    exit 1
fi

# 1. Concatenate sections in numeric/alphabetic order.
#    Section order: 00_preamble, a0_*, a1_..a5_, b0_*, b1_..b3_,
#                   c0_*, c1_..c4_, d0_*, d1_..d3_, 99_*
#    Template file (_TEMPLATE.md) is skipped.
order=$(ls -1 sections/*.md 2>/dev/null | grep -v '/_' | sort)
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

# 1.5 Normalize the concatenated manuscript in-place: replace emoji with
#     text tags, wrap any multi-line \boxed{...} body in \begin{aligned}
#     so xelatex can break the paragraph without killing the \boxed
#     scan, strip stray trailing whitespace.  Saves a backup at
#     manuscript.md.pre_norm for one build.
echo "→ normalize manuscript for pandoc/xelatex"
python3 scripts/normalize_manuscript.py "$OUT_MD"

# 2. Optional PDF via pandoc + xelatex.
#    Uses the system pandoc (pixi adds pandoc 3.9).
if [[ $PDF -eq 1 ]]; then
    command -v pandoc >/dev/null || {
        echo "ERROR: pandoc not found — install via pixi (pandoc dep) or apt" >&2
        exit 1
    }
    echo "  pandoc → PDF (xelatex)"
    # Header-includes:
    #   \emergencystretch=3em allows TeX to stretch inter-word glue on
    #     paragraphs that would otherwise overflow (common with long
    #     inline math + \boxed at narrow margins).
    #   \sloppy disables strict underfull complaints globally.
    HEADER_INC=$(mktemp --suffix=.tex)
    cat > "$HEADER_INC" <<'LATEX'
\emergencystretch=3em
\sloppy
\hbadness=10000
\hfuzz=20pt
LATEX
    if ! pandoc "$OUT_MD" \
            --pdf-engine=xelatex \
            -V geometry:margin=1in \
            -V fontsize=10pt \
            -V mainfont="DejaVu Serif" \
            -V monofont="DejaVu Sans Mono" \
            -V linkcolor=blue \
            -H "$HEADER_INC" \
            --toc --toc-depth=2 \
            --number-sections \
            -o manuscript.pdf 2>&1 | tee /tmp/pandoc_build.log; then
        rm -f "$HEADER_INC"
        echo "  ERROR: xelatex failed — stale manuscript.pdf NOT overwritten"
        echo "  Check /tmp/pandoc_build.log for the scanning error"
        echo "  Common cause: unbalanced \\boxed{} braces in sections/*.md"
        exit 1
    fi
    rm -f "$HEADER_INC"
    echo "→ manuscript.pdf"
fi
