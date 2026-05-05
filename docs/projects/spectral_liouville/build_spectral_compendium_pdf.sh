#!/usr/bin/env bash
# Compile the two merged spectral compendium docs to a single PDF:
# - spectral_solver_design.md    (Part A)
# - spectral_experiments.md      (Part B)
# concatenated and rendered via pandoc + xelatex.
#
# Output: docs/spectral_compendium_2026-05-03.pdf
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

OUT="spectral_compendium_2026-05-03.pdf"
TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

# Build a single combined source. The first doc carries the YAML frontmatter
# for pandoc; we strip any subsequent --- frontmatter blocks.
cat spectral_solver_design.md > "$TMP/combined.md"
{
    echo
    echo
    echo '\clearpage'
    echo
    echo
    # Drop YAML frontmatter of second doc (strip between first two ---)
    awk 'BEGIN{fm=0;done=0} /^---$/ && !done { fm++; if (fm==2) {done=1}; next } done { print }' \
        spectral_experiments.md
} >> "$TMP/combined.md"

pixi run pandoc "$TMP/combined.md" --pdf-engine=xelatex \
    -V mainfont="Times New Roman" \
    -V fontsize=11pt \
    -V geometry=margin=1in \
    --standalone -t latex \
    -o "$TMP/doc.tex"

sed -i 's|\\IfFileExists{footnotehyper.sty}{\\usepackage{footnotehyper}}{\\usepackage{footnote}}|\\IfFileExists{footnotehyper.sty}{\\usepackage{footnotehyper}}{\\IfFileExists{footnote.sty}{\\usepackage{footnote}}{\\providecommand{\\makesavenoteenv}[1]{}}}|' "$TMP/doc.tex"

cd "$TMP"
xelatex -interaction=nonstopmode doc.tex > /dev/null || true
xelatex -interaction=nonstopmode doc.tex > /dev/null || true

if [[ ! -f doc.pdf ]]; then
    echo "xelatex failed to produce PDF" >&2
    exit 1
fi

mv doc.pdf "$HERE/$OUT"
echo "Wrote $HERE/$OUT"
