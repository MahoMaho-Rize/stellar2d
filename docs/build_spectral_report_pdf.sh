#!/usr/bin/env bash
# Compile docs/spectral_stratified_poisson_report_2026-05-03.md → PDF
# via pandoc + xelatex.  Mirrors the toolchain of build_liouville_pdf.sh.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

SRC="spectral_stratified_poisson_report_2026-05-03.md"
OUT="spectral_stratified_poisson_report_2026-05-03.pdf"

TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

pixi run pandoc "$SRC" --pdf-engine=xelatex \
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
