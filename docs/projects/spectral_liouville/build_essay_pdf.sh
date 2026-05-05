#!/usr/bin/env bash
# Compile docs/essay.md → PDF via pandoc + xelatex.
# Uses the same toolchain as build_equations_pdf.sh.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

SRC="essay.md"
OUT="essay.pdf"

TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

pixi run pandoc "$SRC" --pdf-engine=xelatex \
    -V mainfont="Times New Roman" \
    -V fontsize=11pt \
    -V geometry=margin=1in \
    --standalone -t latex \
    -o "$TMP/doc.tex"

# Same footnote.sty workaround as build_equations_pdf.sh
sed -i 's|\\IfFileExists{footnotehyper.sty}{\\usepackage{footnotehyper}}{\\usepackage{footnote}}|\\IfFileExists{footnotehyper.sty}{\\usepackage{footnotehyper}}{\\IfFileExists{footnote.sty}{\\usepackage{footnote}}{\\providecommand{\\makesavenoteenv}[1]{}}}|' "$TMP/doc.tex"

cd "$TMP"
xelatex -interaction=nonstopmode doc.tex > /dev/null
xelatex -interaction=nonstopmode doc.tex > /dev/null

mv doc.pdf "$HERE/$OUT"
echo "Wrote $HERE/$OUT"
