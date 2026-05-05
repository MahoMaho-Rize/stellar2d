#!/usr/bin/env bash
# Compile docs/equations.md → docs/equations.pdf with Times New Roman.
# Requires: pandoc (via pixi) + xelatex + Times New Roman font.
# Gracefully skips `footnote.sty` if Fedora's texlive-mdwtools is absent.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

pixi run pandoc equations.md --pdf-engine=xelatex \
    -V mainfont="Times New Roman" \
    -V fontsize=11pt \
    -V geometry=margin=1in \
    --standalone -t latex \
    -o "$TMP/eq.tex"

# Skip footnote.sty (Fedora ships texlive-mdwtools separately; not needed for
# this doc since no table footnotes are used).
sed -i 's|\\IfFileExists{footnotehyper.sty}{\\usepackage{footnotehyper}}{\\usepackage{footnote}}|\\IfFileExists{footnotehyper.sty}{\\usepackage{footnotehyper}}{\\IfFileExists{footnote.sty}{\\usepackage{footnote}}{\\providecommand{\\makesavenoteenv}[1]{}}}|' "$TMP/eq.tex"

cd "$TMP"
xelatex -interaction=nonstopmode eq.tex > /dev/null
xelatex -interaction=nonstopmode eq.tex > /dev/null

mv eq.pdf "$HERE/equations.pdf"
echo "Wrote $HERE/equations.pdf"
