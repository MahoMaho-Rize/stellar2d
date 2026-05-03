#!/usr/bin/env bash
# Compile paper.md → paper.pdf through Elsevier cas-sc class.
#
# Pipeline:
#   1. Concat 01_*.md ... 12_*.md into body.md (no YAML frontmatter).
#   2. pandoc body.md -> body.tex (fragment, no preamble, no \begin{document}).
#   3. pdflatex paper-cas.tex twice (which \input{body.tex}).
#
# The class files cas-sc.cls, cas-common.sty, cas-model2-names.bst sit in
# this directory, so pdflatex finds them without system install.

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

# 1. Concat sections (no YAML — frontmatter lives in paper-cas.tex).
BODY_MD="body.md"
: > "$BODY_MD"
for f in 01_intro.md 02_setting.md 03_sl_spatial.md 04_gmode_evp.md \
         05_td_mismatch.md 06_assembled_td.md 07_nonlinear.md \
         08_discussion.md 09_conclusions.md 10_refs.md \
         11_appendix_a.md 12_appendix_b.md; do
    if [[ ! -f "$f" ]]; then
        echo "warning: missing $f" >&2
        continue
    fi
    cat "$f" >> "$BODY_MD"
    echo "" >> "$BODY_MD"
done

# Strip "N.M" and "N." numeric prefixes from ATX headings so cas-sc
# can assign its own section numbers.  Matches "^# 1. Title",
# "^## 1.2 Title", "^### 1.2.3 Title".
sed -i -E 's/^(#{1,6})[[:space:]]+[0-9]+(\.[0-9]+){0,2}\.?[[:space:]]+/\1 /' "$BODY_MD"
echo "→ $BODY_MD ($(wc -l < "$BODY_MD") lines)"

# 2. pandoc fragment conversion. No --standalone, no --pdf-engine;
#    emit raw LaTeX fragment that paper-cas.tex will \input.
pixi run pandoc "$BODY_MD" \
    -f markdown+raw_tex+raw_attribute \
    -t latex \
    --top-level-division=section \
    --wrap=preserve \
    -o body.tex

# Rewrite figure paths to absolute (pdflatex runs from $TMP).
sed -i "s|{figures/|{$HERE/figures/|g" body.tex

# 3. pdflatex twice for cross-refs.
TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT
cp paper-cas.tex body.tex cas-sc.cls cas-common.sty cas-model2-names.bst "$TMP/"
cp -r thumbnails "$TMP/"

pushd "$TMP" > /dev/null
xelatex -interaction=nonstopmode paper-cas.tex > xelatex.log 2>&1 || true
xelatex -interaction=nonstopmode paper-cas.tex > xelatex.log 2>&1 || true
popd > /dev/null

if [[ ! -f "$TMP/paper-cas.pdf" ]]; then
    echo "xelatex failed, last 80 lines of log:" >&2
    tail -80 "$TMP/xelatex.log" >&2
    exit 1
fi

mv "$TMP/paper-cas.pdf" "$HERE/paper.pdf"
echo "Wrote $HERE/paper.pdf ($(wc -c < "$HERE/paper.pdf") bytes, $(pdfinfo "$HERE/paper.pdf" 2>/dev/null | grep -oP 'Pages:\s+\K\d+') pages)"
