#!/usr/bin/env bash
# Render docs/cart_ale2_local_convection_2026-05-03.md → PDF via pandoc + xelatex.
#
# Font strategy: Times New Roman for body, STIX Two Text for math/Greek
# glyphs, DejaVu Sans Mono for ASCII box-drawing art.  We preprocess the
# markdown source with sed, wrapping each glyph class in a raw LaTeX
# inline span `[...]{.sym}` / `[...]{.box}`, and register the matching
# font switches through pandoc's Div/Span class mechanism in the header.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

SRC=cart_ale2_local_convection_2026-05-03.md
OUT=cart_ale2_local_convection_2026-05-03.pdf

TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

# Preprocess: wrap each problematic glyph with a LaTeX raw-inline so the
# fallback font only affects that one character.
python3 - "$SRC" "$TMP/pre.md" <<'PY'
import re, sys
src, dst = sys.argv[1:]
SYM = set("⊙⟨⟩≪≫∇∂∈≈≡ΓΣαγδκμπρστ²³⁴⁵⁶⁹₁✅❌▼←→↓↔−×·")
BOX = set("─│┌┐└┘┬")
with open(src, encoding="utf-8") as f:
    text = f.read()
def wrap(ch):
    if ch in SYM:
        return f'`\\sym{{{ch}}}`{{=latex}}'
    if ch in BOX:
        return f'`\\boxd{{{ch}}}`{{=latex}}'
    return ch
# Don't touch content inside fenced code blocks — they should render in the
# monospace stream verbatim via the monofont.  Walk line by line, toggling
# on/off whenever we see a fence.
lines = text.split("\n")
in_code = False
out = []
for ln in lines:
    if ln.startswith("```"):
        in_code = not in_code
        out.append(ln); continue
    if in_code:
        out.append(ln); continue
    # inline code ``...`` also bypassed — keep simple: don't wrap chars
    # inside backtick-delimited runs, so splitting on backtick pairs:
    parts = ln.split("`")
    for i, p in enumerate(parts):
        if i % 2 == 0:              # outside `...`
            parts[i] = "".join(wrap(c) for c in p)
    out.append("`".join(parts))
with open(dst, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
PY

cat > "$TMP/header.tex" <<'EOF'
\usepackage{fontspec}
% DejaVu Sans covers every non-emoji Unicode char we need (arrows, math
% operators, Greek, sub/superscripts, box-drawing). Register bold +
% italic explicitly so \textbf{... → ...} inside section headings uses
% the bold face that has the glyph.
\newfontfamily{\symbolfont}{DejaVu Sans}[
    Scale=0.92,
    BoldFont={DejaVu Sans Bold},
    ItalicFont={DejaVu Sans Oblique},
    BoldItalicFont={DejaVu Sans Bold Oblique}
]
\newcommand{\sym}[1]{{\symbolfont #1}}
\newcommand{\boxd}[1]{{\symbolfont #1}}
EOF

pixi run pandoc "$TMP/pre.md" --pdf-engine=xelatex \
    -V mainfont="Times New Roman" \
    -V monofont="JetBrains Mono" \
    -V fontsize=11pt \
    -V geometry=margin=1in \
    -V colorlinks=true \
    --toc --toc-depth=2 \
    -H "$TMP/header.tex" \
    --standalone -t latex \
    -o "$TMP/doc.tex"

sed -i 's|\\IfFileExists{footnotehyper.sty}{\\usepackage{footnotehyper}}{\\usepackage{footnote}}|\\IfFileExists{footnotehyper.sty}{\\usepackage{footnotehyper}}{\\IfFileExists{footnote.sty}{\\usepackage{footnote}}{\\providecommand{\\makesavenoteenv}[1]{}}}|' "$TMP/doc.tex"

cd "$TMP"
xelatex -interaction=nonstopmode doc.tex > "$TMP/xelatex.log" 2>&1 || true
xelatex -interaction=nonstopmode doc.tex > "$TMP/xelatex.log" 2>&1 || true

# xelatex may exit 0 and produce a near-empty PDF if it hits an error that
# --nonstopmode only "stops on".  A sane PDF is always > 10 KB here.
if [ ! -s doc.pdf ] || [ "$(stat -c%s doc.pdf)" -lt 1000 ]; then
    echo "xelatex produced no usable PDF; tail of log:"
    tail -60 "$TMP/xelatex.log"
    exit 1
fi

mv doc.pdf "$HERE/$OUT"
echo "Wrote $HERE/$OUT"
