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
# fallback font only affects that one character.  Uses a single-pass
# character-level state machine that correctly handles:
#   - fenced code blocks (```...```) — verbatim, monofont handles Unicode
#   - inline code spans (`...`) that may straddle line breaks
#   - everything else — wrap offending chars in \sym{} / \boxd{}
python3 - "$SRC" "$TMP/pre.md" <<'PY'
import sys
src, dst = sys.argv[1:]
SYM = set("⊙⟨⟩≪≫∇∂∈≈≡ΓΣαγδκμπρστ²³⁴⁵⁶⁹₁✅❌▼←→↓↔−×·")
BOX = set("─│┌┐└┘┬")

with open(src, encoding="utf-8") as f:
    text = f.read()

def wrap(ch):
    if ch in SYM: return f'`\\sym{{{ch}}}`{{=latex}}'
    if ch in BOX: return f'`\\boxd{{{ch}}}`{{=latex}}'
    return ch

out = []
i = 0
n = len(text)
# State machine:
#   in_fence: inside ```..``` block (verbatim, no wrapping)
#   in_inline: inside `...` run (verbatim, no wrapping, may cross \n)
in_fence = False
in_inline = False
at_line_start = True
while i < n:
    ch = text[i]
    # Detect opening/closing of fenced block only at line start.
    if at_line_start and text[i:i+3] == "```":
        # Copy the whole fence marker line verbatim.
        eol = text.find("\n", i)
        if eol == -1: eol = n
        out.append(text[i:eol+1])
        in_fence = not in_fence
        i = eol + 1
        at_line_start = True
        continue
    if in_fence:
        out.append(ch)
        at_line_start = (ch == "\n")
        i += 1
        continue
    # Outside fenced block: handle inline code spans.
    if ch == "`":
        in_inline = not in_inline
        out.append(ch)
        i += 1
        at_line_start = False
        continue
    if in_inline:
        out.append(ch)
        at_line_start = (ch == "\n")
        i += 1
        continue
    # Body text: wrap if offending char.
    out.append(wrap(ch))
    at_line_start = (ch == "\n")
    i += 1

with open(dst, "w", encoding="utf-8") as f:
    f.write("".join(out))
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
    --from=markdown-smart \
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
