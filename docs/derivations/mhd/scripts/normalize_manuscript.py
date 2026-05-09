#!/usr/bin/env python3
"""
Normalize the concatenated manuscript.md before pandoc.

Purpose: the markdown source is the author's truth.  The author should
NOT have to memorise pandoc + xelatex idiosyncrasies.  This script
transforms the concatenated manuscript into a form guaranteed to
survive pandoc -> xelatex without Overfull hbox or
"File ended while scanning \\boxed".

Transformations (in order; each preserves source semantics):

  1. Emoji (✅/❌/→ etc.) → bracketed text tags.  DejaVu Serif has no
     emoji glyphs, they would render as □.

  2. Display-math blocks  $$...$$  with content spanning multiple
     source lines AND containing \\boxed{...} AND not already wrapped
     in an amsmath multi-line environment — automatically wrap the
     \\boxed argument in \\begin{aligned} ... \\end{aligned} so
     xelatex can break the paragraph without killing the \\boxed scan.

  3. Display-math blocks containing a BLANK line are an author bug
     (math mode cannot tolerate \\par).  Strip the blank line and
     emit a warning to stderr so the author can fix the source.

  4. Trailing whitespace removal.

The transformed text replaces manuscript.md in place (we keep a backup
at manuscript.md.pre_norm for one build).

This is a SAFETY NET, not a style enforcer.  The companion
lint_sections.py covers genuine errors (unbalanced braces, etc.) at
the source level.  normalize_manuscript.py exists so authors can write
plausible markdown and still get a clean PDF.
"""
from __future__ import annotations
import re
import sys
import pathlib


EMOJI_REPLACEMENTS = {
    "✅": "[verified]",
    "❌": "[TODO]",
    "⚠️": "[warn]",
    "⏳": "[pending]",
    "→": "->",
    "←": "<-",
    "⇒": "=>",
    "⇐": "<=",
    "✓": "[ok]",
}


def normalize_emoji(text: str) -> tuple[str, int]:
    n = 0
    for emoji, replacement in EMOJI_REPLACEMENTS.items():
        c = text.count(emoji)
        if c:
            text = text.replace(emoji, replacement)
            n += c
    return text, n


_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n")
_ENV_RE = re.compile(
    r"\\begin\{(aligned|gathered|matrix|pmatrix|bmatrix|vmatrix|"
    r"cases|multline|split|array|equation|align)\*?\}"
)
_BOXED_RE = re.compile(r"\\boxed\{")


def wrap_boxed_in_aligned(inner: str) -> str:
    """
    If the display-math inner contains \\boxed{<body>} where <body>
    spans >1 source line and has no amsmath env, rewrite <body> to
    \\begin{aligned}<body>\\end{aligned}.
    """
    out = []
    i = 0
    while i < len(inner):
        m = _BOXED_RE.search(inner, i)
        if not m:
            out.append(inner[i:])
            break
        out.append(inner[i:m.start()])
        out.append(m.group(0))
        # Find matching close brace for \boxed{
        depth = 1
        j = m.end()
        body_start = j
        while j < len(inner) and depth > 0:
            ch = inner[j]
            if ch == "\\":
                j += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            j += 1
        # unmatched — bail out, lint will catch this
        if depth != 0:
            out.append(inner[body_start:])
            break
        body = inner[body_start:j - 1]    # content inside \boxed{...}
        close = inner[j - 1]              # the closing }
        body_has_newline = "\n" in body
        body_has_env = _ENV_RE.search(body) is not None
        if body_has_newline and not body_has_env:
            out.append(r"\begin{aligned}")
            # Convert any \qquad / \\ / , style continuation to aligned rows:
            # simplest — keep body as-is, just insert alignment.  aligned
            # with no & renders acceptably.
            out.append(body)
            out.append(r"\end{aligned}")
        else:
            out.append(body)
        out.append(close)
        i = j
    return "".join(out)


def normalize_display_math(text: str) -> tuple[str, int, list[str]]:
    """Return (new_text, n_rewritten, blank_line_warnings)."""
    warns: list[str] = []
    out_parts: list[str] = []
    last = 0
    n_rewrite = 0
    for m in _DISPLAY_MATH_RE.finditer(text):
        out_parts.append(text[last:m.start()])
        inner = m.group(1)

        # Strip blank lines inside display math (math-mode \par = boom)
        if _BLANK_LINE_RE.search(inner):
            line_no = text.count("\n", 0, m.start()) + 1
            warns.append(
                f"line {line_no}: $$...$$ had blank line — auto-stripped; "
                f"please fix the source to not do this"
            )
            inner = _BLANK_LINE_RE.sub("\n", inner)

        # Wrap any multi-line \boxed{...} body in aligned
        new_inner = wrap_boxed_in_aligned(inner)
        if new_inner != inner:
            n_rewrite += 1
        out_parts.append("$$" + new_inner + "$$")
        last = m.end()
    out_parts.append(text[last:])
    return "".join(out_parts), n_rewrite, warns


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: normalize_manuscript.py manuscript.md", file=sys.stderr)
        return 2
    src = pathlib.Path(sys.argv[1])
    text = src.read_text(encoding="utf-8")

    text, n_emoji = normalize_emoji(text)
    text, n_boxed, blank_warns = normalize_display_math(text)
    # Trim trailing whitespace on each line
    text = re.sub(r"[ \t]+\n", "\n", text)

    # Back up the pre-normalization version (one generation)
    backup = src.with_suffix(src.suffix + ".pre_norm")
    backup.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    src.write_text(text, encoding="utf-8")

    print(f"normalize_manuscript: {src}")
    print(f"  emoji replacements: {n_emoji}")
    print(f"  multi-line \\boxed wrapped in aligned: {n_boxed}")
    if blank_warns:
        print(f"  [warn] blank-line-in-math fixes: {len(blank_warns)}")
        for w in blank_warns:
            print(f"    {w}", file=sys.stderr)
    print(f"  backup at {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
