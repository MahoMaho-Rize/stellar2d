#!/usr/bin/env python3
"""
Lint sections/*.md for structural-correctness errors only.

Style preferences (emoji, blockquote length, multi-line \\boxed) are NOT
enforced here — those are handled automatically by
scripts/normalize_manuscript.py at build time.  This lint catches bugs
that no normalizer can fix because they make the author's intent
ambiguous.

Hard errors:

  1. Unbalanced braces inside \\boxed{...} — if the closing } is
     missing, xelatex aborts at EOF with a vague message.  Always a
     bug.

  2. Blank line inside $$...$$ display math — math mode cannot contain
     \\par.  The normalizer strips it but emits a warning; lint
     upgrades this to an error at the source level so the author is
     forced to fix the markdown rather than relying on the fixer.

  3. CJK characters in formal derivation text — the derivation book
     is English-only (DejaVu Serif has no CJK glyphs, and ctex is not
     a standard dependency).  CJK inside an HTML comment
     <!-- ... --> is allowed for scratch drafts.

Usage:
    python3 scripts/lint_sections.py                 # check sections/
    python3 scripts/lint_sections.py path/to/x.md    # check one file

Exit 0 = clean, 1 = any error found.
"""
from __future__ import annotations
import re
import sys
import pathlib


class C:
    RED = "\033[31m" if sys.stderr.isatty() else ""
    GRN = "\033[32m" if sys.stderr.isatty() else ""
    BLD = "\033[1m" if sys.stderr.isatty() else ""
    END = "\033[0m" if sys.stderr.isatty() else ""


def check_boxed_balanced(text: str, path: pathlib.Path,
                         errors: list[str]) -> None:
    i = 0
    while True:
        m = re.search(r"\\boxed\{", text[i:])
        if not m:
            break
        start = i + m.start()
        depth = 1
        j = start + len(m.group(0))
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "\\":
                j += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            j += 1
        if depth != 0:
            line_no = text.count("\n", 0, start) + 1
            snippet = text[start:start + 80].replace("\n", "\\n")
            errors.append(
                f"{path}:{line_no}: \\boxed{{ never closed — {snippet!r}"
            )
        i = max(j, start + 1)


def check_display_math_no_blank_line(text: str, path: pathlib.Path,
                                     errors: list[str]) -> None:
    for m in re.finditer(r"\$\$(.+?)\$\$", text, re.DOTALL):
        if re.search(r"\n[ \t]*\n", m.group(1)):
            line_no = text.count("\n", 0, m.start()) + 1
            errors.append(
                f"{path}:{line_no}: $$...$$ contains blank line — "
                f"breaks math mode; remove the empty line"
            )


_CJK_RE = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")
# Strip HTML comments so CJK inside <!-- ... --> is allowed as
# scratch drafts (see STYLE_GUIDE 2.1).
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def check_no_cjk(text: str, path: pathlib.Path,
                 errors: list[str]) -> None:
    body = _HTML_COMMENT_RE.sub("", text)
    for m in _CJK_RE.finditer(body):
        # Report up to 3 hits per file to keep output readable
        line_no = body.count("\n", 0, m.start()) + 1
        errors.append(
            f"{path}:{line_no}: CJK character {m.group(0)!r} in formal "
            f"derivation text — derivation book is English only "
            f"(see STYLE_GUIDE 2.1).  Move Chinese commentary to "
            f"docs/projects/mhd_verification/ or wrap in <!-- ... -->."
        )
        if len([e for e in errors if "CJK character" in e and str(path) in e]) >= 3:
            errors.append(
                f"{path}: ... (additional CJK hits suppressed; fix the "
                f"above first)"
            )
            return


def lint_file(path: pathlib.Path) -> list[str]:
    errors: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        return [f"{path}: cannot read: {e}"]
    check_boxed_balanced(text, path, errors)
    check_display_math_no_blank_line(text, path, errors)
    check_no_cjk(text, path, errors)
    return errors


def main() -> int:
    argv = sys.argv[1:]
    if argv:
        targets = [pathlib.Path(p) for p in argv]
    else:
        here = pathlib.Path(__file__).resolve().parent
        sections_dir = here.parent / "sections"
        targets = sorted(
            p for p in sections_dir.glob("*.md")
            if not p.name.startswith("_")
        )

    n_err = 0
    for t in targets:
        if not t.exists():
            print(f"{C.RED}MISSING{C.END}: {t}", file=sys.stderr)
            n_err += 1
            continue
        errors = lint_file(t)
        for e in errors:
            print(f"{C.RED}ERROR {C.END}{e}", file=sys.stderr)
        n_err += len(errors)

    n_files = len(targets)
    if n_err > 0:
        print(
            f"\n{C.RED}{C.BLD}lint FAILED{C.END}: "
            f"{n_err} error(s) across {n_files} file(s)",
            file=sys.stderr,
        )
        return 1
    print(f"{C.GRN}lint clean{C.END}: {n_files} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
