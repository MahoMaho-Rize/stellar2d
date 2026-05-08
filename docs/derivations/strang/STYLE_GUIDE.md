# Derivation-book markdown style guide (enforced)

> **`build_manuscript.sh --pdf` refuses to run if any rule in §2 is
> violated.**  Everything else is automatically normalised at build time
> by `scripts/normalize_manuscript.py` — you do not need to memorise
> pandoc + xelatex idiosyncrasies.

This guide governs **formatting** of the formal derivation book.
Content-level rules (derivation-first workflow, sympy verification,
test thresholds) live in `docs/design/derivation_driven_workflow.md`.

This file is the strang-kernel sibling of `docs/mhd_derivations/STYLE_GUIDE.md`.
The two books share the same lint + normaliser toolchain verbatim;
keep them in sync when either is updated.

---

## 1. Division of labour

| Layer | Purpose | Language |
|---|---|---|
| `docs/derivations/strang/sections/*.md` | **Formal derivation book.**  Ships in `manuscript.pdf`. | **English only.** |
| `docs/projects/strang_verification/*.md` | Benchmark logs, milestone records, debugging memos. | Chinese / English mixed OK. |
| `docs/design/*.md` | Engineering design notes, workflow rules. | Chinese / English mixed OK. |
| `src/**/*.cu,cuh,cpp` comments | Code commentary. | Chinese / English mixed OK. |

The derivation book is a published artifact; benchmark logs are an
internal lab notebook.  Keep them on opposite sides of this line.

---

## 2. Hard rules (lint-enforced; build blocks on violation)

### 2.1 English-only in `sections/*.md`

No CJK characters allowed in any `sections/*.md` file.  DejaVu Serif
has no CJK glyphs and the standard `texlive-xetex` install does not
bring in `ctex`; attempting to build with CJK produces either
`ctexhook.sty not found` or silent □ glyphs.

If you want to keep a Chinese draft, do it in a comment block the
normaliser will strip:

```markdown
<!-- scratch (zh): 这里可以写中文草稿 -->
```

Anything outside HTML comments must be English.

### 2.2 Balanced `\boxed{...}` braces

Every `\boxed{` must have a matching `}`.  An unbalanced brace causes
xelatex to fail with `File ended while scanning use of \boxed`, which
points at EOF rather than the actual source line — hard to debug.

### 2.3 No blank line inside `$$...$$`

A blank line inside display math is a LaTeX paragraph break inside
math mode; it cannot be healed by any wrapper.  Do not write it.

---

## 3. Automatically normalised (do NOT worry about)

`scripts/normalize_manuscript.py` runs on the concatenated
`manuscript.md` before pandoc.  It handles:

- **Emoji → text tags**.  `✅` → `[verified]`, `❌` → `[TODO]`, arrows
  (`→ ← ⇒ ⇐`) → ASCII equivalents.  Authors may keep emoji in the
  source if they prefer; the PDF will show text tags.
- **Multi-line `\boxed{...}` → `\boxed{\begin{aligned}...\end{aligned}}`**.
  If the body of `\boxed{...}` spans more than one source line and is
  not already in an amsmath environment, the normaliser wraps it
  automatically so xelatex can break the paragraph without killing
  the `\boxed` scan.
- **Stray trailing whitespace** is stripped.

A pre-norm backup is kept at `manuscript.md.pre_norm` for one build
cycle, in case you want to diff.

---

## 4. Template

New sections start from `sections/_TEMPLATE.md`.  Existing sections
should drift toward that skeleton when edited but need not be
rewritten wholesale.

Key skeleton elements:

```markdown
# X. Section title

> **sympy script:** `scripts/xN_foo.py`
> **verified:**
> - identity 1
> - identity 2
>
> **code checkpoints:**
> - `path/to/kernel.cu::k_foo`
> - `tests/test_foo.cu`

## Motivation / Context
## Main identities
### Identity 1: name
$$\boxed{...}$$
## Degenerate limits / edge cases
## Numerical implementation notes (not in formal derivation)
## Verification checkpoints
```

---

## 5. Toolchain

- **pandoc** ≥ 3.0
- **xelatex** from `texlive-xetex`
- `build_manuscript.sh --pdf` exit code 0 ⟺ clean build; exit 1 on any
  lint error or xelatex failure; **never silently overwrites** the
  previous `manuscript.pdf` on failure.

Header injected automatically:

```latex
\emergencystretch=3em
\sloppy
\hbadness=10000
\hfuzz=20pt
```

These let TeX stretch paragraph glue to absorb the occasional
borderline Overfull hbox without actually exceeding the page margin.

---

## 6. Cheat sheet for longer math

| Want | Do |
|---|---|
| Long formula with text | `\text{...}` inside display math, or `aligned` |
| Several formulas in a row | Separate `$$...$$` blocks, one per line |
| Aligned multi-line formula | `\begin{aligned} ... \end{aligned}` |
| One formula broken across lines | `\begin{multline*} ... \end{multline*}` |
| Boxed multi-line formula | `\boxed{\begin{aligned} ... \end{aligned}}` |

---

## 7. Origin

2026-05-08 session.  The strang derivation book (36 sections) was
drafted simultaneously with the MHD book's adoption of this style
spec.  When the MHD side hardened lint + normaliser into
`docs/mhd_derivations/{STYLE_GUIDE.md,scripts/lint_sections.py,
scripts/normalize_manuscript.py}`, the strang side ported the
identical toolchain so a single PDF build pipeline applies to both
books.

The hard rules (CJK ban, `\boxed{}` balance, no blank line in
`$$...$$`) come from an MHD rebuild that silently produced a broken
PDF: a missing `}` killed xelatex at EOF, long `\boxed{}` blocks
created Overfull hbox, and inline Chinese comments rendered as `□`
because DejaVu Serif lacks CJK glyphs.  This guide + the normaliser
prevent the same class of failure on the strang side.
