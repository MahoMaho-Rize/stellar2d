# X. Section title (matches scripts/xN_foo.py)

> **sympy script:** `scripts/xN_foo.py`
> **verified:**
> - identity 1 — one-line description
> - identity 2 — one-line description
> - identity 3 — one-line description
>
> **code checkpoints:**
> - `src/gpu/explicit/athena_mhd_kernels.cu::k_foo`
> - `tests/test_athena_mhd_foo.cu`

## Motivation / Context

Two to four paragraphs stating what this section is for physically,
and which upstream/downstream sections it connects to.

## Main identities

One subsection per identity.

### Identity 1: name

$$\boxed{\text{single-line formula, or multi-line wrapped in aligned}}$$

Brief explanation if needed.  **Sympy-verified** (or
**Sympy numerically verified** for floating-point sampling checks).

### Identity 2: name

$$\boxed{\begin{aligned}
&\text{line 1} \\
&\text{line 2}
\end{aligned}}$$

## Degenerate limits / edge cases

List the limits under which this section's result must reduce to a
previous section.  Include the sympy reduction check.

## Numerical implementation notes (not in formal derivation)

*Only if an empirically encountered numerical gotcha needs recording.
Otherwise omit this subsection.*

1. **Gotcha name.** Symptom / root cause / fix, one paragraph.
2. **Second gotcha name.**

## Verification checkpoints

- `tests/test_athena_mhd_foo.cu` — which identity is tested, what the
  threshold is, and where the threshold number comes from.
- `tests/test_athena_mhd_bar.cu` — ...

---
<!--
Template usage rules (see ../STYLE_GUIDE.md for the full rationale):
  1. This file is ENGLISH ONLY.  Formal derivation text must not
     contain CJK.  Chinese commentary belongs in
     docs/projects/mhd_verification/ benchmark logs, not here.
  2. \boxed{...} on a single line, OR wrapped in an aligned/multline
     environment.
  3. No blank line inside $$...$$.
  4. Inline $...$ ≤ 120 characters.
  5. No emoji.  Use [verified] / [TODO] text tags.
  6. Keep blockquote `verified:` as a bullet list when longer than
     two items.
-->
