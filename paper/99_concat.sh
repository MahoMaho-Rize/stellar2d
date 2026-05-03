#!/usr/bin/env bash
# Assemble paper.md from section files.
# Run from paper/:  bash 99_concat.sh
set -euo pipefail
cd "$(dirname "$0")"

OUT=paper.md

cat > "$OUT" <<'EOF'
---
title: |
  Operator-Consistency Failure and its Resolution in
  Primitive-Node Pseudo-Spectral Time Stepping of
  Variable-Coefficient Anelastic Flow
author: |
  Kiriko\
  \textit{Department of Astronomy, Tsinghua University}\
  \texttt{github.com/MahoMaho-Rize/stellar2d}
date: 3 May 2026
abstract: |
  Pseudo-spectral methods are widely assumed to retain spectral
  accuracy under smooth variable coefficients.  We show that this
  assumption fails at the discrete operator level.  Initialising a
  standard primitive-node pseudo-spectral anelastic solver with an
  exact g-mode eigenvector of its own discrete eigenproblem produces
  a per-step deviation of $\mathcal{O}(10^{-4})$ that is independent
  of grid resolution, independent of time-step refinement, and
  independent of the boundary-condition formulation.  Over one
  hundred oscillation periods the eigenmode collapses entirely.
  The failure arises from replacing the inverse of a global elliptic
  operator by a pointwise scaling: a structural inconsistency, not a
  singularity or a truncation effect, that persists under both grid
  refinement and small-perturbation limits of the physics amplitude.
  We formalise this as Proposition 1, whose three clauses establish
  the resolution-independent floor, a scaling law
  $c(\rho_0) \sim \|\rho_0'\|_\infty$ in the small-perturbation limit,
  and the structural (not truncation) character of the defect.
  We resolve the failure by assembling $\mathsf{M} = \mathsf{L}^{-1}\mathsf{R}$
  once at setup and applying it as a per-wavenumber dense matrix-vector
  product at every RK4 substage; Theorem 6.1 establishes that this
  preserves any eigenvector of the assembled EVP to within
  $\mathcal{O}(\epsilon_{\mathrm{mach}}\,\kappa(Q))$ round-off per
  step.  The Python implementation reaches $5\times 10^{-18}$ per-step
  deviation, the CUDA implementation $3\times 10^{-15}$, and the spatial
  machinery is externally validated against the GYRE stellar-pulsation
  code to $9.1\times 10^{-9}$ relative error on a Lane--Emden
  $n = 3$ polytrope at $N_r = 96$.  At $N_y = 64$ the assembled
  scheme reaches $3 \times 10^{-18}$ per-step deviation, four
  orders of magnitude below a $\tau$-method Galerkin prototype
  ($2 \times 10^{-14}$), at approximately one-third the
  working-memory footprint and within a factor of two of either
  method's per-substep runtime.  The scheme requires only a single
  setup-time matrix insertion into a pre-existing primitive-node
  pipeline.  Source code and reproducers:
  \url{https://github.com/MahoMaho-Rize/stellar2d}.
---

EOF

for f in 01_intro.md 02_setting.md 03_sl_spatial.md 04_gmode_evp.md \
         05_td_mismatch.md 06_assembled_td.md 07_nonlinear.md \
         08_discussion.md 09_conclusions.md 10_refs.md \
         11_appendix_a.md 12_appendix_b.md; do
    if [[ ! -f "$f" ]]; then
        echo "warning: missing $f" >&2
        continue
    fi
    cat "$f" >> "$OUT"
    echo "" >> "$OUT"
done

echo "→ $OUT ($(wc -l < "$OUT") lines)"
