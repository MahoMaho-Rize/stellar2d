# Paper outline — Sturm–Liouville + Assembled-Matrix Spectral Framework

Target: arXiv preprint, ~25 pages equivalent, self-contained.

## Structure

| Section | File | Content | Target length |
|---|---|---|---|
| Front matter | `paper.md` head (injected by concat) | title, authors, abstract | 1 page |
| 1. Introduction | `01_intro.md` | motivation, Fourier breakdown, contributions, roadmap | 2.5 pages |
| 2. Mathematical setting | `02_setting.md` | anelastic eqns, Lane-Emden bg, reduced pressure | 2 pages |
| 3. SL spatial discretisation | `03_sl_spatial.md` | Liouville, CGL + CC, Poisson 7-step, σ-dichotomy | 4 pages |
| 4. g-mode EVP + GYRE benchmark | `04_gmode_evp.md` | 4-var operator, BCs, 3.6e-5 closure | 3 pages |
| 5. Time-domain operator mismatch | `05_td_mismatch.md` | dev/step 6e-4, Path A/B/C failure modes | 2.5 pages |
| 6. Assembled-matrix TD | `06_assembled_td.md` | Path D: L⁻¹R per kx, 3e-15 CUDA, theorem | 3 pages |
| 7. Nonlinear extension | `07_nonlinear.md` | Phase 3 three-path, IMEX blowup, Path 1 pick | 2.5 pages |
| 8. Discussion | `08_discussion.md` | relation to GYRE/Dedalus, scope, limits | 2 pages |
| 9. Conclusions | `09_conclusions.md` | 4-layer narrative summary | 1 page |
| 10. References | `10_refs.md` | bibliography (plain markdown list) | 1 page |

## Cross-section conventions

- **No internal commit hashes or `docs/*` paths**. External pointer is
  `https://github.com/MahoMaho-Rize/stellar2d` only.
- **No benchmark numbers from Phase 3b onward** — we have linear TD closure
  and Python nonlinear prototype, but CUDA nonlinear benchmarks are
  explicitly listed as future work.
- **Equation numbering per section**: `(3.1), (3.2), ...`
- **Tables / figures**: captioned `Tab. N.k`, `Fig. N.k`, consecutive per section.
- **Provenance**: each data table ends with a short `Setup:` line giving
  ny/nx/dt/n_steps/background so readers can reproduce from the public repo.

## Concat order

```
paper.md = front_matter + 01 + 02 + ... + 10
```

Driven by `99_concat.sh`.
