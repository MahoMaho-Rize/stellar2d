# Benchmark data archive

Curated benchmarks for the stellar2d spectral pipeline validation,
mirroring the recommendations in
`docs/spectral_solver_design.md` §12 (Phase roadmap).

**Status as of 2026-05-03**: initial pass, ~35 MB total.

**How to populate the archive**: run `bash data/benchmarks/download_all.sh`
from the repo root. The script is idempotent (skips files that already
exist) and pulls from Aarhus, VizieR, GitHub (Rayleigh + Dedalus sparse
clones), Springer (Bowman SI), and copies local GYRE + MESA installations.
All source URLs are documented per-benchmark below. The data itself is
gitignored (~35 MB); only this README and `download_all.sh` are tracked.

---

## Directory map

| Directory | Source | Size | Notes |
|-----------|--------|------|-------|
| `gyre_examples/` | `~/gyre/models/` local copy (GYRE bundled) | 9.8 MB | Polytropes, MESA models (bcep/dcep/rgb/solar/spb/sss), FGONG, WDEC, Parfait |
| `fgong_archive/` | Aarhus users-phys.au.dk (Christensen-Dalsgaard) | 2.3 MB | Model S solar reference (fgong.l5bi.d.15, d.15c), cptrho, file-format PDF |
| `kepler_legacy/` | VizieR J/ApJ/835/172 (Lund+ 2017) | 0.5 MB | 66 Kepler solar-like stars, complete frequency tables |
| `bowman2019/` | Springer static-content (Nat Ast 3, 760) | 22 MB | Supplementary PDF — convection-driven g-mode excitation evidence |
| `jones2011/` | github.com/geodynamics/Rayleigh sparse clone | 0.7 MB | Input decks: `j2011_steady_hydro`, `j2011_steady_mhd`, `c2001_case0/1`, `b2010_case0_0T/80T/100T`, `anelastic_dimensional` |
| `dedalus_examples/` | github.com/DedalusProject/dedalus sparse clone | 0.4 MB | Rayleigh-Benard / Lane-Emden ball / rotating convection — Path B reference |
| `mesa_models/test_suite` | symlink to `/home/kiriko/mesa-ref/star/test_suite/` | 100 MB (symlinked) | 105 MESA test cases (1M → 20M, solar, cepheids, dynamo, etc.) |
| `posc/` | astro.up.pt (CoRoT@Portugal / ESTA) | 60 KB | Task 2 landing page + ESTA Task 1 Roadmap PDF; **per-file listing suppressed** — need manual scrape by filename |
| `christensen2001/` | — | 0 | Reference tables only in PEPI 128, 25 (paywalled paper); see Rayleigh input decks in `jones2011/rayleigh_src/input_examples/c2001_case*` |
| `ahlers_rb/`, `kepler_slf/`, `tess_tasc/` | — | 0 (tasc_index.html only) | MAST HLSP requires targeted download by TIC list; too bulky for bulk archive |

---

## Per-benchmark guidance

### 1. GYRE-compatible polytropes (Phase 0 ext+ continuation)

`gyre_examples/poly/` contains HDF5 polytropes for n = 0, 1.5, 3.0, 4.0 and
two-zone n=3.0+3.0 (-jump variant). Use these to extend the Exp K
Chebyshev benchmark beyond just `poly3.txt`:

- `poly/0.0/poly.h5`       — trivial incompressible limit
- `poly/1.5/poly.h5`       — convective core (fractional σ, expected N^-2)
- `poly/3.0/poly.h5`       — Eddington (already frozen in Exp I/J/K)
- `poly/3.0+3.0/poly.h5`   — two-zone smooth
- `poly/3.0+3.0-jump/poly.h5` — two-zone with jump (tests Gibbs handling)
- `poly/4.0/poly.h5`       — edge case, mostly academic

### 2. MESA stellar models (Phase 1 background evolution)

`gyre_examples/mesa/` has five evolved stars ready for GYRE ingestion:

- `solar/solar.mesa`  — Sun-like star, Model S analogue
- `bcep/bcep.mesa`    — β Cep (high-mass pulsator, p-modes)
- `dcep/dcep.mesa`    — δ Cep (classical pulsator)
- `rgb/rgb.mesa`      — red giant (mixed modes)
- `spb/spb.mesa`      — slowly pulsating B star (g-modes, high n_g)
- `sss/sss.mesa`      — speculative super-structure

### 3. Model S (solar seismology reference)

`fgong_archive/fgong.l5bi.d.15c` is the classical Christensen-Dalsgaard
1996 Model S in FGONG format. `file-format.pdf` documents the FGONG
column layout.

### 4. Kepler LEGACY (Lund+ 2017)

`kepler_legacy/table{1,6,7,8}.dat` — 66 Kepler main-sequence stars.
- table1: target summary
- table6: individual mode frequencies
- table7: frequency ratios
- table8: mode IDs

Use for Phase 3 realistic comparison if live mode projection produces
matching spectral content.

### 5. Jones 2011 anelastic benchmark (Phase 2 target)

`jones2011/rayleigh_src/input_examples/`:
- `j2011_steady_hydro_input`  — hydrodynamic case (Icarus 216, 120 Table 5)
- `j2011_steady_mhd_input`    — magnetic case (Table 6)
- `anelastic_dimensional`     — production anelastic setup template

These are Rayleigh (Featherstone+) input decks. The *reference values*
(Table 5-7 of Jones 2011) must be re-keyed from the paper PDF — no
machine-readable table exists publicly.

### 6. Christensen 2001 geodynamo (Phase 2 alt)

`jones2011/rayleigh_src/input_examples/c2001_case{0,1}_input` — these are
the same benchmark cases 0 and 1 from Christensen+ 2001 PEPI 128, 25,
adapted for Rayleigh.

### 7. Dedalus examples (Path B reference + Phase 1 RB baseline)

`dedalus_examples/dedalus_src/examples/`:
- `evp_1d_rayleigh_benard` — 1D eigenvalue, convection onset
- `ivp_2d_rayleigh_benard` — 2D time integration, our Phase 1 target
- `nlbvp_ball_lane_emden`  — Jacobi-weighted Lane-Emden (Path B template)
- `evp_shell_rotating_convection` — 3D shell, Phase 4+ reference
- `evp_disk_pipe_flow`     — EVP on disk basis

### 8. Bowman et al. 2019 SI

`bowman2019/Bowman2019_SI.pdf` — supporting info for the Nature Astronomy
paper on OB-type star SLF variability driven by convective cores. Primary
citation for the Phase 3 "convection drives g-mode excitation" angle.
Light curves themselves are on MAST (not bulk-downloaded here; pull via
`astroquery.mast` keyed on their TIC list in Extended Data Table 1).

---

## What's missing

1. **POSC frequency tables** (Moya 2008 Ap&SS 316, 231) — ESTA server has
   directory listing suppressed. The 5 reference models + 12-code
   frequency tables are only obtainable by email request to A. Moya or
   M. J. P. F. G. Monteiro. Placeholder `posc/ESTA_Task1_Roadmap.pdf` has
   the experimental setup.
2. **Jones 2011 reference tables** — tabulated values for the 9-code
   comparison only exist in the paywalled Icarus paper. Scrape from PDF
   by hand when Phase 2 benchmark is ready.
3. **TESS TASC lightcurves** — MAST HLSP bulk is ~100 GB per sector;
   only download a TIC-specific subset when Phase 3 is ready.
4. **Rogers & Glatzmaier 2006** — no public code / data release.

---

## Provenance

Downloads completed 2026-05-03 under commit `<see git log>`.
All files are small enough to ship with the repo; aggregate size
(excluding symlinked MESA test_suite) ~35 MB.
