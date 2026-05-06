# Route A Survey — 2D Extension of the Andrassy 2022 5-Code Shell-Convection Benchmark

Date: 2026-05-06. Scope: literature + resource audit for Route A ("2D companion to Andrassy et al. 2022"). Corpus: 35 shell-merger PDFs under `../literature_2024-2026/pdfs/` + 4 newly downloaded today (Andrassy 2023 Part I, Leidi 2024 low-Mach, Dethero 2024 2D-vs-3D, Leidi 2026 Phlegethon). Mandate: ground truth, no cheerleading.

---

## §1 — Andrassy 2022 itself: the test problem

Andrassy et al., A&A 659 A193 (2022); arXiv:2111.01165; DOI:10.1051/0004-6361/202142557.

**Geometry**: Cartesian plane-parallel box, `−1 ≤ x,z ≤ 1`, `1 ≤ y ≤ 3` (aspect 2:1 horiz:vert, convective layer depth = 1). Grids 128³ (all 5 codes), 256³ (all 5), 512³ (PPMSTAR full + PROMPI short). Dimensionless, units in Table 1.

**IC**: Lower half y<2 isentropic (γ₀=5/3); upper half y>2 stably stratified (γ₁=1.3), blended across `|y−2|<1/16`. Gravity `g(y)=g₀·f_g(y)·y^(−5/4)` with g₀=1.414870; f_g tapered to 0 near y=1, y=3 (width 1/16) so ρ, p flatten at walls → reflective BC admissible. Two-fluid μ₀=1.848 (conv) / μ₁=1.802 (stable), mass fraction X₁ passive-advected, 1/16 sinusoidal smoothing at y=2 interface. Ideal gas EOS γ=5/3 everywhere in the solver (piecewise γ only in the HSE construction).

**Forcing**: sinusoidal heat source `q̇=q̇₀ sin(8πy)` for `y ∈ [1, 1.125]`, `q̇₀=3.795720×10⁻⁴`, `L=1.2082×10⁻⁴`. **Boosted ×22.5 vs 25 M☉ stellar model**. No cooling, neutrinos, or nuclear source. Resulting rms Ma ≈ 0.04.

**BC**: y-walls reflective (frictionless, non-conductive, zero mass+energy flux); x,z periodic. Initial perturbation `Δρ/ρ₀=5×10⁻⁵·(q̇/q̇₀)·[sin(3πx)+cos(πx)][sin(3πz)−cos(πz)]`; SLH adds 5×10⁻⁷ white noise.

**Duration**: `t_end=2000` ≈ 25 turnovers (`τ_conv=80`). Analysis window 500–2000 (~19 τ_conv, t<500 discarded). Output every 5 units (5123 PROMPI: 1.266).

**Diagnostics** (all horizontally averaged then time-averaged):
| Metric | Definition | Convergence |
|---|---|---|
| `ṽ_rms(y)` | Eq. 15, mass-weighted | within 3σ of 5123 at 1283 |
| rms Ma | ≈ 0.04 | code-independent |
| `y_ub(t)` | max of `∂_y X̃₁` | 14% (1283) → 5% (2563) scatter |
| `H_ub` (boundary thickness) | Eq. 26 | converged ≈ 0.04 at 2563 |
| `M_e(t)` total entrained mass | Eq. 27 | **1283: 9%; 2563: 4% spread** |
| `Ṁ_e(t)` | central diff + τ_av=3τ_conv | converged at 1283 |
| KE spectrum ‖Ψ(k)‖² | 2D FFT at y=1.7 and y=2.7 | Kolmogorov k^(−5/3); 1283 resolves k<5 |
| F_H(y) (enthalpy flux) | Eq. 28 | within statistical scatter |
| F_k(y) (kinetic flux) | Eq. 29 | ≈30× smaller than F_H |
| ⟨ΔA/A⟩, ⟨ΔX₁/X₁⟩ | Fig. 13 | agree within 3σ |

**Entrainment law**: `Ṁ_e ∝ v_rms·Ri_B^(−n)`. Not refit from 5-code data; paper surveys literature: n=0.74±0.04 (Cristini 2019), 0.74±0.01 (Horst 2021), 1.05±0.21 (Meakin & Arnett 2007), 1.32±0.79 (Higl 2021). 30–60% of Ṁ_e attributed to the Meakin & Arnett "heating contribution".

**Action items**: setup is fully reproducible from Eqs. 1–6 + Zenodo 5796842. 2D port requires y-z (not x-y) slice, square domain `−1 ≤ z ≤ 1, 1 ≤ y ≤ 3`. 1283 → 2562 is trivially cheap (~6×10⁴ cells). Bottleneck is multi-species tracer + tapered gravity + reflective walls, not CPU. **Caveat**: 2D turbulence enstrophy cascade k^(−3) differs from 3D k^(−5/3); cite Dethero 2024 for the 2D↔3D entrainment calibration factor.

---

## §2 — Andrassy's own follow-up work 2022 → 2026

**Critical finding**: Andrassy has NOT published a 2D companion to the 5-code test problem. He HAS done a 2.5D (axisymmetric) + 3D study on a different problem (MS core convection). The 5-code test has been re-used once externally, by Leidi+ 2024 (SLH-only methods sweep). Full Andrassy record 2022–2026:

| # | arXiv | Year | Role | Title | Relevance to Route A |
|---|---|---|---|---|---|
| A1 | 2111.01165 | 2022 | **1st author** | Dynamics in a stellar convective layer… Comparison of five 3D codes | **The paper we extend** |
| A2 | 2210.01641 | 2022 | 3rd of 8 (Leidi, Birke, Andrassy, …) | Finite-volume scheme for compressible MHD at low Mach — A&A 668, A143 | SLH-MHD methods; not a benchmark |
| A3 | 2307.04068 | 2023 | **1st author** | Convective core boundary in upper MS stars. Part I: **2.5D and 3D simulations** — A&A (v2 Aug 2024) | **The closest thing to a 2D follow-up**; DIFFERENT problem |
| A4 | 2309.17225 | 2023 | 2nd of 5 (Leidi, Andrassy,…) | Turbulent dynamo at O-burning shell boundary — A&A 679, A132 | MHD extension of 5-code idealized O-shell; Ma≈0.04 |
| A5 | 2402.16706 | 2024 | **2nd of 6** (Leidi, Andrassy,…) | High-order Godunov methods in low-Mach flows — A&A 686, A34 | **Uses Andrassy 2022 IC** with reduced heating for a 3D cost-scaling study |
| A6 | 2407.02566 | 2024 | co-auth | He-WD merger magnetic fields (Pakmor+) — A&A 691, A179 | Not relevant |
| A7 | 2410.07841 | 2024 | co-auth | 3D common envelope (Vetter+) — A&A 2024 | Not relevant |
| A8 | 2504.12213 | 2025 | co-auth | MHD outflows common-envelope (Vetter+) — A&A 698, A133 | Not relevant |
| A9 | 2601.04188 | 2026 | 4th (Gagnier, Leidi, Vetter, Andrassy, Röpke) | Local common-envelope simulations — A&A accepted | Not relevant; Athena++ code |
| A10 | 2604.12672 | 2026 | co-auth (9th of 14) | **Phlegethon code-paper** (Leidi+) | New SLH-successor MHD compressible code; no Andrassy-2022 test run in abstract |

**Absences**: No Part II of the 2023 core-boundary paper (Part I v2 Aug 2024). No code- or data-release paper beyond Zenodo 5796842. No review articles. SLH → Phlegethon (2026) does NOT advertise the Andrassy 2022 test as a benchmark in its abstract.

**Andrassy 2023 (arXiv:2307.04068) — "2.5D" means**: axisymmetric polar grid (`∂/∂φ=0`), all three vector components retained. 15 M☉ ZAMS core convection (not O-shell). Boost b∈{10³,…,10⁶}. Grids Nr∈{128,…,1024}, Nθ=Nr/2. Key result: penetration scales as b^0.285±0.002 in 2.5D vs b^0.295±0.006 in 3D — **statistically agree** on penetration, despite morphology diverging (Fig. 4). The paper's bulk thermal-equilibrium metrics (depth, α) are 2.5D↔3D-robust; turbulence statistics are NOT tested in that paper. This IS a positive methodological precedent for 2D-on-shell-test, with the caveat that our Route A diagnostics include KE spectra and f_d asymmetry — which Andrassy 2023 did NOT cross-check.

**Leidi 2024 (arXiv:2402.16706) re-uses the A22 IC**: "initial conditions … adopted from Andrassy et al. (2022). Here, we opt to reduce the strength of the heat source … to achieve lower convective speeds" (lines 129–130 of our extracted text). Tests 18 reconstruction × Riemann-solver combinations on 2D-KH + 3D-A22-like convection, Ma 10⁻³–10⁻¹. Reports **~4 orders of magnitude cost-at-fixed-accuracy spread**. Does NOT refit Ri_B n — the dissipation-free-reference niche stays open.

Downloaded today to `pdfs/`: andrassy2023_core_boundary_2p5D.pdf, leidi2024_low_mach_godunov.pdf, dethero2024_2d_vs_3d_music.pdf, leidi2026_phlegethon_mhd_code.pdf; txt extractions in `txt/`.

**Niche status**: NOT collapsed. Andrassy+ is on core-MS penetration + MHD; nobody has done a 2D cross-solver companion to A22. Leidi 2024 is the closest external use but is SLH-only methods-internal. Our 2D + multi-solver (anelastic + cart_ale2 + lowmach) entry remains open.

---

## §3 — The Andrassy Zenodo dataset (record 5796842)

URL: https://zenodo.org/records/5796842 (public, accessible). Title matches the paper; 21 coauthors; v2 published 2021-10-29; license **CC-BY 4.0** (allows republishing derivatives). Total **46.5 GB**:
- `1D-profiles.tar.gz` — 381.2 MB (horizontal-averaged radial profiles — our primary target)
- `spectra.tar.gz` — 10.7 MB (KE spectra tables)
- `2D-slices_part{1..4}.tar.gz` — ≈45.9 GB combined (z=0 slice X₁, A, ω, v_y)
- Low/high-res mp4 videos (~200 MB)

Companion analysis notebooks in "CoCoPy" / "CoCoHub" GitHub repo (referenced from the Zenodo record); they ingest `1D-profiles.tar.gz` and reproduce every figure of the paper. Not included: 3D checkpoints, generic initial-condition codes, KE spectra outside y∈{1.7, 2.7}.

**Actions**: download `1D-profiles.tar.gz` + `spectra.tar.gz` (~392 MB). CoCoPy notebooks become our diagnostic-pipeline template — 3σ-band statistics code is already there. Skip the 46 GB slice tarballs (visual-only). Publish our 2D companion as a new Zenodo record + CC-BY.

---

## §4 — ransX toolkit (github.com/mmicromegas/ransX)

Maintainer M. Mocák. Purpose: RANS mean-field post-processing of 2D/3D compressible multi-fluid simulation outputs — transport, flux, variance, and closure-residual equations for mass, momentum, KE/IE/TE, T, H, p, X_i. Python 3 (numpy, scipy, matplotlib); BSD-2-clause; 343 commits, 71.7% Python + 19.6% TeX + 8.5% Fortran.

**Repo state (2025-10-26 last push, via GitHub API)**: 6 stars, 1 fork, 14 open issues. Actively receiving pushes (not abandoned), but effectively single-maintainer with thin adoption — NOT a community standard. Used by Mocák+ 2018 (MNRAS 481 2918 — the original Ne-O merger), Rizzuti+ 2023 & 2024 (PROMPI), Georgy+ 2024 (PROMPI Ne). Referenced in Andrassy 2022 footnote 3 as "the ransX framework"; PROMPI ships ransX-compatible mean-field dumps. Input format: HDF5 / plain text of time-averaged multi-species fields on a regular mesh; each user adapts their code's dump format.

**Actions**: CoCoPy notebook conventions are our primary diagnostics pipeline (exact plots of the Andrassy 2022 paper). Add a ransX-compatible adapter only if we want cross-compatibility with Rizzuti 2024 / Georgy 2024 style plots — that's ~1 week of work and a nice-to-have, not critical.

---

## §5 — 5-code benchmark lineage: the parent setups

Chronology of idealized-O-shell simulations that fed into Andrassy 2022:

| Ref | Year | arXiv | Code | Geometry & resolution | Ma | Entrainment metric | Distinguishing feature vs A22 |
|---|---|---|---|---|---|---|---|
| Meakin & Arnett | 2007 | astro-ph/0611315 | PROMPI (progenitor) | 3D wedge, ~256³-ish | 0.04 | `E = uE/σH`, Ri_B^(-n), **n=1.05±0.21** at 20≲Ri_B≲420 | first paper; introduced Ri_B framework. 2D vs 3D already noted different flow morphology and velocity amplitude. |
| Cristini+ | 2017 | 1610.05173 | PROMPI | 3D spherical, 128³–1024³ | — | boundary widths 10%/30% of H_P top/bottom | first resolution-convergence O-shell; C-burning not O-burning |
| Jones, Andrassy, Sandalski+ | 2017 | 1605.03766 | PPMSTAR | 3D full 4π, 768³ & 1536³ | — (Ma implied ≈0.03) | Ṁ_e ∝ L linearly, ∝ v_shear^3 | established L-linear scaling; 1D exp-diffusion fit f=0.03 |
| Mocák+ | 2018 | 1810.05176 (exact ID TBD, abstract only) | PROMPI, 23 M☉ Ne-O merger | 3D | — | Kolmogorov composition-variance dissipation | first **ransX application**; two-shell merger in single convection zone |
| Cristini+ | 2019 | 1901.10531 | PROMPI | 3D, luminosity-boosted C shell | — | **n ≈ 0.75** | redefined Ri_B exponent from n≈1 to n≈0.75; triggered Higl 2021 reinvestigation |
| Edelmann+ | 2019 | 1903.09392 | SLH | 3D 3 M☉ MS | 10⁻³ | surface wave spectrum (plume-dominated, featureless) | anti-Lecoanet-2021 (strong resonance lines) baseline |
| Horst+ | 2021 | 2107.02133 (already in our PDFs) | SLH low-Mach | 3D 25 M☉ core-He | 10⁻³–10⁻² | **n=0.74±0.01** | first SLH pipeline; validates Andrassy setup family at lower Ma |
| Higl+ | 2021 | 2012.05262 | MAESTRO (!) | 2D, 1.3–3.5 M☉ MS, 20–430 turnovers | — | f_ov=0.010–0.017, **n fit with 1.32±0.79 spread** | **first 2D entrainment paper** in the lineage (MAESTRO = low-Mach compressible, similar to SLH) |
| Scott+ | 2021 | 2103.06196 | Geneva 1D | 1D | — | `A≈2×10⁻⁴, n=1` | folded entrainment results into 1D stellar evo |

**Observations**:
1. A22 setup is essentially the Jones+ 2017 / Andrassy+ 2020 plane-parallel 25 M☉ O-shell simplified for portability; 5-code × 3-resolution was the first multi-code instance.
2. **Higl+ 2021 is the only 2D paper in this lineage**, on 1.3–3.5 M☉ MS core overshoot with MAESTRO — not on the O-shell test. Our 2D O-shell entry still fills a real gap.
3. The n exponent range 0.74 → 1.32 across 7 papers is **the publishable niche** — if 2D anelastic recovers the SLH-family n=0.74, that is a methods-paper result.

**Actions**: queue Higl 2021 (arXiv:2012.05262) for download as a 2D methodology precedent. Probe whether A22 heating can be tuned down (à la Leidi 2024) to hit Ma ≈ 10⁻³ where the anelastic approximation is definitively accurate.

---

## §6 — Complementary 2D/3D idealized-convection benchmarks NOT shell-merger specific

Alternative / sibling test problems that could serve as route B (Boussinesq pseudo-spectral) sanity checks:

| Benchmark | Paper(s) | Code family | Physics | What it tests |
|---|---|---|---|---|
| **MUSIC 2D-vs-3D paired series** | Dethero, Pratt, Vlaykov, Baraffe, Guillet, Goffrey, Le Saux, Morison, A&A 692, A46 (2024), arXiv:2409.09815 | MUSIC (compressible implicit) | 4 stellar setups: 3 M☉ RGB, 1 M☉ pre-MS, Sun, 20 M☉ MS core. Paired 2D↔3D runs. | **Direct quantification of 2D-vs-3D convection differences**. Filling factor "indistinguishable"; plume interaction & penetration parameters systematically differ. **The calibration reference our 2D Route A must cite.** |
| **Anders et al. 2022 Dedalus penetrative-convection benchmark** | arXiv:2110.11356 ApJ 926 169 | Dedalus (spectral anelastic/Boussinesq) | 3D Boussinesq, internal heating, height-dependent κ_rad, thousands of overturns | Gold-standard penetration-depth scaling with penetration parameter ℒ. Boussinesq analog of the Andrassy setup. Direct fit to MESA solar model. |
| **Brown, Oishi, et al. 2020 Boussinesq stratified convection** | ApJ 902 63 (2020) — arXiv:2007.09023 (guess; retrieval failed — to verify) | Dedalus | Boussinesq 3D penetration | Smaller-scale companion to Anders 2022 |
| **Currie, Barker, Lithwick, Pratt 2020** | ApJ 898 6, arXiv:2005.13239 | spectral (Dedalus-class) | A-type star core+envelope 2D/3D | Penetration at Pr≠1, mild stratification |
| **Verhoeven & Stellmach 2014 double-diffusive** | JFM 747 147 | spectralDNS | 3D Boussinesq two-scalar | Double-diffusive layering; staircase structures. Relevant for semi-convection but not direct for shell merger. |
| **Deusebio, Caulfield, Taylor 2015** | JFM 781 298 | spectralDNS | stratified plane Couette | Intermittency boundary, Reynolds/Richardson scaling. 2D/3D DNS reference, **dissipation-free benchmark for Ri_B physics.** |
| **Salehipour, Peltier, Mashayek 2015** | JFM (June 2015) | DNS | turbulent diapycnal stratified shear | Mixing efficiency vs Pr. The closest we have to a "gold-standard Ri_B Boussinesq" mixing-efficiency reference. |
| **Rayleigh code benchmark suite** | https://geodynamics.org/resources/rayleigh | Rayleigh (pseudo-spectral anelastic MHD) | 3D spherical anelastic dynamo | Solar convection zone, Hindman benchmark, Christensen et al. 2001 dynamo benchmark. Not directly applicable to shell convection, but the gold standard for anelastic validation. Active, CIG-maintained. |
| **PADC / ASH solar benchmark** | Brun, Miesch, Toomre 2004 (and many follow-ups) | ASH (anelastic spherical harmonic) | 3D global solar convection | Reference for anelastic in a stratified spherical shell, deep convection. Low-Mach only (10⁻³–10⁻⁴). |

**Actions (Q6)**: Anders+ 2022 (arXiv:2110.11356) is the highest-value target — Boussinesq, Dedalus, 3D, MESA-compared; the 2D reduction is a natural Route B smoke test. Deusebio 2015 + Salehipour 2015 give dissipation-free Ri_B references, but as stratified-shear not heating-driven convection; needs an adapter to map. Dethero 2024 is the mandatory 2D→3D citation. The direct "Boussinesq-spectral vs A22 compressible" comparison is NOT in the literature → Route B is a real (minor) methods contribution.

---

## §7 — Route B (pseudo-spectral Boussinesq) target benchmarks

This ties directly into Q6 but is the more focused "gold-standard Ri_B" line.

**Canonical Boussinesq two-layer stratified mixing-box lineage**:

| Paper | Year | Key contribution |
|---|---|---|
| Linden, JFM 71 385 | 1975 | Laboratory salt-water experiment. First `Ṁ_e ∝ Ri_B^(−1)` entrainment measurement for sharp-interface stratified convection. Referenced by Andrassy 2022 §3.2 as the laboratory analog. |
| Fernando, Ann. Rev. Fluid Mech. 23 455 | 1991 | Review; Ri_B^(−1), Ri_B^(−3/2), Ri_B^(−2) regimes depending on Reynolds, Peclet. |
| Molemaker & Dijkstra, JFM 331 199 | 1997 | First DNS verification of Linden 1975 at moderate Re. |
| Hanson & Fedorovich, JFM 617 | 2008 | LES of penetrative convection in atmospheric boundary layer; n≈−1. |
| Scott, Hirschi, Georgy+ | 2021 | Stellar analog, n=1 asserted (see §5). |
| Fuentes & Cumming, ApJL 902 L35 (2020), arXiv:2004.14030 (verified from ADS export format) | 2020 | Boussinesq semiconvection → mixing staircase → bulk Ri_B^(−1) regime; relevant to our stellar analog. |

**Spectral-code gold-standard references**:
- Deusebio+ 2015 (stratified plane Couette) and Salehipour+ 2015 (diapycnal) are the modern DNS references. **Both use spectralDNS or Channelflow-style codes (Chebyshev×Fourier×Fourier)** at Pr=1–7, Re up to 10⁴.
- Maffioli, Brethouwer, Lindborg JFM 794 (2016) — "mixing efficiency in stratified turbulence" — also DNS reference for the Γ_mix asymptote.
- **All of these are 3D; 2D Boussinesq stratified-convection spectral-gold-standard is not well-populated**, making our `pseudo_spectral_bouss_solver` 2D Route B more of an exploratory sibling than a strict reference.

**Ri_B exponent for penetrative convection — Boussinesq** (answer to Q7 sub-question):
- Linden 1975: n=1 (sharp interface)
- Fernando 1991: n=1 (low Re), n=3/2 (high Re)
- Scott 2021 stellar fit: n=1
- Fuentes & Cumming 2020: n≈1 (semiconvection boundary)
- Andrassy 2022 (compressible): no refit; literature median n=0.74–1.05

**3D anelastic/compressible vs Boussinesq gold standard**: the A22 n=0.74 (Horst 2021 fit) vs Boussinesq n=1 (Linden) discrepancy is small (within individual-simulation scatter) and has **not been explicitly benchmarked** in either direction. Open niche for Route B if run alongside Route A — isolates the "compressible vs incompressible entrainment law" question.

**Actions (Q7)**: match `pseudo_spectral_bouss` IC to the 2-layer Linden/Fernando canonical setup (density discontinuity + bottom heating = Boussinesq reduction of A22). Scan Ri_B ∈ [10, 10³] at Re=10⁴, measure Ṁ_e, fit n. Compare to Linden/Fernando n=1 and to A22 compressible reanalysis. A&A Letters publishable if |n_spec − n_comp| ≥ 0.2.

---

## §8 — Observational data for cross-checking Route A

**Premise check**: Route A is a numerical-methods paper on an idealized setup (Ma≈0.04, heating ×22.5 boosted, ideal gas). It does NOT have a direct observational cross-check. Obs data enters only in a Route A → C extension (parametric grid of realistic pre-SN mixing profiles). The relevant ecosystem for that future step:

| Dataset | Reference | Access | Constraint |
|---|---|---|---|
| Rizzuti 2024 3D shell-merger profile | arXiv:2407.15544 MNRAS 533 3222 | ransX-formatted PROMPI, snapshots not public | Ma, v_rms(r), X_i(r) at 5 hr pre-collapse for 20 M☉ |
| Issa 2025 ×3 mixing-profile parameterisations | arXiv:2509.19240 + odd-Z + ⁴⁴Ti | Zenodo 17576026 | Diffusion-profile ensembles for mppnp post-processing |
| XRISM Cas A odd-Z (P, Cl, K) | XRISM collab 2025 | XRISM public archive (ESA/JAXA) | >6σ P/Cl/K lines; ratios probe merger nucleosynthesis |
| Sato 2025 Cas A Ne-downflow / Si-upflow | in corpus | Chandra ACIS + XRISM | Spatially resolved directional mixing |
| Matsunaga 2024 G359, Sato 2024 N49B, Kuboike 2025 J0550, Terano 2026 G284 | in corpus | XMM/Suzaku public archives | Mg/Ne line ratios as shell-merger tracers |
| Boccioli 2026 Cas A ⁴⁴Ti | in corpus | MESA+CCSN 1D | Ar/Ne ≳ 0.1 ⇒ CO merger |

**Route A action items**: none. Observations are out of scope. Deferred to Route C.

---

## Summary scorecard

| Q | Finding | Route A implication |
|---|---|---|
| Q1 | A22 fully reproducible from Eqs. 1–6 + Zenodo 5796842 | 2D y-z port with tapered gravity + reflective y-walls |
| Q2 | Andrassy 2023 is 2.5D on CORE (not shell); Leidi 2024 uses the shell IC with reduced heat; no Part II yet | niche NOT collapsed; 2D + multi-solver entry is unique |
| Q3 | Zenodo 5796842 public, CC-BY, 46.5 GB; 1D-profiles (381 MB) + spectra (11 MB) are enough | download the small archives, ignore 46 GB slice tarballs |
| Q4 | ransX active (6★, BSD, Python), used by PROMPI cohort | CoCoPy notebooks primary; ransX adapter optional |
| Q5 | 7-paper lineage, **Higl 2021 only 2D (MAESTRO, MS core)**, n-exponent scatter 0.74→1.32 | 2D O-shell fills a real gap; recovery of n=0.74 is publishable |
| Q6 | Anders 2022 (Dedalus Bouss penetration); Dethero 2024 (MUSIC 2D↔3D) | Anders = Route B smoke test; Dethero = mandatory cite |
| Q7 | Linden/Fernando n=1, Deusebio/Salehipour DNS refs, no 2D Bouss spectral gold standard | Route B exploratory only; A&A Letters ceiling |
| Q8 | XRISM / Sato / Matsunaga etc. exist but Route A is methods-only | not in scope for A; deferred to C |

**Verdict (ground truth)**:
1. Route A is viable; niche genuinely empty.
2. Leidi 2024 is the nearest published work; 3D-only, SLH-only; does not encroach on our 2D multi-solver angle.
3. **Biggest risk**: Andrassy may extend A22 to 2D in a Part II; no intel on its progress. Mitigation: M1 (multi-species tracer) running by end of 2026-06.
4. **Biggest limitation**: 2D k^(−3) enstrophy cascade ≠ 3D k^(−5/3); cannot claim 2D reproduces 3D turbulence. Dethero 2024 calibration is required up front.
5. ransX is not the blocker — CoCoPy is the simpler path.
6. Route B ceiling is A&A Letters (no 2D Bouss pseudo-spectral reference to validate against). Deprioritise unless `pseudo_spectral_bouss` is cheap to build.
