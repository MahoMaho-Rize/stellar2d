# Honest 1-Page Assessment — How the Shell-Merger Literature Maps to
# Our "2D Anelastic↔Compressible Benchmark" Proposal

Date: 2026-05-06  
Corpus: 35 papers, 2018-2026 (23 in the target 2024-2026 window)

## Bottom line

**The recent shell-merger literature (2024-2026) does NOT support the research
framing that our project has been building.** There is no open question in the
shell-merger community that is phrased as "when does the anelastic approximation
break down relative to compressible for shell-merger flows?" because **zero
papers in the 2024-2026 corpus use the anelastic approximation for this
problem.** The grep for "anelastic" across 35 extracted PDFs returns 0 hits.

## What the literature IS asking (honest observation)

1. **Does shell merger happen in 3D, and on what timescale?**  
   Answered YES by Yadav 2020 (200 s pre-collapse), Rizzuti 2024 (5 hr pre-collapse, multi-turnover).
2. **What are the correct 3D mixing profiles / entrainment rates?**  
   Post-processed to 3-4 dex yield spread (Issa 2025 ×3).
3. **What does a real 3D progenitor do to the explosion?**  
   Bollig 2021 (1 Bethe achievable), Lella 2026 (GW signature).
4. **How is shell merger observed?**  
   Cas A XRISM P/Cl/K (XRISM 2025), Sato 2025 Ne/Si in Cas A, six SNR papers in 24-26.
5. **Can 1D predictions of merger occurrence be trusted?**  
   Roberti 2025 (209 models, X_C12 & M_CO predictors), Laplace 2024 (compactness).

None of these frame the problem as an anelastic validity question.

## Why the community stays compressible (and we should re-examine our framing)

**Physical reason**: shell-merger flows have Ma ≈ 0.1-0.3 at the active merger
(Yadav 2020 "Ma~0.1 throughout the O shell", Andrassy 2018 "Ma>0.2 locally",
Collins 2018 "Ma_O up to 0.3 in 16-26 M☉ O-shells"). This is the regime where
anelastic's foundational assumption (|δρ/ρ| ~ Ma² << 1) becomes marginal, not
"just-short-of-compressible." The community has collectively converged on
**compressible-with-low-Mach-flux-correction** (SLH AUSM+-up, LHLLD) rather than
anelastic, precisely because they want to keep the door open for the dynamical
merger event itself.

**Numerical reason**: the dominant codes (PROMPI, PPMstar, Prometheus, FLASH,
SLH, MUSIC) are all compressible or low-Mach compressible. The anelastic codes
active in stellar physics (ASH, Rayleigh, MagIC) are all in the solar /
main-sequence / dynamo space where Ma ~ 10⁻⁴ — not pre-SN convection.

## What this means for our project

Our current CLAUDE.md framing is:
> "pseudo-spectral 2D anelastic spectral framework, Chorin splitting, ω²=1.0
> vs analytic 0.8 bias" → suggests a 2D spectral anelastic solver being built
> as a stellar-convection testbed.

The literature suggests three possible pivots, in decreasing order of honesty:

### Pivot A (most honest): reposition the benchmark
Not "anelastic-vs-compressible breakdown threshold for shell merger" (nobody
asks this). Instead:

- **"Reproducibility of shell-merger 3D results across dimensionality and
  approximation" — 2D-compressible vs 2D-anelastic vs 3D-compressible on
  an Andrassy-2022-style idealized setup.** This has a ready-made test problem
  and an already-5-code-validated benchmark dataset (Zenodo 5796842). Our 2D
  anelastic numbers plug in directly as a new data point.
- Target: augment Andrassy et al. 2022 with an anelastic entry. The paper
  itself states the test problem is "accessible to low-Mach approximation
  codes" — anelastic is arguably the next step they invited. This is a
  constructive contribution.

### Pivot B: shift target phase
- Shell mergers (Ma ≈ 0.1-0.3) are the wrong target for anelastic.
- Anelastic's sweet spot is core-H / core-He / main-sequence convection
  (Ma ~ 10⁻³ to 10⁻⁴), exactly where PPMstar MS (Pathak 2025 series), 
  Horst 2021 SLH, and Baraffe/Pratt MUSIC work. Repositioning our benchmark
  toward **main-sequence or core-He convection** where Ma<10⁻² would put
  us in a regime where anelastic IS the natural choice.
- Downside: core-H convection is less "cool" than pre-SN shell merger, and
  the observational fingerprint is indirect (asteroseismology only).

### Pivot C: keep the shell-merger target, drop the anelastic framing
- If the dynamical draw of our code is really the shell-merger problem, drop
  anelastic and go to a low-Mach compressible formulation (like SLH's
  AUSM+-up). That is where the field is. Our existing `anelastic-sl-spectral`
  branch would need substantial re-architecting, but the `cart_ale2` stable
  solver and the `pseudo_spectral` framework already live in this territory.

## Specific papers to read in full before pivoting

1. **Andrassy et al. 2022 (5-code)** — `andrassy2022_5code_comparison.pdf`. 
   The test problem spec + Zenodo data (5796842) is the single best benchmark
   target. They explicitly invite low-Mach codes.
2. **Rizzuti et al. 2024 (PROMPI shell merger)** — `rizzuti2024_3d_shell_merger.pdf`.
   State of the art 3D shell merger. They give Ma, v_rms, entrainment, and
   the ransX github for mean-field diagnostics. If we ever do a 2D proxy,
   we match these numbers.
3. **Issa et al. 2025 ×3** (`issa2025_p_nuclei_3d`, `issa2025_odd_z_3d`,
   `issa2025_44ti`) — show how much nucleosynthesis depends on the mixing
   profile. They parameterize mixing profiles extracted from 3D. If we do an
   anelastic/compressible comparison, the **mixing-profile dependence** is
   where our numbers matter.
4. **Horst 2021 (SLH low-Mach He-shell)** — the closest method to anelastic
   still in compressible form. Ma=10⁻³-10⁻². Direct template for "what low-Mach
   stellar convection looks like."
5. **Leidi 2023 (SLH MHD O-shell)** — a Ma≈0.04 idealized O-shell. If we
   do idealized shell convection (pre-merger) this is the target. Andrassy
   2022 uses the same Andrassy-Edelmann stratification.

## Ground-truth statement

The working assumption of this project — that there is an open "anelastic
breakdown threshold" problem in the shell-merger community that a 2D spectral
code can resolve — **is not supported by the 2024-2026 literature.** The
community has already answered this with their code choices: anelastic is not
the right approximation for shell-merger Mach numbers, and they have picked
low-Mach compressible instead. The valid contributions a 2D anelastic code can
still make are (a) a clean benchmark entry in the Andrassy-2022 test suite
(pivot A), or (b) a move to a lower-Ma regime where anelastic is the
method-of-choice (pivot B), or (c) a methodology change to low-Mach
compressible (pivot C).
