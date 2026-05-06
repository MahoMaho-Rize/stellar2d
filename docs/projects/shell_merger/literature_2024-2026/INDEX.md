# Shell Merger Literature Index (2024-01 → 2026-05)

Compiled 2026-05-06. Corpus: 35 papers downloaded, covering:
- 23 papers 2024-01 to 2026-05 (primary target window)
- 12 pre-2024 foundational references (Yadav 2020, Bollig 2021, Andrassy 2018/2022, Collins 2018, Leidi 2023, Rizzuti 2023, Horst 2021, Ritter 2017/2018, Côté 2019, Pathak 2025) — retained because the 2024-26 papers all cite them as the 3D-simulation benchmarks

PDF files: `pdfs/<name>.pdf`. Text extractions: `txt/<name>.txt`.

## Legend for "type"

- **3D-sim** = full 3D hydro simulation of (part of) a massive star
- **1D-grid** = 1D stellar-evolution grid
- **post-proc** = 1D nucleosynthesis post-processing using 3D-derived mixing
- **obs** = X-ray / observational SNR paper
- **code-paper** = numerical method / code benchmark
- **1D+CCSN** = 1D neutrino-driven explosion survey

## Synthesis table

| # | Paper | Year | Type | Code(s) | D | Shell regime | Key diagnostics | Anel vs Comp? | Data? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Rizzuti et al., MNRAS | 2024 | 3D-sim | **PROMPI** (compressible Euler + 12-isotope net) | 3D, 512×256×1024 & 768×256² | C+Ne+O merger, 20 M☉ solar-Z, 5 hr to collapse | Ma, v_rms, entrainment, ransX mean-field, dipolar asymm. | NO | ransX open source, sim data not public |
| 2 | Roberti et al., A&A | 2025 | 1D-grid (209 models) | MESA / KEPLER / GENEC / FRANEC | 1D | C-O merger occurrence statistics across mass & Z | X_C12, M_CO predictors (X_C12<0.277, M_CO<4.90 M☉) | N/A | — |
| 3 | Roberti et al. (SPAr) | 2025 | 1D-grid | FRANEC 1D | 1D | Proton-capture energy balance in C-O merger | E_SPAr / E_CO ≈ 400 | N/A | — |
| 4 | Georgy et al., MNRAS | 2024 | 3D-sim | **PROMPI** | 3D, 128³ → 1024³ | Ne-burning shell, 15 M☉ | Ma=2-23×10⁻³, v_rms, boost factor 1-1000 | NO | — |
| 5 | Rizzuti et al., MNRAS | 2023 | 3D-sim | **PROMPI** | 3D | Ne-burning complete phase | Ri_B-entrainment, Ma | NO | ransX github |
| 6 | Laplace et al., A&A | 2024 | 1D-grid | MESA 17-50 M☉ | 1D | C/Ne+O mergers → compactness | ξ_2.5, M_Fe core | N/A | Zenodo (MESA profiles) |
| 7 | Brinkman et al., A&A | 2024 | 1D-grid | MESA 20 M☉, 212-iso net | 1D | Asteroseismic mixing → shell merger | f_ov, wind yields | N/A | — |
| 8 | Roberti et al. (γ-process II) | 2024 | 1D CCSN post-proc | FRANEC + CCSN | 1D | γ-process p-nuclei in CO merger | Y_p-nucleus, E_exp | N/A | — |
| 9 | Matsunaga et al., ApJL | 2024 | obs | XMM-Newton | — | G359.0-0.9 Mg-rich SNR | Mg/Ne=1.90 → Ne-shell intrusion | N/A | — |
| 10 | Sato et al., ApJL | 2024 | obs | XMM | — | N49B Mg-rich | Mg/Ne ≳ 1 | N/A | — |
| 11 | Sato et al., Nature(?) | 2025 | obs | XRISM / Chandra | — | Cas A shell merger evidence, ≲10⁴ s pre-collapse | Ne-downflow / Si-upflow | N/A | — |
| 12 | Issa et al. (p-nuclei 3D) | 2025 | **post-proc** | NuGrid mppnp + 3D-informed mixing | 1D network, 3D-macrophys priors | O-C merger p-nuclei | Spread 0.51-1.84 dex from mixing alone | N/A | Zenodo (17576026) |
| 13 | Issa et al. (odd-Z 3D) | 2025 | **post-proc** | NuGrid mppnp + 3D mixing | 1D, 3D priors | O-C merger ⁴⁰K, P, Cl, K, Sc | [K/Fe] spread 1.48 dex, ⁴⁰K spread >3 dex | N/A | Zenodo |
| 14 | Issa et al. (⁴⁴Ti) | 2025/26 | **post-proc** | same | 1D, 3D priors | O-C merger ⁴⁴Ti | Spread 4.78 dex | N/A | Zenodo |
| 15 | Purushothaman et al. | 2025 | 1D+dust | KEPLER+MESA, dust chem | 1D | Shell merger stochasticity → dust mass | M_silicate ~0.02-0.9 M☉, ×2-5 variance | N/A | Zenodo 13645155 |
| 16 | Boccioli et al. (Cas A ⁴⁴Ti) | 2026 | 1D+CCSN | MESA+CCSN 8 model sets | 1D | Cas A Ar/Ne diagnostic | Ar/Ne ≳ 0.1 requires CO merger | N/A | — |
| 17 | Long et al. | 2026 | 1D-grid | MESA He-star 5-65 M☉ | 1D | Shell merger → compactness non-monotonic | ξ, M_Fe | N/A | — |
| 18 | Ferreira et al. (Pop III) | 2026 | 1D-grid | MESA | 1D | Low-mass Pop III shell merger → N-rich | Y_N, M_WD, neutrino flash | N/A | github PopIII |
| 19 | Lella et al. (GW) | 2026 | 3D-CCSN | **Prometheus-Vertex** | 3D, explicit | s12.28 & s18.88 with 7-min O-Ne merger pre-collapse | h_+, h_×, E_GW spectrum | NO | — |
| 20 | Terano et al. (G284) | 2026 | obs | Suzaku + NANTEN CO | — | Shell-merger SNR + γ-ray binary assoc. | Mg/Ne, Si/Mg | N/A | — |
| 21 | XRISM collab. (Cas A Cl/K/P) | 2025 | obs | XRISM | — | Cas A P, Cl, K X-ray detection | >6σ K line | N/A | — |
| 22 | Kuboike et al. (J0550) | 2025 | obs | XMM | — | LMC Mg-rich SNR, progenitor ≲15 M☉ | Mg/Ne ≈ 1, LMC freq 10-40% | N/A | — |
| 23 | Boccioli et al. (1D+) | 2024 | 1D+CCSN | 341 CCSN models | 1D | Remnant mass vs pre-collapse structure | Compactness, NS/BH | N/A | github evanoconnor |
| 24 | Pathak et al. (PPMstar IV) | 2025 | 3D-sim | **PPMstar** | 3D, 25 M☉ MS | Internal gravity waves, SLF variability | Δν spectrum | NO | Zenodo 15679630 |
| 25 | Shimada et al. | 2026 | 3D-MHD | Varma & Müller MHD code | 3D rotating | Rotating O-shell AM transport | Rossby, L_z flux | NO | — |
| 26 | Leidi et al. | 2023 | 3D-MHD | **SLH** (fully compressible MHD) | 3D, 512³ | Idealized 25 M☉ O-shell, MHD | Ma≈0.04, Re, B-field | NO (low-Mach compressible) | — |
| 27 | Andrassy et al. (5-code) | 2022 | **code-paper** | FLASH, MUSIC, PPMSTAR, PROMPI, SLH | 3D, 128³-512³ | Idealized O-shell, box | Mass entrainment, velocity profiles | **MENTIONS low-Mach approx**, NO anelastic | Zenodo 5796842 (sims + scripts!) |
| 28 | Horst et al. (SLH He-shell) | 2021 | 3D-sim | **SLH low-Mach** | 3D | 25 M☉ core-He burning | Ma=10⁻³-10⁻², AUSM+-up vs AUSM+B | NO | Zenodo 4776452 |
| 29 | Yadav et al. | 2020 | 3D-CCSN | **Prometheus** (compressible) | 3D 4π, 18.88 M☉ | O-Ne merger 7 min pre-collapse | Ma≈0.1, large-scale asymm. | NO | — |
| 30 | Bollig et al. | 2021 | 3D-CCSN | **Prometheus-Vertex** | 3D 4π | Same 18.88 M☉, pre+post bounce | Explosion E, NS kick, O-Ne merger | NO | — |
| 31 | Andrassy et al. (C-ingestion) | 2018 | 3D-sim | **PPMstar** | 3D, 768³-1152³ | C-ingestion into O-shell, 25 M☉ (idealized) | Ma up to >0.2, entrainment, L-Ma scaling | NO | Zenodo 2592134 |
| 32 | Collins et al. | 2018 | 1D-grid | KEPLER large grid | 1D | Ma of O/Si shells across 16-26 M☉ | Ma_O up to 0.3, Ma_Si ≲0.15 | N/A | — |
| 33 | Côté et al. (Cr) | 2019 | GCE | NuGrid + OMEGA GCE | — | Si-C merger → [Cr/Fe] overestimate | [Cr/Fe] | N/A | github JINAPyCEE |
| 34 | Ritter et al. (NuGrid II) | 2017 | 1D-grid+nuc | MESA + mppnp | 1D | Large yield grid, O-C mergers → odd-Z | — | N/A | CADC + NuGridPy |
| 35 | Ritter et al. (conv-reactive) | 2018 | 3D + 1D post-proc | **PPMstar** + diffusion | 3D + 1D | C-ingestion conv-reactive, odd-Z prod | OP_s ≈ 7 | NO | — |

## Quantitative corpus statistics

- **Total papers downloaded**: 35
- **Papers that perform 3D hydrodynamic simulations of actual shell interactions** (not just 1D stellar evolution): **8**
  - Rizzuti 2024 (O+Ne+C merger, PROMPI) — the flagship 2024 paper
  - Georgy 2024 (Ne-shell, PROMPI)
  - Yadav 2020 (O-Ne merger, Prometheus)
  - Bollig 2021 (same as Yadav in CCSN context)
  - Lella 2026 (GW signatures, Prometheus-Vertex, O-Ne merger progenitor)
  - Leidi 2023 (MHD O-shell idealized, SLH)
  - Shimada 2026 (MHD rotating O-shell)
  - Andrassy 2018 + 2022 (PPMstar C-ingestion, idealized)
  - Pathak 2025 (PPMstar MS, not a shell merger)
  - Horst 2021 (SLH He-shell, not a shell merger)
  - Rizzuti 2023 (PROMPI Ne complete, not a merger)

  Strict "real shell merger in 3D": **Rizzuti 2024, Yadav 2020, Bollig 2021, Lella 2026** (same progenitor basis as Yadav/Bollig). Plus Andrassy 2018 idealized ingestion runs. → **4 unique real-shell-merger 3D simulations** across the entire 2018-2026 window.

- **Papers using anelastic approximation**: **ZERO**  (grep "anelastic" across all 35 PDFs returns 0 matches)
- **Papers using low-Mach compressible** (AUSM+-up, LHLLD, CABARET…): 3 — SLH family (Horst 2021, Leidi 2023, Andrassy 2022), MUSIC (implicit compressible). 
- **Papers using fully explicit compressible PPM/HLLC**: 5+ — PROMPI (Meakin-Arnett, used by Rizzuti, Georgy, Cristini, Mocák), PPMstar (Andrassy, Herwig, Woodward), Prometheus (Yadav, Bollig, Lella).
- **Papers comparing anelastic vs compressible in same paper**: **ZERO**.
- **Papers comparing different compressible codes**: 1 — Andrassy et al. 2022 (FLASH, MUSIC, PPMSTAR, PROMPI, SLH).
- **Papers that explicitly state "anelastic breakdown threshold is not quantified"**: **ZERO**. Nobody is asking the question; nobody cites it as an open problem in shell-merger literature.
- **Papers with released simulation data**: 
  - Zenodo: Andrassy 2022 (5796842, **full sim outputs + Jupyter notebooks**), Horst 2021 (4776452), Andrassy 2018 (2592134), Laplace 2024 (13645155), Pathak 2025 (15679630), Issa 2025×3 (17576026), Purushothaman 2025 (13645155)
  - GitHub: ransX (github.com/mmicromegas/ransX) — used by Rizzuti, Mocák, Georgy for post-analysis
  - CADC: NuGrid full data
- **Observational papers**: 6 (Sato 2024/2025, Matsunaga 2024, Kuboike 2025, Terano 2026, XRISM 2025, Boccioli 2026)
- **Papers reporting convective Mach numbers in stellar shells**:
  - MS / core-He: 10⁻⁴ to 10⁻³ (Horst 2021, Andrassy 2022)
  - Ne-burning shell: 2-4×10⁻³ (Georgy 2024 nominal), up to 2×10⁻² (boosted)
  - O-burning shell (no merger): 0.04 (Andrassy 2022 idealized), 0.03-0.1 (Collins 2018 across masses)
  - Si shell: ≤0.15 (Collins 2018)
  - **O-Ne active shell merger**: Ma ≈ 0.1-0.3 (Yadav 2020, Andrassy 2018 with boost, Rizzuti 2024)

## Code landscape (what the field actually uses)

| Code | Approach | Used by | Notes |
|------|---------|---------|-------|
| PROMPI | explicit compressible PPM + grid | Rizzuti 2023/24, Georgy 24, Mocák, Cristini | Box-in-star, wedge / partial sphere. Direct descendant of PROMETHEUS. |
| PPMstar | explicit compressible PPM, Cartesian cube-sphere | Andrassy 18/22, Herwig, Woodward, Jones, Pathak 25, Ritter 18 | 4π full-star "Briquette" data format, GPU-friendly |
| Prometheus / Prometheus-Vertex | explicit compressible + neutrino transport | Yadav 20, Bollig 21, Lella 26 | Only code doing self-consistent progenitor→CCSN 3D |
| SLH | fully compressible, **low-Mach-corrected flux**, implicit option, well-balanced gravity | Horst 21, Leidi 23, Andrassy 22, Edelmann | The closest to "low-Mach" in shell-merger space. Still fully compressible. |
| MUSIC | time-implicit fully compressible | Viallet, Baraffe, Pratt, Andrassy 22 | Implicit to bypass acoustic CFL |
| FLASH | compressible AMR | Couch, Müller, Fields, Andrassy 22 | Used for CCSN + pre-SN |
| **any anelastic code** | — | **NOBODY in this corpus** | ASH, Rayleigh, MagIC exist in dynamo/solar-convection literature, but NOT used for pre-SN shell mergers |

The reason is physical: shell mergers operate at Ma ≳ 0.1, which is where the anelastic approximation's "filter acoustics, keep entropy waves" assumption breaks down. The community made an implicit collective decision ≥15 years ago to stay fully compressible (with low-Mach-corrected flux reconstructions where Ma is low enough).

## Observational state 2024-2026

The **observational** narrative has exploded: Cas A XRISM detection of P/Cl/K (XRISM 2025), Sato 2025 Ne-downflow / Si-upflow in Cas A, Matsunaga G359, Sato N49B, Kuboike J0550, Terano G284, Boccioli Ar/Ne ratio. These are strongly pushing theory: shell mergers are no longer a 1D-model curiosity — they leave X-ray fingerprints. This is the big change 2024-2026.

## The "missing paper" — what would a 2D anelastic-vs-compressible shell-merger benchmark add?

**Honest finding**: No paper in this corpus explicitly argues that "the anelastic approximation breaks down at Ma > X during shell merger" because **nobody uses anelastic for this problem anyway**. The question the literature is actively asking is:

1. **Does shell merger really happen in 3D?** (answered YES by Yadav 2020, Rizzuti 2024)
2. **What are the correct mixing profiles / entrainment rates / ingestion rates?** (Issa 2025×3 exposes 3-4 dex spread from 3D macro physics)
3. **What does a real 3D progenitor do to the CCSN explosion?** (Bollig 2021, Lella 2026)
4. **Can we observe it in SNR X-ray?** (Sato, Kuboike, XRISM, Matsunaga — yes)

The community's **stated open problems** (from Issa 2025 abstracts, Rizzuti 2024 conclusions, Georgy 2024 conclusions) are:
- Proper 3D mixing profile shape (downturn at boundary, velocity boost inside shell)
- Ingestion rate dependence on luminosity / Ri_B
- Reproducibility across codes
- How much to trust boosted-luminosity simulations

None of these directly calls for an anelastic benchmark. The Andrassy 2022 5-code test problem (which is the closest thing) explicitly designs the setup to be "accessible to explicit, implicit, and low-Mach codes" — and deliberately leaves anelastic off the list.
