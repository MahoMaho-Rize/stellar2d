# Route B Novelty Check — 2026-05-06

**Author**: Yujian Shi (w/ Claude Code)
**Mandate**: Verify Q5 of `route_B_survey.md` (make-or-break) against arxiv 2023-05 → 2026-05, ADS forward citations of Andrassy 2022 / Couston+ 2017 / Fuentes & Cumming 2020, the Dedalus publications page, and Google Scholar.
**Sources actually reached**: arxiv full-text search engine (multiple queries), Semantic Scholar citations API on `DOI:10.1051/0004-6361/202142557`, `dedalus-project.org/citations/`, Google Scholar constrained queries, full-text analysis of the one existential-threat candidate PDF.

---

## 1. TL;DR verdict: **PARTIALLY SURVIVED — the niche has narrowed but not collapsed**

A new 2025 paper — **Hindman & Fuentes, "Diffusive Braking of Penetrative Convection in Stably-Stratified Fluids", arXiv:2508.10174v2 (4 Nov 2025, submitted to PRF)** — occupies much of the methodological and framing ground we had planned to claim as Route B's contribution: **2D + 3D Boussinesq simulations in Dedalus v3 (pseudospectral), Chebyshev × Fourier, a Ri × Le parameter grid, an entrainment law Vcb(Ri, Le, γ, εT, εC) derived and then validated by h(t) ∝ t^{1/2} measurements in the simulations**. Their "Ri" is a **flux-Richardson number** `Ri = ρ₀cₚκT·N₀² / (gαF*)`, **not** Andrassy 2022's bulk Richardson `Ri_B = ΔB·L/v_rms²`, and they target **compositional** stratification (solute + temperature, Lewis-number driven) rather than Andrassy's **thermal/entropy** stratification. They explicitly do not cite Andrassy 2022 (A&A 659 A193, 5-code benchmark); their Andrassy reference [13] is the 2020 MNRAS C-ingestion paper. They also do not frame their work as a "reference solution" for compressible-code validation — it is a stand-alone diffusive-stalling physics paper.

**Consequence for Route B**: the unique thesis "**first 2D pseudo-spectral Boussinesq entrainment-law reference in Andrassy 2022's parameter regime**" is *not* already published — but we can no longer claim "first pseudo-spectral Boussinesq entrainment-law measurement in 2D+3D with modern spectral code". That phrasing is now Hindman & Fuentes 2025. **Route B must pivot to**: *the dissipation-free thermal Ri_B–E curve at Andrassy 2022's exact benchmark parameter point, framed explicitly as a validation reference for the five compressible codes (FLASH, MUSIC, PPMSTAR, PROMPI, SLH).* The pivot is tight, the contribution remains Letter-grade, but one full section of the paper must be rewritten to position against Hindman & Fuentes 2025 and explain the Ri_B-vs-Ri_flux distinction.

---

## 2. Per-check results

### §1. arxiv full-text search

Queries run on `https://arxiv.org/search/?searchtype=all&query=…`:

| Query | Relevant hits 2023-2026 |
|---|---|
| `pseudo-spectral Boussinesq entrainment stratified convection` | zero |
| `Boussinesq penetrative convection entrainment` | **1** — Hindman & Fuentes 2025, arXiv:2508.10174 ⚠️ |
| `bulk Richardson entrainment convection stellar` | zero post-2021 |
| `entrainment stratified convection Dedalus` | zero |
| `Boussinesq convection stable stratified 2D spectral` | zero |
| `Andrassy 2022 Richardson entrainment` | zero |
| `Fuentes Cumming entrainment penetrative` | error (303) |
| `Leidi Andrassy convection` | 3 — all compressible stellar code work, **none** Boussinesq spectral |
| `convective boundary mixing entrainment rate simulation` | 5 — 4 predate 2024 or irrelevant (Allende 2026 is ice-ocean); only Leidi 2023 and Mao 2023 |

**Threat assessment per hit**:

**(A1, red-flag) Hindman & Fuentes 2025, arXiv:2508.10174** — "Diffusive Braking of Penetrative Convection in Stably-Stratified Fluids" (B. W. Hindman, J. R. Fuentes).
- **Code**: Dedalus v3 pseudospectral, Chebyshev(z)×Fourier(x,y), RK443, 3/2 dealias.
- **Resolution**: 2D at 512×1024 spectral (1536×768 dealiased mesh); 3D at 384³ spectral (576³ dealiased).
- **Setup**: Cartesian 2H×H (2D) or H³ (3D), periodic horizontal, no-penetration stress-free top/bottom. Flux BC on T (fixed F* at bottom, zero at top), zero flux on composition C at bottom, linear gradient C₀ = Ri(1−z) initialised above. Convection grown from diffusive boundary layer; penetrates compositionally stratified layer.
- **Parameter grid**: 2D scan Ri ∈ {1, 3, 5, 7, 10} × Le ∈ {1, 2, 4, 6, 10} at fixed Ra = 10⁸, Pr = 0.1 (25 runs). 3D validation at Ri ∈ {1, 5, 10}, Le = 10.
- **Measured quantities**: interface height h(t), h ∝ t^{1/2} confirmed, Vcb(Ri, Le) fit against analytic law, stalled vs penetrative regime boundary mapped in Ri-Le plane.
- **Cites**: Andrassy+ 2020 MNRAS 491 (C-ingestion), **NOT** Andrassy+ 2022 A&A 659 A193 (5-code benchmark). Their Ri is the flux-Richardson `Ri = ρ₀cₚκT·N₀²/(gαF*)`, not the Andrassy bulk-Richardson `Ri_B = ΔB·L/v_rms²`.
- **Verdict**: **does not publish a Ri_B-E curve in Andrassy's parameter regime**, but does publish a spectral Boussinesq entrainment-rate scan. Substantial overlap on methodology; distinct on both the Ri definition and the target audience.

**(A2, low threat) Allende, Couston, Thalabard, Favier 2026, arXiv:2601.18674** — "Melting dynamics and mixing layer growth near the ice-ocean interface".
- 2D Boussinesq, super-diffusive mixing layer growth h ∝ t^{1.33}. Ice-ocean geophysical context; no stellar framing, no Richardson-entrainment reference claim.
- **Verdict**: **not a threat**. Different physics (phase change), different community.

**(A3, low threat) Leidi, Andrassy, Higl, Edelmann, Röpke 2023, arXiv:2309.17225** — "Turbulent dynamo action and its effects on the mixing at the convective boundary of an idealized oxygen-burning shell".
- 3D MHD compressible (SLH), not Boussinesq spectral. Reports ~20% reduction in entrainment rate from magnetic suppression, at a single parameter point.
- **Verdict**: **not a threat** — compressible, not a reference, not a Ri_B scan.

**(A4, low threat) Mao, Woodward, Herwig 2023, arXiv:2304.10470** — "3D Hydrodynamic Simulations of Massive Main-Sequence Stars. III".
- PPMstar compressible Eulerian. Reports entrainment rates at convective boundary, proposes 1D parameterization.
- **Verdict**: **not a threat** — explicitly compressible, doesn't attempt dissipation-free reference framing.

### §2. ADS / Semantic Scholar forward citations of Andrassy+ 2022

Semantic Scholar `DOI:10.1051/0004-6361/202142557/citations` returned **9 citing papers**. Full list:

1. **Ahlborn, Higl, Andrassy, Leidi, Weiss, Röpke 2026** (A&A) — Kuhfuß convection model validation vs 3D MHD simulations; **compressible**, not spectral, not Ri_B scan. No threat.
2. **Hoai, Nhung, Darriulat, Tan 2025** (ApJ) — AGB Mira light curve shape analysis; observational. No threat.
3. **De Marco, Aleman, Akras 2025** (planetary nebulae review) — review; no simulation. No threat.
4. **Boyd, Calder, Townsley, Zingale 2024** (ApJ) — 3D MAESTROeX convective Urca in a white dwarf; low-Mach compressible. No threat.
5. **Johnston, Michielsen, Anders … 2023** (ApJ) — 1D MESA stellar evolution with time-dependent convective penetration parameterization; **1D only**. No threat.
6. **Bowman 2023** (Ap&SS) — asteroseismology review. No threat.
7. **Bowman, van Saders, Vink 2023** (Galaxies editorial) — introduction. No threat.
8. **Lecoanet, Edelmann 2023** (Galaxies review) — core convection review; cites Andrassy 2022 but does not publish new simulations. No threat.
9. **Mao, Woodward, Herwig 2023** (ApJ) — see §1 entry A4.

**Nothing in the 9-paper forward citation list provides a pseudo-spectral Boussinesq Ri_B-E reference**. Hindman & Fuentes 2025 is **not** in this list because they cite Andrassy 2020 (MNRAS, C-ingestion) rather than Andrassy 2022 (A&A, 5-code) — significant because it means the threat was invisible to forward-citation search.

Couston+ 2017/2018 and Fuentes & Cumming 2020 forward citations: hit Semantic Scholar rate limit during the check; Fuentes-group recent work surfaced via author search alternative routes (see §3). Residual risk flagged in §5.

### §3. Dedalus publications / citations page

`https://dedalus-project.org/citations/` — scraped successfully. 2023–2026 stratified / convection publications listed (not exhaustive — page is manually curated and lags):

1. **Powers, Anders, Brown 2024** (PRF, DOI:10.1103/PhysRevFluids.9.043501) — internally heated fully compressible convection; Rayleigh-Bénard scaling laws. **No stratified stable layer**, no entrainment rate. No threat.
2. **Vasil, Lecoanet, Augustson … Burns … 2024** (Nature) — solar dynamo near surface; global solar, not stratified-convection benchmark. No threat.
3. **Barman, Sahoo 2024** (PoF) — rotating thermal convection with partial stratification; configuration differs substantially from Andrassy O-shell. No threat.
4. **Anders, Jermyn, Lecoanet, Brown 2023** (ApJ 926, 169 "Stellar Convective Penetration") — **3D Boussinesq Dedalus**, Penetration parameter 𝒫 and stiffness 𝒮 parameter-space scan for overshoot fractions. Closest conceptual competitor to Route B from Dedalus community. But: reports **overshoot fraction**, not Ri_B–E curve; stiffness parameter 𝒮 is **not** Andrassy bulk Richardson.
5. **Anders, Lecoanet, Cantiello … 2023** (Nature Astronomy) — photometric variability of massive stars from internal gravity waves; IGW-output focus, not entrainment.
6. **Leclerc, Laibe … 2022** (ApJ) — topological modes in stellar oscillations. Off-topic.
7. **Liu, Julien, Knobloch 2023** (JFM) — salt-finger convection staircases. Off-topic.

The curated Dedalus list **misses Hindman & Fuentes 2025** (published 2025, likely not yet added — the page is updated sporadically). Conclusion: **no Dedalus publication on the curated list provides a Ri_B-E entrainment reference for Andrassy 2022**, but the curated list is incomplete.

### §4. Google Scholar

Constrained queries `"Andrassy 2022" Richardson entrainment` (2023-2026) and `pseudo-spectral stratified entrainment reference` (2023-2026):

- `"Andrassy 2022" Richardson entrainment` — **zero hits** on Scholar for that exact full-text combination in 2023-2026. Consistent with the Semantic Scholar 9-citation list: the 5-code Andrassy paper is cited mostly by descendant compressible-code work, not by Boussinesq spectral work.
- `"bulk Richardson" entrainment spectral Boussinesq` — ~10 hits, all **geophysical / oceanographic** (Lefauve 2024 Comptes Rendus, Pham & Sarkar 2023 Env. Fluid Mech., Zhou & Taylor 2025 JFM, Grace et al. 2023 GRL, Huo et al. 2025 PoF, Rosevear et al. 2026 JGR), **none stellar**. These are tangential — same diagnostic (Ri_B) but atmospheric-boundary-layer or stratified-shear-layer applications, not convective-boundary-mixing for stellar benchmark.
- `pseudo-spectral stratified entrainment reference` — no Andrassy-regime hit; returned Zhang 2025 polito thesis on cloud mixing and Klemmer-Howland 2025 atmospheric LES — both off-topic.

### §5. Optional cross-check (Snoopy, SpectralDNS, PRF, JFM)

- **Snoopy** — no 2023-2026 stratified-convection benchmark publication surfaced. Snoopy remains MRI-shearing-box focused.
- **SpectralDNS / shenfun** — no entrainment benchmark publication surfaced 2023-2026.
- **PRF 2023-2026** — besides Hindman & Fuentes 2025 (submitted to PRF) and Dorel, Le Gal, Le Bars 2023 PRF 8, 103501 (experimental SF₆ gas penetrative convection, cited in Hindman-Fuentes), nothing stellar-facing.
- **JFM 2024-2026** — no Andrassy-regime pseudo-spectral Boussinesq benchmark surfaced. Zhou & Taylor 2025 (JFM) and Liu, Julien, Knobloch 2023 (JFM) both off-topic (shear-flow mixing, salt-finger staircases).

---

## 3. Closest-analog papers, ranked by threat level

| Rank | Paper | Threat level | Why |
|---|---|---|---|
| 1 | **Hindman & Fuentes 2025**, arXiv:2508.10174 | **HIGH — must cite and carefully distinguish** | Same stack (Dedalus v3 pseudospectral, 2D+3D Boussinesq, Ri × param scan, entrainment-law derivation + validation). Different: flux-Ri not bulk-Ri, composition-driven not thermal, no Andrassy 2022 framing, no "reference solution" claim. Route B must open its paper with "unlike Hindman & Fuentes 2025 who measure interface speeds in a composition-gradient setup with flux Richardson, we…" |
| 2 | **Anders, Jermyn, Lecoanet, Brown 2023** (ApJ 926, 169, arXiv:2110.11356) | MEDIUM | Boussinesq Dedalus with stiffness parameter 𝒮 scan (related but not identical to Ri_B). Reports penetration depth / overshoot fraction, not Ri_B–E curve. Same community as Hindman & Fuentes 2025. |
| 3 | **Fuentes & Cumming 2020** (PRF 5, 124501, arXiv:2008.03327) | MEDIUM (prior, not 2024+) | The foundation paper Hindman & Fuentes 2025 extends. Boussinesq Dedalus, depth-vs-stiffness at low Prandtl. Antecedent, must cite. |
| 4 | **Couston, Lecoanet, Favier, Le Bars 2017/2018** (PRF 2, 094804; JFM 854, R3) | MEDIUM (prior) | Dedalus penetrative convection with IGW focus. Conceptual parent of Anders 2023. |
| 5 | **Mao, Woodward, Herwig 2023** III, arXiv:2304.10470 | LOW | Compressible PPMstar MS stars; reports entrainment and 1D parameterization. Different code family, not dissipation-free. |
| 6 | **Leidi, Andrassy, Higl, Edelmann, Röpke 2023**, arXiv:2309.17225 | LOW | Compressible SLH MHD O-shell; single-point ~20% MHD entrainment reduction. Not a scan. |
| 7 | **Ahlborn, Higl, Andrassy, Leidi, Weiss, Röpke 2026** (A&A, 10.1051/0004-6361/202554956) | LOW | Validates Kuhfuß convection model with 3D compressible sims. Complementary (1D model validation), not a spectral reference. |
| 8 | **Dethero, Andrassy, Higl, Röpke 2024**, arXiv:2409.09815 | LOW (already anchored) | Filling-factor / penetration-parameter 2D-vs-3D comparison with MUSIC compressible. **Already cited as the 2D-vs-3D defensive anchor** in route_B_survey Q4. |
| 9 | **Allende, Couston, Thalabard, Favier 2026**, arXiv:2601.18674 | LOW | Ice-ocean Boussinesq mixing layer h∝t^{1.33}. Different community, different framing. |
| 10 | **Dorel, Le Gal, Le Bars 2023** (PRF 8, 103501) | LOW | Lab experiment, not simulation. Cited in Hindman-Fuentes. |

---

## 4. Recommended framing adjustment

Route B's single-sentence novelty claim must change from:

> **Old (pre-check)**: "first dissipation-free 2D pseudo-spectral Boussinesq Ri_B–E curve in the Andrassy 2022 parameter regime"

to:

> **New (post-check)**: "first dissipation-free Ri_B–E reference curve in the exact **thermally-stratified** Andrassy 2022 compressible-benchmark parameter regime, resolving the flux-Richardson vs bulk-Richardson gap left by Hindman & Fuentes 2025 and the stiffness-parameter vs Richardson-number gap left by Anders et al. 2023."

Concretely this means the Route B paper must:

1. Add a full §2 paragraph explicitly discussing Hindman & Fuentes 2025 and Anders et al. 2023, showing the **Ri_flux → Ri_B mapping** (these are related: the flux-Richardson controls input to a stationary-state Ri_B, but the measured Ri_B in a quasi-steady convection layer under flux forcing is *not* Ri_flux; the ratio depends on mixing efficiency γ).
2. Demonstrate the **thermal** Boussinesq variant (b-torque + N²w stratification, not β∆C composition) — this **is** the cheaper setup from route_B_survey Q8.5 option (1), so no code change, but the paper framing must make thermal/composition distinction explicit and explain *why Andrassy 2022 is thermal* (entropy-driven shell convection, Le → ∞ limit of Hindman-Fuentes).
3. Adopt Andrassy 2022's exact Ri_B definition (ΔB · L / v_rms² with L = convective-layer depth, ΔB = integrated interface buoyancy jump) rather than the flux-Richardson, and **publish a conversion table** so Hindman & Fuentes 2025 results can be placed on our plot for comparison.
4. Frame as a **companion to the Andrassy 2022 5-code benchmark**, not a standalone physics paper. This is the single most important framing difference from Hindman & Fuentes 2025, who are a standalone physics paper (diffusive stalling of convection). The Andrassy-benchmark-companion framing is defensible precisely because Hindman & Fuentes don't cite Andrassy 2022.
5. Target **A&A Letters** (or PRF Letter/Rapid Communication) rather than full paper — the "first dissipation-free reference for an established benchmark" framing is Letter-grade; a full entrainment-physics paper would now be a weaker me-too next to Hindman & Fuentes 2025.

**Code-scope implication**: unchanged. The `pseudo_spectral_bouss_solver` plan in route_B_survey Q8 (~2100 LOC, ~2 months) still stands. What changes is the **target parameter-point recipe** (thermal N² stratification, Andrassy's Ri_B definition as diagnostic) and the **positioning in §1-§2 of the paper**. No code rewrite.

**Timeline implication**: add ~1 week for the Hindman-Fuentes 2025 detailed methodology comparison, parameter-point mapping, and framing pivot. Total timeline still ≤ 2.5 months to first draft.

---

## 5. Residual uncertainty

1. **Semantic Scholar citation API rate-limited** (HTTP 429) mid-investigation — I only extracted the first 9 forward citations of Andrassy 2022. S2 may have more indexed but uncached, and may miss citations that reference the paper only in an arXiv version. **Action to close**: re-query with an API key, ideally via ADS `refcode=2022A%26A...659A.193A` citations list manually. Risk: a citing paper from 2025-26 with an Andrassy 2022 reference could still be hiding. Given that Hindman & Fuentes 2025 **do not** cite Andrassy 2022, it's plausible other spectral-Boussinesq groups also don't — so forward-citation search has a genuine blind spot.
2. **ADS full citation list** behind JS render — not scraped. Manual check of `ui.adsabs.harvard.edu/abs/2022A&A...659A.193A/citations` recommended.
3. **Unpublished / in-preparation work**: Fuentes is prolific and has an active collaboration with Anders, Cumming, Hindman, Brown. A follow-up to Hindman & Fuentes 2025 extending to **thermal** stratification (Le → ∞ limit) and/or an Andrassy-2022-framed run is plausible within 2026. **Monitoring recommendation**: arxiv daily listing in physics.flu-dyn and astro-ph.SR for Fuentes, Hindman, Anders, Lecoanet through the end of 2026.
4. **Thesis-level work** (Zhang 2025 polito thesis on stratified cloud mixing surfaced in Scholar) — low probability of being a direct threat but not impossible to contain a relevant measurement.
5. **Non-arXiv venues**: PRF / JFM / ApJ accepted-but-not-yet-arXiv papers fall through this search. Usual arxiv lag ~2-3 weeks is mostly fine; bigger concern is papers that don't go on arxiv at all (rare in this community).
6. **Geophysical-fluid-dynamics crossover**: Lefauve 2024 (Comptes Rendus, geophysical stratified turbulence review), Zhou & Taylor 2025 JFM (Lagrangian view of shear-layer mixing) — conceptually adjacent but do not target stellar benchmark. If a reviewer from that community is assigned, they may question our reference-solution framing relative to the extensive oceanographic pseudo-spectral literature. Counter: we can cite Lefauve 2024 as "this is the stellar-convection analog of a well-established geophysical methodology".
7. **Cannot verify paper text of**: Anders et al. 2023 ApJ 926, 169 at full-text level in this session (only abstract fetched). If they already did a Ri_B-like plot in an appendix, threat #2 rises. Recommend downloading and reading arXiv:2110.11356 before starting Route B code.
8. **Dedalus citations page is curated**, not a database. It misses Hindman & Fuentes 2025. It likely misses additional recent papers. Should not be treated as exhaustive.

---

## Bottom line

**Route B survives, but with a narrower claim and a mandatory new framing section.** The dominant threat is **Hindman & Fuentes 2025 (arXiv:2508.10174)** — pseudospectral Dedalus, 2D+3D Boussinesq, Ri×Le parameter grid, entrainment-law derivation+validation — but they target diffusive stalling in composition-gradient setups with flux-Richardson, explicitly do not cite Andrassy 2022 (A&A 659 A193), and do not frame their work as a 5-code-benchmark reference. That gap is still ours. **Do not pivot to a different project.** Pivot the framing, cite Hindman & Fuentes 2025 as closest analog in §2, emphasize thermal stratification + bulk-Richardson + Andrassy-benchmark-companion positioning, and proceed with the Route B code plan.

**Required reading before starting code**: download and read full text of
- Hindman & Fuentes 2025, arXiv:2508.10174 (downloaded to `literature_2024-2026/pdfs/hindman_fuentes2025_diffusive_braking.pdf`, full text at `txt/hindman_fuentes2025_diffusive_braking.txt`).
- Anders, Jermyn, Lecoanet, Brown 2023, arXiv:2110.11356 (not yet downloaded — should be done before Route B code starts).
- Fuentes & Cumming 2020, arXiv:2008.03327 (foundation paper; download recommended).

**Action taken**: `hindman_fuentes2025_diffusive_braking.pdf` added to `docs/projects/shell_merger/literature_2024-2026/pdfs/`; text extract at `docs/projects/shell_merger/literature_2024-2026/txt/hindman_fuentes2025_diffusive_braking.txt`.

---

## §6. Anders+ 2023 Residual Check (2026-05-06)

Closes §5 residual item #7. PDF downloaded to `literature_2024-2026/pdfs/anders2023_penetration.pdf`, text extract at `literature_2024-2026/txt/anders2023_penetration.txt` (2955 lines, 2.92 MB PDF).

**Paper details**
- Anders, E. H.; Jermyn, A. S.; Lecoanet, D.; Brown, B. P. (2022) "Stellar convective penetration: parameterized theory and dynamical simulations." ApJ 926, 169, arXiv:2110.11356v2 (11 Mar 2022). Received 28 Jul 2021, accepted 2022. Appendix structure: A = Accelerated Evolution technique, B = MESA Implementation, C = Table of Simulation Parameters. **No appendix D, no supplementary materials.**

**What they actually did.** 3D Boussinesq convection in Dedalus with internal heating and a height-dependent radiative conductivity. Equations: incompressible u, temperature with ∇ad source + radiative flux divergence (Eqns. 1-4). Two control-parameter knobs are varied independently: the **penetration parameter** 𝒫 = −(k_CZ Δ∇_CZ)/(k_RZ Δ∇_RZ) (flux-ratio style, Eqn. 15) and the **dynamical stiffness** 𝒮 ≡ N²/f_conv² (Eqn. 27). Extra knobs: freefall Reynolds R ∈ [200, 6400], Prandtl Pr = 0.5 (a single value; see §7 discussion), µ = 10⁻³ (fraction of flux entering through bottom). Parameter scan: 𝒫 ∈ [0.1, 16], 𝒮 ∈ [10², 10⁴] at fixed R = 400 (Case I), plus R-scan at fixed 𝒫 = 4, 𝒮 = 10³. Diagnostics measured: penetration depth δ_p (via three thresholds δ_0.1, δ_0.5, δ_0.9 on the ∇/∇ad profile), dissipation fraction f (fraction of buoyant work dissipated in CZ, Eqn. 11), and ξ (PZ-to-CZ dissipation ratio, Eqn. 13). Outputs: penetration depth δ_p(𝒫, 𝒮, R) saturation curves, plus MESA Sun proof-of-concept with the P-parameterization implemented.

**Does their work contain a Ri_B–E reference curve? NO.** Hard evidence:
1. **No entrainment rate E anywhere.** Paper explicitly separates "penetration" from "entrainment" and rejects the entrainment framing for their problem (line 1366: "not the result of entrainment-induced changes in the Schwarzschild boundaries"). They measure **equilibrium** penetration height δ_p, not a time-derivative h(t) → E.
2. **No bulk Richardson number.** Their 𝒮 ≡ N²/f_conv² is a *frequency-ratio* stiffness, related to but not the same as Andrassy's Ri_B = ΔB·L/v_rms² (which integrates an interface buoyancy jump over a convective layer depth). No conversion table published. Line 322-324 explicitly states: *"a description of the size of a theoretical PZ does not depend on the often-considered stiffness"* — they are arguing δ_p is **independent** of stiffness, the opposite of a Ri-E scaling claim.
3. **§5.4 "Dependence on S" (Fig. 8) is the closest thing**, and it reports δ_0.5 ≈ constant over 𝒮 ∈ [10², 10⁴], with only the PZ-RZ transition layer width δ_0.9 − δ_0.1 scaling as S^{−1/2}. This is a saturation-height vs stiffness plot at one 𝒫 = 4 point, not an entrainment-rate-vs-Ri curve.
4. No dissipation-free framing ("dissipation-free reference curve" is not a concept in the paper; in fact f ∈ [0.6, 0.9] is one of the *primary* measured outputs and their theory explicitly contains a dissipation term).
5. **Does not cite Andrassy 2022** (A&A 659 A193, 5-code benchmark) — not in references. The only Andrassy citations in their bibliography are Horst et al. 2021 (co-authored by Andrassy) and Jones et al. 2017. Same blind spot as Hindman & Fuentes 2025.

**Verdict: Route B SURVIVES.** Anders+ 2023 is a penetration-depth/dissipation paper, not an entrainment-rate paper. Their stiffness 𝒮 is a frequency-ratio quantity that maps to Ri-like concepts only indirectly; their central claim is that δ_p is **independent** of 𝒮, not that it scales with it. They do not compute or plot entrainment rate E at any stratification, do not use Andrassy 2022's Ri_B definition, do not target Andrassy 2022's O-shell benchmark parameter regime (their P-regime is core-convection-massive-star, Pr = 0.5, not stratified O-shell; they explicitly flag Pr ≪ 1 stellar reality as future work in §7). The specific make-or-break deliverable for Route B — **a Ri_B–E(Ri_B) curve at thermally-stratified Andrassy 2022 parameter point** — is untouched by this paper.

**Positioning relative to Route B.** Anders+ 2023 becomes a **secondary competitor / prior-art citation** (Rank 2 in §3's table, unchanged). The primary competitor remains Hindman & Fuentes 2025 (flux-Ri, composition-gradient). Anders+ 2023 should be cited in Route B's §2 "Related Work" for: (i) the Boussinesq-Dedalus infrastructure they established (Route B reuses their code stack); (ii) the distinction between penetration (∇ = ∇_ad beyond Schwarzschild) and entrainment (boundary-erosion) that Route B needs to invoke when explaining why the thermal Ri_B-E curve is a **separate** diagnostic from their 𝒫-based δ_p curve; (iii) the lesson that convective-boundary experiments take thousands of τ_ff to reach equilibrium — Route B's time-budget plan must absorb this finding (also true for Hindman-Fuentes h ∝ t^{1/2} growth). **Framing-wise no change**: Route B's pivot from §4 remains correct, now with one additional cite-and-distinguish paragraph for Anders+ 2023.

**Residual item #7 CLOSED.** Remaining §5 residuals #1, #2, #3, #5 still open but low-probability.
