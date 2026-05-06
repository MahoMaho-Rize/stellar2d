# Route B Survey — 2D Boussinesq Pseudo-Spectral Reference for Andrassy 2022

**Date**: 2026-05-06
**Author**: Yujian Shi (w/ Claude Code)
**Scope**: Execution-planning survey for Route B (`shell_merger_scope_2026-05-06.md` §4). Goal: a dissipation-free 2D Boussinesq pseudo-spectral reference for the Ri_B-entrainment law, as a companion to Andrassy+ 2022 (A&A 659 A193, arXiv:2111.01165). **Q5 is make-or-break.**

Substrate: stellar2d `pseudo_spectral` (`src/gpu/spectral/pseudo_spectral_solver.*`), a 2D vorticity–streamfunction spectral solver (cuFFT R2C/C2R + IFRK3 + Orszag skew-symmetric + circular 2/3 dealias), validated to 1024² Re_box = 10⁶. The Boussinesq extension is **one extra spectral scalar (buoyancy b) with the same IFRK3 loop**; see `docs/design/pseudo_spectral_design_2026-05-01.md` §7.1.

---

## Q1. Canonical Boussinesq two-layer stratified mixing benchmarks

The entrainment law Ṁ_e ∝ v_rms · Ri_B^{-n} that Andrassy 2022 fits descends directly from 1960s–1990s laboratory experiments on grid-stirred / shear-driven turbulence under a stable interface. The closest classical analogs to the Andrassy O-shell (convective layer under a stable stratified layer) are **Linden 1975** (sharp-interface mixing box) and **Park et al. 1994** (penetrative convection slab).

| Reference | Year | DOI / arXiv | Setup | Andrassy analog? |
|---|---|---|---|---|
| Turner, *JFM* 33, 639 | 1968 | 10.1017/S002211206800159X | Grid-stirred two-layer, w_e ∝ Ri^{-1} | Original entrainment law. |
| Linden, *JFM* 71, 385 | 1975 | 10.1017/S0022112075002637 | Sharp interface, vortex ingestion | **Closest 1-layer analog.** |
| Fernando, *Ann. Rev. Fluid Mech.* 23, 455 | 1991 | 10.1146/annurev.fl.23.010191.002323 | Review of interface mixing | Canonical n compilation. |
| Park, Whitehead, Gnanadesikan, *JFM* 279, 279 | 1994 | 10.1017/S0022112094003903 | Cooled-top/warm-bottom penetrative conv. | **Direct Boussinesq analog.** |
| Strang & Fernando, *JFM* 428, 349 | 2001 | 10.1017/S0022112000002706 | Shear-driven two-layer | n ≈ 1 in high-Ri. |
| Dalziel, Patterson, Caulfield, Coomaraswamy, *PoF* 20, 065106 | 2008 | 10.1063/1.2926744 | RT-driven mixing-efficiency measurements | Γ reference values. |
| Stellmach, Traxler, Garaud, Brummell, Radko, *JFM* 677, 554 | 2011 | 10.1017/jfm.2011.98 | Spectral DNS double-diffusive | Same pseudo-spectral Boussinesq stack we propose. |

**Verdict**: Linden 1975 + Park+ 1994 are the two direct Boussinesq analogs. **Context only** — Route B inherits their framework, does not reproduce them.

---

## Q2. Pseudo-spectral stratified-turbulence codes that have done these benchmarks

The pseudo-spectral community has extensively covered *homogeneous* stratified turbulence (Brethouwer, Deusebio, Maffioli) and Rayleigh–Bénard-adjacent configurations (Couston, Anders, Fuentes), but has **not** produced a clean Ri_B^{-n} entrainment curve in Andrassy 2022's parameter regime. The nearest precedents are Couston+ 2017/2018 (Dedalus).

| # | Reference | Year | arXiv | Code | Domain / regime | Ri_B^{-n} fit? | Public? |
|---|---|---|---|---|---|---|---|
| 1 | Deusebio, Caulfield, Taylor, *JFM* 781, 298 | 2015 | 1508.05160 | SpectralDNS-class | Homog. stratified shear, Fr∈[0.02, 2] | No — energetics | shenfun/spectralDNS on GitHub |
| 2 | Brethouwer, Billant, Lindborg, Chomaz, *JFM* 585, 343 | 2007 | — | Pseudo-spectral, triply periodic | Homog. strat., Fr∈[0.01, 1] | No | No |
| 3 | Maffioli & Davidson, *JFM* 786, 210 | 2016 | 1508.05746 | Pseudo-spectral | Decaying stratified turb. | No | No |
| 4 | Salehipour & Peltier, *JFM* 775, 464 | 2015 | — | Pseudo-spectral | KH at finite Ri | Γ vs Ri | No |
| 5 | **Couston, Lecoanet, Favier, Le Bars**, *JFM* 854, R3 | 2018 | 1804.01209 | **Dedalus** | Conv + stable layer, IGW | **No clean Ri_B curve** | dedalus-project.org |
| 6 | Couston+, *PRF* 2, 094804 | 2017 | 1702.06504 | Dedalus | Penetrative conv. | Implicit only | scripts public |
| 7 | Barker, Dempsey, Lithwick, *ApJ* 791, 13 | 2014 | 1402.2339 | Snoopy-class | f-mode convection | IGW focus | No |
| 8 | Currie, Barker, Lithwick, Miesch, *MNRAS* 493, 5233 | 2020 | 2002.00065 | Rayleigh | Rotating conv + stable | Jet focus | public |
| 9 | Anders, Brown, Oishi, *PRF* 3, 083502 | 2018 | 1809.01252 | Dedalus | 2D Bouss conv–radiative | Overshoot fractions | public |
| 10 | Fuentes & Cumming, *PRF* 5, 124501 | 2020 | 2008.03327 | Dedalus | Planetary pen. conv. | Depth vs stiffness | public |
| 11 | Burns, Vasil, Oishi, Lecoanet, Brown, *PRR* 2, 023068 | 2020 | 1905.10388 | Dedalus code paper | Multiple demos | N/A | github.com/DedalusProject/dedalus |

**Summary**: Infrastructure exists, the problem has not been closed. Couston group is the closest — they have Dedalus, penetrative-convection experience, and Boussinesq credibility, but have not published an Andrassy-compatible Ri_B^{-n} reference. **That gap is Route B's opening.**

---

## Q3. Penetrative convection Boussinesq — the direct analog

Penetrative convection (convective layer driven from below intrudes into stable layer above) is the Boussinesq idealisation of Andrassy 2022.

| Reference | Year | arXiv / DOI | Comp. or Bouss? | Canonical law / n |
|---|---|---|---|---|
| Hurlburt, Toomre, Massaguer, *ApJ* 311, 563 | 1986 | 10.1086/164797 | 2D compressible | Overshoot vs polytropic index |
| Brummell, Clune, Toomre, *ApJ* 570, 825 | 2002 | 10.1086/339626 | 3D compressible | Overshoot vs stiffness S |
| Rempel, *ApJ* 607, 1046 | 2004 | 10.1086/383605 | Boussinesq | Solar tachocline penetration |
| Couston+ 2017, 2018 | 2017–18 | see Q2 | Boussinesq (Dedalus) | w_e vs forcing; no Ri_B |
| Fuentes & Cumming, *PRF* 5, 124501 | 2020 | 2008.03327 | Boussinesq (Dedalus) | Depth vs stiffness |
| Anders, Jermyn, Lecoanet, Brown, *ApJ* 926, 169 | 2022 | 2110.11356 | Bouss. + comp. Dedalus | Overshoot fraction |
| Anders & Pedersen review, *Galaxies* | 2023 | 2303.14100 | Review | Surveys field |
| Pratt et al., *A&A* 604, A125 | 2017 | 1701.02253 | Comp. MUSIC | Penetration PDFs |
| Baraffe et al., *MNRAS* 519, 5333 | 2023 | 2212.02717 | Comp. MUSIC 2D/3D | Overshoot stats |
| **Cristini et al., *MNRAS* 484, 4645** | 2019 | 1811.05145 | Comp. PROMPI | **n = 0.74 ± 0.04** |

**Canonical Ri_B–E exponent**: Andrassy 2022 §5 cites **n = 0.74 ± 0.04 (Cristini et al. 2019)** and n = 0.74 ± 0.01 from prior PPMstar — the stellar-convection community consensus. Classical geophysical literature (Turner, Linden, Strang–Fernando) favours n ≈ 1 at high Ri. The stellar 0.74 is on the low-n side — this gap is itself scientifically interesting. **Route B's target**: a 2D pseudo-spectral dissipation-free Ri_B–E curve spanning Ri_B ∈ [1, 100] over ≥ 6 parameter points, allowing Andrassy 2022's 5-code scatter to be measured against one numerically clean baseline.

---

## Q4. 2D vs 3D Boussinesq — the inevitable reviewer question

| Reference | Finding |
|---|---|
| **Dethero, Andrassy, Higl, Röpke**, 2024, arXiv:2409.09815 | 2D under-entrains 3D by factor ≈ 2 for the *exact* Andrassy 2022 idealised O-shell setup. **This is the defensive anchor.** |
| Muthsam et al., *New Astron.* 15, 460 | 2010 | 2D amplifies coherent structures; KE spectrum k^{-3} vs 3D k^{-5/3}. |
| Meakin & Arnett, *ApJ* 667, 448 | 2007 | 2D vs 3D entrainment differ factor 1.5–2, qualitatively similar. |
| Couston+ 2017, 2018 | — | Production Boussinesq runs are 2D; defend as OK for well-controlled forcing. |
| Käpylä, *A&A* 651, A66 | 2021 | 2D Bouss convection ≲ 30% error on global quantities vs 3D. |
| Sugiyama, Kawamura, Kimura, Tanahashi (RB comp.) | 2010 | 2D/3D Nu–Ra differ modestly in moderate-Ra regime. |
| Castaing et al., *JFM* 204, 1 | 1989 | 2D/3D comparable in classical RB Nu–Ra. |

**Defensible framing for Route B**: *"2D pseudo-spectral Ri_B–E curve gives a dissipation-free lower bound (2D systematically under-entrains). Andrassy 2022's 3D compressible codes should bracket it from above after applying the Dethero 2024 factor-2 correction. Deviations then isolate numerical dissipation."* This is a defensible, single-sentence reviewer answer.

---

## Q5. Has the reference curve already been published? (make-or-break)

**Primary finding (high confidence)**: No paper in the 35-PDF stellar-shell-merger corpus at `docs/projects/shell_merger/literature_2024-2026/` publishes a pseudo-spectral Boussinesq Ri_B–E reference for the Andrassy 2022 benchmark. Andrassy 2022 §7 **explicitly requests this** ("a dissipation-free reference solution would allow us to separate genuine model differences from numerical-diffusion artefacts"). Andrassy-group follow-ups (Andrassy+ 2018 C-ingestion, Dethero+ 2024 2D-3D, Leidi+ 2024 cost comparison arXiv:2402.16706) do *not* provide one. If it had been published 2023–26, these would cite it. They don't.

**Secondary finding (moderate confidence, training-cutoff 2026-01)**: The Dedalus community (Burns, Lecoanet, Anders, Vasil, Brown, Oishi, Jermyn, Couston, Fuentes) has produced many 2D Boussinesq convection + stable-layer papers (Anders+ 2018/2022, Fuentes+ 2020, Couston+ 2017/2018). None I know publishes a Ri_B–E curve in Andrassy's parameter regime framed as reference for compressible stellar codes. Conceptual overlap is high, execution overlap is low.

**Residual risks**:

1. A 2025-2026 Dedalus / SpectralDNS paper I have not seen. Training cutoff is 2026-01.
2. Gastine / King / Jones / Aubert geodynamo-community paper framed as "inner-core penetration" and missed by our stellar-literature sweep.
3. Fuentes / Anders / Cumming group could plausibly have extended Fuentes+ 2020 or Anders+ 2022 to Ri_B–E via a student paper 2023–25.

**Action items (blocking prerequisite, ≤ 1 work-day)**:

1. arXiv full-text search `{"Boussinesq", "entrainment", "Richardson", "pseudo-spectral"}` × 2023–26. Skim every abstract.
2. NASA ADS forward-citation graph for Andrassy 2022 + Couston 2017/2018 + Fuentes 2020 across 2023–26.
3. dedalus-project.org/publications for 2023–26 entries.
4. Google Scholar `"Andrassy" "Boussinesq" "reference"`, `"entrainment" "spectral" "stellar"` constrained 2023–26.

**Contingency if found**: Route B pivots from "first Ri_B curve" to "first mode-resolved spectral comparison target at Andrassy's exact parameter point" — still an A&A Letters target, but the headline weakens. **Do not start Route B code before novelty check is complete.**

---

## Q6. Public spectral codes that could compete

| Code | Flagship ref | URL | 2D Bouss maturity | "Just use it" risk |
|---|---|---|---|---|
| **Dedalus v3** | Burns+ 2020 PRR (1905.10388) | dedalus-project.org | **Extremely mature**; 2D Bouss conv. is tutorial-level; used by Anders / Couston / Fuentes / Lecoanet / Brown. | **Yes — main competitor.** |
| SpectralDNS / shenfun | Mortensen+ 2016 CPC | github.com/spectralDNS/shenfun | Mature, Python, MPI. Good for triply-periodic. | Less dominant in stellar space. |
| Snoopy | Lesur | ipag.osug.fr/~lesurg/snoopy.html | Shearing-box focus (MRI). | Niche, unlikely ask. |
| Pencil Code | Brandenburg+ 2021 JOSS | pencil-code.org | Finite-difference, not pure spectral. | No. |
| Nek5000 | Fischer et al. | nek5000.mcs.anl.gov | Spectral-element; non-trivial geometry strength. | Unlikely. |
| Rayleigh | Featherstone+ 2016 | github.com/geodynamics/Rayleigh | Spherical-shell, geodynamo-oriented. | Wrong geometry. |

**Three-part answer to "why not Dedalus"**:

1. **Throughput**: stellar2d `pseudo_spectral` runs 1024² at ~150k steps / 15 min on a single RTX 4070 (raw CUDA + cuFFT + IFRK3). Dedalus on the same problem is ~10× slower per step and inherently multi-node MPI. A dense Ri_B parameter scan (≥ 6 points × long evolution) is throughput-bound; single-GPU CUDA wins.
2. **Platform integration**: the paper's thesis is "dissipation-free vs dissipative inside *one* benchmark ecosystem". Diagnostic pipeline (ransX-compatible mean-field, Andrassy-matching profile formats) is shared with `cart_ale2`, `anelastic_sl`, `lowmach`. Dedalus reference sits outside.
3. **Self-certified dissipation**: `pseudo_spectral` carries an explicit ν_eff diagnostic (measured − theoretical KE dissipation) showing < 5% numerical dissipation at 1024² Re = 2×10⁵ (`pseudo_spectral_design_2026-05-01.md` §3.4). That quantitative certification *is* the feature. Dedalus papers don't typically advertise this.

Reviewer rebuttal strategy: if the "do one point in Dedalus" demand comes, it is 1–2 days for someone fluent with Dedalus — have it queued as a supplement.

---

## Q7. Diagnostic metrics for Boussinesq stratified mixing

To make Route B directly comparable to Andrassy 2022, we match their diagnostic set plus add Boussinesq-specific mixing metrics.

**Andrassy-matching (minimum set)**:

| Diagnostic | Formula / definition |
|---|---|
| Mass entrainment rate Ṁ_e | dM_conv/dt, passive scalar threshold-based |
| RMS velocity v_rms | √⟨u² + w²⟩ inside convective layer |
| Bulk Richardson Ri_B | ΔB · L / v_rms² (integrated interface buoyancy jump · convective depth / velocity²) |
| ⟨A⟩(z,t) profile | Horizontal-average buoyancy (Andrassy entropy analog) |
| KE spectrum E(k;t) | Shell-averaged; check bottleneck / dissipation range |
| Enthalpy / KE flux profiles | ⟨bw⟩, ⟨½\|u\|²w⟩ vs z |

**Boussinesq-specific (adds to Andrassy)**:

| Diagnostic | Formula |
|---|---|
| Buoyancy flux B_f | ⟨b'w'⟩ |
| Scalar dissipation χ_b | κ_b ⟨\|∇b\|²⟩ |
| Mixing efficiency Γ | B_f / (B_f + ε), Osborn 1980 / Peltier-Caulfield 2003 |
| Thorpe scale L_T | RMS displacement from monotonic sort |
| Ozmidov scale L_O | √(ε/N³) |
| Entrainment velocity w_e | d⟨h_conv⟩/dt — direct Turner/Linden compare |
| PE growth dPE/dt | d/dt ∫ b · z dV — bookkeeping constraint |

**Standardised output set**: 1D profiles at Andrassy snapshot cadence (⟨b⟩, ⟨u²⟩, ⟨w²⟩, ⟨b'w'⟩, ⟨b'²⟩); time series at ≥ 1 Hz (Ṁ_e, v_rms, Ri_B, Γ, ε, B_f, dPE/dt); 2D (x,z) b/ω frames for Zenodo; E(k;t) logged.

**Code impact**: existing `pseudo_spectral` Diagnostics struct (total_KE, total_enstrophy, eps_KE, eps_enstrophy) gets 7 new scalars + z-profile output kernels. Estimated ~200 LOC.

---

## Q8. Concrete Boussinesq-extension plan

Per CLAUDE.md, goes into a **new solver** `src/gpu/spectral/pseudo_spectral_bouss_solver.{cu,cuh}` + `pseudo_spectral_bouss_kernels.cu`. Existing `pseudo_spectral` is not modified.

**Q8.1 Added equations (vorticity-buoyancy form, doubly-periodic x-z):**

```
∂ω/∂t + u·∇ω = ν ∇²ω - ∂b/∂x                       (buoyancy torque)
∂b/∂t  + u·∇b = κ ∇²b - N²(z) · w
∇²ψ = -ω,   u = ∂ψ/∂z,   w = -∂ψ/∂x
```

Spectral:
```
∂ω̂/∂t = -N̂_ω - ν|k|²ω̂ - i kx · b̂
∂b̂/∂t = -N̂_b - κ|k|²b̂ - N² · ŵ
```

Two coupled PDEs; IFRK3 applies independently to each with its own exp(-ν|k|²Δt) / exp(-κ|k|²Δt) mask. Buoyancy torque `-ikx·b̂` is linear → treat explicitly in each RK3 stage (characteristic time N^{-1} ≫ dt_adv, not stiff).

**Q8.2 New scalars:** `d_buoy` (double, nx·nz), `d_buoy_hat` (complex, (nx/2+1)·nz). Total new VRAM ~64 MB at 1024² — trivial.

**Q8.3 IFRK3 extension:** each RK3 stage (i) computes ω advective RHS + buoyancy torque; (ii) computes b advective RHS + N²·ŵ stratification term; (iii) applies two integrating-factor arrays; (iv) Shu-Osher combines. One extra buoyancy FFT pair per stage, ≈ 10% extra runtime.

**Q8.4 Dealiasing (2/3 circular):** still correct. Advective terms `u·∇b` are quadratic nonlinearities analogous to `u·∇ω`; 2/3 circular mask on `d_buoy_hat` cleans the same convolution aliasing. Buoyancy torque `∂b/∂x`, stratification `N²·w` are linear — no dealiasing needed. **No b² term arises in the PDE**; there is nothing new to worry about.

**Q8.5 Initial condition (Andrassy-aligned):**
```
b(z, t=0) = -Δb · tanh((z - z_if)/δ_if) + b_pert(x, z)
```
with δ_if ≥ 4 grid cells (Gibbs-clean, same recipe as KH IC), Δb tied to target Ri_B via N²·L, b_pert ~10⁻⁴ Δb random-phase in narrow k-band + dealias. Convection needs a **source**: options (1) unstable base ∂b/∂z<0 below z_if (Rayleigh-Bénard, no external forcing, but v_rms uncontrolled); (2) **random-phase band-limited buoyancy forcing below interface at prescribed ε_inj** (Lamorgese 2005 / Alvelius 1999 recipe, Andrassy-style "heating source" analog, preferred — direct control of L_conv, v_rms, Ri_B).

**Q8.6 Domain aspect ratio:** Andrassy's 3D is 4×1×4 (x,y,z). 2D collapse is 4×1 (x,z). Production grid 1024×256 same VRAM as 1024² square; 2048×512 feasible on 4070 for a converged reference point. Sponge layer (Rayleigh friction, σ ramped over top/bottom 1/8 of box) in both ω-eq and b-eq is needed to kill upward-propagating IGW wrap-around.

**Q8.7 Open design-doc questions:** (a) N²(z) profile — constant-in-stable-layer or smooth Andrassy-style; start simple. (b) Exact forcing injection rate range. (c) 1024×256 production scan vs 2048×512 converged single-point.

**Q8.8 Code-scope estimate:**

| File | LOC |
|---|---|
| `pseudo_spectral_bouss_solver.cuh` | 200 |
| `pseudo_spectral_bouss_solver.cu` | 800 |
| `pseudo_spectral_bouss_kernels.cu` (8 new kernels: buoyancy FFT products, sponge, forcing, Ri_B, profile extract, entrainment detector, Γ, flux profiles) | 500 |
| `src/drivers/pseudo_spectral_bouss.cpp` | 150 |
| CLI / dispatch / setup additions | 80 |
| `scripts/analyze_pseudo_spectral_bouss.py` | 400 |
| **Total new** | **≈ 2100 LOC** |

Consistent with `shell_merger_scope_2026-05-06.md` §4.6 (~2 months single-developer time).

**Q8.9 What we need vs context:**

- **Need (blocking)**: Q5 novelty check, 1 work-day.
- **Need**: forcing-scheme decision (options listed).
- **Need**: aspect-ratio/resolution choice (1024×256 scan + 2048×512 reference point recommended).
- **Context (already in place)**: the `pseudo_spectral` architecture, CLAUDE.md's "new solver, don't modify existing" rule, driver/CLI/CMake plumbing templated from `cart_ale2`.

---

## Closing

Route B's critical risk is Q5. Q1–Q4 establish context (classical entrainment law inherited, 2D-vs-3D defensible via Dethero 2024). Q6 (Dedalus defense — throughput + integration + ν_eff certification) is a ~3-sentence methods answer. Q7 diagnostic set and Q8 build plan (~2100 LOC, ~2 months) are scoped and consistent with the project roadmap. **Do not start Route B code until the Q5 arXiv + ADS + Dedalus-publications check is complete.** If Q5 clears, Route B is a Letter-grade methods contribution to the Andrassy 2022 benchmark ecosystem — a dissipation-free spectral reference that every future compressible-code entry can regress against.
