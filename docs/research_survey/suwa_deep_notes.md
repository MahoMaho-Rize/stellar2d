# Suwa-group deep method notes

Three papers from Yudai Suwa (U-Tokyo / YITP) and collaborators, deep-read
for numerical-method transferability onto `stellar2d` (1D Lagrangian
implicit hydro + 2D ALE + 2D pseudo-spectral + 2D anelastic g-mode EVP;
**no neutrino transport, no MHD**).

---

## 1. Mori, Suwa, Takiwaki (2023) — `2302.00292`
**"Long-term GW asteroseismology of supernova: core collapse to 20 s postbounce"**

- **Numerical method**: 1D spherically symmetric, Eulerian, **general relativistic**
  neutrino-radiation hydrodynamics. Code is the public `GR1D` (O'Connor & Ott
  2010; O'Connor 2015) extended with M1 multi-group neutrino transport (Boltzmann
  to first 2 moments + analytic closure). Metric `ds² = -α²dt² + X²dr² + r²dΩ²`
  with `α=exp(Φ)` and `X = √(1 - 2m/r)` (Eqs. 1-3).
- **Grid / BC**: 1D radial, inner BC at `r=0`, outer BC at matching-to-Schwarzschild
  surface. Neutrino transport is solved out to 600 km; neutrino info read out
  at 500 km. Hydro variables evolved: conserved `U = [D, D Y_e, S^r, τ]`
  (Eqs. 8-12), RHS = Eq. 14 with source + neutrino sink terms `Q^{ν,E}`, `Q^{ν,M}`.
- **Time integration**: explicit hydro (not explicitly stated; GR1D uses HLLE +
  method-of-lines RK). Runs self-consistently from **collapse through bounce to
  20 s postbounce** — longest GW-asteroseismology run to date.
- **Microphysics**: nuclear EOS (standard GR1D table), `NuLib` opacities (18
  log-spaced energy groups 2 MeV to 280 MeV), reactions listed Table II incl.
  `ν_e+n↔p+e⁻`, pair, bremsstrahlung `N+N→N+N+ν+ν̄`, coherent scattering on
  (A,Z), neutrino-electron inelastic scattering.
- **IC**: progenitor is **9.6 M☉ zero-metallicity** (A. Heger 2016 priv. comm.) —
  a low-mass explosion progenitor known to blow up even in 1D.
- **Main diagnostic**: PNS eigenmode frequencies `f_g`, `f_f`, `f_p` vs
  postbounce time, fit with three formulas — compactness `M/R`, surface gravity
  `M/R²`, average density `√(M/R³)` — and a new time-ramp form
  `f(x=t_pb) = a₁ x^{a₄}/(x^{a₄} + a₂) + a₃` (Eq. 23). PNS gravitational mass
  drops to 1.26 M☉, radius shrinks from 100 km to 13 km.
- **Key equations**: metric Eqs. 1-3, hydro flux Eqs. 7-14, GR eigenmode EVP
  Eqs. 15-16 with relativistic Lamb `L²` Eq. 17, Brunt `N²` Eq. 18,
  Schwarzschild discriminant Eq. 20, metric perturbations Eqs. 21-22.
- **What a no-ν-transport code could reproduce**: the **asteroseismology
  post-processing** (GREAT EVP solver on a given ρ(r), Φ(r), Y_e(r) snapshot).
  Feed a pre-built PNS profile (from GR1D output or a toy cooling analytic)
  into a Newtonian Cowling-approximation variant of Eqs. 15-16 — our
  `anelastic_sl_solver` spectral EVP already solves this form for g/f/p modes.
  The 20-s-evolved PNS background is the bottleneck, not the eigenmode.

---

## 2. Shinoda, Suwa et al. (2025) — `2512.11404`
**"Impacts of hydrogen envelope on supernova fallback and the resulting compact remnant masses"**

- **Numerical method**: 1D spherically symmetric, **Eulerian**, explicit finite
  volume. Uses **public `Athena++`** (Stone+20) with Euler integrator `rk1`
  (VL2 tested for robustness), PLM reconstruction (`xorder=2`), **HLLC**
  Riemann solver. No MHD, no neutrino transport.
- **Grid / BC**: 1D spherical, `r ∈ [0, 4.85×10¹⁶ cm]` on **logarithmic mesh**
  N_r = 1000 with Δr_min = 1.4×10⁷ cm. Run duration 10⁶ s. Inner zone replaced
  by a **softened point mass** (core softening method; cubic-spline Price-Monaghan
  2007 potential, softening radius r_s = 3×10⁸ cm ≈ 1.5 M☉) to avoid artificial
  reverse shocks from a hollow sink BC.
- **Time integration**: explicit RK1 / VL2 operator split via Athena++ stock
  integrator.
- **Microphysics**: `γ = 5/3` ideal-gas EOS (note: radiation-pressure fraction
  ~ 0.3 but they ignore it for simplicity), self-gravity via enclosed-mass
  integral `g = -GM_r/r² - ∇φ_pt` (spherical shell), **Q_cool** artificial
  cooling term Eq. 11 with τ_cool = 30 s to suppress thermal-bomb entropy
  contamination of the softened core. **No nuclear network**. Explosion driven
  by **thermal bomb** `E_inj ∈ [10⁴⁸, 10⁵²] erg` injected into innermost 10
  cells at t=0.
- **IC**: Woosley+02 metal-poor solar-metallicity progenitors with
  `M_ZAMS ∈ {18, 20, 24, 28} M☉` at **Z = 10⁻⁴ Z☉**. Two variants per
  progenitor: with H envelope (SNII) and with envelope stripped (SESN) —
  stripped via Matzner-McKee 1999 isentropic helium-core + wind profile
  (Eq. 4).
- **Main diagnostic**: final **remnant mass M_C(E_exp)** and the universal
  transition at `E_exp ≈ 2-3 × E_grav,Hyd` above which reverse shock escapes.
  Fits M_C with Eq. 18 (`f(x) = w/(1+(x/x₁)^{k₁}) + (1-w)/(1+(x/x₂)^{k₂})`,
  parameters given in caption of Fig. 11).
- **Key equations**: hydro Eqs. 1-3, core-softening HSE Eqs. 5-9, envelope
  profile Eq. 4, accretion rate `Ṁ = -4π v_min r²_vmin ρ` Eq. 13.
- **What a no-ν-transport code could reproduce**: **all of it**. This is
  plain ideal-gas hydro + thermal bomb + self-gravity. `stellar2d`'s
  `radial1d` (1D Lagrangian) or `cart_ale2` (if made 1D-spherical) can do
  this directly — swap Helm EOS for γ=5/3, add the Q_cool term Eq. 11, add
  the softened-point-mass potential. The only "missing" piece vs their
  paper is the Woosley+02 progenitor profile, which is a one-time IC load.

---

## 3. Sawada, Suwa (2023) — `2301.03610`
**"Updating the ⁵⁶Ni Problem in Core-collapse Supernova Explosion"**

- **Numerical method**: **1D Lagrangian** spherically symmetric Newtonian
  hydro using `blcode` (prototype of `SNEC`, Morozova+15, available at
  stellarcollapse.org). Explicit artificial-viscosity (Von Neumann-Richtmyer
  1950). Post-processing with `torch` 640-isotope reaction network
  (Timmes 1999).
- **Grid / BC**: 1500 equal-mass Lagrangian shells with mass resolution
  10⁻³ M☉; domain covers `1.5 M☉` outward (i.e., the inner boundary is at
  `M_{s/k_B=4} - 0.5 M☉`). Inner BC: innermost mass shell forbidden from
  shrinking below 50 km (mimics a PNS). Material inside r<200 km flagged as
  "ν-driven wind" and excluded from ejecta.
- **Time integration**: Lagrangian explicit (blcode standard).
- **Microphysics**: **Helmholtz EOS** (Timmes & Swesty 2000) — degenerate
  relativistic e⁻/e⁺, radiation, Boltzmann ions. **Light-bulb neutrino
  heating/cooling** in post-shock region only:
  H = 1.554×10²⁰ (L_ν/10⁵²)(r_ν/100km)⁻²(T_ν/4MeV)² erg/g/s (Eq. 4),
  C = 1.399×10²⁰ (T/2MeV)⁶ erg/g/s (Eq. 5), T_ν fixed at 4 MeV.
  **No transport**. ⁵⁶Ni yield computed by post-process 640-isotope network
  on Lagrangian trajectories.
- **IC**: Sukhbold+18 non-rotating solar-metallicity progenitors,
  `M_ZAMS ∈ {12.3, 16.0, 18.0, 19.5} M☉`, at iron-core-collapse stage.
- **Main diagnostic**: **M(⁵⁶Ni) as a function of explosion-energy growth
  rate Ė_expl**. Key result: outermost ⁵⁶Ni mass coordinate is **insensitive**
  to Ė_expl (~1.65 M☉ robustly where T₉ = 5 is reached), but the **innermost
  ejectable mass coordinate depends on Ė_expl** — slow explosions strand
  inner Ni on the PNS. Fireball relation
  `E_expl = (4π/3)r_sh³ a T⁴ · f(T₉)` Eq. 8 predicts r_{T9=5} ≈
  3.6×10⁸ (E_expl/10⁵¹)^{1/3} cm (Eq. 7). Conclusion: modern slow
  explosions (Ė ≲ 1 Bethe/s) reproduce Type II median M_Ni ≈ 0.03 M☉ but
  **cannot** reproduce stripped-envelope SNe median 0.07 M☉.
- **Key equations**: Lagrangian hydro Eqs. 1-3, heating/cooling Eqs. 4-5,
  fireball E-r-T relation Eqs. 6-8.
- **What a no-ν-transport code could reproduce**: **essentially all of it**.
  This is exactly the niche `radial1d` occupies — 1D Lagrangian implicit
  hydro with Helm EOS. Implementing this paper in `stellar2d` requires:
  (i) load a Sukhbold+18 profile (already compatible, we load MESA); (ii)
  add the light-bulb source terms Eqs. 4-5 (trivial — two lines of kernel);
  (iii) add artificial viscosity or use our JFNK implicit scheme; (iv)
  post-process with a torch-equivalent network (external tool). No new
  solver infrastructure. **Direct replication target**.

---

## Transferability summary for stellar2d

| Paper | No-ν path viable? | Best stellar2d solver | Missing pieces |
|---|---|---|---|
| Mori+23 | Partial (asteroseismology only, not bg) | `anelastic_sl_solver` (EVP) | GR background profile |
| Shinoda+25 | **Yes, fully** | `radial1d` or 1D `cart_ale2` | Q_cool term, softened point mass |
| Sawada+23 | **Yes, fully** | `radial1d` | light-bulb heating/cooling source term, 640-isotope post-proc |

Shinoda+25 and Sawada+23 are both **direct replication targets** for a
code with 1D Lagrangian Newtonian hydro + ideal-gas/Helm EOS + a
prescribed heating/cooling source term. Neither needs neutrino transport.
Mori+23 requires a GR ν-transport code upstream (GR1D-class); only the
GREAT-style eigenmode solver is transferable.
