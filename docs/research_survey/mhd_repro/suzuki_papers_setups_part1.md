# Suzuki-group MHD wind simulations — Reproduction-grade extraction (Part 1)

Two-paper setup dump for 1:1 reproduction by our in-house MHD solver.

- Paper 1: Suzuki, Ohnaka, Yasuda 2025, PASJ — "Red Giant Winds Driven by Alfvén Waves with Magnetic Diffusion" (arXiv:2501.00294v2)
- Paper 2: Shimizu, Shoda, Suzuki 2022, ApJ — "Role of Longitudinal Waves in Alfvén-wave-driven Solar Wind" (arXiv:2203.15280v1)

Both codes are 1D spherically-symmetric super-radially expanding open flux-tube MHD with turbulent Alfvén-wave dissipation closure (Shoda et al. 2018a formalism), Spitzer-Härm thermal conduction, and heating/cooling source terms. Paper 1 extends Paper 2's framework (via Matsuoka et al. 2024) to non-ideal MHD with ambipolar + Ohmic diffusion for weakly-ionized RGB atmospheres.

---

## Paper 1: 2501.00294 (Suzuki+Ohnaka+Yasuda 2025, α Boo / α Tau RGB winds)

### P1. Governing equations

Eqs. (10)–(14), 1D spherical with cross section `A(r) ∝ r² f(r)`:

- Continuity: `dρ/dt + (ρ/(r²f)) ∂/∂r (r² f v_r) = 0`
- Radial momentum: `ρ dv_r/dt = −∂p/∂r − (1/(8π r² f)) ∂/∂r (r² f B_⊥²) + (ρ v_⊥²/(2 r² f)) ∂/∂r (r² f) − ρ GM_*/r²`
- Transverse momentum (vector, two components): `d/dt (ρ √(r²f) v_⊥) = (B_r/(4π)) ∂/∂r (√(r²f) B_⊥) + ρ D_{v⊥}`
- Energy (total):  
  `ρ d/dt [ e + v²/2 + B²/(8πρ) − GM_*/r ] + (1/(r²f)) ∂/∂r [ r² f ((p + B²/8π) v_r − (B_r/4π)(B·v) + F_c ) ] = (1/(r²f)) ∂/∂r [ (η_tot/(4π)) √(r²f) B_⊥ ∂/∂r (√(r²f) B_⊥) ] − Q_rad`
- Transverse induction: `∂B_⊥/∂t = (1/√(r²f)) ∂/∂r [√(r²f) (v_⊥ B_r − v_r B_⊥)] + √(4πρ) D_{B⊥} + (1/√(r²f)) ∂/∂r [η_tot ∂/∂r (√(r²f) B_⊥)]`

- EOS: ideal, `e = p/(ρ(γ−1)) = k_B T/(μ m_u (γ−1))`, γ = 5/3. μ from charge-neutrality + Saha (eq. 25).
- Thermal conduction: Spitzer `F_c = −κ_0 T^{5/2} ∂T/∂r`, κ_0 = 10^{−6} erg cm^{-1} s^{-1} K^{-7/2} (Braginskii 1965).
- Turbulent dissipation closure (Shoda 2018a; eqs. 18–21):  
  `D_{v⊥_i} = −(c_d/(4 λ_⊥)) ( |z_⊥^+| z_⊥i^- + |z_⊥^-| z_⊥i^+ )`,  
  `D_{B⊥_i} = −(c_d/(4 λ_⊥)) ( |z_⊥^+| z_⊥i^- − |z_⊥^-| z_⊥i^+ )`,  
  with Elsässer `z_⊥^± = v_⊥ ∓ B_⊥/√(4πρ)`, `c_d = 0.1`,  
  `λ_⊥(r) = λ_⊥,0 √(r² f(r) / (R_*² f_0))`.

- Dissipation terms (Section 2.7):
  - Ohmic: `η_O ≈ 2.3×10² max(1−x_e, 0)/x_e · √T  [cm² s^{-1}]`  (eq. 26)
  - Ambipolar: `η_AD ≈ 2.1×10^{−16} (max(1−x_e,0)/x_e) · (B[G])²/(ρ[g cm^{-3}])  [cm² s^{-1}]` (eq. 27)
  - `η_tot = η_O + η_AD`
  - No explicit viscosity.

- Cooling `Q_rad`:
  - `T > 1.2×10⁴ K`: optically-thin Sutherland–Dopita 1993, `Q_rad = n_i n_e Γ(T, Z)`.
  - `T ≤ 1.2×10⁴ K`: empirical chromospheric Anderson & Athay 1989 with metallicity scaling and optical-thickness transition (eq. 17):  
    `Q_rad = 4.5×10⁹ (0.2 + 0.8 Z/Z_⊙) min(1, ρ/ρ_cr) erg cm^{-3} s^{-1}`, `ρ_cr = 10^{−16} g cm^{-3}`.
  - Cooling cut-off: `T ≤ T_cut = 0.7 T_eff` (3010 K α Boo, 2730 K α Tau) → Q_rad=0 (numerical stability).
- Ionization: custom Hartmann–Avrett H model; He from LTE Saha; each heavy element j (C, O, Na, Mg, Al, Si, S, K, Ca, Cr, Fe) singly-ionized with combined radiative (`W T_eff`-diluted + ISRF `W_gal=10^{-14}, T_gal=7500 K`) + thermal (LTE Saha) ionization, eq. (24).

### P2. Geometry + grid
- 1D spherical radial `r`, super-radial open flux tube, cross section `A(r) = r² f(r)`.
- Filling factor `f(r)` (eq. 7): `f(r) = [e^{(r−R_*−h)/ζ} + f_0 − (1−f_0) e^{−h/ζ}] / [e^{(r−R_*−h)/ζ} + 1]`, ζ = h/2.
- Scale height `h = (H/R_*) · c_s²/(R_* g)` (eq. 8). For α Boo `h/R_* = 0.55`, α Tau `h/R_* = 0.66`.
- B_r(r) from ∇·B = 0: `B_r(r) = B_{r,0} R_*² f_0 / (r² f(r))` (eq. 9).
- Inner boundary: `r_in = R_*` (photosphere, T = T_eff).
- Outer: `r_out = 31 R_*`.
- Grid: non-uniform radial. Fiducial α Boo: `Δr = 5×10^{-4} R_*` (= 8.8×10³ km) near surface, `Δr = 2×10^{-2} R_*` (= 3.5×10⁵ km) in outer region. Cell count `N = 1800` fiducial. Other runs `N ∈ {1080, 1800, 2500, 3200, 2800, 5300, 10000}` (Tables 3, 4); resolution chosen so that shortest Alfvén wavelength `v_A / ω_max` is resolved by ≥ 4 grid points everywhere.

### P3. Initial condition
- Not spelled out explicitly but implied: hydrostatic atmosphere at T=T_eff anchored on photospheric ρ_0, above which the code relaxes.
- Photospheric density (eq. 1): `ρ_0 ∝ g^{0.6} T_eff^{-2}` normalized to `ρ_{0,⊙} = 2.5×10^{-7} g cm^{-3}` from ATLAS (Kurucz 1979, Castelli+Kurucz 2003). Table 1 gives α Boo ρ_0 = 9.4×10^{-9} g cm^{-3}, α Tau ρ_0 = 6.0×10^{-9} g cm^{-3}.
- Stellar params (Table 1):
  - α Boo (K1.5 III): M_* = 1.0 M_⊙, R_* = 25.4 R_⊙, T_eff = 4300 K, L_* = 195 L_⊙, log g = 1.6.
  - α Tau (K5 III): M_* = 1.3 M_⊙, R_* = 45.2 R_⊙, T_eff = 3900 K, L_* = 418 L_⊙, log g = 1.2.
- Abundances (Table 2): α Boo metal-poor `Z = 0.3 Z_⊙` ([Fe/H] = −0.52); α Tau near-solar `Z = 0.74 Z_⊙` ([Fe/H] = −0.13). Per-element tables given.
- Isothermal sound speed c_{s,0} = 5.4 / 5.1 km s^{-1}; photospheric fluctuation velocity `δv_0` scales as `δv_0 ∝ (T_eff^4/ρ_0)^{1/3}` with solar norm `δv_{0,⊙} = 1.25 km s^{-1}` → α Boo δv_0 = 2.50 km s^{-1}, α Tau 2.56 km s^{-1}.
- Magnetic field norms: photospheric equipartition `B_{r,0}² / (8π) = p_0` (eq. 6), giving B_{r,0} = 262 G (α Boo) / 198 G (α Tau). Observed average `f_0 B_{r,0}` sets `f_0`: α Boo fiducial `f_0 B_{r,0} = 0.65 G` → f_0 = 1/403; α Tau `f_0 B_{r,0} = 0.25 G` → f_0 = 1/790. Additional cases: α Boo 0.37 G, 1.31 G; α Tau 0.50 G.
- Correlation length at photosphere: `λ_⊥,0 = 1.0×10⁶ km` (α Boo), `2.2×10⁶ km` (α Tau), scaled from solar `λ_⊥,0⊙ = 3×10³ km`.

### P4. Boundary conditions
- Inner (r = R_*): fixed T = T_eff. Power-law spectrum `P(ω) ∝ ω^{-1}` applied to all three components of velocity (longitudinal + 2 transverse) with equal amplitude `⟨δv_0²⟩ = ∫ P(ω) dω` between ω_min and ω_max. Frequency range `ω_max = 100 ω_min`, ω_max^{-1} scaled as c_{s,0} R_*²/M_* from solar norm ω_{max,⊙}^{-1} = 0.3 min. → α Boo ω_max^{-1} = 150 min, ω_min^{-1} = 1.5×10⁴ min. α Tau ω_max^{-1} = 340 min, ω_min^{-1} = 3.4×10⁴ min. The spectrum is a stochastic broadband driver (implicit in Matsuoka+2024 / Shimizu 2022 formalism).
- Outer (r = 31 R_*): outflow / free (zero-gradient); not detailed explicitly but implied.

### P5. Time integration + code
- 2nd-order MHD Godunov + Method-of-Characteristics (MoCCT-style) for transverse fields, following Sano, Inutsuka & Miyama 1999.
- Explicit time advance with CFL constraint (not stated numerically, standard MHD-Godunov CFL).
- η_O and η_AD evaluated each step from ionization balance (Section 2.6).
- Run duration `t_sim = 30 R_* / c_{s,0}` → 1135 days (≈3.1 yr) α Boo; 2144 days (≈5.9 yr) α Tau.
- Time averages in `[0.1 t_sim, t_sim]`.
- Code identity: unnamed private Suzuki-group code; direct descendant of Suzuki 2018 / Matsuoka+2024 / Shoda+2018a.

### P6. Parameter sweep

Table 3 — α Boo, 9 runs:

| f_0 B_{r,0} [G] | B-diffusion | Z [Z_⊙] | N |
|---|---|---|---|
| 0.65 | On | 0.3* | 1800 (fiducial) |
| 0.65 | Off | 0.3* | 3200 |
| 0.37 | On | 0.3* | 2500 |
| 1.31 | On | 0.3* | 1080 |
| 0.65 | On | 1 | 1800 |
| 0.65 | On | 10^{-1} | 1800 |
| 0.65 | On | 10^{-2} | 1800 |
| 0.65 | On | 10^{-3} | 1800 |
| 0.65 | On | 0 | 1800 |

Table 4 — α Tau, 3 runs: (0.25 G On 0.74 Z_⊙ N=5300), (0.25 G Off 0.74 Z_⊙ N=10000), (0.50 G On 0.74 Z_⊙ N=2800).

### P7. Key diagnostics

- Mass-loss rate (eq. 29): `Ṁ = 4π ρ v_r r² f`. `Ṁ_sim` is the spatial average over 28 R_* ≤ r ≤ 31 R_*, then time-averaged in [0.1 t_sim, t_sim].
- Alfvénic Poynting luminosity (eq. 31): `L_A = [ −B_r B_⊥ v_⊥ /(4π) + v_r (ρ v_⊥²/2 + B_⊥²/(4π)) ] · 4π r² f`.
- Density-weighted rms transverse velocity (eq. 32): `⟨√v_⊥²⟩ = √( ∫ dt ρ (v_⊥1² + v_⊥2²) / ∫ dt ρ )`.
- Density-weighted longitudinal fluctuation (eq. 33): `⟨√δv_r²⟩ = √( ∫ dt ρ (v_r − ⟨v_r⟩)² / ∫ dt ρ )`.
- Radiation luminosity in T-band (eq. 36): `L_rad(T_1 < T < T_2) = 4π ∫_{T_1<T<T_2} Q_rad r² dr`.
- Magnetic Reynolds number (eq. 28): `Rm_tot = Λ V / η_tot`, with Λ = H_0 = c_{s,0}²/g, V = c_{s,0}`.
- Reimers formula for reference (eq. 30): `Ṁ_Reimers = 4×10^{-13} (R_*/R_⊙)(L_*/L_⊙)(M_*/M_⊙)^{-1} M_⊙ yr^{-1}`.
- Dimensionless: `N_tubes` tube count = `16π R_*² f_0 / d_0²` with d_0 ∝ h ∝ c_s²/g; gives N_⊙ ≈ 1.5×10⁵, α Boo ≈ 2800, α Tau ≈ 950.

### P8. CRITICAL — numerical values to match

α Boo (Table 3):

| Case | ⟨Ṁ⟩ [M_⊙ yr^{-1}] | Ṁ_min, Ṁ_max | ⟨v_out⟩ [km s^{-1}] | ⟨L_{0.5MK+}⟩ [erg s^{-1}] | ⟨L_{20kK–0.5MK}⟩ [erg s^{-1}] |
|---|---|---|---|---|---|
| fid (0.65 G, On, 0.3 Z_⊙) | **3.3×10^{-11}** | 1.0×10^{-12}, 3.8×10^{-10} | **77** | 7.9×10²⁶ | 4.0×10²⁹ |
| (0.65 G, Off, 0.3 Z_⊙) | 5.1×10^{-10} | 9.3×10^{-12}, 4.5×10^{-9} | 45 | 1.0×10²⁸ | 1.3×10³⁰ |
| (0.37 G, On) | 6.1×10^{-12} | 2.8×10^{-13}, 6.2×10^{-11} | 147 | 1.1×10²⁵ | 9.3×10²⁸ |
| (1.31 G, On) | 4.0×10^{-11} | 1.3×10^{-12}, 5.4×10^{-10} | 180 | 2.7×10²⁷ | 6.1×10²⁹ |
| (0.65 G, On, 1 Z_⊙) | 2.6×10^{-11} | 3.2×10^{-13}, 3.7×10^{-10} | 44 | 9.4×10²⁶ | 5.1×10²⁹ |
| (0.65 G, On, 10^{-1} Z_⊙) | 1.9×10^{-11} | ... | 184 | 9.8×10²⁵ | 2.8×10²⁹ |
| (0.65 G, On, 10^{-2} Z_⊙) | 3.2×10^{-12} | ... | 347 | 2.5×10²⁴ | 6.1×10²⁸ |
| (0.65 G, On, 10^{-3} Z_⊙) | 3.5×10^{-13} | ... | 710 | 4.0×10¹⁹ | 6.8×10²⁷ |
| (0.65 G, On, Z=0) | 1.2×10^{-14} | ... | 144 | 0 | 8.9×10²⁶ |

α Tau (Table 4):

| Case | ⟨Ṁ⟩ | ⟨v_out⟩ |
|---|---|---|
| (0.25 G, On, 0.74 Z_⊙) | **1.5×10^{-12}** | **179** |
| (0.25 G, Off, 0.74 Z_⊙) | 7.0×10^{-11} | 12 |
| (0.50 G, On, 0.74 Z_⊙) | **2.0×10^{-11}** | **52** |

Radial-profile targets (Fig. 2 α Boo fiducial vs ideal; Fig. 6 B-scan; Fig. 8 Z-scan; Fig. 9 α Tau):
- Peak time-averaged T ≈ several×10⁴ K at r ≈ 2 R_* (fiducial). Transient hot bubbles up to T > 10⁶ K occupy 16.1% of simulation time for α Boo fiducial.
- rms ⟨v_⊥²⟩ peaks ≈ 40 km s^{-1} at r = (2–3) R_* for α Boo.
- X-ray (T > 2×10⁶ K, > 0.2 keV) luminosity from simulation assuming spherical symmetry: transient maximum 10²⁷–10²⁸ erg s^{-1} (from single-tube scaled by 1/2800 → per-tube max ~10²⁴ erg s^{-1}).
- Density profile for α Boo fiducial: ρ ≈ 10^{-9} g cm^{-3} at photosphere drops to ~10^{-17}–10^{-16} g cm^{-3} at r = 2.6 R_* for cool blobs (Ohnaka MOLsphere comparison).
- Fig. 1 reference boxes: α Boo observed Ṁ = (2.5–4.0)×10^{-11} M_⊙ yr^{-1}; α Tau (1.0–1.6)×10^{-11} (Robinson+1998, Wood+2007) up to 2.8×10^{-11} (Wood+2024).

### P9. 1:1 reproduction checklist
1. 1D spherical MHD in super-radial flux tube with variable cross section `A(r) ∝ r²f(r)` (eq. 7) and `B_r ∝ 1/A`.
2. Ideal-MHD baseline with Godunov + MoCCT (Sano+1999) for `{ρ, v_r, v_⊥, B_⊥, e}`; γ = 5/3.
3. Spitzer thermal conduction `F_c = −κ_0 T^{5/2} ∂T/∂r` with κ_0 = 10^{-6} cgs.
4. Non-ideal terms: Ohmic (eq. 26) and ambipolar (eq. 27) diffusion of B_⊥ in induction AND in energy (Joule heating term in eq. 13), both functions of x_e(T, ρ).
5. Ionization module: H (Hartmann–Avrett 1984), He (LTE Saha), low-FIP heavy elements with radiative + thermal ionization (eq. 24) yielding x_e then μ (eq. 25).
6. Two-piece radiative cooling: Sutherland–Dopita 1993 table for T > 1.2×10⁴ K, scaled Anderson–Athay 1989 chromospheric for T ≤ 1.2×10⁴ K (eq. 17 with Z and ρ-transition), cut off below T_cut = 0.7 T_eff.
7. Shoda+2018a phenomenological Alfvén turbulence dissipation (eqs. 18–21) with c_d = 0.1 and r-dependent λ_⊥.
8. Stochastic photospheric driver with `P(ω) ∝ ω^{-1}` spectrum between ω_min and ω_max = 100 ω_min, applied to both longitudinal v_r and transverse v_⊥ components.
9. Tabulated elemental abundances per Table 2 (C, O, Na, Mg, Al, Si, S, K, Ca, Cr, Fe) scaled by Z.
10. Non-uniform grid resolving shortest Alfvén wavelength with ≥ 4 cells; trial-and-error N.
11. Run to `t_sim = 30 R_*/c_{s,0}`, time-average over `[0.1 t_sim, t_sim]`.

### P10. Gotchas
- Transient "magnetized hot bubbles" cause Ṁ to vary > 2 dex within a single tube; time averaging is essential, and comparison with observed Ṁ must invoke the tube-ensemble argument (N ≈ 2800 tubes on surface → half-hemisphere integrated σ_log ≈ ±4%).
- η_tot fluctuates wildly; arithmetic vs harmonic time-mean of Rm disagree by > 5 dex in corona — must track instantaneous values.
- Cooling cut-off T_cut is a numerical stability knob that biases cool-blob density low vs observation (Ohnaka MOLsphere ≈ 10^{-12}–10^{-11} g cm^{-3} vs simulation 10^{-17}–10^{-16}).
- Helm-like EOS not used; γ=5/3 ideal with μ from ionization. Latent heat of ionization/dissociation is *neglected* — overcools in fact; partially offset by T_cut cutoff.
- Steady-state criterion: inspect that outflowing structures have crossed the domain, typically by t > 0.1 t_sim.
- Resolution tuning is output-dependent: runs with smaller f_0 B_{r,0} (larger Ṁ, shorter λ_Alfvén) require larger N (cf. Table 3: 0.37 G → N=2500 vs 1.31 G → N=1080).

---

## Paper 2: 2203.15280 (Shimizu+Shoda+Suzuki 2022, solar-wind longitudinal waves)

### P1. Governing equations

Eqs. (3)–(8), 1D spherical super-radial flux tube with `h_r=1, h_θ=h_φ = r√f^op` (eq. 2):

- Continuity: `∂ρ/∂t + (1/(r²f^op)) ∂/∂r (ρ v_r r² f^op) = 0`
- Radial momentum (eq. 4): `∂(ρv_r)/∂t + (1/(r²f^op)) ∂/∂r [(ρ v_r² + p_T) r² f^op] = −ρ GM_⊙/r² + ρ v_⊥² d/dr ln √(r²f^op) + 2p d/dr ln √(r²f^op)`
- Transverse momentum (eq. 5): `∂(ρv_⊥)/∂t + (1/(r²f^op)) ∂/∂r [(ρ v_r v_⊥ − B_r B_⊥/(4π)) r² f^op] = (B_r B_⊥/(4π) − ρ v_r v_⊥) d/dr ln √(r²f^op) + ρ D_{v⊥}^{turb}`
- Radial-B (eq. 6): `(1/(r²f^op)) ∂/∂r (B_r r² f^op) = 0`  → `|B_r| r² f^op = |B_{r,⊙}| R_⊙² f_{⊙}^op = Φ_op` (eq. 1)
- Induction (eq. 7): `∂B_⊥/∂t + (1/(r²f^op)) ∂/∂r [(v_r B_⊥ − v_⊥ B_r) r² f^op] = (v_r B_⊥ − v_⊥ B_r) d/dr ln √(r²f^op) + √(4πρ) D_{b⊥}^{turb}`
- Energy (eq. 8): `∂e/∂t + (1/(r²f^op)) ∂/∂r [(e + p_T) v_r − (B_r/4π)(v_⊥·B_⊥)] r² f^op = −ρ v_r GM_⊙/r² + Q_C − Q_R`

with `p_T = p + B_⊥²/8π` (eq. 11), `e = e_int + ρv²/2 + B_⊥²/8π` (eq. 10).

- EOS: partially-ionized H via Saha (eqs. 19–22):  
  `e_int = p/(γ−1) + n_H χ I_H`, χ = ionization degree from Saha-Boltzmann `χ²/(1−χ) = 2/(n_H λ_e³) exp(−I_H/k_BT)`, I_H = 13.6 eV, `p = (1+χ) n_H k_B T`, `λ_e = h/√(2π m_e k_B T)`.
- Thermal conduction (eqs. 12–13): Spitzer–Härm `q_cnd = −min(1, ρ/ρ_cnd) (B_r/|B|) κ_0 T^{5/2} dT/dr`, κ_0 = 10^{-6} erg cm^{-1} s^{-1} K^{-7/2}, ρ_cnd = 10^{-20} g cm^{-3} (Shoda+2020 quenching in low-density regions).
- Radiative cooling (eqs. 14–18):  
  `Q_R = Q_R^thck ξ_rad + Q_R^thin (1 − ξ_rad)`, with `ξ_rad = max(0, 1 − p_chr/p)`, `p_chr = 0.1 p_⊙`.  
  Thick (Gudiksen–Nordlund 2005 exp): `Q_R^thck = (e_int − e_int^ref)/τ_thck`, `τ_thck = 0.1 (ρ/ρ̄)^{-1/2} s`, `ρ̄ = 1.87×10^{-7} g cm^{-3}`, `T^ref(r) = T_⊙`.  
  Thin: `Q_R^thin = Q_GJ(ρ,T) ξ_2 + n_H n_e Λ(T) (1 − ξ_2)`, ξ_2 linear blend around `T_TR = 15000 K` with `ΔT = 5000 K`. Q_GJ = Goodman–Judge 2012 chromospheric; Λ(T) from CHIANTI atomic database.
- Turbulence (eqs. 23–28): Shoda+2018a same as Paper 1 — `D_{v,b}^turb = ∓ c_d/(4λ_⊥) (|z^+| z^- ± |z^-| z^+)` (per θ,φ), `c_d = 0.1`, `λ_⊥ = λ_⊥,⊙ √((r² f^op)/(R_⊙² f_⊙^op))` (eq. 71), `λ_⊥,⊙ = 150 km` (eq. 27).
- No explicit resistivity / ambipolar diffusion (ideal-MHD core; non-ideal pointed out as limitation Section 4.3).

### P2. Geometry + grid
- 1D spherical, polar open flux tube, no rotation. `A(r) ∝ r² f^op(r)`.
- Two-step filling factor (eqs. 29–33): `f^op(r) = f^op_⊙ · f_1^exp(r) · f_2^exp(r)`.
  - `f_1^exp(r) = min[ f^op_cor/f^op_⊙, exp((r−R_⊙)/H_mag) ]`, H_mag = 2.5 H_⊙ = 2.5 a²/g with a = 6.9 km s^{-1}, g = 0.274 km s^{-2}.
  - `f_2^exp(r) = [F(r) + f^op_cor + F(R_⊙)(f^op_cor − 1)] / [f^op_cor (F(r) + 1)]`, `F(r) = exp((r−r_exp)/σ_exp)`, `r_exp/R_⊙ = 1.3`, `σ_exp/R_⊙ = 0.5`.
  - Standard: `f^op_⊙ = 1.00×10^{-3}`, `f^op_cor/f^op_⊙ = 100` (B_⊙ = 1300 G). Weak: `f^op_⊙ = 4.00×10^{-3}`, `f^op_cor/f^op_⊙ = 25` (B_⊙ = 325 G).
- r_in = R_⊙ (photosphere).
- r_out ≈ 95–100 R_⊙ for most cases (Table 1); reduced to 37–39 R_⊙ for runs with highest ⟨δv_∥⟩ amplitudes.
- Grid (eq. 34, Fig. 2): `Δr = max[Δr_m, min(Δr_M, (2 ε_ge/(2+ε_ge))(r − r_ge) + Δr_m)]`, Δr_m = 20 km, Δr_M = 2000 km, ε_ge = 0.01, r_ge = 1.04 R_⊙. Non-uniform stretched grid; finest near photosphere.

### P3. Initial condition
- `T(r=0) = T_⊙ = 5770 K` everywhere (eq. 35).
- Density (eq. 36): `ρ̄_init(r) = max[ ρ_⊙ exp(−(r−R_⊙)/H_⊙), ρ_{w,0} (r/R_⊙ − 1)^{-2.5} ]`, ρ_{w,0} = 10^{-19} g cm^{-3} (standard; reduced to 10^{-25} for B0V06 acoustic-only run); inner hydrostatic switches to outer power-law at `r/R_⊙ − 1 ≈ 0.01`. ρ_⊙ = 1.88 g cm^{-3} (photosphere).
- Stellar: solar. M_⊙, R_⊙, T_eff = 5770 K. Not scanned.

### P4. Boundary conditions
- Inner (r=R_⊙):
  - T fixed at 5770 K.
  - Density perturbation (eq. 40): `ρ_⊙(t) = ρ_⊙ (1 + v_{r,⊙}/a)`, a = 6.9 km s^{-1}.
  - Radial velocity: `v_{r,⊙}(t) = δv_∥,⊙(t)` (eq. 41), broadband longitudinal driver (eq. 42):  
    `δv_∥,⊙ ∝ Σ_{N=0}^{100} sin(2π f_N^l t + φ_N^l) / √(f_N^l)` with random phases, `3.33×10^{-3} Hz ≤ f_N^l ≤ 1.00×10^{-2} Hz` (5 min – 100 s).
  - Transverse driver via outgoing Elsässer `z^+_{⊥,⊙}` (eq. 38):  
    `z^+_{θ,φ,⊙} ∝ Σ_{N=0}^{100} sin(2π f_N^t t + φ_N^t) / √(f_N^t)`, `1.00×10^{-3} Hz ≤ f_N^t ≤ 1.00×10^{-2} Hz`. Incoming component z^- is absorbed via `∂z^-/∂r = 0` (eq. 37, free).
  - Normalization: `⟨δv_⊥,⊙⟩ = 0.6 km s^{-1}` (all cases except B0V06 where it is 0); `⟨δv_∥,⊙⟩` is the scanning parameter, 0–3.0 km s^{-1} (Table 1). Note the "random" peak amplitude is ~√2 × these rms values.
- Outer (r = r_out): free boundary (zero-gradient).

### P5. Time integration + code
- 1D MHD with phenomenological turbulence. The paper does not explicitly name a code or scheme; the Shoda lineage (Shoda+Yokoyama+Suzuki 2018a, 2019; Shoda+2020) uses HLLD-based Godunov with MoCCT-equivalent treatment for transverse Alfvén modes, explicit CFL-limited time stepping. Spitzer conduction is treated sub-cycled in most descendants (not explicit in this paper).
- Simulation wall-time: 4500 min for runs with ⟨δv_∥⟩ = 0–1.2 km s^{-1}; 6000 min for ⟨δv_∥⟩ = 1.5–3.0 km s^{-1} cases.
- Averaging window: last 1500 min before t_end.
- Code: private Suzuki-group / Shoda code; based on the framework of Shoda+Yokoyama+Suzuki 2018a and Shoda+Takasao 2021 super-radial formulation.

### P6. Parameter sweep (Table 1)

14 runs. Label `BxVyy`: x ∈ {0, s, w} (zero, standard, weak B), yy = 10 × ⟨δv_∥,⊙⟩ [km s^{-1}].

| Model | ⟨δv_⊥,⊙⟩ | ⟨δv_∥,⊙⟩ | B_⊙ [G] | f_⊙^op | f_cor^op/f_⊙^op | r_out | Ṁ [M_⊙ yr^{-1}] | v_{r,out} [km/s] |
|---|---|---|---|---|---|---|---|---|
| B0V06 | 0 | 0.6 | 1.3×10^{-4} (≈0) | 1×10^{-3} | 100 | 95.6 R_⊙ | accretion | — |
| BsV00 | 0.6 | 0 | 1300 | 1×10^{-3} | 100 | 99.5 R_⊙ | 1.32×10^{-14} | 688.05 |
| BsV04 | 0.6 | 0.4 | 1300 | 1×10^{-3} | 100 | 99.5 R_⊙ | 1.75×10^{-14} | 687.77 |
| BsV06 | 0.6 | 0.6 | 1300 | 1×10^{-3} | 100 | 99.5 R_⊙ | 1.97×10^{-14} | 697.02 |
| BsV09 | 0.6 | 0.9 | 1300 | 1×10^{-3} | 100 | 99.5 R_⊙ | 2.63×10^{-14} | 701.24 |
| BsV12 | 0.6 | 1.2 | 1300 | 1×10^{-3} | 100 | 99.5 R_⊙ | 3.10×10^{-14} | 716.19 |
| BsV15 | 0.6 | 1.5 | 1300 | 1×10^{-3} | 100 | 99.5 R_⊙ | 3.54×10^{-14} | 691.51 |
| BsV18 | 0.6 | 1.8 | 1300 | 1×10^{-3} | 100 | 39.1 R_⊙ | 4.18×10^{-14} | 633.64 |
| BsV21 | 0.6 | 2.1 | 1300 | 1×10^{-3} | 100 | 39.1 R_⊙ | 4.57×10^{-14} | 635.62 |
| BsV27 | 0.6 | 2.7 | 1300 | 1×10^{-3} | 100 | 39.1 R_⊙ | 5.09×10^{-14} | 560.80 |
| BsV30 | 0.6 | 3.0 | 1300 | 1×10^{-3} | 100 | 39.1 R_⊙ | 4.97×10^{-14} | 561.31 |
| BwV00 | 0.6 | 0 | 325 | 4×10^{-3} | 25 | 95.3 R_⊙ | 1.26×10^{-14} | 586.55 |
| BwV06 | 0.6 | 0.6 | 325 | 4×10^{-3} | 25 | 95.3 R_⊙ | 2.57×10^{-14} | 581.24 |
| BwV18 | 0.6 | 1.8 | 325 | 4×10^{-3} | 25 | 37.8 R_⊙ | 4.76×10^{-14} | 460.09 |

### P7. Key diagnostics

Surface-integrated luminosities (eqs. 48–53):
- Kinetic: `L_K = (1/2) ρ v_r³ · 4π r² f^op`
- Enthalpy: `L_E = γ/(γ−1) · p v_r · 4π r² f^op`
- Alfvénic Poynting: `L_A = [ (1/2) ρ v_⊥² v_r + B_⊥²/(4π) v_r − B_r/(4π)(v_⊥·B_⊥) ] · 4π r² f^op`
- Conductive: `L_C = −q_cnd · 4π r² f^op`
- Gravitational PE: `L_G = ρ v_r (GM_⊙/r) · 4π r² f^op = Ṁ_w · GM_⊙/r`
- Radiation: `L_R(r) = ∫_{r_lch}^r Q_R · 4π r'² f^op dr'`, r_lch − R_⊙ = 0.7 Mm.

Mass-loss (eq. 44): `Ṁ_w = 4π r² f^op ρ v_r`, measured at r = r_out.

Elsässer / wave decomposition (eqs. 69–70): `L_A^± = ρ (z_⊥^±)² (v_r ± v_A) π r² f^op`, v_A = B_r/√(4πρ).

Mode-conversion diagnostic (eqs. 63, 66): ε_{∥↔⊥} = −v_r ∂/∂r(B_⊥²/8π) + ρ v_⊥² (d/dr) ln √(r²f^op) − (B_⊥²/4π) v_r d/dr ln p.

Turbulent-loss integrand (eq. 64): `Q_turb = c_d ρ Σ_i (|z_i^+| (z_i^-)² + |z_i^-| (z_i^+)²) / (4 λ_⊥)`.

Per-unit-mass Alfvén dissipation (eq. 67): `ζ_A = −(1/(4πρ r²f^op)) ∂L_A/∂r`.

Cranmer–Saar scaling (eq. 61): `Ṁ_w ≈ L_{A,cb} / v_g,⊙²`, `v_g,⊙ = √(2GM_⊙/R_⊙) = 617 km s^{-1}`, coronal base at r_cb/R_⊙ = 1.03.

Fit (eq. 46): `ΔṀ_w = 1.41 ⟨δv_∥,⊙⟩^{1.05} × 10^{-14} M_⊙ yr^{-1}` for ⟨δv_∥,⊙⟩ ≤ 2.7 km s^{-1}.

### P8. CRITICAL — numerical values to match
- Ṁ_w values in Table 1 (reproduced above), all in range 1.26–5.09 × 10^{-14} M_⊙ yr^{-1}; v_{r,out} in 460–716 km s^{-1}.
- Baseline Ṁ_w^0 = 1.32 × 10^{-14} M_⊙ yr^{-1} (BsV00).
- Mass-loss enhancement: `ΔṀ_w = Ṁ_w − Ṁ_w^0`, fit `= 1.41 ⟨δv_∥⟩^{1.05} × 10^{-14}` for ⟨δv_∥⟩ ≤ 2.7, saturation above.
- Coronal base (r_cb = 1.03 R_⊙) density (Fig. 4): ρ_cb ≈ 10^{-15} to 4×10^{-15} g cm^{-3} (BsV00 → BsV18 ≈ 4× increase); n_e ≈ 10⁸–10⁹ cm^{-3}.
- Coronal T peak in cases: T ≈ 1.0–1.2 × 10⁶ K at r ≈ 1.05–1.1 R_⊙ (Fig. 3a). (Exact values read from plot; not tabulated.)
- Terminal wind velocities 560–716 km s^{-1}, ≈ v_g,⊙ = 617 km s^{-1}.
- Fig. 6 component fractions (L/L_{A,⊙}): L_{A,cb}/L_{A,⊙} grows from ≈ few % (BsV00) to ≈ 20% at ⟨δv_∥⟩ ≈ 1.8 km s^{-1} then saturates; L_{R,cb}/L_{A,⊙} increases monotonically; L_{G,cb}/L_{A,⊙} ≈ L_{K,out}/L_{A,⊙} ≈ 1–10 %.
- Fig. 11: density-fluctuation rms ⟨δρ⟩/⟨ρ⟩ ≈ 0.03–0.5 near surface, drops to ≈ 0.1 in corona.
- Fig. 12 Elsässer amplitudes at r = 40 R_⊙ (PSP region): `⟨z_⊥^+⟩ ≈ 150–300 km s^{-1}`, `⟨z_⊥^-⟩ ≈ 30–100 km s^{-1}`.
- Appendix A B0V06: acoustic-only → atmosphere heats only to T ≲ 2×10⁵ K, no steady outflow, in-fall (Ṁ < 0 in r/R_⊙ − 1 > 1).

Values graphically (cannot be read to > 2 digits without data): coronal T(r), ρ(r), v_r(r) profiles of Fig. 3, 4; `ζ_A` of Fig. 8; `hβ_r⟩(r)` of Fig. 9; Alfvén Poynting radial decay of Fig. 10.

### P9. 1:1 reproduction checklist
1. 1D super-radial flux tube `A(r) = r² f^op(r)` with two-step f^op (eqs. 29–33) and Kopp-Holzer second expansion.
2. MHD `{ρ, v_r, v_⊥, B_⊥, e}` with B_r = Φ_op / (r² f^op) analytic; solve eqs. 3–8.
3. Partial-ionization H-only EOS: Saha + ionization latent heat in e_int (eqs. 19–22).
4. Spitzer–Härm conduction along B with low-ρ quench at ρ_cnd = 10^{-20} g cm^{-3}.
5. Two-component radiative cooling: optically-thick Newton-cooling (Gudiksen-Nordlund 2005 with τ_thck = 0.1 (ρ/ρ̄)^{-1/2}) in chromosphere smoothly blended (ξ_rad) into optically-thin Goodman–Judge 2012 chromospheric + CHIANTI coronal (ξ_2 around T_TR = 15000 K).
6. Shoda+2018a turbulent Alfvén dissipation closure, c_d = 0.1, λ_⊥,⊙ = 150 km.
7. Photospheric BC: fixed T = 5770 K; free ∂z^-/∂r = 0; outgoing stochastic z^+ driver (f_N^t spectrum, 1e-3 – 1e-2 Hz); longitudinal v_r driver (f_N^l spectrum, 3.33e-3 – 1e-2 Hz); tie ρ_⊙ to v_{r,⊙} via eq. 40.
8. Stretched grid 20 km → 2000 km (eq. 34).
9. Initial hydrostatic isothermal + power-law patch; relax for thousands of minutes.
10. No resistivity, no ambipolar. γ=5/3 for ideal-gas piece; ionization latent heat additional.

### P10. Gotchas
- BsV00 baseline Ṁ_w is ~5× smaller than observed solar (~2×10^{-14} vs 2×10^{-14}–10^{-14} obs OK-ish, but initial outer power-law ρ is 5× lower than observed — quasi-steady state self-adjusts).
- Quasi-steady only after ≳3000 min; must average last 1500 min.
- For high ⟨δv_∥⟩ ≥ 1.8 km s^{-1}, r_out reduced to ~38 R_⊙ to control CFL / reduce cost. Cannot compare v_{r,out} values across the r_out split naively — BsV18 v_{r,out} = 634 km s^{-1} at 39 R_⊙ vs BsV15 = 692 at 100 R_⊙.
- Energy-conservation budget (eq. 59) not perfectly closed; LA,⊙ > LA,cb + LR,cb because L_R below r_lch = R_⊙ + 0.7 Mm is set to zero.
- Conduction quenching at ρ_cnd = 10^{-20} g cm^{-3} is a numerical device for speed; do not turn off!
- Mode-conversion sign: ε_{∥↔⊥} is defined positive for ⊥→∥; negative values in Fig. 7c, 10d indicate "inverse" ∥→⊥ that generates Alfvén waves from acoustic drivers in chromospheric β≈1 layer.
- Longitudinal driver frequency 1 < f_N^l < 10 mHz narrower than transverse 1–10 mHz; matches p-mode 5-min peak.
- Random phases required — deterministic sinusoids will lock into the injection frequencies and give spurious resonances.
- Appendix A B0V06 requires ρ_{w,0} = 10^{-25} g cm^{-3} initial (not 10^{-19}) to avoid initial infall swamping the acoustic-only heating.
