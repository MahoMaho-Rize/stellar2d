# Suzuki-group papers — reproduction-grade setup extraction (Part 2)

Source PDFs:
- `docs/research_survey/pdfs/suzuki/2403.18409.pdf` — Matsuoka, Suzuki, Tokuno & Kakiuchi (2024, submitted ApJ), "Effect of Magnetic diffusion in the Chromosphere on the Solar Wind"
- `docs/research_survey/pdfs/suzuki/2305.12112.pdf` — Suzuki (2023, sole author, ApJ), "MHD in a cylindrical shearing box II: Intermittent Bursts and Substructures in MRI Turbulence"

These two papers use very different setups and are covered independently.

---

## Paper 3: 2403.18409 (Matsuoka+Suzuki+Tokuno+Kakiuchi 2024)

1D non-ideal MHD Alfvén-wave-driven solar wind from photosphere (r = R_sun) to r = 40 R_sun, solved in a super-radially expanding flux tube. Goal is to quantify how Ohmic (eta_O) and ambipolar (eta_AD) diffusion in the partially ionised chromosphere damp Alfvénic Poynting flux and reduce mass-loss rate.

### P1. Governing equations

Written in a 1D expanding flux tube; A(r) = r^2 f, with f the filling factor. Equations (3)-(7):

- Continuity: dt(rho) + (1/(r^2 f)) dr[rho v_r r^2 f] = 0
- Radial momentum: dt(rho v_r) + (1/(r^2 f)) dr[(rho v_r^2 + p + B_perp^2/(8 pi)) r^2 f] = (1/(r^2 f))(rho v_perp^2/2 + p) dr(r^2 f) - rho G M_sun / r^2
- Perpendicular momentum (eq. 5): dt(rho v_perp) + (1/(r^3 f^(3/2))) dr[(rho v_r v_perp - B_r B_perp/(4 pi)) r^3 f^(3/2)] = rho D_{v_perp} (turbulent damping, see below)
- Energy (eq. 6): dt E + (1/(r^2 f)) dr[((E + p_T) v_r - B_r B_perp . v_perp/(4 pi)) r^2 f] = (1/(r^2 f)) dr[(eta_tot/(4 pi)) r sqrt(f) B_perp . dr(B_perp r sqrt(f))] - rho v_r G M_sun / r^2 + Q_rad + Q_cond
- Induction (eq. 7): dt(B_perp) + (1/(r sqrt(f))) dr[(B_perp v_r - B_r v_perp) r sqrt(f)] = sqrt(4 pi rho) D_{b_perp} + (1/(r sqrt(f))) dr[eta_tot dr(B_perp r sqrt(f))]
- Magnetic flux conservation: B_r(r) = B_{r,0} R_sun^2 f_0 / (r^2 f) (eq. 2)
- E = rho e + (1/2) rho v^2 + B_perp^2/(8 pi);  p_T = p + B_perp^2/(8 pi);  p = (rho / (mu m_u)) k_B T (ideal gas, eq. 11).
- EOS: ideal; gamma = 5/3 implied by Spitzer conductivity and by standard Suzuki/Shimizu+2022 heritage (paper text says "second order accuracy in time and space" and inherits Shimizu+2022 numerics). mu computed from ionisation balance (eq. 26).

**Dissipation terms — exact formulas.**

- Ohmic diffusivity (eq. 21): eta_O ≈ 2.3e2 · max((1 - x_e), 0) / x_e · sqrt(T/K)  [cm^2 s^-1]
- Ambipolar diffusivity (eq. 22): eta_AD ≈ 2.1e-16 · (B/G)^2 · max((1 - x_e), 0)^2 / ([rho/(g cm^-3)]^2 · x_e)  [cm^2 s^-1]
- eta_tot = eta_O + eta_AD (eq. 12). Hall term and cross-component AD terms dropped; see Appendix A.
- Ionisation fraction x_e solved from eq. (23) (H, He, He+, He++ + heavy elements C, O, Na, Mg, Al, Si, S, K, Ca, Cr, Fe with Asplund+09 abundances). Photoionisation by T_eff = 5780 K blackbody + interstellar (T_gal = 7500 K, W_gal = 1e-14). Recombination/photoionisation rates from eq. (24) with geometrical dilution factor W(r) (eq. 25).
- Turbulent Alfvén-wave damping (eqs. 17-20): D_{v_perp,i} = -(c_d / (4 lambda_{perp,i})) (|z^+_{perp,i}| z^-_{perp,i} + |z^-_{perp,i}| z^+_{perp,i});  D_{b_perp,i} = -(c_d / (4 lambda_{perp,i})) (|z^+_{perp,i}| z^-_{perp,i} - |z^-_{perp,i}| z^+_{perp,i}); Elsasser variables z^± = v_perp ∓ B_perp/sqrt(4 pi rho); c_d = 0.1; lambda(r) = lambda_0 (r/R_sun) sqrt(f(r)/f_0) with lambda_0 = 1e3 km.

**Radiative cooling Q_rad (optically thick + thin).**

- Optically-thick chromospheric formula (Anderson & Athay 1989, eq. 15): Q_rad = 4.5e9 · rho · min(1, rho/rho_crt),  rho_crt = 1e-16 g cm^-3. Applied for T < T_crt = 1.2e4 K.
- Optically-thin coronal cooling (eq. 16): Q_rad = Lambda(T) n n_e, Lambda from Sutherland & Dopita (1993) tables. Applied for T > T_crt.
- The two regimes are smoothly interpolated in a transition zone around T_crt.

**Thermal conduction (eqs. 13-14).**

- Q_cond = -(1/(r^2 f)) dr(F_c r^2 f); F_c = kappa_0 T^{5/2} dr(T); kappa_0 = 1e6 g cm s^-3 K^{-7/2}. Spitzer-Harm form used over the whole domain (authors note correction is conservative in chromosphere but dominated by other terms).

### P2. Geometry + grid

- 1D, spherical radius, super-radial flux tube. A(r) = r^2 f(r).
- Filling factor (eq. 1):  f(r) = [exp((r - R_sun - h)/sigma) + f_0 - (1 - f_0) exp(-(r - R_sun)/h)] / [exp((r - R_sun - h)/sigma) + 1]
- Parameters: f_0 = f(R_sun) = 1/1265, sigma = h/2, h = 0.042 R_sun.
- Inner boundary: r_in = R_sun = 6.96e10 cm. Outer boundary r_out = 40 R_sun. Above r_out the grid is stretched out to ~80 R_sun with outgoing BC (Suzuki & Inutsuka 2005, 2006).
- Grid spacing below r_out: cell size small enough that authors evaluate L_{A,+,0} at 6 km above the photosphere (average of 4 cells). Paper does not cite total N, but inherits the Shimizu+22 grid: Δr(r=R_sun) ≈ 1.5 km logarithmically-stretched. (Consistency: 6 km = 4-cell average ⇒ Δr ~ 1.5 km at surface.)

### P3. Initial condition

- Hydrostatic isothermal atmosphere at T = T_eff = 5780 K where rho > rho_turn = 2.5e-13 g cm^-3.
- Above rho_turn (ρ < 2.5e-13), density is set higher than HSE to avoid unphysical Alfvén speed (numerical choice to protect dt).
- rho_0 = rho(R_sun) = 2.5e-7 g cm^-3, taken from ATLAS/Kurucz 1979 + Castelli & Kurucz 2003 model.
- Magnetic: radial component B_{r,0} = 1.48 kG at photosphere, determined by the equipartition boundary condition (eq. 27): 8 pi p_0 / B_{r,0}^2 = 1 with p_0 from EOS at T_eff.
- Initial B_perp = 0; Alfvén waves are driven at the inner boundary.

See Figure 1 for the initial density profile (black solid) vs. time-averaged ideal run M0 (blue dash-dotted).

### P4. Boundary conditions

- **Inner (r = R_sun):** T = T_eff = 5780 K fixed; rho = rho_0 fixed; B_{r,0} fixed. Velocity drives waves via a broadband perturbation spectrum (eq. 28): <(dv_0)^2> = integral_{omega_min}^{omega_max} P(omega) dw with 1/omega_min = 30 min, 1/omega_max = 0.3 min; P(omega) ∝ 1/omega (1/f-spectrum); applied to BOTH transverse dv_perp AND longitudinal dv_r. Fiducial amplitude <dv_0> = 1.25 km/s (based on observed 0.37 - 2.4 km/s photospheric velocities: Title+89, Oba+20, Berger+98).
- **Outer (r = r_out = 40 R_sun):** outgoing BC via Suzuki & Inutsuka (2005, 2006) characteristic prescription for waves and mass.

### P5. Time integration + code identity

- Code lineage: Suzuki & Inutsuka (2005, 2006) ideal MHD flux-tube model extended with Shimizu+22 radiative cooling/thermal conduction and now with Ohmic+AD diffusion terms.
- Scheme: Godunov-type second-order (van Leer 1979) for compressible part, treating gas + magnetic pressures (Sano+99 + Suzuki & Inutsuka 2005). MoC (Stone & Norman 1992) for Alfvénic/incompressible part. 2nd order in space AND time.
- Operator split: radiative cooling + thermal conduction updated with time-implicit method (because these times are too short for stable explicit integration at chromosphere/transition-region conditions). Other terms explicit.
- Magnetic diffusion: explicit second-order (inherited from ideal scheme).
- Run time: 6 t_sim, where t_sim = R_sun / c_{s,0} is sound-crossing time at the photosphere. 6 t_sim ≈ 10 Alfvén crossing times of 40 R_sun with <v_A> ≈ 477 km/s.
- Quasi-steady statistics taken over the interval t = 3 t_sim to 6 t_sim.

### P6. Parameter sweep (Table 1)

Six runs:

| model | eta_O | eta_AD | <dv_0> (km/s) |
|---|---|---|---|
| M0 | 0 | 0 | 1.25 (ideal baseline) |
| M1 | on | 0 | 1.25 (Ohmic only) |
| M2 | 0 | on | 1.25 (AD only) |
| M3 | on | on | 1.25 (both, fiducial non-ideal) |
| M3-149 | on | on | 1.49 |
| M3-170 | on | on | 1.70 |

### P7. Key diagnostics — exact formulas

- Alfvén Poynting luminosity (eq. 30): L_A = [v_r (rho v_perp^2/2 + B_perp^2/(4 pi)) - B_r v_perp . B_perp / (4 pi)] · 4 pi r^2 f
- Outgoing/incoming Alfvén (eqs. 39-40): L_{A,+} = rho (z^+_perp)^2 (v_r + v_A) pi r^2 f;  L_{A,-} = rho (z^-_perp)^2 (v_r - v_A) pi r^2 f
- Diffusive Poynting (eq. 35): L_D = [eta_tot / (4 pi r sqrt(f))] B_perp . dr(B_perp r sqrt(f)) · 4 pi r^2 f
- Kinetic luminosity (eq. 31): L_K = (1/2) rho v_r^3 · 4 pi r^2 f
- Enthalpy luminosity (eq. 32): L_E = (gamma/(gamma-1)) p v_r · 4 pi r^2 f
- Gravitational luminosity (eq. 33): L_G = rho v_r · G M_sun/r · 4 pi r^2 f = Mdot · G M_sun / r
- Conductive luminosity (eq. 34): L_C = kappa_0 T^{5/2} dr(T) · 4 pi r^2 f
- Radiative loss (eq. 36): L_R = - integral_r^{r_out} 4 pi r^2 f Q_rad dr
- Mass-loss rate (eq. 37): Mdot = 4 pi r^2 f rho v_r
- Magnetic Reynolds number (eq. 41): R_m = V L / (eta_O + eta_AD); V = 10 km/s, L = 100 km (typical chromospheric velocity + pressure scale height)
- Elsasser reflection ratio: R_E = <(z^-)^2> / <(z^+)^2>
- Density fluctuation (eq. 43): n(r) = (1/<rho>) sqrt(<(rho - <rho>)^2>)

### P8. CRITICAL numerical values to match

From Table 1 (L's in 10^27 erg/s; all <dv_0> = 1.25 km/s except as noted):

| model | L_{A,+,0} | L_{A,tc} | L_{A,out} | L_{K,out} | L_{R,tc} | L_{G,tc} | Mdot [M_sun/yr] |
|---|---|---|---|---|---|---|---|
| M0 | 91.1 | 8.49 | 0.19 | 2.77 | 3.41 | 2.45 | 2.04e-14 |
| M1 | 90.7 | 7.79 | 0.21 | 2.85 | 3.41 | 2.21 | 1.83e-14 |
| M2 | 90.6 | 3.66 | 0.13 | 1.27 | 0.79 | 0.70 | 5.82e-15 |
| M3 | 90.4 | 2.70 | 0.11 | 0.94 | 0.60 | 0.42 | 3.52e-15 |
| M3-149 | 128 | 8.39 | 0.20 | 2.95 | 2.41 | 2.41 | 2.01e-14 |
| M3-170 | 166 | 13.0 | 0.19 | 3.65 | 4.30 | 5.40 | 3.57e-14 |

Key numerical targets:
- **Mdot suppression M3/M0 = 3.52e-15 / 2.04e-14 ≈ 1/6** — non-ideal MHD reduces mass-loss rate by factor ~6.
- **L_{A,tc} suppression M3/M0 ≈ 2.70/8.49 ≈ 1/3** (chromospheric Poynting "survival fraction").
- Reflection fraction at photosphere: 89% of input flux reflected back in M0; 79% in M1; 74% in M3.
- Magnetic Reynolds number (Fig. 2): R_m reaches minimum 1 - 10 at r ≈ 1000 km (r - R_sun ≈ 1.4e-3 R_sun), R_m < 10 throughout middle/upper chromosphere (r - R_sun ≲ 5000 km ≈ 7e-3 R_sun).
- Chromospheric temperature jump: T climbs from ~5780 K (photosphere) through ~5000 K valley to ~20000 K at top of chromosphere (diamonds in Fig. 3). Non-ideal cases M2/M3 shift the T jump outward.
- Peak of L_D (diffusive Poynting dissipation, Fig. 5): at r - R_sun = 2e-4 R_sun in M1/M3 (Ohmic peak) and 1.5e-3 R_sun in M2/M3 (AD peak); L_D amplitude is O(1e26) erg/s (shown multiplied by 1000 in Fig. 5).
- Alfvén-wave amplitude <b_perp> < <v_perp> in chromosphere of M3 (AD-damped) by factor ~2 at r - R_sun ~ 1e-3 R_sun; inverted in the low corona (<b_perp> > <v_perp>) — see Fig. 6.
- Density fluctuation (Fig. 11): n peak ~1 at r - R_sun ≈ 1e-3 R_sun in M0; in M3-149 n is suppressed to ~0.3 - 0.5 there, consistent with Krupar+20 PSP radio observations.

To recover present-day Mdot ≈ 2e-14 M_sun/yr in the non-ideal case (M3-149), the driving amplitude must be raised to <dv_0> = 1.49 km/s — a ~40% increase in input energy (<dv_0^2>) gives a factor-6 Mdot increase.

### P9. 1:1 reproduction checklist

Required physics modules:
- [ ] 1D spherical super-radial flux tube with prescribed A(r) = r^2 f(r)
- [ ] Ideal MHD + radial gravity
- [ ] Ionisation-balance solver for x_e (H, He, He+, He++ + 11 metals; LTE with Saha-like balance including blackbody + galactic photoionisation)
- [ ] Ohmic diffusivity eta_O from x_e, T
- [ ] Ambipolar diffusivity eta_AD from x_e, rho, B
- [ ] Explicit non-ideal induction terms in B_perp equation and corresponding Ohmic+AD heating in energy equation
- [ ] Spitzer-Harm thermal conduction with implicit time step
- [ ] Sutherland-Dopita + Anderson-Athay piecewise radiative cooling with implicit time step
- [ ] Turbulent Alfvén damping (Hossain+95) D_v, D_b with Elsasser variables + c_d=0.1 + lambda(r)
- [ ] Characteristic-based outgoing BC at r_out
- [ ] 1/f-spectrum broadband velocity driver at photosphere, 0.3 - 30 min period, amplitude calibrated to <dv_0> ∈ {1.25, 1.49, 1.70} km/s

The run is 1D (single radial dimension, fluxes in r and perpendicular components treated as 2 transverse vectors). Not 2D/3D; no grid decomposition.

### P10. Gotchas

- Transient ≈ 3 t_sim (t_sim = R_sun/c_{s,0}); statistics averaged over 3-6 t_sim.
- Initial density above rho_turn = 2.5e-13 g cm^-3 is UNPHYSICALLY high to control dt via v_A limit; authors state final steady state does not depend on rho_turn so long as it is small enough.
- Hall term explicitly DROPPED (Sec. 2.3 and Appendix A); also the cross-component "nonlinear" AD terms are dropped — only the isotropic AD term is kept. Hall instability noted as a future 3D question.
- Non-equilibrium ionisation ignored (Section 4.3 caveat): equilibration timescale in upper chromosphere ~1e5 s > Alfvén transit ~1e2-1e3 s. Paper flags this as a limitation, not a bug.
- Longitudinal-wave driving included alongside transverse — shock heating of chromosphere still dominates over AD heating (Arber+16 argument preserved).
- Evaluating L_{A,+,0} at r = R_sun directly gives noise because of boundary reflection coupling; authors use r = 6 km + 4-cell average instead. Without this, L_{A,+,0} is smaller by factor ~1.7 than the naive rho<dv_0^2>v_A estimate.
- 1D flux-tube tends to overestimate density fluctuations (n; Sec. 4.1) because compressible waves are laterally confined.

---

## Paper 4: 2305.12112 (Suzuki 2023)

3D ideal MHD in a local cylindrical shearing box (an extension of Hawley+95 Cartesian shearing-box), unstratified, weak net vertical B_z, with modified radial shearing BC. Goal: expose curvature-driven compressible amplification of B_phi which is absent in Cartesian boxes; produce intermittent bursts and ring-gap substructures in MRI-driven α-disc.

### P1. Governing equations

Ideal MHD in cylindrical (R, phi, z). Eqs. (1)-(5):

- Continuity: drho/dt + rho div(v) = 0 (Lagrangian form)
- Momentum: rho dv/dt = -grad(p + B^2/(8 pi)) + (1/(4 pi))(B . grad) B - rho (G M_star / R^2) Rhat
- Induction: dt B = curl(v × B)
- div(B) = 0
- EOS (locally isothermal, eq. 5-6): p = rho c_s^2(R), with c_s^2 = c_{s,0}^2 (R/R_0)^{-1}

No resistivity, no viscosity, no cooling explicitly — ideal MHD. Numerical dissipation is the only sink; authors quantify it via the variation-rate imbalance in magnetic-energy triangle diagrams (Sec. 3.2, Fig. 4).

EOS: locally isothermal; adiabatic index effectively γ → infinity for the pressure law but energy eq. not solved (isothermal closure).

Equilibrium: sub-Keplerian rotation Omega_eq defined by radial force balance (eq. 8-10):
- -v_{phi,eq}^2 / R = -G M_star / R^2 - (1/rho) dp/dR
- Omega_eq = Omega_K sqrt(1 - 2 f_K)
- f_K = -(1/rho) dp/dR / (2 R Omega_K^2) = (1 + mu) c_s^2 / (2 R^2 Omega_K^2) where rho ∝ R^{-mu}.
- With mu = 1, c_{s,0} = 0.1 R_0 Omega_{K,0} → f_K = c_s^2 / (R^2 Omega_K^2) = 0.01 at R = R_0; Omega_eq,0 = 0.99 Omega_K,0.
- For mu = 1, c_{s,0} = 0.01 R_0 Omega_{K,0} → f_K ≈ 1e-4; Omega_eq,0 ≈ Omega_{K,0}.

Velocities in the corotating frame (eq. 7): dv = v - R Omega_eq phihat; (dv_R, dv_phi, dv_z) = (v_R, v_phi - R Omega_eq, v_z).

### P2. Geometry + grid

**Cylindrical shearing box — "local cylindrical" domain:**

| Model | R_- | R_+ | L_R | phi_- | phi_+ | L_phi | z_- | z_+ | L_z | N_R | N_phi | N_z |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H_0/R_0 = 0.1 | 0.82 R_0 | 1.22 R_0 | 4 H_0 | -pi/12 | pi/12 | (5pi/3) H_0 | -0.2 R_0 | 0.2 R_0 | 4 H_0 | 256 | 320 | 256 |
| H_0/R_0 = 0.01 | 0.980 R_0 | 1.020 R_0 | 4 H_0 | -pi/120 | pi/120 | (5pi/3) H_0 | -0.02 R_0 | 0.02 R_0 | 4 H_0 | 256 | 320 | 256 |
| Cartesian | -2 H_0 | 2 H_0 | 4 H_0 | -(5pi/6) H_0 | (5pi/6) H_0 | (5pi/3) H_0 | -2 H_0 | 2 H_0 | 4 H_0 | 256 | 320 | 256 |

Where H_0 = c_{s,0} / Omega_{K,0} is scale height at R_0. Vertical domain = 4 H_0 (four times larger than S19 = Suzuki+19), to contain more MRI channel wavelengths in saturation.

Grid: Δphi, Δz constant; ΔR ∝ R (so that Δphi·R ≈ ΔR ≈ Δz). Resolution = 64/H_0 in R and z, ≈ 61/H_0 in phi.

Vertical gravity IGNORED (unstratified; gravity only radial). Therefore periodic BC can be used in z.

Units: R_0 = 1, rho_0 = 1, Omega_{K,0} = 1. Magnetic field normalised by R_0 Omega_{K,0} sqrt(4 pi rho_0), which cancels the sqrt(4 pi) factor of Gaussian cgs. One rotation = 2 pi / Omega_{K,0}.

### P3. Initial condition

Uniform weak net vertical B_z with R-profiles from S19 (eqs. 13-15):
- rho_init(R) = rho_{init,0} (R/R_0)^{-1}
- B_{z,init}(R) = B_{z,0,init} (R/R_0)^{-1}
- Plasma beta: beta_{z,init} = 8 pi rho_init c_s^2 / B_z,init^2 = 1e4 (constant initially).

This guarantees constant beta and the B_z ∝ R^{-1} needed for stationary BC-compatible induction solution.

MRI seed: random velocity perturbations added to v_R and v_phi on top of sub-Keplerian equilibrium (v_R, v_phi, v_z) = (0, R Omega_eq, 0).

### P4. Boundary conditions

- **phi (azimuthal):** periodic.
- **z (vertical):** periodic (because unstratified; no z-gravity).
- **R (radial):** modified shearing-radial BC. Conserved-variable shearing periodic (S_±(R, phi, z) = <S(R∓, phi + ΔΩ_eq t, z)> + δS, eq. 11-12). Sign-flip: the shearing-sheet rule is applied on CONSERVED quantities, not primitives, including curvature effects. Amplitude-matching: the rms fluctuation in the ghost cells is rescaled so that <δS^2> at R_{±,g} matches the domain-interior <δS^2> at the adjacent radially-interior cell — prevents mismatched radial BC corruption seen in S19.
- Two matching factors: f_{amp,±} tunes fluctuation amplitude entering the ghost cell. Tuned by "reproduce steady accretion" during inactive-phase mean. For H_0/R_0 = 0.1: f_{amp,+} = 1.02, f_{amp,-} = 0.93; for H_0/R_0 = 0.01: f_{amp,+} = 1.06, f_{amp,-} = 0.96. Sensitivity checked in Appendix A / Table 4 with (1.02, 0.93), (1.06, 0.96), (1, 1).
- f_sh = 0.5 (fixed): fraction of amplitude taken from shifted-sheared cell vs. radially adjacent cell (eq. A2-A3). Acts as mixing parameter.

### P5. Time integration + code identity

- Code: Sano+99 + Evans & Hawley 1988 CMoCCT (constrained-transport method-of-characteristics) for induction; Clarke 1996 Godunov + second-order.
- Azimuthal advection by Omega_eq handled with FARGO (Masset 2000; Benítez-Llambay & Masset 2016) to lift CFL constraint set by background super-Alfvénic azimuthal flow.
- CFL: standard Courant-Friedrichs-Lewy (Courant+28, referenced explicitly).
- Run to t_final = 300 rotations; quasi-saturated state after t ≈ 20 rotations; statistics averaged over t = 25 - 300 rotations (275 rotations of statistics).
- One rotation = 2 pi / Omega_{K,0} (convention, slightly shorter than true 2 pi / Omega_eq,0).

### P6. Parameter sweep

Three primary runs (Table 1):
1. Cylindrical H_0/R_0 = 0.1 (fiducial)
2. Cylindrical H_0/R_0 = 0.01 (near-Cartesian curvature)
3. Cartesian limit R_0 → infinity

Plus four boundary-sensitivity runs with varying f_{amp,±} (Appendix A, Table 4).

Single beta_{z,init} = 1e4 and single resolution (256 × 320 × 256).

### P7. Key diagnostics — exact formulas

- Maxwell stress (eq. 24): alpha_M = -<B_R B_phi>/(4 pi <p>); average over volume + time.
- Reynolds stress (eq. 25): alpha_R = <rho v_R dv_phi>/(4 pi <p>).
- Magnetic-energy variation (eqs. 21-23): shearing terms (i ⇒ j) and compressive terms (i ⇒ i) decompose dt(ln B_i^2):
  - (phi ⇒ R) = (B_R/(4 pi)) dphi(v_R B_phi - v_phi B_R) / (B_R^2/(8 pi))   — shear sourcing R
  - (R ⇒ phi) = -(B_phi/(4 pi)) dR(v_phi B_R) / (B_phi^2/(8 pi))   — winding sourcing phi (radial differential rotation)
  - (phi ⇒ phi) = -(B_phi/(4 pi)) dR(v_R B_phi) / (B_phi^2/(8 pi))   — compressive sourcing phi (THE KEY NEW TERM, set by κ gradient)
  - plus analogous terms for z and self-loops (i ⇒ i).
- Epicyclic frequency (eq. 26): kappa = sqrt(R dR(Omega^2) + 4 Omega^2) = Omega_eq sqrt(1 - 2 f_K). In Cartesian: spatially constant. In cylindrical: kappa ∝ R^{-3/2}.
- Radial displacement (eq. 27): xi_R(t) = integral dt <rho v_R>/<rho>.
- Viscous-type-instability criterion (Appendix B): for alpha_nu ∝ (rho_0/rho)^{q_rho} (B_z^2/B_{z,0}^2)^{q_B}, unstable when q_rho > 1 (eq. 31). Observed q_rho ≈ 1 in simulations — marginally unstable.
- Phase-mixing timescale (eq. 29): tau_comp = (20 pi/3)(H_0/R_0 / 0.1)^{-1} (Δphi_kappa / pi)^{-1} (ΔR/H_0)^{-1} rotations.

### P8. CRITICAL numerical values to match

From Table 3 (time-averaged, t = 25-300 rotations):

| Model | alpha_M | <B_R^2>/(8pi<p>) | <B_phi^2>/(8pi<p>) | <B_z^2>/(8pi<p>) | B_R^2/B_phi^2 | B_z^2/B_phi^2 | alpha_R |
|---|---|---|---|---|---|---|---|
| H/R = 0.1 | 6.22e-2 | 2.78e-2 | 1.11e-1 | 1.25e-2 | 0.250 | 0.112 | 3.36e-3 |
| H/R = 0.01 | 5.60e-2 | 2.51e-2 | 1.00e-1 | 1.14e-2 | 0.250 | 0.113 | 2.10e-3 |
| Cartesian | 7.16e-2 | 2.64e-2 | 1.29e-1 | 1.32e-2 | 0.204 | 0.102 | 1.56e-2 |

Key numerical targets:

- **Time-averaged alpha_M ≈ 0.06 across all three boxes** (similar, despite very different time-variability).
- **Reynolds-stress alpha_R one order lower in cylindrical** (3.4e-3 for H/R=0.1 vs. 1.56e-2 for Cartesian — cylindrical gives ~5× smaller Reynolds stress).
- **B_R^2/B_phi^2 = 0.25 in cylindrical cases, 0.20 in Cartesian** — cylindrical has relatively stronger poloidal compared to toroidal (cf. Fig. 4).
- **Peak burst alpha_M ≈ 0.3 in H_0/R_0 = 0.1**, vs. quiescent alpha_M < 0.05. Burst-to-quiescent ratio ≈ 6-10.
- **Burst period ≈ 50 rotations** (cylindrical 0.1 active phases), vs. typical ~10 rotation period of Cartesian channel-mode cycles. Authors predict tau_comp ≈ 33 rotations for ΔR ≈ 0.2 R_0 phase mismatch, roughly matching the observed cycle.
- **Triangle-diagram compressive amplification** (Fig. 4): [phi ⇒_R phi] = +6.42 (rotation^-1) in H/R=0.1 cylindrical vs. ‑1.05 in Cartesian — CHANGES SIGN because of radial kappa variation; this is the headline result.
- **Sum of B_phi^2 variation arrows in cylindrical = +0.21 (rotation^-1), +0.20 for B_z**; net positive because finite numerical dissipation removes energy. This number is the implicit numerical resistivity — useful for code calibration.
- Ring-gap spacing (Fig. 12): sub-H_0 scale features (~0.1 H_0 to H_0) appear continuously, anti-correlated density vs B_z^2. q_rho ≈ 1 scaling of alpha_M ∝ rho^{-q_rho}.
- Field-line geometry (Fig. 1): predominantly toroidal B_phi with significant fluctuation, tilt angle ~5° relative to phihat.
- Epicyclic oscillation: Omega_eq = 0.99 Omega_K for H_0/R_0 = 0.1 (eq. 9).

### P9. 1:1 reproduction checklist

- [ ] 3D ideal MHD solver in cylindrical coordinates (R, phi, z)
- [ ] Radial gravity only (no z-gravity)
- [ ] Locally-isothermal EOS with c_s^2 ∝ R^{-1}
- [ ] Constrained transport for div(B) = 0
- [ ] CMoCCT upwinding for induction (Sano+99 / Evans+Hawley 1988)
- [ ] FARGO advection for azimuthal Omega_eq transport
- [ ] Shearing-radial BC on conserved variables with curvature, including the separate mean/perturbation treatment (eq. 11-12; Appendix A)
- [ ] Amplitude-rescaling factors f_{amp,±}, f_sh in ghost cells
- [ ] Periodic BC in phi, z
- [ ] Initial weak uniform B_z ∝ R^{-1}, beta = 1e4
- [ ] Random MRI seed in v_R, v_phi
- [ ] Long run (300 rotations); statistics from 25-300

**This is a fully 3D run** (N_R × N_phi × N_z = 256 × 320 × 256). Not 2D, not 2.5D — radial, azimuthal, AND vertical extent are all resolved. Unstratified (no vertical gravity) but spatially 3D.

### P10. Gotchas

- Transient ~20 rotations to quasi-saturated state.
- Time-averaged Maxwell stress converges but TEMPORAL VARIABILITY does not — H_0/R_0 = 0.1 exhibits bursts of amplitude 0.3 with ~50 rotation recurrence; the three runs differ drastically in second moment even though first moment is similar. Statistics converge slowly.
- Reynolds stress alpha_R for cylindrical cases is called "not reliable" by author because definition dv_phi = v_phi - R Omega_eq is ambiguous when bulk rotation profile is self-consistently altered by bursts. Comparing alpha_M is preferred for cross-box comparison.
- Ghost-cell amplitude matching (f_{amp,±}) is hand-tuned; Appendix A Figure 19 shows that changing to (1.06, 0.96) or (1,1) gives modestly different accretion profile (gentle expansion vs. confined) but similar intermittency and time-averaged stresses — not hypersensitive but not parameter-free either.
- Vertical domain L_z = 4 H_0 is FOUR TIMES what S19 used — this paper criticises S19's L_z = H_0 as insufficient for channel-mode saturation.
- Ring-gap saturation level may depend on numerical resolution (Sec. 4.2 noted as future work); current 64/H_0 is "adequate" but not converged.
- Cartesian triangle-diagram shows compressive terms always negative ([y ⇒_x y] = -1.05, [x ⇒_y x] ≈ 0) — the cylindrical/Cartesian result sign-flip is genuine physics from radial kappa gradient, not a numerical artefact.
- Hall term, ambipolar diffusion, and resistivity are NOT included — ideal MHD only. Do not add them if reproducing.
- No stratification and no disc wind launching (unlike global disc sims). Pure local MRI with curvature.

---

End of extraction.
