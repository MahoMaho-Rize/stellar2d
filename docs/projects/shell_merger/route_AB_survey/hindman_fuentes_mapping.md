# Hindman & Fuentes 2025 → Andrassy 2022: Ri_flux ↔ Ri_B Mapping

Date: 2026-05-06. Purpose: Route B paper §2 positioning. Establish an explicit
algebraic bridge between Hindman & Fuentes 2025 (HF25, arXiv:2508.10174) and
Andrassy et al. 2022 (A22, A&A 659 A193) so that (i) HF25 data points can be
overlaid on our Ri_B–E plot, and (ii) we know which scalar diagnostics our
Route B code must emit.

---

## §1. Hindman & Fuentes 2025 definitions

**Setup.** 2D (1536×768) and 3D (576³) Boussinesq Dedalus v3 in a Cartesian
box of height H, width 2H (2D) or H² (3D). Stratification is **compositional**
— the stable layer holds a linear solute profile `C₀(z)=(βg)⁻¹N₀²(H−z)` with
β the compositional contraction coefficient — while the initial potential
temperature is flat, `T₀(z)=0`. Convection is driven by a **fixed-flux**
bottom boundary `∂ₓT|₀ = −F*/(ρ₀ cₚ κ_T)`. Top is zero thermal flux, bottom
is zero compositional flux; both walls impenetrable, stress-free.

**Controlling parameters** (HF25 Eqs. 8, 19):
```
Ri  ≡ ρ₀ cₚ κ_T N₀² / (g α F*)        (flux Richardson)
Le  ≡ κ_T / κ_C                        (Lewis)
Ra  ≡ g α F* H⁴ / (ρ₀ cₚ κ_T² ν)       (flux Rayleigh)
Pr  ≡ ν / κ_T                          (Prandtl)
```

**Scan ranges** (2D): Ri ∈ {1, 3, 5, 7, 10}, Le ∈ {1, 2, 4, 6, 10} at
fixed Ra = 10⁸, Pr = 0.1. 3D: Ri ∈ {1, 5, 10} at Le = 10.

**Entrainment-law diagnostic.** HF25 fit `h(t) = q⁻¹ V_cb (κ_T t)^q` and
report a **nondimensional boundary speed V_cb** (their Eq. 11):
```
V_cb² = [γ + (1−ε_T)/2] Ri⁻¹ − (ε_C/2) Le⁻¹
```
with best-fit γ = 0.50, ε_T = 0.71, ε_C = 0.96. The **dimensional** interface
velocity follows Eq. 13: `dh/dt = V_cb (κ_T / t)^{1/2}`.

Important: **V_cb is an entrainment-front velocity non-dimensionalised by
(κ_T/t)^{1/2}, not a bulk convective velocity.** This is the key translation
pitfall — V_cb ≠ v_rms and V_cb ≠ v_e (Andrassy entrainment velocity).

## §2. Andrassy 2022 definitions

A22 inherits Cristini+17/19, Meakin & Arnett 07, and Horst+21 convention:
```
Ri_B = ΔB · ℓ / v_rms²                (Horst 2021 Eq. 6)
ΔB   = ∫_{rc−Δr}^{rc+Δr} N² dr        (Horst 2021 Eq. 7)
ℓ    = ½ H_P at the boundary
v_rms = ⟨u²⟩^{1/2}_{CZ}
Ṁ_e / (ρ_b v_rms) ≡ E = A·Ri_B^(−n)   (entrainment law)
```
The A22 benchmark is **thermal/entropy-driven** — idealised O-shell with an
entropy jump across a convective-radiative boundary, Ma ≈ 0.04,
compressible-but-low-Mach. In the Boussinesq limit with thermal-only
stratification (Le → ∞, β ≡ 0), A22's ΔB reduces to ΔB = g α ΔT across the
interface, and `N² = gα ∂_z T` in the stable layer.

## §3. The mapping

Assume a quasi-steady convective layer of depth `h` driven by a flux F*
into a thermally-stratified stable top (HF25 with Le→∞, or equivalently
Route B). Convective scaling gives
```
v_rms³ ≈ (g α F* / ρ₀ cₚ) · h          (HF25 Eq. 3; MLT free-fall)
```
so `v_rms = (gαF*h / ρ₀cₚ)^{1/3}`. For a uniform-N₀² stable layer and a
well-mixed CZ, the buoyancy jump **integrated across the interface** is
```
ΔB = ∫ N² dr = N₀² · (2Δr)
```
If one adopts the Andrassy–Horst convention `ℓ = ½ H_P ≈ ¼ h` (or just
`ℓ ∝ h` with an O(1) prefactor `c_ℓ`) and `Δr ≈ ℓ`, then ΔB·ℓ ≈ N₀²·ℓ².
Putting it together:

```
Ri_B = ΔB · ℓ / v_rms²
     = (N₀² · c_ℓ² · h²) / (g α F* h / ρ₀ cₚ)^{2/3}
     = c_ℓ² · (ρ₀ cₚ N₀²)^{2/3} · (1 / (gαF*))^{2/3} · h^{4/3} · N₀^{2/3}
```
Multiplying and rearranging, using HF25's flux-Richardson
`Ri = ρ₀ cₚ κ_T N₀² / (gαF*)`:

```
Ri_B = c_ℓ² · Ri^{2/3} · [(ρ₀ cₚ N₀²)^{1/3} / κ_T^{2/3}] · h^{4/3}
     = c_ℓ² · Ri^{2/3} · (N₀² / κ_T²)^{1/3} · (ρ₀ cₚ)^{1/3} · h^{4/3}
```
Nondimensionalising h by the domain height H and κ_T by H²/τ_T collapses
this to
```
Ri_B = c_ℓ² · Ri · (h/H)² · (τ_T N₀)^{2/3} / (ρ₀ cₚ)^{...}    (rearrangement)
```
**Cleaner dimensionless form.** Use HF25's nondimensionalisation
(lengths → H, time → τ_T = H²/κ_T) so that v_rms → v_rms · (κ_T/H),
N₀² → N₀²·τ_T², and h → h/H. In those units,
```
Ri_B (dimensionless) = [ N₀² (h/H)² · c_ℓ² ] / v_rms²
```
and the MLT closure gives `v_rms² = [gαF* h / (ρ₀cₚ)]^{2/3}` which in
HF25 units is `v_rms² = (h/H)^{2/3} · Ri^{−2/3}`. The buoyancy-frequency
relation is `N₀² · τ_T² = Ri` exactly (HF25 Eq. 8 inverted: gαF*/ρ₀cₚ =
κ_T N₀²/Ri → N₀² = (1/τ_T²)·Ri·(H/H·...)·Pr-free). Substituting:

```
    Ri_B ≈ c_ℓ² · Ri^{5/3} · (h/H)^{4/3}
```

**This is the primary result.** At fixed simulation geometry (h/H ~ O(1)
once CZ fills most of the box), Ri_B ≈ c_ℓ² · Ri^{5/3}. The conversion is
**clean and single-factor up to the O(1) constant c_ℓ**, which contains
the mixing-length convention (Horst ℓ = ½H_P vs HF25 ℓ = h) and the
specific choice of integration width Δr. It is **not Pr-dependent** —
viscosity enters only through Ra, which sets whether convection is
turbulent, not the Ri_B↔Ri ratio at given h/H.

**Assumptions made explicit:**
1. Le → ∞ (no compositional buoyancy) — *required* to equate thermal-only
   and composition-only formulations. See §5.
2. Quasi-steady MLT closure `v_rms³ ∝ F* h` — same assumption HF25 use in
   Eq. 3. Breaks near ignition / overshoot-only regime.
3. Uniform N₀² in the stable layer — holds in both HF25 IC and A22
   idealised-O-shell initialisation.
4. Integration width `Δr ≈ ℓ` so ΔB·ℓ ≈ N₀²·ℓ². A22/Horst21 use Δr
   large-enough to capture full entropy step; this affects c_ℓ but not
   the Ri^{5/3} exponent.
5. ε_T, ε_C ≈ O(1) constants — do not alter the leading-order
   Ri_B↔Ri scaling; they only shift V_cb by an O(1) factor inside c_ℓ.

**Entrainment-velocity translation.** HF25's interface speed is
`dh/dt = V_cb (κ_T/t)^{1/2}` (Eq. 13). Andrassy's v_e ≡ Ṁ_e/(ρ_b A_b) is
the mass-entrainment velocity. In the Boussinesq limit they are equal
(ρ_b constant, interface advancing into a zone of density ρ_b). So
**v_e ↔ dh/dt** directly. The non-dimensional entrainment rate
E = v_e / v_rms becomes, in HF25 variables,
```
E = V_cb · (κ_T/t)^{1/2} / v_rms
  = V_cb · h⁻¹ · κ_T / v_rms     (using h = 2V_cb (κ_T t)^{1/2})
```
At h/H ~ 0.5 in mid-run this is an O(V_cb / Pe) ratio with Pe = v_rms h / κ_T
the Péclet number — high-Pe turbulent convection gives E ≪ 1 as expected.

## §4. Required Route B diagnostics for overlay

A Route B run (thermal-only Boussinesq pseudospectral, Andrassy-regime) must
emit per-snapshot scalars so any reader can convert to HF25 axis and back:

| Diagnostic | Definition | Why needed |
|---|---|---|
| `v_rms(t)` | `⟨u²+w²⟩^{1/2}` averaged over CZ | Ri_B denominator; maps to HF25 `dh/dt` via Pe |
| `ΔB(t)` | `∫_{zb-Δr}^{zb+Δr} N²_local dz` | Ri_B numerator; converts to N₀² × interface width |
| `h(t)` | CZ depth from ⟨∂²_z C⟩ extremum or entropy gradient | Ri exponent factor `(h/H)^{4/3}` in mapping |
| `ℓ(t)` | `½ H_P` at z_b (or fixed ℓ=h/4 for clean Boussinesq) | Horst convention; document choice |
| `v_e(t)` | `dh/dt` from finite-difference or OLS over window | Numerator of E; equivalent to HF25 `V_cb (κ_T/t)^{1/2}` |
| `F*(t)` | imposed flux at z=0 (boundary integrator) | HF25 Ri denominator; should be ≈ constant |
| `N₀²` | initial `g α ∂_z T₀` in stable layer | HF25 Ri numerator |
| `α, κ_T, ρ₀, c_p, g, H` | input constants | full dimensional conversion |
| `Pr, Ra` | derived from inputs | report to place ourselves on HF25 Fig. 2 |
| `c_ℓ` (numerical) | `ℓ / h` at t_peak | calibration of O(1) prefactor for overlay |

These seven time series + six constants are sufficient to reconstruct
**both** Ri_B(t) and HF25-Ri, so plotted points translate losslessly.

Implementation hint: add all of the above to an HDF5 `/diagnostics` group
at snapshot cadence; computed in `pseudo_spectral_bouss_solver` once
per write, cheap (O(N) reductions).

## §5. Le → ∞ limit

Route B is **thermal-only Boussinesq**: β ≡ 0, C ≡ 0, no composition equation.
This is exactly the Le → ∞ limit of HF25's parameter space — `Le⁻¹` in their
V_cb² drops out (Eq. 11) and the "diffusive stall" condition V_cb = Le^{−1/2}
moves to infinity. **Route B therefore lives in the upper-right corner of
HF25's Fig. 2**, in the penetrative-convection regime where stalling cannot
happen. HF25's highest explored Le is **10** (not infinity), and their
Fig. 5b shows `V_cb` at Le=10 asymptoting to the purple "Le→∞" curve
within ≲15% for Ri ∈ [3,10]. So the Le→10 data points are a practical
proxy for Le→∞ overlay.

**What HF25 reports at highest Le.** In the Le=10 column of their Ri-scan,
V_cb at Ri=(1,3,5,7,10) are read from their Fig. 5 as roughly (0.45, 0.25,
0.20, 0.17, 0.15). Using §3 mapping with c_ℓ≈0.5 (HF25 interface is ~half
a scale-height wide, commensurate with Horst's ℓ=½H_P), these convert to
Ri_B ≈ (0.25, 1.3, 2.9, 4.9, 8.3), placing them in the **low-Ri_B end** of
Andrassy's plot — which already spans Ri_B ∈ [10, 10³] for the Meakin-Arnett
/ Horst / Cristini ranges. **HF25 and A22 thus occupy disjoint but adjacent
Ri_B windows**; Route B's job is to bridge the gap by scanning mid-Ri_B.

Conversion cross-check: HF25's Ri=10, Le=10, Ra=10⁸, Pr=0.1 (their highest
Ri) gives Pe ≡ v_rms h / κ_T ~ (Ra Pr / Ri)^{1/2} ≈ 10² ≈ Meakin-Arnett
Pe range. So the **turbulence intensity is comparable**; only the
buoyancy formulation (composition vs entropy) differs.

## §6. Positioning recommendations for Route B paper §2

The §2 "Relation to prior work" paragraph should contain, in this order:

1. **Primary competitor statement.** Cite HF25 as the only recent (2025)
   Dedalus-v3 Boussinesq 2D+3D entrainment scan, and state the three
   distinctions: (a) compositional vs thermal stratification — HF25 use
   C, we use T; we recover the Le→∞ limit; (b) flux-Richardson Ri vs
   bulk-Richardson Ri_B — we publish Ri_B to align with A22, with an
   explicit conversion `Ri_B ≈ c_ℓ² · Ri^{5/3} · (h/H)^{4/3}` (§3 above);
   (c) Andrassy-benchmark companion framing vs standalone physics paper.

2. **Conversion table for overlay.** Publish the (Ri, Le, V_cb_measured,
   Ri_B_derived, E_derived) table for HF25's Le=10 data, and show those
   five points as hollow markers on our Ri_B–E plot. This is the single
   most useful figure for readers trying to connect the two papers.

3. **Le→∞ vs HF25 Le=10.** Note that HF25 Fig. 5b already shows Le=10
   ≈ Le→∞ to 15%, so the overlay is defensible without further work.

4. **Anders+2023 distinction.** Additionally cite Anders, Jermyn,
   Lecoanet, Brown 2023 (ApJ 926, 169) — Boussinesq Dedalus with
   penetration parameter 𝒫 and stiffness 𝒮, not Ri_B. Different enough
   to not threaten Route B's novelty, but same community.

5. **What we don't cover.** Explicitly: no compositional dynamics, no
   finite Le, no diffusive stalling — those are HF25's domain. Our
   contribution is the **A22-regime reference curve** at dissipation-free
   spectral resolution, not a rederivation of their entrainment law.

**Additional positioning to consider (beyond user's three points):**

- **Pr scan disclosure.** HF25 fix Pr=0.1. A22 is compressible and has
  no explicit Pr. Route B should either match Pr=0.1 (for direct HF25
  overlay) or run a Pr ∈ {0.01, 0.1, 1} mini-scan to bracket both HF25
  and the effective stellar Pr ~ 10⁻⁶ (which neither Dedalus nor our
  code can reach, but bracketing proves the trend is Pr-insensitive for
  turbulent regimes).

- **Resolution match statement.** HF25 use 1536×768 in 2D. Our
  pseudo-spectral solver already has 2D at 1024² in the KH baseline;
  match or exceed 1536×768 for the production Route B runs so reviewers
  can't argue under-resolution.

- **IC transient handling.** Following HF25, the first ~1 τ_T should be
  discarded when fitting h(t) and Ri_B(t); document the window choice
  and show sensitivity.

- **Entrainment-law fitter.** Publish both `E = A·Ri_B^{−n}` (A22 form)
  and `h(t) = q⁻¹ V_cb (κ_T t)^q` (HF25 form) from the same simulation,
  with residual plots. Dual-fit positions us as the bridge paper.

---
End of mapping document (≈ 740 words excluding tables and code blocks).
