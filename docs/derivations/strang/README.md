# Strang Derivations — sympy-driven symbolic manuscript

**Date**: 2026-05-08
**Target solver**: `src/gpu/explicit/strang_solver.{cu,cuh}` +
`strang_device.cuh`. 2D Cartesian Strang-split Euler solver —
MUSCL-Hancock predictor with the monotonised-central (MC) slope
limiter, an LM-HLLC Riemann solver at the interface, a perturbation-
variable storage $(\delta\rho, m_x, m_y, \delta E)$ on top of an
isentropic hydrostatic-equilibrium (HSE) background, and operator-
split gravity.

**Status**: **scope approved 2026-05-08**. Phase 1 in progress.

**Completeness target**: match the granularity of
`docs/derivations/mhd/` (33 sections). Final count is **36 sections**
across five parts. The section count is not a fixed target — it is
what the pointwise-strong-form derivation plus alternative-scheme
comparisons produce when pushed to full depth. Estimated effort:
**7-10 working days**.

---

## Core protocol

Every entry in this book obeys the five rules below.

### Rule 1 — Every algebraic identity is verified by sympy

Each section has a sibling script `scripts/<section>.py`. The script
ends with `assert_zero(expr, label)` calls that invoke
`sympy.simplify` and raise on non-zero residuals. Where sympy cannot
simplify a physically-correct identity (nested radicals, transcendental
closures), the script falls back to **numerical random sampling**:
generate ≥ 50 random states from the admissible domain and assert
`abs(residual) < 1e-10`. The fallback is annotated in the markdown as

> _Symbolic simplification intractable — verified by N=200 random
> sample points in the admissible domain (details in the script)._

### Rule 2 — Every script is independently runnable

A script imports only from `_common.py`. It does not reuse
intermediate results computed by any other section's script. Any
single section can be regenerated in isolation.

### Rule 3 — Every section ties back to code and tests

Each `sections/<name>.md` ends with a block

```markdown
> **code checkpoints:**
> `src/gpu/.../<file>.cu::<function>`
> `tests/test_strang_<feature>.cu`

## ✅ Verification checkpoint (to be wired)
{specific invariant the regression test must enforce}
```

The kernel is annotated with back-references of the form

```cpp
// Identity A5-S_star: see docs/derivations/strang/sections/a05_hllc_contact_speed.md :: eq:A5-SM
```

so that any kernel edit can be traced to the derivation section that
justifies it.

### Rule 4 — Strong-form equalities only; weak form as labelled fallback

All derivations are carried out in the **strong** (pointwise) form of
the governing PDEs. Equalities written in this book are of the form
$A(x,t) \equiv B(x,t)$ for every $(x,t)$ in the solution domain, not
$\int A \varphi \,dV = \int B \varphi \,dV$ for all test functions
$\varphi$.

Weak-form fallback is permitted **only** when no closed-form strong-
form identity exists:

1. Non-linear operator identities with no symbolic expansion (e.g.,
   BCH of the full non-linear Strang operators acting on non-smooth
   states).
2. Canonical problems with no closed-form solution (e.g., the
   Woodward-Colella two-shock reflection after wave interactions).
3. The finite-volume cell average $\bar U_i = V_i^{-1} \int_{V_i} U\,dV$
   itself, which is intrinsically an integrated object.

Whenever weak form is invoked the section must:

- Label the equation with `[WEAK]` in the derivation text;
- Include a sympy-driven numerical-consistency check at ≥ 50 random
  smooth states showing agreement to ≤ 10⁻¹⁰;
- State plainly, in one English sentence, **why** strong form is not
  available.

No section may invoke weak form for convenience if a strong-form
derivation exists.

### Rule 5 — Canonical-IC golden values are sympy-generated JSON, not in git

Part D sections dump analytic reference solutions to
`output/<section>.goldens.json`. These JSON files are listed in
`.gitignore`; regression tests (`tests/test_strang_*.cu`) invoke
`bash run_all.sh` as a build-time prerequisite and then read the JSON.
It is **forbidden** for a test source file to hand-compute a Sod
intermediate state, an entropy-wave amplitude, or a Riemann star-
region value — that derivation belongs in the script, not in the
test.

---

## Scope — five parts, 36 sections

The book has five parts. Part A is split into two sub-parts: A-phys
(Euler physics, 6 sections) and A-num (numerical schemes, 8 sections),
in line with Rule 4 requiring **alternative-scheme comparisons** at
full depth so that the book is a standalone numerical reference and
not just a one-kernel justification.

| Part | Scope | # sections |
|---|---|---|
| A-phys | Euler equations, conservative/primitive, flux Jacobian, eigensystem | 6 |
| A-num | Riemann solvers, slope limiters, reconstruction order, time splits | 8 |
| B | Perturbation storage, HSE background, ghost-cell BCs | 6 |
| C | Source terms, CFL, LM-HLLC blending, thermodynamic invariants | 4 |
| D | Canonical ICs with analytic golden values | 7 |
| E | Post-hoc benchmark derivations (scheme characterisation) | 5 |
| **Total** | | **36** |

### Part A-phys — Continuum physics (6 sections)

| # | section | content | code anchor |
|---|---|---|---|
| A1 | `a01_euler_equations.md` | Strong-form mass / momentum / total-energy conservation; flux tensor $F_i(U)$; energy identity $\partial_t E + \partial_i[(E+P) u_i] = 0$; positivity envelope $\rho > 0$, $P > 0$ | `d_euler_flux_{x,y}` |
| A2 | `a02_conservative_primitive.md` | Bijection $(\rho, m_x, m_y, E) \leftrightarrow (\rho, u, v, P)$; Jacobian determinants both directions; degeneracy at $\rho = 0$ floor; positivity preservation under smooth evolution | `d_cons2prim` |
| A3 | `a03_flux_jacobian_x.md` | $A_x = \partial F_x / \partial U$; 4 eigenvalues $\{u-c, u, u, u+c\}$; right/left eigenvector basis; completeness $R L = I$; entropy wave and shear wave share $\lambda = u$ (two-fold degeneracy) | used by A-num |
| A4 | `a04_rotational_covariance.md` | $A_y = R^{-1} A_x R$ with $R = \text{diag}(1, R_{90}, 1)$ the component-rotation matrix; flux rotation identity $F_y(U) = R^{-1} F_x(R U)$ in strong form; consequence: the y-sweep kernel can reuse the x-sweep Riemann solver with a coordinate swap | `k_hllc_update_y` arg order |
| A5 | `a05_entropy_condition.md` | Entropy function $\eta = -\rho s$ with $s = \log(P / \rho^\gamma)$; entropy inequality $\partial_t \eta + \partial_i(\eta u_i) \le 0$ with strict equality on smooth states; Lax entropy condition across each characteristic family | used by A9 (Godunov scheme proof) |
| A6 | `a06_smooth_wave_families.md` | Simple-wave solutions in each characteristic family: genuinely non-linear ($\lambda = u \pm c$) admit rarefactions or shocks; linearly degenerate ($\lambda = u$) admit contact discontinuities only; Riemann invariants $u \pm \frac{2c}{\gamma-1}$ explicit forms | used by A7, D3 |

### Part A-num — Numerical schemes (8 sections)

All four alternative-comparison sections (A-alt1..4) are embedded
here. They are **not optional** — the book is a numerical reference,
so every scheme used or plausibly-relevant at the stellar2d Strang
solver's scope must be derived and cross-compared to the one the
kernel actually uses.

| # | section | content | code anchor |
|---|---|---|---|
| A7 | `a07_riemann_solver_family.md` | **[alt1]** Rusanov / HLLE / HLLC / Roe Riemann solvers derived from the same template (L/R state, flux integral between waves). sympy-proved contact-wave resolution: Rusanov and HLLE smear the contact, HLLC and Roe resolve it exactly in the linear limit; HLLC's relation to Roe when the star region is a contact discontinuity only | `d_lmhllc` (HLLC branch) |
| A8 | `a08_hllc_intermediate_states.md` | HLLC contact speed $S_\star = \frac{p_R - p_L + \rho_L u_L(S_L - u_L) - \rho_R u_R(S_R - u_R)}{\rho_L (S_L - u_L) - \rho_R (S_R - u_R)}$; star pressure $p^\star$ via Toro 10.26; $U_L^\star, U_R^\star$ full conservative forms; consistency $S_L \le S_\star \le S_R$ under admissibility | `d_lmhllc` S* and U* blocks |
| A9 | `a09_wave_speed_estimates.md` | Davis (1988) $S_L = \min(u_L - c_L, u_R - c_R)$, $S_R = \max(u_L + c_L, u_R + c_R)$; Einfeldt entropy fix near sonic points; Roe-average alternative (state-dependent); sympy-proved that Davis bounds Roe conservatively on any admissible pair | `d_lmhllc` $S_L, S_R$ block |
| A10 | `a10_slope_limiter_family.md` | **[alt2]** MC / minmod / van Leer / superbee / Ospre limiters written in Sweby form $\phi(r)$ of the slope ratio $r$. sympy-proved TVD regions for each; MC's symmetry $\phi(1/r) = \phi(r)/r$; comparison chart of limiter aggressiveness vs. TVD strictness; MC chosen for its interior-to-TVD envelope | `d_mc_limit` |
| A11 | `a11_reconstruction_order.md` | **[alt4]** Donor-cell (1st) / MUSCL (2nd) / PPM Colella-Woodward (3rd) / PPM Colella-Sekora (3rd with extrema detection). sympy-derived leading truncation on linear advection; stencil width vs. order trade-off; why MUSCL is preferred for hyperbolic Godunov over PPM when limiter activity is frequent | `k_muscl_hancock_{x,y}` |
| A12 | `a12_muscl_hancock_halfstep.md` | Hancock predictor $U^{n+1/2}_{L/R} = U^n_{L/R} - \frac{\Delta t}{2\Delta x}[F(U^n_R) - F(U^n_L)]$; strong-form proof that on smooth states $U^{n+1/2}$ is 2nd-order in both $\Delta x$ and $\Delta t$; positivity guarantee conditions (ratio of limiter output to CFL) | `k_muscl_hancock_{x,y}` |
| A13 | `a13_time_integrator_family.md` | **[alt3]** Strang split / Lie split / unsplit VL2 (Stone-Gardiner) / RK2-MUSCL. sympy expansion of $e^{\Delta t(X+Y)}$ via BCH on **linear** $X, Y$ (strong form); Strang gives $O(\Delta t^3)$, Lie gives $O(\Delta t^2)$, unsplit VL2 gives $O(\Delta t^3)$ by symmetry. Non-linear Strang order: **[WEAK]** verified numerically on smooth Euler IC | `StrangSolver::step` |
| A14 | `a14_strang_operator_chain.md` | The `step` pipeline $X(\Delta t/2) \, Y(\Delta t/2) \, Y(\Delta t/2) \, X(\Delta t/2)$ equivalence to the canonical $X(\Delta t/2) \, Y(\Delta t) \, X(\Delta t/2)$; proof of symmetry-operator identity $L(\Delta t) L(-\Delta t) = I$; self-adjointness implies order preservation | `StrangSolver::step` |

### Part B — Storage, HSE, boundary conditions (6 sections)

| # | section | content | code anchor |
|---|---|---|---|
| B1 | `b01_perturbation_storage.md` | $(\delta\rho, m_x, m_y, \delta E) \leftrightarrow (\rho, u, v, P)$; momentum stored in full (bg = 0) assumes static HSE; the transformation is strong-form pointwise; consequence: any code that reads $\rho$ must add $\bar\rho(y)$ back before arithmetic | `d_cons2prim`, `k_muscl_hancock_y` |
| B2 | `b02_isentropic_hse.md` | Strong-form ODE $dp/dy = -\rho g$ with $p = K \rho^\gamma$; closed-form solution $\rho(y) = [\rho_0^{\gamma-1} - (\gamma-1) g y / (\gamma K)]^{1/(\gamma-1)}$; atmosphere cut-off $y^\star = \gamma K \rho_0^{\gamma-1} / ((\gamma-1)g)$; dump `b02_hse_profile.goldens.json` for test consumption | `StrangSolver::init` HSE build-up |
| B3 | `b03_face_hse_reconstruction.md` | Using $\bar\rho(y_{\text{face}})$ and $\bar p(y_{\text{face}})$ at the face (not cell centre) in y-sweep MUSCL reconstruction is a **necessary condition** for well-balancing; strong-form proof: on pure HSE ($\delta\rho = \delta P = u = v = 0$) reconstructed L/R face states are algebraically identical, hence the HLLC flux jump vanishes | `k_muscl_hancock_y` face reconstruction |
| B4 | `b04_periodic_x_bc.md` | Periodic-x ghost-cell copy; minimal width $n_g = 2$ for MUSCL; BC applied to both $U$ and face states `wL, wR`; strong-form identity $U_{i<n_g} \equiv U_{i+n_x}$ | `k_ghost_x`, `k_ghost_face_x` |
| B5 | `b05_reflective_y_bc.md` | Bottom reflective: $(\rho, m_x, E) \mapsto (+1)$, $m_y \mapsto (-1)$; strong-form flux-reversal identity $G^y_i(\mathcal{R}U) = \mathcal{R} G^y_i(U)$ with $\mathcal{R} = \text{diag}(1,1,-1,1)$; consequence: normal momentum flux at the wall is exactly zero; reflected HSE remains HSE | `k_ghost_y`, `k_ghost_face_y` bottom branch |
| B6 | `b06_outflow_y_bc.md` | Top outflow: zero-gradient $U_{\text{ghost}} = U_{\text{last phys}}$; characteristic analysis of subsonic outflow (incoming characteristic set to linearly extrapolated); what happens under supersonic outflow (all characteristics out, trivially consistent); comparison with characteristic BC | `k_ghost_y`, `k_ghost_face_y` top branch |

### Part C — Source terms, CFL, LM-HLLC, thermodynamic invariants (4 sections)

| # | section | content | code anchor |
|---|---|---|---|
| C1 | `c01_gravity_source_consistency.md` | $S_{m_y} = -\rho_{\text{tot}} g$, $S_E = -m_y g$; strong-form work identity $\frac{d}{dt}[\text{KE} + \rho g y] = 0$ on the body-force balance; sympy shows that adding a $\rho v \cdot g$ term would double-count work (pre-empts the P32 lowmach-family bug) | `k_hllc_update_y` S_my / S_E |
| C2 | `c02_cfl_bound.md` | Strong-form CFL: $\Delta t \cdot \max\{(|u|+c)/\Delta x + (|v|+c)/\Delta y\} \le \sigma$; linear von-Neumann analysis gives $\sigma = 1$ for both LF and HLLC; Strang-split 2D relaxes the 2D constraint to the max of 1D constraints; the kernel uses $\sigma = 0.4$ | `k_strang_cfl` |
| C3 | `c03_lm_hllc_blending.md` | LM-HLLC pressure-jump blend: $f_M = \text{clamp}(M_{\text{loc}}, M_{\text{cut}}, 1)$, $M_{\text{loc}} = (|u_L| + |u_R|)/(c_L + c_R)$, $M_{\text{cut}} = 10^{-3}$; substitution is $p_R - p_L \to f_M (p_R - p_L)$ **in $S_\star$ only**, not in $p^\star$; sympy-proved reduction to standard HLLC at $M=1$; dispersion analysis in the $M \to 0$ limit shows the pressure-jump contribution is suppressed to $O(f_M)$ | `d_lmhllc` fM block |
| C4 | `c04_entropy_invariant.md` | Smooth-flow entropy invariant: $D_t s = 0$ with $s = P \rho^{-\gamma}$. Strong-form derivation: $D_t (P \rho^{-\gamma}) = $ mass + momentum + energy conservation substituted and simplified to $0$; identity fails at shocks (diffusive by design) | `write_vtk` entropy diagnostic |

### Part D — Canonical IC and analytic goldens (7 sections)

Each script emits `output/d*_goldens.json` (not committed per Rule 5).

| # | section | content | test consumer |
|---|---|---|---|
| D1 | `d01_entropy_wave.md` | $\rho(x,t) = \rho_0 + A \sin(k(x - u_0 t))$, $u = u_0$, $P = P_0$. HLLC degenerates to upwind ($\Delta P = 0$ at every face, $S_\star = u$). Dump $\rho(x)$ at $N=4096$ grid points at $t = L_x / u_0$ | `test_strang_convergence.cu` |
| D2 | `d02_acoustic_linwave.md` | Right-going acoustic mode via A3 right-eigenvector projection: $\delta\rho = A\rho_0 \sin(kx)$, $\delta u = Ac_0 \sin(kx)$, $\delta P = A\gamma P_0 \sin(kx)$. Exact evolution at $t = T = L_x/c_0$: identity. Dump: all 4 fields sampled at $N=4096$ | `test_strang_linwave_convergence.cu` |
| D3 | `d03_sod_shock_tube.md` | Toro's Sod: closed-form intermediate state via Newton-iterated $p^\star$ (sympy-traceable); star quantities, rarefaction head/tail, contact and shock speeds; reference profile at $N=200$ samples in $-0.5 \le x \le 0.5$ at $t = 0.2$ | new `test_strang_sod.cu` |
| D4 | `d04_woodward_colella_blast.md` | Early-time window $t < t_{\text{refl}}$: three independent Riemann problems (at $x=0.1, x=0.9$) with closed-form solutions via A8 star algebra — sympy gives strong-form golden values in this window. Late-time $t = 0.038$ (after reflections and wave interactions): **[WEAK]** — no closed-form; dump `[WEAK]` caveat and a symbolic-early-time-only oracle. Comparison modality: integrated L1 over the reflected window | new `test_strang_wc_blast.cu` |
| D5 | `d05_bubble_init.md` | `init_bubble`: radial entropy boost $\delta s \cdot \exp(-(r/R_0)^2)$ with optional $k$-mode azimuthal perturbation $\varepsilon \cos(k\theta)$; isentropic derivation of $\delta\rho$ from $\delta s$ at constant $P_{\text{bg}}$; $\delta E = 0$ identity from $P = P_{\text{bg}}$; dump canonical IC profile | `test_strang_init.cu` |
| D6 | `d06_hse_zero_perturbation_lock.md` | With $\delta U \equiv 0$, the Strang step preserves the state to machine precision; strong-form proof via B3; round-off upper bound $\varepsilon_{\text{mach}} \cdot N_{\text{step}}$ (accumulation) or $\varepsilon_{\text{mach}} \cdot N_{\text{step}}^{1/2}$ (Kahan) | `test_strang_step.cu` §1 |
| D7 | `d07_reflection_symmetric_ic.md` | IC that satisfies $U(x,y) = \mathcal{R} U(x, -y)$ evolves into $U(x,y,t) = \mathcal{R} U(x, -y, t)$ at all $t$; strong-form proof via B5 + operator-split symmetry; **implies `init_rt_symmetric` must be added to the solver** (per user rule: book is the anchor, solver follows) | new `test_strang_reflection_symmetry.cu` |

### Part E — Post-hoc benchmark / scheme characterisation (5 sections)

Mirrors the F-series of `docs/derivations/mhd/`. Predictions are
derived from the preceding parts so that measurements are compared
against theoretical expectations, not rubber-stamped.

| # | section | prediction | measured by |
|---|---|---|---|
| E1 | `e01_entropy_wave_order.md` | Modified-equation analysis of Hancock-MC on smooth IC: leading truncation is $O(\Delta x^2) + O(\Delta t^2)$; CFL $\sigma = 0.4$ puts both in lock-step; predicted slope $p = 2.0 \pm 0.1$ | `test_strang_convergence.cu` |
| E2 | `e02_linwave_lm_hllc_order.md` | With `use_lm_fix = true`, LM-HLLC suppresses the pressure dissipation that drives acoustic damping; amplitude retention becomes machine-order (artificially super-converged); sympy dispersion in $M \to 0$ proves this. Conclusion: `use_lm_fix = false` is physically mandatory for acoustic convergence tests | `test_strang_linwave_convergence.cu` (must flip flag) |
| E3 | `e03_lm_hllc_nu_eff.md` | Effective numerical viscosity under LM-HLLC: $\nu_{\text{eff}} \approx \Delta x \cdot c \cdot f_M / 2$; at $M = M_{\text{cut}}$ this gives $\nu_{\text{eff}} \approx 5 \times 10^{-4} c \Delta x$; compared to the standard-HLLC floor $\nu_{\text{eff}} \approx c \Delta x / 2$ this is $\sim 10^3 \times$ smaller | new scheme-char probe |
| E4 | `e04_strang_split_source_commutator.md` | The gravity source is absorbed inside $Y$; the true commutator is $[X, Y_{\text{hydro}} + Y_{\text{grav}}]$; BCH expansion on linear $X, Y$ gives leading error $O(\Delta t^3)$ as long as the outer symmetry is preserved; proof that splitting gravity as a third operator would degrade to $O(\Delta t)$ | `StrangSolver::step` |
| E5 | `e05_hse_drift_bound.md` | Long-time HSE drift: $\max_t \|\delta U\|_\infty \le \varepsilon_{\text{mach}} \cdot N_{\text{step}} \cdot (1 + \kappa(\text{HSE flux}))$ where $\kappa$ is the condition number of B3's face-reconstruction identity; dump numerical $\kappa$ at canonical HSE | `test_strang_step.cu` §1 long-time |

---

## File layout

```
docs/derivations/strang/
├── README.md                       (this file)
├── run_all.sh                      run every scripts/*.py (required before ctest)
├── build_manuscript.sh             assemble sections/*.md → manuscript.{md,pdf}
├── .gitignore                      d*_goldens.json  (per Rule 5)
├── scripts/
│   ├── _common.py                  hydro symbol table + helpers
│   ├── a01_euler_equations.py
│   ├── a02_conservative_primitive.py
│   ├── ...
│   └── e05_hse_drift_bound.py      (36 scripts total)
├── sections/
│   ├── 00_preamble.md
│   ├── a01_euler_equations.md
│   ├── ...
│   └── e05_hse_drift_bound.md
└── output/
    ├── *.latex.tex                 sympy-dumped LaTeX snippets (committed)
    ├── *.log                       run_all.sh logs (committed)
    └── d*_goldens.json             golden values (NOT committed)
```

---

## Regression-test reorganisation

| test file | current state | wired to |
|---|---|---|
| `test_strang_init.cu` | 5 ad-hoc sub-tests | §B2 HSE profile, §D5 bubble IC, §D6 HSE zero-lock |
| `test_strang_muscl.cu` | MC limiter + Hancock sanity | §A10 MC TVD identities, §A12 Hancock identity |
| `test_strang_hllc.cu` | HLLC + symmetry + HSE column | §A8 intermediate states, §A9 wave speeds, §C1 HSE flux column |
| `test_strang_unit.cu` | cons2prim + ghosts | §A2 bijection, §B4/B5/B6 ghost-cell BCs |
| `test_strang_step.cu` | HSE + bubble + CFL | §D6 HSE lock + §D5 bubble + §C2 CFL + §E5 drift bound |
| `test_strang_convergence.cu` | entropy wave L1 slope | §D1 goldens + §E1 slope |
| `test_strang_linwave_convergence.cu` | acoustic linwave L1 | §D2 goldens + §E2 (`use_lm_fix = false`) |

**New test files created by this book**:

- `tests/test_strang_sod.cu` — §D3 Sod convergence (reads d03 goldens)
- `tests/test_strang_wc_blast.cu` — §D4 Woodward-Colella (integrated L1, early-window pointwise)
- `tests/test_strang_reflection_symmetry.cu` — §D7 bit-reproducibility

**Solver changes driven by book** (per user rule: book anchors,
solver follows):

- `StrangSolver::init_woodward_colella()` (needed by D4)
- `StrangSolver::init_rt_symmetric()` (needed by D7)

---

## Execution order

1. **Phase 1 — scaffolding** (0.5 day): `scripts/_common.py` with
   hydro-specific symbol table; thin `run_all.sh` / `build_manuscript.sh`
   cloned from MHD; `.gitignore` for d*_goldens.json.
2. **Phase 2 — Part A-phys** (1 day): 6 sections, A1..A6.
3. **Phase 3 — Part A-num** (3 days): 8 sections, A7..A14. Longest
   phase because alt-scheme comparisons (A7, A10, A11, A13) are each
   a complete side-by-side derivation.
4. **Phase 4 — Parts B + C** (1.5 days): 10 sections.
5. **Phase 5 — Part D** (1.5 days): 7 sections, each emits goldens.
6. **Phase 6 — Part E** (1 day): 5 sections.
7. **Phase 7 — test rewire + new IC** (1-2 days, separate PR):
   pull goldens from JSON, add init_wc / init_rt_symmetric.

**Exit criterion**: `bash run_all.sh` reports 36/36 green;
`manuscript.pdf` builds; `ctest -L fast` under strang passes with
every test sourcing goldens via JSON rather than inline constants;
no residual hand-computed analytic expressions in `tests/test_strang_*.cu`;
new IC builders land in `strang_solver.cu` with book-anchored
comments.

---

## Non-goals

- **Do not rewrite the strang solver.** CLAUDE.md rule #1. The book
  is retrofit plus forward-driven: new IC builders added if the book
  needs them (D4, D7), but existing kernel kept intact.
- **No physics extensions** (magnetism, radiation, non-ideal dissipation).
- **No 3D extensions.**
- **No implicit-time-integration derivations.** That is radial1d's book.

---

Phase 1 starts immediately after this README lands.
