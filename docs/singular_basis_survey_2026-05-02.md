---
title: "Spectral methods for singular boundaries: a survey"
subtitle: "Comparing GYRE (stellar pulsation) and Dedalus v3 (general spectral) against the Liouville SL-anelastic design"
author: "stellar2d project / Kiriko"
date: "2026-05-02"
---

# 1. Background and problem statement

## 1.1 Motivation

This project's `pseudo_spectral` solver currently handles uniform-density 2D
incompressible Navier--Stokes on a doubly periodic domain. Extending it to
**anelastic stellar convection** promotes the pressure equation from a
constant-coefficient Laplace operator to a variable-coefficient elliptic
equation:

$$\nabla \cdot \left(\frac{1}{\rho_0(y)} \nabla p\right) = f(x,y)$$

The original design (`docs/anelastic_SL_spectral_design.md`, 2026-05-01)
proposes a Liouville normal-form substitution $\hat p = \sqrt{\rho_0}\, q$
to reduce the variable-coefficient equation to a Schrodinger-type form:

$$q'' + W(y)\, q - k_x^2\, q = g, \qquad
  W(y) = \frac{\rho_0''}{2\rho_0} - \frac{3(\rho_0')^2}{4\rho_0^2}$$

Solving the eigenproblem for $[d^2/dy^2 + W]$ yields a Sturm--Liouville (SL)
basis $(\mu_n, \psi_n)$; all $k_x$ modes share the same basis, and Poisson
inversion reduces to a per-mode scalar division.

## 1.2 The unresolved problem: the $\rho \to 0$ singularity

Phase 0 validation (`docs/anelastic_sl_phase0_2026-05-02.md`) shows:

- The Lane--Emden $n=1.5$ surface $\rho \to 0$ drives $|W| \to 1.35\times 10^6$
  (divergent).
- Truncation to $\rho > 0.01$ (i.e. $r < 0.94$) makes the eigenproblem solvable.
- SL--Poisson end-to-end on a manufactured solution: $\mathrm{err}_{L^2} = 3.7\times 10^{-6}$.
- **Convergence order $\sim -2.4$ (algebraic)**, not the expected exponential.
- Phase 0 E3 confirms: a smooth Gaussian $\rho(y)$ restores exponential convergence.
  The algebraic rate is therefore entirely driven by the singular boundary.

**Core observation.** The Liouville substitution does not eliminate the
$\rho \to 0$ singularity; it merely transports it from a variable-coefficient
elliptic equation to a Schrodinger potential. We still need to address it.

## 1.3 Purpose of this survey

The quantum-mechanics community (Coulomb potentials) and the stellar-pulsation
community (adiabatic oscillations) have accumulated 60 years of standard
techniques for exactly this class of singularity. This report surveys two
representative open-source codes:

1. **GYRE** (Townsend 2013; Fortran)
   - Astrophysical stellar pulsation (adiabatic oscillations).
   - Structurally isomorphic to our problem: a radial SL eigenproblem inside
     a star.
   - Repo: <https://github.com/rhdtownsend/gyre>

2. **Dedalus v3** (Burns et al. 2020; Python)
   - General-purpose spectral PDE framework.
   - Covers Chebyshev / Jacobi / Ball / Disk bases.
   - Repo: <https://github.com/DedalusProject/dedalus>

The central question: **how do these two communities treat $\rho \to 0$-type
boundary singularities, and what can this project borrow?**

# 2. GYRE's strategy

## 2.1 Overall philosophy: **pre-designed variable transformations**

GYRE does not use a Liouville $\sqrt{\rho}$ substitution. It employs a
**non-dimensional geometric substitution**:

$$y_1 = x^{2-\ell}\cdot \frac{\xi_r}{r}, \qquad
  y_2 = x^{2-\ell}\cdot \frac{P'}{\rho g r}$$

where $\xi_r$ is the radial displacement and $P'$ is the pressure perturbation.

**Key point.** This substitution absorbs geometric singularities (origin $r=0$
and surface $\rho \to 0$) **into the definition of the evolved variables**, so
that $y_1, y_2$ remain bounded throughout the radial domain.

*Reference:* `docs/source/ref-guide/osc-equations/dimless-form.rst:24-32`

## 2.2 Surface boundary: analytic asymptotic form

GYRE does not impose Dirichlet or Neumann conditions at the surface. It uses
**physically motivated asymptotic boundary conditions**:

- **DZIEM mode** (Dziembowski 1971):
  $$\left\{1 + \frac{1}{V}[\ldots]\right\} y_1 - y_2 + \ldots = 0$$
- **UNNO mode** (Unno et al. 1989): Eddington approximation.
- **VACUUM mode**: minimal form, $y_2 = 0$.
- **JCD mode**: Jorgensen-Christensen-Dalsgaard detailed form.

*References:* `src/eqns/rad/rad_obound_m.fypp:102-133`,
`src/eqns/rad/gyre/OB_dziem.inc:1-3`

These conditions **all assume a physical limit exists at the surface**, rather
than a divergence. This is the deep reason GYRE can sidestep the $\rho \to 0$
singularity: its non-dimensional variables $y_1, y_2$ are finite at the surface
regardless of what $W$ is doing.

## 2.3 Grid refinement: ODE-eigenvalue criteria

`src/grid/grid_refine_m.fypp` implements adaptive log-grid bisection. The
refinement criterion is driven by ODE eigenvalues $\chi$:

```fortran
! Mechanical / thermal / structural criteria combined
split(j) = split_mech_(...) .OR. split_therm_(...) .OR. split_struct_(...)
```

In $\ln(x)$ space, `dlnx * |chi_imag|` (wavelength criterion) and
`|chi_real|^{-1}` (decay length criterion) are evaluated, so **grid points are
automatically added where oscillations or decay are rapid**.

*References:* `src/grid/grid_refine_m.fypp:78-80, 214-226`

## 2.4 Polytropic model treatment

The Lane--Emden solver uses a **fourth-order Taylor series + LSODAR event
detection**:

```fortran
X_BEG = sqrt(EPSILON(0._RD))   ! ~ 1e-8
y_exp(1) = 1 - x_exp**2/6 + n_poly*x_exp**4/120   ! theta(z) Taylor expansion
```

The key trick: the Lane--Emden RHS has a $1/x^2$ factor that looks singular at
$x=0$, but using $\theta \propto 1 - z^2/6$ near zero makes it finite.

*Reference:* `src/poly/lane_emden_m.fypp:59,109-110`

## 2.5 Discretization: Magnus exponential integrator

No Chebyshev or FEM. GYRE uses **Gauss--Legendre Magnus integrators** (GL2 /
GL4 / GL6):

```fortran
! Diagonalize the ODE as dy_i/dx = lambda_i * y_i, then integrate exponentially
```

*Reference:* `src/diff/magnus_gl2_diff_m.fypp:20-56`

## 2.6 Lessons from GYRE

| Technique | Portability to this project |
|---|---|
| **Pre-designed non-dimensional variables** that absorb singularity | $\star\star\star$ Most central; worth adopting |
| Analytic asymptotic BC (Dziembowski) | $\star\star$ Useful reference; requires physical insight |
| ODE-eigenvalue adaptive grid | $\star$ Complex; not urgent |
| Magnus exponential integrator | -- Orthogonal to spectral methods |

# 3. Dedalus v3's strategy

## 3.1 Overall philosophy: **weighted Jacobi bases**

All of Dedalus's radial bases belong to the Jacobi family $J_n^{(a,b)}(x)$.
The basis functions carry an intrinsic weight $(1-x)^a (1+x)^b$.

**Key insight.** Endpoint singularities can be **fully encoded in the weight
exponents $a, b$**.

- `ChebyshevT`: $a=b=-1/2$, absorbs $(1\pm x)^{-1/2}$ singularities.
- `Ultraspherical(alpha)`: $a=b=\alpha - 1/2$, tunable.
- `Jacobi(a,b)`: arbitrary $(a,b)$.

*Reference:* `dedalus/core/basis.py:435-515`

## 3.2 Ball/Disk basis: regularity at $r=0$

`BallBasis` and `DiskBasis` use a **k parameter** to encode an $r^k$ embedding:

```python
# BallRadialBasis.__init__(..., k=2):
#   basis functions automatically carry an r^k factor
#   Zernike polynomials * r^(alpha + k + ell + regtotal)
```

`regularity_allowed()` automatically filters out physically disallowed
$(\ell, m, \text{regularity})$ combinations.

*Reference:* `dedalus/core/basis.py:3917-4087, 3531-3594`

## 3.3 Boundary conditions: tau / lift method

Instead of Dirichlet, Dedalus uses a lift method:

```python
problem.add_equation("lap(f) + lift(tau) = -f**n")
problem.add_equation("f(r=1) = 0")   # the tau term absorbs boundary residual
```

`LiftJacobi.build_polynomial()` picks a high-order polynomial $\phi_N$; the
boundary residual is projected onto it, avoiding pollution of the eigen-space.

*References:* `dedalus/core/basis.py:790-815`,
`examples/nlbvp_ball_lane_emden/lane_emden.py:67-71`

## 3.4 Variable-coefficient operator matrices

`operator_matrix()` pre-computes variable-coefficient operators on Zernike /
Jacobi bases, including $r^{-k}$ singular operators, without manual
regularization:

*Reference:* `dedalus/core/basis.py:4044-4056`

## 3.5 Things Dedalus does **not** do

- No Kosloff--Tal-Ezer (coordinate stretching) mapping.
- No Liouville $\sqrt{\rho}$ substitution.
- No explicit cutoff.

Everything relies on the Jacobi weight absorbing singular behavior.

## 3.6 Lessons from Dedalus

| Technique | Portability to this project |
|---|---|
| **Jacobi basis with chosen $(a,b)$ weights** | $\star\star\star$ Core idea; structurally similar to GYRE's substitution |
| $r^k$ embedding for Ball/Disk | $\star\star$ Not immediately relevant (we're 2D plane) |
| Tau / lift method for BC | $\star\star$ Cleaner than FD Dirichlet |
| Variable-coefficient operator caching | $\star$ Engineering; later phase |

# 4. Comparison: SL vs GYRE vs Dedalus

## 4.1 Three different philosophies for "variable coefficient + singular boundary"

| Aspect | Project SL (current) | GYRE | Dedalus |
|---|---|---|---|
| Discretization | Interior FD | Magnus exponential | Jacobi spectral |
| Variable substitution | $\hat p = \sqrt{\rho}\, q$ | $y_1 = x^{2-\ell}\xi/r$ | None (basis carries weight) |
| Singularity absorption | **None; truncate $\rho > 0.01$** | Variable definition absorbs | $(1-x)^a (1+x)^b$ absorbs |
| Boundary condition | Dirichlet (interior FD) | Analytic asymptotic (Dziembowski) | Tau / lift |
| Surface singularity | **Regexp to truncation** | Variables finite at surface | Basis handles intrinsically |
| Convergence order | Algebraic $\sim -2.4$ | Analytic (Magnus high-order) | Exponential |

## 4.2 Why is the Liouville $\sqrt{\rho}$ substitution not enough?

The Liouville substitution converts
$$\frac{d}{dy}\left[\frac{1}{\rho}\frac{dp}{dy}\right] = f$$
into
$$q'' + W q = g,$$

but $W = \rho''/(2\rho) - 3(\rho')^2/(4\rho^2)$ **still diverges as $\rho \to 0$**.
We have only moved the singularity; we have not destroyed it.

GYRE and Dedalus instead **ensure no singular term ever appears** in the
discretized operator:

- GYRE: the definitions $y_1 = x^{2-\ell}\xi/r$ and $y_2 = x^{2-\ell}P'/(\rho gr)$
  force the entire LHS to be finite at the surface.
- Dedalus: Jacobi basis $J_n^{(\alpha,\beta)}(x)$ carries $(1-x)^\alpha$
  factors, and if $\alpha$ matches the decay exponent of $\rho$, the
  variable-coefficient ODE is analytically solvable in the weighted inner
  product.

# 5. Three possible paths forward for this project

## 5.1 Path A: **keep Liouville SL, add a GYRE-style substitution**

*Motivation.* Preserve the SL selling point: one set of $(\mu_n, \psi_n)$
serving both Poisson inversion and the g-mode spectrum.

*How.* In addition to $\hat p = \sqrt{\rho}\, q$, add a second substitution
$\tilde\phi = r^\beta\, \phi$, choosing $\beta$ so that the induced operator
$\tilde W$ remains finite as $\rho \to 0$.

*Pros.* Physical elegance preserved; the g-mode spectrum still emerges
automatically as an eigenspectrum of the basis.

*Cons.* Derivation-heavy; requires careful verification that $\tilde W$ is
smooth.

## 5.2 Path B: **switch to Jacobi--Galerkin**

*Motivation.* The Dedalus approach is numerically validated by our own Phase 0
E3 (smooth Gaussian $\rho$ restores exponential convergence).

*How.* Drop the Liouville substitution and work directly in a Jacobi basis
$J_n^{(\alpha,\beta)}(y)$. Pick $\alpha = 1.5$ to match the Lane--Emden $n=1.5$
surface decay exponent.

*Pros.* Simplest engineering; Dedalus can be imported as a library for
validation.

*Cons.* The g-mode spectrum now requires a **separate** generalized eigenproblem
$L\phi = \omega^2 M\phi$. It is no longer a by-product of the basis. The
"unified basis" pitch of the original design weakens.

## 5.3 Path C: **do both A and B, write a method-comparison paper**

*Motivation.* The numerical comparison is itself a contribution.

*How.*

1. Phase 1 in parallel: Boussinesq baseline using Liouville SL and Jacobi.
2. Compare $\mathrm{err}_{L^2}$, wall-clock, and g-mode accuracy.
3. Publish a pure method-comparison paper (JCP fits well).

*Pros.* Most defensible; results cannot be wrong.

*Cons.* ~1.5--2x the workload; less focused narrative.

# 6. Recommendations

## 6.1 Path selection is a function of the paper's angle

| Paper angle | Recommended path |
|---|---|
| "GPU-efficient variable-coefficient Poisson" (JCP methods) | Path B (Jacobi) |
| "Unified spectral basis for Poisson + g-mode spectroscopy" (ApJS astro) | Path A (SL + substitution) |
| "Method comparison for stratified flows" (JCP comparison) | Path C |

## 6.2 Technical risk assessment

**Path A risk.** The substitution $\phi \to r^\beta\phi$ requires $\beta$ to
match the Lane--Emden surface decay exponent of 1.5 with high accuracy. If
mismatched, the singularity residue remains and the convergence is still
algebraic. **Recommended: run a Python verification first to confirm that
$\beta = 1.5$ restores exponential convergence.**

**Path B risk.** Dedalus's EVP solver may need careful tau-parameter tuning on
stiff variable-coefficient problems. That said, Dedalus ships a Lane--Emden
Ball example (`examples/nlbvp_ball_lane_emden/`), indicating the framework
already handles analogous singularities. Engineering risk is low.

**Path C risk.** Largest workload, but risk is distributed.

## 6.3 Suggested execution order

**Phase 0 ext+ (1--2 days, pure Python)**

1. **Use Dedalus as a library, solve Lane--Emden Poisson** (quick validation of
   Path B).
   - Expected: $\mathrm{err}_{L^2} \le 10^{-10}$, exponential convergence.
   - If confirmed: Path B has zero technical risk.
2. **Keep Liouville SL, add $\phi \to r^\beta\phi$ substitution** (Path A
   exploration).
   - Verify that matching $\beta$ restores exponential convergence.
   - If confirmed: Path A has zero technical risk.
3. **Both results on the table, then decide Phase 1's main path.**

This means Phase 1's code commitment will be backed by numerical evidence,
not by a hunch.

# 7. Conclusions

## 7.1 Main findings

1. **Neither GYRE nor Dedalus uses a Liouville $\sqrt{\rho}$ substitution.**
   They absorb the singularity at the variable-definition or basis-weight
   level, from the start, rather than regularizing it after the fact.

2. **The Phase 0 algebraic $-2.4$ convergence is fixable.** E3 already
   demonstrated exponential convergence for a smooth Gaussian $\rho$. The
   problem is that our basis / variable substitution is not aligned with the
   singularity, not that the SL method itself is fundamentally limited.

3. **The g-mode selling point behaves very differently in each path.**
   - Liouville SL: g-mode spectrum is a free by-product (elegant).
   - Jacobi--Galerkin: g-mode requires a separate EVP (standard but loses
     the "unified basis" pitch).

## 7.2 Corrections to the original SL--anelastic design

The mathematics in `docs/anelastic_SL_spectral_design.md` is **entirely correct**
(the Liouville derivation in sect. 3--4 and the SL diagonalization in sect.
4.2 are both rigorous). The engineering assumptions, however, are
**overly optimistic**:

- Section 8.2's "smooth $W$ -> exponential, singular $W$ -> algebraic"
  dichotomy is now empirically confirmed by Phase 0.
- Section 8.1's surface-singularity TODO has been Phase 0-confirmed as the
  primary convergence obstacle.
- Section 7's "three-community gap" narrative remains correct, but the
  reason needs revision:
  it is not "no one thought of SL"; it is **"other communities have already
  adopted simpler approaches that sidestep the Liouville substitution"**.

Proposed revisions:

- Upgrade sections 8.1 and 8.2 from TODOs to **primary challenges**.
- Split Phase 1 into B-quick-validation and A-deep-exploration in parallel.
- Adjust the paper angle from "SL is the optimal method" to "SL vs Jacobi:
  a method comparison".

## 7.3 Direction forward

**Short term (Phase 0 ext+, 1--2 days)**: execute section 6.3 -- Dedalus
Lane--Emden validation plus Liouville + $r^\beta$ substitution experiments.

**Medium term (Phase 1 decision point)**: based on Phase 0 ext+ results,
choose A / B / C.

**Long term (Phase 2-3)**: Boussinesq to anelastic physics extension,
independent of basis choice. g-mode spectrum cross-validation against GYRE
(using Lane--Emden $n=1.5$ as a common benchmark) will provide the final
physical fidelity check.

---

# Appendix A: Key GYRE files

- `src/eqns/rad/rad_obound_m.fypp` -- outer boundary condition dispatch
- `src/eqns/rad/rad_vars_m.fypp` -- non-dimensional variables $y_1, y_2$
- `src/grid/grid_refine_m.fypp` -- adaptive grid refinement
- `src/poly/lane_emden_m.fypp` -- Lane--Emden integrator
- `src/diff/magnus_gl2_diff_m.fypp` -- Magnus GL2/GL4/GL6
- `docs/source/ref-guide/osc-equations/dimless-form.rst` -- theoretical formulation

# Appendix B: Key Dedalus v3 files

- `dedalus/core/basis.py:435-515` -- Jacobi / ChebyshevT / Ultraspherical
- `dedalus/core/basis.py:3917-4087` -- BallBasis / DiskBasis
- `dedalus/core/basis.py:790-815` -- Lift / Tau method
- `dedalus/core/basis.py:4044-4056` -- variable-coefficient operator matrices
- `examples/nlbvp_ball_lane_emden/lane_emden.py` -- Lane--Emden example
- `examples/evp_disk_pipe_flow/pipe_flow.py` -- Disk EVP example

# Appendix C: References

- **Dziembowski (1971)**: Non-radial oscillations of evolved stars. I. Acta Astronomica 21, 289.
- **Unno et al. (1989)**: Nonradial Oscillations of Stars, 2nd ed., University of Tokyo Press.
- **Tassoul (1980)**: Asymptotic approximations for stellar nonradial pulsations. ApJS 43, 469.
- **Townsend & Teitler (2013)**: GYRE: an open-source stellar oscillation code. MNRAS 435, 3406.
- **Burns et al. (2020)**: Dedalus: a flexible framework for numerical simulations with spectral methods. PhRvR 2, 023068.
- **Kosloff & Tal-Ezer (1993)**: A modified Chebyshev pseudospectral method. JCP 104, 457.
- **Boyd (2001)**: Chebyshev and Fourier Spectral Methods. Dover Publications.
