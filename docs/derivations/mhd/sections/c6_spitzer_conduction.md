# C6. Spitzer-Härm anisotropic thermal conduction

> **sympy script:** `scripts/c6_spitzer_conduction.py`
> **verified:** parallel projector $\mathsf{P}_\parallel = \hat{\mathbf{b}}\hat{\mathbf{b}}^{\mathrm{T}}$
> idempotent with $\mathrm{tr}\,\mathsf{P}_\parallel = 1$; parallel /
> perpendicular decomposition of $\nabla T$; Kirchhoff potential
> identity $\mathbf{F}_c = -\nabla[\tfrac{2}{7}\kappa_0 T^{7/2}]$
> in 1D; entropy production $\sigma_\mathrm{cond} = \kappa_\parallel(\hat{\mathbf{b}}\cdot\nabla T)^2/T^2 \ge 0$;
> FTCS stability $\sigma \le 1/2$.
> **code checkpoints:** `athena_mhd_kernels.cu::d_spitzer_heat_flux`,
> `athena_mhd_solver.cu::compute_conduction_dt`,
> `tests/test_athena_mhd_conduction_isothermal.cu`.

## Why Spitzer conduction dominates the corona

In a fully-ionised, magnetised plasma the electron mean-free-path
scales as $\lambda_e \propto T^2/n_e$, while the transit time between
collisions $\tau_c \propto T^{3/2}/n_e$. Combining these with the
electron thermal velocity $v_{th,e} \propto \sqrt{T}$ gives the
Spitzer-Härm (1953) / Braginskii (1965) conductivity

$$\kappa_\parallel(T) = \kappa_0\,T^{5/2},\quad
\kappa_0 \approx 10^{-6}\ \mathrm{erg\,cm^{-1}\,s^{-1}\,K^{-7/2}}.$$

Perpendicular to $\mathbf{B}$, each electron gyrates through
$\Omega_c \tau_c \gg 1$ orbits between collisions, so the
cross-field transport is suppressed by $(\Omega_c\tau_c)^{-2} \sim 10^{-18}$
in a coronal loop. To machine precision the effective conductivity is
**rank-1**:

$$\boxed{\mathbf{F}_c = -\kappa_\parallel(T)\,\hat{\mathbf{b}}\,(\hat{\mathbf{b}}\cdot\nabla T),\qquad \hat{\mathbf{b}} = \mathbf{B}/|\mathbf{B}|.} \quad (\text{C6-Fc})$$

## Parallel / perpendicular decomposition

Sympy-verified the following on a general smooth $(T, \mathbf{B})$:

1. **Idempotent projector.**
   $\mathsf{P}_\parallel \equiv \hat{\mathbf{b}}\hat{\mathbf{b}}^{\mathrm{T}}$
   satisfies $\mathsf{P}_\parallel^2 = \mathsf{P}_\parallel$ and
   $\mathrm{tr}\,\mathsf{P}_\parallel = 1$.
2. **Orthogonal decomposition.** $\nabla T = \mathsf{P}_\parallel\nabla T
   + (\mathsf{I} - \mathsf{P}_\parallel)\nabla T$ with the second piece
   strictly perpendicular to $\hat{\mathbf{b}}$.
3. **Anisotropy bound.** For any $\nabla T$ the flux magnitude is
   capped: $|\mathbf{F}_c| = \kappa_\parallel|\hat{\mathbf{b}}\cdot\nabla T|
   \le \kappa_\parallel|\nabla T|$.

## Kirchhoff potential (1D closed form)

In the smooth isothermal-field (or B-aligned) limit the flux admits a
**gradient-form** potential:

$$\boxed{\mathbf{F}_c = -\nabla\!\left[\tfrac{2}{7}\,\kappa_0\,T^{7/2}\right] \quad\text{(1D / B-aligned smooth flow).}} \quad (\text{C6-Kirchhoff})$$

This is critical for implementation: the nonlinear flux can be
written as $-\kappa_0\partial_x(T^{7/2}) \cdot \tfrac{2}{7}$, which is
a **linear** central-difference operator on the Kirchhoff-transformed
variable $\Theta \equiv \tfrac{2}{7}\kappa_0 T^{7/2}$.
Sympy verifies
$\partial_x(\tfrac{2}{7}\kappa_0 T^{7/2}) = \kappa_0 T^{5/2}\partial_x T$
identically.

## Low-density collisionless quench

At $\rho \lesssim 10^{-20}\,\mathrm{g\,cm^{-3}}$ the mean-free-path
exceeds the scale height and the Spitzer formula over-predicts the
flux by orders of magnitude (Gruzinov-Quataert 2004; Shoda+2018a).
Suzuki 2203.15280 Eq. 12 / Shoda+2020 patch this with a phenomenological
cutoff:

$$\boxed{\mathbf{q}_{\mathrm{cnd}} = -\min\!\Bigl(1,\,\rho/\rho_{\mathrm{cnd}}\Bigr)\,(B_r/|\mathbf{B}|)\,\kappa_0\,T^{5/2}\,\partial_r T,\quad \rho_{\mathrm{cnd}} = 10^{-20}\,\mathrm{g\,cm^{-3}}.} \quad (\text{C6-quench})$$

**Do not drop the quench** — unquenched Spitzer at coronal $\rho$
drives $\Delta t_\mathrm{cond}$ to $\sim 10^{-4}\times$ the hyperbolic
CFL (shown in Shimizu+22 Fig. 2a). The quench is a numerical device
and must be on; flipping it off produces the correct physics (sharper
thermal fronts) but the wall-clock cost is unmanageable.

## Energy-equation coupling

The flux enters the total-energy equation conservatively:

$$\partial_t E + \nabla\!\cdot\!\bigl[(E+p^\star)\mathbf{v}
 - \mathbf{B}(\mathbf{v}\!\cdot\!\mathbf{B}) + \mathbf{F}_c\bigr] = 0. \quad (\text{C6-energy})$$

**Sign.** Heat flows *down* the temperature gradient:
$\mathbf{F}_c\!\cdot\!\nabla T = -\kappa_\parallel(\hat{\mathbf{b}}\!\cdot\!\nabla T)^2 \le 0$
(sympy-verified). Entropy production is strictly non-negative:

$$\sigma_\mathrm{cond} \equiv -\frac{\mathbf{F}_c\!\cdot\!\nabla T}{T^2}
= \frac{\kappa_\parallel}{T^2}(\hat{\mathbf{b}}\!\cdot\!\nabla T)^2 \ge 0. \quad (\text{C6-entropy})$$

This is the 2nd-law certificate. Any kernel regression that flips a
sign on $\mathbf{F}_c$ will show as $\sigma_\mathrm{cond} < 0$ in the
verification test.

## Parabolic CFL (linearised FTCS)

Linearising around a background $T_0$ gives an effective thermal
diffusivity

$$\chi_\mathrm{eff} \equiv \frac{\kappa_0\,T_0^{5/2}}{\rho\,c_v}.$$

Forward-Euler + central-space applied to $\partial_t T = \chi\partial_x^2 T$
has the amplification factor

$$g(\xi) = 1 - 4\sigma\,\sin^2(\xi/2),\qquad
\sigma \equiv \chi\,\Delta t\,/\,\Delta x^2. \quad (\text{C6-FTCS})$$

Sympy-verified: worst case $\xi = \pi$ gives $g = 1 - 4\sigma$;
marginal stability at $\sigma = 1/2$ gives $g = -1$. For the Spitzer
diffusivity the bound becomes

$$\boxed{\Delta t_\mathrm{cond} \le \tfrac{1}{2}\,\min_{\text{cells}} \frac{\rho\,c_v\,\Delta x^2}{\kappa_0\,T^{5/2}}.} \quad (\text{C6-CFL})$$

At chromospheric parameters
$(T \sim 10^5\,\mathrm{K},\ \rho \sim 10^{-10}\,\mathrm{g\,cm^{-3}},\ \Delta x \sim 50\,\mathrm{km})$
this gives $\Delta t_\mathrm{cond} \sim 10^{-2}\,\mathrm{s}$, roughly a
factor of $10^3$ tighter than the hyperbolic CFL. At $T \sim 2\times10^6\,\mathrm{K}$
the factor is $\sim 10^6$.

## RKL2 super-time-stepping

Because the conductive CFL is so tight, all modern stellar-wind codes
use **RKL2 super-time-stepping** (Meyer+2012, based on
Alexiades-Amiez-Gremaud 1996). $N$ Chebyshev sub-stages lift the
explicit diffusion step up to

$$\Delta t_\mathrm{RKL2} \approx \tfrac{N^2 + N}{4}\,\Delta t_\mathrm{cond}.\quad (\text{C6-RKL2})$$

Per hydrodynamic step we pick $N = \lceil\sqrt{4\Delta t_\mathrm{hyp}/\Delta t_\mathrm{cond}}\rceil$
so that one RKL2 block spans the hydro step. The per-block cost is
$N$ conduction-flux evaluations; for $N=20$ the conductive solver
consumes $\sim 10\%$ of the wall-clock — negligible.

RKL2 is a **separate operator** applied after the hyperbolic step
(operator splitting), **not** a modification of the VL2 integrator.

## Implementation recipe (for `athena_mhd` future addition)

1. **Compute $T$** from primitive $p/\rho$ and $\mu$ (partial-ionisation
   μ from §C4 Saha closure).
2. **Evaluate flux at faces**: $\mathbf{F}_c^{i\pm 1/2}$ using
   centred differences of $T$ and the face-averaged $\hat{\mathbf{b}}$.
   Key subtlety: $\hat{\mathbf{b}}$ is a unit vector — compute it by
   first averaging $\mathbf{B}$ to face centres, then normalising.
   Averaging $\hat{\mathbf{b}}$ after normalising at cell centres
   loses rotational symmetry.
3. **Apply quench** per (C6-quench).
4. **Add $\nabla\!\cdot\!\mathbf{F}_c$** to the RHS of the energy
   equation.
5. **RKL2 sub-cycling** if $\Delta t_\mathrm{cond} < 0.1\,\Delta t_\mathrm{hyp}$;
   otherwise integrate in-line with the hyperbolic step.

## ✅ Verification checkpoints

- `tests/test_athena_mhd_conduction_isothermal.cu` — sinusoidal
  $T(x) = T_0(1 + A\cos k x)$ on a uniform $\mathbf{B}_0 = B_0\hat{x}$
  background, no flow. Lock $L^2(T - T_0)$ decay rate matches
  analytical $\exp(-\chi_\mathrm{eff} k^2 t)$ to $<1\%$ at $N = 128$.
- `tests/test_athena_mhd_conduction_anisotropy.cu` — same IC but
  with $\mathbf{B}_0 = B_0\hat{y}$ (perpendicular to $\nabla T$).
  Assert $L^2(T - T_0)$ stays constant over 100 collision times —
  cross-field quench to machine precision.
- `tests/test_athena_mhd_conduction_kirchhoff.cu` — in 1D with
  $T(x) \propto (1 + A\cos k x)$, compare evolution of
  $\tfrac{2}{7}\kappa_0 T^{7/2}$ versus direct nonlinear
  $\kappa_0 T^{5/2}\partial_x T$; lock relative difference $<10^{-12}$.
- `tests/test_athena_mhd_conduction_entropy.cu` — random
  $(T, \mathbf{B}, \nabla T)$ states, 100 samples; assert
  $\sigma_\mathrm{cond} \ge 0$ in every sample. Catches accidental
  sign flips.

Any failure on the anisotropy (perpendicular-quench) test is an
immediate red flag: without rank-1 conductivity, 1D modelling of
coronal loops (Aschwanden 2005 §4) overshoots $T_\mathrm{peak}$ by a
factor of 2–3.

## Numerical implementation notes (not in formal derivation)

Phase B-M4 (`test_athena_mhd_combined.cu`, 10/10 pass) exposed, for
the first time in the combined stack (WB + κ + cooling), a
ghost-cell interaction between the κ operator and the reflective
y-BC.  At the derivation level
$\mathbf{F}_c = -\kappa_\parallel \hat{\mathbf{b}}(\hat{\mathbf{b}}\cdot\nabla T)$
is a continuum identity, but the finite-volume + face-B + cons_to_prim
implementation violates it at reflective walls.

### Symptom

An isothermal magnetised atmosphere has $T = c_s^2$ uniformly, hence
$\nabla T \equiv 0$ and therefore $\mathbf{F}_c \equiv 0$ is expected.
In practice a single call to `apply_conduction(dt)` shifts
$\delta E / E$ to $\sim 4\%$ (well above ULP), while $\delta\rho$,
$|\mathbf{v}|$, and $|\nabla\!\cdot\!\mathbf{B}|$ all stay at machine
precision — **only $E$ is incorrectly modified by κ**.

### Root cause: the ghost $T$ from cons_to_prim is not a scalar mirror

Under reflective y-BC the face-B mirrors antisymmetrically:
$B_{yf}[n_g - 1] = -B_{yf}[n_g + 1]$.  The ghost-cell cell-centered
$B_y$ is therefore

$$B_{y,\mathrm{cc}}^\mathrm{ghost} = \tfrac12(B_{yf}[n_g-1] + B_{yf}[n_g]) = \tfrac12(-B_{0y} + B_{0y}) = 0,$$

while the interior cell-centered value is $B_{y,\mathrm{cc}} = B_{0y}$.
The routine `cons_to_prim` then uses
$p = (\gamma-1)(E - \mathrm{KE} - \mathrm{ME})$ to recover pressure,
and because $\mathrm{ME}^\mathrm{ghost} \ne \mathrm{ME}^\mathrm{interior}$
the ghost $p$ is off by $\tfrac12(\gamma-1) B_{0y}^2$.  Consequently
$T^\mathrm{ghost} = p^\mathrm{ghost}/\rho^\mathrm{ghost} \ne c_s^2$.
When the κ flux kernel reads this contaminated ghost $T$, a spurious
nonzero $\nabla T$ appears at the wall, $\mathbf{F}_c$ drains energy
into the ghost layer, and the uniform-$T$ $\Rightarrow$ zero-flux
expectation is violated.

This is not an error in the §C6 derivation, nor in `cons_to_prim`:
the latter faithfully applies the face-B mirror rule to obtain ghost
$B_\mathrm{cc}$.  The issue is that the "mirror" of $B_\mathrm{cc}$
carries the **antisymmetric** semantics of a vector component normal
to the wall, whereas $T$ is a scalar whose ghost ought to satisfy a
**symmetric** mirror.  Recovering $T$ via the $(p, \rho) \to T$ chain
inadvertently imports the vector mirror semantics of $B$ into a
scalar field.

### Fix: fill $T$ ghost cells independently of cons_to_prim

A family of kernels
`k_athmhd_ghost_T_{y_reflect, y_periodic, y_outflow, x_periodic, x_outflow}`
is introduced.  After `compute_T` and before the κ flux kernel runs,
these overwrite the ghost layer of `T_cc` using the scalar mirror
rule appropriate to each BC.  The κ flux kernel still reads `T_cc`,
but now the ghost $T$ agrees with the interior $T$ at reflective
walls up to ULP, giving $\nabla T|_\mathrm{wall} = 0$ exactly.

### Generalised lesson

Any spatial-flux operator that **reads a cell-centered scalar** (for
example $T$, $\mu$, $Y_e$) must **fill its own ghost layer**
independently of `cons_to_prim`.  The latter carries the
directional-mirror semantics of vector fields ($B$, $\mathbf{v}$) that
cannot be translated losslessly onto a scalar.  Per-cell ODE
operators such as cooling are unaffected (they never cross cells),
but spatial-flux operators — κ conduction, future viscosity,
radiative diffusion — each need an explicit scalar ghost-fill pass.
