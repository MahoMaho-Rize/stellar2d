# B2. Isentropic hydrostatic-equilibrium background

> **sympy script:** `scripts/b02_isentropic_hse.py`
> **generated LaTeX:** `output/b02_isentropic_hse.latex.tex`
> **verifies:** 14 strong-form identities — 1 exponent identity
> ($\gamma/(\gamma-1) - 1 = 1/(\gamma-1)$); 2 parametric
> derivative-chain identities ($dh/dy$, $dp/dh$); 2 HSE-ODE
> identities (parametric and y-world); 2 parametric state
> identities; 2 bottom-BC identities; 1 isentropic-closure
> identity ($P/\rho^\gamma = K$); 1 atmosphere-cutoff identity;
> 3 temperature-lapse identities (parametric, y-world, compact)
> **code checkpoints:**
> `src/gpu/explicit/strang_solver.cu :: StrangSolver::init`
> (host-side HSE build loop, line 686-692)
> `src/gpu/explicit/strang_device.cuh :: d_hse_rho, d_hse_p`
> (device-side HSE evaluation in §B3 face-state reconstruction)

The Strang solver's background $\bar\rho(y), \bar p(y)$ is the
closed-form solution of the **isentropic** hydrostatic-equilibrium
ODE

$$\frac{d\bar p}{dy} \;=\; -\bar\rho\,g, \qquad \bar p \;=\; K\,\bar\rho^{\gamma}, \quad (\text{B2-ODE})$$

with constant gravity $g > 0$ (downward) and a polytropic constant
$K > 0$ (entropy $s = \log K$). The resulting density profile is

$$\bar\rho(y) \;=\; \bigl[\rho_0^{\gamma-1} \;-\; \tfrac{(\gamma-1)\,g}{\gamma\,K}\,y\bigr]^{1/(\gamma-1)}, \quad (\text{B2-rho})$$

and the pressure follows the equation of state. The profile is
**polytropic with adiabatic lapse rate**: the temperature
$T \propto P/\rho = K \rho^{\gamma-1}$ falls linearly with height to
zero at

$$y^\star \;=\; \frac{\gamma\,K\,\rho_0^{\gamma-1}}{(\gamma-1)\,g}. \quad (\text{B2-ystar})$$

The atmosphere has a finite thickness (not a log-pressure profile
as for isothermal gravity); this is intrinsic to polytropic
stratification, not a numerical artefact.

## Strong-form verification path

sympy cannot directly reduce nested fractional powers
$\bigl(h^{1/(\gamma-1)}\bigr)^\gamma \to h^{\gamma/(\gamma-1)}$ for
symbolic $\gamma$, so the derivation is carried in two steps:

**Step 1.** Parametric factorisation through the linear argument
$h(y) = \rho_0^{\gamma-1} - \tfrac{(\gamma-1) g}{\gamma K}\,y$.
Then

$$\bar\rho \;=\; h^{1/(\gamma-1)}, \qquad \bar p \;=\; K\,h^{\gamma/(\gamma-1)},$$

and the ODE is verified by the chain rule:

$$\frac{d\bar p}{dy} \;=\; \frac{d\bar p}{dh}\,\frac{dh}{dy} \;=\; \frac{\gamma K}{\gamma-1}\,h^{1/(\gamma-1)} \cdot \biggl(-\frac{(\gamma-1)\,g}{\gamma\,K}\biggr) \;=\; -g\,h^{1/(\gamma-1)} \;=\; -\bar\rho\,g.$$

Here the key step is the elementary exponent identity
$\gamma/(\gamma-1) - 1 = 1/(\gamma-1)$, which sympy verifies
independently.

**Step 2.** Direct y-world verification of
$d\bar p/dy + \bar\rho g = 0$ via `sp.powdenest(..., force=True)`,
which normalises $(K\gamma)^a$ terms so sympy can cancel them. This
is **strong-form** (the identity is pointwise in $y$); the
`powdenest` call is a sympy-capability workaround per Rule 1, not
a weak-form fallback.

## Isentropic closure

$$s(y) \;=\; \log\bigl(\bar p / \bar\rho^\gamma\bigr) \;=\; \log K \qquad \text{(constant in } y \text{)}. \quad (\text{B2-isentropic})$$

By construction, since $\bar p = K \bar\rho^\gamma$ at every $y$.
The solver's choice of isentropic stratification means every fluid
parcel is marginally convectively stable (Schwarzschild criterion
with zero super-adiabatic gradient): no spurious convection driven
by background thermodynamics.

## Temperature / lapse-rate form

$$\bar p / \bar\rho \;=\; K\,\rho_0^{\gamma-1}\,\bigl(1 - y/y^\star\bigr), \quad (\text{B2-lapse})$$

which is linear in $y$, vanishing at the atmosphere cut-off. For
an ideal gas $p = \rho R T / \mu$, this corresponds to temperature
$T(y) = T_0 (1 - y/y^\star)$ with $T_0 = K \rho_0^{\gamma-1} \mu / R$,
i.e., linearly decreasing with height — the characteristic
adiabatic-atmosphere temperature lapse.

## Atmosphere cut-off

At $y = y^\star$ the argument $h$ of the fractional power vanishes,
so formally $\bar\rho(y^\star) = 0$. The kernel clamps $h \ge 10^{-20}$
(`strang_solver.cu` line 689) so that the solver can run with
computational domains larger than $y^\star$ without NaN. The
physically meaningful region is $0 \le y < y^\star$; tests should
keep $L_y < y^\star$ unless explicitly probing the near-vacuum
asymptote.

## Device-side form

The device functions `d_hse_rho(y, rho0_gm1, coeff, inv_gm1)` and
`d_hse_p(rho, K, gamma)` in `strang_device.cuh` implement the same
formula with pre-computed constants `rho0_gm1 = rho_0^(gamma-1)`,
`coeff = (gamma-1) g / (gamma K)`, and `inv_gm1 = 1/(gamma-1)`.
These are evaluated at face-centre y-coordinates during the y-sweep
MUSCL-Hancock reconstruction (§B3) so that the face pairs see
**identical** HSE background contribution.

## ✅ Verification checkpoint (to be wired)

1. **Host-host consistency.** The C++ init loop in `StrangSolver::init`
   at a cell centre $y_j = y_{\mathrm{lo}} + (j + 1/2)\,\Delta y$ must
   produce $\bar\rho[j], \bar p[j]$ equal to §B2's closed-form
   $\bar\rho(y_j), \bar p(y_j)$ to ULP precision. Test:
   `test_strang_init.cu` §B2-profile-match — compare the init array
   against a host-computed reference over all cells.

2. **Device-device consistency.** At an arbitrary face-centred
   $y_{\mathrm{face}}$, `d_hse_rho` and `d_hse_p` must equal the
   closed-form $\bar\rho(y_{\mathrm{face}})$, $\bar p(y_{\mathrm{face}})$
   to ULP precision. Test: `test_strang_muscl.cu` §B2-hse-face.

3. **HSE ODE residual.** Finite-difference the initialised arrays:
   $|\bar p[j+1] - \bar p[j]|/\Delta y + \bar\rho[j+1/2]\,g$ should
   converge to zero as $\Delta y \to 0$ at second order. Test:
   `test_strang_init.cu` §B2-ode-convergence.

4. **Isentropic closure.** For all $j$, $|\bar p[j] - K \bar\rho[j]^\gamma|
   \le 10\varepsilon_{\mathrm{mach}} \bar p[j]$. Test:
   `test_strang_init.cu` §B2-isentropic.

5. **Goldens dump.** Part D's `d06_hse_zero_perturbation_lock.py`
   (7.6 goldens) emits a reference profile JSON at `N=8192`
   y-points for a canonical HSE setup; the test consumer reads this
   JSON and compares to the kernel's init array.

Failure of (1) is a host-side bug (the closed-form formula was
mistyped or the `j + 0.5` cell-centre convention was violated).
Failure of (2) is a device-side inconsistency between host init and
device re-evaluation (most likely sign error in `coeff` or an
inconsistent definition of `rho0_gm1`). Failure of (3) or (4)
indicates structural loss of the HSE property — triage by
inspecting which profile (rho vs p) went wrong first. Failure of
(5) is rare; it usually means the regression golden was generated
at a different $K, \rho_0$ than the kernel is using.
