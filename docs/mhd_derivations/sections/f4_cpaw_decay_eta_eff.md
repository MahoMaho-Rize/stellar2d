# F4. CPAW 2D long-time decay and $\eta_\mathrm{eff}$ extraction

> **sympy script:** `scripts/f4_cpaw_decay_eta_eff.py`
> **verified:** ideal limit $\eta \to 0$ recovers $\omega = \pm v_A k$;
> weak-$\eta$ expansion $\omega = v_A k - (i/2)\eta k^2 + \mathcal{O}(\eta^2)$;
> amplitude decay $\gamma = \eta k^2 / 2$;
> 2nd-order scaling $\gamma_\mathrm{num}(N) \propto N^{-2}$ via
> modified-equation analysis; two-resolution inversion formula
> recovers scheme order $q$ from $\gamma_1, \gamma_2$.
> **code checkpoints:**
> `tests/test_athena_mhd_cpaw_longtime.cu` — CPAW 2D at
> $N \in \{32, 64, 128\}$, measure amplitude decay, extract $\eta_\mathrm{eff}(N)$.

## Motivation

Phase A4 asks: **what is the numerical resistivity of `athena_mhd` on
linear Alfvén waves?** This number is the single most important
quantitative calibration for later physics work: every 2D Alfvén
turbulence / Shimizu-style run must have physical damping rates
$\gg \gamma_\mathrm{num}(N)$ at the target resolution, or the
reported dissipation is unphysical.

Before running the test we derive **three formulas** that the test
post-processing uses:

1. The resistive dispersion relation for linearised Alfvén waves
   (established by field theory but rarely documented in one place).
2. The weak-$\eta$ amplitude-decay rate $\gamma = \eta k^2 / 2$.
3. The scheme-order inversion formula $p = \log(\gamma_1/\gamma_2) / \log(N_2/N_1)$.

## Q1: Resistive Alfvén dispersion

Linearised transverse MHD on a uniform background
$(\rho_0, B_{r,0}, \mathbf{v}_0 = 0)$ with finite resistivity $\eta$:

$$\partial_t v_\perp = (B_{r,0}/\rho_0)\,\partial_r B_\perp,$$

$$\partial_t B_\perp = B_{r,0}\,\partial_r v_\perp + \eta\,\partial_r^2 B_\perp.$$

Plane-wave ansatz $(v_\perp, B_\perp) = (V, B)\,e^{i(kr - \omega t)}$ gives
the $2\times 2$ dispersion matrix. Setting its determinant to zero
yields

$$\boxed{\omega^2 + i\,\eta\,k^2\,\omega - v_A^2 k^2 = 0,\qquad
v_A \equiv B_{r,0}/\sqrt{\rho_0}.} \quad (\text{F4-disp})$$

Sympy-verified: the ideal limit $\eta \to 0$ recovers the Alfvén
dispersion $\omega = \pm v_A k$.

## Q2: Weak-$\eta$ expansion and amplitude decay

Solving (F4-disp) and expanding around the outgoing branch
$\omega_0 = v_A k$ to first order in $\eta$:

$$\boxed{\omega = v_A k - \tfrac{i}{2}\,\eta\,k^2 + \mathcal{O}(\eta^2),
\qquad \varepsilon_\mathrm{weak} \equiv \eta k / v_A \ll 1.} \quad (\text{F4-weak})$$

Sympy-verified via `sp.series`.

**Physical amplitude decay.** With $e^{-i\omega t} = e^{-iv_A k t} \cdot e^{-(\eta k^2/2) t}$,

$$A(t) = A_0\,\exp(-\gamma\,t),\qquad \gamma = \tfrac{1}{2}\,\eta\,k^2. \quad (\text{F4-decay})$$

Both Elsässer modes $z^\pm$ decay at the same rate $\gamma$ —
resistivity is non-selective between counter-propagating Alfvén
waves.

**Factor of 1/2.** Different conventions absorb or drop the 1/2;
here we keep it explicit because the A4 test measures amplitudes
(not energies) and converts measured $\gamma$ → $\eta$ via the
inverse $\eta = 2\gamma/k^2$.

## Q3: Numerical resistivity from modified-equation analysis

For a 2nd-order Godunov scheme (PLM + HLLD + VL2, see §A6–A8) on
the linear Alfvén equation, standard modified-equation analysis
(LeVeque 2002 §18) gives

$$\boxed{\eta_\mathrm{eff}(h) = C_\mathrm{num}\,h^2\,v_A,\qquad
\gamma_\mathrm{num}(N) = \tfrac{1}{2}\,C_\mathrm{num}\,h^2\,v_A\,k^2 \propto N^{-2}.} \quad (\text{F4-eta-eff})$$

$C_\mathrm{num}$ is a scheme-dependent $\mathcal{O}(1)$ constant that
depends on limiter choice, HLLD branch, and CT corner-EMF weights.
We do **not** compute $C_\mathrm{num}$ from first principles (it would
require von-Neumann analysis of the full VL2 + HLLD + CT loop on the
Alfvén mode — closed-form intractable); instead, we *measure*
$\eta_\mathrm{eff}(N)$ empirically and **check** the $h^2$ scaling.

Sympy-verified: $\mathrm{d}\log\gamma_\mathrm{num}/\mathrm{d}\log h = 2$.

## Q4: Two-resolution inversion for scheme order

Given measured decay rates at two resolutions:

$$\boxed{p = \frac{\log(\gamma_1/\gamma_2)}{\log(N_2/N_1)},} \quad (\text{F4-order})$$

for a scheme with true order $q$ and $\gamma_i \propto N_i^{-q}$, this
formula returns $p = q$ exactly. Sympy-verified.

For our 2nd-order solver we expect $p \approx 2$. Tolerances in the
A4 test: $|p - 2| < 0.3$ on pairs $(32\to64), (64\to128)$.

## Test-pass criteria (A4 CPAW long-time)

For N ∈ {32, 64, 128}, CPAW 2D (§B2, Tóth 2000 §3.2.2) travelling
at $v_A = 1$ over 10 and 50 wave periods:

1. **Amplitude remains finite**: $A(t_\mathrm{end})/A(0) \in [0.1, 1]$
   for every N — no catastrophic blow-up, no complete dissipation.
2. **Monotone decay**: $A(t)$ monotone non-increasing (modulo small
   aliasing noise, similar to §F3 B_cc aliasing).
3. **Scheme-order consistency**: $|p_{32\to64} - 2| < 0.3$ AND
   $|p_{64\to128} - 2| < 0.3$.
4. **$\eta_\mathrm{eff}$ table entries** written to CSV for the
   Phase A results document:

| N | $\gamma_\mathrm{num}$ | $\eta_\mathrm{eff} = 2\gamma/k^2$ | $\mathrm{Re}_\mathrm{num} = v_A L/\eta_\mathrm{eff}$ |
|---|---|---|---|
| 32  | (measured) | (derived via F4-decay) | (physical interpretation) |
| 64  | ... | ... | ... |
| 128 | ... | ... | ... |

## Why this matters for Suzuki physics

For $L = \sqrt{5}$, $v_A = 1$, $k_\mathrm{wave} = 2\pi$ in our CPAW
IC, the weak-$\eta$ criterion $\eta k / v_A \ll 1$ gives
$\eta \ll 0.16$. We expect measured $\eta_\mathrm{eff}(128) \lesssim 10^{-4}$,
corresponding to numerical magnetic Reynolds number $\mathrm{Re}_m
\sim 10^4$ on the domain scale.

When we extend to Suzuki-style problems (C8 chromosphere, E1 driver,
§B1 flux tube), the physical magnetic Reynolds number at typical
parameters is $\mathrm{Re}_m^\mathrm{phys} \sim 10^6$–$10^{10}$.
**Our solver cannot resolve this directly**; physical $\eta$ is
irrelevant because $\mathrm{Re}_m^\mathrm{num} < \mathrm{Re}_m^\mathrm{phys}$.
But for *turbulence* problems where the physical cascade only needs
$\mathrm{Re}_m \gtrsim 10^3$ to set up an inertial range, our
solver at $N \ge 256$ is viable.

F4 gives us the quantitative version of that statement, per
resolution.

## ✅ Verification checkpoints

- `tests/test_athena_mhd_cpaw_longtime.cu` — will implement the A4
  test following this derivation.

Pass criteria map directly from (F4-decay), (F4-eta-eff), (F4-order)
to concrete numerical thresholds in the test file.
