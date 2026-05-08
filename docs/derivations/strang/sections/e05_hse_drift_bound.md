# E5. Long-time HSE drift bound

> **sympy script:** `scripts/e05_hse_drift_bound.py`
> **generated LaTeX:** `output/e05_hse_drift_bound.latex.tex`
> **generated goldens:** `output/e05_hse_drift_bound.goldens.json`
> **verifies:** 2 strong-form identities — 1 linear-summation
> drift bound ($\varepsilon_{\mathrm{mach}} \cdot N_{\mathrm{step}}
> \cdot \kappa(\mathrm{HSE})$); 1 Kahan-summation drift bound
> ($\varepsilon_{\mathrm{mach}} \sqrt{N} \kappa$, not used in
> kernel)
> **code checkpoints:**
> composite of §B2, §B3, §C1, §D6; measured by
> `tests/test_strang_step.cu` §D6-long-time

Quantifies the maximum drift of the stored state away from zero
on pure HSE IC over long time evolution. The kernel uses standard
(non-Kahan) summation for its accumulators, so the drift is
**linear in step count**:

$$\max_{t}\,\|\boldsymbol{\delta U}\|_{\infty}(N_{\mathrm{step}}) \;\leq\; \varepsilon_{\mathrm{mach}}\,N_{\mathrm{step}}\,\kappa(\mathrm{HSE}), \quad (\text{E5-bound})$$

where $\kappa(\mathrm{HSE})$ is a problem-dependent condition
number that captures how much the floating-point errors in
evaluating the HSE flux divergence and gravity source get
amplified by the ratio of background scales across the domain.

## Condition number

The worst case is a floating-point subtraction of two
nearly-equal but large numbers: $\bar p(y_{j+1/2}) - \bar p(y_{j-1/2})$
in the flux divergence minus the gravity-source contribution
$\bar\rho_j g \Delta y$. The relative error in either term is
$\sim \varepsilon_{\mathrm{mach}}$; the absolute error is
$\varepsilon_{\mathrm{mach}} \bar p$ per flux evaluation.

The condition number

$$\kappa(\mathrm{HSE}) \;\sim\; \frac{\max_y \bar\rho(y)}{\min_y \bar\rho(y)} \;\sim\; \biggl(\frac{1}{1 - L_y/y^\star}\biggr)^{1/(\gamma - 1)} \quad (\text{E5-kappa})$$

captures the ratio of the largest background density to the
smallest over the domain. For the canonical HSE setup
($L_y = 1$, $y^\star = 3.5$, $\gamma = 1.4$):

$$\kappa \;=\; \biggl(\frac{1}{1 - 1/3.5}\biggr)^{1/0.4} \;=\; (7/5)^{2.5} \;\approx\; 2.32. \quad (\text{E5-canonical})$$

This is a very gentle condition number — the atmosphere is not
deeply stratified within the canonical domain. For a deeper
domain ($L_y \to y^\star$), $\kappa$ grows rapidly; at
$L_y = 0.99 y^\star$, $\kappa \sim 10^5$ and the HSE drift
becomes a limiting factor.

## Numerical drift prediction

At $\varepsilon_{\mathrm{mach}} = 2.22 \times 10^{-16}$,
$\kappa = 2.32$, $N = 1000$:

- **Linear summation (kernel default):** $5.1 \times 10^{-13}$.
- **Kahan summation (not implemented):** $1.6 \times 10^{-14}$.

The linear bound is well below the typical test tolerance of
$10^{-10}$, so the kernel should pass §D6's long-time test with
ample margin.

## Kahan summation option

If performance could accommodate it, Kahan summation in the flux
divergence accumulator would reduce the drift to
$\varepsilon_{\mathrm{mach}} \sqrt{N} \kappa$:

$$\max_{t}\,\|\boldsymbol{\delta U}\|_{\infty}(N) \;\lesssim\; \varepsilon_{\mathrm{mach}}\,\sqrt{N}\,\kappa(\mathrm{HSE}). \quad (\text{E5-kahan})$$

For $N = 10^6$, this gives $\sim 5 \times 10^{-13}$ vs. linear's
$\sim 5 \times 10^{-10}$ — a factor $10^3$ improvement. The
kernel currently does not use Kahan because the tolerance budget
is ample; a future optimisation could revisit this if very long
HSE evolution becomes a primary use case.

## Drift diagnostics from §D6

§D6's long-time test runs $N = 1000$ steps and requires
$\|\boldsymbol{\delta U}\|_\infty \le 10^{-10}$. §E5 predicts
$\sim 5 \times 10^{-13}$, a factor $200\times$ below tolerance.
If the kernel's measured drift is close to $10^{-10}$ (near
tolerance), the $\kappa$ is much higher than predicted — either
the domain is too deep, or the HSE ODE discretisation is weaker
than 2nd-order.

## Regression-test robustness

The drift test at $N = 10^4$ would bring the predicted drift to
$\sim 5 \times 10^{-12}$ — still well below $10^{-10}$. At
$N = 10^5$, $\sim 5 \times 10^{-11}$ — approaching tolerance.
This is why the §D6 test uses $N = 1000$: a comfortable but
realistic time horizon for a regression check. For convective
evolution tests (§D5 bubble, run time $T = 0.2$, dt $\sim 0.001$
so $N \sim 200$), the drift is negligible at $\sim 10^{-13}$.

## Extending to non-HSE initial conditions

The analysis here is specific to **pure HSE** initial conditions
where the stored state is identically zero. For general ICs with
non-zero perturbations, the drift has two sources:

1. The HSE-preservation drift above ($\varepsilon_{\mathrm{mach}}
   N \kappa$).
2. The physical perturbation evolution's discretisation error
   ($O(\Delta x^2)$ from §E1 modified equation).

The latter dominates for finite-amplitude perturbations. The §E5
bound applies only to the floor below which the kernel cannot
preserve a quiescent HSE state — a diagnostic of the kernel's
round-off accumulation, not of its physical convergence rate.

## ✅ Verification checkpoint (to be wired)

1. **$N = 1000$ drift bound.** Measured
   $\|\boldsymbol{\delta U}\|_\infty$ at $N = 1000$ is in
   $[\varepsilon_{\mathrm{mach}}, 2 \times \varepsilon_{\mathrm{mach}} N \kappa]$
   $= [2.2 \times 10^{-16}, 10^{-12}]$. Test:
   `test_strang_step.cu` §E5-drift-bound.

2. **Linear-in-N growth.** Running at $N \in \{100, 300, 1000, 3000\}$
   and measuring the drift shows linear scaling in $N$. Test:
   `test_strang_step.cu` §E5-linear-scaling.

3. **$\kappa$ scaling with domain depth.** Running the same
   $N = 1000$ test at three domain depths $L_y \in \{0.3, 0.7,
   0.99\} y^\star$ shows drift scaling like $\kappa(L_y)$:
   - $L_y = 0.3 y^\star$: $\kappa \sim 1.06$, drift $\sim 2 \times 10^{-13}$.
   - $L_y = 0.7 y^\star$: $\kappa \sim 11$, drift $\sim 2 \times 10^{-12}$.
   - $L_y = 0.99 y^\star$: $\kappa \sim 10^5$, drift $\sim 2 \times 10^{-8}$.
   Test: scheme-char probe (not a production regression).

Failure of (1) with drift $> 10^{-11}$ at $N = 1000$ indicates
either a bug (likely §C1 or §B3) or a more-stratified test
domain where $\kappa > 10$. Failure of (2) with super-linear
growth (e.g., $N^2$) means the kernel is accumulating a
**systematic** error, not round-off — a structural bug.
