# D6. HSE zero-perturbation lock

> **sympy script:** `scripts/d06_hse_zero_perturbation_lock.py`
> **generated LaTeX:** `output/d06_hse_zero_perturbation_lock.latex.tex`
> **generated goldens:** `output/d06_hse_zero_perturbation_lock.goldens.json`
> **verified:**
> - flux-source cancellation on HSE (y-momentum)
> - energy update zero on HSE
> - density update zero on HSE
> - x-momentum update zero on HSE
>
> **code checkpoints:**
> - whole kernel — this test verifies the **composite** of §B2, §B3, §B4, §B5, §B6, §C1. Any one of these failing breaks the lock. Specifically: init-time HSE build (`StrangSolver::init`), face HSE reconstruction (`k_muscl_hancock_y`), ghost-fill kernels (`k_ghost_x, k_ghost_y`), gravity source application (`k_hllc_update_y`).

The most stringent well-balancing test in the book. With stored
state $(\delta\rho, m_x, m_y, \delta E) = (0, 0, 0, 0)$ at every
cell (pure HSE), the kernel must preserve this state to machine
precision under arbitrary numbers of Strang steps. Any non-zero
drift within $O(\varepsilon_{\mathrm{mach}} N)$ indicates a
composition failure among the HSE-related sections.

## IC

$$(\delta\rho, m_x, m_y, \delta E)(\mathbf{x}, t = 0) \;=\; (0, 0, 0, 0) \quad \forall\,\mathbf{x} \in \text{domain}. \quad (\text{D6-IC})$$

After `StrangSolver::init` builds the HSE background and before
any perturbation is added, the storage buffer **is** the zero
state. `init_bubble()` is **not** called for this test.

## Strong-form balance

The stored $\mathbf{U}_{\mathrm{store}} = \mathbf{0}$ is preserved
because each component's update is identically zero on pure HSE:

| component | flux divergence | source | sum |
|---|---|---|---|
| $\delta\rho$ | $-\partial_x(\rho v) - \partial_y(\rho v) = 0$ (since $v = 0$) | 0 | **0** |
| $m_x$ | $-\partial_x(\rho u^2 + P)$: $P = \bar p$ indep of $x$ → 0; $-\partial_y(\rho u v) = 0$ | 0 | **0** |
| $m_y$ | $-\partial_y(\rho v^2 + P) = -\partial_y \bar p = +\bar\rho g$ | $-\bar\rho g$ | **0** |
| $\delta E$ | $-\partial_x((E+P) u) = 0$; $-\partial_y((E+P) v) = 0$ | $-m_y g = 0$ | **0** |

The non-trivial cancellation is in the y-momentum: the background
pressure gradient contributes $+\bar\rho g$ (via the HSE ODE
$d\bar p/dy = -\bar\rho g$ from §B2), and the gravity source
contributes $-\bar\rho g$. The two cancel exactly in the
continuum limit, and to $O(\Delta y^2)$ under discretisation
(§B3 + §C1 composite).

sympy verifies this cancellation as a symbolic identity:

$$\underbrace{-\frac{\Delta t}{\Delta y}\bigl[\bar p(y_{j+1/2}) - \bar p(y_{j-1/2})\bigr]}_{\text{flux divergence}} \;+\; \underbrace{-\Delta t\,\bar\rho_j\,g}_{\text{gravity source}} \;=\; 0. \quad (\text{D6-balance})$$

## Round-off drift

Under IEEE-754 double precision with standard (non-Kahan)
accumulation, the stored state accumulates round-off linearly in
the number of steps:

$$\|\mathbf{U}_{\mathrm{store}}\|_{\infty}(N_{\mathrm{step}}) \;\lesssim\; \varepsilon_{\mathrm{mach}}\,N_{\mathrm{step}}\,\kappa(\mathrm{HSE}), \quad (\text{D6-drift})$$

where $\kappa(\mathrm{HSE}) \sim O(1)$ is a problem-dependent
condition number (bounded by the ratio $\bar p_{\max}/\bar p_{\min}$
across the domain in the worst case; for the canonical HSE with
$y^\star \gg L_y$ this is $O(1)$).

At $N = 1000$ steps and $\varepsilon_{\mathrm{mach}} \approx 2.22\times
10^{-16}$, the expected drift bound is $\approx 2.2 \times 10^{-13}$.
The kernel's `d_rho, d_mx, d_my, d_E` should remain within this
bound.

If the kernel used **Kahan summation** at every accumulation
site, the drift would be $\varepsilon_{\mathrm{mach}} \sqrt{N}$,
roughly $7 \times 10^{-15}$ at $N = 1000$. The Strang kernel uses
standard (non-Kahan) accumulators, so the linear bound applies.

## Test tolerance

The golden JSON provides `comparison_tolerance = 1e-10`, much
looser than both theoretical bounds. This should always pass;
violations indicate structural issues (not round-off).

## Composite dependencies

This test is the **acceptance criterion** for all of Part B + §C1:

| If failed | Likely root cause |
|---|---|
| drift spikes at step 1 | §B3 face HSE is wrong (cell-centred bg), or §C1 source sign wrong |
| drift grows super-linearly | §B4/B5/B6 ghost fill doesn't preserve zero state (non-zero ghost drives flux) |
| drift is $O(\bar\rho g)$ per step | §C1 source sign flipped (gravity adds, not cancels, flux div) |
| energy component drifts, y-mom stays at 0 | §C1 energy source formula wrong ($-m_y g$ not applied) |
| drift is $O(\Delta y^2 \cdot \bar\rho g / c)$ per step | §B3 face HSE reconstruction is at cell-centred values (wrong face) |

## Golden values dump

Reference HSE profile at $N = 8192$ y-points for canonical
parameters ($\rho_0 = 1, K = 1, g = 1, L_y = 1, \gamma = 1.4$).
The test consumer reads `rho_bar_profile, p_bar_profile` and
compares against the kernel's `d_rho_bar, d_p_bar` to ULP
precision before running any Strang steps (separate §B2-level
check).

## Measurement protocol

1. Initialise kernel with canonical HSE parameters.
2. Confirm `d_rho, d_mx, d_my, d_E` buffers are all-zero.
3. Confirm `d_rho_bar, d_p_bar` match the reference profile to
   ULP precision.
4. Take $N_{\mathrm{step}} = 1000$ Strang steps at default CFL
   ($\sigma = 0.4$).
5. Download `d_rho, d_mx, d_my, d_E`; compute the infinity norm.
6. Required: max-norm drift $\le 10^{-10}$.

## Verification checkpoints

1. **Initial zero state.** After `init()`, the stored state is
   bitwise-zero. Test: `test_strang_init.cu` §D6-init-zero.

2. **Single-step drift.** After one Strang step on HSE IC, the
   max-norm drift is $\le 10 \varepsilon_{\mathrm{mach}} \bar\rho_{\max}$.
   Test: `test_strang_step.cu` §D6-one-step.

3. **Long-time drift.** After $N = 1000$ Strang steps, drift
   $\le 10^{-10}$. Test: `test_strang_step.cu` §D6-long-time.

4. **HSE background self-consistency.** The reference-profile
   `rho_bar_profile, p_bar_profile` matches the kernel's HSE
   background to ULP precision. Test: `test_strang_init.cu`
   §D6-hse-match.

Failure of (2) is a composition bug among §B2-§C1. Use the
failure diagnostics above to triage. Failure of (3) with
linear-in-$N$ drift is expected (round-off); only super-linear
or constant-offset drift indicates a bug. Failure of (4) means
the reference JSON is stale (regenerate via `bash run_all.sh`)
or the kernel's HSE builder parameters don't match the canonical.
