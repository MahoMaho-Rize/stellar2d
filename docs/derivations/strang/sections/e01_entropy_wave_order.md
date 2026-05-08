# E1. Entropy-wave convergence order prediction

> **sympy script:** `scripts/e01_entropy_wave_order.py`
> **generated LaTeX:** `output/e01_entropy_wave_order.latex.tex`
> **verifies:** 6 strong-form identities — 1 at-least-1st-order
> (per-step $O(h)$ coefficient = 0); 1 at-least-2nd-order
> ($O(h^2)$ coefficient = 0); 1 leading $O(h^3)$ coefficient
> identity ($\nu (2\nu^2 - 3\nu + 1) / 12 \cdot u_{xxx}$); 3
> "magic CFL" identities (error vanishes at $\nu = 0, 1/2, 1$)
> **code checkpoints:**
> entire MUSCL-Hancock + MC + HLLC x-sweep pipeline; measured by
> `tests/test_strang_convergence.cu` against §D1 goldens

Modified-equation analysis of the MUSCL-Hancock + MC + HLLC scheme
applied to the smooth entropy-wave IC of §D1. Strong-form Taylor
expansion of the discrete update reveals the **per-step** leading
truncation at $O(\Delta x^3)$, giving a **global** $L^1$ error
scaling of $O(\Delta x^2)$ after $N \sim 1/\Delta x$ steps. The
predicted convergence slope is $p = 2.0$.

## Discrete update

On the linear advection $u_t + u_0 u_x = 0$ (which the entropy
wave obeys):

$$u_j^{n+1} \;=\; u_j^n \;-\; \nu\,\bigl(u_{j+1/2}^L - u_{j-1/2}^L\bigr), \qquad \nu \;=\; \frac{u_0\,\Delta t}{\Delta x}. \quad (\text{E1-update})$$

The face states $u_{j\pm 1/2}^L$ come from MUSCL reconstruction
(§A11) plus Hancock half-step (§A12):

$$u_{j+1/2}^L \;=\; u_j \;+\; \tfrac{1}{2}\bigl(\Delta x - u_0\,\Delta t\bigr)\,\sigma_j, \qquad \sigma_j \;=\; \frac{u_{j+1} - u_{j-1}}{2\,\Delta x}, \quad (\text{E1-face})$$

where the MC limiter (§A10) reduces to the central difference on
smooth data. HLLC on the entropy wave gives pure upwind flux (§D1),
so the discrete update is exactly the formula above.

## Taylor expansion

Expand $u(x_{j\pm 1}, t_n)$ around $(x_j, t_n)$ to $O(\Delta x^5)$,
and $u(x_j, t^{n+1})$ via the exact advection $u(x, t + \Delta t) =
u(x - u_0 \Delta t, t)$. The residual $u_j^{n+1} - u_{\mathrm{exact}}(x_j,
t^{n+1})$, written with $\Delta t = \nu \Delta x / u_0$ so that
all terms are in $\Delta x$:

- $O(\Delta x^1)$ coefficient: **0** (sympy verified)
- $O(\Delta x^2)$ coefficient: **0** (sympy verified)
- $O(\Delta x^3)$ coefficient: $\dfrac{\nu(2\nu^2 - 3\nu + 1)}{12}\,u_{xxx}$

## Leading truncation (strong form)

$$u_j^{n+1} \;-\; u_{\mathrm{exact}}(x_j, t^{n+1}) \;=\; \frac{\nu(2\nu^2 - 3\nu + 1)}{12}\,\Delta x^3\,u_{xxx}(x_j, t_n) \;+\; O(\Delta x^4). \quad (\text{E1-trunc})$$

sympy verifies this coefficient exactly. The factor $\nu(2\nu^2 -
3\nu + 1) = \nu(2\nu - 1)(\nu - 1)$ has three roots at $\nu = 0,
1/2, 1$, so the leading truncation **vanishes** at these CFL
numbers:

- $\nu = 0$: trivial (no time advance, no error).
- $\nu = 1/2$: "magic" CFL where the MUSCL-Hancock face state
  lands exactly at the half-step upwind interpolant.
- $\nu = 1$: courant-number-1 exact advection (the scheme is
  exact for linear advection at $\nu = 1$, which is a classical
  Warming-Beam / Lax-Wendroff property).

For the kernel's $\sigma = 0.4$, $\nu = 0.4$, the coefficient is
$0.4 \cdot (0.32 - 1.2 + 1) / 12 = 0.4 \cdot 0.12 / 12 = 0.004$ —
small but non-zero.

## Global convergence

Per-step error is $O(\Delta x^3)$. Number of steps to reach a
fixed time $T$ is $N = T/\Delta t = T u_0 / (\nu \Delta x)$, which
is $O(1/\Delta x)$. The global error is

$$\|u_{\mathrm{num}} - u_{\mathrm{exact}}\|_{L^1} \;\sim\; N \cdot \Delta x^3 \;=\; O(\Delta x^2). \quad (\text{E1-slope})$$

So the predicted convergence slope is

$$p \;=\; 2.0 \pm 0.1. \quad (\text{E1-pred})$$

## Strang-split compatibility

The §D1 entropy wave has $v = 0$ uniformly. The y-sweep's HLLC
Riemann problems see $v_L = v_R = 0$, which gives identical L/R
states (after accounting for §B3 HSE background, which doesn't
enter a no-gravity, constant-P test). The y-sweep contributes
exactly zero update. Therefore Strang-split entropy-wave
convergence = 1D entropy-wave convergence, giving the same $p = 2.0$.

## Measurement protocol

For $n_x \in \{64, 128, 256, 512\}$:

1. IC from §D1 at `N_ref = 4096` (interpolated / averaged down to
   $n_x$).
2. Run to $T = L_x / u_0 = 1$ with `use_lm_fix = true`
   ($M_{\mathrm{loc}} = 1$ so $f_M = 1$ anyway).
3. Download $\rho(x, y, T)$, project to 1D along a row.
4. Compute $L^1 = \sum_i |\rho_i(T) - \rho_i(0)| \Delta x$.
5. Fit $\log L^1$ vs $\log n_x$; slope expected in $[1.8, 2.2]$.

## ✅ Verification checkpoint (to be wired)

1. **Slope match.** $p \in [1.8, 2.2]$ at the four resolutions.
   Test: `test_strang_convergence.cu` §E1-slope.

2. **Magic CFL.** Running at $\nu = 0.5$ (setting `cfl_number`
   so that the tightest cell has $\nu = 0.5$) should give
   $L^1$ error lower than at $\nu = 0.4$ by a factor of $\sim
   (0.4 \cdot 0.12)/(0.5 \cdot 0)$ — i.e., orders of magnitude
   lower (ideally machine precision). Test:
   `test_strang_convergence.cu` §E1-magic.

3. **No y-variation.** At the measurement time $T$, the kernel's
   $\rho(x, y, T)$ should be independent of $y$ (to round-off
   precision). Failure indicates the y-sweep is contaminating
   the 1D entropy-wave test. Test: §E1-1D-check.

Failure of (1) with slope below 1.8 means the scheme is
effectively 1st-order (e.g., the limiter is clamping in smooth
regions, which would be an MC-limiter bug). Failure of (2) is an
exotic feature test; it confirms the modified-equation analysis
is correct. Failure of (3) indicates a structural y-sweep bug
active even at $v = 0$.
