# D5. Bubble (entropy-boost) canonical IC

> **sympy script:** `scripts/d05_bubble_init.py`
> **generated LaTeX:** `output/d05_bubble_init.latex.tex`
> **generated goldens:** `output/d05_bubble_init.goldens.json`
> **verifies:** 3 strong-form identities — 1 isentropic-to-$\rho$
> map ($\rho' = \bar\rho \exp(-\delta s / \gamma)$ at constant $P$);
> 1 linearised density perturbation; 1 zero $\delta E$ on static
> bubble IC
> **code checkpoints:**
> `src/gpu/explicit/strang_solver.cu :: StrangSolver::init_bubble`
> (line 765; kernel `k_strang_init_bubble` at line 708)

The bubble IC is the canonical convective test for the Strang
kernel: a warm (positive entropy anomaly) "bubble" embedded in
the HSE atmosphere, optionally modulated by an azimuthal mode.
The bubble is initially at pressure equilibrium with the
background, so the only initial perturbation is to density. Under
gravity it becomes buoyant and rises, eventually developing
Rayleigh-Taylor instability if the rise exceeds the atmosphere's
local scale height.

## IC ansatz

$$s(\mathbf{x}) \;=\; s_{\mathrm{bg}} \;+\; \delta s\,\exp\!\bigl(-r^{2}/R_{0}^{2}\bigr)\,\bigl(1 + \epsilon\,\cos(k_\theta \theta)\bigr), \quad (\text{D5-IC})$$

with $r = \sqrt{(x - x_0)^2 + (y - y_0)^2}$,
$\theta = \operatorname{atan2}(y - y_0, x - x_0)$, and the IC
enforces **constant pressure** $P = \bar p(y)$ (equilibrium with
HSE). The azimuthal mode with wavenumber $k_\theta$ and amplitude
$\epsilon$ is optional; with $\epsilon = 0$ the bubble is purely
radial.

## Isentropic closure → density perturbation

Given $s = \log(P \rho^{-\gamma})$ and $P = \bar p$ constant,
$\rho$ must adjust to the new entropy:

$$\rho(\mathbf{x}) \;=\; \bigl(\bar p / \exp(s)\bigr)^{1/\gamma} \;=\; \bar\rho(y)\,\exp\!\bigl(-\delta s_{\mathrm{local}} / \gamma\bigr), \quad (\text{D5-rho})$$

where $\delta s_{\mathrm{local}} = s(\mathbf{x}) - s_{\mathrm{bg}}$
is the local entropy excess. For $\delta s > 0$ (hot bubble),
$\rho < \bar\rho$ (density deficit) — the hot gas is less dense
at the same pressure, producing positive buoyancy.

## Linear-order density perturbation

For small $\delta s$:

$$\frac{\delta\rho}{\bar\rho} \;\approx\; -\frac{\delta s_{\mathrm{local}}}{\gamma} \;+\; O(\delta s^2). \quad (\text{D5-linear})$$

At $\delta s = 0.5$, $\gamma = 1.4$, the linear prediction is
$\delta\rho/\bar\rho \approx -0.357$; the full nonlinear value is
$\exp(-0.5/1.4) - 1 \approx -0.300$, a 16% correction at this
amplitude. Tests using strong bubbles ($\delta s \gtrsim 0.3$)
must use the full nonlinear formula.

## Zero energy perturbation

The stored $\delta E$ decomposes as (§B1):

$$\delta E \;=\; (P - \bar p)/(\gamma - 1) \;+\; \tfrac{1}{2}\rho (u^2 + v^2).$$

At IC, $P = \bar p$ (const-pressure bubble) and $u = v = 0$
(static), so both terms vanish:

$$\delta E \big|_{t=0} \;=\; 0. \quad (\text{D5-dE-zero})$$

This is useful for testing: after `init_bubble()` the stored
$\delta E$ buffer must be bitwise-zero at every cell. Any non-zero
value indicates an arithmetic error in the IC builder.

## Kernel implementation

`k_strang_init_bubble` (line 708-752 of strang_solver.cu) computes:

```cpp
// Simplified view:
double r2 = dx*dx + dy*dy;
double theta = atan2(dy, dx);
double local_ds = delta_s * exp(-r2 / (R0*R0)) *
                  (1 + epsilon * cos(k_theta * theta));
double rho_new = rho_bg * exp(-local_ds / gamma);
// Store perturbation:
d_rho[k] = rho_new - rho_bg;   // delta rho
d_mx[k]  = 0.0;
d_my[k]  = 0.0;
// Pressure-perturbation check:
// P = p_bg, so delta_E = 0 + 0 = 0.
d_E[k]   = 0.0;
```

The stored $\delta\rho$ can be negative (hot bubble has less
density than background). The stored $\delta E = 0$ exactly.

## Canonical parameters

| param | value | role |
|---|---|---|
| $x_0$ | 0.5 | bubble centre x |
| $y_0$ | 0.3 | bubble centre y |
| $R_0$ | 0.1 | bubble radius |
| $\delta s$ | 0.5 | entropy boost |
| $k_\theta$ | 3 | azimuthal mode |
| $\epsilon$ | 0.1 | azimuthal amplitude |

Background parameters (shared with HSE build):
$\rho_0 = 1.0$, $K = 1.0$, $g = 1.0$, $L_y = 1.0$, $\gamma = 1.4$.

The atmosphere cut-off is at $y^\star = \gamma K \rho_0^{\gamma-1} /
((\gamma - 1) g) = 1.4/0.4 = 3.5$, so the bubble is well within
the valid atmosphere.

## Golden values

Golden JSON dumps:

| field | purpose |
|---|---|
| scalar params | feed IC builder |
| `delta_rho_rel_ref_grid_64x64` | reference 2D array for test comparison |
| `delta_rho_rel_at_center` | closed-form $\exp(-\delta s/\gamma) - 1$ for sanity check |

## ✅ Verification checkpoint (to be wired)

1. **IC match.** After `init_bubble()` with canonical params on
   $n_x = n_y = 64$, the stored $\delta\rho / \bar\rho$ 2D array
   matches `delta_rho_rel_ref_grid_64x64` to ULP precision at
   every cell. Test: `test_strang_init.cu` §D5-profile.

2. **$\delta E = 0$.** All cells have stored $\delta E = 0$ to
   bitwise precision. Test: `test_strang_init.cu` §D5-dE-zero.

3. **Centre value.** At the cell containing the bubble centre,
   $\delta\rho / \bar\rho \approx \exp(-\delta s/\gamma) - 1
   \approx -0.300$ to ULP precision. Test:
   `test_strang_init.cu` §D5-center.

4. **Bubble rise rate (downstream).** After $t = 0.2$ of
   evolution, the bubble centre has risen by approximately
   $\Delta y \approx \tfrac{1}{2} g t^2 |\delta\rho/\rho| \approx
   0.2 \cdot 0.5 \cdot 1.0 \cdot 0.3 \approx 0.03$ (buoyancy-
   driven acceleration). Exact check depends on the scheme;
   this is a convergence-sanity rather than a strong-form
   check. Test: `test_strang_step.cu` §D5-rise.

Failure of (1) or (3) indicates an arithmetic bug in
`k_strang_init_bubble`. Failure of (2) means the kernel is
writing non-zero $\delta E$ — usually a cons2prim / delta-E
confusion in the IC code. Failure of (4) is a coarser
downstream check; typical failure modes include
too-aggressive HSE dissipation (bubble dissolves before it
rises) or over-strong entrainment (bubble fragments
prematurely).
