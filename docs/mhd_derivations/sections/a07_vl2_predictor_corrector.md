# A7. Van-Leer 2 predictor-corrector (Stone-Gardiner 2009)

> **sympy script:** `scripts/a7_vl2_predictor_corrector.py`
> **verified:** amplification factor $g(\xi)=1-i\nu\xi-\tfrac{\nu^2}{2}\xi^2+\mathcal{O}(\xi^3)$
> (Lax-Wendroff 2nd-order); $|g|^2\le 1$ on $|\nu|\le 1$;
> truncation error $\mathcal{O}(\Delta t^3 + \Delta t\,\Delta x^2)$.
> **code checkpoints:**
> `athena_mhd_solver.cu::vl2_predictor_step`,
> `athena_mhd_solver.cu::vl2_corrector_step`.

## The scheme

$$\boxed{\mathbf{U}^{\star} = \mathbf{U}^{n} + \tfrac{\Delta t}{2}\mathcal{L}[\mathbf{U}^{n}],\qquad
\mathbf{U}^{n+1} = \mathbf{U}^{n} + \Delta t\,\mathcal{L}[\mathbf{U}^{\star}].}$$

This is **midpoint-RK2** in time, paired with any semi-discrete
operator $\mathcal{L}$ (here: PLM reconstruction + HLLD flux from §A4,
plus CT EMF update from §A5).

## Why VL2 over Strang or CTU

| Feature | VL2 | Strang | CTU |
|---|---|---|---|
| Dimensional coupling | Unsplit, exact | Split, $\mathcal{O}(\Delta t^2)$ error | Unsplit |
| Auxiliary state | $\mathbf{U}^\star$ | per-direction sweep | corner-transport cells |
| CFL limit (unsplit) | $\le 1$ | $\le 1$ in each direction | $\le 1$ (multi-D) |
| Compatibility with CT | Native (Stone-Gardiner 2009) | Requires extra corner EMF gymnastics | Native |
| Kernel complexity | Low (2 stages) | Low (per-direction) | Higher (CTU corner sweeps) |

Stone+08 CTU is more accurate on discontinuities but significantly
more complex; VL2 is the default in Athena++ and what we adopt.

## Fourier (von Neumann) analysis on linear advection

With PLM + upwind flux for $a>0$ on a smooth Fourier mode,

$$g(\xi) = 1 - i\nu\xi - \tfrac{\nu^2}{2}\xi^2 + \mathcal{O}(\xi^3),
\quad \nu \equiv a\Delta t / \Delta x,$$

**matching Lax-Wendroff** at leading orders. Sympy-verified the three
coefficients. Numerical sweep confirms $|g|^2 \le 1$ for
$\nu\in\{0.1,\ldots,1.0\}$; at $\nu = 1.05$ the amplification is
$|g|^2 \approx 1.22 > 1$, confirming the CFL limit is **sharp**.

## Truncation error (A7-consistency)

With the central-flux semi-discrete operator
$\mathcal{L}[U]_j = -a(U_{j+1}-U_{j-1})/(2h)$ acting on smooth $U$,
after two VL2 stages,

$$\boxed{\mathbf{U}^{n+1}_j - U(x_j, t+\Delta t) = \mathcal{O}(\Delta t^3 + \Delta t\,\Delta x^2).}$$

Leading error is a purely dispersive $U_{xxx}\,a(a^2-1)/6$ term,
vanishing at $\nu=1$ (exact for advection on-CFL). Sympy expands the
predictor/corrector composition and confirms the residual has no
$\mathcal{O}(\varepsilon^0 \ldots \varepsilon^2)$ terms (with
$\varepsilon = \Delta t = h$).

## CFL constraint (1D)

$$\Delta t \leq C_{\mathrm{CFL}}\,\frac{\Delta x}
{\max_{\text{cells}}(|v_x| + c_f)},\quad C_{\mathrm{CFL}} \leq 1.$$

See §A8 for multidimensional generalisation and parabolic terms.

## Implementation tips (for `athena_mhd_solver.cu`)

- Save $\mathbf{U}^n$ **before** the predictor so the corrector can
  rewind if a positivity check fails.
- The predictor need **not** compute the flux to full order — a 1st-
  order Godunov flux is enough; the corrector is where the 2nd-order
  accuracy is paid for. Stone+08 explicitly notes this economy.
- For CT, compute **face-centred** $E_z$ at both predictor and
  corrector; average at the corners using Gardiner-Stone 2005 (§A5).

## ✅ Verification checkpoints

- `tests/test_athena_mhd_linear_wave_convergence.cu` — three
  resolutions, fast wave, expect L¹ convergence slope in $[1.9,2.1]$.
- `tests/test_athena_mhd_vl2_stability.cu` — CFL scan
  $\nu\in\{0.5, 0.9, 0.99\}$ stable; $\nu = 1.05$ must blow up.
