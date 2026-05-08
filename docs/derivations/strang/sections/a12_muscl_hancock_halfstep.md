# A12. MUSCL-Hancock half-step predictor

> **sympy script:** `scripts/a12_muscl_hancock_halfstep.py`
> **generated LaTeX:** `output/a12_muscl_hancock_halfstep.latex.tex`
> **verifies:** 3 strong-form identities — Hancock linear-advection
> equivalence ($u_{\mathrm{hancock}}$ equals $u_0 + (\Delta t/2)\,u_t$
> after substituting the PDE constraint $u_t = -a u_x$); 2nd-order
> time-truncation identity ($u_{\mathrm{true}} - u_{\mathrm{hancock}}
> = (\Delta t^2/8)\,a^2 u_{xx} + O(\Delta t^3)$); cell-average
> conservation of the half-step
> **code checkpoints:**
> `src/gpu/explicit/strang_solver.cu :: k_muscl_hancock_x`
> `src/gpu/explicit/strang_solver.cu :: k_muscl_hancock_y`

The Hancock predictor is the second pillar of MUSCL: after
reconstructing face states (§A11), it evolves those states
forward by a **half time step** using the physical flux, producing
time-centred values at $t_n + \Delta t / 2$ to feed the Riemann
solver. This makes MUSCL globally 2nd-order accurate in both
space and time while keeping the stencil local.

## Predictor formulas

For a cell $j$ with reconstructed face states $\mathbf{U}_{j+1/2,L}^n$
(at its right face, left of the Riemann problem) and
$\mathbf{U}_{j-1/2,R}^n$ (at its left face, right of the Riemann
problem),

$$\mathbf{U}^{n+1/2}_{L} \;=\; \mathbf{U}_{j+1/2, L}^{n} - \frac{\Delta t}{2h}\,\bigl[\mathbf{F}(\mathbf{U}_{j+1/2, L}^{n}) - \mathbf{F}(\mathbf{U}_{j-1/2, R}^{n})\bigr],$$

$$\mathbf{U}^{n+1/2}_{R} \;=\; \mathbf{U}_{j-1/2, R}^{n} - \frac{\Delta t}{2h}\,\bigl[\mathbf{F}(\mathbf{U}_{j+1/2, L}^{n}) - \mathbf{F}(\mathbf{U}_{j-1/2, R}^{n})\bigr]. \quad (\text{A12-Hancock})$$

Both face states are updated by the **same** flux-difference term;
this is what enforces cell-average conservation through the
half-step.

## Linear advection consistency (strong form)

For the scalar linear advection $u_t + a u_x = 0$ with constant
$a$, treating face value $u_0$ and its derivatives as independent
symbols and substituting the PDE constraint,

$$u_{\mathrm{hancock}} \;=\; u_0 - \frac{\Delta t}{2}\,a\,u_x \;\xrightarrow{u_t = -a u_x}\; u_0 + \frac{\Delta t}{2}\,u_t. \quad (\text{A12-linear-advect})$$

The right-hand side is the exact midpoint-time value to linear
order in $\Delta t$: $u(x_{j+1/2}, t_n + \Delta t / 2) = u_0 +
(\Delta t/2)\,u_t(x_{j+1/2}, t_n) + O(\Delta t^2)$. Hancock thus
reproduces the exact time-centred face value to linear order,
which is what makes the Godunov scheme 2nd-order accurate in
time under smooth IC.

## Leading 2nd-order truncation error

Expanding the exact midpoint-time value to three Taylor terms,

$$u(x_{j+1/2},\ t_n + \tfrac{\Delta t}{2}) \;=\; u_0 + \tfrac{\Delta t}{2}\,u_t + \tfrac{\Delta t^{2}}{8}\,u_{tt} + O(\Delta t^{3}).$$

Using the chain of PDE-derived identities $u_t = -a u_x$,
$u_{xt} = -a u_{xx}$, $u_{tt} = a^2 u_{xx}$, the difference
between the exact value and the Hancock predictor is

$$u_{\mathrm{true}} - u_{\mathrm{hancock}} \;=\; \tfrac{\Delta t^{2}}{8}\,a^{2}\,u_{xx}(x_{j+1/2}, t_n) + O(\Delta t^{3}). \quad (\text{A12-time-order})$$

**Strong-form verification.** sympy simplifies the difference
directly after substituting the three PDE rules; the result
equals $(\Delta t^2 / 8)\,a^2 u_{xx}$ identically.

This truncation term is a **positive dissipation** coefficient
($(\Delta t^2/8)\,a^2 > 0$), cancelled order-by-order at the
subsequent update step through the HLLC flux jump. The full
scheme, MUSCL-Hancock + HLLC + update, is verified 2nd-order
accurate in §E1 (modified-equation analysis).

## Cell-average conservation

Because both $\mathbf{U}^{n+1/2}_L$ and $\mathbf{U}^{n+1/2}_R$ are
updated with the **same** flux difference, the half-step
preserves cell averages:

$$\tfrac{1}{2}\bigl(\mathbf{U}^{n+1/2}_L + \mathbf{U}^{n+1/2}_R\bigr) \;=\; \tfrac{1}{2}\bigl(\mathbf{U}^{n}_L + \mathbf{U}^{n}_R\bigr) - \frac{\Delta t}{2h}\bigl[\mathbf{F}_R - \mathbf{F}_L\bigr]. \quad (\text{A12-conservation})$$

This is the **finite-volume half-step** identity: a cell-centred
integrator that happens to evolve both face states by the same
amount. It is the reason the kernel can use the Hancock-updated
face states directly as Riemann-problem inputs without worrying
about mid-cell-average drift: conservation is built in.

**Strong-form verification.** sympy directly simplifies the
averaged expression to the expected FV half-step form.

## ✅ Verification checkpoint (to be wired)

The kernel implements §A12 inside `k_muscl_hancock_x/y`. Tests:

1. **Smooth-IC time-order.** Run entropy-wave on
   $\{64^2, 128^2, 256^2, 512^2\}$ with fixed CFL;  $L^1$ should
   decrease as $h^2$ (2nd-order; the time error is subleading
   under Strang splitting). Test: `test_strang_convergence.cu`.

2. **Cell-average consistency.** After one Hancock half-step on
   a smooth IC, manually compute
   $\tfrac{1}{2}(\mathbf{U}^{n+1/2}_L + \mathbf{U}^{n+1/2}_R)$
   and compare to the FV finite-volume half-step formula applied
   to the cell average. Agreement to ULP precision. Test:
   `test_strang_muscl.cu` §A12-cell-avg.

3. **Leading-error magnitude.** On a specific smooth IC with
   known $u_{xx}$, measure $u_{\mathrm{kernel}}^{n+1/2} -
   u_{\mathrm{exact}}$ and verify it agrees with $(\Delta t^2/8)\,
   a^2 u_{xx}$ within 10% (the 10% slack is for accumulated
   round-off over many cells, not theoretical slack). Test:
   `test_strang_muscl.cu` §A12-truncation-check.

Failures of (1) indicate a deeper scheme-order bug, most likely
in the reconstruction of §A11 or the limiter of §A10 interacting
poorly with Hancock. Failure of (2) is a straight bug in the
kernel's flux-divergence computation. Failure of (3) is rare but
possible — it would indicate an arithmetic mistake in the
Hancock half-step formula itself.
