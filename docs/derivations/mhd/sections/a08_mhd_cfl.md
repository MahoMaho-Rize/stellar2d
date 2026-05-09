# A8. MHD CFL time-step constraints

> **sympy script:** `scripts/a8_mhd_cfl.py`
> **verified:** FTCS diffusion amplification worst-case at $\xi=\pi$
> gives $\sigma \le 1/2$; $c_f$ limits in three degenerate cases
> ($B_\perp=0$, $B_x=0$, $\mathbf{B}=0$).
> **code checkpoints:**
> `athena_mhd_kernels.cu::d_mhd_dt_reduction`,
> `athena_mhd_solver.cu::compute_dt`.

## Hyperbolic CFL (fast wave dominant)

Combining §A3 (7-wave eigensystem) with §A7 ($|\nu|\le 1$), the
unsplit multidimensional bound is

$$\boxed{\Delta t_{\mathrm{hyp}} \leq
C_{\mathrm{CFL}}\ \Bigg/ \sum_{d=1}^{D}\max_{\text{cells}}\!\left(\frac{|v_d| + c_{f,d}}{\Delta x_d}\right),\quad C_{\mathrm{CFL}} \leq 1,}$$

where the fast-magnetosonic speed $c_{f,d}$ in direction $d$ is the
positive root of the §A3 discriminant.

## Parabolic CFL (Ohmic + ambipolar diffusion)

For the scalar diffusion $\partial_t U = \eta \partial_x^2 U$ with FTCS
central space + forward Euler time, the amplification factor is
$g(\xi) = 1 - 4\sigma\sin^2(\xi/2)$ with $\sigma = \eta\Delta t/\Delta x^2$.
The worst case $\xi=\pi$ gives $g = 1 - 4\sigma$; stability $|g|\le 1$
requires $\sigma \le 1/2$ (sympy-verified).

Combining Ohmic + ambipolar as $\eta_\text{eff} = \eta_\Omega + \eta_{\mathrm{AD}}$:

$$\boxed{\Delta t_{\mathrm{para}} \leq \tfrac{1}{2}\min_{\text{cells}}\frac{\Delta x^{2}}{\eta_\Omega + \eta_{\mathrm{AD}}}.}$$

## Fast-speed behaviour at degenerate limits (A8-cf-limits)

Sympy-verified:

| Limit | $c_f^2$ |
|---|---|
| $B_\perp = 0$ (tangential-free) | $\max(c_{s_0}^2, c_{Ax}^2)$ |
| $B_x = 0$ (perpendicular only) | $c_{s_0}^2 + c_{A\perp}^2$ |
| $\mathbf{B} = 0$ (hydro) | $c_{s_0}^2$ |

The first two use `numerical fall-back` in sympy because `sp.Max` does
not simplify against the nested radical; 30 random states confirm
agreement to machine precision.

## Combined rule

$$\Delta t = \min\!\left(\Delta t_{\mathrm{hyp}},\,\Delta t_{\mathrm{para}}\right).$$

## Kernel-form reduction

Implementation-wise, per cell:

$$\left(\frac{1}{\Delta t}\right)_{\!i,j,k} =
\frac{|v_x| + c_f}{\Delta x} + \frac{|v_y| + c_f}{\Delta y} + \frac{|v_z| + c_f}{\Delta z} + 2\frac{\eta_\Omega + \eta_{\mathrm{AD}}}{\min(\Delta x,\Delta y,\Delta z)^{2}}.$$

Then $\Delta t^{-1}$ is reduced (`max`) over the grid and the global
$\Delta t$ is $C_\text{CFL} / \Delta t^{-1}_\text{max}$.

## Super-time-stepping (optional, future)

When $\eta$ is large enough that $\Delta t_{\mathrm{para}} \ll \Delta t_{\mathrm{hyp}}$,
the RKL2 super-time-stepping scheme (Meyer+12) advances the parabolic
part with $N$ sub-steps per hyperbolic step at $\mathcal{O}(N^2)$ CFL
relaxation. Not included in the initial `athena_mhd` implementation;
reserved for §C5 future extension when Suzuki turbulent heating is
active.

## ✅ Verification checkpoints

- `tests/test_athena_mhd_cfl_advection.cu` — advection of smooth
  wave at $C_\text{CFL} = 0.95$; no growth. At $C_\text{CFL} = 1.05$
  the test **must** go unstable within 10 steps (positive control).
- `tests/test_athena_mhd_ohm_diffusion.cu` — pure Ohmic field-loop
  decay; $\Delta t_\text{para}$ honored; analytic decay rate matched
  to $< 10^{-5}$.
