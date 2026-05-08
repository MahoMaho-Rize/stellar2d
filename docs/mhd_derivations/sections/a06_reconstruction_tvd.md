# A6. Reconstruction (PLM/PPM) and TVD slope limiters

> **sympy script:** `scripts/a6_reconstruction_tvd.py`
> **verified:** minmod / vL / MC opposite-sign cancellation; Sweby
> TVD region $0\le\varphi(r)\le\min(2,2r)$ for $r\in[0,4]$; PLM
> reconstruction second-order; PPM 4-point interpolant 4th-order
> (exact to cubics); PPM parabola $\int_0^1 W\,d\xi = a_0$.
> **code checkpoints:**
> `athena_mhd_kernels.cu::d_reconstruct_primitive_plm`,
> `athena_mhd_kernels.cu::d_reconstruct_primitive_ppm`,
> `tests/test_athena_mhd_linear_wave_convergence.cu`.

## Why this section matters

VL2 + HLLD without a **monotonic** reconstruction is 2nd-order but
gives spurious oscillations at every shock. The three limiters below
are the only ones we need in practice: **minmod** (safest,
dispersive), **van Leer** (smooth, slightly sharper), **MC** (most
aggressive, Stone+08 default).

A skipped step here directly causes the class of bugs already seen on
stellar2d:

- `cart_ale2` swept-remap used donor-cell then MUSCL without
  characteristic projection — caused the periodic-BC drift at
  `P30/P31`.
- Stone+08 App. A is explicit: **reconstruct in primitive variables,
  then project onto characteristic variables** via the eigenvectors
  of §A3. Skipping the projection converts crisp fast-mode jumps into
  diffusive blobs.

## Definitions

$$\sigma_{\text{minmod}}(a,b) =
\begin{cases} 0, & \mathrm{sign}(a)\neq\mathrm{sign}(b),\\
\mathrm{sign}(a)\,\min(|a|,|b|), & \text{else}.\end{cases}$$

$$\sigma_{\text{vL}}(a,b) =
\begin{cases} 0, & ab\le 0,\\
\dfrac{2ab}{a+b}, & ab>0.\end{cases}$$

$$\sigma_{\text{MC}}(a,b) = \mathrm{sign}(a)\,\min\!\left(2|a|,\tfrac{|a+b|}{2},2|b|\right)
\text{ when } ab>0, \text{ else } 0.$$

## Sweby TVD region (A6-Sweby)

All three limiters satisfy, with $r = b/a$:

$$\boxed{0 \leq \varphi(r) \leq \min(2,\,2r),\quad r\ge 0.}$$

**Sympy-verified** by numerical sweep ($r\in[0,4]$, 401 samples).
Any limiter in this region is TVD for 1D scalar advection with
CFL $\le 1$ (Harten 1983; LeVeque 2002 §16).

## PLM reconstruction and order

$$W^{L}_{i+1/2} = W_i + \tfrac{1}{2}\sigma_i,\quad
\sigma_i = \tfrac{1}{2}(W_{i+1} - W_{i-1}).$$

**Leading error** $W^{L}_{i+1/2} - W(x_{i+1/2}) = -\tfrac{h^{2}}{8}W''(x_{i+1/2}) + \mathcal{O}(h^{3})$,
so the reconstruction is **second order** (no $\mathcal{O}(h)$ term).
Sympy-verified: no $h^0$ or $h^1$ terms in the residual expansion.

## PPM (Colella-Woodward 1984) 4-point interpolant

$$\boxed{W_{i+1/2} = \tfrac{7}{12}(W_i + W_{i+1})
  - \tfrac{1}{12}(W_{i-1} + W_{i+2}),}$$

applied to the **cell-averaged** values. Sympy-verified: no
$\mathcal{O}(h^0..h^3)$ term in the expansion against the smooth
reference; leading error is $-\tfrac{1}{30} h^{4} W^{(4)}$.

The parabolic form inside a cell (Colella-Woodward 1984 Eq. 1.6),

$$W(\xi) = a_L + \xi[\Delta a + a_6(1 - \xi)], \quad
\Delta a = a_R - a_L,\quad a_6 = 6(a_0 - (a_L + a_R)/2),$$

satisfies $\int_0^1 W\,d\xi = a_0$ **exactly** (sympy-verified):
conservation of cell averages is built in.

## Practical recommendation (Stone+08 Appendix A)

1. Reconstruct $(\rho, \mathbf{v}, \mathbf{B}, p)$ in **primitive**
   variables (not conservative). This keeps positivity of $\rho, p$
   easier to enforce.
2. Project onto characteristic variables
   $\delta W^{(k)} = \ell_k \cdot \delta W$ using the left-
   eigenvectors of §A3, limit each wave family separately, then
   project back $\delta W = \sum_k r_k \, \delta W^{(k)}$.
3. Use MC as the default; switch to van Leer near strong shocks if
   MC produces staircase artefacts (rare but documented in Stone+08
   Fig 28).

## ✅ Verification checkpoints

- `tests/test_athena_mhd_linear_wave_convergence.cu` — 3-resolution
  linear fast-wave advection; L¹ convergence slope in $[1.9, 2.1]$.
- `tests/test_athena_mhd_shock_tube.cu` — Brio-Wu, MC limiter, no
  staircase within the fast rarefaction fan.
