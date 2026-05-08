# D3. Sod shock tube canonical IC

> **sympy script:** `scripts/d03_sod_shock_tube.py`
> **generated LaTeX:** `output/d03_sod_shock_tube.latex.tex`
> **generated goldens:** `output/d03_sod_shock_tube.goldens.json`
> **verifies:** 2 strong-form sympy identities (rarefaction and
> shock f-functions vanishing at $p = P_K$); plus closed-form
> Newton solution for $p^\star$ at 15-digit precision
> **code checkpoints:**
> new `init_sod()` IC builder in `src/gpu/explicit/strang_solver.cu`;
> `tests/test_strang_sod.cu` (new test — wire into CMakeLists)

Sod's shock-tube (Sod 1978) is the canonical Riemann problem:
the diaphragm breakup generates a left-going rarefaction, a
contact discontinuity, and a right-going shock. The solution is
closed-form (modulo a transcendental Newton iteration for
$p^\star$), making it an excellent test for shock-capturing
schemes' resolution of all three wave types simultaneously.

## Initial condition

At $t = 0$, discontinuity at $x = 0$ on $x \in [-0.5, 0.5]$:

$$\begin{aligned}\text{Left: }\;&\rho_L = 1.0,\quad u_L = 0,\quad P_L = 1.0, \\ \text{Right: }\;&\rho_R = 0.125,\quad u_R = 0,\quad P_R = 0.1, \\ \gamma &= 1.4.\end{aligned} \quad (\text{D3-IC})$$

## Riemann fan structure

Starting from the $t > 0$ similarity solution (functions of
$\xi = x/t$):

$$\text{L state} \to \text{rarefaction} \to \text{left star} \to \text{contact} \to \text{right star} \to \text{shock} \to \text{R state}. \quad (\text{D3-fan})$$

Five intervals on $\xi$, separated by four wave speeds
($S_{HL} < S_{TL} < S_C < S_R$). The "star" region is the
intermediate state between the contact and the adjacent wave
(left rarefaction tail or right shock), sharing pressure
$p_L^\star = p_R^\star = p^\star$ and velocity $u_L^\star =
u_R^\star = u^\star$ per §A8.

## Star-region equation

The pressure $p^\star$ satisfies the transcendental

$$u_L - u_R \;=\; f_L(p^\star; \rho_L, P_L) \;+\; f_R(p^\star; \rho_R, P_R), \quad (\text{D3-f})$$

where each $f_K$ selects rarefaction or shock depending on the
sign of $p^\star - P_K$:

$$f_K(p) \;=\; \begin{cases}(p - P_K)\,\sqrt{\dfrac{A_K}{p + B_K}}, & p > P_K\;\text{(shock)} \\ \dfrac{2 c_K}{\gamma - 1}\,\bigl[(p / P_K)^{(\gamma-1)/(2\gamma)} - 1\bigr], & p \le P_K\;\text{(rarefaction)}\end{cases}$$

with $A_K = 2/(\rho_K (\gamma+1))$, $B_K = \frac{\gamma-1}{\gamma+1} P_K$.

For Sod, $p^\star < P_L$ (rarefaction left) and $p^\star > P_R$
(shock right).

## Numerical solution

Newton's method on the residual $f_L(p^\star) + f_R(p^\star) - (u_L - u_R)$
(with $u_L = u_R = 0$) converges in $\sim 10$ iterations from the
arithmetic-mean initial guess. The script computes all derived
quantities to full double-precision:

| quantity | value |
|---|---|
| $p^\star$ | 0.303 130 178 050 647 |
| $u^\star$ | 0.927 452 620 048 950 |
| $\rho^\star_L$ | 0.426 319 428 178 495 |
| $\rho^\star_R$ | 0.265 573 711 705 307 |
| $S_{HL}$ (rarefaction head) | $-1.183$ 215 956 619 923 |
| $S_{TL}$ (rarefaction tail) | $-0.070$ 272 812 561 183 |
| $S_C$ (contact) | 0.927 452 620 048 950 |
| $S_R$ (shock) | 1.752 155 732 030 178 |

## Reference profile

At $t = T = 0.2$, sampled at $N = 200$ uniform x-points on
$[-0.5, 0.5]$. The sampling function handles each region:

- Inside the rarefaction fan: Riemann invariants give
  $c(\xi) = \tfrac{2}{\gamma+1}[c_L + \tfrac{\gamma-1}{2}(u_L - \xi)]$,
  $u(\xi) = \tfrac{2}{\gamma+1}[c_L + \tfrac{\gamma-1}{2} u_L + \xi]$,
  with $\rho = \rho_L (c/c_L)^{2/(\gamma-1)}$,
  $P = P_L (c/c_L)^{2\gamma/(\gamma-1)}$.
- Star regions: constant.
- Outside the fan / shock: IC states.

## Verification identities (sympy)

Two closed-form identities are symbolically verified:

1. **f-function rarefaction zero:** $f_{\mathrm{rar}}(P_K; c_K, P_K) = 0$.
   Trivial; sympy simplifies the $(P_K/P_K)^{\alpha} - 1 = 0$ term
   directly.

2. **f-function shock zero:** $f_{\mathrm{shock}}(P_K; \rho_K, P_K) = 0$.
   By the $(p - P_K)$ factor.

These anchor the Newton iteration: at $p^\star = P_L$ or $p^\star
= P_R$, one branch vanishes. For a **constant-pressure** Riemann
problem ($P_L = P_R$, $u_L = u_R$ — the trivial case), the
solution is $p^\star = P_L = P_R$ and both branches contribute
zero, consistent with the trivial Riemann solution being the
IC itself.

The **contact invariant** $p_L^\star = p_R^\star = p^\star$ is a
§A8 strong-form identity, propagated without re-derivation.

## Measurement protocol

1. Initialise kernel with the Sod IC at $n_x = 200$.
2. Evolve for $T = 0.2$.
3. Download $\rho, m_x, m_y, \delta E$; reconstruct $(\rho, u, P)$.
4. Compare against golden `rho_profile, u_profile, P_profile` at
   $x = $ cell centres.
5. Compute $L^1$ norm of the error; required: $L^1 < 10^{-2}$ at
   $n_x = 200$ (the scheme's typical Sod accuracy).

For **convergence test** mode, run at $n_x \in \{100, 200, 400, 800\}$
and fit slope; expected $p \approx 1.0$ (1st order through shocks
and contacts — Godunov limit) with better rate on the rarefaction
interior.

## ✅ Verification checkpoint (to be wired)

1. **IC consistency.** After `init_sod()`, cells with $x_c < 0$
   have $(\rho, u, v, P) = (1.0, 0, 0, 1.0)$ and cells with
   $x_c > 0$ have $(0.125, 0, 0, 0.1)$. Test:
   `test_strang_sod.cu` §D3-IC.

2. **Star-region numerical match.** Measure post-step $p^\star$
   across the contact (middle of the star region); required
   $|p^\star_{\mathrm{measured}} - p^\star_{\mathrm{golden}}| / p^\star_{\mathrm{golden}} < 0.05$
   (5% tolerance for finite-$n_x$ resolution). Test:
   `test_strang_sod.cu` §D3-star-match.

3. **Wave-speed tracking.** Measure the positions of the shock,
   contact, and rarefaction tail at $T = 0.2$; each must lie
   within 1.5 cells of the analytic position $S_\cdot \cdot T$.
   Test: `test_strang_sod.cu` §D3-wave-positions.

4. **Entropy monotonicity at shock.** Across the right shock the
   entropy $s = \log(P/\rho^\gamma)$ must strictly increase
   post-shock vs pre-shock; required $\Delta s > 0$ (§A5 Lax
   condition). Test: `test_strang_sod.cu` §D3-shock-entropy.

Failure of (1) is a direct IC bug. Failure of (2) usually means
the HLLC middle-state formula (§A8) has a bug or the Davis wave
speeds (§A9) are incorrect. Failure of (3) is a typical
convergence-level issue — can be expected at $n_x = 100$ and
lower, but a systematic shock-lag signal indicates a flux
inconsistency. Failure of (4) is a serious Lax-condition
violation; the scheme is producing negative entropy and is
unstable.
