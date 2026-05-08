# C1. Ohmic (resistive) dissipation

> **sympy script:** `scripts/c1_ohmic_dissipation.py`
> **verified:** $\nabla\times(\nabla\times B) = \nabla(\nabla\cdot B) - \nabla^2 B$
> (all 3 components); $\nabla\times(\eta_O J) = \eta_O \nabla\times J + (\nabla\eta_O)\times J$.
> **code checkpoints:** `athena_mhd_kernels.cu::d_ohmic_dissipation`,
> `tests/test_mhd_ohmic_diffusion.cu`.

## Ohm's law with finite conductivity

$$\mathbf{E} = -\mathbf{v}\times\mathbf{B} + \eta_O\,\mathbf{J},
\qquad \mathbf{J} = \nabla\times\mathbf{B}. \quad (\text{C1-Ohm})$$

## Induction equation

**Constant $\eta_O$:**
$$\partial_t\mathbf{B} = \nabla\times(\mathbf{v}\times\mathbf{B}) + \eta_O\,\nabla^2\mathbf{B}. \quad (\text{C1-induction-const})$$

**Variable $\eta_O(\mathbf{r})$:**
$$\partial_t\mathbf{B} = \nabla\times(\mathbf{v}\times\mathbf{B})
+ \eta_O\,\nabla^2\mathbf{B} - (\nabla\eta_O)\times\mathbf{J}. \quad (\text{C1-induction-var})$$

Sympy verified via the vector identity
$\nabla\times(\eta_O\mathbf{J}) = \eta_O\nabla\times\mathbf{J} + (\nabla\eta_O)\times\mathbf{J}$.

## Joule heating

$$\boxed{Q_{\mathrm{Ohm}} = \eta_O |\mathbf{J}|^2 \ge 0.} \quad (\text{C1-Q})$$

Manifestly non-negative; enters internal-energy source.

## CFL for explicit diffusion

$$\Delta t \le \frac{(\Delta x)^2}{2\eta_O}\ (\mathrm{1D}),\
\frac{(\Delta x)^2}{4\eta_O}\ (\mathrm{2D}),\
\frac{(\Delta x)^2}{6\eta_O}\ (\mathrm{3D}). \quad (\text{C1-CFL})$$

At $r \approx 1000$ km in Matsuoka+24, $\eta_O \sim 10^{12}$ cm²/s
and $\Delta x \sim 1$ km, so $\Delta t_{\mathrm{Ohm}} \sim 10^{-4}$ s
— much smaller than hydro CFL. In practice, Matsuoka+24 use **super-
time-stepping** (Alexiades-Amiez-Gremaud 1996) to accelerate the
explicit Ohmic update.

## ✅ Verification

`tests/test_mhd_ohmic_diffusion.cu` — sinusoidal $B_y$ perturbation,
$\eta_O = $ const. Lock $L^2(B_y)$ exponential decay rate matches
analytical $e^{-\eta_O k^2 t}$ to $< 1\%$ at $N = 128$.
