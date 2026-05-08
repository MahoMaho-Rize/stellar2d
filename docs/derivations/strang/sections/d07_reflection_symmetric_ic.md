# D7. Reflection-symmetric IC (bit-reproducibility test)

> **sympy script:** `scripts/d07_reflection_symmetric_ic.py`
> **generated LaTeX:** `output/d07_reflection_symmetric_ic.latex.tex`
> **generated goldens:** `output/d07_reflection_symmetric_ic.goldens.json`
> **verifies:** 13 strong-form identities — 1 involution ($\mathcal{R}_x^2 = \mathbf{I}$);
> 4 $\mathbf{F}_x$ reflection identities; 4 $\mathbf{F}_y$ reflection
> identities; 4 gravity-source invariance identities (all under
> x-reflection)
> **code checkpoints:**
> new `init_rt_symmetric()` IC builder in
> `src/gpu/explicit/strang_solver.cu` (book-anchored: the book
> says this IC must exist); new test
> `tests/test_strang_reflection_symmetry.cu`

An **x-reflection-symmetric** IC must evolve into an x-reflection-
symmetric state at all times. The test: start with
$\mathbf{U}(x, y) = \mathcal{R}_x \mathbf{U}(-x, y)$, evolve,
check that $\mathbf{U}(x, y, t) - \mathcal{R}_x \mathbf{U}(-x, y, t)$
stays at $O(\varepsilon_{\mathrm{mach}} N)$ for any $N$. If the
kernel's operator chain is **asymmetric** in any way (e.g.,
Riemann solver uses L/R asymmetrically, HSE build has $x$-dependent
round-off, ghost-cell fill order introduces drift), this test will
reveal the asymmetry as a non-zero symmetry residual.

This is the cleanest **bit-reproducibility** test for a 2D
shock-capturing kernel: any deviation is a structural bug, not a
physical phenomenon.

## Choice of reflection axis

Gravity in the Strang kernel points in the $-y$ direction. A
$y$-reflection would change the sign of gravity, breaking the
symmetry by the gravity source term. But an **$x$-reflection**
keeps gravity unchanged (it acts only on $y$), so the x-reflection
is compatible with the gravity-driven HSE atmosphere. This is the
axis chosen for §D7.

## x-reflection matrix

Under $x \to -x$: $u \to -u$, $m_x \to -m_x$, all other components
unchanged. On the conservative-state vector:

$$\mathcal{R}_x \;=\; \mathrm{diag}(+1,\, -1,\, +1,\, +1). \quad (\text{D7-R-x})$$

$\mathcal{R}_x^2 = \mathbf{I}$ (involution, sympy verified).

## Flux reflection identities

$$\mathbf{F}_x(\mathcal{R}_x \mathbf{U}) \;=\; \mathrm{diag}(-1, +1, -1, -1)\,\mathbf{F}_x(\mathbf{U}), \quad (\text{D7-flux-x})$$

$$\mathbf{F}_y(\mathcal{R}_x \mathbf{U}) \;=\; \mathcal{R}_x\,\mathbf{F}_y(\mathbf{U}). \quad (\text{D7-flux-y})$$

The $\mathbf{F}_x$ identity has sign pattern $(-1, +1, -1, -1)$
because three of the four $\mathbf{F}_x$ components contain $u$
linearly (mass, x-mom-diagonal absent, xy-mom, energy), and the
middle component $\rho u^2 + P$ is even in $u$.

The $\mathbf{F}_y$ identity is trivial: $\mathbf{F}_y$ has a $u$
factor only in the x-momentum component ($\rho u v$), so under
$u \to -u$ only that component flips — exactly matching
$\mathcal{R}_x$.

Both identities are sympy-verified component-wise.

## Gravity source invariance

$$\mathcal{R}_x \mathbf{S} \;=\; \mathbf{S}. \quad (\text{D7-source})$$

$\mathbf{S} = (0, 0, -\rho g, -m_y g)$ has zero in the
$m_x$-component (the only place $\mathcal{R}_x$ has a sign flip),
and the $m_y$ component $-m_y g$ is unchanged under $m_y
\mapsto m_y$ (which is the $+1$ entry of $\mathcal{R}_x$). The
gravity source is x-reflection invariant.

## Symmetry preservation

Combining:

1. **Initial symmetry.** $\mathbf{U}(x, y, 0) = \mathcal{R}_x
   \mathbf{U}(-x, y, 0)$ by IC construction.
2. **x-sweep preserves x-symmetry.** The x-sweep's Riemann
   problems at $x$ and $-x$ are related by
   $(\mathbf{U}_L(x), \mathbf{U}_R(x)) = (\mathcal{R}_x \mathbf{U}_R(-x),
   \mathcal{R}_x \mathbf{U}_L(-x))$ (the L-R roles are swapped
   by the x-flip), and the HLLC flux respects this by §A8
   symmetry.
3. **y-sweep preserves x-symmetry.** At each $x$, the y-sweep
   operates row-wise and does not mix x-positions. x-symmetry is
   preserved independently at each x.
4. **Gravity preserves x-symmetry.** By the source invariance
   above.

Therefore

$$\mathbf{U}(x, y, 0) \;=\; \mathcal{R}_x \mathbf{U}(-x, y, 0) \;\Longrightarrow\; \mathbf{U}(x, y, t) \;=\; \mathcal{R}_x \mathbf{U}(-x, y, t) \quad \forall\, t. \quad (\text{D7-preserve})$$

This should hold to $O(\varepsilon_{\mathrm{mach}} N)$ for the
kernel (where $N$ is the step count), since the only sources of
symmetry-breaking are floating-point round-off errors, which
accumulate linearly.

## Canonical IC

Two symmetric bubbles at positions $(0.3, 0.3)$ and $(0.7, 0.3)$,
mirror-image across $x = 0.5$:

- Bubble 1: $(x_0, y_0, R_0, \delta s) = (0.3, 0.3, 0.1, 0.5)$.
- Bubble 2: $(x_0, y_0, R_0, \delta s) = (0.7, 0.3, 0.1, 0.5)$.
- HSE background, gravity, etc. as in §D5 canonical.

At $t = 0$: $\rho(0.3, 0.3)$ and $\rho(0.7, 0.3)$ are bitwise
equal (same bubble, mirrored). At the reflection axis $x = 0.5$,
$u(0.5, y, 0) = 0$ by symmetry.

## Test procedure

1. Run `init_rt_symmetric()` (NEW IC builder — book-anchored
   per user rule) with the canonical parameters.
2. Evolve for $T = 0.1$ (roughly 50 steps at CFL 0.4).
3. Download state; compare $\mathbf{U}(x, y, T)$ with
   $\mathcal{R}_x \mathbf{U}(1 - x, y, T)$ cell-by-cell.
4. Required: max residual $\le 10^{-12}$ (much stricter than
   the $\varepsilon_{\mathrm{mach}} N \approx 10^{-14}$ upper
   bound; slack accounts for cumulative round-off through HLLC
   and MUSCL).

## Solver changes needed

Per the user rule "book is the anchor, solver follows": the
kernel currently does not have `init_rt_symmetric()` wired up.
This IC builder must be added to `strang_solver.cu` to satisfy
§D7's test coverage. The book drives this addition — not a
solver-first design decision.

Signature:

```cpp
void StrangSolver::init_rt_symmetric(
    double x0_L, double x0_R,   // bubble centers (mirrored)
    double y0,                  // common y
    double R0,                  // common radius
    double delta_s              // common entropy boost
);
```

Both bubbles use the same $y_0, R_0, \delta s$; only the $x_0$
differs. The asserted symmetry is $x_0^L + x_0^R = L_x$.

## ✅ Verification checkpoint (to be wired)

1. **IC symmetry.** After `init_rt_symmetric(0.3, 0.7, 0.3, 0.1, 0.5)`,
   $\delta\rho(x, y) - \delta\rho(1-x, y)$ is zero to ULP
   precision at every cell pair. Test:
   `test_strang_reflection_symmetry.cu` §D7-IC-sym.

2. **x-flip of $m_x$ on IC.** The initial $m_x$ is zero on
   symmetric IC (static bubbles, no flow), so $m_x(x, y) +
   m_x(1-x, y) = 0$ trivially. After one step, $m_x$ is small
   but $m_x(x) + m_x(1-x)$ stays at $O(\varepsilon_{\mathrm{mach}})$.
   Test: §D7-x-mom-antisymmetry.

3. **Long-time symmetry.** After 50 steps, max $(\mathbf{U}(x,y,t) -
   \mathcal{R}_x \mathbf{U}(1-x, y, t))$ stays $\le 10^{-12}$.
   Test: §D7-long-time.

4. **Azimuthal mode preservation.** If the IC includes a single
   bubble with azimuthal perturbation $\cos(k\theta)$ for even
   $k$ (e.g., $k = 2$), the x-reflection of this perturbation
   is itself, so the symmetry is exactly preserved. Test:
   §D7-azim-mode (extension).

Failure of (1) is an IC builder bug. Failure of (2) with non-
zero drift in $m_x$ on the IC would indicate the solver is
producing asymmetric momenta from symmetric IC — HLLC L/R bias.
Failure of (3) with steady linear drift in the symmetry residual
means the kernel has a structural asymmetry (most likely in the
ghost-cell fill order or Riemann-solver L/R handling). Failure
of (4) is a special test for the azimuthal IC; if (1)-(3) pass
but (4) fails, there is a specific angular-mode bug.
