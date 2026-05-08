# D4. Woodward-Colella two-blast-wave interaction

> **sympy script:** `scripts/d04_woodward_colella_blast.py`
> **generated LaTeX:** `output/d04_woodward_colella_blast.latex.tex`
> **generated goldens:** `output/d04_woodward_colella_blast.goldens.json`
> **verifies:** 2 closed-form Riemann-problem solutions (at
> $x = 0.1$ and $x = 0.9$, early-time window) via §D3's Newton
> routine; late-time profile is **[WEAK]** per Rule 4 (no closed
> form after shock-shock collision)
> **code checkpoints:**
> new `init_woodward_colella()` IC builder in
> `src/gpu/explicit/strang_solver.cu`; `tests/test_strang_wc_blast.cu`
> wired to golden JSON

The Woodward & Colella (1984) "two blast waves" test has three
high-pressure / low-pressure / high-pressure regions separated by
two diaphragms. Upon release, two shocks propagate inward into the
low-pressure middle region, collide, reflect off the walls, and
interact with each other in a highly nonlinear fashion. The test
is a standard benchmark for:

1. **Shock-capturing accuracy at high pressure ratios**
   ($P_L / P_M = 10^5$).
2. **Robustness of Riemann solvers under wave collision** (two
   strong shocks meeting head-on).
3. **Boundary-reflection handling** (both outer boundaries reflect
   the trailing rarefaction and returning shocks).

## Initial condition

On $x \in [0, 1]$ with reflective walls at both ends:

$$\begin{aligned}0 \le x < 0.1 \;&:\; \rho = 1,\; u = 0,\; P = 1000 \\ 0.1 \le x < 0.9 \;&:\; \rho = 1,\; u = 0,\; P = 0.01 \\ 0.9 \le x \le 1.0 \;&:\; \rho = 1,\; u = 0,\; P = 100 \\ \gamma \;&=\; 1.4.\end{aligned} \quad (\text{D4-IC})$$

## Early-time window: two independent Riemann problems

For $t < t_{\mathrm{collision}} \approx 0.026$, the inward-moving
shocks have not yet met and the two Riemann problems (at
$x = 0.1$ and $x = 0.9$) evolve independently. Using §D3's
Newton routine, the script computes both star-region states
closed-form:

| quantity | Left (x = 0.1, L-blast vs. M) | Right (x = 0.9, M vs. R-blast) |
|---|---|---|
| $p^\star$ | 460.89 | 46.10 |
| $u^\star$ | $+19.60$ | $-6.20$ |
| $\rho^\star_L$ | 0.575 | 5.992 |
| $\rho^\star_R$ | 5.999 | 0.575 |
| outward shock speed | $S_R = 23.52$ (rightward) | $S_L = -7.44$ (leftward) |

Both are strong shocks (post-shock density compression $\sim 6\times$,
close to the strong-shock limit of $(\gamma+1)/(\gamma-1) = 6$ for
$\gamma = 1.4$). The post-shock velocities are high-Mach.

## Collision time

Inward-moving shock speeds from above: $+23.52$ from the left
blast, $-7.44$ from the right blast. They collide when their
positions equalise:

$$0.1 + 23.52\,t \;=\; 0.9 + (-7.44)\,t \;\;\Longrightarrow\;\; t_{\mathrm{collision}} \;\approx\; \frac{0.8}{30.96} \;\approx\; 0.026. \quad (\text{D4-t-coll})$$

The test's standard final time is $T = 0.038$, which is **after**
the collision — the test specifically probes the post-collision
regime.

## Late-time window: **[WEAK]**

After shock-shock collision, no closed-form analytic solution
exists. The test validates by:

1. **High-resolution reference run.** Run the same IC on
   $n_x = 3200$ cells (far beyond the converged resolution) and
   treat this as "truth".
2. **L1-integrated comparison.** For lower-resolution runs
   ($n_x = 100, 200, 400, 800$), compute the $L^1$ norm of the
   density error against the $n_x = 3200$ reference. Expect
   convergence rate $p \sim 1$ (shocks dominate; 1st-order under
   Godunov-limit analysis).
3. **Feature preservation.** Visual inspection of the density
   profile at $T = 0.038$ should show:
   - A persistent high-density peak near $x \approx 0.65$ (post-
     collision contact discontinuity).
   - Multiple weaker shock fronts from the wave interactions.
   - The characteristic two-peak structure of the Woodward-Colella
     solution.

Per Rule 4, the late-time reference is **[WEAK]**: it is
numerical, not symbolic. The goldens JSON marks this explicitly:

```json
"WEAK_caveat": "Late-time t = 0.038 profile has NO closed-form
solution; reference comparison is against a high-resolution
(N >= 3200) run, L1 diff measured against lower-resolution
runs."
```

The early-window verification (closed-form Riemann solutions at
$t \lesssim 0.025$) is strong-form and provides the pointwise
test anchor.

## Measurement protocol

**Early window test** (strong-form oracle, $t = 0.02$):
- Run $n_x = 1000$.
- Compare at $t = 0.02$ (before collision). Expected: two
  independent Riemann fans; star-region values match the closed-
  form predictions within 5%.

**Late window test** (weak-form, $t = 0.038$):
- Run the reference at $n_x = 3200$.
- Run test resolutions at $n_x \in \{100, 200, 400, 800\}$.
- Compute $L^1$ norm of $\rho$ against the reference.
- Fit slope; expected $p \sim 0.8{-}1.1$ (somewhere between 1st-
  order for shocks and a sub-linear rate from wave-interaction
  complexity).

## ✅ Verification checkpoint (to be wired)

1. **Early-window star-region match.** At $t = 0.02$, measured
   $p^\star$ at $x = 0.3$ (within the left blast's post-shock
   region) agrees with closed-form $p^\star \approx 461$ within
   3%. Test: `test_strang_wc_blast.cu` §D4-early.

2. **Shock position tracking.** At $t = 0.02$, the left-blast's
   outgoing shock has reached $x \approx 0.1 + 23.52 \cdot 0.02 =
   0.57$; tolerance $\pm 2$ cells. Test:
   `test_strang_wc_blast.cu` §D4-shock-position.

3. **Late-window L1 convergence.** $L^1$ slope over four
   resolutions in $[0.7, 1.2]$. Test:
   `test_strang_wc_blast.cu` §D4-late-convergence.

4. **Entropy floor preservation.** No cell has $P \le 0$ at any
   time during the simulation. Test:
   `test_strang_wc_blast.cu` §D4-positivity.

5. **Reflective-wall energy conservation.** $\int_V (E + \rho g y) dV$
   (with $g = 0$ for this test) stays within $10\varepsilon_{\mathrm{mach}}
   N$ of the initial value. Test:
   `test_strang_wc_blast.cu` §D4-energy-conservation.

Failure of (1) is a Riemann-solver bug (probably §A8 formula
error). Failure of (2) is a scheme-speed bug (wrong wave-speed
estimation in §A9). Failure of (3) with a slope far below 0.7 is
a deeper convergence issue. Failure of (4) indicates positivity
preservation is broken — the HLLC or floor logic has a bug
triggered at high pressure ratios. Failure of (5) points to a
wall-BC bug (§B5 reflective); the blast-wave geometry stresses
this more than simpler tests.
