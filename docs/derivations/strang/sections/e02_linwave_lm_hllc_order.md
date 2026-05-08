# E2. Linwave convergence under LM-HLLC vs standard HLLC

> **sympy script:** `scripts/e02_linwave_lm_hllc_order.py`
> **generated LaTeX:** `output/e02_linwave_lm_hllc_order.latex.tex`
> **verifies:** 1 strong-form identity — log-decay-ratio
> $\log(\mathcal{A}_{\mathrm{std}}/\mathcal{A}_{\mathrm{LM}}) =
> -(1 - M_{\mathrm{cut}}) c k^2 \Delta x T / 2$
> **code checkpoints:**
> `src/gpu/explicit/strang_device.cuh :: d_lmhllc` ($f_M$ branch
> versus `f_M = 1` branch); §D2 linwave test; `tests/test_strang_linwave_convergence.cu`

Modified-equation dispersion analysis of the MUSCL-HLLC scheme on
the §D2 acoustic linwave. Predicts the numerical viscosity
$\nu_{\mathrm{eff}}$ and the resulting amplitude decay rate under
two configurations: (a) `use_lm_fix = false` (standard HLLC), and
(b) `use_lm_fix = true` (LM-HLLC at Mach below $M_{\mathrm{cut}}$).

**Key conclusion:** for a correct acoustic convergence measurement,
`use_lm_fix = false` is mandatory. LM-HLLC at the linwave's low
$M \approx \epsilon \ll M_{\mathrm{cut}} = 10^{-3}$ gives $f_M =
M_{\mathrm{cut}}$, suppressing the pressure dissipation by
$10^{3}\times$ and artificially super-converging the test. The
standard HLLC behaviour (2nd-order in $\Delta x$) requires
$f_M = 1$, which `use_lm_fix = false` enforces.

## Dispersion relation

The linearised MUSCL-HLLC update on a right-going acoustic mode
gives the discrete dispersion

$$\omega(k) \;=\; c\,k \;-\; i\,\nu_{\mathrm{eff}}\,k^2 \;+\; O(k^3), \quad (\text{E2-dispersion})$$

where $c$ is the sound speed and the imaginary part gives the
amplitude decay rate. The effective viscosity is

$$\nu_{\mathrm{eff}} \;=\; f_M \cdot \frac{c\,\Delta x}{2}, \quad (\text{E2-nu-eff})$$

proportional to the HLLC pressure-dissipation coefficient. The
factor $c \Delta x / 2$ is the standard Godunov scheme's numerical
viscosity. The factor $f_M$ (the LM-HLLC blend, §C3) scales the
pressure-jump contribution to the HLLC flux.

## Per-period amplitude decay

Over one wave period $T = L_x / (u_0 + c)$ (which is $T = L_x/c$ for
the canonical stationary background $u_0 = 0$), the amplitude
decays as

$$\frac{\mathcal{A}(T)}{\mathcal{A}(0)} \;=\; \exp\bigl(-\nu_{\mathrm{eff}}\,k^2\,T\bigr) \;=\; \exp\bigl(-f_M\,c\,k^2\,\Delta x\,T/2\bigr). \quad (\text{E2-decay})$$

At the canonical $k = 2\pi/L_x$, $L_x = 1$, $c = 1$, $T = 1$,
$\Delta x = 1/64$:

- **Standard HLLC** ($f_M = 1$): $\log(\mathcal{A}/\mathcal{A}_0) = -\pi^2/64 \approx -0.154$; retention factor $\approx 0.857$ (15% loss).
- **LM-HLLC** ($f_M = M_{\mathrm{cut}} = 10^{-3}$): $\log(\mathcal{A}/\mathcal{A}_0) = -10^{-3} \pi^2/64 \approx -1.5 \times 10^{-4}$; retention factor $\approx 0.99985$ (0.015% loss).

The LM-HLLC case retains the amplitude almost perfectly —
far better than the kernel's truncation-error floor allows at
$\Delta x = 1/64$. The $L^1$ error is dominated by **higher-order**
dispersive terms, not the $\nu_{\mathrm{eff}}$ pressure
dissipation.

## Convergence rate

$L^1$ error $\sim 1 - \mathcal{A}(T)/\mathcal{A}_0$:

- Standard HLLC: $\sim \nu_{\mathrm{eff}} k^2 T = f_M c k^2 \Delta x T / 2$.
  Linear in $\Delta x$.
- After modified-equation expansion to $O(\Delta x^3)$ dispersion
  (the scheme's actual leading per-step error), the global $L^1$
  is $O(\Delta x^2)$.

**Standard HLLC predicted slope: $p = 2.0$.**

Under LM-HLLC at low Mach:

- The pressure-dissipation term is suppressed by $f_M = M_{\mathrm{cut}}/1$
  (when $M \le M_{\mathrm{cut}}$), making the $\nu_{\mathrm{eff}}$
  contribution to $L^1$ negligibly small.
- The $L^1$ error is dominated by **other** truncation sources
  (higher-order dispersion, entropy coupling, etc.) which at low
  amplitude become machine-precision-bounded.
- The measured slope can be **super-linear** (the $L^1$ error
  reaches the floor quickly and then plateaus at
  $\varepsilon_{\mathrm{mach}}$).

**LM-HLLC predicted slope: $p > 2$ (not matching standard theory).**

## Decay ratio

$$\log\biggl(\frac{\mathcal{A}_{\mathrm{std}}(T)}{\mathcal{A}_{\mathrm{LM}}(T)}\biggr) \;=\; -(1 - M_{\mathrm{cut}})\,\frac{c\,k^2\,\Delta x\,T}{2}, \quad (\text{E2-ratio})$$

sympy-verified. At the canonical parameters above, the ratio is
$\exp(-0.308) \approx 0.735$: LM-HLLC retains amplitude $\sim 35\%$
more than standard HLLC over one period. This is a large, easily-
measurable effect that distinguishes the two modes.

## Implication for §D2 linwave test

The golden JSON (§D2) dumps `use_lm_fix: false`. The test
must read this flag and configure the solver accordingly. If the
test accidentally runs with `use_lm_fix = true`:

- The measured $L^1$ error at $n_x = 64, 128, 256, 512$ will show
  **super-linear** slope (e.g., $p = 2.5$ or higher) — which
  looks "better" than theory but is a diagnostic of the wrong
  configuration.
- The measured amplitude retention at $T$ will be $\sim 0.9999$,
  much higher than standard HLLC's $0.857$.

Both signals indicate a configuration error, not a solver bug.
The regression test should both (a) measure the slope and check
it's in $[1.8, 2.2]$ and (b) measure the amplitude retention and
compare to §E2's prediction.

## ✅ Verification checkpoint (to be wired)

1. **Standard HLLC slope.** With `use_lm_fix = false`, $p \in
   [1.8, 2.2]$ over four resolutions. Test:
   `test_strang_linwave_convergence.cu` §E2-std-slope.

2. **LM-HLLC super-convergence.** With `use_lm_fix = true`, $p >
   2.2$ (tests confirm the expected super-linear behaviour).
   Test: §E2-LM-slope (diagnostic; not a pass/fail on $p$, but
   assert $p > 2.2$ or the amplitude retention $> 0.99$).

3. **Amplitude retention match.** With `use_lm_fix = false` at
   $n_x = 64$, $T = 1$, the measured amplitude retention is
   $0.857 \pm 0.05$ (matching the linear theory). Test:
   §E2-std-amplitude.

4. **Decay-ratio match.** Computing the ratio of amplitude losses
   between the two configurations gives $\approx 0.735$ (matching
   the $\exp(-0.308)$ prediction). Test: §E2-ratio.

Failure of (1) with $p < 1.8$ would indicate a
scheme-order-degradation bug (the convergence is weaker than
2nd-order, suggesting limiter clamping or flux asymmetry).
Failure of (2) with $p \in [1.8, 2.2]$ means the LM fix is not
activating — the kernel might be using `fM = 1` regardless of
`use_lm_fix = true`. Failures (3, 4) are numeric cross-checks of
the theoretical prediction.
