# E3. LM-HLLC effective numerical viscosity

> **sympy script:** `scripts/e03_lm_hllc_nu_eff.py`
> **generated LaTeX:** `output/e03_lm_hllc_nu_eff.latex.tex`
> **verified:**
> - 1 $\nu_{\mathrm{eff}}$ ratio ($\nu_{\mathrm{LM}}/\nu_{\mathrm{std}} = M$)
> - 1 $\mathrm{Re}_{\mathrm{eff}}$ under LM ($= 2 N M_{\mathrm{conv}}/M_{\mathrm{loc}}$)
> - 1 under standard HLLC ($= 2 N M_{\mathrm{conv}}$)
> - 1 Re-ratio ($\mathrm{Re}_{\mathrm{LM}}/\mathrm{Re}_{\mathrm{std}} = 1/M_{\mathrm{loc}}$)
> - 1 clamped-regime identity
>
> **code checkpoints:**
> - `src/gpu/explicit/strang_device.cuh :: d_lmhllc` ($f_M$ clamp logic, lines 115-122); measurement via a dedicated scheme- characterisation test at Andrassy-style low-Mach convection parameters

Quantifies the practical advantage of LM-HLLC over standard HLLC
for low-Mach convective flows — specifically, the **effective
Reynolds number** the scheme supports at a fixed grid resolution.
Standard HLLC's pressure-jump dissipation dominates at low Mach,
restricting the effective viscosity to $\sim c \Delta x$; LM-HLLC
reduces this to $\sim M_{\mathrm{loc}} c \Delta x$, improving
$\mathrm{Re}_{\mathrm{eff}}$ by a factor $1/M_{\mathrm{loc}}$.

## Effective numerical viscosity

From §E2's dispersion analysis:

$$\nu_{\mathrm{eff}}^{\mathrm{LM}} \;=\; M_{\mathrm{loc}}\,\frac{c\,\Delta x}{2}, \qquad \nu_{\mathrm{eff}}^{\mathrm{std}} \;=\; \frac{c\,\Delta x}{2}. \quad (\text{E3-nu-eff})$$

The ratio $\nu_{\mathrm{LM}} / \nu_{\mathrm{std}} = M_{\mathrm{loc}}$
is the fundamental LM-HLLC advantage: the pressure-dissipation
strength is reduced to the physical Mach level, matching what
the continuum flow actually contains.

## Effective Reynolds number

For a convective flow at speed $u_{\mathrm{conv}}$ across domain
$L$:

$$\mathrm{Re}_{\mathrm{eff}} \;=\; \frac{L\,u_{\mathrm{conv}}}{\nu_{\mathrm{eff}}}, \qquad N \;=\; L / \Delta x, \;\; M_{\mathrm{conv}} \;=\; u_{\mathrm{conv}} / c.$$

**Standard HLLC:**
$$\mathrm{Re}_{\mathrm{eff}}^{\mathrm{std}} \;=\; 2\,N\,M_{\mathrm{conv}}.$$

**LM-HLLC (active regime, $M > M_{\mathrm{cut}}$):**
$$\mathrm{Re}_{\mathrm{eff}}^{\mathrm{LM}} \;=\; 2\,N\,\frac{M_{\mathrm{conv}}}{M_{\mathrm{loc}}}. \quad (\text{E3-Re-eff})$$

The LM advantage:

$$\frac{\mathrm{Re}_{\mathrm{eff}}^{\mathrm{LM}}}{\mathrm{Re}_{\mathrm{eff}}^{\mathrm{std}}} \;=\; \frac{1}{M_{\mathrm{loc}}}. \quad (\text{E3-advantage})$$

At the canonical Andrassy stratified-convection ambient
$M_{\mathrm{loc}} \sim 10^{-3}$, this gives a $1000\times$
Reynolds-number boost — the LM fix is the **reason** one can
simulate meaningful turbulence in the low-Mach regime at
affordable resolutions.

## Clamped regime ($M < M_{\mathrm{cut}}$)

When the local Mach drops below $M_{\mathrm{cut}} = 10^{-3}$, the
blend factor $f_M$ clamps at $M_{\mathrm{cut}}$:

$$\nu_{\mathrm{eff}}^{\mathrm{LM,\;clamp}} \;=\; M_{\mathrm{cut}}\,\frac{c\,\Delta x}{2}, \qquad \mathrm{Re}_{\mathrm{eff}}^{\mathrm{LM,\;clamp}} \;=\; 2\,N\,\frac{M_{\mathrm{conv}}}{M_{\mathrm{cut}}}. \quad (\text{E3-clamp})$$

At $M_{\mathrm{loc}} = M_{\mathrm{cut}} = 10^{-3}$, the clamp is
not yet active and the regime is a smooth transition. Below
$M_{\mathrm{cut}}$, the LM reduction saturates — Reynolds number
grows no further as $M_{\mathrm{loc}}$ decreases. This is by
design: $M_{\mathrm{cut}}$ provides a stability floor to prevent
the scheme from going negative-viscosity at vanishing Mach.

## Numerical example

At $n_x = 512$, $L = 1$, $c = 1$, $u_{\mathrm{conv}} = M_{\mathrm{conv}} c$
with $M_{\mathrm{conv}} = M_{\mathrm{loc}} = 10^{-2}$ (typical for
stellar convection at the top of the convection zone):

| scheme | $\nu_{\mathrm{eff}}$ | $\mathrm{Re}_{\mathrm{eff}}$ | improvement |
|---|---|---|---|
| standard HLLC | $5 \times 10^{-4}$ | 10.24 | 1 |
| LM-HLLC | $5 \times 10^{-6}$ | 1024.00 | $100\times$ |

The LM-HLLC scheme supports $100\times$ higher effective Reynolds
at the same grid. For turbulent flows where Reynolds drives
self-similar cascades, this is a dramatic expansion of the
accessible regime.

## Application to Strang kernel

The stellar2d Strang kernel uses LM-HLLC by default (`use_lm_fix
= true` at all production sites; `use_lm_fix = false` is a test
flag only). The kernel's intended use case is stratified
atmospheres with convective motions at $M \sim 10^{-2}$ to
$10^{-3}$; LM-HLLC is the appropriate choice for these
applications.

For tests that probe the **scheme's convergence theory**
(§D2 linwave), `use_lm_fix = false` is used to recover the
standard HLLC behaviour; this is a configuration choice, not a
kernel defect.

## Verification checkpoints

1. **$\nu_{\mathrm{eff}}$ measurement under standard HLLC.**
   On a smoothly-driven low-Mach flow ($u_{\mathrm{conv}} = 10^{-2} c$)
   at $n_x = 512$ with `use_lm_fix = false`, measure the decay
   rate of a tagged perturbation; extract $\nu_{\mathrm{eff}}$
   from the decay coefficient. Expected $\approx c \Delta x / 2
   = 5 \times 10^{-4}$ (well above the physical viscosity $\nu
   = 0$). Test: scheme-char probe.

2. **$\nu_{\mathrm{eff}}$ measurement under LM-HLLC.** Same
   setup with `use_lm_fix = true`; expected $\nu_{\mathrm{eff}}
   \approx M_{\mathrm{loc}} c \Delta x / 2 = 5 \times 10^{-6}$
   — $100\times$ lower. Test: scheme-char probe.

3. **Ratio check.** The measured ratio $\nu_{\mathrm{std}}/\nu_{\mathrm{LM}}
   \approx 100$ matches $1/M_{\mathrm{loc}} = 100$. Test:
   scheme-char probe.

4. **Reynolds crossover at $M_{\mathrm{cut}}$.** Running at
   progressively lower $M$ and measuring $\nu_{\mathrm{eff}}$,
   observe the clamp activate around $M_{\mathrm{loc}} = 10^{-3}$.
   Below, $\nu_{\mathrm{eff}}$ stays constant at the clamped
   value. Test: scheme-char probe sweep over $M_{\mathrm{loc}}$.

Failure of (1) or (2) far from the predicted $\nu_{\mathrm{eff}}$
indicates a dispersion-analysis error; revisit §E2. Failure of
(3) is a direct diagnostic of LM branch breakage.
