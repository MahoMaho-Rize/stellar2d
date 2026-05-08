# C3. LM-HLLC blending

> **sympy script:** `scripts/c03_lm_hllc_blending.py`
> **generated LaTeX:** `output/c03_lm_hllc_blending.latex.tex`
> **verifies:** 7 strong-form identities — 1 transonic ($M = 1$)
> reduction to standard HLLC; 1 linearity identity ($\partial
> S_\star / \partial f_M$); 3 reflective-BC identities
> ($p_R - p_L = 0$, $f_M$ invariance, wall $S_\star = 0$); 1
> dispersion-ratio identity ($\sim 1/M$ suppression); 1
> Mach-cutoff clamp identity
> **code checkpoints:**
> `src/gpu/explicit/strang_device.cuh :: d_lmhllc`
> (lines 107-122: `fM = fmin(1.0, fmax(M_local, M_cutoff))`;
> line 127-128: `S_star = (fM * (PR - PL) + ...)/denom`)

Standard HLLC injects a pressure-jump term $p_R - p_L$ into the
contact-wave speed $S_\star$ (§A8). For low-Mach convective flows
this pressure jump is dominated by the **hydrostatic** component
of pressure ($\delta P = O(\rho g H)$, which is $O(\rho c^2)$), not
by the physical convective signal. The resulting numerical
dissipation scales as $\rho c^3 M$, but the physical convective
flux scales as $\rho c^3 M^3$, so the numerical dissipation
overwhelms the signal by a factor $M^{-2}$.

The LM-HLLC fix multiplies the pressure jump by a Mach-based blend
factor $f_M$, reducing the dissipation by a factor $f_M \sim M$,
which matches the physical scaling. At transonic and supersonic
speeds ($M \ge 1$), $f_M = 1$ recovers standard HLLC. At arbitrary
low Mach, a floor $M_{\mathrm{cut}} = 10^{-3}$ retains enough
dissipation for stability.

## LM-HLLC contact wave formula

$$S_{\star} \;=\; \frac{f_M\,(P_R - P_L) \;+\; \rho_L u_L (S_L - u_L) \;-\; \rho_R u_R (S_R - u_R)}{\rho_L (S_L - u_L) \;-\; \rho_R (S_R - u_R)}. \quad (\text{C3-S-star-LM})$$

The **only** modification relative to §A8's standard HLLC is
$P_R - P_L \to f_M (P_R - P_L)$ in the numerator. The denominator
is unchanged, and the Davis wave-speed estimates $S_L, S_R$
(§A9) are unchanged.

## Mach-based blend factor

$$f_M \;=\; \mathrm{clamp}(M_{\mathrm{local}},\, M_{\mathrm{cut}},\, 1), \qquad M_{\mathrm{local}} \;=\; \frac{|u_L| + |u_R|}{c_L + c_R}, \qquad M_{\mathrm{cut}} \;=\; 10^{-3}. \quad (\text{C3-fM})$$

- $M_{\mathrm{local}}$ is an average Mach at the face, symmetric
  in $L, R$ (important for well-balancing — an asymmetric choice
  would break the symmetry of the numerical flux).
- $M_{\mathrm{cut}} = 10^{-3}$ is the lower floor. It sets a
  minimum pressure dissipation so that even at vanishing Mach
  the scheme is stable (against purely internal-energy sources
  that would otherwise destabilise via negative effective
  viscosity).
- $f_M \le 1$ always, so LM-HLLC is strictly a **reduction** of
  standard HLLC dissipation, never an enhancement.

## Reduction at $M = 1$

$$f_M\big|_{M = 1} \;=\; 1 \;\Longrightarrow\; S_\star^{\mathrm{LM}} \;=\; S_\star^{\mathrm{HLLC}}. \quad (\text{C3-M1})$$

sympy verifies this as a direct substitution. At $M \ge 1$
(transonic and supersonic), LM-HLLC is exactly standard HLLC: the
blend adds zero error in the physically-important shock regime.

## Reflective-BC invariance

Under the reflective BC substitution (§B5): $\rho_L = \rho_R$,
$u_L = -u_R$, $P_L = P_R$. The pressure jump vanishes identically:

$$P_R - P_L \big|_{\mathrm{reflective}} \;=\; 0, \quad (\text{C3-reflective-pressure-jump})$$

so the $f_M$ factor multiplying it is irrelevant. LM-HLLC and
standard HLLC give identical $S_\star$ on reflective L/R pairs —
LM-HLLC preserves §B5's wall-symmetry exactly:

$$S_\star^{\mathrm{LM}}\big|_{\mathrm{reflective}} \;=\; S_\star^{\mathrm{HLLC}}\big|_{\mathrm{reflective}} \;=\; 0. \quad (\text{C3-wall-symmetry})$$

The third identity (sympy verified) uses the additional fact that
Davis wave speeds on the reflective pair satisfy $S_L = -S_R$ by
$\mathbf{u} \mapsto -\mathbf{u}$ symmetry, which makes the full
$S_\star$ numerator vanish.

## Low-Mach dispersion suppression

Scaling argument (§E3 quantifies this as effective viscosity):

| Regime | standard HLLC dissipation | LM-HLLC dissipation | ratio |
|---|---|---|---|
| $M \sim 1$ | $\sim \rho c^3$ | $\sim \rho c^3$ | 1 |
| $M \ll 1$ | $\sim \rho c^3 \cdot M$ | $\sim \rho c^3 \cdot M^2$ | $M^{-1}$ |
| $M \le M_{\mathrm{cut}}$ | $\sim \rho c^3 \cdot M$ | $\sim \rho c^3 \cdot M_{\mathrm{cut}}$ | $M / M_{\mathrm{cut}}$ |

At the atmospheric Mach $M \sim 10^{-2}$, LM-HLLC suppresses
pressure dissipation by $M / M_{\mathrm{cut}} \sim 10$ without
going to zero (which would be numerically unstable for some
operators). sympy verifies the $1/M$ ratio factor between the
two dispersions (dimensional analysis on Mach-linear dispersion
from §E3).

## Implications for acoustic convergence tests

An **acoustic wave** has velocity perturbation $\delta u = O(M c)$
and pressure perturbation $\delta P = O(M \rho c^2)$ — **both**
are of order $M$. The true physical decay rate under HLLC is
proportional to the numerical pressure dissipation $\sim c \delta P
= O(M \rho c^3)$. With LM-HLLC enabled ($f_M \to M$), the
dissipation drops to $O(M^2 \rho c^3)$: the acoustic wave is
artificially amplified (relative to standard HLLC, the expected
$\sim M^2$ decay becomes $\sim M^3$).

For **§D2's acoustic linwave test** (convergence order of HLLC on
a known acoustic mode), `use_lm_fix` must be **disabled** so the
measured convergence rate reflects the standard HLLC theory (2nd
order in $\Delta x$, with $\nu_{\mathrm{eff}} = c \Delta x / 2$).
With `use_lm_fix = true` the measured convergence would be
artificially super-linear because the LM-HLLC dissipation is
suppressed below the leading-order truncation of the spatial
reconstruction. This is what the kernel's `use_lm_fix = false`
branch (line 120-121) enables for testing:

```cpp
if (use_lm_fix) {
    // ... compute M_local and fM = clamp(M_local, M_cut, 1)
} else {
    fM = 1.0;
}
```

## Implications for convective tests

For **§D5's bubble test** (low-Mach convective flow over HSE),
`use_lm_fix = true` is physically mandatory: without it, the
convective signal is swamped by the pressure-dissipation artefact
and the bubble rises too slowly or fragments prematurely. This is
the opposite of the acoustic case above — for each test class,
the correct flag value follows from the pressure-dissipation
structure of the expected solution.

## Robustness against strong shocks

At a strong shock, $M_{\mathrm{local}} \approx (|u_L| + |u_R|) /
(c_L + c_R) \sim 1$ (since the post-shock velocity is of order
$c$), so $f_M \to 1$ and LM-HLLC recovers standard HLLC. The
modification is inactive where full HLLC dissipation is needed.
This is a design feature: LM-HLLC is a **low-Mach correction**, not
a shock-capturing modification.

## ✅ Verification checkpoint (to be wired)

1. **$f_M = 1$ regime.** On a strong-shock Sod IC (§D3), verify
   LM-HLLC produces bit-identical output to standard HLLC
   (`use_lm_fix = true` vs `false` comparison). Test:
   `test_strang_hllc.cu` §C3-shock-equiv.

2. **Reflective BC symmetry.** On a symmetric IC (§D7) with
   `use_lm_fix = true`, the solution preserves reflection
   symmetry to ULP precision (the $f_M$ factor does not break
   it). Test: `test_strang_reflection_symmetry.cu` §C3-sym.

3. **Low-Mach dispersion scaling.** On a low-Mach acoustic wave
   ($M = 10^{-2}$), measure the effective numerical viscosity
   with `use_lm_fix = false` vs `true`; the ratio should be
   close to $M / M_{\mathrm{cut}} = 10$ (§E3 quantifies). Test:
   `test_strang_linwave_convergence.cu` §C3-nu-ratio.

4. **HSE preservation with LM fix.** On pure HSE with
   `use_lm_fix = true`, the state stays at
   $O(\varepsilon_{\mathrm{mach}} N)$ drift — the LM fix does
   not destroy well-balancing. Test: `test_strang_step.cu` §C3-hse-lm.

Failure of (1) is an arithmetic bug in the LM branch. Failure of
(2) means $f_M$ is asymmetric between $L$ and $R$. Failure of
(3) would indicate an incorrect $M_{\mathrm{local}}$ formula.
Failure of (4) is rare — it would mean the $f_M$ multiplication
breaks the §B3 WB guarantee, which should be impossible because
$P_R - P_L = 0$ on pure HSE (symmetric MUSCL reconstruction) makes
the $f_M$ factor irrelevant.
