# D3. MRI stress decomposition $\alpha_{\mathrm{SS}} = \alpha_R + \alpha_M$

> **sympy script:** `scripts/d3_mri_stress_decomposition.py`
> **verified:** Maxwell stress antisymmetric under $B_R\to-B_R$,
> invariant under $(B_R,B_\phi)\to(-B_R,-B_\phi)$; Reynolds stress
> antisymmetric under $v_R\to-v_R$; MRI dispersion vanishes at
> Balbus-Hawley extremum ($k_*^2 v_A^2 = 15/16\,\Omega^2$,
> $\gamma_{\max} = 3/4\,\Omega$).
> **code checkpoints:** `tests/test_mhd_mri_growth_rate.cu`,
> `tests/test_mhd_mri_stress_decomp.cu`.

## Shakura-Sunyaev stress

$$T_{R\phi} = \underbrace{\rho v_R\,\delta v_\phi}_{\text{Reynolds}} + \underbrace{(-B_R B_\phi)}_{\text{Maxwell}}. \quad (\text{D3-stress})$$

$$\alpha_{\mathrm{SS}} = \frac{\langle T_{R\phi}\rangle}{\langle p\rangle} = \alpha_R + \alpha_M.$$

## Suzuki 2023 sign-quadrant decomposition

Maxwell stress is:
- **antisymmetric** under single flip $B_R \to -B_R$: $M_{R\phi}\to -M_{R\phi}$.
- **invariant** under simultaneous flip $(B_R, B_\phi) \to (-B_R, -B_\phi)$.

This pair of symmetries is what makes the Suzuki 2305.12112 "triangle
diagnostic" meaningful: the four sign quadrants $(\pm B_R, \pm B_\phi)$
carry physically distinct stress contributions. The cylindrical vs
Cartesian sign-flip result in that paper (headline $[\phi \Rightarrow_R \phi]$
arrow $+6.42$ vs $-1.05$) is a direct consequence of this symmetry
structure.

## Balbus-Hawley MRI dispersion

For vertical $\mathbf{B}_0 = B_0\hat{z}$ and Keplerian $q = 3/2$
($\kappa^2 = \Omega^2$):

$$\omega^4 - \omega^2(\kappa^2 + 2k^2v_A^2) + k^2v_A^2(k^2v_A^2 + \kappa^2 - 4\Omega^2) = 0. \quad (\text{D3-BH-disp})$$

Instability requires $k^2 v_A^2 < 3\Omega^2$. The most-unstable mode:

$$\boxed{k_*^2 v_A^2 = \tfrac{15}{16}\Omega^2,\qquad \gamma_{\max} = \tfrac{3}{4}\Omega.} \quad (\text{D3-MRI-max})$$

**Sympy verified** by direct substitution into the dispersion
polynomial. This gives the MRI timescale $\sim 4/(3\Omega) \approx 1/3$
of an orbital period.

## Parseval for Maxwell

$$\frac{1}{V}\int B_R(\mathbf{x}) B_\phi(\mathbf{x})\,d^3x = \sum_{\mathbf{k}} \mathrm{Re}[\hat{B}_R(\mathbf{k})\hat{B}_\phi^*(\mathbf{k})]$$

Allows FFT-based computation of $\alpha_M$ at arbitrary resolution.

## ✅ Verification

**`tests/test_mhd_mri_growth_rate.cu`** — seed a linear $B_y$
perturbation at $k_*$ with amplitude $10^{-6}B_0$; lock measured
growth rate matches $(3/4)\Omega$ to $<1\%$ over $5/\Omega$.

**`tests/test_mhd_mri_stress_decomp.cu`** — in nonlinear saturated
state, lock $\alpha_M/\alpha_R \sim 3-5$ (Stone+96, Suzuki+23
Cartesian baseline). Suzuki+23 reports $\alpha_M \approx 0.072$,
$\alpha_R \approx 0.016$ for Cartesian Keplerian baseline.
