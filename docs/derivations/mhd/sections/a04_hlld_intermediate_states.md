# A4. HLLD intermediate states (Miyoshi-Kusano 2005)

> **sympy script:** `scripts/a4_hlld_intermediate_states.py`
> **verified:** $p^{\star}_{\text{tot,L}} = p^{\star}_{\text{tot,R}}$
> symbolically; Rankine–Hugoniot outer-wave jumps numerically
> (20 trials, max err $3\times10^{-15}$).
> **code checkpoints:** `athena_mhd_kernels.cu::d_hlld_flux`.

## Wave fan

HLLD partitions the Riemann fan between $S_L$ and $S_R$ into four
intermediate plateaus separated by $S^\star_L, S_M, S^\star_R$ —
outer fast, Alfvén, contact, Alfvén, outer fast.

## Contact speed (MK Eq.\ 38)

$$\boxed{S_M = \frac{(S_R - v_{xR})\rho_R v_{xR} - (S_L - v_{xL})\rho_L v_{xL}
 - p^{\star}_{\text{tot,R}} + p^{\star}_{\text{tot,L}}}
 {(S_R - v_{xR})\rho_R - (S_L - v_{xL})\rho_L}.}$$

## Star-region state

$\rho^{\star}_K = \rho_K(S_K - v_{xK})/(S_K - S_M)$ (MK Eq.\ 43);
$p^{\star}_{\text{tot}} = p^{\star}_{\text{tot,L}} + \rho_L(S_L - v_{xL})(S_M - v_{xL})
= p^{\star}_{\text{tot,R}} + \rho_R(S_R - v_{xR})(S_M - v_{xR})$
(MK Eq.\ 41). **Sympy symbolically verifies** the L/R equality.

$B^{\star}_{yK}, v^{\star}_{yK}$ from MK Eqs.\ 44-47 (see script).
Alfvén speeds $S^{\star}_{L,R} = S_M \mp |B_x|/\sqrt{\rho^\star_{L,R}}$.

## Critical implementation note

**HLLD star state is NOT an EOS state.** Do NOT compute $\mathbf{F}^\star_K$
via $p = (\gamma-1)(E^\star - \tfrac{1}{2}\rho|\mathbf{v}|^2 - \tfrac{1}{2}|\mathbf{B}|^2)$
— it gives the wrong pressure. Use MK Eq.\ 64 directly with
$p^\star_{\text{tot}}$ from (A4-Ptot-star). Numerical verification
confirms: RH residuals are $\mathcal{O}(1)$ with EOS inversion,
$\mathcal{O}(10^{-15})$ with direct MK formula.

## ✅ Verification

`tests/test_athena_mhd_hlld.cu` — Brio-Wu shock tube match Stone+08
Fig 28 to $L^1 < 2\%$ at $N=512$.
