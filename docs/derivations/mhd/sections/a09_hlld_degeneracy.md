# A9. HLLD degenerate branches

> **sympy script:** `scripts/a9_hlld_degeneracy.py`
> **verified:** on the Alfvén locus $(S_K - v_{xK})^2 = B_x^2/\rho_K$,
> numerator of $B^\star_{yK}$ vanishes and denominator reduces to
> $\rho_K(S_K-v_{xK})(v_{xK} - S_M)$; $D_{S_M} = 0$ when both sides
> of the Riemann interface are identical.
> **code checkpoints:**
> `athena_mhd_kernels.cu::d_hlld_flux` (branch dispatch).

## Three degeneracies of the generic HLLD

The §A4 formulas $B^\star_{yK}, v^\star_{yK}$ blow up in three
cases. The kernel must dispatch on these **before** evaluating the
generic formulas.

| Branch | Condition | Fallback |
|---|---|---|
| **D1** | $B_x = 0$ | HLLC-MHD (Li 2005), 3-wave fan |
| **D2** | $(S_K - v_{xK})^2 = B_x^2/\rho_K$ | Upstream $B_y, v_y, B_z, v_z$ unchanged |
| **D3** | $S_L \to S_R$ and sides identical | Pure HLL (2-wave average) |

## D1 — Zero longitudinal field ($B_x = 0$)

$$B_x = 0 \Longrightarrow S^\star_L = S^\star_R = S_M,\ \mathbf{U}^\star_K = \mathbf{U}^{\star\star}_K,$$

so the two-plateau star structure collapses to one plateau. The
algorithm reduces to **HLLC-MHD** (Li 2005 Appendix A). Recommended
implementation: detect $|B_x| < \varepsilon \cdot (c_{s_0} + c_{A\perp})\sqrt{\rho}$
and dispatch.

## D2 — Alfvén locus removable singularity

The MK Eq. 44 denominator
$\rho_K(S_K-v_{xK})(S_K-S_M) - B_x^2$ vanishes on the locus
$(S_K-v_{xK})^2 = B_x^2/\rho_K$, where the upstream state already
sits on the Alfvén characteristic.

**Sympy-verified** along the locus:
- Numerator $\rho_K(S_K-v_{xK})^2 - B_x^2 \to 0$.
- Denominator reduces to $\rho_K(S_K-v_{xK})(v_{xK} - S_M)$.

So both vanish, and $B^\star_{yK}$ is $0/0$ — *removable*. The physical
content is that the transverse field does not jump across an Alfvén
wave that coincides with the upstream state.

**Kernel regularisation:**

$$\boxed{\begin{aligned}
&|\text{den}| < \epsilon\sqrt{\rho_K}\,|B_x| \\
&\quad\Longrightarrow\ B^\star_{yK} \leftarrow B_{yK},\ v^\star_{yK} \leftarrow v_{yK} \\
&\quad\text{(and same for } z\text{).}
\end{aligned}}$$

with $\epsilon \sim 10^{-12}$ in double precision. Falling through to
the upstream state is continuous with the generic formula.

## D3 — $S_L \approx S_R$ (vacuum / strongly-aligned)

When $\|(S_R - v_{xR})\rho_R - (S_L - v_{xL})\rho_L\| < \epsilon$, the
$S_M$ denominator vanishes. Physically: the Riemann fan has collapsed
(L = R), so there is no jump to resolve. Fall back to pure HLL:

$$\mathbf{F}_{\mathrm{HLL}} = \frac{S_R \mathbf{F}_L - S_L \mathbf{F}_R + S_L S_R(\mathbf{U}_R - \mathbf{U}_L)}{S_R - S_L}.$$

Pure HLL is diffusive but unconditionally stable, a safe last-resort.

## Implementation order (most to least specific)

```
if |D_SM| < ε:
    return HLL_flux()                  # D3
if |B_x| < ε * (cs0 + cA⊥)*sqrt(ρ):
    return HLLC_MHD_flux()             # D1
compute generic star states (§A4)
if |den_K| < ε * sqrt(ρ_K) * |B_x|:    # D2
    set B_y*_K, B_z*_K, v_y*_K, v_z*_K
      to upstream values
return HLLD_flux(generic)
```

## Why this matters

Without these dispatches, the Brio-Wu shock tube **will** produce NaN
at $t \sim 0.05$ when a rarefaction fan grazes the Alfvén locus.
Documented in Miyoshi-Kusano 2005 Sec. 3.4 and in Athena++
`src/hydro/rsolvers/mhd/hlld.cpp::HLLDTransport` branch logic.

## ✅ Verification checkpoints

- `tests/test_athena_mhd_brio_wu.cu` — Brio-Wu, 512 cells, match
  Stone+08 Fig 28 to $L^1 < 2\%$.
- `tests/test_athena_mhd_degenerate_Bx0.cu` — synthetic state with
  $B_x = 10^{-14}$, must not produce NaN and must match pure-HLLC
  hydrodynamic flux to $10^{-10}$.
