# B1. Perturbation storage bijection

> **sympy script:** `scripts/b01_perturbation_storage.py`
> **generated LaTeX:** `output/b01_perturbation_storage.latex.tex`
> **verified:**
> - 4 round-trip (forward + reverse) identity identities
> - 1 pressure-perturbation split
> - 8 zero-perturbation invariant identities (4 forward + 4 reverse)
> - 1 Jacobian-determinant positivity identity
>
> **code checkpoints:**
> - `src/gpu/explicit/strang_device.cuh :: d_cons2prim`
> - `src/gpu/explicit/strang_solver.cu :: k_strang_init_bubble`
> - every site that does `+ d_rho_bar[j_phys]` or `+ d_p_bar[j_phys]/gm1`

The Strang solver stores the **perturbation** of the four
conservative variables above the isentropic HSE background
$(\bar\rho(y), \bar p(y))$. The stored state $\mathbf{U}_{\text{store}}
= (\delta\rho, m_x, m_y, \delta E)$ lives in RAM; full conservative
state $\mathbf{U} = (\rho, m_x, m_y, E_{\mathrm{tot}})$ is
reconstructed at every arithmetic site by adding back the background.
This is a **numerical well-balancing trick**: on pure HSE the stored
state is identically zero, so round-off is $O(\varepsilon_{\mathrm{mach}})$
rather than $O(\varepsilon_{\mathrm{mach}} \cdot \bar\rho_{\mathrm{bot}})$.
The code-level cost is one add/sub per read; the pay-off is
HSE-drift bounded by §E5's condition-number estimate.

## Storage map

$$\mathbf{U}_{\text{store}} \;=\; \begin{pmatrix}\delta\rho \\ m_x \\ m_y \\ \delta E\end{pmatrix} \;=\; \begin{pmatrix}\rho - \bar\rho(y) \\ \rho u \\ \rho v \\ E_{\mathrm{tot}} - \bar p(y)/(\gamma-1)\end{pmatrix}. \quad (\text{B1-store})$$

The momentum is stored in full because the HSE background is static
($\bar u \equiv 0$, $\bar v \equiv 0$). For total energy the
background is a pure internal-energy term $\bar p / (\gamma - 1)$
(no background kinetic energy).

## Decode map (reverse)

At every device kernel that performs arithmetic on full-state
quantities, the stored state is decoded as

$$\rho = \delta\rho + \bar\rho, \qquad u = m_x / \rho, \qquad v = m_y / \rho, \qquad P = (\gamma - 1)\,\bigl[\delta E + \bar p/(\gamma - 1)\bigr] - \tfrac{1}{2}\rho(u^2 + v^2). \quad (\text{B1-decode})$$

This is a straightforward inversion. The sympy script confirms that
round-tripping $\mathbf{W} = (\rho, u, v, P) \to \mathbf{U}_{\text{store}}
\to \mathbf{W}$ is the identity in strong form.

## Pressure-perturbation identity

A direct algebraic rearrangement of the decode map gives

$$\delta P \;\equiv\; P - \bar p \;=\; (\gamma - 1)\,\bigl[\delta E - \tfrac{1}{2}\rho(u^2 + v^2)\bigr]. \quad (\text{B1-dP})$$

This identity is the **reason** the solver stores $\delta E$ and not
full $E_{\mathrm{tot}}$: on a pure acoustic wave with amplitude
$\delta P = O(\varepsilon)$, the stored $\delta E$ is also
$O(\varepsilon)$ — the background $\bar p/(\gamma - 1)$ cancels
exactly. If the solver stored $E_{\mathrm{tot}}$ directly, a linear
wave with amplitude $10^{-6}$ on top of $\bar p = 1$ would be
represented as $E_{\mathrm{tot}} \approx 2.5 + 10^{-6}$ (floating-
point subtracting two nearly-equal large numbers), losing 6 digits
of precision at every flux-assembly site.

## Zero-perturbation invariant

$$\delta\rho = m_x = m_y = \delta E = 0 \;\Longleftrightarrow\; (\rho, u, v, P) = (\bar\rho, 0, 0, \bar p). \quad (\text{B1-zero-pert})$$

This is the well-balancing statement at the storage level. The
kernel's treatment is that any all-zeros state represents pure HSE
and is preserved identically by the flux-assembly and update code
(the HSE-consistency of the face reconstruction is §B3). **Strong-
form verification.** sympy substitutes both directions and simplifies
to zero; both substitutions verify independently.

## Positive-definite Jacobian

The differential of the storage map has Jacobian

$$\frac{\partial \mathbf{U}_{\text{store}}}{\partial \mathbf{W}} \;=\; \begin{pmatrix}1 & 0 & 0 & 0 \\ u & \rho & 0 & 0 \\ v & 0 & \rho & 0 \\ \tfrac{1}{2}(u^2+v^2) & \rho u & \rho v & 1/(\gamma-1)\end{pmatrix}, \qquad \det = \frac{\rho^{2}}{\gamma-1} > 0 \quad \text{on} \;\; \rho > 0. \quad (\text{B1-jacobian})$$

The determinant is identical to $\det \partial \mathbf{U} / \partial
\mathbf{W}$ from §A2 because subtracting the background is a pure
translation — it does not change the differential. The map is
therefore a smooth diffeomorphism on the admissible domain
$\{\rho > 0, P > 0\}$; there is no branch point or coordinate
singularity away from the floor state.

## Consequence: canonical decode-before-arithmetic pattern

Every device kernel that reads the stored state performs

```cpp
// Standard pattern throughout strang_solver.cu
double rho = d_rho[k] + d_rho_bar[j_phys];
double u   = d_mx[k] / rho;
double v   = d_my[k] / rho;
double E_t = d_E[k]  + d_p_bar[j_phys] / gm1;
double P   = gm1 * (E_t - 0.5 * rho * (u*u + v*v));
```

This pattern appears in `k_muscl_hancock_x/y`, `k_hllc_update_x/y`,
`k_strang_cfl`, `k_strang_init_bubble`, and the host-side VTK/I/O
and diagnostics loops. The pattern is implicit in §B1-decode and
is the only correct way to compute a full-state quantity from
stored state. Any kernel that fails to add back $\bar\rho(y)$ or
$\bar p(y)/(\gamma - 1)$ will compute with negative or wrong values.

## Verification checkpoints

1. **Round-trip precision.** Starting from random primitive state
   $(\rho, u, v, P)$ within the HSE admissibility envelope, encode
   to $\mathbf{U}_{\text{store}}$ and decode back. Required agreement:
   $|\Delta \mathbf{W}| / |\mathbf{W}| \le 2\varepsilon_{\mathrm{mach}}$
   (2 ULP, accounting for two subtract-add pairs). Test:
   `test_strang_unit.cu` §B1-roundtrip.

2. **HSE zero-storage check.** After `init()` builds the HSE
   background and before any `init_bubble()` or IC perturbation is
   added, the stored state $\mathbf{U}_{\text{store}}$ is identically
   zero. Required: `cudaMemcpy` of the state buffer is checked
   bitwise-equal to a 4-field all-zero buffer. Test:
   `test_strang_init.cu` §B1-hse-zero.

3. **Pressure-perturbation formula.** Given a specific
   $(\rho, u, v, P)$ and its stored $\delta E$, verify that
   $P - \bar p = (\gamma - 1)\,[\delta E - \tfrac{1}{2}\rho(u^2+v^2)]$
   to ULP precision. Test: `test_strang_unit.cu` §B1-dP-formula.

Failure of (1) or (3) indicates an arithmetic bug in `d_cons2prim`
or the encode side. Failure of (2) is structural and would indicate
the HSE builder itself is inconsistent with §B2's closed-form
background (a §B2-level bug, not §B1).
