# A6. Simple-wave families (rarefactions, contacts, shocks)

> **sympy script:** `scripts/a06_smooth_wave_families.py`
> **generated LaTeX:** `output/a06_smooth_wave_families.latex.tex`
> **verifies:** 13 symbolic strong-form identities (Riemann invariants
> for all three wave families; genuine nonlinearity of 1- and
> 3-family; linear degeneracy of 2-family) + 4 numerical-fallback
> strong-form checks (mass / momentum / energy Rankine-Hugoniot
> jumps, Prandtl mass-flux equality) at 80 random admissible shock
> states
> **code checkpoints:**
> §A8 (HLLC intermediate states use the 1-shock / 3-shock Hugoniot
> relations derived here)
> §D3 (Sod shock-tube reference profile uses the rarefaction
> integration curves $J_k^{(i)} = \text{const}$)

The Godunov finite-volume kernel does not resolve individual waves;
it resolves Riemann problems at every cell face. But every
non-trivial piece of the Riemann solver — the HLLC contact speed
(§A8), the Davis wave-speed bounds (§A9), the Sod analytic reference
(§D3) — is built from the three wave families of this section. This
section derives the characteristic structure in strong form, using
both the genuine-nonlinearity framework (Lax 1957) and the explicit
Rankine-Hugoniot locus (Toro 2009 §4.2).

The tangential velocity $v$ plays no algebraic role in the x-sweep
apart from the shear wave (§A6.3 below). We therefore present the
derivation in the 1D projection $(\rho, u, p)$; the shear wave is
handled separately.

## Right eigenvectors in primitive form

Using the primitive state $\mathbf{W} = (\rho, u, v, p)^\top$ and
the sound speed $c = \sqrt{\gamma p / \rho}$,

$$R_1^{(\mathrm{prim})} = \begin{pmatrix}1\\ -c/\rho\\ 0\\ c^{2}\end{pmatrix},\qquad
R_{2a}^{(\mathrm{prim})} = \begin{pmatrix}1\\ 0\\ 0\\ 0\end{pmatrix},\qquad
R_{2b}^{(\mathrm{prim})} = \begin{pmatrix}0\\ 0\\ 1\\ 0\end{pmatrix},\qquad
R_3^{(\mathrm{prim})} = \begin{pmatrix}1\\ +c/\rho\\ 0\\ c^{2}\end{pmatrix}. \quad (\text{A6-R-prim})$$

These are the primitive-space counterparts of the conservative
eigenvectors of §A3; the mapping between them is $R^{(\mathrm{cons})}_k
= \partial \mathbf{U}/\partial \mathbf{W} \cdot R_k^{(\mathrm{prim})}$.
The two-fold degeneracy of the $\lambda = u$ eigenspace is split
into an **entropy** direction $R_{2a}$ (density contrast at
constant $u, v, p$) and a **shear** direction $R_{2b}$ (tangential-
velocity contrast at constant $\rho, u, p$).

## Riemann invariants

**1-family ($\lambda = u - c$, acoustic left):**

$$J_1^{(1)} \;=\; u + \frac{2c}{\gamma - 1}, \qquad J_1^{(2)} \;=\; \frac{p}{\rho^{\gamma}}. \quad (\text{A6-RI-1})$$

**3-family ($\lambda = u + c$, acoustic right):**

$$J_3^{(1)} \;=\; u - \frac{2c}{\gamma - 1}, \qquad J_3^{(2)} \;=\; \frac{p}{\rho^{\gamma}}. \quad (\text{A6-RI-3})$$

**2-family ($\lambda = u$, entropy + shear):**

- Entropy wave: $u$ and $p$ constant through; $\rho$ may jump.
- Shear wave: $\rho$, $u$, $p$ all constant; only tangential $v$
  jumps.

**Strong-form verification.** For each invariant $J_k^{(i)}$ and
eigenvector $R_k$,

$$\nabla_{\mathbf{W}} J_k^{(i)} \;\cdot\; R_k \;\overset{\text{sp.simplify}}{\longrightarrow}\; 0.$$

9 scalar identities, all verified.

## Genuine nonlinearity and linear degeneracy

The Lax classification requires computing
$\nabla_{\mathbf{W}} \lambda_k \cdot R_k$ for each family:

$$\nabla_{\mathbf{W}} \lambda_1 \cdot R_1 \;=\; -\frac{(\gamma + 1)\,c}{2\rho} \;\neq\; 0 \qquad \text{(1-family, genuinely nonlinear)}, \quad (\text{A6-gn-1})$$

$$\nabla_{\mathbf{W}} \lambda_3 \cdot R_3 \;=\; +\frac{(\gamma + 1)\,c}{2\rho} \;\neq\; 0 \qquad \text{(3-family, genuinely nonlinear)}, \quad (\text{A6-gn-3})$$

$$\nabla_{\mathbf{W}} \lambda_2 \cdot R_{2a} \;=\; 0, \quad \nabla_{\mathbf{W}} \lambda_2 \cdot R_{2b} \;=\; 0 \qquad \text{(2-family, linearly degenerate)}. \quad (\text{A6-ld})$$

**Consequence for the Riemann problem.** The acoustic 1- and
3-families admit genuine non-linear solutions: either rarefactions
(smooth, self-similar fans) or shocks (jump discontinuities). The
entropy/shear 2-family admits only contact discontinuities — a jump
where the velocity and pressure are continuous. This is the
structural fact §A7 and §A8 exploit when constructing the HLLC
flux.

## Rarefaction integration curves

Through a genuinely-nonlinear rarefaction fan, the invariants
$J_k^{(1)}, J_k^{(2)}$ are simultaneously constant. Writing the
fan parametrised by $\xi = (x - x_0)/t$ in the 1-family,

$$J_1^{(1)} = u + \frac{2c}{\gamma - 1} = \text{const},\qquad J_1^{(2)} = \frac{p}{\rho^{\gamma}} = \text{const}, \qquad \xi = u - c.$$

Combining the three relations gives explicit expressions for
$\rho(\xi), u(\xi), p(\xi)$ through the fan; these are the
formulas §D3 (Sod) uses to sample the analytic rarefaction profile.

## Rankine-Hugoniot jump conditions

For a 1D shock of speed $\sigma$ separating pre-shock state
$(\rho_L, u_L, p_L)$ from post-shock state $(\rho_R, u_R, p_R)$,

$$\sigma\,[\rho] \;=\; [\rho u], \qquad
\sigma\,[\rho u] \;=\; [\rho u^{2} + p], \qquad
\sigma\,[E] \;=\; [(E + p)\,u], \quad (\text{A6-RH})$$

with $[f] = f_R - f_L$ and $E = p/(\gamma-1) + \tfrac{1}{2}\rho u^2$.

**Hugoniot locus for a 1-shock (Toro 2009 §4.2.1).** Given
$p^\star > p_L$ (compressive shock), the post-shock state is

$$\rho^{\star}_L \;=\; \rho_L\,\frac{p^{\star}/p_L + (\gamma - 1)/(\gamma + 1)}{(\gamma - 1)/(\gamma + 1)\,p^{\star}/p_L + 1}, \quad (\text{A6-hug-rho})$$

$$u^{\star} \;=\; u_L - (p^{\star} - p_L)\sqrt{\frac{A}{p^{\star} + B}}, \qquad A = \frac{2}{(\gamma + 1)\rho_L},\quad B = \frac{\gamma - 1}{\gamma + 1}\,p_L, \quad (\text{A6-hug-u})$$

$$\sigma \;=\; u_L - c_L\sqrt{\frac{\gamma + 1}{2\gamma}\,\frac{p^{\star}}{p_L} + \frac{\gamma - 1}{2\gamma}}. \quad (\text{A6-hug-sigma})$$

**Strong-form verification with numerical fallback.** Substituting
$\rho^\star_L, u^\star, \sigma$ back into the three RH jump
equations yields expressions with nested square roots that exceed
`sp.simplify`'s sqrt-denest capability.

> _Per Rule 1, when sp.simplify cannot reach 0 on a strong-form
> pointwise identity that is physically correct, the script falls
> back to numerical random sampling at $N = 80$ admissible shock
> states with $\rho_L \in [0.1, 10]$, $|u_L| \in [0, 3]$,
> $p_L \in [0.1, 10]$, $p^\star/p_L \in [1.05, 20]$ (compressive),
> $\gamma \in \{1.4, 5/3, 2\}$. Tolerance $10^{-9}$. All four
> identities (mass jump, momentum jump, energy jump, Prandtl
> mass-flux equality $\rho_L(u_L - \sigma) = \rho^\star(u^\star -
> \sigma)$) pass. Note this is **not a weak-form step**: the
> identities are strong-form pointwise; the numerical fallback is
> a sympy-capability workaround, not a distributional relaxation._

Achieved residual sizes (maxima across the 80-sample ensemble):

| Identity | Max $|\text{residual}|$ | Tolerance |
|---|---|---|
| Mass jump | $3 \times 10^{-14}$ | $10^{-9}$ |
| Momentum jump | $2 \times 10^{-13}$ | $10^{-9}$ |
| Energy jump | $1 \times 10^{-12}$ | $10^{-9}$ |
| Prandtl flux | $4 \times 10^{-14}$ | $10^{-9}$ |

All four are several orders of magnitude below the tolerance.

## Contact discontinuity

Across a 2-family discontinuity (contact wave), the strong-form
conditions are

$$\sigma \;=\; u_L \;=\; u_R \;\equiv\; u^{\star}, \qquad p_L \;=\; p_R \;\equiv\; p^{\star}, \quad (\text{A6-contact})$$

while $\rho_L$ and $\rho_R$ are unrelated (density contrast
permitted) and $v_L$ and $v_R$ are unrelated (shear permitted).
This is the identity §A8 uses to define the HLLC intermediate
states: the two star-region cells share $u^\star$ and $p^\star$
but carry independent densities and tangential velocities.

## ✅ Verification checkpoint (to be wired)

The kernel does not encode §A6 explicitly; the HLLC solver (§A8)
uses the star-region fact $u^\star_L = u^\star_R$ and
$p^\star_L = p^\star_R$ as its defining algebra. §A6's results are
verified at the solver level through §D3 and §D4:

1. **Sod shock-tube reference.** §D3 dumps the analytic
   $\{\rho, u, p\}(x, t = 0.2)$ profile computed from the 1-family
   rarefaction fan + 2-family contact + 3-family shock Rankine-
   Hugoniot locus. Comparison to the simulation output gives the
   convergence slope expected in §E1.

2. **Tangential-velocity invariance.** Across the contact, §A6
   permits shear but not pressure / normal-velocity jumps. Test
   `test_strang_hllc.cu` §A6-contact block: initialise a stationary
   contact with $\rho_L \ne \rho_R$, $v_L \ne v_R$, $u_L = u_R$,
   $p_L = p_R$; verify that the evolved $u$ and $p$ fields stay
   constant to ULP, while $\rho$ and $v$ are advected through the
   moving contact at speed $u$.

Failure on (1) indicates the Riemann solver's star-region algebra
is wrong (§A8 bug). Failure on (2) indicates the HLLC treatment of
the tangential velocity is incorrect — specifically, that the
shear wave is being spuriously excited by acoustic-family mixing,
which diagnoses a bug in the §A3 eigenvector basis or in the
slope-limiter interaction with the shear component (§A10).
