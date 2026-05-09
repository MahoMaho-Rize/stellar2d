# A8. HLLC intermediate states

> **sympy script:** `scripts/a08_hllc_intermediate_states.py`
> **generated LaTeX:** `output/a08_hllc_intermediate_states.latex.tex`
> **verified:**
> - 17 strong-form pointwise identities — $p^\star_L = p^\star_R$
> - mass / momentum-x / momentum-y / energy Rankine- Hugoniot across $S_L$ and $S_R$ (8 identities)
> - HLLC flux in the left-star and right-star branches reduces to $F(\mathbf{U}^\star_L)$ and $F(\mathbf{U}^\star_R)$ (8 components)
>
> **code checkpoints:**
> - `src/gpu/explicit/strang_device.cuh :: d_lmhllc` (the $S_\star$ formula and $U^\star_L, U^\star_R$ constructions in the kernel are LM-scaled via $f_M$; the LM factor is derived separately in §C3)

The HLLC Riemann solver resolves three waves: the left and right
acoustics at speeds $S_L, S_R$ bounding the fan, and an interior
contact at speed $S_\star$. Between the two acoustic waves, the
flow is divided into two **star regions** with shared normal
velocity $u^\star = S_\star$ and shared pressure $p^\star$, but
distinct densities $\rho^\star_L, \rho^\star_R$ and tangential
velocities $v^\star_L = v_L, v^\star_R = v_R$ (linearly-degenerate
contact carries only density and tangential-velocity contrast,
§A6).

This section derives all six star-region quantities — $S_\star,
p^\star, \rho^\star_L, \rho^\star_R, \mathbf{U}^\star_L,
\mathbf{U}^\star_R$ — in strong form, then verifies the Rankine-
Hugoniot jump conditions on every wave (16 scalar identities,
one per conservative component per wave side).

## Contact speed $S_\star$

From the Rankine-Hugoniot mass-jump equations across $S_L$ and
$S_R$, plus the pressure-balance condition $p^\star_L = p^\star_R$
(which is imposed as the **definition** of the contact wave — §A6,
linear degeneracy), algebraic elimination yields Toro's formula
(2009 eq. 10.37):

$$S_{\star} \;=\; \frac{p_R - p_L + \rho_L u_L (S_L - u_L) - \rho_R u_R (S_R - u_R)}{\rho_L (S_L - u_L) - \rho_R (S_R - u_R)}. \quad (\text{A8-Sstar})$$

Denominator note: $\rho_L (S_L - u_L) < 0$ and $\rho_R (S_R - u_R)
> 0$ on admissible states (since $S_L < u_L$ and $S_R > u_R$);
hence the denominator is strictly negative, never vanishing on an
admissible state.

## Star pressure $p^\star$

Two equivalent expressions (Toro 2009 eq. 10.38), one from each
side of the contact:

$$p^{\star} \;=\; p_L + \rho_L (u_L - S_L)(u_L - S_{\star}) \;=\; p_R + \rho_R (u_R - S_R)(u_R - S_{\star}). \quad (\text{A8-pstar})$$

**Strong-form consistency.** sympy verifies that substituting the
$S_\star$ formula into both expressions yields the same value:

$$p^\star_L - p^\star_R \;\overset{\text{sp.simplify}}{\longrightarrow}\; 0.$$

## Star densities

From the mass Rankine-Hugoniot across $S_L$ and $S_R$ (Toro 2009
eq. 10.36):

$$\rho^{\star}_L \;=\; \rho_L\,\frac{S_L - u_L}{S_L - S_{\star}}, \qquad \rho^{\star}_R \;=\; \rho_R\,\frac{S_R - u_R}{S_R - S_{\star}}. \quad (\text{A8-rhostar})$$

**Strong-form verification.** sympy substitutes $\rho^\star_L$ and
$\rho^\star_R$ into the mass jump condition $S_K[\rho] = [\rho u_n]$
across each wave and `sp.simplify`-reduces to $0$.

## Star momenta

Normal momentum in each star state is $\rho^\star$ times the
common normal velocity $u^\star = S_\star$:

$$(\rho u)^{\star}_K \;=\; \rho^{\star}_K\,S_{\star}. \quad (\text{A8-momstar-n})$$

Tangential momentum is unchanged across an acoustic wave
(linear-degeneracy of the tangential component along 2-family):

$$(\rho v)^{\star}_K \;=\; \rho^{\star}_K\,v_K \qquad K \in \{L, R\}. \quad (\text{A8-momstar-t})$$

**Strong-form verification.** Both momentum jump conditions verified
on each acoustic wave (4 identities × 2 waves = 8 identities). The
tangential-momentum identities use $v^\star_K = v_K$ as the
unchanged-tangential-velocity condition.

## Star energies

From the total-energy Rankine-Hugoniot (Toro 2009 eq. 10.39), after
substituting the already-derived star densities and momenta:

$$E^{\star}_K \;=\; \rho^{\star}_K\,\left[\,\frac{E_K}{\rho_K} \;+\; (S_{\star} - u_K)\left(S_{\star} + \frac{p_K}{\rho_K\,(S_K - u_K)}\right)\right] \qquad K \in \{L, R\}. \quad (\text{A8-Estar})$$

**Strong-form verification.** sympy verifies $S_K[E] = [(E+p) u_n]$
on each acoustic wave (2 identities, one per side).

## HLLC numerical flux

The full piecewise definition, with branches selected by the sign
of the local wave speeds:

$$F^{\mathrm{HLLC}} \;=\; \begin{cases}
\mathbf{F}_L & 0 \le S_L, \\[2pt]
\mathbf{F}_L + S_L (\mathbf{U}^{\star}_L - \mathbf{U}_L) & S_L \le 0 \le S_{\star}, \\[2pt]
\mathbf{F}_R + S_R (\mathbf{U}^{\star}_R - \mathbf{U}_R) & S_{\star} \le 0 \le S_R, \\[2pt]
\mathbf{F}_R & S_R \le 0.
\end{cases} \quad (\text{A8-HLLC})$$

**Strong-form consistency.** In the left-star branch, the flux
simplifies to

$$\mathbf{F}_L + S_L (\mathbf{U}^{\star}_L - \mathbf{U}_L) \;=\; \mathbf{F}(\mathbf{U}^{\star}_L) \;=\; \begin{pmatrix}\rho^{\star}_L S_{\star}\\ \rho^{\star}_L S_{\star}^{2} + p^{\star}\\ \rho^{\star}_L S_{\star} v_L\\ (E^{\star}_L + p^{\star}) S_{\star}\end{pmatrix}.$$

sympy verifies all 4 components. Analogous result for the right-
star branch. This means the HLLC flux is equivalently the **Euler
flux evaluated at the star state** in each subsonic branch — a
structurally stronger result than the general HLL template, and
the reason HLLC exactly resolves isolated contacts (§A7).

## Verification checkpoints

The §A8 identities are implemented inside `d_lmhllc`. The
regression tests should check:

1. **S_\star computation.** For random admissible L/R states with
   Davis-bounded $S_L, S_R$ (§A9), the kernel's $S_\star$ must
   match the analytic formula to ULP precision. Test:
   `test_strang_hllc.cu` §A8-Sstar block.

2. **Pressure consistency.** Verify that $p^\star$ computed from
   the L-side formula and from the R-side formula agree to ULP.
   Any disagreement larger than $10\varepsilon_{\mathrm{mach}}$
   indicates a bug in either $S_\star$ or in the star-pressure
   algebra. Test: `test_strang_hllc.cu` §A8-pstar block.

3. **Star-state reconstruction.** Given L, R, $S_L, S_R, S_\star,
   p^\star$, verify that the kernel's reconstructed $\mathbf{U}^\star_L$
   satisfies $\rho^\star_L \cdot S_\star$ = normal-momentum component
   and $\rho^\star_L \cdot v_L$ = tangential-momentum component to
   ULP. Test: `test_strang_hllc.cu` §A8-U-star block.

4. **Contact-wave resolution.** With $u_L = u_R = 0$, $p_L = p_R$,
   $\rho_L = 1$, $\rho_R = 5$, `d_lmhllc` must return
   $(F_\rho, F_{\rho u}, F_{\rho v}, F_E) = (0, p_L, 0, 0)$ to ULP.
   Strengthens the A7 scorecard test: if the kernel returns
   non-zero mass flux, the bug is localised to the $S_\star$
   formula or to the left-star branch selection. Test:
   `test_strang_hllc.cu` §A8-contact block.

Failures of (1) or (2) indicate algebraic bugs in the $S_\star /
p^\star$ computation — the most common form is a sign error on
the denominator (see Toro §10.5.1 for the canonical form).
Failures of (3) would affect the energy balance and show up as
energy-conservation drift in long-time simulations. Failure of
(4) is the smoking-gun regression for HLLC's contact resolution
and must be fixed before any convection or stratified benchmark
can be run.
