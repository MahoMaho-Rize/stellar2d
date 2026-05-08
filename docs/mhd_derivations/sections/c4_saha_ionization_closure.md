# C4. Saha closure for $\eta_O$ and $\eta_A$

> **sympy script:** `scripts/c4_saha_ionization_closure.py`
> **verified:** Saha low-T limit $x_e \to 0$; high-T limit $x_e \to 1$.
> **code checkpoints:** `athena_mhd_kernels.cu::d_saha_eta`,
> `tests/test_mhd_saha_table.cu`.

## Saha equation (pure hydrogen, LTE)

$$\boxed{\frac{x_e^2}{1-x_e} = \frac{1}{n_H}\!\left(\frac{m_e k_B T}{2\pi\hbar^2}\right)^{3/2}\!e^{-\chi_H/k_B T},} \quad (\text{C4-Saha})$$

with $\chi_H = 13.6$ eV and $x_e = n_e/(n_e + n_H)$. **Sympy verified**
$x_e \to 0$ as $T \to 0$ and $x_e \to 1$ as $T \to \infty$.

## Diffusivity closures (Draine 1983 / Choi+09)

**Ohmic** (electron-neutral collisions):

$$\eta_O \approx 234\,\sqrt{T/10^4\,\mathrm{K}}\,x_e^{-1}\ \mathrm{cm^2/s}. \quad (\text{C4-etaO})$$

**Ambipolar** (ion-neutral drift):

$$\eta_A = \frac{|\mathbf{B}|^2}{\rho_i\rho_n\gamma_{in}},\qquad
\gamma_{in} \approx 3.5\times 10^{13}\ \mathrm{cm^3/g/s}. \quad (\text{C4-etaA})$$

These are the forms used in Suzuki+25 (RGB winds) and Matsuoka+24
(solar chromosphere).

## Weakly-ionised limit ($x_e \ll 1$)

$$\eta_O \sim \frac{234\sqrt{T/10^4}}{x_e},\qquad
\eta_A \sim \frac{|\mathbf{B}|^2}{x_e\rho^2\gamma_{in}}.$$

**Both diverge** as $x_e \to 0$ — non-ideal effects are maximal at
the chromosphere base where most hydrogen is neutral. This is why
the RGB winds in Suzuki+25 get their 15× $\dot M$ suppression from
ambipolar: it shuts down high-frequency Alfvén wave transmission
at the chromospheric floor.

## Magnetic Reynolds numbers

$$R_m^{\mathrm{Ohm}} = \frac{L v_A}{\eta_O},\qquad
R_m^{\mathrm{amb}} = \frac{L v_A\rho}{\eta_A|\mathbf{B}|^2}.$$

**Ideal MHD requires $R_m \gg 1$.** Matsuoka+24 reports $R_m \sim 1-10$
at $r \approx 1000$ km in the solar chromosphere — non-ideal MHD
kicks in below this altitude.

## Kernel implementation

Pre-tabulate $(\eta_O, \eta_A)$ as functions of $(\rho, T, |\mathbf{B}|)$
on a 3D table to avoid evaluating the Saha equation every timestep.
Suzuki+25 uses a 128×128×64 log-spaced table; cost is negligible
once built.

## ✅ Verification

`tests/test_mhd_saha_table.cu` — verify tabulated $(\eta_O, \eta_A)$
match Draine 1983 Table 2 to $<1\%$ at five reference $(T, \rho)$
pairs: $(10^4, 10^{-9})$, $(6000, 10^{-7})$, $(4500, 10^{-8})$,
$(3000, 10^{-11})$, $(10^5, 10^{-12})$.
