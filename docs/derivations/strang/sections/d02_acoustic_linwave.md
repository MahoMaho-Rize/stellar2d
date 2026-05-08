# D2. Acoustic linear-wave canonical IC

> **sympy script:** `scripts/d02_acoustic_linwave.py`
> **generated LaTeX:** `output/d02_acoustic_linwave.latex.tex`
> **generated goldens:** `output/d02_acoustic_linwave.goldens.json`
> **verifies:** 11 strong-form identities — 1 adiabatic relation
> ($\delta P = c_0^2 \delta\rho$); 3 linearised PDE components
> (mass, x-momentum, pressure equation); 1 $O(\epsilon)$ non-linear
> residual vanishing; 1 periodicity at $T = L_x/(u_0+c_0)$; 1
> phase-speed documentation; 4 right-eigenvector projection
> identities
> **code checkpoints:**
> `src/gpu/explicit/strang_solver.cu` — new `init_linwave()` IC
> builder; `tests/test_strang_linwave_convergence.cu` must set
> `use_lm_fix = false` (golden JSON includes the flag)

A right-going acoustic wave from §A3's $(u+c)$ eigenvector,
linearised around a stationary uniform background. This is the
canonical test for a shock-capturing scheme's **acoustic
convergence** — the scheme's pressure-diffusion truncation drives
an amplitude decay that, at a known rate, sets the convergence
slope for smooth acoustic flows.

Critically, this test requires `use_lm_fix = false`. At the
amplitude $\epsilon = 10^{-6}$ chosen below, the local Mach
$M = \epsilon$ is far below $M_{\mathrm{cut}} = 10^{-3}$, so
LM-HLLC (§C3) would reduce pressure dissipation by a factor $M /
M_{\mathrm{cut}} = 10^{-3}$, artificially super-converging the
linwave and hiding the true HLLC behaviour. §E2 quantifies this.

## IC ansatz

$$\begin{aligned}\delta\rho / \rho_0 \;&=\; \epsilon\,\sin\bigl(k(x - (u_0 + c_0) t)\bigr), \\ \delta u / c_0 \;&=\; \epsilon\,\sin\bigl(k(x - (u_0 + c_0) t)\bigr), \\ \delta v \;&=\; 0, \\ \delta P / (\gamma P_0) \;&=\; \epsilon\,\sin\bigl(k(x - (u_0 + c_0) t)\bigr),\end{aligned} \quad (\text{D2-IC})$$

with $c_0 = \sqrt{\gamma P_0 / \rho_0}$ and phase $k L_x = 2\pi$.

## Eigenvector projection

The amplitudes $(\rho_0, c_0, 0, \gamma P_0)^{\mathsf T}$ (in
primitive $(\rho, u, v, P)$ order) are **parallel** to the right
eigenvector of $\mathcal{A}_x$ at eigenvalue $u_0 + c_0$ (§A3).
The IC is a pure acoustic mode with no projection onto the
entropy ($\lambda = u_0$), shear ($\lambda = u_0$), or left-acoustic
($\lambda = u_0 - c_0$) components.

## Adiabatic relation

$$\delta P \;=\; c_0^2\,\delta\rho, \quad (\text{D2-adiabatic})$$

which is the standard acoustic wave relation: pressure and density
perturbations are in phase, with proportionality $c_0^2$. sympy
verifies this directly from the IC amplitudes.

## Linearised PDE satisfaction

The linearised Euler system around $(\rho_0, u_0, 0, P_0)$:

$$\begin{aligned}\partial_t \delta\rho + \rho_0 \partial_x (\delta u) + u_0 \partial_x (\delta\rho) \;&=\; 0, \\ \rho_0 \partial_t (\delta u) + \rho_0 u_0 \partial_x (\delta u) + \partial_x (\delta P) \;&=\; 0, \\ \partial_t (\delta P) + u_0 \partial_x (\delta P) + \gamma P_0 \partial_x (\delta u) \;&=\; 0,\end{aligned}$$

each verified by sympy against the ansatz. These are satisfied
exactly at all $\epsilon$ for the linear system.

## Non-linear residual

Substituting the ansatz into the **full non-linear** Euler PDE and
expanding in $\epsilon$, the $O(\epsilon)$ coefficient of the mass
residual vanishes (sympy verified). The $O(\epsilon^2)$ residual is
non-zero — it represents acoustic self-steepening (a small-amplitude
wave eventually steepens into a shock). For short evolution
$t \lesssim 1/(k \epsilon c_0)$ (the non-linear breaking time) and
small $\epsilon$, this residual is negligible, so the ansatz is an
excellent approximation to a true Euler solution.

At $\epsilon = 10^{-6}$, the breaking time $\sim 10^6 / (k c_0)$ is
astronomically long relative to the one-period evolution $T \sim
1/c_0$, so non-linear effects are invisible in the test.

## Periodicity

At $t = T = L_x / (u_0 + c_0)$ the phase has advanced by $k L_x =
2\pi$, so all four perturbation fields return to their initial
values. sympy verifies this directly.

## HLLC response at this IC

Unlike the §D1 entropy wave, here the face states have non-zero
pressure jump $\delta P_R - \delta P_L = O(\epsilon)$ at every
interface. The HLLC pressure-dissipation term contributes a
truncation error proportional to $\Delta x \cdot c \cdot \delta P$,
giving the leading $O(\Delta x^2)$ amplitude decay and hence
2nd-order $L^1$ convergence. §E2 derives the exact expression for
the numerical viscosity $\nu_{\mathrm{eff}}$ and the resulting
convergence rate.

## LM-HLLC interaction (critical!)

With `use_lm_fix = true`, the blend factor $f_M = \max(M_{\mathrm{local}},
M_{\mathrm{cut}}, 1)$ becomes $f_M = M_{\mathrm{cut}} = 10^{-3}$ at
$M \le M_{\mathrm{cut}}$. This multiplies the pressure-jump
dissipation by $10^{-3}$, artificially amplifying the wave
(relative to standard HLLC). The measured convergence rate would be
super-2nd-order (the kernel's error becomes machine-order for most
resolutions), which is **not** the standard HLLC convergence
theory. See §E2 for the derivation.

**For this test**, `use_lm_fix = false` is mandatory. The golden
JSON dumps `use_lm_fix: false` as a flag the test reads to
configure the solver.

## Golden values dump

| parameter | value |
|---|---|
| $\rho_0$ | 1.0 |
| $P_0$ | $1/\gamma$ |
| $u_0$ | 0.0 (stationary background) |
| $c_0$ | 1.0 |
| $\epsilon$ | $10^{-6}$ |
| $L_x$ | 1.0 |
| $\gamma$ | 1.4 |
| $k$ | $2\pi / L_x$ |
| $N_{\mathrm{ref}}$ | 4096 |
| $T$ | $1/c_0 = 1.0$ |
| `delta_rho_initial[4096]` | linear ansatz $\rho_0 \epsilon \sin(kx)$ |
| `delta_u_initial[4096]` | $c_0 \epsilon \sin(kx)$ |
| `delta_P_initial[4096]` | $\gamma P_0 \epsilon \sin(kx)$ |
| `delta_rho_final_at_T[4096]` | identical to initial |
| `L1_expected_error_at_T` | 0.0 |
| `use_lm_fix` | **false** |

## Measurement protocol

For $n_x \in \{64, 128, 256, 512\}$:

1. `init_linwave()` with the canonical parameters; `use_lm_fix = false`.
2. Evolve for $T = L_x / (u_0 + c_0) = 1.0$.
3. Download $\delta\rho(x, y, T)$.
4. Compute $L^1 = \sum_i |\delta\rho_i(T) - \delta\rho_i(0)|$.
5. Fit slope; expected $p \approx 2.0$ (2nd-order).

## ✅ Verification checkpoint (to be wired)

1. **IC consistency.** After `init_linwave()`, the face-state
   reconstruction (§A11) yields $(\rho_0 + \delta\rho, u_0 + \delta u, 0,
   P_0 + \delta P)$ with the linear ansatz profile, to ULP precision.
   Test: `test_strang_linwave_convergence.cu` §D2-IC.

2. **LM-flag flag consistency.** The test configures the kernel
   with `use_lm_fix = false` as specified in golden JSON. Test:
   `test_strang_linwave_convergence.cu` reads flag → sets solver.

3. **L1 convergence slope.** Slope over 4 resolutions in $[1.8, 2.2]$.
   Test: §E2 slope check.

4. **No growth.** The wave amplitude at $T$ must not exceed the IC
   amplitude by more than $10^{-12}$ (numerical dispersion
   tolerance). Growth would indicate an unstable scheme. Test:
   `test_strang_linwave_convergence.cu` §D2-no-growth.

Failure of (2) is a test-configuration bug — confirm the test
actually sets `use_lm_fix = false`. Failure of (3) with
`use_lm_fix = true` would show super-2nd-order convergence (the
wrong answer for this test — see §E2). Failure of (4) is a
deep-solver bug; usually it means the scheme is producing
negative dissipation on low-Mach acoustic perturbations, which is
non-physical.
