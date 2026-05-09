# E1-T7 analytic-on-mesh error decomposition

> **script:** `scripts/e1_t7_solver_vs_analytic.py`
> **data:** `build/t7_timeseries.csv` (dumped by
> `tests/test_athena_mhd_driver.cu::test_T7_wkb_amplitude_growth`).
> **purpose:** isolate the true solver discretisation error from RMS
> sampling-window bias by evaluating the Hankel analytic solution
> on the SAME `(y_k, t_i)` mesh the solver sampled.

## Motivation

Before this decomposition the E1-T7 threshold was attributed to PLM
dissipation via the F5 O((kΔy)⁴) amplitude-retention formula.  A
convergence scan however showed the solver error **saturates around
−8%** in Ny ∈ [192, 384]; a PLM-limited error would decrease as
(Δy)² for amplitude retention (or (Δy)³ for retention × N_steps).
The saturation rules out PLM as the dominant mechanism.

To separate concerns we ran the Hankel analytic solution on the
identical sample mesh the solver uses and computed the same RMS
statistic.

## Analytic solution

Linear Alfvén wave in an isothermal stratified atmosphere,
bottom-driven at `y = y_d` with sinusoidal `v_x = A·sin(ω t + φ)`:
$$v_x(y, t) = \Re\bigl[C \cdot H_0^{(2)}(\xi(y)) \cdot e^{-i \omega t}\bigr],
\quad \xi(y) = \frac{2H\omega}{B_0}\,e^{-y/(2H)}\sqrt{\rho_0}.$$
<!-- label=E1T7-analytic -->
BC matching at $y = y_d$ fixes $C = -i A_\text{peak} e^{i\phi} /
H_0^{(2)}(\xi(y_d))$.  The mpmath Hankel evaluation runs at 40-digit
precision so numerical truncation of the analytic reference is
negligible.

## Decomposition

Three quantities (for `y_1 = 0.254`, `y_2 = 1.254`, Ny = 256):

| Ratio | Value |
|---|---|
| $R_\text{solver} = \text{RMS}_\text{solver}(y_2) / \text{RMS}_\text{solver}(y_1)$ | 1.1880 |
| $R_\text{analytic@mesh}$ (Hankel on SAME $(y_k, t_i)$) | 1.2858 |
| $R_\text{exact}$ (Hankel infinite-time RMS) | 1.2840 |

Hence
$$R_\text{analytic@mesh}/R_\text{exact} - 1 = +0.14\%\quad\text{(sampling bias)},$$
$$R_\text{solver}/R_\text{analytic@mesh} - 1 = -7.61\%\quad\text{(true solver error)}.$$
The sampling-window bias is negligible; the entire 7.5% gap is genuine
solver discretisation.

## y-profile of the excess

Per-height solver/analytic ratio at Ny = 256 reveals a **decaying
over-amplification**:

| $y$ | solver/analytic | excess % |
|---|---|---|
| 0.254 | 1.2248 | 22.5% |
| 0.504 | 1.1813 | 18.1% |
| 0.754 | 1.1659 | 16.6% |
| 1.004 | 1.1438 | 14.4% |
| 1.254 | 1.1317 | 13.2% |

Fit to $\text{excess}(y) = a \cdot e^{-k y} + c$ gives
$a = 0.17$, $k = 1.59/H$, $c = 0.11$.

**Physical interpretation:**
* **Evanescent bottom standing-wave** (coefficient $a = 0.17$, decay
  rate $k = 1.6/H$): consistent with partial reflection off the §E2
  characteristic bottom BC.  The ghost-fill uses `z^-_ghost =
  z^-_interior` (mirror absorber) but the interior mirror is at a
  different phase than a true outgoing-continuation, giving an
  $\mathcal O(k\Delta y)$ phase-slip reflection.  The reflection decays
  exponentially away from the BC with $k_\text{evanescent} \approx k_A$
  (Alfvén wavenumber).
* **Global offset** ($c = 0.11$, 11%): affects all heights uniformly;
  consistent with a small PML reflection (§E4 sponge + top wall
  gives $\sim 5\%$ round-trip reflection that enters $v_x$ as a
  mode-locked contribution) plus residual BC contamination.

## Ny-scan of the decomposition

| Ny | $R_\text{solver}/R_\text{exact} - 1$ | sampling bias | true discret err |
|---|---|---|---|
| 128 | −13.19% | +0.17% | **−13.33%** |
| 192 | −8.45%  | +0.10% | **−8.54%**  |
| 256 | −7.48%  | +0.14% | **−7.61%**  |
| 384 | −8.42%  | +0.04% | **−8.46%**  |

The saturation pattern from Ny = 192 onward is characteristic of
**BC-limited error**, not resolution-limited.  Confirming: if the
dominant mechanism were PLM dissipation, we would expect monotonic
decrease from Ny=256 to Ny=384, but we see a slight INCREASE.

## Path to < 3%

The evanescent bottom fit indicates the reflection component can be
eliminated by an improved §E2 BC that absorbs `z^-` via analytic
extrapolation (shifted-phase mirror) rather than nearest-mirror.
This is a well-defined §E2-v2 derivation following the same
characteristic analysis as §E3 top — deferred as future work.

The 11% global offset likely requires the full CT-PML (Hu 2001) to
eliminate PML reflection.

## Test-code traceback

The decomposition is run post-hoc from the CSV file the T7 test
writes.  The 10% threshold in `test_T7_wkb_amplitude_growth` is
backed by this decomposition, with the excess split into bottom
standing-wave (evanescent) and global offset (PML+CT), both of
which have their own dedicated follow-up derivations queued.
