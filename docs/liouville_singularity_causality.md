---
title: "On the artificial nature of the Liouville potential singularity and its implications for basis design"
author: "stellar2d project"
date: "2 May 2026"
---

# 1. The causality inversion: QM vs astrophysics

The Liouville normal-form reduction transforms the variable-coefficient pressure equation

$$\nabla \cdot \left(\frac{1}{\rho_0(y)}\,\nabla p\right) = f$$

into a Schrodinger-type equation

$$q'' + W(y)\,q - k_x^2\,q = g, \qquad W(y) = \frac{\rho_0''}{2\rho_0} - \frac{3(\rho_0')^2}{4\rho_0^2},$$

via the substitution $\hat{p} = \sqrt{\rho_0}\,q$.

The resulting inverse-square singularity $W \propto -c/(R-y)^2$ near the stellar surface ($\rho_0 \to 0$) is mathematically identical to structures encountered in quantum mechanics (Coulomb potential, centrifugal barrier).  However, the **causal relationship** between the singularity and the physics is reversed.


## 1.1 In quantum mechanics: potential is primary

The logical chain in QM is:

$$V(r) \;\;\text{(physical input)} \;\;\longrightarrow\;\; \psi(r) \to 0 \;\;\text{(mathematical consequence)}$$

The inverse-square or Coulomb potential is the *fundamental physical quantity*.  The wavefunction (and hence the probability density $|\psi|^2$) vanishes at certain boundaries *because* the potential forces it to.  Singular potentials are ontologically real: they represent actual physical interactions (nuclear charge, angular momentum barrier).

Techniques such as Coulomb-Sturmian bases ($r^\ell e^{-\alpha r} L_n$), DVR on Laguerre grids, and R-matrix matching are designed to accommodate a singularity that *cannot be removed* because it is a property of the physical system.


## 1.2 In astrophysics: density is primary

In the stellar structure problem, the causal chain is reversed:

$$\rho_0(y) \to 0 \;\;\text{(physical input)} \;\;\longrightarrow\;\; W(y) \to -\infty \;\;\text{(mathematical artifact)}$$

The density profile $\rho_0(y)$ is determined by hydrostatic equilibrium (Lane-Emden, MESA models, etc.).  The stellar surface is simply where the gas runs out --- there is no "infinite potential barrier" in the physics.  The original elliptic operator $\nabla \cdot (\rho_0^{-1} \nabla p)$ is a *degenerate elliptic* operator at the surface: it loses ellipticity smoothly as $\rho_0 \to 0$, without any true singularity in the PDE.

The divergence of $W(y)$ is entirely manufactured by the $\sqrt{\rho_0}$ substitution.  It is a coordinate singularity, not a physical one.


# 2. Implications for numerical treatment

This distinction has concrete consequences for the choice of regularisation strategy.


## 2.1 QM methods are applicable but over-engineered

Since the mathematical structure is identical, all QM techniques for inverse-square potentials (Sturmian bases, DVR, matched asymptotics) *work*.  Phase 0 E3 confirms that the convergence degradation is caused entirely by the singularity: a smooth $\rho_0$ restores exponential convergence.

However, QM methods are designed to *live with* an irreducible singularity.  In this problem, the singularity is reducible --- it was introduced by a specific variable substitution and can in principle be eliminated by a better one.  Applying QM techniques without recognising this amounts to treating a coordinate artifact as a fundamental constraint.


## 2.2 The singularity can be eliminated at the source

Since $W(y)$ diverges only because $\sqrt{\rho_0}$ vanishes, three strategies are available that have no QM analogue:

**Strategy A: Augmented Liouville substitution.**
Replace $\hat{p} = \sqrt{\rho_0}\,q$ with $\hat{p} = \rho_0^s\,q$ for some $s \neq 1/2$.  The induced potential becomes

$$W_s(y) = s(1-s)\left(\frac{\rho_0'}{\rho_0}\right)^2 + s\,\frac{\rho_0''}{\rho_0} - \frac{k_x^2}{\rho_0^{2s-1}}\,\delta_{s \neq 1/2}.$$

The key question is whether there exists an $s$ such that:
  1. $W_s$ remains bounded as $\rho_0 \to 0$, and
  2. the resulting operator still separates into a $k_x$-independent part plus a scalar shift.

For the standard Liouville form ($s = 1/2$), condition (2) is satisfied but (1) is not.  Verifying whether an alternative $s$ can satisfy both is a short calculation that should precede any engineering work.

**Strategy B: Absorb the singularity into the basis weight.**
Following Dedalus, use a Jacobi basis $J_n^{(\alpha,\beta)}$ with weight exponents chosen to match the decay rate of $\rho_0$ near the surface.  For Lane-Emden $n=3/2$: $\rho_0 \propto (R-y)^{3/2}$, so $\alpha = 3/4$ (the indicial exponent of the SL equation at the singular endpoint) is the natural choice.  This does not remove the singularity from $W$ but ensures the basis functions carry the correct asymptotic behavior, restoring spectral convergence.

**Strategy C: Pre-multiply by the singular factor.**
Define $\tilde{\psi}_n(y) = (R-y)^{3/4}\,\phi_n(y)$ where $\phi_n$ is smooth.  The SL eigenvalue problem for $\phi_n$ has a bounded effective potential $\tilde{W}$.  This is the GYRE philosophy applied to the Liouville framework: absorb the singularity into the variable definition rather than the basis or the substitution.

All three strategies exploit the fact that the singularity is artificial.  In QM, only strategies B and C are available (the potential cannot be substituted away).


# 3. The indicial equation

For the Lane-Emden $n=3/2$ surface, $\rho_0 \propto (R-y)^{3/2}$, and

$$W(y) \;\approx\; -\frac{3}{16(R-y)^2} \qquad (y \to R).$$

The indicial equation at the regular singular point is

$$\alpha(\alpha - 1) - \frac{3}{16} = 0 \qquad \Longrightarrow \qquad \alpha = \frac{3}{4} \;\;\text{or}\;\; \alpha = \frac{1}{4}.$$

The physically admissible (square-integrable) solution behaves as $(R-y)^{3/4}$.  This exponent is the key input for strategies B and C above.


# 4. Recommended verification sequence

Before committing to a particular strategy in CUDA:

1. **Verify Strategy A analytically**: compute $W_s$ for general $s$ and check whether the $k_x$-independence property (the core selling point of Liouville) survives for $s \neq 1/2$.  If it does not, Strategy A is ruled out.

2. **Verify Strategy C numerically (Python)**: solve the SL eigenvalue problem for $\tilde{W}$ after the $(R-y)^{3/4}$ extraction.  If $\tilde{W}$ is bounded and the convergence order improves to exponential, Strategy C is confirmed.

3. **Benchmark Strategy B as a fallback**: use scipy Jacobi quadrature with $\alpha = 3/4$ and measure convergence.  This requires the least mathematical novelty but sacrifices the g-mode byproduct.

The choice between B and C determines the paper angle: B leads to a methods-comparison paper (JCP), C preserves the unified-basis narrative (ApJS).


# 5. Summary

| Aspect | Quantum mechanics | Astrophysical Liouville |
|---|---|---|
| Fundamental quantity | Potential $V(r)$ | Density $\rho_0(y)$ |
| Singularity origin | Physical (Coulomb, centrifugal) | Artificial ($\sqrt{\rho_0}$ substitution) |
| Can singularity be removed? | No | Yes (in principle) |
| Standard treatment | Sturmian basis, DVR, R-matrix | Applicable but over-engineered |
| Optimal treatment | N/A | Eliminate at source (Strategy A/C) |
| Indicial exponent | $\ell$ (angular momentum) | $3/4$ (Lane-Emden $n=3/2$) |
