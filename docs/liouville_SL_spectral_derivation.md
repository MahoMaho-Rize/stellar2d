---
title: |
  Liouville Normal-Form Reduction of the Variable-Density Pressure Poisson Equation
  and Its Sturm--Liouville Spectral Diagonalisation on GPUs
author: |
  Technical Report --- stellar2d project,
  \texttt{pseudo-astro-explore} branch
date: 2 May 2026
geometry: margin=1in
fontsize: 11pt
mainfont: "Times New Roman"
header-includes: |
  \usepackage{amsmath,amssymb,amsthm}
  \usepackage{bm}
  \newtheorem{theorem}{Theorem}
  \newtheorem{proposition}{Proposition}
  \newtheorem{remark}{Remark}
  \newcommand{\dd}{\mathrm{d}}
  \newcommand{\pp}{\partial}
  \newcommand{\rhob}{\rho_0}
  \newcommand{\phat}{\hat{p}}
  \newcommand{\fhat}{\hat{f}}
  \renewcommand{\Re}{\operatorname{Re}}
---

# 1. Introduction

Pseudo-spectral methods enjoy exponential convergence and exact differentiation in Fourier space, making them the method of choice for incompressible turbulence at moderate Reynolds numbers.  Their efficiency rests on a critical structural property: the pressure Poisson equation $\nabla^{2}p = f$ is *diagonal* in the Fourier basis, since each $e^{i\bm{k}\cdot\bm{x}}$ is an eigenfunction of the Laplacian with eigenvalue $-|\bm{k}|^{2}$.  The Poisson solve thus reduces to a pointwise division in spectral space, at cost $\mathcal{O}(N\log N)$ via the FFT.

When the fluid is density-stratified --- as in stellar convection zones, where $\rho$ varies over many orders of magnitude --- the constant-coefficient Laplacian is replaced by a *variable-coefficient* elliptic operator:

$$\nabla \cdot \!\left(\frac{1}{\rhob(y)}\,\nabla p\right) = f(\bm{x}). \tag{1}$$

Fourier modes are no longer eigenfunctions of this operator.  In spectral space the solve becomes a *dense* matrix equation (a convolution of $\widehat{1/\rhob}$ with $|\bm{k}|^{2}\phat$), destroying the $\mathcal{O}(N\log N)$ advantage.

Existing anelastic/pseudo-incompressible spectral codes (ASH, Rayleigh) circumvent this by retaining Fourier or spherical-harmonic expansions in the *homogeneous* directions and discretising the *inhomogeneous* (radial/vertical) direction with Chebyshev collocation or finite differences, yielding banded matrix systems solved by direct factorisation.  This approach is effective but abandons spectral representation in the stratified direction.

In this note we show that a classical technique from mathematical physics --- the **Liouville normal-form transformation** --- reduces equation (1) to a *constant-coefficient Laplacian plus a potential*, whose Sturm--Liouville (SL) eigenfunctions furnish a basis that **simultaneously diagonalises the operator for every horizontal wavenumber**.  The resulting algorithm replaces the banded solve with a dense matrix--vector product (the SL forward/inverse transform), which maps naturally onto GPU tensor cores via batched GEMM.  We provide a self-contained derivation, analyse the spectral properties of the Liouville potential for polytropic stellar models, and discuss the practical trade-offs relative to Chebyshev--tridiagonal and multigrid alternatives.


# 2. Problem Formulation

## 2.1 The variable-density pressure equation

Consider the momentum equation for a variable-density incompressible flow in a domain periodic in $x$ and bounded (or periodic) in $y$:

$$\rhob(y)\,\frac{\pp \bm{u}}{\pp t} + \rhob(y)\,\bm{u}\cdot\nabla\bm{u} = -\nabla p + \bm{F}, \qquad \nabla\cdot\bm{u} = 0. \tag{2}$$

Dividing by $\rhob$ and taking the divergence, the incompressibility constraint enforces

$$\nabla\cdot\!\left(\frac{1}{\rhob(y)}\,\nabla p\right) = \nabla\cdot\!\left(\frac{\bm{F}}{\rhob} - \bm{u}\cdot\nabla\bm{u}\right) \equiv f(\bm{x}). \tag{3}$$

For the anelastic system $\nabla\cdot(\rhob\bm{u})=0$, the same operator arises after an analogous projection; the mathematical structure is identical.

## 2.2 Fourier reduction in the homogeneous direction

Since $\rhob$ depends only on $y$, the $x$-direction remains homogeneous.  Expanding $p(x,y) = \sum_{k_x} \phat(k_x,y)\,e^{ik_x x}$ and likewise for $f$, each horizontal mode satisfies a one-dimensional ODE:

$$\frac{\dd}{\dd y}\!\left[\frac{1}{\rhob}\,\frac{\dd \phat}{\dd y}\right] - \frac{k_x^{2}}{\rhob}\,\phat = \fhat(k_x,y). \tag{4}$$

This is a *family* of boundary-value problems parameterised by $k_x$, with the variable coefficient $1/\rhob(y)$ appearing in both the diffusion and the reaction terms.  A direct Sturm--Liouville expansion of (4) would require a *different* set of eigenfunctions for each $k_x$ (since the eigenvalue problem depends on $k_x$ through the reaction term), defeating the purpose of a precomputed spectral basis.


# 3. The Liouville Normal-Form Transformation

## 3.1 Change of dependent variable

We introduce the substitution

$$\boxed{\phat(y) = \sqrt{\rhob(y)}\;\, q(y),} \tag{5}$$

and compute the transformed operator.

\begin{proposition}[Liouville reduction]
Under the substitution~(5),
$$\frac{\dd}{\dd y}\!\left[\frac{1}{\rhob}\,\frac{\dd \phat}{\dd y}\right] = \frac{1}{\sqrt{\rhob}}\,\bigl[q'' + W(y)\,q\bigr], \tag{6}$$
where the \textbf{Liouville potential} is
$$W(y) \;=\; \frac{\rhob''}{2\,\rhob} - \frac{3\,(\rhob')^{2}}{4\,\rhob^{2}}. \tag{7}$$
\end{proposition}

\begin{proof}
Differentiating $\phat = \sqrt{\rhob}\,q$:
\begin{align}
\phat' &= \frac{\rhob'}{2\sqrt{\rhob}}\,q + \sqrt{\rhob}\,q', \tag{8a}\\[4pt]
\frac{1}{\rhob}\,\phat' &= \frac{\rhob'}{2\,\rhob^{3/2}}\,q + \frac{q'}{\sqrt{\rhob}}. \tag{8b}
\end{align}
Differentiating~(8b) with respect to~$y$:
\begin{align}
\frac{\dd}{\dd y}\!\left[\frac{1}{\rhob}\,\phat'\right] &= \underbrace{\frac{\dd}{\dd y}\!\left[\frac{\rhob'}{2\,\rhob^{3/2}}\,q\right]}_{I_1} + \underbrace{\frac{\dd}{\dd y}\!\left[\frac{q'}{\sqrt{\rhob}}\right]}_{I_2}. \tag{9}
\end{align}
For~$I_1$:
$$I_1 = \left[\frac{\rhob''}{2\,\rhob^{3/2}} - \frac{3(\rhob')^{2}}{4\,\rhob^{5/2}}\right]q + \frac{\rhob'}{2\,\rhob^{3/2}}\,q'.$$
For~$I_2$:
$$I_2 = -\frac{\rhob'}{2\,\rhob^{3/2}}\,q' + \frac{q''}{\sqrt{\rhob}}.$$
The $q'$ terms cancel, yielding
$$I_1 + I_2 = \frac{q''}{\sqrt{\rhob}} + \left[\frac{\rhob''}{2\,\rhob^{3/2}} - \frac{3(\rhob')^{2}}{4\,\rhob^{5/2}}\right]q = \frac{1}{\sqrt{\rhob}}\left[q'' + W(y)\,q\right]. \qquad \qed$$
\end{proof}

## 3.2 The reduced equation

Substituting (5) and (6) into (4) and multiplying through by $\sqrt{\rhob}$:

$$\boxed{q'' + W(y)\,q - k_x^{2}\,q = g(y), \qquad g \equiv \sqrt{\rhob}\;\fhat.} \tag{10}$$

Defining the **Liouville--Schr\"{o}dinger operator**

$$\mathcal{T} \equiv \frac{\dd^{2}}{\dd y^{2}} + W(y), \tag{11}$$

equation (10) becomes

$$\bigl[\mathcal{T} - k_x^{2}\bigr]\,q = g. \tag{12}$$

**Crucially, $\mathcal{T}$ does not depend on $k_x$.** The horizontal wavenumber enters only as an additive shift of the eigenvalues.  This is the key structural property that enables a single precomputed spectral basis to serve all $k_x$ simultaneously.


# 4. Spectral Diagonalisation

## 4.1 The eigenvalue problem

Consider the Sturm--Liouville (equivalently, time-independent Schr\"{o}dinger) eigenvalue problem for $\mathcal{T}$:

$$\mathcal{T}\,\psi_n \;=\; \psi_n'' + W(y)\,\psi_n \;=\; -\mu_n\,\psi_n, \qquad n = 0, 1, 2, \ldots \tag{13}$$

with appropriate boundary conditions (periodic, Dirichlet, or Neumann in $y$).  By standard SL theory:

1. The eigenvalues $\{\mu_n\}$ are real and form a non-decreasing sequence $\mu_0 \le \mu_1 \le \mu_2 \le \cdots$.
2. The eigenfunctions $\{\psi_n\}$ form a complete orthonormal basis of $L^{2}([0,L_y])$: $\langle\psi_m,\psi_n\rangle = \int_0^{L_y}\psi_m\,\psi_n\,\dd y = \delta_{mn}$.
3. When $W \equiv 0$ (uniform density), $\psi_n$ reduces to the standard Fourier modes and $\mu_n = n^{2}\pi^{2}/L_y^{2}$.

\begin{theorem}[Universal diagonalisation]
The eigenfunctions $\{\psi_n\}$ of $\mathcal{T}$ simultaneously diagonalise the operator $\mathcal{T} - k_x^{2}$ for every $k_x$:
$$\bigl[\mathcal{T} - k_x^{2}\bigr]\,\psi_n = -(\mu_n + k_x^{2})\,\psi_n. \tag{14}$$
\end{theorem}

\begin{proof}
Immediate from the linearity of $\mathcal{T}$ and the fact that $k_x^{2}$ is a scalar multiple of the identity.
\end{proof}

## 4.2 Solution by eigenfunction expansion

Expand $q = \sum_n a_n\,\psi_n$ and $g = \sum_n g_n\,\psi_n$, where $g_n = \langle g,\psi_n\rangle$.  Substituting into (12):

$$\sum_n a_n\,\bigl[-(\mu_n + k_x^{2})\bigr]\,\psi_n = \sum_n g_n\,\psi_n.$$

By orthonormality,

$$\boxed{a_n(k_x) = -\frac{g_n(k_x)}{\mu_n + k_x^{2}}.} \tag{15}$$

**There is no mode coupling.** The variable coefficient $1/\rhob(y)$ is entirely encoded in the eigendata $\{(\mu_n,\psi_n)\}$.  The solve for each $(k_x, n)$ pair is a scalar division, structurally identical to the uniform-density Fourier--Poisson solve $\phat(\bm{k}) = -\fhat(\bm{k})/|\bm{k}|^{2}$.


# 5. Complete Algorithm

## 5.1 Precomputation (once, or whenever $\rhob$ changes)

1. Evaluate $W(y)$ from $\rhob(y)$ via equation (7).
2. Solve the 1D Schr\"{o}dinger eigenvalue problem (13) for $\{(\mu_n, \psi_n)\}_{n=0}^{N_y-1}$.
3. Store the transform matrix $\Psi_{in} = \psi_n(y_i)$, $\,\Psi \in \mathbb{R}^{N_y \times N_y}$.
4. Precompute $\sqrt{\rhob(y_i)}$ as a vector.

## 5.2 Per-timestep Poisson solve

Given the right-hand side $f(x,y)$ on an $N_x \times N_y$ grid:

\medskip
\begin{center}
\renewcommand{\arraystretch}{1.3}
\begin{tabular}{clll}
\hline
Step & Operation & Formula & Cost \\
\hline
1 & FFT in $x$ & $f(x,y) \to \fhat(k_x,y)$ & $\mathcal{O}(N_xN_y\log N_x)$ \\
2 & Weight & $g = \sqrt{\rhob}\cdot\fhat$ & $\mathcal{O}(N_xN_y)$ \\
3 & Forward SL transform & $G = \Psi^{\!\top} g$ & $\mathcal{O}(N_y^{2}N_x)$ \\
4 & Pointwise division & $Q_n(k_x) = -G_n/({\mu_n+k_x^{2}})$ & $\mathcal{O}(N_xN_y)$ \\
5 & Inverse SL transform & $q = \Psi\, Q$ & $\mathcal{O}(N_y^{2}N_x)$ \\
6 & Weight & $\phat = \sqrt{\rhob}\cdot q$ & $\mathcal{O}(N_xN_y)$ \\
7 & IFFT in $x$ & $\phat(k_x,y) \to p(x,y)$ & $\mathcal{O}(N_xN_y\log N_x)$ \\
\hline
\end{tabular}
\end{center}
\medskip

The dominant cost is the pair of dense matrix--vector products in steps 3 and 5, giving a total complexity of $\mathcal{O}(N_y^{2}N_x)$ per Poisson solve.

## 5.3 Comparison of asymptotic and practical costs

\medskip
\begin{center}
\renewcommand{\arraystretch}{1.3}
\begin{tabular}{lccl}
\hline
Method & $y$-direction & Total ($N{\times}N$) & GPU suitability \\
\hline
Fourier ($\rhob{=}\mathrm{const}$) & division & $N^{2}\!\log N$ & cuFFT (optimal) \\
Chebyshev + tridiagonal & Thomas alg. & $N^{2}$ & serial per column \\
\textbf{SL-GEMM (this work)} & \textbf{GEMM} & $\bm{N^{3}}$ & \textbf{cuBLAS batched (high AI)} \\
Iterative (CG / multigrid) & SpMV $\times k$ & $N^{2}k$ & depends on $k$ \\
\hline
\end{tabular}
\end{center}
\medskip

Although the SL-GEMM approach has higher asymptotic complexity, the dense matrix multiply achieves near-peak arithmetic throughput on GPUs (high arithmetic intensity, fully parallel, amenable to tensor cores).  For $N_y \lesssim 4096$, the wall-clock time of a batched DGEMM can be competitive with or faster than the latency-bound Thomas algorithm.


# 6. Properties of the Liouville Potential

## 6.1 General form

For a general smooth density profile $\rhob(y) > 0$, the potential (7) can be rewritten as

$$W = \frac{1}{2}\,\frac{\dd^{2}}{\dd y^{2}}\!\ln\rhob - \frac{1}{4}\left(\frac{\dd}{\dd y}\ln\rhob\right)^{\!2}, \tag{16}$$

or equivalently

$$W = -\frac{\bigl(\sqrt{\rhob}\bigr)''}{\sqrt{\rhob}}. \tag{17}$$

The latter reveals $W$ as the *Schwarz potential* of the map $y \mapsto \sqrt{\rhob(y)}$, connecting it to the classical Schwarzian derivative in complex analysis.

## 6.2 Polytropic stellar models

For a polytropic equation of state $P = K\rhob^{1+1/n}$ in hydrostatic equilibrium with uniform gravity $g$:

$$\rhob(y) = \rhob_c\!\left[1 - \frac{(\gamma-1)\,g\,y}{\gamma\,K\,\rhob_c^{\gamma-1}}\right]^{1/(\gamma-1)}, \tag{18}$$

where $\gamma = 1 + 1/n$.  Substituting into (7):

- For $n = 1$ ($\gamma = 2$): $\rhob$ is linear in $y$, so $\rhob'' = 0$ and $W = -3(\rhob')^{2}/(4\rhob^{2})$, which diverges as $\rhob \to 0$ near the stellar surface.

- For $n = 3/2$ ($\gamma = 5/3$): $\rhob \propto (1 - y/R)^{3/2}$.  Near the surface $y \to R$: $\rhob \propto (R-y)^{3/2}$, giving $W \propto -3/(16(R-y)^{2})$, an inverse-square singularity.

## 6.3 Implications for convergence

The inverse-square singularity $W \propto c/(R-y)^{2}$ near the stellar surface is integrable but not bounded.  The SL eigenfunctions near the singular endpoint behave as $(R-y)^{\alpha}$ with $\alpha$ determined by the indicial equation $\alpha(\alpha-1) + c = 0$.  This is the same structure as the quantum-mechanical Coulomb or centrifugal-barrier problem, and the eigenfunction expansion retains spectral (superalgebraic) convergence for sufficiently smooth source terms $g(y)$, though the rate may be slower than in the bounded-$W$ case.

Practical mitigation strategies include:

1. **Surface excision**: truncate the domain at $\rhob = \epsilon\,\rhob_c$ (standard in stellar simulations; cf.\ atmosphere/sponge layers).
2. **Regularisation**: replace $\rhob$ by $\max(\rhob, \epsilon)$ near the surface, preserving smoothness.
3. **Matched asymptotic basis**: use analytically known singular eigenfunctions near the boundary, matched to numerically computed smooth eigenfunctions in the interior.


# 7. Connection to Stellar g-modes

The eigenvalue problem (13) is mathematically identical to the vertical structure equation for *internal gravity waves* (g-modes) in a stratified atmosphere.  The Brunt--V\"{a}is\"{a}l\"{a} frequency $N^{2}(y)$, which governs the oscillation spectrum, is related to the density stratification by

$$N^{2} = -g\!\left(\frac{\dd\ln\rhob}{\dd y} - \frac{1}{\gamma}\,\frac{\dd\ln P_0}{\dd y}\right). \tag{19}$$

For an isentropic background, $N^{2} = 0$ and $W$ reduces to a purely geometric quantity.  For a non-isentropic stratification, $W$ incorporates both the geometric and the buoyancy contributions.

\begin{remark}
The eigenvalues $\{\mu_n\}$ of the Liouville--Schr\"{o}dinger operator $\mathcal{T}$ are directly related to the squared g-mode frequencies.  The Poisson-solver precomputation therefore yields the g-mode spectrum as a \textbf{free byproduct}, with no additional eigenvalue solve required.
\end{remark}


# 8. GPU Implementation Considerations

## 8.1 Batched GEMM formulation

Steps 3 and 5 of the algorithm (Section 5.2) can be expressed as a single matrix--matrix multiplication:

$$\bm{G} = \Psi^{\!\top}\!\cdot\bm{g}, \qquad \bm{q} = \Psi\cdot\bm{Q},$$

where $\bm{g}, \bm{G}, \bm{Q}, \bm{q} \in \mathbb{R}^{N_y \times N_x}$.  On modern GPUs (NVIDIA Ampere/Hopper), FP64 GEMM achieves over 1 TFLOP/s, and the arithmetic intensity of GEMM ($\mathcal{O}(N)$ flops per byte read) ensures compute-bound execution for $N_y \ge 256$.

## 8.2 Workflow integration

The SL-GEMM Poisson solver slots into an existing pseudo-spectral time integrator (e.g., IFRK3 with cuFFT) by replacing the spectral-space division $\phat = -\fhat/|\bm{k}|^{2}$ with the sequence:
$$\text{cuFFT (R2C in } x\text{)} \;\to\; \text{cuBLAS DGEMM} \;\to\; \text{pointwise divide} \;\to\; \text{cuBLAS DGEMM} \;\to\; \text{cuFFT (C2R in } x\text{)}.$$

All existing infrastructure --- IFRK3 time integration, skew-symmetric convection, 2/3 dealiasing, VRAM frame buffering --- is reused without modification.

## 8.3 Memory requirements

The transform matrix $\Psi$ requires $N_y^{2}$ doubles $= 32$ MiB for $N_y = 2048$.  This is modest compared to the flow-field arrays (approximately 100 MiB each at $2048^{2}$) and the VRAM frame buffer (approximately 10 GiB).


# 9. Relation to Existing Methods

The approach presented here does not appear in the existing literature in the specific combination proposed (Liouville reduction + SL spectral basis + GPU GEMM for stratified convection).  The individual ingredients are well established:

- **Liouville normal form**: standard in mathematical physics (Sturm 1836; Liouville 1837), see Zettl (2005) for a modern treatment.
- **SL eigenfunction expansions for elliptic PDEs**: discussed by Boyd (2001, *Chebyshev and Fourier Spectral Methods*) as theoretically elegant but rarely implemented due to the lack of a fast transform.
- **Anelastic pseudo-spectral solvers**: ASH (Clune et al.\ 1999), Rayleigh (Featherstone and Hindman 2016) use spherical harmonics in angle + Chebyshev/FD in radius.
- **GPU pseudo-spectral DNS**: numerous codes for homogeneous turbulence (cuFFT-based), none for stratified/anelastic flows.

The *gap* arises because the three communities (SL theory, stellar spectral methods, GPU computing) have not intersected:

1. The stellar spectral community (ASH/Rayleigh) operates on CPUs where GEMM is not competitive with banded solves.
2. The GPU spectral community solves only uniform-density problems where FFT suffices.
3. The variable-density GPU community uses finite-volume methods with multigrid, not spectral methods.


# 10. Summary

We have shown that:

1. The Liouville substitution $\phat = \sqrt{\rhob}\,q$ reduces the variable-coefficient pressure Poisson equation to a constant-coefficient Laplacian plus a potential $W(y)$.

2. The Sturm--Liouville eigenfunctions of $\dd^{2}/\dd y^{2} + W(y)$ **simultaneously diagonalise** the operator for all horizontal wavenumbers $k_x$, with eigenvalues $-(\mu_n + k_x^{2})$.

3. The resulting Poisson solve is a sequence of FFT, GEMM, pointwise division, GEMM, IFFT, naturally suited to GPU execution.

4. The eigenvalues $\{\mu_n\}$ are directly related to stellar g-mode frequencies, providing an astrophysical diagnostic as a free byproduct.

5. The potential $W(y)$ may exhibit inverse-square singularities near stellar surfaces, requiring excision or regularisation.

The practical viability of this approach hinges on a GPU benchmark comparing SL-GEMM against Chebyshev--tridiagonal and multigrid alternatives, and on the convergence behaviour of the SL expansion for realistic stellar density profiles.  These constitute the immediate next steps toward a publication-ready implementation.


# References

- Boyd, J. P. (2001). *Chebyshev and Fourier Spectral Methods*, 2nd ed. Dover.
- Clune, T. C. et al. (1999). Computational Aspects of a Code to Study Rotating Turbulent Convection in Spherical Shells. Parallel Computing, 25, 361.
- Featherstone, N. A. and Hindman, B. W. (2016). The Spectral Amplitude of Stellar Convection and Its Scaling in the High-Rayleigh-Number Regime. ApJ, 818, 32.
- Liouville, J. (1837). Sur le developpement des fonctions... J. Math. Pures Appl., 1, 253.
- Sturm, C. (1836). Memoire sur les equations differentielles lineaires du second ordre. J. Math. Pures Appl., 1, 106.
- Zettl, A. (2005). *Sturm--Liouville Theory*. AMS Mathematical Surveys and Monographs, Vol. 121.
