# 3. Sturm--Liouville spatial discretisation

## 3.1 The Liouville substitution

The elliptic operator in the reduced-pressure projection,
$\mathcal L_\pi = \nabla\!\cdot\!(\rho_0 \nabla \,\cdot\,)$, fails to be
diagonalised by Fourier modes because $\rho_0 = \rho_0(y)$ multiplies
the radial derivative.  The Liouville substitution
$\pi(x,y) = \rho_0(y)^{-1/2}\, q(x,y)$ rewrites the operator in terms
of $q$ as

$$
\frac{1}{\sqrt{\rho_0}}\,\bigl[\partial_x^2 + \partial_y^2 + W(y)\bigr]\, q
\;=\; \frac{1}{\sqrt{\rho_0}}\,\mathcal L_\pi\,\pi,
\qquad
W(y) \;=\; \frac{1}{2}\,\frac{\rho_0''}{\rho_0}
\;-\; \frac{1}{4}\,\left(\frac{\rho_0'}{\rho_0}\right)^{\!2}.
\tag{3.1}
$$

Taking the horizontal Fourier transform $q(x,y) = \sum_k \hat q_k(y)\,
e^{ik_x x}$, the reduced-pressure projection per wavenumber becomes

$$
-\hat q_k''(y) \;+\; \bigl[\,k_x^2 \;-\; W(y)\,\bigr]\, \hat q_k(y) \;=\; g_k(y),
\tag{3.2}
$$

a one-dimensional Schrödinger equation with Dirichlet boundary
conditions $\hat q_k(0) = \hat q_k(L_y) = 0$.  The coefficient
$W(y)$ is time-independent; $k_x^2$ enters only through the diagonal
shift.

## 3.2 The Sturm--Liouville basis

Equation (3.2) defines a Sturm--Liouville problem with background
weight $W(y)$.  Consider the eigenvalue problem

$$
-\psi_n''(y) \;-\; W(y)\,\psi_n(y) \;=\; \mu_n\, \psi_n(y),
\qquad
\psi_n(0) = \psi_n(L_y) = 0,
\tag{3.3}
$$

for $n = 0, 1, \dots$.  By Sturm--Liouville theory the eigenvalues
$\{\mu_n\}$ are real, separated, and bounded below; the
eigenfunctions $\{\psi_n\}$ form a complete orthonormal basis of
$L^2\bigl([0,L_y]\bigr)$ under the standard inner product.  (Here the
weight function in the SL sense is unity because the substitution
has already absorbed $\rho_0$ into the definition of $q$.  The
equivalent form acting on $V$ directly has $\rho_0$ as the weight.)

For fixed $(\mu_n,\psi_n)$, the solution of (3.2) is obtained by
Galerkin projection:

$$
\hat q_k(y) \;=\; \sum_n \frac{\,\langle g_k,\psi_n\rangle\,}{\mu_n + k_x^2}\,\psi_n(y),
\tag{3.4}
$$

where $\langle f,g\rangle = \int_0^{L_y} f(y)\,g(y)\,dy$.  The
right-hand side inner product is computed once per time step, per
wavenumber, by a Clenshaw--Curtis quadrature on the CGL grid
(Section 3.3); the division by $\mu_n + k_x^2$ is a diagonal
inversion; the basis multiplication reconstructs $\hat q_k$ in
real space.  Converting back via $\pi = \rho_0^{-1/2}\,q$ gives the
reduced pressure.

**The key observation.**  The eigenpair $\{(\mu_n,\psi_n)\}$ depends
only on the background $\rho_0$ (through $W$), not on $k_x$ or on
the right-hand side $g_k$.  It is computed once, at setup time, by a
standard one-dimensional SL solver.  After that, the cost per
pressure projection per time step is the cost of one forward SL
synthesis, one diagonal divide, and one backward SL synthesis ---
all $\mathcal O(N_y^2)$ per wavenumber in the naive implementation,
or $\mathcal O(N_y \log N_y)$ per wavenumber with fast-SL machinery
(which we do not use; see Section 8 for discussion).  The horizontal
direction remains $\mathcal O(N_x \log N_x)$ by FFT.  The net cost is
$\mathcal O(N_y^2\, N_x) + \mathcal O(N_y\, N_x \log N_x)$, which for
the GPU-relevant regime $N_y \lesssim 128$ is entirely bandwidth-bound
rather than arithmetic-bound and fits the Fourier pseudo-spectral
cost model.

## 3.3 Chebyshev--Gauss--Lobatto discretisation

We collocate (3.3) on the Chebyshev--Gauss--Lobatto grid
$y_j = (1 - \cos(j\pi/N))\,L_y/2$, $j = 0, 1, \dots, N$, with
$N \equiv N_y - 1$.  The Chebyshev differentiation matrix $D$ of
Trefethen is assembled on this grid.  Quadrature weights are
Clenshaw--Curtis,

$$
w_j \;=\; \frac{2\,L_y}{N}\,
\Biggl[\,1 \;-\; \sum_{m=1}^{\lfloor N/2 \rfloor}
      \frac{b_m}{4 m^2 - 1}\, \cos\!\Bigl(\frac{2mj\pi}{N}\Bigr)\Biggr],
\qquad
b_m = \begin{cases} 1, & 2m = N,\\ 2, & \text{otherwise},\end{cases}
\tag{3.5}
$$

with endpoint weights halved.  These weights integrate polynomials
of degree $\le N$ exactly and provide the correct discrete inner
product on the CGL grid.

Dirichlet boundary conditions are imposed by restricting to the
interior, $j = 1, \dots, N-1$.  The discretised SL operator is

$$
\mathsf L_{\mathrm{SL}}
\;=\;
-\,\bigl[D^2\bigr]_{\mathrm{int}}
\;-\;
\mathrm{diag}\bigl(W(y_{\mathrm{int}})\bigr),
\tag{3.6}
$$

an $(N-1)\times(N-1)$ dense matrix.  Its eigendecomposition
$\mathsf L_{\mathrm{SL}}\,\psi_n = \mu_n\,\psi_n$ is performed once at setup
by LAPACK \texttt{dgeev} (or \texttt{cusolverDnXgeev} in the CUDA
implementation); eigenvectors are orthonormalised against the
Clenshaw--Curtis discrete inner product.  The subsequent
time-stepping uses only the precomputed $\{\mu_n,\psi_n\}_{n=0}^{N-2}$
and the corresponding synthesis/analysis matrices.

## 3.4 The reduced-pressure projection, in seven steps

The anelastic pressure projection step, as implemented, consists of:

1. **Compute divergence** on the CGL grid:
   $d_{ij} = \partial_x u + \partial_y v$ (with $\partial_x$ via FFT,
   $\partial_y$ via $D$), modulated by $\rho_0$ to form the anelastic
   mass-flux divergence $d_{\rho,ij} = \partial_x(\rho_0 u) +
   \partial_y(\rho_0 v)$.

2. **Transform to Fourier** in $x$: $\hat d_{\rho,k}(y) =
   \mathrm{FFT}_x[d_\rho](k,y)$, plus 2/3 dealiasing.

3. **SL-analysis**: for each $k_x$ bin, project onto the SL basis,
   $\hat d_{\rho,k,n} = \sum_j \psi_n(y_j)\,\hat d_{\rho,k}(y_j)\,w_j$.

4. **Diagonal solve** in SL space:
   $\hat \pi_{k,n} = \hat d_{\rho,k,n} / (\mu_n + k_x^2)$.

5. **SL-synthesis**: $\hat\pi_k(y_j) = \sum_n \psi_n(y_j)\,
   \hat\pi_{k,n}$.

6. **Inverse FFT**: $\pi(x,y) = \mathrm{IFFT}_x[\hat\pi](x,y)$, with
   2/3 dealiasing on output.

7. **Project velocity**: $u \leftarrow u - \partial_x \pi$,
   $v \leftarrow v - \partial_y \pi$, enforcing $v(y=0) = v(y=L_y) = 0$
   on the walls.

Each step is $\mathcal O(N_x N_y \log N_x)$ or $\mathcal O(N_x N_y^2)$
in the worst case.  The dominant cost on the GPU is step 7's inverse
FFT and the $D$ multiplication in step 1; the SL analysis and
synthesis are single-precision-storable dense matrix-vector products
and easily bandwidth-saturate on modern hardware.

## 3.5 Convergence behaviour and the polytropic-index dichotomy

For the Poisson-only test we manufacture a solution $\pi^\star(x,y)$
and measure the relative $L^2$ error of the numerical $\pi$ against
$\pi^\star$ on the CGL grid.  Two convergence regimes appear depending
on the smoothness of $W(y)$ at the surface $y = L_y$.

**Integer polytropic index.**  For the standard Lane--Emden $n=1$
and $n=3$ profiles the surface singularity of $\rho_0$ is polynomial
with integer exponent, and $W(y)$ is analytic in the open domain
with only a removable coordinate singularity at the wall.  Chebyshev
collocation of (3.6) achieves exponential convergence: relative
error in the solved $\pi$ reaches $10^{-13}$ at $N_y \approx 32$ and
further $N_y$ refinement is quadrature-noise limited.

**Fractional polytropic index.**  For $n = 3/2$ (our primary test
case) and for the convective $n = 0$ through $n = 1$ range that
arises in realistic stellar envelopes, the surface exponent is a
half-integer and $W(y)$ has a boundary branch-point singularity.
Chebyshev collocation then exhibits only algebraic $\mathcal O(N_y^{-2\sigma-1})$
convergence with $\sigma$ the fractional surface exponent.  For $n =
3/2$ with surface cut $\rho_{\mathrm{cut}} = 0.05$, an $N_y = 64$ grid
reaches $10^{-6}$ relative Poisson error, which is adequate for
anelastic DNS with time-stepping errors at the $10^{-5}$ level, and
further refinement saturates around $N_y \approx 128$ due to a
combination of ill-conditioning in the SL eigenproblem near the
wall and finite-precision arithmetic in the dense eigendecomposition.

For completeness we note that the Chebyshev method is not the only
choice that works on the $n = 3/2$ case: a Liouville prefactor
$\rho_0^\alpha$ with a choice of $\alpha$ matched to the surface
exponent recovers polynomial-exact basis functions and restores
exponential convergence.  We do not use this extension here
because, as will be seen in Section 5, the time-domain closure
problem is independent of the spatial error floor once $N_y$ is
above $\sim 48$.

## 3.6 Connection to the g-mode eigenproblem

The same machinery organises the g-mode generalised eigenproblem
(2.2)--(2.3).  On the CGL grid with interior restriction, assemble

$$
\mathsf L \;=\; -D\,\mathrm{diag}(\rho_0)\,D + k_x^2\,\mathrm{diag}(\rho_0),
\qquad
\mathsf R \;=\; k_x^2\,\mathrm{diag}(N^2\,\rho_0),
\tag{3.7}
$$

and solve $\mathsf R\,V = \omega^2\,\mathsf L\,V$ by LAPACK
\texttt{dggev}.  This matrix $\mathsf L$ is \emph{not} the same as the
SL discretisation (3.6): the latter acts on the Liouville-transformed
variable $q = \rho_0^{1/2}\,v$ while the former acts on $V = v$ directly.
The eigenvalues coincide by construction; the eigenvectors differ by
the Liouville prefactor.  The practical choice --- whether to solve
for $v$ on the CGL grid or for $q$ and transform back --- is made by
matching convenience with the rest of the code: our two-dimensional
solver stores $v$, $b$ in the original frame and applies the SL
basis only in the pressure projection.

For the GYRE benchmark of Section 4 we extend (3.7) to the full
four-variable adiabatic non-rotating system, keeping the same CGL
discretisation and boundary-condition structure.
