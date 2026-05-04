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

The eigenpair $\{(\mu_n,\psi_n)\}$ depends only on the background
$\rho_0$ (through $W$), not on $k_x$ or on the right-hand side $g_k$,
and is computed once at setup time by a standard one-dimensional SL
solver.  After that, the cost per
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
$\mathsf L_{\mathrm{SL}}\,\psi_n = \mu_n\,\psi_n$ is performed once at
setup by a standard dense eigensolver; eigenvectors are
orthonormalised against the Clenshaw--Curtis discrete inner product.
The subsequent time-stepping uses only the precomputed
$\{\mu_n,\psi_n\}_{n=0}^{N-2}$ and the corresponding synthesis/analysis
matrices.  Implementation details are deferred to Appendix A.

## 3.4 Algorithmic pressure projection

Given the velocity field $(u,v)$ on the CGL grid, the reduced-pressure
projection computes $(u', v') = \mathcal P(u,v)$ such that
$\partial_x(\rho_0 u') + \partial_y(\rho_0 v') = 0$ and
$v'|_{y=0,L_y} = 0$, through the composed mapping
$$
\mathcal P \;=\;
\bigl(I - \nabla \circ
\mathcal F_x^{-1}\,\Psi\,\Lambda^{-1}\,\Psi^\top\,\mathcal F_x
\circ \nabla\!\cdot(\rho_0\,\cdot)\bigr),
\tag{3.6*}
$$
where $\mathcal F_x$ is the discrete Fourier transform in $x$, $\Psi$
is the matrix of SL eigenvectors $\psi_n(y_j)$, and $\Lambda =
\mathrm{diag}(\mu_n + k_x^2)$ is the diagonal of SL eigenvalues
shifted by the horizontal wavenumber.  The action of $\mathcal P$ on
$(u, v)$ costs $\mathcal O(N_x N_y^2) + \mathcal O(N_x N_y \log N_x)$:
the former from the dense SL analysis/synthesis, the latter from the
FFT in $x$.  On the GPU, the SL analysis and synthesis saturate memory
bandwidth, and the overall cost is dominated by the 2D FFT for the
GPU-relevant regime $N_y \lesssim 128$.

## 3.5 Convergence behaviour

The spatial discretisation exhibits two convergence regimes depending
on the surface smoothness of $\rho_0$.  For integer polytropic
indices ($n = 1, 3$), Chebyshev collocation achieves exponential
convergence --- relative Poisson error reaches $10^{-13}$ at
$N_y \approx 32$.  For half-integer indices ($n = 3/2$), the
surface branch-point reduces convergence to algebraic,
$\mathcal O(N_y^{-2\sigma-1})$ with $\sigma$ the fractional exponent;
$N_y = 64$ suffices to reach $10^{-6}$ relative Poisson error, which
is adequate for the temporal-closure analysis of Sections 5--6.
The detailed derivation of the convergence dichotomy and the
alternative Liouville-prefactor basis that restores exponential
convergence on half-integer profiles are given in Appendix A.2.
The *spatial* discretisation thus closes to machine precision on
integer profiles and to $10^{-6}$ on half-integer profiles; any
larger discrepancy observed in the time-domain loop (Section 5) is
by elimination not a spatial truncation error.

## 3.6 Connection to the g-mode eigenproblem

The same machinery organises the g-mode generalised eigenproblem
(2.2)--(2.3).  On the CGL grid with interior restriction, assemble

$$
\mathsf L \;=\; -D\,\mathrm{diag}(\rho_0)\,D + k_x^2\,\mathrm{diag}(\rho_0),
\qquad
\mathsf R \;=\; k_x^2\,\mathrm{diag}(N^2\,\rho_0),
\tag{3.7}
$$

and solve $\mathsf R\,V = \omega^2\,\mathsf L\,V$ by a dense
generalised eigensolver.  This matrix $\mathsf L$ is \emph{not} the
same as the
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
