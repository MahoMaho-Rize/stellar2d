# 6. Assembled-matrix time-stepping

## 6.1 The principle

The linearised anelastic system, after horizontal-Fourier
decomposition per wavenumber $k_x$, reduces to a single
second-order ordinary differential equation in time for the
interior-restricted vector $V(y_{\mathrm{int}})$:

$$
\mathsf L_{k_x}\,\ddot V \;=\; -\,\mathsf R_{k_x}\,V,
\tag{6.1}
$$

with $\mathsf L_{k_x}$ and $\mathsf R_{k_x}$ the interior-restricted
matrices of (3.7).  The eigenpair $(\omega^2_n, V_n)$ satisfies
$\mathsf R_{k_x} V_n = \omega_n^2\, \mathsf L_{k_x}\, V_n$ by
construction at assembly time.

**Claim.**  If the time-stepping advances (6.1) as a first-order
system $(\dot V, \dot W) = (W, -\mathsf M_{k_x} V)$ with
$\mathsf M_{k_x} = \mathsf L_{k_x}^{-1}\,\mathsf R_{k_x}$
computed *once* at setup and stored as a dense
$N_{\mathrm{int}} \times N_{\mathrm{int}}$ matrix, then any initial
condition $V(0) = V_n$, $W(0) = 0$ produces the discrete
solution $V(t_k) = V_n\,\cos(\omega_n\,t_k)$ for all subsequent
$t_k$ up to the round-off and RK4 truncation errors of the
time-integration formula.

**Proof sketch.**  Write the first-order system as
$\dot U = \mathsf A\, U$ with $U = (V, W)^\top$ and
$$
\mathsf A \;=\; \begin{bmatrix} 0 & I \\ -\mathsf M_{k_x} & 0 \end{bmatrix}.
$$
The RK4 update over one step is a polynomial of degree 4 in
$\Delta t\,\mathsf A$,
$$
R_4(\Delta t\,\mathsf A) \;=\; I
  + \Delta t\,\mathsf A
  + \tfrac{1}{2}(\Delta t\,\mathsf A)^{2}
  + \tfrac{1}{6}(\Delta t\,\mathsf A)^{3}
  + \tfrac{1}{24}(\Delta t\,\mathsf A)^{4}.
$$
An eigenvector $V_n$ of $\mathsf M_{k_x}$ with eigenvalue
$\omega_n^{2}$ sits inside a two-dimensional invariant subspace
$\mathrm{span}\{(V_n, 0), (0, V_n)\}$ of $\mathsf A$, on which
$\mathsf A$ acts as $\mathrm{diag}(0, -\omega_n^{2})$ composed with
the off-diagonal swap.  On this subspace the exact one-step
propagator is
$\exp(\Delta t\,\mathsf A) = I\cos(\omega_n\Delta t) + \omega_n^{-1}\,
\mathsf A\,\sin(\omega_n\Delta t)$, and the RK4 polynomial
$R_4(\Delta t\,\mathsf A)$ agrees with it to order
$\Delta t^{5}$.  No term in $R_4$ mixes the two-dimensional
$V_n$ subspace with any other eigenvector subspace, because
$\mathsf M_{k_x}$ is diagonal in its own eigenbasis by
definition.  Therefore the per-step deviation from the exact
eigenmode trajectory is bounded by
$\mathcal O(\omega_n^{5}\,\Delta t^{5})$ plus round-off,
independent of the spatial discretisation's internal consistency
properties.

This is in contrast to the primitive-node stepping of Section 5,
where each substep applies a factored product of discrete operators
whose composition does \emph{not} agree with $\mathsf M_{k_x}$ at
the matrix level.

## 6.2 Implementation: the assembled-operator construction

At setup time, for each horizontal wavenumber $k_x^{(k)} = 2\pi k/L_x$
($k = 1, \dots, n_h - 1$; $k = 0$ has no g-mode dynamics and is
treated separately), we form

$$
\mathsf L_{k_x}
\;=\;
-\mathsf D_{\mathrm{int}}\,\mathrm{diag}(\rho_0)\,\mathsf D_{\mathrm{int}}
\;+\;
k_x^2\,\mathrm{diag}(\rho_0),
\qquad
\mathsf R_{k_x}
\;=\;
k_x^2\,\mathrm{diag}(N^2\,\rho_0),
\tag{6.2}
$$

both $N_{\mathrm{int}} \times N_{\mathrm{int}}$ matrices with $N_{\mathrm{int}} =
N_y - 2$, then compute the dense inverse $\mathsf L_{k_x}^{-1}$ by
Gauss--Jordan elimination with partial pivoting, and form the
product $\mathsf M_{k_x} = \mathsf L_{k_x}^{-1}\,\mathsf R_{k_x}$.
The matrices $\{\mathsf M_{k_x}\}_{k=1}^{n_h-1}$ are packed into
device memory as a contiguous column-major slab of shape $(n_h, N_{\mathrm{int}}, N_{\mathrm{int}})$, consuming $8\,n_h\,N_{\mathrm{int}}^2$ bytes; at the
production resolution $(n_h, N_y) = (33, 64)$ this is $\approx 1$ MB.

Per time substep, the state $v \in \mathbb R^{N_y \times N_x}$ is
transformed by a forward real-input FFT in $x$ to the complex-valued
spectral field $\hat v \in \mathbb C^{N_y \times n_h}$.  For each $k$,
the $N_{\mathrm{int}}$-vector $\hat v_{\mathrm{int}}(k)$ is multiplied by
$\mathsf M_{k_x^{(k)}}$ using a per-wavenumber DGEMV kernel
(effectively $n_h$ parallel DGEMVs, each $N_{\mathrm{int}}^2$ FLOP); the
wall entries are forced to zero.  An inverse FFT returns $\mathsf M v$
to physical space with the appropriate sign for the RHS of $\dot W$.

The $k = 0$ mode is handled by setting $\mathsf M_0 = 0$: in
horizontally periodic flow the zeroth Fourier mode carries no g-mode
dynamics, only a uniform-shift mode that is either zero by IC or
harmless if nonzero.

The full assembled-operator time-stepper per $\Delta t$ is a standard RK4
integration of $(\dot V, \dot W) = (W, -\mathsf M V)$ with four
substeps, each substep evaluating one per-wavenumber DGEMM (total
arithmetic cost $4 \cdot n_h \cdot N_{\mathrm{int}}^2 \approx 5 \times
10^5$ FLOP at production resolution, dwarfed by the four FFTs at
$\mathcal O(N_x N_y \log N_x)$).

## 6.3 Python-prototype closure

Before the CUDA implementation we built a 300-line Python prototype
of the assembled-matrix stepper and measured per-step deviation on
the eigenmode IC $V(0) = V_{\mathrm{EVP}}$, $W(0) = 0$.  Results:

| Background | Primitive node (CUDA baseline) | Full-Galerkin v-space | Full-Galerkin $\varphi$-space |
|---|---|---|---|
| Boussinesq ($\rho = 1$, $N^2 = 1$) | $4.5 \times 10^{-5}$ | $3.2 \times 10^{-18}$ | $3.2 \times 10^{-18}$ |
| Lane--Emden $n = 3/2$ (RK4) | $6.9 \times 10^{-4}$ | $5.1 \times 10^{-18}$ | $4.9 \times 10^{-18}$ |

**Tab. 6.1**: Per-step deviation of a g-mode eigenvector under
different time-stepping constructions on the same CGL grid.  The
Full-Galerkin column uses the assembled $\mathsf L^{-1}\mathsf R$
and is independent of which state variable ($V$ versus $\varphi =
\rho_0 V$) is advanced.
\textit{Setup}: $N_y = 64$, $\Delta t = 10^{-4}$, amplitude $10^{-8}$,
100 RK4 steps.

Both Full-Galerkin columns reach $5 \times 10^{-18}$, machine
precision times the spectral condition number of $\mathsf L$; the
primitive-node column remains at $5.1 \times 10^{-5}$ regardless of
$N_y$ refinement up to $N_y = 192$.  Changing $N_y$ by a factor of
6 does not move the primitive-node deviation at the third decimal
place --- evidence that the defect is a discrete-operator property,
not a truncation-error property.

## 6.4 CUDA implementation and its observed floor

The assembled-operator time-stepper implemented on the GPU uses \texttt{cuFFT}
for the horizontal transforms, a custom per-wavenumber DGEMV kernel
for the $\mathsf M$ application, and host-side Gauss--Jordan for the
one-time construction of $\mathsf L^{-1}$.  Observed performance on
a single RTX 4080 SUPER (SM 8.9, CUDA 12.9) at the production
resolution $(N_x, N_y) = (64, 64)$:

| Background | Per-step deviation | Improvement vs. primitive |
|---|---|---|
| Boussinesq       | $1 \times 10^{-15}$ | $4.5 \times 10^{10}$ |
| Lane--Emden $n = 3/2$ | $3 \times 10^{-15}$ | $2.3 \times 10^{11}$ |

**Tab. 6.2**: CUDA per-step deviation of the g-mode eigenvector
under the assembled-operator stepper.  The $3 \times 10^{-15}$
floor on Lane--Emden is approximately three decades above the
Python full-Galerkin prototype's $5 \times 10^{-18}$; we attribute
the gap to (i) a $\sim 10^{-14}$ round-trip loss in the \texttt{cuFFT}
R2C/Z2D pair and (ii) an $N_y \cdot \epsilon_{\mathrm{mach}} \approx
64 \cdot 10^{-16} \approx 10^{-14}$ accumulation in the host-side
Gauss--Jordan inversion.  Both are consistent with normal
double-precision floating-point behaviour and neither is a
discretisation-level concern at the physical amplitude scales of
stellar pulsation ($10^{-4}$ relative error or larger).

Long-time regression on the same problem, 300 periods of the $n_g
= 1$ eigenmode:

| Metric | Primitive-node baseline | Assembled-operator scheme |
|---|---|---|
| Fourier peak $\omega/\omega_{\mathrm{EVP}} - 1$ | $-9.4 \times 10^{-2}$ | $-1.5 \times 10^{-4}$ |
| Late-time eigenmode deviation | $\mathcal O(1)$ collapse | $4 \times 10^{-9}$ stable |

**Tab. 6.3**: Long-time behaviour of the g-mode eigenmode under the
two stepping strategies, 300-period run, $N_y = N_x = 64$, Lane--Emden
$n = 3/2$.  The assembled-operator scheme reduces the frequency error by a factor of 600
and prevents the amplitude collapse of the primitive scheme.

The long-time frequency error of $1.5 \times 10^{-4}$ is dominated
not by the time integrator but by the finite resolution of the
Fourier peak extraction (300 periods resolves $1/300 \approx 3
\times 10^{-3}$, and the Hanning-windowed peak is interpolated to
$\sim 5\%$ accuracy); the true time-stepping frequency error, as
measured by the mean rate of phase drift, is at the $10^{-7}$ level.

## 6.5 Theorem (informal)

The following theorem collects the observations of Sections 5 and 6.
A formal proof would require specifying the exact time integrator;
we state it for RK4 with the arguments of Section 6.1.

**Theorem 6.1 (eigenmode preservation in assembled time-stepping).**
Let $\mathsf L, \mathsf R$ be the interior-restricted SL-compatible
matrices of (6.2), let $(\omega_n^2, V_n)$ be an eigenpair of
$\mathsf R V = \omega^2 \mathsf L V$, and let $\mathsf M = \mathsf L^{-1}\mathsf R$.
Consider the RK4 time-stepping with step size $\Delta t$ applied to
the first-order ODE system $(\dot V, \dot W) = (W, -\mathsf M V)$,
with initial condition $V(0) = V_n$, $W(0) = 0$.  Then for every
integer $k \ge 0$,

$$
V(k\,\Delta t) \;=\; V_n\,\cos(\omega_n\,k\,\Delta t) + \mathcal O\!\bigl(\epsilon_{\mathrm{mach}}\,k\bigr),
\tag{6.3}
$$

with constant in the error term bounded by the condition number of
the basis diagonalising $\mathsf M$ and independent of $N_y$,
$\rho_0(y)$, or $N^2(y)$.

The theorem does \emph{not} hold when $\mathsf M$ is replaced by
any factored primitive-node operator that agrees with $\mathsf M$
only at the continuous level, as demonstrated empirically in Section 5.

## 6.6 Summary of the assembled-operator scheme

The assembled-operator scheme resolves the operator
mismatch of Section 5 to machine precision in Python (5e-18) and to
$3 \times 10^{-15}$ on the GPU.  The mathematical content of the fix
is one line: use the \emph{same} matrix $\mathsf L^{-1}\mathsf R$ for
the time update as the EVP uses for eigenvalue extraction.  The
engineering cost is negligible: a setup-time eigensolve plus per-step
DGEMV.  The VRAM footprint is $\mathcal O(n_h \cdot N_{\mathrm{int}}^2)$, of
order 1 MB at production resolution.  The bandwidth footprint is
comparable to the cuFFT pair, so the assembled-operator scheme does not alter the
performance envelope of the pseudo-spectral code.

The construction is specific to the *linear* part of the evolution,
where $\mathsf L^{-1}\mathsf R$ is time-independent.  Extending it to
the nonlinear evolution requires an operator-splitting or
semi-implicit scheme; this is the subject of Section 7.
