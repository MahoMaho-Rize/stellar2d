# 5. The time-domain operator mismatch

## 5.1 The expectation, and its empirical failure

Suppose the spatial discretisation of Section 3 is correct: the
eigenpair $(\omega^2_{\mathrm{EVP}}, V_{\mathrm{EVP}})$ satisfies
$\mathsf R\, V_{\mathrm{EVP}} = \omega^2_{\mathrm{EVP}}\, \mathsf L\, V_{\mathrm{EVP}}$
to machine precision.  Initialise the time-stepping loop of the
anelastic code with $v(x, y, 0) = V_{\mathrm{EVP}}(y)\,\cos(k_x^{(1)} x)$,
$w(x, y, 0) = 0$ (where $w = \partial_t v$), $b(x, y, 0) = 0$.  Let
the code advance by standard RK3 or RK4 pseudo-spectral time-stepping,
projecting onto the divergence-free subspace after each substep.
The expected behaviour is

$$
v(x, y, t) \;=\; V_{\mathrm{EVP}}(y)\,\cos(k_x^{(1)} x)\,\cos(\omega_{\mathrm{EVP}}\, t)
\;+\; \mathcal O\!\bigl(\epsilon_{\mathrm{mach}} \cdot t / \Delta t\bigr),
\tag{5.1}
$$

a harmonic oscillation whose deviation from the initial eigenmode
grows only at the floating-point error rate.  In the Boussinesq
baseline ($\rho_0 \equiv 1$, $N^2 \equiv 1$) this is what we observe:
the per-step deviation $\mathrm{dev}(t_n) = \|v(\cdot,\cdot,t_n) -
a(t_n)\,V_{\mathrm{EVP}}\|_{L^2} / \|V_{\mathrm{EVP}}\|_{L^2}$ (with $a$ the
weighted projection coefficient) grows at $4.5 \times 10^{-5}$ per
step, consistent with a finite but acceptable mixture of truncation
and round-off.

On the Lane--Emden $n = 3/2$ stratification the same code, same
discretisation, same spatial validation certificate, produces a
per-step deviation of $6.9 \times 10^{-4}$: four orders of magnitude
worse than Boussinesq, on a problem whose spatial closure certificate
guarantees machine-precision eigenvector recovery.  Over 100 periods
the deviation grows to $\mathcal O(1)$ --- the eigenmode has been
entirely replaced by a mixture of other modes --- and the
Fourier-peak frequency of the radial velocity at the domain centre
drifts by $-9.4\%$ from the EVP-predicted value, well outside any
acceptable physical range.

This is the failure mode that Section 5 diagnoses.  It is not a
spatial-resolution problem: increasing $N_y$ from 48 to 256 leaves
the per-step deviation at $\sim 6 \times 10^{-4}$.  It is not a time-
step problem: reducing $\Delta t$ by a factor of 16 leaves the
per-step deviation unchanged.  It is not a boundary-condition
problem: alternative BC formulations (regular-at-origin with $1/r$
series match; vacuum-at-surface versus stress-free) change the
deviation at the third significant figure.  It is a discrete-
operator-consistency problem.

## 5.2 The discrete variable-coefficient Leibniz defect

The linearised $v$-equation takes the form (after elimination of $u$
and $\pi$ via continuity and momentum-$u$)

$$
\partial_t w \;=\; -\,L^{-1}\,R\,v,
\qquad
L \;=\; -\partial_y\bigl(\rho_0\,\partial_y\bigr) + k_x^2\,\rho_0,
\qquad
R \;=\; k_x^2\,N^2\,\rho_0,
\tag{5.2}
$$

with $w = \partial_t v$, and is followed by a Poisson projection
onto the divergence-free subspace.  The CUDA time-stepping code does
not apply the composite operator $L^{-1} R$ in one step; it applies
its factors in sequence, in *primitive-node* form:

1. Compute $\partial_y v$ using the Chebyshev matrix $D$: one matrix
   multiply by $\mathsf D$.

2. Multiply pointwise by $\rho_0$: one diagonal multiply.

3. Compute $\partial_y(\rho_0 \partial_y v)$ using $\mathsf D$ again:
   second matrix multiply.

4. Add $k_x^2 \rho_0 v$ and divide by $\rho_0$ (effectively inverting
   the mass matrix of the continuous form), giving the momentum
   right-hand side.

5. Add the buoyancy source $+b$ in step $w$; separately, update $b$
   by $-N^2\, v$ pointwise plus advection.

6. Pressure-project.

The sequence is a correct discretisation of the *continuous*
operator $L$; it is \emph{not} a correct discretisation of the
*discrete* matrix $L$.  At the continuous level,

$$
-\frac{d}{dy}\bigl(\rho_0 \frac{dv}{dy}\bigr) \;=\;
-\rho_0 v'' \;-\; \rho_0' v',
\tag{5.3}
$$

a Leibniz identity that holds pointwise.  At the discrete level with
Chebyshev $\mathsf D$ on the CGL grid, the same identity fails:
$\mathsf D^2 \cdot \mathrm{diag}(\rho_0)$ and $\mathrm{diag}(\rho_0)
\cdot \mathsf D^2$ do not commute, and neither agrees with
$-\mathsf D \cdot \mathrm{diag}(\rho_0) \cdot \mathsf D$ (the form that
appears in assembled $\mathsf L$).  The defect

$$
\Delta_{L}(\rho_0)
\;\equiv\;
\mathsf D \cdot \mathrm{diag}(\rho_0) \cdot \mathsf D
\;-\;
\mathrm{diag}(\rho_0) \cdot \mathsf D^2
\;-\;
\mathrm{diag}(\rho_0') \cdot \mathsf D
\tag{5.4}
$$

is exactly zero as an operator on the continuous polynomial
functions of degree $\le N$, but is nonzero as a matrix: it is in
the null space of the test functions but not in the null space of
an arbitrary discrete state vector.  Its two-norm scales as
$\|\rho_0'\|_\infty$ times the condition number of $\mathsf D$,
giving the $\mathcal O(10^{-4})$ per-step leak observed on Lane--Emden.

A second, smaller defect enters through the pointwise multiplication
by $\rho_0'/\rho_0$ in the pressure-gradient buoyancy term; this is
$\mathcal O(10^{-5})$ on the same background and cannot explain the
leading deviation.

## 5.3 Why changing the spatial basis does not fix the defect

The natural response to seeing a $\rho_0$-dependent discrete
inconsistency is to change the spatial basis so that $\rho_0$
disappears from the leading operator.  Under the Liouville
substitution $\varphi = \rho_0\, V$, the continuous linear problem
becomes

$$
-\varphi''(y) \;+\; k_x^2\,\varphi(y) \;=\;
\frac{k_x^2\,N^2(y)}{\omega^2}\,\varphi(y),
\tag{5.5}
$$

in which $\rho_0$ has dropped out of the elliptic operator.  We
consider two variants of this idea --- call them the
*Fourier-basis EVP substitution* and the *SL-basis EVP substitution*
--- and run the same eigenmode-preservation diagnostic on each.

**Fourier-basis EVP substitution.**  Replace the CGL discretisation
of the eigenvalue problem by a sine-Fourier basis $\varphi_n =
\sin(n\pi y/L_y)$ on the interior, diagonalising $-\partial_{yy}$.
The resulting EVP is $n_{\mathrm{modes}} \times n_{\mathrm{modes}}$ rather
than $N_y \times N_y$ and converges to machine precision on the
Boussinesq problem.  For the Lane--Emden time-domain evolution,
however, the per-step deviation drops only from $6.9 \times 10^{-4}$
to $6.0 \times 10^{-4}$: a 13% improvement, far from closure.

**SL-basis EVP substitution.**  Replace the sine basis above by the
SL basis $\{\psi_n\}$ of Section 3.2 --- so that the eigenvalue
problem and the pressure projection use the *same* inner-product
and basis.  The per-step deviation drops to $5.9 \times 10^{-4}$:
a 4% improvement, again far from closure.

The explanation is that the time-stepping residual is not in the
EVP basis at all.  Even if $\rho_0$ is absent from the EVP operator,
it is present in the $(u,v,b)$ time-stepping equations through:

  - the $\rho_0'/\rho_0$ coefficient in the pressure-gradient
    buoyancy,
  - the $N^2(y)$ pointwise multiplication in the buoyancy equation
    (equivalent under the discrete Leibniz defect when $N^2 \propto
    -\rho_0'/\rho_0$),
  - the $\rho_0$-weighted divergence that drives the Poisson
    projection.

None of these enter the EVP operator, because the EVP eliminates
them algebraically; all of them enter the time-domain loop at the
discrete level, where their Leibniz defect does not cancel.  The
two EVP basis substitutions fix one leak out of three, leaving the
other two.

## 5.4 Why changing the time-stepping state variable does not fix the defect either

A third experiment changes not the basis but the variable advanced
in time.  Call this the *reduced-variable time-stepping*: replace
$v$ by $\varphi = \rho_0 v$ and advance $\varphi$ in place of $v$.
Continuously this is exactly the Liouville reduction of
Section 2.4; the equation for $\varphi$ is Boussinesq-like with no
$\rho_0$ in the leading operator, so one expects the deviation to
drop to Boussinesq levels.

A Python prototype using the same reduced 1D setup as the EVP
substitutions measures a 23% improvement over primitive-node: from
$6.9 \times 10^{-4}$ to $5.3 \times 10^{-4}$.  Still far from
closure.  The explanation is that while the elliptic operator in
$\varphi$-space has no $\rho_0$, the \emph{buoyancy source}
$\partial_t b = -N^2 v$ still requires pointwise multiplication by
$N^2(y)$ to advance $b$; this pointwise product is a scattering
operator on the discrete eigenvector basis and carries its own
Leibniz-like defect.  Changing the state variable moves the leak
around without removing it.

**Assessment.**  Three experimental variations --- two EVP basis
substitutions and one state-variable substitution --- covering the
principal spatial-discretisation and variable-change options for the
time-stepping all fail to close the mismatch at the $10^{-12}$ level
that internal consistency would demand.  The leaks are uniformly in
the $5$ to $7 \times 10^{-4}$ range, irrespective of which
$\rho_0$-dependent term is the one hidden in the leading operator.
This pattern identifies the pathology as structural rather than
basis-dependent: the primitive-node time-stepping, regardless of
which variables are advanced, cannot reproduce the discrete
eigenvectors of its own eigenproblem.  The fix must come from the
time-stepping side, not the spatial side.

## 5.5 Summary of the diagnosis

The anelastic time-stepping loop, written in primitive-node form
(apply $\mathsf D$; multiply by $\rho_0$; multiply by $N^2$; project),
is not the same discrete operator as $\mathsf L^{-1}\mathsf R$, even
when both are derived from the same continuous equations and the
same Chebyshev machinery.  The discrepancy is a variable-coefficient
Leibniz defect: $\mathsf D\,\mathrm{diag}(\rho_0)\,\mathsf D \ne
\mathrm{diag}(\rho_0)\,\mathsf D^2 + \mathrm{diag}(\rho_0')\,\mathsf D$
at the discrete level.  It produces a $6.9 \times 10^{-4}$ per-step
deviation of the discrete eigenvector on Lane--Emden $n = 3/2$,
independent of $N_y$ or $\Delta t$.  Changing the spatial basis of
the EVP improves by 4--13%; changing the time-stepping state
variable improves by 23%; none of these reaches closure.  The
resolution, developed in Section 6, is to assemble the discrete
$\mathsf L^{-1}\mathsf R$ explicitly and apply it as a single
matrix-vector product per wavenumber per time substep.
