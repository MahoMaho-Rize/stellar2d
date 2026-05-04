# Equation Reference for stellar2d

All numbered equations in this document are referenced by source code comments
of the form `// Eq. (X.Y)`. This is the authoritative specification — if code
disagrees with this document, the code has a bug.

Coordinate system: axisymmetric spherical $(r, \theta)$, with
$\partial/\partial\phi = 0$ and $v_\phi = 0$. $\theta \in [0, \pi]$ is the
polar angle measured from the north pole.

---

## §1 Governing Equations

### Conservative variables

$$
\mathbf{U} = \begin{pmatrix} \rho \\ \rho v_r \\ \rho v_\theta \\ \rho E \end{pmatrix}
\tag{1.1}
$$

where total specific energy $E = e + \tfrac{1}{2}(v_r^2 + v_\theta^2)$ and $e$
is specific internal energy.

### Ideal gas equation of state

$$
P = (\gamma - 1)\,\rho\,e
\tag{1.2}
$$

### Sound speed

$$
c_s = \sqrt{\gamma\,P / \rho}
\tag{1.3}
$$

### Euler equations in conservative form (spherical, axisymmetric)

$$
\frac{\partial \mathbf{U}}{\partial t}
+ \frac{1}{r^2}\frac{\partial(r^2 \mathbf{F})}{\partial r}
+ \frac{1}{r\sin\theta}\frac{\partial(\sin\theta\,\mathbf{G})}{\partial\theta}
= \mathbf{S}
\tag{1.4}
$$

Radial flux:

$$
\mathbf{F} = \begin{pmatrix}
\rho v_r \\
\rho v_r^2 + P \\
\rho v_r v_\theta \\
(\rho E + P)\,v_r
\end{pmatrix}
\tag{1.5}
$$

Theta flux:

$$
\mathbf{G} = \begin{pmatrix}
\rho v_\theta \\
\rho v_r v_\theta \\
\rho v_\theta^2 + P \\
(\rho E + P)\,v_\theta
\end{pmatrix}
\tag{1.6}
$$

Geometric + gravity source:

$$
\mathbf{S} = \begin{pmatrix}
0 \\
\displaystyle\frac{\rho v_\theta^2}{r} + \frac{2P}{r}
  - \rho\frac{\partial\Phi}{\partial r} \\
\displaystyle\frac{P\cot\theta}{r} - \frac{\rho v_r v_\theta}{r}
  - \frac{\rho}{r}\frac{\partial\Phi}{\partial\theta} \\
-\rho\,\mathbf{v}\cdot\nabla\Phi
\end{pmatrix}
\tag{1.7}
$$

Note: the $2P/r$ and $P\cot\theta/r$ terms are geometric consequences of
expressing the divergence in spherical coordinates, not additional physics.

### Poisson equation (self-gravity)

$$
\nabla^2\Phi = 4\pi G\rho
\tag{1.8}
$$

Expanded in axisymmetric spherical coordinates:

$$
\frac{1}{r^2}\frac{\partial}{\partial r}\!\left(r^2\frac{\partial\Phi}{\partial r}\right)
+ \frac{1}{r^2\sin\theta}\frac{\partial}{\partial\theta}\!\left(\sin\theta\,\frac{\partial\Phi}{\partial\theta}\right)
= 4\pi G\rho
\tag{1.9}
$$

---

## §2 Finite Volume Discretization

Structured grid with $N_r \times N_\theta$ cells. Indices: $i = 0,\ldots,N_r-1$
(radial), $j = 0,\ldots,N_\theta-1$ (polar). Cell interfaces at
$r_{i+1/2}$, $\theta_{j+1/2}$.

### Logarithmic radial mesh

$$
r_{i+1/2} = R_{\mathrm{outer}}\left(\frac{i}{N_r}\right)^\alpha
\tag{2.1}
$$

### Cell volume (azimuthal $2\pi$ factor omitted; cancels in all FV updates)

$$
V_{ij} = \frac{r_{i+1/2}^3 - r_{i-1/2}^3}{3}
         \bigl(\cos\theta_{j-1/2} - \cos\theta_{j+1/2}\bigr)
\tag{2.2}
$$

### Radial face area

$$
A^r_{i+1/2,\,j} = r_{i+1/2}^2
  \bigl(\cos\theta_{j-1/2} - \cos\theta_{j+1/2}\bigr)
\tag{2.3}
$$

### Theta face area

$$
A^\theta_{i,\,j+1/2} = \sin\theta_{j+1/2}\;
  \frac{r_{i+1/2}^2 - r_{i-1/2}^2}{2}
\tag{2.4}
$$

### Semi-discrete finite volume update

$$
\frac{d\mathbf{U}_{ij}}{dt}
= -\frac{1}{V_{ij}}\Bigl[
    A^r_{i+1/2,j}\,\hat{\mathbf{F}}_{i+1/2,j}
  - A^r_{i-1/2,j}\,\hat{\mathbf{F}}_{i-1/2,j}
  + A^\theta_{i,j+1/2}\,\hat{\mathbf{G}}_{i,j+1/2}
  - A^\theta_{i,j-1/2}\,\hat{\mathbf{G}}_{i,j-1/2}
\Bigr]
+ \mathbf{S}_{ij}
\tag{2.5}
$$

---

## §3 MUSCL Reconstruction

### Minmod limiter

$$
\mathrm{minmod}(a, b) =
\begin{cases}
\mathrm{sgn}(a)\min(|a|,|b|) & \text{if } ab > 0 \\
0 & \text{otherwise}
\end{cases}
\tag{3.1}
$$

### Van Leer limiter

$$
\mathrm{vanleer}(a, b) =
\begin{cases}
\dfrac{2ab}{a + b} & \text{if } ab > 0 \\
0 & \text{otherwise}
\end{cases}
\tag{3.2}
$$

### Limited slope at cell $i$

$$
\sigma_i = \mathrm{limiter}\bigl(w_i - w_{i-1},\; w_{i+1} - w_i\bigr)
\tag{3.3}
$$

### Reconstructed left state at face $i+1/2$

$$
w^L_{i+1/2} = w_i + \tfrac{1}{2}\,\sigma_i
\tag{3.4}
$$

### Reconstructed right state at face $i+1/2$

$$
w^R_{i+1/2} = w_{i+1} - \tfrac{1}{2}\,\sigma_{i+1}
\tag{3.5}
$$

---

## §4 HLLC Riemann Solver

Reference: Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics",
3rd ed., §10.4.

### Wave speed estimates

$$
S_L = \min(u_L - c_L,\; u_R - c_R), \qquad
S_R = \max(u_L + c_L,\; u_R + c_R)
\tag{4.1}
$$

### Contact wave speed

$$
S^* = \frac{P_R - P_L + \rho_L u_L(S_L - u_L) - \rho_R u_R(S_R - u_R)}
           {\rho_L(S_L - u_L) - \rho_R(S_R - u_R)}
\tag{4.2}
$$

### Total energy density

$$
\mathcal{E} = \frac{P}{\gamma - 1} + \frac{1}{2}\rho(u^2 + v_t^2)
\tag{4.3}
$$

### Physical flux (in the normal direction $u$, tangential $v_t$)

$$
\mathbf{F} = \begin{pmatrix}
\rho u \\ \rho u^2 + P \\ \rho u\,v_t \\ (\mathcal{E}+P)\,u
\end{pmatrix}
\tag{4.4}
$$

For radial faces: $u = v_r$, $v_t = v_\theta$.
For theta faces: $u = v_\theta$, $v_t = v_r$.

### HLLC star-state density

$$
\rho^*_K = \rho_K \frac{S_K - u_K}{S_K - S^*}
\tag{4.5}
$$

### HLLC star-state energy

$$
\mathcal{E}^*_K = \rho^*_K \left[
  \frac{\mathcal{E}_K}{\rho_K}
  + (S^* - u_K)\!\left(S^* + \frac{P_K}{\rho_K(S_K - u_K)}\right)
\right]
\tag{4.6}
$$

### HLLC numerical flux

$$
\hat{\mathbf{F}} =
\begin{cases}
\mathbf{F}_L & \text{if } S_L \geq 0 \\
\mathbf{F}_L + S_L(\mathbf{U}^*_L - \mathbf{U}_L) & \text{if } S_L < 0 \leq S^* \\
\mathbf{F}_R + S_R(\mathbf{U}^*_R - \mathbf{U}_R) & \text{if } S^* < 0 \leq S_R \\
\mathbf{F}_R & \text{if } S_R < 0
\end{cases}
\tag{4.7}
$$

---

## §5 Geometric Source Terms

These arise from expanding $\nabla\cdot$ in spherical coordinates. They must be
discretized in a **volume-consistent** manner to achieve well-balanced
equilibrium (see §5.3).

### Radial momentum geometric source (continuous)

$$
S^{\mathrm{geom}}_{m_r} = \frac{\rho v_\theta^2}{r} + \frac{2P}{r}
\tag{5.1}
$$

### Theta momentum geometric source (continuous)

$$
S^{\mathrm{geom}}_{m_\theta} = \frac{P\cot\theta}{r} - \frac{\rho v_r v_\theta}{r}
\tag{5.2}
$$

### Volume-consistent discrete geometric source (well-balanced)

The continuous $2P/r$ source must be discretized consistently with the FV
divergence operator. We replace the point-value $2P/r$ with its volume average:

$$
\left\langle\frac{2P}{r}\right\rangle_{ij}
= \frac{1}{V_{ij}} \int_{\text{cell}} \frac{2P}{r}\, r^2 \sin\theta\, dr\, d\theta
= \frac{P_{ij}}{V_{ij}}
  \bigl(r_{i+1/2}^2 - r_{i-1/2}^2\bigr)
  \bigl(\cos\theta_{j-1/2} - \cos\theta_{j+1/2}\bigr)
\tag{5.3}
$$

(assuming $P$ approximately uniform within the cell).

Similarly, the $P\cot\theta/r$ source:

$$
\left\langle\frac{P\cot\theta}{r}\right\rangle_{ij}
= \frac{P_{ij}}{V_{ij}}
  \frac{r_{i+1/2}^2 - r_{i-1/2}^2}{2}
  \bigl(\sin\theta_{j-1/2} - \sin\theta_{j+1/2}\bigr)
\tag{5.4}
$$

(derived from $\int \cos\theta\, d\theta = \sin\theta$).

---

## §6 Gravity

### Gravity source — radial momentum

$$
S^g_{m_r} = -\rho\,\frac{\partial\Phi}{\partial r}
\tag{6.1}
$$

### Gravity source — theta momentum

$$
S^g_{m_\theta} = -\frac{\rho}{r}\,\frac{\partial\Phi}{\partial\theta}
\tag{6.2}
$$

### Gravity source — energy

$$
S^g_E = -\rho\left(v_r\frac{\partial\Phi}{\partial r}
  + \frac{v_\theta}{r}\frac{\partial\Phi}{\partial\theta}\right)
\tag{6.3}
$$

### Discrete Laplacian — radial part

$$
L^r_{ij} = \frac{1}{r_i^2}\,
\frac{r_{i+1/2}^2\,\dfrac{\Phi_{i+1,j}-\Phi_{ij}}{\delta r^+_i}
    - r_{i-1/2}^2\,\dfrac{\Phi_{ij}-\Phi_{i-1,j}}{\delta r^-_i}}
     {\Delta r_i}
\tag{6.4}
$$

where $\delta r^+_i = r^c_{i+1} - r^c_i$, $\delta r^-_i = r^c_i - r^c_{i-1}$,
$\Delta r_i = r^f_{i+1} - r^f_i$ (face-to-face width).

### Discrete Laplacian — theta part

$$
L^\theta_{ij} = \frac{1}{r_i^2 \sin\theta_j}\,
\frac{\sin\theta_{j+1/2}\,\dfrac{\Phi_{i,j+1}-\Phi_{ij}}{\delta\theta^+_j}
    - \sin\theta_{j-1/2}\,\dfrac{\Phi_{ij}-\Phi_{i,j-1}}{\delta\theta^-_j}}
     {\Delta\theta_j}
\tag{6.5}
$$

where $\delta\theta^\pm_j$ are center-to-center distances, $\Delta\theta_j$ is
face-to-face width.

### Poisson boundary conditions

Dirichlet at outer boundary (monopole approximation):

$$
\Phi(R_\mathrm{outer}) = -\frac{G\,M_\mathrm{total}}{R_\mathrm{outer}}
\tag{6.6}
$$

Neumann at center ($r = 0$, symmetry):

$$
\left.\frac{\partial\Phi}{\partial r}\right|_{r=0} = 0
\tag{6.7}
$$

Neumann at poles ($\theta = 0, \pi$, axial symmetry):

$$
\left.\frac{\partial\Phi}{\partial\theta}\right|_{\theta=0,\pi} = 0
\tag{6.8}
$$

### Gravity gradient at cell center (face-based interpolation)

Face gradients:

$$
\left.\frac{\partial\Phi}{\partial r}\right|_{i+1/2}
= \frac{\Phi_{i+1,j} - \Phi_{ij}}{r^c_{i+1} - r^c_i}
\tag{6.9a}
$$

Cell-center gradient by weighted average of flanking face gradients:

$$
\left.\frac{\partial\Phi}{\partial r}\right|_i
= \frac{\delta r^+_i \cdot g_{i-1/2} + \delta r^-_i \cdot g_{i+1/2}}
       {\delta r^+_i + \delta r^-_i}
\tag{6.9b}
$$

At $i=0$: use $g_{i-1/2}=0$ (Neumann, Eq. 6.7). At $i = N_r - 1$: use the
Dirichlet value (Eq. 6.6) to form $g_{i+1/2}$.

Theta gradient analogously:

$$
\left.\frac{\partial\Phi}{\partial\theta}\right|_{j+1/2}
= \frac{\Phi_{i,j+1} - \Phi_{ij}}{\theta^c_{j+1} - \theta^c_j}
\tag{6.10a}
$$

$$
\left.\frac{\partial\Phi}{\partial\theta}\right|_j
= \frac{\delta\theta^+_j \cdot g^\theta_{j-1/2}
      + \delta\theta^-_j \cdot g^\theta_{j+1/2}}
       {\delta\theta^+_j + \delta\theta^-_j}
\tag{6.10b}
$$

At $j=0$ and $j=N_\theta-1$: use $g^\theta = 0$ (Neumann, Eq. 6.8).

---

## §7 Time Integration

### RK2 — Heun's method

Stage 1:

$$
\mathbf{U}^* = \mathbf{U}^n + \Delta t\,\mathbf{R}(\mathbf{U}^n)
\tag{7.1}
$$

Stage 2:

$$
\mathbf{U}^{**} = \mathbf{U}^* + \Delta t\,\mathbf{R}(\mathbf{U}^*)
\tag{7.2}
$$

Average:

$$
\mathbf{U}^{n+1} = \tfrac{1}{2}\bigl(\mathbf{U}^n + \mathbf{U}^{**}\bigr)
\tag{7.3}
$$

where $\mathbf{R}(\mathbf{U})$ includes flux divergence, geometric source, and
gravity source.

### CFL condition

$$
\Delta t = C_\mathrm{CFL} \cdot \min_{i,j}\left[
  \min\!\left(
    \frac{\Delta r_i}{|v_r| + c_s},\;
    \frac{r_i\,\Delta\theta_j}{|v_\theta| + c_s}
  \right)
\right]
\tag{7.4}
$$

---

## §8 Boundary Conditions

### Inner boundary ($r = 0$): reflecting symmetry

For ghost cell at layer $g$ ($g = 1, 2, \ldots$):

$$
\rho_{-g,j} = \rho_{g-1,j}, \quad
(\rho v_r)_{-g,j} = -(\rho v_r)_{g-1,j}, \quad
(\rho v_\theta)_{-g,j} = (\rho v_\theta)_{g-1,j}, \quad
(\rho E)_{-g,j} = (\rho E)_{g-1,j}
\tag{8.1}
$$

### Outer boundary ($r = R$): zero-gradient outflow

$$
\mathbf{U}_{N_r+g,\,j} = \mathbf{U}_{N_r-1,\,j}
\tag{8.2}
$$

### Axis boundaries ($\theta = 0$ and $\theta = \pi$): reflecting symmetry in $v_\theta$

For $\theta = 0$ (north pole), ghost layer $g$:

$$
\rho_{i,-g} = \rho_{i,g-1}, \quad
(\rho v_r)_{i,-g} = (\rho v_r)_{i,g-1}, \quad
(\rho v_\theta)_{i,-g} = -(\rho v_\theta)_{i,g-1}, \quad
(\rho E)_{i,-g} = (\rho E)_{i,g-1}
\tag{8.3}
$$

South pole ($\theta = \pi$) identical with indices mirrored from $j = N_\theta - 1$.

---

## §9 Initial Conditions

### Lane-Emden equation

$$
\frac{1}{\xi^2}\frac{d}{d\xi}\!\left(\xi^2\frac{d\Theta}{d\xi}\right) + \Theta^n = 0,
\qquad \Theta(0) = 1, \quad \Theta'(0) = 0
\tag{9.1}
$$

### Lane-Emden length scale

$$
\alpha^2 = \frac{(n+1)\,K\,\rho_c^{1/n - 1}}{4\pi G}
\tag{9.2}
$$

### Stellar radius

$$
R_\star = \alpha\,\xi_1
\tag{9.3}
$$

where $\xi_1$ is the first zero of $\Theta(\xi)$.

### Density from Lane-Emden solution

$$
\rho(r) = \rho_c\,\Theta\!\left(\frac{r}{\alpha}\right)^n
\tag{9.4}
$$

### Polytropic pressure

$$
P = K\,\rho^{1 + 1/n}
\tag{9.5}
$$

### Sedov blast energy deposition

$$
P_\mathrm{blast} = (\gamma - 1)\,\rho_0\,\frac{E_\mathrm{blast}}{V_\mathrm{blast}}
\tag{9.6}
$$

where $V_\mathrm{blast}$ uses the same volume convention as the code (no $2\pi$
azimuthal factor).

### Evrard collapse density profile

$$
\rho(r) = \frac{M}{2\pi R^3}\,\frac{1}{r}, \qquad r < R
\tag{9.7}
$$

### Jeans instability perturbation

$$
\rho = \rho_0\bigl[1 + \epsilon\cos(k_r r)\cos(k_\theta\theta)\bigr]
\tag{9.8}
$$

---

## §10 Low-Mach Implicit Solver (JFNK)

The low-Mach solver uses Backward Euler with Jacobian-Free Newton-Krylov
(JFNK) to take time steps unconstrained by the acoustic CFL.

### Backward Euler nonlinear system

$$
F(U^{n+1}) = \frac{U^{n+1} - U^n}{\Delta t} - R(U^{n+1}) = 0
\tag{10.1}
$$

where $R(U)$ is the spatial residual (§1, §5, §6) using 1st-order upwind
advection and central-difference pressure/gravity gradients.

### Newton iteration

$$
J_k\,\delta U = -F(U^k), \qquad U^{k+1} = U^k + \alpha\,\delta U
\tag{10.2}
$$

with backtracking line search on the scaled merit function $\|L^{-1}F\|_2$.

### Jacobian-free matrix-vector product

$$
J\,v \approx \frac{F(U^k + \varepsilon v) - F(U^k)}{\varepsilon},
\qquad \varepsilon = \sqrt{\epsilon_{\mathrm{mach}}}\,\frac{1 + \|U\|}{\|v\|}
\tag{10.3}
$$

### FGMRES linear solver

Right-preconditioned FGMRES(120) with Eisenstat-Walker adaptive forcing
(initial $\eta = 10^{-2}$, EW Choice 2).

### 1D radial gravity

Gravity is computed from angle-averaged density via cumulative mass integral
(no Poisson solve, no GMG noise):

$$
g_r(r_i) = -\frac{G\,M(<r_i)}{r_i^2}, \qquad
M(<r_i) = \sum_{k<i} \bar{\rho}_k\,V_k
\tag{10.4}
$$

where $\bar{\rho}_k$ is the $\theta$-averaged density at shell $k$.

### Well-balanced residual (reference-state subtraction)

The momentum residual uses perturbation form to achieve $R(U_{\mathrm{HSE}}) = 0$
to machine precision:

$$
F_r = -\nabla P' + \rho'\,g_0(r) + \rho\,g'(r) + S_{\mathrm{geom}}
\tag{10.5}
$$

where primed quantities are deviations from the HSE reference state.

### Atmosphere treatment

Cells with $\rho_0 < 10^{-6}\,\rho_{\max}$ are treated as atmosphere:
$R = 0$ in the residual kernel, so Newton does not evolve them.

---

## §11 Physics-Based Preconditioner (PBP)

The JFNK preconditioner $M \approx J$ uses the block structure of the
Jacobian to incorporate a pressure Poisson solve, enabling GMRES convergence
in $O(10)$ iterations regardless of $\Delta t$.

### Jacobian block structure

The 4-DOF Jacobian has the block form:

$$
J = \begin{pmatrix}
A_\rho & 0 & 0 & 0 \\
g_0 & A_{vr} & 0 & B_r \\
0 & 0 & A_{v\theta} & B_\theta \\
0 & C_r & C_\theta & A_E
\end{pmatrix}
\tag{11.1}
$$

where $A_v = -(1/\Delta t + \text{advection rate})$ (diagonal-dominant),
$B = -(\gamma-1)\nabla$ (pressure gradient), and $C = -P\,\nabla\cdot(1/\rho)$
(compression work).

### Schur complement

Eliminating velocity from the energy equation gives the pressure Schur
complement:

$$
S = A_E - C\,A_v^{-1}\,B \approx \nabla\cdot\!\left(\frac{\gamma-1}{A_p}\nabla\right)
\tag{11.2}
$$

which is a variable-coefficient Poisson operator, solved by geometric
multigrid (GMG).

### PBP preconditioner application ($y = M^{-1}x$)

Given input $x = (x_\rho, x_{mr}, x_{m\theta}, x_E)$:

**Step 1 — Momentum predict** (2-DOF block-tridiagonal r-line solve):

$$
\tilde{v}_r, \tilde{v}_\theta = A_v^{-1}\,(x_{mr}, x_{m\theta}) / \rho
\tag{11.3}
$$

**Step 2 — Pressure Poisson** (GMG, 3 V-cycles):

$$
\nabla\cdot\!\left(\frac{1}{A_p}\nabla\,\delta p\right) = \nabla\cdot\tilde{v}
\tag{11.4}
$$

**Step 3 — Velocity correct and assemble**:

$$
y_{mr} = \rho\left(\tilde{v}_r - \frac{1}{A_p}\frac{\partial\,\delta p}{\partial r}\right),
\qquad
y_{m\theta} = \rho\left(\tilde{v}_\theta - \frac{1}{A_p}\frac{1}{r}\frac{\partial\,\delta p}{\partial\theta}\right)
\tag{11.5}
$$

$$
y_\rho = (J^{-1})_{00}\,x_\rho, \qquad y_E = (J^{-1})_{33}\,x_E
\tag{11.6}
$$

where $(J^{-1})_{00}$ and $(J^{-1})_{33}$ are diagonal elements of the
block-Jacobi inverse (point Jacobi on $\rho$ and energy).

### Why this works

The preconditioner runs one cycle of the projection method *inside* the
Krylov solver. The GMG pressure Poisson directly resolves the global
pressure coupling that GMRES alone would need $O(N)$ iterations to discover.
Because this is only a preconditioner, approximation errors do not affect
the final solution — the outer JFNK iteration converges to the exact
Backward Euler solution regardless of preconditioner quality.

---

## §12 FAS Nonlinear Multigrid (Polar Grid)

The FAS (Full Approximation Scheme) solver operates on the same polar grid as
the explicit solver (§1–§8), using nonlinear multigrid with block-Jacobi
smoothing.

### Well-balanced flux-level P₀ subtraction

The momentum flux divergence subtracts background pressure at each face to
achieve $R(U_\mathrm{HSE}) = 0$ to machine precision. For regular cells
($i \geq 1$):

$$
(\nabla\cdot F)_{m_r} = -\frac{1}{V_{ij}}\Bigl[
  A^r_{i+1/2}\bigl(\hat F^{m_r}_{i+1/2} - P^0_{i+1/2}\bigr)
- A^r_{i-1/2}\bigl(\hat F^{m_r}_{i-1/2} - P^0_{i-1/2}\bigr)
+ A^\theta_{j+1/2}\hat G^{m_r}_{j+1/2}
- A^\theta_{j-1/2}\hat G^{m_r}_{j-1/2}
\Bigr]
\tag{12.1}
$$

where $P^0_{i+1/2} = \tfrac{1}{2}(P_0(r_i) + P_0(r_{i+1}))$ is the
face-averaged HSE pressure. The theta momentum equation subtracts $P^0$
from the $\theta$-face flux analogously.

### Well-balanced gravity source (perturbation split)

$$
S^g_{m_r} = \rho'\,g_{0,r} + \rho\,g'_r,
\qquad
\rho' = \rho - \rho_0, \quad g' = g - g_0
\tag{12.2}
$$

where $\rho_0, g_0$ are the HSE reference state. At equilibrium,
$\rho' = 0$ and $g' = 0$, so the source vanishes exactly.

### HSE defect subtraction

On coarse grids the discrete flux-level WB is not exact due to averaging.
The solver precomputes and subtracts the frozen defect:

$$
R_\mathrm{corrected}(U) = R(U) - R(U_0)
\tag{12.3}
$$

ensuring $R_\mathrm{corrected}(U_0) = 0$ on every grid level.

### Origin cell ($i = 0$)

The innermost cell is a pie-slice wedge touching $r = 0$. It has no inner
radial face (zero area). Geometry:

$$
V_{0j} = \tfrac{1}{3}\,r_{1/2}^3\,(\cos\theta_{j-1/2} - \cos\theta_{j+1/2}),
\qquad
A^r_{1/2,j} = r_{1/2}^2\,(\cos\theta_{j-1/2} - \cos\theta_{j+1/2})
\tag{12.4}
$$

Volume-averaged $\langle 1/r \rangle$ for geometric source:

$$
\left\langle \frac{1}{r} \right\rangle_{0j} = \frac{3}{2\,r_{1/2}}
\tag{12.5}
$$

First-order HLLC is used at the origin (stencil too short for MUSCL).
WB P₀ subtraction applies to the single outer radial face:

$$
(\nabla\cdot F)_{m_r}\big|_{i=0} = -\frac{1}{V_{0j}}
  \Bigl[A^r_{1/2}\bigl(\hat F^{m_r}_{1/2} - P^0_{1/2}\bigr)
  + A^\theta_{j+1/2}\hat G^{m_r}_{j+1/2}
  - A^\theta_{j-1/2}\hat G^{m_r}_{j-1/2}\Bigr]
\tag{12.6}
$$

### MUSCL reconstruction (perturbation form)

Slopes are computed on perturbation variables, then face states are
restored using the face-centered HSE background:

$$
\sigma^\rho_i = \mathrm{limiter}\bigl(\rho'_i - \rho'_{i-1},\;
  \rho'_{i+1} - \rho'_i\bigr)
\tag{12.7a}
$$

$$
\rho^L_{i+1/2} = \rho_0(r_{i+1/2}) + \rho'_i + \tfrac{1}{2}\sigma^\rho_i,
\qquad
\rho^R_{i+1/2} = \rho_0(r_{i+1/2}) + \rho'_{i+1} - \tfrac{1}{2}\sigma^\rho_{i+1}
\tag{12.7b}
$$

where $\rho_0(r_{i+1/2}) = \tfrac{1}{2}(\rho_{0,i} + \rho_{0,i+1})$ is
the face-centered background. Pressure is treated identically.

### FAS restrict (volume-weighted)

Fine-to-coarse state transfer with 2×2 coarsening:

$$
U^H_{I,J} = \frac{\sum_{(i,j)\in\mathcal{C}(I,J)} U^h_{ij}\,V^h_{ij}}
                  {\sum_{(i,j)\in\mathcal{C}(I,J)} V^h_{ij}}
\tag{12.8}
$$

where $\mathcal{C}(I,J)$ denotes the 2×2 fine children of coarse cell
$(I,J)$.

### FAS tau correction

The coarse-level FAS right-hand side incorporates the restricted
fine-level defect so that the coarse solve drives the fine-level residual
to zero:

$$
\tau_H = \frac{U_H}{\Delta t} - R(U_H) + \bar{d}_H
\tag{12.9}
$$

where $\bar{d}_H$ is the restricted fine defect (Eq. 12.8 applied to
$F(U^h) = R(U^h) - U^h/\Delta t + \tau^h$).

### FAS prolongate (limited injection)

Coarse correction is piecewise-constant, clamped for safety:

$$
\delta U_H = U_H^{\mathrm{solved}} - U_H^{\mathrm{restricted}}
\tag{12.10a}
$$

$$
U^h_{ij} \mathrel{+}= \mathrm{clamp}(\delta U_H,\;
  -\beta\,s_{ij},\; +\beta\,s_{ij}),
\qquad \beta = 0.5
\tag{12.10b}
$$

where $s_{ij}$ is a local scale ($\rho$, $\rho c_s$, or $\rho E$
depending on the equation).

### Block-Jacobi smoother

$$
U^{k+1} = U^k - \omega\,J_\mathrm{diag}^{-1}\,F(U^k),
\qquad \omega = 0.8
\tag{12.11}
$$

The $4\times4$ diagonal Jacobian block at each cell is:

$$
J_\mathrm{diag} = -\Bigl(\frac{1}{\Delta t} + s_r\Bigr)I
  + \text{off-diagonal couplings}
\tag{12.12}
$$

where $s_r = 2\bigl[(|v_r|+c_s)/\Delta r + (|v_\theta|+c_s)/(r\Delta\theta)\bigr]$
is the spectral radius estimate (factor 2 for MUSCL doubling).
Off-diagonal entries capture pressure–energy coupling
($\partial(\nabla P)/\partial(\rho E)$) and gravity–density coupling
($\partial S^g/\partial\rho$). The block is inverted per cell via
4×4 Gauss–Jordan elimination.

### Smooth floor

Density and internal energy are kept non-negative by a $C^\infty$ smooth
approximation to $\max(x,\varepsilon)$:

$$
\mathrm{floor}(x) = \tfrac{1}{2}\bigl(x + \sqrt{x^2 + 4\varepsilon^2}\bigr),
\qquad \varepsilon = 10^{-20}
\tag{12.13}
$$

Applied first to $\rho$, then to the internal energy
$e_\mathrm{int} = \rho E - \tfrac{1}{2}\rho v^2$.

### CFL condition (polar grid)

$$
\Delta t = C_\mathrm{CFL}\;\min_{i,j}\left[\min\!\left(
  \frac{\Delta r_i}{|v_r|+c_s},\;
  \frac{r_i\,\Delta\theta_j}{|v_\theta|+c_s}
\right)\right]
\tag{12.14}
$$

Atmosphere cells ($\rho_0 < 10^{-6}\rho_{\max}$) are excluded from the
CFL computation.

---

## §13 Strang Cartesian Solver

The Strang solver operates on a uniform Cartesian grid $(x,y)$,
$[0, L_x]\times[0, L_y]$, with cell sizes $\Delta x = L_x/N_x$,
$\Delta y = L_y/N_y$ and $n_g = 2$ ghost layers.

### Perturbation storage

The solver stores perturbations from a 1-D isentropic HSE background:

$$
\rho'_{ij} = \rho_{ij} - \bar\rho(y_j), \qquad
E'_{ij} = E_{ij} - \frac{\bar p(y_j)}{\gamma - 1}
\tag{13.1}
$$

Momentum $(\rho u, \rho v)$ is stored in total form (background velocity
is zero).

### Isentropic HSE background

$$
\bar\rho(y) = \Bigl[\rho_0^{\gamma-1}
  - \frac{(\gamma-1)\,g\,y}{\gamma\,K}\Bigr]^{1/(\gamma-1)},
\qquad
\bar p(y) = K\,\bar\rho(y)^\gamma
\tag{13.2}
$$

where $\rho_0$ is the base density, $K$ the polytropic constant, $g$
the uniform gravitational acceleration, and $\gamma$ the adiabatic index.

### MC (Monotonised Central) limiter

$$
\mathrm{MC}(a,b) = \begin{cases}
\mathrm{sgn}(a)\;\min\!\bigl(\tfrac{1}{2}|a+b|,\; 2|a|,\; 2|b|\bigr)
  & \text{if } ab > 0 \\
0 & \text{otherwise}
\end{cases}
\tag{13.3}
$$

### Well-balanced y-sweep MUSCL reconstruction

Slopes are computed on perturbation quantities; face states are restored
using face-centred HSE values:

$$
\sigma^\rho_j = \mathrm{MC}(\rho'_j - \rho'_{j-1},\;
  \rho'_{j+1} - \rho'_j)
\tag{13.4a}
$$

$$
\sigma^P_j = \mathrm{MC}(P'_j - P'_{j-1},\;
  P'_{j+1} - P'_j), \qquad
P'_j = P_j - \bar p(y_j)
\tag{13.4b}
$$

$$
\rho^L_{j+1/2} = \bar\rho(y_{j+1/2}) + \rho'_j + \tfrac{1}{2}\sigma^\rho_j
\tag{13.4c}
$$

$$
P^L_{j+1/2} = \bar p(y_{j+1/2}) + P'_j + \tfrac{1}{2}\sigma^P_j
\tag{13.4d}
$$

The x-sweep uses no WB (background is y-dependent only); slopes and
reconstruction are on total variables.

### MUSCL-Hancock predictor

Both left and right face states of a cell are evolved by half a time step
using the within-cell flux difference:

$$
\mathbf{U}^{n+1/2}_\mathrm{face} =
  \mathbf{U}^n_\mathrm{face}
  + \frac{\Delta t}{2\Delta x}
    \bigl[\mathbf{F}(\mathbf{W}_L) - \mathbf{F}(\mathbf{W}_R)\bigr]
\tag{13.5}
$$

where $\mathbf{W}_L, \mathbf{W}_R$ are the MUSCL-reconstructed left and
right primitives of the same cell. In the y-sweep, the gravity source
$-\rho g$ is included in the Hancock half-step to balance
$\partial\bar p/\partial y$:

$$
(\rho v)^{n+1/2}_\mathrm{face} \mathrel{+}= -\tfrac{1}{2}\Delta t\,\rho\,g
\tag{13.6}
$$

### Strang splitting

$$
\mathbf{U}^{n+1} = \mathcal{L}_x(\tfrac{\Delta t}{2})
  \circ \mathcal{L}_y(\tfrac{\Delta t}{2})
  \circ \mathcal{L}_y(\tfrac{\Delta t}{2})
  \circ \mathcal{L}_x(\tfrac{\Delta t}{2})\;\mathbf{U}^n
\tag{13.7}
$$

Each $\mathcal{L}$ is a MUSCL-Hancock + LM-HLLC unsplit sweep in the
corresponding direction.

### Boundary conditions (Cartesian)

**x-periodic** ($n_g$ ghost layers):

$$
U_{-g,\,j} = U_{N_x-g,\,j}, \qquad
U_{N_x+g,\,j} = U_{g,\,j}
\tag{13.8a}
$$

**y-bottom reflective** ($y = 0$):

$$
\rho_{i,-g} = \rho_{i,g-1}, \quad
(\rho u)_{i,-g} = (\rho u)_{i,g-1}, \quad
(\rho v)_{i,-g} = -(\rho v)_{i,g-1}, \quad
E_{i,-g} = E_{i,g-1}
\tag{13.8b}
$$

**y-top outflow** (zero-gradient):

$$
U_{i,\,N_y+g} = U_{i,\,N_y-1}
\tag{13.8c}
$$

### CFL condition (Cartesian)

$$
\Delta t = C_\mathrm{CFL}\;\min_{i,j}\left[\min\!\left(
  \frac{\Delta x}{|u| + c_s},\;
  \frac{\Delta y}{|v| + c_s}
\right)\right]
\tag{13.9}
$$

---

## §14 Low-Mach HLLC (LM-HLLC)

The LM-HLLC Riemann solver modifies the standard HLLC (§4) to reduce
excessive numerical diffusion at low Mach number. The modification scales
the pressure jump in the contact wave speed estimate.

### Local Mach number

$$
M_\mathrm{local} = \frac{|u_L| + |u_R|}{c_L + c_R}
\tag{14.1}
$$

### Blending factor

$$
f(M) = \mathrm{clamp}(M_\mathrm{local},\; M_\mathrm{cutoff},\; 1)
\tag{14.2}
$$

with $M_\mathrm{cutoff} = 10^{-3}$. At $M \gg M_\mathrm{cutoff}$,
$f = 1$ and standard HLLC is recovered. At $M < M_\mathrm{cutoff}$,
$f = M_\mathrm{cutoff}$ and the acoustic dissipation is suppressed.

### Modified contact wave speed

In the Strang solver:

$$
S^* = \frac{f(M)(P_R - P_L) + \rho_L u_L(S_L - u_L) - \rho_R u_R(S_R - u_R)}
           {\rho_L(S_L - u_L) - \rho_R(S_R - u_R)}
\tag{14.3}
$$

In the FAS solver (equivalent, alternative formulation):

$$
P_{L,\mathrm{mod}} = \bar P + \tfrac{1}{2}f(P_L - P_R), \qquad
P_{R,\mathrm{mod}} = \bar P - \tfrac{1}{2}f(P_L - P_R)
\tag{14.4a}
$$

$$
S^* = \frac{P_{R,\mathrm{mod}} - P_{L,\mathrm{mod}}
  + \rho_L u_L(S_L-u_L) - \rho_R u_R(S_R-u_R)}
  {\rho_L(S_L-u_L) - \rho_R(S_R-u_R)}
\tag{14.4b}
$$

where $\bar P = \tfrac{1}{2}(P_L + P_R)$. The two formulations are
algebraically identical.

### Physical interpretation

At low Mach number, the pressure jump across a contact wave is $O(M^2 P)$
while the acoustic pressure fluctuation is $O(M P)$. Standard HLLC treats
both at the same order, creating $O(1/M)$ numerical dissipation. Scaling
by $f(M)$ reduces the effective pressure jump, suppressing the spurious
acoustic mode and allowing buoyancy-driven flows to develop correctly.

Consequence: sound waves with $M < M_\mathrm{cutoff}$ are damped. For
stellar convection ($M \sim 10^{-3}$–$10^{-1}$), this is acceptable
because the dynamics are buoyancy-dominated.

---

## §15 Cartesian Compatible Lagrangian (`cart_lag`)

Staggered quadrilateral mesh on $[0, L_x] \times [0, L_y]$. Cell-centered
thermodynamic variables $(\rho, P, Q, e_\text{int})$; node-centered kinematic
variables $(X, Y, v_x, v_y, F_x, F_y)$. Subcell forces distribute the cell's
pressure + AV work to its 4 corner nodes, then `atomicAdd` to the node
force buffer. Energy is updated from node displacements dotted with subcell
forces (Caramana-Shashkov-Whalen 1998, JCP 144) — **compatible** in the sense
that cell internal energy change equals the work that the cell does on its
4 corner nodes, to the bit.

### Node mass

$$
m_n = \tfrac{1}{4} \sum_{c\ \text{adj}\ n} m_c
\tag{15.1}
$$

Under periodic BC the adjacent-cell wrap uses modular indexing; node
copies at `in = nnx-1` and `jn = nny-1` receive `m_n` equal to their
origin (0-indexed) partner, because they each independently iterate over
the same wrap set. **Diagnostics must skip those duplicates** when
computing $\sum m_n$ to avoid double counting (see P31).

### Cell area via shoelace (for pre-Lagrangian volume)

$$
V_c = \tfrac{1}{2}\left| \sum_{k=0}^{3} X_k Y_{k+1} - X_{k+1} Y_k \right|
\tag{15.2}
$$

indices modulo 4.

### Divergence-consistent strain rate

$$
s = -\frac{1}{V_c} \sum_\text{edges} \tfrac{1}{2}(v_a + v_b) \cdot \hat n_{ab}\,\ell_{ab}
\tag{15.3}
$$

Compression-only (`s > 0`) triggers artificial viscosity; dilation
`s < 0` yields $Q = 0$.

### von Neumann–Richtmyer artificial viscosity

$$
Q = \rho\left( C_\text{quad}\, (sL)^2 + C_\text{lin}\, c_s\, sL \right) \cdot w_\text{shear}
\tag{15.4}
$$

with $L = \sqrt{V_c}$. The shear weight $w_\text{shear} \in [0, 1]$ (default 1)
optionally reduces $Q$ in shear-dominated cells using an estimate of
$\partial u/\partial x$, $\partial v/\partial y$ from corner velocity
differences (`--shear-aware-av`). Defaults $C_\text{quad} = 2$, $C_\text{lin} = 0.5$.

### Subcell edge normal force (per cell, per edge $k$)

$$
\mathbf{a}_k = (P + Q)\,(\Delta y_k,\; -\Delta x_k), \qquad
\Delta x_k = X_{k+1} - X_k,\; \Delta y_k = Y_{k+1} - Y_k
\tag{15.5}
$$

### Corner subcell force (contributes to node via `atomicAdd`)

$$
\mathbf{F}_\text{sub}^{(k)} = \tfrac{1}{2}(\mathbf{a}_{k-1} + \mathbf{a}_k)
\tag{15.6}
$$

indices modulo 4. For uniform $P$ on a Cartesian cell the four corner
subcell forces sum to zero around every interior node — this is the
basis of the uniform-advection invariance check (P30, P31).

### Kick-drift-kick node update

$$
\begin{aligned}
\tfrac{1}{2}\Delta v &= \frac{F_n}{m_n} \tfrac{1}{2}\Delta t \\
\Delta X &= (v_n + \tfrac{1}{2}\Delta v) \Delta t \\
v_n^{n+1} &= v_n^n + \Delta v
\end{aligned}
\tag{15.7}
$$

### Compatible energy update (per cell)

$$
m_c \Delta e_c = -\sum_{k=0}^{3} \mathbf{F}_\text{sub}^{(k)} \cdot \Delta X_k^{(\text{node})}
\tag{15.8}
$$

Energy removed from node KE equals energy deposited in cell IE, modulo machine epsilon.

### Minimum-height CFL

$$
\Delta t = \mathrm{CFL} \cdot \min_c \frac{h_c}{c_s + |\mathbf{v}|} \cdot \min\!\left(1, \frac{1}{f_\text{comp}\, s L / c_s}\right)
\tag{15.9}
$$

where $h_c$ is the minimum perpendicular height in the (possibly deformed)
quadrilateral, and $f_\text{comp}$ is a compression-strain safety fraction
(default 0.25) to avoid grid inversion under strong shocks.

---

## §16 Eulerian Rezone + Swept-Edge Remap (`cart_ale`, `cart_ale2`)

ALE step = Lagrangian substep (§15) → rezone → remap. Rezone: snap every
node back to its uniform Cartesian position $(X_0, Y_0)$. Remap: transfer
cell-centered conserved quantities $\{dm, dm \cdot e_\text{int}, p_x, p_y\}$
across each edge via the **signed swept area** between the Lagrangian-
displaced edge and the original edge.

### Swept-region signed area (east face of cell $(i,j)$)

$$
A_\text{sw} = \tfrac{1}{2} \sum_{k=0}^{3} (X_k Y_{k+1} - X_{k+1} Y_k),\quad
\text{vertices}\ (A_\text{old}, A_\text{new}, B_\text{new}, B_\text{old})
\tag{16.1}
$$

$A_\text{sw} > 0$: sweep moved into cell $R$ (donor = $L$);
$A_\text{sw} < 0$: sweep moved into cell $L$ (donor = $R$).

### First-order donor-cell flux

$$
\Delta f = \frac{f_\text{donor}}{V_\text{donor}}\,|A_\text{sw}|,\quad
f \in \{dm,\ dm \cdot e_\text{int},\ p_x,\ p_y\}
\tag{16.2}
$$

with a safety clamp $|A_\text{sw}|/V_\text{donor} \le 1/2$.

### MUSCL 2nd-order reconstruction (Kucharik–Shashkov 2012)

Per-cell limited slopes on the reference mesh:

$$
\partial_x f_c = \phi\!\left(\frac{f_{c+\hat x} - f_c}{\Delta x},\; \frac{f_c - f_{c-\hat x}}{\Delta x}\right)
\tag{16.3}
$$

same for $\partial_y f_c$. Limiter $\phi$: `minmod`, van Leer (default),
or MC (`--remap-limiter`). Face-centroid evaluation:

$$
\hat f(\xi, \eta) = f_c + \partial_x f_c \cdot (\xi - x_c) + \partial_y f_c \cdot (\eta - y_c)
\tag{16.4}
$$

evaluated at the swept-region centroid $(\xi, \eta) = \tfrac{1}{4}\sum A_k$.

### Remap update

$$
\Delta f_\text{donor} = -\Delta f, \quad \Delta f_\text{acceptor} = +\Delta f
\tag{16.5}
$$

Discrete conservation is **exact** because the same flux value is
subtracted from donor and added to acceptor.

### Periodic BC wrap edge

Under `--bc-x periodic`, the east-face edge $ic = n_x - 1$ is **not** a
wall but a wrap connecting cell $n_x - 1$ with cell $0$. Remap kernels
extend their edge count from $(n_x - 1) \cdot n_y$ to $n_x \cdot n_y$
(and similarly in $y$), with `cR_idx = (ic + 1) \bmod n_x`. The
face-centroid-relative coordinate $s_x = (c_x - x_\text{donor})/\Delta x$
must be wrapped: if $s_x > 0.5$ then $s_x \mathrel{-}= n_x$, else if $s_x
< -0.5$ then $s_x \mathrel{+}= n_x$ (see P30).

### Node velocity rebuild (mass-weighted)

Post-remap node velocity is built from new cell-centered momentum and
mass density:

$$
v_n = \frac{\sum_{c\ \text{adj}\ n} \tfrac{1}{4} p_c}{\sum_{c\ \text{adj}\ n} \tfrac{1}{4} m_c}
\tag{16.6}
$$

Momentum-conservative: $\sum_n m_n v_n = \sum_c p_c$ to machine precision.

---

## §17 Cart_ale2 PPM Reconstruction Variants

Extends §16 to piecewise-parabolic remap (Colella-Woodward 1984) with
optional extremum-preserving limiter (Colella-Sekora 2008), primitive-
variable space, and characteristic-variable projection (Stone et al.
2008 Appendix A, adiabatic hydro branch of `athena/reconstruct/ppm.cpp`).

### PPM 4-point interpolant

$$
f_{i+1/2} = \tfrac{1}{12}\bigl(7(f_i + f_{i+1}) - (f_{i-1} + f_{i+2})\bigr)
\tag{17.1}
$$

Yielding initial left/right face values $f_L = f_{i-1/2}$, $f_R = f_{i+1/2}$.

### Classical CW monotonization

$$
\begin{aligned}
&\text{if } (f_L - f_c)(f_c - f_R) \le 0:\ f_L = f_R = f_c \\
&\text{else if } \Delta f \cdot \Delta_6 > (\Delta f)^2:\ f_L = 3 f_c - 2 f_R \\
&\text{else if } \Delta f \cdot \Delta_6 < -(\Delta f)^2:\ f_R = 3 f_c - 2 f_L
\end{aligned}
\tag{17.2}
$$

with $\Delta f = f_R - f_L$, $\Delta_6 = 6(f_c - \tfrac{1}{2}(f_L + f_R))$.
CW clamps smooth extrema (penalizes KH roll-up).

### Colella-Sekora extremum-preserving limiter

Two stages (`--ppm-limiter cs`, default):

**Stage 1** — face-value correction at smooth extrema, using five-point
second-derivative stencils:
$d^2_{i-1} = f_{i-2} + f_i - 2 f_{i-1}$ and likewise $d^2_i$, $d^2_{i+1}$.
At face $f_{L}$ (i.e. $f_{i-1/2}$), if $(f_L - f_{i-1})(f_i - f_L) < 0$:

$$
f_L \leftarrow \tfrac{1}{2}(f_{i-1} + f_i) - q_d / 6
\tag{17.3}
$$

where $q_a = 3(f_{i-1} + f_i - 2 f_L)$ and

$$
q_d = \begin{cases}
\operatorname{sgn}(q_a)\,\min(C_2 |d^2_{i-1}|, C_2 |d^2_i|, |q_a|) & \text{if signs agree} \\
0 & \text{otherwise}
\end{cases}
\tag{17.4}
$$

with $C_2 = 1.25$. Analogous at $f_R$.

**Stage 2** — parabolic coefficient limiting with smooth-vs-shock
discrimination. Let $d^2_f = 6(f_L + f_R - 2 f_c)$. If signs of $d^2_{i-1},
d^2_i, d^2_{i+1}, d^2_f$ all agree,

$$
q_e = \operatorname{sgn}(d^2_f)\,\min(C_2|d^2_{i-1}|, C_2|d^2_i|, C_2|d^2_{i+1}|, |d^2_f|),\quad
\rho_r = q_e / d^2_f
\tag{17.5}
$$

At a local extremum $(f_c - f_L)(f_R - f_c) \le 0$ and $\rho_r < 1 -
\epsilon$: scale parabola smoothly ($f_L \leftarrow f_c - \rho_r(f_c - f_L)$,
$f_R \leftarrow f_c + \rho_r(f_R - f_c)$). Otherwise apply classical CW
overshoot clamp. The result: smooth peaks survive, true shocks still
clamp.

### Primitive vs conservative variable space

`--ppm-space cons`: reconstruct the 4 conserved densities
$\{\rho, \rho e_\text{int}, \rho v_x, \rho v_y\}$. Unstable on smooth shear
layers where $\rho v_x$ crosses zero (overshoot → negative mass × negative
velocity → negative pressure).

`--ppm-space prim` (default): reconstruct the 4 primitives
$\{\rho, P, v_x, v_y\}$. At swept centroid evaluate primitives then
convert to conserved flux:

$$
\begin{aligned}
\Delta \widehat{dm}  &= \hat\rho\,|A_\text{sw}| \\
\Delta \widehat{p_x} &= \hat\rho\,\hat v_x\,|A_\text{sw}| \\
\Delta \widehat{p_y} &= \hat\rho\,\hat v_y\,|A_\text{sw}| \\
\Delta \widehat{ie}  &= \hat P\,|A_\text{sw}| / (\gamma - 1)
\end{aligned}
\tag{17.6}
$$

The same flux value is subtracted from donor and added to acceptor,
preserving discrete conservation exactly.

### Characteristic projection (default on, `--ppm-char`)

Before PPM reconstruction, project the 5-point primitive stencil into
local characteristic variables using cell-$i$'s $(\rho_i, a_i)$ where
$a_i = \sqrt{\gamma P_i / \rho_i}$. For x-direction sweep:

$$
\begin{aligned}
w_0 &= \tfrac{1}{2}(P/a^2 - \rho_i\, v_x / a_i)      & \text{(left acoustic)}\\
w_1 &= \rho - P/a^2                                   & \text{(entropy)}\\
w_2 &= v_y                                            & \text{(shear)}\\
w_3 &= \tfrac{1}{2}(P/a^2 + \rho_i\, v_x / a_i)      & \text{(right acoustic)}
\end{aligned}
\tag{17.7}
$$

Apply PPM + CS limiter to each $w_n$ independently. Project back using
the same $(\rho_i, a_i)$ basis:

$$
\begin{aligned}
\rho_f  &= w_0^f + w_1^f + w_3^f \\
v_x^f   &= a_i(w_3^f - w_0^f)/\rho_i \\
v_y^f   &= w_2^f \\
P_f     &= a_i^2(w_0^f + w_3^f)
\end{aligned}
\tag{17.8}
$$

For y-direction sweep, swap the role of $v_x$ and $v_y$ in the projection
(i.e. pass $v_y$ where $v_x$ appears in Eq. 17.7 and vice versa), then
unpack with the same swap.

Rationale: in a smooth shear layer, acoustic and shear modes are
independent — projecting and limiting each mode separately prevents
shear gradients from triggering pressure overshoots. Mirrors Athena
`ppm.cpp:99-107` + `characteristic.cpp:235-256, 470-492`.

### KH benchmark (reality check)

The stack (17.1)–(17.8) keeps the Lecoanet canonical KH (Athena iprob=4
geometry, $k = 1$, $v_\text{flow} = 1$, $P_0 = 10$) stable to $t = 5$ at
256×512, with IE conserved to 10 digits and total $E$ drift ~1.2%.
**However**: the kinetic-energy spectrum at $k = 7$, 512×1024 falls as
$\sim k^{-10}$ past the injection scale, not $k^{-3}$. The effective
Reynolds number of this ALE stack is bounded by the Caramana subcell
force dissipation, which is tens of times stronger than an HLLC
face flux. `cart_ale2` is therefore a **compressible stellar
convection / PdV pulsation** tool, not a 2D turbulence benchmark
tool. For Kraichnan cascade studies use the `pseudo_spectral` branch.

---

## §18 Pseudo-Spectral Incompressible NS (`pseudo_spectral`)

2D doubly-periodic Navier--Stokes in vorticity--streamfunction form, solved
on a uniform $(N_x, N_y)$ grid by cuFFT R2C/C2R with Orszag skew-symmetric
convection + IFRK3 integrating-factor viscosity + circular 2/3 dealiasing.
See `docs/pseudo_spectral_design_2026-05-01.md` for benchmarking and
background.

### Vorticity--streamfunction primitives

$$
\frac{\partial \omega}{\partial t} + \mathbf{u}\cdot\nabla\omega
  = \nu\,\nabla^{2}\omega,
\qquad
\nabla^{2}\psi = -\omega,
\qquad
u = \partial_y\psi,\;\; v = -\partial_x\psi.
\tag{18.1}
$$

In spectral space the Poisson solve is diagonal:
$\hat\psi(\mathbf{k}) = \hat\omega(\mathbf{k})/|\mathbf{k}|^{2}$ (with the
$\mathbf{k}=0$ mode set to zero to fix the constant gauge).

### Circular 2/3 dealiasing

Let $k_{\max} = \min(N_x, N_y)\pi / L$ (assuming $L_x = L_y = L$).  The
dealias mask zeroes out all modes with $|\mathbf{k}| > \tfrac{2}{3}k_{\max}$.

$$
\hat\omega(\mathbf{k}) \gets
\begin{cases}
\hat\omega(\mathbf{k}) & |\mathbf{k}| \le \tfrac{2}{3}k_{\max}\\
0 & \text{otherwise}
\end{cases}
\tag{18.2}
$$

A circular rather than axis-aligned truncation avoids anisotropic
under-resolution in the corners of $\mathbf{k}$-space.

### Orszag skew-symmetric convection

Two mathematically equivalent forms of $\mathbf{u}\cdot\nabla\omega$:

$$
N_A(\omega) \equiv \mathbf{u}\cdot\nabla\omega,
\qquad
N_C(\omega) \equiv \nabla\cdot(\mathbf{u}\,\omega)
\tag{18.3}
$$

agree for $\nabla\cdot\mathbf{u}=0$ in the continuum.  After FFT truncation
the two discretisations differ by $\mathcal{O}(\Delta k)$ terms that act
as numerical dissipation.  Orszag (1971) shows the symmetrised form
$N_S = \tfrac{1}{2}(N_A + N_C)$ is an anti-symmetric operator in the
truncated spectral representation, so it exactly conserves both enstrophy
$\sum |\hat\omega|^{2}$ and kinetic energy $\sum |\hat\omega|^{2}/|\mathbf{k}|^{2}$
at the discrete level.

$$
\hat N_S(\mathbf{k}) = \tfrac{1}{2}\bigl[\hat N_A(\mathbf{k}) + \hat N_C(\mathbf{k})\bigr]
\tag{18.4}
$$

CLI flag `--ps-adv-only` substitutes the advective-only form $\hat N_A$
for reference.

### Integrating-factor RK3 (IFRK3)

Substitute $\hat\omega(\mathbf{k}, t) = e^{-\nu|\mathbf{k}|^{2} t}\,
\hat w(\mathbf{k}, t)$.  The viscous term is absorbed exactly; the evolution
equation for $\hat w$ reads

$$
\frac{\dd \hat w}{\dd t} = e^{+\nu|\mathbf{k}|^{2} t}\,
\bigl[-\hat N_S(\hat\omega)\bigr]
\tag{18.5}
$$

Integrated by Shu--Osher SSP-RK3 with integrating-factor weights
$E \equiv e^{-\nu|\mathbf{k}|^{2}\Delta t}$:

$$
\begin{aligned}
\hat\omega_{1}      &= E\,\hat\omega_{n} + \Delta t\,E\,\bigl[-\hat N_S(\hat\omega_{n})\bigr],\\
\hat\omega_{2}      &= \tfrac{3}{4}E^{1/2}\hat\omega_{n}
                       + \tfrac{1}{4}E^{-1/2}\hat\omega_{1}
                       + \tfrac{1}{4}\Delta t\,E^{-1/2}\,\bigl[-\hat N_S(\hat\omega_{1})\bigr],\\
\hat\omega_{n+1}    &= \tfrac{1}{3}E\,\hat\omega_{n}
                       + \tfrac{2}{3}E^{1/2}\hat\omega_{2}
                       + \tfrac{2}{3}\Delta t\,E^{1/2}\,\bigl[-\hat N_S(\hat\omega_{2})\bigr].
\end{aligned}
\tag{18.6}
$$

Viscous stability becomes unconditional (viscosity is mode-by-mode exact),
so $\Delta t$ is set solely by the advective CFL $\Delta t \le C_{\text{CFL}}
\,\Delta x / \max|\mathbf{u}|$, $C_{\text{CFL}} \approx 0.5$.

### Effective viscosity diagnostic

The actual dissipation rate of enstrophy $\mathcal{E} \equiv \tfrac{1}{2}\sum|\hat\omega|^{2}$
is compared against the prescribed $\nu$ via

$$
\nu_{\text{eff}} \equiv -\frac{\dd\mathcal{E}/\dd t}
                            {2\sum|\mathbf{k}|^{2}\,|\hat\omega(\mathbf{k})|^{2}}.
\tag{18.7}
$$

A healthy run has $\nu_{\text{eff}}/\nu \to 1$ in the statistically steady
regime.  Deviation signals numerical dissipation leakage (typically from
insufficient dealiasing or $\Delta t$ too large).

### Energy spectrum (ring average)

$$
E(k) = \frac{1}{2}\sum_{k - \tfrac{1}{2} < |\mathbf{k}'| \le k + \tfrac{1}{2}}
        \frac{|\hat\omega(\mathbf{k}')|^{2}}{|\mathbf{k}'|^{2}}
\tag{18.8}
$$

with $k$ binned in integer shells.  The forced-turbulence benchmark
(`--test forced_turb`) shows a clean $k^{-3}$ enstrophy cascade past the
injection scale and a modest $k^{-5/3}$ inverse-cascade tail.


---

## §19 Sturm--Liouville Spectral Basis (Phase 0 ext+ — 1D offline)

This section documents the 1D Python solvers developed during Phase 0 ext+
(2026-05-02..03) to certify the spectral approach for the
reduced-pressure Poisson problem and the GYRE adiabatic pulsation
eigenvalue problem.  These routines do **not** ship in the GPU solver —
they are regression oracles and building blocks for the Phase 1
Fourier--Chebyshev 2D solver.  The authoritative derivation of the
spectral formulation and its convergence analysis is
`docs/spectral_stratified_poisson_report_2026-05-03.md`; this section
records the code-level contract.

### Reduced-pressure Poisson operator (Fourier in $x$)

Writing the reduced pressure $\pi = p/\rho_{0}$, the variable-density
Poisson equation reads
$\nabla\!\cdot(\rho_{0}\nabla\pi) = \tilde f$, and its Fourier mode
$\hat\pi(k_x, r)$ satisfies

$$
\frac{\dd}{\dd r}\!\left[\rho_{0}(r)\,\frac{\dd\hat\pi}{\dd r}\right]
- k_x^{2}\,\rho_{0}(r)\,\hat\pi = \hat{\tilde f}(k_x, r),
\qquad r \in [a, b].
\tag{19.1}
$$

### Chebyshev--Gauss--Lobatto collocation

On the CGL grid $\xi_{j} = \cos(j\pi/N)$, $j = 0,\ldots,N$, mapped affinely
to $r\in[a,b]$, and with $D$ the Trefethen spectral differentiation matrix
(size $(N+1)\times(N+1)$):

$$
L_{N} = D\,R_{\rho}\,D - k_{x}^{2}\,R_{\rho},
\qquad R_{\rho} \equiv \operatorname{diag}\bigl(\rho_{0}(r_{j})\bigr).
\tag{19.2}
$$

Dirichlet BCs are imposed by strong collocation (unit rows at $j=0, N$).

### Convergence regime (integer vs fractional surface exponent)

For $\rho_{0}(r) \sim (R - r)^{\sigma}$ near the surface:

$$
\|\pi_{N} - \pi_{\text{exact}}\|_{\infty} \sim
\begin{cases}
\rho^{N} & \sigma \in \mathbb{Z}_{\ge 0} \quad\text{(spectral)}\\
N^{-\sigma - 1/2} & \sigma \notin \mathbb{Z} \quad\text{(algebraic, Trefethen Thm 7.2)}
\end{cases}
\tag{19.3}
$$

Lane--Emden $n = 3$ (Eddington, $\sigma = 3$) is the only standard
physically-motivated polytrope in the spectral-convergence branch;
$n = 3/2$ (convective core) is in the $N^{-2}$ algebraic branch and
requires a Jacobi-weighted basis (outside the current implementation).

### Liouville potential (reference-only, not used in Phase 1)

Under $\hat\pi = \rho_{0}^{-1/2}\,q$, equation (19.1) transforms to

$$
q'' + \widetilde W(r)\,q - k_x^{2}\,q = \tilde g,
\qquad \widetilde W = \frac{\rho_{0}''}{2\rho_{0}} - \frac{(\rho_{0}')^{2}}{4\rho_{0}^{2}},
\tag{19.4}
$$

exhibiting the $k_x$-independent Schr\"odinger structure.  Near the surface
$\widetilde W \sim \sigma(\sigma - 2)/[4(R - r)^{2}]$.  The Liouville
programme's original promise — "one basis diagonalises Poisson and the
g-mode operator simultaneously" — fails in the strong sense: the Poisson
operator's optimal prefactor is $\alpha_{\star}(\text{Poisson}) = 1 - \sigma/2$
whereas the g-mode operator has $\beta_{\star}(\text{g-mode}) = \ell + 1$
(different singular point, $r = 0$ vs $r = R$).  The operational consequence
is that the Chebyshev mesh is shared but the operators are assembled and
factorised independently; see the technical report §7 for the full analysis.

### GYRE 4-variable adiabatic pulsation equations

For $\ell$-mode oscillations of a non-rotating star, with
$\alpha_{\text{grv}} = 1$ (full self-gravity), using the Unno et al.\
(1989) non-dimensional variables $(y_{1}, y_{2}, y_{3}, y_{4})$:

$$
\begin{aligned}
x\,\tfrac{\dd y_{1}}{\dd x} &=
  (V_{g} - \ell - 1)\,y_{1}
  + \Bigl(\tfrac{\lambda}{c_{1}\omega^{2}} - V_{g}\Bigr)\,y_{2}
  + \tfrac{\lambda}{c_{1}\omega^{2}}\,y_{3},\\
x\,\tfrac{\dd y_{2}}{\dd x} &=
  (c_{1}\omega^{2} - A^{\star}_{\text{iso}})\,y_{1}
  + (A^{\star} - U + 3 - \ell)\,y_{2}
  - y_{4},\\
x\,\tfrac{\dd y_{3}}{\dd x} &=
  (3 - U - \ell)\,y_{3} + y_{4},\\
x\,\tfrac{\dd y_{4}}{\dd x} &=
  U A^{\star}\,y_{1} + U V_{g}\,y_{2}
  + \lambda\,y_{3} + (2 - U - \ell)\,y_{4},
\end{aligned}
\tag{19.5}
$$

with $x = r/R$, $\lambda = \ell(\ell+1)$, $y_{3} = \Phi'/(gr)$,
$y_{4} = (\dd\Phi'/\dd r)/g$.  Structure coefficients
$V_{g} = V/\Gamma_{1}$, $A^{\star}$, $U$, $c_{1}$, $\Gamma_{1}$ follow
the Unno et al.\ definitions and are read from the GYRE native
polytropic file (`/tmp/gyre_run/poly3.txt` for Lane--Emden $n = 3$).

### Regularity and vacuum boundary conditions

Regularity at $x = 0$ and vacuum at $x = 1$:

$$
\begin{aligned}
\text{inner: }\quad
  & c_{1}\omega^{2} y_{1} - \ell y_{2} - \ell y_{3} = 0,\qquad \ell y_{3} - y_{4} = 0,\\
\text{outer: }\quad
  & y_{1} - y_{2} = 0,\qquad U y_{1} + (\ell + 1) y_{3} + y_{4} = 0.
\end{aligned}
\tag{19.6}
$$

### Generalised eigenvalue problem and benchmark

After Chebyshev collocation on $x \in [0.01, 0.999]$ with $N+1$ nodes,
(19.5) assembles into a $4(N+1)\times 4(N+1)$ generalised eigenproblem

$$
\mathsf{P}\,\mathbf{u} = \omega^{-2}\,\mathsf{Q}\,\mathbf{u},
\tag{19.7}
$$

in which the inverse-$\omega^{2}$ structure of equation (19.5a) requires
the generalised (linearised) form rather than a standard linear eigenproblem.
At $N = 48$ ($192$ DOF) the first radial-order g-mode frequency
$\omega^{2}_{n_{g}=1}(\ell=1)$ agrees with the GYRE reference to
$5.9\times 10^{-7}$; the maximum relative error over the first ten
radial orders is $1.5\times 10^{-6}$.  A $21\times$ reduction in DOF and
$350\times$ lower maximum error compared with a staggered second-order
finite-difference discretisation at $N_{r} = 1024$.

### Barycentric Lagrange evaluation

The $N + 1$ Chebyshev coefficients $\{a_{n}\}$ or nodal values
$\{u_{j}\}$ define a continuous representation that is evaluable at any
$r^{\star}$ via the barycentric formula

$$
u(r^{\star}) = \frac{\sum_{j} w_{j}\,u_{j}/(r^{\star} - r_{j})}
                    {\sum_{j} w_{j}/(r^{\star} - r_{j})},
\qquad
w_{j} = (-1)^{j}\,c_{j},
\tag{19.8}
$$

with $c_{0} = c_{N} = 1/2$, $c_{j} = 1$ otherwise (Berrut \&
Trefethen 2004).  The evaluation is rounding-error stable, with error
$\le 10^{-12}$, and is the mechanism by which a low-$N$ Chebyshev
representation supports arbitrary-resolution output rendering.

### Reproducibility

Frozen regression oracles (`python <script>.py --verify` exits zero):

- `scripts/gmode_exp_i_gyre_compat.py` — 2-var Cowling FD, GYRE Cowling
  benchmark, max rel.\ diff.\ $5.6\times 10^{-4}$ at $N_{r} = 1024$.
- `scripts/gmode_exp_j_full_gyre_compat.py` — 4-var full gravity FD,
  GYRE full benchmark, max rel.\ diff.\ $5.3\times 10^{-4}$.
- `scripts/gmode_exp_k_chebyshev_full.py` — 4-var full gravity Chebyshev,
  GYRE full benchmark, max rel.\ diff.\ $1.5\times 10^{-6}$ at $N = 48$.
- `scripts/spectral_analytical_ceiling.py` — three analytic ceiling
  tests (manufactured Poisson, harmonic oscillator, Dirichlet
  Laplacian); all reach $10^{-13}$ to $10^{-15}$ at $N \lesssim 64$.

The continuous-representation demonstration
(`scripts/spectral_resolution_demo.py`) shows the $N = 48$ Chebyshev
basis produces the same eigenfunction at 4096-point barycentric
sampling as the FD $N_{r} = 1024$ solution at the same points, to
within the $N = 48$ discretisation error of $3.4\times 10^{-3}$.


---

## Equation Index

| Eq. | Description | Source file(s) |
|-----|-------------|---------------|
| 1.1 | Conservative variables $\mathbf{U}$ | `state.h`, `fas_common.cuh` |
| 1.2 | Ideal gas EOS | `eos.h`, `strang_device.cuh:d_cons2prim` |
| 1.3 | Sound speed | `eos.h`, `strang_device.cuh`, `fas_hllc.cuh` |
| 1.4 | Euler equations (spherical) | `hydro/flux.cpp`, `fas_residual.cu` |
| 1.5 | Radial flux | `hydro/riemann.cpp`, `fas_hllc.cuh` |
| 1.6 | Theta flux | `hydro/riemann.cpp`, `fas_hllc.cuh` |
| 1.7 | Geometric + gravity source | `hydro/flux.cpp`, `fas_residual.cu` |
| 1.8–1.9 | Poisson equation | `gravity/poisson.cpp`, `gmg_gpu.cu` |
| 2.1 | Logarithmic radial mesh | `grid.cpp` |
| 2.2 | Cell volume | `grid.cpp`, `fas_solver.cu` |
| 2.3 | Radial face area | `grid.cpp`, `fas_solver.cu` |
| 2.4 | Theta face area | `grid.cpp`, `fas_solver.cu` |
| 2.5 | FV update | `hydro/flux.cpp`, `fas_residual.cu` |
| 3.1 | Minmod limiter | `hydro/reconstruct.cpp`, `fas_hllc.cuh` |
| 3.2 | Van Leer limiter | `hydro/reconstruct.cpp` |
| 3.3–3.5 | MUSCL reconstruction | `hydro/reconstruct.cpp`, `fas_residual.cu` |
| 4.1–4.7 | HLLC Riemann solver | `hydro/riemann.cpp`, `fas_hllc.cuh`, `strang_device.cuh` |
| 5.1–5.4 | Geometric source (volume-consistent) | `hydro/flux.cpp`, `fas_residual.cu` |
| 6.1–6.3 | Gravity source | `hydro/flux.cpp`, `fas_residual.cu`, `lm_residual.cu` |
| 6.4–6.5 | Discrete Laplacian | `gravity/poisson.cpp`, `gmg_gpu.cu` |
| 6.6–6.8 | Poisson BCs | `gravity/poisson.cpp`, `gmg_gpu.cu` |
| 6.9–6.10 | Gravity gradient | `hydro/flux.cpp`, `fas_residual.cu` |
| 7.1–7.3 | RK2 (Heun's method) | `hydro/integrate.cpp` |
| 7.4 | CFL condition | `hydro/integrate.cpp` |
| 8.1–8.3 | Polar BCs | `bc/boundary.cpp`, `fas_residual.cu` |
| 9.1–9.8 | Initial conditions | `init/lane_emden.cpp`, `init/sedov.cpp`, `init/jeans.cpp`, `init/evrard.cpp` |
| 10.1–10.5 | Low-Mach JFNK solver | `lm_residual.cu`, `lm_krylov.cu`, `lowmach_solver.cu` |
| 11.1–11.6 | Physics-based preconditioner | `lm_precond.cu` |
| 12.1–12.3 | FAS WB residual | `fas_residual.cu` |
| 12.4–12.6 | FAS origin cell | `fas_residual.cu:k_fas_residual_origin` |
| 12.7 | FAS perturbation MUSCL | `fas_residual.cu` |
| 12.8 | FAS restrict | `fas_multigrid.cu:k_fas_restrict_state` |
| 12.9 | FAS tau correction | `fas_multigrid.cu:k_fas_assemble_coarse_rhs` |
| 12.10 | FAS prolongate (limited) | `fas_multigrid.cu:k_fas_prolongate_correct` |
| 12.11–12.12 | Block-Jacobi smoother | `fas_smoothers.cu:k_fas_smooth_blkjac` |
| 12.13 | Smooth floor | `fas_residual.cu:k_fas_floor` |
| 12.14 | CFL (polar) | `fas_residual.cu:k_fas_cfl` |
| 13.1 | Perturbation storage | `strang_solver.cuh` |
| 13.2 | Isentropic HSE background | `strang_device.cuh:d_hse_rho`, `d_hse_p` |
| 13.3 | MC limiter | `strang_device.cuh:d_mc_limit` |
| 13.4 | WB y-sweep MUSCL | `strang_solver.cu:k_strang_sweep_y` |
| 13.5–13.6 | Hancock predictor | `strang_solver.cu:k_strang_sweep_x`, `k_strang_sweep_y` |
| 13.7 | Strang splitting | `strang_solver.cu:StrangSolver::step` |
| 13.8 | Cartesian BCs | `strang_solver.cu:k_ghost_x`, `k_ghost_y` |
| 13.9 | CFL (Cartesian) | `strang_solver.cu:StrangSolver::compute_dt` |
| 14.1–14.2 | LM-HLLC Mach blending | `strang_device.cuh:d_lmhllc`, `fas_hllc.cuh:fas_hllc_lm` |
| 14.3 | LM-HLLC $S^*$ (Strang) | `strang_device.cuh:d_lmhllc` |
| 14.4 | LM-HLLC $S^*$ (FAS) | `fas_hllc.cuh:fas_hllc_lm` |
| 15.1 | Node mass (cart_lag/ale) | `cart_lag_kernels.cu:k_clag_node_mass`, `cart_ale2_kernels.cu:k_cale2_node_mass` |
| 15.2 | Shoelace cell area | `cart_ale_kernels.cu:k_cale_geometry`, `cart_ale2_kernels.cu:k_cale2_geometry` |
| 15.3 | Divergence-consistent strain | `cart_ale2_kernels.cu:k_cale2_eos_and_q` |
| 15.4 | Artificial viscosity | `cart_ale2_kernels.cu:k_cale2_eos_and_q` |
| 15.5–15.6 | Subcell edge / corner force | `cart_ale2_kernels.cu:k_cale2_node_forces` |
| 15.7 | Kick-drift-kick node update | `cart_ale2_kernels.cu:k_cale2_node_update` |
| 15.8 | Compatible energy update | `cart_ale2_kernels.cu:k_cale2_energy_update` |
| 15.9 | Minimum-height CFL | `cart_ale2_kernels.cu:k_cale2_cfl` |
| 16.1 | Swept-region signed area | `cart_ale2_kernels.cu:swept_quad_signed` |
| 16.2 | Donor-cell flux | `cart_ale2_kernels.cu:k_cale2_remap_east`/`north` |
| 16.3 | Limited slopes | `cart_ale2_kernels.cu:k_cale2_slopes_minmod` |
| 16.4 | Face centroid MUSCL eval | `cart_ale2_kernels.cu:k_cale2_remap_*_2nd` |
| 16.5 | Conservative remap update | `cart_ale2_kernels.cu:k_cale2_remap_*` (all variants) |
| 16.6 | Mass-weighted rebuild | `cart_ale2_kernels.cu:k_cale2_rebuild_node_v` |
| 17.1 | PPM 4-point interpolant | `cart_ale2_kernels.cu:k_cale2_ppm_reconstruct` |
| 17.2 | CW monotonization | `cart_ale2_kernels.cu:ppm_monotonize` |
| 17.3–17.5 | Colella-Sekora limiter | `cart_ale2_kernels.cu:ppm_cs_limit` |
| 17.6 | Primitive→conservative flux | `cart_ale2_kernels.cu:k_cale2_remap_*_ppm_prim` |
| 17.7–17.8 | Characteristic projection | `cart_ale2_kernels.cu:char_project_x`, `char_unproject_x`, `k_cale2_ppm_reconstruct_char` |
| 18.1 | Vorticity--streamfunction NS | `pseudo_spectral_kernels.cu:k_spec_uv`, `k_spec_grad_omega` |
| 18.2 | Circular 2/3 dealias | `pseudo_spectral_kernels.cu:k_init_wavenumbers`, `k_apply_dealias` |
| 18.3–18.4 | Orszag skew-symmetric convection | `pseudo_spectral_kernels.cu:k_compute_skew_nonlinear`, `k_form_rhs_skew`; adv-only: `k_compute_adv_nonlinear`, `k_form_rhs_adv_only` |
| 18.5–18.6 | IFRK3 integrating-factor RK3 | `pseudo_spectral_kernels.cu:k_ifrk_combine`, `pseudo_spectral_solver.cu:PseudoSpectralSolver::step` |
| 18.7 | Effective viscosity diagnostic | `pseudo_spectral_kernels.cu:k_reduce_diag`, `k_reduce_k2E` |
| 18.8 | Ring-averaged energy spectrum | `pseudo_spectral_kernels.cu:k_reduce_spectrum_bins`; post: `scripts/spectrum_pseudo_spectral.py` |
| 19.1–19.2 | Reduced-pressure Chebyshev operator | `scripts/spectral_liouville_convergence_v2.py`, `reduced_pressure_chebyshev.py` |
| 19.3 | Polytropic index convergence dichotomy | `scripts/spectral_liouville_convergence_v2.py` (empirical); `docs/polytropic_index_spectral_convergence_2026-05-03.md` (analysis) |
| 19.4 | Liouville potential + α★/β★ prefactors | `scripts/spectral_liouville_beta_derivation.py`, `spectral_liouville_beta_gmode_check.py` |
| 19.5–19.6 | GYRE 4-var adiabatic pulsation + BCs | `scripts/gmode_infra.py:solve_gmode_full_gyre_compat` (FD), `gmode_exp_k_chebyshev_full.py:solve_gmode_full_chebyshev` (spectral) |
| 19.7 | Generalised eigenvalue problem | `scripts/gmode_exp_k_chebyshev_full.py` (Chebyshev), `gmode_exp_j_full_gyre_compat.py` (FD reference) |
| 19.8 | Barycentric Lagrange evaluation | `scripts/spectral_resolution_demo.py` (uses `scipy.interpolate.BarycentricInterpolator`) |
