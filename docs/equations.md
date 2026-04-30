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
