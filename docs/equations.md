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
0 \\[4pt]
\displaystyle\frac{\rho v_\theta^2}{r} + \frac{2P}{r}
  - \rho\frac{\partial\Phi}{\partial r} \\[8pt]
\displaystyle\frac{P\cot\theta}{r} - \frac{\rho v_r v_\theta}{r}
  - \frac{\rho}{r}\frac{\partial\Phi}{\partial\theta} \\[8pt]
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
