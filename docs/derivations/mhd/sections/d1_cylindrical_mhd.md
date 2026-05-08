# D1. Ideal MHD in cylindrical $(R, \phi, z)$ coordinates

> **sympy script:** `scripts/d1_cylindrical_mhd.py`
> **verified:** displays cylindrical divergence, curl, and axisymmetric
> reductions. No new nontrivial identities beyond Part A; this section
> is the reference for D2 and D3.
> **code checkpoints:** `athena_mhd_cylindrical_solver.cu` (future).

## Cylindrical operators

$$\nabla\cdot\mathbf{v} = \frac{1}{R}\partial_R(R v_R) + \frac{1}{R}\partial_\phi v_\phi + \partial_z v_z.$$

$$(\nabla\times\mathbf{B})_R = \frac{1}{R}\partial_\phi B_z - \partial_z B_\phi,\
(\nabla\times\mathbf{B})_\phi = \partial_z B_R - \partial_R B_z,\
(\nabla\times\mathbf{B})_z = \frac{1}{R}\!\left[\partial_R(R B_\phi) - \partial_\phi B_R\right].$$

## Radial momentum (axisymmetric)

$$\partial_t(\rho v_R) + \frac{1}{R}\partial_R(R\rho v_R^2) + \partial_z(\rho v_R v_z)
= \boxed{\frac{\rho v_\phi^2}{R}} - \partial_R p - \rho\partial_R\Phi_{\mathrm{grav}}. \quad (\text{D1-mom-R})$$

The $\rho v_\phi^2/R$ term is the **Christoffel-symbol / centrifugal
source**. It is what keeps a rotating disk in equilibrium against gravity.
In a kernel, it is the most common source of subtle bugs — easy to
drop when porting Cartesian kernels.

## Axisymmetric radial-flux conservation

$\partial_\phi = \partial_z = 0$ with $\mathbf{B} = B_R \hat{R}$
forces $\nabla\cdot\mathbf{B} = (1/R)\partial_R(R B_R) = 0$, so
$R\cdot B_R(R) = \text{const}$ — the cylindrical analog of the
spherical $r^2 B_r$ conservation.

## ✅ Verification

`tests/test_mhd_cyl_centrifugal.cu` — rotating equilibrium disk
with $v_\phi = \sqrt{GM/R}$. Lock $\max|v_R|/c_s < 10^{-4}$ over
10 rotations. Failure means the centrifugal source term is missing
or mis-signed.
