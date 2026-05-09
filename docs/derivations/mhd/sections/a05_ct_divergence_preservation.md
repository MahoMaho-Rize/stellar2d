# A5. Constrained Transport and discrete $\nabla\cdot\mathbf{B}=0$

> **sympy script:** `scripts/a5_ct_divergence_preservation.py`
> **verified:** CT telescoping identity
> $(\nabla\cdot\mathbf{B})^{n+1} - (\nabla\cdot\mathbf{B})^n = 0$;
> Gardiner-Stone 2005 corner-EMF averaging is 2nd-order accurate.
> **code checkpoints:** `athena_mhd_solver.cu::update_face_B_with_emf`,
> `tests/test_athena_mhd_field_loop.cu`.

## Grid layout

Yee-like staggered: $B_x$ on vertical faces $(i\pm\tfrac{1}{2}, j)$,
$B_y$ on horizontal faces $(i, j\pm\tfrac{1}{2})$, $E_z$ on corners.

## CT update (Evans-Hawley 1988)

$$B_x^{i\pm 1/2, j, n+1} = B_x^{i\pm 1/2, j, n} - \frac{\Delta t}{\Delta y}(E_z^{i\pm 1/2, j+1/2} - E_z^{i\pm 1/2, j-1/2}),$$
$$B_y^{i, j\pm 1/2, n+1} = B_y^{i, j\pm 1/2, n} + \frac{\Delta t}{\Delta x}(E_z^{i+1/2, j\pm 1/2} - E_z^{i-1/2, j\pm 1/2}).$$

## The central identity

$$\boxed{(\nabla\cdot\mathbf{B})^{n+1}_{i,j} = (\nabla\cdot\mathbf{B})^n_{i,j}}$$

**Sympy symbolically verifies** this as a pure telescoping identity —
the four corner-EMF contributions cancel exactly, regardless of what
values the EMFs take. This is why CT is qualitatively different from
Dedner GLM: ∇·B = 0 is an **exact** property, not a truncation
error.

## Gardiner-Stone 2005 corner-EMF averaging

$$E_z^{i+1/2, j+1/2} = \tfrac{1}{4}(E_z^{x,i+1/2,j} + E_z^{x,i+1/2,j+1} + E_z^{y,i,j+1/2} + E_z^{y,i+1,j+1/2}).$$

**Sympy-verified** via Taylor expansion: leading error is
$\tfrac{h^2}{8}(\partial_x^2 E_z + \partial_y^2 E_z) + \mathcal{O}(h^4)$,
no $\mathcal{O}(h)$ term — matches 2nd-order Godunov accuracy.

## ✅ Verification

`tests/test_athena_mhd_field_loop.cu` — GS05 field-loop, 10
crossings, lock $\max|\nabla\cdot\mathbf{B}| < 10\varepsilon_{\mathrm{mach}}$
at every step.
