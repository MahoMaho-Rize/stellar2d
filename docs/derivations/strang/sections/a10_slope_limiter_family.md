# A10. Slope-limiter family (MC / minmod / van Leer / superbee / Ospre)

> **sympy script:** `scripts/a10_slope_limiter_family.py`
> **generated LaTeX:** `output/a10_slope_limiter_family.latex.tex`
> **verifies:** 28 strong-form identities — second-order
> consistency $\phi(1) = 1$ for all 5 limiters; spot values at
> $r \in \{-1/2, 3/2, 2, 3, 4\}$; zero-at-extremum property for
> minmod/VL/MC/superbee; explicit Ospre non-TVD demonstration at
> $r = -1/2$; 6 numerical-fallback identities (symmetry $\phi(1/r)
> = \phi(r)/r$ for all 5 limiters + MC kernel/Sweby equivalence) at
> $\ge 100$ random samples each, all residuals $\le 10^{-15}$
> **code checkpoints:**
> `src/gpu/explicit/strang_device.cuh :: d_mc_limit`

This is the **second alternative-scheme comparison** section of the
book. The Strang kernel uses the MC limiter exclusively, but a
derivation book that does not place MC in the wider family of
Sweby-form TVD limiters cannot answer the practical question
"would van Leer be enough?" or "is superbee too aggressive for
stratified convection?". All five canonical limiters are derived
here, their TVD regions compared, and the kernel's two-argument
form `d_mc_limit(a, b)` is proven equivalent to its Sweby-form
counterpart.

## Sweby form and TVD constraints

Let $r = \Delta_L / \Delta_R$ be the ratio of the two consecutive
slope estimates (left difference over right difference) at a cell
centre. A Sweby-form limiter $\phi(r)$ selects the effective slope
$\phi(r)\,\Delta_R$ subject to the total-variation-diminishing
(TVD) constraints of Sweby (1984):

$$0 \;\le\; \phi(r) \;\le\; \min(2r, 2) \quad (r \ge 0), \qquad \phi(r) = 0 \quad (r < 0). \quad (\text{A10-Sweby-region})$$

The zero-at-negative-$r$ condition means **local extrema get flat
reconstructions** (no overshoot). Second-order consistency on
smooth flow requires

$$\phi(1) \;=\; 1. \quad (\text{A10-second-order})$$

This is the one constraint every limiter below obeys; differences
lie in how they fill the admissible TVD region.

## Limiter family

| limiter | Sweby form $\phi(r)$ | behaviour |
|---|---|---|
| minmod | $\max(0,\ \min(1,\ r))$ | most diffusive; TVD-strict |
| van Leer | $(r + \lvert r\rvert)/(1 + \lvert r\rvert)$ | smooth $C^1$; TVD-strict |
| MC | $\max(0,\ \min(2r,\ (1+r)/2,\ 2))$ | kernel's choice; TVD-strict |
| superbee | $\max(0,\ \min(2r, 1),\ \min(r, 2))$ | most aggressive; TVD-strict |
| Ospre | $\tfrac{3}{2}\cdot(r^{2} + r)/(r^{2} + r + 1)$ | rational; extends into $r < 0$ (**not strictly TVD**) |

**Ospre non-TVD note.** Unlike the other four, Ospre is a rational
function that does not enforce $\phi(r) = 0$ for $r < 0$. At
$r = -1/2$, $\phi_{\mathrm{Ospre}}(-1/2) = -1/2$, violating the
Sweby condition $\phi \ge 0$. Ospre trades strict TVD for third-
order smooth-extremum resolution; it is listed here for comparison
but is not a drop-in replacement for MC in stellar2d.

## Spot values and aggressiveness ranking

At $r = 3/2$ (moderate positive gradient, a typical value in
smooth-but-non-trivial flow):

$$\phi_{\mathrm{minmod}}(3/2) = 1 \;<\; \phi_{\mathrm{Ospre}}(3/2) = 45/38 \approx 1.184 \;<\; \phi_{\mathrm{VL}}(3/2) = 6/5 \;<\; \phi_{\mathrm{MC}}(3/2) = 5/4 \;<\; \phi_{\mathrm{superbee}}(3/2) = 3/2. \quad (\text{A10-comparison})$$

Each $\phi$ value is verified algebraically.

At $r = 2$ (steep gradient), the $(1+r)/2$ constraint of MC binds:
$\phi_{\mathrm{MC}}(2) = 3/2$. MC's hard cap $\phi \le 2$ only
binds for $r \ge 3$; verified at $r = 3$ and $r = 4$ explicitly.
At $r = 2$, superbee hits the upper TVD envelope ($\phi = 2$),
while minmod stays at $\phi = 1$ (TVD maximal diffusion), and
van Leer sits at $\phi = 4/3$ (moderate).

## Zero-at-extremum

At any local extremum, the two neighbouring slopes have opposite
sign; hence $r < 0$. All four non-Ospre limiters return $\phi = 0$
there, giving a flat reconstruction at the extremum — no overshoot,
no oscillation. Ospre's explicit failure at $r = -1/2$ is noted
above.

## Symmetry property

For all five limiters (including Ospre),

$$\phi(1/r) \;=\; \phi(r)/r \qquad (r > 0). \quad (\text{A10-symmetry})$$

This ensures the reconstruction is **direction-symmetric**:
reversing the ordering of the two neighbours gives the same
effective slope magnitude. Verified by numerical random sampling
at 100 positive values of $r$ with atol $10^{-12}$ (sympy's
Min/Max simplifier cannot reduce this identity symbolically; this
is a sympy-capability workaround, not a weak-form step).

## Kernel's two-argument form is Sweby-form MC

The kernel stores two consecutive slope differences $a, b$ and
returns the limited slope directly (not the ratio):

$$\mathrm{d\_mc\_limit}(a, b) \;=\;
\begin{cases}
\mathrm{sign}(a)\,\min\!\bigl(\tfrac{|a+b|}{2},\ 2|a|,\ 2|b|\bigr) & \mathrm{sign}(a)\,\mathrm{sign}(b) > 0, \\[2pt]
0 & \text{otherwise.}
\end{cases} \quad (\text{A10-kernel-form})$$

**Equivalence.** For the same-sign branch ($a > 0$, $b > 0$
without loss of generality), the kernel's $\min\bigl(\tfrac{a+b}{2},
2a, 2b\bigr)$ equals $\phi_{\mathrm{MC}}(b/a) \cdot a$ at every
admissible $(a, b)$ — verified at 100 random positive pairs,
max residual $10^{-15}$.

The opposite-sign branch returns $0$ directly, which matches the
zero-at-extremum property of $\phi_{\mathrm{MC}}(r)$ at $r < 0$.

## ✅ Verification checkpoint (to be wired)

The kernel's `d_mc_limit` is a simple two-line function. Tests:

1. **Kernel identity at canonical points.** At $a = b$ (smooth,
   $r = 1$), `d_mc_limit(a, a) == a` exactly. At $a = 0$ or
   $b = 0$, `d_mc_limit == 0`. At opposite signs,
   `d_mc_limit(+a, -b) == 0`. These are the three corner cases that
   define the limiter's behaviour. Test: `test_strang_muscl.cu`
   §A10-corner-cases.

2. **Kernel-vs-Sweby equivalence.** For 100 random same-sign pairs
   $(a, b)$, `d_mc_limit(a, b)` must equal
   $a \cdot \phi_{\mathrm{MC}}(b/a)$ to ULP precision. Test:
   `test_strang_muscl.cu` §A10-Sweby-equivalence.

3. **TVD region.** For 100 random $r > 0$ via the kernel's
   two-argument form at some $(a, b)$ with $r = b/a$, the returned
   $\phi = \text{kernel output}/a$ must satisfy $0 \le \phi \le
   \min(2r, 2)$. Test: `test_strang_muscl.cu` §A10-TVD-region.

Failure of (1) indicates a sign-handling bug; failure of (2)
indicates an arithmetic error in the min-of-three; failure of
(3) would mean the kernel violates TVD, which would show up as
spurious oscillations in a shock-tube test (§D3).
