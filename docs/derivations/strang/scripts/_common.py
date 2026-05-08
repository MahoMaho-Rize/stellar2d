r"""
docs/derivations/strang/scripts/_common.py — shared sympy utilities
for the Strang-split Euler derivation book.

Every section script imports from here. Design:
- Hydrodynamics-only symbol table (no magnetic / radiation clutter).
- Background HSE symbols (rho_bar, p_bar) live alongside perturbation
  symbols (delta_rho, delta_E), matching the perturbation-form storage
  of the Strang solver.
- Numerical-scheme scalars: CFL sigma, Mach cutoff M_cut, slope-ratio
  r used in Sweby-form limiter analysis.
- LatexDump / assert_zero helpers identical in spirit to the MHD book,
  kept source-compatible so a shared utility module can be extracted
  later if multiple derivation books converge.
- Optional JSON dump helper (GoldensDump) for Part D sections.
  GoldensDump targets output/<stem>.goldens.json; this file is in
  .gitignore and consumed at ctest runtime.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import sympy as sp

# ════════════════════════════════════════════════════════════
# Symbol inventory — hydrodynamics only.
#
# Convention: all scalar state variables are positive where physical.
# Perturbation variables (delta_*) are signed.  "bar" suffix denotes
# the HSE background, which depends only on y in the Strang solver.
# ════════════════════════════════════════════════════════════

# ---------- scalar state ----------
rho = sp.Symbol("rho", positive=True)
p = sp.Symbol("p", positive=True)              # gas pressure
T = sp.Symbol("T", positive=True)
e = sp.Symbol("e", positive=True)              # specific internal energy
E = sp.Symbol("E", positive=True)              # total energy per unit volume
gamma = sp.Symbol("gamma", positive=True)       # ratio of specific heats
c_sound = sp.Symbol("c", positive=True)         # adiabatic sound speed
s_entropy = sp.Symbol("s", real=True)           # specific entropy s = ln(P/rho^gamma)

# ---------- velocities and momenta ----------
u, v_vel, w_vel = sp.symbols("u v w", real=True)  # 1-D primitive velocities
v_x, v_y, v_z = sp.symbols("v_x v_y v_z", real=True)
m_x, m_y, m_z = sp.symbols("m_x m_y m_z", real=True)

# ---------- perturbation-form storage (Strang solver convention) ----------
# Using Function for rho_bar / p_bar lets us write rho_bar(y).
rho_bar = sp.Function("rho_bar")               # background density rho_bar(y)
p_bar = sp.Function("p_bar")                   # background pressure p_bar(y)
delta_rho = sp.Symbol("delta_rho", real=True)
delta_E = sp.Symbol("delta_E", real=True)

# ---------- left / right Riemann states ----------
rho_L, rho_R = sp.symbols("rho_L rho_R", positive=True)
u_L, u_R = sp.symbols("u_L u_R", real=True)       # normal velocity
v_L, v_R = sp.symbols("v_L v_R", real=True)       # tangential velocity
p_L, p_R = sp.symbols("p_L p_R", positive=True)
E_L, E_R = sp.symbols("E_L E_R", positive=True)   # total energy / unit volume
c_L, c_R = sp.symbols("c_L c_R", positive=True)   # sound speeds

# ---------- HLLC star-region quantities ----------
S_L, S_R, S_star = sp.symbols("S_L S_R S_star", real=True)
p_star = sp.Symbol("p_star", positive=True)
rho_starL, rho_starR = sp.symbols("rho_{*L} rho_{*R}", positive=True)

# ---------- coordinates and time ----------
x, y, z, t = sp.symbols("x y z t", real=True)

# ---------- wavenumber and mode labels ----------
k, k_x, k_y = sp.symbols("k k_x k_y", real=True)
omega = sp.Symbol("omega", real=True)           # dispersion-relation frequency

# ---------- numerical scheme scalars ----------
dx = sp.Symbol("Delta_x", positive=True)
dy = sp.Symbol("Delta_y", positive=True)
dt = sp.Symbol("Delta_t", positive=True)
sigma_cfl = sp.Symbol("sigma", positive=True)   # CFL safety factor
r_slope = sp.Symbol("r", real=True)             # slope ratio in Sweby form
M_loc = sp.Symbol("M", nonnegative=True)        # local Mach number
M_cut = sp.Symbol("M_cut", positive=True)       # LM-HLLC Mach cutoff
fM = sp.Symbol("f_M", nonnegative=True)         # LM-HLLC pressure-blend factor

# ---------- gravity and HSE constants ----------
g_grav = sp.Symbol("g", positive=True)          # uniform downward gravity
K_poly = sp.Symbol("K", positive=True)          # isentropic constant p = K rho^gamma
rho_0 = sp.Symbol("rho_0", positive=True)       # bottom-of-atmosphere density
y_atm_cut = sp.Symbol("y_star", positive=True)  # atmosphere cutoff

# ---------- limiter: two neighbouring slope differences ----------
a_slope, b_slope = sp.symbols("a b", real=True)

# ---------- wave amplitude (for linwave / entropy-wave ICs) ----------
A_amp = sp.Symbol("A", positive=True)
rho_ref = sp.Symbol("rho_bg", positive=True)    # background density for IC
p_ref = sp.Symbol("P_bg", positive=True)         # background pressure for IC
u_ref = sp.Symbol("u_bg", real=True)             # background velocity for IC
c_ref = sp.Symbol("c_bg", positive=True)         # background sound speed for IC


# ════════════════════════════════════════════════════════════
# Vector / tensor helpers (2D-oriented, but 3-component kept
# because the Strang state carries (rho, m_x, m_y, E) and the
# Riemann solver treats v, w as tangential).
# ════════════════════════════════════════════════════════════
def vec3(a, b, c=0):
    """Return a 3-vector as a sp.Matrix column."""
    return sp.Matrix([a, b, c])


def dot_prod(a, b):
    return (a.T @ b)[0, 0]


def div_cart_2d(Vx, Vy, coords=(x, y)):
    """Divergence of a 2D vector field (Vx, Vy) in Cartesian."""
    X, Y = coords
    return sp.diff(Vx, X) + sp.diff(Vy, Y)


def grad_cart_2d(f, coords=(x, y)):
    return sp.Matrix([sp.diff(f, c) for c in coords])


def div_cart_3d(Vx, Vy, Vz, coords=(x, y, z)):
    """Divergence of a 3D vector field."""
    X, Y, Z = coords
    return sp.diff(Vx, X) + sp.diff(Vy, Y) + sp.diff(Vz, Z)


# ════════════════════════════════════════════════════════════
# Euler flux and state helpers
#
# These are the canonical flux vectors derived in A1 and reused
# across the Riemann-solver sections (A7-A9), the Hancock predictor
# (A12), and the CFL bound (C2).
# ════════════════════════════════════════════════════════════
def total_energy_sym(rho_, u_, v_, p_, gamma_):
    """Total energy per unit volume E = P/(gamma-1) + rho*(u^2+v^2)/2."""
    return p_ / (gamma_ - 1) + sp.Rational(1, 2) * rho_ * (u_**2 + v_**2)


def flux_x_euler(rho_, u_, v_, p_, gamma_):
    """2D Euler flux in x-direction, (F0, F1, F2, F3)."""
    E_ = total_energy_sym(rho_, u_, v_, p_, gamma_)
    return sp.Matrix([
        rho_ * u_,
        rho_ * u_**2 + p_,
        rho_ * u_ * v_,
        (E_ + p_) * u_,
    ])


def flux_y_euler(rho_, u_, v_, p_, gamma_):
    """2D Euler flux in y-direction."""
    E_ = total_energy_sym(rho_, u_, v_, p_, gamma_)
    return sp.Matrix([
        rho_ * v_,
        rho_ * u_ * v_,
        rho_ * v_**2 + p_,
        (E_ + p_) * v_,
    ])


def cons_from_prim(rho_, u_, v_, p_, gamma_):
    """Conservative (rho, rho*u, rho*v, E) from primitive (rho, u, v, p)."""
    return sp.Matrix([
        rho_,
        rho_ * u_,
        rho_ * v_,
        total_energy_sym(rho_, u_, v_, p_, gamma_),
    ])


# ════════════════════════════════════════════════════════════
# LaTeX dump, JSON goldens, and assertion utilities
# ════════════════════════════════════════════════════════════
OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class LatexDump:
    """Accumulate LaTeX equations; write to output/<stem>.latex.tex."""

    def __init__(self, script_path: str | Path):
        stem = Path(script_path).stem
        self.out_path = OUT_DIR / f"{stem}.latex.tex"
        self.entries: list[tuple[str, str, str | None]] = []
        self.script_stem = stem

    def add(self, name: str, latex: str, label: str | None = None):
        self.entries.append((name, latex, label))

    def add_expr(self, name: str, expr, label: str | None = None):
        self.entries.append((name, sp.latex(expr), label))

    def add_equation(self, name: str, lhs, rhs, label: str | None = None):
        self.entries.append(
            (name, f"{sp.latex(lhs)} = {sp.latex(rhs)}", label)
        )

    def write(self):
        lines = [
            f"% Auto-generated by scripts/{self.script_stem}.py",
            f"% Do not edit manually.",
            "",
        ]
        for name, latex, label in self.entries:
            lines.append(f"% {name}")
            if label:
                lines.append(
                    "\\begin{equation}\\label{" + label + "}\n"
                    + latex
                    + "\n\\end{equation}"
                )
            else:
                lines.append("\\begin{equation*}\n" + latex + "\n\\end{equation*}")
            lines.append("")
        self.out_path.write_text("\n".join(lines))
        print(f"  wrote {self.out_path.relative_to(OUT_DIR.parent)}")


class GoldensDump:
    """Dump numerical golden values to output/<stem>.goldens.json.

    Part-D scripts call .add(key, value) with scalar floats, list-of-
    floats, or nested dicts.  The resulting JSON is consumed by
    tests/test_strang_*.cu.

    Per Rule 5: *.goldens.json is .gitignored; `bash run_all.sh` is a
    prerequisite of `ctest`.
    """

    def __init__(self, script_path: str | Path):
        stem = Path(script_path).stem
        self.out_path = OUT_DIR / f"{stem}.goldens.json"
        self.payload: dict = {}
        self.script_stem = stem

    def add(self, key: str, value):
        self.payload[key] = self._coerce(value)

    @staticmethod
    def _coerce(v):
        if isinstance(v, (int, float, str, bool)) or v is None:
            return v
        if isinstance(v, (list, tuple)):
            return [GoldensDump._coerce(x) for x in v]
        if isinstance(v, dict):
            return {k: GoldensDump._coerce(x) for k, x in v.items()}
        try:
            return float(v)
        except (TypeError, ValueError):
            return str(v)

    def write(self):
        self.out_path.write_text(json.dumps(self.payload, indent=2, sort_keys=True))
        print(f"  wrote {self.out_path.relative_to(OUT_DIR.parent)}")


def assert_zero(expr, name: str, trig: bool = False, verbose: bool = True):
    """Assert sp.simplify(expr) == 0; raise with context on failure."""
    simplified = sp.simplify(expr)
    if trig:
        simplified = sp.trigsimp(simplified)
    if simplified != 0:
        msg = (
            f"[FAIL] {name}: expected 0, got {simplified}\n"
            f"       raw expr: {expr}"
        )
        print(msg, file=sys.stderr)
        raise AssertionError(msg)
    if verbose:
        print(f"  [OK] {name} — sympy verified to 0.")


def assert_zero_numeric(expr, substitutions_iter, name: str, atol: float = 1e-10,
                         verbose: bool = True):
    """Fallback verification when sp.simplify cannot reach 0.

    Iterate over `substitutions_iter` (each a dict symbol->value),
    substitute, and assert magnitude below `atol`.  Caller supplies
    an ensemble of N>=50 admissible sample points.
    """
    worst = 0.0
    n_samples = 0
    for subs in substitutions_iter:
        val = float(sp.N(expr.subs(subs)))
        n_samples += 1
        if abs(val) > worst:
            worst = abs(val)
        if abs(val) > atol:
            msg = (
                f"[FAIL-num] {name}: |residual| = {abs(val):.3e} > atol {atol:.1e}\n"
                f"            at subs = {subs}"
            )
            print(msg, file=sys.stderr)
            raise AssertionError(msg)
    if verbose:
        print(f"  [OK-num] {name} — {n_samples} samples, max|residual| = {worst:.3e}.")


def banner(title: str):
    """Pretty-print a section banner to stdout."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
