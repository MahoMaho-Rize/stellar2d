"""
stellar2d test infrastructure.

Provides fixtures for:
  - Running the stellar2d binary in a temp directory
  - Parsing VTK output files
  - Parsing stdout diagnostics (mass, energy per step)
"""
import json
import os
import re
import subprocess
import tempfile

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY_CPU = os.path.join(REPO_ROOT, "build_cpu", "stellar2d")
BASELINE_DIR = os.path.join(REPO_ROOT, "tests", "baselines")


def _find_binary():
    for candidate in [BINARY_CPU,
                      os.path.join(REPO_ROOT, "build", "stellar2d")]:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    pytest.skip("No stellar2d binary found — build first")


# ── VTK reader ──────────────────────────────────────────────────────

def read_vtk(path):
    with open(path) as f:
        lines = f.readlines()

    for l in lines:
        if l.startswith("DIMENSIONS"):
            dims = list(map(int, l.split()[1:]))
            nt = dims[0] - 1
            nr = dims[1] - 1
            break

    idx = next(i for i, l in enumerate(lines) if l.startswith("CELL_DATA"))
    ncells = int(lines[idx].split()[1])

    data = {}
    i = idx + 1
    while i < len(lines):
        if lines[i].startswith("SCALARS"):
            name = lines[i].split()[1]
            i += 2
            vals = []
            while len(vals) < ncells:
                vals.extend(map(float, lines[i].split()))
                i += 1
            data[name] = np.array(vals).reshape(nr, nt)
        elif lines[i].startswith("VECTORS"):
            name = lines[i].split()[1]
            i += 1
            vals = []
            while len(vals) < ncells * 3:
                vals.extend(map(float, lines[i].split()))
                i += 1
            data[name] = np.array(vals).reshape(nr, nt, 3)
        else:
            i += 1

    idx_pts = next(i for i, l in enumerate(lines) if l.startswith("POINTS"))
    npts = int(lines[idx_pts].split()[1])
    coords = []
    j = idx_pts + 1
    while len(coords) < npts * 3:
        coords.extend(map(float, lines[j].split()))
        j += 1
    coords = np.array(coords).reshape(nr + 1, nt + 1, 3)

    x_n = coords[:, :, 0]
    z_n = coords[:, :, 2]
    r_nodes = np.sqrt(x_n**2 + z_n**2)
    r_cell = 0.25 * (r_nodes[:-1, :-1] + r_nodes[1:, :-1]
                      + r_nodes[:-1, 1:] + r_nodes[1:, 1:])

    theta_nodes = np.arctan2(x_n, z_n)  # atan2(sin, cos) = theta from pole
    theta_cell = 0.25 * (theta_nodes[:-1, :-1] + theta_nodes[1:, :-1]
                          + theta_nodes[:-1, 1:] + theta_nodes[1:, 1:])

    return {"nr": nr, "nt": nt, "r": r_cell, "theta": theta_cell,
            "coords": coords, **data}


# ── Stdout parser ───────────────────────────────────────────────────

_STEP_RE = re.compile(
    r"Step\s+(\d+)\s+t\s*=\s*([\d.eE+-]+)\s+dt\s*=\s*([\d.eE+-]+)"
    r"\s+M\s*=\s*([\d.eE+-]+)\s+E\s*=\s*([\d.eE+-]+)"
)
_FINAL_RE = re.compile(
    r"Final:\s+step\s+(\d+)\s+t\s*=\s*([\d.eE+-]+)"
    r"\s+M\s*=\s*([\d.eE+-]+)\s+E\s*=\s*([\d.eE+-]+)"
)


def parse_stdout(text):
    records = []
    for m in _STEP_RE.finditer(text):
        records.append({
            "step": int(m.group(1)),
            "t": float(m.group(2)),
            "dt": float(m.group(3)),
            "mass": float(m.group(4)),
            "energy": float(m.group(5)),
        })
    m = _FINAL_RE.search(text)
    if m:
        records.append({
            "step": int(m.group(1)),
            "t": float(m.group(2)),
            "dt": 0.0,
            "mass": float(m.group(3)),
            "energy": float(m.group(4)),
        })
    return records


# ── Runner fixture ──────────────────────────────────────────────────

class StellarRun:
    """Result of a stellar2d run."""

    def __init__(self, workdir, stdout, stderr, returncode):
        self.workdir = workdir
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self._diagnostics = None

    @property
    def diagnostics(self):
        if self._diagnostics is None:
            self._diagnostics = parse_stdout(self.stdout)
        return self._diagnostics

    def vtk(self, name="output_final.vtk"):
        direct = os.path.join(self.workdir, name)
        if os.path.exists(direct):
            return read_vtk(direct)
        runs_dir = os.path.join(self.workdir, "runs")
        if os.path.isdir(runs_dir):
            for sub in sorted(os.listdir(runs_dir)):
                candidate = os.path.join(runs_dir, sub, name)
                if os.path.exists(candidate):
                    return read_vtk(candidate)
        return read_vtk(direct)

    @property
    def mass_drift(self):
        d = self.diagnostics
        if len(d) < 2:
            return 0.0
        M0 = d[0]["mass"]
        return max(abs((r["mass"] - M0) / M0) for r in d)

    @property
    def energy_drift(self):
        d = self.diagnostics
        if len(d) < 2:
            return 0.0
        E0 = d[0]["energy"]
        if abs(E0) < 1e-30:
            return 0.0
        return max(abs((r["energy"] - E0) / E0) for r in d)


@pytest.fixture(scope="session")
def stellar2d_bin():
    return _find_binary()


def run_stellar2d(binary, args, timeout=120):
    tmpdir = tempfile.mkdtemp(prefix="stellar2d_test_")
    cmd = [binary] + args
    result = subprocess.run(
        cmd, cwd=tmpdir, capture_output=True, text=True, timeout=timeout
    )
    return StellarRun(tmpdir, result.stdout, result.stderr, result.returncode)


@pytest.fixture(scope="session")
def run(stellar2d_bin):
    def _run(args, timeout=120):
        return run_stellar2d(stellar2d_bin, args, timeout)
    return _run


# ── Lane-Emden analytic solution ────────────────────────────────────

def solve_lane_emden(n_poly=1.5, dxi=0.001):
    xi_arr, theta_arr = [0.0], [1.0]
    xi, theta, dtheta = 1e-10, 1.0, 0.0
    while theta > 0 and xi < 100:
        def f2(x, t, dt):
            if x < 1e-10:
                return -t / 3.0
            return -(t**n_poly if t > 0 else 0.0) - 2 * dt / x

        k1y1, k1y2 = dtheta, f2(xi, theta, dtheta)
        k2y1 = dtheta + 0.5 * dxi * k1y2
        k2y2 = f2(xi + 0.5 * dxi, theta + 0.5 * dxi * k1y1, dtheta + 0.5 * dxi * k1y2)
        k3y1 = dtheta + 0.5 * dxi * k2y2
        k3y2 = f2(xi + 0.5 * dxi, theta + 0.5 * dxi * k2y1, dtheta + 0.5 * dxi * k2y2)
        k4y1 = dtheta + dxi * k3y2
        k4y2 = f2(xi + dxi, theta + dxi * k3y1, dtheta + dxi * k3y2)
        theta += dxi / 6 * (k1y1 + 2 * k2y1 + 2 * k3y1 + k4y1)
        dtheta += dxi / 6 * (k1y2 + 2 * k2y2 + 2 * k3y2 + k4y2)
        xi += dxi
        xi_arr.append(xi)
        theta_arr.append(max(theta, 0.0))
        if theta <= 0:
            break
    return np.array(xi_arr), np.array(theta_arr), xi


# ── Baseline I/O ────────────────────────────────────────────────────

def load_baseline(name):
    path = os.path.join(BASELINE_DIR, f"{name}.json")
    if not os.path.exists(path):
        pytest.skip(f"Baseline {name}.json not found — run pytest --update-baselines")
    with open(path) as f:
        return json.load(f)


def save_baseline(name, data):
    os.makedirs(BASELINE_DIR, exist_ok=True)
    path = os.path.join(BASELINE_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def pytest_addoption(parser):
    parser.addoption("--update-baselines", action="store_true", default=False,
                     help="Re-generate regression baselines instead of comparing.")
