"""Shared VTK loader + last-frame helper for cart_ale2 standard tests."""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np

# Reuse the andrassy VTK parser — same cart_ale2 writer format.
_this = Path(__file__).resolve()
sys.path.insert(0, str(_this.parent.parent / "andrassy2022"))
from diagnose import parse_vtk  # noqa: E402


def latest_vtk(run_dir: str) -> Path:
    """Return the highest-numbered cart_ale2 frame.

    Skips `output_final.vtk` — that is written by the axisymmetric grid
    path in main.cpp and contains zeros for cart_ale2 (no mapping to the
    axisymmetric state struct). The driver-emitted frames are
    output_0000.vtk, output_0001.vtk, ..."""
    cand = sorted(glob.glob(f"{run_dir}/output_*.vtk"))
    cand = [c for c in cand if "final" not in c]
    if not cand:
        raise FileNotFoundError(f"no output_NNNN.vtk in {run_dir}")
    return Path(cand[-1])


def frame_time(vtk_path: Path, vtk_dt: float) -> float:
    """Extract numeric frame index and multiply by vtk_dt."""
    m = re.search(r"output_(\d+)\.vtk", vtk_path.name)
    if not m:
        return 0.0
    return (int(m.group(1)) + 1) * vtk_dt


def cell_centers(data: dict) -> tuple[np.ndarray, np.ndarray]:
    x_nodes = np.asarray(data["x_nodes"], dtype=float)
    y_nodes = np.asarray(data["y_nodes"], dtype=float)
    xc = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    yc = 0.5 * (y_nodes[:-1] + y_nodes[1:])
    return xc, yc


def pass_fail(name: str, value: float, tol: float, unit: str = "") -> bool:
    ok = abs(value) <= tol
    tag = "PASS" if ok else "FAIL"
    print(f"  {tag}  {name} = {value:.4e}{unit}  (tol {tol:.2e})")
    return ok
