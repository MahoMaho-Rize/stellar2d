"""Minimal test utilities for stellar2d pytest tst/ framework.

Design goals (Athena++ / AthenaK inspired, but GPU-only, no MPI):
  - Every test function owns a ``run_base`` under ``tst/bin/<testname>/``.
  - ``run_stellar2d()`` shells out to the compiled binary, captures output.
  - ``read_error_dat()`` parses the ``# schema: ...`` line and returns a dict
    keyed by schema column name — never drift if a new column is added.
  - No analytic solvers in Python. C++ ``compute_*_error`` owns the error.

Keep this file thin.  If a helper grows past ~20 lines, move it to its own
module under ``tst/``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent
BIN_DEFAULT = REPO / "build" / "stellar2d"


def stellar2d_binary() -> Path:
    """Path to the compiled binary, overridable via STELLAR2D_BIN env."""
    override = os.environ.get("STELLAR2D_BIN")
    if override:
        return Path(override)
    return BIN_DEFAULT


def fresh_run_base(test_name: str) -> Path:
    """Return ``tst/bin/<test_name>/``, scrubbed of any prior contents."""
    path = REPO / "tst" / "bin" / test_name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_stellar2d(args: list[str], run_base: Path,
                  env: dict[str, str] | None = None,
                  timeout: float = 120.0) -> subprocess.CompletedProcess:
    """Invoke stellar2d with ``args`` under CWD == run_base.

    Always passes ``--run-base <run_base>`` so outputs (CSV / VTK / .dat)
    land inside the isolated directory.  Merges stderr into stdout and
    returns the CompletedProcess; raises on non-zero exit.
    """
    bin_path = stellar2d_binary()
    if not bin_path.exists():
        raise RuntimeError(
            f"stellar2d binary not found at {bin_path}. "
            f"Build first (cmake + make) or set STELLAR2D_BIN.")
    # Tier B-1: stellar2d now requires a "run" subcommand before flags.
    # See docs/design/cli_unification_plan_2026-05-09.md.
    full_args = [str(bin_path), "run"] + args + ["--run-base", str(run_base)]
    cp = subprocess.run(
        full_args,
        cwd=run_base,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if cp.returncode != 0:
        msg = (f"stellar2d failed (rc={cp.returncode})\n"
               f"  cmd: {' '.join(full_args)}\n"
               f"  stdout tail:\n{cp.stdout[-800:]}\n"
               f"  stderr tail:\n{cp.stderr[-800:]}\n")
        raise RuntimeError(msg)
    return cp


def find_error_dat(run_base: Path, name: str) -> Path:
    """Locate ``<name>-errors.dat`` under ``run_base`` — it may live either
    directly in run_base or in a timestamped subdirectory the solver
    created (e.g. ``entropy_wave_128x128_YYYYMMDD_HHMMSS/``).

    Raises if zero or >1 matches are found.  Returns the single .dat path.
    """
    candidates = list(run_base.rglob(f"{name}-errors.dat"))
    if not candidates:
        raise FileNotFoundError(
            f"no {name}-errors.dat under {run_base}. Did --compute-error "
            f"make it into the argv? Check the solver's run.log.")
    if len(candidates) > 1:
        raise RuntimeError(
            f"multiple {name}-errors.dat under {run_base}: {candidates}")
    return candidates[0]


def read_error_dat(path: Path) -> list[dict[str, Any]]:
    """Parse a ``*-errors.dat`` file emitted by a solver's compute_*_error.

    Expects the first line to be ``# schema: col1 col2 ...`` — each data
    row is split on whitespace and zipped against the schema names.
    Columns whose name is a small integer list are parsed as ints; the
    rest as floats.  Returns one dict per data row.
    """
    INT_FIELDS = {"Nx", "Ny", "Ncycle", "k"}
    rows: list[dict[str, Any]] = []
    schema: list[str] | None = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("# schema:"):
            schema = line.removeprefix("# schema:").split()
            continue
        if line.startswith("#"):
            continue
        if schema is None:
            raise RuntimeError(f"{path}: data row before schema line")
        toks = line.split()
        if len(toks) != len(schema):
            raise RuntimeError(
                f"{path}: row has {len(toks)} cols, schema has {len(schema)}")
        row: dict[str, Any] = {}
        for name, tok in zip(schema, toks):
            if name in INT_FIELDS:
                row[name] = int(tok)
            else:
                row[name] = float(tok)
        rows.append(row)
    return rows
