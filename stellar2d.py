#!/usr/bin/env python3
"""stellar2d -- unified CLI entry point.

Usage:
    ./stellar2d.py build   [--gpu] [--amgx-dir PATH] [--debug]
    ./stellar2d.py run     [--test CASE] [--solver SOLVER] ... (simulation flags)
    ./stellar2d.py plot    <run_dir> [--gif]
    ./stellar2d.py animate <run_dir> [-o FILE] [--skip N] [--fps N]
    ./stellar2d.py mach    <run_dir> [--plot]
    ./stellar2d.py viewer  [--port PORT]
    ./stellar2d.py test    [--gpu] [--slow]
    ./stellar2d.py clean
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIXI_ENV = ROOT / ".pixi" / "envs" / "default"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _pixi_available():
    """Check if pixi environment is installed."""
    return (PIXI_ENV / "bin").is_dir()


def _pixi_wrap(cmd):
    """Wrap a command to run inside pixi environment if available."""
    if _pixi_available():
        return ["pixi", "run"] + cmd
    return cmd


def _run(cmd, *, cwd=None, check=True, capture=False, pixi=False, **kw):
    """Run a shell command with pretty printing.

    If pixi=True and a pixi env exists, the command is wrapped with
    ``pixi run`` so that the correct toolchain (nvcc, cmake, etc.) is
    on PATH.
    """
    if pixi:
        cmd = _pixi_wrap(cmd)
    label = " ".join(str(c) for c in cmd)
    print(f"\033[36m> {label}\033[0m")
    return subprocess.run(
        cmd,
        cwd=cwd or ROOT,
        check=check,
        capture_output=capture,
        text=True if capture else None,
        **kw,
    )


def _find_binary():
    """Locate the stellar2d binary (GPU build preferred)."""
    for d in ["build", "build-cpu"]:
        p = ROOT / d / "stellar2d"
        if p.is_file():
            return p
    return None


def _nproc():
    return os.cpu_count() or 4


# ---------------------------------------------------------------------------
# sub-commands
# ---------------------------------------------------------------------------


def cmd_build(args):
    """Configure and build the C++ simulation binary."""
    build_dir = ROOT / ("build" if args.gpu else "build-cpu")
    build_dir.mkdir(exist_ok=True)

    cmake_args = [
        "cmake",
        str(ROOT),
        f"-DCMAKE_BUILD_TYPE={'Debug' if args.debug else 'Release'}",
    ]
    if args.gpu:
        cmake_args.append("-DUSE_GPU=ON")
        cmake_args.append("-DCMAKE_CUDA_ARCHITECTURES=89")
    if args.amgx_dir:
        cmake_args.append(f"-DAMGX_DIR={args.amgx_dir}")

    _run(cmake_args, cwd=build_dir, pixi=True)
    _run(["cmake", "--build", ".", f"-j{_nproc()}"], cwd=build_dir, pixi=True)
    print(f"\n\033[32mBuild complete: {build_dir / 'stellar2d'}\033[0m")


def cmd_run(args):
    """Run a simulation."""
    binary = _find_binary()
    if binary is None:
        print("Error: stellar2d binary not found. Run `./stellar2d.py build` first.")
        sys.exit(1)

    cmd = [str(binary), "run"]
    cmd += ["--test", args.test]
    cmd += ["--nr", str(args.nr)]
    cmd += ["--ntheta", str(args.ntheta)]
    cmd += ["--tend", str(args.tend)]
    cmd += ["--cfl", str(args.cfl)]
    cmd += ["--output-interval", str(args.output_interval)]
    if args.solver:
        cmd += ["--solver", args.solver]
    if args.precond:
        cmd += ["--precond", args.precond]
    if args.mesh:
        cmd += ["--mesh", args.mesh]

    _run(cmd)


def cmd_plot(args):
    """Render VTK frames as PNG (density + velocity + Mach)."""
    script = ROOT / "scripts" / "plot_frames.py"
    cmd = [sys.executable, str(script), args.run_dir]
    if args.gif:
        cmd.append("--gif")
    _run(cmd)


def cmd_animate(args):
    """Generate a publication-quality GIF animation."""
    script = ROOT / "scripts" / "animate_evolution.py"
    cmd = [sys.executable, str(script), args.run_dir, args.output, str(args.skip)]
    _run(cmd)


def cmd_mach(args):
    """Analyze Mach number evolution across frames."""
    script = ROOT / "scripts" / "check_mach.py"
    cmd = [sys.executable, str(script), args.run_dir]
    if args.plot:
        cmd.append("--plot")
    _run(cmd)


def cmd_viewer(args):
    """Start the React + Three.js web viewer (dev server)."""
    fe = ROOT / "frontend"
    if not (fe / "node_modules").is_dir():
        print("Installing frontend dependencies...")
        _run(["pnpm", "install"], cwd=fe)
    _run(["pnpm", "dev", "--port", str(args.port)], cwd=fe)


def cmd_test(args):
    """Run the test suite (C++ and/or Python)."""
    ran = False

    # C++ tests
    build_dir = ROOT / ("build" if args.gpu else "build-cpu")
    for name in ["test_unit", "test_exact", "test_pitfalls"]:
        binary = build_dir / name
        if binary.is_file():
            _run([str(binary)], check=False, pixi=True)
            ran = True

    if args.gpu:
        for name in ["test_lowmach", "test_mini"]:
            binary = build_dir / name
            if binary.is_file():
                _run([str(binary)], check=False, pixi=True)
                ran = True

    # Python tests
    pytest_cmd = [sys.executable, "-m", "pytest", "-v"]
    if not args.slow:
        pytest_cmd += ["-m", "not slow"]
    _run(pytest_cmd, check=False, pixi=True)
    ran = True

    if not ran:
        print("No test binaries found. Run `./stellar2d.py build` first.")
        sys.exit(1)


def cmd_clean(args):
    """Remove build artifacts."""
    for d in ["build", "build-cpu"]:
        p = ROOT / d
        if p.is_dir():
            print(f"Removing {p}")
            shutil.rmtree(p)
    print("Clean complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        prog="stellar2d.py",
        description="stellar2d: 2D axisymmetric stellar simulation toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  ./stellar2d.py build --gpu                    Build with CUDA support
  ./stellar2d.py run --test bubble --solver fas  Run bubble convection (FAS)
  ./stellar2d.py run --solver lowmach --nr 64   Run low-Mach solver
  ./stellar2d.py plot runs/my_run --gif         Render frames + GIF
  ./stellar2d.py animate runs/my_run            Generate evolution GIF
  ./stellar2d.py mach runs/my_run --plot        Mach number analysis
  ./stellar2d.py viewer                         Start web viewer
  ./stellar2d.py test                           Run test suite

pixi integration:
  If a pixi environment is detected (.pixi/envs/default/), build and
  test commands automatically run inside it (nvcc, cmake, python, etc.).
  To set up:  pixi install
""",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # -- build --
    sp = sub.add_parser("build", help="Configure and compile the C++ binary")
    sp.add_argument(
        "--gpu", action="store_true", help="Enable CUDA GPU solvers (FAS, LowMach)"
    )
    sp.add_argument(
        "--amgx-dir",
        default=None,
        help="Path to AmgX install (enables compressible GPU solver)",
    )
    sp.add_argument(
        "--debug", action="store_true", help="Debug build (-O0, sanitizers)"
    )
    sp.set_defaults(func=cmd_build)

    # -- run --
    sp = sub.add_parser("run", help="Run a simulation")
    sp.add_argument(
        "--test",
        default="lane_emden",
        choices=[
            "lane_emden",
            "lane_emden_perturbed",
            "bubble",
            "sedov",
            "jeans",
            "evrard",
        ],
        help="Test case / initial condition (default: lane_emden)",
    )
    sp.add_argument(
        "--solver",
        default=None,
        choices=["compressible", "lowmach", "fas"],
        help="Solver backend (default: compressible; lowmach/fas require GPU build)",
    )
    sp.add_argument(
        "--precond",
        default=None,
        choices=[
            "none",
            "block_jacobi",
            "simple",
            "line_jacobi",
            "block_schur",
            "combined",
            "pbp",
        ],
        help="Preconditioner for implicit solvers (default: line_jacobi)",
    )
    sp.add_argument(
        "--mesh",
        default=None,
        choices=["log", "equimass"],
        help="Mesh type (default: log)",
    )
    sp.add_argument("--nr", type=int, default=128, help="Radial cells (default: 128)")
    sp.add_argument("--ntheta", type=int, default=64, help="Polar cells (default: 64)")
    sp.add_argument("--tend", type=float, default=1.0, help="End time (default: 1.0)")
    sp.add_argument("--cfl", type=float, default=0.4, help="CFL number (default: 0.4)")
    sp.add_argument(
        "--output-interval",
        type=int,
        default=100,
        help="Steps between VTK snapshots (default: 100)",
    )
    sp.set_defaults(func=cmd_run)

    # -- plot --
    sp = sub.add_parser("plot", help="Render VTK frames as PNG (3-panel)")
    sp.add_argument("run_dir", help="Path to run directory with VTK files")
    sp.add_argument("--gif", action="store_true", help="Also assemble frames into GIF")
    sp.set_defaults(func=cmd_plot)

    # -- animate --
    sp = sub.add_parser("animate", help="Generate publication-quality evolution GIF")
    sp.add_argument("run_dir", help="Path to run directory with VTK files")
    sp.add_argument(
        "-o",
        "--output",
        default="evolution.gif",
        help="Output GIF path (default: evolution.gif)",
    )
    sp.add_argument(
        "--skip", type=int, default=2, help="Frame skip interval (default: 2)"
    )
    sp.add_argument("--fps", type=int, default=12, help="Animation FPS (default: 12)")
    sp.set_defaults(func=cmd_animate)

    # -- mach --
    sp = sub.add_parser("mach", help="Analyze Mach number evolution")
    sp.add_argument("run_dir", help="Path to run directory with VTK files")
    sp.add_argument("--plot", action="store_true", help="Save mach_history.png plot")
    sp.set_defaults(func=cmd_mach)

    # -- viewer --
    sp = sub.add_parser("viewer", help="Start React + Three.js web viewer")
    sp.add_argument(
        "--port", type=int, default=5173, help="Dev server port (default: 5173)"
    )
    sp.set_defaults(func=cmd_viewer)

    # -- test --
    sp = sub.add_parser("test", help="Run C++ and Python test suites")
    sp.add_argument("--gpu", action="store_true", help="Also run GPU-only tests")
    sp.add_argument(
        "--slow", action="store_true", help="Include slow convergence tests"
    )
    sp.set_defaults(func=cmd_test)

    # -- clean --
    sp = sub.add_parser("clean", help="Remove build directories")
    sp.set_defaults(func=cmd_clean)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
