"""pytest-level fixtures for the stellar2d tst/ suite.

Keep this small — bespoke fixtures live in the per-solver subdir if needed.
"""
from __future__ import annotations

import pytest

from testutils import stellar2d_binary


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "fast: short test (< 30 s), runs in default CI / ctest -L fast")
    config.addinivalue_line(
        "markers",
        "slow: multi-minute test, opt-in via `-m slow`")
    config.addinivalue_line(
        "markers",
        "scan: hours-scale parameter sweep, always manual")


@pytest.fixture(scope="session", autouse=True)
def _require_stellar2d_binary():
    """Fail the whole session loudly if the binary is missing — no test
    should silently skip because of a build issue."""
    bin_path = stellar2d_binary()
    if not bin_path.exists():
        pytest.fail(
            f"stellar2d binary missing at {bin_path}. "
            f"Build the project first (cd build && make) or set "
            f"STELLAR2D_BIN to an existing path.",
            pytrace=False)
