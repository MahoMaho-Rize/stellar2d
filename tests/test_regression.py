"""
Golden-value regression tests.

Compare the final state (mass, energy, density extrema, velocity extrema)
against stored baseline values. This catches any unintentional change in
solver output.

Run with --update-baselines to regenerate baselines from the current binary.
"""
import numpy as np
import pytest

from conftest import load_baseline, save_baseline

CASES = [
    {
        "name": "lane_emden_16x8",
        "args": ["--test", "lane_emden", "--nr", "16", "--ntheta", "8",
                 "--tend", "0.01", "--output-interval", "5"],
    },
    {
        "name": "sedov_16x8",
        "args": ["--test", "sedov", "--nr", "16", "--ntheta", "8",
                 "--tend", "0.002", "--output-interval", "5"],
    },
    {
        "name": "jeans_16x8",
        "args": ["--test", "jeans", "--nr", "16", "--ntheta", "8",
                 "--tend", "0.01", "--output-interval", "5"],
    },
    {
        "name": "evrard_16x8",
        "args": ["--test", "evrard", "--nr", "16", "--ntheta", "8",
                 "--tend", "0.002", "--output-interval", "5"],
    },
]


def _extract_fingerprint(result):
    vtk = result.vtk()
    d = result.diagnostics
    final = d[-1] if d else {}
    return {
        "final_mass": final.get("mass"),
        "final_energy": final.get("energy"),
        "final_step": final.get("step"),
        "rho_min": float(np.min(vtk["density"])),
        "rho_max": float(np.max(vtk["density"])),
        "P_min": float(np.min(vtk["pressure"])),
        "P_max": float(np.max(vtk["pressure"])),
        "phi_min": float(np.min(vtk["phi"])),
        "phi_max": float(np.max(vtk["phi"])),
        "v_max": float(np.max(np.sqrt(
            vtk["velocity"][:, :, 0]**2 + vtk["velocity"][:, :, 2]**2
        ))),
    }


@pytest.fixture(scope="module", params=CASES, ids=[c["name"] for c in CASES])
def regression_case(request, run):
    case = request.param
    result = run(case["args"])
    assert result.returncode == 0, result.stderr
    return case["name"], result


class TestRegression:

    RTOL = 1e-8

    def test_golden_values(self, regression_case, request):
        name, result = regression_case
        fp = _extract_fingerprint(result)

        if request.config.getoption("--update-baselines"):
            save_baseline(name, fp)
            pytest.skip(f"Baseline {name} updated")

        baseline = load_baseline(name)

        for key in baseline:
            got = fp[key]
            expected = baseline[key]
            if expected is None or got is None:
                continue
            if expected == 0:
                assert abs(got) < 1e-15, f"{name}/{key}: expected 0, got {got}"
            else:
                rel = abs(got - expected) / abs(expected)
                assert rel < self.RTOL, \
                    f"{name}/{key}: {got} vs baseline {expected} (rel={rel:.2e})"
