"""
Hydrostatic equilibrium integration test (black-box).

Runs the full binary and checks that the Lane-Emden solution doesn't
blow up entirely. The proper HSE residual test is in test_exact.cpp
(Layer 2, C++). This Python test is a lightweight regression guard.
"""
import numpy as np
import pytest

from conftest import solve_lane_emden


@pytest.fixture(scope="module")
def hse_run(run):
    return run(["--test", "lane_emden", "--nr", "32", "--ntheta", "16",
                "--tend", "0.001", "--output-interval", "5"])


def _analytic_density(r_1d):
    n_poly, rho_c, K, G = 1.5, 1.0, 1.0, 1.0
    xi_arr, theta_arr, xi1 = solve_lane_emden(n_poly)
    alpha = np.sqrt((n_poly + 1) * K * rho_c ** (1.0 / n_poly - 1) / (4 * np.pi * G))
    r_analytic = xi_arr * alpha
    rho_analytic = rho_c * np.where(theta_arr > 0, theta_arr ** n_poly, 0.0)
    return np.interp(r_1d, r_analytic, rho_analytic, right=0.0)


class TestHSEStability:

    def test_run_succeeds(self, hse_run):
        assert hse_run.returncode == 0, hse_run.stderr

    def test_central_density_preserved(self, hse_run):
        """Central density should stay within 50% of initial."""
        vtk = hse_run.vtk()
        j_eq = vtk["nt"] // 2
        rho_center = vtk["density"][0, j_eq]
        assert rho_center > 0.5, f"central ρ = {rho_center:.3e}, expected ~1.0"
        assert rho_center < 2.0, f"central ρ = {rho_center:.3e}, blew up"

    def test_no_nan(self, hse_run):
        """No NaN in output."""
        vtk = hse_run.vtk()
        assert not np.any(np.isnan(vtk["density"]))
        assert not np.any(np.isnan(vtk["pressure"]))
        assert not np.any(np.isnan(vtk["phi"]))

    def test_positive_density(self, hse_run):
        """Density should remain positive everywhere."""
        vtk = hse_run.vtk()
        assert np.all(vtk["density"] > 0), "negative density detected"
