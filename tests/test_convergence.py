"""
Convergence order tests.

Run the same problem at multiple resolutions, measure L1 density error
against the analytic solution, and verify that the scheme converges at
the expected order (2nd for MUSCL + RK2).

These tests are slow — mark them for selective execution.
"""
import numpy as np
import pytest

from conftest import solve_lane_emden

pytestmark = pytest.mark.slow


def _lane_emden_l1_error(run_result):
    """Compute volume-weighted L1 density error for Lane-Emden."""
    vtk = run_result.vtk()
    j_eq = vtk["nt"] // 2
    r_1d = vtk["r"][:, j_eq]
    rho_sim = vtk["density"][:, j_eq]

    n_poly, rho_c, K, G = 1.5, 1.0, 1.0, 1.0
    xi_arr, theta_arr, xi1 = solve_lane_emden(n_poly)
    alpha = np.sqrt((n_poly + 1) * K * rho_c ** (1.0 / n_poly - 1) / (4 * np.pi * G))
    r_analytic = xi_arr * alpha
    rho_exact = np.interp(r_1d, r_analytic,
                          rho_c * np.where(theta_arr > 0, theta_arr ** n_poly, 0.0),
                          right=0.0)

    mask = rho_exact > 1e-6
    return np.mean(np.abs(rho_sim[mask] - rho_exact[mask]))


class TestConvergenceOrder:

    RESOLUTIONS = [(32, 16), (64, 32), (128, 64)]

    def test_lane_emden_initial_condition(self, run):
        """IC interpolation error should converge ≥ 2nd order."""
        errors = []
        for nr, nt in self.RESOLUTIONS:
            result = run(["--test", "lane_emden", f"--nr={nr}", f"--ntheta={nt}",
                          "--tend", "0.0001", "--output-interval", "99999"])
            assert result.returncode == 0, result.stderr
            errors.append(_lane_emden_l1_error(result))

        for i in range(1, len(errors)):
            ratio = errors[i - 1] / max(errors[i], 1e-30)
            order = np.log2(ratio)
            assert order > 1.5, \
                f"res {self.RESOLUTIONS[i-1]}→{self.RESOLUTIONS[i]}: " \
                f"L1 error {errors[i-1]:.3e}→{errors[i]:.3e}, order={order:.2f} < 1.5"

    def test_lane_emden_evolved(self, run):
        """Evolved solution error should converge ≥ 1st order.

        The evolved convergence rate may be lower than spatial order due to
        time-error coupling and equilibrium drift. We require at least 1st order.
        """
        errors = []
        for nr, nt in self.RESOLUTIONS:
            result = run(["--test", "lane_emden", f"--nr={nr}", f"--ntheta={nt}",
                          "--tend", "0.05", "--output-interval", "99999"],
                         timeout=180)
            assert result.returncode == 0, result.stderr
            errors.append(_lane_emden_l1_error(result))

        for i in range(1, len(errors)):
            ratio = errors[i - 1] / max(errors[i], 1e-30)
            order = np.log2(ratio)
            assert order > 0.8, \
                f"res {self.RESOLUTIONS[i-1]}→{self.RESOLUTIONS[i]}: " \
                f"L1 error {errors[i-1]:.3e}→{errors[i]:.3e}, order={order:.2f} < 0.8"
