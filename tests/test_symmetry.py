"""
Symmetry tests.

For spherically-symmetric initial conditions (Lane-Emden, Sedov),
the IC should be perfectly θ-independent.  The evolved solution breaks
symmetry due to discretization on a structured (r,θ) grid, but this
breakage should converge away with resolution.

Fast tests check IC symmetry only.  Evolved convergence is in test_convergence.
"""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def lane_emden_ic(run):
    return run(["--test", "lane_emden", "--nr", "32", "--ntheta", "16",
                "--tend", "0.0001", "--output-interval", "999"])


@pytest.fixture(scope="module")
def sedov_ic(run):
    return run(["--test", "sedov", "--nr", "32", "--ntheta", "16",
                "--tend", "0.0001", "--output-interval", "999"])


@pytest.fixture(scope="module")
def jeans_ic(run):
    return run(["--test", "jeans", "--nr", "32", "--ntheta", "16",
                "--tend", "0.0001", "--output-interval", "999"])


def _theta_variation(vtk, field, rho_floor=1e-6):
    """Return max relative θ-variation per radial shell, for cells above floor."""
    arr = vtk[field]
    nr = arr.shape[0]
    variations = []
    for i in range(nr):
        row = arr[i, :]
        mean = np.mean(row)
        if abs(mean) < rho_floor:
            continue
        variations.append(np.max(np.abs(row - mean)) / abs(mean))
    return np.max(variations) if variations else 0.0


class TestICSymmetry:
    """Initial conditions should be perfectly θ-independent for 1D problems."""

    def test_lane_emden_runs(self, lane_emden_ic):
        assert lane_emden_ic.returncode == 0, lane_emden_ic.stderr

    def test_lane_emden_density_ic(self, lane_emden_ic):
        vtk = lane_emden_ic.vtk("output_0000.vtk")
        var = _theta_variation(vtk, "density")
        assert var < 1e-14, f"IC density θ-variation = {var:.3e}"

    def test_lane_emden_pressure_ic(self, lane_emden_ic):
        vtk = lane_emden_ic.vtk("output_0000.vtk")
        var = _theta_variation(vtk, "pressure")
        assert var < 1e-14, f"IC pressure θ-variation = {var:.3e}"

    def test_lane_emden_phi_ic(self, lane_emden_ic):
        vtk = lane_emden_ic.vtk("output_0000.vtk")
        var = _theta_variation(vtk, "phi")
        assert var < 1e-6, f"IC phi θ-variation = {var:.3e}"

    def test_lane_emden_zero_velocity_ic(self, lane_emden_ic):
        vtk = lane_emden_ic.vtk("output_0000.vtk")
        v = vtk["velocity"]
        v_mag = np.sqrt(v[:, :, 0]**2 + v[:, :, 2]**2)
        assert np.max(v_mag) < 1e-14, f"IC max |v| = {np.max(v_mag):.3e}"

    def test_sedov_density_ic(self, sedov_ic):
        assert sedov_ic.returncode == 0
        vtk = sedov_ic.vtk("output_0000.vtk")
        var = _theta_variation(vtk, "density")
        assert var < 1e-14, f"Sedov IC density θ-variation = {var:.3e}"

    def test_jeans_has_theta_dependence(self, jeans_ic):
        """Jeans IC has cos(k_theta * theta) perturbation — NOT θ-symmetric."""
        assert jeans_ic.returncode == 0
        vtk = jeans_ic.vtk("output_0000.vtk")
        var = _theta_variation(vtk, "density")
        assert var > 1e-6, "Jeans IC should have θ-dependent perturbation"
