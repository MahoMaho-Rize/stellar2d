"""
Conservation tests: mass and energy should be conserved (or bounded) for all test cases.

Each test runs a short simulation at low resolution and checks that the relative
drift in total mass and total energy stays below a threshold.

Mass is not exactly conserved because the outer boundary uses outflow BCs
which allow mass flux through the domain boundary.  The thresholds here
reflect the expected leakage for each test configuration.
"""
import pytest


@pytest.fixture(scope="module")
def lane_emden_run(run):
    return run(["--test", "lane_emden", "--nr", "32", "--ntheta", "16",
                "--tend", "0.01", "--output-interval", "5"])


@pytest.fixture(scope="module")
def sedov_run(run):
    return run(["--test", "sedov", "--nr", "32", "--ntheta", "16",
                "--tend", "0.005", "--output-interval", "5"])


@pytest.fixture(scope="module")
def jeans_run(run):
    return run(["--test", "jeans", "--nr", "32", "--ntheta", "16",
                "--tend", "0.01", "--output-interval", "5"])


@pytest.fixture(scope="module")
def evrard_run(run):
    return run(["--test", "evrard", "--nr", "32", "--ntheta", "16",
                "--tend", "0.005", "--output-interval", "5"])


class TestMassConservation:
    """Mass should stay bounded.  Outflow BCs leak, so we allow small drift."""

    def test_lane_emden(self, lane_emden_run):
        assert lane_emden_run.returncode == 0, lane_emden_run.stderr
        assert lane_emden_run.mass_drift < 1e-4

    def test_sedov(self, sedov_run):
        assert sedov_run.returncode == 0, sedov_run.stderr
        assert sedov_run.mass_drift < 1e-3

    def test_jeans(self, jeans_run):
        assert jeans_run.returncode == 0, jeans_run.stderr
        assert jeans_run.mass_drift < 1e-3

    def test_evrard(self, evrard_run):
        assert evrard_run.returncode == 0, evrard_run.stderr
        assert evrard_run.mass_drift < 1e-2


class TestEnergyConservation:
    """Total energy (KE + thermal + gravitational) should be bounded.

    For explicit RK2 + self-gravity the discrete energy is not exactly
    conserved, but drift should be small at low CFL.
    """

    def test_lane_emden(self, lane_emden_run):
        assert lane_emden_run.returncode == 0
        assert lane_emden_run.energy_drift < 5e-2

    def test_sedov(self, sedov_run):
        assert sedov_run.returncode == 0
        assert sedov_run.energy_drift < 0.1

    def test_jeans(self, jeans_run):
        assert jeans_run.returncode == 0
        assert jeans_run.energy_drift < 5e-2

    def test_evrard(self, evrard_run):
        assert evrard_run.returncode == 0
        assert evrard_run.energy_drift < 0.2


class TestRunExits:
    """Binary should exit cleanly."""

    def test_lane_emden_exit(self, lane_emden_run):
        assert lane_emden_run.returncode == 0

    def test_sedov_exit(self, sedov_run):
        assert sedov_run.returncode == 0

    def test_jeans_exit(self, jeans_run):
        assert jeans_run.returncode == 0

    def test_evrard_exit(self, evrard_run):
        assert evrard_run.returncode == 0
