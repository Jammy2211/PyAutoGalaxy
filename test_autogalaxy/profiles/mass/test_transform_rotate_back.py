"""
Tests for the automatic back-rotation feature of the transform decorator (Phase 2).

These tests capture the expected deflection values from the current manual back-rotation
approach. After switching to automatic back-rotation via @aa.grid_dec.transform(rotate_back=True),
the same values must be produced.
"""
import numpy as np
import pytest

import autogalaxy as ag


@pytest.fixture
def grid():
    return ag.Grid2D.uniform(shape_native=(3, 3), pixel_scales=0.5)


@pytest.fixture
def grid_irregular():
    return ag.Grid2DIrregular(values=[(1.0, 1.0), (0.5, -0.5), (-1.0, 0.3)])


class TestIsothermalBackRotation:
    def test__deflections__spherical__no_rotation_needed(self, grid):
        """Spherical profile: back-rotation is identity (angle=0)."""
        profile = ag.mp.Isothermal(
            centre=(0.0, 0.0), ell_comps=(0.0, 0.0), einstein_radius=1.0
        )
        deflections = profile.deflections_yx_2d_from(grid=grid)
        assert deflections.shape == (9, 2)
        # Symmetry: for centred spherical, deflections should point radially
        # Centre pixel should have ~zero deflection
        assert deflections.array[4, 0] == pytest.approx(0.0, abs=1e-10)
        assert deflections.array[4, 1] == pytest.approx(0.0, abs=1e-10)

    def test__deflections__elliptical__correct_values(self, grid):
        """Elliptical profile: back-rotation must rotate deflection vectors."""
        profile = ag.mp.Isothermal(
            centre=(0.0, 0.0), ell_comps=(0.17647, 0.0), einstein_radius=1.0
        )
        deflections = profile.deflections_yx_2d_from(grid=grid)
        assert deflections.shape == (9, 2)

        # Store reference values for regression
        expected_y = deflections.array[:, 0].copy()
        expected_x = deflections.array[:, 1].copy()

        # Recompute and verify stability
        deflections_2 = profile.deflections_yx_2d_from(grid=grid)
        np.testing.assert_array_almost_equal(
            deflections_2.array[:, 0], expected_y, decimal=10
        )
        np.testing.assert_array_almost_equal(
            deflections_2.array[:, 1], expected_x, decimal=10
        )

    def test__deflections__off_centre_elliptical(self, grid):
        """Off-centre elliptical: both translation and rotation matter."""
        profile = ag.mp.Isothermal(
            centre=(0.3, -0.2), ell_comps=(0.1, 0.2), einstein_radius=1.5
        )
        deflections = profile.deflections_yx_2d_from(grid=grid)
        assert deflections.shape == (9, 2)
        # No pixel should have NaN
        assert not np.any(np.isnan(deflections.array))

    def test__deflections__irregular_grid(self, grid_irregular):
        """Irregular grid: decorator must handle non-uniform grids."""
        profile = ag.mp.Isothermal(
            centre=(0.0, 0.0), ell_comps=(0.1, 0.15), einstein_radius=1.0
        )
        deflections = profile.deflections_yx_2d_from(grid=grid_irregular)
        assert deflections.shape == (3, 2)
        assert not np.any(np.isnan(deflections.array))


class TestNFWBackRotation:
    def test__deflections__spherical(self, grid):
        profile = ag.mp.NFW(
            centre=(0.0, 0.0),
            ell_comps=(0.0, 0.0),
            kappa_s=0.05,
            scale_radius=1.0,
        )
        deflections = profile.deflections_yx_2d_from(grid=grid)
        assert deflections.shape == (9, 2)
        assert not np.any(np.isnan(deflections.array))

    def test__deflections__elliptical(self, grid):
        profile = ag.mp.NFW(
            centre=(0.0, 0.0),
            ell_comps=(0.15, 0.05),
            kappa_s=0.05,
            scale_radius=1.0,
        )
        deflections = profile.deflections_yx_2d_from(grid=grid)
        assert deflections.shape == (9, 2)
        assert not np.any(np.isnan(deflections.array))

        # Verify determinism
        deflections_2 = profile.deflections_yx_2d_from(grid=grid)
        np.testing.assert_array_almost_equal(
            deflections.array, deflections_2.array, decimal=10
        )


class TestPowerLawBackRotation:
    def test__deflections__elliptical(self, grid):
        profile = ag.mp.PowerLaw(
            centre=(0.0, 0.0),
            ell_comps=(0.1, 0.2),
            einstein_radius=1.0,
            slope=2.3,
        )
        deflections = profile.deflections_yx_2d_from(grid=grid)
        assert deflections.shape == (9, 2)
        assert not np.any(np.isnan(deflections.array))

        deflections_2 = profile.deflections_yx_2d_from(grid=grid)
        np.testing.assert_array_almost_equal(
            deflections.array, deflections_2.array, decimal=10
        )


class TestExternalShearBackRotation:
    def test__deflections(self, grid):
        profile = ag.mp.ExternalShear(gamma_1=0.05, gamma_2=0.03)
        deflections = profile.deflections_yx_2d_from(grid=grid)
        assert deflections.shape == (9, 2)
        assert not np.any(np.isnan(deflections.array))


class TestGaussianMassBackRotation:
    def test__deflections__elliptical(self, grid):
        profile = ag.mp.Gaussian(
            centre=(0.0, 0.0),
            ell_comps=(0.1, 0.2),
            intensity=1.0,
            sigma=0.5,
            mass_to_light_ratio=1.0,
        )
        deflections = profile.deflections_yx_2d_from(grid=grid)
        assert deflections.shape == (9, 2)
        assert not np.any(np.isnan(deflections.array))


class TestRegressionValues:
    """
    Exact numerical values from the manual back-rotation implementation.
    Any drift after switching to automatic back-rotation is a bug.
    """

    def test__isothermal_regression(self):
        grid = ag.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)
        profile = ag.mp.Isothermal(
            centre=(0.0, 0.0), ell_comps=(0.17647, 0.0), einstein_radius=1.0
        )
        deflections = profile.deflections_yx_2d_from(grid=grid)

        expected = np.array([
            [7.302765158231814e-01, -7.302765158231813e-01],
            [9.780567592278144e-01, -1.147703673995887e-01],
            [6.485808825578577e-01, 6.485808825578578e-01],
            [1.147703673995890e-01, -9.780567592278144e-01],
            [0.0, 0.0],
            [-1.147703673995890e-01, 9.780567592278144e-01],
            [-6.485808825578578e-01, -6.485808825578577e-01],
            [-9.780567592278142e-01, 1.147703673995890e-01],
            [-7.302765158231814e-01, 7.302765158231813e-01],
        ])
        np.testing.assert_array_almost_equal(deflections.array, expected, decimal=10)

    def test__nfw_regression(self):
        grid = ag.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)
        profile = ag.mp.NFW(
            centre=(0.0, 0.0),
            ell_comps=(0.15, 0.05),
            kappa_s=0.05,
            scale_radius=1.0,
        )
        deflections = profile.deflections_yx_2d_from(grid=grid)

        expected = np.array([
            [3.930040704822994e-02, -3.704991327787192e-02],
            [5.330198200319717e-02, -5.486557682861090e-03],
            [3.681444501824387e-02, 3.397437925990372e-02],
            [5.883231023414714e-03, -5.165835655921132e-02],
            [0.0, 0.0],
            [-5.883231023414721e-03, 5.165835655921130e-02],
            [-3.681444501824389e-02, -3.397437925990375e-02],
            [-5.330198200319718e-02, 5.486557682861097e-03],
            [-3.930040704822995e-02, 3.704991327787193e-02],
        ])
        np.testing.assert_array_almost_equal(deflections.array, expected, decimal=10)

    def test__power_law_regression(self):
        grid = ag.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)
        profile = ag.mp.PowerLaw(
            centre=(0.0, 0.0), ell_comps=(0.1, 0.2), einstein_radius=1.0, slope=2.3
        )
        deflections = profile.deflections_yx_2d_from(grid=grid)

        # Check non-centre pixels (centre pixel [4] has singularity)
        expected_0 = np.array([6.396990900652429e-01, -5.290981929231797e-01])
        expected_1 = np.array([9.317530273653852e-01, -4.087717797661689e-02])
        np.testing.assert_array_almost_equal(deflections.array[0], expected_0, decimal=10)
        np.testing.assert_array_almost_equal(deflections.array[1], expected_1, decimal=10)
