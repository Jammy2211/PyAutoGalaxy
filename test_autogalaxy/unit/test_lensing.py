import typing

import numpy as np
import pytest

from pyquad import quad_grid
from skimage import measure

import autogalaxy as ag
from autoarray.structures import grids
from autogalaxy import lensing
from autogalaxy.profiles import geometry_profiles


class MockEllipticalIsothermal(
    geometry_profiles.EllipticalProfile, lensing.LensingObject
):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
    ):
        """
        Abstract class for elliptical mass profiles.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps : (float, float)
            The first and second ellipticity components of the elliptical coordinate system, where
            fac = (1 - axis_ratio) / (1 + axis_ratio), ellip_y = fac * sin(2*phi) and ellip_x = fac * cos(2*phi).
        """
        super(MockEllipticalIsothermal, self).__init__(
            centre=centre, elliptical_comps=elliptical_comps
        )
        self.einstein_radius = einstein_radius

    @property
    def einstein_radius_rescaled(self):
        """Rescale the einstein radius by slope and axis_ratio, to reduce its degeneracy with other mass-profiles
        parameters"""
        return (1.0 / (1 + self.axis_ratio)) * self.einstein_radius

    def convergence_func(self, grid_radius):
        return self.einstein_radius_rescaled * (grid_radius ** 2) ** (-0.5)

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def convergence_from_grid(self, grid):
        """ Calculate the projected convergence at a given set of arc-second gridded coordinates.

        The `grid_like_to_structure` decorator reshapes the ndarrays the convergence is outputted on. See \
        *ag.grid_like_to_structure* for a description of the output.

        Parameters
        ----------
        grid : ag.Grid2D
            The grid of (y,x) arc-second coordinates the convergence is computed on.

        """

        covnergence_grid = np.zeros(grid.shape[0])

        grid_eta = self.grid_to_elliptical_radii(grid)

        for i in range(grid.shape[0]):
            covnergence_grid[i] = self.convergence_func(grid_eta[i])

        return covnergence_grid

    @staticmethod
    def potential_func(u, y, x, axis_ratio):
        eta_u = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        return (
            (eta_u / u)
            * (eta_u) ** -1.0
            * eta_u
            / ((1 - (1 - axis_ratio ** 2) * u) ** 0.5)
        )

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def potential_from_grid(self, grid):
        """
        Calculate the potential at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : ag.Grid2D
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """

        potential_grid = quad_grid(
            self.potential_func, 0.0, 1.0, grid, args=(self.axis_ratio)
        )[0]

        return self.einstein_radius_rescaled * self.axis_ratio * potential_grid

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        For coordinates (0.0, 0.0) the analytic calculation of the deflection angle gives a NaN. Therefore, \
        coordinates at (0.0, 0.0) are shifted slightly to (1.0e-8, 1.0e-8).

        Parameters
        ----------
        grid : ag.Grid2D
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
        factor = (
            2.0
            * self.einstein_radius_rescaled
            * self.axis_ratio
            / np.sqrt(1 - self.axis_ratio ** 2)
        )

        psi = np.sqrt(
            np.add(
                np.multiply(self.axis_ratio ** 2, np.square(grid[:, 1])),
                np.square(grid[:, 0]),
            )
        )

        deflection_y = np.arctanh(
            np.divide(np.multiply(np.sqrt(1 - self.axis_ratio ** 2), grid[:, 0]), psi)
        )
        deflection_x = np.arctan(
            np.divide(np.multiply(np.sqrt(1 - self.axis_ratio ** 2), grid[:, 1]), psi)
        )
        return self.rotate_grid_from_profile(
            np.multiply(factor, np.vstack((deflection_y, deflection_x)).T)
        )

    @property
    def is_point_mass(self):
        return False

    @property
    def mass_profiles(self):
        return [self]

    @property
    def mass_profile_centres(self):
        return [self.centre]


class MockSphericalIsothermal(MockEllipticalIsothermal):
    def __init__(
        self,
        centre: typing.Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
    ):
        """
        Abstract class for elliptical mass profiles.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps : (float, float)
            The first and second ellipticity components of the elliptical coordinate system, where
            fac = (1 - axis_ratio) / (1 + axis_ratio), ellip_y = fac * sin(2*phi) and ellip_x = fac * cos(2*phi).
        """
        super(MockSphericalIsothermal, self).__init__(
            centre=centre, elliptical_comps=(0.0, 0.0), einstein_radius=einstein_radius
        )

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def potential_from_grid(self, grid):
        """
        Calculate the potential at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : ag.Grid2D
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
        eta = self.grid_to_elliptical_radii(grid)
        return 2.0 * self.einstein_radius_rescaled * eta

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : ag.Grid2D
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
        return self.grid_to_grid_cartesian(
            grid=grid,
            radius=np.full(grid.shape[0], 2.0 * self.einstein_radius_rescaled),
        )


class MockGalaxy(lensing.LensingObject):
    def __init__(self, mass_profiles):
        self._mass_profiles = mass_profiles

    @property
    def mass_profiles(self):
        return self._mass_profiles

    @property
    def mass_profile_centres(self):
        return [mass_profile.centre for mass_profile in self.mass_profiles]


class TestDeflectionsMagnitudes:
    def test__compare_sis_deflection_magnitudes_to_known_values(self):
        sis = MockSphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

        grid = ag.Grid2DIrregularGrouped([(1.0, 0.0), (0.0, 1.0)])

        deflection_magnitudes = sis.deflection_magnitudes_from_grid(grid=grid)

        assert deflection_magnitudes == pytest.approx(np.array([1.0, 1.0]), 1.0e-4)

        sis = MockSphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        grid = ag.Grid2DIrregularGrouped([(2.0, 0.0), (0.0, 2.0)])

        deflection_magnitudes = sis.deflection_magnitudes_from_grid(grid=grid)

        assert deflection_magnitudes == pytest.approx(np.array([2.0, 2.0]), 1.0e-4)

        grid = ag.Grid2D.uniform(shape_native=(5, 5), pixel_scales=0.1, sub_size=1)

        deflections = sis.deflections_from_grid(grid=grid)
        magitudes_manual = np.sqrt(
            np.square(deflections[:, 0]) + np.square(deflections[:, 1])
        )

        deflection_magnitudes = sis.deflection_magnitudes_from_grid(grid=grid)

        assert deflection_magnitudes == pytest.approx(magitudes_manual, 1.0e-4)


class TestDeflectionsViaPotential:
    def test__compare_sis_deflections_via_potential_and_calculation(self):
        sis = MockSphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        grid = ag.Grid2D.uniform(shape_native=(10, 10), pixel_scales=0.05, sub_size=1)

        deflections_via_calculation = sis.deflections_from_grid(grid=grid)

        deflections_via_potential = sis.deflections_via_potential_from_grid(grid=grid)

        mean_error = np.mean(
            deflections_via_potential.slim - deflections_via_calculation.slim
        )

        assert mean_error < 1e-4

    def test__compare_sie_at_phi_45__deflections_via_potential_and_calculation(self):
        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.111111, 0.0), einstein_radius=2.0
        )

        grid = ag.Grid2D.uniform(shape_native=(10, 10), pixel_scales=0.05, sub_size=1)

        deflections_via_calculation = sie.deflections_from_grid(grid=grid)

        deflections_via_potential = sie.deflections_via_potential_from_grid(grid=grid)

        mean_error = np.mean(
            deflections_via_potential.slim - deflections_via_calculation.slim
        )

        assert mean_error < 1e-4

    def test__compare_sie_at_phi_0__deflections_via_potential_and_calculation(self):
        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.0, -0.111111), einstein_radius=2.0
        )

        grid = ag.Grid2D.uniform(shape_native=(10, 10), pixel_scales=0.05, sub_size=1)

        deflections_via_calculation = sie.deflections_from_grid(grid=grid)

        deflections_via_potential = sie.deflections_via_potential_from_grid(grid=grid)

        mean_error = np.mean(
            deflections_via_potential.slim - deflections_via_calculation.slim
        )

        assert mean_error < 1e-4


class TestJacobian:
    def test__jacobian_components(self):
        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.0, -0.111111), einstein_radius=2.0
        )

        grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05, sub_size=1)

        jacobian = sie.jacobian_from_grid(grid=grid)

        A_12 = jacobian[0][1]
        A_21 = jacobian[1][0]

        mean_error = np.mean(A_12.slim - A_21.slim)

        assert mean_error < 1e-4

        grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05, sub_size=2)

        jacobian = sie.jacobian_from_grid(grid=grid)

        A_12 = jacobian[0][1]
        A_21 = jacobian[1][0]

        mean_error = np.mean(A_12.slim - A_21.slim)

        assert mean_error < 1e-4


class TestHessian:
    def test__hessian_from_grid(self):

        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.0, -0.111111), einstein_radius=2.0
        )

        grid = ag.Grid2DIrregularGrouped(grid=[[(0.5, 0.5)], [(1.0, 1.0)]])

        hessian_yy, hessian_xy, hessian_yx, hessian_xx = sie.hessian_from_grid(
            grid=grid
        )

        print(hessian_yy, hessian_xy, hessian_yx, hessian_xx)

        assert hessian_yy == pytest.approx(np.array([1.3883822, 0.694127]), 1.0e-4)
        assert hessian_xy == pytest.approx(np.array([-1.388124, -0.694094]), 1.0e-4)
        assert hessian_yx == pytest.approx(np.array([-1.388165, -0.694099]), 1.0e-4)
        assert hessian_xx == pytest.approx(np.array([1.3883824, 0.694127]), 1.0e-4)

        grid = ag.Grid2DIrregularGrouped(grid=[[(1.0, 0.0)], [(0.0, 1.0)]])

        hessian_yy, hessian_xy, hessian_yx, hessian_xx = sie.hessian_from_grid(
            grid=grid
        )

        assert hessian_yy == pytest.approx(np.array([0.0, 1.777699]), 1.0e-4)
        assert hessian_xy == pytest.approx(np.array([0.0, 0.0]), 1.0e-4)
        assert hessian_yx == pytest.approx(np.array([0.0, 0.0]), 1.0e-4)
        assert hessian_xx == pytest.approx(np.array([2.22209, 0.0]), 1.0e-4)


class TestConvergence:
    def test__convergence_via_hessian_from_grid(self):

        buffer = 0.0001
        grid = ag.Grid2DIrregularGrouped(
            grid=[
                [(1.075, -0.125)],
                [(-0.875, -0.075)],
                [(-0.925, -0.075)],
                [(0.075, 0.925)],
            ]
        )

        sis = MockEllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.001, 0.001), einstein_radius=1.0
        )

        convergence = sis.convergence_via_hessian_from_grid(grid=grid, buffer=buffer)

        assert convergence.in_grouped_list[0][0] == pytest.approx(0.461447, 1.0e-4)
        assert convergence.in_grouped_list[1][0] == pytest.approx(0.568875, 1.0e-4)
        assert convergence.in_grouped_list[2][0] == pytest.approx(0.538326, 1.0e-4)
        assert convergence.in_grouped_list[3][0] == pytest.approx(0.539390, 1.0e-4)

        sis = ag.mp.EllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.3, 0.4), einstein_radius=1.5
        )

        print(sis.convergence_from_grid(grid=grid))

        convergence = sis.convergence_via_hessian_from_grid(grid=grid, buffer=buffer)

        print(convergence)

        assert convergence.in_grouped_list[0][0] == pytest.approx(0.35313, 1.0e-4)
        assert convergence.in_grouped_list[1][0] == pytest.approx(0.46030, 1.0e-4)
        assert convergence.in_grouped_list[2][0] == pytest.approx(0.43484, 1.0e-4)
        assert convergence.in_grouped_list[3][0] == pytest.approx(1.00492, 1.0e-4)


class TestShear:
    def test__shear_via_hessian_from_grid(self):

        buffer = 0.00001
        grid = ag.Grid2DIrregularGrouped(
            grid=[
                [(1.075, -0.125)],
                [(-0.875, -0.075)],
                [(-0.925, -0.075)],
                [(0.075, 0.925)],
            ]
        )

        sis = ag.mp.EllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.001, 0.001), einstein_radius=1.0
        )

        shear = sis.shear_via_hessian_from_grid(grid=grid, buffer=buffer)

        assert shear.in_grouped_list[0][0] == pytest.approx(0.461447, 1.0e-4)
        assert shear.in_grouped_list[1][0] == pytest.approx(0.568875, 1.0e-4)
        assert shear.in_grouped_list[2][0] == pytest.approx(0.538326, 1.0e-4)
        assert shear.in_grouped_list[3][0] == pytest.approx(0.539390, 1.0e-4)

        sis = ag.mp.EllipticalIsothermal(
            centre=(0.2, 0.1), elliptical_comps=(0.3, 0.4), einstein_radius=1.5
        )

        shear = sis.shear_from_grid(grid=grid)
        print((shear[:, 0] ** 2 + shear[:, 1] ** 2) ** 0.5)

        shear = sis.shear_via_hessian_from_grid(grid=grid, buffer=buffer)

        assert shear.in_grouped_list[0][0] == pytest.approx(0.41597, 1.0e-4)
        assert shear.in_grouped_list[1][0] == pytest.approx(0.38299, 1.0e-4)
        assert shear.in_grouped_list[2][0] == pytest.approx(0.36522, 1.0e-4)
        assert shear.in_grouped_list[3][0] == pytest.approx(0.82750, 1.0e-4)


class TestMagnification:
    def test__compare_magnification_from_eigen_values_and_from_determinant(self):
        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.0, -0.111111), einstein_radius=2.0
        )

        grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05, sub_size=1)

        magnification_via_determinant = sie.magnification_from_grid(grid=grid)

        tangential_eigen_value = sie.tangential_eigen_value_from_grid(grid=grid)

        radal_eigen_value = sie.radial_eigen_value_from_grid(grid=grid)

        magnification_via_eigen_values = 1 / (
            tangential_eigen_value * radal_eigen_value
        )

        mean_error = np.mean(
            magnification_via_determinant.slim - magnification_via_eigen_values.slim
        )

        assert mean_error < 1e-4

        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.0, -0.111111), einstein_radius=2.0
        )

        grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05, sub_size=2)

        magnification_via_determinant = sie.magnification_from_grid(grid=grid)

        tangential_eigen_value = sie.tangential_eigen_value_from_grid(grid=grid)

        radal_eigen_value = sie.radial_eigen_value_from_grid(grid=grid)

        magnification_via_eigen_values = 1 / (
            tangential_eigen_value * radal_eigen_value
        )

        mean_error = np.mean(
            magnification_via_determinant.slim - magnification_via_eigen_values.slim
        )

        assert mean_error < 1e-4

    def test__compare_magnification_from_determinant_and_from_convergence_and_shear(
        self,
    ):
        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.0, -0.111111), einstein_radius=2.0
        )

        grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05, sub_size=1)

        magnification_via_determinant = sie.magnification_from_grid(grid=grid)

        convergence = sie.convergence_via_jacobian_from_grid(grid=grid)

        shear = sie.shear_via_jacobian_from_grid(grid=grid)

        magnification_via_convergence_and_shear = 1 / (
            (1 - convergence) ** 2 - shear ** 2
        )

        mean_error = np.mean(
            magnification_via_determinant.slim
            - magnification_via_convergence_and_shear.slim
        )

        assert mean_error < 1e-4

        grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05, sub_size=2)

        magnification_via_determinant = sie.magnification_from_grid(grid=grid)

        convergence = sie.convergence_via_jacobian_from_grid(grid=grid)

        shear = sie.shear_via_jacobian_from_grid(grid=grid)

        magnification_via_convergence_and_shear = 1 / (
            (1 - convergence) ** 2 - shear ** 2
        )

        mean_error = np.mean(
            magnification_via_determinant.slim
            - magnification_via_convergence_and_shear.slim
        )

        assert mean_error < 1e-4

    def test__magnification_via_hessian_from_grid(self):

        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.0, -0.111111), einstein_radius=2.0
        )

        grid = ag.Grid2DIrregularGrouped(grid=[[(0.5, 0.5)], [(1.0, 1.0)]])

        magnification = sie.magnification_via_hessian_from_grid(grid=grid)

        assert magnification.in_grouped_list[0][0] == pytest.approx(-0.56303, 1.0e-4)
        assert magnification.in_grouped_list[1][0] == pytest.approx(-2.57591, 1.0e-4)


def critical_curve_via_magnification_from(mass_profile, grid):
    magnification = mass_profile.magnification_from_grid(grid=grid)

    inverse_magnification = 1 / magnification

    critical_curves_indices = measure.find_contours(inverse_magnification.native, 0)

    no_critical_curves = len(critical_curves_indices)
    contours = []
    critical_curves = []

    for jj in np.arange(no_critical_curves):
        contours.append(critical_curves_indices[jj])
        contour_x, contour_y = contours[jj].T
        pixel_coord = np.stack((contour_x, contour_y), axis=-1)

        critical_curve = grid.mask.grid_scaled_from_grid_pixels_1d_for_marching_squares(
            grid_pixels_1d=pixel_coord, shape_native=magnification.sub_shape_native
        )

        critical_curves.append(critical_curve)

    return critical_curves


def caustics_via_magnification_from(mass_profile, grid):
    caustics = []

    critical_curves = critical_curve_via_magnification_from(
        mass_profile=mass_profile, grid=grid
    )

    for i in range(len(critical_curves)):
        critical_curve = critical_curves[i]

        deflections_1d = mass_profile.deflections_from_grid(grid=critical_curve)

        caustic = critical_curve - deflections_1d

        caustics.append(caustic)

    return caustics


class TestConvergenceViajacobian:
    def test__compare_sis_convergence_via_jacobian_and_calculation(self):
        sis = MockSphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        grid = ag.Grid2D.uniform(shape_native=(20, 20), pixel_scales=0.05, sub_size=1)

        convergence_via_calculation = sis.convergence_from_grid(grid=grid)

        convergence_via_jacobian = sis.convergence_via_jacobian_from_grid(grid=grid)

        mean_error = np.mean(
            convergence_via_jacobian.slim - convergence_via_calculation.slim
        )

        assert convergence_via_jacobian.native_binned.shape == (20, 20)
        assert mean_error < 1e-1

        mean_error = np.mean(
            convergence_via_jacobian.slim - convergence_via_calculation.slim
        )

        assert mean_error < 1e-1

    def test__compare_sie_at_phi_45__convergence_via_jacobian_and_calculation(self):
        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.111111, 0.0), einstein_radius=2.0
        )

        grid = ag.Grid2D.uniform(shape_native=(20, 20), pixel_scales=0.05, sub_size=1)

        convergence_via_calculation = sie.convergence_from_grid(grid=grid)

        convergence_via_jacobian = sie.convergence_via_jacobian_from_grid(grid=grid)

        mean_error = np.mean(
            convergence_via_jacobian.slim - convergence_via_calculation.slim
        )

        assert mean_error < 1e-1


class TestEvaluationGrid:
    def test__grid_changes_resolution_based_on_pixel_scale_input(self):
        @lensing.evaluation_grid
        def mock_func(lensing_obj, grid, pixel_scale=0.05):
            return grid

        grid = ag.Grid2D.uniform(shape_native=(4, 4), pixel_scales=0.05)

        evaluation_grid = mock_func(lensing_obj=None, grid=grid, pixel_scale=0.05)

        assert (evaluation_grid == grid).all()

        evaluation_grid = mock_func(lensing_obj=None, grid=grid, pixel_scale=0.1)
        downscaled_grid = ag.Grid2D.uniform(shape_native=(2, 2), pixel_scales=0.1)

        assert (evaluation_grid == downscaled_grid).all()

        evaluation_grid = mock_func(lensing_obj=None, grid=grid, pixel_scale=0.025)
        upscaled_grid = ag.Grid2D.uniform(shape_native=(8, 8), pixel_scales=0.025)

        assert (evaluation_grid == upscaled_grid).all()

        evaluation_grid = mock_func(lensing_obj=None, grid=grid, pixel_scale=0.03)
        upscaled_grid = ag.Grid2D.uniform(shape_native=(6, 6), pixel_scales=0.03)

        assert (evaluation_grid == upscaled_grid).all()

    def test__grid_changes_to_uniform_and_zoomed_in_if_masked(self):
        @lensing.evaluation_grid
        def mock_func(lensing_obj, grid, pixel_scale=0.05):
            return grid

        mask = ag.Mask2D.circular(shape_native=(11, 11), pixel_scales=1.0, radius=3.0)

        grid = ag.Grid2D.from_mask(mask=mask)

        evaluation_grid = mock_func(lensing_obj=None, grid=grid, pixel_scale=1.0)
        grid_uniform = ag.Grid2D.uniform(shape_native=(7, 7), pixel_scales=1.0)

        assert (evaluation_grid[0] == np.array([3.0, -3.0])).all()
        assert (evaluation_grid == grid_uniform).all()

        mask = ag.Mask2D.circular(
            shape_native=(29, 29), pixel_scales=1.0, radius=3.0, centre=(5.0, 5.0)
        )

        grid = ag.Grid2D.from_mask(mask=mask)

        evaluation_grid = mock_func(lensing_obj=None, grid=grid, pixel_scale=1.0)
        grid_uniform = ag.Grid2D.uniform(
            shape_native=(7, 7), pixel_scales=1.0, origin=(5.0, 5.0)
        )

        assert (evaluation_grid[0] == np.array([8.0, 2.0])).all()
        assert (evaluation_grid == grid_uniform).all()


class TestCriticalCurvesAndCaustics:
    def test_compare_magnification_from_determinant_and_from_convergence_and_shear(
        self,
    ):
        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.0, -0.111111), einstein_radius=2.0
        )

        grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05, sub_size=2)

        magnification_via_determinant = sie.magnification_from_grid(grid=grid)

        convergence = sie.convergence_via_jacobian_from_grid(grid=grid)

        shear = sie.shear_via_jacobian_from_grid(grid=grid)

        magnification_via_convergence_and_shear = 1 / (
            (1 - convergence) ** 2 - shear ** 2
        )

        mean_error = np.mean(
            magnification_via_determinant - magnification_via_convergence_and_shear
        )

        assert mean_error < 1e-2

    def test__tangential_critical_curve_radii__spherical_isothermal(self):

        sis = MockSphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        grid = ag.Grid2D.uniform(shape_native=(15, 15), pixel_scales=0.3)

        critical_curves = sis.critical_curves_from_grid(grid=grid)

        tangential_critical_curve = np.asarray(critical_curves.in_grouped_list[0])

        x_critical_tangential, y_critical_tangential = (
            tangential_critical_curve[:, 1],
            tangential_critical_curve[:, 0],
        )

        assert np.mean(
            x_critical_tangential ** 2 + y_critical_tangential ** 2
        ) == pytest.approx(sis.einstein_radius ** 2, 5e-1)

    def test__tangential_critical_curve_centres__spherical_isothermal(self):
        sis = MockSphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

        critical_curves = sis.critical_curves_from_grid(grid=grid)

        tangential_critical_curve = np.asarray(critical_curves.in_grouped_list[0])

        y_centre = np.mean(tangential_critical_curve[:, 0])
        x_centre = np.mean(tangential_critical_curve[:, 1])

        assert -0.03 < y_centre < 0.03
        assert -0.03 < x_centre < 0.03

        sis = MockSphericalIsothermal(centre=(0.5, 1.0), einstein_radius=2.0)

        critical_curves = sis.critical_curves_from_grid(grid=grid)

        tangential_critical_curve = np.asarray(critical_curves.in_grouped_list[0])

        y_centre = np.mean(tangential_critical_curve[:, 0])
        x_centre = np.mean(tangential_critical_curve[:, 1])

        assert 0.47 < y_centre < 0.53
        assert 0.97 < x_centre < 1.03

    def test__radial_critical_curve_centres__spherical_isothermal(self):

        sis = MockSphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

        critical_curves = sis.critical_curves_from_grid(grid=grid)

        radial_critical_curve = np.asarray(critical_curves.in_grouped_list[1])

        y_centre = np.mean(radial_critical_curve[:, 0])
        x_centre = np.mean(radial_critical_curve[:, 1])

        assert -0.05 < y_centre < 0.05
        assert -0.05 < x_centre < 0.05

        sis = MockSphericalIsothermal(centre=(0.5, 1.0), einstein_radius=2.0)

        critical_curves = sis.critical_curves_from_grid(grid=grid)

        radial_critical_curve = np.asarray(critical_curves.in_grouped_list[1])

        y_centre = np.mean(radial_critical_curve[:, 0])
        x_centre = np.mean(radial_critical_curve[:, 1])

        assert 0.45 < y_centre < 0.55
        assert 0.95 < x_centre < 1.05

    def test__tangential_caustic_centres__spherical_isothermal(self):
        sis = MockSphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

        caustics = sis.caustics_from_grid(grid=grid)

        tangential_caustic = np.asarray(caustics.in_grouped_list[0])

        y_centre = np.mean(tangential_caustic[:, 0])
        x_centre = np.mean(tangential_caustic[:, 1])

        assert -0.03 < y_centre < 0.03
        assert -0.03 < x_centre < 0.03

        sis = MockSphericalIsothermal(centre=(0.5, 1.0), einstein_radius=2.0)

        caustics = sis.caustics_from_grid(grid=grid)

        tangential_caustic = np.asarray(caustics.in_grouped_list[0])

        y_centre = np.mean(tangential_caustic[:, 0])
        x_centre = np.mean(tangential_caustic[:, 1])

        assert 0.47 < y_centre < 0.53
        assert 0.97 < x_centre < 1.03

    def test__radial_caustics_radii__spherical_isothermal(self):
        sis = MockSphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        grid = ag.Grid2D.uniform(shape_native=(20, 20), pixel_scales=0.2)

        caustics = sis.caustics_from_grid(grid=grid)

        caustic_radial = np.asarray(caustics.in_grouped_list[1])

        x_caustic_radial, y_caustic_radial = (
            caustic_radial[:, 1],
            caustic_radial[:, 0],
        )

        assert np.mean(x_caustic_radial ** 2 + y_caustic_radial ** 2) == pytest.approx(
            sis.einstein_radius ** 2, 5e-1
        )

    def test__radial_caustic_centres__spherical_isothermal(self):
        sis = MockSphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

        caustics = sis.caustics_from_grid(grid=grid)

        radial_caustic = np.asarray(caustics.in_grouped_list[1])

        y_centre = np.mean(radial_caustic[:, 0])
        x_centre = np.mean(radial_caustic[:, 1])

        assert -0.2 < y_centre < 0.2
        assert -0.35 < x_centre < 0.35

        sis = MockSphericalIsothermal(centre=(0.5, 1.0), einstein_radius=2.0)

        caustics = sis.caustics_from_grid(grid=grid)

        radial_caustic = np.asarray(caustics.in_grouped_list[1])

        y_centre = np.mean(radial_caustic[:, 0])
        x_centre = np.mean(radial_caustic[:, 1])

        assert 0.3 < y_centre < 0.7
        assert 0.7 < x_centre < 1.2

    def test__compare_tangential_critical_curves_from_magnification_and_eigen_values(
        self,
    ):
        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), einstein_radius=2, elliptical_comps=(0.109423, -0.019294)
        )

        grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

        tangential_critical_curve_from_magnification = critical_curve_via_magnification_from(
            mass_profile=sie, grid=grid
        )[
            0
        ]

        tangential_critical_curve = sie.tangential_critical_curve_from_grid(
            grid=grid, pixel_scale=0.2
        )

        assert tangential_critical_curve == pytest.approx(
            tangential_critical_curve_from_magnification, 5e-1
        )

        tangential_critical_curve_from_magnification = critical_curve_via_magnification_from(
            mass_profile=sie, grid=grid
        )[
            0
        ]

        tangential_critical_curve = sie.tangential_critical_curve_from_grid(
            grid=grid, pixel_scale=0.2
        )

        assert tangential_critical_curve == pytest.approx(
            tangential_critical_curve_from_magnification, 5e-1
        )

    def test__compare_radial_critical_curves_from_magnification_and_eigen_values(self):

        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), einstein_radius=2, elliptical_comps=(0.109423, -0.019294)
        )

        grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

        critical_curve_radial_from_magnification = critical_curve_via_magnification_from(
            mass_profile=sie, grid=grid
        )[
            1
        ]

        radial_critical_curve = sie.radial_critical_curve_from_grid(grid=grid)

        assert sum(critical_curve_radial_from_magnification) == pytest.approx(
            sum(radial_critical_curve), abs=0.7
        )

    def test__compare_tangential_caustic_from_magnification_and_eigen_values(self):
        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), einstein_radius=2, elliptical_comps=(0.109423, -0.019294)
        )

        grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

        tangential_caustic_from_magnification = caustics_via_magnification_from(
            mass_profile=sie, grid=grid
        )[0]

        tangential_caustic = sie.tangential_caustic_from_grid(
            grid=grid, pixel_scale=0.2
        )

        assert sum(tangential_caustic) == pytest.approx(
            sum(tangential_caustic_from_magnification), 5e-1
        )

    def test__compare_radial_caustic_from_magnification_and_eigen_values__grid(self):
        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), einstein_radius=2, elliptical_comps=(0.109423, -0.019294)
        )

        grid = ag.Grid2D.uniform(shape_native=(60, 60), pixel_scales=0.08)

        caustic_radial_from_magnification = caustics_via_magnification_from(
            mass_profile=sie, grid=grid
        )[1]

        radial_caustic = sie.radial_caustic_from_grid(grid=grid, pixel_scale=0.08)

        assert sum(radial_caustic) == pytest.approx(
            sum(caustic_radial_from_magnification), 7e-1
        )


class TestEinsteinRadiusMass:
    def test__tangential_critical_curve_area_from_critical_curve_and_calculation__spherical_isothermal(
        self,
    ):
        sis = MockSphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

        area_calc = np.pi * sis.einstein_radius ** 2

        area_within_tangential_critical_curve = sis.area_within_tangential_critical_curve_from_grid(
            grid=grid
        )

        assert area_within_tangential_critical_curve == pytest.approx(area_calc, 1e-1)

    def test__einstein_radius_from_tangential_critical_curve_values(self):

        grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

        sis = MockSphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        einstein_radius = sis.einstein_radius_from_grid(grid=grid)

        assert einstein_radius == pytest.approx(2.0, 1e-1)

        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), einstein_radius=2.0, elliptical_comps=(0.0, -0.25)
        )

        einstein_radius = sie.einstein_radius_from_grid(grid=grid)

        assert einstein_radius == pytest.approx(1.9360, 1e-1)

    def test__einstein_mass_from_tangential_critical_curve_values(self):

        grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

        sis = MockSphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        einstein_mass = sis.einstein_mass_angular_from_grid(grid=grid)

        assert einstein_mass == pytest.approx(np.pi * 2.0 ** 2.0, 1e-1)


class TestGridBinning:
    def test__binning_works_on_all_from_grid_methods(self):
        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.0, -0.111111), einstein_radius=2.0
        )

        grid = ag.Grid2D.uniform(shape_native=(10, 10), pixel_scales=0.05, sub_size=2)

        deflections = sie.deflections_via_potential_from_grid(grid=grid)

        deflections_first_binned_pixel = (
            deflections[0] + deflections[1] + deflections[2] + deflections[3]
        ) / 4

        assert deflections.slim_binned[0] == pytest.approx(
            deflections_first_binned_pixel, 1e-4
        )

        deflections_100th_binned_pixel = (
            deflections[399] + deflections[398] + deflections[397] + deflections[396]
        ) / 4

        assert deflections.slim_binned[99] == pytest.approx(
            deflections_100th_binned_pixel, 1e-4
        )

        jacobian = sie.jacobian_from_grid(grid=grid)

        jacobian_1st_pixel_binned_up = (
            jacobian[0][0][0]
            + jacobian[0][0][1]
            + jacobian[0][0][2]
            + jacobian[0][0][3]
        ) / 4

        assert jacobian[0][0].native_binned.shape == (10, 10)
        assert jacobian[0][0].sub_shape_native == (20, 20)
        assert jacobian[0][0].slim_binned[0] == pytest.approx(
            jacobian_1st_pixel_binned_up, 1e-4
        )

        jacobian_last_pixel_binned_up = (
            jacobian[0][0][399]
            + jacobian[0][0][398]
            + jacobian[0][0][397]
            + jacobian[0][0][396]
        ) / 4

        assert jacobian[0][0].slim_binned[99] == pytest.approx(
            jacobian_last_pixel_binned_up, 1e-4
        )

        shear_via_jacobian = sie.shear_via_jacobian_from_grid(grid=grid)

        shear_1st_pixel_binned_up = (
            shear_via_jacobian[0]
            + shear_via_jacobian[1]
            + shear_via_jacobian[2]
            + shear_via_jacobian[3]
        ) / 4

        assert shear_via_jacobian.slim_binned[0] == pytest.approx(
            shear_1st_pixel_binned_up, 1e-4
        )

        shear_last_pixel_binned_up = (
            shear_via_jacobian[399]
            + shear_via_jacobian[398]
            + shear_via_jacobian[397]
            + shear_via_jacobian[396]
        ) / 4

        assert shear_via_jacobian.slim_binned[99] == pytest.approx(
            shear_last_pixel_binned_up, 1e-4
        )

        tangential_eigen_values = sie.tangential_eigen_value_from_grid(grid=grid)

        first_pixel_binned_up = (
            tangential_eigen_values[0]
            + tangential_eigen_values[1]
            + tangential_eigen_values[2]
            + tangential_eigen_values[3]
        ) / 4

        assert tangential_eigen_values.slim_binned[0] == pytest.approx(
            first_pixel_binned_up, 1e-4
        )

        pixel_10000_from_av_sub_grid = (
            tangential_eigen_values[399]
            + tangential_eigen_values[398]
            + tangential_eigen_values[397]
            + tangential_eigen_values[396]
        ) / 4

        assert tangential_eigen_values.slim_binned[99] == pytest.approx(
            pixel_10000_from_av_sub_grid, 1e-4
        )

        radial_eigen_values = sie.radial_eigen_value_from_grid(grid=grid)

        first_pixel_binned_up = (
            radial_eigen_values[0]
            + radial_eigen_values[1]
            + radial_eigen_values[2]
            + radial_eigen_values[3]
        ) / 4

        assert radial_eigen_values.slim_binned[0] == pytest.approx(
            first_pixel_binned_up, 1e-4
        )

        pixel_10000_from_av_sub_grid = (
            radial_eigen_values[399]
            + radial_eigen_values[398]
            + radial_eigen_values[397]
            + radial_eigen_values[396]
        ) / 4

        assert radial_eigen_values.slim_binned[99] == pytest.approx(
            pixel_10000_from_av_sub_grid, 1e-4
        )
