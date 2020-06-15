import os

from autoconf import conf
import autofit as af
import autogalaxy as ag
import numpy as np
import pytest
from astropy import cosmology as cosmo
from autoarray.structures import grids
from autogalaxy import lensing
from autogalaxy.profiles import geometry_profiles
from pyquad import quad_grid
from skimage import measure
from test_autogalaxy import mock
import typing


@pytest.fixture(autouse=True)
def reset_config():
    """
    Use configuration from the default path. You may want to change this to set a specific path.
    """
    test_path = "{}/config/lensing".format(os.path.dirname(os.path.realpath(__file__)))
    conf.instance = conf.Config(config_path=test_path)


class MockEllipticalIsothermal(
    geometry_profiles.EllipticalProfile, lensing.LensingObject
):
    @af.map_types
    def __init__(
        self,
        centre: ag.dim.Position = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
        einstein_radius: ag.dim.Length = 1.0,
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
    def unit_mass(self):
        return "angular"

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

        The *grid_like_to_structure* decorator reshapes the NumPy arrays the convergence is outputted on. See \
        *ag.grid_like_to_structure* for a description of the output.

        Parameters
        ----------
        grid : ag.Grid
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
        grid : ag.Grid
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
        grid : ag.Grid
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
    @af.map_types
    def __init__(
        self, centre: ag.dim.Position = (0.0, 0.0), einstein_radius: ag.dim.Length = 1.0
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
        grid : ag.Grid
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
        grid : ag.Grid
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

        grid = ag.GridCoordinates([(1.0, 0.0), (0.0, 1.0)])

        deflection_magnitudes = sis.deflection_magnitudes_from_grid(grid=grid)

        assert deflection_magnitudes == pytest.approx(np.array([1.0, 1.0]), 1.0e-4)

        sis = MockSphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        grid = ag.GridCoordinates([(2.0, 0.0), (0.0, 2.0)])

        deflection_magnitudes = sis.deflection_magnitudes_from_grid(grid=grid)

        assert deflection_magnitudes == pytest.approx(np.array([2.0, 2.0]), 1.0e-4)

        grid = ag.Grid.uniform(shape_2d=(5, 5), pixel_scales=0.1, sub_size=1)

        deflections = sis.deflections_from_grid(grid=grid)
        magitudes_manual = np.sqrt(
            np.square(deflections[:, 0]) + np.square(deflections[:, 1])
        )

        deflection_magnitudes = sis.deflection_magnitudes_from_grid(grid=grid)

        assert deflection_magnitudes == pytest.approx(magitudes_manual, 1.0e-4)


class TestDeflectionsViaPotential:
    def test__compare_sis_deflections_via_potential_and_calculation(self):
        sis = MockSphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        grid = ag.Grid.uniform(shape_2d=(10, 10), pixel_scales=0.05, sub_size=1)

        deflections_via_calculation = sis.deflections_from_grid(grid=grid)

        deflections_via_potential = sis.deflections_via_potential_from_grid(grid=grid)

        mean_error = np.mean(
            deflections_via_potential.in_1d - deflections_via_calculation.in_1d
        )

        assert mean_error < 1e-4

    def test__compare_sie_at_phi_45__deflections_via_potential_and_calculation(self):

        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.111111, 0.0), einstein_radius=2.0
        )

        grid = ag.Grid.uniform(shape_2d=(10, 10), pixel_scales=0.05, sub_size=1)

        deflections_via_calculation = sie.deflections_from_grid(grid=grid)

        deflections_via_potential = sie.deflections_via_potential_from_grid(grid=grid)

        mean_error = np.mean(
            deflections_via_potential.in_1d - deflections_via_calculation.in_1d
        )

        assert mean_error < 1e-4

    def test__compare_sie_at_phi_0__deflections_via_potential_and_calculation(self):

        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.0, -0.111111), einstein_radius=2.0
        )

        grid = ag.Grid.uniform(shape_2d=(10, 10), pixel_scales=0.05, sub_size=1)

        deflections_via_calculation = sie.deflections_from_grid(grid=grid)

        deflections_via_potential = sie.deflections_via_potential_from_grid(grid=grid)

        mean_error = np.mean(
            deflections_via_potential.in_1d - deflections_via_calculation.in_1d
        )

        assert mean_error < 1e-4


class TestJacobian:
    def test__jacobian_components(self):
        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.0, -0.111111), einstein_radius=2.0
        )

        grid = ag.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.05, sub_size=1)

        jacobian = sie.jacobian_from_grid(grid=grid)

        A_12 = jacobian[0][1]
        A_21 = jacobian[1][0]

        mean_error = np.mean(A_12.in_1d - A_21.in_1d)

        assert mean_error < 1e-4

        grid = ag.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.05, sub_size=2)

        jacobian = sie.jacobian_from_grid(grid=grid)

        A_12 = jacobian[0][1]
        A_21 = jacobian[1][0]

        mean_error = np.mean(A_12.in_1d - A_21.in_1d)

        assert mean_error < 1e-4


class TestMagnification:
    def test__compare_magnification_from_eigen_values_and_from_determinant(self):
        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.0, -0.111111), einstein_radius=2.0
        )

        grid = ag.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.05, sub_size=1)

        magnification_via_determinant = sie.magnification_from_grid(grid=grid)

        tangential_eigen_value = sie.tangential_eigen_value_from_grid(grid=grid)

        radal_eigen_value = sie.radial_eigen_value_from_grid(grid=grid)

        magnification_via_eigen_values = 1 / (
            tangential_eigen_value * radal_eigen_value
        )

        mean_error = np.mean(
            magnification_via_determinant.in_1d - magnification_via_eigen_values.in_1d
        )

        assert mean_error < 1e-4

        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.0, -0.111111), einstein_radius=2.0
        )

        grid = ag.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.05, sub_size=2)

        magnification_via_determinant = sie.magnification_from_grid(grid=grid)

        tangential_eigen_value = sie.tangential_eigen_value_from_grid(grid=grid)

        radal_eigen_value = sie.radial_eigen_value_from_grid(grid=grid)

        magnification_via_eigen_values = 1 / (
            tangential_eigen_value * radal_eigen_value
        )

        mean_error = np.mean(
            magnification_via_determinant.in_1d - magnification_via_eigen_values.in_1d
        )

        assert mean_error < 1e-4

    def test__compare_magnification_from_determinant_and_from_convergence_and_shear(
        self
    ):
        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.0, -0.111111), einstein_radius=2.0
        )

        grid = ag.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.05, sub_size=1)

        magnification_via_determinant = sie.magnification_from_grid(grid=grid)

        convergence = sie.convergence_via_jacobian_from_grid(grid=grid)

        shear = sie.shear_via_jacobian_from_grid(grid=grid)

        magnification_via_convergence_and_shear = 1 / (
            (1 - convergence) ** 2 - shear ** 2
        )

        mean_error = np.mean(
            magnification_via_determinant.in_1d
            - magnification_via_convergence_and_shear.in_1d
        )

        assert mean_error < 1e-4

        grid = ag.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.05, sub_size=2)

        magnification_via_determinant = sie.magnification_from_grid(grid=grid)

        convergence = sie.convergence_via_jacobian_from_grid(grid=grid)

        shear = sie.shear_via_jacobian_from_grid(grid=grid)

        magnification_via_convergence_and_shear = 1 / (
            (1 - convergence) ** 2 - shear ** 2
        )

        mean_error = np.mean(
            magnification_via_determinant.in_1d
            - magnification_via_convergence_and_shear.in_1d
        )

        assert mean_error < 1e-4


class TestBoundingBox:
    def test__mass_profile_bounding_box__is_drawn_around_centres_of_mass_profies(self):
        sis = MockSphericalIsothermal(centre=(0.0, 0.0))

        assert sis.mass_profile_bounding_box == [0.0, 0.0, 0.0, 0.0]

        sis_0 = MockSphericalIsothermal(centre=(1.0, 1.0))

        sis_1 = MockSphericalIsothermal(centre=(-1.0, -1.0))

        galaxy = MockGalaxy(mass_profiles=[sis_0, sis_1])

        assert galaxy.mass_profile_bounding_box == [-1.0, 1.0, -1.0, 1.0]

        sis_0 = MockSphericalIsothermal(centre=(8.0, -6.0))

        sis_1 = MockSphericalIsothermal(centre=(4.0, 10.0))

        galaxy = MockGalaxy(mass_profiles=[sis_0, sis_1])

        assert galaxy.mass_profile_bounding_box == [4.0, 8.0, -6.0, 10.0]

        sis_0 = MockSphericalIsothermal(centre=(8.0, -6.0))

        sis_1 = MockSphericalIsothermal(centre=(4.0, 10.0))

        sis_2 = MockSphericalIsothermal(centre=(18.0, -16.0))

        sis_3 = MockSphericalIsothermal(centre=(0.0, 90.0))

        galaxy = MockGalaxy(mass_profiles=[sis_0, sis_1, sis_2, sis_3])

        assert galaxy.mass_profile_bounding_box == [0.0, 18.0, -16.0, 90.0]

    def test__convergence_bounding_box_for_single_mass_profile__extends_to_threshold(
        self
    ):

        sis = MockSphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

        assert sis.convergence_bounding_box(
            convergence_threshold=0.02
        ) == pytest.approx([-25.0, 25.0, -25.0, 25.0], 1.0e-4)

        sis = MockSphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

        assert sis.convergence_bounding_box(convergence_threshold=0.1) == pytest.approx(
            [-5.0, 5.0, -5.0, 5.0], 1.0e-4
        )

        sis = MockSphericalIsothermal(centre=(5.0, 5.0), einstein_radius=1.0)

        assert sis.convergence_bounding_box(convergence_threshold=0.1) == pytest.approx(
            [0.0, 10.0, 0.0, 10.0], 1.0e-4
        )

    def test__convergence_bounding_box__mass_profiles_are_only_point_masses__uses_their_einstein_radii(
        self
    ):

        point_mass_0 = ag.mp.PointMass(einstein_radius=0.1)

        convergence_bounding_box = point_mass_0.convergence_bounding_box()

        assert convergence_bounding_box == pytest.approx([-0.3, 0.3, -0.3, 0.3], 1.0e-4)

        galaxy = MockGalaxy(mass_profiles=[point_mass_0])

        convergence_bounding_box = galaxy.convergence_bounding_box()

        assert convergence_bounding_box == pytest.approx([-0.3, 0.3, -0.3, 0.3], 1.0e-4)

        point_mass_1 = ag.mp.PointMass(einstein_radius=0.3)

        galaxy = MockGalaxy(mass_profiles=[point_mass_0, point_mass_1])

        convergence_bounding_box = galaxy.convergence_bounding_box()

        assert convergence_bounding_box == pytest.approx([-1.2, 1.2, -1.2, 1.2], 1.0e-4)

        sis_0 = ag.mp.SphericalIsothermal(einstein_radius=2.0)
        galaxy = ag.Galaxy(
            redshift=0.5, point0=point_mass_0, point1=point_mass_1, sis0=sis_0
        )

        convergence_bounding_box = galaxy.convergence_bounding_box()

        assert convergence_bounding_box != pytest.approx([-1.2, 1.2, -1.2, 1.2], 1.0e-4)


def critical_curve_via_magnification_from_mass_profile_and_grid(mass_profile, grid):
    magnification = mass_profile.magnification_from_grid(grid=grid)

    inverse_magnification = 1 / magnification

    critical_curves_indices = measure.find_contours(inverse_magnification.in_2d, 0)

    no_critical_curves = len(critical_curves_indices)
    contours = []
    critical_curves = []

    for jj in np.arange(no_critical_curves):
        contours.append(critical_curves_indices[jj])
        contour_x, contour_y = contours[jj].T
        pixel_coord = np.stack((contour_x, contour_y), axis=-1)

        critical_curve = grid.geometry.grid_scaled_from_grid_pixels_1d_for_marching_squares(
            grid_pixels_1d=pixel_coord, shape_2d=magnification.sub_shape_2d
        )

        critical_curves.append(critical_curve)

    return critical_curves


def caustics_via_magnification_from_mass_profile_and_grid(mass_profile, grid):
    caustics = []

    critical_curves = critical_curve_via_magnification_from_mass_profile_and_grid(
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

        grid = ag.Grid.uniform(shape_2d=(20, 20), pixel_scales=0.05, sub_size=1)

        convergence_via_calculation = sis.convergence_from_grid(grid=grid)

        convergence_via_jacobian = sis.convergence_via_jacobian_from_grid(grid=grid)

        mean_error = np.mean(
            convergence_via_jacobian.in_1d - convergence_via_calculation.in_1d
        )

        assert convergence_via_jacobian.in_2d_binned.shape == (20, 20)
        assert mean_error < 1e-1

        mean_error = np.mean(
            convergence_via_jacobian.in_1d - convergence_via_calculation.in_1d
        )

        assert mean_error < 1e-1

    def test__compare_sie_at_phi_45__convergence_via_jacobian_and_calculation(self):
        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.111111, 0.0), einstein_radius=2.0
        )

        grid = ag.Grid.uniform(shape_2d=(20, 20), pixel_scales=0.05, sub_size=1)

        convergence_via_calculation = sie.convergence_from_grid(grid=grid)

        convergence_via_jacobian = sie.convergence_via_jacobian_from_grid(grid=grid)

        mean_error = np.mean(
            convergence_via_jacobian.in_1d - convergence_via_calculation.in_1d
        )

        assert mean_error < 1e-1


class TestCriticalCurvesAndCaustics:
    def test_compare_magnification_from_determinant_and_from_convergence_and_shear(
        self
    ):
        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.0, -0.111111), einstein_radius=2.0
        )

        grid = ag.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.05, sub_size=2)

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

        tangential_critical_curve = np.asarray(sis.critical_curves.in_list[0])

        x_critical_tangential, y_critical_tangential = (
            tangential_critical_curve[:, 1],
            tangential_critical_curve[:, 0],
        )

        assert np.mean(
            x_critical_tangential ** 2 + y_critical_tangential ** 2
        ) == pytest.approx(sis.einstein_radius ** 2, 5e-1)

    def test__tangential_critical_curve_centres__spherical_isothermal(self):
        sis = MockSphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        tangential_critical_curve = np.asarray(sis.critical_curves.in_list[0])

        y_centre = np.mean(tangential_critical_curve[:, 0])
        x_centre = np.mean(tangential_critical_curve[:, 1])

        assert -0.03 < y_centre < 0.03
        assert -0.03 < x_centre < 0.03

        sis = MockSphericalIsothermal(centre=(0.5, 1.0), einstein_radius=2.0)

        tangential_critical_curve = np.asarray(sis.critical_curves.in_list[0])

        y_centre = np.mean(tangential_critical_curve[:, 0])
        x_centre = np.mean(tangential_critical_curve[:, 1])

        assert 0.47 < y_centre < 0.53
        assert 0.97 < x_centre < 1.03

    def test__radial_critical_curve_centres__spherical_isothermal(self):
        sis = MockSphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        radial_critical_curve = np.asarray(sis.critical_curves.in_list[1])

        y_centre = np.mean(radial_critical_curve[:, 0])
        x_centre = np.mean(radial_critical_curve[:, 1])

        assert -0.05 < y_centre < 0.05
        assert -0.05 < x_centre < 0.05

        sis = MockSphericalIsothermal(centre=(0.5, 1.0), einstein_radius=2.0)

        radial_critical_curve = np.asarray(sis.critical_curves.in_list[1])

        y_centre = np.mean(radial_critical_curve[:, 0])
        x_centre = np.mean(radial_critical_curve[:, 1])

        assert 0.45 < y_centre < 0.55
        assert 0.95 < x_centre < 1.05

    def test__tangential_caustic_centres__spherical_isothermal(self):
        sis = MockSphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        tangential_caustic = np.asarray(sis.caustics.in_list[0])

        y_centre = np.mean(tangential_caustic[:, 0])
        x_centre = np.mean(tangential_caustic[:, 1])

        assert -0.03 < y_centre < 0.03
        assert -0.03 < x_centre < 0.03

        sis = MockSphericalIsothermal(centre=(0.5, 1.0), einstein_radius=2.0)

        tangential_caustic = np.asarray(sis.caustics.in_list[0])

        y_centre = np.mean(tangential_caustic[:, 0])
        x_centre = np.mean(tangential_caustic[:, 1])

        assert 0.47 < y_centre < 0.53
        assert 0.97 < x_centre < 1.03

    def test__radial_caustics_radii__spherical_isothermal(self):
        sis = MockSphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        caustic_radial = np.asarray(sis.caustics.in_list[1])

        x_caustic_radial, y_caustic_radial = (
            caustic_radial[:, 1],
            caustic_radial[:, 0],
        )

        assert np.mean(x_caustic_radial ** 2 + y_caustic_radial ** 2) == pytest.approx(
            sis.einstein_radius ** 2, 5e-1
        )

    def test__radial_caustic_centres__spherical_isothermal(self):
        sis = MockSphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        radial_caustic = np.asarray(sis.caustics.in_list[1])

        y_centre = np.mean(radial_caustic[:, 0])
        x_centre = np.mean(radial_caustic[:, 1])

        assert -0.2 < y_centre < 0.2
        assert -0.35 < x_centre < 0.35

        radial_caustic = np.asarray(sis.caustics.in_list[1])

        y_centre = np.mean(radial_caustic[:, 0])
        x_centre = np.mean(radial_caustic[:, 1])

        assert -0.2 < y_centre < 0.2
        assert -0.4 < x_centre < 0.4

        sis = MockSphericalIsothermal(centre=(0.5, 1.0), einstein_radius=2.0)

        radial_caustic = np.asarray(sis.caustics.in_list[1])

        y_centre = np.mean(radial_caustic[:, 0])
        x_centre = np.mean(radial_caustic[:, 1])

        assert 0.3 < y_centre < 0.7
        assert 0.8 < x_centre < 1.2

    def test__compare_tangential_critical_curves_from_magnification_and_eigen_values(
        self
    ):

        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), einstein_radius=2, elliptical_comps=(0.109423, -0.019294)
        )

        tangential_critical_curve_from_magnification = critical_curve_via_magnification_from_mass_profile_and_grid(
            mass_profile=sie, grid=sie.calculation_grid
        )[
            0
        ]

        assert sie.tangential_critical_curve == pytest.approx(
            tangential_critical_curve_from_magnification, 5e-1
        )

        tangential_critical_curve_from_magnification = critical_curve_via_magnification_from_mass_profile_and_grid(
            mass_profile=sie, grid=sie.calculation_grid
        )[
            0
        ]

        assert sie.tangential_critical_curve == pytest.approx(
            tangential_critical_curve_from_magnification, 5e-1
        )

    def test__compare_radial_critical_curves_from_magnification_and_eigen_values(self):
        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), einstein_radius=2, elliptical_comps=(0.109423, -0.019294)
        )

        critical_curve_radial_from_magnification = critical_curve_via_magnification_from_mass_profile_and_grid(
            mass_profile=sie, grid=sie.calculation_grid
        )[
            1
        ]

        assert sum(critical_curve_radial_from_magnification) == pytest.approx(
            sum(sie.radial_critical_curve), abs=0.7
        )

    def test__compare_tangential_caustic_from_magnification_and_eigen_values(self):
        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), einstein_radius=2, elliptical_comps=(0.109423, -0.019294)
        )

        tangential_caustic_from_magnification = caustics_via_magnification_from_mass_profile_and_grid(
            mass_profile=sie, grid=sie.calculation_grid
        )[
            0
        ]

        assert sum(sie.tangential_caustic) == pytest.approx(
            sum(tangential_caustic_from_magnification), 5e-1
        )

    def test__compare_radial_caustic_from_magnification_and_eigen_values__grid(self):
        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), einstein_radius=2, elliptical_comps=(0.109423, -0.019294)
        )

        caustic_radial_from_magnification = caustics_via_magnification_from_mass_profile_and_grid(
            mass_profile=sie, grid=sie.calculation_grid
        )[
            1
        ]

        assert sum(sie.radial_caustic) == pytest.approx(
            sum(caustic_radial_from_magnification), 7e-1
        )


class TestEinsteinRadiusMassfrom:
    def test__tangential_critical_curve_area_from_critical_curve_and_calculation__spherical_isothermal(
        self
    ):
        sis = MockSphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        area_calc = np.pi * sis.einstein_radius ** 2

        assert sis.area_within_tangential_critical_curve == pytest.approx(
            area_calc, 1e-1
        )

        area_calc = np.pi * sis.einstein_radius ** 2

        assert sis.area_within_tangential_critical_curve == pytest.approx(
            area_calc, 1e-1
        )

    def test__einstein_radius_from_tangential_critical_curve__spherical_isothermal(
        self
    ):
        sis = MockSphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        einstein_radius = sis.einstein_radius_in_units(unit_length="arcsec")

        assert einstein_radius == pytest.approx(2.0, 1e-1)

        cosmology = mock.MockCosmology(arcsec_per_kpc=2.0, kpc_per_arcsec=0.5)

        einstein_radius = sis.einstein_radius_in_units(
            unit_length="kpc", redshift_object=2.0, cosmology=cosmology
        )

        assert einstein_radius == pytest.approx(1.0, 1e-1)

    def test__compare_einstein_radius_from_tangential_critical_curve_and_rescaled__sie(
        self
    ):

        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), einstein_radius=2.0, elliptical_comps=(0.0, -0.25)
        )

        einstein_radius = sie.einstein_radius_in_units(unit_length="arcsec")

        assert einstein_radius == pytest.approx(1.9360, 1e-1)

        cosmology = mock.MockCosmology(arcsec_per_kpc=2.0, kpc_per_arcsec=0.5)

        einstein_radius = sie.einstein_radius_in_units(
            unit_length="kpc", redshift_object=2.0, cosmology=cosmology
        )

        assert einstein_radius == pytest.approx(0.5 * 1.9360, 1e-1)

    def test__einstein_mass_from_tangential_critical_curve_and_kappa__spherical_isothermal(
        self
    ):
        sis = MockSphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        einstein_mass = sis.einstein_mass_in_units(
            redshift_object=1, redshift_source=2, unit_mass="angular"
        )

        assert einstein_mass == pytest.approx(np.pi * 2.0 ** 2.0, 1e-1)

        cosmology = mock.MockCosmology(
            kpc_per_arcsec=1.0, arcsec_per_kpc=1.0, critical_surface_density=0.5
        )

        einstein_mass = sis.einstein_mass_in_units(
            redshift_object=1,
            redshift_source=2,
            unit_mass="solMass",
            cosmology=cosmology,
        )

        assert einstein_mass == pytest.approx(2.0 * np.pi * 2.0 ** 2.0, 1e-1)

    def test__einstein_mass_from_tangential_critical_curve_and_radius_rescaled_calc__sie(
        self
    ):
        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), einstein_radius=2.0, elliptical_comps=(0.0, -0.25)
        )

        einstein_mass_from_critical_curve = sie.einstein_mass_in_units(
            redshift_object=1,
            redshift_source=2,
            unit_mass="solMass",
            cosmology=cosmo.Planck15,
        )

        einstein_radius = sie.einstein_radius_in_units(unit_length="arcsec")

        sigma_crit = ag.util.cosmology.critical_surface_density_between_redshifts_from(
            redshift_0=1,
            redshift_1=2,
            unit_length="arcsec",
            unit_mass="solMass",
            cosmology=cosmo.Planck15,
        )

        einstein_mass_vie_einstein_radius = np.pi * einstein_radius ** 2 * sigma_crit

        assert einstein_mass_vie_einstein_radius == pytest.approx(
            einstein_mass_from_critical_curve, 1e-1
        )


class TestGridBinning:
    def test__binning_works_on_all_from_grid_methods(self):
        sie = MockEllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.0, -0.111111), einstein_radius=2.0
        )

        grid = ag.Grid.uniform(shape_2d=(10, 10), pixel_scales=0.05, sub_size=2)

        deflections = sie.deflections_via_potential_from_grid(grid=grid)

        deflections_first_binned_pixel = (
            deflections[0] + deflections[1] + deflections[2] + deflections[3]
        ) / 4

        assert deflections.in_1d_binned[0] == pytest.approx(
            deflections_first_binned_pixel, 1e-4
        )

        deflections_100th_binned_pixel = (
            deflections[399] + deflections[398] + deflections[397] + deflections[396]
        ) / 4

        assert deflections.in_1d_binned[99] == pytest.approx(
            deflections_100th_binned_pixel, 1e-4
        )

        jacobian = sie.jacobian_from_grid(grid=grid)

        jacobian_1st_pixel_binned_up = (
            jacobian[0][0][0]
            + jacobian[0][0][1]
            + jacobian[0][0][2]
            + jacobian[0][0][3]
        ) / 4

        assert jacobian[0][0].in_2d_binned.shape == (10, 10)
        assert jacobian[0][0].sub_shape_2d == (20, 20)
        assert jacobian[0][0].in_1d_binned[0] == pytest.approx(
            jacobian_1st_pixel_binned_up, 1e-4
        )

        jacobian_last_pixel_binned_up = (
            jacobian[0][0][399]
            + jacobian[0][0][398]
            + jacobian[0][0][397]
            + jacobian[0][0][396]
        ) / 4

        assert jacobian[0][0].in_1d_binned[99] == pytest.approx(
            jacobian_last_pixel_binned_up, 1e-4
        )

        shear_via_jacobian = sie.shear_via_jacobian_from_grid(grid=grid)

        shear_1st_pixel_binned_up = (
            shear_via_jacobian[0]
            + shear_via_jacobian[1]
            + shear_via_jacobian[2]
            + shear_via_jacobian[3]
        ) / 4

        assert shear_via_jacobian.in_1d_binned[0] == pytest.approx(
            shear_1st_pixel_binned_up, 1e-4
        )

        shear_last_pixel_binned_up = (
            shear_via_jacobian[399]
            + shear_via_jacobian[398]
            + shear_via_jacobian[397]
            + shear_via_jacobian[396]
        ) / 4

        assert shear_via_jacobian.in_1d_binned[99] == pytest.approx(
            shear_last_pixel_binned_up, 1e-4
        )

        tangential_eigen_values = sie.tangential_eigen_value_from_grid(grid=grid)

        first_pixel_binned_up = (
            tangential_eigen_values[0]
            + tangential_eigen_values[1]
            + tangential_eigen_values[2]
            + tangential_eigen_values[3]
        ) / 4

        assert tangential_eigen_values.in_1d_binned[0] == pytest.approx(
            first_pixel_binned_up, 1e-4
        )

        pixel_10000_from_av_sub_grid = (
            tangential_eigen_values[399]
            + tangential_eigen_values[398]
            + tangential_eigen_values[397]
            + tangential_eigen_values[396]
        ) / 4

        assert tangential_eigen_values.in_1d_binned[99] == pytest.approx(
            pixel_10000_from_av_sub_grid, 1e-4
        )

        radial_eigen_values = sie.radial_eigen_value_from_grid(grid=grid)

        first_pixel_binned_up = (
            radial_eigen_values[0]
            + radial_eigen_values[1]
            + radial_eigen_values[2]
            + radial_eigen_values[3]
        ) / 4

        assert radial_eigen_values.in_1d_binned[0] == pytest.approx(
            first_pixel_binned_up, 1e-4
        )

        pixel_10000_from_av_sub_grid = (
            radial_eigen_values[399]
            + radial_eigen_values[398]
            + radial_eigen_values[397]
            + radial_eigen_values[396]
        ) / 4

        assert radial_eigen_values.in_1d_binned[99] == pytest.approx(
            pixel_10000_from_av_sub_grid, 1e-4
        )
