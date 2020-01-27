import numpy as np
import autofit as af
from scipy.optimize import root_scalar
from astropy import cosmology as cosmo
from skimage import measure

from autoarray.util import array_util
from autoarray.structures import grids
from autoastro import dimensions as dim
from autoastro.util import cosmology_util


class LensingObject(object):
    @property
    def mass_profiles(self):
        raise NotImplementedError("mass profiles list should be overriden")

    def convergence_func(self, grid_radius):
        raise NotImplementedError("convergence_func should be overridden")

    def convergence_from_grid(self, grid):
        raise NotImplementedError("convergence_from_grid should be overridden")

    def potential_func(self, u, y, x):
        raise NotImplementedError("potential_func should be overridden")

    def potential_from_grid(self, grid):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement potential_from_grid"
        )

    def deflections_from_grid(self, grid):
        raise NotImplementedError("deflections_from_grid should be overridden")

    @property
    def mass_profile_centres(self):
        raise NotImplementedError("mass profile centres should be overridden")

    @property
    def unit_length(self):
        raise NotImplementedError("unit_length should be overridden")

    @property
    def unit_mass(self):
        raise NotImplementedError("unit_mass should be overridden")

    def mass_integral(self, x):
        """Routine to integrate an elliptical light profiles - set axis ratio to 1 to compute the luminosity within a \
        circle"""
        return 2 * np.pi * x * self.convergence_func(grid_radius=x)

    def deflection_magnitudes_from_grid(self, grid):
        deflections = self.deflections_from_grid(grid=grid)
        return deflections.distances_from_coordinate(coordinate=(0.0, 0.0))

    def deflections_via_potential_from_grid(self, grid):

        potential = self.potential_from_grid(grid=grid)

        deflections_y_2d = np.gradient(potential.in_2d, grid.in_2d[:, 0, 0], axis=0)
        deflections_x_2d = np.gradient(potential.in_2d, grid.in_2d[0, :, 1], axis=1)

        return grid.mapping.grid_stored_1d_from_sub_grid_2d(
            sub_grid_2d=np.stack((deflections_y_2d, deflections_x_2d), axis=-1)
        )

    def jacobian_a11_from_grid(self, grid):

        deflections = self.deflections_from_grid(grid=grid)

        return grid.mapping.array_stored_1d_from_sub_array_2d(
            sub_array_2d=1.0
            - np.gradient(deflections.in_2d[:, :, 1], grid.in_2d[0, :, 1], axis=1)
        )

    def jacobian_a12_from_grid(self, grid):

        deflections = self.deflections_from_grid(grid=grid)

        return grid.mapping.array_stored_1d_from_sub_array_2d(
            sub_array_2d=-1.0
            * np.gradient(deflections.in_2d[:, :, 1], grid.in_2d[:, 0, 0], axis=0)
        )

    def jacobian_a21_from_grid(self, grid):

        deflections = self.deflections_from_grid(grid=grid)

        return grid.mapping.array_stored_1d_from_sub_array_2d(
            sub_array_2d=-1.0
            * np.gradient(deflections.in_2d[:, :, 0], grid.in_2d[0, :, 1], axis=1)
        )

    def jacobian_a22_from_grid(self, grid):

        deflections = self.deflections_from_grid(grid=grid)

        return grid.mapping.array_stored_1d_from_sub_array_2d(
            sub_array_2d=1
            - np.gradient(deflections.in_2d[:, :, 0], grid.in_2d[:, 0, 0], axis=0)
        )

    def jacobian_from_grid(self, grid):

        a11 = self.jacobian_a11_from_grid(grid=grid)

        a12 = self.jacobian_a12_from_grid(grid=grid)

        a21 = self.jacobian_a21_from_grid(grid=grid)

        a22 = self.jacobian_a22_from_grid(grid=grid)

        return [[a11, a12], [a21, a22]]

    def convergence_via_jacobian_from_grid(self, grid):

        jacobian = self.jacobian_from_grid(grid=grid)

        convergence = 1 - 0.5 * (jacobian[0][0] + jacobian[1][1])

        return grid.mapping.array_stored_1d_from_sub_array_1d(sub_array_1d=convergence)

    def shear_via_jacobian_from_grid(self, grid):

        jacobian = self.jacobian_from_grid(grid=grid)

        gamma_1 = 0.5 * (jacobian[1][1] - jacobian[0][0])
        gamma_2 = -0.5 * (jacobian[0][1] + jacobian[1][0])

        return grid.mapping.array_stored_1d_from_sub_array_1d(
            sub_array_1d=(gamma_1 ** 2 + gamma_2 ** 2) ** 0.5
        )

    def tangential_eigen_value_from_grid(self, grid):

        convergence = self.convergence_via_jacobian_from_grid(grid=grid)

        shear = self.shear_via_jacobian_from_grid(grid=grid)

        return grid.mapping.array_stored_1d_from_sub_array_1d(
            sub_array_1d=1 - convergence - shear
        )

    def radial_eigen_value_from_grid(self, grid):

        convergence = self.convergence_via_jacobian_from_grid(grid=grid)

        shear = self.shear_via_jacobian_from_grid(grid=grid)

        return grid.mapping.array_stored_1d_from_sub_array_1d(
            sub_array_1d=1 - convergence + shear
        )

    def magnification_from_grid(self, grid):

        jacobian = self.jacobian_from_grid(grid=grid)

        det_jacobian = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]

        return grid.mapping.array_stored_1d_from_sub_array_1d(
            sub_array_1d=1 / det_jacobian
        )

    @property
    def mass_profile_bounding_box(self):
        y_min = np.min(list(map(lambda centre: centre[0], self.mass_profile_centres)))
        y_max = np.max(list(map(lambda centre: centre[0], self.mass_profile_centres)))
        x_min = np.min(list(map(lambda centre: centre[1], self.mass_profile_centres)))
        x_max = np.max(list(map(lambda centre: centre[1], self.mass_profile_centres)))
        return [y_min, y_max, x_min, x_max]

    def convergence_bounding_box(self, convergence_threshold=0.02):

        if all(mass_profile.is_point_mass for mass_profile in self.mass_profiles):
            einstein_radius = sum(
                [
                    mass_profile.einstein_radius
                    for mass_profile in self.mass_profiles
                    if mass_profile.is_point_mass
                ]
            )
            return [
                -3.0 * einstein_radius,
                3.0 * einstein_radius,
                -3.0 * einstein_radius,
                3.0 * einstein_radius,
            ]

        [y_min, y_max, x_min, x_max] = self.mass_profile_bounding_box

        def func_for_y_min(y):
            grid = grids.GridIrregular.manual_1d(grid=[[y, x_max], [y, x_min]])
            return np.max(self.convergence_from_grid(grid=grid) - convergence_threshold)

        convergence_y_min = root_scalar(func_for_y_min, bracket=[y_min, -1000.0]).root

        def func_for_y_max(y):
            grid = grids.GridIrregular.manual_1d(grid=[[y, x_max], [y, x_min]])
            return np.min(self.convergence_from_grid(grid=grid) - convergence_threshold)

        convergence_y_max = root_scalar(func_for_y_max, bracket=[y_max, 1000.0]).root

        def func_for_x_min(x):
            grid = grids.GridIrregular.manual_1d(grid=[[y_max, x], [y_min, x]])
            return np.max(self.convergence_from_grid(grid=grid) - convergence_threshold)

        convergence_x_min = root_scalar(func_for_x_min, bracket=[x_min, -1000.0]).root

        def func_for_x_max(x):
            grid = grids.GridIrregular.manual_1d(grid=[[y_max, x], [y_min, x]])
            return np.min(self.convergence_from_grid(grid=grid) - convergence_threshold)

        convergence_x_max = root_scalar(func_for_x_max, bracket=[x_max, 1000.0]).root

        return [
            convergence_y_min,
            convergence_y_max,
            convergence_x_min,
            convergence_x_max,
        ]

    @property
    def calculation_grid(self):

        convergence_threshold = af.conf.instance.general.get(
            "calculation_grid", "convergence_threshold", float
        )

        pixels = af.conf.instance.general.get("calculation_grid", "pixels", int)

        # TODO : The error is raised for point mass profile which does not have a convergence, need to think how to
        # TODO : better deal with point masses.

        bounding_box = self.convergence_bounding_box(
            convergence_threshold=convergence_threshold
        )

        return grids.Grid.bounding_box(
            bounding_box=bounding_box,
            shape_2d=(pixels, pixels),
            buffer_around_corners=True,
        )

    @property
    def tangential_critical_curve(self):

        grid = self.calculation_grid

        tangential_eigen_values = self.tangential_eigen_value_from_grid(grid=grid)

        tangential_critical_curve_indices = measure.find_contours(
            tangential_eigen_values.in_2d, 0
        )

        if len(tangential_critical_curve_indices) == 0:
            return []

        tangential_critical_curve = grid.geometry.grid_scaled_from_grid_pixels_1d_for_marching_squares(
            grid_pixels_1d=tangential_critical_curve_indices[0],
            shape_2d=tangential_eigen_values.sub_shape_2d,
        )

        return grids.GridIrregular(grid=tangential_critical_curve)

    @property
    def radial_critical_curve(self):

        grid = self.calculation_grid

        radial_eigen_values = self.radial_eigen_value_from_grid(grid=grid)

        radial_critical_curve_indices = measure.find_contours(
            radial_eigen_values.in_2d, 0
        )

        if len(radial_critical_curve_indices) == 0:
            return []

        radial_critical_curve = grid.geometry.grid_scaled_from_grid_pixels_1d_for_marching_squares(
            grid_pixels_1d=radial_critical_curve_indices[0],
            shape_2d=radial_eigen_values.sub_shape_2d,
        )

        return grids.GridIrregular(grid=radial_critical_curve)

    @property
    @array_util.Memoizer()
    def critical_curves(self):
        return [self.tangential_critical_curve, self.radial_critical_curve]

    @property
    def tangential_caustic(self):

        tangential_critical_curve = self.tangential_critical_curve

        if len(tangential_critical_curve) == 0:
            return []

        deflections_critical_curve = self.deflections_from_grid(
            grid=tangential_critical_curve
        )

        return tangential_critical_curve - deflections_critical_curve

    @property
    def radial_caustic(self):

        radial_critical_curve = self.radial_critical_curve

        if len(radial_critical_curve) == 0:
            return []

        deflections_critical_curve = self.deflections_from_grid(
            grid=radial_critical_curve
        )

        return radial_critical_curve - deflections_critical_curve

    @property
    @array_util.Memoizer()
    def caustics(self):
        return [self.tangential_caustic, self.radial_caustic]

    @property
    @array_util.Memoizer()
    def area_within_tangential_critical_curve(self):

        tangential_critical_curve = self.tangential_critical_curve
        x, y = tangential_critical_curve[:, 0], tangential_critical_curve[:, 1]

        return np.abs(0.5 * np.sum(y[:-1] * np.diff(x) - x[:-1] * np.diff(y)))

    def einstein_radius_in_units(
        self, unit_length="arcsec", redshift_object=None, cosmology=cosmo.Planck15
    ):

        einstein_radius = dim.Length(
            value=np.sqrt(self.area_within_tangential_critical_curve / np.pi),
            unit_length=self.unit_length,
        )

        if unit_length is "kpc":

            kpc_per_arcsec = cosmology_util.kpc_per_arcsec_from_redshift_and_cosmology(
                redshift=redshift_object, cosmology=cosmology
            )

        else:

            kpc_per_arcsec = None

        return einstein_radius.convert(
            unit_length=unit_length, kpc_per_arcsec=kpc_per_arcsec
        )

    def einstein_mass_in_units(
        self,
        redshift_object=None,
        redshift_source=None,
        cosmology=cosmo.Planck15,
        unit_mass="solMass",
    ):

        einstein_radius = self.einstein_radius_in_units()
        einstein_mass = dim.Mass(np.pi * (einstein_radius ** 2))

        if unit_mass is "solMass":

            critical_surface_density = cosmology_util.critical_surface_density_between_redshifts_from_redshifts_and_cosmology(
                redshift_0=redshift_object,
                redshift_1=redshift_source,
                cosmology=cosmology,
                unit_length=self.unit_length,
                unit_mass=unit_mass,
            )

        else:

            critical_surface_density = None

        return einstein_mass.convert(
            unit_mass=unit_mass, critical_surface_density=critical_surface_density
        )
