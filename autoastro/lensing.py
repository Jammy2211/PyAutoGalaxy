import numpy as np
from astropy import cosmology as cosmo
from skimage import measure

from autoarray.structures import grids
from autoastro import dimensions as dim
from autoastro.util import cosmology_util


class LensingObject(object):

    axis_ratio = None

    def convergence_func(self, eta):
        raise NotImplementedError("surface_density_func should be overridden")

    def convergence_from_grid(self, grid):
        raise NotImplementedError("surface_density_from_grid should be overridden")

    def potential_func(self, u, y, x):
        raise NotImplementedError("potential_func should be overridden")

    def potential_from_grid(self, grid):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement potential_from_grid"
        )

    def deflections_from_grid(self, grid):
        raise NotImplementedError("deflections_from_grid should be overridden")

    @property
    def unit_length(self):
        raise NotImplementedError("unit_length should be overridden")

    @property
    def unit_mass(self):
        raise NotImplementedError("unit_mass should be overridden")

    def mass_integral(self, x):
        """Routine to integrate an elliptical light profiles - set axis ratio to 1 to compute the luminosity within a \
        circle"""
        return 2 * np.pi * x * self.convergence_func(x)

    def deflections_via_potential_from_grid(self, grid):

        potential = self.potential_from_grid(grid=grid)

        deflections_y_2d = np.gradient(potential.in_2d, grid.in_2d[:, 0, 0], axis=0)
        deflections_x_2d = np.gradient(potential.in_2d, grid.in_2d[0, :, 1], axis=1)

        return grid.mapping.grid_from_sub_grid_2d(
            sub_grid_2d=np.stack((deflections_y_2d, deflections_x_2d), axis=-1)
        )

    def jacobian_a11_from_grid(self, grid):

        deflections = self.deflections_from_grid(grid=grid)

        return grid.mapping.array_from_sub_array_2d(
            sub_array_2d=1.0
            - np.gradient(deflections.in_2d[:, :, 1], grid.in_2d[0, :, 1], axis=1)
        )

    def jacobian_a12_from_grid(self, grid):

        deflections = self.deflections_from_grid(grid=grid)

        return grid.mapping.array_from_sub_array_2d(
            sub_array_2d=-1.0
            * np.gradient(deflections.in_2d[:, :, 1], grid.in_2d[:, 0, 0], axis=0)
        )

    def jacobian_a21_from_grid(self, grid):

        deflections = self.deflections_from_grid(grid=grid)

        return grid.mapping.array_from_sub_array_2d(
            sub_array_2d=-1.0
            * np.gradient(deflections.in_2d[:, :, 0], grid.in_2d[0, :, 1], axis=1)
        )

    def jacobian_a22_from_grid(self, grid):

        deflections = self.deflections_from_grid(grid=grid)

        return grid.mapping.array_from_sub_array_2d(
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

        return grid.mapping.array_from_sub_array_1d(sub_array_1d=convergence)

    def shear_via_jacobian_from_grid(self, grid):

        jacobian = self.jacobian_from_grid(grid=grid)

        gamma_1 = 0.5 * (jacobian[1][1] - jacobian[0][0])
        gamma_2 = -0.5 * (jacobian[0][1] + jacobian[1][0])

        return grid.mapping.array_from_sub_array_1d(
            sub_array_1d=(gamma_1 ** 2 + gamma_2 ** 2) ** 0.5
        )

    def tangential_eigen_value_from_grid(self, grid):

        convergence = self.convergence_via_jacobian_from_grid(grid=grid)

        shear = self.shear_via_jacobian_from_grid(grid=grid)

        return grid.mapping.array_from_sub_array_1d(
            sub_array_1d=1 - convergence - shear
        )

    def radial_eigen_value_from_grid(self, grid):

        convergence = self.convergence_via_jacobian_from_grid(grid=grid)

        shear = self.shear_via_jacobian_from_grid(grid=grid)

        return grid.mapping.array_from_sub_array_1d(
            sub_array_1d=1 - convergence + shear
        )

    def magnification_from_grid(self, grid):

        jacobian = self.jacobian_from_grid(grid=grid)

        det_jacobian = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]

        return grid.mapping.array_from_sub_array_1d(sub_array_1d=1 / det_jacobian)

    def tangential_critical_curve_from_grid(self, grid):

        tangential_eigen_values = self.tangential_eigen_value_from_grid(grid=grid)

        tangential_critical_curve_indices = measure.find_contours(
            tangential_eigen_values.in_2d, 0
        )

        if len(tangential_critical_curve_indices) == 0:
            return []

        tangential_critical_curve = grid.geometry.grid_arcsec_from_grid_pixels_1d_for_marching_squares(
            grid_pixels_1d=tangential_critical_curve_indices[0],
            shape_2d=tangential_eigen_values.sub_shape_2d,
        )

        return grids.GridIrregular(grid=tangential_critical_curve)

    def radial_critical_curve_from_grid(self, grid):

        radial_eigen_values = self.radial_eigen_value_from_grid(grid=grid)

        radial_critical_curve_indices = measure.find_contours(
            radial_eigen_values.in_2d, 0
        )

        if len(radial_critical_curve_indices) == 0:
            return []

        radial_critical_curve = grid.geometry.grid_arcsec_from_grid_pixels_1d_for_marching_squares(
            grid_pixels_1d=radial_critical_curve_indices[0],
            shape_2d=radial_eigen_values.sub_shape_2d,
        )

        return grids.GridIrregular(grid=radial_critical_curve)

    def tangential_caustic_from_grid(self, grid):

        tangential_critical_curve = self.tangential_critical_curve_from_grid(grid=grid)

        if len(tangential_critical_curve) == 0:
            return []

        deflections_critical_curve = self.deflections_from_grid(
            grid=tangential_critical_curve
        )

        return tangential_critical_curve - deflections_critical_curve

    def radial_caustic_from_grid(self, grid):

        radial_critical_curve = self.radial_critical_curve_from_grid(grid=grid)

        if len(radial_critical_curve) == 0:
            return []

        deflections_critical_curve = self.deflections_from_grid(
            grid=radial_critical_curve
        )

        return radial_critical_curve - deflections_critical_curve

    def critical_curves_from_grid(self, grid):
        return [
            self.tangential_critical_curve_from_grid(grid=grid),
            self.radial_critical_curve_from_grid(grid=grid),
        ]

    def caustics_from_grid(self, grid):
        return [
            self.tangential_caustic_from_grid(grid=grid),
            self.radial_caustic_from_grid(grid=grid),
        ]

    def area_within_tangential_critical_curve(self, grid):

        critical_curve = self.critical_curves_from_grid(grid=grid)[0]
        x, y = critical_curve[:, 0], critical_curve[:, 1]

        return np.abs(0.5 * np.sum(y[:-1] * np.diff(x) - x[:-1] * np.diff(y)))

    def einstein_radius_in_units(
        self, grid, unit_length="arcsec", redshift_object=None, cosmology=cosmo.Planck15
    ):

        area = self.area_within_tangential_critical_curve(grid=grid)

        einstein_radius = dim.Length(
            value=np.sqrt(area / np.pi), unit_length=self.unit_length
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
        grid,
        redshift_object=None,
        redshift_source=None,
        cosmology=cosmo.Planck15,
        unit_mass="solMass",
        **kwargs,
    ):
        radius = self.einstein_radius_in_units(grid=grid)
        einstein_mass = dim.Mass(np.pi * (radius ** 2))

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
