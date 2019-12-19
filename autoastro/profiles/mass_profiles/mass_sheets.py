import numpy as np
from astropy import cosmology as cosmo

import autofit as af
from autoarray.structures import arrays, grids
from autoastro import dimensions as dim
from autoastro.profiles import geometry_profiles
from autoastro.profiles import mass_profiles as mp


class MassSheet(geometry_profiles.SphericalProfile, mp.MassProfile):
    @af.map_types
    def __init__(self, centre: dim.Position = (0.0, 0.0), kappa: float = 0.0):
        """
        Represents a mass-sheet

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        kappa : float
            The magnitude of the convergence of the mass-sheet.
        """
        super(MassSheet, self).__init__(centre=centre)
        self.kappa = kappa

    def convergence_func(self, grid_radius):
        return 0.0

    @grids.convert_positions_to_grid
    def convergence_from_grid(self, grid):
        return grid.mapping.array_stored_1d_from_sub_array_1d(
            sub_array_1d=np.full(shape=grid.sub_shape_1d, fill_value=self.kappa)
        )

    @grids.convert_positions_to_grid
    def potential_from_grid(self, grid):
        return grid.mapping.array_stored_1d_from_sub_array_1d(
            sub_array_1d=np.zeros(shape=grid.sub_shape_1d)
        )

    @grids.convert_positions_to_grid
    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def deflections_from_grid(self, grid):
        grid_radii = self.grid_to_grid_radii(grid=grid)
        return self.grid_to_grid_cartesian(grid=grid, radius=self.kappa * grid_radii)


# noinspection PyAbstractClass
class ExternalShear(geometry_profiles.EllipticalProfile, mp.MassProfile):
    @af.map_types
    def __init__(self, magnitude: float = 0.2, phi: float = 0.0):
        """
        An external shear term, to model the line-of-sight contribution of other galaxies / satellites.

        The shear angle phi is defined in the direction of stretching of the image. Therefore, if an object located \
        outside the lens is responsible for the shear, it will be offset 90 degrees from the value of phi.

        Parameters
        ----------
        magnitude : float
            The overall magnitude of the shear (gamma).
        phi : float
            The rotation axis of the shear.
        """

        super(ExternalShear, self).__init__(centre=(0.0, 0.0), phi=phi, axis_ratio=1.0)
        self.magnitude = magnitude

    def convergence_func(self, grid_radius):
        return 0.0

    def average_convergence_of_1_radius_in_units(
        self,
        unit_length="arcsec",
        redshift_profile=None,
        cosmology=cosmo.Planck15,
        **kwargs,
    ):
        return dim.Length(value=0.0, unit_length=self.unit_length)

    @grids.convert_positions_to_grid
    def convergence_from_grid(self, grid):
        return grid.mapping.array_stored_1d_from_sub_array_1d(
            sub_array_1d=np.zeros(shape=grid.sub_shape_1d)
        )

    @grids.convert_positions_to_grid
    def potential_from_grid(self, grid):
        return grid.mapping.array_stored_1d_from_sub_array_1d(
            sub_array_1d=np.zeros(shape=grid.sub_shape_1d)
        )

    @grids.convert_positions_to_grid
    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
        deflection_y = -np.multiply(self.magnitude, grid[:, 0])
        deflection_x = np.multiply(self.magnitude, grid[:, 1])
        return self.rotate_grid_from_profile(np.vstack((deflection_y, deflection_x)).T)
