import numpy as np
from autoarray.structures import grids
from autogalaxy.profiles import geometry_profiles
from autogalaxy.profiles import mass_profiles as mp
from autogalaxy import convert
import typing

from scipy.interpolate import griddata
from autogalaxy import exc


class MassSheet(geometry_profiles.SphericalProfile, mp.MassProfile):
    def __init__(
        self, centre: typing.Tuple[float, float] = (0.0, 0.0), kappa: float = 0.0
    ):
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

    @grids.grid_like_to_structure
    def convergence_from_grid(self, grid):
        return np.full(shape=grid.shape[0], fill_value=self.kappa)

    @grids.grid_like_to_structure
    def potential_from_grid(self, grid):
        return np.zeros(shape=grid.shape[0])

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def deflections_from_grid(self, grid):
        grid_radii = self.grid_to_grid_radii(grid=grid)
        return self.grid_to_grid_cartesian(grid=grid, radius=self.kappa * grid_radii)

    @property
    def is_mass_sheet(self):
        return True


# noinspection PyAbstractClass
class ExternalShear(geometry_profiles.EllipticalProfile, mp.MassProfile):
    def __init__(self, elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0)):
        """
        An `ExternalShear` term, to model the line-of-sight contribution of other galaxies / satellites.

        The shear angle phi is defined in the direction of stretching of the image. Therefore, if an object located \
        outside the lens is responsible for the shear, it will be offset 90 degrees from the value of phi.

        Parameters
        ----------
        magnitude : float
            The overall magnitude of the shear (gamma).
        phi : float
            The rotation axis of the shear.
        """

        super(ExternalShear, self).__init__(
            centre=(0.0, 0.0), elliptical_comps=elliptical_comps
        )

        magnitude, phi = convert.shear_magnitude_and_phi_from(
            elliptical_comps=elliptical_comps
        )

        self.magnitude = magnitude
        self.phi = phi

    def convergence_func(self, grid_radius):
        return 0.0

    def average_convergence_of_1_radius(self):
        return 0.0

    @grids.grid_like_to_structure
    def convergence_from_grid(self, grid):
        return np.zeros(shape=grid.shape[0])

    @grids.grid_like_to_structure
    def potential_from_grid(self, grid):
        return np.zeros(shape=grid.shape[0])

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def deflections_from_grid(self, grid):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : aa.Grid2D
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
        deflection_y = -np.multiply(self.magnitude, grid[:, 0])
        deflection_x = np.multiply(self.magnitude, grid[:, 1])
        return self.rotate_grid_from_profile(np.vstack((deflection_y, deflection_x)).T)


class InputDeflections(mp.MassProfile):
    def __init__(
        self,
        deflections_y,
        deflections_x,
        image_plane_grid,
        preload_grid=None,
        preload_blurring_grid=None,
        #      normalization_scale: float = 1.0,
    ):
        """
        Represents a known deflection angle map (e.g. from an already performed lens model or particle simulation
        of a mass distribution) which can be used for model fitting.

        The image-plane grid of the delflection angles is used to align an input grid to the input deflections, so that
        a new deflection angle map can be computed via interpolation using the scipy.interpolate.griddata method.

        A normalization scale can be included, which scales the overall normalization of the deflection angle map
        interpolated by a multiplicative factor.

        Parameters
        ----------
        deflections_y : aa.Array2D
            The input array of the y components of the deflection angles.
        deflections_x : aa.Array2D
            The input array of the x components of the deflection angles.
        image_plane_grid : aa.Grid2D
            The image-plane grid from which the deflection angles are defined.
        grid_interp : aa.Grid2D
            The grid that interpolated quantities are computed on. If this is input in advance, the interpolation
            weights can be precomputed to speed up the calculation time.
        normalization_scale : float
            The calculated deflection angles are multiplied by this factor scaling their values up and doown.
        """
        super().__init__()

        self.deflections_y = deflections_y
        self.deflections_x = deflections_x

        self.image_plane_grid = image_plane_grid

        self.centre = image_plane_grid.origin

        self.preload_grid = preload_grid
        self.preload_deflections = None
        self.preload_blurring_grid = preload_blurring_grid
        self.preload_blurring_deflections = None

        if self.preload_grid is not None:
            self.normalization_scale = 1.0
            self.preload_deflections = self.deflections_from_grid(grid=preload_grid)

        if self.preload_blurring_grid is not None:
            self.normalization_scale = 1.0
            self.preload_blurring_deflections = self.deflections_from_grid(
                grid=preload_blurring_grid
            )

        self.normalization_scale = 1.0  # normalization_scale

    @grids.grid_like_to_structure
    def convergence_from_grid(self, grid):
        return self.convergence_via_jacobian_from_grid(grid=grid)

    @grids.grid_like_to_structure
    def potential_from_grid(self, grid):
        return np.zeros(shape=grid.shape[0])

    @grids.grid_like_to_structure
    def deflections_from_grid(self, grid):

        if self.preload_grid is not None and self.preload_deflections is not None:

            if grid.sub_shape_slim == self.preload_grid.sub_shape_slim:
                if np.allclose(grid, self.preload_grid, 1e-8):
                    return self.normalization_scale * self.preload_deflections

        if (
            self.preload_blurring_grid is not None
            and self.preload_blurring_deflections is not None
        ):

            if grid.sub_shape_slim == self.preload_blurring_grid.sub_shape_slim:
                if np.allclose(grid, self.preload_blurring_grid, 1e-8):
                    return self.normalization_scale * self.preload_blurring_deflections

        deflections_y = self.normalization_scale * griddata(
            points=self.image_plane_grid, values=self.deflections_y, xi=grid
        )
        deflections_x = self.normalization_scale * griddata(
            points=self.image_plane_grid, values=self.deflections_x, xi=grid
        )

        if np.isnan(deflections_y).any() or np.isnan(deflections_x).any():
            raise exc.ProfileException(
                "The grid input into the DefectionsInput.deflections_from_grid() method has (y,x)"
                "coodinates extending beyond the input image_plane_grid."
                ""
                "Update the image_plane_grid to include deflection angles reaching to larger"
                "radii or reduce the input grid. "
            )

        return np.stack((deflections_y, deflections_x), axis=-1)
