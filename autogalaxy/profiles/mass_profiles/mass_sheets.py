import numpy as np
from scipy.interpolate import griddata
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass_profiles import MassProfile

from autogalaxy import convert
from autogalaxy import exc


class MassSheet(MassProfile):
    def __init__(self, centre: Tuple[float, float] = (0.0, 0.0), kappa: float = 0.0):
        """
        Represents a mass-sheet

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        kappa
            The magnitude of the convergence of the mass-sheet.
        """
        super().__init__(centre=centre, elliptical_comps=(0.0, 0.0))
        self.kappa = kappa

    def convergence_func(self, grid_radius: float) -> float:
        return 0.0

    @aa.grid_dec.grid_2d_to_structure
    def convergence_2d_from(self, grid: aa.type.Grid2DLike):
        return np.full(shape=grid.shape[0], fill_value=self.kappa)

    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        return np.zeros(shape=grid.shape[0])

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        grid_radii = self.grid_to_grid_radii(grid=grid)
        return self.grid_to_grid_cartesian(grid=grid, radius=self.kappa * grid_radii)


# noinspection PyAbstractClass
class ExternalShear(MassProfile):
    def __init__(self, elliptical_comps: Tuple[float, float] = (0.0, 0.0)):
        """
        An `ExternalShear` term, to model the line-of-sight contribution of other galaxies / satellites.

        The shear angle is defined in the direction of stretching of the image. Therefore, if an object located \
        outside the lens is responsible for the shear, it will be offset 90 degrees from the value of angle.

        Parameters
        ----------
        magnitude
            The overall magnitude of the shear (gamma).
        angle
            The rotation axis of the shear.
        """

        super().__init__(centre=(0.0, 0.0), elliptical_comps=elliptical_comps)

    @property
    def magnitude(self):
        return convert.shear_magnitude_from(elliptical_comps=self.elliptical_comps)

    @property
    def angle(self):
        return convert.shear_angle_from(elliptical_comps=self.elliptical_comps)

    def convergence_func(self, grid_radius: float) -> float:
        return 0.0

    def average_convergence_of_1_radius(self):
        return 0.0

    @aa.grid_dec.grid_2d_to_structure
    def convergence_2d_from(self, grid: aa.type.Grid2DLike):
        return np.zeros(shape=grid.shape[0])

    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        return np.zeros(shape=grid.shape[0])

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
        deflection_y = -np.multiply(self.magnitude, grid[:, 0])
        deflection_x = np.multiply(self.magnitude, grid[:, 1])
        return self.rotate_grid_from_reference_frame(
            np.vstack((deflection_y, deflection_x)).T
        )


class InputDeflections(MassProfile):
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
        image_plane_grid
            The image-plane grid from which the deflection angles are defined.
        grid_interp : aa.Grid2D
            The grid that interpolated quantities are computed on. If this is input in advance, the interpolation
            weight_list can be precomputed to speed up the calculation time.
        normalization_scale
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
            self.preload_deflections = self.deflections_yx_2d_from(grid=preload_grid)

        if self.preload_blurring_grid is not None:
            self.normalization_scale = 1.0
            self.preload_blurring_deflections = self.deflections_yx_2d_from(
                grid=preload_blurring_grid
            )

        self.normalization_scale = 1.0  # normalization_scale

    @aa.grid_dec.grid_2d_to_structure
    def convergence_2d_from(self, grid: aa.type.Grid2DLike):
        return self.convergence_2d_via_jacobian_from(grid=grid)

    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        return np.zeros(shape=grid.shape[0])

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):

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
                "The grid input into the DefectionsInput.deflections_yx_2d_from() method has (y,x)"
                "coodinates extending beyond the input image_plane_grid."
                ""
                "Update the image_plane_grid to include deflection angles reaching to larger"
                "radii or reduce the input grid. "
            )

        return np.stack((deflections_y, deflections_x), axis=-1)
