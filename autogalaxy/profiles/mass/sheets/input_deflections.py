import numpy as np
from scipy.interpolate import griddata

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile

from autogalaxy import exc


class InputDeflections(MassProfile):
    def __init__(
        self,
        deflections_y: aa.Array2D,
        deflections_x: aa.Array2D,
        image_plane_grid: aa.type.Grid2DLike,
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
            try:
                if grid.sub_shape_slim == self.preload_grid.sub_shape_slim:
                    if np.allclose(grid, self.preload_grid, 1e-8):
                        return self.normalization_scale * self.preload_deflections
            except AttributeError:
                pass

        if (
            self.preload_blurring_grid is not None
            and self.preload_blurring_deflections is not None
        ):
            try:
                if grid.sub_shape_slim == self.preload_blurring_grid.sub_shape_slim:
                    if np.allclose(grid, self.preload_blurring_grid, 1e-8):
                        return (
                            self.normalization_scale * self.preload_blurring_deflections
                        )
            except AttributeError:
                pass

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
