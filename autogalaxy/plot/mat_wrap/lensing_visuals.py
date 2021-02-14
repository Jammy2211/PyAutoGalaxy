from autoarray.mask import mask_2d
from autoarray.structures.arrays.two_d import array_2d
from autoarray.structures.grids.two_d import grid_2d
from autoarray.structures.grids.two_d import grid_2d_irregular
from autoarray.structures.vector_fields import vector_field_irregular
from autoarray.plot.mat_wrap import visuals as vis

from matplotlib import patches as ptch
import typing
from typing import List


class Visuals1D(vis.Visuals1D):

    pass


class Visuals2D(vis.Visuals2D):
    def __init__(
        self,
        origin: grid_2d.Grid2D = None,
        border: grid_2d.Grid2D = None,
        mask: mask_2d.Mask2D = None,
        positions: grid_2d_irregular.Grid2DIrregular = None,
        grid: grid_2d.Grid2D = None,
        pixelization_grid: grid_2d.Grid2D = None,
        vector_field: vector_field_irregular.VectorField2DIrregular = None,
        patches: typing.Union[ptch.Patch] = None,
        array_overlay: array_2d.Array2D = None,
        light_profile_centres: grid_2d_irregular.Grid2DIrregular = None,
        mass_profile_centres: grid_2d_irregular.Grid2DIrregular = None,
        multiple_images: grid_2d_irregular.Grid2DIrregular = None,
        critical_curves: grid_2d_irregular.Grid2DIrregular = None,
        caustics: grid_2d_irregular.Grid2DIrregular = None,
        indexes: typing.Union[List[int], List[List[int]]] = None,
        pixelization_indexes: typing.Union[List[int], List[List[int]]] = None,
    ):

        super().__init__(
            mask=mask,
            positions=positions,
            grid=grid,
            pixelization_grid=pixelization_grid,
            vector_field=vector_field,
            patches=patches,
            array_overlay=array_overlay,
            origin=origin,
            border=border,
            indexes=indexes,
            pixelization_indexes=pixelization_indexes,
        )

        self.light_profile_centres = light_profile_centres
        self.mass_profile_centres = mass_profile_centres
        self.multiple_images = multiple_images
        self.critical_curves = critical_curves
        self.caustics = caustics

    def plot_via_plotter(self, plotter, grid_indexes=None, mapper=None):

        super().plot_via_plotter(
            plotter=plotter, grid_indexes=grid_indexes, mapper=mapper
        )

        if self.light_profile_centres is not None:
            plotter.light_profile_centres_scatter.scatter_grid(
                grid=self.light_profile_centres
            )

        if self.mass_profile_centres is not None:
            plotter.mass_profile_centres_scatter.scatter_grid(
                grid=self.mass_profile_centres
            )

        if self.multiple_images is not None:
            plotter.multiple_images_scatter.scatter_grid(grid=self.multiple_images)

        if self.critical_curves is not None:
            try:
                plotter.critical_curves_plot.plot_grid(grid=self.critical_curves)
            except TypeError:
                pass

        if self.caustics is not None:
            try:
                plotter.caustics_plot.plot_grid(grid=self.caustics)
            except TypeError:
                pass
