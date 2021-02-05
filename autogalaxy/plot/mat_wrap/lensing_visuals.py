from autoarray.mask import mask_2d
from autoarray.structures import arrays, grids, vector_fields
from autoarray.plot.mat_wrap import visuals as vis

from matplotlib import patches as ptch
import typing
from typing import List


class Visuals1D(vis.Visuals1D):

    pass


class Visuals2D(vis.Visuals2D):
    def __init__(
        self,
        origin: grids.Grid2D = None,
        border: grids.Grid2D = None,
        mask: mask_2d.Mask2D = None,
        positions: grids.Grid2DIrregular = None,
        grid: grids.Grid2D = None,
        pixelization_grid: grids.Grid2D = None,
        vector_field: vector_fields.VectorField2DIrregular = None,
        patches: typing.Union[ptch.Patch] = None,
        array_overlay: arrays.Array2D = None,
        light_profile_centres: grids.Grid2DIrregular = None,
        mass_profile_centres: grids.Grid2DIrregular = None,
        multiple_images: grids.Grid2DIrregular = None,
        critical_curves: grids.Grid2DIrregular = None,
        caustics: grids.Grid2DIrregular = None,
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
            plotter.light_profile_centres_scatter.scatter_grid_grouped(
                grid_grouped=self.light_profile_centres
            )

        if self.mass_profile_centres is not None:
            plotter.mass_profile_centres_scatter.scatter_grid_grouped(
                grid_grouped=self.mass_profile_centres
            )

        if self.multiple_images is not None:
            plotter.multiple_images_scatter.scatter_grid_grouped(
                grid_grouped=self.multiple_images
            )

        if self.critical_curves is not None:
            plotter.critical_curves_plot.plot_grid_grouped(
                grid_grouped=self.critical_curves
            )

        if self.caustics is not None:
            plotter.caustics_plot.plot_grid_grouped(grid_grouped=self.caustics)
