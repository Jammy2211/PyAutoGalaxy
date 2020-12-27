import inspect
from functools import wraps

import numpy as np

from autoarray.inversion import mappers
from autoarray.plot.plotter import plotter as aa_plotter
from autoarray.plot.mat_wrap import mat_base, mat_structure, mat_obj
from autogalaxy.plot.mat_wrap import lensing_mat_obj as lmo
from autogalaxy.plot.plotter import lensing_include


class LensingPlotter(aa_plotter.Plotter):
    def __init__(
        self,
        units: mat_base.Units = mat_base.Units(),
        figure: mat_base.Figure = mat_base.Figure(),
        cmap: mat_base.Cmap = mat_base.Cmap(),
        colorbar: mat_base.Colorbar = mat_base.Colorbar(),
        tickparams: mat_base.TickParams = mat_base.TickParams(),
        yticks: mat_base.YTicks = mat_base.YTicks(),
        xticks: mat_base.XTicks = mat_base.XTicks(),
        title: mat_base.Title = mat_base.Title(),
        ylabel: mat_base.YLabel = mat_base.YLabel(),
        xlabel: mat_base.XLabel = mat_base.XLabel(),
        legend: mat_base.Legend = mat_base.Legend(),
        output: mat_base.Output = mat_base.Output(),
        array_overlay: mat_structure.ArrayOverlay = mat_structure.ArrayOverlay(),
        grid_scatter: mat_structure.GridScatter = mat_structure.GridScatter(),
        line_plot: mat_structure.LinePlot = mat_structure.LinePlot(),
        vector_field_quiver: mat_structure.VectorFieldQuiver = mat_structure.VectorFieldQuiver(),
        patch_overlay: mat_structure.PatchOverlay = mat_structure.PatchOverlay(),
        voronoi_drawer: mat_structure.VoronoiDrawer = mat_structure.VoronoiDrawer(),
        origin_scatter: mat_obj.OriginScatter = mat_obj.OriginScatter(),
        mask_scatter: mat_obj.MaskScatter = mat_obj.MaskScatter(),
        border_scatter: mat_obj.BorderScatter = mat_obj.BorderScatter(),
        positions_scatter: mat_obj.PositionsScatter = mat_obj.PositionsScatter(),
        index_scatter: mat_obj.IndexScatter = mat_obj.IndexScatter(),
        pixelization_grid_scatter: mat_obj.PixelizationGridScatter = mat_obj.PixelizationGridScatter(),
        light_profile_centres_scatter: lmo.LightProfileCentresScatter = lmo.LightProfileCentresScatter(),
        mass_profile_centres_scatter: lmo.MassProfileCentresScatter = lmo.MassProfileCentresScatter(),
        multiple_images_scatter: lmo.MultipleImagesScatter = lmo.MultipleImagesScatter(),
        critical_curves_plot: lmo.CriticalCurvesPlot = lmo.CriticalCurvesPlot(),
        caustics_plot: lmo.CausticsPlot = lmo.CausticsPlot(),
    ):
        """
        Visualizes data structures (e.g an `Array`, `Grid`, `VectorField`, etc.) using Matplotlib.
        
        The `Plotter` is passed objects from the `mat_wrap` package which wrap matplotlib plot functions and 
        customize the appearance of the plots of the data structure. If the values of these matplotlib wrapper 
        objects are not manually specified, they assume the default values provided in 
        the `config.visualize.mat_*` `.ini` config files.
        
        The following data structures can be plotted using the following matplotlib functions:
        
        - `Array`:, using `plt.imshow`.
        - `Grid`: using `plt.scatter`.
        - `Line`: using `plt.plot`, `plt.semilogy`, `plt.loglog` or `plt.scatter`.
        - `VectorField`: using `plt.quiver`.
        - `RectangularMapper`: using `plt.imshow`.
        - `VoronoiMapper`: using `plt.fill`.
        
        Parameters
        ----------
        units : mat_base.Units
          The units of the figure used to plot the data structure which sets the y and x ticks and labels.
        figure : mat_base.Figure
          Opens the matplotlib figure before plotting via `plt.figure` and closes it once plotting is complete
          via `plt.close`
        cmap : mat_base.Cmap
          Customizes the colormap of the plot and its normalization via matplotlib `colors` objects such 
          as `colors.Normalize` and `colors.LogNorm`.
        colorbar : mat_base.Colorbar
          Plots the colorbar of the plot via `plt.colorbar` and customizes its appearance and ticks using methods
          like `cb.set_yticklabels` and `cb.ax.tick_params`.
        tickparams : mat_base.TickParams
          Customizes the appearances of the y and x ticks on the plot, (e.g. the fontsize), using `plt.tick_params`.
        yticks : mat_base.YTicks
          Sets the yticks of the plot, including scaling them to new units depending on the `Units` object, via
          `plt.yticks`.
        xticks : mat_base.XTicks
          Sets the xticks of the plot, including scaling them to new units depending on the `Units` object, via
          `plt.xticks`.
        title : mat_base.Title
          Sets the figure title and customizes its appearance using `plt.title`.        
        ylabel : mat_base.YLabel
          Sets the figure ylabel and customizes its appearance using `plt.ylabel`.
        xlabel : mat_base.XLabel
          Sets the figure xlabel and customizes its appearance using `plt.xlabel`.
        legend : mat_base.Legend
          Sets whether the plot inclues a legend and customizes its appearance and labels using `plt.legend`.
        output : mat_base.Output
          Sets if the figure is displayed on the user's screen or output to `.png` using `plt.show` and `plt.savefig`
        array_overlay: mat_structure.ArrayOverlay
          Overlays an input `Array` over the figure using `plt.imshow`.
        grid_scatter : mat_structure.GridScatter
          Scatters a `Grid` of (y,x) coordinates over the figure using `plt.scatter`.
        line_plot: mat_structure.LinePlot
          Plots lines of data (e.g. a y versus x plot via `plt.plot`, vertical lines via `plt.avxline`, etc.)
        vector_field_quiver: mat_structure.VectorFieldQuiver
          Plots a `VectorField` object using the matplotlib function `plt.quiver`.
        patch_overlay: mat_structure.PatchOverlay
          Overlays matplotlib `patches.Patch` objects over the figure, such as an `Ellipse`.
        voronoi_drawer: mat_structure.VoronoiDrawer
          Draws a colored Voronoi mesh of pixels using `plt.fill`.
        origin_scatter : mat_obj.OriginScatter
          Scatters the (y,x) origin of the data structure on the figure.
        mask_scatter : mat_obj.MaskScatter
          Scatters an input `Mask2d` over the plotted data structure's figure.
        border_scatter : mat_obj.BorderScatter
          Scatters the border of an input `Mask2d` over the plotted data structure's figure.
        positions_scatter : mat_obj.PositionsScatter
          Scatters specific (y,x) coordinates input as a `GridIrregular` object over the figure.
        index_scatter : mat_obj.IndexScatter
          Scatters specific coordinates of an input `Grid` based on input values of the `Grid`'s 1D or 2D indexes.
        pixelization_grid_scatter : mat_obj.PixelizationGridScatter
          Scatters the `PixelizationGrid` of a `Pixelization` object.
        light_profile_centres_scatter : lensing_mat_obj.LightProfileCentresScatter
          Scatters the (y,x) centres of all `LightProfile`'s in the plotted object (e.g. a `Tracer`).
        mass_profile_centres_scatter : lensing_mat_obj.MassProfileCentresScatter
          Scatters the (y,x) centres of all `MassProfile`'s in the plotted object (e.g. a `Tracer`).
        light_profile_centres_scatter : lensing_mat_obj.MultipleImagesScatter
          Scatters the (y,x) coordinates of the multiple image locations of the lens mass model.
        critical_curves_plot : lensing_mat_obj.CriticalCurvesPlot
            Plots the critical curves of the lens mass model as colored lines.
        caustics_plot : lensing_mat_obj.CauticsPlot
            Plots the caustics of the lens mass model as colored lines.
        """

        self.light_profile_centres_scatter = light_profile_centres_scatter
        self.mass_profile_centres_scatter = mass_profile_centres_scatter
        self.multiple_images_scatter = multiple_images_scatter
        self.critical_curves_plot = critical_curves_plot
        self.caustics_plot = caustics_plot

        super(LensingPlotter, self).__init__(
            units=units,
            figure=figure,
            cmap=cmap,
            colorbar=colorbar,
            legend=legend,
            title=title,
            tickparams=tickparams,
            yticks=yticks,
            xticks=xticks,
            ylabel=ylabel,
            xlabel=xlabel,
            output=output,
            origin_scatter=origin_scatter,
            mask_scatter=mask_scatter,
            border_scatter=border_scatter,
            grid_scatter=grid_scatter,
            positions_scatter=positions_scatter,
            index_scatter=index_scatter,
            pixelization_grid_scatter=pixelization_grid_scatter,
            vector_field_quiver=vector_field_quiver,
            patch_overlay=patch_overlay,
            array_overlay=array_overlay,
            line_plot=line_plot,
            voronoi_drawer=voronoi_drawer,
        )

    def plot_lensing_attributes(
        self,
        light_profile_centres=None,
        mass_profile_centres=None,
        multiple_images=None,
        critical_curves=None,
        caustics=None,
    ):

        if light_profile_centres is not None:
            self.light_profile_centres_scatter.scatter_grid_grouped(
                grid_grouped=light_profile_centres
            )

        if mass_profile_centres is not None:
            self.mass_profile_centres_scatter.scatter_grid_grouped(
                grid_grouped=mass_profile_centres
            )

        if multiple_images is not None:
            self.multiple_images_scatter.scatter_grid_grouped(
                grid_grouped=multiple_images
            )

        if critical_curves is not None:
            self.critical_curves_plot.plot_grid_grouped(grid_grouped=critical_curves)

        if caustics is not None:
            self.caustics_plot.plot_grid_grouped(grid_grouped=caustics)

    def plot_array(
        self,
        array,
        mask=None,
        positions=None,
        grid=None,
        lines=None,
        vector_field=None,
        extent_manual=None,
        patches=None,
        array_overlay=None,
        light_profile_centres=None,
        mass_profile_centres=None,
        multiple_images=None,
        critical_curves=None,
        caustics=None,
        include_origin=False,
        include_border=False,
        bypass_output=False,
    ):
        """Plot an array of data as a figure.

        Parameters
        -----------
        array : data.array.aa.Scaled
            The 2D array of data which is plotted.
        origin : (float, float).
            The origin of the coordinate system of the array, which is plotted as an 'x' on the image if input.
        mask : data.array.mask.Mask2D
            The mask applied to the array, the edge of which is plotted as a set of points over the plotted array.
        border : bool
            If a mask is supplied, its borders pixels (e.g. the exterior edge) is plotted if this is `True`.
        positions : [[]]
            Lists of (y,x) coordinates on the image which are plotted as colored dots, to highlight specific pixels.
        grid : data.array.aa.Grid
            A grid of (y,x) coordinates which may be plotted over the plotted array.

        """

        if array is None or np.all(array == 0):
            return

        array = array.in_1d_binned

        if array.mask.is_all_false:
            buffer = 0
        else:
            buffer = 1

        array = array.zoomed_around_mask(buffer=buffer)

        super(LensingPlotter, self).plot_array(
            array=array,
            mask=mask,
            positions=positions,
            grid=grid,
            lines=lines,
            vector_field=vector_field,
            patches=patches,
            array_overlay=array_overlay,
            extent_manual=extent_manual,
            include_origin=include_origin,
            include_border=include_border,
            bypass_output=True,
        )

        self.plot_lensing_attributes(
            light_profile_centres=light_profile_centres,
            mass_profile_centres=mass_profile_centres,
            multiple_images=multiple_images,
            critical_curves=critical_curves,
            caustics=caustics,
        )

        if not bypass_output:
            self.output.to_figure(structure=array)

        if not self.for_subplot and not bypass_output:
            self.figure.close()

    def plot_grid(
        self,
        grid,
        color_array=None,
        axis_limits=None,
        indexes=None,
        positions=None,
        light_profile_centres=None,
        mass_profile_centres=None,
        multiple_images=None,
        critical_curves=None,
        caustics=None,
        include_origin=False,
        include_border=False,
        symmetric_around_centre=True,
        bypass_output=False,
    ):
        """Plot a grid of (y,x) Cartesian coordinates as a scatter plotter of points.

        Parameters
        -----------
        grid : data.array.aa.Grid
            The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2).
        axis_limits : []
            The axis limits of the figure on which the grid is plotted, following [xmin, xmax, ymin, ymax].
        indexes : []
            A set of points that are plotted in a different colour for emphasis (e.g. to show the mappings between \
            different planes).
        as_subplot : bool
            Whether the grid is plotted as part of a subplot, in which case the grid figure is not opened / closed.
        label_yunits : str
            The label of the unit_label of the y / x axis of the plots.

        """

        super(LensingPlotter, self).plot_grid(
            grid=grid,
            color_array=color_array,
            axis_limits=axis_limits,
            indexes=indexes,
            positions=positions,
            symmetric_around_centre=symmetric_around_centre,
            include_origin=include_origin,
            include_border=include_border,
            bypass_output=True,
        )

        self.plot_lensing_attributes(
            light_profile_centres=light_profile_centres,
            mass_profile_centres=mass_profile_centres,
            multiple_images=multiple_images,
            critical_curves=critical_curves,
            caustics=caustics,
        )

        if not bypass_output:
            self.output.to_figure(structure=grid)

        if not self.for_subplot and not bypass_output:
            self.figure.close()

    def plot_line(
        self,
        y,
        x,
        label=None,
        plot_axis_type="semilogy",
        vertical_lines=None,
        vertical_line_labels=None,
        bypass_output=False,
    ):

        super(LensingPlotter, self).plot_line(
            y=y,
            x=x,
            label=label,
            plot_axis_type=plot_axis_type,
            vertical_lines=vertical_lines,
            vertical_line_labels=vertical_line_labels,
            bypass_output=True,
        )

        if not bypass_output:
            self.output.to_figure(structure=None)

        if not self.for_subplot and not bypass_output:
            self.figure.close()

    def plot_mapper(
        self,
        mapper,
        source_pixel_values=None,
        positions=None,
        light_profile_centres=None,
        mass_profile_centres=None,
        multiple_images=None,
        critical_curves=None,
        caustics=None,
        include_origin=False,
        include_pixelization_grid=False,
        include_grid=False,
        include_border=False,
        image_pixel_indexes=None,
        source_pixel_indexes=None,
        bypass_output=False,
    ):

        if isinstance(mapper, mappers.MapperRectangular):

            self.plot_rectangular_mapper(
                mapper=mapper,
                source_pixel_values=source_pixel_values,
                positions=positions,
                light_profile_centres=light_profile_centres,
                mass_profile_centres=mass_profile_centres,
                multiple_images=multiple_images,
                critical_curves=critical_curves,
                caustics=caustics,
                include_origin=include_origin,
                include_grid=include_grid,
                include_pixelization_grid=include_pixelization_grid,
                include_border=include_border,
                image_pixel_indexes=image_pixel_indexes,
                source_pixel_indexes=source_pixel_indexes,
            )

        else:

            self.plot_voronoi_mapper(
                mapper=mapper,
                source_pixel_values=source_pixel_values,
                positions=positions,
                light_profile_centres=light_profile_centres,
                mass_profile_centres=mass_profile_centres,
                multiple_images=multiple_images,
                critical_curves=critical_curves,
                caustics=caustics,
                include_origin=include_origin,
                include_grid=include_grid,
                include_pixelization_grid=include_pixelization_grid,
                include_border=include_border,
                image_pixel_indexes=image_pixel_indexes,
                source_pixel_indexes=source_pixel_indexes,
            )

    def plot_rectangular_mapper(
        self,
        mapper,
        source_pixel_values=None,
        positions=None,
        light_profile_centres=None,
        mass_profile_centres=None,
        multiple_images=None,
        critical_curves=None,
        caustics=None,
        include_origin=False,
        include_pixelization_grid=False,
        include_grid=False,
        include_border=False,
        image_pixel_indexes=None,
        source_pixel_indexes=None,
        bypass_output=False,
    ):

        super(LensingPlotter, self).plot_rectangular_mapper(
            mapper=mapper,
            source_pixel_values=source_pixel_values,
            positions=positions,
            include_origin=include_origin,
            include_pixelization_grid=include_pixelization_grid,
            include_grid=include_grid,
            include_border=include_border,
            image_pixel_indexes=image_pixel_indexes,
            source_pixel_indexes=source_pixel_indexes,
            bypass_output=True,
        )

        self.plot_lensing_attributes(
            light_profile_centres=light_profile_centres,
            mass_profile_centres=mass_profile_centres,
            multiple_images=multiple_images,
            critical_curves=critical_curves,
            caustics=caustics,
        )

        if not bypass_output:
            self.output.to_figure(structure=None)

        if not self.for_subplot and not bypass_output:
            self.figure.close()

    def plot_voronoi_mapper(
        self,
        mapper,
        source_pixel_values=None,
        positions=None,
        light_profile_centres=None,
        mass_profile_centres=None,
        multiple_images=None,
        critical_curves=None,
        caustics=None,
        include_origin=False,
        include_pixelization_grid=False,
        include_grid=False,
        include_border=False,
        image_pixel_indexes=None,
        source_pixel_indexes=None,
        bypass_output=False,
    ):

        super(LensingPlotter, self).plot_voronoi_mapper(
            mapper=mapper,
            source_pixel_values=source_pixel_values,
            positions=positions,
            include_origin=include_origin,
            include_pixelization_grid=include_pixelization_grid,
            include_grid=include_grid,
            include_border=include_border,
            image_pixel_indexes=image_pixel_indexes,
            source_pixel_indexes=source_pixel_indexes,
            bypass_output=True,
        )

        self.plot_lensing_attributes(
            light_profile_centres=light_profile_centres,
            mass_profile_centres=mass_profile_centres,
            multiple_images=multiple_images,
            critical_curves=critical_curves,
            caustics=caustics,
        )

        if not bypass_output:
            self.output.to_figure(structure=None)

        if not self.for_subplot and not bypass_output:
            self.figure.close()


def set_plotter_for_figure(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        plotter_key = aa_plotter.plotter_key_from_dictionary(dictionary=kwargs)

        if plotter_key is not None:
            plotter = kwargs[plotter_key]
        else:
            plotter = LensingPlotter()
            plotter_key = "plotter"

        plotter.for_subplot = False

        kwargs[plotter_key] = plotter

        return func(*args, **kwargs)

    return wrapper


def set_plotter_for_subplot(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        plotter_key = aa_plotter.plotter_key_from_dictionary(dictionary=kwargs)

        if plotter_key is not None:
            plotter = kwargs[plotter_key]
        else:
            plotter = LensingPlotter()
            plotter_key = "plotter"

        plotter.for_subplot = True

        kwargs[plotter_key] = plotter

        return func(*args, **kwargs)

    return wrapper


def plot_array(
    array,
    mask=None,
    positions=None,
    grid=None,
    vector_field=None,
    patches=None,
    array_overlay=None,
    extent_manual=None,
    light_profile_centres=None,
    mass_profile_centres=None,
    multiple_images=None,
    critical_curves=None,
    caustics=None,
    include=None,
    plotter=None,
):
    if include is None:
        include = lensing_include.Include()

    if plotter is None:
        plotter = LensingPlotter()

    plotter.plot_array(
        array=array,
        mask=mask,
        vector_field=vector_field,
        patches=patches,
        array_overlay=array_overlay,
        extent_manual=extent_manual,
        light_profile_centres=light_profile_centres,
        mass_profile_centres=mass_profile_centres,
        multiple_images=multiple_images,
        critical_curves=critical_curves,
        caustics=caustics,
        positions=positions,
        grid=grid,
        include_origin=include.origin,
        include_border=include.border,
    )


def plot_grid(
    grid,
    color_array=None,
    axis_limits=None,
    indexes=None,
    positions=None,
    light_profile_centres=None,
    mass_profile_centres=None,
    multiple_images=None,
    critical_curves=None,
    caustics=None,
    symmetric_around_centre=True,
    include=None,
    plotter=None,
):
    if include is None:
        include = lensing_include.Include()

    if plotter is None:
        plotter = LensingPlotter()

    plotter.plot_grid(
        grid=grid,
        color_array=color_array,
        axis_limits=axis_limits,
        indexes=indexes,
        positions=positions,
        light_profile_centres=light_profile_centres,
        mass_profile_centres=mass_profile_centres,
        multiple_images=multiple_images,
        critical_curves=critical_curves,
        caustics=caustics,
        symmetric_around_centre=symmetric_around_centre,
        include_origin=include.origin,
        include_border=include.border,
    )


def plot_line(
    y,
    x,
    label=None,
    plot_axis_type="semilogy",
    vertical_lines=None,
    vertical_line_labels=None,
    plotter=None,
):
    if plotter is None:
        plotter = LensingPlotter()

    plotter.plot_line(
        y=y,
        x=x,
        label=label,
        plot_axis_type=plot_axis_type,
        vertical_lines=vertical_lines,
        vertical_line_labels=vertical_line_labels,
    )


def plot_mapper_obj(
    mapper,
    light_profile_centres=None,
    mass_profile_centres=None,
    multiple_images=None,
    critical_curves=None,
    caustics=None,
    image_pixel_indexes=None,
    source_pixel_indexes=None,
    include=None,
    plotter=None,
):
    if include is None:
        include = lensing_include.Include()

    if plotter is None:
        plotter = LensingPlotter()

    plotter.plot_mapper(
        mapper=mapper,
        include_grid=include.inversion_grid,
        include_pixelization_grid=include.inversion_pixelization_grid,
        include_border=include.inversion_border,
        light_profile_centres=light_profile_centres,
        mass_profile_centres=mass_profile_centres,
        multiple_images=multiple_images,
        critical_curves=critical_curves,
        caustics=caustics,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
    )
