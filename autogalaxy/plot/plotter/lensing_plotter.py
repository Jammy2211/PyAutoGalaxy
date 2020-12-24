import inspect
from functools import wraps

import numpy as np

from autoarray.inversion import mappers
from autoarray.plot.plotter import plotter as aa_plotter
from autogalaxy.plot.mat_wrap import lensing_mat_obj
from autogalaxy.plot.plotter import lensing_include


class LensingPlotter(aa_plotter.AbstractPlotter):
    def __init__(
        self,
        module=None,
        units=None,
        figure=None,
        cmap=None,
        cb=None,
        legend=None,
        title=None,
        tickparams=None,
        yticks=None,
        xticks=None,
        ylabel=None,
        xlabel=None,
        output=None,
        origin_scatter=None,
        mask_scatter=None,
        border_scatter=None,
        grid_scatter=None,
        positions_scatter=None,
        index_scatter=None,
        pixelization_grid_scatter=None,
        vector_quiver=None,
        patcher=None,
        array_over=None,
        line=None,
        voronoi_drawer=None,
        light_profile_centres_scatter=None,
        mass_profile_centres_scatter=None,
        multiple_images_scatter=None,
        critical_curves_line=None,
        caustics_line=None,
    ):

        super(LensingPlotter, self).__init__(
            module=module,
            units=units,
            figure=figure,
            cmap=cmap,
            cb=cb,
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
            vector_quiver=vector_quiver,
            patcher=patcher,
            array_over=array_over,
            line=line,
            voronoi_drawer=voronoi_drawer,
        )

        if isinstance(self, Plotter):
            use_subplot_defaults = False
        else:
            use_subplot_defaults = True

        self.light_profile_centres_scatter = (
            light_profile_centres_scatter
            if light_profile_centres_scatter is not None
            else lensing_mat_obj.LightProfileCentreScatter(
                use_subplot_defaults=use_subplot_defaults
            )
        )

        self.mass_profile_centres_scatter = (
            mass_profile_centres_scatter
            if mass_profile_centres_scatter is not None
            else lensing_mat_obj.MassProfileCentreScatter(
                use_subplot_defaults=use_subplot_defaults
            )
        )

        self.multiple_images_scatter = (
            multiple_images_scatter
            if multiple_images_scatter is not None
            else lensing_mat_obj.MultipleImagesScatter(
                use_subplot_defaults=use_subplot_defaults
            )
        )

        self.critical_curves_line = (
            critical_curves_line
            if critical_curves_line is not None
            else lensing_mat_obj.CriticalCurvesLine(
                use_subplot_defaults=use_subplot_defaults
            )
        )

        self.caustics_line = (
            caustics_line
            if caustics_line is not None
            else lensing_mat_obj.CausticsLine(use_subplot_defaults=use_subplot_defaults)
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
            self.critical_curves_line.plot_grid_grouped(grid_grouped=critical_curves)

        if caustics is not None:
            self.caustics_line.plot_grid_grouped(grid_grouped=caustics)

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
        array_over=None,
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
        settings : PlotterSettings
            Settings
        include : PlotterInclude
            Include
        labels : PlotterLabels
            labels
        outputs : PlotterOutputs
            outputs
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
            array_over=array_over,
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

        if not isinstance(self, SubPlotter) and not bypass_output:
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

        if not isinstance(self, SubPlotter) and not bypass_output:
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

        if not isinstance(self, SubPlotter) and not bypass_output:
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

        if not isinstance(self, SubPlotter) and not bypass_output:
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

        if not isinstance(self, SubPlotter) and not bypass_output:
            self.figure.close()


class Plotter(LensingPlotter, aa_plotter.Plotter):
    def __init__(
        self,
        module=None,
        units=None,
        figure=None,
        cmap=None,
        cb=None,
        title=None,
        tickparams=None,
        yticks=None,
        xticks=None,
        ylabel=None,
        xlabel=None,
        legend=None,
        output=None,
        origin_scatter=None,
        mask_scatter=None,
        border_scatter=None,
        grid_scatter=None,
        positions_scatter=None,
        index_scatter=None,
        pixelization_grid_scatter=None,
        vector_quiver=None,
        patcher=None,
        array_overer=None,
        line=None,
        voronoi_drawer=None,
        light_profile_centres_scatter=None,
        mass_profile_centres_scatter=None,
        multiple_images_scatter=None,
        critical_curves_line=None,
        caustics_line=None,
    ):
        super(Plotter, self).__init__(
            module=module,
            units=units,
            figure=figure,
            cmap=cmap,
            cb=cb,
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
            vector_quiver=vector_quiver,
            patcher=patcher,
            array_over=array_overer,
            line=line,
            voronoi_drawer=voronoi_drawer,
            light_profile_centres_scatter=light_profile_centres_scatter,
            mass_profile_centres_scatter=mass_profile_centres_scatter,
            multiple_images_scatter=multiple_images_scatter,
            critical_curves_line=critical_curves_line,
            caustics_line=caustics_line,
        )


class SubPlotter(LensingPlotter, aa_plotter.SubPlotter):
    def __init__(
        self,
        module=None,
        units=None,
        figure=None,
        cmap=None,
        cb=None,
        legend=None,
        title=None,
        tickparams=None,
        yticks=None,
        xticks=None,
        ylabel=None,
        xlabel=None,
        output=None,
        origin_scatter=None,
        mask_scatter=None,
        border_scatter=None,
        grid_scatter=None,
        positions_scatter=None,
        index_scatter=None,
        pixelization_grid_scatter=None,
        vector_quiver=None,
        patcher=None,
        array_over=None,
        line=None,
        voronoi_drawer=None,
        light_profile_centres_scatter=None,
        mass_profile_centres_scatter=None,
        multiple_images_scatter=None,
        critical_curves_line=None,
        caustics_line=None,
    ):
        super(SubPlotter, self).__init__(
            module=module,
            units=units,
            figure=figure,
            cmap=cmap,
            cb=cb,
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
            array_over=array_over,
            vector_quiver=vector_quiver,
            patcher=patcher,
            line=line,
            voronoi_drawer=voronoi_drawer,
            light_profile_centres_scatter=light_profile_centres_scatter,
            mass_profile_centres_scatter=mass_profile_centres_scatter,
            multiple_images_scatter=multiple_images_scatter,
            critical_curves_line=critical_curves_line,
            caustics_line=caustics_line,
        )


def set_include_and_plotter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        include_key = aa_plotter.include_key_from_dictionary(dictionary=kwargs)

        if include_key is not None:
            include = kwargs[include_key]
        else:
            include = lensing_include.Include()
            include_key = "include"

        kwargs[include_key] = include

        plotter_key = aa_plotter.plotter_key_from_dictionary(dictionary=kwargs)

        if plotter_key is not None:
            plotter = kwargs[plotter_key]
        else:
            plotter = Plotter(module=inspect.getmodule(func))
            plotter_key = "plotter"

        kwargs[plotter_key] = plotter

        return func(*args, **kwargs)

    return wrapper


def set_include_and_sub_plotter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        include_key = aa_plotter.include_key_from_dictionary(dictionary=kwargs)

        if include_key is not None:
            include = kwargs[include_key]
        else:
            include = lensing_include.Include()
            include_key = "include"

        kwargs[include_key] = include

        sub_plotter_key = aa_plotter.plotter_key_from_dictionary(dictionary=kwargs)

        if sub_plotter_key is not None:
            plotter = kwargs[sub_plotter_key]
        else:
            plotter = SubPlotter(module=inspect.getmodule(func))
            sub_plotter_key = "sub_plotter"

        kwargs[sub_plotter_key] = plotter

        return func(*args, **kwargs)

    return wrapper


def plot_array(
    array,
    mask=None,
    positions=None,
    grid=None,
    vector_field=None,
    patches=None,
    array_over=None,
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
        plotter = Plotter()

    plotter.plot_array(
        array=array,
        mask=mask,
        vector_field=vector_field,
        patches=patches,
        array_over=array_over,
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
        plotter = Plotter()

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
        plotter = Plotter()

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
        plotter = Plotter()

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
