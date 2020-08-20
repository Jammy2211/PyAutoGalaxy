import copy
from functools import wraps
import inspect

import numpy as np
from autoconf import conf
from autoarray.inversion import mappers
from autoarray.plot import plotters
from autogalaxy import lensing
from autogalaxy.plot import lensing_mat_objs


class LensingPlotter(plotters.AbstractPlotter):
    def __init__(
        self,
        module=None,
        units=None,
        figure=None,
        cmap=None,
        cb=None,
        legend=None,
        ticks=None,
        labels=None,
        output=None,
        origin_scatterer=None,
        mask_scatterer=None,
        border_scatterer=None,
        grid_scatterer=None,
        positions_scatterer=None,
        index_scatterer=None,
        pixelization_grid_scatterer=None,
        array_overlayer=None,
        liner=None,
        voronoi_drawer=None,
        light_profile_centres_scatterer=None,
        mass_profile_centres_scatterer=None,
        multiple_images_scatterer=None,
        critical_curves_liner=None,
        caustics_liner=None,
    ):

        super(LensingPlotter, self).__init__(
            module=module,
            units=units,
            figure=figure,
            cmap=cmap,
            cb=cb,
            legend=legend,
            ticks=ticks,
            labels=labels,
            output=output,
            origin_scatterer=origin_scatterer,
            mask_scatterer=mask_scatterer,
            border_scatterer=border_scatterer,
            grid_scatterer=grid_scatterer,
            positions_scatterer=positions_scatterer,
            index_scatterer=index_scatterer,
            pixelization_grid_scatterer=pixelization_grid_scatterer,
            array_overlayer=array_overlayer,
            liner=liner,
            voronoi_drawer=voronoi_drawer,
        )

        if isinstance(self, Plotter):
            from_subplot_config = False
        else:
            from_subplot_config = True

        self.light_profile_centres_scatterer = (
            light_profile_centres_scatterer
            if light_profile_centres_scatterer is not None
            else lensing_mat_objs.LightProfileCentreScatterer(
                from_subplot_config=from_subplot_config
            )
        )

        self.mass_profile_centres_scatterer = (
            mass_profile_centres_scatterer
            if mass_profile_centres_scatterer is not None
            else lensing_mat_objs.MassProfileCentreScatterer(
                from_subplot_config=from_subplot_config
            )
        )

        self.multiple_images_scatterer = (
            multiple_images_scatterer
            if multiple_images_scatterer is not None
            else lensing_mat_objs.MultipleImagesScatterer(
                from_subplot_config=from_subplot_config
            )
        )

        self.critical_curves_liner = (
            critical_curves_liner
            if critical_curves_liner is not None
            else lensing_mat_objs.CriticalCurvesLiner(
                from_subplot_config=from_subplot_config
            )
        )

        self.caustics_liner = (
            caustics_liner
            if caustics_liner is not None
            else lensing_mat_objs.CausticsLiner(from_subplot_config=from_subplot_config)
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
            self.light_profile_centres_scatterer.scatter_coordinates(
                coordinates=light_profile_centres
            )

        if mass_profile_centres is not None:
            self.mass_profile_centres_scatterer.scatter_coordinates(
                coordinates=mass_profile_centres
            )

        if multiple_images is not None:
            self.multiple_images_scatterer.scatter_coordinates(
                coordinates=multiple_images
            )

        if critical_curves is not None:
            self.critical_curves_liner.draw_coordinates(coordinates=critical_curves)

        if caustics is not None:
            self.caustics_liner.draw_coordinates(coordinates=caustics)

    def plot_array(
        self,
        array,
        mask=None,
        positions=None,
        grid=None,
        lines=None,
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
        mask : data.array.mask.Mask
            The mask applied to the array, the edge of which is plotted as a set of points over the plotted array.
        extract_array_from_mask : bool
            The plotter array is extracted using the mask, such that masked values are plotted as zeros. This ensures \
            bright features outside the mask do not impact the color map of the plotters.
        zoom_around_mask : bool
            If True, the 2D region of the array corresponding to the rectangle encompassing all unmasked values is \
            plotted, thereby zooming into the region of interest.
        border : bool
            If a mask is supplied, its borders pixels (e.g. the exterior edge) is plotted if this is *True*.
        positions : [[]]
            Lists of (y,x) coordinates on the image which are plotted as colored dots, to highlight specific pixels.
        grid : data.array.aa.Grid
            A grid of (y,x) coordinates which may be plotted over the plotted array.
        as_subplot : bool
            Whether the array is plotted as part of a subplot, in which case the grid figure is not opened / closed.
        unit_label : str
            The label for the unit_label of the y / x axis of the plots.
        unit_conversion_factor : float or None
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        figsize : (int, int)
            The size of the figure in (rows, columns).
        aspect : str
            The aspect ratio of the array, specifically whether it is forced to be square ('equal') or adapts its size to \
            the figure size ('auto').
        cmap : str
            The colormap the array is plotted using, which may be chosen from the standard matplotlib colormaps.
        norm : str
            The normalization of the colormap used to plotters the image, specifically whether it is linear ('linear'), log \
            ('log') or a symmetric log normalization ('symmetric_log').
        norm_min : float or None
            The minimum array value the colormap map spans (all values below this value are plotted the same color).
        norm_max : float or None
            The maximum array value the colormap map spans (all values above this value are plotted the same color).
        linthresh : float
            For the 'symmetric_log' colormap normalization ,this specifies the range of values within which the colormap \
            is linear.
        linscale : float
            For the 'symmetric_log' colormap normalization, this allowws the linear range set by linthresh to be stretched \
            relative to the logarithmic range.
        cb_ticksize : int
            The size of the tick labels on the colorbar.
        cb_fraction : float
            The fraction of the figure that the colorbar takes up, which resizes the colorbar relative to the figure.
        cb_pad : float
            Pads the color bar in the figure, which resizes the colorbar relative to the figure.
        xsize : int
            The fontsize of the x axes label.
        ysize : int
            The fontsize of the y axes label.
        xyticksize : int
            The font size of the x and y ticks on the figure axes.
        mask_scatterer : int
            The size of the points plotted to show the mask.
        xticks_manual :  [] or None
            If input, the xticks do not use the array's default xticks but instead overwrite them as these values.
        yticks_manual :  [] or None
            If input, the yticks do not use the array's default yticks but instead overwrite them as these values.
        output_path : str
            The path on the hard-disk where the figure is output.
        output_filename : str
            The filename of the figure that is output.
        output_format : str
            The format the figue is output:
            'show' - display on computer screen.
            'png' - output to hard-disk as a png.
            'fits' - output to hard-disk as a fits file.'

        Returns
        --------
        None

        Examples
        --------
            plotters.plot_array(
            array=image, origin=(0.0, 0.0), mask=circular_mask,
            border=False, points=[[1.0, 1.0], [2.0, 2.0]], grid=None, as_subplot=False,
            unit_label='scaled', kpc_per_arcsec=None, figsize=(7,7), aspect='auto',
            cmap='jet', norm='linear, norm_min=None, norm_max=None, linthresh=None, linscale=None,
            cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
            title='Image', titlesize=16, xsize=16, ysize=16, xyticksize=16,
            mask_scatterer=10, border_pointsize=2, position_pointsize=10, grid_pointsize=10,
            xticks_manual=None, yticks_manual=None,
            output_path='/path/to/output', output_format='png', output_filename='image')
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
            array_overlay=array_overlay,
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
        """Plot a grid of (y,x) Cartesian coordinates as a scatter plotters of points.

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
        unit_conversion_factor : float
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        figsize : (int, int)
            The size of the figure in (rows, columns).
        pointsize : int
            The size of the points plotted on the grid.
        xyticksize : int
            The font size of the x and y ticks on the figure axes.
        title : str
            The text of the title.
        titlesize : int
            The size of of the title of the figure.
        xsize : int
            The fontsize of the x axes label.
        ysize : int
            The fontsize of the y axes label.
        output_path : str
            The path on the hard-disk where the figure is output.
        output_filename : str
            The filename of the figure that is output.
        output_format : str
            The format the figue is output:
            'show' - display on computer screen.
            'png' - output to hard-disk as a png.
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


class Plotter(LensingPlotter, plotters.Plotter):
    def __init__(
        self,
        module=None,
        units=None,
        figure=None,
        cmap=None,
        cb=None,
        ticks=None,
        labels=None,
        legend=None,
        output=None,
        origin_scatterer=None,
        mask_scatterer=None,
        border_scatterer=None,
        grid_scatterer=None,
        positions_scatterer=None,
        index_scatterer=None,
        pixelization_grid_scatterer=None,
        array_overlayer=None,
        liner=None,
        voronoi_drawer=None,
        light_profile_centres_scatterer=None,
        mass_profile_centres_scatterer=None,
        multiple_images_scatterer=None,
        critical_curves_liner=None,
        caustics_liner=None,
    ):

        super(Plotter, self).__init__(
            module=module,
            units=units,
            figure=figure,
            cmap=cmap,
            cb=cb,
            legend=legend,
            ticks=ticks,
            labels=labels,
            output=output,
            origin_scatterer=origin_scatterer,
            mask_scatterer=mask_scatterer,
            border_scatterer=border_scatterer,
            grid_scatterer=grid_scatterer,
            positions_scatterer=positions_scatterer,
            index_scatterer=index_scatterer,
            pixelization_grid_scatterer=pixelization_grid_scatterer,
            array_overlayer=array_overlayer,
            liner=liner,
            voronoi_drawer=voronoi_drawer,
            light_profile_centres_scatterer=light_profile_centres_scatterer,
            mass_profile_centres_scatterer=mass_profile_centres_scatterer,
            multiple_images_scatterer=multiple_images_scatterer,
            critical_curves_liner=critical_curves_liner,
            caustics_liner=caustics_liner,
        )


class SubPlotter(LensingPlotter, plotters.SubPlotter):
    def __init__(
        self,
        module=None,
        units=None,
        figure=None,
        cmap=None,
        cb=None,
        legend=None,
        ticks=None,
        labels=None,
        output=None,
        origin_scatterer=None,
        mask_scatterer=None,
        border_scatterer=None,
        grid_scatterer=None,
        positions_scatterer=None,
        index_scatterer=None,
        pixelization_grid_scatterer=None,
        array_overlayer=None,
        liner=None,
        voronoi_drawer=None,
        light_profile_centres_scatterer=None,
        mass_profile_centres_scatterer=None,
        multiple_images_scatterer=None,
        critical_curves_liner=None,
        caustics_liner=None,
    ):

        super(SubPlotter, self).__init__(
            module=module,
            units=units,
            figure=figure,
            cmap=cmap,
            cb=cb,
            legend=legend,
            ticks=ticks,
            labels=labels,
            output=output,
            origin_scatterer=origin_scatterer,
            mask_scatterer=mask_scatterer,
            border_scatterer=border_scatterer,
            grid_scatterer=grid_scatterer,
            positions_scatterer=positions_scatterer,
            index_scatterer=index_scatterer,
            pixelization_grid_scatterer=pixelization_grid_scatterer,
            array_overlayer=array_overlayer,
            liner=liner,
            voronoi_drawer=voronoi_drawer,
            light_profile_centres_scatterer=light_profile_centres_scatterer,
            mass_profile_centres_scatterer=mass_profile_centres_scatterer,
            multiple_images_scatterer=multiple_images_scatterer,
            critical_curves_liner=critical_curves_liner,
            caustics_liner=caustics_liner,
        )


class Include(plotters.Include):
    def __init__(
        self,
        origin=None,
        mask=None,
        grid=None,
        border=None,
        positions=None,
        light_profile_centres=None,
        mass_profile_centres=None,
        critical_curves=None,
        caustics=None,
        multiple_images=None,
        inversion_pixelization_grid=None,
        inversion_grid=None,
        inversion_border=None,
        inversion_image_pixelization_grid=None,
        preloaded_critical_curves=None,
        preload_caustics=None,
    ):

        super(Include, self).__init__(
            origin=origin,
            mask=mask,
            grid=grid,
            border=border,
            inversion_pixelization_grid=inversion_pixelization_grid,
            inversion_grid=inversion_grid,
            inversion_border=inversion_border,
            inversion_image_pixelization_grid=inversion_image_pixelization_grid,
        )

        self.positions = self.load_include(value=positions, name="positions")
        self.light_profile_centres = self.load_include(
            value=light_profile_centres, name="light_profile_centres"
        )
        self.mass_profile_centres = self.load_include(
            value=mass_profile_centres, name="mass_profile_centres"
        )
        self.critical_curves = self.load_include(
            value=critical_curves, name="critical_curves"
        )

        self.caustics = self.load_include(value=caustics, name="caustics")
        self.multiple_images = self.load_include(
            value=multiple_images, name="multiple_images"
        )

        self.preloaded_critical_curves = preloaded_critical_curves
        self.preloaded_caustics = preload_caustics

    @staticmethod
    def load_include(value, name):

        return (
            conf.instance.visualize_general.get(
                section_name="include", attribute_name=name, attribute_type=bool
            )
            if value is None
            else value
        )

    def positions_from_masked_dataset(self, masked_dataset):

        if self.positions:
            return masked_dataset.positions

    def light_profile_centres_from_obj(self, obj):

        if self.light_profile_centres:
            return obj.light_profile_centres

    def mass_profile_centres_from_obj(self, obj):

        if self.mass_profile_centres:
            return obj.mass_profile_centres

    def critical_curves_from_obj(self, obj):

        if not hasattr(obj, "has_mass_profile"):
            return None

        if not self.critical_curves or not obj.has_mass_profile:
            return None

        if self.preloaded_caustics is not None:
            return self.preloaded_critical_curves

        if isinstance(obj, lensing.LensingObject):
            try:
                return obj.critical_curves
            except ValueError:
                print(
                    "Critical curve could not be calculated due to an unphysical mass model"
                )

    def caustics_from_obj(self, obj):

        if not hasattr(obj, "has_mass_profile"):
            return None

        if not self.caustics or not obj.has_mass_profile:
            return None

        if self.preloaded_caustics is not None:
            return self.preloaded_caustics

        if isinstance(obj, lensing.LensingObject):

            try:
                return obj.caustics
            except ValueError:
                print(
                    "Caustics could not be calculated due to an unphysical mass model"
                )

    def positions_from_fit(self, fit):
        """Get the masks of the fit if the masks should be plotted on the fit.

        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractLensHyperFit
            The fit to the datas, which includes a lisrt of every model image, residual_map, chi-squareds, etc.
        mask : bool
            If *True*, the masks is plotted on the fit's datas.
        """
        if self.positions:
            try:
                return fit.settings_masked_dataset.positions
            except AttributeError:
                return None

    def traced_grid_of_plane_from_fit_and_plane_index(self, fit, plane_index):

        if self.positions is True:
            return fit.tracer.traced_grids_of_planes_from_grid(grid=fit.grid)[
                plane_index
            ]

    def positions_of_plane_from_fit_and_plane_index(self, fit, plane_index):

        if self.positions is True:
            positions = self.positions_from_fit(fit=fit)
            if positions is None:
                return None

            return fit.tracer.traced_grids_of_planes_from_grid(grid=positions)[
                plane_index
            ]

    def inversion_image_pixelization_grid_from_fit(self, fit):

        if fit.inversion is not None:
            if self.inversion_image_pixelization_grid:
                if fit.inversion.mapper.is_image_plane_pixelization:
                    return fit.tracer.sparse_image_plane_grids_of_planes_from_grid(
                        grid=fit.grid
                    )[-1]

    def new_include_with_preloaded_critical_curves_and_caustics(
        self, preloaded_critical_curves, preloaded_caustics
    ):

        include = copy.deepcopy(self)
        include.preloaded_critical_curves = preloaded_critical_curves
        include.preloaded_caustics = preloaded_caustics

        return include


def set_include_and_plotter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        include_key = plotters.include_key_from_dictionary(dictionary=kwargs)

        if include_key is not None:
            include = kwargs[include_key]
        else:
            include = Include()
            include_key = "include"

        kwargs[include_key] = include

        plotter_key = plotters.plotter_key_from_dictionary(dictionary=kwargs)

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

        include_key = plotters.include_key_from_dictionary(dictionary=kwargs)

        if include_key is not None:
            include = kwargs[include_key]
        else:
            include = Include()
            include_key = "include"

        kwargs[include_key] = include

        sub_plotter_key = plotters.plotter_key_from_dictionary(dictionary=kwargs)

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
    array_overlay=None,
    light_profile_centres=None,
    mass_profile_centres=None,
    multiple_images=None,
    critical_curves=None,
    caustics=None,
    include=None,
    plotter=None,
):

    if include is None:
        include = Include()

    if plotter is None:
        plotter = Plotter()

    plotter.plot_array(
        array=array,
        mask=mask,
        array_overlay=array_overlay,
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
        include = Include()

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
        include = Include()

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
