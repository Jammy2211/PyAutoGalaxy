import autoarray.plot as aplt
from autogalaxy.plot.mat_wrap import lensing_wrap as lw


class MatPlot1D(aplt.MatPlot1D):
    def __init__(
        self,
        units: aplt.Units = aplt.Units(),
        figure: aplt.Figure = aplt.Figure(),
        axis: aplt.Axis = aplt.Axis(),
        cmap: aplt.Cmap = aplt.Cmap(),
        colorbar: aplt.Colorbar = aplt.Colorbar(),
        colorbar_tickparams: aplt.ColorbarTickParams = aplt.ColorbarTickParams(),
        tickparams: aplt.TickParams = aplt.TickParams(),
        yticks: aplt.YTicks = aplt.YTicks(),
        xticks: aplt.XTicks = aplt.XTicks(),
        title: aplt.Title = aplt.Title(),
        ylabel: aplt.YLabel = aplt.YLabel(),
        xlabel: aplt.XLabel = aplt.XLabel(),
        text: aplt.Text = aplt.Text(),
        legend: aplt.Legend = aplt.Legend(),
        output: aplt.Output = aplt.Output(),
        yx_plot: aplt.YXPlot = aplt.YXPlot(),
        half_light_radius_axvline: lw.HalfLightRadiusAXVLine = lw.HalfLightRadiusAXVLine(),
        einstein_radius_axvline: lw.EinsteinRadiusAXVLine = lw.EinsteinRadiusAXVLine(),
        model_fluxes_yx_scatter: lw.ModelFluxesYXScatter = lw.ModelFluxesYXScatter(),
        fill_between: aplt.FillBetween = aplt.FillBetween(),
    ):
        """
        Visualizes 1D data structures as a y versus x plot using Matplotlib.

        The `Plotter` is passed objects from the `wrap_base` package which wrap matplotlib plot functions and customize
        the appearance of the plots of the data structure. If the values of these matplotlib wrapper objects are not
        manually specified, they assume the default values provided in the `config.visualize.mat_*` `.ini` config files.

        Parameters
        ----------
        units : aplt.Units
            The units of the figure used to plot the data structure which sets the y and x ticks and labels.
        figure : aplt.Figure
            Opens the matplotlib figure before plotting via `plt.figure` and closes it once plotting is complete
            via `plt.close`.
        axis : aplt.Axis
            Sets the extent of the figure axis via `plt.axis` and allows for a manual axis range.
        cmap : aplt.Cmap
            Customizes the colormap of the plot and its normalization via matplotlib `colors` objects such
            as `colors.Normalize` and `colors.LogNorm`.
        colorbar : aplt.Colorbar
            Plots the colorbar of the plot via `plt.colorbar` and customizes its tick labels and values using method
            like `cb.set_yticklabels`.
        colorbar_tickparams : aplt.ColorbarTickParams
            Customizes the yticks of the colorbar plotted via `plt.colorbar`.
        tickparams : aplt.TickParams
            Customizes the appearances of the y and x ticks on the plot, (e.g. the fontsize), using `plt.tick_params`.
        yticks : aplt.YTicks
            Sets the yticks of the plot, including scaling them to new units depending on the `Units` object, via
            `plt.yticks`.
        xticks : aplt.XTicks
            Sets the xticks of the plot, including scaling them to new units depending on the `Units` object, via
            `plt.xticks`.
        title : aplt.Title
            Sets the figure title and customizes its appearance using `plt.title`.
        ylabel : aplt.YLabel
            Sets the figure ylabel and customizes its appearance using `plt.ylabel`.
        xlabel : aplt.XLabel
            Sets the figure xlabel and customizes its appearance using `plt.xlabel`.
        legend : aplt.Legend
            Sets whether the plot inclues a legend and customizes its appearance and labels using `plt.legend`.
        output : aplt.Output
            Sets if the figure is displayed on the user's screen or output to `.png` using `plt.show` and `plt.savefig`
        yx_plot : aplt.YXPlot
            Sets how the y versus x plot appears, for example if it each axis is linear or log, using `plt.plot`.
        half_light_radius_axvline : lw.HalfLightRadiusAXVLine
            Sets how a vertical line representing the half light radius of a `LightProfile` is plotted on the figure
            using the `plt.axvline` method.
        half_light_radius_axvline : lw.HalfLightRadiusAXVLine
            Sets how a vertical line representing the Einstein radius of a `LensingObj` (e.g. a `MassProfile`) is
            plotted on the figure using the `plt.axvline` method.
        """

        super().__init__(
            units=units,
            figure=figure,
            axis=axis,
            cmap=cmap,
            colorbar=colorbar,
            colorbar_tickparams=colorbar_tickparams,
            tickparams=tickparams,
            yticks=yticks,
            xticks=xticks,
            title=title,
            ylabel=ylabel,
            xlabel=xlabel,
            text=text,
            legend=legend,
            output=output,
            yx_plot=yx_plot,
            fill_between=fill_between,
        )

        self.half_light_radius_axvline = half_light_radius_axvline
        self.einstein_radius_axvline = einstein_radius_axvline
        self.model_fluxes_yx_scatter = model_fluxes_yx_scatter

    def set_for_multi_plot(self, is_for_multi_plot: bool, color: str):
        """
        Sets the `is_for_subplot` attribute for every `MatWrap` object in this `MatPlot` object by updating
        the `is_for_subplot`. By changing this tag:

            - The [subplot] section of the config file of every `MatWrap` object is used instead of [figure].
            - Calls which output or close the matplotlib figure are over-ridden so that the subplot is not removed.

        Parameters
        ----------
        is_for_subplot : bool
            The entry the `is_for_subplot` attribute of every `MatWrap` object is set too.
        """

        super().set_for_multi_plot(is_for_multi_plot=is_for_multi_plot, color=color)

        self.half_light_radius_axvline.kwargs["c"] = color
        self.einstein_radius_axvline.kwargs["c"] = color

        self.half_light_radius_axvline.no_label = True
        self.einstein_radius_axvline.no_label = True


class MatPlot2D(aplt.MatPlot2D):
    def __init__(
        self,
        units: aplt.Units = aplt.Units(),
        figure: aplt.Figure = aplt.Figure(),
        axis: aplt.Axis = aplt.Axis(),
        cmap: aplt.Cmap = aplt.Cmap(),
        colorbar: aplt.Colorbar = aplt.Colorbar(),
        colorbar_tickparams: aplt.ColorbarTickParams = aplt.ColorbarTickParams(),
        tickparams: aplt.TickParams = aplt.TickParams(),
        yticks: aplt.YTicks = aplt.YTicks(),
        xticks: aplt.XTicks = aplt.XTicks(),
        title: aplt.Title = aplt.Title(),
        ylabel: aplt.YLabel = aplt.YLabel(),
        xlabel: aplt.XLabel = aplt.XLabel(),
        text: aplt.Text = aplt.Text(),
        legend: aplt.Legend = aplt.Legend(),
        output: aplt.Output = aplt.Output(),
        array_overlay: aplt.ArrayOverlay = aplt.ArrayOverlay(),
        grid_scatter: aplt.GridScatter = aplt.GridScatter(),
        grid_plot: aplt.GridPlot = aplt.GridPlot(),
        vector_field_quiver: aplt.VectorFieldQuiver = aplt.VectorFieldQuiver(),
        patch_overlay: aplt.PatchOverlay = aplt.PatchOverlay(),
        voronoi_drawer: aplt.VoronoiDrawer = aplt.VoronoiDrawer(),
        origin_scatter: aplt.OriginScatter = aplt.OriginScatter(),
        mask_scatter: aplt.MaskScatter = aplt.MaskScatter(),
        border_scatter: aplt.BorderScatter = aplt.BorderScatter(),
        positions_scatter: aplt.PositionsScatter = aplt.PositionsScatter(),
        index_scatter: aplt.IndexScatter = aplt.IndexScatter(),
        pixelization_grid_scatter: aplt.PixelizationGridScatter = aplt.PixelizationGridScatter(),
        light_profile_centres_scatter: lw.LightProfileCentresScatter = lw.LightProfileCentresScatter(),
        mass_profile_centres_scatter: lw.MassProfileCentresScatter = lw.MassProfileCentresScatter(),
        multiple_images_scatter: lw.MultipleImagesScatter = lw.MultipleImagesScatter(),
        critical_curves_plot: lw.CriticalCurvesPlot = lw.CriticalCurvesPlot(),
        caustics_plot: lw.CausticsPlot = lw.CausticsPlot(),
    ):
        """
        Visualizes data structures (e.g an `Array2D`, `Grid2D`, `VectorField`, etc.) using Matplotlib.
        
        The `Plotter` is passed objects from the `mat_wrap` package which wrap matplotlib plot functions and 
        customize the appearance of the plots of the data structure. If the values of these matplotlib wrapper 
        objects are not manually specified, they assume the default values provided in 
        the `config.visualize.mat_*` `.ini` config files.
        
        The following data structures can be plotted using the following matplotlib functions:
        
        - `Array2D`:, using `plt.imshow`.
        - `Grid2D`: using `plt.scatter`.
        - `Line`: using `plt.plot`, `plt.semilogy`, `plt.loglog` or `plt.scatter`.
        - `VectorField`: using `plt.quiver`.
        - `RectangularMapper`: using `plt.imshow`.
        - `VoronoiMapper`: using `plt.fill`.
        
        Parameters
        ----------
        units : aplt.Units
          The units of the figure used to plot the data structure which sets the y and x ticks and labels.
        figure : aplt.Figure
          Opens the matplotlib figure before plotting via `plt.figure` and closes it once plotting is complete
          via `plt.close`.
        axis : mat_wrap.Axis
            Sets the extent of the figure axis via `plt.axis` and allows for a manual axis range.
        cmap : aplt.Cmap
          Customizes the colormap of the plot and its normalization via matplotlib `colors` objects such 
          as `colors.Normalize` and `colors.LogNorm`.
        colorbar : mat_wrap.Colorbar
            Plots the colorbar of the plot via `plt.colorbar` and customizes its tick labels and values using method
            like `cb.set_yticklabels`.
        colorbar_tickparams : mat_wrap.ColorbarTickParams
            Customizes the yticks of the colorbar plotted via `plt.colorbar`.
        tickparams : aplt.TickParams
          Customizes the appearances of the y and x ticks on the plot, (e.g. the fontsize), using `plt.tick_params`.
        yticks : aplt.YTicks
          Sets the yticks of the plot, including scaling them to new units depending on the `Units` object, via
          `plt.yticks`.
        xticks : aplt.XTicks
          Sets the xticks of the plot, including scaling them to new units depending on the `Units` object, via
          `plt.xticks`.
        title : aplt.Title
          Sets the figure title and customizes its appearance using `plt.title`.        
        ylabel : aplt.YLabel
          Sets the figure ylabel and customizes its appearance using `plt.ylabel`.
        xlabel : aplt.XLabel
          Sets the figure xlabel and customizes its appearance using `plt.xlabel`.
        legend : aplt.Legend
          Sets whether the plot inclues a legend and customizes its appearance and labels using `plt.legend`.
        output : aplt.Output
          Sets if the figure is displayed on the user's screen or output to `.png` using `plt.show` and `plt.savefig`
        array_overlay: aplt.ArrayOverlay
          Overlays an input `Array2D` over the figure using `plt.imshow`.
        grid_scatter : aplt.GridScatter
          Scatters a `Grid2D` of (y,x) coordinates over the figure using `plt.scatter`.
        grid_plot: aplt.LinePlot
          Plots lines of data (e.g. a y versus x plot via `plt.plot`, vertical lines via `plt.avxline`, etc.)
        vector_field_quiver: aplt.VectorFieldQuiver
          Plots a `VectorField` object using the matplotlib function `plt.quiver`.
        patch_overlay: aplt.PatchOverlay
          Overlays matplotlib `patches.Patch` objects over the figure, such as an `Ellipse`.
        voronoi_drawer: aplt.VoronoiDrawer
          Draws a colored Voronoi mesh of pixels using `plt.fill`.
        origin_scatter : aplt.OriginScatter
          Scatters the (y,x) origin of the data structure on the figure.
        mask_scatter : aplt.MaskScatter
          Scatters an input `Mask2d` over the plotted data structure's figure.
        border_scatter : aplt.BorderScatter
          Scatters the border of an input `Mask2d` over the plotted data structure's figure.
        positions_scatter : aplt.PositionsScatter
          Scatters specific (y,x) coordinates input as a `Grid2DIrregular` object over the figure.
        index_scatter : aplt.IndexScatter
          Scatters specific coordinates of an input `Grid2D` based on input values of the `Grid2D`'s 1D or 2D indexes.
        pixelization_grid_scatter : aplt.PixelizationGridScatter
          Scatters the `PixelizationGrid` of a `Pixelization` object.
        light_profile_centres_scatter : lensing_aplt.LightProfileCentresScatter
          Scatters the (y,x) centres of all `LightProfile`'s in the plotted object (e.g. a `Tracer`).
        mass_profile_centres_scatter : lensing_aplt.MassProfileCentresScatter
          Scatters the (y,x) centres of all `MassProfile`'s in the plotted object (e.g. a `Tracer`).
        light_profile_centres_scatter : lensing_aplt.MultipleImagesScatter
          Scatters the (y,x) coordinates of the multiple image locations of the lens mass model.
        critical_curves_plot : lensing_aplt.CriticalCurvesPlot
            Plots the critical curves of the lens mass model as colored lines.
        caustics_plot : lensing_aplt.CauticsPlot
            Plots the caustics of the lens mass model as colored lines.
        """

        self.light_profile_centres_scatter = light_profile_centres_scatter
        self.mass_profile_centres_scatter = mass_profile_centres_scatter
        self.multiple_images_scatter = multiple_images_scatter
        self.critical_curves_plot = critical_curves_plot
        self.caustics_plot = caustics_plot

        super(MatPlot2D, self).__init__(
            units=units,
            figure=figure,
            axis=axis,
            cmap=cmap,
            colorbar=colorbar,
            colorbar_tickparams=colorbar_tickparams,
            legend=legend,
            title=title,
            tickparams=tickparams,
            yticks=yticks,
            xticks=xticks,
            ylabel=ylabel,
            xlabel=xlabel,
            text=text,
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
            grid_plot=grid_plot,
            voronoi_drawer=voronoi_drawer,
        )
