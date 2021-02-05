from autoarray.plot.mat_wrap import mat_plot
from autoarray.plot.mat_wrap.wrap import wrap_base, wrap_2d
from autogalaxy.plot.mat_wrap import lensing_wrap


class MatPlot1D(mat_plot.MatPlot1D):

    pass


class MatPlot2D(mat_plot.MatPlot2D):
    def __init__(
        self,
        units: wrap_base.Units = wrap_base.Units(),
        figure: wrap_base.Figure = wrap_base.Figure(),
        axis: wrap_base.Axis = wrap_base.Axis(),
        cmap: wrap_base.Cmap = wrap_base.Cmap(),
        colorbar: wrap_base.Colorbar = wrap_base.Colorbar(),
        colorbar_tickparams: wrap_base.ColorbarTickParams = wrap_base.ColorbarTickParams(),
        tickparams: wrap_base.TickParams = wrap_base.TickParams(),
        yticks: wrap_base.YTicks = wrap_base.YTicks(),
        xticks: wrap_base.XTicks = wrap_base.XTicks(),
        title: wrap_base.Title = wrap_base.Title(),
        ylabel: wrap_base.YLabel = wrap_base.YLabel(),
        xlabel: wrap_base.XLabel = wrap_base.XLabel(),
        legend: wrap_base.Legend = wrap_base.Legend(),
        output: wrap_base.Output = wrap_base.Output(),
        array_overlay: wrap_2d.ArrayOverlay = wrap_2d.ArrayOverlay(),
        grid_scatter: wrap_2d.GridScatter = wrap_2d.GridScatter(),
        grid_plot: wrap_2d.GridPlot = wrap_2d.GridPlot(),
        vector_field_quiver: wrap_2d.VectorFieldQuiver = wrap_2d.VectorFieldQuiver(),
        patch_overlay: wrap_2d.PatchOverlay = wrap_2d.PatchOverlay(),
        voronoi_drawer: wrap_2d.VoronoiDrawer = wrap_2d.VoronoiDrawer(),
        origin_scatter: wrap_2d.OriginScatter = wrap_2d.OriginScatter(),
        mask_scatter: wrap_2d.MaskScatter = wrap_2d.MaskScatter(),
        border_scatter: wrap_2d.BorderScatter = wrap_2d.BorderScatter(),
        positions_scatter: wrap_2d.PositionsScatter = wrap_2d.PositionsScatter(),
        index_scatter: wrap_2d.IndexScatter = wrap_2d.IndexScatter(),
        pixelization_grid_scatter: wrap_2d.PixelizationGridScatter = wrap_2d.PixelizationGridScatter(),
        light_profile_centres_scatter: lensing_wrap.LightProfileCentresScatter = lensing_wrap.LightProfileCentresScatter(),
        mass_profile_centres_scatter: lensing_wrap.MassProfileCentresScatter = lensing_wrap.MassProfileCentresScatter(),
        multiple_images_scatter: lensing_wrap.MultipleImagesScatter = lensing_wrap.MultipleImagesScatter(),
        critical_curves_plot: lensing_wrap.CriticalCurvesPlot = lensing_wrap.CriticalCurvesPlot(),
        caustics_plot: lensing_wrap.CausticsPlot = lensing_wrap.CausticsPlot(),
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
        units : wrap_base.Units
          The units of the figure used to plot the data structure which sets the y and x ticks and labels.
        figure : wrap_base.Figure
          Opens the matplotlib figure before plotting via `plt.figure` and closes it once plotting is complete
          via `plt.close`.
        axis : mat_wrap.Axis
            Sets the extent of the figure axis via `plt.axis` and allows for a manual axis range.
        cmap : wrap_base.Cmap
          Customizes the colormap of the plot and its normalization via matplotlib `colors` objects such 
          as `colors.Normalize` and `colors.LogNorm`.
        colorbar : mat_wrap.Colorbar
            Plots the colorbar of the plot via `plt.colorbar` and customizes its tick labels and values using method
            like `cb.set_yticklabels`.
        colorbar_tickparams : mat_wrap.ColorbarTickParams
            Customizes the yticks of the colorbar plotted via `plt.colorbar`.
        tickparams : wrap_base.TickParams
          Customizes the appearances of the y and x ticks on the plot, (e.g. the fontsize), using `plt.tick_params`.
        yticks : wrap_base.YTicks
          Sets the yticks of the plot, including scaling them to new units depending on the `Units` object, via
          `plt.yticks`.
        xticks : wrap_base.XTicks
          Sets the xticks of the plot, including scaling them to new units depending on the `Units` object, via
          `plt.xticks`.
        title : wrap_base.Title
          Sets the figure title and customizes its appearance using `plt.title`.        
        ylabel : wrap_base.YLabel
          Sets the figure ylabel and customizes its appearance using `plt.ylabel`.
        xlabel : wrap_base.XLabel
          Sets the figure xlabel and customizes its appearance using `plt.xlabel`.
        legend : wrap_base.Legend
          Sets whether the plot inclues a legend and customizes its appearance and labels using `plt.legend`.
        output : wrap_base.Output
          Sets if the figure is displayed on the user's screen or output to `.png` using `plt.show` and `plt.savefig`
        array_overlay: wrap_2d.ArrayOverlay
          Overlays an input `Array2D` over the figure using `plt.imshow`.
        grid_scatter : wrap_2d.GridScatter
          Scatters a `Grid2D` of (y,x) coordinates over the figure using `plt.scatter`.
        grid_plot: wrap_2d.LinePlot
          Plots lines of data (e.g. a y versus x plot via `plt.plot`, vertical lines via `plt.avxline`, etc.)
        vector_field_quiver: wrap_2d.VectorFieldQuiver
          Plots a `VectorField` object using the matplotlib function `plt.quiver`.
        patch_overlay: wrap_2d.PatchOverlay
          Overlays matplotlib `patches.Patch` objects over the figure, such as an `Ellipse`.
        voronoi_drawer: wrap_2d.VoronoiDrawer
          Draws a colored Voronoi mesh of pixels using `plt.fill`.
        origin_scatter : wrap_2d.OriginScatter
          Scatters the (y,x) origin of the data structure on the figure.
        mask_scatter : wrap_2d.MaskScatter
          Scatters an input `Mask2d` over the plotted data structure's figure.
        border_scatter : wrap_2d.BorderScatter
          Scatters the border of an input `Mask2d` over the plotted data structure's figure.
        positions_scatter : wrap_2d.PositionsScatter
          Scatters specific (y,x) coordinates input as a `Grid2DIrregular` object over the figure.
        index_scatter : wrap_2d.IndexScatter
          Scatters specific coordinates of an input `Grid2D` based on input values of the `Grid2D`'s 1D or 2D indexes.
        pixelization_grid_scatter : wrap_2d.PixelizationGridScatter
          Scatters the `PixelizationGrid` of a `Pixelization` object.
        light_profile_centres_scatter : lensing_wrap_2d.LightProfileCentresScatter
          Scatters the (y,x) centres of all `LightProfile`'s in the plotted object (e.g. a `Tracer`).
        mass_profile_centres_scatter : lensing_wrap_2d.MassProfileCentresScatter
          Scatters the (y,x) centres of all `MassProfile`'s in the plotted object (e.g. a `Tracer`).
        light_profile_centres_scatter : lensing_wrap_2d.MultipleImagesScatter
          Scatters the (y,x) coordinates of the multiple image locations of the lens mass model.
        critical_curves_plot : lensing_wrap_2d.CriticalCurvesPlot
            Plots the critical curves of the lens mass model as colored lines.
        caustics_plot : lensing_wrap_2d.CauticsPlot
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
