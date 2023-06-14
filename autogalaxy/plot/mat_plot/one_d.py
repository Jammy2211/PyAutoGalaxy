from typing import Optional

import autoarray.plot as aplt
from autogalaxy.plot import wrap as w


class MatPlot1D(aplt.MatPlot1D):
    def __init__(
        self,
        units: Optional[aplt.Units] = None,
        figure: Optional[aplt.Figure] = None,
        axis: Optional[aplt.Axis] = None,
        cmap: Optional[aplt.Cmap] = None,
        colorbar: Optional[aplt.Colorbar] = None,
        colorbar_tickparams: Optional[aplt.ColorbarTickParams] = None,
        tickparams: Optional[aplt.TickParams] = None,
        yticks: Optional[aplt.YTicks] = None,
        xticks: Optional[aplt.XTicks] = None,
        title: Optional[aplt.Title] = None,
        ylabel: Optional[aplt.YLabel] = None,
        xlabel: Optional[aplt.XLabel] = None,
        text: Optional[aplt.Text] = None,
        legend: Optional[aplt.Legend] = None,
        output: Optional[aplt.Output] = None,
        yx_plot: Optional[aplt.YXPlot] = None,
        fill_between: Optional[aplt.FillBetween] = None,
        half_light_radius_axvline: Optional[w.HalfLightRadiusAXVLine] = None,
        einstein_radius_axvline: Optional[w.EinsteinRadiusAXVLine] = None,
        model_fluxes_yx_scatter: Optional[w.ModelFluxesYXScatter] = None,
    ):
        """
        Visualizes 1D data structures as a y versus x plot using Matplotlib.

        The `Plotter` is passed objects from the `wrap_base` package which wrap matplotlib plot functions and customize
        the appearance of the plots of the data structure. If the values of these matplotlib wrapper objects are not
        manually specified, they assume the default values provided in the `config.visualize.mat_*` `.ini` config files.

        Parameters
        ----------
        units
            The units of the figure used to plot the data structure which sets the y and x ticks and labels.
        figure
            Opens the matplotlib figure before plotting via `plt.figure` and closes it once plotting is complete
            via `plt.close`.
        axis
            Sets the extent of the figure axis via `plt.axis` and allows for a manual axis range.
        cmap
            Customizes the colormap of the plot and its normalization via matplotlib `colors` objects such
            as `colors.Normalize` and `colors.LogNorm`.
        colorbar
            Plots the colorbar of the plot via `plt.colorbar` and customizes its tick labels and values using method
            like `cb.set_yticklabels`.
        colorbar_tickparams
            Customizes the yticks of the colorbar plotted via `plt.colorbar`.
        tickparams
            Customizes the appearances of the y and x ticks on the plot, (e.g. the fontsize), using `plt.tick_params`.
        yticks
            Sets the yticks of the plot, including scaling them to new units depending on the `Units` object, via
            `plt.yticks`.
        xticks
            Sets the xticks of the plot, including scaling them to new units depending on the `Units` object, via
            `plt.xticks`.
        title
            Sets the figure title and customizes its appearance using `plt.title`.
        ylabel
            Sets the figure ylabel and customizes its appearance using `plt.ylabel`.
        xlabel
            Sets the figure xlabel and customizes its appearance using `plt.xlabel`.
        legend
            Sets whether the plot inclues a legend and customizes its appearance and labels using `plt.legend`.
        output
            Sets if the figure is displayed on the user's screen or output to `.png` using `plt.show` and `plt.savefig`
        yx_plot
            Sets how the y versus x plot appears, for example if it each axis is linear or log, using `plt.plot`.
        half_light_radius_axvline
            Sets how a vertical line representing the half light radius of a `LightProfile` is plotted on the figure
            using the `plt.axvline` method.
        half_light_radius_axvline
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

        self.half_light_radius_axvline = (
            half_light_radius_axvline or w.HalfLightRadiusAXVLine(is_default=True)
        )
        self.einstein_radius_axvline = (
            einstein_radius_axvline or w.EinsteinRadiusAXVLine(is_default=True)
        )
        self.model_fluxes_yx_scatter = (
            model_fluxes_yx_scatter or w.ModelFluxesYXScatter(is_default=True)
        )

    def set_for_multi_plot(
        self, is_for_multi_plot: bool, color: str, xticks=None, yticks=None
    ):
        """
        Sets the `is_for_subplot` attribute for every `MatWrap` object in this `MatPlot` object by updating
        the `is_for_subplot`. By changing this tag:

            - The subplot: section of the config file of every `MatWrap` object is used instead of figure:.
            - Calls which output or close the matplotlib figure are over-ridden so that the subplot is not removed.

        Parameters
        ----------
        is_for_subplot
            The entry the `is_for_subplot` attribute of every `MatWrap` object is set too.
        """

        super().set_for_multi_plot(
            is_for_multi_plot=is_for_multi_plot,
            color=color,
            xticks=xticks,
            yticks=yticks,
        )

        self.half_light_radius_axvline.kwargs["c"] = color
        self.einstein_radius_axvline.kwargs["c"] = color

        self.half_light_radius_axvline.no_label = True
        self.einstein_radius_axvline.no_label = True
