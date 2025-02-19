from os import path

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.interferometer.fit_interferometer import FitInterferometer
from autogalaxy.interferometer.plot.fit_interferometer_plotters import (
    FitInterferometerPlotter,
)
from autogalaxy.analysis.plotter_interface import PlotterInterface

from autogalaxy.analysis.plotter_interface import plot_setting


class PlotterInterfaceInterferometer(PlotterInterface):
    def interferometer(self, dataset: aa.Interferometer):
        """
        Visualizes an `Interferometer` dataset object.

        Images are output to the `image` folder of the `image_path` in a subfolder called `interferometer`. When
        used with a non-linear search the `image_path` is the output folder of the non-linear search.

        Visualization includes individual images of attributes of the dataset (e.g. the visibilities, noise map,
        uv-wavelengths) and a subplot of all these attributes on the same figure.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under the
        [dataset] header.

        Parameters
        ----------
        dataset
            The interferometer dataset whose attributes are visualized.
        """

        def should_plot(name):
            return plot_setting(section=["dataset", "interferometer"], name=name)

        mat_plot_1d = self.mat_plot_1d_from()
        mat_plot_2d = self.mat_plot_2d_from()

        dataset_plotter = aplt.InterferometerPlotter(
            dataset=dataset,
            include_2d=self.include_2d,
            mat_plot_1d=mat_plot_1d,
            mat_plot_2d=mat_plot_2d,
        )

        if should_plot("subplot_dataset"):
            dataset_plotter.subplot_dataset()

    def fit_interferometer(
        self,
        fit: FitInterferometer,
    ):
        """
        Visualizes a `FitInterferometer` object, which fits an interferometer dataset.

        Images are output to the `image` folder of the `image_path` in a subfolder called `fit`. When
        used with a non-linear search the `image_path` points to the search's results folder and this function
        visualizes the maximum log likelihood `FitInterferometer` inferred by the search so far.

        Visualization includes individual images of attributes of the `FitInterferometer` (e.g. the model data,
        residual map) and a subplot of all `FitInterferometer`'s images on the same figure.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under the
        [fit] header.

        Parameters
        ----------
        fit
            The maximum log likelihood `FitInterferometer` of the non-linear search which is used to plot the fit.
        """

        def should_plot(name):
            return plot_setting(section=["fit", "fit_interferometer"], name=name)

        mat_plot_1d = self.mat_plot_1d_from()
        mat_plot_2d = self.mat_plot_2d_from()

        fit_plotter = FitInterferometerPlotter(
            fit=fit,
            include_2d=self.include_2d,
            mat_plot_1d=mat_plot_1d,
            mat_plot_2d=mat_plot_2d,
        )

        if should_plot("subplot_fit"):
            fit_plotter.subplot_fit()

        if should_plot("subplot_fit_real_space"):
            fit_plotter.subplot_fit_real_space()

