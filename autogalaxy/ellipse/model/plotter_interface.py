from typing import List

from autoconf.fitsable import hdu_list_for_output_from

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.ellipse.fit_ellipse import FitEllipse
from autogalaxy.ellipse.plot.fit_ellipse_plotters import FitEllipsePlotter
from autogalaxy.analysis.plotter_interface import PlotterInterface

from autogalaxy.analysis.plotter_interface import plot_setting


class PlotterInterfaceEllipse(PlotterInterface):
    def imaging(self, dataset: aa.Imaging):
        """
        Visualizes an `Imaging` dataset object.

        Images are output to the `image` folder of the `image_path`. When used with a non-linear search the `image_path`
        is the output folder of the non-linear search.

        Visualization includes individual images of attributes of the dataset (e.g. the image, noise map, PSF) and a
        subplot of all these attributes on the same figure.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under the
        [dataset] header.

        Parameters
        ----------
        dataset
            The imaging dataset whose attributes are visualized.
        """

        def should_plot(name):
            return plot_setting(section=["dataset", "imaging"], name=name)

        mat_plot_2d = self.mat_plot_2d_from()

        dataset_plotter = aplt.ImagingPlotter(
            dataset=dataset,
            mat_plot_2d=mat_plot_2d,
        )

        if should_plot("subplot_dataset"):
            dataset_plotter.subplot_dataset()

        image_list = [
            dataset.data.native,
            dataset.noise_map.native,
        ]

        hdu_list = hdu_list_for_output_from(
            values_list=[image_list[0].mask.astype("float")] + image_list,
            ext_name_list=[
                "mask",
                "data",
                "noise_map",
            ],
            header_dict=dataset.mask.header_dict,
        )

        hdu_list.writeto(self.image_path / "dataset.fits", overwrite=True)

    def fit_ellipse(
        self,
        fit_list: List[FitEllipse],
    ):
        """
        Visualizes a `FitEllipse` object, which fits an imaging dataset.

        Images are output to the `image` folder of the `image_path`. When used with a non-linear search the `image_path`
        points to the search's results folder and this function visualizes the maximum log likelihood `FitEllipse`
        inferred by the search so far.

        Visualization includes a subplot of individual images of attributes of the `FitEllipse` (e.g. the model data,
        residual map).

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under
        the `fit` and `fit_ellipse` headers.

        Parameters
        ----------
        fit
            The maximum log likelihood `FitEllipse` of the non-linear search which is used to plot the fit.
        """

        def should_plot(name):
            return plot_setting(section=["fit", "fit_ellipse"], name=name)

        mat_plot_2d = self.mat_plot_2d_from()

        fit_plotter = FitEllipsePlotter(
            fit_list=fit_list,
            mat_plot_2d=mat_plot_2d,
        )

        fit_plotter.figures_2d(
            data=should_plot("data"),
            ellipse_residuals=should_plot("ellipse_residuals"),
        )

        if should_plot("data_no_ellipse"):
            fit_plotter.figures_2d(
                data=True,
                disable_data_contours=True,
            )

        if should_plot("subplot_fit_ellipse"):

            fit_plotter.subplot_fit_ellipse()

        fit_plotter.mat_plot_2d.use_log10 = True

        fit_plotter.figures_2d(data=should_plot("data"))

        if should_plot("data_no_ellipse"):
            fit_plotter.figures_2d(
                data=True,
                disable_data_contours=True,
            )
