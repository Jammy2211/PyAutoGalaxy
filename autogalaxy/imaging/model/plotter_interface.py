from os import path
from typing import ClassVar, List

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.imaging.fit_imaging import FitImaging
from autogalaxy.imaging.plot.fit_imaging_plotters import FitImagingPlotter
from autogalaxy.analysis.plotter_interface import PlotterInterface

from autogalaxy.analysis.plotter_interface import plot_setting


def fits_to_fits(
    should_plot: bool,
    fit: FitImaging,
    mat_plot_2d: aplt.MatPlot2D,
    fit_plotter_cls: ClassVar,
):
    """
    Output attributes of a `FitImaging` to .fits format.

    This function is separated on its own so that it can be called by `PyAutoLens` and therefore avoid repeating
    large amounts of code for visualization.

    Parameters
    ----------
    should_plot
        The function which inspects the configuration files to determine if a .fits file should be output.
    fit
        The fit to output to a .fits file.
    mat_plot_2d
        The 2D matplotlib plot used to create the .fits files.
    fit_plotter_cls
        The plotter class used to create the .fits files.
    """

    if should_plot("fits_fit"):
        multi_plotter = aplt.MultiFigurePlotter(
            plotter_list=[fit_plotter_cls(fit=fit, mat_plot_2d=mat_plot_2d)] * 4,
        )

        multi_plotter.output_to_fits(
            func_name_list=["figures_2d"] * len(multi_plotter.plotter_list),
            figure_name_list=[
                "model_image",
                "residual_map",
                "normalized_residual_map",
                "chi_squared_map",
            ],
            #                tag_list=[name for name, galaxy in galaxies.items()],
            tag_list=[
                "model_image",
                "residual_map",
                "normalized_residual_map",
                "chi_squared_map",
            ],
            filename="fit",
            remove_fits_first=True,
        )

    if should_plot("fits_model_galaxy_images"):
        multi_plotter = aplt.MultiFigurePlotter(
            plotter_list=[
                aplt.Array2DPlotter(array=image, mat_plot_2d=mat_plot_2d)
                for (galaxy, image) in fit.galaxy_model_image_dict.items()
            ],
        )

        multi_plotter.output_to_fits(
            func_name_list=["figure_2d"] * len(multi_plotter.plotter_list),
            figure_name_list=[None] * len(multi_plotter.plotter_list),
            #                tag_list=[name for name, galaxy in galaxies.items()],
            tag_list=[f"galaxy_{i}" for i in range(len(multi_plotter.plotter_list))],
            filename="model_galaxy_images",
            remove_fits_first=True,
        )


class PlotterInterfaceImaging(PlotterInterface):
    def imaging(self, dataset: aa.Imaging):
        """
        Output visualization of an `Imaging` dataset, typically before a model-fit is performed.

        Images are output to the `image` folder of the `image_path`. When used with a non-linear search the `image_path`
        is the output folder of the non-linear search.

        Visualization includes a subplot of the individual images of attributes of the dataset (e.g. the image,
        noise map, PSF).

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under
        the `dataset` and `imaging` headers.

        Parameters
        ----------
        dataset
            The imaging dataset which is visualized.
        """

        def should_plot(name):
            return plot_setting(section=["dataset", "imaging"], name=name)

        mat_plot_2d = self.mat_plot_2d_from()

        dataset_plotter = aplt.ImagingPlotter(
            dataset=dataset, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        if should_plot("subplot_dataset"):
            dataset_plotter.subplot_dataset()

    def fit_imaging(
        self,
        fit: FitImaging,
    ):
        """
        Visualizes a `FitImaging` object, which fits an imaging dataset.

        Images are output to the `image` folder of the `image_path`. When used with a non-linear search the `image_path`
        points to the search's results folder and this function visualizes the maximum log likelihood `FitImaging`
        inferred by the search so far.

        Visualization includes a subplot of individual images of attributes of the `FitImaging` (e.g. the model data,
        residual map) and .fits files containing its attributes grouped together.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under
        the `fit` and `fit_imaging` headers.

        Parameters
        ----------
        fit
            The maximum log likelihood `FitImaging` of the non-linear search which is used to plot the fit.
        """

        def should_plot(name):
            return plot_setting(section=["fit", "fit_imaging"], name=name)

        mat_plot_2d = self.mat_plot_2d_from()

        fit_plotter = FitImagingPlotter(
            fit=fit, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        if should_plot("subplot_fit"):
            fit_plotter.subplot_fit()

        if should_plot("subplot_of_galaxies"):
            fit_plotter.subplot_of_galaxies()

        fits_to_fits(
            should_plot=should_plot,
            fit=fit,
            mat_plot_2d=mat_plot_2d,
            fit_plotter_cls=FitImagingPlotter,
        )

    def imaging_combined(self, dataset_list: List[aa.Imaging]):
        """
        Output visualization of all `Imaging` datasets in a summed combined analysis, typically before a model-fit
        is performed.

        Images are output to the `image` folder of the `image_path`. When used with a non-linear search the `image_path`
        is the output folder of the non-linear search.

        Visualization includes a single subplot of individual images of attributes of each dataset (e.g. the image,
        noise map,  PSF), such that the full suite of multiple datasets can be viewed on the same figure.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under
        the `dataset` header.

        Parameters
        ----------
        dataset
            The list of imaging datasets which are visualized.
        """

        def should_plot(name):
            return plot_setting(section=["dataset", "imaging"], name=name)

        mat_plot_2d = self.mat_plot_2d_from()

        dataset_plotter_list = [
            aplt.ImagingPlotter(
                dataset=dataset, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
            )
            for dataset in dataset_list
        ]

        subplot_shape = (len(dataset_list), 4)

        multi_plotter = aplt.MultiFigurePlotter(
            plotter_list=dataset_plotter_list, subplot_shape=subplot_shape
        )

        if should_plot("subplot_dataset"):
            multi_plotter.subplot_of_figures_multi(
                func_name_list=["figures_2d"] * 4,
                figure_name_list=["data", "noise_map", "signal_to_noise_map", "psf"],
                filename_suffix="dataset_combined",
            )

            for plotter in multi_plotter.plotter_list:
                plotter.mat_plot_2d.use_log10 = True

            multi_plotter.subplot_of_figures_multi(
                func_name_list=["figures_2d"] * 4,
                figure_name_list=["data", "noise_map", "signal_to_noise_map", "psf"],
                filename_suffix="dataset_combined_log10",
            )

    def fit_imaging_combined(self, fit_list: List[FitImaging]):
        """
        Output visualization of all `FitImaging` objects in a summed combined analysis, typically during or after a
        model-fit is performed.

        Images are output to the `image` folder of the `image_path`. When used with a non-linear search the `image_path`
        is the output folder of the non-linear search.

        Visualization includes a single subplot of individual images of attributes of each fit (e.g. data,
        normalized residual-map), such that the full suite of multiple datasets can be viewed on the same figure.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under
        the `fit` header.

        Parameters
        ----------
        fit
            The list of imaging fits which are visualized.
        """

        def should_plot(name):
            return plot_setting(section=["fit", "fit_imaging"], name=name)

        mat_plot_2d = self.mat_plot_2d_from()

        fit_plotter_list = [
            FitImagingPlotter(
                fit=fit, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
            )
            for fit in fit_list
        ]

        subplot_shape = (len(fit_list), 5)

        multi_plotter = aplt.MultiFigurePlotter(
            plotter_list=fit_plotter_list, subplot_shape=subplot_shape
        )

        if should_plot("subplot_fit"):

            def make_subplot_fit(filename_suffix):
                multi_plotter.subplot_of_figures_multi(
                    func_name_list=["figures_2d"] * 4,
                    figure_name_list=[
                        "data",
                        "signal_to_noise_map",
                        "model_image",
                        "normalized_residual_map",
                    ],
                    filename_suffix=filename_suffix,
                    number_subplots=len(fit_list) * 5,
                    close_subplot=False,
                )

                for plotter in multi_plotter.plotter_list:
                    plotter.mat_plot_2d.cmap.kwargs["vmin"] = -1.0
                    plotter.mat_plot_2d.cmap.kwargs["vmax"] = 1.0

                multi_plotter.subplot_of_figures_multi(
                    func_name_list=["figures_2d"],
                    figure_name_list=[
                        "normalized_residual_map",
                    ],
                    filename_suffix=filename_suffix,
                    number_subplots=len(fit_list) * 5,
                    subplot_index_offset=4,
                    open_subplot=False,
                )

                for plotter in multi_plotter.plotter_list:
                    plotter.mat_plot_2d.cmap.kwargs["vmin"] = None
                    plotter.mat_plot_2d.cmap.kwargs["vmax"] = None

            make_subplot_fit(filename_suffix="fit_combined")

            for plotter in multi_plotter.plotter_list:
                plotter.mat_plot_2d.use_log10 = True

            make_subplot_fit(filename_suffix="fit_combined_log10")
