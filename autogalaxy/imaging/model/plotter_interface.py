from pathlib import Path
from typing import List

from autoconf.fitsable import hdu_list_for_output_from

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.imaging.fit_imaging import FitImaging
from autogalaxy.imaging.plot.fit_imaging_plotters import FitImagingPlotter
from autogalaxy.analysis.plotter_interface import PlotterInterface

from autogalaxy.analysis.plotter_interface import plot_setting


def fits_to_fits(
    should_plot: bool,
    image_path: Path,
    fit: FitImaging,
):
    """
    Output attributes of a `FitImaging` to .fits format.

    This function is separated on its own so that it can be called by `PyAutoLens` and therefore avoid repeating
    large amounts of code for visualization.

    Parameters
    ----------
    should_plot
        The function which inspects the configuration files to determine if a .fits file should be output.
    image_path
        The path the .fits files are output and the name of the .fits files.
    fit
        The fit to output to a .fits file.
    """

    if should_plot("fits_fit"):

        image_list = [
            fit.model_data.native_for_fits,
            fit.residual_map.native_for_fits,
            fit.normalized_residual_map.native_for_fits,
            fit.chi_squared_map.native_for_fits,
        ]

        hdu_list = hdu_list_for_output_from(
            values_list=[
                image_list[0].mask.astype("float"),
            ]
            + image_list,
            ext_name_list=[
                "mask",
                "model_data",
                "residual_map",
                "normalized_residual_map",
                "chi_squared_map",
            ],
            header_dict=fit.mask.header_dict,
        )

        hdu_list.writeto(image_path / "fit.fits", overwrite=True)

    if should_plot("fits_model_galaxy_images"):
        number_plots = len(fit.galaxy_model_image_dict.keys()) + 1

        image_list = [
            image.native_for_fits for image in fit.galaxy_model_image_dict.values()
        ]

        hdu_list = hdu_list_for_output_from(
            values_list=[image_list[0].mask.astype("float")] + image_list,
            ext_name_list=[
                "mask",
            ]
            + [f"galaxy_{i}" for i in range(number_plots)],
            header_dict=fit.mask.header_dict,
        )

        hdu_list.writeto(image_path / "model_galaxy_images.fits", overwrite=True)


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
            dataset=dataset,
            mat_plot_2d=mat_plot_2d,
        )

        if should_plot("subplot_dataset"):
            dataset_plotter.subplot_dataset()

        if should_plot("fits_dataset"):
            image_list = [
                dataset.data.native_for_fits,
                dataset.noise_map.native_for_fits,
                dataset.psf.native_for_fits,
                dataset.grids.lp.over_sample_size.native_for_fits.astype("float"),
                dataset.grids.pixelization.over_sample_size.native_for_fits.astype(
                    "float"
                ),
            ]

            hdu_list = hdu_list_for_output_from(
                values_list=[image_list[0].mask.astype("float")] + image_list,
                ext_name_list=[
                    "mask",
                    "data",
                    "noise_map",
                    "psf",
                    "over_sample_size_lp",
                    "over_sample_size_pixelization",
                ],
                header_dict=dataset.mask.header_dict,
            )

            hdu_list.writeto(self.image_path / "dataset.fits", overwrite=True)

    def fit_imaging(self, fit: FitImaging, quick_update: bool = False):
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
            fit=fit,
            mat_plot_2d=mat_plot_2d,
        )

        if should_plot("subplot_fit") or quick_update:
            fit_plotter.subplot_fit()

        if quick_update:
            return

        if should_plot("subplot_of_galaxies"):
            fit_plotter.subplot_of_galaxies()

        fits_to_fits(
            should_plot=should_plot,
            image_path=self.image_path,
            fit=fit,
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
                dataset=dataset,
                mat_plot_2d=mat_plot_2d,
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
                fit=fit,
                mat_plot_2d=mat_plot_2d,
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
