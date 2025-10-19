from pathlib import Path

from autoconf.fitsable import hdu_list_for_output_from

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.interferometer.fit_interferometer import FitInterferometer
from autogalaxy.interferometer.plot.fit_interferometer_plotters import (
    FitInterferometerPlotter,
)
from autogalaxy.analysis.plotter_interface import PlotterInterface

from autogalaxy.analysis.plotter_interface import plot_setting


def fits_to_fits(
    should_plot: bool,
    image_path: Path,
    fit: FitInterferometer,
):
    """
    Output attributes of a `FitInterferometer` to .fits format.

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

    if should_plot("fits_model_galaxy_images"):

        image_list = [
            image.native_for_fits for image in fit.galaxy_model_image_dict.values()
        ]

        hdu_list = hdu_list_for_output_from(
            values_list=[image_list[0].mask.astype("float")] + image_list,
            ext_name_list=["mask"]
            + [f"galaxy_{i}" for i in range(len(fit.galaxy_model_image_dict.values()))],
            header_dict=fit.dataset.real_space_mask.header_dict,
        )

        hdu_list.writeto(image_path / "model_galaxy_images.fits", overwrite=True)

    if should_plot("fits_dirty_images"):

        image_list = [
            fit.dirty_image.native_for_fits,
            fit.dirty_noise_map.native_for_fits,
            fit.dirty_model_image.native_for_fits,
            fit.dirty_residual_map.native_for_fits,
            fit.dirty_normalized_residual_map.native_for_fits,
            fit.dirty_chi_squared_map.native_for_fits,
        ]

        hdu_list = hdu_list_for_output_from(
            values_list=[image_list[0].mask.astype("float")] + image_list,
            ext_name_list=["mask"]
            + [
                "dirty_image",
                "dirty_noise_map",
                "dirty_model_image",
                "dirty_residual_map",
                "dirty_normalized_residual_map",
                "dirty_chi_squared_map",
            ],
            header_dict=fit.dataset.real_space_mask.header_dict,
        )

        hdu_list.writeto(image_path / "fit_dirty_images.fits", overwrite=True)


class PlotterInterfaceInterferometer(PlotterInterface):
    def interferometer(self, dataset: aa.Interferometer):
        """
        Visualizes an `Interferometer` dataset object.

        Images are output to the `image` folder of the `image_path`. When used with a non-linear search the `image_path`
        is the output folder of the non-linear search.

        Visualization includes a subplot of individual images of attributes of the dataset (e.g. the visibilities,
        noise map, uv-wavelengths).

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under
        the `dataset` and `interferometer` headers.

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
            mat_plot_1d=mat_plot_1d,
            mat_plot_2d=mat_plot_2d,
        )

        if should_plot("subplot_dataset"):
            dataset_plotter.subplot_dataset()

        if should_plot("fits_dataset"):

            hdu_list = hdu_list_for_output_from(
                values_list=[
                    dataset.real_space_mask.astype("float"),
                    dataset.data.in_array,
                    dataset.noise_map.in_array,
                    dataset.uv_wavelengths,
                ],
                ext_name_list=["mask", "data", "noise_map", "uv_wavelengths"],
                header_dict=dataset.real_space_mask.header_dict,
            )

            hdu_list.writeto(self.image_path / "dataset.fits", overwrite=True)

    def fit_interferometer(
        self,
        fit: FitInterferometer,
        quick_update: bool = False,
    ):
        """
        Visualizes a `FitInterferometer` object, which fits an interferometer dataset.

        Images are output to the `image` folder of the `image_path`. When used with a non-linear search the `image_path`
        points to the search's results folder and this function visualizes the maximum log likelihood `FitInterferometer`
        inferred by the search so far.

        Visualization includes a subplot of individual images of attributes of the `FitInterferometer` (e.g. the model
        data, residual map) and .fits files containing its attributes grouped together.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under
        the `fit` and `fit_interferometer` headers.

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
            mat_plot_1d=mat_plot_1d,
            mat_plot_2d=mat_plot_2d,
        )

        if should_plot("subplot_fit") or quick_update:
            fit_plotter.subplot_fit()

        if should_plot("subplot_fit_dirty_images") or quick_update:
            fit_plotter.subplot_fit_dirty_images()

        if quick_update:
            return

        if should_plot("subplot_fit_real_space"):
            fit_plotter.subplot_fit_real_space()

        fits_to_fits(
            should_plot=should_plot,
            image_path=self.image_path,
            fit=fit,
        )
