from pathlib import Path

from autoconf.fitsable import hdu_list_for_output_from

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.interferometer.fit_interferometer import FitInterferometer
from autogalaxy.interferometer.plot import fit_interferometer_plots
from autogalaxy.analysis.plotter_interface import PlotterInterface, plot_setting


def fits_to_fits(
    should_plot: bool,
    image_path: Path,
    fit: FitInterferometer,
):
    if should_plot("fits_galaxy_images"):

        image_list = [image.native_for_fits for image in fit.galaxy_image_dict.values()]

        hdu_list = hdu_list_for_output_from(
            values_list=[image_list[0].mask.astype("float")] + image_list,
            ext_name_list=["mask"]
            + [f"galaxy_{i}" for i in range(len(fit.galaxy_image_dict.values()))],
            header_dict=fit.dataset.real_space_mask.header_dict,
        )

        hdu_list.writeto(image_path / "galaxy_images.fits", overwrite=True)

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
        def should_plot(name):
            return plot_setting(section=["dataset", "interferometer"], name=name)

        output = self.output_from()

        dataset_plotter = aplt.InterferometerPlotter(
            dataset=dataset,
            output=output,
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
        def should_plot(name):
            return plot_setting(section=["fit", "fit_interferometer"], name=name)

        if should_plot("subplot_fit") or quick_update:
            fit_interferometer_plots.subplot_fit(
                fit=fit,
                output_path=self.image_path,
                output_format=self.fmt,
            )

        if should_plot("subplot_fit_dirty_images") or quick_update:
            fit_interferometer_plots.subplot_fit_dirty_images(
                fit=fit,
                output_path=self.image_path,
                output_format=self.fmt,
            )

        if quick_update:
            return

        if should_plot("subplot_fit_real_space"):
            fit_interferometer_plots.subplot_fit_real_space(
                fit=fit,
                output_path=self.image_path,
                output_format=self.fmt,
            )

        fits_to_fits(
            should_plot=should_plot,
            image_path=self.image_path,
            fit=fit,
        )
