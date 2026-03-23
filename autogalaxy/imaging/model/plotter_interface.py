import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

from autoconf.fitsable import hdu_list_for_output_from

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.imaging.fit_imaging import FitImaging
from autogalaxy.imaging.plot import fit_imaging_plots
from autogalaxy.analysis.plotter_interface import PlotterInterface, plot_setting
from autogalaxy.plot.plot_utils import _save_subplot, plot_array


def fits_to_fits(should_plot, image_path: Path, fit: FitImaging):
    if should_plot("fits_fit"):
        image_list = [
            fit.model_data.native_for_fits,
            fit.residual_map.native_for_fits,
            fit.normalized_residual_map.native_for_fits,
            fit.chi_squared_map.native_for_fits,
        ]

        hdu_list = hdu_list_for_output_from(
            values_list=[image_list[0].mask.astype("float")] + image_list,
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

    if should_plot("fits_galaxy_images"):
        number_plots = len(fit.galaxy_image_dict.keys()) + 1
        image_list = [image.native_for_fits for image in fit.galaxy_image_dict.values()]
        hdu_list = hdu_list_for_output_from(
            values_list=[image_list[0].mask.astype("float")] + image_list,
            ext_name_list=["mask"] + [f"galaxy_{i}" for i in range(number_plots)],
            header_dict=fit.mask.header_dict,
        )
        hdu_list.writeto(image_path / "galaxy_images.fits", overwrite=True)

    if should_plot("fits_model_galaxy_images"):
        number_plots = len(fit.galaxy_model_image_dict.keys()) + 1
        image_list = [image.native_for_fits for image in fit.galaxy_model_image_dict.values()]
        hdu_list = hdu_list_for_output_from(
            values_list=[image_list[0].mask.astype("float")] + image_list,
            ext_name_list=["mask"] + [f"galaxy_{i}" for i in range(number_plots)],
            header_dict=fit.mask.header_dict,
        )
        hdu_list.writeto(image_path / "model_galaxy_images.fits", overwrite=True)


class PlotterInterfaceImaging(PlotterInterface):
    def imaging(self, dataset: aa.Imaging):
        def should_plot(name):
            return plot_setting(section=["dataset", "imaging"], name=name)

        if should_plot("subplot_dataset"):
            panels = [
                (dataset.data, "Data"),
                (dataset.noise_map, "Noise Map"),
                (dataset.signal_to_noise_map, "Signal-To-Noise Map"),
            ]
            try:
                panels.append((dataset.psf.kernel, "PSF"))
            except Exception:
                pass
            n = len(panels)
            fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
            axes_flat = list(axes.flatten()) if n > 1 else [axes]
            for i, (array, title) in enumerate(panels):
                plot_array(array, title, ax=axes_flat[i])
            plt.tight_layout()
            _save_subplot(fig, self.image_path, "subplot_dataset", self.fmt)

        if should_plot("fits_dataset"):
            image_list = [
                dataset.data.native_for_fits,
                dataset.noise_map.native_for_fits,
                dataset.psf.kernel.native_for_fits,
                dataset.grids.lp.over_sample_size.native_for_fits.astype("float"),
                dataset.grids.pixelization.over_sample_size.native_for_fits.astype("float"),
            ]

            hdu_list = hdu_list_for_output_from(
                values_list=[image_list[0].mask.astype("float")] + image_list,
                ext_name_list=["mask", "data", "noise_map", "psf",
                               "over_sample_size_lp", "over_sample_size_pixelization"],
                header_dict=dataset.mask.header_dict,
            )
            hdu_list.writeto(self.image_path / "dataset.fits", overwrite=True)

    def fit_imaging(self, fit: FitImaging, quick_update: bool = False):
        def should_plot(name):
            return plot_setting(section=["fit", "fit_imaging"], name=name)

        if should_plot("subplot_fit") or quick_update:
            fit_imaging_plots.subplot_fit(
                fit=fit,
                output_path=self.image_path,
                output_format=self.fmt,
            )

        if quick_update:
            return

        if should_plot("subplot_of_galaxies"):
            galaxy_indices = list(range(len(fit.galaxies)))
            for galaxy_index in galaxy_indices:
                fit_imaging_plots.subplot_of_galaxy(
                    fit=fit,
                    galaxy_index=galaxy_index,
                    output_path=self.image_path,
                    output_format=self.fmt,
                )

        fits_to_fits(should_plot=should_plot, image_path=self.image_path, fit=fit)

    def imaging_combined(self, dataset_list: List[aa.Imaging]):
        def should_plot(name):
            return plot_setting(section=["dataset", "imaging"], name=name)

        if should_plot("subplot_dataset"):
            n = len(dataset_list)
            fig, axes = plt.subplots(n, 4, figsize=(28, 7 * n))
            if n == 1:
                axes = [axes]

            for i, dataset in enumerate(dataset_list):
                plot_array(dataset.data, "Data", ax=axes[i][0])
                plot_array(dataset.noise_map, "Noise Map", ax=axes[i][1])
                plot_array(dataset.signal_to_noise_map, "Signal-To-Noise Map", ax=axes[i][2])

            plt.tight_layout()
            _save_subplot(fig, self.image_path, "subplot_dataset_combined", self.fmt)

    def fit_imaging_combined(self, fit_list: List[FitImaging]):
        def should_plot(name):
            return plot_setting(section=["fit", "fit_imaging"], name=name)

        output = self.output_from()

        if should_plot("subplot_fit"):
            n = len(fit_list)
            fig, axes = plt.subplots(n, 5, figsize=(35, 7 * n))
            if n == 1:
                axes = [axes]

            for i, fit in enumerate(fit_list):
                plot_array(fit.data, "Data", ax=axes[i][0])
                plot_array(fit.signal_to_noise_map, "Signal-To-Noise Map", ax=axes[i][1])
                plot_array(fit.model_data, "Model Image", ax=axes[i][2])
                plot_array(fit.normalized_residual_map, "Normalized Residual Map", ax=axes[i][3])
                plot_array(fit.chi_squared_map, "Chi-Squared Map", ax=axes[i][4])

            plt.tight_layout()
            _save_subplot(fig, self.image_path, "subplot_fit_combined", self.fmt)
