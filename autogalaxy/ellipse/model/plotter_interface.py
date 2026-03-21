from typing import List

from autoconf.fitsable import hdu_list_for_output_from

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.ellipse.fit_ellipse import FitEllipse
from autogalaxy.ellipse.plot import fit_ellipse_plots
from autogalaxy.analysis.plotter_interface import PlotterInterface, plot_setting


class PlotterInterfaceEllipse(PlotterInterface):
    def imaging(self, dataset: aa.Imaging):
        def should_plot(name):
            return plot_setting(section=["dataset", "imaging"], name=name)

        output = self.output_from()

        dataset_plotter = aplt.ImagingPlotter(
            dataset=dataset,
            output=output,
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
        def should_plot(name):
            return plot_setting(section=["fit", "fit_ellipse"], name=name)

        if should_plot("data"):
            fit_ellipse_plots.plot_data(
                fit_list=fit_list,
                output_path=self.image_path,
                output_format=self.fmt,
            )

        if should_plot("ellipse_residuals"):
            fit_ellipse_plots.plot_ellipse_residuals(
                fit_list=fit_list,
                output_path=self.image_path,
                output_format=self.fmt,
            )

        if should_plot("data_no_ellipse"):
            fit_ellipse_plots.plot_data(
                fit_list=fit_list,
                output_path=self.image_path,
                output_format=self.fmt,
                disable_data_contours=True,
            )

        if should_plot("subplot_fit_ellipse"):
            fit_ellipse_plots.subplot_fit_ellipse(
                fit_list=fit_list,
                output_path=self.image_path,
                output_format=self.fmt,
            )

        fit_ellipse_plots.plot_data(
            fit_list=fit_list,
            output_path=self.image_path,
            output_format=self.fmt,
            use_log10=True,
        )

        if should_plot("data_no_ellipse"):
            fit_ellipse_plots.plot_data(
                fit_list=fit_list,
                output_path=self.image_path,
                output_format=self.fmt,
                use_log10=True,
                disable_data_contours=True,
            )
