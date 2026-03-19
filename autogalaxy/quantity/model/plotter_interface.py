from autoconf.fitsable import hdu_list_for_output_from

from autogalaxy.quantity.dataset_quantity import DatasetQuantity
from autogalaxy.quantity.fit_quantity import FitQuantity
from autogalaxy.quantity.plot.fit_quantity_plotters import FitQuantityPlotter
from autogalaxy.analysis.plotter_interface import PlotterInterface, plot_setting


class PlotterInterfaceQuantity(PlotterInterface):
    def dataset_quantity(self, dataset: DatasetQuantity):
        image_list = [
            dataset.data.native_for_fits,
            dataset.noise_map.native_for_fits,
        ]

        hdu_list = hdu_list_for_output_from(
            values_list=[
                image_list[0].mask.astype("float"),
            ]
            + image_list,
            ext_name_list=[
                "mask",
                "data",
                "noise_map",
            ],
            header_dict=dataset.mask.header_dict,
        )

        hdu_list.writeto(self.image_path / "dataset.fits", overwrite=True)

    def fit_quantity(
        self,
        fit: FitQuantity,
        fit_quanaity_plotter_cls=FitQuantityPlotter,
    ):
        def should_plot(name):
            return plot_setting(section="fit_quantity", name=name)

        output = self.output_from()

        fit_quantity_plotter = fit_quanaity_plotter_cls(
            fit=fit,
            output=output,
        )

        if should_plot("subplot_fit"):
            fit_quantity_plotter.subplot_fit()
