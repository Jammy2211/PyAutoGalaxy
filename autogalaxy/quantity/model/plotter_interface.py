from autoconf.fitsable import hdu_list_for_output_from

from autogalaxy.quantity.dataset_quantity import DatasetQuantity
from autogalaxy.quantity.fit_quantity import FitQuantity
from autogalaxy.quantity.plot.fit_quantity_plotters import FitQuantityPlotter
from autogalaxy.analysis.plotter_interface import PlotterInterface
from autogalaxy.analysis.plotter_interface import plot_setting
from autogalaxy.plot.visuals.two_d import Visuals2D


class PlotterInterfaceQuantity(PlotterInterface):
    def dataset_quantity(self, dataset: DatasetQuantity):
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
        visuals_2d: Visuals2D = None,
        fit_quanaity_plotter_cls=FitQuantityPlotter,
    ):
        """
        Visualizes a `FitQuantity` object, which fits a quantity of a light or mass profile (e.g. an image, potential)
        to the same quantity of another light or mass profile.

        Images are output to the `image` folder of the `image_path`. When used with a non-linear search the `image_path`
        points to the search's results folder and this function visualizes the maximum log likelihood `FitQuantity`
        inferred by the search so far.

        Visualization includes a subplot of individual images of attributes of the `FitQuantity` (e.g. the model data,
        residual map).

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under
        the `fit_quantity` header.

        Parameters
        ----------
        fit
            The maximum log likelihood `FitQuantity` of the non-linear search which is used to plot the fit.
        visuals_2d
            An object containing attributes which may be plotted over the figure (e.g. the centres of mass and light
            profiles).
        """

        def should_plot(name):
            return plot_setting(section="fit_quantity", name=name)

        mat_plot_2d = self.mat_plot_2d_from()

        fit_quantity_plotter = fit_quanaity_plotter_cls(
            fit=fit,
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
        )

        if should_plot("subplot_fit"):
            fit_quantity_plotter.subplot_fit()
