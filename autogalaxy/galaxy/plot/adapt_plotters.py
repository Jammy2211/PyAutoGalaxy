from typing import Dict, List

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.plot.mat_plot.two_d import MatPlot2D


class AdaptPlotter(Plotter):
    def __init__(
        self,
        mat_plot_2d: MatPlot2D = None,
    ):
        super().__init__(mat_plot_2d=mat_plot_2d)

    def figure_model_image(self, model_image: aa.Array2D):
        self._plot_array(
            array=model_image,
            auto_labels=aplt.AutoLabels(
                title="adapt image", filename="adapt_model_image"
            ),
        )

    def figure_galaxy_image(self, galaxy_image: aa.Array2D):
        self._plot_array(
            array=galaxy_image,
            auto_labels=aplt.AutoLabels(
                title="galaxy Image", filename="adapt_galaxy_image"
            ),
        )

    def subplot_adapt_images(
        self, adapt_galaxy_name_image_dict: Dict[Galaxy, aa.Array2D]
    ):
        if adapt_galaxy_name_image_dict is None:
            return

        self.open_subplot_figure(number_subplots=len(adapt_galaxy_name_image_dict))

        for path, galaxy_image in adapt_galaxy_name_image_dict.items():
            self.figure_galaxy_image(galaxy_image=galaxy_image)

        self.mat_plot_2d.output.subplot_to_figure(auto_filename="subplot_adapt_images")

        self.close_subplot_figure()
