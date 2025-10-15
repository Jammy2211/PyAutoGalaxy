from typing import Dict, List

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.plot.mat_plot.two_d import MatPlot2D
from autogalaxy.plot.visuals.two_d import Visuals2D


class AdaptPlotter(Plotter):
    def __init__(
        self,
        mat_plot_2d: MatPlot2D = None,
        visuals_2d: Visuals2D = None,
    ):
        super().__init__(mat_plot_2d=mat_plot_2d, visuals_2d=visuals_2d)

    def figure_model_image(self, model_image: aa.Array2D):
        """
        Plots the adapt model image (e.g. sum of all individual galaxy model images).

        Parameters
        ----------
        model_image
            The adapt model image that is plotted.
        """

        self.mat_plot_2d.plot_array(
            array=model_image,
            visuals_2d=self.visuals_2d,
            auto_labels=aplt.AutoLabels(
                title="adapt image", filename="adapt_model_image"
            ),
        )

    def figure_galaxy_image(self, galaxy_image: aa.Array2D):
        """
        Plot the galaxy image of a galaxy.

        Parameters
        ----------
        galaxy_image
            The galaxy image that is plotted.
        """
        self.mat_plot_2d.plot_array(
            array=galaxy_image,
            visuals_2d=self.visuals_2d,
            auto_labels=aplt.AutoLabels(
                title="galaxy Image", filename="adapt_galaxy_image"
            ),
        )

    def subplot_adapt_images(
        self, adapt_galaxy_name_image_dict: Dict[Galaxy, aa.Array2D]
    ):
        """
        Plots a subplot of the galaxy image of all galaxies.

        This uses the `adapt_galaxy_name_image_dict` which is a dictionary mapping each galaxy to its corresponding
        to galaxy image.

        Parameters
        ----------
        adapt_galaxy_name_image_dict
            A dictionary mapping each galaxy to its corresponding to galaxy image.
        """
        if adapt_galaxy_name_image_dict is None:
            return

        self.open_subplot_figure(number_subplots=len(adapt_galaxy_name_image_dict))

        for path, galaxy_image in adapt_galaxy_name_image_dict.items():
            self.figure_galaxy_image(galaxy_image=galaxy_image)

        self.mat_plot_2d.output.subplot_to_figure(auto_filename="subplot_adapt_images")

        self.close_subplot_figure()
