from typing import Dict, List

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.plot.mat_plot.two_d import MatPlot2D
from autogalaxy.plot.visuals.two_d import Visuals2D
from autogalaxy.plot.include.two_d import Include2D


class AdaptPlotter(Plotter):
    def __init__(
        self,
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
    ):
        super().__init__(
            mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
        )

    def get_visuals_2d(self) -> Visuals2D:
        return self.visuals_2d

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
            visuals_2d=self.get_visuals_2d(),
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
            visuals_2d=self.get_visuals_2d(),
            auto_labels=aplt.AutoLabels(
                title="galaxy Image", filename="adapt_galaxy_image"
            ),
        )

    def subplot_images_of_galaxies(
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

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename="subplot_adapt_images_of_galaxies"
        )

        self.close_subplot_figure()

    def subplot_contribution_map_list(
        self, contribution_map_list_list: List[aa.Array2D]
    ):
        """
        Plots a subplot of the contribution maps of all galaxies.

        This uses the `contribution_map_list` which is a list of each galaxy's corresponding contribution map.

        Parameters
        ----------
        contribution_map_list_list
            A list of each galaxy's corresponding contribution map.
        """
        contribution_maps = [
            contribution_map
            for contribution_map in contribution_map_list_list
            if contribution_map is not None
        ]

        number_subplots = len(contribution_maps)

        if number_subplots == 0:
            return

        self.open_subplot_figure(number_subplots=number_subplots)

        for contribution_map_array in contribution_maps:
            self.figure_contribution_map(contribution_map=contribution_map_array)

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename="subplot_contribution_map_list"
        )

        self.close_subplot_figure()

    def figure_contribution_map(self, contribution_map: aa.Array2D):
        """
        Plot the contribution map of a galaxy.

        Parameters
        ----------
        contribution_map
            The contribution map that is plotted.
        """
        self.mat_plot_2d.plot_array(
            array=contribution_map,
            visuals_2d=self.get_visuals_2d(),
            auto_labels=aplt.AutoLabels(
                title="Contribution Map", filename="contribution_map_2d"
            ),
        )
