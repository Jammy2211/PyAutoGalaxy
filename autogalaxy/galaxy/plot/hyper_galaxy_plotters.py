from typing import Dict, List

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.plot.mat_wrap.mat_plot import MatPlot2D
from autogalaxy.plot.mat_wrap.visuals import Visuals2D
from autogalaxy.plot.mat_wrap.include import Include2D


class HyperPlotter(Plotter):
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

    def figure_hyper_model_image(self, hyper_model_image: aa.Array2D):
        """
        Plot the hyper model image of a hyper galaxy.

        Parameters
        -----------
        hyper_model_image
            The hyper model image that is plotted.
        """

        self.mat_plot_2d.plot_array(
            array=hyper_model_image,
            visuals_2d=self.get_visuals_2d(),
            auto_labels=aplt.AutoLabels(
                title="Hyper Model Image", filename="hyper_model_image"
            ),
        )

    def figure_hyper_galaxy_image(self, galaxy_image: aa.Array2D):
        """
        Plot the hyper galaxy image of a hyper galaxy.

        Parameters
        -----------
        galaxy_image
            The hyper galaxy image that is plotted.
        """
        self.mat_plot_2d.plot_array(
            array=galaxy_image,
            visuals_2d=self.get_visuals_2d(),
            auto_labels=aplt.AutoLabels(
                title="Hyper Galaxy Image", filename="hyper_galaxy_image"
            ),
        )

    def figure_contribution_map(self, contribution_map: aa.Array2D):
        """
        Plot the contribution map of a hyper galaxy.

        Parameters
        -----------
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

    def subplot_hyper_images_of_galaxies(
        self, hyper_galaxy_image_path_dict: Dict[Galaxy, aa.Array2D]
    ):
        """
        Plots a subplot of the hyper galaxy image of all hyper galaxies.

        This uses the `hyper_galaxy_image_path_dict` which is a dictionary mapping each galaxy to its corresponding
        to hyper galaxy image.

        Parameters
        ----------
        hyper_galaxy_image_path_dict
            A dictionary mapping each galaxy to its corresponding to hyper galaxy image.
        """
        if hyper_galaxy_image_path_dict is None:
            return

        self.open_subplot_figure(number_subplots=len(hyper_galaxy_image_path_dict))

        for path, galaxy_image in hyper_galaxy_image_path_dict.items():

            self.figure_hyper_galaxy_image(galaxy_image=galaxy_image)

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename="subplot_hyper_images_of_galaxies"
        )

        self.close_subplot_figure()

    def subplot_contribution_map_list(
        self, contribution_map_list_list: List[aa.Array2D]
    ):
        """
        Plots a subplot of the contribution maps of all hyper galaxies.

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
