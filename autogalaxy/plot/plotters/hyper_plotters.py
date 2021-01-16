from autoarray.plot.plotters import abstract_plotters
from autoarray.plot.mat_wrap import mat_plot as mp
from autogalaxy.plot.mat_wrap import lensing_mat_plot, lensing_include, lensing_visuals


class HyperPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        mat_plot_2d: lensing_mat_plot.MatPlot2D = lensing_mat_plot.MatPlot2D(),
        visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
        include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    ):
        super().__init__(
            mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
        )

    @property
    def visuals_with_include_2d(self) -> lensing_visuals.Visuals2D:
        """
        Extracts from a `Structure` attributes that can be plotted and return them in a `Visuals` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From an `AbstractStructure` the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the structure's coordinate system.
        - mask: the mask of the structure.
        - border: the border of the structure's mask.

        Parameters
        ----------
        structure : abstract_structure.AbstractStructure
            The structure whose attributes are extracted for plotting.

        Returns
        -------
        vis.Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """
        return self.visuals_2d

    def figure_hyper_model_image(self, hyper_model_image):
        """Plot the image of a hyper_galaxies galaxy image.

        Set *autogalaxy.datas.arrays.mat_plot_2d.mat_plot_2d* for a description of all input parameters not described below.

        Parameters
        -----------
        hyper_galaxy_image : datas.imaging.datas.Imaging
            The hyper_galaxies galaxy image.
        origin : True
            If true, the origin of the datas's coordinate system is plotted as a 'x'.
        """
        self.mat_plot_2d.plot_array(
            array=hyper_model_image,
            visuals_2d=self.visuals_with_include_2d,
            auto_labels=mp.AutoLabels(
                title="Hyper Model Image", filename="hyper_model_image"
            ),
        )

    def figure_hyper_galaxy_image(self, galaxy_image):
        """Plot the image of a hyper_galaxies galaxy image.

        Set *autogalaxy.datas.arrays.mat_plot_2d.mat_plot_2d* for a description of all input parameters not described below.

        Parameters
        -----------
        hyper_galaxy_image : datas.imaging.datas.Imaging
            The hyper_galaxies galaxy image.
        origin : True
            If true, the origin of the datas's coordinate system is plotted as a 'x'.
        """
        self.mat_plot_2d.plot_array(
            array=galaxy_image,
            visuals_2d=self.visuals_with_include_2d,
            auto_labels=mp.AutoLabels(
                title="Hyper Galaxy Image", filename="hyper_galaxy_image"
            ),
        )

    def figure_contribution_map(self, contribution_map_in):
        """Plot the summed contribution maps of a hyper_galaxies-fit.

        Set *autogalaxy.datas.arrays.mat_plot_2d.mat_plot_2d* for a description of all input parameters not described below.

        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractLensHyperFit
            The hyper_galaxies-fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
        image_index : int
            The index of the datas in the datas-set of which the contribution_maps are plotted.
        """
        self.mat_plot_2d.plot_array(
            array=contribution_map_in,
            visuals_2d=self.visuals_with_include_2d,
            auto_labels=mp.AutoLabels(
                title="Contribution Map", filename="contribution_map"
            ),
        )

    def subplot_hyper_images_of_galaxies(self, hyper_galaxy_image_path_dict):

        if hyper_galaxy_image_path_dict is None:
            return

        self.open_subplot_figure(number_subplots=len(hyper_galaxy_image_path_dict))

        for path, galaxy_image in hyper_galaxy_image_path_dict.items():

            self.figure_hyper_galaxy_image(galaxy_image=galaxy_image)

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename="subplot_hyper_images_of_galaxies"
        )

        self.close_subplot_figure()

    def subplot_contribution_maps_of_galaxies(self, contribution_maps_of_galaxies):

        contribution_maps = [
            contribution_map
            for contribution_map in contribution_maps_of_galaxies
            if contribution_map is not None
        ]

        number_subplots = len(contribution_maps)

        if number_subplots == 0:
            return

        self.open_subplot_figure(number_subplots=number_subplots)

        for contribution_map_array in contribution_maps:

            self.figure_contribution_map(contribution_map_in=contribution_map_array)

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename="subplot_contribution_maps_of_galaxies"
        )

        self.close_subplot_figure()
