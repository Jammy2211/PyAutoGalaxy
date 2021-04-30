import numpy as np
from autoarray.plot.mat_wrap import mat_plot
from autoarray.plot import inversion_plotters, fit_imaging_plotters
from autogalaxy.plot.mat_wrap import lensing_mat_plot, lensing_include, lensing_visuals
from autogalaxy.profiles import light_profiles, mass_profiles
from autogalaxy.fit import fit_imaging


class FitImagingPlotter(fit_imaging_plotters.AbstractFitImagingPlotter):
    def __init__(
        self,
        fit: fit_imaging.FitImaging,
        mat_plot_2d: lensing_mat_plot.MatPlot2D = lensing_mat_plot.MatPlot2D(),
        visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
        include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    ):

        super().__init__(
            fit=fit,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

    @property
    def plane(self):
        return self.fit.plane

    @property
    def visuals_with_include_2d(self):

        visuals_2d = super().visuals_with_include_2d

        return visuals_2d + visuals_2d.__class__(
            light_profile_centres=self.extract_2d(
                "light_profile_centres",
                self.plane.extract_attribute(
                    cls=light_profiles.LightProfile, attr_name="centre"
                ),
            ),
            mass_profile_centres=self.extract_2d(
                "mass_profile_centres",
                self.plane.extract_attribute(
                    cls=mass_profiles.MassProfile, attr_name="centre"
                ),
            ),
        )

    @property
    def inversion_plotter(self):
        return inversion_plotters.InversionPlotter(
            inversion=self.fit.inversion,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_with_include_2d,
            include_2d=self.include_2d,
        )

    def galaxy_indexes_from_galaxy_index(self, galaxy_index):

        if galaxy_index is None:
            return range(len(self.fit.galaxies))
        else:
            return [galaxy_index]

    def figures_2d_of_galaxies(
        self, subtracted_image=False, model_image=False, galaxy_index=None
    ):

        galaxy_indexes = self.galaxy_indexes_from_galaxy_index(
            galaxy_index=galaxy_index
        )

        for galaxy_index in galaxy_indexes:

            if subtracted_image:

                self.mat_plot_2d.cmap.kwargs["vmin"] = np.max(
                    self.fit.model_images_of_galaxies[galaxy_index]
                )
                self.mat_plot_2d.cmap.kwargs["vmin"] = np.min(
                    self.fit.model_images_of_galaxies[galaxy_index]
                )

                self.mat_plot_2d.plot_array(
                    array=self.fit.subtracted_images_of_galaxies[galaxy_index],
                    visuals_2d=self.visuals_with_include_2d,
                    auto_labels=mat_plot.AutoLabels(
                        title=f"Subtracted Image of Galaxy {galaxy_index}",
                        filename=f"subtracted_image_of_galaxy_{galaxy_index}",
                    ),
                )

            if model_image:

                self.mat_plot_2d.plot_array(
                    array=self.fit.model_images_of_galaxies[galaxy_index],
                    visuals_2d=self.visuals_with_include_2d,
                    auto_labels=mat_plot.AutoLabels(
                        title=f"Model Image of Galaxy {galaxy_index}",
                        filename=f"model_image_of_galaxy_{galaxy_index}",
                    ),
                )

    def subplots_of_galaxies(self, galaxy_index=None):
        """Plot the model data of an analysis, using the *Fitter* class object.

        The visualization and output type can be fully customized.

        Parameters
        -----------
        fit : autogalaxy.lens.fitting.Fitter
            Class containing fit between the model data and observed lens data (including residual_map, chi_squared_map etc.)
        output_path : str
            The path where the data is output if the output_type is a file format (e.g. png, fits)
        output_filename : str
            The name of the file that is output, if the output_type is a file format (e.g. png, fits)
        output_format : str
            How the data is output. File formats (e.g. png, fits) output the data to harddisk. 'show' displays the data \
            in the python interpreter window.
        """

        galaxy_indexes = self.galaxy_indexes_from_galaxy_index(
            galaxy_index=galaxy_index
        )

        for galaxy_index in galaxy_indexes:

            self.open_subplot_figure(number_subplots=4)

            self.figures_2d(image=True)
            self.figures_2d_of_galaxies(
                galaxy_index=galaxy_index, subtracted_image=True
            )
            self.figures_2d_of_galaxies(galaxy_index=galaxy_index)

            if self.plane.has_pixelization:
                self.inversion_plotter.figures_2d(reconstruction=True)

            self.mat_plot_2d.output.subplot_to_figure(
                auto_filename=f"subplot_of_galaxy_{galaxy_index}"
            )
            self.close_subplot_figure()
