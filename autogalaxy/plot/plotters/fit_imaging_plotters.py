import numpy as np
from autoarray.plot.plotters import fit_imaging_plotters
from autoarray.plot.plotters import inversion_plotters
from autoarray.plot.plotters import abstract_plotters
from autogalaxy.plot.mat_wrap import lensing_mat_plot, lensing_include, lensing_visuals
from autogalaxy.fit import fit as f


class FitImagingPlotter(fit_imaging_plotters.AbstractFitImagingPlotter):
    def __init__(
        self,
        fit: f.FitImaging,
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
                "mass_profile_centres", self.plane.light_profile_centres
            ),
            mass_profile_centres=self.extract_2d(
                "mass_profile_centres", self.plane.mass_profile_centres
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

    @abstract_plotters.for_figure_with_index
    def figure_subtracted_image_of_galaxy(self, galaxy_index):
        """Plot the model image of a specific plane of a lens fit.

        Set *autogalaxy.datas.arrays.mat_plot_2d.mat_plot_2d* for a description of all input parameters not described below.

        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractFitter
            The fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
        image_index : int
            The index of the datas in the datas-set of which the model image is plotted.
        galaxy_indexes : int
            The plane from which the model image is generated.
        """

        if len(self.fit.galaxies) > 1:

            other_galaxies_model_images = [
                model_image
                for i, model_image in enumerate(self.fit.model_images_of_galaxies)
                if i != galaxy_index
            ]

            subtracted_image = self.fit.image - sum(other_galaxies_model_images)

        else:

            subtracted_image = self.fit.image

        self.mat_plot_2d.cmap.kwargs["vmin"] = np.max(
            self.fit.model_images_of_galaxies[galaxy_index]
        )
        self.mat_plot_2d.cmap.kwargs["vmin"] = np.min(
            self.fit.model_images_of_galaxies[galaxy_index]
        )

        self.mat_plot_2d.plot_array(
            array=subtracted_image, visuals_2d=self.visuals_with_include_2d
        )

    @abstract_plotters.for_figure_with_index
    def figure_model_image_of_galaxy(self, galaxy_index):
        """Plot the model image of a specific plane of a lens fit.

        Set *autogalaxy.datas.arrays.mat_plot_2d.mat_plot_2d* for a description of all input parameters not described below.

        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractFitter
            The fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
        galaxy_indexes : [int]
            The plane from which the model image is generated.
        """

        self.mat_plot_2d.plot_array(
            array=self.fit.model_images_of_galaxies[galaxy_index],
            visuals_2d=self.visuals_with_include_2d,
        )

    def figure_individuals(
        self,
        plot_image=False,
        plot_noise_map=False,
        plot_signal_to_noise_map=False,
        plot_model_image=False,
        plot_residual_map=False,
        plot_normalized_residual_map=False,
        plot_chi_squared_map=False,
        plot_subtracted_images_of_galaxies=False,
        plot_model_images_of_galaxies=False,
    ):
        """Plot the model datas_ of an analysis, using the *Fitter* class object.

        The visualization and output type can be fully customized.

        Parameters
        -----------
        fit : autogalaxy.lens.fitting.Fitter
            Class containing fit between the model datas_ and observed lens datas_ (including residual_map, chi_squared_map etc.)
        output_path : str
            The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
        output_format : str
            How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
            in the python interpreter window.
        """

        super(FitImagingPlotter, self).figure_individuals(
            plot_image=plot_image,
            plot_noise_map=plot_noise_map,
            plot_signal_to_noise_map=plot_signal_to_noise_map,
            plot_model_image=plot_model_image,
            plot_residual_map=plot_residual_map,
            plot_normalized_residual_map=plot_normalized_residual_map,
            plot_chi_squared_map=plot_chi_squared_map,
        )

        if plot_subtracted_images_of_galaxies:

            for galaxy_index in range(len(self.fit.galaxies)):

                self.figure_subtracted_image_of_galaxy(galaxy_index=galaxy_index)

        if plot_model_images_of_galaxies:

            for galaxy_index in range(len(self.fit.galaxies)):

                self.figure_model_image_of_galaxy(galaxy_index=galaxy_index)

    def subplots_of_all_galaxies(self):

        for galaxy_index in range(len(self.fit.galaxies)):

            if (
                self.fit.galaxies[galaxy_index].has_light_profile
                or self.fit.galaxies[galaxy_index].has_pixelization
            ):

                self.subplot_of_galaxy(galaxy_index=galaxy_index)

    @abstract_plotters.for_subplot_with_index
    def subplot_of_galaxy(self, galaxy_index):
        """Plot the model datas_ of an analysis, using the *Fitter* class object.

        The visualization and output type can be fully customized.

        Parameters
        -----------
        fit : autogalaxy.lens.fitting.Fitter
            Class containing fit between the model datas_ and observed lens datas_ (including residual_map, chi_squared_map etc.)
        output_path : str
            The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
        output_filename : str
            The name of the file that is output, if the output_type is a file format (e.g. png, fits)
        output_format : str
            How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
            in the python interpreter window.
        """

        number_subplots = 4

        self.open_subplot_figure(number_subplots=number_subplots)

        self.setup_subplot(number_subplots=number_subplots, subplot_index=1)
        self.figure_image()

        self.setup_subplot(number_subplots=number_subplots, subplot_index=2)
        self.figure_subtracted_image_of_galaxy(galaxy_index=galaxy_index)

        self.setup_subplot(number_subplots=number_subplots, subplot_index=3)
        self.figure_model_image_of_galaxy(galaxy_index=galaxy_index)

        if self.plane.has_pixelization:

            aspect_inv = self.mat_plot_2d.figure.aspect_for_subplot_from_grid(
                grid=self.fit.inversion.mapper.source_full_grid
            )

            self.setup_subplot(
                number_subplots=number_subplots,
                subplot_index=4,
                aspect=float(aspect_inv),
            )
            self.inversion_plotter.figure_reconstruction()

        self.mat_plot_2d.output.subplot_to_figure()
        self.mat_plot_2d.figure.close()
