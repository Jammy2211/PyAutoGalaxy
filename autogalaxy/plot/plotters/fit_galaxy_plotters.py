from autoarray.plot.plotters import fit_imaging_plotters
from autoarray.plot.plotters import structure_plotters
from autogalaxy import exc
from autoarray.plot.plotters import abstract_plotters
from autogalaxy.plot.mat_wrap import lensing_mat_plot, lensing_include, lensing_visuals

import copy


class FitGalaxyPlotter(fit_imaging_plotters.FitImagingPlotter):
    def __init__(
        self,
        fit,
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
    def visuals_with_include_2d(self):

        return self.visuals_2d + self.visuals_2d.__class__()

    @abstract_plotters.for_figure
    def figure_galaxy_data_array(self, galaxy_data):

        if galaxy_data.use_image:
            title = "Galaxy Data Image"
        elif galaxy_data.use_convergence:
            title = "Galaxy Data Convergence"
        elif galaxy_data.use_potential:
            title = "Galaxy Data Potential"
        elif galaxy_data.use_deflections_y:
            title = "Galaxy Data Deflections (y)"
        elif galaxy_data.use_deflections_x:
            title = "Galaxy Data Deflections (x)"
        else:
            raise exc.PlottingException(
                "The galaxy data arrays does not have a `True` use_profile_type"
            )

        self.mat_plot_2d.plot_array(
            array=galaxy_data.image, visuals_2d=self.visuals_with_include_2d
        )

    def figure_individuals(
        self,
        plot_image=False,
        plot_noise_map=False,
        plot_model_image=False,
        plot_residual_map=False,
        plot_normalized_residual_map=False,
        plot_chi_squared_map=False,
    ):

        if plot_image:

            self.figure_galaxy_data_array(galaxy_data=self.fit.masked_galaxy_dataset)

        super(FitGalaxyPlotter, self).figure_individuals(
            plot_noise_map=plot_noise_map,
            plot_model_image=plot_model_image,
            plot_residual_map=plot_residual_map,
            plot_normalized_residual_map=plot_normalized_residual_map,
            plot_chi_squared_map=plot_chi_squared_map,
        )

    @abstract_plotters.for_subplot
    def subplot_fit_galaxy(self):
        number_subplots = 4

        self.open_subplot_figure(number_subplots=number_subplots)

        self.setup_subplot(number_subplots=number_subplots, subplot_index=1)

        self.figure_galaxy_data_array(galaxy_data=self.fit.masked_galaxy_dataset)

        self.setup_subplot(number_subplots=number_subplots, subplot_index=2)

        self.figure_model_image()

        self.setup_subplot(number_subplots=number_subplots, subplot_index=3)

        self.figure_residual_map()

        self.setup_subplot(number_subplots=number_subplots, subplot_index=4)

        self.figure_chi_squared_map()

        self.mat_plot_2d.output.subplot_to_figure()

        self.mat_plot_2d.figure.close()
