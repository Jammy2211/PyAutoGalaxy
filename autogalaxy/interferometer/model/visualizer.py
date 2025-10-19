import autofit as af

from autogalaxy.interferometer.model.plotter_interface import (
    PlotterInterfaceInterferometer,
)
from autogalaxy import exc


class VisualizerInterferometer(af.Visualizer):
    @staticmethod
    def visualize_before_fit(
        analysis,
        paths: af.AbstractPaths,
        model: af.AbstractPriorModel,
    ):
        """
        PyAutoFit calls this function immediately before the non-linear search begins.

        It visualizes objects which do not change throughout the model fit like the dataset.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        """

        PlotterInterface = PlotterInterfaceInterferometer(
            image_path=paths.image_path, title_prefix=analysis.title_prefix
        )

        PlotterInterface.interferometer(dataset=analysis.interferometer)

        if analysis.adapt_images is not None:
            PlotterInterface.adapt_images(adapt_images=analysis.adapt_images)

    @staticmethod
    def visualize(
        analysis,
        paths: af.DirectoryPaths,
        instance: af.ModelInstance,
        during_analysis: bool,
        quick_update: bool = False,
    ):
        """
        Outputs images of the maximum log likelihood model inferred by the model-fit. This function is called
        throughout the non-linear search at input intervals, and therefore provides on-the-fly visualization of how
        well the model-fit is going.

        The visualization performed by this function includes:

        - Images of the best-fit galaxies, including the images of each galaxy.

        - Images of the best-fit `FitInterferometer`, including the model-image, residuals and chi-squared of its fit
          to the imaging data.

        - The adapt-images of the model-fit showing how the galaxies are used to represent different galaxies in
          the dataset.

        - If adapt features are used to scale the noise, a `FitInterferometer` with these features turned off may be
          output, to indicate how much these features are altering the dataset.

        The images output by this function are customized using the file `config/visualize/plots.yaml`.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization, and the pickled objects used by the aggregator output by this function.
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).
        """
        fit = analysis.fit_from(instance=instance)

        plotter_interface = PlotterInterfaceInterferometer(
            image_path=paths.image_path, title_prefix=analysis.title_prefix
        )

        try:
            plotter_interface.fit_interferometer(
                fit=fit,
                quick_update=quick_update,
            )
        except exc.InversionException:
            pass

        if quick_update:
            return

        galaxies = fit.galaxies_linear_light_profiles_to_light_profiles

        plotter_interface.galaxies(
            galaxies=galaxies,
            grid=fit.grids.lp,
        )

        if fit.inversion is not None:
            try:
                plotter_interface.inversion(
                    inversion=fit.inversion,
                )
            except IndexError:
                pass
