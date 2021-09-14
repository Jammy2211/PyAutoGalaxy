from astropy import cosmology as cosmo

import autofit as af
import autoarray as aa

from autogalaxy.analysis.analysis import AnalysisDataset
from autogalaxy.interferometer.model.result import ResultInterferometer
from autogalaxy.analysis.visualizer import Visualizer
from autogalaxy.interferometer.fit_interferometer import FitInterferometer
from autogalaxy.galaxy.galaxy import Galaxy

from autogalaxy import exc


class AnalysisInterferometer(AnalysisDataset):
    def __init__(
        self,
        dataset,
        hyper_dataset_result=None,
        cosmology=cosmo.Planck15,
        settings_pixelization=aa.SettingsPixelization(),
        settings_inversion=aa.SettingsInversion(),
    ):

        super().__init__(
            dataset=dataset,
            hyper_dataset_result=hyper_dataset_result,
            cosmology=cosmology,
            settings_pixelization=settings_pixelization,
            settings_inversion=settings_inversion,
        )

        if self.hyper_dataset_result is not None:

            self.set_hyper_dataset(result=self.hyper_dataset_result)

        else:

            self.hyper_galaxy_visibilities_path_dict = None
            self.hyper_model_visibilities = None

    def set_hyper_dataset(self, result):

        super().set_hyper_dataset(result=result)

        self.hyper_model_visibilities = result.hyper_model_visibilities
        self.hyper_galaxy_visibilities_path_dict = (
            result.hyper_galaxy_visibilities_path_dict
        )

    @property
    def interferometer(self):
        return self.dataset

    def log_likelihood_function(self, instance):
        """
        Determine the fit of a lens galaxy and source galaxy to the interferometer in this lens.

        Parameters
        ----------
        instance
            A model instance with attributes

        Returns
        -------
        fit : Fit
            A fractional value indicating how well this model fit and the model interferometer itself
        """

        self.associate_hyper_images(instance=instance)
        plane = self.plane_for_instance(instance=instance)

        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        try:
            fit = self.fit_interferometer_for_plane(
                plane=plane, hyper_background_noise=hyper_background_noise
            )

            return fit.figure_of_merit
        except (
            exc.PixelizationException,
            exc.InversionException,
            exc.GridException,
        ) as e:
            raise exc.FitException from e

    def associate_hyper_visibilities(
        self, instance: af.ModelInstance
    ) -> af.ModelInstance:
        """
        Takes visibilities from the last result, if there is one, and associates them with galaxies in this search
        where full-path galaxy names match.

        If the galaxy collection has a different name then an association is not made.

        e.g.
        galaxies.lens will match with:
            galaxies.lens
        but not with:
            galaxies.lens
            galaxies.source

        Parameters
        ----------
        instance
            A model instance with 0 or more galaxies in its tree

        Returns
        -------
        instance
           The input instance with visibilities associated with galaxies where possible.
        """
        if self.hyper_galaxy_visibilities_path_dict is not None:
            for galaxy_path, galaxy in instance.path_instance_tuples_for_class(Galaxy):
                if galaxy_path in self.hyper_galaxy_visibilities_path_dict:
                    galaxy.hyper_model_visibilities = self.hyper_model_visibilities
                    galaxy.hyper_galaxy_visibilities = self.hyper_galaxy_visibilities_path_dict[
                        galaxy_path
                    ]

        return instance

    def fit_interferometer_for_plane(
        self, plane, hyper_background_noise, use_hyper_scalings=True
    ):

        return FitInterferometer(
            interferometer=self.dataset,
            plane=plane,
            hyper_background_noise=hyper_background_noise,
            use_hyper_scalings=use_hyper_scalings,
            settings_pixelization=self.settings_pixelization,
            settings_inversion=self.settings_inversion,
        )

    def visualize(self, paths: af.DirectoryPaths, instance, during_analysis):

        self.associate_hyper_images(instance=instance)
        plane = self.plane_for_instance(instance=instance)
        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        fit = self.fit_interferometer_for_plane(
            plane=plane, hyper_background_noise=hyper_background_noise
        )

        visualizer = Visualizer(visualize_path=paths.image_path)
        visualizer.visualize_interferometer(interferometer=self.interferometer)
        visualizer.visualize_fit_interferometer(
            fit=fit, during_analysis=during_analysis
        )
        if fit.inversion is not None:
            visualizer.visualize_inversion(
                inversion=fit.inversion, during_analysis=during_analysis
            )

        visualizer.visualize_hyper_images(
            hyper_galaxy_image_path_dict=self.hyper_galaxy_image_path_dict,
            hyper_model_image=self.hyper_model_image,
            plane=plane,
        )

        if visualizer.plot_fit_no_hyper:
            fit = self.fit_interferometer_for_plane(
                plane=plane, hyper_background_noise=None, use_hyper_scalings=False
            )

            visualizer.visualize_fit_interferometer(
                fit=fit, during_analysis=during_analysis, subfolders="fit_no_hyper"
            )

    def make_result(
        self, samples: af.PDFSamples, model: af.Collection, search: af.NonLinearSearch
    ):
        return ResultInterferometer(
            samples=samples, model=model, analysis=self, search=search
        )

    def save_attributes_for_aggregator(self, paths: af.DirectoryPaths):

        super().save_attributes_for_aggregator(paths=paths)

        paths.save_object("uv_wavelengths", self.dataset.uv_wavelengths)
        paths.save_object("real_space_mask", self.dataset.real_space_mask)
