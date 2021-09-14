from astropy import cosmology as cosmo

import autofit as af
import autoarray as aa

from autogalaxy.analysis.analysis import AnalysisDataset
from autogalaxy.imaging.model.result import ResultImaging
from autogalaxy.analysis.visualizer import Visualizer
from autogalaxy.imaging.fit_imaging import FitImaging

from autogalaxy import exc


class AnalysisImaging(AnalysisDataset):
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

    @property
    def imaging(self):
        return self.dataset

    def log_likelihood_function(self, instance):
        """
        Determine the fit of a lens galaxy and source galaxy to the imaging in this lens.

        Parameters
        ----------
        instance
            A model instance with attributes

        Returns
        -------
        fit : Fit
            A fractional value indicating how well this model fit and the model imaging itself
        """

        self.associate_hyper_images(instance=instance)
        plane = self.plane_for_instance(instance=instance)

        hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)

        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        try:
            fit = self.fit_imaging_for_plane(
                plane=plane,
                hyper_image_sky=hyper_image_sky,
                hyper_background_noise=hyper_background_noise,
            )

            return fit.figure_of_merit
        except (
            exc.PixelizationException,
            exc.InversionException,
            exc.GridException,
        ) as e:
            raise exc.FitException from e

    def fit_imaging_for_plane(
        self, plane, hyper_image_sky, hyper_background_noise, use_hyper_scalings=True
    ):

        return FitImaging(
            imaging=self.dataset,
            plane=plane,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            use_hyper_scalings=use_hyper_scalings,
            settings_pixelization=self.settings_pixelization,
            settings_inversion=self.settings_inversion,
        )

    def visualize(self, paths: af.DirectoryPaths, instance, during_analysis):

        instance = self.associate_hyper_images(instance=instance)
        plane = self.plane_for_instance(instance=instance)
        hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)
        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        fit = self.fit_imaging_for_plane(
            plane=plane,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        visualizer = Visualizer(visualize_path=paths.image_path)
        visualizer.visualize_imaging(imaging=self.imaging)
        visualizer.visualize_fit_imaging(fit=fit, during_analysis=during_analysis)

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

            fit = self.fit_imaging_for_plane(
                plane=plane,
                hyper_image_sky=None,
                hyper_background_noise=None,
                use_hyper_scalings=False,
            )

            visualizer.visualize_fit_imaging(
                fit=fit, during_analysis=during_analysis, subfolders="fit_no_hyper"
            )

    def make_result(
        self, samples: af.PDFSamples, model: af.Collection, search: af.NonLinearSearch
    ):
        return ResultImaging(samples=samples, model=model, analysis=self, search=search)

    def save_attributes_for_aggregator(self, paths: af.DirectoryPaths):

        super().save_attributes_for_aggregator(paths=paths)

        paths.save_object("psf", self.dataset.psf_unormalized)
        paths.save_object("mask", self.dataset.mask)
