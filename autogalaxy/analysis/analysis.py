from astropy import cosmology as cosmo

import autofit as af
import autoarray as aa

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.plane.plane import Plane


class Analysis(af.Analysis):
    def __init__(self, hyper_dataset_result=None, cosmology=cosmo.Planck15):
        self.hyper_dataset_result = hyper_dataset_result
        self.cosmology = cosmology


class AnalysisDataset(Analysis):
    def __init__(
        self,
        dataset,
        hyper_dataset_result=None,
        cosmology=cosmo.Planck15,
        settings_pixelization=aa.pix.SettingsPixelization(),
        settings_inversion=aa.SettingsInversion(),
    ):

        super().__init__(hyper_dataset_result=hyper_dataset_result, cosmology=cosmology)

        self.dataset = dataset

        if self.hyper_dataset_result is not None:

            if hyper_dataset_result.search is not None:
                hyper_dataset_result.search.paths = None

            self.set_hyper_dataset(result=self.hyper_dataset_result)

        else:

            self.hyper_galaxy_image_path_dict = None
            self.hyper_model_image = None

        self.settings_pixelization = settings_pixelization
        self.settings_inversion = settings_inversion

        self.preloads = aa.Preloads()

    def set_hyper_dataset(self, result):

        self.hyper_galaxy_image_path_dict = result.hyper_galaxy_image_path_dict
        self.hyper_model_image = result.hyper_model_image

    def hyper_image_sky_for_instance(self, instance):

        if hasattr(instance, "hyper_image_sky"):
            return instance.hyper_image_sky

    def hyper_background_noise_for_instance(self, instance):

        if hasattr(instance, "hyper_background_noise"):
            return instance.hyper_background_noise

    def plane_for_instance(self, instance):
        return Plane(galaxies=instance.galaxies)

    def associate_hyper_images(self, instance: af.ModelInstance) -> af.ModelInstance:
        """
        Takes images from the last result, if there is one, and associates them with galaxies in this search
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
           The input instance with images associated with galaxies where possible.
        """

        if self.hyper_galaxy_image_path_dict is not None:

            for galaxy_path, galaxy in instance.path_instance_tuples_for_class(Galaxy):
                if galaxy_path in self.hyper_galaxy_image_path_dict:
                    galaxy.hyper_model_image = self.hyper_model_image

                    galaxy.hyper_galaxy_image = self.hyper_galaxy_image_path_dict[
                        galaxy_path
                    ]

        return instance

    def save_attributes_for_aggregator(self, paths: af.DirectoryPaths):

        paths.save_object("data", self.dataset.data)
        paths.save_object("noise_map", self.dataset.noise_map)
        paths.save_object("settings_dataset", self.dataset.settings)
        paths.save_object("settings_inversion", self.settings_inversion)
        paths.save_object("settings_pixelization", self.settings_pixelization)

        paths.save_object("cosmology", self.cosmology)

        if self.hyper_model_image is not None:
            paths.save_object("hyper_model_image", self.hyper_model_image)

        if self.hyper_galaxy_image_path_dict is not None:
            paths.save_object(
                "hyper_galaxy_image_path_dict", self.hyper_galaxy_image_path_dict
            )
