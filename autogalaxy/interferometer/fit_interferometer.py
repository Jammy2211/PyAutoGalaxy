import numpy as np
from typing import Dict, Optional

from autoconf import cached_property

import autoarray as aa

from autogalaxy.abstract_fit import AbstractFit
from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.hyper.hyper_data import HyperBackgroundNoise
from autogalaxy.plane.plane import Plane
from autogalaxy.plane.to_inversion import PlaneToInversion
from autogalaxy.profiles.light_profiles.light_profiles_linear import LightProfileLinear


class FitInterferometer(aa.FitInterferometer, AbstractFit):
    def __init__(
        self,
        dataset: aa.Interferometer,
        plane: Plane,
        hyper_background_noise: HyperBackgroundNoise = None,
        use_hyper_scaling: bool = True,
        settings_pixelization: aa.SettingsPixelization = aa.SettingsPixelization(),
        settings_inversion: aa.SettingsInversion = aa.SettingsInversion(),
        preloads: aa.Preloads = aa.Preloads(),
        profiling_dict: Optional[Dict] = None,
    ):
        """
        An  lens fitter, which contains the plane's used to perform the fit and functions to manipulate \
        the lens dataset's hyper_galaxies.

        Parameters
        -----------
        plane : plane.Tracer
            The plane, which describes the ray-tracing and strong lens configuration.
        """

        super().__init__(
            dataset=dataset, use_mask_in_fit=False, profiling_dict=profiling_dict
        )
        AbstractFit.__init__(self=self, model_obj=plane, settings_inversion=settings_inversion)

        self.plane = plane

        self.hyper_background_noise = hyper_background_noise

        self.use_hyper_scaling = use_hyper_scaling

        self.settings_pixelization = settings_pixelization
        self.settings_inversion = settings_inversion

        self.preloads = preloads

    @property
    def noise_map(self):
        """
        Returns the interferometer's noise-map, which may have a hyper scaling performed which increase the noise in
        regions of the data that are poorly fitted in order to avoid overfitting.
        """
        if self.use_hyper_scaling and self.hyper_background_noise is not None:

            return self.hyper_background_noise.hyper_noise_map_complex_from(
                noise_map=self.dataset.noise_map
            )

        return self.dataset.noise_map

    @property
    def profile_visibilities(self):
        """
        Returns the visibilities of every light profile in the plane, which are computed by performing a Fourier
        transform to the sum of light profile images.
        """
        return self.plane.visibilities_from(
            grid=self.dataset.grid, transformer=self.dataset.transformer
        )

    @property
    def profile_subtracted_visibilities(self):
        """
        Returns the interferomter dataset's visibilities with all transformed light profile images in the fit's
        plane subtracted.
        """
        return self.visibilities - self.profile_visibilities

    @cached_property
    def inversion(self):
        """
        If the plane has linear objects which are used to fit the data (e.g. a pixelization) this function returns
        the linear inversion.

        The image passed to this function is the dataset's image with all light profile images of the plane subtracted.
        """
        if self.perform_inversion:

            plane_to_inversion = PlaneToInversion(plane=self.plane)

            return plane_to_inversion.inversion_interferometer_from(
                dataset=self.dataset,
                visibilities=self.profile_subtracted_visibilities,
                noise_map=self.noise_map,
                w_tilde=self.w_tilde,
                settings_pixelization=self.settings_pixelization,
                settings_inversion=self.settings_inversion,
                preloads=self.preloads,
            )

    @property
    def model_data(self):
        """
        Returns the model-image that is used to fit the data.

        If the plane does not have any linear objects and therefore omits an inversion, the model image is the
        sum of all light profile images.

        If a inversion is included it is the sum of this sum and the inversion's reconstruction of the image.
        """

        if self.plane.has(cls=aa.pix.Pixelization) or self.plane.has(
            cls=LightProfileLinear
        ):

            return self.profile_visibilities + self.inversion.mapped_reconstructed_data

        return self.profile_visibilities

    @property
    def grid(self):
        return self.interferometer.grid

    @property
    def galaxies(self):
        return self.plane.galaxies

    @property
    def galaxy_model_image_dict(self) -> {Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """
        galaxy_model_image_dict = self.plane.galaxy_image_2d_dict_from(grid=self.grid)

        galaxy_linear_obj_image_dict = self.galaxy_linear_obj_data_dict_from(
            use_image=True
        )

        return {**galaxy_model_image_dict, **galaxy_linear_obj_image_dict}

    @property
    def galaxy_model_visibilities_dict(self) -> {Galaxy: np.ndarray}:

        galaxy_model_visibilities_dict = self.plane.galaxy_visibilities_dict_from(
            grid=self.interferometer.grid, transformer=self.interferometer.transformer
        )

        galaxy_linear_obj_data_dict = self.galaxy_linear_obj_data_dict_from(
            use_image=False
        )

        return {**galaxy_model_visibilities_dict, **galaxy_linear_obj_data_dict}

    @property
    def model_visibilities_of_galaxies_list(self):
        return list(self.galaxy_model_visibilities_dict.values())

    def refit_with_new_preloads(self, preloads, settings_inversion=None):

        if self.profiling_dict is not None:
            profiling_dict = {}
        else:
            profiling_dict = None

        if settings_inversion is None:
            settings_inversion = self.settings_inversion

        return FitInterferometer(
            dataset=self.interferometer,
            plane=self.plane,
            hyper_background_noise=self.hyper_background_noise,
            use_hyper_scaling=self.use_hyper_scaling,
            settings_pixelization=self.settings_pixelization,
            settings_inversion=settings_inversion,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )
