import numpy as np
from typing import Dict, Optional

from autoconf import cached_property

import autoarray as aa

from autogalaxy.galaxy.galaxy import Galaxy


class FitInterferometer(aa.FitInterferometer):
    def __init__(
        self,
        dataset,
        plane,
        hyper_background_noise=None,
        use_hyper_scaling=True,
        settings_pixelization=aa.SettingsPixelization(),
        settings_inversion=aa.SettingsInversion(),
        preloads=aa.Preloads(),
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
        return self.plane.visibilities_via_transformer_from(
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
        if self.plane.has_pixelization:

            if self.settings_inversion.use_w_tilde:
                w_tilde = self.dataset.w_tilde
            else:
                w_tilde = None

            return self.plane.inversion_interferometer_from(
                grid=self.dataset.grid_inversion,
                visibilities=self.profile_subtracted_visibilities,
                noise_map=self.noise_map,
                transformer=self.dataset.transformer,
                w_tilde=w_tilde,
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

        if self.plane.has_pixelization:

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

        for path, image in galaxy_model_image_dict.items():
            galaxy_model_image_dict[path] = image.binned

        # TODO : Extend to multiple inversioons across Planes

        for galaxy in self.galaxies:

            if galaxy.has_pixelization:

                galaxy_model_image_dict.update(
                    {galaxy: self.inversion.mapped_reconstructed_image}
                )

        return galaxy_model_image_dict

    @property
    def galaxy_model_visibilities_dict(self) -> {Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """
        galaxy_model_visibilities_dict = self.plane.galaxy_visibilities_dict_via_transformer_from(
            grid=self.interferometer.grid, transformer=self.interferometer.transformer
        )

        # TODO : Extend to multiple inversioons across Planes

        for galaxy in self.galaxies:

            if galaxy.has_pixelization:

                galaxy_model_visibilities_dict.update(
                    {galaxy: self.inversion.mapped_reconstructed_data}
                )

        return galaxy_model_visibilities_dict

    def model_visibilities_of_galaxies(self):

        model_visibilities_of_galaxies = self.plane.visibilities_list_via_transformer_from(
            grid=self.interferometer.grid, transformer=self.interferometer.transformer
        )

        for (galaxy_index, galaxy) in enumerate(self.galaxies):

            if galaxy.has_pixelization:

                model_visibilities_of_galaxies[
                    galaxy_index
                ] += self.inversion.mapped_reconstructed_image

        return model_visibilities_of_galaxies

    @property
    def total_mappers(self):
        return 1

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
