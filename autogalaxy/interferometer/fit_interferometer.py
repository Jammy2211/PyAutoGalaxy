import numpy as np

import autoarray as aa

from autogalaxy.galaxy.galaxy import Galaxy


class FitInterferometer(aa.FitInterferometer):
    def __init__(
        self,
        dataset,
        plane,
        hyper_background_noise=None,
        use_hyper_scalings=True,
        settings_pixelization=aa.SettingsPixelization(),
        settings_inversion=aa.SettingsInversion(),
    ):
        """ An  lens fitter, which contains the plane's used to perform the fit and functions to manipulate \
        the lens dataset's hyper_galaxies.

        Parameters
        -----------
        plane : plane.Tracer
            The plane, which describes the ray-tracing and strong lens configuration.
        """

        if use_hyper_scalings:

            if hyper_background_noise is not None:
                noise_map = hyper_background_noise.hyper_noise_map_complex_from(
                    noise_map=dataset.noise_map
                )
            else:
                noise_map = dataset.noise_map

        else:

            noise_map = dataset.noise_map

        self.plane = plane

        self.profile_visibilities = self.plane.visibilities_via_transformer_from(
            grid=dataset.grid, transformer=dataset.transformer
        )

        self.profile_subtracted_visibilities = (
            dataset.visibilities - self.profile_visibilities
        )

        if not plane.has_pixelization:

            inversion = None
            model_visibilities = self.profile_visibilities

        else:

            inversion = plane.inversion_interferometer_from(
                grid=dataset.grid_inversion,
                visibilities=self.profile_subtracted_visibilities,
                noise_map=noise_map,
                transformer=dataset.transformer,
                settings_pixelization=settings_pixelization,
                settings_inversion=settings_inversion,
            )

            model_visibilities = (
                self.profile_visibilities + inversion.mapped_reconstructed_data
            )

        fit = aa.FitDataComplex(
            data=dataset.visibilities,
            noise_map=noise_map,
            model_data=model_visibilities,
            inversion=inversion,
            use_mask_in_fit=False,
        )

        super().__init__(dataset=dataset, fit=fit)

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
