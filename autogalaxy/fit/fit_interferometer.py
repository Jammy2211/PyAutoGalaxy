import numpy as np

from autoconf import conf
from autoarray.fit import fit as aa_fit
from autoarray.inversion import pixelizations as pix, inversions as inv
from autogalaxy.galaxy import galaxy as g


class FitInterferometer(aa_fit.FitInterferometer):
    def __init__(
        self,
        interferometer,
        plane,
        hyper_background_noise=None,
        use_hyper_scalings=True,
        settings_pixelization=pix.SettingsPixelization(),
        settings_inversion=inv.SettingsInversion(),
    ):
        """ An  lens fitter, which contains the plane's used to perform the fit and functions to manipulate \
        the lens dataset's hyper_galaxies.

        Parameters
        -----------
        plane : plane.Tracer
            The plane, which describes the ray-tracing and strong lens configuration.
        scaled_array_2d_from_array_1d : func
            A function which maps the 1D lens hyper_galaxies to its unmasked 2D arrays.
        """

        if use_hyper_scalings:

            if hyper_background_noise is not None:
                noise_map = hyper_background_noise.hyper_noise_map_from_complex_noise_map(
                    noise_map=interferometer.noise_map
                )
            else:
                noise_map = interferometer.noise_map

            if hyper_background_noise is not None:

                interferometer = interferometer.modify_noise_map(noise_map=noise_map)

        else:

            noise_map = interferometer.noise_map

        self.plane = plane

        self.profile_visibilities = plane.profile_visibilities_from_grid_and_transformer(
            grid=interferometer.grid, transformer=interferometer.transformer
        )

        self.profile_subtracted_visibilities = (
            interferometer.visibilities - self.profile_visibilities
        )

        if not plane.has_pixelization:

            inversion = None
            model_visibilities = self.profile_visibilities

        else:

            inversion = plane.inversion_interferometer_from_grid_and_data(
                grid=interferometer.grid_inversion,
                visibilities=self.profile_subtracted_visibilities,
                noise_map=noise_map,
                transformer=interferometer.transformer,
                settings_pixelization=settings_pixelization,
                settings_inversion=settings_inversion,
            )

            model_visibilities = (
                self.profile_visibilities + inversion.mapped_reconstructed_visibilities
            )

        super().__init__(
            interferometer=interferometer,
            model_visibilities=model_visibilities,
            inversion=inversion,
            use_mask_in_fit=False,
        )

    @property
    def grid(self):
        return self.interferometer.grid

    @property
    def galaxies(self):
        return self.plane.galaxies

    @property
    def galaxy_model_image_dict(self) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """
        galaxy_model_image_dict = self.plane.galaxy_image_dict_from_grid(grid=self.grid)

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
    def galaxy_model_visibilities_dict(self) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """
        galaxy_model_visibilities_dict = self.plane.galaxy_profile_visibilities_dict_from_grid_and_transformer(
            grid=self.interferometer.grid, transformer=self.interferometer.transformer
        )

        # TODO : Extend to multiple inversioons across Planes

        for galaxy in self.galaxies:

            if galaxy.has_pixelization:

                galaxy_model_visibilities_dict.update(
                    {galaxy: self.inversion.mapped_reconstructed_visibilities}
                )

        return galaxy_model_visibilities_dict

    def model_visibilities_of_galaxies(self):

        model_visibilities_of_galaxies = self.plane.profile_visibilities_of_galaxies_from_grid_and_transformer(
            grid=self.interferometer.grid, transformer=self.interferometer.transformer
        )

        for (galaxy_index, galaxy) in enumerate(self.galaxies):

            if galaxy.has_pixelization:

                model_visibilities_of_galaxies[
                    galaxy_index
                ] += self.inversion.mapped_reconstructed_image

        return model_visibilities_of_galaxies

    @property
    def total_inversions(self):
        return 1
