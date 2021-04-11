import numpy as np

from autoconf import conf
from autoarray.fit import fit as aa_fit
from autoarray.inversion import pixelizations as pix, inversions as inv
from autogalaxy.galaxy import galaxy as g


class FitImaging(aa_fit.FitImaging):
    def __init__(
        self,
        imaging,
        plane,
        hyper_image_sky=None,
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

        self.plane = plane

        if use_hyper_scalings:

            image = hyper_image_from_image_and_hyper_image_sky(
                image=imaging.image, hyper_image_sky=hyper_image_sky
            )

            noise_map = hyper_noise_map_from_noise_map_plane_and_hyper_background_noise(
                noise_map=imaging.noise_map,
                plane=plane,
                hyper_background_noise=hyper_background_noise,
            )

            if (
                plane.has_hyper_galaxy
                or hyper_image_sky is not None
                or hyper_background_noise is not None
            ):

                imaging = imaging.modify_image_and_noise_map(
                    image=image, noise_map=noise_map
                )

        else:

            image = imaging.image
            noise_map = imaging.noise_map

        self.blurred_image = plane.blurred_image_2d_from_grid_and_convolver(
            grid=imaging.grid,
            convolver=imaging.convolver,
            blurring_grid=imaging.blurring_grid,
        )

        self.profile_subtracted_image = image - self.blurred_image

        if not plane.has_pixelization:

            inversion = None
            model_image = self.blurred_image

        else:

            inversion = plane.inversion_imaging_from_grid_and_data(
                grid=imaging.grid_inversion,
                image=self.profile_subtracted_image,
                noise_map=noise_map,
                convolver=imaging.convolver,
                settings_pixelization=settings_pixelization,
                settings_inversion=settings_inversion,
            )

            model_image = self.blurred_image + inversion.mapped_reconstructed_image

        super().__init__(
            imaging=imaging,
            model_image=model_image,
            inversion=inversion,
            use_mask_in_fit=False,
        )

    @property
    def galaxies(self):
        return self.plane.galaxies

    @property
    def grid(self):
        return self.imaging.grid

    @property
    def galaxy_model_image_dict(self) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """

        galaxy_model_image_dict = self.plane.galaxy_blurred_image_dict_from_grid_and_convolver(
            grid=self.grid,
            convolver=self.imaging.convolver,
            blurring_grid=self.imaging.blurring_grid,
        )

        for galaxy in self.galaxies:

            if galaxy.has_pixelization:

                galaxy_model_image_dict.update(
                    {galaxy: self.inversion.mapped_reconstructed_image}
                )

        return galaxy_model_image_dict

    @property
    def model_images_of_galaxies(self):

        model_images_of_galaxies = self.plane.blurred_images_of_galaxies_from_grid_and_psf(
            grid=self.grid,
            psf=self.imaging.psf,
            blurring_grid=self.imaging.blurring_grid,
        )

        for galaxy_index, galaxy in enumerate(self.galaxies):

            if galaxy.has_pixelization:

                model_images_of_galaxies[
                    galaxy_index
                ] += self.inversion.mapped_reconstructed_image

        return model_images_of_galaxies

    @property
    def subtracted_images_of_galaxies(self):

        subtracted_images_of_galaxies = []

        model_images_of_galaxies = self.model_images_of_galaxies

        for galaxy_index in range(len(self.galaxies)):

            other_galaxies_model_images = [
                model_image
                for i, model_image in enumerate(model_images_of_galaxies)
                if i != galaxy_index
            ]

            subtracted_image = self.image - sum(other_galaxies_model_images)

            subtracted_images_of_galaxies.append(subtracted_image)

        return subtracted_images_of_galaxies

    @property
    def unmasked_blurred_image(self):
        return self.plane.unmasked_blurred_image_2d_from_grid_and_psf(
            grid=self.grid, psf=self.imaging.psf
        )

    @property
    def unmasked_blurred_image_of_galaxies(self):
        return self.plane.unmasked_blurred_image_of_galaxies_from_grid_and_psf(
            grid=self.grid, psf=self.imaging.psf
        )

    @property
    def total_inversions(self):
        return 1


def hyper_image_from_image_and_hyper_image_sky(image, hyper_image_sky):

    if hyper_image_sky is not None:
        return hyper_image_sky.hyper_image_from_image(image=image)
    else:
        return image


def hyper_noise_map_from_noise_map_plane_and_hyper_background_noise(
    noise_map, plane, hyper_background_noise
):

    hyper_noise_map = plane.hyper_noise_map_from_noise_map(noise_map=noise_map)

    if hyper_background_noise is not None:
        noise_map = hyper_background_noise.hyper_noise_map_from_noise_map(
            noise_map=noise_map
        )

    if hyper_noise_map is not None:
        noise_map = noise_map + hyper_noise_map
        noise_map_limit = conf.instance["general"]["hyper"]["hyper_noise_limit"]
        noise_map[noise_map > noise_map_limit] = noise_map_limit

    return noise_map
