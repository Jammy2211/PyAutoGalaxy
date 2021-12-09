import numpy as np

from autoconf import conf
import autoarray as aa

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.plane.plane import Plane


class FitImaging(aa.FitImaging):
    def __init__(
        self,
        dataset: aa.Imaging,
        plane: Plane,
        hyper_image_sky=None,
        hyper_background_noise=None,
        use_hyper_scalings: bool = True,
        settings_pixelization: aa.SettingsPixelization = aa.SettingsPixelization(),
        settings_inversion: aa.SettingsInversion = aa.SettingsInversion(),
    ):
        """ An lens fitter, which contains the plane's used to perform the fit and functions to manipulate \
        the lens dataset's hyper_galaxies.

        Parameters
        -----------
        plane
            The plane of galaxies whose model images are used to fit the imaging data.
        """

        self.plane = plane

        if use_hyper_scalings:

            image = hyper_image_from(
                image=dataset.image, hyper_image_sky=hyper_image_sky
            )

            noise_map = hyper_noise_map_from(
                noise_map=dataset.noise_map,
                plane=plane,
                hyper_background_noise=hyper_background_noise,
            )

        else:

            image = dataset.image
            noise_map = dataset.noise_map

        self.blurred_image = self.plane.blurred_image_2d_via_convolver_from(
            grid=dataset.grid,
            convolver=dataset.convolver,
            blurring_grid=dataset.blurring_grid,
        )

        self.profile_subtracted_image = image - self.blurred_image

        if not plane.has_pixelization:

            inversion = None
            model_image = self.blurred_image

        else:

            inversion = plane.inversion_imaging_from(
                grid=dataset.grid_inversion,
                image=self.profile_subtracted_image,
                noise_map=noise_map,
                convolver=dataset.convolver,
                w_tilde=dataset.w_tilde,
                settings_pixelization=settings_pixelization,
                settings_inversion=settings_inversion,
            )

            model_image = self.blurred_image + inversion.mapped_reconstructed_image

        fit = aa.FitData(
            data=image,
            noise_map=noise_map,
            model_data=model_image,
            mask=dataset.mask,
            inversion=inversion,
            use_mask_in_fit=False,
        )

        super().__init__(dataset=dataset, fit=fit)

    @property
    def galaxies(self):
        return self.plane.galaxies

    @property
    def grid(self):
        return self.imaging.grid

    @property
    def galaxy_model_image_dict(self) -> {Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """

        galaxy_model_image_dict = self.plane.galaxy_blurred_image_2d_dict_via_convolver_from(
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

        model_images_of_galaxies = self.plane.blurred_image_2d_list_via_psf_from(
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
        return self.plane.unmasked_blurred_image_2d_via_psf_from(
            grid=self.grid, psf=self.imaging.psf
        )

    @property
    def unmasked_blurred_image_of_galaxies(self):
        return self.plane.unmasked_blurred_image_2d_list_via_psf_from(
            grid=self.grid, psf=self.imaging.psf
        )

    @property
    def total_mappers(self):
        return 1


def hyper_image_from(image, hyper_image_sky):

    if hyper_image_sky is not None:
        return hyper_image_sky.hyper_image_from(image=image)
    else:
        return image


def hyper_noise_map_from(noise_map, plane, hyper_background_noise):

    hyper_noise_map = plane.hyper_noise_map_from(noise_map=noise_map)

    if hyper_background_noise is not None:
        noise_map = hyper_background_noise.hyper_noise_map_from(noise_map=noise_map)

    if hyper_noise_map is not None:
        noise_map = noise_map + hyper_noise_map
        noise_map_limit = conf.instance["general"]["hyper"]["hyper_noise_limit"]
        noise_map[noise_map > noise_map_limit] = noise_map_limit

    return noise_map
