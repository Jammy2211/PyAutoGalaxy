import autofit as af
from autoarray.structures.arrays.two_d import array_2d
from autoarray.structures import visibilities
from autoconf import conf
import numpy as np
from autogalaxy.galaxy import galaxy as g


class Result(af.Result):
    def __init__(self, samples, model, analysis, search):
        """
        The results of a `NonLinearSearch` performed by a search.

        Parameters
        ----------
        samples : af.Samples
            A class containing the samples of the non-linear search, including methods to get the maximum log
            likelihood model, errors, etc.
        model : af.ModelMapper
            The model used in this result model-fit.
        analysis : Analysis
            The Analysis class used by this model-fit to fit the model to the data.
        search : af.NonLinearSearch
            The `NonLinearSearch` search used by this model fit.
        """
        super().__init__(samples=samples, model=model, search=search)

        self.analysis = analysis

    @property
    def max_log_likelihood_plane(self):

        instance = self.analysis.associate_hyper_images(instance=self.instance)

        return self.analysis.plane_for_instance(instance=instance)

    @property
    def path_galaxy_tuples(self) -> [(str, g.Galaxy)]:
        """
        Tuples associating the names of galaxies with instances from the best fit
        """
        return self.instance.path_instance_tuples_for_class(cls=g.Galaxy)


class ResultDataset(Result):
    @property
    def max_log_likelihood_fit(self):

        hyper_image_sky = self.analysis.hyper_image_sky_for_instance(
            instance=self.instance
        )

        hyper_background_noise = self.analysis.hyper_background_noise_for_instance(
            instance=self.instance
        )

        return self.analysis.fit_imaging_for_plane(
            plane=self.max_log_likelihood_plane,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

    @property
    def mask(self):
        return self.max_log_likelihood_fit.mask

    @property
    def dataset(self):
        return self.max_log_likelihood_fit.dataset

    @property
    def pixelization(self):
        for galaxy in self.max_log_likelihood_fit.galaxies:
            if galaxy.pixelization is not None:
                return galaxy.pixelization

    def image_for_galaxy(self, galaxy: g.Galaxy) -> np.ndarray:
        """
        Parameters
        ----------
        galaxy
            A galaxy used in this search

        Returns
        -------
        ndarray or None
            A numpy arrays giving the model image of that galaxy
        """
        return self.max_log_likelihood_fit.galaxy_model_image_dict[galaxy]

    @property
    def image_galaxy_dict(self) -> {str: g.Galaxy}:
        """
        A dictionary associating galaxy names with model images of those galaxies
        """
        return {
            galaxy_path: self.image_for_galaxy(galaxy)
            for galaxy_path, galaxy in self.path_galaxy_tuples
        }

    @property
    def hyper_galaxy_image_path_dict(self):
        """
        A dictionary associating 1D hyper_galaxies galaxy images with their names.
        """

        hyper_minimum_percent = conf.instance["general"]["hyper"][
            "hyper_minimum_percent"
        ]

        hyper_galaxy_image_path_dict = {}

        for path, galaxy in self.path_galaxy_tuples:

            galaxy_image = self.image_galaxy_dict[path]

            if not np.all(galaxy_image == 0):
                minimum_galaxy_value = hyper_minimum_percent * max(galaxy_image)
                galaxy_image[galaxy_image < minimum_galaxy_value] = minimum_galaxy_value

            hyper_galaxy_image_path_dict[path] = galaxy_image

        return hyper_galaxy_image_path_dict

    @property
    def hyper_model_image(self):

        hyper_model_image = array_2d.Array2D.manual_mask(
            array=np.zeros(self.mask.mask_sub_1.pixels_in_mask),
            mask=self.mask.mask_sub_1,
        )

        for path, galaxy in self.path_galaxy_tuples:
            hyper_model_image += self.hyper_galaxy_image_path_dict[path]

        return hyper_model_image


class ResultImaging(ResultDataset):
    @property
    def max_log_likelihood_fit(self):

        hyper_image_sky = self.analysis.hyper_image_sky_for_instance(
            instance=self.instance
        )

        hyper_background_noise = self.analysis.hyper_background_noise_for_instance(
            instance=self.instance
        )

        return self.analysis.fit_imaging_for_plane(
            plane=self.max_log_likelihood_plane,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

    @property
    def unmasked_model_image(self):
        return self.max_log_likelihood_fit.unmasked_blurred_image

    @property
    def unmasked_model_image_of_galaxies(self):
        return self.max_log_likelihood_fit.unmasked_blurred_image_of_galaxies


class ResultInterferometer(ResultDataset):
    @property
    def max_log_likelihood_fit(self):

        hyper_background_noise = self.analysis.hyper_background_noise_for_instance(
            instance=self.instance
        )

        return self.analysis.fit_interferometer_for_tracer(
            plane=self.max_log_likelihood_tracer,
            hyper_background_noise=hyper_background_noise,
        )

    @property
    def real_space_mask(self):
        return self.max_log_likelihood_fit.interferometer.real_space_mask

    @property
    def unmasked_model_visibilities(self):
        return self.max_log_likelihood_fit.unmasked_blurred_image

    @property
    def unmasked_model_visibilities_of_galaxies(self):
        return self.max_log_likelihood_fit.unmasked_blurred_image_of_galaxies

    def visibilities_for_galaxy(self, galaxy: g.Galaxy) -> np.ndarray:
        """
        Parameters
        ----------
        galaxy
            A galaxy used in this search

        Returns
        -------
        ndarray or None
            A numpy arrays giving the model visibilities of that galaxy
        """
        return self.max_log_likelihood_fit.galaxy_model_visibilities_dict[galaxy]

    @property
    def visibilities_galaxy_dict(self) -> {str: g.Galaxy}:
        """
        A dictionary associating galaxy names with model visibilities of those galaxies
        """
        return {
            galaxy_path: self.visibilities_for_galaxy(galaxy)
            for galaxy_path, galaxy in self.path_galaxy_tuples
        }

    @property
    def hyper_galaxy_visibilities_path_dict(self):
        """
        A dictionary associating 1D hyper_galaxies galaxy visibilities with their names.
        """

        hyper_galaxy_visibilities_path_dict = {}

        for path, galaxy in self.path_galaxy_tuples:
            hyper_galaxy_visibilities_path_dict[path] = self.visibilities_galaxy_dict[
                path
            ]

        return hyper_galaxy_visibilities_path_dict

    @property
    def hyper_model_visibilities(self):

        hyper_model_visibilities = visibilities.Visibilities.zeros(
            shape_slim=(self.max_log_likelihood_fit.visibilities.shape_slim,)
        )

        for path, galaxy in self.path_galaxy_tuples:
            hyper_model_visibilities += self.hyper_galaxy_visibilities_path_dict[path]

        return hyper_model_visibilities
