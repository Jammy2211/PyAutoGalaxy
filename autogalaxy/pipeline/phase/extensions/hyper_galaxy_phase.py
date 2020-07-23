import copy

import autofit as af
import numpy as np
from autoarray.fit import fit as aa_fit
from autogalaxy.fit import fit
from autogalaxy.galaxy import galaxy as g
from autogalaxy.hyper import hyper_data as hd
from autogalaxy.pipeline import visualizer

from .hyper_phase import HyperPhase


class Analysis(af.Analysis):
    def __init__(
        self, masked_imaging, hyper_model_image, hyper_galaxy_image, image_path
    ):
        """
        An analysis to fit the noise for a single galaxy image.
        Parameters
        ----------
        masked_imaging: LensData
            lens dataset, including an image and noise
        hyper_model_image: ndarray
            An image produce of the overall system by a model
        hyper_galaxy_image: ndarray
            The contribution of one galaxy to the model image
        """

        super().__init__()

        self.masked_imaging = masked_imaging

        self.visualizer = visualizer.HyperGalaxyVisualizer(image_path=image_path)

        self.hyper_model_image = hyper_model_image
        self.hyper_galaxy_image = hyper_galaxy_image

    def visualize(self, instance, during_analysis):

        if self.visualizer.plot_hyper_galaxy_subplot:
            hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)

            hyper_background_noise = self.hyper_background_noise_for_instance(
                instance=instance
            )

            contribution_map = instance.hyper_galaxy.contribution_map_from_hyper_images(
                hyper_model_image=self.hyper_model_image,
                hyper_galaxy_image=self.hyper_galaxy_image,
            )

            fit_normal = aa_fit.FitImaging(
                masked_imaging=self.masked_imaging, model_image=self.hyper_model_image
            )

            fit_hyper = self.fit_for_hyper_galaxy(
                hyper_galaxy=instance.hyper_galaxy,
                hyper_image_sky=hyper_image_sky,
                hyper_background_noise=hyper_background_noise,
            )

            self.visualizer.visualize_hyper_galaxy(
                fit=fit_normal,
                hyper_fit=fit_hyper,
                galaxy_image=self.hyper_galaxy_image,
                contribution_map_in=contribution_map,
            )

    def log_likelihood_function(self, instance):
        """
        Fit the model image to the real image by scaling the hyper_galaxies noise.
        Parameters
        ----------
        instance: ModelInstance
            A model instance with a hyper_galaxies galaxy property
        Returns
        -------
        fit: float
        """

        hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)

        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        fit = self.fit_for_hyper_galaxy(
            hyper_galaxy=instance.hyper_galaxy,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        return fit.figure_of_merit

    @staticmethod
    def hyper_image_sky_for_instance(instance):
        if hasattr(instance, "hyper_image_sky"):
            return instance.hyper_image_sky

    @staticmethod
    def hyper_background_noise_for_instance(instance):
        if hasattr(instance, "hyper_background_noise"):
            return instance.hyper_background_noise

    def fit_for_hyper_galaxy(
        self, hyper_galaxy, hyper_image_sky, hyper_background_noise
    ):

        image = fit.hyper_image_from_image_and_hyper_image_sky(
            image=self.masked_imaging.image, hyper_image_sky=hyper_image_sky
        )

        if hyper_background_noise is not None:
            noise_map = hyper_background_noise.hyper_noise_map_from_noise_map(
                noise_map=self.masked_imaging.noise_map
            )
        else:
            noise_map = self.masked_imaging.noise_map

        hyper_noise_map = hyper_galaxy.hyper_noise_map_from_hyper_images_and_noise_map(
            hyper_model_image=self.hyper_model_image,
            hyper_galaxy_image=self.hyper_galaxy_image,
            noise_map=self.masked_imaging.noise_map,
        )

        noise_map = noise_map + hyper_noise_map

        masked_imaging = self.masked_imaging.modify_image_and_noise_map(
            image=image, noise_map=noise_map
        )

        return aa_fit.FitImaging(
            masked_imaging=masked_imaging, model_image=self.hyper_model_image
        )

    @classmethod
    def describe(cls, instance):
        return "Running hyper_galaxies galaxy fit for HyperGalaxy:\n{}".format(
            instance.hyper_galaxy
        )


class HyperGalaxyPhase(HyperPhase):
    Analysis = Analysis

    def __init__(self, phase, search, include_sky_background, include_noise_background):

        super().__init__(phase=phase, search=search, hyper_name="hyper_galaxy")
        self.include_sky_background = include_sky_background
        self.include_noise_background = include_noise_background

    def run_hyper(
        self, dataset, results: af.ResultsCollection, info=None, pickle_files=None
    ):
        """
        Run a fit for each galaxy from the previous phase.
        Parameters
        ----------
        dataset: LensData
        results: ResultsCollection
            Results from all previous phases
        Returns
        -------
        results: HyperGalaxyResults
            A collection of results, with one item per a galaxy
        """

        self.results = results

        phase = self.make_hyper_phase()

        masked_imaging = results.last.masked_dataset

        hyper_result = copy.deepcopy(results.last)
        hyper_result.model = hyper_result.model.copy_with_fixed_priors(
            hyper_result.instance
        )

        hyper_result.analysis.hyper_model_image = results.last.hyper_model_image
        hyper_result.analysis.hyper_galaxy_image_path_dict = (
            results.last.hyper_galaxy_image_path_dict
        )

        for path, galaxy in results.last.path_galaxy_tuples:

            model = copy.deepcopy(phase.model)

            search = self.search.copy_with_name_extension(extension=path[-1])

            # TODO : This is a HACK :O

            model.galaxies = []

            model.hyper_galaxy = g.HyperGalaxy

            if self.include_sky_background:
                model.hyper_image_sky = hd.HyperImageSky

            if self.include_noise_background:
                model.hyper_background_noise = hd.HyperBackgroundNoise

            # If arrays is all zeros, galaxy did not have image in previous phase and
            # shoumasked_imaging be ignored
            if not np.all(
                hyper_result.analysis.hyper_galaxy_image_path_dict[path] == 0
            ):
                hyper_model_image = hyper_result.analysis.hyper_model_image

                analysis = self.Analysis(
                    masked_imaging=masked_imaging,
                    hyper_model_image=hyper_model_image,
                    hyper_galaxy_image=hyper_result.analysis.hyper_galaxy_image_path_dict[
                        path
                    ],
                    image_path=search.paths.image_path,
                )

                result = search.fit(model=model, analysis=analysis)

                def transfer_field(name):
                    if hasattr(result._instance, name):
                        setattr(
                            hyper_result.instance.object_for_path(path),
                            name,
                            getattr(result._instance, name),
                        )
                        setattr(
                            hyper_result.model.object_for_path(path),
                            name,
                            getattr(result.model, name),
                        )

                transfer_field("hyper_galaxy")

                hyper_result.instance.hyper_image_sky = getattr(
                    result._instance, "hyper_image_sky"
                )
                hyper_result.model.hyper_image_sky = getattr(
                    result.model, "hyper_image_sky"
                )

                hyper_result.instance.hyper_background_noise = getattr(
                    result._instance, "hyper_background_noise"
                )
                hyper_result.model.hyper_background_noise = getattr(
                    result.model, "hyper_background_noise"
                )

        return hyper_result
