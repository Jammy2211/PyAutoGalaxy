from astropy import cosmology as cosmo
import json
import numpy as np
from typing import Optional, Union
from os import path

from autoconf import conf
import autofit as af
import autoarray as aa

from autogalaxy import exc
from autogalaxy.hyper.hyper_data import HyperImageSky
from autogalaxy.hyper.hyper_data import HyperBackgroundNoise
from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.plane.plane import Plane
from autogalaxy.analysis.result import ResultDataset


class Analysis(af.Analysis):
    def __init__(self, cosmology=cosmo.Planck15):
        """
        Analysis classes are used by PyAutoFit to fit a model to a dataset via a non-linear search.

        This abstract Analysis class for all model-fits which fit galaxies (or objects containing galaxies like a
        plane), but does not perform a model-fit by itself (and is therefore only inherited from).

        This class stores the Cosmology used for the analysis and hyper datasets used for certain model classes.

        Parameters
        ----------
        cosmology
            The AstroPy Cosmology assumed for this analysis.
        """
        self.cosmology = cosmology

    def plane_via_instance_from(self, instance: af.ModelInstance) -> Plane:
        """
        Create a `Plane` from the galaxies contained in a model instance.

        Parameters
        ----------
        instance
            An instance of the model that is fitted to the data by this analysis (whose parameters may have been set
            via a non-linear search).

        Returns
        -------
        An instance of the Plane class that is used to then fit the dataset.
        """
        if hasattr(instance, "clumps"):
            return Plane(galaxies=instance.galaxies + instance.clumps)
        return Plane(galaxies=instance.galaxies)


class AnalysisDataset(Analysis):
    def __init__(
        self,
        dataset: Union[aa.Imaging, aa.Interferometer],
        hyper_dataset_result: ResultDataset = None,
        cosmology=cosmo.Planck15,
        settings_pixelization: aa.SettingsPixelization = None,
        settings_inversion: aa.SettingsInversion = None,
    ):
        """
        Abstract Analysis class for all model-fits which fit galaxies (or objects containing galaxies like a plane)
        to a dataset, like imaging or interferometer data.

        This class stores the settings used to perform the model-fit for certain components of the model (e.g. a
        pixelization or inversion), the Cosmology used for the analysis and hyper datasets used for certain model
        classes.

        Parameters
        ----------
        dataset
            The dataset that is the model is fitted too.
        hyper_dataset_result
            The hyper-model image and hyper galaxies images of a previous result in a model-fitting pipeline, which are
            used by certain classes for adapting the analysis to the properties of the dataset.
        cosmology
            The Cosmology assumed for this analysis.
        settings_pixelization
            settings controlling how a pixelization is fitted during the model-fit, for example if a border is used
            when creating the pixelization.
        settings_inversion
            Settings controlling how an inversion is fitted during the model-fit, for example which linear algebra
            formalism is used.
        """
        super().__init__(cosmology=cosmology)

        self.dataset = dataset
        self.hyper_dataset_result = hyper_dataset_result

        if self.hyper_dataset_result is not None:

            if hyper_dataset_result.search is not None:
                hyper_dataset_result.search.paths = None

            self.set_hyper_dataset(result=self.hyper_dataset_result)

        else:

            self.hyper_galaxy_image_path_dict = None
            self.hyper_model_image = None

        self.settings_pixelization = settings_pixelization or aa.SettingsPixelization()
        self.settings_inversion = settings_inversion or aa.SettingsInversion()

        self.preloads = aa.Preloads()

    def set_hyper_dataset(self, result: ResultDataset) -> None:
        """
        Using a the result of a previous model-fit, set the hyper-dataset for this analysis. This is used to adapt
        aspects of the model (e.g. the pixelization, regularization scheme) to the properties of the dataset being
        fitted.

        This passes the hyper model image and hyper galaxy images of the previous fit. These represent where different
        galaxies in the dataset are located and thus allows the fit to adapt different aspects of the model to different
        galaxies in the data.

        Parameters
        ----------
        result
            The result of a previous model-fit which contains the model image and model galaxy images of a fit to
            the dataset, which set up the hyper dataset. These are used by certain classes for adapting the analysis
            to the properties of the dataset.
        """
        hyper_galaxy_image_path_dict = result.hyper_galaxy_image_path_dict
        hyper_model_image = result.hyper_model_image

        self.hyper_galaxy_image_path_dict = hyper_galaxy_image_path_dict
        self.hyper_model_image = hyper_model_image

    def hyper_image_sky_via_instance_from(
        self, instance: af.ModelInstance
    ) -> Optional[HyperImageSky]:
        """
        If the model instance contains a `HyperImageSky` attribute, which adds a free parameter to the model that
        scales the background sky, return this attribute. Otherwise a None is returned.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        An instance of the hyper image sky class that scales the sky background.
        """
        if hasattr(instance, "hyper_image_sky"):
            return instance.hyper_image_sky

    def hyper_background_noise_via_instance_from(
        self, instance: af.ModelInstance
    ) -> Optional[HyperBackgroundNoise]:
        """
        If the model instance contains a `HyperBackgroundNoise` attribute, which adds a free parameter to the model that
        scales the background noise, return this attribute. Otherwise a None is returned.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        An instance of the hyper background noise class that scales the background noise.
        """
        if hasattr(instance, "hyper_background_noise"):
            return instance.hyper_background_noise

    def instance_with_associated_hyper_images_from(
        self, instance: af.ModelInstance
    ) -> af.ModelInstance:
        """
        Using the model image and galaxy images that were set up as the hyper dataset, associate the galaxy images
        of that result with the galaxies in this model fit.

        Association is performed based on galaxy names, whereby if the name of a galaxy in this search matches the
        full-path name of galaxies in the hyper dataset the galaxy image is passed.

        If the galaxy collection has a different name then an association is not made.

        For example, `galaxies.lens` will match with:
            `galaxies.lens`
        but not with:
            `galaxies.source`

        Parameters
        ----------
        instance
        An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search), which has 0 or more galaxies in its tree.

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
        """
        Before the model-fit via the non-linear search begins, this routine saves attributes of the `Analysis` object
        to the `pickles` folder such that they can be load after the analysis using PyAutoFit's database and aggregator
        tools.

        For this analysis the following are output:

        - The dataset's data.
        - The dataset's noise-map.
        - The settings associated with the dataset.
        - The settings associated with the inversion.
        - The settings associated with the pixelization.
        - The Cosmology.
        - The hyper dataset's model image and galaxy images, if used.

        It is common for these attributes to be loaded by many of the template aggregator functions given in the
        `aggregator` modules. For example, when using the database tools to reperform a fit, this will by default
        load the dataset, settings and other attributes necessary to perform a fit using the attributes output by
        this function.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored, visualization,
            and the pickled objects used by the aggregator output by this function.
        """
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

    def output_or_check_figure_of_merit_sanity(
        self, paths: af.DirectoryPaths, result: af.Result
    ):
        """
        Changes to the PyAutoGalaxy source code may inadvertantly change the numerics of how a log likelihood is
        computed. Equally, one may set off a model-fit that resumes from previous results, but change the settings of
        the pixelization or inversion in a way that changes the log likelihood function.

        This function performs an optional sanity check, which raises an exception if the log likelihood calculation
        changes, to ensure a model-fit is not resumed with a different likelihood calculation to the previous run.

        If the model-fit has not been performed before (e.g. it is not a resume) this function outputs
        the `figure_of_merit` (e.g. the log likelihood) of the maximum log likelihood model at the end of the model-fit.

        If the model-fit is a resume, it loads this `figure_of_merit` and compares it against a new value computed for
        the resumed run (again using the maximum log likelihood model inferred). If the two likelihoods do not agree
        and therefore the log likelihood function has changed, an exception is raised and the code execution terminated.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored, visualization,
            and pickled objects used by the database and aggregator.
        result
            The result containing the maximum log likelihood fit of the model.
        """
        figure_of_merit = result.max_log_likelihood_fit.figure_of_merit

        figure_of_merit_sanity_file = path.join(
            paths.output_path, "figure_of_merit_sanity.json"
        )

        if not path.exists(figure_of_merit_sanity_file):

            with open(figure_of_merit_sanity_file, "w+") as f:
                json.dump(figure_of_merit, f)

        else:

            with open(figure_of_merit_sanity_file) as json_file:
                figure_of_merit_sanity = json.load(json_file)

            if conf.instance["general"]["test"]["check_figure_of_merit_sanity"]:

                if not np.isclose(figure_of_merit, figure_of_merit_sanity):

                    raise exc.AnalysisException(
                        "Figure of merit sanity check failed. "
                        ""
                        "This means that the existing results of a model fit used a different "
                        "likelihood function compared to the one implemented now.\n\n"
                        f"Old Figure of Merit = {figure_of_merit_sanity}\n"
                        f"New Figure of Merit = {figure_of_merit}"
                    )
