import numpy as np
import os
from typing import Dict, Optional

import autofit as af
import autoarray as aa

from autoarray.exc import PixelizationException

from autogalaxy.analysis.analysis import AnalysisDataset
from autogalaxy.analysis.preloads import Preloads
from autogalaxy.cosmology.lensing import LensingCosmology
from autogalaxy.cosmology.wrap import Planck15
from autogalaxy.imaging.model.visualizer import VisualizerImaging
from autogalaxy.imaging.model.result import ResultImaging
from autogalaxy.imaging.fit_imaging import FitImaging
from autogalaxy.plane.plane import Plane

from autogalaxy import exc


class AnalysisImaging(AnalysisDataset):
    def __init__(
        self,
        dataset: aa.Imaging,
        adapt_result: ResultImaging = None,
        cosmology: LensingCosmology = Planck15(),
        settings_pixelization: aa.SettingsPixelization = None,
        settings_inversion: aa.SettingsInversion = None,
    ):
        """
        Analysis classes are used by PyAutoFit to fit a model to a dataset via a non-linear search.

        An Analysis class defines the `log_likelihood_function` which fits the model to the dataset and returns the
        log likelihood value defining how well the model fitted the data. The Analysis class handles many other tasks,
        such as visualization, outputting results to hard-disk and storing results in a format that can be loaded after
        the model-fit is complete using PyAutoFit's database tools.

        This Analysis class is used for model-fits which fit galaxies (or objects containing galaxies like a `Plane`)
        to an imaging dataset.

        This class stores the settings used to perform the model-fit for certain components of the model (e.g. a
        pixelization or inversion), the Cosmology used for the analysis and adapt datasets used for certain model
        classes.

        Parameters
        ----------
        dataset
            The `Imaging` dataset that the model is fitted too.
        adapt_result
            The adapt-model image and galaxies images of a previous result in a model-fitting pipeline, which are
            used by certain classes for adapting the analysis to the properties of the dataset.
        cosmology
            The Cosmology assumed for this analysis.
        settings_pixelization
            Settings controlling how a pixelization is fitted for example if a border is used when creating the
            pixelization.
        settings_inversion
            Settings controlling how an inversion is fitted for example which linear algebra formalism is used.
        """
        super().__init__(
            dataset=dataset,
            adapt_result=adapt_result,
            cosmology=cosmology,
            settings_pixelization=settings_pixelization,
            settings_inversion=settings_inversion,
        )

    @property
    def imaging(self):
        return self.dataset

    def modify_before_fit(self, paths: af.DirectoryPaths, model: af.Collection):
        """
        PyAutoFit calls this function immediately before the non-linear search begins, therefore it can be used to
        perform tasks using the final model parameterization.

        This function checks that the adapt-dataset is consistent with previous adapt-datasets if the model-fit is
        being resumed from a previous run, and it visualizes objects which do not change throughout the model fit
        like the dataset.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The PyAutoFit model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        """

        super().modify_before_fit(paths=paths, model=model)

        if not paths.is_complete:
            self.set_preloads(paths=paths, model=model)

        return self

    def log_likelihood_function(self, instance: af.ModelInstance) -> float:
        """
        Given an instance of the model, where the model parameters are set via a non-linear search, fit the model
        instance to the imaging dataset.

        This function returns a log likelihood which is used by the non-linear search to guide the model-fit.

        For this analysis class, this function performs the following steps:

        1) If the analysis has a adapt dataset, associated the model galaxy images of this dataset to the galaxies in
           the model instance.

        2) Extract attributes which model aspects of the data reductions, like the scaling the background sky
           and background noise.

        3) Extracts all galaxies from the model instance and set up a `Plane`.

        4) Use the `Plane` and other attributes to create a `FitImaging` object, which performs steps such as creating
           model images of every galaxy in the plane, blurring them with the imaging dataset's PSF and computing residuals,
           a chi-squared statistic and the log likelihood.

        Certain models will fail to fit the dataset and raise an exception. For example if an `Inversion` is used, the
        linear algebra calculation may be invalid and raise an Exception. In such circumstances the model is discarded
        and its likelihood value is passed to the non-linear search in a way that it ignores it (for example, using a
        value of -1.0e99).

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        float
            The log likelihood indicating how well this model instance fitted the imaging data.
        """

        try:
            return self.fit_imaging_via_instance_from(instance=instance).figure_of_merit
        except (
            PixelizationException,
            exc.PixelizationException,
            exc.InversionException,
            exc.GridException,
            ValueError,
            np.linalg.LinAlgError,
            OverflowError,
        ) as e:
            raise exc.FitException from e

    def fit_imaging_via_instance_from(
        self,
        instance: af.ModelInstance,
        preload_overwrite: Optional[Preloads] = None,
        profiling_dict: Optional[Dict] = None,
    ) -> FitImaging:
        """
        Given a model instance create a `FitImaging` object.

        This function is used in the `log_likelihood_function` to fit the model to the imaging data and compute the
        log likelihood.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).
        preload_overwrite
            If a `Preload` object is input this is used instead of the preloads stored as an attribute in the analysis.
        profiling_dict
            A dictionary which times functions called to fit the model to data, for profiling.

        Returns
        -------
        FitImaging
            The fit of the plane to the imaging dataset, which includes the log likelihood.
        """
        instance = self.instance_with_associated_adapt_images_from(instance=instance)

        plane = self.plane_via_instance_from(instance=instance)

        return self.fit_imaging_via_plane_from(
            plane=plane,
            preload_overwrite=preload_overwrite,
            profiling_dict=profiling_dict,
        )

    def fit_imaging_via_plane_from(
        self,
        plane: Plane,
        preload_overwrite: Optional[Preloads] = None,
        profiling_dict: Optional[Dict] = None,
    ) -> FitImaging:
        """
        Given a `Plane`, which the analysis constructs from a model instance, create a `FitImaging` object.

        This function is used in the `log_likelihood_function` to fit the model to the imaging data and compute the
        log likelihood.

        Parameters
        ----------
        plane
            The plane of galaxies whose model images are used to fit the imaging data.
        preload_overwrite
            If a `Preload` object is input this is used instead of the preloads stored as an attribute in the analysis.
        profiling_dict
            A dictionary which times functions called to fit the model to data, for profiling.

        Returns
        -------
        FitImaging
            The fit of the plane to the imaging dataset, which includes the log likelihood.
        """

        preloads = self.preloads if preload_overwrite is None else preload_overwrite

        return FitImaging(
            dataset=self.dataset,
            plane=plane,
            settings_pixelization=self.settings_pixelization,
            settings_inversion=self.settings_inversion,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )

    @property
    def fit_func(self):
        return self.fit_imaging_via_instance_from

    def visualize_before_fit(self, paths: af.DirectoryPaths, model: af.Collection):
        """
        PyAutoFit calls this function immediately before the non-linear search begins.

        It visualizes objects which do not change throughout the model fit like the dataset.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The PyAutoFit model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        """
        if not self.should_visualize(paths=paths):
            return

        visualizer = VisualizerImaging(visualize_path=paths.image_path)

        visualizer.visualize_imaging(dataset=self.dataset)

        visualizer.visualize_adapt_images(
            adapt_galaxy_image_path_dict=self.adapt_galaxy_image_path_dict,
            adapt_model_image=self.adapt_model_image,
        )

    def visualize(
        self,
        paths: af.DirectoryPaths,
        instance: af.ModelInstance,
        during_analysis: bool,
    ):
        """
        Output images of the maximum log likelihood model inferred by the model-fit. This function is called throughout
        the non-linear search at regular intervals, and therefore provides on-the-fly visualization of how well the
        model-fit is going.

        The visualization performed by this function includes:

        - Images of the best-fit `Plane`, including the images of each of its galaxies.

        - Images of the best-fit `FitImaging`, including the model-image, residuals and chi-squared of its fit to
          the imaging data.

        - The adapt-images of the model-fit showing how the galaxies are used to represent different galaxies in
          the dataset.

        The images output by this function are customized using the file `config/visualize/plots.ini`.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization, and the pickled objects used by the aggregator output by this function.
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).
        during_analysis
            If True the visualization is being performed midway through the non-linear search before it is finished,
            which may change which images are output.
        """

        if not self.should_visualize(paths=paths):
            return

        instance = self.instance_with_associated_adapt_images_from(instance=instance)
        plane = self.plane_via_instance_from(instance=instance)

        fit = self.fit_imaging_via_plane_from(
            plane=plane,
        )

        visualizer = VisualizerImaging(visualize_path=paths.image_path)
        visualizer.visualize_imaging(dataset=self.dataset)

        try:
            visualizer.visualize_fit_imaging(fit=fit, during_analysis=during_analysis)
        except exc.InversionException:
            pass

        plane = fit.plane_linear_light_profiles_to_light_profiles

        visualizer.visualize_plane(
            plane=plane, grid=fit.grid, during_analysis=during_analysis
        )
        visualizer.visualize_galaxies(
            galaxies=plane.galaxies, grid=fit.grid, during_analysis=during_analysis
        )
        if fit.inversion is not None:
            visualizer.visualize_inversion(
                inversion=fit.inversion, during_analysis=during_analysis
            )

        visualizer.visualize_adapt_images(
            adapt_galaxy_image_path_dict=self.adapt_galaxy_image_path_dict,
            adapt_model_image=self.adapt_model_image,
        )

    def make_result(
        self,
        samples: af.SamplesPDF,
        model: af.Collection,
        sigma=1.0,
        use_errors=True,
        use_widths=False,
    ) -> ResultImaging:
        """
        After the non-linear search is complete create its `Result`, which includes:

        - The samples of the non-linear search (E.g. MCMC chains, nested sampling samples) which are used to compute
          the maximum likelihood model, posteriors and other properties.

        - The model used to fit the data, which uses the samples to create specific instances of the model (e.g.
          an instance of the maximum log likelihood model).

        - The non-linear search used to perform the model fit.

        The `ResultImaging` object contains a number of methods which use the above objects to create the max
        log likelihood `Plane`, `FitImaging`, adapt-galaxy images,etc.

        Parameters
        ----------
        samples
            A PyAutoFit object which contains the samples of the non-linear search, for example the chains of an MCMC
            run of samples of the nested sampler.
        model
            The PyAutoFit model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        search
            The non-linear search used to perform this model-fit.

        Returns
        -------
        ResultImaging
            The result of fitting the model to the imaging dataset, via a non-linear search.
        """
        return ResultImaging(samples=samples, model=model, analysis=self)

    def save_attributes_for_aggregator(self, paths: af.DirectoryPaths):
        """
         Before the non-linear search begins, this routine saves attributes of the `Analysis` object to the `pickles`
         folder such that they can be loaded after the analysis using PyAutoFit's database and aggregator tools.

         For this analysis, it uses the `AnalysisDataset` object's method to output the following:

         - The dataset's data.
         - The dataset's noise-map.
         - The settings associated with the dataset.
         - The settings associated with the inversion.
         - The settings associated with the pixelization.
         - The Cosmology.
         - The adapt dataset's model image and galaxy images, if used.

         This function also outputs attributes specific to an imaging dataset:

        - Its PSF.
        - Its mask.

         It is common for these attributes to be loaded by many of the template aggregator functions given in the
         `aggregator` modules. For example, when using the database tools to perform a fit, the default behaviour is for
         the dataset, settings and other attributes necessary to perform the fit to be loaded via the pickle files
         output by this function.

         Parameters
         ----------
         paths
             The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored, visualization,
             and the pickled objects used by the aggregator output by this function.
        """
        super().save_attributes_for_aggregator(paths=paths)

        paths.save_object("psf", self.dataset.psf)
        paths.save_object("mask", self.dataset.mask)
