from typing import Dict, Optional

import autofit as af

from autogalaxy.analysis.preloads import Preloads
from autogalaxy.legacy.hyper_data import HyperImageSky
from autogalaxy.legacy.hyper_data import HyperBackgroundNoise
from autogalaxy.legacy.imaging.fit_imaging import FitImaging
from autogalaxy.legacy.imaging.model.result import ResultImaging
from autogalaxy.legacy.plane import Plane

from autogalaxy.imaging.model.analysis import AnalysisImaging as AnalysisImagingBase


class AnalysisImaging(AnalysisImagingBase):
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

    def fit_imaging_via_instance_from(
        self,
        instance: af.ModelInstance,
        use_hyper_scaling: bool = True,
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
        use_hyper_scaling
            If false, the scaling of the background sky and noise are not performed irrespective of the model components
            themselves.
        preload_overwrite
            If a `Preload` object is input this is used instead of the preloads stored as an attribute in the analysis.
        profiling_dict
            A dictionary which times functions called to fit the model to data, for profiling.

        Returns
        -------
        FitImaging
            The fit of the plane to the imaging dataset, which includes the log likelihood.
        """
        instance = self.instance_with_associated_hyper_images_from(instance=instance)

        hyper_image_sky = self.hyper_image_sky_via_instance_from(instance=instance)

        hyper_background_noise = self.hyper_background_noise_via_instance_from(
            instance=instance
        )

        plane = self.plane_via_instance_from(instance=instance)

        return self.fit_imaging_via_plane_from(
            plane=plane,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            use_hyper_scaling=use_hyper_scaling,
            preload_overwrite=preload_overwrite,
            profiling_dict=profiling_dict,
        )

    def fit_imaging_via_plane_from(
        self,
        plane: Plane,
        hyper_image_sky: Optional[HyperImageSky],
        hyper_background_noise: Optional[HyperBackgroundNoise],
        use_hyper_scaling: bool = True,
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
        hyper_image_sky
            A model component which scales the background sky level of the data before computing the log likelihood.
        hyper_background_noise
            A model component which scales the background noise level of the data before computing the log likelihood.
        use_hyper_scaling
            If false, the scaling of the background sky and noise are not performed irrespective of the model components
            themselves.
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
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            use_hyper_scaling=use_hyper_scaling,
            settings_pixelization=self.settings_pixelization,
            settings_inversion=self.settings_inversion,
            preloads=preloads,
            profiling_dict=profiling_dict,
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
        log likelihood `Plane`, `FitImaging`, hyper-galaxy images,etc.

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
