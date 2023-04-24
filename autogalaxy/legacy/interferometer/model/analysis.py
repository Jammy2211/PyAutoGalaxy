import logging
from typing import Optional

import autofit as af

from autogalaxy.analysis.preloads import Preloads
from autogalaxy.legacy.interferometer.model.result import ResultInterferometer
from autogalaxy.legacy.interferometer.fit_interferometer import FitInterferometer
from autogalaxy.legacy.hyper_data import HyperBackgroundNoise
from autogalaxy.plane.plane import Plane

from autogalaxy.interferometer.model.analysis import (
    AnalysisInterferometer as AnalysisInterferometerBase,
)

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class AnalysisInterferometer(AnalysisInterferometerBase):
    def fit_interferometer_via_instance_from(
        self,
        instance: af.ModelInstance,
        use_hyper_scaling: bool = True,
        preload_overwrite: Optional[Preloads] = None,
    ) -> FitInterferometer:
        """
        Given a model instance create a `FitInterferometer` object.

        This function is used in the `log_likelihood_function` to fit the model to the interferometer data and compute
        the log likelihood.

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
        FitInterferometer
            The fit of the plane to the interferometer dataset, which includes the log likelihood.
        """
        instance = self.instance_with_associated_adapt_images_from(instance=instance)

        hyper_background_noise = self.hyper_background_noise_via_instance_from(
            instance=instance
        )

        plane = self.plane_via_instance_from(instance=instance)

        return self.fit_interferometer_via_plane_from(
            plane=plane,
            hyper_background_noise=hyper_background_noise,
            use_hyper_scaling=use_hyper_scaling,
        )

    def fit_interferometer_via_plane_from(
        self,
        plane: Plane,
        hyper_background_noise: Optional[HyperBackgroundNoise],
        use_hyper_scaling: bool = True,
        preload_overwrite: Optional[Preloads] = None,
    ) -> FitInterferometer:
        """
        Given a `Plane`, which the analysis constructs from a model instance, create a `FitInterferometer` object.

        This function is used in the `log_likelihood_function` to fit the model to the interferometer data and compute
        the log likelihood.

        Parameters
        ----------
        plane
            The plane of galaxies whose model images are used to fit the interferometer data.
        hyper_background_noise
            A model component which scales the background noise level of the data before computing the log likelihood.
        use_hyper_scaling
            If false, the scaling of the background noise is not performed irrespective of the model components
            themselves.

        Returns
        -------
        FitInterferometer
            The fit of the plane to the interferometer dataset, which includes the log likelihood.
        """

        preloads = self.preloads if preload_overwrite is None else preload_overwrite

        return FitInterferometer(
            dataset=self.dataset,
            plane=plane,
            hyper_background_noise=hyper_background_noise,
            use_hyper_scaling=use_hyper_scaling,
            settings_pixelization=self.settings_pixelization,
            settings_inversion=self.settings_inversion,
            preloads=preloads,
        )

    def make_result(
        self,
        samples: af.SamplesPDF,
        model: af.Collection,
        sigma=1.0,
        use_errors=True,
        use_widths=False,
    ) -> ResultInterferometer:
        """
        After the non-linear search is complete create its `Result`, which includes:

        - The samples of the non-linear search (E.g. MCMC chains, nested sampling samples) which are used to compute
          the maximum likelihood model, posteriors and other properties.

        - The model used to fit the data, which uses the samples to create specific instances of the model (e.g.
          an instance of the maximum log likelihood model).

        - The non-linear search used to perform the model fit.

        The `ResultInterferometer` object contains a number of methods which use the above objects to create the max
        log likelihood `Plane`, `FitInterferometer`, adapt-galaxy images,etc.

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
        ResultInterferometer
            The result of fitting the model to the interferometer dataset, via a non-linear search.
        """
        return ResultInterferometer(samples=samples, model=model, analysis=self)
