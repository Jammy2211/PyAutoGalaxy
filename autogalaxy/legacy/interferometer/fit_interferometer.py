import numpy as np
from typing import Dict, List, Optional

from autoconf import cached_property

import autoarray as aa

from autogalaxy.abstract_fit import AbstractFitInversion
from autogalaxy.analysis.preloads import Preloads
from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.legacy.hyper_data import HyperBackgroundNoise
from autogalaxy.plane.plane import Plane
from autogalaxy.plane.to_inversion import PlaneToInversion
from autogalaxy.profiles.light.linear import LightProfileLinear

from autogalaxy.interferometer.fit_interferometer import (
    FitInterferometer as FitInterferometerBase,
)


class FitInterferometer(FitInterferometerBase):
    def __init__(
        self,
        dataset: aa.Interferometer,
        plane: Plane,
        hyper_background_noise: HyperBackgroundNoise = None,
        use_hyper_scaling: bool = True,
        settings_pixelization: aa.SettingsPixelization = aa.SettingsPixelization(),
        settings_inversion: aa.SettingsInversion = aa.SettingsInversion(),
        preloads: aa.Preloads = Preloads(),
        profiling_dict: Optional[Dict] = None,
    ):
        """
        Fits an interferometer dataset using a `Plane` object.

        The fit performs the following steps:

        1) Compute the sum of all images of galaxy light profiles in the `Plane`.

        2) Fourier transform this image with the transformer object and `uv_wavelengths` to create
           the `profile_visibilities`.

        3) Subtract these visibilities from the `data` to create the `profile_subtracted_visibilities`.

        4) If the `Plane` has any linear algebra objects (e.g. linear light profiles, a pixelization / regulariation)
           fit the `profile_subtracted_visibilities` with these objects via an inversion.

        5) Compute the `model_data` as the sum of the `profile_visibilities` and `reconstructed_data` of the inversion
           (if an inversion is not performed the `model_data` is only the `profile_visibilities`.

        6) Subtract the `model_data` from the data and compute the residuals, chi-squared and likelihood via the
           noise-map (if an inversion is performed the `log_evidence`, including addition terms describing the linear
           algebra solution, is computed).

        When performing a model-fit` via ` AnalysisInterferometer` object the `figure_of_merit` of
        this `FitInterferometer` object is called and returned in the `log_likelihood_function`.

        Parameters
        ----------
        dataset
            The interfometer dataset which is fitted by the galaxies in the plane.
        plane
            The plane of galaxies whose light profile images are used to fit the interferometer data.
        hyper_background_noise
            If included, adds a noise-scaling term to the background noise.
        use_hyper_scaling
            If set to False, the hyper scaling functions (e.g. the `hyper_background_noise`) are
            omitted irrespective of their inputs.
        settings_pixelization
            Settings controlling how a pixelization is fitted for example if a border is used when creating the
            pixelization.
        settings_inversion
            Settings controlling how an inversion is fitted for example which linear algebra formalism is used.
        preloads
            Contains preloaded calculations (e.g. linear algebra matrices) which can skip certain calculations in
            the fit.
        profiling_dict
            A dictionary which if passed to the fit records how long fucntion calls which have the `profile_func`
            decorator take to run.
        """

        super().__init__(
            dataset=dataset,
            plane=plane,
            settings_pixelization=settings_pixelization,
            settings_inversion=settings_inversion,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )

        self.hyper_background_noise = hyper_background_noise

        self.use_hyper_scaling = use_hyper_scaling

    @property
    def noise_map(self) -> aa.VisibilitiesNoiseMap:
        """
        Returns the interferometer's noise-map, which may have a hyper scaling performed which increase the noise in
        regions of the data that are poorly fitted in order to avoid overfitting.
        """
        if self.use_hyper_scaling and self.hyper_background_noise is not None:
            return self.hyper_background_noise.hyper_noise_map_complex_from(
                noise_map=self.dataset.noise_map
            )

        return self.dataset.noise_map

    def refit_with_new_preloads(
        self,
        preloads: Preloads,
        settings_inversion: Optional[aa.SettingsInversion] = None,
    ) -> "FitInterferometer":
        """
        Returns a new fit which uses the dataset, plane and other objects of this fit, but uses a different set of
        preloads input into this function.

        This is used when setting up the preloads objects, to concisely test how using different preloads objects
        changes the attributes of the fit.

        Parameters
        ----------
        preloads
            The new preloads which are used to refit the data using the
        settings_inversion
            Settings controlling how an inversion is fitted for example which linear algebra formalism is used.

        Returns
        -------
        A new fit which has used new preloads input into this function but the same dataset, plane and other settings.
        """
        if self.profiling_dict is not None:
            profiling_dict = {}
        else:
            profiling_dict = None

        if settings_inversion is None:
            settings_inversion = self.settings_inversion

        return FitInterferometer(
            dataset=self.interferometer,
            plane=self.plane,
            hyper_background_noise=self.hyper_background_noise,
            use_hyper_scaling=self.use_hyper_scaling,
            settings_pixelization=self.settings_pixelization,
            settings_inversion=settings_inversion,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )
