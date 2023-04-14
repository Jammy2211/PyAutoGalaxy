import numpy as np
from typing import Dict, List, Optional

from autoconf import conf
from autoconf import cached_property

import autoarray as aa

from autogalaxy.abstract_fit import AbstractFitInversion
from autogalaxy.analysis.preloads import Preloads
from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.legacy.hyper_data import HyperImageSky
from autogalaxy.legacy.hyper_data import HyperBackgroundNoise
from autogalaxy.plane.plane import Plane
from autogalaxy.plane.to_inversion import PlaneToInversion
from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.light.linear import LightProfileLinear
from autogalaxy.profiles.light.operated.abstract import LightProfileOperated

from autogalaxy.imaging.fit_imaging import FitImaging as FitImagingBase


class FitImaging(FitImagingBase):
    def __init__(
        self,
        dataset: aa.Imaging,
        plane: Plane,
        hyper_image_sky: Optional[HyperImageSky] = None,
        hyper_background_noise: Optional[HyperBackgroundNoise] = None,
        use_hyper_scaling: bool = True,
        settings_pixelization: aa.SettingsPixelization = aa.SettingsPixelization(),
        settings_inversion: aa.SettingsInversion = aa.SettingsInversion(),
        preloads: aa.Preloads = Preloads(),
        profiling_dict: Optional[Dict] = None,
    ):
        """
        Fits an imaging dataset using a `Plane` object.

        The fit performs the following steps:

        1) Compute the sum of all images of galaxy light profiles in the `Plane`.

        2) Blur this with the imaging PSF to created the `blurred_image`.

        3) Subtract this image from the `data` to create the `profile_subtracted_image`.

        4) If the `Plane` has any linear algebra objects (e.g. linear light profiles, a pixelization / regulariation)
           fit the `profile_subtracted_image` with these objects via an inversion.

        5) Compute the `model_data` as the sum of the `blurred_image` and `reconstructed_data` of the inversion (if
           an inversion is not performed the `model_data` is only the `blurred_image`.

        6) Subtract the `model_data` from the data and compute the residuals, chi-squared and likelihood via the
           noise-map (if an inversion is performed the `log_evidence`, including additional terms describing the linear
           algebra solution, is computed).

        When performing a `model-fit`via an `AnalysisImaging` object the `figure_of_merit` of this `FitImaging` object
        is called and returned in the `log_likelihood_function`.

        Parameters
        ----------
        dataset
            The imaging dataset which is fitted by the galaxies in the plane.
        plane
            The plane of galaxies whose light profile images are used to fit the imaging data.
        hyper_image_sky
            If included, accounts for the background sky in the fit.
        hyper_background_noise
            If included, adds a noise-scaling term to the background to account for an inaacurate background sky model.
        use_hyper_scaling
            If set to False, the hyper scaling functions (e.g. the `hyper_image_sky` / `hyper_background_noise`) are
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
            preloads=preloads,
            settings_pixelization=settings_pixelization,
            settings_inversion=settings_inversion,
            profiling_dict=profiling_dict,
        )

        self.hyper_image_sky = hyper_image_sky
        self.hyper_background_noise = hyper_background_noise

        self.use_hyper_scaling = use_hyper_scaling

    @property
    def data(self) -> aa.Array2D:
        """
        Returns the imaging data, which may have a hyper scaling performed which rescales the background sky level
        in order to account for uncertainty in the background sky subtraction.
        """
        if self.use_hyper_scaling:

            return hyper_image_from(
                image=self.dataset.image, hyper_image_sky=self.hyper_image_sky
            )

        return self.dataset.data

    @property
    def noise_map(self) -> aa.Array2D:
        """
        Returns the imaging noise-map, which may have a hyper scaling performed which increase the noise in regions of
        the data that are poorly fitted in order to avoid overfitting.
        """
        if self.use_hyper_scaling:

            return hyper_noise_map_from(
                noise_map=self.dataset.noise_map,
                model_obj=self.plane,
                hyper_background_noise=self.hyper_background_noise,
            )

        return self.dataset.noise_map

    def refit_with_new_preloads(
        self,
        preloads: Preloads,
        settings_inversion: Optional[aa.SettingsInversion] = None,
    ) -> "FitImaging":
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
        profiling_dict = {} if self.profiling_dict is not None else None

        settings_inversion = (
            self.settings_inversion
            if settings_inversion is None
            else settings_inversion
        )

        return FitImaging(
            dataset=self.imaging,
            plane=self.plane,
            hyper_image_sky=self.hyper_image_sky,
            hyper_background_noise=self.hyper_background_noise,
            use_hyper_scaling=self.use_hyper_scaling,
            settings_pixelization=self.settings_pixelization,
            settings_inversion=settings_inversion,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )


def hyper_image_from(image: aa.Array2D, hyper_image_sky: HyperImageSky) -> aa.Array2D:
    """
    Returns a `hyper_image` from the input data's `image` used in a fit.

    The `hyper_image` is the image with its background sky subtraction changed based on the `hyper_image_sky` object,
    which scales the sky level using a `sky_scale` parameter.

    Parameters
    ----------
    image
        The image of the data whose background sky is changed.
    hyper_image_sky
        The model image describing how much the background sky level is scaled.

    Returns
    -------
    The image whose background sky is scaled.
    """
    if hyper_image_sky is not None:
        return hyper_image_sky.hyper_image_from(image=image)
    return image


def hyper_noise_map_from(
    noise_map: aa.Array2D, model_obj, hyper_background_noise: HyperBackgroundNoise
) -> aa.Array2D:
    """
    Returns a `hyper_noise_map` from the input data's `noise_map` used in a fit.

    The `hyper_noise_map` is the noise-map with a background value added or subtracted from it, based on
    the `hyper_background_noise` object, which scales the noise level using a `noise_scale` parameter.

    Parameters
    ----------
    image
        The image of the data whose background sky is changed.
    hyper_image_sky
        The model image describing how much the background sky level is scaled.

    Returns
    -------
    The image whose background sky is scaled.
    """
    hyper_noise_map = model_obj.hyper_noise_map_from(noise_map=noise_map)

    if hyper_background_noise is not None:
        noise_map = hyper_background_noise.hyper_noise_map_from(noise_map=noise_map)

    if hyper_noise_map is not None:
        noise_map = noise_map + hyper_noise_map
        noise_map_limit = conf.instance["general"]["hyper"]["hyper_noise_limit"]
        noise_map[noise_map > noise_map_limit] = noise_map_limit

    return noise_map
