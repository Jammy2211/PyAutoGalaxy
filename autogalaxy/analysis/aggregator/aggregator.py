import autofit as af
import autogalaxy as ag
from autofit.database.model.fit import Fit

from typing import Optional

from functools import partial


class ImagingAgg:
    def __init__(self, aggregator: af.Aggregator):

        self.aggregator = aggregator

    def imaging_gen(self, settings_imaging: Optional[ag.SettingsImaging] = None):
        """
        Returns a generator of `Imaging` objects from an input aggregator, which generates a list of the
        `Imaging` objects for every set of results loaded in the aggregator.

        This is performed by mapping the `imaging_from_agg_obj` with the aggregator, which sets up each
        imaging using only generators ensuring that manipulating the imaging of large sets of results is done in a
        memory efficient way.

        Parameters
        ----------
        aggregator : af.Aggregator
            A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits."""

        func = partial(self._imaging_gen, settings_imaging=settings_imaging)

        return self.aggregator.map(func=func)

    def _imaging_gen(
        self, fit: Fit, settings_imaging: Optional[ag.SettingsImaging] = None
    ):
        """
        Returns a `Imaging` object from an aggregator's `SearchOutput` class, which we call an 'agg_obj' to describe
        that it acts as the aggregator object for one result in the `Aggregator`. This uses the aggregator's generator
        outputs such that the function can use the `Aggregator`'s map function to to create a `Imaging` generator.

         The `Imaging` is created following the same method as the PyAutoGalaxy `Search` classes, including using the
        `SettingsImaging` instance output by the Search to load inputs of the `Imaging` (e.g. psf_shape_2d).

        Parameters
        ----------
        fit : af.SearchOutput
            A PyAutoFit aggregator's SearchOutput object containing the generators of the results of PyAutoGalaxy model-fits.
        """

        data = fit.value(name="data")
        noise_map = fit.value(name="noise_map")
        psf = fit.value(name="psf")
        settings_imaging = settings_imaging or fit.value(name="settings_dataset")

        imaging = ag.Imaging(
            image=data,
            noise_map=noise_map,
            psf=psf,
            settings=settings_imaging,
            setup_convolver=True,
        )

        imaging.apply_settings(settings=settings_imaging)

        return imaging


class InterferometerAgg:
    def __init__(self, aggregator: af.Aggregator):

        self.aggregator = aggregator

    def interferometer_gen(
        self,
        real_space_mask: Optional[ag.Mask2D] = None,
        settings_interferometer: Optional[ag.SettingsInterferometer] = None,
    ):
        """
        Returns a generator of `Interferometer` objects from an input aggregator, which generates a list of the
        `Interferometer` objects for every set of results loaded in the aggregator.

        This is performed by mapping the `interferometer_from_agg_obj` with the aggregator, which sets up each
        interferometer object using only generators ensuring that manipulating the interferometer objects of large
        sets of results is done in a memory efficient  way.

        Parameters
        ----------
        aggregator : af.Aggregator
            A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits."""

        func = partial(
            self._interferometer_gen,
            real_space_mask=real_space_mask,
            settings_interferometer=settings_interferometer,
        )

        return self.aggregator.map(func=func)

    def _interferometer_gen(
        self,
        fit: Fit,
        real_space_mask: Optional[ag.Mask2D] = None,
        settings_interferometer: Optional[ag.SettingsInterferometer] = None,
    ):
        """
        Returns a `Interferometer` object from an aggregator's `SearchOutput` class, which we call an 'agg_obj' to
        describe that it acts as the aggregator object for one result in the `Aggregator`. This uses the aggregator's
        generator outputs such that the function can use the `Aggregator`'s map function to to create a
        `Interferometer` generator.

        The `Interferometer` is created following the same method as the PyAutoGalaxy `Search` classes, including
        using the `SettingsInterferometer` instance output by the Search to load inputs of the `Interferometer`
        (e.g. psf_shape_2d).

        Parameters
        ----------
        fit : af.SearchOutput
            A PyAutoFit aggregator's SearchOutput object containing the generators of the results of PyAutoGalaxy
            model-fits.
        """

        data = fit.value(name="data")
        noise_map = fit.value(name="noise_map")
        uv_wavelengths = fit.value(name="uv_wavelengths")
        real_space_mask = real_space_mask or fit.value(name="real_space_mask")
        settings_interferometer = settings_interferometer or fit.value(
            name="settings_dataset"
        )

        interferometer = ag.Interferometer(
            visibilities=data,
            noise_map=noise_map,
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
        )

        interferometer = interferometer.apply_settings(settings=settings_interferometer)

        return interferometer
