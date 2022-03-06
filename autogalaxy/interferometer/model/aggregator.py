from functools import partial
from typing import Optional

import autofit as af
import autoarray as aa


def _interferometer_from(
    fit: af.Fit,
    real_space_mask: Optional[aa.Mask2D] = None,
    settings_interferometer: Optional[aa.SettingsInterferometer] = None,
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
    fit : ImaginSearchOutput
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

    interferometer = aa.Interferometer(
        visibilities=data,
        noise_map=noise_map,
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
    )

    interferometer = interferometer.apply_settings(settings=settings_interferometer)

    return interferometer


class InterferometerAgg:
    def __init__(self, aggregator: af.Aggregator):

        self.aggregator = aggregator

    def interferometer_gen_from(
        self,
        real_space_mask: Optional[aa.Mask2D] = None,
        settings_interferometer: Optional[aa.SettingsInterferometer] = None,
    ):
        """
        Returns a generator of `Interferometer` objects from an input aggregator, which generates a list of the
        `Interferometer` objects for every set of results loaded in the aggregator.

        This is performed by mapping the `interferometer_from_agg_obj` with the aggregator, which sets up each
        interferometer object using only generators ensuring that manipulating the interferometer objects of large
        sets of results is done in a memory efficient  way.

        Parameters
        ----------
        aggregator : ImaginAggregator
            A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits."""

        func = partial(
            _interferometer_from,
            real_space_mask=real_space_mask,
            settings_interferometer=settings_interferometer,
        )

        return self.aggregator.map(func=func)
