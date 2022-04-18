from functools import partial
from typing import Optional

import autofit as af
import autoarray as aa


def _imaging_from(fit: af.Fit, settings_imaging: Optional[aa.SettingsImaging] = None):
    """
    Returns a `Imaging` object from an aggregator's `SearchOutput` class, which we call an 'agg_obj' to describe
    that it acts as the aggregator object for one result in the `Aggregator`. This uses the aggregator's generator
    outputs such that the function can use the `Aggregator`'s map function to to create a `Imaging` generator.

     The `Imaging` is created following the same method as the PyAutoGalaxy `Search` classes, including using the
    `SettingsImaging` instance output by the Search to load inputs of the `Imaging` (e.g. psf_shape_2d).

    Parameters
    ----------
    fit : ImaginSearchOutput
        A PyAutoFit aggregator's SearchOutput object containing the generators of the results of PyAutoGalaxy model-fits.
    """

    data = fit.value(name="data")
    noise_map = fit.value(name="noise_map")
    psf = fit.value(name="psf")
    settings_imaging = settings_imaging or fit.value(name="settings_dataset")

    if not hasattr(settings_imaging, "relative_accuracy"):
        settings_imaging.relative_accuracy = None

    imaging = aa.Imaging(
        image=data,
        noise_map=noise_map,
        psf=psf,
        settings=settings_imaging,
        pad_for_convolver=True,
    )

    imaging.apply_settings(settings=settings_imaging)

    return imaging


class ImagingAgg:
    def __init__(self, aggregator: af.Aggregator):

        self.aggregator = aggregator

    def imaging_gen_from(self, settings_imaging: Optional[aa.SettingsImaging] = None):
        """
        Returns a generator of `Imaging` objects from an input aggregator, which generates a list of the
        `Imaging` objects for every set of results loaded in the aggregator.

        This is performed by mapping the `imaging_from_agg_obj` with the aggregator, which sets up each
        imaging using only generators ensuring that manipulating the imaging of large sets of results is done in a
        memory efficient way.

        Parameters
        ----------
        aggregator : ImaginAggregator
            A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits."""

        func = partial(_imaging_from, settings_imaging=settings_imaging)

        return self.aggregator.map(func=func)
