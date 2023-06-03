from functools import partial
from typing import Optional

import autofit as af
import autoarray as aa


def _imaging_from(fit: af.Fit, settings_dataset: Optional[aa.SettingsImaging] = None):
    """
    Returns a `Imaging` object from an aggregator's `SearchOutput` class, which we call an 'agg_obj' to describe
    that it acts as the aggregator object for one result in the `Aggregator`. This uses the aggregator's generator
    outputs such that the function can use the `Aggregator`'s map function to create a `Imaging` generator.

     The `Imaging` is created following the same method as the PyAutoGalaxy `Search` classes, including using the
    `SettingsImaging` instance output by the Search to load inputs of the `Imaging` (e.g. psf_shape_2d).

    Parameters
    ----------
    fit
        A PyAutoFit aggregator's SearchOutput object containing the generators of the results of model-fits.
    """

    data = fit.value(name="data")
    noise_map = fit.value(name="noise_map")
    psf = fit.value(name="psf")

    settings_dataset = settings_dataset or fit.value(name="settings_dataset")

    if not hasattr(settings_dataset, "relative_accuracy"):
        settings_dataset.relative_accuracy = None

    if hasattr(settings_dataset, "sub_size_inversion"):
        settings_dataset.sub_size_pixelization = settings_dataset.sub_size_inversion

    if hasattr(settings_dataset, "grid_inversion_class"):
        settings_dataset.grid_pixelization_class = settings_dataset.grid_inversion_class

    dataset = aa.Imaging(
        data=data,
        noise_map=noise_map,
        psf=psf,
        settings=settings_dataset,
        pad_for_convolver=True,
    )

    dataset.apply_settings(settings=settings_dataset)

    return dataset


class ImagingAgg:
    def __init__(self, aggregator: af.Aggregator):
        self.aggregator = aggregator

    def dataset_gen_from(self, settings_dataset: Optional[aa.SettingsImaging] = None):
        """
        Returns a generator of `Imaging` objects from an input aggregator, which generates a list of the
        `Imaging` objects for every set of results loaded in the aggregator.

        This is performed by mapping the `imaging_from_agg_obj` with the aggregator, which sets up each
        imaging using only generators ensuring that manipulating the imaging of large sets of results is done in a
        memory efficient way.

        Parameters
        ----------
        aggregator
            A PyAutoFit aggregator object containing the results of model-fits.
        """

        func = partial(_imaging_from, settings_dataset=settings_dataset)

        return self.aggregator.map(func=func)
