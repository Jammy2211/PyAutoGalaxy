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

    data = aa.Array2D.from_primary_hdu(primary_hdu=fit.value(name="dataset.data"))
    noise_map = aa.Array2D.from_primary_hdu(
        primary_hdu=fit.value(name="dataset.noise_map")
    )
    psf = aa.Kernel2D.from_primary_hdu(primary_hdu=fit.value(name="dataset.psf"))

    settings_dataset = settings_dataset or fit.value(name="dataset.settings")

    dataset = aa.Imaging(
        data=data,
        noise_map=noise_map,
        psf=psf,
        settings=settings_dataset,
        check_noise_map=False,
    )

    mask = aa.Mask2D.from_primary_hdu(primary_hdu=fit.value(name="dataset.mask"))
    dataset = dataset.apply_mask(mask=mask)

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
